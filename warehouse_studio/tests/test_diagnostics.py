"""diagnostics: per-query rules, cost, operator profile, EXPLAIN pre-flight."""

import json
from unittest import mock

from django.test import SimpleTestCase, override_settings

from warehouse_studio.services import diagnostics as dg
from warehouse_studio.services import executor, sources
from warehouse_studio.services.connection import Context

GB = 1024 ** 3
QID = '01c82df1-0005-ac42-0001-425a016cd052'
SETTINGS = {'credit_price_usd': 2.0, 'toolkit_database': 'HOSPITALS', 'toolkit_schema': 'ANALYTICS',
            'database': 'HOSPITALS'}


def base(**kw):
    row = {'execution_status': 'SUCCESS', 'total_elapsed_time': 1000, 'compilation_time': 100, 'execution_time': 900,
           'warehouse_size': 'X-Small', 'bytes_scanned': 1000, 'query_type': 'SELECT', 'query_text': 'select a from t'}
    row.update(kw)
    return row


def titles(q, ops=None):
    return [i['title'] for i in dg.diagnose(q, ops)]


class DiagnoseTests(SimpleTestCase):
    def test_healthy(self):
        self.assertEqual(titles(base()), ['Looks healthy'])

    def test_metadata_only(self):
        self.assertEqual(titles(base(warehouse_size=None, bytes_scanned=0)), ['Looks healthy', 'Metadata-only'])

    def test_oom(self):
        out = dg.diagnose(base(execution_status='FAIL', error_message='Out of memory on node'))
        self.assertEqual((out[0]['title'], out[0]['severity']), ('Out of memory', 'CRITICAL'))

    def test_spill_levels(self):
        self.assertEqual(dg.diagnose(base(bytes_spilled_to_remote_storage=2 * GB))[0]['title'], 'Remote spill')
        self.assertEqual(dg.diagnose(base(bytes_spilled_to_local_storage=2 * GB))[0]['severity'], 'HIGH')
        self.assertEqual(dg.diagnose(base(bytes_spilled_to_local_storage=10))[0]['title'], 'Minor local spill')
        both = titles(base(bytes_spilled_to_remote_storage=GB, bytes_spilled_to_local_storage=GB))
        self.assertIn('Remote spill', both)
        self.assertNotIn('Local spill', both)

    def test_skew(self):
        self.assertIn('Possible join skew', titles(base(bytes_spilled_to_local_storage=5, bytes_sent_over_the_network=2 * GB)))
        self.assertNotIn('Possible join skew', titles(base(bytes_sent_over_the_network=2 * GB)))

    def test_pruning(self):
        out = dg.diagnose(base(partitions_scanned=1990, partitions_total=2000))
        self.assertEqual((out[0]['title'], out[0]['severity'], out[0]['metric']), ('Poor partition pruning', 'HIGH', '100%'))
        self.assertEqual(dg.diagnose(base(partitions_scanned=1700, partitions_total=2000))[0]['severity'], 'MEDIUM')
        self.assertNotIn('Poor partition pruning', titles(base(partitions_scanned=999, partitions_total=999)))
        self.assertNotIn('Poor partition pruning', titles(base(partitions_scanned=100, partitions_total=2000)))

    def test_compilation(self):
        self.assertEqual(dg.diagnose(base(compilation_time=2000, execution_time=100))[0]['severity'], 'MEDIUM')
        self.assertEqual(dg.diagnose(base(compilation_time=20000, execution_time=100))[0]['severity'], 'HIGH')
        self.assertNotIn('Compilation dominates', titles(base(compilation_time=900, execution_time=100)))

    def test_queue(self):
        out = dg.diagnose(base(queued_overload_time=5000, total_elapsed_time=6000))
        self.assertEqual((out[0]['title'], out[0]['severity'], out[0]['category']), ('Waited in the queue', 'HIGH', 'concurrency'))
        self.assertEqual(dg.diagnose(base(queued_overload_time=2000, total_elapsed_time=5000))[0]['severity'], 'MEDIUM')

    def test_provisioning_and_blocked(self):
        self.assertIn('Warehouse start-up', titles(base(queued_provisioning_time=2500)))
        out = dg.diagnose(base(transaction_blocked_time=400000))
        self.assertEqual((out[0]['title'], out[0]['severity']), ('Blocked by a lock', 'HIGH'))

    def test_select_star_and_big_result(self):
        self.assertIn('SELECT *', titles(base(query_text='SELECT  *  FROM t', bytes_scanned=2 * GB)))
        self.assertNotIn('SELECT *', titles(base(query_text='select * from t', bytes_scanned=10)))
        self.assertIn('Very large result', titles(base(rows_produced=2_000_000)))
        self.assertNotIn('Very large result', titles(base(rows_produced=2_000_000, query_type='INSERT')))

    def test_cache(self):
        out = dg.diagnose(base(percentage_scanned_from_cache=0.9))
        self.assertEqual(out[0]['title'], 'Served from warehouse cache')
        self.assertEqual(out[0]['severity'], 'INFO')

    def test_operator_rules(self):
        ops = [{'operator_type': 'CartesianJoin', 'operator_id': 4, 'operator_statistics': {}},
               {'operator_type': 'Join', 'operator_id': 2,
                'operator_statistics': {'input_rows': 1000, 'output_rows': 2_000_000}}]
        out = dg.diagnose(base(), ops)
        self.assertEqual([i['title'] for i in out], ['Cartesian join', 'Exploding join'])
        self.assertIn('#4', out[0]['detail'])
        small = [{'operator_type': 'Join', 'operator_id': 2, 'operator_statistics': {'input_rows': 10, 'output_rows': 1000}}]
        self.assertEqual(titles(base(), small), ['Looks healthy'])

    def test_sorted_by_severity(self):
        out = dg.diagnose(base(bytes_spilled_to_local_storage=10, bytes_spilled_to_remote_storage=GB,
                               queued_provisioning_time=5000, compilation_time=2000, execution_time=100))
        order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']
        sev = [order.index(i['severity']) for i in out]
        self.assertEqual(sev, sorted(sev))

    def test_handles_missing_and_garbage(self):
        self.assertEqual(dg.diagnose({}), [])
        dg.diagnose({'total_elapsed_time': 'abc', 'execution_status': 'SUCCESS'})


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class CostTests(SimpleTestCase):
    def test_estimated(self):
        out = dg.query_cost(base(execution_time=3_600_000, warehouse_size='Medium', credits_used_cloud_services=0.5))
        self.assertEqual((out['credits'], out['cloud_credits'], out['estimated'], out['credits_per_hour']), (4.0, 0.5, True, 4.0))
        self.assertEqual(out['usd'], 9.0)

    def test_attributed(self):
        out = dg.query_cost(base(), attributed_credits=0.25)
        self.assertEqual((out['credits'], out['estimated'], out['usd']), (0.25, False, 0.5))

    def test_formatting(self):
        self.assertEqual(dg.fmt_bytes(512), '512 B')
        self.assertEqual(dg.fmt_bytes(1536), '1.5 KB')
        self.assertEqual(dg.fmt_bytes(GB * 3), '3.0 GB')
        self.assertEqual(dg.fmt_ms(250), '250 ms')
        self.assertEqual(dg.fmt_ms(2500), '2.5 s')
        self.assertEqual(dg.fmt_ms(125_000), '2m 5s')
        self.assertEqual(dg.fmt_ms(3_725_000), '1h 2m')


class ProfileTests(SimpleTestCase):
    ROWS = [
        {'step_id': 1, 'operator_id': 0, 'parent_operators': None, 'operator_type': 'Result',
         'operator_statistics': json.dumps({'output_rows': 10}), 'execution_time_breakdown': json.dumps({'overall_percentage': 0.05}),
         'operator_attributes': json.dumps({'expressions': ['A']})},
        {'step_id': 1, 'operator_id': 1, 'parent_operators': json.dumps([0]), 'operator_type': 'Join',
         'operator_statistics': {'input_rows': 100, 'output_rows': 10, 'spilling': {'bytes_spilled_local_storage': 50}},
         'execution_time_breakdown': {'overall_percentage': 0.6}, 'operator_attributes': {'join_condition': 'a = b'}},
        {'step_id': 1, 'operator_id': 2, 'parent_operators': [1], 'operator_type': 'TableScan',
         'operator_statistics': {'output_rows': 100, 'spilling': {'bytes_spilled_remote_storage': 25}},
         'execution_time_breakdown': {'overall_percentage': 0.35},
         'operator_attributes': {'table_name': 'DB.S.' + 'T' * 300}},
        {'step_id': 1, 'operator_id': 3, 'parent_operators': [1], 'operator_type': 'Filter',
         'operator_statistics': None, 'execution_time_breakdown': None, 'operator_attributes': 'not-json'},
    ]

    def test_analyze(self):
        out = dg.analyze_profile(self.ROWS)
        self.assertEqual(out['operators'], 4)
        self.assertEqual(out['steps'], 1)
        self.assertEqual(out['hottest'], ['1:1', '1:2', '1:0'])
        self.assertEqual(out['edges'], [{'from': '1:1', 'to': '1:0', 'rows': 10}, {'from': '1:2', 'to': '1:1', 'rows': 100},
                                        {'from': '1:3', 'to': '1:1', 'rows': None}])
        self.assertEqual(out['spilled_bytes'], 75)
        nodes = {n['id']: n for n in out['nodes']}
        self.assertAlmostEqual(nodes['1:1']['percent'], 60.0)
        self.assertEqual(nodes['1:1']['label'], 'a = b')
        self.assertEqual(len(nodes['1:2']['label']), 160)
        self.assertEqual(nodes['1:0']['label'], '')
        self.assertEqual(nodes['1:3']['label'], '')
        self.assertEqual(dg.analyze_profile([])['hottest'], [])

    @override_settings(WAREHOUSE_STUDIO=SETTINGS)
    def test_load_profile_toolkit_and_fallback(self):
        with mock.patch.object(executor, 'records', return_value=self.ROWS[:1]) as rec:
            with mock.patch.object(sources, 'probe', return_value={'toolkit_procedures': {'SP_WS_QUERY_PROFILE': True}}):
                dg.load_profile(QID)
            self.assertEqual(rec.call_args[0][:2], ('CALL HOSPITALS.ANALYTICS.SP_WS_QUERY_PROFILE(%s)', (QID,)))
            with mock.patch.object(sources, 'probe', return_value={}):
                self.assertEqual(dg.load_profile(QID.upper())['operators'], 1)
            self.assertEqual(rec.call_args[0][:2], ('SELECT * FROM TABLE(GET_QUERY_OPERATOR_STATS(%s))', (QID,)))
        with self.assertRaises(executor.QueryError):
            dg.load_profile("x'); drop")


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class LoadQueryTests(SimpleTestCase):
    def test_account_view_first(self):
        with mock.patch.object(sources, 'relation', return_value='REL'), \
                mock.patch.object(executor, 'records', return_value=[{'query_id': QID}]) as rec:
            out = dg.load_query(QID)
        self.assertEqual(out, {'query_id': QID, 'source': 'account_usage', 'complete': True})
        self.assertIn('FROM REL WHERE query_id = %s', rec.call_args[0][0])

    def test_falls_back_to_information_schema(self):
        calls = []

        def fake(sql, params=None, **kw):
            calls.append(sql)
            if 'QUERY_HISTORY_BY_USER' in sql:
                raise executor.QueryError('nope')
            return [{'query_id': QID}]
        with mock.patch.object(sources, 'relation', side_effect=sources.SourceUnavailable('query_history')), \
                mock.patch.object(executor, 'records', side_effect=fake):
            out = dg.load_query(QID)
        self.assertEqual(out['source'], 'information_schema')
        self.assertFalse(out['complete'])
        self.assertIn('NULL AS partitions_scanned', calls[-1])
        self.assertIn('"HOSPITALS".INFORMATION_SCHEMA.QUERY_HISTORY(', calls[-1])

    def test_not_found(self):
        with mock.patch.object(sources, 'relation', return_value='REL'), \
                mock.patch.object(executor, 'records', return_value=[]):
            self.assertIsNone(dg.load_query(QID))

    def test_attributed_credits(self):
        with mock.patch.object(sources, 'relation', return_value='REL'):
            with mock.patch.object(executor, 'records', return_value=[{'c': 0.25}]):
                self.assertEqual(dg.attributed_credits(QID), 0.25)
            with mock.patch.object(executor, 'records', return_value=[{'c': None}]):
                self.assertIsNone(dg.attributed_credits(QID))
            with mock.patch.object(executor, 'records', side_effect=executor.QueryError('x')):
                self.assertIsNone(dg.attributed_credits(QID))


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class ExplainTests(SimpleTestCase):
    PLAN = {'GlobalStats': {'partitionsTotal': 2000, 'partitionsAssigned': 1900, 'bytesAssigned': 200 * 1024 ** 2},
            'Operations': [[{'id': 0, 'operation': 'Result'},
                            {'id': 1, 'operation': 'CartesianJoin', 'parentOperators': [0]},
                            {'id': 2, 'operation': 'TableScan', 'objects': ['DB.S.T'], 'partitionsAssigned': 1900,
                             'partitionsTotal': 2000, 'parentOperators': [1]}]]}

    def test_parse(self):
        out = dg.parse_explain(json.dumps(self.PLAN))
        self.assertEqual(out['partitions_total'], 2000)
        self.assertEqual(out['prune_ratio'], 0.05)
        self.assertEqual(out['tables'], ['DB.S.T'])
        self.assertEqual([w['severity'] for w in out['warnings']], ['CRITICAL', 'HIGH'])
        self.assertIn('DB.S.T', out['warnings'][1]['text'])
        self.assertEqual(len(out['operations']), 3)

    def test_parse_empty(self):
        out = dg.parse_explain({})
        self.assertEqual((out['partitions_total'], out['prune_ratio'], out['warnings']), (0, None, []))
        self.assertEqual(dg.parse_explain('')['operations'], [])

    def test_estimate(self):
        running = dg.estimate_cost(200 * 1024 ** 2, {'size': 'X-Small', 'state': 'STARTED'})
        self.assertEqual((running['seconds'], running['billed_seconds']), (2.0, 2.0))
        self.assertFalse(running['resume_minimum_applies'])
        cold = dg.estimate_cost(200 * 1024 ** 2, {'size': 'X-Small', 'state': 'SUSPENDED'})
        self.assertEqual(cold['billed_seconds'], 60.0)
        self.assertTrue(cold['resume_minimum_applies'])
        self.assertAlmostEqual(cold['credits'], 60 / 3600, places=6)
        self.assertAlmostEqual(cold['usd'], 2 * 60 / 3600, places=4)
        medium = dg.estimate_cost(400 * 1024 ** 2, {'size': 'Medium'})
        self.assertEqual((medium['seconds'], medium['credits_per_hour']), (1.0, 4.0))
        self.assertEqual(dg.estimate_cost(0, None)['credits_per_hour'], 1.0)
        self.assertEqual(dg.estimate_cost(100, {'size': 'X-Small'}, throughput_per_cph=100)['seconds'], 1.0)

    def test_preflight(self):
        with mock.patch.object(executor, 'run', return_value=mock.Mock(rows=[[json.dumps(self.PLAN)]])) as run:
            out = dg.preflight('select 1', context=Context(warehouse='w'),
                               warehouses=[{'name': 'W', 'size': 'Medium', 'state': 'STARTED'}])
        self.assertEqual(run.call_args[0][0], 'EXPLAIN USING JSON select 1')
        self.assertEqual(out['estimate']['credits_per_hour'], 4.0)
        with mock.patch.object(executor, 'run', return_value=mock.Mock(rows=[])):
            out = dg.preflight('select 1', context=Context())
        self.assertEqual(out['estimate']['warehouse_size'], 'X-Small')
