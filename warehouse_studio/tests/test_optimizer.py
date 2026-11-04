"""optimizer: registry, parameter validation, CALL building, findings SQL, roll-ups."""

import re
from unittest import mock

from django.test import SimpleTestCase, override_settings

from warehouse_studio.services import optimizer, sources
from warehouse_studio.services.optimizer import Param

SETTINGS = {'toolkit_database': 'HOSPITALS', 'toolkit_schema': 'ANALYTICS', 'credit_price_usd': 2.0}
CALL_RE = re.compile(r'^CALL HOSPITALS\.ANALYTICS\.SP_[A-Z_]+\(((%\(p\d+\)s)(, %\(p\d+\)s)*)?\)$')


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class RegistryTests(SimpleTestCase):
    def test_keys_unique_and_counts(self):
        keys = [p.key for p in optimizer.DETECTORS + optimizer.REPORTS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(optimizer.DETECTORS), 12)
        self.assertEqual(len(optimizer.REPORTS), 6)

    def test_every_param(self):
        for proc in optimizer.ALL.values():
            for p in proc.params:
                with self.subTest(proc=proc.key, param=p.name):
                    if p.required:
                        with self.assertRaises(ValueError):
                            p.coerce(None)
                    else:
                        self.assertEqual(p.coerce(None), p.default)
                        self.assertEqual(p.coerce(''), p.default)
                    if p.type in ('int', 'float'):
                        with self.assertRaises(ValueError):
                            p.coerce('abc')
                        if p.min is not None:
                            self.assertEqual(p.coerce(p.min), p.min)
                            with self.assertRaises(ValueError):
                                p.coerce(p.min - 1)
                        if p.max is not None:
                            self.assertEqual(p.coerce(p.max), p.max)
                            with self.assertRaises(ValueError):
                                p.coerce(p.max + 1)
                        if p.default is not None:
                            self.assertTrue(p.min is None or p.default >= p.min)
                            self.assertTrue(p.max is None or p.default <= p.max)

    def test_int_coercion(self):
        p = Param('X', 'X', 'int', 1, 0, 100)
        self.assertEqual(p.coerce('7'), 7)
        self.assertEqual(p.coerce('7.9'), 7)
        self.assertIsInstance(Param('F', 'F', 'float', 0.5).coerce('1'), float)

    def test_identifier_param(self):
        p = Param('T', 'Table', 'identifier', required=True)
        for good in ('t', 'db.s.t', 'DB_1.S$X.T'):
            self.assertEqual(p.coerce(good), good)
        for bad in ('a.b.c.d', 'x; drop table y', '"quoted"', "t'", '1abc', 'a..b'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                p.coerce(bad)

    def test_text_param_length(self):
        p = Param('K', 'Key', 'text')
        self.assertEqual(p.coerce('(a, b)'), '(a, b)')
        with self.assertRaises(ValueError):
            p.coerce('x' * 2001)

    def test_as_dict(self):
        d = optimizer.get('poor_pruning').as_dict()
        self.assertEqual(d['fq'], 'HOSPITALS.ANALYTICS.SP_DETECT_POOR_PRUNING')
        self.assertEqual([p['name'] for p in d['params']], ['P_LOOKBACK_DAYS', 'P_MIN_PARTITIONS', 'P_MIN_RATIO'])

    def test_get_unknown(self):
        with self.assertRaises(ValueError):
            optimizer.get('nope')

    def test_build_call_every_proc(self):
        for proc in optimizer.ALL.values():
            raw = {p.name: 'DB.S.T' for p in proc.params if p.required}
            with self.subTest(proc=proc.key):
                sql, binds, clean = optimizer.build_call(proc, raw)
                self.assertRegex(sql, CALL_RE)
                self.assertIn(proc.procedure, sql)
                self.assertEqual(len(binds), len(proc.params))
                self.assertEqual(list(clean), [p.name for p in proc.params])

    def test_build_call_values_only_in_binds(self):
        sql, binds, clean = optimizer.build_call(optimizer.get('clustering_report'),
                                                 {'P_TABLE': 'db.s.t', 'P_KEY': "(a); drop table x --"})
        self.assertNotIn('drop', sql)
        self.assertEqual(binds, {'p0': 'db.s.t', 'p1': "(a); drop table x --"})
        sql, binds, _ = optimizer.build_call(optimizer.get('warehouse_settings'), {})
        self.assertEqual((sql, binds), ('CALL HOSPITALS.ANALYTICS.SP_DETECT_WAREHOUSE_SETTINGS()', {}))

    def test_build_call_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            optimizer.build_call(optimizer.get('join_explosion'), {'P_LOOKBACK_DAYS': 30})  # max 14
        with self.assertRaises(ValueError):
            optimizer.build_call(optimizer.get('sos_cost'), {})

    def test_callable_procedures(self):
        names = optimizer.callable_procedures()
        self.assertIn('HOSPITALS.ANALYTICS.SP_RUN_ALL_DETECTORS', names)
        self.assertIn('HOSPITALS.ANALYTICS.SP_WS_QUERY_PROFILE', names)
        self.assertEqual(len(names), 18 + len(optimizer.HELPERS))
        self.assertNotIn('HOSPITALS.ANALYTICS.REVOKE_ALL_AI_ACCESS', names)


class OutcomeTests(SimpleTestCase):
    def test_status_json_string(self):
        out = optimizer.parse_outcome(optimizer.get('queueing'), {'rows': [['{"detector": "x", "findings": 3}']]})
        self.assertEqual(out, {'type': 'status', 'value': {'detector': 'x', 'findings': 3}})

    def test_wrapped_json(self):
        out = optimizer.parse_outcome(optimizer.get('sos_cost'), {'rows': [[{'$json': {'a': 1}}]]})
        self.assertEqual(out, {'type': 'variant', 'value': {'a': 1}})

    def test_non_json_string(self):
        self.assertEqual(optimizer.parse_outcome(optimizer.get('queueing'), {'rows': [['plain']]})['value'], 'plain')

    def test_empty(self):
        self.assertIsNone(optimizer.parse_outcome(optimizer.get('queueing'), {'rows': []})['value'])

    def test_table(self):
        res = {'columns': [{'name': 'A'}], 'rows': [[1], [2]], 'total': 2}
        self.assertEqual(optimizer.parse_outcome(optimizer.get('full_analysis'), res),
                         {'type': 'table', 'columns': [{'name': 'A'}], 'rows': [[1], [2]], 'total': 2})


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class FilterTests(SimpleTestCase):
    def test_defaults(self):
        where, binds = optimizer._filters({})
        self.assertEqual(binds, {'days': 30})
        self.assertIn('%(days)s', where)

    def test_days_clamped(self):
        self.assertEqual(optimizer._filters({'days': '9999'})[1]['days'], 365)
        self.assertEqual(optimizer._filters({'days': '-3'})[1]['days'], 1)
        self.assertEqual(optimizer._filters({'days': 'abc'})[1]['days'], 30)

    def test_sanitises_everything(self):
        evil = "x' OR 1=1 --"
        where, binds = optimizer._filters({
            'severity': ['high', 'Critical', evil], 'detector': ['sp_detect_queueing', evil, 'SP_UPPER'],
            'object_type': 'query,' + evil, 'q': evil, 'ids': ['1', '22', evil, '3x']})
        self.assertNotIn(evil, where)
        self.assertNotIn('OR 1=1', where)
        self.assertEqual([v for k, v in binds.items() if k.startswith('sev')], ['HIGH', 'CRITICAL'])
        self.assertEqual([v for k, v in binds.items() if k.startswith('det')], ['sp_detect_queueing'])
        self.assertEqual([v for k, v in binds.items() if k.startswith('obj')], ['QUERY'])
        self.assertEqual(binds['q'], f'%{evil}%')
        self.assertIn('f.finding_id IN (1, 22)', where)

    def test_comma_strings_and_long_query(self):
        _, binds = optimizer._filters({'severity': 'low,medium', 'q': 'x' * 500})
        self.assertEqual(binds['sev0'], 'LOW')
        self.assertEqual(binds['sev1'], 'MEDIUM')
        self.assertEqual(len(binds['q']), 202)

    def test_as_list(self):
        self.assertEqual(optimizer._as_list(None), [])
        self.assertEqual(optimizer._as_list(''), [])
        self.assertEqual(optimizer._as_list(' a, ,b '), ['a', 'b'])
        self.assertEqual(optimizer._as_list(['a', ' ']), ['a'])

    def test_cph_case(self):
        sql = optimizer._cph_case('x')
        self.assertIn("WHEN 'X-SMALL' THEN 1", sql)
        self.assertIn("WHEN '6X-LARGE' THEN 512", sql)
        self.assertEqual(sql.count("WHEN 'MEDIUM'"), 1)


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class FindingsTests(SimpleTestCase):
    def setUp(self):
        p1 = mock.patch.object(sources, 'toolkit_object', side_effect=lambda n, state=None: f'HOSPITALS.ANALYTICS.{n}')
        self.records = mock.patch.object(optimizer.executor, 'records')
        p1.start()
        self.rec = self.records.start()
        self.addCleanup(p1.stop)
        self.addCleanup(self.records.stop)

    def test_findings(self):
        self.rec.return_value = [{'finding_id': 1, 'est_credits': 0.5, 'total_count': 41},
                                 {'finding_id': 2, 'est_credits': None, 'total_count': 41}]
        out = optimizer.findings({'limit': '9999', 'offset': '5', 'sort': 'cost'})
        sql, binds = self.rec.call_args[0]
        self.assertIn('FROM HOSPITALS.ANALYTICS.TUNING_FINDINGS f', sql)
        self.assertIn('ORDER BY est_credits DESC', sql)
        self.assertEqual((binds['limit'], binds['offset']), (500, 5))
        self.assertEqual(out['total'], 41)
        self.assertEqual([f['est_usd'] for f in out['findings']], [1.0, 0.0])
        self.assertNotIn('total_count', out['findings'][0])

    def test_sort_fallback_and_bad_paging(self):
        self.rec.return_value = []
        out = optimizer.findings({'sort': 'f.finding_id; drop table x', 'limit': 'x'})
        sql, binds = self.rec.call_args[0]
        self.assertIn("CASE f.severity WHEN 'CRITICAL'", sql)
        self.assertNotIn('drop table', sql)
        self.assertEqual((binds['limit'], binds['offset']), (50, 0))
        self.assertEqual(out['total'], 0)

    def test_every_sort_key_is_known_sql(self):
        self.rec.return_value = []
        for key, clause in optimizer.SORTS.items():
            optimizer.findings({'sort': key})
            self.assertIn(clause, self.rec.call_args[0][0])

    def test_summary(self):
        self.rec.return_value = [
            {'detector': 'a', 'severity': 'HIGH', 'object_type': 'QUERY', 'day': '2026-11-01', 'findings': 3,
             'wasted_seconds': 100, 'est_credits': 1, 'objects': 3, 'last_detected': '2026-11-01T10:00:00'},
            {'detector': 'a', 'severity': 'LOW', 'object_type': 'QUERY', 'day': '2026-11-02', 'findings': 2,
             'wasted_seconds': None, 'est_credits': None, 'objects': 2, 'last_detected': '2026-11-02T10:00:00'},
            {'detector': 'b', 'severity': 'critical', 'object_type': 'WAREHOUSE', 'day': '2026-11-02', 'findings': 1,
             'wasted_seconds': 500, 'est_credits': 2, 'objects': 1, 'last_detected': None},
        ]
        s = optimizer.summary({})
        self.assertEqual(s['by_severity'], {'CRITICAL': 1, 'HIGH': 3, 'MEDIUM': 0, 'LOW': 2})
        self.assertEqual(s['total'], 6)
        self.assertEqual([d['detector'] for d in s['by_detector']], ['b', 'a'])
        a = s['by_detector'][1]
        self.assertEqual((a['findings'], a['wasted_seconds'], a['est_usd']), (5, 100.0, 2.0))
        self.assertEqual(a['last_detected'], '2026-11-02T10:00:00')
        self.assertEqual([d['day'] for d in s['by_day']], ['2026-11-01', '2026-11-02'])
        self.assertEqual(s['by_day'][1]['CRITICAL'], 1)
        self.assertEqual((s['wasted_seconds'], s['est_credits'], s['est_usd']), (600.0, 3.0, 6.0))
        self.assertEqual(s['health_score'], optimizer.health_score(s['by_severity']))

    def test_health_score(self):
        self.assertEqual(optimizer.health_score({}), 100)
        prev = 100
        for n in (1, 5, 20, 100, 1000):
            score = optimizer.health_score({'HIGH': n})
            self.assertLess(score, prev)
            self.assertGreaterEqual(score, 0)
            prev = score
        self.assertLess(optimizer.health_score({'CRITICAL': 1}), optimizer.health_score({'LOW': 1}))
        self.assertEqual(optimizer.health_score({'BOGUS': 99, 'low': 0}), 100)

    def test_recent_runs(self):
        self.rec.return_value = [{'run_id': 1}]
        self.assertEqual(optimizer.recent_runs(99999), [{'run_id': 1}])
        self.assertEqual(self.rec.call_args[0][1], {'n': 500})

    def test_schedule(self):
        with mock.patch.object(optimizer.executor, 'run') as run:
            run.return_value = mock.Mock(rows=[[{'state': 'suspended'}]])
            self.assertEqual(optimizer.schedule('resume'), {'state': 'suspended'})
            self.assertEqual(run.call_args[0], ('CALL HOSPITALS.ANALYTICS.SP_WS_DETECTOR_SCHEDULE(%s)', ('RESUME',)))
            run.return_value = mock.Mock(rows=[['text']])
            self.assertEqual(optimizer.schedule(), {'raw': 'text'})
        with self.assertRaises(ValueError):
            optimizer.schedule('DROP')


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class StartTests(SimpleTestCase):
    def test_missing_toolkit(self):
        with mock.patch.object(sources, 'probe', return_value={'toolkit_procedures': {}}):
            with self.assertRaises(sources.SourceUnavailable):
                optimizer.start('queueing', {})

    def test_submits(self):
        with mock.patch.object(sources, 'probe', return_value={'toolkit_procedures': {'SP_DETECT_QUEUEING': True}}), \
                mock.patch.object(optimizer.executor, 'submit', return_value={'query_id': 'q', 'context': {}}) as sub:
            out = optimizer.start('queueing', {'P_LOOKBACK_DAYS': '3'}, user='ann')
        sub.assert_called_once()
        self.assertEqual(sub.call_args[0][0], 'CALL HOSPITALS.ANALYTICS.SP_DETECT_QUEUEING(%(p0)s)')
        self.assertEqual(sub.call_args[1]['params'], {'p0': 3})
        self.assertEqual(sub.call_args[1]['tag'], 'warehouse-studio:optimizer:queueing:ann')
        self.assertEqual(out['params'], {'P_LOOKBACK_DAYS': 3})
        self.assertEqual(out['procedure']['key'], 'queueing')

    def test_invalid_params_before_submit(self):
        with mock.patch.object(sources, 'probe', return_value={'toolkit_procedures': {'SP_DETECT_QUEUEING': True}}), \
                mock.patch.object(optimizer.executor, 'submit') as sub:
            with self.assertRaises(ValueError):
                optimizer.start('queueing', {'P_LOOKBACK_DAYS': 'lots'})
        sub.assert_not_called()
