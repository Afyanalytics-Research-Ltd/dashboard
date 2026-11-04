"""Warehouse Studio JSON API — service layer mocked, Django stack real."""

import json
import uuid
from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import AuditLog
from warehouse_studio.models import DetectorRun, FindingTriage, QueryRun, Worksheet
from warehouse_studio.services import (catalog, cost, diagnostics, executor, history, optimizer, sources)
from warehouse_studio.services import session as session_svc
from warehouse_studio.services.connection import Context, StudioConnectionError

User = get_user_model()
QID = '01c82df1-0005-ac42-0001-425a016cd052'
LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache', 'LOCATION': 'ws-api-tests'}}
SETTINGS = {'toolkit_database': 'HOSPITALS', 'toolkit_schema': 'ANALYTICS', 'database': 'HOSPITALS',
            'schema': 'ANALYTICS', 'role': 'DATAANALYSTS', 'warehouse': 'COMPUTE_WH', 'allow_writes': False,
            'credit_price_usd': 2.0}


def url(name, **kw):
    return reverse(f'warehouse_studio:{name}', kwargs=kw or None)


@override_settings(CACHES=LOCMEM, WAREHOUSE_STUDIO=SETTINGS)
class ApiTestCase(TestCase):
    def setUp(self):
        cache.clear()
        self.admin = User.objects.create_user('admin', 'a@t.io', 'pw-123456', is_superuser=True, is_staff=True)
        self.other_admin = User.objects.create_user('admin2', 'b@t.io', 'pw-123456', is_superuser=True, is_staff=True)
        self.plain = User.objects.create_user('plain', 'c@t.io', 'pw-123456')
        self.client.force_login(self.admin)

    def patch(self, target, attr, **kw):
        p = mock.patch.object(target, attr, **kw)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def post(self, name, data=None, raw=None, **kw):
        body = raw if raw is not None else json.dumps(data or {})
        return self.client.post(url(name, **kw), body, content_type='application/json')

    def get(self, name, params=None, **kw):
        return self.client.get(url(name, **kw), params or {})

    def err(self, resp, status, code):
        self.assertEqual(resp.status_code, status, resp.content[:500])
        payload = resp.json()['error']
        self.assertEqual(payload['code'], code)
        return payload

    def audit(self, **kw):
        return AuditLog.objects.filter(resource='Warehouse Studio', **kw)


class AuthTests(ApiTestCase):
    def test_anonymous(self):
        self.client.logout()
        self.err(self.get('api_session'), 401, 'auth')

    def test_non_superuser(self):
        self.client.force_login(self.plain)
        self.err(self.get('api_session'), 403, 'forbidden')
        self.err(self.post('api_query_submit', {'sql': 'select 1'}), 403, 'forbidden')

    def test_method_not_allowed(self):
        self.assertEqual(self.get('api_query_submit').status_code, 405)
        self.assertEqual(self.client.delete(url('api_session')).status_code, 405)


class ErrorMappingTests(ApiTestCase):
    def test_mapping(self):
        info = self.patch(session_svc, 'info')
        cases = [
            (executor.QueryError('bad at line 2 at position 4', errno=1, sqlstate='42000', query_id=QID), 422, 'query'),
            (sources.SourceUnavailable('query_history'), 409, 'toolkit'),
            (StudioConnectionError('down'), 503, 'connection'),
            (ValueError('nope'), 400, 'invalid'),
            (catalog.CatalogError('bad ident'), 400, 'invalid'),
        ]
        for exc, status, code in cases:
            info.side_effect = exc
            with self.subTest(code=code):
                payload = self.err(self.get('api_session'), status, code)
                if code == 'query':
                    self.assertEqual(payload['position'], {'line': 2, 'column': 4})
                    self.assertEqual(payload['query_id'], QID)
                if code == 'toolkit':
                    self.assertEqual(payload['toolkit'], 'HOSPITALS.ANALYTICS')

    def test_unexpected_does_not_leak(self):
        self.patch(session_svc, 'info', side_effect=RuntimeError('password=hunter2'))
        with self.assertLogs('warehouse_studio.api', level='ERROR'):
            payload = self.err(self.get('api_session'), 500, 'server')
        self.assertNotIn('hunter2', payload['message'])

    def test_bad_json(self):
        self.err(self.post('api_query_submit', raw='{nope'), 400, 'invalid')
        self.err(self.post('api_query_submit', raw='[1, 2]'), 400, 'invalid')

    def test_bad_context(self):
        self.patch(session_svc, 'info', return_value={})
        self.err(self.get('api_session', {'role': 'x"y'}), 400, 'invalid')


class SessionTests(ApiTestCase):
    def test_session(self):
        info = self.patch(session_svc, 'info', return_value={'connected': True})
        resp = self.get('api_session', {'warehouse': 'BIG_WH', 'refresh': '1'})
        self.assertEqual(resp.json(), {'connected': True})
        ctx, = info.call_args[0]
        self.assertEqual((ctx.warehouse, ctx.role), ('BIG_WH', 'DATAANALYSTS'))
        self.assertTrue(info.call_args[1]['refresh'])
        self.assertEqual(self.get('api_root').status_code, 200)


class QueryTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.submit = self.patch(executor, 'submit', return_value={'query_id': QID, 'context': {'role': 'DATAANALYSTS'}})

    def test_split(self):
        resp = self.post('api_query_split', {'sql': 'select 1; -- c\nshow tables'})
        self.assertEqual([s['keyword'] for s in resp.json()['statements']], ['SELECT', 'SHOW'])

    def test_submit(self):
        ws = Worksheet.objects.create(owner=self.admin, title='w')
        resp = self.post('api_query_submit', {'sql': 'select 1;', 'context': {'warehouse': 'X'}, 'worksheet_id': str(ws.id)})
        self.assertEqual(resp.status_code, 202)
        data = resp.json()
        self.assertEqual((data['query_id'], data['kind'], data['keyword'], data['statement']), (QID, 'read', 'SELECT', 'select 1'))
        args, kw = self.submit.call_args
        self.assertEqual(args, ('select 1',))
        self.assertEqual(kw['context'], Context('DATAANALYSTS', 'X', 'HOSPITALS', 'ANALYTICS'))
        self.assertEqual(kw['tag'], 'warehouse-studio:worksheet:admin')
        run = QueryRun.objects.get(pk=data['run_id'])
        self.assertEqual((run.status, run.query_id, run.worksheet_id, run.user), ('RUNNING', QID, ws.id, self.admin))
        self.assertTrue(self.audit(action='query', detail__contains=QID).exists())

    def test_others_worksheet_not_attached(self):
        ws = Worksheet.objects.create(owner=self.other_admin, title='w')
        data = self.post('api_query_submit', {'sql': 'select 1', 'worksheet_id': str(ws.id)}).json()
        self.assertIsNone(QueryRun.objects.get(pk=data['run_id']).worksheet_id)

    def test_blocked(self):
        for sql, code in (('delete from t', 'write'), ('select 1; select 2', 'multiple'), ('use role sysadmin', 'session'),
                          ('grant select on t to role r', 'admin'), ('call some.other.proc()', 'call'), ('  ', 'empty')):
            with self.subTest(sql=sql):
                payload = self.err(self.post('api_query_submit', {'sql': sql, 'mode': 'write'}), 400, code)
                self.assertIn('message', payload)
        self.submit.assert_not_called()
        self.assertEqual(QueryRun.objects.filter(status='BLOCKED').count(), 6)

    @override_settings(WAREHOUSE_STUDIO={**SETTINGS, 'allow_writes': True})
    def test_write_mode_unlocked(self):
        self.assertEqual(self.post('api_query_submit', {'sql': 'delete from t', 'mode': 'write'}).status_code, 202)
        self.err(self.post('api_query_submit', {'sql': 'delete from t'}), 400, 'write')
        self.err(self.post('api_query_submit', {'sql': 'create user x', 'mode': 'write'}), 400, 'admin')

    def test_toolkit_call_allowed_in_read(self):
        for sql in ('call HOSPITALS.ANALYTICS.SP_DETECT_QUEUEING(7)', 'CALL sp_ws_warehouses()'):
            with self.subTest(sql=sql):
                self.assertEqual(self.post('api_query_submit', {'sql': sql}).status_code, 202)

    def test_submit_query_error(self):
        self.submit.side_effect = executor.QueryError('error line 1 at position 7', query_id=QID)
        payload = self.err(self.post('api_query_submit', {'sql': 'select x'}), 422, 'query')
        self.assertEqual(payload['position'], {'line': 1, 'column': 7})
        run = QueryRun.objects.get()
        self.assertEqual((run.status, run.query_id), ('FAILED', QID))
        self.assertIsNotNone(run.finished_at)

    def test_submit_connection_error(self):
        self.submit.side_effect = StudioConnectionError('Snowflake is not configured')
        self.err(self.post('api_query_submit', {'sql': 'select 1'}), 503, 'connection')

    def make_run(self, **kw):
        return QueryRun.objects.create(user=self.admin, sql='select 1', query_id=QID, status='RUNNING', **kw)

    def test_status_transitions(self):
        status = self.patch(executor, 'status')
        run = self.make_run()
        status.return_value = {'query_id': QID, 'status': 'RUNNING', 'running': True, 'done': False, 'ok': False, 'error': None}
        self.assertEqual(self.get('api_query_status', query_id=QID).json()['status'], 'RUNNING')
        run.refresh_from_db()
        self.assertEqual(run.status, 'RUNNING')
        status.return_value = {'query_id': QID, 'status': 'SUCCESS', 'running': False, 'done': True, 'ok': True, 'error': None}
        self.get('api_query_status', query_id=QID)
        run.refresh_from_db()
        self.assertEqual(run.status, 'SUCCESS')
        self.assertIsNotNone(run.elapsed_ms)

    def test_status_failed_and_aborted(self):
        status = self.patch(executor, 'status')
        for st, expected in (('FAILED_WITH_ERROR', 'FAILED'), ('ABORTED', 'CANCELLED')):
            QueryRun.objects.all().delete()
            run = self.make_run()
            status.return_value = {'query_id': QID, 'status': st, 'running': False, 'done': True, 'ok': False,
                                   'error': {'message': 'Division by zero'}}
            self.get('api_query_status', query_id=QID)
            run.refresh_from_db()
            self.assertEqual((run.status, run.error), (expected, 'Division by zero'))

    def test_status_does_not_touch_other_users_runs(self):
        self.patch(executor, 'status', return_value={'query_id': QID, 'status': 'SUCCESS', 'running': False,
                                                     'done': True, 'ok': True, 'error': None})
        run = QueryRun.objects.create(user=self.other_admin, sql='x', query_id=QID, status='RUNNING')
        self.get('api_query_status', query_id=QID)
        run.refresh_from_db()
        self.assertEqual(run.status, 'RUNNING')

    def test_results(self):
        results = self.patch(executor, 'results', return_value={'query_id': QID, 'total': 5, 'fetched': 5, 'rows': [[1]]})
        run = self.make_run()
        resp = self.get('api_query_results', {'offset': '-4', 'limit': '999999'}, query_id=QID)
        self.assertEqual(resp.json()['rows'], [[1]])
        self.assertEqual(results.call_args[1], {'offset': 0, 'limit': 10_000})
        run.refresh_from_db()
        self.assertEqual((run.rows, run.status), (5, 'SUCCESS'))

    def test_cancel(self):
        self.patch(executor, 'cancel', return_value={'query_id': QID, 'message': 'cancelled'})
        run = self.make_run()
        self.assertEqual(self.post('api_query_cancel', query_id=QID).json()['message'], 'cancelled')
        run.refresh_from_db()
        self.assertEqual(run.status, 'CANCELLED')

    def test_invalid_query_id(self):
        self.err(self.get('api_query_status', query_id='nope'), 422, 'query')

    def test_profile(self):
        self.patch(diagnostics, 'load_profile', return_value={'nodes': []})
        self.assertEqual(self.get('api_query_profile', query_id=QID).json(), {'nodes': []})

    def test_history_and_runs(self):
        search = self.patch(history, 'search', return_value={'rows': []})
        self.get('api_history', {'scope': 'account', 'q': 'x'})
        self.assertEqual(search.call_args[0][0], {'scope': 'account', 'q': 'x'})
        self.make_run()
        QueryRun.objects.create(user=self.other_admin, sql='secret', query_id=QID)
        runs = self.get('api_runs').json()['runs']
        self.assertEqual([r['sql'] for r in runs], ['select 1'])


class ExportTests(ApiTestCase):
    COLS = [{'name': 'ID', 'family': 'number'}, {'name': 'NOTE', 'family': 'text'}, {'name': 'OBJ', 'family': 'json'}]

    def export(self, fmt, batches):
        self.patch(executor, 'iter_export', return_value=iter([(self.COLS, b) for b in batches]))
        return self.get('api_query_export', {'format': fmt}, query_id=QID)

    def content(self, resp):
        return b''.join(resp.streaming_content).decode()

    def test_csv(self):
        resp = self.export('csv', [[(Decimal('1'), 'a,b', '{"k": 1}')], [(2, None, None)]])
        self.assertEqual(resp['Content-Type'], 'text/csv')
        self.assertIn('attachment; filename="query-01c82df1.csv"', resp['Content-Disposition'])
        self.assertEqual(self.content(resp).splitlines(), ['ID,NOTE,OBJ', '1,"a,b","{""k"": 1}"', '2,,'])
        self.assertTrue(self.audit(action='export').exists())

    def test_tsv(self):
        self.assertEqual(self.content(self.export('tsv', [[(1, 'x', None)]])).splitlines(), ['ID\tNOTE\tOBJ', '1\tx\t'])

    def test_json(self):
        body = self.content(self.export('json', [[(1, 'x', '[1]')], [(2, 'y', None)], [(3, 'z', None)]]))
        self.assertEqual(json.loads(body), [{'ID': 1, 'NOTE': 'x', 'OBJ': [1]}, {'ID': 2, 'NOTE': 'y', 'OBJ': None},
                                            {'ID': 3, 'NOTE': 'z', 'OBJ': None}])

    def test_ndjson(self):
        lines = self.content(self.export('ndjson', [[(1, 'x', None), (2, 'y', None)]])).splitlines()
        self.assertEqual([json.loads(line)['ID'] for line in lines], [1, 2])

    def test_empty(self):
        self.assertEqual(self.content(self.export('json', [])), '[]')
        self.assertEqual(self.content(self.export('csv', [])), '')

    def test_bad_format(self):
        self.err(self.get('api_query_export', {'format': 'xlsx'}, query_id=QID), 400, 'invalid')

    def test_error_before_stream(self):
        def gen():
            raise executor.QueryError('Result for query has expired')
            yield  # pragma: no cover
        self.patch(executor, 'iter_export', return_value=gen())
        self.err(self.get('api_query_export', {'format': 'csv'}, query_id=QID), 422, 'query')


class DetailExplainTests(ApiTestCase):
    def test_details(self):
        q = {'query_id': QID, 'execution_status': 'SUCCESS', 'warehouse_size': 'X-Small', 'execution_time': 3_600_000,
             'bytes_scanned': 1, 'total_elapsed_time': 3_600_000, 'compilation_time': 1}
        self.patch(diagnostics, 'load_query', return_value=q)
        self.patch(diagnostics, 'load_profile', return_value={'nodes': [
            {'type': 'CartesianJoin', 'operator_id': 3, 'stats': {}}]})
        self.patch(diagnostics, 'attributed_credits', return_value=None)
        data = self.get('api_query_details', query_id=QID).json()
        self.assertEqual(data['insights'][0]['title'], 'Cartesian join')
        self.assertEqual((data['cost']['credits'], data['cost']['usd']), (1.0, 2.0))
        self.assertTrue(data['profile_available'])
        self.assertIsNone(data['profile_error'])

    def test_details_profile_unavailable(self):
        self.patch(diagnostics, 'load_query', return_value={'query_id': QID, 'execution_status': 'SUCCESS'})
        self.patch(diagnostics, 'load_profile', side_effect=executor.QueryError('Query too old'))
        self.patch(diagnostics, 'attributed_credits', return_value=0.5)
        data = self.get('api_query_details', query_id=QID).json()
        self.assertEqual(data['profile_error'], 'Query too old')
        self.assertFalse(data['profile_available'])
        self.assertFalse(data['cost']['estimated'])

    def test_details_not_found(self):
        self.patch(diagnostics, 'load_query', return_value=None)
        self.err(self.get('api_query_details', query_id=QID), 404, 'not_found')

    def test_explain(self):
        pre = self.patch(diagnostics, 'preflight', return_value={'estimate': {}})
        self.patch(cost, 'list_warehouses', side_effect=RuntimeError('no'))
        resp = self.post('api_query_explain', {'sql': 'select 1;', 'context': {'warehouse': 'W'}})
        self.assertEqual(resp.json(), {'estimate': {}})
        self.assertEqual(pre.call_args[0], ('select 1',))
        self.assertEqual(pre.call_args[1]['context'].warehouse, 'W')
        self.assertEqual(pre.call_args[1]['warehouses'], [])

    def test_explain_rejects(self):
        self.err(self.post('api_query_explain', {'sql': 'show tables'}), 400, 'explain')
        self.err(self.post('api_query_explain', {'sql': 'delete from t'}), 400, 'write')
        self.err(self.post('api_query_explain', {'sql': 'call HOSPITALS.ANALYTICS.SP_WS_WAREHOUSES()'}), 400, 'call')


class WorksheetTests(ApiTestCase):
    def test_crud(self):
        resp = self.post('api_worksheets', {'title': 'T' * 300, 'sql': 'select 1', 'context': {'warehouse': 'W'}, 'folder': 'f'})
        self.assertEqual(resp.status_code, 201)
        ws = resp.json()
        self.assertEqual(len(ws['title']), 200)
        self.assertEqual(ws['context']['warehouse'], 'W')
        listing = self.get('api_worksheets').json()
        self.assertEqual(len(listing['worksheets']), 1)
        self.assertEqual(listing['worksheets'][0]['preview'], 'select 1')
        self.assertNotIn('sql', listing['worksheets'][0])
        detail = url('api_worksheet', pk=ws['id'])
        r = self.client.patch(detail, json.dumps({'title': '', 'sql': 'select 2', 'starred': True, 'shared': 1,
                                                  'context': {'schema': 'S'}}), content_type='application/json')
        data = r.json()
        self.assertEqual((data['title'], data['sql'], data['starred'], data['shared'], data['context']['schema']),
                         ('Untitled worksheet', 'select 2', True, True, 'S'))
        self.assertEqual(self.client.get(detail).json()['sql'], 'select 2')
        self.assertEqual(self.client.delete(detail).json(), {'deleted': True})
        self.assertFalse(Worksheet.objects.exists())

    def test_ownership(self):
        private = Worksheet.objects.create(owner=self.other_admin, title='p', sql='secret')
        shared = Worksheet.objects.create(owner=self.other_admin, title='s', sql='shared sql', shared=True)
        self.err(self.client.get(url('api_worksheet', pk=private.id)), 403, 'forbidden')
        self.assertEqual(self.client.get(url('api_worksheet', pk=shared.id)).json()['sql'], 'shared sql')
        self.err(self.client.patch(url('api_worksheet', pk=shared.id), '{"sql": "x"}', content_type='application/json'),
                 403, 'forbidden')
        self.err(self.client.delete(url('api_worksheet', pk=shared.id)), 403, 'forbidden')
        listing = self.get('api_worksheets').json()
        self.assertEqual([w['title'] for w in listing['shared']], ['s'])
        self.assertEqual(self.client.get(url('api_worksheet', pk=uuid.uuid4())).status_code, 404)

    def test_create_bad_context(self):
        self.err(self.post('api_worksheets', {'context': {'database': 'a"b'}}), 400, 'invalid')


class CatalogProcedureTests(ApiTestCase):
    def test_catalog(self):
        dbs = self.patch(catalog, 'databases', return_value=[{'name': 'H'}])
        self.assertEqual(self.get('api_databases', {'refresh': '1'}).json(), {'databases': [{'name': 'H'}]})
        self.assertTrue(dbs.call_args[1]['refresh'])
        sch = self.patch(catalog, 'schemas', return_value=[])
        self.get('api_schemas', {'database': 'H'})
        self.assertEqual(sch.call_args[0][0], 'H')
        obj = self.patch(catalog, 'objects', return_value={})
        self.get('api_objects', {'database': 'H', 'schema_name': 'S'})
        self.assertEqual(obj.call_args[0][:2], ('H', 'S'))
        cols = self.patch(catalog, 'columns', return_value=[])
        self.get('api_columns', {'database': 'H', 'schema_name': 'S', 'table': 'T'})
        self.assertEqual(cols.call_args[0][:3], ('H', 'S', 'T'))
        self.get('api_columns', {'database': 'H', 'schema_name': 'S'})
        self.assertIsNone(cols.call_args[0][2])

    def test_ddl_validation(self):
        self.err(self.get('api_ddl', {'type': 'USER', 'name': 'x'}), 400, 'invalid')
        self.patch(executor, 'run', return_value=mock.Mock(rows=[['create view v']]))
        self.assertEqual(self.get('api_ddl', {'type': 'VIEW', 'name': 'v'}).json(), {'ddl': 'create view v'})

    def body(self, name='SP_DETECT_QUEUEING', **kw):
        return {'database': 'hospitals', 'schema': 'analytics', 'name': name,
                'params': [{'name': 'P_LOOKBACK_DAYS', 'type': 'NUMBER', 'has_default': False}], 'values': ['7'], **kw}

    def test_procedure_allow_list(self):
        submit = self.patch(executor, 'submit', return_value={'query_id': QID, 'context': {}})
        resp = self.post('api_procedure_call', self.body())
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(resp.json()['sql'], 'CALL "hospitals"."analytics"."SP_DETECT_QUEUEING"(%s)')
        self.assertEqual(submit.call_args[1]['params'], (7,))
        self.assertEqual(QueryRun.objects.get().kind, 'procedure')
        self.assertTrue(self.audit(action='call').exists())
        self.err(self.post('api_procedure_call', self.body('REVOKE_ALL_AI_ACCESS', mode='write')), 400, 'call')
        self.err(self.post('api_procedure_call', self.body(params='x')), 400, 'invalid')
        self.err(self.post('api_procedure_call', self.body(values=['abc'])), 400, 'invalid')
        self.assertEqual(submit.call_count, 1)

    @override_settings(WAREHOUSE_STUDIO={**SETTINGS, 'allow_writes': True})
    def test_procedure_write_mode(self):
        self.patch(executor, 'submit', return_value={'query_id': QID, 'context': {}})
        payload = self.err(self.post('api_procedure_call', self.body('OTHER')), 400, 'call')
        self.assertIn('Unlock write mode', payload['message'])
        self.assertEqual(self.post('api_procedure_call', self.body('OTHER', mode='write')).status_code, 202)


class CostEndpointTests(ApiTestCase):
    def test_endpoints(self):
        cases = [('api_cost_overview', 'overview', {'days': '14'}, ('14',)),
                 ('api_cost_heatmap', 'heatmap', {}, (30,)),
                 ('api_cost_warehouses', 'warehouses', {'days': '7'}, ('7',)),
                 ('api_cost_attribution', 'attribution', {'days': '7', 'limit': '500'}, ('7', 100)),
                 ('api_cost_top_queries', 'top_queries', {'by': 'spill'}, (7, 'spill', 50)),
                 ('api_cost_storage', 'storage', {'limit': '1'}, (5,))]
        for name, fn, params, expected in cases:
            with self.subTest(name=name):
                m = self.patch(cost, fn, return_value={'ok': fn})
                self.assertEqual(self.get(name, params).json(), {'ok': fn})
                self.assertEqual(m.call_args[0], expected)

    def test_monitors_and_refresh(self):
        self.patch(cost, 'resource_monitors', return_value=[{'name': 'M'}])
        self.assertEqual(self.get('api_cost_monitors').json(), {'monitors': [{'name': 'M'}]})
        clear = self.patch(cost, 'clear_cache')
        self.assertEqual(self.post('api_cost_refresh').json(), {'refreshed': True})
        clear.assert_called_once()

    def test_unavailable(self):
        self.patch(cost, 'overview', side_effect=sources.SourceUnavailable('metering_daily'))
        payload = self.err(self.get('api_cost_overview'), 409, 'toolkit')
        self.assertEqual(payload['source'], 'metering_daily')


class OptimizerTests(ApiTestCase):
    STATE = {'toolkit_installed': False, 'toolkit_procedures': {'SP_DETECT_QUEUEING': True}, 'toolkit_relations': {}}

    def test_catalog(self):
        probe = self.patch(sources, 'probe', return_value=self.STATE)
        DetectorRun.objects.create(user=self.admin, key='queueing', query_id='old', status='SUCCESS')
        newer = DetectorRun.objects.create(user=self.admin, key='queueing', query_id='new', status='SUCCESS')
        data = self.get('api_optimizer', {'refresh': '1'}).json()
        self.assertTrue(probe.call_args[1]['force'])
        flags = {d['key']: d['installed'] for d in data['detectors']}
        self.assertTrue(flags['queueing'])
        self.assertFalse(flags['spill_and_oom'])
        self.assertEqual(len(data['detectors']), 12)
        self.assertEqual(len(data['reports']), 6)
        self.assertEqual(data['last_runs']['queueing']['id'], newer.id)
        self.assertIn('warehouse_studio_install', data['install_command'])

    def test_run_and_poll_success(self):
        start = self.patch(optimizer, 'start', return_value={
            'query_id': QID, 'context': {'role': 'R'}, 'sql': 'CALL X(%(p0)s)', 'params': {'P_LOOKBACK_DAYS': 3},
            'procedure': {}})
        resp = self.post('api_optimizer_run', {'key': 'queueing', 'params': {'P_LOOKBACK_DAYS': '3'}})
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(start.call_args[0], ('queueing', {'P_LOOKBACK_DAYS': '3'}))
        self.assertEqual(start.call_args[1], {'user': 'admin'})
        run_id = resp.json()['run']['id']
        self.assertEqual(QueryRun.objects.get().kind, 'optimizer')
        self.assertTrue(self.audit(action='optimizer').exists())

        status = self.patch(executor, 'status', return_value={'status': 'RUNNING', 'done': False, 'ok': False})
        data = self.get('api_optimizer_run_detail', run_id=run_id).json()
        self.assertEqual((data['status'], data['live_status']), ('RUNNING', 'RUNNING'))

        status.return_value = {'status': 'SUCCESS', 'done': True, 'ok': True}
        self.patch(executor, 'results', return_value={'rows': [['{"detector": "sp_detect_queueing", "findings": 4}']],
                                                       'total': 1})
        clear = self.patch(cost, 'clear_cache')
        data = self.get('api_optimizer_run_detail', run_id=run_id).json()
        self.assertEqual(data['status'], 'SUCCESS')
        self.assertEqual(data['outcome'], {'type': 'status', 'value': {'detector': 'sp_detect_queueing', 'findings': 4}})
        clear.assert_called_once()
        self.assertEqual(QueryRun.objects.get().status, 'SUCCESS')
        status.reset_mock()
        self.get('api_optimizer_run_detail', run_id=run_id)
        status.assert_not_called()  # finished runs are not re-polled

    def test_poll_failure(self):
        run = DetectorRun.objects.create(user=self.admin, key='queueing', query_id=QID)
        self.patch(executor, 'status', return_value={'status': 'FAILED_WITH_ERROR', 'done': True, 'ok': False,
                                                     'error': {'message': 'Insufficient privileges'}})
        data = self.get('api_optimizer_run_detail', run_id=run.id).json()
        self.assertEqual((data['status'], data['error']), ('FAILED', 'Insufficient privileges'))
        self.assertEqual(self.get('api_optimizer_run_detail', run_id=9999).status_code, 404)

    def test_run_errors(self):
        self.err(self.post('api_optimizer_run', {'key': 'nope'}), 400, 'invalid')
        self.patch(optimizer, 'start', side_effect=sources.SourceUnavailable('sp_detect_queueing'))
        self.err(self.post('api_optimizer_run', {'key': 'queueing'}), 409, 'toolkit')
        self.assertFalse(DetectorRun.objects.exists())

    def test_runs_list(self):
        DetectorRun.objects.create(user=self.admin, key='a', query_id='1')
        DetectorRun.objects.create(user=self.admin, key='b', query_id='2')
        self.assertEqual([r['key'] for r in self.get('api_optimizer_runs', {'key': 'b'}).json()['runs']], ['b'])

    def findings(self, ids):
        return self.patch(optimizer, 'findings', return_value={
            'findings': [{'finding_id': i} for i in ids], 'total': len(ids), 'limit': 50, 'offset': 0})

    def test_findings_with_triage(self):
        FindingTriage.objects.create(finding_id=1, status='resolved', note='done')
        f = self.findings([1, 2])
        data = self.get('api_findings', {'severity': ['HIGH', 'LOW'], 'days': '7'}).json()
        self.assertEqual(f.call_args[0][0]['severity'], ['HIGH', 'LOW'])
        self.assertEqual(f.call_args[0][0]['days'], '7')
        self.assertEqual(data['findings'][0]['triage']['status'], 'resolved')
        self.assertIsNone(data['findings'][1]['triage'])

    def test_findings_triage_filter(self):
        f = self.findings([1])
        data = self.get('api_findings', {'triage': 'ignored'}).json()
        self.assertEqual(data['findings'], [])
        f.assert_not_called()
        FindingTriage.objects.create(finding_id=5, status='ignored')
        self.get('api_findings', {'triage': 'ignored'})
        self.assertEqual(f.call_args[0][0]['ids'], [5])

    def test_findings_open_filter(self):
        FindingTriage.objects.create(finding_id=1, status='resolved')
        FindingTriage.objects.create(finding_id=3, status='open')
        self.findings([1, 2, 3])
        data = self.get('api_findings', {'triage': 'open'}).json()
        self.assertEqual([x['finding_id'] for x in data['findings']], [2, 3])

    def test_summary(self):
        FindingTriage.objects.create(finding_id=1, status='resolved')
        FindingTriage.objects.create(finding_id=2, status='resolved')
        FindingTriage.objects.create(finding_id=3, status='ignored')
        self.patch(optimizer, 'summary', return_value={'total': 3})
        self.assertEqual(self.get('api_findings_summary').json(), {'total': 3, 'triage': {'resolved': 2, 'ignored': 1}})

    def test_batches_and_schedule(self):
        rr = self.patch(optimizer, 'recent_runs', return_value=[{'run_id': 1}])
        self.assertEqual(self.get('api_optimizer_batches', {'limit': '9999'}).json(), {'runs': [{'run_id': 1}]})
        self.assertEqual(rr.call_args[0], (500,))
        sched = self.patch(optimizer, 'schedule', return_value={'state': 'suspended'})
        self.assertEqual(self.get('api_optimizer_schedule').json(), {'state': 'suspended'})
        self.assertEqual(sched.call_args[0], ('STATUS',))
        self.assertFalse(self.audit(action='schedule').exists())
        self.post('api_optimizer_schedule', {'action': 'RESUME'})
        self.assertEqual(sched.call_args[0], ('RESUME',))
        self.assertTrue(self.audit(action='schedule', detail='RESUME').exists())

    def test_triage(self):
        data = self.post('api_finding_triage', {'status': 'in_progress', 'note': 'n', 'assignee': 'ann'}, finding_id=7).json()
        self.assertEqual((data['status'], data['note'], data['assignee'], data['updated_by']), ('in_progress', 'n', 'ann', 'admin'))
        self.post('api_finding_triage', {'status': 'resolved'}, finding_id=7)
        t = FindingTriage.objects.get(finding_id=7)
        self.assertEqual((t.status, t.note), ('resolved', 'n'))
        self.err(self.post('api_finding_triage', {'status': 'deleted'}, finding_id=7), 400, 'invalid')

    def test_bulk_triage(self):
        FindingTriage.objects.create(finding_id=1, status='open')
        data = self.post('api_findings_triage_bulk', {'status': 'ignored', 'ids': [1, '2', 'x', -3], 'note': 'noise'}).json()
        self.assertEqual(data, {'updated': 2, 'status': 'ignored'})
        self.assertEqual(sorted(FindingTriage.objects.filter(status='ignored').values_list('finding_id', flat=True)), [1, 2])
        self.err(self.post('api_findings_triage_bulk', {'status': 'gone', 'ids': [1]}), 400, 'invalid')
