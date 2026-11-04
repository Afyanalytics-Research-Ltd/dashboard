"""Live integration tests against a real Snowflake account.

Skipped unless ``WAREHOUSE_STUDIO_LIVE_TESTS=1``.  They only *read*: every
statement is a SELECT / SHOW / EXPLAIN / CALL of a read-only toolkit
procedure.  Point them at an account with::

    WAREHOUSE_STUDIO_LIVE_TESTS=1 python manage.py test warehouse_studio.tests.test_live

Cost Center tests need either the toolkit (install.sql) or a role that can
read SNOWFLAKE.ACCOUNT_USAGE; they skip themselves otherwise.
"""

import os
import time
import unittest

from django.core.cache import cache
from django.test import SimpleTestCase

from warehouse_studio.services import (catalog, cost, diagnostics, executor, history, optimizer, session, sources,
                                       sql_guard)
from warehouse_studio.services.connection import default_context, reset_pool

LIVE = os.getenv('WAREHOUSE_STUDIO_LIVE_TESTS') == '1'


@unittest.skipUnless(LIVE, 'set WAREHOUSE_STUDIO_LIVE_TESTS=1 to run against Snowflake')
class LiveSnowflakeTests(SimpleTestCase):
    databases = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        reset_pool(None)
        cache.clear()
        cls.state = sources.probe(force=True)
        cls.cost_ok = cls.state['toolkit_installed'] or cls.state['account_usage']

    @classmethod
    def tearDownClass(cls):
        reset_pool(None)
        super().tearDownClass()

    def wait(self, qid, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = executor.status(qid)
            if st['done']:
                return st
            time.sleep(0.5)
        self.fail(f'query {qid} did not finish')

    # -------------------------------------------------------------- session / catalog

    def test_session_info(self):
        info = session.info(refresh=True)
        self.assertTrue(info['connected'])
        self.assertTrue(info['user'])
        self.assertIn(info['role'], info['available_roles'])
        self.assertIsInstance(info['warehouses'], list)
        self.assertIn('toolkit_installed', info['toolkit'])

    def test_catalog_walk(self):
        dbs = catalog.databases()
        self.assertTrue(dbs)
        ctx = default_context()
        db = ctx.database or dbs[0]['name']
        schemas = catalog.schemas(db)
        self.assertTrue(schemas)
        names = [s['name'] for s in schemas]
        schema = ctx.schema if ctx.schema in names else names[0]
        objs = catalog.objects(db, schema)
        self.assertIsInstance(objs, dict)
        tables = objs.get('TABLE') or objs.get('VIEW') or []
        if tables:
            cols = catalog.columns(db, schema, tables[0]['name'])
            self.assertTrue(cols)
            ddl = catalog.ddl('TABLE' if objs.get('TABLE') else 'VIEW', f'"{db}"."{schema}"."{tables[0]["name"]}"')
            self.assertIn('create', ddl.lower())

    # -------------------------------------------------------------- async execution

    def test_submit_poll_results(self):
        (st, verdict), = sql_guard.check('select seq4() as n, n * 2 as doubled, current_user() as who, '
                                         'parse_json(\'{"a": 1}\') as obj from table(generator(rowcount => 1234))')
        self.assertEqual(verdict.kind, 'read')
        started = executor.submit(st.text)
        status = self.wait(started['query_id'])
        self.assertTrue(status['ok'], status)
        page = executor.results(started['query_id'], offset=1000, limit=500)
        self.assertEqual(page['total'], 1234)
        self.assertEqual(page['returned'], 234)
        self.assertEqual([c['name'] for c in page['columns']], ['N', 'DOUBLED', 'WHO', 'OBJ'])
        self.assertEqual(page['columns'][3]['family'], 'json')
        self.assertEqual(page['rows'][0][3], {'$json': {'a': 1}})
        rows = [r for _, chunk in executor.iter_export(started['query_id'], batch=500) for r in chunk]
        self.assertEqual(len(rows), 1234)

    def test_compile_error_has_position(self):
        started = None
        try:
            started = executor.submit('select nope_col from table(generator(rowcount => 1))')
        except executor.QueryError as exc:
            self.assertIsNotNone(exc.position)
            return
        st = self.wait(started['query_id'])
        self.assertFalse(st['ok'])
        self.assertIsNotNone(st['error']['position'], st)

    def test_cancel(self):
        started = executor.submit('select system$wait(30)')
        time.sleep(1.5)
        executor.cancel(started['query_id'])
        st = self.wait(started['query_id'], timeout=60)
        self.assertFalse(st['ok'])

    def test_explain_preflight(self):
        ctx = default_context()
        plan = diagnostics.preflight('select count(*) from snowflake_sample_data.tpch_sf1.lineitem '
                                     "where l_shipdate >= '1998-01-01'", context=ctx, warehouses=cost.list_warehouses())
        self.assertGreater(plan['partitions_total'], 0)
        self.assertIn('estimate', plan)
        self.assertGreaterEqual(plan['estimate']['credits'], 0)

    # -------------------------------------------------------------- history / diagnostics

    def test_live_history_and_details(self):
        started = executor.submit('select 42 as answer')
        self.wait(started['query_id'])
        res = history.search({'scope': 'live', 'hours': 1, 'q': started['query_id']})
        self.assertEqual(res['scope'], 'live')
        detail = None
        for _ in range(10):
            detail = diagnostics.load_query(started['query_id'])
            if detail:
                break
            time.sleep(1)
        self.assertIsNotNone(detail)
        self.assertEqual(detail['query_id'], started['query_id'])
        self.assertTrue(diagnostics.diagnose(detail))
        prof = diagnostics.load_profile(started['query_id'])
        self.assertIn('nodes', prof)

    # -------------------------------------------------------------- cost center

    def _need_cost(self):
        if not self.cost_ok:
            self.skipTest('no toolkit and no ACCOUNT_USAGE access for this role')

    def test_cost_overview(self):
        self._need_cost()
        data = cost.overview(30)
        for key in ('daily_by_service', 'daily_by_warehouse', 'storage', 'queries', 'month_daily'):
            self.assertIsInstance(data[key], list, f'{key}: {data[key]}')
        self.assertGreaterEqual(data['total_credits'], 0)
        self.assertIsNotNone(data['forecast'])

    def test_cost_heatmap(self):
        self._need_cost()
        data = cost.heatmap(30)
        self.assertIsInstance(data['cells'], list)
        for c in data['cells']:
            self.assertTrue(1 <= int(c['dow']) <= 7)
            self.assertTrue(0 <= int(c['hour']) <= 23)

    def test_cost_warehouses(self):
        self._need_cost()
        data = cost.warehouses(14)
        self.assertEqual(data['errors'], {})
        self.assertTrue(data['warehouses'])

    def test_cost_attribution(self):
        self._need_cost()
        data = cost.attribution(7)
        for dim in ('user', 'role', 'warehouse', 'database', 'query_tag'):
            self.assertIsInstance(data[dim], list)

    def test_cost_top_queries(self):
        self._need_cost()
        for by in ('credits', 'elapsed', 'spill'):
            data = cost.top_queries(7, by, 10)
            self.assertLessEqual(len(data['queries']), 10)

    def test_cost_storage(self):
        self._need_cost()
        data = cost.storage(10)
        for key in ('daily', 'databases', 'tables'):
            self.assertIsInstance(data[key], list, f'{key}: {data[key]}')

    def test_account_history(self):
        self._need_cost()
        res = history.search({'scope': 'account', 'hours': 48, 'limit': 20, 'sort': 'duration'})
        self.assertLessEqual(len(res['rows']), 20)

    def test_monitors(self):
        self._need_cost()
        self.assertIsInstance(cost.resource_monitors(), list)

    # -------------------------------------------------------------- toolkit

    def test_toolkit_findings_and_reports(self):
        if not self.state['toolkit_installed']:
            self.skipTest('toolkit not installed')
        summary = optimizer.summary({'days': 30})
        self.assertIn('health_score', summary)
        page = optimizer.findings({'days': 30, 'limit': 5})
        self.assertLessEqual(len(page['findings']), 5)
        started = optimizer.start('warehouse_settings', {})
        st = self.wait(started['query_id'], timeout=180)
        self.assertTrue(st['ok'], st)
        outcome = optimizer.parse_outcome(optimizer.get('warehouse_settings'), executor.results(started['query_id']))
        self.assertEqual(outcome['value']['detector'], 'sp_detect_warehouse_settings')
        self.assertIn(optimizer.schedule('STATUS').get('state'), ('started', 'suspended'))
