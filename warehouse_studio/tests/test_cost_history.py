"""cost (forecast, roll-ups, caching) and history (filters, SQL building) — executor mocked."""

import datetime as dt
from unittest import mock

from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from warehouse_studio.services import cost, executor, history, sources

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache', 'LOCATION': 'ws-cost-tests'}}
SETTINGS = {'credit_price_usd': 2.0, 'storage_price_per_tb_month_usd': 20.0, 'pool_size': 4,
            'database': 'HOSPITALS', 'toolkit_database': 'HOSPITALS', 'toolkit_schema': 'ANALYTICS'}
STATE = {'toolkit_relations': {}, 'toolkit_procedures': {}, 'account_usage': True}


def days(start, n, credits):
    return [{'day': (start + dt.timedelta(days=i)).isoformat(), 'credits': credits} for i in range(n)]


class ForecastTests(SimpleTestCase):
    def test_mid_month(self):
        today = dt.date(2026, 11, 4)
        rows = days(dt.date(2026, 10, 1), 31, 2) + days(dt.date(2026, 11, 1), 3, 3) + [{'day': today, 'credits': 1}]
        f = cost.month_forecast(rows, today=today)
        self.assertEqual(f['month_to_date'], 10)
        self.assertEqual(f['last_month'], 62)
        self.assertAlmostEqual(f['run_rate_per_day'], 17 / 7, places=4)
        self.assertAlmostEqual(f['projected_month'], 9 + 17 / 7 * 27, places=3)
        self.assertEqual(f['last_month_to_date'], 8)
        self.assertEqual(f['pace_vs_last_month'], 0.25)
        self.assertEqual((f['days_in_month'], f['day_of_month']), (30, 4))

    def test_today_already_above_run_rate(self):
        today = dt.date(2026, 11, 4)
        rows = days(dt.date(2026, 10, 28), 7, 1) + [{'day': today.isoformat() + 'T00:00:00', 'credits': 5}]
        f = cost.month_forecast(rows, today=today)
        # MTD 3 complete days (1 each) + today 5; today counts as-is because it beat the 1/day run-rate.
        self.assertAlmostEqual(f['projected_month'], 3 + 5 + 1 * 26, places=4)

    def test_first_of_month(self):
        today = dt.date(2026, 12, 1)
        rows = days(dt.date(2026, 11, 1), 30, 1) + [{'day': today, 'credits': 0.5}]
        f = cost.month_forecast(rows, today=today)
        self.assertEqual(f['month_to_date'], 0.5)
        self.assertEqual(f['last_month'], 30)
        self.assertEqual(f['run_rate_per_day'], 1)
        self.assertEqual(f['projected_month'], 31)
        self.assertEqual(f['pace_vs_last_month'], -0.5)

    def test_last_day_of_month_and_no_history(self):
        today = dt.date(2026, 12, 31)
        f = cost.month_forecast(days(dt.date(2026, 12, 1), 31, 1), today=today)
        self.assertEqual((f['month_to_date'], f['projected_month'], f['last_month']), (31, 31, 0))
        self.assertIsNone(f['pace_vs_last_month'])

    def test_short_previous_month(self):
        today = dt.date(2026, 3, 31)
        f = cost.month_forecast(days(dt.date(2026, 2, 1), 28, 1), today=today)
        self.assertEqual(f['last_month_to_date'], 28)

    def test_empty(self):
        f = cost.month_forecast([], today=dt.date(2026, 1, 15))
        self.assertEqual((f['month_to_date'], f['projected_month'], f['run_rate_per_day']), (0, 0, 0))


class HelperTests(SimpleTestCase):
    def test_clamp_days(self):
        self.assertEqual(cost.clamp_days('x'), 30)
        self.assertEqual(cost.clamp_days(None, default=7), 7)
        self.assertEqual(cost.clamp_days(0), 1)
        self.assertEqual(cost.clamp_days('999'), 365)
        self.assertEqual(cost.clamp_days(50, maximum=30), 30)

    def test_is_dim_row(self):
        cols = ['user_name', 'role_name', 'warehouse_name', 'database_name', 'query_tag']
        gids = {'user_name': 15, 'role_name': 23, 'warehouse_name': 27, 'database_name': 29, 'query_tag': 30}
        for col, gid in gids.items():
            for other, ogid in gids.items():
                with self.subTest(col=col, gid=ogid):
                    self.assertEqual(cost._is_dim_row({'gid': ogid}, col, cols), col == other)

    @override_settings(WAREHOUSE_STUDIO=SETTINGS)
    def test_parallel_captures_errors(self):
        def unavailable():
            raise sources.SourceUnavailable('query_history')

        def query_error():
            raise executor.QueryError('Object does not exist')

        def boom():
            raise RuntimeError('kaput')
        out = cost._parallel({'ok': lambda: [1], 'u': unavailable, 'q': query_error, 'b': boom})
        self.assertEqual(out['ok'], [1])
        self.assertTrue(out['u']['unavailable'])
        self.assertIn('toolkit', out['u']['error'])
        self.assertEqual(out['q'], {'error': 'Object does not exist'})
        self.assertEqual(out['b'], {'error': 'kaput'})


@override_settings(CACHES=LOCMEM, WAREHOUSE_STUDIO=SETTINGS)
class CachingTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_cached_and_generation_bump(self):
        fn = mock.Mock(side_effect=[1, 2, 3])
        self.assertEqual(cost._cached('k', fn), 1)
        self.assertEqual(cost._cached('k', fn), 1)
        self.assertEqual(cost._cached('other', fn), 2)
        cost.clear_cache()
        self.assertEqual(cost._cached('k', fn), 3)
        cost.clear_cache()  # incr path once the key exists
        self.assertEqual(cache.get(cost.GEN_KEY), 2)


def fake_records(routes):
    def fake(sql, params=None, **kw):
        for needle, rows in routes:
            if needle in sql:
                if isinstance(rows, Exception):
                    raise rows
                return rows
        raise AssertionError(f'unexpected SQL: {sql[:200]}')
    return fake


@override_settings(CACHES=LOCMEM, WAREHOUSE_STUDIO=SETTINGS)
class CostQueryTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        for target, attr, kw in ((sources, 'probe', {'return_value': STATE}),
                                 (sources, 'relation', {'side_effect': lambda name, state=None: f'REL_{name}'})):
            p = mock.patch.object(target, attr, **kw)
            p.start()
            self.addCleanup(p.stop)

    def patch_records(self, routes):
        p = mock.patch.object(executor, 'records', side_effect=fake_records(routes))
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def test_overview(self):
        today = dt.date.today()
        rec = self.patch_records([
            ('service_type', [{'day': '2026-11-01', 'service_type': 'WAREHOUSE_METERING', 'credits': 3, 'used': 3},
                              {'day': '2026-11-02', 'service_type': 'WAREHOUSE_METERING', 'credits': 2, 'used': 2},
                              {'day': '2026-11-02', 'service_type': 'PIPE', 'credits': 1, 'used': 1}]),
            ("DATE_TRUNC('month'", [{'day': today.isoformat(), 'credits': 6}]),
            ('AS compute', [{'day': '2026-11-01', 'warehouse_name': 'A', 'credits': 1},
                            {'day': '2026-11-01', 'warehouse_name': 'B', 'credits': 4}]),
            ('stage_bytes', [{'day': '2026-11-01', 'storage_bytes': 1024 ** 4, 'stage_bytes': 0, 'failsafe_bytes': 0},
                             {'day': '2026-11-02', 'storage_bytes': 2 * 1024 ** 4, 'stage_bytes': 0, 'failsafe_bytes': 0}]),
            ('APPROX_PERCENTILE', [{'day': '2026-11-01', 'queries': 10}]),
        ])
        out = cost.overview('7')
        self.assertEqual(out['days'], 7)
        self.assertEqual(out['total_credits'], 6)
        self.assertEqual(out['total_usd'], 12)
        self.assertEqual([s['service_type'] for s in out['services']], ['WAREHOUSE_METERING', 'PIPE'])
        self.assertEqual([w['warehouse_name'] for w in out['warehouses']], ['B', 'A'])
        self.assertEqual(out['storage_tb'], 2)
        self.assertEqual(out['storage_monthly_usd'], 40)
        self.assertEqual(out['forecast']['month_to_date'], 6)
        sql_with_days = [c for c in rec.call_args_list if c[0][1] == {'d': 7}]
        self.assertTrue(sql_with_days)
        self.assertTrue(all('REL_' in c[0][0] for c in rec.call_args_list))
        cost.overview(7)  # cached
        self.assertEqual(rec.call_count, 5)

    def test_overview_partial_failure(self):
        self.patch_records([
            ('service_type', executor.QueryError('denied')),
            ("DATE_TRUNC('month'", sources.SourceUnavailable('metering_daily')),
            ('AS compute', []), ('stage_bytes', []), ('APPROX_PERCENTILE', []),
        ])
        out = cost.overview(30)
        self.assertEqual(out['daily_by_service'], {'error': 'denied'})
        self.assertTrue(out['month_daily']['unavailable'])
        self.assertIsNone(out['forecast'])
        self.assertEqual((out['total_credits'], out['storage_tb']), (0, 0))

    def test_heatmap(self):
        rec = self.patch_records([('DAYOFWEEKISO', [{'dow': 1, 'hour': 9, 'credits': 2.5}])])
        out = cost.heatmap(500)
        self.assertEqual(out['days'], 90)
        self.assertEqual(out['cells'][0]['credits'], 2.5)
        self.assertEqual(rec.call_args[0][1], {'d': 90})

    def test_warehouses(self):
        settings_rows = [{'name': 'A', 'size': 'X-Small', 'auto_suspend': 600}, {'name': 'IDLE', 'size': 'Small'}]
        exc = executor.QueryError('no events')
        self.patch_records([
            ('idle_hour_credits', [{'warehouse_name': 'A', 'compute_credits': 10, 'cloud_credits': 1, 'active_hours': 12,
                                    'queries': 50, 'exec_seconds': 3600, 'queued_seconds': 5, 'remote_spill_bytes': 0,
                                    'failed': 2, 'idle_hour_credits': 3, 'idle_hours': 4}]),
            ('TO_DATE(start_time) AS day', [{'warehouse_name': 'A', 'day': '2026-11-01', 'credits': 4}]),
            ('avg_running', [{'warehouse_name': 'A', 'avg_running': 0.5}]),
            ('RESUME_WAREHOUSE', exc),
        ])
        with mock.patch.object(cost, 'list_warehouses', return_value=settings_rows):
            out = cost.warehouses(90)
        self.assertEqual(out['days'], 30)
        a, idle = out['warehouses']
        self.assertEqual(a['name'], 'A')
        self.assertEqual((a['usd'], a['utilization'], a['credits_per_hour']), (22.0, 0.1, 1.0))
        self.assertEqual(a['load'], {'warehouse_name': 'A', 'avg_running': 0.5})
        self.assertIsNone(a['events'])
        self.assertEqual(a['daily'], [{'day': '2026-11-01', 'credits': 4.0}])
        self.assertEqual((idle['name'], idle['compute_credits'], idle['utilization']), ('IDLE', 0, None))
        self.assertEqual(out['errors'], {'events': str(exc)})

    def test_attribution(self):
        self.patch_records([('GROUPING SETS', [
            {'gid': 15, 'user_name': 'ANN', 'credits': 1, 'queries': 3, 'exec_seconds': 9},
            {'gid': 15, 'user_name': 'BOB', 'credits': 5, 'queries': 1, 'exec_seconds': 2},
            {'gid': 23, 'role_name': 'R', 'credits': 6, 'queries': 4, 'exec_seconds': 11},
            {'gid': 29, 'database_name': None, 'credits': 2, 'queries': 1, 'exec_seconds': 1},
            {'gid': 30, 'query_tag': '', 'credits': 4, 'queries': 4, 'exec_seconds': 4},
        ])])
        out = cost.attribution(7, limit=2)
        self.assertEqual([u['name'] for u in out['user']], ['BOB', 'ANN'])
        self.assertEqual(out['role'], [{'name': 'R', 'credits': 6, 'queries': 4, 'exec_seconds': 11}])
        self.assertEqual(out['database'][0]['name'], '(none)')
        self.assertEqual(out['query_tag'][0]['name'], '(none)')
        self.assertEqual(out['warehouse'], [])

    def test_top_queries(self):
        rec = self.patch_records([('query_parameterized_hash AS shape', [
            {'shape': 'h', 'credits': 0.5, 'users': '["A", "B"]'}, {'shape': 'g', 'credits': None, 'users': 'bad json'}])])
        out = cost.top_queries(7, 'spill', 10)
        self.assertEqual(out['queries'][0]['users'], ['A', 'B'])
        self.assertEqual(out['queries'][0]['usd'], 1.0)
        self.assertEqual(out['queries'][1]['users'], 'bad json')
        sql, binds = rec.call_args[0][:2]
        self.assertIn('ORDER BY spilled_bytes DESC', sql)
        self.assertIn('LEFT JOIN REL_query_attribution', sql)
        self.assertEqual(binds, {'d': 7, 'n': 10})
        cost.top_queries(7, "credits; drop table x", 10)
        self.assertIn('ORDER BY credits DESC', rec.call_args[0][0])

    def test_top_queries_without_attribution(self):
        def rel(name, state=None):
            if name == 'query_attribution':
                raise sources.SourceUnavailable(name)
            return f'REL_{name}'
        rec = self.patch_records([('AS shape', [])])
        with mock.patch.object(sources, 'relation', side_effect=rel):
            cost.top_queries(7, 'credits', 5)
        sql = rec.call_args[0][0]
        self.assertIn('0 AS credits', sql)
        self.assertNotIn('JOIN', sql)

    def test_storage(self):
        tb = 1024 ** 4
        self.patch_records([
            ('stage_bytes', [{'day': '2026-11-01'}]),
            ('average_database_bytes', [{'database_name': 'H', 'bytes': 5}]),
            ('path_to_object', [{'path_to_object': 'H.S.T', 'total_bytes': 2 * tb, 'time_travel_bytes': tb / 2,
                                 'failsafe_bytes': tb / 2}]),
        ])
        out = cost.storage(1)
        t = out['tables'][0]
        self.assertEqual((t['est_monthly_usd'], t['overhead_monthly_usd']), (40.0, 20.0))
        self.assertEqual(out['storage_price_per_tb_month_usd'], 20.0)


@override_settings(CACHES=LOCMEM, WAREHOUSE_STUDIO=SETTINGS)
class WarehouseListTests(SimpleTestCase):
    def test_toolkit_procedure(self):
        state = {'toolkit_procedures': {'SP_WS_WAREHOUSES': True}}
        with mock.patch.object(sources, 'probe', return_value=state), \
                mock.patch.object(executor, 'records', return_value=[{'name': 'W', 'size': 'Large', 'auto_suspend': '60'}]) as rec:
            out = cost.list_warehouses()
        self.assertEqual(rec.call_args[0][0], 'CALL HOSPITALS.ANALYTICS.SP_WS_WAREHOUSES()')
        self.assertEqual((out[0]['auto_suspend'], out[0]['credits_per_hour']), (60, 8.0))

    def test_show_fallback(self):
        raw = [{'name': 'W', 'size': 'X-Small', 'auto_suspend': None, 'state': 'SUSPENDED', 'extra': 'dropped'},
               {'name': 'V', 'size': 'Small', 'auto_suspend': 'null'}, {'name': 'U', 'size': 'Small', 'auto_suspend': 'x'}]
        with mock.patch.object(sources, 'probe', return_value={}), \
                mock.patch.object(executor, 'records', return_value=raw) as rec:
            out = cost.list_warehouses()
        self.assertEqual(rec.call_args[0][0], 'SHOW WAREHOUSES')
        self.assertNotIn('extra', out[0])
        self.assertEqual([w['auto_suspend'] for w in out], [None, None, None])
        self.assertEqual(out[0]['state'], 'SUSPENDED')

    def test_monitors(self):
        with mock.patch.object(sources, 'probe', return_value={}), \
                mock.patch.object(executor, 'records', return_value=[{'name': 'M', 'credit_quota': '210.00', 'used_credits': None}]):
            out = cost.resource_monitors()
        self.assertEqual((out[0]['credit_quota'], out[0]['used_credits']), (210.0, 0.0))
        with mock.patch.object(sources, 'probe', return_value={'toolkit_procedures': {'SP_WS_RESOURCE_MONITORS': True}}), \
                mock.patch.object(executor, 'records', return_value=[]) as rec:
            cost.resource_monitors()
        self.assertEqual(rec.call_args[0][0], 'CALL HOSPITALS.ANALYTICS.SP_WS_RESOURCE_MONITORS()')


@override_settings(CACHES=LOCMEM, WAREHOUSE_STUDIO=SETTINGS)
class SourcesTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_relation_resolution(self):
        tk = {'toolkit_relations': {'WS_QUERY_HISTORY': True}, 'account_usage': True}
        self.assertEqual(sources.relation('query_history', tk), 'HOSPITALS.ANALYTICS.WS_QUERY_HISTORY')
        self.assertIn('SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY', sources.relation('warehouse_metering', tk))
        with self.assertRaises(sources.SourceUnavailable):
            sources.relation('storage_daily', {'toolkit_relations': {}, 'account_usage': False})
        with self.assertRaises(KeyError):
            sources.relation('nope', tk)

    def test_toolkit_object(self):
        st = {'toolkit_procedures': {'SP_X': True}, 'toolkit_relations': {'T': False}}
        self.assertEqual(sources.toolkit_object('SP_X', st), 'HOSPITALS.ANALYTICS.SP_X')
        with self.assertRaises(sources.SourceUnavailable):
            sources.toolkit_object('T', st)

    def test_probe(self):
        from warehouse_studio.services import installer

        def rec(sql, params=None, **kw):
            return [{'name': n, 'kind': 'x'} for n in installer.EXPECTED_PROCEDURES + installer.EXPECTED_RELATIONS]
        with mock.patch.object(executor, 'records', side_effect=rec), \
                mock.patch.object(executor, 'run', side_effect=executor.QueryError('no access')):
            st = sources.probe(force=True)
        self.assertTrue(st['toolkit_installed'])
        self.assertFalse(st['account_usage'])
        self.assertEqual(st['toolkit_missing'], [])
        with mock.patch.object(executor, 'records') as r2:
            self.assertEqual(sources.probe(), st)  # cached
            r2.assert_not_called()
        sources.invalidate()
        with mock.patch.object(executor, 'records', side_effect=executor.QueryError('no db')), \
                mock.patch.object(executor, 'run', return_value=None):
            st = sources.probe()
        self.assertFalse(st['toolkit_installed'])
        self.assertTrue(st['account_usage'])
        self.assertEqual(st['error'], 'no db')


@override_settings(CACHES=LOCMEM, WAREHOUSE_STUDIO=SETTINGS)
class SessionInfoTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def fake(self, warehouses_error=False):
        def rec(sql, params=None, **kw):
            if sql == 'SHOW WAREHOUSES':
                if warehouses_error:
                    raise executor.QueryError('no')
                return [{'name': 'W', 'size': 'X-Small', 'state': 'SUSPENDED', 'type': 'STANDARD', 'auto_suspend': 60,
                         'owner': 'dropped'}]
            return [{'account_locator': 'MW1', 'user': 'U', 'role': 'R', 'roles': '["R", "PUBLIC"]'}]
        return rec

    def test_info(self):
        from warehouse_studio.services import session
        with mock.patch.object(executor, 'records', side_effect=self.fake()) as rec, \
                mock.patch.object(sources, 'probe', return_value={'toolkit_installed': True}) as probe:
            info = session.info()
            self.assertEqual(info['available_roles'], ['PUBLIC', 'R'])
            self.assertEqual(info['warehouses'], [{'name': 'W', 'size': 'X-Small', 'state': 'SUSPENDED',
                                                   'type': 'STANDARD', 'auto_suspend': 60}])
            self.assertEqual(info['toolkit'], {'toolkit_installed': True})
            self.assertEqual((info['credit_price_usd'], info['allow_writes']), (2.0, False))
            self.assertIn('busy', info['pool'])
            self.assertNotIn('roles', info)
            calls = rec.call_count
            session.info()                       # cached
            self.assertEqual(rec.call_count, calls)
            with mock.patch.object(sources, 'invalidate') as inv:
                session.info(refresh=True)
            inv.assert_called_once()
            self.assertTrue(probe.call_args[1]['force'])
            self.assertGreater(rec.call_count, calls)

    def test_info_without_warehouse_visibility(self):
        from warehouse_studio.services import session
        with mock.patch.object(executor, 'records', side_effect=self.fake(warehouses_error=True)), \
                mock.patch.object(sources, 'probe', return_value={}):
            self.assertEqual(session.info()['warehouses'], [])


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class HistoryTests(SimpleTestCase):
    EVIL = "x' OR 1=1 --"

    def test_where_defaults(self):
        binds = {}
        where = history._where({}, binds)
        self.assertEqual(binds, {'hours': 24})
        self.assertTrue(any('NOT IN' in w for w in where))

    def test_where_filters_use_binds(self):
        binds = {}
        where = history._where({'hours': '99999', 'status': 'failed', 'user': self.EVIL, 'warehouse': 'wh',
                                'role': 'r', 'database': 'd', 'type': 'select', 'q': self.EVIL, 'min_ms': '500',
                                'spilled': '1', 'hide_system': '0'}, binds)
        text = ' AND '.join(where)
        self.assertNotIn('OR 1=1', text)
        self.assertEqual(binds['hours'], 720)
        self.assertEqual(binds['user'], self.EVIL.upper())
        self.assertEqual(binds['q'], f'%{self.EVIL}%')
        self.assertEqual(binds['min_ms'], 500)
        self.assertIn("execution_status IN ('FAIL'", text)
        self.assertIn('UPPER(user_name) = %(user)s', text)
        self.assertIn('bytes_spilled_to_local_storage', text)
        self.assertNotIn('NOT IN', text)

    def test_where_query_id_and_bad_values(self):
        binds = {}
        where = history._where({'q': '01C82DF1-0005-AC42-0001-425A016CD052', 'status': 'bogus', 'min_ms': 'x',
                                'hours': 'y'}, binds)
        self.assertEqual(binds['qid'], '01c82df1-0005-ac42-0001-425a016cd052')
        self.assertIn('query_id = %(qid)s', where)
        self.assertEqual(binds['hours'], 24)
        self.assertNotIn('min_ms', binds)

    def test_live_columns(self):
        out = history.live_columns('query_id, partitions_scanned, bytes_written')
        self.assertEqual(out, 'query_id, NULL AS partitions_scanned, NULL AS bytes_written')

    def test_search_live(self):
        with mock.patch.object(executor, 'records', return_value=[
                {'query_text': 'x' * 5000, 'credits': None, 'est_credits': 0.5}]) as rec:
            out = history.search({'hours': '1000', 'sort': 'spill', 'spilled': '1', 'limit': '5000'})
        sql, binds = rec.call_args[0][:2]
        self.assertEqual(out['scope'], 'live')
        self.assertTrue(sql.startswith('SELECT * FROM (SELECT'))
        self.assertIn('TABLE("HOSPITALS".INFORMATION_SCHEMA.QUERY_HISTORY(', sql)
        self.assertIn('NULL AS bytes_spilled_to_local_storage', sql)
        self.assertIn('ORDER BY start_time DESC', sql)
        where = sql.split(') WHERE ', 1)[1]
        self.assertNotIn('bytes_spilled', where)
        self.assertEqual((binds['hours'], binds['limit']), (168, 1000))
        row = out['rows'][0]
        self.assertEqual(row['usd'], 1.0)
        self.assertEqual(len(row['query_text']), 4001)

    def test_search_account(self):
        with mock.patch.object(sources, 'probe', return_value=STATE), \
                mock.patch.object(sources, 'relation', side_effect=lambda n, s=None: f'REL_{n}'), \
                mock.patch.object(executor, 'records', return_value=[{'query_text': None, 'credits': 0.25, 'est_credits': 9}]) as rec:
            out = history.search({'scope': 'account', 'sort': 'duration', 'hours': '48'})
        sql = rec.call_args[0][0]
        self.assertIn('FROM REL_query_history q LEFT JOIN', sql)
        self.assertIn('REL_query_attribution', sql)
        self.assertIn('ORDER BY total_elapsed_time DESC', sql)
        self.assertEqual(out['rows'][0]['usd'], 0.5)
        self.assertEqual(out['rows'][0]['query_text'], '')

    def test_search_account_without_attribution(self):
        def rel(n, s=None):
            if n == 'query_attribution':
                raise sources.SourceUnavailable(n)
            return f'REL_{n}'
        with mock.patch.object(sources, 'probe', return_value=STATE), \
                mock.patch.object(sources, 'relation', side_effect=rel), \
                mock.patch.object(executor, 'records', return_value=[]) as rec:
            history.search({'scope': 'account'})
        self.assertIn('NULL AS credits', rec.call_args[0][0])
        self.assertNotIn('JOIN', rec.call_args[0][0])
