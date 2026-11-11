"""executor: sync run, async submit/status/results/cancel/export — against a fake driver."""

from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace
from unittest import mock

from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from warehouse_studio.services import executor
from warehouse_studio.services.connection import Context

QID = '01c82df1-0005-ac42-0001-425a016cd052'
LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache', 'LOCATION': 'ws-tests'}}


class DriverError(Exception):
    def __init__(self, msg, errno=None, sqlstate=None, sfqid=None):
        super().__init__(msg)
        self.msg, self.errno, self.sqlstate, self.sfqid = msg, errno, sqlstate, sfqid


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.description = None
        self.rows = []
        self.sfqid = None
        self.rowcount = -1
        self.closed = False

    def execute(self, sql, params=None, timeout=None, _statement_params=None):
        self.db.calls.append(('execute', sql, params, _statement_params))
        if self.db.fail:
            raise self.db.fail
        self.description, self.rows = self.db.result
        self.rowcount = len(self.rows)
        self.sfqid = QID

    def execute_async(self, sql, params=None, _statement_params=None):
        self.db.calls.append(('execute_async', sql, params, _statement_params))
        if self.db.fail:
            raise self.db.fail
        self.sfqid = self.db.async_qid

    def get_results_from_sfqid(self, qid):
        self.db.calls.append(('get_results', qid))
        if self.db.fail:
            raise self.db.fail
        self.description, self.rows = self.db.result
        self.rowcount = self.db.rowcount if self.db.rowcount is not None else len(self.rows)

    def fetchall(self):
        out, self.rows = self.rows, []
        return out

    def fetchmany(self, n):
        out, self.rows = self.rows[:n], self.rows[n:]
        return out

    def close(self):
        self.closed = True


class FakeConn:
    def __init__(self, db):
        self.db = db

    def cursor(self):
        c = FakeCursor(self.db)
        self.db.cursors.append(c)
        return c

    def get_query_status(self, qid):
        self.db.calls.append(('status', qid))
        return SimpleNamespace(name=self.db.status)

    def get_query_status_throw_if_error(self, qid):
        if self.db.status_error:
            raise self.db.status_error


class FakeDB:
    def __init__(self):
        self.calls, self.cursors, self.contexts = [], [], []
        self.result = ([('N', 0, None, None, 38, 0, False)], [(1,)])
        self.fail = None
        self.async_qid = QID
        self.status = 'SUCCESS'
        self.status_error = None
        self.rowcount = None

    @contextmanager
    def connection(self, context=None):
        self.contexts.append(context)
        yield FakeConn(self)


@override_settings(CACHES=LOCMEM, WAREHOUSE_STUDIO={'max_result_rows': 5, 'max_export_rows': 7,
                                                     'account': 'a', 'user': 'u', 'password': 'p', 'role': 'R'})
class ExecutorTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.db = FakeDB()
        patcher = mock.patch.object(executor, 'get_pool', return_value=self.db)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ------------------------------------------------------------ helpers

    def test_valid_query_id(self):
        self.assertEqual(executor.valid_query_id(QID.upper()), QID)
        for bad in ('', None, 'abc', QID + 'x', "'; drop table x; --"):
            with self.subTest(bad=bad), self.assertRaises(executor.QueryError):
                executor.valid_query_id(bad)

    def test_statement_tag(self):
        self.assertEqual(executor.statement_tag('worksheet', 'ann'), 'warehouse-studio:worksheet:ann')
        self.assertEqual(executor.statement_tag('cost'), 'warehouse-studio:cost')

    def test_query_error_position(self):
        err = executor._wrap(DriverError('SQL compilation error: error line 2 at position 7', 1003, '42000', QID))
        self.assertEqual(err.as_dict(), {'message': 'SQL compilation error: error line 2 at position 7', 'errno': 1003,
                                         'sqlstate': '42000', 'query_id': QID, 'position': {'line': 2, 'column': 7}})
        self.assertIs(executor._wrap(err), err)

    # ------------------------------------------------------------ sync

    def test_run_and_records(self):
        self.db.result = ([('ID', 0, None, None, 38, 0, False), ('OBJ', 9, None, None, None, None, True)],
                          [(Decimal('1'), '{"a": 2}'), (Decimal('2'), None)])
        res = executor.run('select 1', ('x',), tag='t')
        self.assertEqual(res.rows, [[1, {'a': 2}], [2, None]])
        self.assertEqual(res.query_id, QID)
        _, sql, params, sp = self.db.calls[0]
        self.assertEqual((sql, params, sp), ('select 1', ('x',), {'QUERY_TAG': 't'}))
        self.assertEqual(executor.records('select 1'), [{'id': 1, 'obj': {'a': 2}}, {'id': 2, 'obj': None}])
        self.assertTrue(all(c.closed for c in self.db.cursors))

    def test_run_default_tag_and_context(self):
        executor.run('select 1')
        self.assertEqual(self.db.calls[0][3], {'QUERY_TAG': 'warehouse-studio:internal'})
        self.assertEqual(self.db.contexts[0].role, 'R')

    def test_run_max_rows(self):
        self.db.result = ([('N', 0)], [(i,) for i in range(10)])
        self.assertEqual(len(executor.run('x', max_rows=3).rows), 3)

    def test_run_wraps_errors(self):
        self.db.fail = DriverError('boom at line 1 at position 3', 2000, '42S02')
        with self.assertRaises(executor.QueryError) as cm:
            executor.run('select')
        self.assertEqual((cm.exception.errno, cm.exception.sqlstate), (2000, '42S02'))
        self.assertTrue(self.db.cursors[0].closed)

    # ------------------------------------------------------------ async

    def test_submit(self):
        ctx = Context(warehouse='W')
        out = executor.submit('select 1', context=ctx, tag='tag-x', params={'a': 1})
        self.assertEqual(out, {'query_id': QID, 'context': ctx.as_dict()})
        self.assertEqual(self.db.calls[0], ('execute_async', 'select 1', {'a': 1}, {'QUERY_TAG': 'tag-x'}))
        self.assertEqual(self.db.contexts[0], ctx)

    def test_submit_default_tag(self):
        executor.submit('select 1')
        self.assertEqual(self.db.calls[0][3], {'QUERY_TAG': 'warehouse-studio:worksheet'})

    def test_submit_error(self):
        self.db.fail = DriverError("SQL compilation error:\nerror line 1 at position 7\ninvalid identifier 'X'", 904, '42000', QID)
        with self.assertRaises(executor.QueryError) as cm:
            executor.submit('select x')
        self.assertEqual(cm.exception.position, {'line': 1, 'column': 7})
        self.assertEqual(cm.exception.query_id, QID)

    def test_submit_without_qid(self):
        self.db.async_qid = None
        with self.assertRaises(executor.QueryError):
            executor.submit('select 1')

    def test_status_running(self):
        for st in ('RUNNING', 'QUEUED', 'RESUMING_WAREHOUSE', 'NO_DATA', 'BLOCKED'):
            self.db.status = st
            with self.subTest(st=st):
                out = executor.status(QID)
                self.assertTrue(out['running'])
                self.assertFalse(out['done'])
                self.assertIsNone(out['error'])

    def test_status_success(self):
        self.assertEqual(executor.status(QID), {'query_id': QID, 'status': 'SUCCESS', 'running': False, 'done': True,
                                                'ok': True, 'error': None})

    def test_status_failed_with_message(self):
        self.db.status = 'FAILED_WITH_ERROR'
        self.db.status_error = DriverError('Division by zero', 100051, '22012', QID)
        out = executor.status(QID)
        self.assertFalse(out['ok'])
        self.assertTrue(out['done'])
        self.assertEqual(out['error']['message'], 'Division by zero')
        self.assertEqual(out['error']['errno'], 100051)

    def test_status_aborted_without_exception(self):
        self.db.status = 'ABORTED'
        out = executor.status(QID)
        self.assertEqual(out['error']['message'], 'Query aborted.')

    def test_status_driver_error(self):
        conn_err = DriverError('nope')

        def boom(qid):
            raise conn_err
        with mock.patch.object(FakeConn, 'get_query_status', side_effect=boom):
            with self.assertRaises(executor.QueryError):
                executor.status(QID)

    def test_results_paging_and_cache(self):
        self.db.result = ([('N', 0, None, None, 38, 0, False)], [(i,) for i in range(4)])
        p1 = executor.results(QID, offset=0, limit=3)
        self.assertEqual(p1['rows'], [[0], [1], [2]])
        self.assertEqual((p1['total'], p1['fetched'], p1['truncated'], p1['returned']), (4, 4, False, 3))
        p2 = executor.results(QID, offset=3, limit=3)
        self.assertEqual(p2['rows'], [[3]])
        self.assertEqual(sum(1 for c in self.db.calls if c[0] == 'get_results'), 1)
        executor.forget(QID)
        executor.results(QID)
        self.assertEqual(sum(1 for c in self.db.calls if c[0] == 'get_results'), 2)

    def test_results_truncated(self):
        self.db.result = ([('N', 0)], [(i,) for i in range(9)])
        self.db.rowcount = 9
        res = executor.results(QID, limit=100)
        self.assertTrue(res['truncated'])
        self.assertEqual(res['fetched'], 5)
        self.assertEqual(res['total'], 9)
        self.assertEqual(res['max_rows'], 5)

    def test_results_truncated_unknown_total(self):
        self.db.result = ([('N', 0)], [(i,) for i in range(9)])
        self.db.rowcount = -1
        self.assertIsNone(executor.results(QID)['total'])

    def test_results_limits_clamped(self):
        self.db.result = ([('N', 0)], [(1,)])
        res = executor.results(QID, offset=-5, limit=0)
        self.assertEqual((res['offset'], res['limit']), (0, 1))

    def test_results_error(self):
        self.db.fail = DriverError('Query not found', 1)
        with self.assertRaises(executor.QueryError):
            executor.results(QID)

    def test_cancel(self):
        self.db.result = ([('R', 2)], [('query cancelled',)])
        out = executor.cancel(QID)
        self.assertEqual(out, {'query_id': QID, 'message': 'query cancelled'})
        _, sql, params, _ = self.db.calls[0]
        self.assertEqual(sql, 'SELECT SYSTEM$CANCEL_QUERY(%s)')
        self.assertEqual(params, (QID,))

    def test_iter_export(self):
        self.db.result = ([('N', 0)], [(i,) for i in range(20)])
        batches = list(executor.iter_export(QID, batch=3))
        sizes = [len(rows) for _, rows in batches]
        self.assertEqual(sizes, [3, 3, 1])  # capped at max_export_rows=7
        self.assertEqual(batches[0][0][0]['name'], 'N')

    def test_iter_export_explicit_max(self):
        self.db.result = ([('N', 0)], [(i,) for i in range(4)])
        self.assertEqual(sum(len(r) for _, r in executor.iter_export(QID, batch=10, max_rows=100)), 4)

    def test_iter_export_error(self):
        self.db.fail = DriverError('gone')
        with self.assertRaises(executor.QueryError):
            list(executor.iter_export(QID))
