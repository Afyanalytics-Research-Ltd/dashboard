"""Internal read queries survive one dropped network connection; real errors are not retried."""

from unittest import mock

from django.test import SimpleTestCase

from warehouse_studio.services import connection, executor


class FakeCursor:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.description = [('X', 0, None, None, 38, 0, True)]
        self.sfqid = '01c82df1-0005-ac42-0001-425a016cd052'

    def execute(self, *a, **k):
        b = self.behaviour.pop(0)
        if isinstance(b, Exception):
            raise b

    def fetchall(self):
        return [(1,)]

    def fetchmany(self, n):
        return [(1,)]

    def close(self):
        pass


class FakeConn:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.closed = False

    def cursor(self):
        return FakeCursor(self.behaviour)

    def is_closed(self):
        return self.closed

    def close(self):
        self.closed = True


class ResilienceTests(SimpleTestCase):

    def _pool(self, behaviour):
        made = []

        class Connector:
            @staticmethod
            def connect(**kw):
                c = FakeConn(behaviour)
                made.append(c)
                return c

        cfg = mock.Mock(pool_size=4, configured=True, query_tag_prefix='ws', statement_timeout_seconds=60,
                        role='R', warehouse='W', database='D', schema='S', private_key_path='', password='p',
                        authenticator='', account='a', user='u')
        pool = connection.ConnectionPool(config=cfg, connector=Connector)
        return pool, made

    def test_transient_error_retried_on_fresh_session(self):
        pool, made = self._pool([Exception('Response ended prematurely'), None])
        with mock.patch.object(executor, 'get_pool', return_value=pool), \
                mock.patch.object(executor, 'default_context', return_value=connection.Context()), \
                mock.patch.object(executor, 'statement_tag', return_value='t'), \
                mock.patch.object(executor.time, 'sleep'):
            res = executor.run('select 1')
        self.assertEqual(res.rows, [[1]])
        self.assertEqual(len(made), 2, 'the broken session must be discarded and a new one opened')
        self.assertTrue(made[0].closed)

    def test_real_errors_not_retried(self):
        pool, made = self._pool([Exception("SQL compilation error: invalid identifier 'X'"), None])
        with mock.patch.object(executor, 'get_pool', return_value=pool), \
                mock.patch.object(executor, 'default_context', return_value=connection.Context()), \
                mock.patch.object(executor, 'statement_tag', return_value='t'):
            with self.assertRaises(executor.QueryError):
                executor.run('select x')
        self.assertEqual(len(made), 1)

    def test_gives_up_after_one_retry(self):
        pool, made = self._pool([Exception('Connection reset by peer'), Exception('Connection reset by peer')])
        with mock.patch.object(executor, 'get_pool', return_value=pool), \
                mock.patch.object(executor, 'default_context', return_value=connection.Context()), \
                mock.patch.object(executor, 'statement_tag', return_value='t'), \
                mock.patch.object(executor.time, 'sleep'):
            with self.assertRaises(executor.QueryError):
                executor.run('select 1')
        self.assertEqual(len(made), 2)

    def test_is_transient(self):
        self.assertTrue(executor.is_transient(Exception('Max retries exceeded with url')))
        self.assertFalse(executor.is_transient(Exception('Insufficient privileges')))
