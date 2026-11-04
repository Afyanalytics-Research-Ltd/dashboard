"""conf + connection pool."""

import os
import threading
from unittest import mock

from django.conf import settings
from django.test import SimpleTestCase, override_settings

from warehouse_studio import conf
from warehouse_studio.conf import StudioConfig, credits_per_hour, get_config
from warehouse_studio.services import connection as cn
from warehouse_studio.services.connection import (ConnectionPool, Context, StudioConnectionError, connect_kwargs,
                                                  default_context, get_pool, reset_pool)

CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith(('SNOWFLAKE_', 'WAREHOUSE_STUDIO_'))}


class ConfTests(SimpleTestCase):
    def test_env_fallback(self):
        env = {**CLEAN_ENV, 'SNOWFLAKE_ACCOUNT': 'acc', 'SNOWFLAKE_USER': 'u', 'SNOWFLAKE_PASSWORD': 'p',
               'SNOWFLAKE_ROLE': 'R', 'WAREHOUSE_STUDIO_CREDIT_PRICE_USD': '3.5', 'WAREHOUSE_STUDIO_ALLOW_WRITES': 'yes',
               'WAREHOUSE_STUDIO_POOL_SIZE': '0'}
        with mock.patch.dict(os.environ, env, clear=True), override_settings(WAREHOUSE_STUDIO={}):
            cfg = get_config()
        self.assertEqual((cfg.account, cfg.user, cfg.password, cfg.role), ('acc', 'u', 'p', 'R'))
        self.assertEqual(cfg.credit_price_usd, 3.5)
        self.assertTrue(cfg.allow_writes)
        self.assertEqual(cfg.pool_size, 1)  # clamped
        self.assertTrue(cfg.configured)
        self.assertEqual(cfg.toolkit, 'HOSPITALS.ANALYTICS')

    def test_studio_env_beats_platform_env_and_settings_beat_env(self):
        env = {**CLEAN_ENV, 'SNOWFLAKE_USER': 'platform', 'WAREHOUSE_STUDIO_SNOWFLAKE_USER': 'studio',
               'SNOWFLAKE_ROLE': 'ENVROLE'}
        with mock.patch.dict(os.environ, env, clear=True), override_settings(WAREHOUSE_STUDIO={'role': 'SETROLE'}):
            cfg = get_config()
        self.assertEqual(cfg.user, 'studio')
        self.assertEqual(cfg.role, 'SETROLE')

    def test_relative_key_path_made_absolute(self):
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True), \
                override_settings(WAREHOUSE_STUDIO={'private_key_path': 'keys/k.p8'}):
            cfg = get_config()
        self.assertTrue(os.path.isabs(cfg.private_key_path))
        self.assertTrue(cfg.private_key_path.endswith(os.path.join('keys', 'k.p8')))
        self.assertTrue(cfg.private_key_path.startswith(str(settings.BASE_DIR)))

    def test_bad_numbers_fall_back(self):
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True), \
                override_settings(WAREHOUSE_STUDIO={'credit_price_usd': 'abc', 'max_result_rows': 'x'}):
            cfg = get_config()
        self.assertEqual(cfg.credit_price_usd, 2.60)
        self.assertEqual(cfg.max_result_rows, 10_000)

    def test_invalid_toolkit_identifier(self):
        for bad in ({'toolkit_database': 'X; DROP'}, {'toolkit_schema': 'a.b'}):
            with self.subTest(bad=bad), override_settings(WAREHOUSE_STUDIO=bad):
                with self.assertRaises(ValueError):
                    get_config()

    def test_toolkit_uppercased(self):
        with override_settings(WAREHOUSE_STUDIO={'toolkit_database': 'mydb', 'toolkit_schema': 'tk'}):
            self.assertEqual(get_config().toolkit, 'MYDB.TK')

    def test_not_configured(self):
        self.assertFalse(StudioConfig().configured)
        self.assertTrue(StudioConfig(account='a', user='u', authenticator='externalbrowser').configured)

    def test_credits_per_hour(self):
        for size, cph in (('X-Small', 1), ('XSMALL', 1), ('Small', 2), ('Medium', 4), ('Large', 8), ('X-Large', 16),
                          ('2X-Large', 32), ('3X-Large', 64), ('4X-Large', 128), ('5X-Large', 256), ('6X-Large', 512),
                          ('x small', 1), (None, 0), ('', 0), ('Huge', 0)):
            with self.subTest(size=size):
                self.assertEqual(credits_per_hour(size), cph)

    def test_bool_helper(self):
        self.assertTrue(conf._bool('on'))
        self.assertFalse(conf._bool('nope'))
        self.assertTrue(conf._bool(None, True))


class ContextTests(SimpleTestCase):
    def test_defaults_and_overrides(self):
        base = Context(role='R', warehouse='W', database='D', schema='S')
        ctx = Context.from_dict({'warehouse': 'other', 'schema': None}, default=base)
        self.assertEqual(ctx.as_dict(), {'role': 'R', 'warehouse': 'other', 'database': 'D', 'schema': 'S'})
        self.assertEqual(Context.from_dict(None), Context())

    def test_strips_wrapping_quotes(self):
        self.assertEqual(Context.from_dict({'database': '"My Db"'}).database, 'My Db')

    def test_rejects_quotes_and_control_chars(self):
        for bad in ('a"b', 'x\ny', 'a\x00', 'z' * 300):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Context.from_dict({'schema': bad})

    def test_hashable_for_buckets(self):
        self.assertEqual(len({Context('a'), Context('a'), Context('b')}), 2)


class ConnectKwargsTests(SimpleTestCase):
    def cfg(self, **kw):
        return StudioConfig(account='acc', user='u', **kw)

    def test_password(self):
        kw = connect_kwargs(self.cfg(password='pw'), Context(role='DATAANALYSTS', warehouse='my wh', database='Mixed'))
        self.assertEqual(kw['password'], 'pw')
        self.assertEqual(kw['role'], 'DATAANALYSTS')
        self.assertEqual(kw['warehouse'], '"my wh"')
        self.assertEqual(kw['database'], 'Mixed')  # plain identifier passes bare
        self.assertNotIn('schema', kw)
        sp = kw['session_parameters']
        self.assertEqual(sp['QUERY_TAG'], 'warehouse-studio')
        self.assertEqual(sp['STATEMENT_TIMEOUT_IN_SECONDS'], 900)
        self.assertFalse(sp['ABORT_DETACHED_QUERY'])
        self.assertEqual(kw['application'], cn.APPLICATION_NAME)

    def test_key_pair_preferred(self):
        with mock.patch.object(cn, '_load_private_key', return_value=b'DER') as load:
            kw = connect_kwargs(self.cfg(password='pw', private_key_path='/k.p8', private_key_passphrase='pp'), Context())
        load.assert_called_once_with('/k.p8', 'pp')
        self.assertEqual(kw['private_key'], b'DER')
        self.assertNotIn('password', kw)

    def test_authenticator(self):
        kw = connect_kwargs(self.cfg(authenticator='externalbrowser'), Context())
        self.assertEqual(kw['authenticator'], 'externalbrowser')
        self.assertNotIn('password', kw)

    def test_query_tag_override(self):
        self.assertEqual(connect_kwargs(self.cfg(password='p'), Context(), query_tag='t')['session_parameters']['QUERY_TAG'], 't')

    def test_unconfigured(self):
        with self.assertRaises(StudioConnectionError):
            connect_kwargs(StudioConfig(), Context())

    def test_default_context(self):
        self.assertEqual(default_context(StudioConfig(role='R', warehouse='W')), Context('R', 'W', '', ''))


class FakeConn:
    def __init__(self, n):
        self.n, self.closed = n, False

    def is_closed(self):
        return self.closed

    def close(self):
        self.closed = True


class FakeConnector:
    def __init__(self, fail=False):
        self.made, self.fail, self.calls = [], fail, []

    def connect(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError('bad creds')
        c = FakeConn(len(self.made))
        self.made.append(c)
        return c


class OperationalError(Exception):
    pass


class PoolTests(SimpleTestCase):
    def make(self, size=3, fail=False):
        self.connector = FakeConnector(fail)
        return ConnectionPool(StudioConfig(account='a', user='u', password='p', pool_size=size), connector=self.connector)

    def test_reuse_per_context(self):
        pool = self.make()
        a = Context(warehouse='A')
        with pool.connection(a) as c1:
            pass
        with pool.connection(a) as c2:
            pass
        self.assertIs(c1, c2)
        with pool.connection(Context(warehouse='B')) as c3:
            pass
        self.assertIsNot(c1, c3)
        self.assertEqual(len(self.connector.made), 2)
        self.assertEqual(self.connector.calls[1]['warehouse'], 'B')

    def test_exclusive_checkout(self):
        pool = self.make()
        ctx = Context()
        c1, c2 = pool.acquire(ctx), pool.acquire(ctx)
        self.assertIsNot(c1, c2)
        self.assertEqual(pool.stats()['busy'], 2)
        pool.release(ctx, c1)
        pool.release(ctx, c2)
        self.assertEqual(pool.stats(), {'busy': 0, 'idle': 2, 'contexts': 1, 'size': 3})

    def test_cap_evicts_idle_from_other_context(self):
        pool = self.make(size=2)
        for name in ('A', 'B'):
            with pool.connection(Context(warehouse=name)):
                pass
        with pool.connection(Context(warehouse='C')):
            self.assertEqual(sum(c.closed for c in self.connector.made), 1)
            self.assertTrue(self.connector.made[0].closed)  # least recently used bucket
        self.assertEqual(pool.stats()['idle'], 2)

    def test_dead_connection_replaced(self):
        pool = self.make()
        with pool.connection() as c1:
            pass
        c1.closed = True
        with pool.connection() as c2:
            pass
        self.assertIsNot(c1, c2)

    def test_discard_on_connection_failure(self):
        pool = self.make()
        with self.assertRaises(OperationalError):
            with pool.connection() as c1:
                raise OperationalError('gone')
        self.assertTrue(c1.closed)
        self.assertEqual(pool.stats()['idle'], 0)
        with self.assertRaises(ValueError):
            with pool.connection() as c2:
                raise ValueError('ordinary error keeps the session')
        self.assertFalse(c2.closed)
        self.assertEqual(pool.stats()['idle'], 1)

    def test_message_based_failure_detection(self):
        self.assertTrue(cn._is_connection_failure(Exception('Session no longer exists. New login required')))
        self.assertTrue(cn._is_connection_failure(Exception('Authentication token has expired')))
        self.assertFalse(cn._is_connection_failure(Exception('syntax error')))

    def test_timeout_when_exhausted(self):
        pool = self.make(size=1)
        held = pool.acquire(Context())
        with self.assertRaises(StudioConnectionError):
            pool.acquire(Context(), timeout=0.05)
        pool.release(Context(), held)

    def test_waiter_wakes_on_release(self):
        pool = self.make(size=1)
        held = pool.acquire(Context())
        got = []
        t = threading.Thread(target=lambda: got.append(pool.acquire(Context(), timeout=5)))
        t.start()
        pool.release(Context(), held)
        t.join(5)
        self.assertEqual(got, [held])

    def test_connect_failure_frees_slot(self):
        pool = self.make(size=1, fail=True)
        with self.assertRaises(StudioConnectionError) as cm:
            pool.acquire(Context())
        self.assertIn('bad creds', str(cm.exception))
        self.assertEqual(pool.stats()['busy'], 0)

    def test_close_all(self):
        pool = self.make()
        with pool.connection():
            pass
        pool.close_all()
        self.assertTrue(self.connector.made[0].closed)
        self.assertEqual(pool.stats()['idle'], 0)

    def test_reset_pool(self):
        fake = self.make()
        try:
            reset_pool(fake)
            self.assertIs(get_pool(), fake)
            with fake.connection():
                pass
            reset_pool(None)
            self.assertTrue(self.connector.made[0].closed)
            self.assertIsInstance(get_pool(), ConnectionPool)
            self.assertIsNot(get_pool(), fake)
        finally:
            reset_pool(None)

    def test_clean_error(self):
        e = Exception('first line\nsecond')
        self.assertEqual(cn._clean_error(e), 'first line')
