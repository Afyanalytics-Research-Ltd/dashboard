"""SPA shell view, URL routing, and the warehouse_studio_install management command."""

import io
import os
import tempfile
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.management import CommandError, call_command
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import resolve, reverse

from warehouse_studio import api, views
from warehouse_studio.management.commands import warehouse_studio_install as cmd
from warehouse_studio.services import installer

User = get_user_model()
SETTINGS = {'toolkit_database': 'HOSPITALS', 'toolkit_schema': 'ANALYTICS', 'credit_price_usd': 2.5,
            'allow_writes': False, 'account': 'acc', 'user': 'u', 'password': 'p'}


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class ShellViewTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user('admin', 'a@t.io', 'pw-123456', is_superuser=True, is_staff=True)
        self.plain = User.objects.create_user('plain', 'p@t.io', 'pw-123456')

    def test_superuser_gets_shell(self):
        self.client.force_login(self.admin)
        resp = self.client.get('/warehouse-studio/')
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, 'warehouse_studio/app.html')
        cfg = resp.context['studio_config']
        self.assertEqual(cfg['root'], '/warehouse-studio/')
        self.assertEqual(cfg['api'], '/warehouse-studio/api/')
        self.assertEqual((cfg['user'], cfg['toolkit'], cfg['creditPrice'], cfg['allowWrites'], cfg['configured']),
                         ('admin', 'HOSPITALS.ANALYTICS', 2.5, False, True))
        self.assertEqual(resp.context['sidebar_section'], 'warehouse_studio')
        self.assertContains(resp, 'type="application/json"')
        self.assertContains(resp, '/warehouse-studio/api/')

    def test_deep_links_render_shell(self):
        self.client.force_login(self.admin)
        for path in ('/warehouse-studio/optimizer', '/warehouse-studio/worksheets/abc', '/warehouse-studio/cost/storage',
                     '/warehouse-studio/query/01c82df1-0005-ac42-0001-425a016cd052'):
            with self.subTest(path=path):
                resp = self.client.get(path)
                self.assertEqual(resp.status_code, 200)
                self.assertTemplateUsed(resp, 'warehouse_studio/app.html')

    def test_anonymous_redirected_to_login(self):
        resp = self.client.get('/warehouse-studio/optimizer')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('next=/warehouse-studio/optimizer', resp['Location'])

    def test_non_superuser_forbidden(self):
        self.client.force_login(self.plain)
        self.assertEqual(self.client.get('/warehouse-studio/').status_code, 403)


class RoutingTests(SimpleTestCase):
    def test_api_not_swallowed_by_spa(self):
        self.assertIs(resolve('/warehouse-studio/api/').func, api.session_view)
        self.assertIs(resolve('/warehouse-studio/api/optimizer/findings/').func, api.optimizer_findings)
        self.assertIs(resolve('/warehouse-studio/api/query/abc/status/').func, api.query_status)
        match = resolve('/warehouse-studio/cost')
        self.assertEqual(match.url_name, 'app')
        self.assertEqual(match.func.view_class, views.StudioAppView)

    def test_unknown_api_path_is_404_not_shell(self):
        from django.urls import Resolver404
        with self.assertRaises(Resolver404):
            resolve('/warehouse-studio/api/does-not-exist/')

    def test_reverse(self):
        self.assertEqual(reverse('warehouse_studio:home'), '/warehouse-studio/')
        self.assertEqual(reverse('warehouse_studio:api_finding_triage', kwargs={'finding_id': 3}),
                         '/warehouse-studio/api/optimizer/findings/3/triage/')


class EnvFileTests(SimpleTestCase):
    def test_read_env_file(self):
        with tempfile.NamedTemporaryFile('w', suffix='.env', delete=False) as fh:
            fh.write('# comment\n\nexport SNOWFLAKE_USER="ADMIN"\nSNOWFLAKE_ROLE=\'ACCOUNTADMIN\'\nBROKEN LINE\n'
                     'SNOWFLAKE_PASSWORD=a=b\n')
        self.addCleanup(os.unlink, fh.name)
        self.assertEqual(cmd._read_env_file(fh.name),
                         {'SNOWFLAKE_USER': 'ADMIN', 'SNOWFLAKE_ROLE': 'ACCOUNTADMIN', 'SNOWFLAKE_PASSWORD': 'a=b'})


class FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@override_settings(WAREHOUSE_STUDIO=SETTINGS)
class InstallCommandTests(SimpleTestCase):
    def run_cmd(self, *args):
        out = io.StringIO()
        call_command('warehouse_studio_install', *args, stdout=out)
        return out.getvalue()

    def env_file(self, **values):
        with tempfile.NamedTemporaryFile('w', suffix='.env', delete=False) as fh:
            for k, v in values.items():
                fh.write(f'SNOWFLAKE_{k}={v}\n')
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_dry_run(self):
        out = self.run_cmd('--dry-run', '--app-role', 'DATAANALYSTS', '--task-warehouse', 'COMPUTE_WH')
        self.assertEqual(out.count('-- ['), 82)
        self.assertIn('82 statements rendered (dry run).', out)
        self.assertRegex(out, r'GRANT USAGE ON PROCEDURE HOSPITALS\.ANALYTICS\.SP_RUN_ALL_DETECTORS\(NUMBER\)\s+TO ROLE DATAANALYSTS;')

    def test_dry_run_other_target(self):
        out = self.run_cmd('--dry-run', '--database', 'mydb', '--schema', 'tk', '--app-role', 'r', '--task-warehouse', 'w')
        self.assertIn('CREATE TABLE IF NOT EXISTS MYDB.TK.TUNING_FINDINGS', out)
        self.assertNotIn('HOSPITALS.ANALYTICS', out)

    def test_bad_identifier(self):
        with self.assertRaises(CommandError):
            self.run_cmd('--dry-run', '--database', 'X; DROP DATABASE Y', '--app-role', 'r', '--task-warehouse', 'w')

    def test_uninstall_keeps_tables_unless_purge(self):
        base = ('--uninstall', '--dry-run', '--app-role', 'r', '--task-warehouse', 'w')
        out = self.run_cmd(*base)
        self.assertNotIn('DROP TABLE', out)
        self.assertIn('DROP VIEW IF EXISTS HOSPITALS.ANALYTICS.V_TUNING_FINDINGS', out)
        self.assertIn('DROP TASK IF EXISTS', out)
        purged = self.run_cmd(*base, '--purge')
        sql_lines = [line for line in purged.splitlines() if line.startswith('DROP TABLE IF EXISTS')]
        self.assertEqual(len(sql_lines), 3)

    def test_missing_credentials(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('SNOWFLAKE_ADMIN_')}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch('snowflake.connector.connect') as connect:
            with self.assertRaises(CommandError):
                self.run_cmd('--verify', '--app-role', 'r', '--task-warehouse', 'w')
            with self.assertRaises(CommandError):
                self.run_cmd('--verify', '--app-role', 'r', '--task-warehouse', 'w',
                             '--env-file', self.env_file(ACCOUNT='a', USER='u'))
        connect.assert_not_called()

    def test_verify(self):
        conn = FakeConn()
        report = {'procedures': {'SP_A': True}, 'relations': {'T': False}, 'installed': False, 'missing': ['T']}
        with mock.patch('snowflake.connector.connect', return_value=conn) as connect, \
                mock.patch.object(installer, 'verify', return_value=report) as verify:
            out = self.run_cmd('--verify', '--app-role', 'r', '--task-warehouse', 'w',
                               '--env-file', self.env_file(ACCOUNT='acc', USER='admin', PASSWORD='pw', WAREHOUSE='WH'))
        kw = connect.call_args[1]
        self.assertEqual((kw['account'], kw['user'], kw['password'], kw['role'], kw['warehouse']),
                         ('acc', 'admin', 'pw', 'ACCOUNTADMIN', 'WH'))
        self.assertEqual(verify.call_args[0][1:], ('HOSPITALS', 'ANALYTICS'))
        self.assertIn('Missing: T', out)
        self.assertTrue(conn.closed)

    def test_install_with_key_pair(self):
        conn = FakeConn()
        ok = {'installed': True, 'missing': [], 'procedures': {}, 'relations': {}}

        def fake_apply(c, steps, on_step=None):
            on_step({'step': 1, 'summary': 'USE ROLE ACCOUNTADMIN', 'ok': True})
            return []
        with mock.patch('snowflake.connector.connect', return_value=conn) as connect, \
                mock.patch.object(cmd, '_load_private_key', return_value=b'DER') as load, \
                mock.patch.object(installer, 'apply', side_effect=fake_apply) as apply, \
                mock.patch.object(installer, 'verify', return_value=ok):
            out = self.run_cmd('--app-role', 'DATAANALYSTS', '--task-warehouse', 'COMPUTE_WH',
                               '--env-file', self.env_file(ACCOUNT='acc', USER='admin', PRIVATE_KEY_PATH='/k.p8',
                                                           ROLE='SYSADMIN'))
        self.assertEqual(connect.call_args[1]['private_key'], b'DER')
        self.assertEqual(connect.call_args[1]['role'], 'SYSADMIN')
        self.assertNotIn('password', connect.call_args[1])
        load.assert_called_once_with('/k.p8', '')
        self.assertEqual(len(apply.call_args[0][1]), 82)
        self.assertIn('Installed the toolkit in HOSPITALS.ANALYTICS (82 statements).', out)
        self.assertTrue(conn.closed)

    def test_install_reports_missing_after_apply(self):
        with mock.patch('snowflake.connector.connect', return_value=FakeConn()), \
                mock.patch.object(installer, 'apply', return_value=[]), \
                mock.patch.object(installer, 'verify', return_value={'installed': False, 'missing': ['SP_X']}):
            with self.assertRaisesMessage(CommandError, 'SP_X'):
                self.run_cmd('--app-role', 'r', '--task-warehouse', 'w',
                             '--env-file', self.env_file(ACCOUNT='a', USER='u', PASSWORD='p'))

    def test_install_step_failure(self):
        with mock.patch('snowflake.connector.connect', return_value=FakeConn()), \
                mock.patch.object(installer, 'apply', side_effect=installer.InstallError('Step 4 failed')):
            with self.assertRaisesMessage(CommandError, 'Step 4 failed'):
                self.run_cmd('--app-role', 'r', '--task-warehouse', 'w',
                             '--env-file', self.env_file(ACCOUNT='a', USER='u', PASSWORD='p'))

    def test_connect_failure(self):
        with mock.patch('snowflake.connector.connect', side_effect=RuntimeError('Incorrect username or password')):
            with self.assertRaisesMessage(CommandError, 'Incorrect username or password'):
                self.run_cmd('--verify', '--app-role', 'r', '--task-warehouse', 'w',
                             '--env-file', self.env_file(ACCOUNT='a', USER='u', PASSWORD='p'))

    def test_env_file_never_mixes_with_ambient_admin_env(self):
        env = {**os.environ, 'SNOWFLAKE_ADMIN_PRIVATE_KEY_PATH': '/ambient/key.p8'}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch('snowflake.connector.connect', return_value=FakeConn()) as connect, \
                mock.patch.object(cmd, '_load_private_key') as load, \
                mock.patch.object(installer, 'verify', return_value={'installed': True, 'missing': [],
                                                                     'procedures': {}, 'relations': {}}):
            self.run_cmd('--verify', '--app-role', 'r', '--task-warehouse', 'w',
                         '--env-file', self.env_file(ACCOUNT='a', USER='u', PASSWORD='p'))
        load.assert_not_called()
        self.assertEqual(connect.call_args[1]['password'], 'p')

    def test_admin_env_vars(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('SNOWFLAKE_ADMIN_')}
        env.update({'SNOWFLAKE_ADMIN_ACCOUNT': 'envacc', 'SNOWFLAKE_ADMIN_USER': 'envuser',
                    'SNOWFLAKE_ADMIN_PASSWORD': 'envpw'})
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch('snowflake.connector.connect', return_value=FakeConn()) as connect, \
                mock.patch.object(installer, 'verify', return_value={'installed': True, 'missing': [],
                                                                     'procedures': {}, 'relations': {}}):
            out = self.run_cmd('--verify', '--app-role', 'r', '--task-warehouse', 'w')
        self.assertEqual(connect.call_args[1]['user'], 'envuser')
        self.assertIn('Toolkit installed.', out)
