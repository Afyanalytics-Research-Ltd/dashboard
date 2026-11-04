"""installer: template rendering, step parsing, apply/verify against a fake connection."""

from unittest import mock

from django.test import SimpleTestCase

from warehouse_studio.services import installer
from warehouse_studio.services.installer import InstallError, Step

VALUES = {'DB': 'hospitals', 'SCHEMA': 'Analytics', 'APP_ROLE': 'DATAANALYSTS', 'TASK_WAREHOUSE': 'COMPUTE_WH'}


class RenderTests(SimpleTestCase):
    def test_substitutes_upper_cased(self):
        out = installer.render('USE {{DB}}.{{SCHEMA}} TO {{APP_ROLE}} ON {{TASK_WAREHOUSE}}', VALUES)
        self.assertEqual(out, 'USE HOSPITALS.ANALYTICS TO DATAANALYSTS ON COMPUTE_WH')

    def test_rejects_injection(self):
        for key, bad in (('DB', 'X; DROP DATABASE Y'), ('SCHEMA', 'a.b'), ('APP_ROLE', '"r"'), ('TASK_WAREHOUSE', ''),
                         ('DB', '1abc')):
            with self.subTest(key=key, bad=bad), self.assertRaises(InstallError):
                installer.render('{{DB}}', {**VALUES, key: bad})

    def test_missing_value(self):
        with self.assertRaises(InstallError):
            installer.render('{{DB}}', {'DB': 'X'})

    def test_unknown_placeholder(self):
        with self.assertRaises(InstallError):
            installer.render('{{NOPE}}', VALUES)


class StepTests(SimpleTestCase):
    def test_install_steps(self):
        steps = installer.load_steps('install.sql', VALUES)
        self.assertEqual(len(steps), 82)
        self.assertEqual([s.index for s in steps], list(range(1, 83)))
        joined = '\n'.join(s.sql for s in steps)
        self.assertNotIn('{{', joined)
        self.assertEqual(steps[0].summary, 'USE ROLE ACCOUNTADMIN')
        procs = [s for s in steps if s.sql.startswith('CREATE OR REPLACE PROCEDURE')]
        self.assertEqual(len(procs), len(installer.EXPECTED_PROCEDURES))
        for p in procs:
            self.assertIn('EXECUTE AS OWNER', p.sql)
            self.assertTrue(p.sql.rstrip().endswith('$$'))
        grants = [s for s in steps if s.sql.startswith('GRANT USAGE ON PROCEDURE')]
        self.assertEqual(len(grants), len(installer.EXPECTED_PROCEDURES))
        # Every expected object is created by the script.
        for name in installer.EXPECTED_PROCEDURES + installer.EXPECTED_RELATIONS:
            self.assertIn(f'HOSPITALS.ANALYTICS.{name}', joined)
        # Nothing in the toolkit alters warehouses or tables outside its own task.
        self.assertNotIn('ALTER WAREHOUSE', joined.replace("'ALTER WAREHOUSE \"'", ''))

    def test_uninstall_steps(self):
        steps = installer.load_steps('uninstall.sql', VALUES)
        self.assertTrue(all(s.sql.startswith(('USE ROLE', 'DROP')) for s in steps))
        self.assertEqual(sum(s.sql.startswith('DROP TABLE') for s in steps), 3)

    def test_missing_file(self):
        with self.assertRaises(InstallError):
            installer.load_steps('nope.sql', VALUES)

    def test_summaries(self):
        cases = {
            'CREATE OR REPLACE PROCEDURE D.S.P(A NUMBER) RETURNS VARCHAR AS $$x$$': 'CREATE OR REPLACE PROCEDURE D.S.P',
            'CREATE TABLE IF NOT EXISTS D.S.T (A INT)': 'CREATE TABLE IF NOT EXISTS D.S.T',
            'GRANT USAGE ON PROCEDURE D.S.P(NUMBER, NUMBER) TO ROLE R': 'GRANT USAGE ON PROCEDURE D.S.P(NUMBER, NUMBER)',
            'GRANT SELECT ON VIEW D.S.V TO ROLE R': 'GRANT SELECT ON VIEW D.S.V',
            'DROP PROCEDURE IF EXISTS D.S.P(NUMBER)': 'DROP PROCEDURE IF EXISTS D.S.P(NUMBER)',
            'select ' + 'x' * 200: ('select ' + 'x' * 200)[:90],
        }
        for sql, summary in cases.items():
            with self.subTest(sql=sql[:40]):
                self.assertEqual(Step(1, sql).summary, summary)


class FakeCursor:
    def __init__(self, fail_on=None, proc_rows=(), rel_rows=()):
        self.fail_on, self.proc_rows, self.rel_rows = fail_on, proc_rows, rel_rows
        self.executed, self.description, self.closed, self._rows = [], None, False, []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if self.fail_on and self.fail_on in sql:
            err = Exception('raw')
            err.msg = 'Insufficient privileges'
            raise err
        self.description = [('status',)]
        if 'PROCEDURES' in sql:
            self._rows = list(self.proc_rows)
        elif 'TABLES' in sql:
            self._rows = list(self.rel_rows)
        else:
            self._rows = [('Statement executed successfully.',)]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def close(self):
        self.closed = True


class ApplyVerifyTests(SimpleTestCase):
    def test_apply_all(self):
        cur = FakeCursor()
        conn = mock.Mock(cursor=mock.Mock(return_value=cur))
        seen = []
        out = installer.apply(conn, [Step(1, 'select 1'), Step(2, 'select 2')], on_step=seen.append)
        self.assertEqual([o['ok'] for o in out], [True, True])
        self.assertEqual(out[0]['message'], 'Statement executed successfully.')
        self.assertEqual(seen, out)
        self.assertTrue(cur.closed)

    def test_apply_stops_at_first_failure(self):
        cur = FakeCursor(fail_on='bad')
        conn = mock.Mock(cursor=mock.Mock(return_value=cur))
        seen = []
        with self.assertRaises(InstallError) as cm:
            installer.apply(conn, [Step(1, 'select 1'), Step(2, 'bad one'), Step(3, 'select 3')], on_step=seen.append)
        self.assertIn('Step 2', str(cm.exception))
        self.assertIn('Insufficient privileges', str(cm.exception))
        self.assertEqual([o['ok'] for o in seen], [True, False])
        self.assertEqual(len(cur.executed), 2)
        self.assertTrue(cur.closed)

    def test_verify(self):
        cur = FakeCursor(proc_rows=[(p,) for p in installer.EXPECTED_PROCEDURES],
                         rel_rows=[(r,) for r in installer.EXPECTED_RELATIONS[:-1]])
        conn = mock.Mock(cursor=mock.Mock(return_value=cur))
        report = installer.verify(conn, 'HOSPITALS', 'ANALYTICS')
        self.assertFalse(report['installed'])
        self.assertEqual(report['missing'], [installer.EXPECTED_RELATIONS[-1]])
        self.assertTrue(all(report['procedures'].values()))
        self.assertEqual(cur.executed[0][1], ('ANALYTICS',))
        self.assertIn('"HOSPITALS".INFORMATION_SCHEMA.PROCEDURES', cur.executed[0][0])

    def test_verify_complete(self):
        cur = FakeCursor(proc_rows=[(p,) for p in installer.EXPECTED_PROCEDURES],
                         rel_rows=[(r,) for r in installer.EXPECTED_RELATIONS])
        report = installer.verify(mock.Mock(cursor=mock.Mock(return_value=cur)), 'HOSPITALS', 'ANALYTICS')
        self.assertTrue(report['installed'])
        self.assertEqual(report['missing'], [])
