"""catalog: identifier quoting, signatures, argument binding, listings."""

from unittest import mock

from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from warehouse_studio.services import catalog, executor
from warehouse_studio.services.catalog import CatalogError

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache', 'LOCATION': 'ws-cat-tests'}}


class QuoteTests(SimpleTestCase):
    def test_quote(self):
        self.assertEqual(catalog.quote('HOSPITALS'), '"HOSPITALS"')
        self.assertEqual(catalog.quote('my "db"'), '"my ""db"""')

    def test_quote_rejects(self):
        for bad in ('', None, 'x\ny', 'a' * 256):
            with self.subTest(bad=bad), self.assertRaises(CatalogError):
                catalog.quote(bad)


class SignatureTests(SimpleTestCase):
    def test_parse(self):
        self.assertEqual(catalog.parse_signature('(DB_NAME VARCHAR)'),
                         [{'name': 'DB_NAME', 'type': 'VARCHAR', 'has_default': False}])
        self.assertEqual(catalog.parse_signature('(A NUMBER(38,0), B FLOAT, "C" TIMESTAMP_LTZ(9))'),
                         [{'name': 'A', 'type': 'NUMBER(38,0)', 'has_default': False},
                          {'name': 'B', 'type': 'FLOAT', 'has_default': False},
                          {'name': 'C', 'type': 'TIMESTAMP_LTZ(9)', 'has_default': False}])

    def test_defaults_and_bare_types(self):
        self.assertEqual(catalog.parse_signature('(VARCHAR, DEFAULT NUMBER)'),
                         [{'name': 'ARG1', 'type': 'VARCHAR', 'has_default': False},
                          {'name': 'ARG2', 'type': 'NUMBER', 'has_default': True}])
        self.assertEqual(catalog.parse_signature('(P_DAYS NUMBER DEFAULT 7)')[0]['type'], 'NUMBER DEFAULT 7')

    def test_empty(self):
        for sig in (None, '', '()', '(  )'):
            self.assertEqual(catalog.parse_signature(sig), [])


class BindTests(SimpleTestCase):
    def test_numbers(self):
        self.assertEqual(catalog.bind_value('NUMBER(38,0)', '42'), (42, '%s'))
        self.assertEqual(catalog.bind_value('NUMBER', '-4.5'), (-4.5, '%s'))
        self.assertEqual(catalog.bind_value('FLOAT', '1e3'), (1000.0, '%s'))
        for t, bad in (('NUMBER', '1; drop'), ('INT', 'abc'), ('FLOAT', 'x')):
            with self.subTest(t=t), self.assertRaises(CatalogError):
                catalog.bind_value(t, bad)

    def test_boolean(self):
        self.assertEqual(catalog.bind_value('BOOLEAN', 'true')[0], True)
        self.assertEqual(catalog.bind_value('BOOLEAN', 'no')[0], False)

    def test_json(self):
        self.assertEqual(catalog.bind_value('ARRAY', '["a"]'), ('["a"]', 'PARSE_JSON(%s)::ARRAY'))
        self.assertEqual(catalog.bind_value('OBJECT', {'a': 1}), ('{"a": 1}', 'PARSE_JSON(%s)::OBJECT'))
        self.assertEqual(catalog.bind_value('VARIANT', '1')[1], 'PARSE_JSON(%s)::VARIANT')
        with self.assertRaises(CatalogError):
            catalog.bind_value('ARRAY', '[oops')

    def test_text_and_empty(self):
        self.assertEqual(catalog.bind_value('VARCHAR', "it's"), ("it's", '%s'))
        self.assertEqual(catalog.bind_value('NUMBER', ''), (None, '%s'))
        self.assertEqual(catalog.bind_value(None, 'x'), ('x', '%s'))

    def test_build_call(self):
        params = catalog.parse_signature('(A NUMBER, B ARRAY, DEFAULT C VARCHAR)')
        sql, binds = catalog.build_procedure_call('HOSPITALS', 'my schema', 'SP', params, ['5', '[1]'])
        self.assertEqual(sql, 'CALL "HOSPITALS"."my schema"."SP"(%s, PARSE_JSON(%s)::ARRAY)')
        self.assertEqual(binds, (5, '[1]'))
        sql, binds = catalog.build_procedure_call('D', 'S', 'P', params, ['5', '[1]', 'x'])
        self.assertEqual(binds, (5, '[1]', 'x'))

    def test_build_call_required_missing_is_null(self):
        params = catalog.parse_signature('(A NUMBER, B VARCHAR)')
        sql, binds = catalog.build_procedure_call('D', 'S', 'P', params, [])
        self.assertEqual((sql, binds), ('CALL "D"."S"."P"(%s, %s)', (None, None)))

    def test_build_call_injection_in_names_is_quoted(self):
        sql, _ = catalog.build_procedure_call('D', 'S', 'P"(); drop table x; --', [], [])
        self.assertEqual(sql, 'CALL "D"."S"."P""(); drop table x; --"()')


@override_settings(CACHES=LOCMEM)
class ListingTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        p = mock.patch.object(executor, 'records')
        self.rec = p.start()
        self.addCleanup(p.stop)

    def test_databases_cached_and_refresh(self):
        self.rec.return_value = [{'name': 'HOSPITALS', 'kind': None, 'owner': 'O', 'comment': '', 'created_on': None,
                                  'origin': ''}]
        out = catalog.databases()
        self.assertEqual(out[0]['kind'], 'STANDARD')
        catalog.databases()
        self.assertEqual(self.rec.call_count, 1)
        catalog.databases(refresh=True)
        self.assertEqual(self.rec.call_count, 2)
        self.assertEqual(self.rec.call_args[0][0], 'SHOW DATABASES')

    def test_schemas_hide_information_schema(self):
        self.rec.return_value = [{'name': 'INFORMATION_SCHEMA'}, {'name': 'ANALYTICS', 'owner': 'O'}]
        self.assertEqual([s['name'] for s in catalog.schemas('HOSPITALS')], ['ANALYTICS'])
        self.assertEqual(self.rec.call_args[0][0], 'SHOW SCHEMAS IN DATABASE "HOSPITALS"')

    def test_objects(self):
        def fake(sql, params=None, **kw):
            if 'INFORMATION_SCHEMA.TABLES' in sql:
                return [{'name': 'T', 'table_type': 'BASE TABLE', 'row_count': 5, 'bytes': 10, 'last_altered': None,
                         'comment': None, 'clustering_key': None, 'is_transient': 'YES', 'retention_time': 1},
                        {'name': 'V', 'table_type': 'VIEW', 'row_count': None, 'bytes': None, 'last_altered': None,
                         'comment': None, 'clustering_key': None, 'is_transient': 'NO', 'retention_time': 1},
                        {'name': 'DT', 'table_type': 'DYNAMIC TABLE', 'row_count': 1, 'bytes': 1, 'last_altered': None,
                         'comment': None, 'clustering_key': None, 'is_transient': 'NO', 'retention_time': 1}]
            if 'PROCEDURES' in sql:
                return [{'name': 'SP', 'argument_signature': '(A NUMBER)', 'data_type': 'VARCHAR', 'language': 'SQL',
                         'comment': None}]
            if 'FUNCTIONS' in sql:
                raise executor.QueryError('no privilege')
            if sql.startswith('SHOW TASKS'):
                return [{'name': 'TSK', 'state': 'suspended', 'schedule': 'cron'}]
            return []
        self.rec.side_effect = fake
        out = catalog.objects('HOSPITALS', 'ANALYTICS')
        self.assertEqual(out['TABLE'][0]['transient'], True)
        self.assertEqual(out['VIEW'][0]['name'], 'V')
        self.assertEqual(out['DYNAMIC_TABLE'][0]['name'], 'DT')
        self.assertEqual(out['PROCEDURE'][0]['params'], [{'name': 'A', 'type': 'NUMBER', 'has_default': False}])
        self.assertEqual(out['FUNCTION'], [])
        self.assertEqual(out['TASK'][0]['state'], 'suspended')
        self.assertNotIn('STAGE', out)
        tables_call = [c for c in self.rec.call_args_list if 'TABLES' in c[0][0]][0]
        self.assertEqual(tables_call[0][1], ('ANALYTICS',))
        self.assertIn('"HOSPITALS".INFORMATION_SCHEMA.TABLES', tables_call[0][0])

    def test_columns(self):
        self.rec.return_value = [{'table_name': 'T', 'column_name': 'A'}]
        catalog.columns('D', 'S', 'T')
        sql, params = self.rec.call_args[0][:2]
        self.assertEqual(params, ('S', 'T'))
        self.assertNotIn('LIMIT', sql)
        catalog.columns('D', 'S')
        sql, params = self.rec.call_args[0][:2]
        self.assertEqual(params, ('S',))
        self.assertIn('LIMIT 20000', sql)

    def test_ddl(self):
        with mock.patch.object(executor, 'run') as run:
            run.return_value = mock.Mock(rows=[['create table t (a int);']])
            self.assertEqual(catalog.ddl('table', '"D"."S"."T"'), 'create table t (a int);')
            self.assertEqual(run.call_args[0], ('SELECT GET_DDL(%s, %s, TRUE)', ('TABLE', '"D"."S"."T"')))
            self.assertEqual(catalog.ddl('file format', 'X'), 'create table t (a int);')
            self.assertEqual(run.call_args[0][1][0], 'FILE_FORMAT')
            run.return_value = mock.Mock(rows=[])
            self.assertEqual(catalog.ddl('VIEW', 'v'), '')
        for t, name in (('USER', 'x'), ('ROLE', 'r'), ('TABLE', ''), ('TABLE', 'a\x00b')):
            with self.subTest(t=t, name=name), self.assertRaises(CatalogError):
                catalog.ddl(t, name)
