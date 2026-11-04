"""sql_guard: lexer, statement splitter, classifier and policy."""

from django.test import SimpleTestCase

from warehouse_studio.services import installer
from warehouse_studio.services.sql_guard import (READ, WRITE, SqlGuardError, check, classify, error_position,
                                                 qualify_procedure, split_statements, statement_at, tokenize)

VALUES = {'DB': 'HOSPITALS', 'SCHEMA': 'ANALYTICS', 'APP_ROLE': 'DATAANALYSTS', 'TASK_WAREHOUSE': 'COMPUTE_WH'}


def kinds(sql):
    return [(t.kind, t.text) for t in tokenize(sql) if t.kind != 'ws']


def texts(sql):
    return [s.text for s in split_statements(sql)]


def verdict(sql):
    return classify(split_statements(sql)[0])


class TokenizerTests(SimpleTestCase):
    def test_single_quoted_with_doubled_quote(self):
        self.assertEqual(kinds("'it''s; ok'"), [('string', "'it''s; ok'")])

    def test_backslash_escape(self):
        self.assertEqual(kinds(r"'a\'b' x"), [('string', r"'a\'b'"), ('word', 'x')])

    def test_quoted_identifier_with_doubled_quotes(self):
        self.assertEqual(kinds('"my ""col"""'), [('ident', '"my ""col"""')])

    def test_dollar_body(self):
        toks = kinds('$$ select 1; drop table x; $$ ;')
        self.assertEqual(toks[0][0], 'dollar')
        self.assertIn('drop table', toks[0][1])
        self.assertEqual(toks[1], ('punct', ';'))

    def test_comments(self):
        toks = kinds('-- a ; b\n// c ; d\n/* e ; f */ select')
        self.assertEqual([k for k, _ in toks], ['comment', 'comment', 'comment', 'word'])

    def test_unterminated_string_runs_to_end(self):
        toks = tokenize("select 'abc; def")
        self.assertEqual(toks[-1].kind, 'string')
        self.assertEqual(toks[-1].end, len("select 'abc; def"))

    def test_unterminated_comment_and_ident(self):
        self.assertEqual(tokenize('/* never closed')[0].kind, 'comment')
        self.assertEqual(tokenize('"never closed')[0].kind, 'ident')
        self.assertEqual(tokenize('$$ never closed')[0].kind, 'dollar')

    def test_numbers_words_punct(self):
        self.assertEqual(kinds('a1 3.5e2 .5 ;'), [('word', 'a1'), ('number', '3.5e2'), ('number', '.5'), ('punct', ';')])

    def test_offsets_cover_input(self):
        sql = "select 'x' -- c\n, \"y\" from t;"
        toks = tokenize(sql)
        self.assertEqual(''.join(t.text for t in toks), sql)


class SplitTests(SimpleTestCase):
    def test_multiple_and_empty(self):
        self.assertEqual(texts('select 1; select 2;;  ; select 3'), ['select 1', 'select 2', 'select 3'])

    def test_comment_only_pieces_dropped(self):
        self.assertEqual(texts('-- just a comment;\n/* x */ ; select 1;'), ['select 1'])

    def test_semicolons_in_strings_comments_dollar(self):
        sql = "select ';' as a /* ; */ -- ;\n; select $$;$$"
        self.assertEqual(len(split_statements(sql)), 2)

    def test_offsets(self):
        sql = '  select 1 ;\n  select 2'
        st = split_statements(sql)
        self.assertEqual(sql[st[1].start:st[1].end], 'select 2')

    def test_begin_end_block_nested(self):
        sql = """BEGIN
  IF (1 = 1) THEN
    LET x := 1;
  END IF;
  FOR i IN 1 TO 3 DO
    LET y := CASE WHEN i = 1 THEN 1 ELSE 2 END;
  END FOR;
  LOOP
    BREAK;
  END LOOP;
  CREATE TABLE IF NOT EXISTS t (a int);
  RETURN 1;
END;
select 2"""
        st = split_statements(sql)
        self.assertEqual(len(st), 2, [s.text for s in st])
        self.assertTrue(st[0].text.startswith('BEGIN'))
        self.assertTrue(st[0].text.endswith('END'))
        self.assertEqual(st[1].text, 'select 2')

    def test_declare_block(self):
        sql = 'DECLARE v NUMBER DEFAULT 1; BEGIN v := v + 1; RETURN v; END; select 1'
        self.assertEqual(len(split_statements(sql)), 2)

    def test_begin_transaction_is_not_block(self):
        self.assertEqual(texts('BEGIN; insert into t values (1); COMMIT;'), ['BEGIN', 'insert into t values (1)', 'COMMIT'])
        self.assertEqual(len(split_statements('BEGIN TRANSACTION; select 1; COMMIT')), 3)
        self.assertEqual(len(split_statements('BEGIN WORK; select 1')), 2)

    def test_create_procedure_dollar_body_is_one_statement(self):
        sql = ("CREATE OR REPLACE PROCEDURE p() RETURNS VARCHAR LANGUAGE SQL AS $$ BEGIN "
               "INSERT INTO t VALUES (1); RETURN 'x'; END; $$; select 1")
        st = split_statements(sql)
        self.assertEqual(len(st), 2)
        self.assertTrue(st[0].text.endswith('$$'))

    def test_install_sql_renders_82_statements(self):
        steps = installer.load_steps('install.sql', VALUES)
        self.assertEqual(len(steps), 82)
        self.assertTrue(all(s.sql.strip() for s in steps))

    def test_statement_at(self):
        sql = 'select 1;\nselect 2;\nselect 3'
        self.assertEqual(statement_at(sql, 0).text, 'select 1')
        self.assertEqual(statement_at(sql, sql.index('2')).text, 'select 2')
        self.assertEqual(statement_at(sql, len(sql)).text, 'select 3')
        self.assertIsNone(statement_at('-- nothing', 3))

    def test_keyword_property(self):
        self.assertEqual(split_statements('/* c */ select 1')[0].keyword, 'SELECT')


class ClassifyTests(SimpleTestCase):
    def test_reads(self):
        for sql in ('select 1', 'SHOW TABLES', 'desc table t', 'describe table t', 'explain select 1',
                    'list @s', 'ls @s', 'values (1)', 'with a as (select 1) select * from a',
                    '(select 1) union (select 2)'):
            with self.subTest(sql=sql):
                self.assertEqual(verdict(sql).kind, 'read')

    def test_with_insert_is_write(self):
        self.assertEqual(verdict('with a as (select 1) insert into t select * from a').kind, 'write')

    def test_with_nested_dml_word_in_parens_is_read(self):
        self.assertEqual(verdict('with a as (select "update" from t) select * from a').kind, 'read')

    def test_call_names(self):
        self.assertEqual(verdict('call p()').procedure, 'P')
        self.assertEqual(verdict('call s.p(1)').procedure, 'S.P')
        self.assertEqual(verdict('CALL db.s.p(1)').procedure, 'DB.S.P')
        self.assertEqual(verdict('call "My Db"."s".p()').procedure, 'My Db.s.P')

    def test_session(self):
        for sql in ('use warehouse x', 'set x = 1', 'unset x', 'alter session set query_tag = 1'):
            with self.subTest(sql=sql):
                self.assertEqual(verdict(sql).kind, 'session')

    def test_admin(self):
        for sql in ('grant select on t to role r', 'revoke select on t from role r', 'create user bob',
                    'alter user bob set password = 1', 'drop role r', 'create or replace role r',
                    'create api integration x', 'create storage integration x', 'create security integration x',
                    'alter account set x = 1', 'create network policy p'):
            with self.subTest(sql=sql):
                self.assertEqual(verdict(sql).kind, 'admin')

    def test_writes(self):
        for sql in ('create table t (a int)', 'create or replace temporary table t as select 1', 'drop table t',
                    'alter warehouse w set auto_suspend = 60', 'insert into t values (1)', 'update t set a = 1',
                    'delete from t', 'merge into t using s on 1=1 when matched then delete', 'truncate table t',
                    'copy into t from @s', 'undrop table t', 'commit', 'execute immediate $$select 1$$',
                    'BEGIN select 1; END'):
            with self.subTest(sql=sql):
                self.assertEqual(verdict(sql).kind, 'write')

    def test_stateful_system_functions(self):
        self.assertEqual(verdict('select system$cancel_all_queries(1)').kind, 'write')
        self.assertEqual(verdict('select SYSTEM$ABORT_SESSION(1)').kind, 'write')
        self.assertEqual(verdict("select system$clustering_information('t')").kind, 'read')
        self.assertEqual(verdict("select system$explain_plan_json('select 1')").kind, 'read')

    def test_dml_words_as_data_do_not_block(self):
        for sql in ('select update, delete, "CREATE" from t', "select 'drop table x' as s",
                    'select 1 -- delete everything\n', 'select /* insert */ 1'):
            with self.subTest(sql=sql):
                self.assertEqual(verdict(sql).kind, 'read')

    def test_unknown_keyword_is_write(self):
        self.assertEqual(verdict('frobnicate t').kind, 'write')

    def test_parse_errors(self):
        with self.assertRaises(SqlGuardError):
            verdict('call ()')
        with self.assertRaises(SqlGuardError):
            verdict('; 1')  # punct then number: no keyword

    def test_qualify_procedure(self):
        self.assertEqual(qualify_procedure('p', 'db', 's'), 'DB.S.P')
        self.assertEqual(qualify_procedure('x.p', 'db', 's'), 'DB.X.P')
        self.assertEqual(qualify_procedure('a.b.c', 'db', 's'), 'A.B.C')


class CheckPolicyTests(SimpleTestCase):
    ALLOW = frozenset({'HOSPITALS.ANALYTICS.SP_OK'})

    def test_read_ok(self):
        (st, v), = check('select 1;')
        self.assertEqual((st.text, v.kind), ('select 1', 'read'))

    def test_empty(self):
        for sql in ('', '   ', '-- only comment'):
            with self.assertRaises(SqlGuardError) as cm:
                check(sql)
            self.assertEqual(cm.exception.code, 'empty')

    def test_multiple(self):
        with self.assertRaises(SqlGuardError) as cm:
            check('select 1; select 2')
        self.assertEqual(cm.exception.code, 'multiple')
        self.assertEqual(len(check('select 1; select 2', single=False)), 2)

    def test_write_matrix(self):
        sql = 'insert into t values (1)'
        for mode, allow in ((READ, False), (READ, True), (WRITE, False)):
            with self.subTest(mode=mode, allow=allow):
                with self.assertRaises(SqlGuardError) as cm:
                    check(sql, mode=mode, allow_writes=allow)
                self.assertEqual(cm.exception.code, 'write')
        self.assertEqual(check(sql, mode=WRITE, allow_writes=True)[0][1].kind, 'write')

    def test_write_hint_text(self):
        with self.assertRaises(SqlGuardError) as cm:
            check('drop table t', allow_writes=True)
        self.assertIn('Unlock write mode', str(cm.exception))
        with self.assertRaises(SqlGuardError) as cm:
            check('drop table t', allow_writes=False)
        self.assertIn('read-only', str(cm.exception))

    def test_admin_never(self):
        with self.assertRaises(SqlGuardError) as cm:
            check('create user x', mode=WRITE, allow_writes=True)
        self.assertEqual(cm.exception.code, 'admin')

    def test_session_never(self):
        with self.assertRaises(SqlGuardError) as cm:
            check('use role accountadmin', mode=WRITE, allow_writes=True)
        self.assertEqual(cm.exception.code, 'session')

    def test_call_allow_list(self):
        (_, v), = check('call sp_ok(1)', callable_procedures=self.ALLOW, database='hospitals', schema='analytics')
        self.assertEqual(v.procedure, 'HOSPITALS.ANALYTICS.SP_OK')
        with self.assertRaises(SqlGuardError) as cm:
            check('call other()', callable_procedures=self.ALLOW, database='hospitals', schema='analytics')
        self.assertEqual(cm.exception.code, 'call')
        self.assertEqual(check('call other()', mode=WRITE, allow_writes=True, database='d', schema='s')[0][1].procedure,
                         'D.S.OTHER')

    def test_statement_attached_to_error(self):
        with self.assertRaises(SqlGuardError) as cm:
            check('delete from t')
        self.assertEqual(cm.exception.statement, 'delete from t')


class ErrorPositionTests(SimpleTestCase):
    def test_found(self):
        self.assertEqual(error_position("SQL compilation error: error line 3 at position 14\ninvalid identifier 'X'"),
                         {'line': 3, 'column': 14})

    def test_missing(self):
        self.assertIsNone(error_position('boom'))
        self.assertIsNone(error_position(None))


class RegressionTests(SimpleTestCase):
    """Bugs found while porting the splitter to JavaScript."""

    def test_cursor_for_inside_block_does_not_nest(self):
        sql = ('DECLARE\n  c1 CURSOR FOR SELECT 1;\nBEGIN\n  FOR r IN c1 DO\n    LET x := 1;\n  END FOR;\n'
               '  RETURN 1;\nEND;\nSELECT 2;')
        self.assertEqual(len(split_statements(sql)), 2)
        self.assertEqual(texts(sql)[1], 'SELECT 2')

    def test_let_cursor_for_in_block(self):
        sql = 'BEGIN\n  LET c1 CURSOR FOR rs;\n  FOR rec IN c1 DO\n    NULL;\n  END FOR;\nEND;\nSELECT 3;'
        self.assertEqual(texts(sql)[-1], 'SELECT 3')

    def test_unicode_identifiers_lex_as_words(self):
        toks = kinds('select café, naïve_col from tablé')
        self.assertIn(('word', 'café'), toks)
        self.assertIn(('word', 'naïve_col'), toks)
        self.assertIn(('word', 'tablé'), toks)
        self.assertEqual(verdict('select café from t').kind, 'read')
