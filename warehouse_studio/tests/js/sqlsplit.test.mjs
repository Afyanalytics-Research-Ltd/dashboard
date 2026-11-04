import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
  tokenize, splitStatements, statementAt, posFromOffset, offsetFromPos, errorPosInDoc, isExplainable,
} from '../../../static/warehouse_studio/js/lib/sqlsplit.js';

const texts = (sql) => splitStatements(sql).map((s) => s.text);
const kinds = (sql) => tokenize(sql).filter((t) => t.kind !== 'ws').map((t) => t.kind);

test('tokenize: strings with doubled quotes and backslashes', () => {
  const toks = tokenize("select 'it''s', 'a\\'b' x");
  const strs = toks.filter((t) => t.kind === 'string').map((t) => t.text);
  assert.deepEqual(strs, ["'it''s'", "'a\\'b'"]);
  assert.equal(toks.at(-1).text, 'x');
});

test('tokenize: quoted identifiers with doubled quotes', () => {
  const t = tokenize('select "a""b" from "My Table"').filter((x) => x.kind === 'ident').map((x) => x.text);
  assert.deepEqual(t, ['"a""b"', '"My Table"']);
});

test('tokenize: comments of all three kinds and dollar bodies', () => {
  assert.deepEqual(kinds('-- hi ;\nselect 1 // x;\n/* ; */ $$ a; b $$'),
    ['comment', 'word', 'number', 'comment', 'comment', 'dollar']);
});

test('tokenize: unterminated strings, idents, comments, dollar run to end', () => {
  for (const sql of ["select 'abc", 'select "abc', 'select /* abc', 'select $$ abc']) {
    const toks = tokenize(sql);
    assert.equal(toks.at(-1).end, sql.length, sql);
  }
});

test('tokenize: numbers and offsets cover the input exactly', () => {
  const sql = 'select 1.5e3, .5, 42 from t where a>=3;';
  const toks = tokenize(sql);
  assert.deepEqual(toks.filter((t) => t.kind === 'number').map((t) => t.text), ['1.5e3', '.5', '42', '3']);
  assert.equal(toks.map((t) => t.text).join(''), sql);
  for (let i = 1; i < toks.length; i += 1) assert.equal(toks[i].start, toks[i - 1].end);
});

test('split: multiple statements, empty and comment-only pieces dropped', () => {
  assert.deepEqual(texts('select 1; select 2;;  ; -- only a comment\n; select 3'), ['select 1', 'select 2', 'select 3']);
  assert.deepEqual(texts(''), []);
  assert.deepEqual(texts('  -- nothing\n /* here */ '), []);
});

test('split: semicolons inside strings, idents, comments and $$ do not split', () => {
  const sql = "select ';' as a, \"x;y\" from t -- ; no\n where b = 1; create procedure p() returns int language sql as $$ begin return 1; end; $$;";
  const st = texts(sql);
  assert.equal(st.length, 2);
  assert.ok(st[1].startsWith('create procedure'));
  assert.ok(st[1].endsWith('$$'));
});

test('split: anonymous BEGIN..END block with nested IF/FOR/CASE stays whole', () => {
  const block = `begin
  let x := 1;
  if (x > 0) then
    x := 2;
  end if;
  for i in 1 to 3 do
    x := x + i;
  end for;
  select case when x > 1 then 1 else 0 end;
  return x;
end`;
  const st = splitStatements(`${block}; select 9`);
  assert.equal(st.length, 2);
  assert.equal(st[0].text, block);
  assert.equal(st[1].text, 'select 9');
});

test('split: LOOP / WHILE / REPEAT blocks', () => {
  const sql = 'begin while (true) do break; end while; loop break; end loop; repeat x := 1; until (true) end repeat; end; select 1';
  assert.equal(splitStatements(sql).length, 2);
});

test('split: IF NOT EXISTS inside a block does not open a nesting level', () => {
  const sql = 'begin create table if not exists t (a int); insert into t values (1); end; select 2';
  const st = texts(sql);
  assert.equal(st.length, 2);
  assert.equal(st[1], 'select 2');
});

test('split: BEGIN; / BEGIN TRANSACTION / BEGIN WORK are transactions', () => {
  assert.deepEqual(texts('begin; insert into t values (1); commit;'), ['begin', 'insert into t values (1)', 'commit']);
  assert.deepEqual(texts('begin transaction; select 1; commit'), ['begin transaction', 'select 1', 'commit']);
  assert.deepEqual(texts('BEGIN WORK; select 1'), ['BEGIN WORK', 'select 1']);
  assert.deepEqual(texts('begin'), ['begin']);
});

test('split: DECLARE … BEGIN … END is one statement', () => {
  const sql = `declare
  v number default 1;
  c cursor for select 1;
begin
  v := v + 1;
  return v;
end;
select 'after'`;
  const st = texts(sql);
  assert.equal(st.length, 2);
  assert.ok(st[0].startsWith('declare'));
  assert.ok(st[0].endsWith('end'));
  assert.equal(st[1], "select 'after'");
});

test('split: offsets point into the original text and keyword is set', () => {
  const sql = '  -- lead\n  SELECT 1 ;\n\n(select 2) union all select 3';
  const st = splitStatements(sql);
  assert.equal(st.length, 2);
  for (const s of st) assert.equal(sql.slice(s.start, s.end), s.text);
  assert.equal(st[0].keyword, 'SELECT');
  assert.equal(st[1].keyword, 'SELECT');
});

test('split: the install.sql toolkit splits into the same 82 statements as the server', () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const raw = readFileSync(join(here, '../../../snowflake/warehouse_studio/install.sql'), 'utf8');
  const sql = raw.replace(/\{\{DB\}\}/g, 'HOSPITALS').replace(/\{\{SCHEMA\}\}/g, 'ANALYTICS')
    .replace(/\{\{APP_ROLE\}\}/g, 'DATAANALYSTS').replace(/\{\{TASK_WAREHOUSE\}\}/g, 'COMPUTE_WH');
  const st = splitStatements(sql);
  assert.equal(st.length, 82);
  assert.ok(st.every((s) => !/^\s*$/.test(s.text)));
  assert.equal(st[0].text, 'USE ROLE ACCOUNTADMIN');
});

test('split: LET … CURSOR FOR inside a block does not open a loop', () => {
  const sql = 'begin let c1 cursor for select 1; for r in c1 do x := 1; end for; end; select 2';
  assert.deepEqual(texts(sql).at(-1), 'select 2');
  assert.equal(splitStatements(sql).length, 2);
});

test('statementAt picks the statement under or before the cursor', () => {
  const sql = 'select 1;\nselect 2;\n\nselect 3';
  assert.equal(statementAt(sql, 0).text, 'select 1');
  assert.equal(statementAt(sql, 13).text, 'select 2');
  assert.equal(statementAt(sql, 20).text, 'select 2'); // blank line after statement 2
  assert.equal(statementAt(sql, sql.length).text, 'select 3');
  assert.equal(statementAt('-- nothing', 3), null);
});

test('positions round-trip and error positions map into the document', () => {
  const doc = 'select 1;\n\nselect\n  nope from t';
  for (const off of [0, 5, 9, 10, 11, 20, doc.length]) assert.equal(offsetFromPos(doc, posFromOffset(doc, off)), off);
  const start = doc.indexOf('select\n');
  assert.deepEqual(errorPosInDoc(doc, start, { line: 2, column: 2 }), { line: 3, ch: 2 });
  assert.deepEqual(errorPosInDoc(doc, start, { line: 1, column: 3 }), { line: 2, ch: 3 });
  assert.equal(errorPosInDoc(doc, 0, null), null);
});

test('isExplainable only for queries', () => {
  assert.ok(isExplainable({ keyword: 'SELECT' }));
  assert.ok(isExplainable({ keyword: 'with' }));
  assert.ok(!isExplainable({ keyword: 'SHOW' }));
  assert.ok(!isExplainable(null));
});
