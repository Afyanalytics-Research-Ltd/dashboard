import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as g from '../../../static/warehouse_studio/js/lib/gridcore.js';

const cols = [
  { name: 'ID', family: 'number' }, { name: 'NAME', family: 'text' }, { name: 'TS', family: 'timestamp' },
  { name: 'OBJ', family: 'json' }, { name: 'OK', family: 'boolean' },
];
const rows = [
  [3, 'beta', '2026-01-03T00:00:00+00:00', { $json: { a: 1 } }, true],
  [1, 'Alpha', '2026-01-01T00:00:00+00:00', null, false],
  [null, 'gamma', null, { $json: [1, 2] }, null],
  [2, null, '2026-01-02T00:00:00+00:00', { $json: 'x' }, true],
  [10, 'item 10', '2025-12-31T23:00:00-02:00', null, false],
];

test('cellValue / cellText unwrap API wrappers', () => {
  assert.deepEqual(g.cellValue({ $json: { a: 1 } }), { a: 1 });
  assert.equal(g.cellValue({ $binary: 'BEEF' }), 'BEEF');
  assert.equal(g.cellText(null), '');
  assert.equal(g.cellText({ $json: { a: 1 } }), '{"a":1}');
  assert.equal(g.cellText({ $json: null }), 'null');
  assert.equal(g.cellText(false), 'false');
  assert.equal(g.cellText('12345678901234567890'), '12345678901234567890');
});

test('numeric sort, NULLs last in both directions', () => {
  assert.deepEqual(g.sortIndex(rows, 0, 'asc', 'number'), [1, 3, 0, 4, 2]);
  assert.deepEqual(g.sortIndex(rows, 0, 'desc', 'number'), [4, 0, 3, 1, 2]);
});

test('text sort is case-insensitive and natural', () => {
  assert.deepEqual(g.sortIndex(rows, 1, 'asc', 'text'), [1, 0, 2, 4, 3]);
  const r = [['item 10'], ['item 9'], ['item 100']];
  assert.deepEqual(g.sortIndex(r, 0, 'asc', 'text'), [1, 0, 2]);
});

test('timestamp sort honours offsets', () => {
  // 2025-12-31T23:00-02:00 == 2026-01-01T01:00Z -> after Alpha (2026-01-01T00:00Z)
  assert.deepEqual(g.sortIndex(rows, 2, 'asc', 'timestamp'), [1, 4, 3, 0, 2]);
});

test('big integers sent as strings sort exactly', () => {
  const r = [['9007199254740993'], ['9007199254740992'], [5], ['-12345678901234567890']];
  assert.deepEqual(g.sortIndex(r, 0, 'asc', 'number'), [3, 2, 1, 0]);
});

test('sort is stable and respects a base index', () => {
  const r = [[1, 'a'], [1, 'b'], [0, 'c'], [1, 'd']];
  assert.deepEqual(g.sortIndex(r, 0, 'asc', 'number'), [2, 0, 1, 3]);
  assert.deepEqual(g.sortIndex(r, 0, 'desc', 'number', [3, 2, 1]), [3, 1, 2]);
});

test('parseFilter grammar', () => {
  assert.deepEqual(g.parseFilter(''), null);
  assert.deepEqual(g.parseFilter('null'), { op: 'null' });
  assert.deepEqual(g.parseFilter('!null'), { op: 'notnull' });
  assert.deepEqual(g.parseFilter('>= 10'), { op: '>=', value: '10' });
  assert.deepEqual(g.parseFilter('<>3'), { op: '!=', value: '3' });
  assert.deepEqual(g.parseFilter('1..5'), { op: 'range', lo: 1, hi: 5 });
  assert.equal(g.parseFilter('/^a/').op, 'regex');
  assert.equal(g.parseFilter('/[/').op, 'contains');   // invalid regex falls back
  assert.deepEqual(g.parseFilter('=Beta'), { op: 'eq', value: 'beta' });
  assert.deepEqual(g.parseFilter('!ga'), { op: 'notcontains', value: 'ga' });
  assert.deepEqual(g.parseFilter('ALP'), { op: 'contains', value: 'alp' });
});

test('filterIndex across columns and families', () => {
  const fam = cols.map((c) => c.family);
  assert.deepEqual(g.filterIndex(rows, { 1: 'a' }, fam), [0, 1, 2]);
  assert.deepEqual(g.filterIndex(rows, { 0: '>2' }, fam), [0, 4]);
  assert.deepEqual(g.filterIndex(rows, { 0: '2..3' }, fam), [0, 3]);
  assert.deepEqual(g.filterIndex(rows, { 0: 'null' }, fam), [2]);
  assert.deepEqual(g.filterIndex(rows, { 1: '!null', 0: '<5' }, fam), [0, 1]);
  assert.deepEqual(g.filterIndex(rows, { 3: '"a"' }, fam), [0]);
  assert.deepEqual(g.filterIndex(rows, { 2: '>=2026-01-02' }, fam), [0, 3]);
  assert.deepEqual(g.filterIndex(rows, { 2: '>2026-01-02' }, fam), [0]);
  assert.deepEqual(g.filterIndex(rows, { 1: '/^(alpha|beta)$/' }, fam), [0, 1]);
  assert.deepEqual(g.filterIndex(rows, { 1: '!a' }, fam), [3, 4]);
  assert.deepEqual(g.filterIndex(rows, {}, fam), [0, 1, 2, 3, 4]);
  assert.deepEqual(g.filterIndex(rows, { 1: '' }, fam, [4, 0]), [4, 0]);
});

test('profileColumn for numbers', () => {
  const p = g.profileColumn(rows, 0, 'number');
  assert.equal(p.count, 5);
  assert.equal(p.nulls, 1);
  assert.equal(p.nullPct, 0.2);
  assert.equal(p.distinct, 4);
  assert.equal(p.unique, true);
  assert.equal(p.min, '1');
  assert.equal(p.max, '10');
  assert.equal(p.sum, 16);
  assert.equal(p.mean, 4);
  assert.equal(p.histogram.reduce((a, b) => a + b.count, 0), 4);
  assert.ok(p.stddev > 0);
});

test('profileColumn for text with top values', () => {
  const r = [['a'], ['b'], ['a'], [null], ['ccc']];
  const p = g.profileColumn(r, 0, 'text');
  assert.equal(p.top[0].value, 'a');
  assert.equal(p.top[0].count, 2);
  assert.equal(p.top[0].share, 0.5);
  assert.equal(p.unique, false);
  assert.equal(p.minLength, 1);
  assert.equal(p.maxLength, 3);
  const empty = g.profileColumn([], 0, 'text');
  assert.equal(empty.count, 0);
  assert.equal(empty.min, null);
});

test('histogram edge cases', () => {
  assert.deepEqual(g.histogram([], 5), []);
  assert.deepEqual(g.histogram([7, 7, 7], 5), [{ lo: 7, hi: 7, count: 3 }]);
  const h = g.histogram([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5);
  assert.equal(h.length, 5);
  assert.equal(h.at(-1).hi, 10);
  assert.equal(h.reduce((a, b) => a + b.count, 0), 11);
});

test('CSV escaping: quotes, commas, newlines, padding, NULL as empty', () => {
  const c = [{ name: 'a,b' }, { name: 'q' }];
  const r = [['he said "hi"', 'line1\nline2'], [null, ' padded'], [{ $json: { k: 'v,w' } }, 'plain']];
  assert.equal(g.toCSV(c, r), '"a,b",q\n"he said ""hi""","line1\nline2"\n," padded"\n"{""k"":""v,w""}",plain');
  assert.equal(g.toCSV(c, r, { header: false, idx: [2], cols: [1] }), 'plain');
});

test('TSV flattens tabs and newlines', () => {
  const c = [{ name: 'x\ty' }, { name: 'z' }];
  const r = [['a\tb', 'c\r\nd'], [null, 1]];
  assert.equal(g.toTSV(c, r), 'x y\tz\na b\tc d\n\t1');
});

test('Markdown escapes pipes, backslashes and newlines; numbers right-aligned', () => {
  const c = [{ name: 'n', family: 'number' }, { name: 't|x', family: 'text' }];
  const r = [[1, 'a|b'], [null, 'c\\d\ne']];
  assert.equal(g.toMarkdown(c, r), '| n | t\\|x |\n| ---: | --- |\n| 1 | a\\|b |\n|  | c\\\\d<br>e |');
  assert.equal(g.toMarkdown(c, r, { maxRows: 1 }).split('\n').length, 3);
});

test('JSON export unwraps VARIANT and keeps nulls', () => {
  const out = JSON.parse(g.toJSON(cols.slice(0, 4), rows.slice(0, 2).map((r) => r.slice(0, 4))));
  assert.deepEqual(out[0], { ID: 3, NAME: 'beta', TS: '2026-01-03T00:00:00+00:00', OBJ: { a: 1 } });
  assert.equal(out[1].OBJ, null);
});

test('autoChart picks time series, category bars, scatter', () => {
  const ts = g.autoChart([{ family: 'date' }, { family: 'text' }, { family: 'number' }], [['2026-01-01', 'a', 1], ['2026-01-02', 'b', 2]]);
  assert.deepEqual([ts.type, ts.x, ts.y, ts.group], ['area', 0, [2], 1]);
  const bars = g.autoChart([{ family: 'text' }, { family: 'number' }, { family: 'number' }]);
  assert.deepEqual([bars.type, bars.x, bars.y], ['bar', 0, [1]]);
  const sc = g.autoChart([{ family: 'number' }, { family: 'number' }]);
  assert.equal(sc.type, 'scatter');
  assert.equal(g.autoChart([{ family: 'text' }]).ok, false);
});

test('chartSeries sums per x/group, sorts x, folds groups beyond 8 into Other', () => {
  const r = [['2026-01-02', 'a', 1], ['2026-01-01', 'a', 2], ['2026-01-01', 'a', 3], ['2026-01-01', 'b', '4']];
  const s = g.chartSeries(r, { x: 0, y: [2], group: 1, xFamily: 'date' });
  assert.deepEqual(s.xs, ['2026-01-01', '2026-01-02']);
  assert.deepEqual(s.series.find((x) => x.name === 'a').data, [5, 1]);
  assert.deepEqual(s.series.find((x) => x.name === 'b').data, [4, 0]);
  const many = [...Array(12)].map((_, i) => ['x', `g${i}`, i + 1]);
  const m = g.chartSeries(many, { x: 0, y: [2], group: 1 });
  assert.equal(m.series.length, 8);
  assert.equal(m.series.at(-1).name, 'Other');
  assert.equal(m.series.reduce((a, x) => a + x.data[0], 0), 78);
  const noGroup = g.chartSeries([['a', 1, 2]], { x: 0, y: [1, 2] });
  assert.deepEqual(noGroup.series.map((x) => x.name), ['1', '2']);
});
