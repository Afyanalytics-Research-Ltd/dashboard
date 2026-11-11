import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as x from '../../../static/warehouse_studio/js/lib/explorercore.js';

const GB = 1024 ** 3;

test('queryBadges: remote spill wins over local', () => {
  const b = x.queryBadges({ bytes_spilled_to_remote_storage: 5, bytes_spilled_to_local_storage: 10 });
  assert.deepEqual(b.map((v) => v.key), ['remote_spill']);
  assert.equal(b[0].severity, 'CRITICAL');
});

test('queryBadges: local spill severity scales with size', () => {
  assert.equal(x.queryBadges({ bytes_spilled_to_local_storage: 100 })[0].severity, 'LOW');
  assert.equal(x.queryBadges({ bytes_spilled_to_local_storage: 2 * GB })[0].severity, 'HIGH');
});

test('queryBadges: queued, compile-heavy, pruning, cold start', () => {
  const keys = x.queryBadges({
    total_elapsed_time: 10000, queued_overload_time: 7000, compilation_time: 2000, execution_time: 500,
    partitions_scanned: 990, partitions_total: 1000, queued_provisioning_time: 3000,
  }).map((b) => [b.key, b.severity]);
  assert.deepEqual(keys, [['queued', 'HIGH'], ['compile', 'MEDIUM'], ['pruning', 'HIGH'], ['cold', 'LOW']]);
});

test('queryBadges: healthy / small / null rows have none', () => {
  assert.deepEqual(x.queryBadges({ total_elapsed_time: 100, execution_time: 100 }), []);
  assert.deepEqual(x.queryBadges({ partitions_scanned: 9, partitions_total: 10 }), []);  // small table
  assert.deepEqual(x.queryBadges({ queued_overload_time: 500, total_elapsed_time: 600 }), []); // < 1s
  assert.deepEqual(x.queryBadges(null), []);
  assert.deepEqual(x.queryBadges({ partitions_scanned: null, partitions_total: null }), []);
});

test('isFailed / isRunning', () => {
  assert.ok(x.isFailed({ execution_status: 'FAIL' }));
  assert.ok(x.isFailed({ execution_status: 'FAILED_WITH_ERROR' }));
  assert.ok(!x.isFailed({ execution_status: 'SUCCESS' }));
  assert.ok(x.isRunning({ execution_status: 'RESUMING_WAREHOUSE' }));
  assert.ok(!x.isRunning({}));
});

test('durationBar is log-scaled and bounded', () => {
  assert.equal(x.durationBar(0, 1000), 0);
  assert.equal(x.durationBar(1000, 1000), 1);
  assert.ok(x.durationBar(10, 1000) > 0.3, 'short queries still visible');
  assert.ok(x.durationBar(10, 1000) < x.durationBar(100, 1000));
  assert.equal(x.durationBar(5000, 1000), 1);
  assert.equal(x.durationBar(null, null), 0);
});

test('percentile interpolates', () => {
  assert.equal(x.percentile([1, 2, 3, 4], 0.5), 2.5);
  assert.equal(x.percentile([5], 0.95), 5);
  assert.equal(x.percentile([], 0.5), null);
  assert.equal(x.percentile([1, 'x', 3], 1), 3);
});

test('historyKpis', () => {
  const k = x.historyKpis([
    { total_elapsed_time: 100, execution_status: 'SUCCESS', usd: 0.5, credits: 0.1, bytes_scanned: 10 },
    { total_elapsed_time: 300, execution_status: 'FAIL', usd: 1, est_credits: 0.2, bytes_spilled_to_local_storage: 4 },
  ]);
  assert.equal(k.count, 2);
  assert.equal(k.failed, 1);
  assert.equal(k.failedRatio, 0.5);
  assert.equal(k.p50, 200);
  assert.equal(k.usd, 1.5);
  assert.ok(Math.abs(k.credits - 0.3) < 1e-9);
  assert.equal(k.spilled, 1);
  assert.equal(x.historyKpis([]).failedRatio, 0);
  assert.equal(x.historyKpis(undefined).count, 0);
});

test('diffLines: LCS keeps common lines and marks edits', () => {
  const d = x.diffLines('select a\nfrom t\nwhere x = 1', 'select a, b\nfrom t\nwhere x = 1\nlimit 5');
  assert.deepEqual(d.map((r) => r.op), ['del', 'add', 'same', 'same', 'add']);
  const s = x.diffStats(d);
  assert.equal(s.same, 2);
  assert.equal(s.add, 2);
  assert.equal(s.del, 1);
  assert.equal(s.identical, false);
  assert.equal(s.similarity, 0.5);
});

test('diffLines: identical, empty and CRLF', () => {
  assert.ok(x.diffStats(x.diffLines('a\r\nb', 'a\nb')).identical);
  assert.ok(x.diffStats(x.diffLines('', '')).identical);
  const d = x.diffLines('', 'x');
  assert.deepEqual(d.map((r) => r.op), ['del', 'add']);
  assert.equal(x.diffStats([]).similarity, 1);
});

test('diffLines: line numbers on both sides', () => {
  const d = x.diffLines('a\nb\nc', 'a\nc');
  assert.deepEqual(d, [{ op: 'same', a: 1, b: 1, text: 'a' }, { op: 'del', a: 2, text: 'b' }, { op: 'same', a: 3, b: 2, text: 'c' }]);
});

test('normalizeSql collapses whitespace only', () => {
  assert.equal(x.normalizeSql('select  a ,\tb   \nfrom t  '), 'select a , b\nfrom t');
});

test('compareMetrics verdicts honour direction and tolerance', () => {
  const a = { total_elapsed_time: 1000, bytes_scanned: 100, rows_produced: 5, usd: 1, bytes_spilled_to_remote_storage: 0 };
  const b = { total_elapsed_time: 500, bytes_scanned: 101, rows_produced: 7, usd: 2, bytes_spilled_to_remote_storage: 0 };
  const byKey = Object.fromEntries(x.compareMetrics(a, b).map((m) => [m.key, m]));
  assert.equal(byKey.total_elapsed_time.verdict, 'better');
  assert.equal(byKey.total_elapsed_time.delta, -500);
  assert.equal(byKey.total_elapsed_time.ratio, -0.5);
  assert.equal(byKey.bytes_scanned.verdict, 'same');          // within 2 %
  assert.equal(byKey.rows_produced.verdict, 'changed');       // no better direction
  assert.equal(byKey.usd.verdict, 'worse');
  assert.equal(byKey.bytes_spilled_to_remote_storage.verdict, 'same');
  assert.equal(byKey.partitions_scanned.verdict, 'n/a');      // missing on both
  assert.equal(byKey.partitions_scanned.delta, null);
});

test('compareMetrics: from zero is an infinite ratio', () => {
  const [m] = x.compareMetrics({ q: 0 }, { q: 5 }, [{ key: 'q', label: 'Q', kind: 'int', better: 'lower' }]);
  assert.equal(m.ratio, Infinity);
  assert.equal(m.verdict, 'worse');
});

test('parseHistoryFilters sanitises junk and applies defaults', () => {
  const f = x.parseHistoryFilters(new URLSearchParams('scope=evil&status=nope&sort=drop&hours=-3&min_ms=abc&limit=99999&spilled=1&hide_system=0'));
  assert.equal(f.scope, 'live');
  assert.equal(f.status, '');
  assert.equal(f.sort, 'start');
  assert.equal(f.hours, '24');
  assert.equal(f.min_ms, '');
  assert.equal(f.limit, '1000');
  assert.equal(f.spilled, '', 'live scope cannot filter spill');
  assert.equal(f.hide_system, '0');
});

test('parseHistoryFilters caps hours by scope and keeps valid values', () => {
  assert.equal(x.parseHistoryFilters({ scope: 'live', hours: '999' }).hours, String(24 * 7));
  const f = x.parseHistoryFilters({ scope: 'account', hours: '999', status: 'failed', sort: 'spill', spilled: 'true', warehouse: 'COMPUTE_WH', min_ms: '1500' });
  assert.equal(f.hours, '720');
  assert.equal(f.status, 'FAILED');
  assert.equal(f.sort, 'spill');
  assert.equal(f.spilled, '1');
  assert.equal(f.warehouse, 'COMPUTE_WH');
  assert.equal(f.min_ms, '1500');
  assert.equal(x.parseHistoryFilters(null).scope, 'live');
});

test('historyQueryPatch drops defaults; round-trip through the URL', () => {
  const f = x.parseHistoryFilters({ scope: 'account', status: 'FAILED', q: 'select' });
  const patch = x.historyQueryPatch(f);
  assert.equal(patch.hours, null);
  assert.equal(patch.hide_system, null);
  assert.equal(patch.scope, 'account');
  const sp = new URLSearchParams(Object.entries(patch).filter(([, v]) => v != null));
  assert.deepEqual(x.parseHistoryFilters(sp), f);
});

test('historyApiParams drops empties', () => {
  assert.deepEqual(x.historyApiParams({ a: '', b: null, c: '1', d: 0 }), { c: '1', d: 0 });
});

test('parseSignature mirrors the server parser', () => {
  assert.deepEqual(x.parseSignature('(A VARCHAR, B NUMBER(38,0), DEFAULT C FLOAT)'), [
    { name: 'A', type: 'VARCHAR', has_default: false },
    { name: 'B', type: 'NUMBER(38,0)', has_default: false },
    { name: 'C', type: 'FLOAT', has_default: true },
  ]);
  assert.deepEqual(x.parseSignature('()'), []);
  assert.deepEqual(x.parseSignature(null), []);
  assert.deepEqual(x.parseSignature('(VARCHAR)'), [{ name: 'ARG1', type: 'VARCHAR', has_default: false }]);
  assert.deepEqual(x.parseSignature('("p" ARRAY)'), [{ name: 'p', type: 'ARRAY', has_default: false }]);
});

test('inputKind', () => {
  assert.equal(x.inputKind('NUMBER(38,0)'), 'int');
  assert.equal(x.inputKind('float'), 'float');
  assert.equal(x.inputKind('BOOLEAN'), 'boolean');
  assert.equal(x.inputKind('VARIANT'), 'json');
  assert.equal(x.inputKind('TIMESTAMP_LTZ'), 'datetime');
  assert.equal(x.inputKind('VARCHAR(10)'), 'text');
  assert.equal(x.inputKind(undefined), 'text');
});

test('coerceParam mirrors bind_value rules', () => {
  const p = (type) => ({ name: 'P', type });
  assert.deepEqual(x.coerceParam(p('NUMBER'), ' 42 '), { ok: true, value: 42 });
  assert.deepEqual(x.coerceParam(p('NUMBER'), '4.5'), { ok: true, value: 4.5 });
  assert.equal(x.coerceParam(p('NUMBER'), '1e3').ok, false);
  assert.equal(x.coerceParam(p('NUMBER'), 'abc').ok, false);
  assert.deepEqual(x.coerceParam(p('FLOAT'), '0.5'), { ok: true, value: 0.5 });
  assert.equal(x.coerceParam(p('FLOAT'), 'NaN').ok, false);
  assert.deepEqual(x.coerceParam(p('BOOLEAN'), 'yes'), { ok: true, value: true });
  assert.deepEqual(x.coerceParam(p('BOOLEAN'), 'off'), { ok: true, value: false });
  assert.deepEqual(x.coerceParam(p('BOOLEAN'), false), { ok: true, value: false });
  assert.deepEqual(x.coerceParam(p('ARRAY'), '[1,2]'), { ok: true, value: '[1,2]' });
  assert.deepEqual(x.coerceParam(p('OBJECT'), { a: 1 }), { ok: true, value: '{"a":1}' });
  assert.equal(x.coerceParam(p('VARIANT'), '{bad').ok, false);
  assert.deepEqual(x.coerceParam(p('VARCHAR'), 'x'), { ok: true, value: 'x' });
  assert.equal(x.coerceParam(p('VARCHAR'), '  ').empty, true);
});

test('validateParams: required, defaults, trailing defaults dropped', () => {
  const params = [
    { name: 'A', type: 'NUMBER', has_default: false },
    { name: 'B', type: 'FLOAT', has_default: true },
    { name: 'C', type: 'VARCHAR', has_default: true },
  ];
  assert.deepEqual(x.validateParams(params, ['7', '', '']), { ok: true, values: [7], errors: [] });
  assert.deepEqual(x.validateParams(params, ['7', '', 'z']), { ok: true, values: [7, null, 'z'], errors: [] });
  const bad = x.validateParams(params, ['', 'x', '']);
  assert.equal(bad.ok, false);
  assert.equal(bad.errors.length, 2);
  assert.deepEqual(x.validateParams([], []), { ok: true, values: [], errors: [] });
});

test('procedureDdlName quotes and strips type precision', () => {
  assert.equal(x.procedureDdlName('HOSPITALS', 'ANALYTICS', 'SP_X', [{ type: 'NUMBER(38,0)' }, { type: 'VARCHAR' }]),
    '"HOSPITALS"."ANALYTICS"."SP_X"(NUMBER, VARCHAR)');
  assert.equal(x.procedureDdlName('d', 's', 'p"q', []), '"d"."s"."p""q"()');
});

test('groupObjects orders kinds, filters, sorts and drops empty groups', () => {
  const objs = {
    VIEW: [{ name: 'v_b' }, { name: 'v_a' }],
    TABLE: [{ name: 'small', rows: 1, bytes: 10 }, { name: 'big', rows: 100, bytes: 5 }],
    PROCEDURE: [{ name: 'sp' }],
    WEIRD_KIND: [{ name: 'w' }],
    STAGE: [],
  };
  const g = x.groupObjects(objs);
  assert.deepEqual(g.map((k) => k.kind), ['TABLE', 'VIEW', 'PROCEDURE', 'WEIRD_KIND']);
  assert.deepEqual(g[1].items.map((o) => o.name), ['v_a', 'v_b']);
  assert.equal(g[3].label, 'weird kind');
  assert.deepEqual(x.groupObjects(objs, { sort: 'rows' })[0].items.map((o) => o.name), ['big', 'small']);
  assert.deepEqual(x.groupObjects(objs, { sort: 'size' })[0].items.map((o) => o.name), ['small', 'big']);
  const f = x.groupObjects(objs, { q: 'V_' });
  assert.deepEqual(f.map((k) => k.kind), ['VIEW']);
  assert.deepEqual(x.groupObjects(null), []);
});

test('objectTotals', () => {
  assert.deepEqual(x.objectTotals({ TABLE: [{ rows: 2, bytes: 3 }, { rows: null }], VIEW: [{}] }), { count: 3, rows: 2, bytes: 3 });
});

test('quoteIdent', () => {
  assert.equal(x.quoteIdent('Db', 'my"schema', 'T'), '"Db"."my""schema"."T"');
  assert.equal(x.quoteIdent('a', '', null), '"a"');
});

test('detectorOutcome recognises detector and batch JSON', () => {
  assert.deepEqual(x.detectorOutcome('{"detector":"sp_detect_queueing","findings":3}'),
    { kind: 'detector', detector: 'sp_detect_queueing', findings: 3, raw: { detector: 'sp_detect_queueing', findings: 3 } });
  assert.equal(x.detectorOutcome({ $json: { batch_id: 'x', new_findings: 4, failed: 1 } }).findings, 4);
  assert.equal(x.detectorOutcome('not json'), null);
  assert.equal(x.detectorOutcome([1]), null);
  assert.equal(x.detectorOutcome({ other: 1 }), null);
  assert.equal(x.detectorOutcome(null), null);
});
