// node --test n8n_studio/tests/js
import test from 'node:test';
import assert from 'node:assert/strict';
import { deepDiff, lineDiff, pathString } from '../../../static/n8n_studio/js/core/jsondiff.js';
import { UndoStack } from '../../../static/n8n_studio/js/core/history.js';
import { parseCSV, csvToRecords, toCSV, inferType, coerce, detectDelimiter } from '../../../static/n8n_studio/js/core/csv.js';
import { parseCron, nextRuns, describeCron } from '../../../static/n8n_studio/js/core/cron.js';
import { score, rank } from '../../../static/n8n_studio/js/core/fuzzy.js';
import { isVisible, descriptionFor, defaults, visibleProps, subtitle } from '../../../static/n8n_studio/js/core/params.js';
import * as X from '../../../static/n8n_studio/js/core/execution.js';

test('deepDiff', () => {
  const d = deepDiff({ a: 1, b: [1, 2], c: { x: 1 } }, { a: 2, b: [1], c: { x: 1, y: 2 } });
  assert.deepEqual(d.map((x) => [pathString(x.path), x.kind]), [['a', 'changed'], ['b[1]', 'removed'], ['c.y', 'added']]);
  assert.deepEqual(deepDiff({ a: [1] }, { a: [1] }), []);
  assert.deepEqual(deepDiff(null, { a: 1 }).length, 1);
});

test('lineDiff', () => {
  const d = lineDiff('a\nb\nc', 'a\nx\nc');
  assert.deepEqual(d.map((x) => x.op + x.text), [' a', '-b', '+x', ' c']);
});

test('UndoStack with coalescing', () => {
  const u = new UndoStack({ coalesceMs: 1000 });
  u.push({ v: 0 }, { group: 'type', now: 0 });
  u.push({ v: 1 }, { group: 'type', now: 100 });
  u.push({ v: 2 }, { group: 'other', now: 200 });
  assert.deepEqual(u.undo({ v: 3 }), { v: 2 });
  assert.deepEqual(u.undo({ v: 2 }), { v: 0 });
  assert.equal(u.undo({ v: 0 }), null);
  assert.deepEqual(u.redo({ v: 0 }), { v: 2 });
  assert.ok(u.canRedo);
  u.push({ v: 9 });
  assert.equal(u.canRedo, false);
});

test('CSV: BOM, quotes, newlines, delimiters', () => {
  const text = '﻿name;note\r\n"Ann";"says ""hi""\nthere"\r\nBob;\r\n';
  assert.equal(detectDelimiter(text.slice(1)), ';');
  const rows = parseCSV(text);
  assert.deepEqual(rows, [['name', 'note'], ['Ann', 'says "hi"\nthere'], ['Bob', '']]);
  const { headers, records } = csvToRecords('a,b\n1,2\n3');
  assert.deepEqual(headers, ['a', 'b']);
  assert.deepEqual(records, [{ a: '1', b: '2' }, { a: '3', b: '' }]);
  const out = toCSV([{ a: 'x,y', b: 'q"q' }, { a: { j: 1 }, b: null }], ['a', 'b']);
  assert.equal(out, 'a,b\r\n"x,y","q""q"\r\n"{""j"":1}",');
  assert.deepEqual(csvToRecords(out).records[0], { a: 'x,y', b: 'q"q' });
  assert.ok(toCSV([], ['a'], { bom: true }).startsWith('﻿'));
});

test('CSV type inference and coercion', () => {
  assert.equal(inferType(['1', '2.5', '']), 'number');
  assert.equal(inferType(['true', 'FALSE']), 'boolean');
  assert.equal(inferType(['2024-01-02', '2024-01-03T10:00:00Z']), 'date');
  assert.equal(inferType(['x', '1']), 'string');
  assert.equal(coerce('3', 'number'), 3);
  assert.equal(coerce('', 'number'), null);
  assert.equal(coerce('yes', 'boolean'), true);
});

test('cron parse, next runs, describe', () => {
  assert.throws(() => parseCron('* *'));
  assert.throws(() => parseCron('61 * * * *'));
  const from = new Date(2026, 0, 1, 10, 7, 30);
  const runs = nextRuns('*/15 * * * *', 3, from);
  assert.deepEqual(runs.map((d) => [d.getHours(), d.getMinutes()]), [[10, 15], [10, 30], [10, 45]]);
  const wk = nextRuns('0 9 * * 1-5', 2, new Date(2026, 0, 2, 12)); // Fri noon
  assert.deepEqual(wk.map((d) => d.getDay()), [1, 2]);
  assert.equal(nextRuns('@daily', 1, from)[0].getDate(), 2);
  assert.equal(describeCron('*/5 * * * *'), 'Every 5 minutes');
  assert.equal(describeCron('0 9 * * 1-5'), 'Weekdays at 09:00');
  assert.equal(describeCron('30 8 * * 1,3'), 'Mon, Wed at 08:30');
  assert.equal(describeCron('0 0 1 * *'), 'Monthly on day 1 at 00:00');
  assert.match(describeCron('garbage'), /^Cron:/);
});

test('fuzzy ranking', () => {
  assert.ok(score('slack', 'Slack') > score('slack', 'Slack Trigger'));
  assert.ok(score('ht', 'HTTP Request') > score('ht', 'Hacker News Trigger'));
  assert.equal(score('zzz', 'Slack'), -Infinity);
  const r = rank([{ n: 'Google Sheets' }, { n: 'Gmail' }, { n: 'GitHub' }], 'gsh', (x) => [x.n]);
  assert.equal(r[0].n, 'Google Sheets');
});

test('displayOptions: show/hide, root paths, @version, _cnd', () => {
  const p = { displayOptions: { show: { resource: ['message'], '@version': [{ _cnd: { gte: 2 } }] }, hide: { '/mode': ['raw'] } } };
  assert.ok(isVisible(p, { resource: 'message' }, { mode: 'x' }, { typeVersion: 2 }));
  assert.ok(!isVisible(p, { resource: 'message' }, { mode: 'x' }, { typeVersion: 1 }));
  assert.ok(!isVisible(p, { resource: 'message' }, { mode: 'raw' }, { typeVersion: 3 }));
  assert.ok(!isVisible(p, { resource: 'channel' }, {}, { typeVersion: 3 }));
  assert.ok(isVisible(p, {}, {}, { typeVersion: 3, defaults: { resource: 'message' } }), 'falls back to defaults');
  assert.ok(isVisible({ displayOptions: { show: { a: [true] } } }, { a: 'true' }) === true);
});

test('descriptionFor picks by version', () => {
  const descs = [{ version: [1, 1.1], tag: 'old' }, { version: [2, 3], tag: 'new' }];
  assert.equal(descriptionFor(descs, 1.1).tag, 'old');
  assert.equal(descriptionFor(descs, 3).tag, 'new');
  assert.equal(descriptionFor(descs, 9).tag, 'new');
  assert.equal(descriptionFor([], 1), null);
});

test('defaults resolve dependent properties', () => {
  const props = [
    { name: 'resource', type: 'options', default: 'message' },
    { name: 'operation', type: 'options', default: 'send', displayOptions: { show: { resource: ['message'] } } },
    { name: 'operation', type: 'options', default: 'create', displayOptions: { show: { resource: ['channel'] } } },
    { name: 'notice', type: 'notice', default: '' },
  ];
  assert.deepEqual(defaults(props, 1), { resource: 'message', operation: 'send' });
  assert.deepEqual(defaults(props, 1, { resource: 'channel' }), { resource: 'message', operation: 'create' });
  const vis = visibleProps(props, { resource: 'channel' }, { resource: 'channel' }, { typeVersion: 1 });
  assert.deepEqual(vis.map((p) => p.default), ['message', 'create', '']);
});

test('subtitle template', () => {
  const desc = { subtitle: '={{$parameter["operation"] + ": " + $parameter["resource"]}}' };
  assert.equal(subtitle(desc, { operation: 'send', resource: 'message' }), 'send: message');
  assert.equal(subtitle({}, { operation: 'get' }), 'get');
  assert.equal(subtitle({ subtitle: 'static' }, {}), 'static');
});

const exec = {
  id: '7', status: 'error', workflowId: 'w1', startedAt: '2026-01-01T00:00:00.000Z', stoppedAt: '2026-01-01T00:00:02.500Z',
  data: {
    resultData: {
      lastNodeExecuted: 'HTTP',
      error: { message: 'Request failed with status code 404 for id 123', node: { name: 'HTTP' } },
      runData: {
        Trigger: [{ startTime: 1, executionTime: 1, executionStatus: 'success', source: [], data: { main: [[{ json: { a: 1 } }, { json: { a: 2 } }]] } }],
        HTTP: [{ startTime: 2, executionTime: 40, executionStatus: 'error', error: { message: 'boom' }, source: [{ previousNode: 'Trigger' }], data: { main: [[]] } }],
      },
    },
    executionData: { runtimeData: { triggerNode: { name: 'Trigger' } } },
  },
};

test('execution summaries, status map, errors', () => {
  const s = X.nodeSummaries(exec);
  assert.deepEqual(s.map((x) => [x.name, x.status, x.items]), [['Trigger', 'success', 2], ['HTTP', 'error', 0]]);
  const m = X.nodeStatusMap(exec);
  assert.equal(m.get('HTTP').status, 'error');
  assert.equal(X.durationMs(exec), 2500);
  assert.equal(X.mainError(exec).node, 'HTTP');
  assert.equal(X.triggerNode(exec), 'Trigger');
  assert.equal(X.edgeCounts(exec).get('Trigger\u00000'), 2);
});

test('errorSignature groups similar errors', () => {
  assert.equal(X.errorSignature('Request failed with status code 404 for id 123'), X.errorSignature('Request failed with status code 500 for id 9'));
  assert.equal(X.errorSignature('Bad "x" at https://a.b/c'), 'Bad <str> at <url>');
  const groups = X.groupFailures([exec, { ...exec, id: '8', startedAt: '2026-01-02T00:00:00Z' }, { ...exec, id: '9', status: 'success' }]);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].count, 2);
  assert.equal(groups[0].last, '2026-01-02T00:00:00Z');
});

test('compareExecutions and pin data', () => {
  const b = JSON.parse(JSON.stringify(exec));
  b.data.resultData.runData.Trigger[0].data.main[0][0].json.a = 5;
  const c = X.compareExecutions(exec, b);
  const t = c.find((x) => x.name === 'Trigger');
  assert.equal(t.diff.length, 1);
  assert.deepEqual(X.toPinData(exec, ['Trigger']), { Trigger: [{ json: { a: 1 } }, { json: { a: 2 } }] });
  assert.deepEqual(X.toPinData(exec), { Trigger: [{ json: { a: 1 } }, { json: { a: 2 } }] });
});

test('schemaOf and itemsTable', () => {
  const items = [{ json: { a: 1, b: { c: 'x' } } }, { json: { a: 'two', d: [{ e: 1 }] } }];
  const sch = X.schemaOf(items);
  assert.deepEqual(sch.find((f) => f.path === 'a').types.sort(), ['number', 'string']);
  assert.ok(sch.some((f) => f.path === 'b.c'));
  assert.ok(sch.some((f) => f.path === 'd.0.e'));
  assert.deepEqual(X.itemsTable(items).columns, ['a', 'b', 'd']);
});

test('timeBuckets, percentile, workflowStats', () => {
  const end = Date.parse('2026-01-01T12:00:00Z');
  const list = [
    { status: 'success', startedAt: '2026-01-01T11:30:00Z', stoppedAt: '2026-01-01T11:30:01Z', workflowId: 'a' },
    { status: 'error', startedAt: '2026-01-01T11:40:00Z', stoppedAt: '2026-01-01T11:40:03Z', workflowId: 'a' },
    { status: 'success', startedAt: '2025-12-01T00:00:00Z', stoppedAt: '2025-12-01T00:00:01Z', workflowId: 'b' },
  ];
  const b = X.timeBuckets(list, { buckets: 12, end, span: 12 * 3600e3 });
  assert.equal(b.reduce((a, x) => a + x.success + x.error, 0), 2);
  assert.equal(X.percentile([1, 2, 3, 4], 50), 2);
  assert.equal(X.percentile([], 50), null);
  const st = X.workflowStats(list).find((s) => s.workflowId === 'a');
  assert.equal(st.failureRate, 0.5);
  assert.equal(st.p95, 3000);
  assert.equal(st.lastStatus, 'error');
});

test('groupFailures merges messages that differ in one token', () => {
  const mk = (id, msg, node = 'Score', wf = 'w1') => ({ id: String(id), workflowId: wf, status: 'error', startedAt: `2026-01-0${id}T00:00:00Z`, data: { resultData: { error: { message: msg, node: { name: node } } } } });
  const groups = X.groupFailures([
    mk(1, 'Amount 9500 exceeds the limit for carl'),
    mk(2, 'Amount 9999 exceeds the limit for eve'),
    mk(3, 'Amount 1 exceeds the limit for eve', 'Other'),
    mk(4, 'Totally different failure here now ok'),
  ]);
  const merged = groups.find((g) => g.count === 2);
  assert.ok(merged, JSON.stringify(groups.map((g) => g.signature)));
  assert.equal(merged.signature, 'Amount <n> exceeds the limit for <x>');
  assert.deepEqual(merged.ids.sort(), ['1', '2']);
  assert.equal(groups.length, 3, 'different node and different message stay separate');
});
