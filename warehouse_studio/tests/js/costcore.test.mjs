import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as c from '../../../static/warehouse_studio/js/lib/costcore.js';

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≈ ${b}`);

test('effectivePrice: override wins, junk falls back', () => {
  assert.equal(c.effectivePrice(3.1, 2), 3.1);
  assert.equal(c.effectivePrice('4', 2), 4);
  assert.equal(c.effectivePrice(null, 2), 2);
  assert.equal(c.effectivePrice('', 2), 2);
  assert.equal(c.effectivePrice(-1, 2), 2);
  assert.equal(c.effectivePrice('abc', 0), 2.6);
  assert.equal(c.toUsd(10, 2.5), 25);
  assert.equal(c.toUsd(null, 2.5), 0);
  assert.equal(c.perMonth(7, 7), 30);
  assert.equal(c.perMonth(5, 0), 0);
});

test('isError / listOr distinguish per-card failures from data', () => {
  assert.ok(c.isError({ error: 'x' }));
  assert.ok(!c.isError([]));
  assert.ok(!c.isError(null));
  assert.deepEqual(c.listOr({ error: 'x' }), []);
  assert.deepEqual(c.listOr([1]), [1]);
});

test('dailyTotals sums and sorts by day', () => {
  const r = c.dailyTotals([
    { day: '2026-11-02', credits: 1 }, { day: '2026-11-01', credits: 2 }, { day: '2026-11-02T00:00:00', credits: 3 },
  ]);
  assert.deepEqual(r, [{ day: '2026-11-01', credits: 2 }, { day: '2026-11-02', credits: 4 }]);
  assert.deepEqual(c.dailyTotals({ error: 'x' }), []);
});

test('cumulative', () => {
  assert.deepEqual(c.cumulative([{ x: 1, y: 2 }, { x: 2, y: 3 }, { x: 3, y: null }]).map((p) => p.y), [2, 5, 5]);
});

test('pivotDaily orders keys by total and applies labels', () => {
  const p = c.pivotDaily([
    { day: '2026-11-01', service_type: 'PIPE', credits: 1 },
    { day: '2026-11-01', service_type: 'WAREHOUSE_METERING', credits: 5 },
    { day: '2026-11-02', service_type: 'WAREHOUSE_METERING', credits: 5 },
    { day: '2026-11-02', service_type: null, credits: 0.5 },
  ], (r) => r.service_type, 'credits', c.serviceLabel);
  assert.deepEqual(p.keys, ['Warehouse metering', 'Pipe', '(none)']);
  assert.equal(p.rows.length, 4);
  assert.equal(p.totals.get('Warehouse metering'), 10);
});

test('monthBurnup aligns months and projects at the 7-day run rate', () => {
  const today = new Date(Date.UTC(2026, 10, 10, 15)); // Nov 10 2026
  const daily = [];
  for (let d = 1; d <= 31; d += 1) daily.push({ day: `2026-10-${String(d).padStart(2, '0')}`, credits: 1 });
  for (let d = 1; d <= 10; d += 1) daily.push({ day: `2026-11-${String(d).padStart(2, '0')}`, credits: 2 });
  const b = c.monthBurnup(daily, today);
  assert.equal(b.daysInMonth, 30);
  assert.equal(b.thisMonth.length, 10);
  assert.equal(b.mtd, 20);
  assert.equal(b.lastMonth.length, 31);
  assert.equal(b.lastMonthTotal, 31);
  assert.equal(b.rate, 2);                         // Nov 3..9
  assert.equal(b.forecast[0].x, 10);
  assert.equal(b.forecast[0].y, 20);
  assert.equal(b.forecast.at(-1).x, 30);
  assert.equal(b.projected, 20 + 2 * 20);
  assert.equal(b.lastSamePoint, 10);
  assert.equal(b.pace, 1);
});

test('monthBurnup on the 1st reaches back into last month for the run rate', () => {
  const today = new Date(Date.UTC(2026, 11, 1, 1)); // Dec 1
  const daily = [{ day: '2026-11-30', credits: 7 }, { day: '2026-11-24', credits: 7 }, { day: '2026-12-01', credits: 1 }];
  const b = c.monthBurnup(daily, today);
  assert.equal(b.mtd, 1);
  assert.equal(b.rate, 2);
  assert.equal(b.lastMonth.length, 30);
  assert.equal(b.forecast.length, 31);
  const empty = c.monthBurnup([], today);
  assert.equal(empty.projected, 0);
  assert.equal(empty.pace, null);
});

test('monthBurnup can be anchored to the server projection', () => {
  const today = new Date(Date.UTC(2026, 10, 10, 15));
  const daily = [];
  for (let d = 1; d <= 10; d += 1) daily.push({ day: `2026-11-${String(d).padStart(2, '0')}`, credits: 2 });
  const b = c.monthBurnup(daily, today, { projected: 80 });
  assert.equal(b.mtd, 20);
  assert.equal(b.rate, 3);                      // (80 - 20) / 20 remaining days
  near(b.projected, 80);
  assert.equal(c.monthBurnup(daily, today, { projected: 5 }).rate, 2);       // below MTD: ignored
  assert.equal(c.monthBurnup(daily, today, { projected: null }).rate, 2);    // absent: run-rate
  const last = c.monthBurnup([{ day: '2026-11-30', credits: 4 }], new Date(Date.UTC(2026, 10, 30)), { projected: 9 });
  assert.equal(last.rate, 0);                   // last day: nothing left to forecast
  assert.equal(last.forecast.length, 1);
});

test('foldTopN folds the tail into Other', () => {
  const items = [1, 9, 3, 7, 5, 0].map((v, i) => ({ name: `n${i}`, value: v }));
  const f = c.foldTopN(items, 3);
  assert.deepEqual(f.map((i) => i.name), ['n1', 'n3', 'n4', 'Other']);
  assert.equal(f[3].value, 4);
  assert.equal(f[3].count, 2);
  assert.equal(c.foldTopN(items, 10).length, 5); // zero dropped, no Other
});

test('heatmapInsights: peak, weekend, night and business shares', () => {
  const cells = [
    { dow: 1, hour: 9, value: 6 },    // business
    { dow: 6, hour: 2, value: 3 },    // weekend + night
    { dow: 3, hour: 22, value: 1 },   // night
    { dow: 9, hour: 2, value: 100 },  // junk, ignored
  ];
  const i = c.heatmapInsights(cells);
  assert.equal(i.total, 10);
  assert.deepEqual(i.peak, { dow: 1, hour: 9, value: 6 });
  near(i.weekendShare, 0.3);
  near(i.nightShare, 0.4);
  near(i.businessShare, 0.6);
  assert.equal(i.busiestDow, 1);
  assert.equal(i.busiestHour, 9);
  const z = c.heatmapInsights([]);
  assert.equal(z.peak, null);
  assert.equal(z.weekendShare, 0);
});

test('classifyUtilization bands', () => {
  assert.equal(c.classifyUtilization(null), 'unknown');
  assert.equal(c.classifyUtilization(0.05), 'idle');
  assert.equal(c.classifyUtilization(0.2), 'low');
  assert.equal(c.classifyUtilization(0.9), 'healthy');
  assert.equal(c.classifyUtilization(2), 'saturated');
});

test('sizes: stepping and SQL keywords', () => {
  assert.equal(c.sizeStep('Small', 1), 'Medium');
  assert.equal(c.sizeStep('X-Small', -1), null);
  assert.equal(c.sizeStep('6X-Large', 1), null);
  assert.equal(c.sizeStep('nope', 1), null);
  assert.equal(c.sizeKeyword('X-Small'), 'XSMALL');
  assert.equal(c.sizeKeyword('2X-Large'), 'XXLARGE');
  assert.equal(c.sizeKeyword('4X-Large'), 'X4LARGE');
  assert.equal(c.sizeKeyword('?'), null);
});

test('alterWarehouseSql quotes names and only sets what changed', () => {
  assert.equal(c.alterWarehouseSql('COMPUTE_WH', { autoSuspend: 60 }), 'ALTER WAREHOUSE "COMPUTE_WH" SET AUTO_SUSPEND = 60;');
  assert.equal(c.alterWarehouseSql('we"ird', { size: 'Medium', autoSuspend: 120 }), 'ALTER WAREHOUSE "we""ird" SET WAREHOUSE_SIZE = MEDIUM AUTO_SUSPEND = 120;');
  assert.equal(c.alterWarehouseSql('X', {}), '');
  assert.equal(c.alterWarehouseSql('X', { maxClusters: 2 }), 'ALTER WAREHOUSE "X" SET MAX_CLUSTER_COUNT = 2;');
});

test('recommendations: long auto-suspend is the top saving', () => {
  const w = { name: 'WH', compute_credits: 100, idle_hour_credits: 40, exec_seconds: 1000, queued_seconds: 0,
    remote_spill_bytes: 0, utilization: 0.5, settings: { size: 'X-Small', auto_suspend: 600, auto_resume: 'true' } };
  const recs = c.recommendations(w, { price: 2, days: 30 });
  assert.equal(recs[0].key, 'auto_suspend');
  assert.equal(recs[0].severity, 'MEDIUM');
  near(recs[0].saveCredits, 36);
  near(recs[0].saveUsdMonth, 72);
  assert.match(recs[0].sql, /AUTO_SUSPEND = 60/);
  assert.match(recs[0].title, /600s → 60s/);
});

test('recommendations: never-suspending warehouse is HIGH and assumes an hour tail', () => {
  const w = { name: 'WH', compute_credits: 100, idle_hour_credits: 60, settings: { size: 'X-Small', auto_suspend: null } };
  const r = c.recommendations(w, { price: 1, days: 30 }).find((x) => x.key === 'auto_suspend');
  assert.equal(r.severity, 'HIGH');
  assert.match(r.title, /never/);
  near(r.saveCredits, 60 - 60 * (60 / 3600));
});

test('recommendations: awake, queueing, spill, size-down, auto-resume', () => {
  const awake = c.recommendations({ name: 'A', compute_credits: 10, idle_hour_credits: 5, settings: { size: 'Small', auto_suspend: 60 } });
  assert.ok(awake.some((r) => r.key === 'awake'));
  const q = c.recommendations({ name: 'Q', compute_credits: 10, queued_seconds: 5000, exec_seconds: 6000, settings: { size: 'Small', auto_suspend: 60, max_cluster_count: 1 } });
  const qr = q.find((r) => r.key === 'queueing');
  assert.equal(qr.severity, 'HIGH');
  assert.match(qr.sql, /MAX_CLUSTER_COUNT = 2/);
  const sp = c.recommendations({ name: 'S', compute_credits: 10, remote_spill_bytes: 5, utilization: 0.01, settings: { size: 'Medium', auto_suspend: 60 } });
  assert.match(sp.find((r) => r.key === 'spill').sql, /WAREHOUSE_SIZE = LARGE/);
  assert.ok(!sp.some((r) => r.key === 'size_down'), 'never size down a spilling warehouse');
  const down = c.recommendations({ name: 'D', compute_credits: 100, idle_hour_credits: 50, utilization: 0.05, settings: { size: 'Large', auto_suspend: 60 } });
  const d = down.find((r) => r.key === 'size_down');
  assert.match(d.sql, /WAREHOUSE_SIZE = MEDIUM/);
  near(d.saveCredits, 25);
  const small = c.recommendations({ name: 'X', utilization: 0.01, settings: { size: 'X-Small', auto_suspend: 60 } });
  assert.ok(!small.some((r) => r.key === 'size_down'));
  const ar = c.recommendations({ name: 'R', settings: { size: 'Small', auto_suspend: 60, auto_resume: 'false' } });
  assert.equal(ar[0].key, 'auto_resume');
  assert.deepEqual(c.recommendations(null), []);
  assert.deepEqual(c.recommendations({ name: 'nosettings', compute_credits: 5 }), []);
});

test('storageFix produces quoted retention and transient-swap SQL', () => {
  const f = c.storageFix({ path_to_object: 'DB.SC.T', database_name: 'DB', schema_name: 'SC', table_name: 'T', overhead_multiple: 2.5 });
  assert.equal(f.retention, 'ALTER TABLE "DB"."SC"."T" SET DATA_RETENTION_TIME_IN_DAYS = 1;');
  assert.match(f.transient, /CREATE TRANSIENT TABLE "DB"."SC"."T_TRANSIENT" AS SELECT \* FROM "DB"."SC"."T";/);
  assert.match(f.transient, /SWAP WITH/);
  assert.ok(f.flagged);
  assert.ok(!c.storageFix({ overhead_multiple: 0.5, path_to_object: 'a.b.c' }).flagged);
});

test('monitor thresholds and level', () => {
  assert.deepEqual(c.parsePercents('90%,75%'), [75, 90]);
  assert.deepEqual(c.parsePercents(null), []);
  const m = c.monitorStatus({ credit_quota: 200, used_credits: 160, notify_at: '75%', suspend_at: '100%', suspend_immediately_at: '110%' });
  near(m.usedPct, 80);
  assert.equal(m.level, 'warning');
  assert.deepEqual(m.suspend, [100, 110]);
  assert.equal(c.monitorStatus({ credit_quota: 100, used_credits: 101, suspend_at: '100%' }).level, 'critical');
  assert.equal(c.monitorStatus({ credit_quota: 0, used_credits: 5 }).level, 'ok');
});

test('sparkPath geometry', () => {
  assert.deepEqual(c.sparkPath([]), { line: '', area: '' });
  const one = c.sparkPath([5], 100, 20);
  assert.match(one.line, /^M2,10 L98,10$/);
  const p = c.sparkPath([0, 10], 100, 20, 0);
  assert.equal(p.line, 'M0,20 L100,0');
  assert.ok(p.area.endsWith('Z'));
});

test('serviceLabel, filterWarehouses and sortWarehouses', () => {
  assert.equal(c.serviceLabel('SNOWPARK_CONTAINER_SERVICES'), 'Snowpark container services');
  const list = [
    { name: 'B', compute_credits: 5, idle_hour_credits: 4, utilization: 0.2, settings: { state: 'STARTED', size: 'Small' } },
    { name: 'A', compute_credits: 50, idle_hour_credits: 1, utilization: null, settings: { state: 'SUSPENDED', size: 'X-Small' } },
    { name: 'C', compute_credits: 0, settings: {} },
  ];
  assert.deepEqual(c.filterWarehouses(list, { state: 'started' }).map((w) => w.name), ['B']);
  assert.deepEqual(c.filterWarehouses(list, { size: 'X-Small' }).map((w) => w.name), ['A']);
  assert.deepEqual(c.filterWarehouses(list, { q: 'c' }).map((w) => w.name), ['C']);
  assert.deepEqual(c.sortWarehouses(list, 'credits').map((w) => w.name), ['A', 'B', 'C']);
  assert.deepEqual(c.sortWarehouses(list, 'idle').map((w) => w.name), ['B', 'A', 'C']);
  assert.deepEqual(c.sortWarehouses(list, 'name').map((w) => w.name), ['A', 'B', 'C']);
  assert.equal(c.sortWarehouses(list, 'utilization')[0].name, 'B');
});
