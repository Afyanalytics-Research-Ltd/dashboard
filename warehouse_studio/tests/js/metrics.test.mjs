import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as m from '../../../static/warehouse_studio/js/lib/metrics.js';

test('timeBreakdown shares sum to 1 and drop zero parts', () => {
  const parts = m.timeBreakdown({ compilation_time: 100, execution_time: 300, queued_overload_time: 0 });
  assert.deepEqual(parts.map((p) => p.label), ['Compile', 'Execute']);
  assert.equal(parts.reduce((a, p) => a + p.share, 0), 1);
  assert.equal(parts[1].share, 0.75);
  assert.deepEqual(m.timeBreakdown({}), []);
  assert.deepEqual(m.timeBreakdown(null), []);
});

test('healthScore matches the server curve', () => {
  assert.equal(m.healthScore({}), 100);
  assert.equal(m.healthScore({ CRITICAL: 0, HIGH: 0 }), 100);
  assert.equal(m.healthScore({ CRITICAL: 1 }), Math.round(100 / (1 + 10 / 25)));
  assert.ok(m.healthScore({ HIGH: 10 }) > m.healthScore({ HIGH: 20 }));
  assert.ok(m.healthScore({ LOW: 1000 }) > 0);
  assert.equal(m.healthScore({ high: 5 }), m.healthScore({ HIGH: 5 }));
});

test('fixesToReach plans critical first and reports reachability', () => {
  const res = m.fixesToReach({ CRITICAL: 2, HIGH: 5, MEDIUM: 10 }, 80);
  assert.equal(res.plan[0].severity, 'CRITICAL');
  assert.ok(res.reachable);
  assert.ok(res.score >= 80);
  const none = m.fixesToReach({}, 80);
  assert.deepEqual(none.plan, []);
  assert.ok(none.reachable);
});

test('rankFindings orders by severity, then money, then time', () => {
  const rows = [
    { id: 1, severity: 'LOW', est_usd: 100 },
    { id: 2, severity: 'CRITICAL', est_usd: 1 },
    { id: 3, severity: 'HIGH', est_usd: 5, wasted_seconds: 10 },
    { id: 4, severity: 'HIGH', est_usd: 5, wasted_seconds: 99 },
    { id: 5, severity: 'HIGH', est_usd: 50 },
  ];
  assert.deepEqual(m.rankFindings(rows).map((r) => r.id), [2, 5, 4, 3, 1]);
  assert.deepEqual(m.rankFindings(null), []);
});

test('whatIf: shorter auto-suspend shrinks idle burn only', () => {
  const w = { compute_credits: 100, idle_hour_credits: 40, settings: { size: 'X-Small', auto_suspend: 600 } };
  const r = m.whatIf(w, { autoSuspend: 60 });
  assert.equal(r.busy, 60);
  assert.ok(Math.abs(r.idleProjected - 4) < 1e-9);
  assert.ok(Math.abs(r.projected - 64) < 1e-9);
  assert.ok(Math.abs(r.saved - 36) < 1e-9);
  assert.ok(r.savedRatio > 0.35);
});

test('whatIf: sizing up keeps busy credits, doubles idle, halves runtime', () => {
  const w = { compute_credits: 100, idle_hour_credits: 10, settings: { size: 'Small', auto_suspend: 60 } };
  const r = m.whatIf(w, { size: 'Medium' });
  assert.equal(r.runtimeFactor, 0.5);
  assert.ok(Math.abs(r.idleProjected - 20) < 1e-9);
  assert.ok(Math.abs(r.projected - 110) < 1e-9);
});

test('whatIf: auto_suspend floors at 60 and missing inputs are safe', () => {
  const r = m.whatIf({ compute_credits: 10, idle_hour_credits: 50, settings: { auto_suspend: 0 } }, { autoSuspend: 1 });
  assert.equal(r.newAutoSuspend, 60);
  assert.ok(r.idleCurrent <= 10);
  const z = m.whatIf(undefined, {});
  assert.equal(z.current, 0);
  assert.equal(z.savedRatio, 0);
});

test('projectMonth uses the last 7 complete days', () => {
  const today = new Date(Date.UTC(2026, 10, 10, 12)); // Nov 10 2026
  const daily = [];
  for (let d = 1; d <= 10; d += 1) daily.push({ day: `2026-11-${String(d).padStart(2, '0')}`, credits: d <= 9 ? 2 : 1 });
  const p = m.projectMonth(daily, today);
  assert.equal(p.mtd, 19);
  assert.equal(p.rate, 2);
  assert.equal(p.days, 30);
  assert.equal(p.remaining, 20);
  assert.equal(p.projected, 19 + 40);
  assert.equal(p.points.length, 20);
  assert.equal(p.points[0].x, '2026-11-11');
});

test('savingsByDetector aggregates and sorts', () => {
  const s = m.savingsByDetector([
    { detector: 'a', est_usd: 1, wasted_seconds: 5 }, { detector: 'b', est_usd: 3 }, { detector: 'a', est_usd: 4 },
  ]);
  assert.deepEqual(s.map((x) => [x.detector, x.findings, x.usd]), [['a', 2, 5], ['b', 1, 3]]);
});
