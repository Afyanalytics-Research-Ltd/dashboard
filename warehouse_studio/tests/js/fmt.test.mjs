import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fmt from '../../../static/warehouse_studio/js/lib/fmt.js';

test('fmtInt / fmtNum handle junk', () => {
  assert.equal(fmt.fmtInt(1234.6), '1,235');
  assert.equal(fmt.fmtInt(null), '—');
  assert.equal(fmt.fmtInt('abc'), '—');
  assert.equal(fmt.fmtNum(1.23456), '1.23');
  assert.equal(fmt.fmtNum(5), '5');
  assert.equal(fmt.num('', 7), 7);
});

test('compact', () => {
  assert.equal(fmt.compact(999), '999');
  assert.equal(fmt.compact(1234), '1.2K');
  assert.equal(fmt.compact(3_400_000), '3.4M');
  assert.equal(fmt.compact(2_000_000_000), '2B');
  assert.equal(fmt.compact(1.5), '1.5');
  assert.equal(fmt.compact(-2500), '-2.5K');
});

test('credits precision ladder', () => {
  assert.equal(fmt.credits(0), '0');
  assert.equal(fmt.credits(0.0004), '<0.001');
  assert.equal(fmt.credits(0.1234), '0.123');
  assert.equal(fmt.credits(12.345), '12.35');
  assert.equal(fmt.credits(1234.4), '1,234');
});

test('usd', () => {
  assert.equal(fmt.usd(0), '$0.00');
  assert.equal(fmt.usd(0.004), '<$0.01');
  assert.equal(fmt.usd(12.5), '$12.50');
  assert.equal(fmt.usd(12345.67), '$12,346');
  assert.equal(fmt.usd(-3), '-$3.00');
  assert.equal(fmt.usd(undefined), '—');
});

test('bytes', () => {
  assert.equal(fmt.bytes(512), '512 B');
  assert.equal(fmt.bytes(1536), '1.5 KB');
  assert.equal(fmt.bytes(5 * 1024 ** 3), '5 GB');
  assert.equal(fmt.bytes(1024 ** 5 * 3), '3 PB');
});

test('ms durations', () => {
  assert.equal(fmt.ms(850), '850 ms');
  assert.equal(fmt.ms(4200), '4.2 s');
  assert.equal(fmt.ms(65_000), '1m 05s');
  assert.equal(fmt.ms(2 * 3600_000 + 7 * 60_000), '2h 07m');
  assert.equal(fmt.ms(3 * 86400_000 + 4 * 3600_000), '3d 4h');
  assert.equal(fmt.ms(-5), '0 ms');
  assert.equal(fmt.seconds(90), '1m 30s');
});

test('pct and relTime', () => {
  assert.equal(fmt.pct(0.256), '26%');
  assert.equal(fmt.pct(0.256, 1), '25.6%');
  const now = Date.parse('2026-11-04T12:00:00Z');
  assert.equal(fmt.relTime('2026-11-04T11:59:50Z', now), 'just now');
  assert.equal(fmt.relTime('2026-11-04T11:50:00Z', now), '10 min ago');
  assert.equal(fmt.relTime('2026-11-04T09:00:00Z', now), '3 hr ago');
  assert.equal(fmt.relTime('2026-11-01T12:00:00Z', now), '3 days ago');
  assert.equal(fmt.relTime('2026-11-04T15:00:00Z', now), 'in 3 hr');
  assert.equal(fmt.relTime(null, now), '—');
});

test('severity and sizes', () => {
  assert.deepEqual(['LOW', 'CRITICAL', 'x', 'HIGH'].sort((a, b) => fmt.sevRank(a) - fmt.sevRank(b)), ['CRITICAL', 'HIGH', 'LOW', 'x']);
  assert.equal(fmt.creditsPerHour('X-Small'), 1);
  assert.equal(fmt.creditsPerHour('Large'), 8);
  assert.equal(fmt.creditsPerHour('2X-Large'), 32);
  assert.equal(fmt.creditsPerHour('weird'), 0);
  assert.equal(fmt.detectorLabel('sp_detect_long_compilation'), 'Long compilation');
  assert.equal(fmt.plural(1, 'finding'), '1 finding');
  assert.equal(fmt.plural(3, 'finding'), '3 findings');
  assert.equal(fmt.oneLine('select\n  *\n from x', 100), 'select * from x');
  assert.equal(fmt.truncate('abcdef', 4), 'abc…');
});
