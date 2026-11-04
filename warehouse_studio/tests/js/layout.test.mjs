import { test } from 'node:test';
import assert from 'node:assert/strict';
import { layoutDag, crossings, edgeWidth } from '../../../static/warehouse_studio/js/lib/layout.js';

// Plan shape: Result <- Aggregate <- Join <- (Filter <- ScanA, ScanB)
const nodes = ['res', 'agg', 'join', 'filter', 'scanA', 'scanB'].map((id) => ({ id }));
const edges = [
  { from: 'agg', to: 'res' }, { from: 'join', to: 'agg' }, { from: 'filter', to: 'join' },
  { from: 'scanB', to: 'join' }, { from: 'scanA', to: 'filter' },
];

test('root on top, leaves below, longest path layering', () => {
  const l = layoutDag(nodes, edges);
  const layerOf = (id) => l.positions.get(id).layer;
  assert.equal(layerOf('res'), 0);
  assert.equal(layerOf('agg'), 1);
  assert.equal(layerOf('join'), 2);
  assert.equal(layerOf('filter'), 3);
  assert.equal(layerOf('scanB'), 3);
  assert.equal(layerOf('scanA'), 4);
  for (const e of edges) assert.ok(l.positions.get(e.from).y > l.positions.get(e.to).y, `${e.from} below ${e.to}`);
});

test('no overlaps within a layer and the canvas covers every node', () => {
  const l = layoutDag(nodes, edges, { nodeW: 100, nodeH: 40, gapX: 10, gapY: 20, pad: 5 });
  for (const layer of l.layers) {
    const xs = layer.map((id) => l.positions.get(id).x).sort((a, b) => a - b);
    for (let i = 1; i < xs.length; i += 1) assert.ok(xs[i] - xs[i - 1] >= 110);
  }
  for (const p of l.positions.values()) {
    assert.ok(p.x >= 0 && p.x + 100 <= l.width + 0.001);
    assert.ok(p.y >= 0 && p.y + 40 <= l.height + 0.001);
  }
});

test('barycenter sweeps remove avoidable crossings', () => {
  // Two independent chains whose leaves are listed in the "wrong" order.
  const e = [{ from: 'a2', to: 'r2' }, { from: 'a1', to: 'r1' }];
  const l = layoutDag([{ id: 'r1' }, { id: 'r2' }, { id: 'a2' }, { id: 'a1' }], e);
  assert.equal(crossings(l, e), 0);
});

test('a crossing fixture is detected by crossings()', () => {
  const e = [{ from: 'a', to: 'r2' }, { from: 'b', to: 'r1' }];
  const fake = { positions: new Map([
    ['r1', { layer: 0, order: 0 }], ['r2', { layer: 0, order: 1 }], ['a', { layer: 1, order: 0 }], ['b', { layer: 1, order: 1 }],
  ]) };
  assert.equal(crossings(fake, e), 1);
});

test('ignores unknown endpoints, self loops and cycles', () => {
  const l = layoutDag([{ id: 'a' }, { id: 'b' }], [{ from: 'a', to: 'zzz' }, { from: 'a', to: 'a' }, { from: 'a', to: 'b' }, { from: 'b', to: 'a' }]);
  assert.equal(l.positions.size, 2);
  assert.ok(Number.isFinite(l.height));
});

test('empty graph', () => {
  const l = layoutDag([], []);
  assert.equal(l.positions.size, 0);
  assert.ok(l.width > 0);
});

test('edgeWidth is monotone and bounded', () => {
  assert.equal(edgeWidth(0, 1000), 1.5);
  assert.ok(edgeWidth(10, 1000) < edgeWidth(1000, 1000));
  assert.ok(Math.abs(edgeWidth(1000, 1000) - 9) < 1e-9);
  assert.ok(edgeWidth(5, 0) <= 9);
});
