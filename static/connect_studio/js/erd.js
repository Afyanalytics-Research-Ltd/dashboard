// Interactive ERD: force layout for related tables, grid for loners, column-anchored edges,
// pan / zoom / drag, neighbour highlighting, declared vs inferred relationships.

import { h, s, icon, mount } from '../../ui/js/dom.js';

const W = 230;
const HEAD = 34;
const ROW = 19;
const MAX_COLS = 9;

function visibleColumns(t, related) {
  const cols = t.columns;
  const score = (c) => (c.pk ? 0 : related.has(c.name) ? 1 : 2);
  const sorted = [...cols].sort((a, b) => score(a) - score(b) || a.position - b.position);
  return { shown: sorted.slice(0, MAX_COLS), more: Math.max(0, cols.length - MAX_COLS) };
}

function layout(nodes, edges) {
  // deterministic pseudo-random
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const linked = new Set(edges.flatMap((e) => [e.a, e.b]));
  const conn = nodes.filter((n) => linked.has(n.key));
  const lone = nodes.filter((n) => !linked.has(n.key));
  const area = Math.max(1, conn.length) * 260 * 230;
  const k = Math.sqrt(area / Math.max(1, conn.length));
  conn.forEach((n, i) => {
    const a = (i / Math.max(1, conn.length)) * Math.PI * 2;
    n.x = Math.cos(a) * k * 2 + rnd() * 10;
    n.y = Math.sin(a) * k * 2 + rnd() * 10;
  });
  const idx = Object.fromEntries(conn.map((n) => [n.key, n]));
  let temp = k * 2;
  const iters = conn.length > 120 ? 160 : 320;
  for (let it = 0; it < iters; it++) {
    for (const n of conn) { n.dx = 0; n.dy = 0; }
    for (let i = 0; i < conn.length; i++) {
      for (let j = i + 1; j < conn.length; j++) {
        const a = conn[i]; const b = conn[j];
        let dx = a.x - b.x; let dy = a.y - b.y;
        let d = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const minD = (a.r + b.r) * 1.2;
        const f = (k * k) / d + (d < minD ? (minD - d) * 4 : 0);
        dx /= d; dy /= d;
        a.dx += dx * f; a.dy += dy * f; b.dx -= dx * f; b.dy -= dy * f;
      }
    }
    for (const e of edges) {
      const a = idx[e.a]; const b = idx[e.b];
      if (!a || !b || a === b) continue;
      let dx = a.x - b.x; let dy = a.y - b.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (d * d) / k;
      dx /= d; dy /= d;
      a.dx -= dx * f; a.dy -= dy * f; b.dx += dx * f; b.dy += dy * f;
    }
    for (const n of conn) {
      n.dx -= n.x * 0.02; n.dy -= n.y * 0.02; // gravity
      const d = Math.sqrt(n.dx * n.dx + n.dy * n.dy) || 0.01;
      n.x += (n.dx / d) * Math.min(d, temp);
      n.y += (n.dy / d) * Math.min(d, temp);
    }
    temp *= 0.985;
  }
  // normalize: top-left at 40,40
  let minX = Infinity; let minY = Infinity; let maxY = -Infinity; let maxX = -Infinity;
  for (const n of conn) { minX = Math.min(minX, n.x - n.w / 2); minY = Math.min(minY, n.y - n.h / 2); maxY = Math.max(maxY, n.y + n.h / 2); maxX = Math.max(maxX, n.x + n.w / 2); }
  if (!conn.length) { minX = 0; minY = 0; maxY = 0; maxX = 900; }
  for (const n of conn) { n.x = n.x - n.w / 2 - minX + 40; n.y = n.y - n.h / 2 - minY + 40; }
  // loners in a grid below
  const cols = Math.max(3, Math.floor(Math.max(900, maxX - minX) / (W + 40)));
  let y0 = conn.length ? maxY - minY + 110 : 40;
  let rowH = 0;
  lone.forEach((n, i) => {
    const c = i % cols;
    if (c === 0 && i) { y0 += rowH + 40; rowH = 0; }
    n.x = 40 + c * (W + 40);
    n.y = y0;
    rowH = Math.max(rowH, n.h);
  });
}

/** erd(snapshot, { onOpen(tableKey), height }) → element */
export function erd(snapshot, { onOpen, height = 640 } = {}) {
  let showInferred = true;
  const rels = snapshot.relationships || [];
  const relatedCols = {};
  for (const r of rels) {
    (relatedCols[r.from] ||= new Set()); r.from_columns.forEach((c) => relatedCols[r.from].add(c));
    (relatedCols[r.to] ||= new Set()); r.to_columns.forEach((c) => relatedCols[r.to].add(c));
  }
  const nodes = snapshot.tables.map((t) => {
    const { shown, more } = visibleColumns(t, relatedCols[t.key] || new Set());
    const hgt = HEAD + shown.length * ROW + (more ? ROW : 0) + 8;
    return { key: t.key, t, shown, more, w: W, h: hgt, r: Math.max(W, hgt) / 2 };
  });
  const byKey = Object.fromEntries(nodes.map((n) => [n.key, n]));
  const edges = rels.filter((r) => byKey[r.from] && byKey[r.to]).map((r) => ({ a: r.from, b: r.to, r }));
  layout(nodes, edges);

  const view = s('g');
  const edgeLayer = s('g');
  const nodeLayer = s('g');
  view.append(edgeLayer, nodeLayer);
  const svg = s('svg', {}, s('defs',
    s('marker', { id: 'cs-erd-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
      s('path', { d: 'M0,0 L10,5 L0,10 z', fill: '#94a3b8' }))), view);
  let tx = 0; let ty = 0; let scale = 1;
  const apply = () => view.setAttribute('transform', `translate(${tx},${ty}) scale(${scale})`);

  const colY = (n, col) => {
    const i = n.shown.findIndex((c) => c.name === col);
    return n.y + HEAD + (i < 0 ? n.shown.length * ROW / 2 : i * ROW + ROW / 2) + 4;
  };
  const edgeEls = [];
  function drawEdges() {
    mount(edgeLayer);
    edgeEls.length = 0;
    for (const e of edges) {
      if (!showInferred && e.r.kind === 'inferred') continue;
      const a = byKey[e.a]; const b = byKey[e.b];
      const ya = colY(a, e.r.from_columns[0]);
      const yb = colY(b, e.r.to_columns[0]);
      let d;
      if (a === b) {
        d = `M${a.x + a.w},${ya} C${a.x + a.w + 50},${ya} ${a.x + a.w + 50},${yb} ${a.x + a.w},${yb}`;
      } else {
        const aRight = a.x + a.w / 2 < b.x + b.w / 2;
        const x1 = aRight ? a.x + a.w : a.x;
        const x2 = aRight ? b.x : b.x + b.w;
        const dx = Math.max(40, Math.abs(x2 - x1) / 2);
        d = `M${x1},${ya} C${x1 + (aRight ? dx : -dx)},${ya} ${x2 + (aRight ? -dx : dx)},${yb} ${x2},${yb}`;
      }
      // full class list: for SVG a `class` attr replaces the classes from the tag
      const p = s('path', { d, class: ['cs-erd-edge', e.r.kind === 'inferred' && 'inferred'], 'marker-end': 'url(#cs-erd-arrow)' },
        s('title', `${e.a}.${e.r.from_columns.join(',')} → ${e.b}.${e.r.to_columns.join(',')} · ${e.r.kind}${e.r.kind === 'inferred' ? ` (${Math.round(e.r.confidence * 100)}%)` : ''}`));
      edgeEls.push({ el: p, e });
      edgeLayer.appendChild(p);
    }
  }

  const nodeEls = {};
  for (const n of nodes) {
    const g = s('g.cs-erd-node', { transform: `translate(${n.x},${n.y})` },
      s('rect.box', { width: n.w, height: n.h, rx: 10 }),
      s('path.hdr', { d: `M0,10 a10,10 0 0 1 10,-10 h${n.w - 20} a10,10 0 0 1 10,10 v${HEAD - 10} h-${n.w} z` }),
      s('text.title', { x: 12, y: 21 }, n.key.length > 24 ? `${n.key.slice(0, 23)}…` : n.key),
      n.t.rows != null && s('text.rows', { x: n.w - 10, y: 21, 'text-anchor': 'end' }, n.t.rows >= 1000 ? `${(n.t.rows / 1000).toFixed(n.t.rows >= 1e5 ? 0 : 1)}k` : String(n.t.rows)),
      n.shown.map((c, i) => s('g',
        s('text.col', { x: 12, y: HEAD + i * ROW + 16 }, `${c.pk ? '🔑 ' : c.references ? '↗ ' : ''}${c.name.length > 20 ? `${c.name.slice(0, 19)}…` : c.name}`),
        s('text.ctype', { x: n.w - 10, y: HEAD + i * ROW + 16, 'text-anchor': 'end' }, (c.type || c.family || '').toLowerCase().slice(0, 12)))),
      n.more && s('text.more', { x: 12, y: HEAD + n.shown.length * ROW + 15 }, `+${n.more} more columns`));
    nodeEls[n.key] = g;
    nodeLayer.appendChild(g);
    g.addEventListener('mouseenter', () => highlight(n.key));
    g.addEventListener('mouseleave', () => highlight(null));
    g.addEventListener('dblclick', () => onOpen?.(n.key));
    // drag node
    g.addEventListener('pointerdown', (ev) => {
      ev.stopPropagation();
      const start = { x: ev.clientX, y: ev.clientY, nx: n.x, ny: n.y };
      let moved = false;
      const move = (e2) => {
        moved = true;
        n.x = start.nx + (e2.clientX - start.x) / scale;
        n.y = start.ny + (e2.clientY - start.y) / scale;
        g.setAttribute('transform', `translate(${n.x},${n.y})`);
        drawEdges();
        highlight(n.key);
      };
      const up = () => {
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        if (!moved) onOpen?.(n.key);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    });
  }

  function highlight(key) {
    const neigh = new Set([key]);
    if (key) for (const { e } of edgeEls) { if (e.a === key) neigh.add(e.b); if (e.b === key) neigh.add(e.a); }
    for (const [k, g] of Object.entries(nodeEls)) {
      g.classList.toggle('hl', !!key && k === key);
      g.classList.toggle('dim', !!key && !neigh.has(k));
    }
    for (const { el, e } of edgeEls) {
      const on = key && (e.a === key || e.b === key);
      el.classList.toggle('hl', !!on);
      el.classList.toggle('dim', !!key && !on);
    }
  }

  const wrap = h('div.cs-erd', { style: { height: `${height}px` } }, svg);
  function fit() {
    const r = wrap.getBoundingClientRect();
    if (!r.width) return;
    let maxX = 0; let maxY = 0;
    for (const n of nodes) { maxX = Math.max(maxX, n.x + n.w); maxY = Math.max(maxY, n.y + n.h); }
    scale = Math.min(1.25, Math.max(0.15, Math.min((r.width - 40) / (maxX + 40), (r.height - 40) / (maxY + 40))));
    tx = (r.width - (maxX + 40) * scale) / 2;
    ty = Math.max(10, (r.height - (maxY + 40) * scale) / 2);
    apply();
  }
  // pan & zoom
  wrap.addEventListener('pointerdown', (ev) => {
    const start = { x: ev.clientX, y: ev.clientY, tx, ty };
    wrap.classList.add('dragging');
    const move = (e2) => { tx = start.tx + e2.clientX - start.x; ty = start.ty + e2.clientY - start.y; apply(); };
    const up = () => { wrap.classList.remove('dragging'); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });
  wrap.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const r = wrap.getBoundingClientRect();
    const mx = ev.clientX - r.left; const my = ev.clientY - r.top;
    const f = Math.exp(-ev.deltaY * 0.0015);
    const ns = Math.min(2.5, Math.max(0.1, scale * f));
    tx = mx - ((mx - tx) * ns) / scale;
    ty = my - ((my - ty) * ns) / scale;
    scale = ns;
    apply();
  }, { passive: false });
  const zoom = (f) => {
    const r = wrap.getBoundingClientRect();
    const mx = r.width / 2; const my = r.height / 2;
    const ns = Math.min(2.5, Math.max(0.1, scale * f));
    tx = mx - ((mx - tx) * ns) / scale; ty = my - ((my - ty) * ns) / scale; scale = ns; apply();
  };
  const inferredBtn = h('button.cs-icon-btn.on', { type: 'button', title: 'Show inferred relationships', onclick: () => {
    showInferred = !showInferred; inferredBtn.classList.toggle('on', showInferred); drawEdges();
  } }, icon('stars'));
  wrap.append(
    h('div.cs-erd-tools',
      h('button.cs-icon-btn', { type: 'button', title: 'Zoom in', onclick: (e) => { e.stopPropagation(); zoom(1.25); } }, icon('plus-lg')),
      h('button.cs-icon-btn', { type: 'button', title: 'Zoom out', onclick: (e) => { e.stopPropagation(); zoom(0.8); } }, icon('dash-lg')),
      h('button.cs-icon-btn', { type: 'button', title: 'Fit to screen', onclick: (e) => { e.stopPropagation(); fit(); } }, icon('arrows-fullscreen')),
      inferredBtn),
    h('div.cs-erd-legend', h('span', h('i'), 'declared FK'), h('span', h('i.inf'), 'inferred by dbatlas'), h('span', 'drag tables · scroll to zoom · click to open')));
  wrap.querySelectorAll('.cs-erd-tools button').forEach((b) => b.addEventListener('pointerdown', (e) => e.stopPropagation()));
  drawEdges();
  apply();
  requestAnimationFrame(() => requestAnimationFrame(fit));
  return wrap;
}
