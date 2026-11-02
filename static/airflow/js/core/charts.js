// Dependency-free SVG charts: tooltips, donut, sparkline, run bars, gantt,
// DAG graph (layered layout + pan/zoom), grid matrix and calendar heatmap.

import { h, s, icon, mount } from './dom.js';
import { stateInfo } from './ui.js';
import { fmtDate, fmtDuration, fmtShortDate, toDate, fmtDay } from './format.js';

// ------------------------------------------------------------------ tooltip

let tipEl;
function tipNode() {
  if (!tipEl) {
    tipEl = h('div.af-tooltip', { role: 'tooltip' });
    document.body.appendChild(tipEl);
  }
  return tipEl;
}
function placeTip(e) {
  const t = tipNode();
  const pad = 14;
  let x = e.clientX + pad;
  let y = e.clientY + pad;
  if (x + t.offsetWidth > window.innerWidth - 8) x = e.clientX - t.offsetWidth - pad;
  if (y + t.offsetHeight > window.innerHeight - 8) y = e.clientY - t.offsetHeight - pad;
  t.style.transform = `translate(${x}px, ${y}px)`;
}
/** Attach a rich tooltip; content() returns a node or string. */
export function tip(el, content) {
  el.addEventListener('mouseenter', (e) => {
    const t = tipNode();
    mount(t, typeof content === 'function' ? content() : content);
    t.classList.add('show');
    placeTip(e);
  });
  el.addEventListener('mousemove', placeTip);
  el.addEventListener('mouseleave', hideTip);
  return el;
}
export function hideTip() { tipEl?.classList.remove('show'); }

export function tipRows(title, rows) {
  return h('div',
    title && h('div.af-tip-title', title),
    rows.filter(Boolean).map(([k, v]) => h('div.af-tip-row', h('span', k), h('b', v ?? '—'))));
}

// ------------------------------------------------------------------ donut

/** donut([{label, value, color}], {size, thickness, centerLabel, centerSub}) */
export function donut(segments, { size = 150, thickness = 20, centerLabel, centerSub, onClick } = {}) {
  const total = segments.reduce((a, b) => a + b.value, 0);
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  const svg = s('svg', { viewBox: `0 0 ${size} ${size}`, width: size, height: size, class: 'af-donut', role: 'img' });
  svg.appendChild(s('circle', { cx: size / 2, cy: size / 2, r, fill: 'none', stroke: '#eef2f7', 'stroke-width': thickness }));
  let acc = 0;
  for (const seg of segments) {
    if (!seg.value) continue;
    const len = (seg.value / total) * c;
    const arc = s('circle', {
      cx: size / 2, cy: size / 2, r, fill: 'none', stroke: seg.color, 'stroke-width': thickness,
      'stroke-dasharray': `${Math.max(len - 2, 0.5)} ${c}`, 'stroke-dashoffset': -acc,
      transform: `rotate(-90 ${size / 2} ${size / 2})`, class: 'af-donut-seg',
      style: onClick ? 'cursor:pointer' : '',
    });
    tip(arc, () => tipRows(seg.label, [['Count', seg.value.toLocaleString()], ['Share', `${((seg.value / total) * 100).toFixed(1)}%`]]));
    if (onClick) arc.addEventListener('click', () => onClick(seg));
    svg.appendChild(arc);
    acc += len;
  }
  svg.appendChild(s('text', { x: '50%', y: '48%', 'text-anchor': 'middle', class: 'af-donut-value' }, centerLabel ?? total.toLocaleString()));
  if (centerSub) svg.appendChild(s('text', { x: '50%', y: '62%', 'text-anchor': 'middle', class: 'af-donut-sub' }, centerSub));
  return svg;
}

export function legend(items, { onClick } = {}) {
  return h('div.af-legend', items.map((it) => h(onClick ? 'button.af-legend-item' : 'span.af-legend-item', {
    type: onClick ? 'button' : undefined, onclick: onClick ? () => onClick(it) : undefined,
  }, h('span.af-dot', { style: { background: it.color } }), it.label, it.value != null && h('b', it.value.toLocaleString()))));
}

// ------------------------------------------------------------------ sparkline

export function sparkline(values, { width = 120, height = 28, color = '#0072CE' } = {}) {
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, width, height, class: 'af-spark' });
  if (values.length < 2) return svg;
  const max = Math.max(...values, 1);
  const step = width / (values.length - 1);
  const pts = values.map((v, i) => `${(i * step).toFixed(1)},${(height - 2 - (v / max) * (height - 4)).toFixed(1)}`);
  svg.appendChild(s('polyline', { points: `0,${height} ${pts.join(' ')} ${width},${height}`, fill: color, opacity: 0.12, stroke: 'none' }));
  svg.appendChild(s('polyline', { points: pts.join(' '), fill: 'none', stroke: color, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }));
  return svg;
}

// ------------------------------------------------------------------ run bars

/** Duration bar per run (oldest -> newest), colored by state. runs: [{dag_run_id, state, duration, start_date, end_date, run_after}] */
export function runBars(runs, { height = 140, onClick, selected } = {}) {
  const data = [...runs].reverse();
  const durs = data.map((r) => r.duration ?? (r.start_date ? ((toDate(r.end_date) || new Date()) - toDate(r.start_date)) / 1000 : 0));
  const max = Math.max(...durs, 1);
  const wrap = h('div.af-runbars', { style: { height: `${height}px` } });
  const axis = h('div.af-runbars-axis', h('span', fmtDuration(max)), h('span', fmtDuration(max / 2)), h('span', '0'));
  const bars = h('div.af-runbars-bars');
  data.forEach((r, i) => {
    const info = stateInfo(r.state);
    const bar = h('button.af-runbar', {
      type: 'button',
      class: [selected === r.dag_run_id && 'selected', r.state === 'running' && 'af-pulse'],
      style: { height: `${Math.max(3, (durs[i] / max) * 100)}%`, background: info.color },
      'aria-label': `${r.dag_run_id}: ${r.state}`,
      onclick: () => onClick?.(r),
    });
    tip(bar, () => tipRows(r.dag_run_id, [
      ['State', info.label], ['Run after', fmtDate(r.run_after)], ['Started', fmtDate(r.start_date)],
      ['Duration', fmtDuration(durs[i])], ['Type', r.run_type],
    ]));
    bars.appendChild(h('div.af-runbar-slot', bar));
  });
  wrap.append(axis, bars);
  return wrap;
}

// ------------------------------------------------------------------ gantt

/** Gantt of task instances. tis: [{task_id, map_index, state, queued_when, start_date, end_date, try_number}] */
export function gantt(tis, { onClick, rowHeight = 24 } = {}) {
  const rows = tis.filter((t) => t.start_date || t.queued_when)
    .sort((a, b) => (toDate(a.start_date || a.queued_when) - toDate(b.start_date || b.queued_when)));
  if (!rows.length) return h('div.af-empty.af-empty-sm', icon('bar-chart-steps', 'af-empty-icon'), h('div.af-empty-title', 'No task has started yet'));
  const now = new Date();
  const t0 = Math.min(...rows.map((t) => toDate(t.queued_when || t.start_date).getTime()));
  const t1 = Math.max(...rows.map((t) => (toDate(t.end_date) || now).getTime()));
  const span = Math.max(t1 - t0, 1000);
  const pct = (d) => (((toDate(d) || now).getTime() - t0) / span) * 100;

  const ticks = h('div.af-gantt-ticks', [0, 0.25, 0.5, 0.75, 1].map((f) =>
    h('span', { style: { left: `${f * 100}%` } }, fmtDuration((span / 1000) * f))));
  const body = h('div.af-gantt-body');
  for (const t of rows) {
    const info = stateInfo(t.state);
    const label = t.map_index >= 0 ? `${t.task_id} [${t.map_index}]` : t.task_id;
    const queuedLeft = t.queued_when ? pct(t.queued_when) : null;
    const startLeft = t.start_date ? pct(t.start_date) : null;
    const end = pct(t.end_date);
    const track = h('div.af-gantt-track');
    if (queuedLeft != null && (startLeft == null || startLeft > queuedLeft)) {
      track.appendChild(h('div.af-gantt-queued', { style: { left: `${queuedLeft}%`, width: `${Math.max((startLeft ?? end) - queuedLeft, 0.2)}%` } }));
    }
    if (startLeft != null) {
      const bar = h('div.af-gantt-bar', {
        class: t.state === 'running' && 'af-pulse',
        style: { left: `${startLeft}%`, width: `${Math.max(end - startLeft, 0.3)}%`, background: info.color },
      });
      track.appendChild(bar);
    }
    const row = h('div.af-gantt-row', { style: { height: `${rowHeight}px` }, onclick: () => onClick?.(t), class: onClick && 'clickable' },
      h('div.af-gantt-label', { title: label }, h('span.af-dot', { style: { background: info.color } }), label),
      track);
    tip(row, () => tipRows(label, [
      ['State', info.label], ['Queued', fmtDate(t.queued_when)], ['Started', fmtDate(t.start_date)],
      ['Ended', fmtDate(t.end_date)], ['Duration', fmtDuration(t.duration ?? (t.start_date ? ((toDate(t.end_date) || now) - toDate(t.start_date)) / 1000 : null))],
      ['Try', t.try_number],
    ]));
    body.appendChild(row);
  }
  return h('div.af-gantt', h('div.af-gantt-head', h('div.af-gantt-label', 'Task'), ticks), body);
}

// ------------------------------------------------------------------ DAG graph

const NODE_W = 210;
const NODE_H = 46;
const GAP_X = 70;
const GAP_Y = 18;

/** Layered (Sugiyama-style) layout: longest-path layering + barycenter ordering. */
export function layoutGraph(tasks) {
  const ids = tasks.map((t) => t.task_id);
  const known = new Set(ids);
  const down = new Map(ids.map((id) => [id, []]));
  const up = new Map(ids.map((id) => [id, []]));
  for (const t of tasks) {
    for (const d of t.downstream_task_ids || []) {
      if (!known.has(d)) continue;
      down.get(t.task_id).push(d);
      up.get(d).push(t.task_id);
    }
  }
  // longest path layering (Kahn)
  const indeg = new Map(ids.map((id) => [id, up.get(id).length]));
  const layer = new Map(ids.map((id) => [id, 0]));
  const queue = ids.filter((id) => indeg.get(id) === 0);
  const order = [];
  while (queue.length) {
    const id = queue.shift();
    order.push(id);
    for (const d of down.get(id)) {
      layer.set(d, Math.max(layer.get(d), layer.get(id) + 1));
      indeg.set(d, indeg.get(d) - 1);
      if (indeg.get(d) === 0) queue.push(d);
    }
  }
  ids.filter((id) => !order.includes(id)).forEach((id) => order.push(id)); // cycles (shouldn't happen)

  const layers = [];
  for (const id of order) (layers[layer.get(id)] ||= []).push(id);
  const pos = new Map();
  const reindex = () => layers.forEach((l) => l.forEach((id, i) => pos.set(id, i)));
  reindex();
  const bary = (id, nbrs) => {
    const ns = nbrs.get(id);
    return ns.length ? ns.reduce((a, n) => a + pos.get(n), 0) / ns.length : pos.get(id);
  };
  for (let pass = 0; pass < 6; pass++) {
    const downward = pass % 2 === 0;
    const seq = downward ? layers.slice(1) : layers.slice(0, -1).reverse();
    for (const l of seq) {
      const b = new Map(l.map((id) => [id, bary(id, downward ? up : down)]));
      l.sort((a, c) => b.get(a) - b.get(c));
      l.forEach((id, i) => pos.set(id, i));
    }
  }
  const maxLen = Math.max(...layers.map((l) => l.length), 1);
  const nodes = new Map();
  layers.forEach((l, li) => {
    const offset = ((maxLen - l.length) * (NODE_H + GAP_Y)) / 2;
    l.forEach((id, i) => nodes.set(id, { id, x: li * (NODE_W + GAP_X), y: offset + i * (NODE_H + GAP_Y), layer: li }));
  });
  const edges = [];
  for (const [id, ds] of down) ds.forEach((d) => edges.push({ from: id, to: d }));
  return {
    nodes, edges, up, down,
    width: layers.length * (NODE_W + GAP_X) - GAP_X,
    height: maxLen * (NODE_H + GAP_Y) - GAP_Y,
  };
}

function collect(start, map) {
  const seen = new Set();
  const stack = [start];
  while (stack.length) {
    const id = stack.pop();
    for (const n of map.get(id) || []) if (!seen.has(n)) { seen.add(n); stack.push(n); }
  }
  return seen;
}

/**
 * Interactive DAG graph.
 * dagGraph(tasks, { states: {task_id: state}, onNodeClick(task), selected, height })
 * returns {el, setStates(map), highlight(text)}
 */
export function dagGraph(tasks, { states = {}, onNodeClick, height = 460, selected } = {}) {
  if (!tasks.length) return { el: h('div.af-empty', h('div.af-empty-title', 'This DAG has no tasks')), setStates() {}, highlight() {} };
  const layout = layoutGraph(tasks);
  const byId = new Map(tasks.map((t) => [t.task_id, t]));
  const PAD = 30;
  const svg = s('svg', { class: 'af-graph-svg', width: '100%', height: '100%' });
  const defs = s('defs');
  defs.appendChild(s('marker', { id: 'af-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
    s('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#94a3b8' })));
  defs.appendChild(s('marker', { id: 'af-arrow-hl', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
    s('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#0072CE' })));
  svg.appendChild(defs);
  const world = s('g');
  svg.appendChild(world);

  const edgeEls = [];
  for (const e of layout.edges) {
    const a = layout.nodes.get(e.from);
    const b = layout.nodes.get(e.to);
    const x1 = a.x + NODE_W; const y1 = a.y + NODE_H / 2;
    const x2 = b.x - 2; const y2 = b.y + NODE_H / 2;
    const mx = (x1 + x2) / 2;
    const path = s('path', { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, class: 'af-edge', 'marker-end': 'url(#af-arrow)' });
    world.appendChild(path);
    edgeEls.push({ ...e, el: path });
  }

  const nodeEls = new Map();
  for (const [id, n] of layout.nodes) {
    const t = byId.get(id);
    const g = s('g', { class: 'af-node', transform: `translate(${n.x},${n.y})`, tabindex: 0, role: 'button', 'aria-label': id });
    const rect = s('rect', { width: NODE_W, height: NODE_H, rx: 8, class: 'af-node-rect' });
    const bar = s('rect', { width: 6, height: NODE_H, rx: 3, class: 'af-node-bar' });
    const label = s('text', { x: 16, y: 20, class: 'af-node-label' }, id.length > 26 ? `${id.slice(0, 25)}…` : id);
    const sub = s('text', { x: 16, y: 36, class: 'af-node-sub' }, [t.operator_name || '', t.is_mapped ? '· mapped' : ''].join(' '));
    const stateTxt = s('text', { x: NODE_W - 10, y: 36, 'text-anchor': 'end', class: 'af-node-state' }, '');
    g.append(rect, bar, label, sub, stateTxt);
    g.addEventListener('click', () => onNodeClick?.(t));
    g.addEventListener('keydown', (e) => { if (e.key === 'Enter') onNodeClick?.(t); });
    g.addEventListener('mouseenter', () => focusNode(id));
    g.addEventListener('mouseleave', () => focusNode(null));
    tip(g, () => tipRows(id, [
      ['Operator', t.operator_name], ['State', stateInfo(currentStates[id]).label],
      ['Upstream', layout.up.get(id).length], ['Downstream', layout.down.get(id).length],
      ['Trigger rule', t.trigger_rule], ['Retries', t.retries], ['Pool', t.pool],
    ]));
    world.appendChild(g);
    nodeEls.set(id, { g, bar, stateTxt });
  }

  let currentStates = {};
  function setStates(map) {
    currentStates = map || {};
    for (const [id, { g, bar, stateTxt }] of nodeEls) {
      const st = currentStates[id];
      const info = stateInfo(st);
      bar.setAttribute('fill', st ? info.color : '#cbd5e1');
      g.style.setProperty('--af-node-color', st ? info.color : '#cbd5e1');
      g.classList.toggle('af-node-running', st === 'running');
      stateTxt.textContent = st ? info.label : '';
      stateTxt.setAttribute('fill', info.color);
    }
  }
  function focusNode(id) {
    hideTip();
    if (!id) {
      nodeEls.forEach(({ g }) => g.classList.remove('dim', 'hl'));
      edgeEls.forEach(({ el }) => { el.classList.remove('dim', 'hl'); el.setAttribute('marker-end', 'url(#af-arrow)'); });
      return;
    }
    const related = new Set([id, ...collect(id, layout.up), ...collect(id, layout.down)]);
    nodeEls.forEach(({ g }, nid) => { g.classList.toggle('dim', !related.has(nid)); g.classList.toggle('hl', nid === id); });
    edgeEls.forEach(({ el, from, to }) => {
      const on = related.has(from) && related.has(to);
      el.classList.toggle('dim', !on);
      el.classList.toggle('hl', on);
      el.setAttribute('marker-end', on ? 'url(#af-arrow-hl)' : 'url(#af-arrow)');
    });
  }
  function highlight(text) {
    const q = (text || '').toLowerCase();
    nodeEls.forEach(({ g }, id) => g.classList.toggle('af-node-match', !!q && id.toLowerCase().includes(q)));
  }
  if (selected) nodeEls.get(selected)?.g.classList.add('af-node-selected');

  // ---- pan & zoom
  let view = { x: 0, y: 0, k: 1 };
  const apply = () => world.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`);
  const container = h('div.af-graph', { style: { height: `${height}px` } }, svg);
  function fit() {
    const w = container.clientWidth || 800;
    const hh = container.clientHeight || height;
    const k = Math.min((w - PAD * 2) / Math.max(layout.width, 1), (hh - PAD * 2) / Math.max(layout.height, 1), 1.4);
    view = { k, x: (w - layout.width * k) / 2, y: (hh - layout.height * k) / 2 };
    apply();
  }
  function zoom(factor, cx, cy) {
    const k = Math.min(Math.max(view.k * factor, 0.15), 3);
    const f = k / view.k;
    view = { k, x: cx - (cx - view.x) * f, y: cy - (cy - view.y) * f };
    apply();
  }
  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = svg.getBoundingClientRect();
    zoom(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  let drag = null;
  svg.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.af-node')) return;
    drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    svg.setPointerCapture(e.pointerId);
    container.classList.add('dragging');
  });
  svg.addEventListener('pointermove', (e) => {
    if (!drag) return;
    view.x = drag.vx + e.clientX - drag.x;
    view.y = drag.vy + e.clientY - drag.y;
    apply();
  });
  const endDrag = () => { drag = null; container.classList.remove('dragging'); };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);

  const controls = h('div.af-graph-controls',
    h('button.af-icon-btn', { title: 'Zoom in', onclick: () => zoom(1.25, container.clientWidth / 2, container.clientHeight / 2) }, icon('zoom-in')),
    h('button.af-icon-btn', { title: 'Zoom out', onclick: () => zoom(0.8, container.clientWidth / 2, container.clientHeight / 2) }, icon('zoom-out')),
    h('button.af-icon-btn', { title: 'Fit to view', onclick: fit }, icon('arrows-fullscreen')),
    h('button.af-icon-btn', {
      title: 'Download SVG',
      onclick: () => {
        const clone = svg.cloneNode(true);
        clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        clone.querySelector('g').setAttribute('transform', `translate(${PAD},${PAD})`);
        clone.setAttribute('viewBox', `0 0 ${layout.width + PAD * 2} ${layout.height + PAD * 2}`);
        const style = `<style>.af-node-rect{fill:#fff;stroke:#cbd5e1}.af-node-label{font:600 12px Inter,sans-serif;fill:#1e293b}.af-node-sub,.af-node-state{font:10px Inter,sans-serif;fill:#64748b}.af-edge{fill:none;stroke:#94a3b8;stroke-width:1.5}</style>`;
        const blob = new Blob([clone.outerHTML.replace('>', `>${style}`)], { type: 'image/svg+xml' });
        const a = h('a', { href: URL.createObjectURL(blob), download: 'dag-graph.svg' });
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      },
    }, icon('download')));
  container.appendChild(controls);
  const ro = new ResizeObserver(() => { if (!container.dataset.fitted && container.clientWidth) { container.dataset.fitted = '1'; fit(); } });
  ro.observe(container);
  setStates(states);
  return { el: container, setStates, highlight, fit };
}

// ------------------------------------------------------------------ grid matrix

/**
 * Airflow-style grid: rows = tasks, columns = runs (oldest -> newest).
 * gridMatrix({ runs, taskIds, cell(runId, taskId) -> {state, count, states} , onCell(run, taskId), onRun(run) })
 */
export function gridMatrix({ runs, taskIds, cell, onCell, onRun }) {
  const cols = [...runs].reverse();
  const table = h('div.af-grid', { style: { '--af-grid-cols': cols.length } });
  // header: run duration bars
  const durs = cols.map((r) => r.duration ?? 0);
  const max = Math.max(...durs, 1);
  table.appendChild(h('div.af-grid-row.af-grid-head',
    h('div.af-grid-label', h('span.text-muted.small', `${cols.length} runs →`)),
    h('div.af-grid-cells', cols.map((r, i) => {
      const info = stateInfo(r.state);
      const b = h('button.af-grid-runbar', {
        type: 'button', onclick: () => onRun?.(r),
        style: { height: `${Math.max(4, (durs[i] / max) * 56)}px`, background: info.color },
      });
      tip(b, () => tipRows(r.dag_run_id, [['State', info.label], ['Run after', fmtShortDate(r.run_after)], ['Duration', fmtDuration(r.duration)]]));
      return h('div.af-grid-col', b);
    }))));
  for (const taskId of taskIds) {
    table.appendChild(h('div.af-grid-row',
      h('div.af-grid-label', { title: taskId }, taskId),
      h('div.af-grid-cells', cols.map((r) => {
        const c = cell(r.dag_run_id, taskId);
        const info = stateInfo(c?.state);
        const sq = h('button.af-grid-cell', {
          type: 'button',
          class: [!c && 'empty', c?.state === 'running' && 'af-pulse'],
          style: c ? { background: info.color } : null,
          onclick: () => c && onCell?.(r, taskId),
        }, c?.count > 1 ? h('span.af-grid-count', String(c.count)) : null);
        if (c) {
          tip(sq, () => tipRows(taskId, [
            ['Run', r.dag_run_id], ['State', info.label],
            c.count > 1 && ['Mapped', `${c.count} instances`],
            c.duration != null && ['Duration', fmtDuration(c.duration)],
            c.try_number && ['Try', c.try_number],
          ]));
        }
        return h('div.af-grid-col', sq);
      }))));
  }
  return h('div.af-grid-scroll', table);
}

// ------------------------------------------------------------------ calendar heatmap

/** days: Map('YYYY-MM-DD' -> {success, failed, other}) over the last `weeks` weeks. */
export function calendarHeatmap(days, { weeks = 53, onDay } = {}) {
  const cell = 12;
  const gap = 3;
  const end = new Date();
  end.setHours(0, 0, 0, 0);
  const start = new Date(end);
  start.setDate(start.getDate() - (weeks * 7 - 1) - start.getDay());
  const svg = s('svg', { class: 'af-cal', viewBox: `0 0 ${weeks * (cell + gap) + 30} ${7 * (cell + gap) + 20}` });
  const maxTotal = Math.max(1, ...[...days.values()].map((d) => d.success + d.failed + d.other));
  ['Mon', 'Wed', 'Fri'].forEach((d, i) => svg.appendChild(s('text', { x: 0, y: 20 + (i * 2 + 1) * (cell + gap) + 9, class: 'af-cal-lbl' }, d)));
  let lastMonth = -1;
  for (let d = new Date(start), i = 0; d <= end; d.setDate(d.getDate() + 1), i++) {
    const w = Math.floor(i / 7);
    const dow = d.getDay();
    const key = fmtDay(d);
    const v = days.get(key);
    const total = v ? v.success + v.failed + v.other : 0;
    let fill = '#eef2f7';
    if (v) {
      if (v.failed) fill = v.failed >= total / 2 ? '#dc2626' : '#f97316';
      else fill = `rgba(22,163,74,${0.3 + 0.7 * (total / maxTotal)})`;
    }
    const x = 30 + w * (cell + gap);
    if (d.getMonth() !== lastMonth && dow === 0) {
      lastMonth = d.getMonth();
      svg.appendChild(s('text', { x, y: 10, class: 'af-cal-lbl' }, d.toLocaleString(undefined, { month: 'short' })));
    }
    const rect = s('rect', { x, y: 18 + dow * (cell + gap), width: cell, height: cell, rx: 2, fill, class: v ? 'af-cal-day has' : 'af-cal-day' });
    const dayCopy = new Date(d);
    tip(rect, () => tipRows(dayCopy.toDateString(), v ? [['Runs', total], ['Success', v.success], ['Failed', v.failed], ['Other', v.other]] : [['Runs', 0]]));
    if (v && onDay) rect.addEventListener('click', () => onDay(key));
    svg.appendChild(rect);
  }
  return h('div.af-cal-wrap', svg);
}

// ------------------------------------------------------------------ stacked bar (state breakdown)

export function stateBar(counts, { height = 8 } = {}) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const bar = h('div.af-statebar', { style: { height: `${height}px` } });
  if (!total) { bar.classList.add('empty'); return bar; }
  for (const [state, n] of Object.entries(counts)) {
    if (!n) continue;
    const seg = h('span', { style: { width: `${(n / total) * 100}%`, background: stateInfo(state).color } });
    tip(seg, () => tipRows(stateInfo(state).label, [['Count', n], ['Share', `${((n / total) * 100).toFixed(1)}%`]]));
    bar.appendChild(seg);
  }
  return bar;
}
