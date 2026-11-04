// Workflow canvas: HTML nodes on a pan/zoom layer with SVG connections.
//
//   const cv = createCanvas(el, { readOnly, onChange(wf, meta), onOpenNode(name), onAddRequest(req),
//                                 onSelect(names), onContextMenu(e, target) })
//   cv.setWorkflow(wf); cv.fitView(); cv.setStatus(nodeStatusMap, edgeCounts)
//
// The page owns the workflow; the canvas emits immutable updates through onChange.

import { h, s, icon, mount } from '../../../ui/js/dom.js';
import { nodeIcon, entryFor, displayName } from '../core/catalog.js';
import { edges as edgeList, isSticky, nodeSize, bounds, snap, addConnection, removeConnection, NODE_W, NODE_H, isTriggerType } from '../core/workflow.js';
import { subtitle as nodeSubtitle } from '../core/params.js';

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 2.5;
const PORT_GAP = 22;

// ------------------------------------------------------------------ ports

const asType = (p) => (typeof p === 'string' ? p : p?.type || 'main');

/** Port lists for a node: {inputs: [{type, label, index}], outputs: [...]} */
export function portsFor(node, wf) {
  const entry = entryFor(node.type);
  const list = edgeList(wf);
  const inbound = list.filter((e) => e.to === node.name);
  const outbound = list.filter((e) => e.from === node.name);
  const resolve = (decl, connected, isOut) => {
    let ports = Array.isArray(decl) ? decl.map((p) => ({ type: asType(p), label: typeof p === 'object' ? p.displayName : null, required: p?.required })) : null;
    if (!ports) {
      // Dynamic (expression) ports: infer from connections and well-known parameters.
      const maxMain = Math.max(-1, ...connected.filter((e) => (isOut ? e.fromType : e.toType) === 'main').map((e) => (isOut ? e.fromIndex : e.toIndex)));
      let mainCount = Math.max(maxMain + 1, isOut ? 1 : (entry?.trigger ? 0 : 1));
      if (isOut && /\.switch$/.test(node.type)) {
        const rules = node.parameters?.rules?.values?.length ?? node.parameters?.numberOutputs ?? 4;
        mainCount = Math.max(mainCount, rules + (node.parameters?.options?.fallbackOutput === 'extra' ? 1 : 0));
      }
      ports = Array.from({ length: mainCount }, () => ({ type: 'main' }));
      if (!isOut && /langchain\.(agent|chainLlm|chainRetrievalQa|chainSummarization|informationExtractor|textClassifier|sentimentAnalysis)/.test(node.type)) {
        ports.push({ type: 'ai_languageModel', label: 'Model', required: true });
        if (/agent/.test(node.type)) ports.push({ type: 'ai_memory', label: 'Memory' }, { type: 'ai_tool', label: 'Tool' });
      }
      if (isOut && /langchain/.test(node.type) && !entry?.trigger && !/agent|chain|Extractor|Classifier|sentiment/i.test(node.type)) {
        const t = (outbound[0]?.fromType) || null;
        if (t && t !== 'main') ports = [{ type: t }];
      }
    }
    // Make sure every connected port exists.
    for (const e of connected) {
      const type = isOut ? e.fromType : e.toType;
      const index = isOut ? e.fromIndex : e.toIndex;
      const same = ports.filter((p) => p.type === type);
      for (let i = same.length; i <= index; i++) ports.push({ type });
    }
    const counters = {};
    return ports.map((p) => ({ ...p, index: (counters[p.type] = (counters[p.type] ?? -1) + 1) }));
  };
  const outputNames = Array.isArray(entry?.outputNames) ? entry.outputNames : /\.if$/.test(node.type) ? ['true', 'false'] : /\.filter$/.test(node.type) && node.parameters?.options?.allowFallthrough ? ['kept', 'discarded'] : null;
  const outputs = resolve(entry?.outputs, outbound, true).map((p) => ({ ...p, label: p.label || (p.type === 'main' && outputNames ? outputNames[p.index] : null) }));
  const inputs = resolve(entry?.inputs, inbound, false);
  return { inputs, outputs };
}

const isSubNode = (ports) => ports.outputs.length > 0 && ports.outputs.every((p) => p.type !== 'main');

function portXY(node, ports, dir, type, index) {
  const [x, y] = node.position || [0, 0];
  const [w, hgt] = nodeSize(node);
  const sub = isSubNode(ports);
  if (dir === 'out') {
    if (type !== 'main') return [x + w / 2, sub ? y : y + hgt];
    const mains = ports.outputs.filter((p) => p.type === 'main');
    const offset = (index - (mains.length - 1) / 2) * PORT_GAP * (mains.length > 3 ? 0.8 : 1);
    return [x + w, y + hgt / 2 + offset];
  }
  if (type !== 'main') {
    const subs = ports.inputs.filter((p) => p.type !== 'main');
    const i = subs.findIndex((p) => p.type === type && p.index === index);
    const span = w + Math.max(0, subs.length - 2) * 40;
    return [x + w / 2 - span / 2 + (span / (subs.length + 1)) * (i + 1), y + hgt];
  }
  const mains = ports.inputs.filter((p) => p.type === 'main');
  return [x, y + hgt / 2 + (index - (mains.length - 1) / 2) * PORT_GAP];
}

function edgePath([x1, y1], [x2, y2], vertical = false) {
  if (vertical) {
    const dy = Math.max(40, Math.abs(y2 - y1) / 2);
    return `M${x1},${y1} C${x1},${y1 - dy} ${x2},${y2 + dy} ${x2},${y2}`;
  }
  if (x2 >= x1 + 20) {
    const dx = Math.max(40, (x2 - x1) / 2);
    return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
  }
  // backwards edge: loop below
  const drop = Math.max(y1, y2) + 80;
  return `M${x1},${y1} C${x1 + 80},${y1} ${x1 + 80},${drop} ${(x1 + x2) / 2},${drop} S${x2 - 80},${y2} ${x2},${y2}`;
}

// ------------------------------------------------------------------ sticky markdown (tiny, safe)

function stickyMarkdown(text) {
  const root = h('div.n8s-sticky-md');
  for (const block of String(text || '').split(/\n{2,}/)) {
    const lines = block.split('\n');
    const heading = /^(#{1,6})\s+(.*)$/.exec(lines[0]);
    if (heading) {
      root.appendChild(h(`h${Math.min(6, heading[1].length + 1)}`, inline(heading[2])));
      if (lines.length > 1) root.appendChild(h('p', inline(lines.slice(1).join(' '))));
    } else if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
      root.appendChild(h('ul', lines.map((l) => h('li', inline(l.replace(/^\s*[-*]\s+/, ''))))));
    } else root.appendChild(h('p', lines.map((l, i) => [i ? h('br') : null, inline(l)])));
  }
  return root;
}

function inline(text) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\((https?:[^)\s]+)\)|_[^_]+_)/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith('**')) out.push(h('b', t.slice(2, -2)));
    else if (t.startsWith('`')) out.push(h('code', t.slice(1, -1)));
    else if (t.startsWith('[')) out.push(h('a', { href: m[2], target: '_blank', rel: 'noopener noreferrer' }, t.slice(1, t.indexOf(']'))));
    else out.push(h('i', t.slice(1, -1)));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

const STICKY_COLORS = ['#fff5d6', '#ffe0d2', '#ffd6e0', '#e6f7d9', '#d7ecff', '#ece0ff', '#eceff3'];

// ------------------------------------------------------------------ canvas

export function createCanvas(container, opts = {}) {
  const state = {
    wf: { nodes: [], connections: {} },
    view: { x: 0, y: 0, k: 1 },
    selected: new Set(),
    selectedEdge: null,
    status: new Map(),
    edgeCounts: new Map(),
    readOnly: !!opts.readOnly,
    dark: !!opts.dark,
    highlight: null,
    ports: new Map(),
  };

  const edgeLayer = s('svg.n8s-edges', { width: 1, height: 1 });
  const nodeLayer = h('div.n8s-nodes');
  const world = h('div.n8s-world', edgeLayer, nodeLayer);
  const marquee = h('div.n8s-marquee');
  const mini = h('canvas.n8s-minimap', { width: 200, height: 130, title: 'Minimap — click to jump' });
  const zoomLabel = h('span.n8s-zoom-label', '100%');
  const controls = h('div.n8s-canvas-controls',
    h('button.n8s-cbtn', { type: 'button', title: 'Zoom in (+)', onclick: () => zoomBy(1.2) }, icon('zoom-in')),
    h('button.n8s-cbtn', { type: 'button', title: 'Zoom out (-)', onclick: () => zoomBy(1 / 1.2) }, icon('zoom-out')),
    h('button.n8s-cbtn', { type: 'button', title: 'Reset zoom (0)', onclick: () => zoomTo(1) }, zoomLabel),
    h('button.n8s-cbtn', { type: 'button', title: 'Fit view (1)', onclick: () => fitView() }, icon('arrows-fullscreen')),
    h('button.n8s-cbtn', { type: 'button', title: 'Toggle minimap (M)', onclick: () => mini.classList.toggle('d-none') }, icon('map')));
  const root = h('div.n8s-canvas', { tabindex: 0, class: [state.readOnly && 'readonly', state.dark && 'dark'] }, world, marquee, mini, controls);
  mount(container, root);

  // ---------------------------------------------------------------- transforms

  const applyView = () => {
    const { x, y, k } = state.view;
    world.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
    root.style.setProperty('--grid-size', `${20 * k}px`);
    root.style.setProperty('--grid-x', `${x}px`);
    root.style.setProperty('--grid-y', `${y}px`);
    root.classList.toggle('zoomed-out', k < 0.45);
    mount(zoomLabel, `${Math.round(k * 100)}%`);
    drawMinimap();
  };

  const toWorld = (cx, cy) => {
    const r = root.getBoundingClientRect();
    return [(cx - r.left - state.view.x) / state.view.k, (cy - r.top - state.view.y) / state.view.k];
  };

  function zoomTo(k, center) {
    const r = root.getBoundingClientRect();
    const [cx, cy] = center || [r.left + r.width / 2, r.top + r.height / 2];
    const [wx, wy] = toWorld(cx, cy);
    state.view.k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, k));
    state.view.x = cx - r.left - wx * state.view.k;
    state.view.y = cy - r.top - wy * state.view.k;
    applyView();
  }
  const zoomBy = (f, center) => zoomTo(state.view.k * f, center);

  function fitView(names) {
    const nodes = (state.wf.nodes || []).filter((n) => !names || names.includes(n.name));
    const r = root.getBoundingClientRect();
    if (!nodes.length || !r.width) { state.view = { x: r.width / 2 - 50, y: r.height / 2 - 50, k: 1 }; applyView(); return; }
    const b = bounds(nodes);
    const pad = 80;
    const k = Math.min(1.2, Math.max(MIN_ZOOM, Math.min((r.width - pad * 2) / Math.max(b.w, 1), (r.height - pad * 2) / Math.max(b.h, 1))));
    state.view = { k, x: r.width / 2 - (b.x + b.w / 2) * k, y: r.height / 2 - (b.y + b.h / 2) * k };
    applyView();
  }

  function center(name) {
    const n = state.wf.nodes.find((x) => x.name === name);
    if (!n) return;
    const r = root.getBoundingClientRect();
    const [w, hgt] = nodeSize(n);
    state.view.x = r.width / 2 - (n.position[0] + w / 2) * state.view.k;
    state.view.y = r.height / 2 - (n.position[1] + hgt / 2) * state.view.k;
    applyView();
  }

  // ---------------------------------------------------------------- rendering

  const nodeEls = new Map();

  function renderNode(node) {
    if (isSticky(node)) return renderSticky(node);
    const entry = entryFor(node.type);
    const ports = portsFor(node, state.wf);
    state.ports.set(node.name, ports);
    const st = state.status.get(node.name);
    const trigger = isTriggerType(node.type, entry);
    const sub = isSubNode(ports);
    const pinned = !!state.wf.pinData?.[node.name]?.length;
    const el = h('div.n8s-node', {
      class: [trigger && 'trigger', sub && 'subnode', node.disabled && 'disabled', st && `st-${st.status}`, state.selected.has(node.name) && 'selected', state.highlight?.has?.(node.name) && 'hl'],
      style: { transform: `translate(${node.position[0]}px, ${node.position[1]}px)`, '--node-color': entry?.iconColor ? null : undefined },
      dataset: { name: node.name },
    },
    h('div.n8s-node-box',
      nodeIcon(node.type, { size: sub ? 34 : 44 }),
      trigger && h('span.n8s-node-bolt', { title: 'Trigger' }, icon('lightning-charge-fill')),
      node.disabled && h('span.n8s-node-off', icon('slash-circle')),
      pinned && h('span.n8s-node-pin', { title: 'Pinned data' }, icon('pin-angle-fill')),
      node.notes && node.notesInFlow !== false && h('span.n8s-node-note', { title: node.notes }, icon('chat-left-text')),
      st && h('span.n8s-node-status', { title: st.error?.message || `${st.items} items` },
        st.status === 'error' ? icon('exclamation-triangle-fill') : st.status === 'running' ? icon('arrow-repeat', 'n8s-spin') : icon('check-lg')),
      st?.runs > 1 && h('span.n8s-node-runs', `${st.runs}×`)),
    h('div.n8s-node-label',
      h('div.n8s-node-name', node.name),
      h('div.n8s-node-sub', st?.status === 'error' ? (st.error?.message || 'Error').slice(0, 60) : nodeSubtitle(entry, node.parameters) || '')),
    ports.inputs.map((p) => portEl(node, ports, 'in', p)),
    ports.outputs.map((p) => portEl(node, ports, 'out', p)));
    return el;
  }

  function portEl(node, ports, dir, p) {
    const [x, y] = portXY(node, ports, dir, p.type, p.index);
    const lx = x - node.position[0];
    const ly = y - node.position[1];
    const main = p.type === 'main';
    return h('div.n8s-port', {
      class: [dir, main ? 'main' : 'aux', p.required && 'required'],
      style: { left: `${lx}px`, top: `${ly}px` },
      dataset: { node: node.name, dir, type: p.type, index: p.index },
      title: `${p.label || (main ? (dir === 'in' ? 'Input' : 'Output') : p.type.replace('ai_', ''))}${state.readOnly ? '' : dir === 'out' ? ' — drag to connect' : ''}`,
    },
    !main && dir === 'in' && h('span.n8s-port-label', p.label || p.type.replace('ai_', '').replace(/([A-Z])/g, ' $1').trim()),
    main && p.label && h('span.n8s-port-label.side', p.label),
    dir === 'out' && main && !state.readOnly && h('button.n8s-port-plus', { type: 'button', title: 'Add a node here', tabindex: -1 }, icon('plus')));
  }

  function renderSticky(node) {
    const p = node.parameters || {};
    const [w, hgt] = nodeSize(node);
    const colorIdx = typeof p.color === 'number' ? p.color - 1 : 0;
    const el = h('div.n8s-sticky', {
      class: [state.selected.has(node.name) && 'selected'],
      style: { transform: `translate(${node.position[0]}px, ${node.position[1]}px)`, width: `${w}px`, height: `${hgt}px`, background: STICKY_COLORS[colorIdx] || STICKY_COLORS[0] },
      dataset: { name: node.name, sticky: '1' },
    }, stickyMarkdown(p.content), !state.readOnly && h('div.n8s-sticky-resize', { dataset: { resize: node.name } }));
    return el;
  }

  function renderNodes() {
    nodeEls.clear();
    const stickies = state.wf.nodes.filter(isSticky);
    const others = state.wf.nodes.filter((n) => !isSticky(n));
    mount(nodeLayer, [...stickies, ...others].map((n) => {
      const el = renderNode(n);
      nodeEls.set(n.name, el);
      return el;
    }));
  }

  function renderEdges() {
    const byName = new Map(state.wf.nodes.map((n) => [n.name, n]));
    const paths = [];
    for (const e of edgeList(state.wf)) {
      const a = byName.get(e.from);
      const b = byName.get(e.to);
      if (!a || !b) continue;
      const pa = state.ports.get(a.name) || portsFor(a, state.wf);
      const pb = state.ports.get(b.name) || portsFor(b, state.wf);
      const p1 = portXY(a, pa, 'out', e.fromType, e.fromIndex);
      const p2 = portXY(b, pb, 'in', e.toType, e.toIndex);
      const vertical = e.fromType !== 'main';
      const d = vertical ? edgePath(p1, p2, true) : edgePath(p1, p2);
      const key = `${e.from}\u0000${e.fromType}\u0000${e.fromIndex}\u0000${e.to}\u0000${e.toType}\u0000${e.toIndex}`;
      const count = e.fromType === 'main' ? state.edgeCounts.get(`${e.from}\u0000${e.fromIndex}`) : null;
      const st = state.status.get(e.from);
      const sel = state.selectedEdge === key;
      // s() replaces (not merges) an SVG element's class, so list them all here.
      const g = s('g', { class: ['n8s-edge', vertical && 'aux', sel && 'selected', st && `st-${st.status}`, count && 'flowed', b.disabled && 'to-disabled'], dataset: { key } },
        s('path.n8s-edge-hit', { d }),
        s('path.n8s-edge-line', { d }),
        count != null && s('text.n8s-edge-count', { x: (p1[0] + p2[0]) / 2, y: (p1[1] + p2[1]) / 2 - 8 }, `${count} item${count === 1 ? '' : 's'}`));
      paths.push(g);
    }
    if (drag?.kind === 'connect' && drag.to) {
      paths.push(s('path.n8s-edge-temp', { d: drag.fromType === 'main' ? edgePath(drag.from, drag.to) : edgePath(drag.from, drag.to, true) }));
    }
    const b = bounds(state.wf.nodes);
    edgeLayer.setAttribute('viewBox', `${b.x - 2000} ${b.y - 2000} ${b.w + 4000} ${b.h + 4000}`);
    edgeLayer.setAttribute('width', b.w + 4000);
    edgeLayer.setAttribute('height', b.h + 4000);
    edgeLayer.style.left = `${b.x - 2000}px`;
    edgeLayer.style.top = `${b.y - 2000}px`;
    mount(edgeLayer, paths);
    edgeTools();
  }

  // Hover/selected edge toolbar: insert node, delete.
  const edgeBar = h('div.n8s-edge-bar.d-none',
    h('button.n8s-cbtn', { type: 'button', title: 'Insert node', onclick: () => { const e = decodeEdge(edgeBar.dataset.key); hideEdgeBar(); opts.onAddRequest?.({ splitEdge: e, at: edgeBar.world }); } }, icon('plus-lg')),
    h('button.n8s-cbtn', { type: 'button', title: 'Delete connection', onclick: () => { const e = decodeEdge(edgeBar.dataset.key); hideEdgeBar(); change(removeConnection(state.wf, e), { label: 'Delete connection' }); } }, icon('trash')));
  world.appendChild(edgeBar);
  const hideEdgeBar = () => edgeBar.classList.add('d-none');
  function edgeTools() {
    if (!state.selectedEdge) hideEdgeBar();
  }
  const decodeEdge = (key) => {
    const [from, fromType, fromIndex, to, toType, toIndex] = key.split('\u0000');
    return { from, fromType, fromIndex: Number(fromIndex), to, toType, toIndex: Number(toIndex) };
  };

  function render() {
    renderNodes();
    renderEdges();
    drawMinimap();
  }

  // ---------------------------------------------------------------- minimap

  function drawMinimap() {
    if (mini.classList.contains('d-none') || !mini.isConnected) return;
    const ctx = mini.getContext('2d');
    const W = mini.width;
    const H = mini.height;
    ctx.clearRect(0, 0, W, H);
    const nodes = state.wf.nodes;
    if (!nodes.length) return;
    const r = root.getBoundingClientRect();
    const vb = { x: -state.view.x / state.view.k, y: -state.view.y / state.view.k, w: r.width / state.view.k, h: r.height / state.view.k };
    const b = bounds(nodes);
    const x1 = Math.min(b.x, vb.x) - 40;
    const y1 = Math.min(b.y, vb.y) - 40;
    const x2 = Math.max(b.x + b.w, vb.x + vb.w) + 40;
    const y2 = Math.max(b.y + b.h, vb.y + vb.h) + 40;
    const sc = Math.min(W / (x2 - x1), H / (y2 - y1));
    mini.mapping = { x1, y1, sc };
    for (const n of nodes) {
      const [w, hgt] = nodeSize(n);
      const st = state.status.get(n.name)?.status;
      ctx.fillStyle = isSticky(n) ? 'rgba(250, 204, 21, .35)' : st === 'error' ? '#ef4444' : st === 'success' ? '#22c55e' : state.selected.has(n.name) ? '#ff6d5a' : (state.dark ? '#64748b' : '#94a3b8');
      ctx.fillRect((n.position[0] - x1) * sc, (n.position[1] - y1) * sc, Math.max(2, w * sc), Math.max(2, hgt * sc));
    }
    ctx.strokeStyle = '#ff6d5a';
    ctx.lineWidth = 1.5;
    ctx.strokeRect((vb.x - x1) * sc, (vb.y - y1) * sc, vb.w * sc, vb.h * sc);
  }
  mini.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    const m = mini.mapping;
    if (!m) return;
    const rect = mini.getBoundingClientRect();
    const move = (ev) => {
      const wx = (ev.clientX - rect.left) / m.sc + m.x1;
      const wy = (ev.clientY - rect.top) / m.sc + m.y1;
      const r = root.getBoundingClientRect();
      state.view.x = r.width / 2 - wx * state.view.k;
      state.view.y = r.height / 2 - wy * state.view.k;
      applyView();
    };
    move(e);
    const up = () => { removeEventListener('mousemove', move); removeEventListener('mouseup', up); };
    addEventListener('mousemove', move);
    addEventListener('mouseup', up);
  });

  // ---------------------------------------------------------------- selection

  function select(names, { add = false } = {}) {
    if (!add) state.selected.clear();
    for (const n of names) state.selected.add(n);
    state.selectedEdge = null;
    for (const [name, el] of nodeEls) el.classList.toggle('selected', state.selected.has(name));
    edgeLayer.querySelectorAll('.n8s-edge.selected').forEach((g) => g.classList.remove('selected'));
    hideEdgeBar();
    drawMinimap();
    opts.onSelect?.([...state.selected]);
  }

  function change(next, meta = {}) {
    if (state.readOnly) return;
    opts.onChange?.(next, meta);
  }

  // ---------------------------------------------------------------- pointer interaction

  let drag = null;
  let spaceDown = false;

  root.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      zoomBy(Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0025)), [e.clientX, e.clientY]);
    } else {
      state.view.x -= e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX;
      state.view.y -= e.shiftKey && !e.deltaX ? 0 : e.deltaY;
      applyView();
    }
  }, { passive: false });

  root.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.n8s-canvas-controls,.n8s-minimap,.n8s-edge-bar')) return;
    root.focus({ preventScroll: true });
    const plus = e.target.closest('.n8s-port-plus');
    const port = e.target.closest('.n8s-port');
    const resize = e.target.closest('.n8s-sticky-resize');
    const nodeEl = e.target.closest('.n8s-node,.n8s-sticky');
    const edgeEl = e.target.closest('.n8s-edge');
    const [wx, wy] = toWorld(e.clientX, e.clientY);

    if (e.button === 1 || e.button === 2 || spaceDown || (e.button === 0 && !nodeEl && !port && !edgeEl && !e.shiftKey && !e.ctrlKey && !e.metaKey)) {
      if (e.button === 2 && (nodeEl || edgeEl)) return; // context menu
      drag = { kind: 'pan', sx: e.clientX, sy: e.clientY, vx: state.view.x, vy: state.view.y, moved: false, button: e.button };
      root.setPointerCapture(e.pointerId);
      root.classList.add('panning');
      return;
    }
    if (e.button !== 0) return;
    if (plus && !state.readOnly) {
      const p = plus.closest('.n8s-port');
      e.stopPropagation();
      const from = { node: p.dataset.node, type: p.dataset.type, index: Number(p.dataset.index) };
      const n = state.wf.nodes.find((x) => x.name === from.node);
      opts.onAddRequest?.({ from, at: [n.position[0] + 220, n.position[1] + (from.index - 0) * 40] });
      return;
    }
    if (port && !state.readOnly) {
      const node = state.wf.nodes.find((n) => n.name === port.dataset.node);
      const ports = state.ports.get(node.name);
      const dir = port.dataset.dir;
      const type = port.dataset.type;
      const index = Number(port.dataset.index);
      if (dir === 'out') {
        drag = { kind: 'connect', node: node.name, fromType: type, fromIndex: index, from: portXY(node, ports, 'out', type, index), to: null };
      } else {
        // Dragging from an input re-wires an existing connection (or starts one backwards).
        const existing = edgeList(state.wf).find((x) => x.to === node.name && x.toType === type && x.toIndex === index);
        if (existing) {
          const src = state.wf.nodes.find((n) => n.name === existing.from);
          change(removeConnection(state.wf, existing), { label: 'Rewire', group: 'rewire' });
          drag = { kind: 'connect', node: src.name, fromType: existing.fromType, fromIndex: existing.fromIndex, from: portXY(src, state.ports.get(src.name) || portsFor(src, state.wf), 'out', existing.fromType, existing.fromIndex), to: [wx, wy] };
        } else {
          drag = { kind: 'connect-in', node: node.name, toType: type, toIndex: index };
        }
      }
      root.setPointerCapture(e.pointerId);
      return;
    }
    if (resize && !state.readOnly) {
      const node = state.wf.nodes.find((n) => n.name === resize.dataset.resize);
      drag = { kind: 'resize', name: node.name, sx: wx, sy: wy, w: nodeSize(node)[0], h: nodeSize(node)[1] };
      root.setPointerCapture(e.pointerId);
      return;
    }
    if (nodeEl) {
      const name = nodeEl.dataset.name;
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        if (state.selected.has(name)) { state.selected.delete(name); select([...state.selected]); } else select([name], { add: true });
      } else if (!state.selected.has(name)) select([name]);
      if (state.readOnly) return;
      const moving = [...state.selected];
      drag = {
        kind: 'move', sx: wx, sy: wy, moved: false, alt: e.altKey,
        start: new Map(state.wf.nodes.filter((n) => moving.includes(n.name)).map((n) => [n.name, [...n.position]])),
      };
      root.setPointerCapture(e.pointerId);
      return;
    }
    if (edgeEl) {
      const key = edgeEl.dataset.key;
      select([]);
      state.selectedEdge = key;
      edgeEl.classList.add('selected');
      if (!state.readOnly) {
        edgeBar.dataset.key = key;
        edgeBar.world = [wx - 50, wy - 50];
        Object.assign(edgeBar.style, { left: `${wx}px`, top: `${wy}px` });
        edgeBar.classList.remove('d-none');
      }
      return;
    }
    // shift/ctrl + drag on background: marquee select
    drag = { kind: 'marquee', sx: e.clientX, sy: e.clientY, add: e.shiftKey || e.ctrlKey || e.metaKey, base: new Set(state.selected) };
    root.setPointerCapture(e.pointerId);
  });

  root.addEventListener('pointermove', (e) => {
    if (!drag) {
      return;
    }
    const [wx, wy] = toWorld(e.clientX, e.clientY);
    if (drag.kind === 'pan') {
      drag.moved = drag.moved || Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 3;
      state.view.x = drag.vx + e.clientX - drag.sx;
      state.view.y = drag.vy + e.clientY - drag.sy;
      applyView();
    } else if (drag.kind === 'move') {
      const dx = wx - drag.sx;
      const dy = wy - drag.sy;
      if (!drag.moved && Math.hypot(dx, dy) < 3 / state.view.k) return;
      drag.moved = true;
      for (const n of state.wf.nodes) {
        const p = drag.start.get(n.name);
        if (!p) continue;
        n.position = [e.altKey ? p[0] + dx : snap(p[0] + dx), e.altKey ? p[1] + dy : snap(p[1] + dy)];
        const el = nodeEls.get(n.name);
        if (el) el.style.transform = `translate(${n.position[0]}px, ${n.position[1]}px)`;
      }
      renderEdges();
    } else if (drag.kind === 'connect') {
      drag.to = [wx, wy];
      const hover = document.elementFromPoint(e.clientX, e.clientY)?.closest('.n8s-port.in,.n8s-node');
      nodeLayer.querySelectorAll('.drop-target').forEach((x) => x.classList.remove('drop-target'));
      if (hover && hover.dataset.node !== drag.node && hover.dataset.name !== drag.node) hover.classList.add('drop-target');
      renderEdges();
    } else if (drag.kind === 'resize') {
      const node = state.wf.nodes.find((n) => n.name === drag.name);
      node.parameters = { ...node.parameters, width: Math.max(150, snap(drag.w + wx - drag.sx)), height: Math.max(80, snap(drag.h + wy - drag.sy)) };
      const el = nodeEls.get(node.name);
      el.style.width = `${node.parameters.width}px`;
      el.style.height = `${node.parameters.height}px`;
      drag.moved = true;
    } else if (drag.kind === 'marquee') {
      const r = root.getBoundingClientRect();
      const x = Math.min(e.clientX, drag.sx) - r.left;
      const y = Math.min(e.clientY, drag.sy) - r.top;
      const w = Math.abs(e.clientX - drag.sx);
      const hh = Math.abs(e.clientY - drag.sy);
      Object.assign(marquee.style, { display: 'block', left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${hh}px` });
      const [ax, ay] = toWorld(Math.min(e.clientX, drag.sx), Math.min(e.clientY, drag.sy));
      const [bx, by] = toWorld(Math.max(e.clientX, drag.sx), Math.max(e.clientY, drag.sy));
      const hit = state.wf.nodes.filter((n) => {
        const [nw, nh] = nodeSize(n);
        return n.position[0] < bx && n.position[0] + nw > ax && n.position[1] < by && n.position[1] + nh > ay;
      }).map((n) => n.name);
      state.selected = new Set(drag.add ? [...drag.base, ...hit] : hit);
      for (const [name, el] of nodeEls) el.classList.toggle('selected', state.selected.has(name));
    }
  });

  const endDrag = (e) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    root.classList.remove('panning');
    marquee.style.display = 'none';
    nodeLayer.querySelectorAll('.drop-target').forEach((x) => x.classList.remove('drop-target'));
    if (d.kind === 'pan' && !d.moved && d.button === 0) {
      select([]);
      opts.onBackgroundClick?.();
    } else if (d.kind === 'move' && d.moved) {
      const positions = new Map(state.wf.nodes.map((n) => [n.name, n.position]));
      for (const [name, p] of d.start) state.wf.nodes.find((n) => n.name === name).position = p;
      change({ ...state.wf, nodes: state.wf.nodes.map((n) => ({ ...n, position: positions.get(n.name) })) }, { label: 'Move', group: `move:${[...d.start.keys()].join(',')}` });
    } else if (d.kind === 'resize' && d.moved) {
      change({ ...state.wf, nodes: state.wf.nodes.map((n) => ({ ...n })) }, { label: 'Resize note' });
    } else if (d.kind === 'connect') {
      const target = document.elementFromPoint(e.clientX, e.clientY);
      const port = target?.closest('.n8s-port.in');
      const nodeEl = target?.closest('.n8s-node');
      let edge = null;
      if (port && port.dataset.node !== d.node) {
        edge = { from: d.node, fromType: d.fromType, fromIndex: d.fromIndex, to: port.dataset.node, toType: port.dataset.type, toIndex: Number(port.dataset.index) };
      } else if (nodeEl && nodeEl.dataset.name !== d.node) {
        const ports = state.ports.get(nodeEl.dataset.name);
        const match = ports?.inputs.find((p) => p.type === d.fromType);
        if (match) edge = { from: d.node, fromType: d.fromType, fromIndex: d.fromIndex, to: nodeEl.dataset.name, toType: match.type, toIndex: match.index };
      }
      if (edge && edge.fromType !== edge.toType) edge = null;
      if (edge) change(addConnection(state.wf, edge), { label: 'Connect' });
      else if (!target?.closest('.n8s-node')) {
        opts.onAddRequest?.({ from: { node: d.node, type: d.fromType, index: d.fromIndex }, at: d.to ? [snap(d.to[0]), snap(d.to[1] - NODE_H / 2)] : null });
      }
      renderEdges();
    } else if (d.kind === 'connect-in') {
      opts.onAddRequest?.({ to: { node: d.node, type: d.toType, index: d.toIndex } });
    } else if (d.kind === 'marquee') {
      opts.onSelect?.([...state.selected]);
      drawMinimap();
    }
  };
  root.addEventListener('pointerup', endDrag);
  root.addEventListener('pointercancel', endDrag);

  // Pointer capture retargets click events to the root, so resolve what is under the pointer.
  const hitAt = (e) => document.elementFromPoint(e.clientX, e.clientY) || e.target;
  root.addEventListener('dblclick', (e) => {
    const target = hitAt(e);
    if (target.closest('.n8s-canvas-controls,.n8s-minimap,.n8s-edge-bar')) return;
    const nodeEl = target.closest('.n8s-node,.n8s-sticky');
    if (nodeEl) opts.onOpenNode?.(nodeEl.dataset.name);
    else if (!state.readOnly) opts.onAddRequest?.({ at: toWorld(e.clientX, e.clientY).map((v) => snap(v - 50)) });
  });

  root.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const nodeEl = hitAt(e).closest('.n8s-node,.n8s-sticky');
    if (nodeEl && !state.selected.has(nodeEl.dataset.name)) select([nodeEl.dataset.name]);
    opts.onContextMenu?.(e, { node: nodeEl?.dataset.name || null, world: toWorld(e.clientX, e.clientY) });
  });

  // Touch: pinch zoom with two pointers.
  const touches = new Map();
  root.addEventListener('touchstart', (e) => { for (const t of e.changedTouches) touches.set(t.identifier, [t.clientX, t.clientY]); }, { passive: true });
  root.addEventListener('touchmove', (e) => {
    if (e.touches.length !== 2) return;
    const [a, b] = [...e.touches];
    const pa = touches.get(a.identifier);
    const pb = touches.get(b.identifier);
    if (pa && pb) {
      const before = Math.hypot(pa[0] - pb[0], pa[1] - pb[1]);
      const after = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (before > 0) zoomBy(after / before, [(a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2]);
    }
    touches.set(a.identifier, [a.clientX, a.clientY]);
    touches.set(b.identifier, [b.clientX, b.clientY]);
  }, { passive: true });
  root.addEventListener('touchend', (e) => { for (const t of e.changedTouches) touches.delete(t.identifier); });

  const keydown = (e) => { if (e.code === 'Space' && !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName) && !e.target.isContentEditable) { spaceDown = true; root.classList.add('space'); } };
  const keyup = (e) => { if (e.code === 'Space') { spaceDown = false; root.classList.remove('space'); } };
  addEventListener('keydown', keydown);
  addEventListener('keyup', keyup);
  const ro = new ResizeObserver(() => drawMinimap());
  ro.observe(root);

  // ---------------------------------------------------------------- API

  return {
    el: root,
    setWorkflow(wf, { fit = false } = {}) {
      state.wf = wf;
      for (const n of [...state.selected]) if (!wf.nodes.some((x) => x.name === n)) state.selected.delete(n);
      render();
      if (fit) requestAnimationFrame(() => fitView());
    },
    setStatus(statusMap, edgeCounts) {
      state.status = statusMap || new Map();
      state.edgeCounts = edgeCounts || new Map();
      render();
    },
    setHighlight(names) { state.highlight = names ? new Set(names) : null; render(); },
    setReadOnly(v) { state.readOnly = v; root.classList.toggle('readonly', v); render(); },
    setDark(v) { state.dark = v; root.classList.toggle('dark', v); drawMinimap(); },
    select,
    selection: () => [...state.selected],
    selectedEdge: () => (state.selectedEdge ? decodeEdge(state.selectedEdge) : null),
    fitView,
    zoomTo,
    zoomBy,
    center,
    toWorld,
    viewCenter() {
      const r = root.getBoundingClientRect();
      return toWorld(r.left + r.width / 2, r.top + r.height / 2).map((v) => snap(v - 50));
    },
    nodeElement: (name) => nodeEls.get(name),
    flash(name) {
      const el = nodeEls.get(name);
      if (!el) return;
      el.classList.remove('flash');
      void el.offsetWidth;
      el.classList.add('flash');
    },
    destroy() {
      removeEventListener('keydown', keydown);
      removeEventListener('keyup', keyup);
      ro.disconnect();
    },
    displayName,
    NODE_W,
  };
}
