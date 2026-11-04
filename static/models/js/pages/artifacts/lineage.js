// Incremental lineage graph: start at one artifact version, expand one hop at a time.
//
// Nodes: artifacts ("a:<entity>/<project>/<collection>:v<N>") and runs ("r:<entity>/<project>/<run>").
// Edges always point in data-flow order: producer run -> artifact -> consumer run.

import { h, s, icon, mount } from '../../../../ui/js/dom.js';
import { layoutGraph, tip, tipRows, hideTip } from '../../../../ui/js/charts.js';
import { toast, toastError } from '../../../../ui/js/ui.js';
import { navigate } from '../../../../ui/js/router.js';
import { gql } from '../../core/api.js';
import { runStateInfo } from '../../core/wb.js';
import { typeLook } from './shared.js';

const NODE_W = 210;
const NODE_H = 46;

const ART_FIELDS = `id versionIndex state artifactType { name } artifactSequence { name project { name entityName } }
  aliases { alias }
  createdBy { __typename ... on Run { name displayName state jobType project { name entityName } } }`;

const artKey = (a) => {
  const pr = a.artifactSequence?.project || {};
  return `a:${pr.entityName}/${pr.name}/${a.artifactSequence?.name}:v${a.versionIndex}`;
};
const runKey = (r) => `r:${r.project?.entityName}/${r.project?.name}/${r.name}`;

/**
 * lineageGraph({ artifact, urls, height }) -> { el }
 *   artifact: the focus version (needs id, versionIndex, artifactSequence/project, createdBy, usedRuns)
 */
export function lineageGraph({ artifact, entity, project, collection, urls, height = 460 }) {
  const nodes = new Map(); // key -> {key, kind, label, sub, data, expanded}
  const edges = new Set(); // "from>to"
  const container = h('div.ma-lineage', { style: { height: `${height}px` } });
  const legend = h('div.ma-lineage-legend',
    h('span', h('span.ma-ln-dot.art'), 'Artifact'), h('span', h('span.ma-ln-dot.run'), 'Run'),
    h('span.text-muted', icon('plus-circle', 'me-1'), 'Click ⊕ to expand a node one hop · drag to pan · scroll to zoom'));
  const wrap = h('div.ma-lineage-wrap', container, legend);

  const focusKey = `a:${entity}/${project}/${collection}:v${artifact.versionIndex}`;
  addArtifact({ ...artifact, artifactSequence: { name: collection, project: { name: project, entityName: entity } } }, true);
  if (artifact.createdBy?.__typename === 'Run') {
    const r = { ...artifact.createdBy, project: artifact.createdBy.project || { name: project, entityName: entity } };
    addRun(r);
    edges.add(`${runKey(r)}>${focusKey}`);
  }
  for (const r of artifact.usedRuns || []) {
    addRun(r);
    edges.add(`${focusKey}>${runKey(r)}`);
  }
  nodes.get(focusKey).expanded = true;

  function addArtifact(a, focus = false) {
    const key = artKey(a);
    if (!nodes.has(key)) {
      nodes.set(key, { key, kind: 'artifact', label: `${a.artifactSequence?.name}:v${a.versionIndex}`, sub: a.artifactType?.name || '', data: a, focus, aliases: (a.aliases || []).map((x) => x.alias).filter((x) => !/^v\d+$/.test(x)) });
    }
    return key;
  }
  function addRun(r) {
    const key = runKey(r);
    if (!nodes.has(key)) nodes.set(key, { key, kind: 'run', label: r.displayName || r.name, sub: r.jobType || 'run', data: r });
    return key;
  }

  async function expand(node) {
    if (node.expanded || node.loading) return;
    node.loading = true;
    draw();
    mount(stat, h('span', h('span.spinner-border.spinner-border-sm.me-1'), `Expanding ${node.label}…`));
    const slow = setTimeout(() => toast('The W&B server is slow to answer — the graph will update when it does.', 'info', { timeout: 4000 }), 8000);
    const before = nodes.size;
    try {
      if (node.kind === 'run') {
        const pr = node.data.project || { name: project, entityName: entity };
        const d = await gql(`query RunArts($e: String!, $p: String!, $r: String!) { project(entityName: $e, name: $p) { run(name: $r) {
          inputArtifacts(first: 50) { edges { node { ${ART_FIELDS} } } }
          outputArtifacts(first: 50) { edges { node { ${ART_FIELDS} } } } } } }`, { e: pr.entityName, p: pr.name, r: node.data.name });
        const run = d.project?.run;
        for (const e of run?.inputArtifacts?.edges || []) edges.add(`${addArtifact(e.node)}>${node.key}`);
        for (const e of run?.outputArtifacts?.edges || []) edges.add(`${node.key}>${addArtifact(e.node)}`);
      } else {
        const d = await gql(`query ArtLin($id: ID!) { artifact(id: $id) { ${ART_FIELDS}
          usedBy(first: 50) { edges { node { name displayName state jobType project { name entityName } } } } } }`, { id: node.data.id });
        const a = d.artifact;
        if (a?.createdBy?.__typename === 'Run') edges.add(`${addRun(a.createdBy)}>${node.key}`);
        for (const e of a?.usedBy?.edges || []) edges.add(`${node.key}>${addRun(e.node)}`);
      }
      node.expanded = true;
      if (nodes.size === before) toast(`${node.label} has no other ${node.kind === 'run' ? 'inputs or outputs' : 'producers or consumers'}.`, 'info', { timeout: 2500 });
    } catch (err) {
      toastError(err, 'Could not expand lineage');
    } finally {
      clearTimeout(slow);
      node.loading = false;
      draw();
    }
  }

  function open(node) {
    if (node.kind === 'run') {
      const pr = node.data.project || { name: project, entityName: entity };
      navigate(urls.run(pr.entityName, pr.name, node.data.name));
    } else {
      const pr = node.data.artifactSequence?.project || { name: project, entityName: entity };
      navigate(urls.artifact(pr.entityName, pr.name, node.data.artifactType?.name || node.sub, node.data.artifactSequence.name, `v${node.data.versionIndex}`, 'lineage'));
    }
  }

  // ---- render (pan/zoom SVG)
  let view = null;
  const svg = s('svg', { class: 'ma-lineage-svg', width: '100%', height: '100%' });
  const world = s('g');
  const defs = s('defs', s('marker', { id: 'ma-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
    s('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#94a3b8' })));
  svg.append(defs, world);
  container.appendChild(svg);
  const controls = h('div.ma-lineage-controls',
    h('button.af-icon-btn', { title: 'Zoom in', onclick: () => zoomBy(1.25) }, icon('zoom-in')),
    h('button.af-icon-btn', { title: 'Zoom out', onclick: () => zoomBy(0.8) }, icon('zoom-out')),
    h('button.af-icon-btn', { title: 'Fit', onclick: () => fit() }, icon('arrows-fullscreen')),
    h('button.af-icon-btn', { title: 'Expand everything one hop', onclick: () => Promise.all([...nodes.values()].filter((n) => !n.expanded).slice(0, 12).map(expand)) }, icon('diagram-3')));
  container.appendChild(controls);
  const stat = h('div.ma-lineage-stat');
  container.appendChild(stat);

  let layout = null;
  const apply = () => view && world.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`);
  function fit() {
    if (!layout) return;
    const w = container.clientWidth || 800;
    const hh = container.clientHeight || height;
    const k = Math.min((w - 60) / Math.max(layout.width, 1), (hh - 60) / Math.max(layout.height, 1), 1.3);
    view = { k, x: (w - layout.width * k) / 2, y: (hh - layout.height * k) / 2 };
    apply();
  }
  function zoomBy(f, cx = container.clientWidth / 2, cy = container.clientHeight / 2) {
    if (!view) return;
    const k = Math.min(Math.max(view.k * f, 0.15), 3);
    const r = k / view.k;
    view = { k, x: cx - (cx - view.x) * r, y: cy - (cy - view.y) * r };
    apply();
  }
  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = svg.getBoundingClientRect();
    zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  let drag = null;
  svg.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.ma-ln-node')) return;
    drag = { x: e.clientX, y: e.clientY, vx: view?.x || 0, vy: view?.y || 0 };
    svg.setPointerCapture(e.pointerId);
    container.classList.add('dragging');
  });
  svg.addEventListener('pointermove', (e) => { if (drag && view) { view.x = drag.vx + e.clientX - drag.x; view.y = drag.vy + e.clientY - drag.y; apply(); } });
  const end = () => { drag = null; container.classList.remove('dragging'); };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', end);

  function draw() {
    hideTip();
    const down = new Map([...nodes.keys()].map((k) => [k, []]));
    for (const e of edges) {
      const [a, b] = e.split('>');
      if (down.has(a) && nodes.has(b)) down.get(a).push(b);
    }
    layout = layoutGraph([...nodes.keys()].map((k) => ({ task_id: k, downstream_task_ids: down.get(k) })));
    const prevNodes = world.querySelectorAll('.ma-ln-node').length;
    world.replaceChildren();
    for (const e of edges) {
      const [a, b] = e.split('>');
      const pa = layout.nodes.get(a);
      const pb = layout.nodes.get(b);
      if (!pa || !pb) continue;
      const x1 = pa.x + NODE_W; const y1 = pa.y + NODE_H / 2; const x2 = pb.x - 2; const y2 = pb.y + NODE_H / 2; const mx = (x1 + x2) / 2;
      world.appendChild(s('path', { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, class: 'ma-ln-edge', 'marker-end': 'url(#ma-arrow)' }));
    }
    for (const [key, p] of layout.nodes) {
      const n = nodes.get(key);
      const isArt = n.kind === 'artifact';
      const look = isArt ? typeLook(n.sub) : null;
      const color = isArt ? look.color : runStateInfo(n.data.state).color;
      const g = s('g', { class: ['ma-ln-node', isArt ? 'art' : 'run', n.focus && 'focus', n.loading && 'loading'].filter(Boolean).join(' '), transform: `translate(${p.x},${p.y})`, tabindex: 0, role: 'button', 'aria-label': n.label });
      g.appendChild(s('rect', { width: NODE_W, height: NODE_H, rx: isArt ? 8 : 23, class: 'ma-ln-rect', style: `--c:${color}` }));
      g.appendChild(s('rect', { width: 6, height: NODE_H - 12, x: 8, y: 6, rx: 3, fill: color }));
      g.appendChild(s('text', { x: 22, y: 19, class: 'ma-ln-label' }, n.label.length > 24 ? `${n.label.slice(0, 23)}…` : n.label));
      g.appendChild(s('text', { x: 22, y: 35, class: 'ma-ln-sub' }, `${isArt ? n.sub : n.sub}${n.aliases?.length ? ` · ${n.aliases.join(', ')}` : ''}`.slice(0, 32)));
      if (!n.expanded) {
        const plus = s('g', { class: 'ma-ln-expand', transform: `translate(${NODE_W - 16},${NODE_H / 2})` },
          s('circle', { r: 9 }), s('path', { d: 'M -4 0 L 4 0 M 0 -4 L 0 4' }));
        plus.addEventListener('click', (e) => { e.stopPropagation(); expand(n); });
        g.appendChild(plus);
      }
      g.addEventListener('click', () => open(n));
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(n); if (e.key === '+') expand(n); });
      tip(g, () => tipRows(n.label, isArt
        ? [['Type', n.sub], ['Version', `v${n.data.versionIndex}`], ['Aliases', n.aliases?.join(', ') || '—'], ['', n.expanded ? 'Click to open' : 'Click to open · ⊕ expand']]
        : [['Run', n.data.name], ['Job type', n.data.jobType || '—'], ['State', runStateInfo(n.data.state).label], ['', n.expanded ? 'Click to open' : 'Click to open · ⊕ expand']]));
      world.appendChild(g);
    }
    mount(stat, `${[...nodes.values()].filter((n) => n.kind === 'artifact').length} artifacts · ${[...nodes.values()].filter((n) => n.kind === 'run').length} runs`);
    if (!view || world.querySelectorAll('.ma-ln-node').length !== prevNodes) requestAnimationFrame(fit);
  }

  const ro = new ResizeObserver(() => { if (!container.dataset.ready && container.clientWidth) { container.dataset.ready = '1'; draw(); } });
  ro.observe(container);
  return { el: wrap, expand: (key) => nodes.get(key) && expand(nodes.get(key)), nodes };
}
