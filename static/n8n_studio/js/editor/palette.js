// Node creator: fuzzy search over the catalog with categories, recent nodes,
// keyboard navigation, and smart wiring (from an output, into an input, or
// splitting an existing connection).

import { h, icon, mount, prefs, debounce } from '../../../ui/js/dom.js';
import { modal } from '../../../ui/js/ui.js';
import { catalog, nodeIcon, categoryOf, PALETTE_ORDER } from '../core/catalog.js';
import { rank } from '../core/fuzzy.js';
import { uniqueName, uuid, addConnection, removeConnection, snap, triggers, STICKY } from '../core/workflow.js';
import { stripHtml } from './paramform.js';

const RECENT_KEY = 'n8s:recent-nodes';
const WEBHOOKISH = /webhook|formTrigger|chatTrigger|\.wait$|mcpTrigger/i;

/** Build a node of `type` placed at `at` with n8n defaults. */
export function makeNode(type, wf, at = [0, 0]) {
  const entry = catalog.nodes.get(type);
  const base = entry?.defaults?.name || entry?.displayName || type.split('.').pop();
  const node = {
    id: uuid(),
    name: uniqueName(base, wf.nodes.map((n) => n.name)),
    type,
    typeVersion: entry?.defaultVersion ?? Math.max(...(entry?.versions || [1])),
    position: [snap(at[0]), snap(at[1])],
    parameters: {},
  };
  if (type === STICKY) Object.assign(node, { parameters: { content: '## Note\nDouble-click to edit.', height: 200, width: 280, color: 1 } });
  if (WEBHOOKISH.test(type)) node.webhookId = uuid();
  return node;
}

/** Insert a node and wire it according to the request. Returns {wf, node}. */
export function insertNode(wf, type, req = {}) {
  let at = req.at;
  if (!at && req.from) {
    const src = wf.nodes.find((n) => n.name === req.from.node);
    at = src ? [src.position[0] + 220, src.position[1] + (req.from.type === 'main' ? 0 : 200)] : [0, 0];
  }
  if (!at && req.to) {
    const dst = wf.nodes.find((n) => n.name === req.to.node);
    at = dst ? (req.to.type === 'main' ? [dst.position[0] - 220, dst.position[1]] : [dst.position[0], dst.position[1] + 180]) : [0, 0];
  }
  if (!at && req.splitEdge) {
    const a = wf.nodes.find((n) => n.name === req.splitEdge.from);
    const b = wf.nodes.find((n) => n.name === req.splitEdge.to);
    at = a && b ? [(a.position[0] + b.position[0]) / 2, (a.position[1] + b.position[1]) / 2] : [0, 0];
  }
  const node = makeNode(type, wf, at || [0, 0]);
  // nudge away from an exactly overlapping node
  while (wf.nodes.some((n) => n.position[0] === node.position[0] && n.position[1] === node.position[1])) node.position = [node.position[0], node.position[1] + 120];
  let next = { ...wf, nodes: [...wf.nodes, node] };
  if (req.from) next = addConnection(next, { from: req.from.node, fromType: req.from.type, fromIndex: req.from.index, to: node.name, toType: req.from.type, toIndex: 0 });
  if (req.to) next = addConnection(next, { from: node.name, fromType: req.to.type, fromIndex: 0, to: req.to.node, toType: req.to.type, toIndex: req.to.index });
  if (req.splitEdge) {
    const e = req.splitEdge;
    next = removeConnection(next, e);
    next = addConnection(next, { from: e.from, fromType: e.fromType, fromIndex: e.fromIndex, to: node.name, toType: 'main', toIndex: 0 });
    next = addConnection(next, { from: node.name, fromType: 'main', fromIndex: 0, to: e.to, toType: e.toType, toIndex: e.toIndex });
  }
  const recent = [type, ...prefs.get(RECENT_KEY, []).filter((t) => t !== type)].slice(0, 8);
  prefs.set(RECENT_KEY, recent);
  return { wf: next, node };
}

/** Open the creator. Resolves with the chosen type (or null). */
export function openPalette(wf, req = {}) {
  return new Promise((resolve) => {
    let chosen = null;
    const all = [...catalog.nodes.values()].filter((e) => !e.hidden);
    const wantType = req.from?.type && req.from.type !== 'main' ? req.from.type : req.to?.type && req.to.type !== 'main' ? req.to.type : null;
    const fromSub = wantType && wantType === req.from?.type;
    const pool = wantType
      ? all.filter((e) => {
        const ports = fromSub ? e.inputs : e.outputs;
        // Agents/chains declare inputs dynamically (an expression), so include them by name.
        if (!Array.isArray(ports)) return fromSub && /langchain\.(agent|chain|informationExtractor|textClassifier|sentimentAnalysis)/.test(e.name);
        return ports.some((p) => (typeof p === 'string' ? p : p?.type) === wantType);
      })
      : all;
    const noTrigger = !triggers(wf, catalog.nodes).length;
    const search = h('input.form-control.n8s-pal-search', { type: 'search', placeholder: wantType ? `Search ${wantType.replace('ai_', '')} nodes…` : 'Search 900+ nodes — e.g. "slack", "http", "if"…', autofocus: true });
    const list = h('div.n8s-pal-list', { role: 'listbox' });
    const cats = h('div.n8s-pal-cats');
    let category = null;
    let items = [];
    let cursor = 0;

    const row = (e) => h('button.n8s-pal-item', {
      type: 'button', role: 'option', dataset: { type: e.name },
      onclick: () => { chosen = e.name; m.close(); },
      onmousemove: () => { const i = items.indexOf(e); if (i !== cursor) { cursor = i; highlight(); } },
    }, nodeIcon(e.name, { size: 30 }),
    h('div.n8s-pal-text', h('div.n8s-pal-name', e.displayName, e.trigger && h('span.n8s-pal-badge', 'trigger')), h('div.n8s-pal-desc', stripHtml(e.description || '').slice(0, 120))));

    function render() {
      const q = search.value.trim();
      if (q) {
        items = rank(pool, q, (e) => [e.displayName, e.name.split('.').pop(), ...(e.codex?.alias || [])], 60);
        mount(cats);
      } else {
        const groups = new Map(PALETTE_ORDER.map((c) => [c, []]));
        for (const e of pool) groups.get(categoryOf(e))?.push(e);
        if (!category) {
          const recent = prefs.get(RECENT_KEY, []).map((t) => catalog.nodes.get(t)).filter((e) => e && pool.includes(e));
          const order = noTrigger && !req.from ? PALETTE_ORDER : PALETTE_ORDER.filter((c) => c !== 'Triggers').concat('Triggers');
          mount(cats, order.filter((c) => groups.get(c)?.length).map((c) => h('button.n8s-pal-cat', { type: 'button', onclick: () => { category = c; render(); } },
            icon({ Triggers: 'lightning-charge', AI: 'stars', Flow: 'signpost-split', 'Data transformation': 'shuffle', Core: 'cpu', Helpers: 'tools', Apps: 'grid-3x3-gap' }[c]),
            h('span', c), h('span.n8s-pal-count', String(groups.get(c).length)))),
          !wantType && h('button.n8s-pal-cat', { type: 'button', onclick: () => { chosen = STICKY; m.close(); } }, icon('sticky'), h('span', 'Sticky note')));
          items = recent.length ? recent : (noTrigger ? groups.get('Triggers') : groups.get('Core')).slice(0, 12);
          mount(list, h('div.n8s-pal-section', recent.length ? 'Recently used' : noTrigger ? 'Start with a trigger' : 'Popular'), items.map(row));
          cursor = 0;
          highlight();
          return;
        }
        items = groups.get(category) || [];
        mount(cats, h('button.n8s-pal-back', { type: 'button', onclick: () => { category = null; render(); } }, icon('arrow-left'), ` ${category}`));
      }
      cursor = 0;
      mount(list, items.length ? items.map(row) : h('div.n8s-pal-empty', 'No nodes match. Community nodes appear once installed (Packages page).'));
      highlight();
    }

    function highlight() {
      const els = list.querySelectorAll('.n8s-pal-item');
      els.forEach((el, i) => el.classList.toggle('active', i === cursor));
      els[cursor]?.scrollIntoView({ block: 'nearest' });
    }

    search.addEventListener('input', debounce(render, 60));
    search.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') { e.preventDefault(); cursor = Math.min(items.length - 1, cursor + 1); highlight(); }
      if (e.key === 'ArrowUp') { e.preventDefault(); cursor = Math.max(0, cursor - 1); highlight(); }
      if (e.key === 'Enter' && items[cursor]) { e.preventDefault(); chosen = items[cursor].name; m.close(); }
      if (e.key === 'Backspace' && !search.value && category) { category = null; render(); }
    });

    const m = modal({
      title: req.splitEdge ? 'Insert a node' : req.from ? `Add a node after “${req.from.node}”` : req.to ? `Add a ${req.to.type === 'main' ? 'node' : req.to.type.replace('ai_', '')} for “${req.to.node}”` : 'Add a node',
      icon: 'plus-circle',
      size: 'lg',
      body: h('div.n8s-pal', search, cats, list, h('div.n8s-pal-foot', h('span', h('kbd', '↑'), h('kbd', '↓'), ' navigate'), h('span', h('kbd', 'Enter'), ' add'), h('span', h('kbd', 'Esc'), ' close'), h('span.ms-auto', `${pool.length} nodes · catalog: ${catalog.source}`))),
      onClose: () => resolve(chosen),
    });
    m.el.classList.add('n8s-pal-modal');
    render();
    setTimeout(() => search.focus(), 30);
  });
}
