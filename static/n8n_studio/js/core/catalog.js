// Node-type catalog in the browser: a Map(type -> index entry) loaded once from
// /automations/studio/catalog/, plus lazily fetched full descriptions per type.

import { h, icon } from '../../../ui/js/dom.js';
import { studio, iconUrl, enc } from './api.js';
import { descriptionFor } from './params.js';

let indexPromise = null;
const descCache = new Map();

export const catalog = {
  nodes: new Map(),
  credentials: new Map(),
  source: null,
  ready: false,
};

export function loadCatalog({ refresh = false } = {}) {
  if (indexPromise && !refresh) return indexPromise;
  indexPromise = studio.get('/catalog/', refresh ? { refresh: 1 } : undefined).then((data) => {
    catalog.nodes = new Map(data.nodes.map((n) => [n.name, n]));
    catalog.credentials = new Map(data.credentials.map((c) => [c.name, c]));
    catalog.source = data.source;
    catalog.ready = true;
    if (refresh) descCache.clear();
    return catalog;
  }).catch((err) => { indexPromise = null; throw err; });
  return indexPromise;
}

/** All description entries for a type (one per version range). */
export function loadDescriptions(type) {
  if (!descCache.has(type)) {
    descCache.set(type, studio.get(`/catalog/nodes/${enc(type)}/`).catch((err) => { descCache.delete(type); throw err; }));
  }
  return descCache.get(type);
}

export async function loadDescription(type, typeVersion) {
  return descriptionFor(await loadDescriptions(type), typeVersion);
}

const credCache = new Map();
export function loadCredentialType(name) {
  if (!credCache.has(name)) credCache.set(name, studio.get(`/catalog/credentials/${enc(name)}/`).catch((e) => { credCache.delete(name); throw e; }));
  return credCache.get(name);
}

// ------------------------------------------------------------------ presentation

const CATEGORY_COLORS = {
  trigger: '#ff6d5a', AI: '#8b5cf6', 'Core Nodes': '#0ea5e9', 'Data & Storage': '#10b981', Communication: '#f59e0b',
  Development: '#64748b', Productivity: '#06b6d4', Marketing: '#ec4899', Sales: '#22c55e', Finance: '#eab308',
  Analytics: '#6366f1', 'Utility': '#94a3b8', Miscellaneous: '#94a3b8', 'Cybersecurity': '#ef4444', HITL: '#f97316',
};

export function entryFor(type) {
  return catalog.nodes.get(type) || null;
}

export function displayName(type) {
  return entryFor(type)?.displayName || String(type || '').split('.').pop().replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, (c) => c.toUpperCase());
}

// n8n's named icon colours (iconColor on node descriptions).
const ICON_COLORS = {
  amber: '#d97706', green: '#16a34a', 'dark-green': '#166534', 'light-green': '#22c55e', blue: '#2563eb', 'dark-blue': '#1e3a8a',
  'light-blue': '#0ea5e9', azure: '#0284c7', magenta: '#c026d3', purple: '#7c3aed', pink: '#db2777', red: '#ef4444', crimson: '#dc2626',
  orange: '#f97316', 'orange-red': '#ea580c', black: '#111827', gray: '#6b7280', primary: '#ff6d5a',
};

export function nodeColor(type) {
  const e = entryFor(type);
  if (!e) return '#94a3b8';
  if (e.iconColor && ICON_COLORS[e.iconColor]) return ICON_COLORS[e.iconColor];
  if (e.trigger) return CATEGORY_COLORS.trigger;
  const cat = e.codex?.categories?.[0];
  return CATEGORY_COLORS[cat] || '#64748b';
}

// n8n 2.x core nodes reference icons bundled in its editor ("node:webhook"); map them to glyphs.
const NODE_GLYPHS = {
  aggregate: 'collection', 'ai-agent': 'robot', 'ai-transform': 'magic', 'basic-llm-chain': 'link-45deg', calculator: 'calculator',
  'call-n8n-sub-workflow-tool': 'box-arrow-in-right', 'character-text-splitter': 'scissors', 'chat-memory-manager': 'memory', 'chat-trigger': 'chat-dots',
  code: 'code-slash', 'compare-datasets': 'layout-split', compression: 'file-zip', 'contextual-compression-retriever': 'funnel', 'convert-to-file': 'file-earmark-arrow-down',
  'data-table': 'table', 'date-and-time': 'calendar-event', 'default-data-loader': 'box-arrow-in-down', 'edit-fields': 'pencil-square', 'edit-image': 'image',
  'error-trigger': 'bug', 'execute-sub-workflow': 'box-arrow-in-right', 'execution-data': 'bookmark-star', 'extract-from-file': 'file-earmark-arrow-up', filter: 'funnel',
  'form-trigger': 'ui-checks', ftp: 'hdd-network', guardrails: 'shield-check', html: 'filetype-html', 'http-request': 'globe2', if: 'signpost-split',
  'information-extractor': 'card-text', 'item-list-output-parser': 'list-ol', limit: 'sort-numeric-down', 'loop-over-items': 'arrow-repeat', 'manual-trigger': 'hand-index',
  markdown: 'markdown', merge: 'union', 'model-selector': 'toggles', 'multiquery-retriever': 'search', n8n: 'diagram-3', 'n8n-trigger': 'lightning-charge',
  'no-operation': 'arrow-right', 'question-and-answer-chain': 'patch-question', 'read-write-files-from-disk': 'device-hdd', 'recursive-character-text-splitter': 'scissors',
  'remove-duplicates': 'subtract', 'rename-keys': 'input-cursor-text', 'respond-to-webhook': 'reply', 'rss-feed-trigger': 'rss', 'rss-read': 'rss',
  'schedule-trigger': 'clock', 'send-mail': 'envelope', 'sentiment-analysis': 'emoji-smile', 'simple-memory': 'memory', 'simple-vector-store': 'grid-3x3',
  sort: 'sort-down', 'split-out': 'diagram-2', 'sse-trigger': 'broadcast-pin', ssh: 'terminal', 'stop-and-error': 'exclamation-octagon',
  'structured-output-parser': 'braces', 'sub-workflow-trigger': 'box-arrow-in-down-right', 'summarization-chain': 'text-paragraph', summarize: 'sigma', switch: 'signpost-2',
  'text-classifier': 'tags', 'think-tool': 'lightbulb', 'token-splitter': 'scissors', totp: 'shield-lock', 'track-time-saved': 'stopwatch',
  'vector-store-question-answer-tool': 'patch-question', 'vector-store-retriever': 'search', wait: 'hourglass-split', webhook: 'broadcast', 'workflow-retriever': 'diagram-3', xml: 'filetype-xml',
};

const FALLBACK_ICONS = {
  'n8n-nodes-base.stickyNote': 'sticky', 'n8n-nodes-base.code': 'code-slash', 'n8n-nodes-base.if': 'signpost-split',
  'n8n-nodes-base.switch': 'signpost-2', 'n8n-nodes-base.merge': 'union', 'n8n-nodes-base.set': 'pencil-square',
  'n8n-nodes-base.httpRequest': 'globe2', 'n8n-nodes-base.webhook': 'broadcast', 'n8n-nodes-base.scheduleTrigger': 'clock',
  'n8n-nodes-base.manualTrigger': 'hand-index', 'n8n-nodes-base.wait': 'hourglass-split', 'n8n-nodes-base.noOp': 'dash-circle',
  'n8n-nodes-base.executeWorkflow': 'box-arrow-in-right', 'n8n-nodes-base.splitInBatches': 'arrow-repeat',
  'n8n-nodes-base.filter': 'funnel', 'n8n-nodes-base.respondToWebhook': 'reply', 'n8n-nodes-base.errorTrigger': 'bug',
};

function iconPath(entry, dark = false) {
  const u = entry?.iconUrl;
  if (!u) return null;
  if (typeof u === 'string') return u;
  return (dark ? u.dark : u.light) || u.light || u.dark || null;
}

/** <span class="n8s-icon"> with the node's SVG (proxied) or a coloured glyph. */
export function nodeIcon(type, { size = 28, dark = false, title } = {}) {
  const entry = entryFor(type);
  const path = iconPath(entry, dark);
  const wrap = h('span.n8s-icon', { style: { width: `${size}px`, height: `${size}px`, fontSize: `${Math.round(size * 0.62)}px`, color: nodeColor(type) }, title: title || displayName(type) });
  if (path) {
    const img = h('img', { src: iconUrl(path), alt: '', loading: 'lazy', decoding: 'async', draggable: 'false' });
    img.addEventListener('error', () => { img.replaceWith(icon(glyphFor(type, entry))); }, { once: true });
    wrap.appendChild(img);
  } else {
    wrap.appendChild(icon(glyphFor(type, entry)));
  }
  return wrap;
}

function glyphFor(type, entry) {
  const ref = typeof entry?.icon === 'string' ? entry.icon : '';
  if (ref.startsWith('node:') && NODE_GLYPHS[ref.slice(5)]) return NODE_GLYPHS[ref.slice(5)];
  if (ref.startsWith('fa:') && faToBi(ref.slice(3))) return faToBi(ref.slice(3));
  return FALLBACK_ICONS[type] || (entry?.trigger ? 'lightning-charge' : /langchain/.test(type) ? 'stars' : 'box');
}

const FA_BI = {
  'arrow-right': 'arrow-right', 'check-double': 'check2-all', cut: 'scissors', 'exchange-alt': 'arrow-left-right', 'file-export': 'box-arrow-up',
  'file-import': 'box-arrow-in-down', 'file-pdf': 'file-earmark-pdf', inbox: 'inbox', 'network-wired': 'diagram-3', play: 'play-fill', 'sticky-note': 'sticky',
  'th-large': 'grid', code: 'code-slash', 'map-signs': 'signpost-split', 'random': 'shuffle', 'sync': 'arrow-repeat', 'clock': 'clock',
  'pause-circle': 'pause-circle', 'pen': 'pen', 'filter': 'funnel', 'globe': 'globe2', 'sort-amount-down': 'sort-down',
  'file': 'file-earmark', 'database': 'database', 'table': 'table', 'calendar': 'calendar', 'envelope': 'envelope',
  'terminal': 'terminal', 'robot': 'robot', 'brain': 'cpu', 'comments': 'chat-dots', 'bolt': 'lightning-charge',
  'key': 'key', 'lock': 'lock', 'cog': 'gear', 'cogs': 'gears', 'search': 'search', 'list': 'list-ul', 'link': 'link-45deg',
  'tools': 'tools', 'hourglass': 'hourglass', 'arrow-right': 'arrow-right', 'check': 'check2', 'stop': 'stop-circle',
};
const faToBi = (fa) => (fa ? FA_BI[fa] || null : null);

export function credentialIcon(name, size = 22) {
  const c = catalog.credentials.get(name);
  const path = iconPath(c);
  if (path) return h('span.n8s-icon', { style: { width: `${size}px`, height: `${size}px` } }, h('img', { src: iconUrl(path), alt: '', loading: 'lazy' }));
  const node = [...(c?.supportedNodes || [])][0];
  if (node && catalog.nodes.has(node)) return nodeIcon(node, { size });
  return h('span.n8s-icon', { style: { width: `${size}px`, height: `${size}px` } }, icon('key'));
}

export function credentialDisplayName(name) {
  return catalog.credentials.get(name)?.displayName || name;
}

/** Palette categories derived from codex data. */
export function categoryOf(entry) {
  if (entry.trigger) return 'Triggers';
  const cats = entry.codex?.categories || [];
  if (cats.includes('AI') || /langchain/.test(entry.name)) return 'AI';
  if (cats.includes('Core Nodes')) {
    const sub = entry.codex?.subcategories?.['Core Nodes']?.[0];
    if (sub === 'Flow') return 'Flow';
    if (sub === 'Data Transformation') return 'Data transformation';
    if (sub === 'Helpers') return 'Helpers';
    return 'Core';
  }
  return 'Apps';
}

export const PALETTE_ORDER = ['Triggers', 'AI', 'Flow', 'Data transformation', 'Core', 'Helpers', 'Apps'];
