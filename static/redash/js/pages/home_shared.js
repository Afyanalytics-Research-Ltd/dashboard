// Helpers shared by the home, activity and system pages (not a route).

import { h, icon } from '../../../ui/js/dom.js';
import { urls } from '../core/urls.js';

const OBJECT_LABELS = {
  query: 'query', dashboard: 'dashboard', visualization: 'visualization', widget: 'widget',
  data_source: 'data source', datasource: 'data source', alert: 'alert', destination: 'alert destination',
  user: 'user', group: 'group', query_snippet: 'query snippet', rq_status: 'queue status',
  outdated_queries: 'outdated queries', page: 'page', api_key: 'API key', query_result: 'query result',
  settings: 'settings', organization: 'organization',
};

const ACTIONS = {
  create: { verb: 'created', icon: 'plus-circle', tone: 'green' },
  edit: { verb: 'edited', icon: 'pencil-square', tone: 'blue' },
  update: { verb: 'updated', icon: 'pencil-square', tone: 'blue' },
  view: { verb: 'viewed', icon: 'eye', tone: 'slate' },
  view_source: { verb: 'viewed the SQL of', icon: 'code-slash', tone: 'slate' },
  list: { verb: 'listed', icon: 'list-ul', tone: 'slate' },
  execute_query: { verb: 'ran a query on', icon: 'play-circle', tone: 'teal' },
  execute: { verb: 'executed', icon: 'play-circle', tone: 'teal' },
  search: { verb: 'searched', icon: 'search', tone: 'slate' },
  archive: { verb: 'archived', icon: 'archive', tone: 'amber' },
  delete: { verb: 'deleted', icon: 'trash', tone: 'red' },
  fork: { verb: 'forked', icon: 'diagram-2', tone: 'purple' },
  share: { verb: 'shared', icon: 'share', tone: 'purple' },
  unshare: { verb: 'stopped sharing', icon: 'share', tone: 'amber' },
  favorite: { verb: 'starred', icon: 'star', tone: 'amber' },
  unfavorite: { verb: 'unstarred', icon: 'star', tone: 'slate' },
  load_favorites: { verb: 'opened favourite', icon: 'star', tone: 'slate' },
  activate_api_key: { verb: 'used an API key for', icon: 'key', tone: 'slate' },
  regenerate_api_key: { verb: 'regenerated the API key of', icon: 'key', tone: 'amber' },
  add_member: { verb: 'added a member to', icon: 'person-plus', tone: 'green' },
  remove_member: { verb: 'removed a member from', icon: 'person-dash', tone: 'amber' },
  add_data_source: { verb: 'granted a data source to', icon: 'database-add', tone: 'green' },
  remove_data_source: { verb: 'revoked a data source from', icon: 'database-dash', tone: 'amber' },
  change_data_source_permission: { verb: 'changed data source access of', icon: 'shield', tone: 'blue' },
  pause: { verb: 'paused', icon: 'pause-circle', tone: 'amber' },
  resume: { verb: 'resumed', icon: 'play', tone: 'green' },
  test: { verb: 'tested', icon: 'plug', tone: 'blue' },
  mute: { verb: 'muted', icon: 'bell-slash', tone: 'amber' },
  unmute: { verb: 'unmuted', icon: 'bell', tone: 'green' },
  subscribe: { verb: 'subscribed to', icon: 'bell', tone: 'green' },
  invite: { verb: 'invited', icon: 'envelope', tone: 'green' },
  login: { verb: 'signed in', icon: 'box-arrow-in-right', tone: 'slate' },
};

export const actionInfo = (action) => ACTIONS[action] || { verb: String(action || 'did').replace(/_/g, ' '), icon: 'dot', tone: 'slate' };
export const objectLabel = (type) => OBJECT_LABELS[type] || String(type || '').replace(/_/g, ' ');
export const ACTION_KEYS = Object.keys(ACTIONS);
/** Bookkeeping events that drown out meaningful activity. */
export const NOISE_ACTIONS = new Set(['list', 'load_favorites', 'activate_api_key']);

/** Link to the object of an event inside Studio, or null. */
export function eventHref(e) {
  const id = e.object_id;
  if (!id || !/^\d+$/.test(String(id))) return null;
  switch (e.object_type) {
    case 'query': return urls.query(id);
    case 'dashboard': return urls.dashboard(id);
    case 'data_source':
    case 'datasource': return urls.dataSource(id);
    case 'alert': return urls.alert(id);
    case 'user': return urls.user(id);
    case 'group': return urls.group(id);
    default: return null;
  }
}

/** The event rendered as a sentence: "admin ran a query on data source #1". */
export function eventSentence(e, { names = {} } = {}) {
  const info = actionInfo(e.action);
  const who = e.user_name?.startsWith('ApiKey') ? h('span.rd-ev-apikey', icon('key'), ' API key') : h('b', e.user_name || 'Someone');
  // List events carry a route ("admin/data_sources") rather than an id in object_id.
  const id = /^\d+$/.test(String(e.object_id ?? '')) ? e.object_id : null;
  const label = objectLabel(e.object_type);
  const name = names[`${e.object_type}:${id}`];
  const plural = e.action === 'list' || e.action === 'load_favorites';
  const target = id
    ? (eventHref(e) ? h('a', { href: eventHref(e) }, name || `${label} #${id}`) : h('span', `${label} #${id}`))
    : h('span', plural && !/s$/.test(label) ? `${label}s` : label);
  const parts = [who, ` ${info.verb} `, target];
  if (e.action === 'search' && e.details?.term) parts.push(h('span', ' for ', h('q', e.details.term)));
  if (e.action === 'execute_query' && e.details?.query_id) parts.push(h('span.text-muted', ' (', h('a', { href: urls.query(e.details.query_id) }, `query #${e.details.query_id}`), ')'));
  return h('span.rd-ev-text', parts);
}

/** "Today", "Yesterday" or a date label for grouping. */
export function dayLabel(iso) {
  const d = new Date(iso);
  const today = new Date();
  const start = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((start(today) - start(d)) / 86400000);
  if (diff === 0) return 'Today';
  if (diff === 1) return 'Yesterday';
  if (diff < 7) return d.toLocaleDateString(undefined, { weekday: 'long' });
  return d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: today.getFullYear() === d.getFullYear() ? undefined : 'numeric' });
}

export const greeting = () => {
  const hr = new Date().getHours();
  return hr < 5 ? 'Working late' : hr < 12 ? 'Good morning' : hr < 18 ? 'Good afternoon' : 'Good evening';
};

const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** A number that counts up from 0 (or its previous value) when first shown. */
export function countUp(value, { duration = 700, format = (n) => Math.round(n).toLocaleString() } = {}) {
  const target = Number(value);
  const el = h('span.rdh-count', { dataset: { value: String(value ?? '') } });
  if (!Number.isFinite(target)) { el.textContent = value ?? '—'; return el; }
  if (reducedMotion() || target === 0) { el.textContent = format(target); return el; }
  el.textContent = format(0);
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - (1 - t) ** 3;
    el.textContent = format(target * eased);
    if (t < 1 && el.isConnected !== false) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  return el;
}

/** Seconds -> "1 min", "2 h" etc. for queue ages. */
export const shortAge = (seconds) => {
  if (seconds == null) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)} h`;
  return `${(seconds / 86400).toFixed(1)} d`;
};
