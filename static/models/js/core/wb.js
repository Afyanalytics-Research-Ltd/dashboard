// W&B domain helpers: run states, config/summary parsing, run filters, metric keys.

import { h, icon } from '../../../ui/js/dom.js';
import { parseJSON } from './api.js';

export const RUN_STATES = {
  running: { label: 'Running', color: '#0891b2', bg: '#cffafe', icon: 'arrow-repeat' },
  finished: { label: 'Finished', color: '#16a34a', bg: '#dcfce7', icon: 'check-circle-fill' },
  crashed: { label: 'Crashed', color: '#dc2626', bg: '#fee2e2', icon: 'x-octagon-fill' },
  failed: { label: 'Failed', color: '#dc2626', bg: '#fee2e2', icon: 'x-circle-fill' },
  killed: { label: 'Killed', color: '#ea580c', bg: '#ffedd5', icon: 'slash-circle' },
  preempted: { label: 'Preempted', color: '#a16207', bg: '#fef3c7', icon: 'pause-circle' },
  pending: { label: 'Pending', color: '#64748b', bg: '#e2e8f0', icon: 'hourglass-split' },
};
export const runStateInfo = (s) => RUN_STATES[s] || { label: s || 'unknown', color: '#94a3b8', bg: '#f1f5f9', icon: 'circle' };

export function runStateBadge(state) {
  const i = runStateInfo(state);
  return h('span.af-badge', { class: state === 'running' && 'af-pulse', style: { color: i.color, background: i.bg } },
    h('span.af-dot', { style: { background: i.color } }), i.label);
}

/** Stable, distinguishable colour for a run (by its index or id hash). */
export const RUN_COLORS = ['#0072CE', '#E5484D', '#0BB99F', '#F5A623', '#7C3AED', '#12A594', '#D6409F', '#3E63DD', '#E54D2E', '#8E4EC6',
  '#46A758', '#FFB224', '#0091FF', '#D6409F', '#978365', '#5B5BD6', '#29A383', '#E93D82', '#6E56CF', '#AD7F58'];
export function runColor(run, index) {
  if (index != null) return RUN_COLORS[index % RUN_COLORS.length];
  let hsh = 0;
  for (const ch of String(run?.name || run)) hsh = (hsh * 31 + ch.charCodeAt(0)) | 0;
  return RUN_COLORS[Math.abs(hsh) % RUN_COLORS.length];
}

/**
 * W&B config is {key: {value, desc}} (and may be nested). Flatten to {"a.b": value}, skipping _wandb internals.
 */
export function parseConfig(raw) {
  const cfg = parseJSON(raw, {});
  const out = {};
  const walk = (obj, prefix) => {
    for (const [k, v] of Object.entries(obj || {})) {
      if (!prefix && k.startsWith('_wandb')) continue;
      const key = prefix ? `${prefix}.${k}` : k;
      if (v && typeof v === 'object' && !Array.isArray(v) && !('_type' in v)) walk(v, key);
      else out[key] = v;
    }
  };
  for (const [k, v] of Object.entries(cfg)) {
    if (k.startsWith('_wandb')) continue;
    const value = v && typeof v === 'object' && 'value' in v ? v.value : v;
    if (value && typeof value === 'object' && !Array.isArray(value)) walk(value, k);
    else out[k] = value;
  }
  return out;
}

/** Summary metrics: drop internal keys and media/table objects (keep scalars for tables & charts). */
export function parseSummary(raw, { scalarsOnly = true } = {}) {
  const s = parseJSON(raw, {});
  const out = {};
  for (const [k, v] of Object.entries(s)) {
    if (k.startsWith('_') && k !== '_runtime' && k !== '_step') continue;
    if (scalarsOnly && v && typeof v === 'object') continue;
    out[k] = v;
  }
  return out;
}

/** Media/table entries in a summary: [{key, type, value}] where type is image-file, table-file, histogram... */
export function summaryMedia(raw) {
  const s = parseJSON(raw, {});
  return Object.entries(s).filter(([, v]) => v && typeof v === 'object' && v._type).map(([key, v]) => ({ key, type: v._type, value: v }));
}

export function displayName(run) {
  return run?.displayName || run?.name || '—';
}

export function fmtMetric(v) {
  if (v == null || v === '') return '—';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
    return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

// ------------------------------------------------------------------ server-side run filters

/**
 * Build W&B's Mongo-style run filter from UI conditions.
 * condition: { field: 'state' | 'tags' | 'group' | 'jobType' | 'config.<k>' | 'summary.<k>' | 'displayName' | 'createdAt' | 'sweep', op, value }
 * op: '=', '!=', '>', '>=', '<', '<=', 'in', 'contains', 'exists'
 */
export function buildFilter(conditions = [], { search, includeCrashed = true } = {}) {
  const and = [];
  for (const c of conditions) {
    if (c.value === '' && c.op !== 'exists') continue;
    let field = c.field;
    if (field.startsWith('config.')) field = `config.${field.slice(7)}.value`;
    else if (field.startsWith('summary.')) field = `summary_metrics.${field.slice(8)}`;
    const num = c.value !== '' && !Number.isNaN(Number(c.value)) && typeof c.value !== 'boolean' ? Number(c.value) : c.value;
    switch (c.op) {
      case '=': and.push({ [field]: num }); break;
      case '!=': and.push({ [field]: { $ne: num } }); break;
      case '>': and.push({ [field]: { $gt: num } }); break;
      case '>=': and.push({ [field]: { $gte: num } }); break;
      case '<': and.push({ [field]: { $lt: num } }); break;
      case '<=': and.push({ [field]: { $lte: num } }); break;
      case 'in': and.push({ [field]: { $in: String(c.value).split(',').map((x) => x.trim()).filter(Boolean) } }); break;
      case 'contains': and.push({ [field]: { $regex: escapeRegex(String(c.value)) } }); break;
      case 'exists': and.push({ [field]: { $exists: true } }); break;
      default: break;
    }
  }
  if (search) and.push({ $or: [{ displayName: { $regex: escapeRegex(search) } }, { name: { $regex: escapeRegex(search) } }, { tags: { $in: [search] } }] });
  if (!includeCrashed) and.push({ state: { $nin: ['crashed', 'failed'] } });
  return and.length ? JSON.stringify({ $and: and }) : JSON.stringify({});
}

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** W&B order string: '-createdAt', '+summary_metrics.val/accuracy', '-config.lr.value'. */
export function orderString(key, dir = 'desc') {
  let field = key;
  if (key.startsWith('config.')) field = `config.${key.slice(7)}.value`;
  else if (key.startsWith('summary.')) field = `summary_metrics.${key.slice(8)}`;
  return `${dir === 'asc' ? '+' : '-'}${field}`;
}

// ------------------------------------------------------------------ metric keys

/** Group metric names by their prefix ("train/loss" -> "train"). Ungrouped go under "Charts". */
export function groupMetrics(keys) {
  const groups = new Map();
  for (const k of keys) {
    if (k.startsWith('_') || k.startsWith('system/')) continue;
    const g = k.includes('/') ? k.split('/')[0] : 'Charts';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(k);
  }
  for (const list of groups.values()) list.sort();
  return [...groups.entries()].sort(([a], [b]) => (a === 'Charts' ? 1 : b === 'Charts' ? -1 : a.localeCompare(b)));
}

export function runLink(entity, project, run) {
  return `/p/${encodeURIComponent(entity)}/${encodeURIComponent(project)}/runs/${encodeURIComponent(run.name)}`;
}

export function stateIcon(state) {
  return icon(runStateInfo(state).icon);
}
