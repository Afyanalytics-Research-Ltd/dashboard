// Redash query parameters: detection, definitions, input widgets and serialization.
//
// A parameter definition (query.options.parameters[i]) looks like
//   { name, title, type, value, global, enumOptions, queryId, multiValuesOptions, locals }
// Types: text, number, enum, query, date, datetime-local, datetime-with-seconds,
//        date-range, datetime-range, datetime-range-with-seconds.
// Dynamic date values are tokens Redash understands (d_now, d_last_30_days, ...);
// they are stored as-is and resolved to concrete dates only when executing.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { rd, enc } from './api.js';

export const PARAM_TYPES = [
  { value: 'text', label: 'Text' },
  { value: 'number', label: 'Number' },
  { value: 'enum', label: 'Dropdown list' },
  { value: 'query', label: 'Query-based dropdown' },
  { value: 'date', label: 'Date' },
  { value: 'datetime-local', label: 'Date and time' },
  { value: 'datetime-with-seconds', label: 'Date and time (with seconds)' },
  { value: 'date-range', label: 'Date range' },
  { value: 'datetime-range', label: 'Date and time range' },
  { value: 'datetime-range-with-seconds', label: 'Date and time range (with seconds)' },
];

const isRange = (type) => type?.endsWith('range') || type?.includes('range-');
const isDateType = (type) => type?.startsWith('date');
const withTime = (type) => type?.startsWith('datetime');
const withSeconds = (type) => type?.endsWith('with-seconds');

/** Unique {{ name }} occurrences in query text (Mustache sections like {{#x}} are ignored). */
export function detectParams(sql) {
  const names = [];
  for (const m of String(sql || '').matchAll(/\{\{\s*([^#^/{}\s][^{}]*?)\s*\}\}/g)) {
    const name = m[1].replace(/\.(start|end)$/, '');
    if (!names.includes(name)) names.push(name);
  }
  return names;
}

/** Merge detected names with existing definitions: keep known, add new as text, drop missing. */
export function syncParams(sql, existing = []) {
  const byName = new Map(existing.map((p) => [p.name, p]));
  return detectParams(sql).map((name) => byName.get(name) || {
    name, title: titleize(name), type: 'text', value: '', global: false, parentQueryId: undefined, locals: [],
  });
}

const titleize = (s) => s.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

// ------------------------------------------------------------------ dynamic dates

const pad = (n) => String(n).padStart(2, '0');
const ymd = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const ymdhm = (d, secs) => `${ymd(d)} ${pad(d.getHours())}:${pad(d.getMinutes())}${secs ? `:${pad(d.getSeconds())}` : ''}`;
const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate());
const endOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate(), 23, 59, 59);
const addDays = (d, n) => new Date(d.getFullYear(), d.getMonth(), d.getDate() + n);

export const DYNAMIC_DATES = [
  { value: 'd_now', label: 'Today / now', resolve: () => new Date() },
  { value: 'd_yesterday', label: 'Yesterday', resolve: () => addDays(new Date(), -1) },
];

export const DYNAMIC_RANGES = [
  { value: 'd_today', label: 'Today', resolve: (n) => [startOfDay(n), endOfDay(n)] },
  { value: 'd_yesterday', label: 'Yesterday', resolve: (n) => [startOfDay(addDays(n, -1)), endOfDay(addDays(n, -1))] },
  { value: 'd_this_week', label: 'This week', resolve: (n) => [startOfDay(addDays(n, -n.getDay())), endOfDay(addDays(n, 6 - n.getDay()))] },
  { value: 'd_this_month', label: 'This month', resolve: (n) => [new Date(n.getFullYear(), n.getMonth(), 1), endOfDay(new Date(n.getFullYear(), n.getMonth() + 1, 0))] },
  { value: 'd_this_year', label: 'This year', resolve: (n) => [new Date(n.getFullYear(), 0, 1), endOfDay(new Date(n.getFullYear(), 11, 31))] },
  { value: 'd_last_week', label: 'Last week', resolve: (n) => [startOfDay(addDays(n, -n.getDay() - 7)), endOfDay(addDays(n, -n.getDay() - 1))] },
  { value: 'd_last_month', label: 'Last month', resolve: (n) => [new Date(n.getFullYear(), n.getMonth() - 1, 1), endOfDay(new Date(n.getFullYear(), n.getMonth(), 0))] },
  { value: 'd_last_year', label: 'Last year', resolve: (n) => [new Date(n.getFullYear() - 1, 0, 1), endOfDay(new Date(n.getFullYear() - 1, 11, 31))] },
  { value: 'd_last_7_days', label: 'Last 7 days', resolve: (n) => [startOfDay(addDays(n, -7)), endOfDay(n)] },
  { value: 'd_last_14_days', label: 'Last 14 days', resolve: (n) => [startOfDay(addDays(n, -14)), endOfDay(n)] },
  { value: 'd_last_30_days', label: 'Last 30 days', resolve: (n) => [startOfDay(addDays(n, -30)), endOfDay(n)] },
  { value: 'd_last_60_days', label: 'Last 60 days', resolve: (n) => [startOfDay(addDays(n, -60)), endOfDay(n)] },
  { value: 'd_last_90_days', label: 'Last 90 days', resolve: (n) => [startOfDay(addDays(n, -90)), endOfDay(n)] },
  { value: 'd_last_12_months', label: 'Last 12 months', resolve: (n) => [new Date(n.getFullYear() - 1, n.getMonth(), n.getDate()), endOfDay(n)] },
];

function fmtFor(type, d) {
  if (!withTime(type)) return ymd(d);
  return ymdhm(d, withSeconds(type));
}

/** Concrete API value for a parameter (resolves dynamic dates, splits multi-values). */
export function resolveValue(param, value = param.value) {
  const now = new Date();
  if (isDateType(param.type) && isRange(param.type)) {
    if (typeof value === 'string') {
      const dyn = DYNAMIC_RANGES.find((d) => d.value === value);
      if (dyn) {
        const [a, b] = dyn.resolve(now);
        return { start: fmtFor(param.type, a), end: fmtFor(param.type, b) };
      }
    }
    return value && typeof value === 'object' ? { start: value.start || '', end: value.end || '' } : value;
  }
  if (isDateType(param.type) && typeof value === 'string') {
    const dyn = DYNAMIC_DATES.find((d) => d.value === value);
    if (dyn) return fmtFor(param.type, dyn.resolve(now));
  }
  if (param.type === 'number' && value !== '' && value != null) return Number(value);
  return value;
}

/** {name: concreteValue} for POST /api/queries/{id}/results. */
export function serializeParams(params, values = {}) {
  const out = {};
  for (const p of params || []) out[p.name] = resolveValue(p, values[p.name] ?? p.value);
  return out;
}

export function missingParams(params, values = {}) {
  return (params || []).filter((p) => {
    const v = values[p.name] ?? p.value;
    if (v == null || v === '') return true;
    if (Array.isArray(v)) return !v.length;
    if (typeof v === 'object') return !v.start || !v.end;
    return false;
  }).map((p) => p.title || p.name);
}

/** Human summary of a parameter value (for chips and URLs). */
export function describeValue(param, value = param.value) {
  if (value == null || value === '') return '—';
  const dyn = [...DYNAMIC_DATES, ...DYNAMIC_RANGES].find((d) => d.value === value);
  if (dyn) return dyn.label;
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'object') return `${value.start || '…'} → ${value.end || '…'}`;
  return String(value);
}

// ------------------------------------------------------------------ widgets

const enumOptions = (p) => String(p.enumOptions || '').split('\n').map((s) => s.trim()).filter(Boolean);

function inputType(type) {
  if (!withTime(type)) return 'date';
  return 'datetime-local';
}
const toInput = (type, v) => (v ? String(v).replace(' ', 'T').slice(0, withSeconds(type) ? 19 : 16) : '');
const fromInput = (type, v) => (v ? v.replace('T', ' ') : '');

/**
 * Input widget for one parameter. onChange(value) fires on every edit.
 * dropdownFetcher(param) -> Promise<[{name, value}]> loads query-based dropdown options.
 */
export function paramInput(param, value, onChange, { dropdownFetcher } = {}) {
  const multi = !!param.multiValuesOptions;
  const id = `rdp-${param.name}-${Math.random().toString(36).slice(2, 6)}`;
  switch (param.type) {
    case 'number': {
      const el = h('input.form-control.form-control-sm', { id, type: 'number', value: value ?? '', step: 'any' });
      el.addEventListener('input', debounce(() => onChange(el.value === '' ? '' : Number(el.value)), 150));
      return el;
    }
    case 'enum':
    case 'query': {
      const sel = h('select.form-select.form-select-sm', { id, multiple: multi, size: multi ? 4 : undefined });
      const fill = (opts) => {
        const current = multi ? new Set(Array.isArray(value) ? value.map(String) : [String(value)]) : new Set([String(value ?? '')]);
        mount(sel,
          !multi && h('option', { value: '' }, '— choose —'),
          opts.map((o) => h('option', { value: o.value, selected: current.has(String(o.value)) }, o.name)));
      };
      if (param.type === 'enum') fill(enumOptions(param).map((o) => ({ name: o, value: o })));
      else {
        mount(sel, h('option', 'Loading options…'));
        (dropdownFetcher ? dropdownFetcher(param) : fetchDropdown(param))
          .then(fill)
          .catch((err) => mount(sel, h('option', `Could not load options: ${err.message}`)));
      }
      sel.addEventListener('change', () => onChange(multi ? [...sel.selectedOptions].map((o) => o.value) : sel.value));
      return sel;
    }
    case 'date':
    case 'datetime-local':
    case 'datetime-with-seconds': {
      const isDyn = DYNAMIC_DATES.some((d) => d.value === value);
      const inp = h('input.form-control.form-control-sm', { id, type: inputType(param.type), step: withSeconds(param.type) ? 1 : undefined, value: isDyn ? '' : toInput(param.type, value) });
      const quick = h('select.form-select.form-select-sm.rd-date-quick', { title: 'Dynamic value' },
        h('option', { value: '' }, 'Fixed'),
        DYNAMIC_DATES.map((d) => h('option', { value: d.value, selected: d.value === value }, d.label)));
      inp.addEventListener('change', () => { quick.value = ''; onChange(fromInput(param.type, inp.value)); });
      quick.addEventListener('change', () => { if (quick.value) { inp.value = ''; onChange(quick.value); } });
      return h('div.rd-param-date', inp, quick);
    }
    default: {
      if (isRange(param.type)) return rangeInput(param, value, onChange, id);
      const el = h('input.form-control.form-control-sm', { id, type: 'text', value: Array.isArray(value) ? value.join(',') : value ?? '', placeholder: multi ? 'comma, separated' : '' });
      el.addEventListener('input', debounce(() => onChange(multi ? el.value.split(',').map((s) => s.trim()).filter(Boolean) : el.value), 150));
      return el;
    }
  }
}

function rangeInput(param, value, onChange, id) {
  const dyn = typeof value === 'string' ? value : '';
  const v = value && typeof value === 'object' ? value : {};
  const a = h('input.form-control.form-control-sm', { id, type: inputType(param.type), step: withSeconds(param.type) ? 1 : undefined, value: toInput(param.type, v.start), 'aria-label': 'Start' });
  const b = h('input.form-control.form-control-sm', { type: inputType(param.type), step: withSeconds(param.type) ? 1 : undefined, value: toInput(param.type, v.end), 'aria-label': 'End' });
  const presets = h('select.form-select.form-select-sm.rd-date-quick', { title: 'Quick ranges' },
    h('option', { value: '' }, 'Custom range'),
    DYNAMIC_RANGES.map((d) => h('option', { value: d.value, selected: d.value === dyn }, d.label)));
  const sync = () => {
    presets.value = '';
    onChange({ start: fromInput(param.type, a.value), end: fromInput(param.type, b.value) });
  };
  a.addEventListener('change', sync);
  b.addEventListener('change', sync);
  presets.addEventListener('change', () => {
    if (!presets.value) return;
    const [s, e] = DYNAMIC_RANGES.find((d) => d.value === presets.value).resolve(new Date());
    a.value = toInput(param.type, fmtFor(param.type, s));
    b.value = toInput(param.type, fmtFor(param.type, e));
    onChange(presets.value);
  });
  if (dyn) {
    const found = DYNAMIC_RANGES.find((d) => d.value === dyn);
    if (found) {
      const [s, e] = found.resolve(new Date());
      a.value = toInput(param.type, fmtFor(param.type, s));
      b.value = toInput(param.type, fmtFor(param.type, e));
    }
  }
  return h('div.rd-param-range', presets, a, h('span.rd-range-sep', icon('arrow-right')), b);
}

const dropdownCache = new Map();
/** Options for a query-based dropdown: GET /api/queries/{id}/dropdowns/{dropdown_id} when nested, else /dropdown. */
export function fetchDropdown(param, { ownerQueryId } = {}) {
  const key = `${ownerQueryId || ''}:${param.queryId}`;
  if (!dropdownCache.has(key)) {
    const path = ownerQueryId
      ? `/api/queries/${enc(ownerQueryId)}/dropdowns/${enc(param.queryId)}`
      : `/api/queries/${enc(param.queryId)}/dropdown`;
    dropdownCache.set(key, rd.get(path).catch((err) => { dropdownCache.delete(key); throw err; }));
  }
  return dropdownCache.get(key);
}

/**
 * A horizontal bar of parameter inputs. values is mutated in place.
 * Returns {el, values, setValues(obj)}; onApply() fires on Enter / Apply.
 */
export function paramBar(params, values, { onChange, onApply, applyLabel = 'Apply', ownerQueryId, sticky = false, compact = false } = {}) {
  const el = h('div.rd-params', { class: [sticky && 'rd-params-sticky', compact && 'rd-params-compact'] });
  const applyBtn = h('button.btn.btn-sm.btn-primary.rd-params-apply', { type: 'button', onclick: () => onApply?.(values), disabled: true },
    icon('play-fill', 'me-1'), applyLabel);
  let dirty = false;
  const markDirty = () => { dirty = true; applyBtn.disabled = false; el.classList.add('dirty'); };
  const render = () => {
    mount(el, params.map((p) => {
      const input = paramInput(p, values[p.name] ?? p.value, (v) => { values[p.name] = v; markDirty(); onChange?.(p, v); },
        { dropdownFetcher: (pp) => fetchDropdown(pp, { ownerQueryId }) });
      return h('label.rd-param', h('span.rd-param-label', p.title || p.name), input);
    }), onApply && h('div.rd-param-actions', applyBtn));
  };
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.matches('input')) { e.preventDefault(); onApply?.(values); }
  });
  render();
  return {
    el,
    values,
    get dirty() { return dirty; },
    reset() { dirty = false; applyBtn.disabled = true; el.classList.remove('dirty'); },
    setValues(next) { Object.assign(values, next); render(); },
  };
}

/** Editor for one parameter's definition (title, type, options...). Returns {el, value()}. */
export function paramDefinitionForm(param, { queries = [] } = {}) {
  const def = structuredClone(param);
  const title = h('input.form-control.form-control-sm', { value: def.title || '' });
  const type = h('select.form-select.form-select-sm', PARAM_TYPES.map((t) => h('option', { value: t.value, selected: t.value === def.type }, t.label)));
  const enumTa = h('textarea.form-control.form-control-sm.af-mono', { rows: 4, placeholder: 'One option per line' }, def.enumOptions || '');
  const querySel = h('select.form-select.form-select-sm',
    h('option', { value: '' }, '— choose the query that returns the options —'),
    queries.map((q) => h('option', { value: q.id, selected: Number(def.queryId) === q.id }, `#${q.id} ${q.name}`)));
  const multi = h('input.form-check-input', { type: 'checkbox', checked: !!def.multiValuesOptions });
  const quote = h('select.form-select.form-select-sm',
    [['', 'None'], ["'", "Single quote ( ' )"], ['"', 'Double quote ( " )']].map(([v, l]) => h('option', { value: v, selected: (def.multiValuesOptions?.quoteCharacter ?? '') === v }, l)));
  const extra = h('div.rd-paramdef-extra');
  const renderExtra = () => {
    const t = type.value;
    mount(extra,
      t === 'enum' && h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Options'), enumTa),
      t === 'query' && h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Options query'), querySel,
        h('div.form-text.small', 'The query must return a "value" column (and optionally "name").')),
      (t === 'enum' || t === 'query' || t === 'text') && h('div.af-form-row',
        h('label.form-check.small', multi, h('span.ms-2', 'Allow multiple values'))),
      (t === 'enum' || t === 'query') && h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Quote each value with'), quote));
  };
  type.addEventListener('change', renderExtra);
  renderExtra();
  return {
    el: h('div.af-form',
      h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Keyword'), h('code.rd-param-keyword', `{{ ${def.name} }}`)),
      h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Title'), title),
      h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Type'), type),
      extra),
    value() {
      const out = { ...def, title: title.value || def.name, type: type.value };
      delete out.enumOptions; delete out.queryId;
      if (out.type === 'enum') out.enumOptions = enumTa.value;
      if (out.type === 'query') out.queryId = querySel.value ? Number(querySel.value) : null;
      if (multi.checked && ['enum', 'query', 'text'].includes(out.type)) {
        out.multiValuesOptions = { prefix: '', suffix: '', separator: ',', quoteCharacter: quote.value };
      } else delete out.multiValuesOptions;
      if (out.type !== def.type) out.value = isRange(out.type) ? 'd_last_30_days' : isDateType(out.type) ? 'd_now' : '';
      return out;
    },
  };
}
