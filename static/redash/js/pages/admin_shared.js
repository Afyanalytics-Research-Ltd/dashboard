// Helpers shared by the Studio admin pages (alerts, data sources, destinations,
// users, groups, snippets, settings). Not a route.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { modal, emptyState } from '../../../ui/js/ui.js';
import { rd, enc, redashLink } from '../core/api.js';
import { dataSourceIcon } from '../core/common.js';

// Redash returns secret option values masked with this placeholder; sending it back
// unchanged tells Redash to keep the stored secret (ConfigurationContainer.update).
export const SECRET_PLACEHOLDER = '--------';

const humanize = (key) => key.replace(/File$/, '').replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2')
  .replace(/^./, (c) => c.toUpperCase());

const LONG_FIELDS = /(key|cert|certificate|json|credentials|private|query|template|body)/i;

/** Redash may declare `secret` as a list or a single string. */
const secretList = (schema) => (Array.isArray(schema.secret) ? schema.secret : schema.secret ? [schema.secret] : []);

/**
 * Form generated from a Redash configuration_schema (data sources and destinations).
 *   configForm(schema, values, { isNew })
 * Handles order, required, secret (masked), extendedEnum, booleans, numbers, *File uploads,
 * defaults and extra_options (collapsed "Advanced" section).
 * Returns { el, values() -> options (throws on validation error), isDirty() }.
 */
export function configForm(schema = {}, values = {}, { isNew = false } = {}) {
  const props = schema.properties || {};
  const required = new Set(schema.required || []);
  const secrets = new Set(secretList(schema));
  const extra = new Set(schema.extra_options || []);
  const order = [...(schema.order || []), ...Object.keys(props).filter((k) => !(schema.order || []).includes(k))]
    .filter((k) => props[k]);
  const getters = {};
  const initial = {};

  const field = (key) => {
    const p = props[key];
    const id = `rd-cfg-${key}-${Math.random().toString(36).slice(2, 6)}`;
    const label = p.title || humanize(key);
    const isSecret = secrets.has(key);
    const current = values[key] ?? (isNew ? p.default : undefined);
    initial[key] = current;
    let input;
    let get;
    if (p.extendedEnum || p.enum) {
      const opts = (p.extendedEnum || p.enum.map((v) => ({ value: v, name: v })));
      input = h('select.form-select.form-select-sm', { id },
        !required.has(key) && h('option', { value: '' }, '— not set —'),
        opts.map((o) => h('option', { value: o.value, selected: String(o.value) === String(current ?? '') }, o.name)));
      get = () => input.value || undefined;
    } else if (p.type === 'boolean') {
      input = h('div.form-check.form-switch', h('input.form-check-input', { id, type: 'checkbox', checked: !!current }));
      get = () => input.querySelector('input').checked;
    } else if (p.type === 'number') {
      input = h('input.form-control.form-control-sm', { id, type: 'number', step: 'any', value: current ?? '', placeholder: p.default != null ? `default ${p.default}` : '' });
      get = () => (input.value === '' ? undefined : Number(input.value));
    } else if (key.endsWith('File')) {
      // Redash stores uploaded files base64-encoded in the option value.
      let encoded = current;
      const status = h('span.small.text-muted', current ? (current === SECRET_PLACEHOLDER ? 'A file is stored (hidden).' : 'A file is stored.') : 'No file');
      const file = h('input.form-control.form-control-sm', { id, type: 'file' });
      file.addEventListener('change', () => {
        const f = file.files[0];
        if (!f) return;
        const reader = new FileReader();
        reader.onload = () => { encoded = String(reader.result).split(',')[1]; mount(status, `${f.name} (${Math.ceil(f.size / 1024)} KB) ready to upload`); };
        reader.readAsDataURL(f);
      });
      input = h('div.d-flex.flex-column.gap-1', file, status);
      get = () => encoded || undefined;
    } else {
      const isLong = LONG_FIELDS.test(key) && !isSecret && !/(user|name|host|port|region|url)/i.test(key);
      const attrs = {
        id, value: current ?? '', placeholder: p.default != null ? String(p.default) : '',
        autocomplete: isSecret ? 'new-password' : 'off', spellcheck: 'false',
      };
      if (isSecret) {
        const inp = h('input.form-control.form-control-sm', { ...attrs, type: 'password', value: current === SECRET_PLACEHOLDER ? '' : current ?? '', placeholder: current === SECRET_PLACEHOLDER ? '•••••••• (unchanged — type to replace)' : attrs.placeholder });
        const reveal = h('button.btn.btn-sm.btn-outline-secondary', {
          type: 'button', title: 'Show / hide',
          onclick: () => { inp.type = inp.type === 'password' ? 'text' : 'password'; mount(reveal, icon(inp.type === 'password' ? 'eye' : 'eye-slash')); },
        }, icon('eye'));
        input = h('div.input-group.input-group-sm', inp, reveal);
        get = () => (inp.value === '' ? (current === SECRET_PLACEHOLDER ? SECRET_PLACEHOLDER : undefined) : inp.value);
      } else if (isLong) {
        input = h('textarea.form-control.form-control-sm.af-mono', { ...attrs, rows: 4 }, current ?? '');
        get = () => (input.value === '' ? undefined : input.value);
      } else {
        input = h('input.form-control.form-control-sm', { ...attrs, type: 'text' });
        get = () => (input.value === '' ? undefined : input.value);
      }
    }
    getters[key] = () => {
      const v = get();
      if (required.has(key) && (v === undefined || v === '') && !(isSecret && current === SECRET_PLACEHOLDER)) {
        throw new Error(`${label} is required.`);
      }
      return v;
    };
    return h('div.af-form-row', { class: p.type === 'boolean' && 'af-form-row-inline rd-cfg-bool' },
      h('label.form-label.small.fw-semibold', { for: id }, label, required.has(key) && h('span.text-danger', ' *'),
        isSecret && h('span.rd-secret-pill', { title: 'Stored encrypted; never shown again after saving' }, icon('lock-fill'), ' secret')),
      input,
      p.description && h('div.form-text.small', p.description));
  };

  const main = order.filter((k) => !extra.has(k));
  const adv = order.filter((k) => extra.has(k));
  const advBody = h('div.af-form.rd-cfg-adv-body', adv.map(field));
  const el = h('div.af-form.rd-cfg',
    main.length ? main.map(field) : h('div.af-callout.af-callout-info', icon('info-circle'), h('span', 'This type needs no connection settings.')),
    adv.length && h('details.rd-cfg-adv', h('summary', icon('sliders', 'me-1'), `Advanced settings (${adv.length})`), advBody));

  return {
    el,
    values() {
      const out = {};
      for (const [k, get] of Object.entries(getters)) {
        const v = get();
        if (v !== undefined) out[k] = v;
      }
      return out;
    },
    isDirty() {
      try {
        return Object.entries(getters).some(([k, get]) => {
          const v = get();
          return (v ?? '') !== (initial[k] ?? '') && !(v === SECRET_PLACEHOLDER && initial[k] === SECRET_PLACEHOLDER);
        });
      } catch { return true; }
    },
  };
}

/**
 * Data sources attached to a group, with view_only.
 * GET /api/groups/<id>/data_sources 500s in Redash when a group has duplicate
 * permission rows for one data source ("Multiple rows were found for one()").
 * In that case rebuild the list from each data source's own `groups` map.
 * Resolves { sources, degraded } where degraded=true means the fallback was used.
 */
export async function groupDataSources(groupId, { signal } = {}) {
  try {
    return { sources: await rd.get(`/api/groups/${enc(groupId)}/data_sources`, null, { signal }), degraded: false };
  } catch (err) {
    if (err.name === 'AbortError' || err.status !== 500) throw err;
    const all = await rd.get('/api/data_sources', null, { signal });
    const full = await Promise.all(all.map((d) => rd.get(`/api/data_sources/${enc(d.id)}`, null, { signal })));
    const sources = full.filter((d) => d.groups && String(groupId) in d.groups)
      .map((d) => ({ ...d, view_only: !!d.groups[groupId] }));
    return { sources, degraded: true };
  }
}

/** Logo for a data source / destination type, served by Redash; falls back to an icon. */
export function typeLogo(type, kind = 'db', size = 28) {
  const src = redashLink(kind === 'db' ? `/static/images/db-logos/${type}.png` : `/static/images/destinations/${type}.png`);
  const fallback = icon(kind === 'db' ? dataSourceIcon(type) : 'send');
  if (!src) return h('span.rd-type-logo', { style: { width: `${size}px`, height: `${size}px` } }, fallback);
  const img = h('img', { src, alt: '', width: size, height: size, decoding: 'async' });
  const wrap = h('span.rd-type-logo', { style: { width: `${size}px`, height: `${size}px` } }, img);
  img.addEventListener('error', () => mount(wrap, fallback));
  return wrap;
}

/** Searchable grid of types. Resolves with the chosen type object (or null). */
export function pickType(types, { title = 'Choose a type', kind = 'db', popular = [] } = {}) {
  return new Promise((resolve) => {
    let chosen = null;
    const grid = h('div.rd-type-grid');
    const search = h('input.form-control', { type: 'search', placeholder: `Search ${types.length} types…`, autofocus: true });
    const draw = (q = '') => {
      const ql = q.toLowerCase();
      const list = types.filter((t) => !ql || t.name.toLowerCase().includes(ql) || t.type.toLowerCase().includes(ql))
        .sort((a, b) => (popular.indexOf(b.type) > -1) - (popular.indexOf(a.type) > -1) || a.name.localeCompare(b.name));
      mount(grid, list.length ? list.map((t) => h('button.rd-type-card', {
        type: 'button', title: t.name,
        onclick: () => { chosen = t; m.close(); },
      }, typeLogo(t.type, kind, 32), h('span.rd-type-name', t.name), popular.includes(t.type) && h('span.rd-type-pop', 'popular')))
        : emptyState({ icon: 'search', title: 'No matching types' }));
    };
    search.addEventListener('input', debounce(() => draw(search.value.trim()), 80));
    search.addEventListener('keydown', (e) => { if (e.key === 'Enter') grid.querySelector('.rd-type-card')?.click(); });
    const m = modal({
      title, icon: 'plus-circle', size: 'lg',
      body: h('div.d-flex.flex-column.gap-3', search, grid),
      onClose: () => resolve(chosen),
    });
    draw();
  });
}

/** Case-insensitive match of q against any of the given fields of an object. */
export function matches(obj, q, fields) {
  if (!q) return true;
  const ql = q.toLowerCase();
  return fields.some((f) => {
    const v = typeof f === 'function' ? f(obj) : obj[f];
    return v != null && String(v).toLowerCase().includes(ql);
  });
}

/** Highlight the first occurrence of q inside text. */
export function hl(text, q) {
  const s = String(text ?? '');
  if (!q) return s;
  const i = s.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return s;
  return [s.slice(0, i), h('mark.rd-hl', s.slice(i, i + q.length)), s.slice(i + q.length)];
}

/** Small status pill: tone = ok | bad | warn | muted | info */
export function pill(text, tone = 'muted', ic) {
  return h(`span.rd-pill.rd-pill-${tone}`, ic && icon(ic), text);
}

export const ALERT_STATES = {
  ok: { label: 'OK', tone: 'ok', icon: 'check-circle-fill' },
  triggered: { label: 'Triggered', tone: 'bad', icon: 'exclamation-triangle-fill' },
  unknown: { label: 'Unknown', tone: 'muted', icon: 'question-circle' },
};

export function alertStateBadge(state, { muted } = {}) {
  const s = ALERT_STATES[state] || ALERT_STATES.unknown;
  return h('span.d-inline-flex.gap-1.align-items-center', pill(s.label, s.tone, s.icon), muted && pill('Muted', 'warn', 'bell-slash-fill'));
}

export const OPERATORS = [
  { value: '>', label: 'greater than', symbol: '>' },
  { value: '>=', label: 'greater than or equal to', symbol: '≥' },
  { value: '<', label: 'less than', symbol: '<' },
  { value: '<=', label: 'less than or equal to', symbol: '≤' },
  { value: '==', label: 'equals', symbol: '=' },
  { value: '!=', label: 'does not equal', symbol: '≠' },
];
const LEGACY_OPS = { 'greater than': '>', 'less than': '<', equals: '==' };
export const normalizeOp = (op) => LEGACY_OPS[op] || op;
export const opInfo = (op) => OPERATORS.find((o) => o.value === normalizeOp(op)) || { value: op, label: op, symbol: op };

/**
 * Client-side copy of Redash's alert evaluation (models.next_state): numbers compare
 * numerically, booleans as lowercase strings, everything else as strings.
 */
export function evaluateAlert(op, value, threshold) {
  if (value === undefined) return 'unknown';
  const fn = {
    '>': (a, b) => a > b, '>=': (a, b) => a >= b, '<': (a, b) => a < b,
    '<=': (a, b) => a <= b, '==': (a, b) => a === b, '!=': (a, b) => a !== b,
  }[normalizeOp(op)];
  if (!fn) return 'unknown';
  let v = value;
  let t = threshold;
  if (typeof v === 'boolean') v = String(v);
  else if (v !== null && v !== '' && !Number.isNaN(Number(v))) {
    v = Number(v);
    if (t === '' || t == null || Number.isNaN(Number(t))) return 'unknown';
    t = Number(t);
  } else {
    v = String(v);
    t = String(t ?? '');
  }
  return fn(v, t) ? 'triggered' : 'ok';
}

/** Rearm (seconds) <-> {mode, amount, unit}. */
export const REARM_UNITS = [{ value: 60, label: 'minutes' }, { value: 3600, label: 'hours' }, { value: 86400, label: 'days' }];
export function rearmParts(rearm) {
  if (!rearm) return { mode: 'once', amount: 1, unit: 3600 };
  if (rearm === 1) return { mode: 'each', amount: 1, unit: 60 };
  const unit = [...REARM_UNITS].reverse().find((u) => rearm % u.value === 0)?.value || 60;
  return { mode: 'every', amount: Math.max(1, Math.round(rearm / unit)), unit };
}
export function rearmText(rearm) {
  const p = rearmParts(rearm);
  if (p.mode === 'once') return 'Just once, until back to normal';
  if (p.mode === 'each') return 'Each time the alert is evaluated';
  const u = REARM_UNITS.find((x) => x.value === p.unit);
  return `At most every ${p.amount} ${p.amount === 1 ? u.label.replace(/s$/, '') : u.label}`;
}

export const TEMPLATE_VARS = [
  ['ALERT_NAME', 'Alert name'], ['ALERT_URL', 'Link to the alert'], ['ALERT_STATUS', 'TRIGGERED / OK / UNKNOWN'],
  ['ALERT_CONDITION', 'Operator, e.g. >'], ['ALERT_THRESHOLD', 'Threshold value'], ['QUERY_NAME', 'Query name'],
  ['QUERY_URL', 'Link to the query'], ['QUERY_RESULT_VALUE', 'Value checked by the alert'],
  ['QUERY_RESULT_ROWS', 'All result rows'], ['QUERY_RESULT_COLS', 'All result columns'],
];

/** Minimal mustache: {{VAR}} / {{{VAR}}} plus {{#ROWS}}…{{/ROWS}} sections over arrays. */
export function renderMustache(template, ctx) {
  const lookup = (scope, key) => (key === '.' ? scope : key.split('.').reduce((o, k) => (o == null ? undefined : o[k]), scope));
  const render = (tpl, scope) => tpl
    .replace(/\{\{#\s*([\w.]+)\s*\}\}([\s\S]*?)\{\{\/\s*\1\s*\}\}/g, (_, key, inner) => {
      const v = lookup(scope, key) ?? lookup(ctx, key);
      if (Array.isArray(v)) return v.map((item) => render(inner, typeof item === 'object' ? { ...ctx, ...item } : { ...ctx, '.': item })).join('');
      return v ? render(inner, scope) : '';
    })
    .replace(/\{\{\{?\s*([\w.]+)\s*\}?\}\}/g, (_, key) => {
      const v = lookup(scope, key) ?? lookup(ctx, key);
      if (v == null) return '';
      return typeof v === 'object' ? JSON.stringify(v) : String(v);
    });
  return render(String(template || ''), ctx);
}
