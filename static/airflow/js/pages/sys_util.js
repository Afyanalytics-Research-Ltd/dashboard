// Helpers shared by the system pages: a URL-synced filter bar for limit/offset list endpoints.

import { h, icon, debounce } from '../core/dom.js';
import { toLocalInput, fromLocalInput } from '../core/format.js';

/**
 * Read filter state from the URL. defs: [{name, type}] — 'list' types use repeated keys.
 */
export function readState(query, defs, extra = {}) {
  const state = {};
  for (const d of defs) {
    if (d.type === 'list') state[d.name] = query.getAll(d.name);
    else state[d.name] = query.get(d.name) || '';
  }
  for (const [k, v] of Object.entries(extra)) state[k] = query.get(k) ?? v;
  return state;
}

/** Convert filter state to API params (drops empties, numbers stay strings — the proxy forwards them). */
export function toParams(state, defs) {
  const params = {};
  for (const d of defs) {
    const v = state[d.name];
    if (v == null || v === '' || (Array.isArray(v) && !v.length)) continue;
    params[d.api || d.name] = v;
  }
  return params;
}

/**
 * Filter bar. defs: [{name, label, type: text|number|select|datetime|list, options, placeholder, advanced}]
 * onChange(state) fires (debounced for typing) with the full state.
 */
export function filterBar(defs, state, onChange) {
  const fire = () => onChange({ ...state });
  const fireSoon = debounce(fire, 350);
  const field = (d) => {
    const id = `sysf-${d.name}`;
    let input;
    if (d.type === 'select') {
      input = h('select.form-select.form-select-sm', {
        id,
        onchange: (e) => { state[d.name] = e.target.value; fire(); },
      }, h('option', { value: '' }, d.any || 'Any'), (d.options || []).map((o) => {
        const opt = typeof o === 'object' ? o : { value: o, label: o };
        return h('option', { value: opt.value, selected: String(state[d.name]) === String(opt.value) }, opt.label);
      }));
    } else if (d.type === 'datetime') {
      input = h('input.form-control.form-control-sm', {
        id, type: 'datetime-local', value: toLocalInput(state[d.name]),
        onchange: (e) => { state[d.name] = fromLocalInput(e.target.value) || ''; fire(); },
      });
    } else if (d.type === 'list') {
      input = h('input.form-control.form-control-sm', {
        id, value: (state[d.name] || []).join(', '), placeholder: d.placeholder || 'comma, separated',
        oninput: (e) => { state[d.name] = e.target.value.split(',').map((x) => x.trim()).filter(Boolean); fireSoon(); },
      });
    } else {
      input = h('input.form-control.form-control-sm', {
        id, type: d.type === 'number' ? 'number' : 'search', value: state[d.name] || '', placeholder: d.placeholder || '',
        oninput: (e) => { state[d.name] = e.target.value.trim(); fireSoon(); },
      });
    }
    return h('div.sys-filter', { class: d.wide && 'sys-filter-wide' }, h('label.sys-filter-label', { for: id }, d.label), input);
  };
  const basic = defs.filter((d) => !d.advanced);
  const adv = defs.filter((d) => d.advanced);
  const advActive = adv.some((d) => (Array.isArray(state[d.name]) ? state[d.name].length : state[d.name]));
  const advBox = h('div.sys-filters.sys-filters-adv', { class: !advActive && 'd-none' }, adv.map(field));
  const resetBtn = h('button.btn.btn-sm.btn-link.text-decoration-none', {
    type: 'button',
    onclick: () => {
      defs.forEach((d) => { state[d.name] = d.type === 'list' ? [] : ''; });
      wrap.replaceWith(filterBar(defs, state, onChange));
      fire();
    },
  }, icon('x-circle', 'me-1'), 'Reset');
  const wrap = h('div.sys-filterbar',
    h('div.sys-filters', basic.map(field),
      h('div.sys-filter-actions',
        adv.length ? h('button.btn.btn-sm.btn-outline-secondary', {
          type: 'button', onclick: () => advBox.classList.toggle('d-none'),
        }, icon('funnel', 'me-1'), `More filters (${adv.length})`) : null,
        resetBtn)),
    adv.length ? advBox : null);
  return wrap;
}

/** Show a JSON-ish string as parsed JSON when possible. */
export function parseMaybeJson(value) {
  if (value == null || typeof value !== 'string') return value;
  const t = value.trim();
  if (!(t.startsWith('{') || t.startsWith('['))) return value;
  try { return JSON.parse(t); } catch { return value; }
}
