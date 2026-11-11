// Helpers shared by the admin pages (connections, variables, pools, xcoms,
// assets, backfills, required actions). Not a route.

import { h, icon, mount } from '../core/dom.js';
import { modal, toast, dataTable, buildForm } from '../core/ui.js';
import { truncate } from '../core/format.js';

/** Trigger a browser download of `data` as pretty JSON. */
export function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = h('a', { href: URL.createObjectURL(blob), download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** Let the user pick a .json file; resolves to {name, data} (or null when cancelled). */
export function pickJsonFile() {
  return new Promise((resolve, reject) => {
    const input = h('input', { type: 'file', accept: '.json,application/json', style: 'display:none' });
    input.addEventListener('change', async () => {
      const file = input.files?.[0];
      input.remove();
      if (!file) { resolve(null); return; }
      try {
        resolve({ name: file.name, data: JSON.parse(await file.text()) });
      } catch (err) {
        reject(new Error(`${file.name} is not valid JSON: ${err.message}`));
      }
    });
    document.body.appendChild(input);
    input.click();
  });
}

/** Summarise a BulkResponse ({create|update|delete: {success, errors}}) as a toast. */
export function bulkToast(res, verb) {
  const parts = ['create', 'update', 'delete'].map((k) => res?.[k]).filter(Boolean);
  const ok = parts.reduce((n, p) => n + (p.success || []).length, 0);
  const errors = parts.flatMap((p) => p.errors || []);
  if (errors.length) {
    const first = errors[0];
    const msg = typeof first === 'object' ? (first.error || first.detail || JSON.stringify(first)) : String(first);
    toast(`${ok} ${verb}, ${errors.length} failed — ${truncate(msg, 160)}`, ok ? 'warning' : 'error', { title: `Bulk ${verb}` });
  } else {
    toast(`${ok} item(s) ${verb}`, 'success');
  }
  return { ok, errors };
}

/** Stable pastel colour for a string (connection types, groups). */
export function hashColor(text) {
  let hash = 0;
  for (const ch of String(text)) hash = (hash * 31 + ch.charCodeAt(0)) | 0;
  const hue = Math.abs(hash) % 360;
  return { color: `hsl(${hue} 55% 32%)`, background: `hsl(${hue} 70% 94%)`, border: `hsl(${hue} 50% 85%)` };
}

export function typeBadge(text) {
  if (!text) return h('span.text-muted', '—');
  const c = hashColor(text);
  return h('span.af-type-badge', { style: { color: c.color, background: c.background, borderColor: c.border } }, text);
}

/** Try to parse a string as a JSON object/array (not bare scalars). */
export function parseJsonish(value) {
  if (value && typeof value === 'object') return { ok: true, value };
  if (typeof value !== 'string') return { ok: false };
  const t = value.trim();
  if (!(t.startsWith('{') || t.startsWith('['))) return { ok: false };
  try { return { ok: true, value: JSON.parse(t) }; } catch { return { ok: false }; }
}

export const isMasked = (v) => typeof v === 'string' && /^\*{3,}$/.test(v.trim());

/** True when a value still holds Airflow's "***" redaction anywhere inside it. */
export const containsMask = (v) => typeof v === 'string' && /(^|["'\s:=])\*{3}(["'\s,}\]]|$)/.test(v);

/** Equal as JSON when both parse, otherwise as trimmed text. */
export function sameValue(a, b) {
  const pa = parseJsonish(a);
  const pb = parseJsonish(b);
  if (pa.ok && pb.ok) return JSON.stringify(pa.value) === JSON.stringify(pb.value);
  return String(a ?? '').trim() === String(b ?? '').trim();
}

export const MASK_ERROR = 'This value still contains masked placeholders (***). Airflow would store them literally and destroy the real secret — replace every *** with the real value, or leave the value unchanged.';

/** Compact one-line preview of any value. */
export function valuePreview(value, max = 90) {
  if (value == null) return h('span.text-muted', 'null');
  if (isMasked(value)) return h('span.af-masked', icon('eye-slash'), ' masked');
  const parsed = parseJsonish(value);
  if (parsed.ok) {
    const v = parsed.value;
    const label = Array.isArray(v) ? `array[${v.length}]` : `object{${Object.keys(v).length}}`;
    return h('span.af-val-preview', h('span.af-json-pill', label), h('code.af-inline-json', truncate(JSON.stringify(v), max)));
  }
  return h('code.af-inline-json', truncate(String(value), max));
}

/**
 * Import confirmation dialog: preview the parsed entities and choose
 * action_on_existence. onConfirm(entities, onExistence) performs the request.
 */
export function importDialog({ title, entities, columns, rowKey, onConfirm }) {
  const opts = buildForm([{
    name: 'action_on_existence', label: 'If an item already exists', type: 'select',
    options: [
      { value: 'fail', label: 'Fail the whole import' },
      { value: 'skip', label: 'Skip existing items' },
      { value: 'overwrite', label: 'Overwrite existing items' },
    ],
    value: 'skip',
  }]);
  modal({
    title, icon: 'upload', size: 'lg',
    subtitle: `${entities.length} item(s) found in file`,
    body: h('div', opts.el,
      h('div.af-preview', dataTable({ dense: true, rows: entities.slice(0, 200), rowKey, columns })),
      entities.length > 200 && h('div.small.text-muted.mt-1', `…and ${entities.length - 200} more`)),
    actions: [
      { label: 'Cancel' },
      {
        label: `Import ${entities.length}`, variant: 'primary', icon: 'upload',
        onClick: async (close) => {
          await onConfirm(entities, opts.values().action_on_existence);
          close();
        },
      },
    ],
  });
}

/** Find the <input>/<select>/<textarea> rendered for a buildForm field by label text. */
export function fieldControl(formEl, labelText) {
  const label = [...formEl.querySelectorAll('label.form-label')].find((l) => l.textContent.replace('*', '').trim() === labelText);
  return label ? formEl.querySelector(`#${CSS.escape(label.htmlFor)}`) : null;
}

/** Always-one-selected toggle group (core chips() allows deselecting). */
export function segmented(options, value, onChange) {
  let current = value;
  const bar = h('div.af-seg', { role: 'radiogroup' });
  const draw = () => mount(bar, options.map((o) => h('button.af-seg-btn', {
    type: 'button', role: 'radio', 'aria-checked': o.value === current, class: o.value === current && 'active',
    onclick: () => { if (o.value === current) return; current = o.value; draw(); onChange(o.value); },
  }, o.icon && icon(o.icon, 'me-1'), o.label)));
  draw();
  return bar;
}

export function linkList(items, empty = 'None') {
  if (!items?.length) return h('span.text-muted', empty);
  return h('div.af-link-list', items);
}

export { mount };
