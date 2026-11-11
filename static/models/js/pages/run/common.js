// Helpers shared by the run page tabs: file index, inline editing, tag editor, YAML.

import { h, icon, mount } from '../../../../ui/js/dom.js';
import { toastError } from '../../../../ui/js/ui.js';
import { gql, collect } from '../../core/api.js';

/** Every file of a run (paged), cached per run. Returns Map(name -> {name, sizeBytes, mimetype, directUrl, url, updatedAt}). */
const fileIndexCache = new Map();
export function fileIndex(entity, project, runName, { refresh = false } = {}) {
  const key = `${entity}/${project}/${runName}`;
  if (!refresh && fileIndexCache.has(key)) return fileIndexCache.get(key);
  const p = collect((after) => gql(`query RunFiles($e: String!, $p: String!, $r: String!, $after: String) {
    project(entityName: $e, name: $p) { run(name: $r) { files(first: 500, after: $after) {
      pageInfo { hasNextPage endCursor }
      edges { node { id name sizeBytes mimetype directUrl url updatedAt } } } } } }`,
  { e: entity, p: project, r: runName, after }), (d) => d.project?.run?.files, { max: 20000 })
    .then((nodes) => new Map(nodes.map((n) => [n.name, n])))
    .catch((err) => { fileIndexCache.delete(key); throw err; });
  fileIndexCache.set(key, p);
  return p;
}

/** Click-to-edit text; Enter saves, Escape cancels; onSave returns a promise (reverts on failure). */
export function inlineText(value, onSave, { placeholder = 'Add…', multiline = false, className = '', title = 'Click to edit' } = {}) {
  const wrap = h('span.msr-inline', { class: className });
  let current = value || '';
  const view = () => mount(wrap,
    h('span.msr-inline-text', { class: !current && 'empty', tabindex: 0, title }, current || placeholder),
    icon('pencil', 'msr-inline-pencil'));
  const edit = () => {
    const input = multiline
      ? h('textarea.form-control.form-control-sm.msr-inline-input', { rows: 3 }, current)
      : h('input.form-control.form-control-sm.msr-inline-input', { value: current, placeholder });
    mount(wrap, input);
    input.focus();
    input.select?.();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const next = input.value.trim();
      if (!save || next === current) { view(); return; }
      const prev = current;
      current = next;
      view();
      try { await onSave(next); } catch (err) { current = prev; view(); toastError(err, 'Could not save'); }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (!multiline || e.ctrlKey || e.metaKey)) { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
  };
  wrap.addEventListener('click', (e) => { if (!e.target.closest('input,textarea')) edit(); });
  wrap.addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target.classList.contains('msr-inline-text')) edit(); });
  view();
  return wrap;
}

/** Tag chips with add/remove. onSave(tags) persists; reverts on failure. */
export function tagEditor(tags, onSave, { suggestions = [] } = {}) {
  let current = [...(tags || [])];
  const el = h('div.msr-tags');
  const persist = async (next) => {
    const prev = current;
    current = next;
    render();
    try { await onSave(next); } catch (err) { current = prev; render(); toastError(err, 'Could not save tags'); }
  };
  const render = () => mount(el,
    current.map((t) => h('span.ms-tag.msr-tag', icon('tag'), t,
      h('button.msr-tag-x', { type: 'button', title: `Remove ${t}`, 'aria-label': `Remove tag ${t}`, onclick: () => persist(current.filter((x) => x !== t)) }, icon('x')))),
    h('button.msr-tag-add', {
      type: 'button',
      onclick: (e) => {
        const btn = e.currentTarget;
        const listId = `msr-tags-${Math.random().toString(36).slice(2, 7)}`;
        const input = h('input.form-control.form-control-sm.msr-tag-input', { placeholder: 'tag', list: listId, 'aria-label': 'New tag' });
        const dl = h('datalist', { id: listId }, suggestions.filter((s) => !current.includes(s)).map((s) => h('option', { value: s })));
        btn.replaceWith(input, dl);
        input.focus();
        let done = false;
        const finish = (save) => {
          if (done) return;
          done = true;
          const v = input.value.trim();
          dl.remove();
          if (save && v && !current.includes(v)) persist([...current, v]); else render();
        };
        input.addEventListener('keydown', (ev) => {
          if (ev.key === 'Enter') { ev.preventDefault(); finish(true); }
          if (ev.key === 'Escape') { ev.preventDefault(); ev.stopPropagation(); finish(false); }
        });
        input.addEventListener('blur', () => finish(true));
      },
    }, icon('plus'), current.length ? '' : ' Add tag'));
  render();
  return el;
}

/** Minimal YAML emitter for config export (scalars, arrays, nested objects). */
export function toYaml(obj, indent = 0) {
  const pad = '  '.repeat(indent);
  const scalar = (v) => {
    if (v == null) return 'null';
    if (typeof v === 'number' || typeof v === 'boolean') return String(v);
    const s = String(v);
    return /^[\w.\-/]+$/.test(s) && !/^(true|false|null|yes|no|on|off|\d.*)$/i.test(s) ? s : JSON.stringify(s);
  };
  if (Array.isArray(obj)) {
    if (!obj.length) return `${pad}[]`;
    return obj.map((v) => (v && typeof v === 'object' ? `${pad}-\n${toYaml(v, indent + 1)}` : `${pad}- ${scalar(v)}`)).join('\n');
  }
  return Object.entries(obj || {}).map(([k, v]) => {
    const key = /^[\w.\-/]+$/.test(k) ? k : JSON.stringify(k);
    if (v && typeof v === 'object' && (Array.isArray(v) ? v.length : Object.keys(v).length)) return `${pad}${key}:\n${toYaml(v, indent + 1)}`;
    if (v && typeof v === 'object') return `${pad}${key}: ${Array.isArray(v) ? '[]' : '{}'}`;
    return `${pad}${key}: ${scalar(v)}`;
  }).join('\n');
}

/** Un-flatten {"a.b": 1} into {a: {b: 1}} for YAML/JSON export. */
export function unflatten(flat) {
  const out = {};
  for (const [k, v] of Object.entries(flat || {})) {
    const parts = k.split('.');
    let cur = out;
    parts.forEach((p, i) => {
      if (i === parts.length - 1) cur[p] = v;
      else cur = cur[p] = cur[p] && typeof cur[p] === 'object' ? cur[p] : {};
    });
  }
  return out;
}

export function fmtBytesShort(n) {
  if (n == null) return '—';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = Number(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i ? 1 : 0)} ${u[i]}`;
}

/** Highlight occurrences of q (case-insensitive) in text. Returns nodes. */
export function highlight(text, q) {
  const s = String(text ?? '');
  if (!q) return s;
  const lower = s.toLowerCase();
  const out = [];
  let i = 0;
  for (;;) {
    const j = lower.indexOf(q, i);
    if (j < 0) { out.push(s.slice(i)); break; }
    if (j > i) out.push(s.slice(i, j));
    out.push(h('mark', s.slice(j, j + q.length)));
    i = j + q.length;
  }
  return out;
}

/** Mask class colours (index = class id). Teal first: stands out on typical red/skin/greyscale imagery. */
export const CLASS_COLORS = [[0, 0, 0], [11, 185, 159], [124, 58, 237], [0, 114, 206], [245, 166, 35], [214, 64, 159], [70, 167, 88], [229, 72, 77], [62, 99, 221], [255, 178, 36]];
/** Box colours (index = class id). Bright yellow first for contrast against masks and red objects. */
export const BOX_COLORS = [[250, 204, 21], [250, 204, 21], [34, 211, 238], [244, 114, 182], [163, 230, 53], [251, 146, 60], [167, 139, 250], [248, 113, 113], [96, 165, 250], [52, 211, 153]];
