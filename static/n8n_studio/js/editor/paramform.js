// Schema-driven parameter form for n8n node properties.
//
//   const form = paramForm({ props, values, typeVersion, onChange(values), preview(expr) })
//   form.el
//
// Supports string/number/boolean/options/multiOptions/collection/fixedCollection/json/
// color/dateTime/resourceLocator/assignmentCollection/filter/notice, an expression
// toggle on every field with sandboxed live preview, and drag-to-map drops.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { copyButton } from '../../../ui/js/ui.js';
import { visibleProps, defaults } from '../core/params.js';
import { clone, uuid } from '../core/workflow.js';
import { isExpression, previewText } from '../core/sandbox.js';

const FIELD_TYPES_WITH_RERENDER = new Set(['options', 'boolean', 'multiOptions']);

const getAt = (obj, path) => path.reduce((o, k) => (o == null ? undefined : o[k]), obj);
function setAt(obj, path, value) {
  let o = obj;
  for (let i = 0; i < path.length - 1; i++) {
    const k = path[i];
    if (o[k] == null || typeof o[k] !== 'object') o[k] = typeof path[i + 1] === 'number' ? [] : {};
    o = o[k];
  }
  if (value === undefined) {
    if (Array.isArray(o)) o.splice(path[path.length - 1], 1);
    else delete o[path[path.length - 1]];
  } else o[path[path.length - 1]] = value;
}

const label = (p) => p.displayName || p.name;

export function paramForm({ props, values, typeVersion, onChange, preview, readOnly = false, credentialsSlot }) {
  const state = { values: clone(values || {}) };
  const root = h('div.n8s-form');
  const emit = () => onChange?.(clone(state.values));
  const emitDebounced = debounce(emit, 250);

  // ------------------------------------------------------------------ field shell

  function fieldShell(p, path, editor, { canExpr = true, value } = {}) {
    const isExpr = isExpression(value);
    const toggle = canExpr && !readOnly && h('div.n8s-fx', { role: 'group' },
      h('button', { type: 'button', class: !isExpr && 'on', title: 'Fixed value', onclick: () => { if (isExpr) { setAt(state.values, path, stripExpr(value, p)); rerender(); emit(); } } }, 'Fixed'),
      h('button', { type: 'button', class: isExpr && 'on', title: 'Expression', onclick: () => { if (!isExpr) { setAt(state.values, path, toExpr(value)); rerender(path); emit(); } } }, 'Expression'));
    const head = h('div.n8s-field-head',
      h('label.n8s-field-label', label(p), p.required && h('span.text-danger', ' *')),
      p.description && h('span.n8s-field-hint', { title: stripHtml(p.description) }, icon('question-circle')),
      h('div.flex-grow-1'),
      value !== undefined && p.default !== undefined && JSON.stringify(value) !== JSON.stringify(p.default) && !readOnly
        && h('button.n8s-field-reset', { type: 'button', title: 'Reset to default', onclick: () => { setAt(state.values, path, undefined); rerender(); emit(); } }, icon('arrow-counterclockwise')),
      toggle);
    const wrap = h('div.n8s-field', { class: isExpr && 'expr', dataset: { path: path.join('.') } }, head, isExpr ? exprEditor(path, value) : editor);
    if (p.hint && !isExpr) wrap.appendChild(h('div.n8s-field-note', stripHtml(p.hint)));
    // Drag-to-map: dropping {{ … }} turns the field into an expression.
    if (canExpr && !readOnly) {
      wrap.addEventListener('dragover', (e) => { if (e.dataTransfer.types.includes('text/plain')) { e.preventDefault(); wrap.classList.add('drop'); } });
      wrap.addEventListener('dragleave', () => wrap.classList.remove('drop'));
      wrap.addEventListener('drop', (e) => {
        const text = e.dataTransfer.getData('text/plain');
        wrap.classList.remove('drop');
        if (!/\{\{.*\}\}/s.test(text)) return;
        e.preventDefault();
        const cur = getAt(state.values, path);
        const next = isExpression(cur) ? `${cur}${cur.endsWith(' ') ? '' : ' '}${text}` : `=${text}`;
        setAt(state.values, path, next);
        rerender(path);
        emit();
      });
    }
    return wrap;
  }

  function exprEditor(path, value) {
    const ta = h('textarea.form-control.form-control-sm.n8s-expr-input', { rows: Math.min(8, Math.max(2, String(value).split('\n').length)), spellcheck: 'false', readonly: readOnly }, String(value).slice(1));
    const out = h('div.n8s-expr-preview', h('span.text-muted', preview ? 'Result preview…' : 'Run the workflow once to preview expressions against real data.'));
    const run = debounce(async () => {
      if (!preview) return;
      const r = await preview(`=${ta.value}`);
      mount(out, h('div', { class: r.ok ? 'ok' : 'err' }, h('span.n8s-expr-type', r.ok ? r.type : 'error'), h('pre', previewText(r).slice(0, 2000))));
    }, 300);
    ta.addEventListener('input', () => { setAt(state.values, path, `=${ta.value}`); emitDebounced(); run(); });
    run();
    return h('div.n8s-expr', ta, out, h('div.n8s-expr-help', 'Use ', h('code', '{{ $json.field }}'), ', ', h('code', "{{ $('Node').item.json.x }}"), ', ', h('code', '{{ $now }}'), ' — or drag a field from the input panel.'));
  }

  const toExpr = (v) => (v == null || v === '' ? '=' : typeof v === 'object' ? `={{ ${JSON.stringify(v)} }}` : `=${v}`);
  const stripExpr = (v, p) => {
    const raw = String(v).slice(1);
    if (/^\{\{[\s\S]*\}\}$/.test(raw.trim())) return p.default;
    if (p.type === 'number') return Number(raw) || p.default;
    if (p.type === 'boolean') return raw === 'true';
    return raw;
  };

  // ------------------------------------------------------------------ editors

  function editorFor(p, path, value, local) {
    const t = p.type;
    const v = value === undefined ? p.default : value;
    const opts = p.typeOptions || {};
    if (isExpression(v) && !['collection', 'fixedCollection', 'notice', 'assignmentCollection', 'filter'].includes(t)) {
      return fieldShell(p, path, null, { value: v });
    }
    switch (t) {
      case 'notice':
      case 'callout':
        return h('div.n8s-notice', icon('info-circle', 'me-1'), stripHtml(p.displayName));
      case 'hidden':
      case 'curlImport':
      case 'button':
        return null;
      case 'boolean': {
        const input = h('input.form-check-input', { type: 'checkbox', role: 'switch', checked: !!v, disabled: readOnly });
        input.addEventListener('change', () => { setAt(state.values, path, input.checked); rerender(); emit(); });
        return fieldShell(p, path, h('div.form-check.form-switch', input), { value });
      }
      case 'number': {
        const input = h('input.form-control.form-control-sm', { type: 'number', value: v ?? '', min: opts.minValue, max: opts.maxValue, step: opts.numberPrecision ? 10 ** -opts.numberPrecision : 'any', readonly: readOnly });
        input.addEventListener('input', () => { setAt(state.values, path, input.value === '' ? undefined : Number(input.value)); emitDebounced(); });
        return fieldShell(p, path, input, { value });
      }
      case 'options': {
        const options = p.options || [];
        if (!options.length || opts.loadOptionsMethod || opts.loadOptions) {
          const input = h('input.form-control.form-control-sm', { value: v ?? '', placeholder: 'Value (options load at runtime in n8n)', readonly: readOnly });
          input.addEventListener('input', () => { setAt(state.values, path, input.value); emitDebounced(); });
          return fieldShell(p, path, h('div', input, h('div.n8s-field-note', icon('cloud', 'me-1'), 'Dynamic list — n8n loads these options from the service.')), { value });
        }
        const sel = h('select.form-select.form-select-sm', { disabled: readOnly },
          options.map((o) => h('option', { value: JSON.stringify(o.value), selected: o.value === v, title: o.description ? stripHtml(o.description) : undefined }, o.name)));
        sel.addEventListener('change', () => { setAt(state.values, path, JSON.parse(sel.value)); rerender(); emit(); });
        const desc = options.find((o) => o.value === v)?.description;
        return fieldShell(p, path, h('div', sel, desc && h('div.n8s-field-note', stripHtml(desc))), { value });
      }
      case 'multiOptions': {
        const chosen = new Set(Array.isArray(v) ? v : []);
        const box = h('div.n8s-multi', (p.options || []).map((o) => {
          const cb = h('input.form-check-input', { type: 'checkbox', checked: chosen.has(o.value), disabled: readOnly });
          cb.addEventListener('change', () => { cb.checked ? chosen.add(o.value) : chosen.delete(o.value); setAt(state.values, path, [...chosen]); rerender(); emit(); });
          return h('label.n8s-multi-opt', cb, h('span', o.name));
        }));
        if (!(p.options || []).length) return fieldShell(p, path, h('div.n8s-field-note', 'Options load at runtime in n8n — switch to Expression to set values.'), { value });
        return fieldShell(p, path, box, { value });
      }
      case 'color': {
        const input = h('input.form-control.form-control-sm.form-control-color', { type: 'color', value: v || '#ff6d5a', disabled: readOnly });
        input.addEventListener('input', () => { setAt(state.values, path, input.value); emitDebounced(); });
        return fieldShell(p, path, input, { value });
      }
      case 'dateTime': {
        const input = h('input.form-control.form-control-sm', { type: 'datetime-local', value: v ? String(v).slice(0, 16) : '', readonly: readOnly });
        input.addEventListener('change', () => { setAt(state.values, path, input.value ? new Date(input.value).toISOString() : ''); emit(); });
        return fieldShell(p, path, input, { value });
      }
      case 'json':
        return fieldShell(p, path, codeArea(path, typeof v === 'string' ? v : JSON.stringify(v ?? {}, null, 2), 'json', 6), { value });
      case 'resourceLocator':
        return fieldShell(p, path, resourceLocator(p, path, v), { value, canExpr: false });
      case 'collection':
        return collection(p, path, v || {}, local);
      case 'fixedCollection':
        return fixedCollection(p, path, v || {}, local);
      case 'assignmentCollection':
        return assignments(p, path, v || { assignments: [] });
      case 'filter':
        return filter(p, path, v || { conditions: [], combinator: 'and' });
      case 'workflowSelector':
        return fieldShell(p, path, plainInput(path, typeof v === 'object' ? v?.value : v, 'Workflow ID'), { value });
      case 'credentialsSelect':
        return fieldShell(p, path, plainInput(path, v, 'Credential type'), { value });
      case 'string':
      default: {
        if (t !== 'string' && typeof v === 'object' && v !== null) {
          return fieldShell(p, path, codeArea(path, JSON.stringify(v, null, 2), 'json', 6, true), { value });
        }
        if (opts.editor || opts.rows > 4 || /code|query|html|sql/i.test(p.name) && opts.rows) {
          const lang = opts.editorLanguage || (/sql/i.test(opts.editor || p.name) ? 'sql' : /html/i.test(opts.editor || '') ? 'html' : /python/i.test(opts.editorLanguage || p.name) ? 'python' : 'javascript');
          return fieldShell(p, path, codeArea(path, v ?? '', lang, opts.rows || 12), { value, canExpr: !opts.editor });
        }
        if (opts.rows) {
          const ta = h('textarea.form-control.form-control-sm', { rows: opts.rows, placeholder: p.placeholder, readonly: readOnly }, v ?? '');
          ta.addEventListener('input', () => { setAt(state.values, path, ta.value); emitDebounced(); });
          return fieldShell(p, path, ta, { value });
        }
        const input = h('input.form-control.form-control-sm', { type: opts.password ? 'password' : 'text', value: v ?? '', placeholder: p.placeholder, readonly: readOnly, autocomplete: 'off' });
        input.addEventListener('input', () => { setAt(state.values, path, input.value); emitDebounced(); });
        return fieldShell(p, path, input, { value });
      }
    }
  }

  function plainInput(path, v, placeholder) {
    const input = h('input.form-control.form-control-sm', { value: v ?? '', placeholder, readonly: readOnly });
    input.addEventListener('input', () => { setAt(state.values, path, input.value); emitDebounced(); });
    return input;
  }

  function codeArea(path, text, lang, rows, parseJson = false) {
    const ta = h('textarea.form-control.form-control-sm.n8s-code', { rows, spellcheck: 'false', readonly: readOnly, dataset: { lang } }, text);
    const status = h('span.n8s-code-status');
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') { e.preventDefault(); ta.setRangeText('  ', ta.selectionStart, ta.selectionEnd, 'end'); }
    });
    ta.addEventListener('input', () => {
      let v = ta.value;
      if (parseJson) { try { v = JSON.parse(ta.value); mount(status, ''); } catch (err) { mount(status, h('span.text-danger', err.message)); return; } }
      if (lang === 'json' && !parseJson) { try { JSON.parse(ta.value); mount(status, h('span.text-success', icon('check2'), ' valid JSON')); } catch (err) { mount(status, h('span.text-danger', err.message)); } }
      setAt(state.values, path, v);
      emitDebounced();
    });
    const lines = h('div.n8s-code-meta', h('span.n8s-code-lang', lang), status, h('div.flex-grow-1'), copyButton(() => ta.value), h('button.af-link-btn', { type: 'button', title: 'Expand editor', onclick: () => ta.classList.toggle('tall') }, icon('arrows-angle-expand')));
    return h('div.n8s-code-wrap', ta, lines);
  }

  function resourceLocator(p, path, v) {
    const val = v && typeof v === 'object' && v.__rl ? v : { __rl: true, mode: p.modes?.[0]?.name || 'id', value: typeof v === 'string' ? v : '' };
    const modes = p.modes || [{ name: 'id', displayName: 'ID' }, { name: 'url', displayName: 'By URL' }];
    const sel = h('select.form-select.form-select-sm.n8s-rl-mode', { disabled: readOnly }, modes.map((m) => h('option', { value: m.name, selected: m.name === val.mode }, m.displayName || m.name)));
    const input = h('input.form-control.form-control-sm', { value: typeof val.value === 'string' ? val.value : JSON.stringify(val.value), placeholder: modes.find((m) => m.name === val.mode)?.placeholder || 'Value or {{ expression }}', readonly: readOnly });
    const update = () => {
      const raw = input.value;
      setAt(state.values, path, { __rl: true, mode: sel.value, value: raw, ...(val.cachedResultName && sel.value === val.mode ? { cachedResultName: val.cachedResultName } : {}) });
      emitDebounced();
    };
    sel.addEventListener('change', update);
    input.addEventListener('input', update);
    return h('div.n8s-rl', sel, input, val.cachedResultName && h('div.n8s-field-note', icon('bookmark', 'me-1'), val.cachedResultName));
  }

  function collection(p, path, v, local) {
    const options = (p.options || []).filter((o) => o.type !== 'notice');
    const present = options.filter((o) => o.name in v);
    const addable = options.filter((o) => !(o.name in v));
    const items = visibleProps(present, v, state.values, { typeVersion }).map((o) => {
      const child = editorFor(o, [...path, o.name], v[o.name], v);
      return child && h('div.n8s-coll-item', child, !readOnly && h('button.n8s-coll-del', { type: 'button', title: 'Remove', onclick: () => { setAt(state.values, [...path, o.name], undefined); rerender(); emit(); } }, icon('x-lg')));
    });
    const add = !readOnly && addable.length && h('select.form-select.form-select-sm.n8s-coll-add', {
      onchange: (e) => {
        const o = addable.find((x) => x.name === e.target.value);
        if (!o) return;
        setAt(state.values, [...path, o.name], clone(o.default ?? (o.type === 'boolean' ? false : o.type === 'number' ? 0 : '')));
        rerender([...path, o.name]);
        emit();
      },
    }, h('option', { value: '' }, `+ ${p.placeholder || `Add ${label(p).toLowerCase()}`}`), addable.map((o) => h('option', { value: o.name, title: stripHtml(o.description || '') }, label(o))));
    return h('fieldset.n8s-coll', h('legend', label(p), p.description && h('span.n8s-field-hint', { title: stripHtml(p.description) }, icon('question-circle'))), items, add);
  }

  function fixedCollection(p, path, v) {
    const multiple = !!p.typeOptions?.multipleValues;
    const groups = p.options || [];
    const blocks = [];
    for (const g of groups) {
      const gv = v[g.name];
      const entries = multiple ? (Array.isArray(gv) ? gv : []) : gv ? [gv] : [];
      entries.forEach((entry, i) => {
        const ePath = multiple ? [...path, g.name, i] : [...path, g.name];
        const props = visibleProps(g.values || [], entry, state.values, { typeVersion });
        blocks.push(h('div.n8s-fc-entry',
          h('div.n8s-fc-head', h('span.n8s-fc-title', g.displayName || g.name, multiple ? ` #${i + 1}` : ''),
            !readOnly && multiple && i > 0 && h('button.n8s-coll-del', { type: 'button', title: 'Move up', onclick: () => { const arr = getAt(state.values, [...path, g.name]); [arr[i - 1], arr[i]] = [arr[i], arr[i - 1]]; rerender(); emit(); } }, icon('arrow-up')),
            !readOnly && h('button.n8s-coll-del', { type: 'button', title: 'Remove', onclick: () => { setAt(state.values, ePath, undefined); if (!multiple) setAt(state.values, [...path, g.name], undefined); rerender(); emit(); } }, icon('trash'))),
          props.map((cp) => editorFor(cp, [...ePath, cp.name], entry[cp.name], entry))));
      });
    }
    const addable = groups.filter((g) => multiple || !v[g.name]);
    const add = !readOnly && addable.length && h('div.n8s-fc-add', addable.map((g) => h('button.btn.btn-sm.btn-outline-secondary', {
      type: 'button',
      onclick: () => {
        const fresh = defaults(g.values || [], typeVersion);
        if (multiple) {
          const arr = getAt(state.values, [...path, g.name]);
          setAt(state.values, [...path, g.name], [...(Array.isArray(arr) ? arr : []), fresh]);
        } else setAt(state.values, [...path, g.name], fresh);
        rerender();
        emit();
      },
    }, icon('plus-lg', 'me-1'), p.placeholder && addable.length === 1 ? p.placeholder : `Add ${g.displayName || g.name}`)));
    return h('fieldset.n8s-coll.n8s-fc', h('legend', label(p), p.description && h('span.n8s-field-hint', { title: stripHtml(p.description) }, icon('question-circle'))), blocks, add);
  }

  const ASSIGN_TYPES = ['string', 'number', 'boolean', 'array', 'object'];
  function assignments(p, path, v) {
    const list = Array.isArray(v.assignments) ? v.assignments : [];
    const rows = list.map((a, i) => {
      const name = h('input.form-control.form-control-sm', { value: a.name ?? '', placeholder: 'name', readonly: readOnly });
      const type = h('select.form-select.form-select-sm', { disabled: readOnly }, ASSIGN_TYPES.map((t) => h('option', { value: t, selected: t === (a.type || 'string') }, t)));
      const val = h('input.form-control.form-control-sm.n8s-mono-input', { value: a.value == null ? '' : typeof a.value === 'object' ? JSON.stringify(a.value) : String(a.value), placeholder: 'value or ={{ expression }}', readonly: readOnly });
      const sync = () => {
        let value = val.value;
        if (!value.startsWith('=')) {
          if (type.value === 'number' && value !== '') value = Number(value);
          else if (type.value === 'boolean') value = value === 'true';
        }
        setAt(state.values, [...path, 'assignments', i], { id: a.id || uuid(), name: name.value, value, type: type.value });
        emitDebounced();
      };
      [name, val].forEach((x) => x.addEventListener('input', sync));
      type.addEventListener('change', sync);
      val.addEventListener('drop', (e) => {
        const text = e.dataTransfer.getData('text/plain');
        if (!/\{\{/.test(text)) return;
        e.preventDefault();
        val.value = val.value.startsWith('=') ? `${val.value} ${text}` : `=${text}`;
        sync();
      });
      return h('div.n8s-assign', name, type, val, !readOnly && h('button.n8s-coll-del', { type: 'button', title: 'Remove', onclick: () => { setAt(state.values, [...path, 'assignments', i], undefined); rerender(); emit(); } }, icon('trash')));
    });
    return h('fieldset.n8s-coll', h('legend', label(p)), h('div.n8s-assign-head', h('span', 'Name'), h('span', 'Type'), h('span', 'Value')), rows,
      !readOnly && h('button.btn.btn-sm.btn-outline-secondary.mt-1', {
        type: 'button',
        onclick: () => { setAt(state.values, [...path, 'assignments'], [...list, { id: uuid(), name: '', value: '', type: 'string' }]); rerender(); emit(); },
      }, icon('plus-lg', 'me-1'), 'Add field'));
  }

  const OPS = [
    ['string', ['equals', 'notEquals', 'contains', 'notContains', 'startsWith', 'notStartsWith', 'endsWith', 'notEndsWith', 'regex', 'notRegex', 'exists', 'notExists', 'empty', 'notEmpty']],
    ['number', ['equals', 'notEquals', 'gt', 'lt', 'gte', 'lte', 'exists', 'notExists', 'empty', 'notEmpty']],
    ['dateTime', ['equals', 'notEquals', 'after', 'before', 'afterOrEquals', 'beforeOrEquals', 'exists', 'notExists', 'empty', 'notEmpty']],
    ['boolean', ['true', 'false', 'equals', 'notEquals', 'exists', 'notExists', 'empty', 'notEmpty']],
    ['array', ['contains', 'notContains', 'lengthEquals', 'lengthNotEquals', 'lengthGt', 'lengthLt', 'lengthGte', 'lengthLte', 'exists', 'notExists', 'empty', 'notEmpty']],
    ['object', ['exists', 'notExists', 'empty', 'notEmpty']],
  ];
  const SINGLE = new Set(['exists', 'notExists', 'empty', 'notEmpty', 'true', 'false']);
  function filter(p, path, v) {
    const conds = Array.isArray(v.conditions) ? v.conditions : [];
    const rows = conds.map((c, i) => {
      const left = h('input.form-control.form-control-sm.n8s-mono-input', { value: c.leftValue ?? '', placeholder: '={{ $json.field }}', readonly: readOnly });
      const op = h('select.form-select.form-select-sm', { disabled: readOnly }, OPS.map(([type, ops]) => h('optgroup', { label: type }, ops.map((o) => h('option', { value: `${type}:${o}`, selected: c.operator?.type === type && c.operator?.operation === o }, `${type} ${o.replace(/([A-Z])/g, ' $1').toLowerCase()}`)))));
      const right = h('input.form-control.form-control-sm.n8s-mono-input', { value: c.rightValue ?? '', placeholder: 'value', readonly: readOnly, disabled: SINGLE.has(c.operator?.operation) });
      const sync = () => {
        const [type, operation] = op.value.split(':');
        right.disabled = SINGLE.has(operation);
        setAt(state.values, [...path, 'conditions', i], { id: c.id || uuid(), leftValue: left.value, rightValue: right.disabled ? '' : right.value, operator: { type, operation, ...(SINGLE.has(operation) ? { singleValue: true } : {}) } });
        emitDebounced();
      };
      [left, right].forEach((x) => {
        x.addEventListener('input', sync);
        x.addEventListener('drop', (e) => {
          const text = e.dataTransfer.getData('text/plain');
          if (!/\{\{/.test(text)) return;
          e.preventDefault();
          x.value = `=${text}`;
          sync();
        });
      });
      op.addEventListener('change', sync);
      return h('div.n8s-cond', left, op, right, !readOnly && h('button.n8s-coll-del', { type: 'button', onclick: () => { setAt(state.values, [...path, 'conditions', i], undefined); rerender(); emit(); } }, icon('trash')));
    });
    const comb = h('select.form-select.form-select-sm.n8s-comb', { disabled: readOnly }, ['and', 'or'].map((x) => h('option', { value: x, selected: (v.combinator || 'and') === x }, x.toUpperCase())));
    comb.addEventListener('change', () => { setAt(state.values, [...path, 'combinator'], comb.value); emit(); });
    return h('fieldset.n8s-coll', h('legend', label(p), conds.length > 1 && h('span.ms-2', 'match ', comb)), rows,
      !readOnly && h('button.btn.btn-sm.btn-outline-secondary.mt-1', {
        type: 'button',
        onclick: () => {
          const next = { ...v, conditions: [...conds, { id: uuid(), leftValue: '', rightValue: '', operator: { type: 'string', operation: 'equals' } }], combinator: v.combinator || 'and' };
          if (!next.options) next.options = { caseSensitive: true, leftValue: '', typeValidation: 'strict', version: 2 };
          setAt(state.values, path, next);
          rerender();
          emit();
        },
      }, icon('plus-lg', 'me-1'), 'Add condition'));
  }

  // ------------------------------------------------------------------ render

  let focusPath = null;
  function rerender(path) {
    if (path) focusPath = path.join('.');
    const active = document.activeElement;
    const activePath = active && root.contains(active) ? active.closest('.n8s-field')?.dataset.path : null;
    const visible = visibleProps(props, state.values, state.values, { typeVersion, defaults: defaults(props, typeVersion, state.values) });
    mount(root,
      credentialsSlot || null,
      visible.map((p) => editorFor(p, [p.name], state.values[p.name], state.values)),
      !visible.length && h('div.n8s-field-note', 'This node has no parameters.'));
    const target = focusPath || activePath;
    if (target) {
      const el = root.querySelector(`.n8s-field[data-path="${CSS.escape(target)}"] :is(input,textarea,select)`);
      el?.focus({ preventScroll: true });
    }
    focusPath = null;
  }
  rerender();

  return {
    el: root,
    values: () => clone(state.values),
    set(values) { state.values = clone(values || {}); rerender(); },
  };
}

export function stripHtml(s) {
  return String(s ?? '').replace(/<[^>]*>/g, '').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
}

export { FIELD_TYPES_WITH_RERENDER };
