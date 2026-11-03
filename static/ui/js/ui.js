// Shared UI components for the pipeline app. Everything returns DOM nodes.

import { h, icon, clear, mount, debounce, prefs } from './dom.js';
import { fmtDate, relTime } from './format.js';

// ------------------------------------------------------------------ states

export const STATES = {
  success: { color: '#16a34a', bg: '#dcfce7', label: 'success', icon: 'check-circle-fill' },
  failed: { color: '#dc2626', bg: '#fee2e2', label: 'failed', icon: 'x-circle-fill' },
  running: { color: '#0891b2', bg: '#cffafe', label: 'running', icon: 'arrow-repeat' },
  queued: { color: '#64748b', bg: '#e2e8f0', label: 'queued', icon: 'hourglass-split' },
  scheduled: { color: '#a16207', bg: '#fef3c7', label: 'scheduled', icon: 'calendar-event' },
  up_for_retry: { color: '#ca8a04', bg: '#fef9c3', label: 'up for retry', icon: 'arrow-counterclockwise' },
  up_for_reschedule: { color: '#0d9488', bg: '#ccfbf1', label: 'up for reschedule', icon: 'clock-history' },
  upstream_failed: { color: '#ea580c', bg: '#ffedd5', label: 'upstream failed', icon: 'exclamation-triangle-fill' },
  skipped: { color: '#db2777', bg: '#fce7f3', label: 'skipped', icon: 'skip-forward-fill' },
  deferred: { color: '#7c3aed', bg: '#ede9fe', label: 'deferred', icon: 'pause-circle' },
  removed: { color: '#94a3b8', bg: '#f1f5f9', label: 'removed', icon: 'trash' },
  restarting: { color: '#9333ea', bg: '#f3e8ff', label: 'restarting', icon: 'bootstrap-reboot' },
  paused: { color: '#64748b', bg: '#f1f5f9', label: 'paused', icon: 'pause-fill' },
  active: { color: '#0072CE', bg: '#e8f4ff', label: 'active', icon: 'play-fill' },
  none: { color: '#94a3b8', bg: '#f8fafc', label: 'no status', icon: 'circle' },
};

export const RUN_STATES = ['queued', 'running', 'success', 'failed'];
export const TASK_STATES = ['success', 'failed', 'running', 'queued', 'scheduled', 'up_for_retry',
  'up_for_reschedule', 'upstream_failed', 'skipped', 'deferred', 'removed', 'restarting'];

export const stateInfo = (state) => STATES[state || 'none'] || STATES.none;

export function stateBadge(state, { size = '' } = {}) {
  const info = stateInfo(state);
  return h('span.af-badge', {
    class: [size && `af-badge-${size}`, state === 'running' && 'af-pulse'],
    style: { color: info.color, background: info.bg },
    title: info.label,
  }, h('span.af-dot', { style: { background: info.color } }), info.label);
}

export function stateDot(state, title) {
  const info = stateInfo(state);
  return h('span.af-dot', {
    class: state === 'running' && 'af-pulse',
    style: { background: info.color },
    title: title || info.label,
  });
}

// ------------------------------------------------------------------ toasts

let toastHost;
export function toast(message, type = 'info', { title, timeout = 4500, action } = {}) {
  if (!toastHost) {
    toastHost = h('div.af-toasts', { 'aria-live': 'polite' });
    document.body.appendChild(toastHost);
  }
  const icons = { success: 'check-circle-fill', error: 'x-octagon-fill', warning: 'exclamation-triangle-fill', info: 'info-circle-fill' };
  const close = () => {
    el.classList.add('af-toast-out');
    setTimeout(() => el.remove(), 200);
  };
  const el = h(`div.af-toast.af-toast-${type}`, { role: 'status' },
    icon(icons[type] || icons.info, 'af-toast-icon'),
    h('div.af-toast-body',
      title && h('div.af-toast-title', title),
      h('div', String(message)),
      action && h('button.af-link-btn', { onclick: () => { action.onClick(); close(); } }, action.label)),
    h('button.af-toast-close', { 'aria-label': 'Dismiss', onclick: close }, icon('x')));
  toastHost.appendChild(el);
  if (timeout) setTimeout(close, type === 'error' ? timeout * 2 : timeout);
  return close;
}

export function toastError(err, title = 'Something went wrong') {
  if (err?.name === 'AbortError') return;
  console.error(err);
  toast(err?.message || String(err), 'error', { title });
}

// ------------------------------------------------------------------ overlays

const overlayStack = [];
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && overlayStack.length) {
    const top = overlayStack[overlayStack.length - 1];
    if (!top.persistent) { e.preventDefault(); top.close(); }
  }
});

function trapFocus(container) {
  setTimeout(() => {
    const target = container.querySelector('[autofocus], input:not([type=hidden]), textarea, select, button.btn-primary, button');
    target?.focus();
  }, 30);
}

/**
 * modal({ title, body, size: 'sm'|'lg'|'xl', actions: [{label, variant, onClick(close)}] })
 * onClick may return a promise; the button shows a spinner and the modal stays open on error.
 */
export function modal({ title, subtitle, body, size = '', actions = [], onClose, persistent = false, icon: ic } = {}) {
  const prevFocus = document.activeElement;
  const footer = actions.length ? h('div.af-modal-footer') : null;
  const backdrop = h('div.af-backdrop', { onmousedown: (e) => { if (e.target === backdrop && !persistent) close(); } });
  const dialog = h(`div.af-modal${size ? `.af-modal-${size}` : ''}`, { role: 'dialog', 'aria-modal': 'true' },
    h('div.af-modal-header',
      h('div', h('h5.af-modal-title', ic && icon(ic, 'me-2'), title), subtitle && h('div.af-modal-sub', subtitle)),
      h('button.af-icon-btn', { 'aria-label': 'Close', onclick: () => close() }, icon('x-lg'))),
    h('div.af-modal-body', body),
    footer);
  backdrop.appendChild(dialog);

  const entry = { close, persistent };
  let closed = false;
  function close(result) {
    if (closed) return;
    closed = true;
    overlayStack.splice(overlayStack.indexOf(entry), 1);
    backdrop.classList.add('af-fade-out');
    setTimeout(() => backdrop.remove(), 150);
    onClose?.(result);
    prevFocus?.focus?.();
  }

  for (const action of actions) {
    const btn = h(`button.btn.btn-sm.btn-${action.variant || 'outline-secondary'}`, {
      type: 'button',
      disabled: action.disabled,
      onclick: async () => {
        if (!action.onClick) return close();
        btn.disabled = true;
        const original = [...btn.childNodes];
        mount(btn, spinner('sm'), ' ', action.label);
        try {
          await action.onClick(close, btn);
        } catch (err) {
          toastError(err, action.errorTitle || `${action.label} failed`);
        } finally {
          if (!closed) { btn.disabled = false; mount(btn, ...original); }
        }
      },
    }, action.icon && icon(action.icon, 'me-1'), action.label);
    footer.appendChild(btn);
  }

  document.body.appendChild(backdrop);
  overlayStack.push(entry);
  trapFocus(dialog);
  return { close, el: dialog, body: dialog.querySelector('.af-modal-body') };
}

export function confirmDialog({ title = 'Are you sure?', message, confirmLabel = 'Confirm', danger = false, details } = {}) {
  return new Promise((resolve) => {
    let answered = false;
    modal({
      title,
      size: 'sm',
      icon: danger ? 'exclamation-triangle-fill' : 'question-circle',
      body: h('div', h('p.mb-2', message), details),
      actions: [
        { label: 'Cancel' },
        { label: confirmLabel, variant: danger ? 'danger' : 'primary', onClick: (close) => { answered = true; resolve(true); close(); } },
      ],
      onClose: () => { if (!answered) resolve(false); },
    });
  });
}

/** Right-hand slide-over panel. Returns {close, body, setTitle}. */
export function drawer({ title, subtitle, body, width = 760, actions, onClose } = {}) {
  const prevFocus = document.activeElement;
  const titleEl = h('h5.af-drawer-title', title);
  const subEl = h('div.af-drawer-sub', subtitle);
  const bodyEl = h('div.af-drawer-body', body);
  const backdrop = h('div.af-backdrop.af-backdrop-drawer', { onmousedown: (e) => { if (e.target === backdrop) close(); } });
  const panel = h('aside.af-drawer', { style: { width: `min(${width}px, 100vw)` }, role: 'dialog', 'aria-modal': 'true' },
    h('div.af-drawer-header',
      h('div.af-drawer-heading', titleEl, subEl),
      h('div.d-flex.gap-1.align-items-center', actions, h('button.af-icon-btn', { 'aria-label': 'Close', onclick: () => close() }, icon('x-lg')))),
    bodyEl);
  backdrop.appendChild(panel);
  const entry = { close, persistent: false };
  let closed = false;
  const cleanups = [];
  function close() {
    if (closed) return;
    closed = true;
    cleanups.forEach((fn) => fn());
    overlayStack.splice(overlayStack.indexOf(entry), 1);
    panel.classList.add('af-drawer-out');
    backdrop.classList.add('af-fade-out');
    setTimeout(() => backdrop.remove(), 200);
    onClose?.();
    prevFocus?.focus?.();
  }
  document.body.appendChild(backdrop);
  overlayStack.push(entry);
  return {
    close,
    body: bodyEl,
    el: panel,
    isOpen: () => !closed,
    onCleanup: (fn) => cleanups.push(fn),
    setTitle: (t, sub) => { mount(titleEl, t); if (sub !== undefined) mount(subEl, sub); },
  };
}

// ------------------------------------------------------------------ forms

/**
 * formModal({ title, fields, submitLabel, onSubmit(values, close) })
 * field: { name, label, type: text|number|password|textarea|json|select|checkbox|datetime|tags|static,
 *          value, options: [{value,label}] | string[], required, help, placeholder, readonly, rows }
 */
export function formModal({ title, subtitle, fields, submitLabel = 'Save', onSubmit, size = '', danger = false, icon: ic, extra } = {}) {
  const form = buildForm(fields);
  const errorBox = h('div.alert.alert-danger.py-2.small.d-none');
  const m = modal({
    title, subtitle, size, icon: ic,
    body: h('div', errorBox, form.el, extra),
    actions: [
      { label: 'Cancel' },
      {
        label: submitLabel,
        variant: danger ? 'danger' : 'primary',
        onClick: async (close) => {
          errorBox.classList.add('d-none');
          let values;
          try {
            values = form.values();
          } catch (err) {
            mount(errorBox, err.message);
            errorBox.classList.remove('d-none');
            return;
          }
          try {
            await onSubmit(values, close);
          } catch (err) {
            if (err?.name === 'AbortError') return;
            mount(errorBox, icon('exclamation-octagon', 'me-1'), err.message || String(err));
            errorBox.classList.remove('d-none');
          }
        },
      },
    ],
  });
  form.el.addEventListener('submit', (e) => {
    e.preventDefault();
    m.el.querySelector('.af-modal-footer .btn-primary, .af-modal-footer .btn-danger')?.click();
  });
  return { ...m, form };
}

export function buildForm(fields) {
  const getters = {};
  const el = h('form.af-form', { novalidate: true });
  for (const f of fields) {
    if (f.type === 'section') {
      el.appendChild(h('div.af-form-section', f.label));
      continue;
    }
    const id = `af-f-${f.name}-${Math.random().toString(36).slice(2, 7)}`;
    let input;
    let get;
    const value = f.value ?? '';
    switch (f.type) {
      case 'textarea':
        input = h('textarea.form-control.form-control-sm', { id, rows: f.rows || 4, placeholder: f.placeholder, readonly: f.readonly }, value);
        get = () => input.value;
        break;
      case 'json': {
        const editor = jsonEditor(f.value ?? {}, { rows: f.rows || 8, allowEmpty: !f.required });
        input = editor.el;
        get = () => editor.value();
        break;
      }
      case 'select': {
        const opts = (f.options || []).map((o) => (typeof o === 'object' ? o : { value: o, label: o }));
        input = h('select.form-select.form-select-sm', { id, disabled: f.readonly },
          opts.map((o) => h('option', { value: o.value, selected: String(o.value) === String(value) }, o.label)));
        get = () => input.value;
        break;
      }
      case 'checkbox':
        input = h('div.form-check.form-switch',
          h('input.form-check-input', { id, type: 'checkbox', checked: !!f.value, disabled: f.readonly }),
          h('label.form-check-label.small', { for: id }, f.checkLabel || ''));
        get = () => input.querySelector('input').checked;
        break;
      case 'datetime':
        input = h('input.form-control.form-control-sm', { id, type: 'datetime-local', step: 1, value: f.value || '' });
        get = () => (input.value ? new Date(input.value).toISOString() : null);
        break;
      case 'tags':
        input = h('input.form-control.form-control-sm', { id, value: Array.isArray(value) ? value.join(', ') : value, placeholder: f.placeholder || 'comma, separated' });
        get = () => input.value.split(',').map((t) => t.trim()).filter(Boolean);
        break;
      case 'static':
        input = h('div.form-control-plaintext.form-control-sm.af-mono', value);
        get = () => f.value;
        break;
      default:
        input = h('input.form-control.form-control-sm', {
          id, type: f.type || 'text', value, placeholder: f.placeholder, readonly: f.readonly,
          autocomplete: f.type === 'password' ? 'new-password' : 'off', min: f.min, max: f.max, step: f.step,
        });
        get = () => (f.type === 'number' ? (input.value === '' ? null : Number(input.value)) : input.value);
    }
    getters[f.name] = () => {
      const v = get();
      if (f.required && (v == null || v === '' || (Array.isArray(v) && !v.length))) {
        throw new Error(`${f.label || f.name} is required.`);
      }
      return v;
    };
    el.appendChild(h('div.af-form-row', { class: f.type === 'checkbox' && 'af-form-row-inline' },
      f.type !== 'checkbox' || f.label ? h('label.form-label.small.fw-semibold', { for: id }, f.label, f.required && h('span.text-danger', ' *')) : null,
      input,
      f.help && h('div.form-text.small', f.help)));
  }
  el.appendChild(h('button', { type: 'submit', hidden: true }));
  return {
    el,
    values() {
      const out = {};
      for (const [name, get] of Object.entries(getters)) out[name] = get();
      return out;
    },
  };
}

/** Textarea JSON editor with live validation, format and Tab support. */
export function jsonEditor(initial, { rows = 8, allowEmpty = true, onChange } = {}) {
  const text = typeof initial === 'string' ? initial : JSON.stringify(initial, null, 2);
  const status = h('span.af-json-status');
  const ta = h('textarea.form-control.form-control-sm.af-mono.af-json-input', { rows, spellcheck: 'false' }, text);
  const validate = () => {
    const v = ta.value.trim();
    if (!v && allowEmpty) { mount(status, h('span.text-muted', 'empty')); ta.classList.remove('is-invalid'); return true; }
    try {
      JSON.parse(v);
      mount(status, h('span.text-success', icon('check2'), ' valid JSON'));
      ta.classList.remove('is-invalid');
      return true;
    } catch (err) {
      mount(status, h('span.text-danger', icon('x'), ` ${err.message}`));
      ta.classList.add('is-invalid');
      return false;
    }
  };
  ta.addEventListener('input', debounce(() => { validate(); onChange?.(ta.value); }, 200));
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const { selectionStart: a, selectionEnd: b } = ta;
      ta.setRangeText('  ', a, b, 'end');
    }
  });
  const fmt = h('button.af-link-btn', {
    type: 'button',
    onclick: () => { try { ta.value = JSON.stringify(JSON.parse(ta.value), null, 2); validate(); } catch { validate(); } },
  }, icon('braces'), ' Format');
  validate();
  return {
    el: h('div.af-json-editor', ta, h('div.af-json-bar', status, fmt)),
    textarea: ta,
    value() {
      const v = ta.value.trim();
      if (!v) {
        if (allowEmpty) return null;
        throw new Error('JSON value is required.');
      }
      try { return JSON.parse(v); } catch (err) { throw new Error(`Invalid JSON: ${err.message}`); }
    },
    set(v) { ta.value = typeof v === 'string' ? v : JSON.stringify(v, null, 2); validate(); },
  };
}

// ------------------------------------------------------------------ data display

/** Collapsible JSON tree. */
export function jsonView(value, { open = 2, root = true } = {}) {
  const node = (v, depth, key) => {
    const keyEl = key != null ? h('span.af-jv-key', `${JSON.stringify(key).slice(1, -1)}: `) : null;
    if (v === null) return h('div.af-jv-row', keyEl, h('span.af-jv-null', 'null'));
    if (typeof v !== 'object') {
      const cls = typeof v === 'string' ? 'af-jv-str' : typeof v === 'number' ? 'af-jv-num' : 'af-jv-bool';
      return h('div.af-jv-row', keyEl, h(`span.${cls}`, typeof v === 'string' ? JSON.stringify(v) : String(v)));
    }
    const isArr = Array.isArray(v);
    const entries = isArr ? v.map((x, i) => [i, x]) : Object.entries(v);
    if (!entries.length) return h('div.af-jv-row', keyEl, h('span.af-jv-punct', isArr ? '[]' : '{}'));
    const children = h('div.af-jv-children');
    let rendered = false;
    const toggle = h('button.af-jv-toggle', { type: 'button' }, icon('caret-right-fill'));
    const summary = h('span.af-jv-summary', isArr ? `[${entries.length}]` : `{${entries.length}}`);
    const wrap = h('div.af-jv-node', h('div.af-jv-row', toggle, keyEl, summary), children);
    const setOpen = (o) => {
      if (o && !rendered) { entries.forEach(([k, x]) => children.appendChild(node(x, depth + 1, isArr ? null : k))); rendered = true; }
      wrap.classList.toggle('open', o);
    };
    toggle.addEventListener('click', () => setOpen(!wrap.classList.contains('open')));
    summary.addEventListener('click', () => setOpen(!wrap.classList.contains('open')));
    setOpen(depth < open);
    return wrap;
  };
  const tree = h('div.af-jv', node(value, 0, null));
  if (!root) return tree;
  const text = JSON.stringify(value, null, 2);
  return h('div.af-jv-wrap',
    h('div.af-jv-tools', copyButton(text, 'Copy JSON')),
    tree);
}

/** Read-only code block with line numbers and light Python/SQL/JSON highlighting. */
export function codeBlock(text, { lang = 'python', highlightLines = [], maxHeight } = {}) {
  const lines = String(text ?? '').split('\n');
  const body = h('div.af-code-lines');
  lines.forEach((line, i) => {
    body.appendChild(h('div.af-code-line', { class: highlightLines.includes(i + 1) && 'af-code-hl', id: `L${i + 1}` },
      h('span.af-code-num', String(i + 1)),
      h('span.af-code-text', highlight(line, lang))));
  });
  return h('div.af-code', { style: maxHeight ? { maxHeight } : null },
    h('div.af-code-tools', h('span.af-code-lang', lang), copyButton(text, 'Copy')),
    body);
}

const PY_KW = /\b(def|class|return|import|from|as|with|for|in|if|elif|else|try|except|finally|raise|while|lambda|yield|None|True|False|and|or|not|is|pass|break|continue|global|nonlocal|async|await|assert|del)\b/;
const SQL_KW = /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP|BY|ORDER|LIMIT|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|AS|AND|OR|NOT|NULL|IS|IN|WITH|UNION|ALL|DISTINCT|CASE|WHEN|THEN|ELSE|END|MERGE|USING)\b/i;

function highlight(line, lang) {
  if (lang !== 'python' && lang !== 'sql') return line || ' ';
  const frag = document.createDocumentFragment();
  const commentRe = lang === 'python' ? /#.*$/ : /--.*$/;
  const tokenRe = /("""|'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|@\w+|\b\d+(?:\.\d+)?\b|\b\w+\b)/g;
  let code = line;
  let comment = '';
  const cm = line.match(commentRe);
  if (cm && !/["']/.test(line.slice(0, cm.index))) { code = line.slice(0, cm.index); comment = cm[0]; }
  let last = 0;
  for (const m of code.matchAll(tokenRe)) {
    if (m.index > last) frag.appendChild(document.createTextNode(code.slice(last, m.index)));
    const t = m[0];
    let cls = null;
    if (/^["']/.test(t)) cls = 'af-tk-str';
    else if (t[0] === '@') cls = 'af-tk-dec';
    else if (/^\d/.test(t)) cls = 'af-tk-num';
    else if ((lang === 'python' ? PY_KW : SQL_KW).test(t) && t.length > 1) cls = 'af-tk-kw';
    frag.appendChild(cls ? h(`span.${cls}`, t) : document.createTextNode(t));
    last = m.index + t.length;
  }
  if (last < code.length) frag.appendChild(document.createTextNode(code.slice(last)));
  if (comment) frag.appendChild(h('span.af-tk-com', comment));
  if (!line) frag.appendChild(document.createTextNode(' '));
  return frag;
}

export function copyButton(text, label = '') {
  const btn = h('button.af-copy-btn', {
    type: 'button',
    title: 'Copy to clipboard',
    onclick: async (e) => {
      e.stopPropagation();
      const value = typeof text === 'function' ? text() : text;
      try {
        await navigator.clipboard.writeText(value);
      } catch {
        const ta = h('textarea', { style: 'position:fixed;opacity:0' }, value);
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        ta.remove();
      }
      mount(btn, icon('check2'), label && ` Copied`);
      btn.classList.add('copied');
      setTimeout(() => { mount(btn, icon('clipboard'), label && ` ${label}`); btn.classList.remove('copied'); }, 1400);
    },
  }, icon('clipboard'), label && ` ${label}`);
  return btn;
}

/** Definition grid: kv([['Owner', 'airflow'], ['Tags', node]]) */
export function kv(rows, { cols = 2 } = {}) {
  return h('dl.af-kv', { style: { '--af-kv-cols': cols } },
    rows.filter(Boolean).map(([k, v, opts = {}]) => h('div.af-kv-item', { class: opts.wide && 'af-kv-wide' },
      h('dt', k),
      h('dd', v == null || v === '' ? h('span.text-muted', '—') : v))));
}

/** <time> that keeps its relative label fresh (see startClock). */
export function timeAgo(value, { abs = false } = {}) {
  if (!value) return h('span.text-muted', '—');
  return h('time.af-time', { datetime: value, title: fmtDate(value), dataset: { rel: '1' } },
    abs ? fmtDate(value) : relTime(value));
}

export function startClock() {
  setInterval(() => {
    document.querySelectorAll('time[data-rel]').forEach((t) => { t.textContent = relTime(t.getAttribute('datetime')); });
  }, 15000);
}

export function spinner(size = '') {
  return h(`span.spinner-border${size ? `.spinner-border-${size}` : ''}.af-spinner`, { role: 'status', 'aria-label': 'Loading' });
}

export function skeleton(rows = 5, { height = 18 } = {}) {
  return h('div.af-skeleton-wrap', Array.from({ length: rows }, (_, i) =>
    h('div.af-skeleton', { style: { height: `${height}px`, width: `${70 + ((i * 37) % 30)}%` } })));
}

export function emptyState({ icon: ic = 'inbox', title = 'Nothing here', message, action } = {}) {
  return h('div.af-empty', icon(ic, 'af-empty-icon'), h('div.af-empty-title', title), message && h('div.af-empty-msg', message), action);
}

export function errorState(err, retry) {
  const status = err?.status;
  const service = (window.APP_UI || window.AIRFLOW_UI || {}).service || 'Airflow';
  const hint = status === 404 ? 'It may have been deleted, or the id is mistyped.'
    : status === 403 ? `The ${service} account used by the dashboard lacks permission for this.`
      : status === 0 || status >= 500 ? `${service} may be unreachable. Check the health page.` : null;
  return h('div.af-error',
    icon('exclamation-octagon', 'af-error-icon'),
    h('div',
      h('div.af-error-title', status ? `Error ${status}` : 'Error'),
      h('div.af-error-msg', err?.message || String(err)),
      hint && h('div.af-error-hint', hint),
      retry && h('button.btn.btn-sm.btn-outline-danger.mt-2', { onclick: retry }, icon('arrow-clockwise', 'me-1'), 'Retry')));
}

/** Render async content into el with skeleton + error handling. */
export async function loadInto(el, loader, { rows = 5 } = {}) {
  mount(el, skeleton(rows));
  try {
    const content = await loader();
    mount(el, content);
  } catch (err) {
    if (err?.name === 'AbortError') return;
    mount(el, errorState(err, () => loadInto(el, loader, { rows })));
  }
}

// ------------------------------------------------------------------ layout pieces

export function pageHeader({ title, icon: ic, subtitle, actions, crumbs, badges } = {}) {
  return h('div.af-page-header',
    crumbs && h('nav.af-crumbs', crumbs.map((c, i) => [
      i ? icon('chevron-right', 'af-crumb-sep') : null,
      c.href ? h('a', { href: c.href, dataset: { link: '' } }, c.label) : h('span', c.label),
    ])),
    h('div.af-page-header-row',
      h('div.af-page-heading',
        h('h1.af-page-title', ic && h('span.af-page-icon', icon(ic)), h('span.af-page-title-text', title), badges),
        subtitle && h('div.af-page-sub', subtitle)),
      actions && h('div.af-page-actions', actions)));
}

export function card({ title, icon: ic, actions, body, cls = '', flush = false } = {}) {
  return h('section.af-card', { class: cls },
    (title || actions) && h('header.af-card-header',
      h('h2.af-card-title', ic && icon(ic, 'me-2'), title),
      actions && h('div.af-card-actions', actions)),
    h('div.af-card-body', { class: flush && 'af-card-flush' }, body));
}

export function statCard({ label, value, icon: ic, tone = 'blue', sub, onClick, href } = {}) {
  const tag = href ? 'a.af-stat' : onClick ? 'button.af-stat' : 'div.af-stat';
  return h(tag, { class: `af-tone-${tone}`, href, dataset: href ? { link: '' } : undefined, onclick: onClick, type: onClick ? 'button' : undefined },
    h('div.af-stat-icon', icon(ic)),
    h('div.af-stat-text', h('div.af-stat-label', label), h('div.af-stat-value', value), sub && h('div.af-stat-sub', sub)));
}

export function tabs(items, active, onChange) {
  const bar = h('div.af-tabs', { role: 'tablist' });
  const render = (current) => {
    mount(bar, items.map((t) => h('button.af-tab', {
      type: 'button', role: 'tab', class: t.id === current && 'active', 'aria-selected': t.id === current,
      onclick: () => { render(t.id); onChange(t.id); },
    }, t.icon && icon(t.icon, 'me-1'), t.label, t.count != null && h('span.af-tab-count', String(t.count)))));
  };
  render(active);
  bar.setActive = render;
  return bar;
}

/** Chip-style segmented filter. options: [{value, label, color}] */
export function chips(options, value, onChange, { multi = false } = {}) {
  let current = multi ? new Set(value || []) : value;
  const bar = h('div.af-chips');
  const render = () => mount(bar, options.map((o) => {
    const on = multi ? current.has(o.value) : current === o.value;
    return h('button.af-chip', {
      type: 'button',
      class: on && 'active',
      onclick: () => {
        if (multi) { on ? current.delete(o.value) : current.add(o.value); onChange([...current]); }
        else { current = on && o.value !== '' ? '' : o.value; onChange(current); }
        render();
      },
    }, o.color && h('span.af-dot', { style: { background: o.color } }), o.label, o.count != null && h('span.af-chip-count', String(o.count)));
  }));
  render();
  return bar;
}

export function searchInput({ placeholder = 'Search…', value = '', onInput, delay = 250, autofocus = false, shortcut = true } = {}) {
  const input = h('input.form-control.form-control-sm', { type: 'search', placeholder, value, autofocus, dataset: shortcut ? { searchShortcut: '' } : undefined });
  input.addEventListener('input', debounce(() => onInput(input.value.trim()), delay));
  return h('div.af-search', icon('search', 'af-search-icon'), input, shortcut && h('kbd.af-kbd-hint', '/'));
}

/** Dropdown menu anchored to a button. items: [{label, icon, onClick, danger, href, divider, disabled}] */
export function menuButton(label, items, { variant = 'outline-secondary', icon: ic = 'three-dots-vertical', align = 'end', title } = {}) {
  const btn = h(`button.btn.btn-sm.btn-${variant}`, { type: 'button', title: title || (typeof label === 'string' ? label : 'More actions'), 'aria-haspopup': 'menu' },
    ic && icon(ic), label && h('span.ms-1', label));
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    openMenu(btn, typeof items === 'function' ? items() : items, align);
  });
  return btn;
}

let openMenuEl = null;
export function openMenu(anchor, items, align = 'end') {
  closeMenu();
  const menu = h('div.af-menu', { role: 'menu' }, items.filter(Boolean).map((it) => {
    if (it.divider) return h('div.af-menu-divider');
    if (it.header) return h('div.af-menu-header', it.header);
    const tag = it.href ? 'a.af-menu-item' : 'button.af-menu-item';
    return h(tag, {
      role: 'menuitem', type: it.href ? undefined : 'button', href: it.href,
      target: it.external ? '_blank' : undefined, rel: it.external ? 'noopener' : undefined,
      dataset: it.href && !it.external ? { link: '' } : undefined,
      class: it.danger && 'danger', disabled: it.disabled,
      onclick: () => { closeMenu(); it.onClick?.(); },
    }, it.icon && icon(it.icon), h('span', it.label), it.external && icon('box-arrow-up-right', 'ms-auto small'));
  }));
  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  const mw = menu.offsetWidth;
  const mh = menu.offsetHeight;
  let left = align === 'end' ? r.right - mw : r.left;
  left = Math.max(8, Math.min(left, window.innerWidth - mw - 8));
  let top = r.bottom + 4;
  if (top + mh > window.innerHeight - 8) top = Math.max(8, r.top - mh - 4);
  Object.assign(menu.style, { left: `${left}px`, top: `${top}px` });
  openMenuEl = menu;
  openMenuAnchor = anchor;
  menu.querySelector('.af-menu-item')?.focus({ preventScroll: true });
}
let openMenuAnchor = null;
export function closeMenu() { openMenuEl?.remove(); openMenuEl = null; openMenuAnchor = null; }
// Outside clicks close the menu. The click that opened it (bubbling up from its anchor)
// and clicks on non-item parts of the menu (headers, dividers) are ignored, so callers
// don't need to stopPropagation() themselves.
document.addEventListener('click', (e) => {
  if (!openMenuEl || !(e.target instanceof Node)) return;
  if (openMenuAnchor?.contains(e.target) || openMenuEl.contains(e.target)) return;
  closeMenu();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });
window.addEventListener('resize', closeMenu);
window.addEventListener('scroll', (e) => {
  if (openMenuEl && e.target instanceof Node && openMenuEl.contains(e.target)) return;
  closeMenu();
}, true);

// ------------------------------------------------------------------ tables

/**
 * dataTable({ columns, rows, rowKey, sort: {key, dir}, onSort(key, dir), onRowClick(row),
 *             selectable, selected: Set, onSelect(set), empty, rowClass(row), dense })
 * column: { key, label, render(row) -> node|string, sortable: true|'field', width, align, cls, title }
 */
export function dataTable(opts) {
  const { columns, rows, rowKey = (r) => r.id, sort, onSort, onRowClick, selectable, empty, rowClass, dense } = opts;
  const selected = opts.selected || new Set();
  const notify = () => opts.onSelect?.(selected);

  if (!rows.length) return empty || emptyState({ title: 'No results', message: 'Try adjusting the filters.' });

  const allKeys = rows.map(rowKey);
  const headCheck = selectable && h('input.form-check-input', {
    type: 'checkbox', 'aria-label': 'Select all',
    checked: allKeys.length && allKeys.every((k) => selected.has(k)),
    onclick: (e) => {
      e.stopPropagation();
      allKeys.forEach((k) => (e.target.checked ? selected.add(k) : selected.delete(k)));
      table.querySelectorAll('tbody .form-check-input').forEach((c) => { c.checked = e.target.checked; });
      table.querySelectorAll('tbody tr').forEach((tr) => tr.classList.toggle('af-row-selected', e.target.checked));
      notify();
    },
  });

  const thead = h('thead', h('tr',
    selectable && h('th.af-col-check', headCheck),
    columns.map((c) => {
      const sortKey = c.sortable === true ? c.key : c.sortable;
      const isSorted = sortKey && sort?.key === sortKey;
      return h('th', {
        style: { width: c.width, textAlign: c.align },
        class: [sortKey && 'af-sortable', isSorted && 'af-sorted', c.cls],
        title: c.title,
        onclick: sortKey ? () => onSort?.(sortKey, isSorted && sort.dir === 'asc' ? 'desc' : 'asc') : undefined,
      }, c.label, sortKey && icon(isSorted ? (sort.dir === 'asc' ? 'sort-up' : 'sort-down') : 'chevron-expand', 'af-sort-icon'));
    })));

  const tbody = h('tbody', rows.map((row) => {
    const key = rowKey(row);
    const tr = h('tr', {
      class: [onRowClick && 'af-row-click', selected.has(key) && 'af-row-selected', rowClass?.(row)],
      onclick: onRowClick ? (e) => {
        if (e.target.closest('a,button,input,.af-no-row-click')) return;
        onRowClick(row, e);
      } : undefined,
    },
    selectable && h('td.af-col-check', h('input.form-check-input', {
      type: 'checkbox', checked: selected.has(key), 'aria-label': 'Select row',
      onclick: (e) => {
        e.stopPropagation();
        e.target.checked ? selected.add(key) : selected.delete(key);
        tr.classList.toggle('af-row-selected', e.target.checked);
        headCheck.checked = allKeys.every((k) => selected.has(k));
        notify();
      },
    })),
    columns.map((c) => h('td', { style: { textAlign: c.align }, class: c.cls }, c.render ? c.render(row) : fmtCell(row[c.key]))));
    return tr;
  }));

  const table = h('table.af-table', { class: dense && 'af-table-dense' }, thead, tbody);
  return h('div.af-table-wrap', table);
}

function fmtCell(v) {
  if (v == null || v === '') return h('span.text-muted', '—');
  if (typeof v === 'boolean') return v ? icon('check-lg', 'text-success') : icon('dash', 'text-muted');
  if (typeof v === 'object') return h('code.af-inline-json', JSON.stringify(v));
  return String(v);
}

/** Pagination footer for limit/offset APIs. */
export function pager({ total, limit, offset, onChange, sizes = [25, 50, 100] }) {
  const pages = Math.max(1, Math.ceil(total / limit));
  const page = Math.floor(offset / limit) + 1;
  const go = (p) => onChange((Math.min(Math.max(p, 1), pages) - 1) * limit, limit);
  const from = total ? offset + 1 : 0;
  const to = Math.min(offset + limit, total);
  const nums = [];
  for (let p = Math.max(1, page - 2); p <= Math.min(pages, page + 2); p++) nums.push(p);
  return h('div.af-pager',
    h('span.af-pager-info', `${from.toLocaleString()}–${to.toLocaleString()} of ${total.toLocaleString()}`),
    h('div.af-pager-controls',
      h('select.form-select.form-select-sm.af-pager-size', {
        'aria-label': 'Rows per page',
        onchange: (e) => onChange(0, Number(e.target.value)),
      }, sizes.map((n) => h('option', { value: n, selected: n === limit }, `${n} / page`))),
      h('button.af-page-btn', { disabled: page <= 1, onclick: () => go(1), title: 'First' }, icon('chevron-double-left')),
      h('button.af-page-btn', { disabled: page <= 1, onclick: () => go(page - 1), title: 'Previous' }, icon('chevron-left')),
      nums.map((p) => h('button.af-page-btn', { class: p === page && 'active', onclick: () => go(p) }, String(p))),
      h('button.af-page-btn', { disabled: page >= pages, onclick: () => go(page + 1), title: 'Next' }, icon('chevron-right')),
      h('button.af-page-btn', { disabled: page >= pages, onclick: () => go(pages), title: 'Last' }, icon('chevron-double-right'))));
}

/** Sticky action bar shown when rows are selected. */
export function selectionBar(count, actions, onClear) {
  if (!count) return h('div.af-selection-bar.d-none');
  return h('div.af-selection-bar',
    h('span.fw-semibold', `${count} selected`),
    h('div.d-flex.gap-2.flex-wrap', actions),
    h('button.af-link-btn', { onclick: onClear }, 'Clear'));
}

// ------------------------------------------------------------------ auto refresh

const INTERVALS = [0, 5, 10, 30, 60];
/**
 * Auto-refresh control. onTick() is called every N seconds while the tab is visible.
 * Returns {el, stop, trigger}. Register stop with the page cleanup.
 */
export function refreshControl(onTick, { key = 'refresh', initial = 10 } = {}) {
  let seconds = prefs.get(`${key}:interval`, initial);
  let timer = null;
  let busy = false;
  const label = h('span');
  const btn = h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Refresh now (R)', onclick: () => tick(true) }, icon('arrow-clockwise'));
  const select = h('button.btn.btn-sm.btn-outline-secondary.af-refresh-select', {
    type: 'button',
    title: 'Auto-refresh interval',
    onclick: (e) => {
      e.stopPropagation();
      openMenu(select, INTERVALS.map((s) => ({
        label: s ? `Every ${s}s` : 'Off', icon: s === seconds ? 'check2' : 'blank',
        onClick: () => { seconds = s; prefs.set(`${key}:interval`, s); schedule(); renderLabel(); },
      })));
    },
  }, icon('stopwatch'), ' ', label);
  const renderLabel = () => mount(label, seconds ? `${seconds}s` : 'Off');
  async function tick(manual = false) {
    if (busy) return;
    if (!manual && document.hidden) return;
    busy = true;
    btn.classList.add('af-spinning');
    try { await onTick(); } finally {
      busy = false;
      btn.classList.remove('af-spinning');
    }
  }
  function schedule() {
    clearInterval(timer);
    if (seconds) timer = setInterval(() => tick(false), seconds * 1000);
  }
  renderLabel();
  schedule();
  const el = h('div.btn-group.af-refresh', btn, select);
  return { el, stop: () => clearInterval(timer), trigger: () => tick(true) };
}

export { clear, mount };
