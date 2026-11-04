// Automation Studio widgets built on the shared af-* toolkit.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { modal, toast, toastError, errorState, skeleton, spinner, confirmDialog } from '../../../ui/js/ui.js';
import { n8n, enc, featureName, pool, n8nLink } from './api.js';
import { statusInfo } from './execution.js';
import { urls } from './urls.js';
import { patchWorkflow } from './store.js';

export function execBadge(status, { size = '' } = {}) {
  const info = statusInfo(status);
  return h('span.n8s-badge', { class: [`n8s-st-${status || 'unknown'}`, size && `n8s-badge-${size}`], title: info.label },
    icon(info.icon, status === 'running' ? 'n8s-spin' : ''), h('span', info.label));
}

export function execDot(status) {
  return h('span.n8s-dot', { class: `n8s-st-${status || 'unknown'}`, title: statusInfo(status).label });
}

export function tagPills(tags, { onClick } = {}) {
  if (!tags?.length) return null;
  return h('span.n8s-tags', tags.map((t) => h(onClick ? 'button.n8s-tag' : 'span.n8s-tag', {
    type: onClick ? 'button' : undefined,
    style: { '--tag-hue': hue(t.name) },
    onclick: onClick ? (e) => { e.stopPropagation(); onClick(t); } : undefined,
  }, t.name)));
}

/** Stable hue (0-359) for a string, so a tag always gets the same colour. */
export function hue(text) {
  let x = 0;
  for (const ch of String(text)) x = (x * 31 + ch.charCodeAt(0)) >>> 0;
  return x % 360;
}

export function kbd(...keys) {
  return h('span.n8s-kbd', keys.map((k) => h('kbd', k)));
}

export const MOD = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl';

// ------------------------------------------------------------------ licence-gated features

export function licenseGate(err, { title, alternative } = {}) {
  const feature = featureName(err.license);
  return h('div.n8s-gate',
    h('div.n8s-gate-icon', icon('shield-lock')),
    h('div.n8s-gate-body',
      h('div.n8s-gate-title', title || `${feature} is not included in this n8n licence`),
      h('div.n8s-gate-msg', `n8n answered: “${err.message}”. Studio shows this feature as soon as the instance's licence enables `, h('code', err.license), '.'),
      alternative && h('div.n8s-gate-alt', icon('lightbulb', 'me-1'), alternative),
      h('div.n8s-gate-actions',
        h('a.btn.btn-sm.btn-outline-secondary', { href: n8nLink('/settings/usage'), target: '_blank', rel: 'noopener' }, icon('box-arrow-up-right', 'me-1'), 'Licence & usage in n8n'),
        h('a.btn.btn-sm.btn-link', { href: 'https://docs.n8n.io/hosting/community-edition-features/', target: '_blank', rel: 'noopener' }, 'Which features need a licence?'))));
}

/** Like ui.loadInto but renders a licence gate for feature-gated 403s. */
export async function gated(el, loader, opts = {}) {
  mount(el, skeleton(opts.rows || 5));
  try {
    mount(el, await loader());
  } catch (err) {
    if (err?.name === 'AbortError') return;
    if (err?.license) mount(el, licenseGate(err, opts));
    else mount(el, errorState(err, () => gated(el, loader, opts)));
  }
}

// ------------------------------------------------------------------ prompts

export function promptText({ title, label, value = '', placeholder, help, okLabel = 'Save', multiline = false, validate } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const input = multiline
      ? h('textarea.form-control.form-control-sm', { rows: 5, placeholder }, value)
      : h('input.form-control.form-control-sm', { value, placeholder, autofocus: true });
    const err = h('div.text-danger.small.mt-1');
    const m = modal({
      title, size: 'sm',
      body: h('form', { onsubmit: (e) => { e.preventDefault(); m.el.querySelector('.btn-primary')?.click(); } },
        label && h('label.form-label.small.fw-semibold', label), input, help && h('div.form-text.small', help), err),
      actions: [
        { label: 'Cancel' },
        {
          label: okLabel, variant: 'primary',
          onClick: (close) => {
            const v = input.value.trim();
            const problem = validate?.(v);
            if (problem) { mount(err, problem); return; }
            done = true; resolve(v); close();
          },
        },
      ],
      onClose: () => { if (!done) resolve(null); },
    });
    setTimeout(() => { input.focus(); input.select?.(); }, 40);
  });
}

export function pickFile(accept = '*/*') {
  return new Promise((resolve) => {
    const input = h('input', { type: 'file', accept, style: 'display:none' });
    input.addEventListener('change', () => { resolve(input.files[0] || null); input.remove(); });
    document.body.appendChild(input);
    input.click();
  });
}

export const readFileText = (file) => new Promise((resolve, reject) => {
  const r = new FileReader();
  r.onload = () => resolve(String(r.result));
  r.onerror = () => reject(r.error);
  r.readAsText(file);
});

// ------------------------------------------------------------------ bulk operations

/**
 * Run `run(item)` over items with a progress modal, per-item results and "retry failed".
 * Resolves {ok: [...], failed: [...]} when the modal closes.
 */
export function bulkRun({ title, items, label, run, concurrency = 4, verb = 'Processing' }) {
  return new Promise((resolve) => {
    const bar = h('div.progress-bar', { style: { width: '0%' } });
    const counter = h('span', `0 / ${items.length}`);
    const list = h('div.n8s-bulk-list');
    const rows = new Map(items.map((it) => {
      const row = h('div.n8s-bulk-row', h('span.n8s-bulk-state', spinner('sm')), h('span.n8s-bulk-label', label(it)), h('span.n8s-bulk-msg'));
      list.appendChild(row);
      return [it, row];
    }));
    let results = [];
    const m = modal({
      title, size: 'lg', icon: 'lightning-charge', persistent: true,
      body: h('div', h('div.d-flex.justify-content-between.small.mb-1', h('span', verb, '…'), counter), h('div.progress.mb-3', { style: 'height:6px' }, bar), list),
      actions: [{ label: 'Close', variant: 'primary', onClick: (close) => close() }],
      onClose: () => resolve({ ok: results.filter((r) => r.ok).map((r) => r.item), failed: results.filter((r) => !r.ok) }),
    });
    const go = async (subset) => {
      for (const it of subset) mount(rows.get(it).querySelector('.n8s-bulk-state'), spinner('sm'));
      const out = await pool(subset, async (it) => {
        const v = await run(it);
        const row = rows.get(it);
        mount(row.querySelector('.n8s-bulk-state'), icon('check-circle-fill', 'text-success'));
        mount(row.querySelector('.n8s-bulk-msg'));
        return v;
      }, {
        concurrency,
        onProgress: (done, total) => { bar.style.width = `${(done / total) * 100}%`; mount(counter, `${done} / ${total}`); },
      });
      for (const r of out) {
        if (!r.ok) {
          const row = rows.get(r.item);
          mount(row.querySelector('.n8s-bulk-state'), icon('x-circle-fill', 'text-danger'));
          mount(row.querySelector('.n8s-bulk-msg'), r.error?.message || String(r.error));
        }
      }
      results = [...results.filter((x) => !subset.includes(x.item)), ...out];
      const failed = results.filter((r) => !r.ok);
      const footer = m.el.querySelector('.af-modal-footer');
      footer.querySelector('.n8s-retry')?.remove();
      if (failed.length) {
        footer.prepend(h('button.btn.btn-sm.btn-outline-danger.n8s-retry', { type: 'button', onclick: () => go(failed.map((f) => f.item)) }, icon('arrow-clockwise', 'me-1'), `Retry ${failed.length} failed`));
      }
      mount(counter, `${results.filter((r) => r.ok).length} ok · ${failed.length} failed`);
    };
    go(items);
  });
}

// ------------------------------------------------------------------ workflow helpers

export function workflowLink(wf, { fallbackId } = {}) {
  const id = wf?.id ?? fallbackId;
  if (!id) return h('span.text-muted', '—');
  return h('a.n8s-wf-link', { href: urls.workflow(id), title: wf?.name || id }, wf?.name || h('span.af-mono', id));
}

/** Toggle switch that (de)activates a workflow through the API. */
export function activeSwitch(wf, { onChange, size = '' } = {}) {
  const input = h('input.form-check-input', { type: 'checkbox', role: 'switch', checked: !!wf.active, 'aria-label': 'Active', disabled: wf.isArchived });
  const wrap = h('label.form-check.form-switch.n8s-switch.m-0', { class: size, title: wf.isArchived ? 'Archived workflows cannot be activated' : wf.active ? 'Active — click to deactivate' : 'Inactive — click to activate' }, input);
  wrap.addEventListener('click', (e) => e.stopPropagation());
  input.addEventListener('change', async () => {
    const want = input.checked;
    input.disabled = true;
    try {
      const updated = await n8n.post(`/workflows/${enc(wf.id)}/${want ? 'activate' : 'deactivate'}`, {});
      Object.assign(wf, { active: updated.active, versionId: updated.versionId, activeVersionId: updated.activeVersionId });
      patchWorkflow(updated);
      input.checked = !!updated.active;
      toast(`${wf.name} ${updated.active ? 'activated' : 'deactivated'}`, 'success', { timeout: 2500 });
      onChange?.(updated);
    } catch (err) {
      input.checked = !want;
      toastError(err, want ? 'Could not activate' : 'Could not deactivate');
    } finally {
      input.disabled = !!wf.isArchived;
    }
  });
  return wrap;
}

export async function confirmDelete(what, name, { details } = {}) {
  return confirmDialog({ title: `Delete ${what}?`, message: h('span', 'This permanently deletes ', h('b', name), '.'), confirmLabel: 'Delete', danger: true, details });
}

/** Debounced auto-save indicator: "Saved · 3s ago", "Unsaved changes", "Saving…". */
export function saveIndicator() {
  const el = h('span.n8s-save-ind');
  const set = (state, extra) => {
    el.dataset.state = state;
    const map = { saved: ['check2-circle', 'Saved'], dirty: ['circle-fill', 'Unsaved changes'], saving: ['arrow-repeat', 'Saving…'], error: ['exclamation-triangle', 'Save failed'], draft: ['clock-history', 'Draft restored'] };
    const [ic, text] = map[state] || map.saved;
    mount(el, icon(ic, state === 'saving' ? 'n8s-spin' : ''), h('span', extra || text));
  };
  set('saved');
  return { el, set };
}

export function userLabel(u) {
  if (!u) return '—';
  const name = [u.firstName, u.lastName].filter(Boolean).join(' ');
  return name || u.email || u.id;
}

export function avatar(u, size = 24) {
  const text = (userLabel(u) || '?').split(/\s+/).map((p) => p[0]).join('').slice(0, 2).toUpperCase();
  return h('span.n8s-avatar', { style: { width: `${size}px`, height: `${size}px`, '--hue': hue(u?.email || u?.id || text) } }, text);
}

export function emptyHint(iconName, title, message, action) {
  return h('div.n8s-empty', h('div.n8s-empty-art', icon(iconName)), h('div.n8s-empty-title', title), message && h('div.n8s-empty-msg', message), action && h('div.mt-2', action));
}

/** Inline-editable title: click to edit, Enter to save, Esc to cancel. */
export function inlineEdit(value, onSave, { placeholder = 'Untitled', cls = '' } = {}) {
  const text = h('span.n8s-inline-text', { tabindex: 0, class: cls, title: 'Click to rename' }, value || placeholder);
  const wrap = h('span.n8s-inline', text, icon('pencil', 'n8s-inline-pencil'));
  const edit = () => {
    const input = h('input.form-control.form-control-sm.n8s-inline-input', { value: text.textContent });
    const finish = async (save) => {
      if (input.dataset.done) return;
      input.dataset.done = '1';
      const v = input.value.trim();
      if (save && v && v !== text.textContent) {
        try { await onSave(v); text.textContent = v; } catch (err) { toastError(err, 'Rename failed'); }
      }
      input.replaceWith(text);
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') finish(true); if (e.key === 'Escape') finish(false); });
    input.addEventListener('blur', () => finish(true));
    text.replaceWith(input);
    input.focus();
    input.select();
  };
  text.addEventListener('click', edit);
  text.addEventListener('keydown', (e) => { if (e.key === 'Enter') edit(); });
  wrap.setValue = (v) => { text.textContent = v || placeholder; };
  return wrap;
}

/** Async-searchable select over a list of {value, label, icon?, sub?}. */
export function comboBox({ options, value, onChange, placeholder = 'Search…', allowEmpty = true, emptyLabel = '— none —' }) {
  const btn = h('button.form-select.form-select-sm.n8s-combo-btn', { type: 'button' });
  let current = value;
  const label = () => {
    const opt = options.find((o) => String(o.value) === String(current));
    mount(btn, opt ? [opt.icon || null, h('span', opt.label)] : h('span.text-muted', current ? String(current) : emptyLabel));
  };
  label();
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const search = h('input.form-control.form-control-sm', { placeholder, autofocus: true });
    const listEl = h('div.n8s-combo-list');
    const pop = h('div.n8s-combo-pop', search, listEl);
    const render = () => {
      const q = search.value.toLowerCase();
      const opts = [allowEmpty ? { value: '', label: emptyLabel } : null, ...options].filter(Boolean)
        .filter((o) => !q || `${o.label} ${o.sub || ''} ${o.value}`.toLowerCase().includes(q)).slice(0, 200);
      mount(listEl, opts.map((o) => h('button.n8s-combo-item', {
        type: 'button', class: String(o.value) === String(current) && 'active',
        onclick: () => { current = o.value; label(); close(); onChange(o.value, o); },
      }, o.icon || null, h('span.flex-grow-1', o.label, o.sub && h('div.n8s-combo-sub', o.sub)))));
    };
    const close = () => { pop.remove(); document.removeEventListener('mousedown', outside, true); };
    const outside = (ev) => { if (!pop.contains(ev.target) && ev.target !== btn) close(); };
    search.addEventListener('input', debounce(render, 80));
    search.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') close();
      if (ev.key === 'Enter') { ev.preventDefault(); listEl.querySelector('.n8s-combo-item')?.click(); }
    });
    document.body.appendChild(pop);
    const r = btn.getBoundingClientRect();
    Object.assign(pop.style, { left: `${Math.min(r.left, innerWidth - 330)}px`, top: `${Math.min(r.bottom + 4, innerHeight - 320)}px`, minWidth: `${Math.max(r.width, 280)}px` });
    render();
    setTimeout(() => search.focus(), 10);
    setTimeout(() => document.addEventListener('mousedown', outside, true), 0);
  });
  return { el: btn, set: (v) => { current = v; label(); }, get: () => current };
}

