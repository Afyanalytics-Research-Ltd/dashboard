// Shared helpers for Model Studio admin pages (teams, users, system, cleanup, explorer, project settings).

import { h, icon, mount, debounce } from '../../../../ui/js/dom.js';
import { modal, toast } from '../../../../ui/js/ui.js';
import { gql, collect } from '../../core/api.js';

// ------------------------------------------------------------------ cached context

let ctxCache = null;
/** viewer + organization + teams (entities the viewer can act on). Cached for 60 s. */
export async function adminContext({ fresh = false, signal } = {}) {
  if (!fresh && ctxCache && Date.now() - ctxCache.at < 60000) return ctxCache.value;
  const d = await gql(`query AdminCtx { viewer { id username name email admin entity photoUrl
      teams(first: 100) { edges { node { id name isTeam photoUrl memberCount projectCount storageBytes defaultAccess } } }
      organizations { id name orgType } } }`, {}, { signal });
  const v = d.viewer || {};
  const value = {
    viewer: v,
    org: (v.organizations || [])[0] || null,
    entities: (v.teams?.edges || []).map((e) => e.node),
  };
  ctxCache = { at: Date.now(), value };
  return value;
}
export const invalidateAdminContext = () => { ctxCache = null; };

/** Every project of every entity the viewer can see: [{entityName, name, id, totalRuns}]. */
export async function allProjects({ signal } = {}) {
  const { entities } = await adminContext({ signal });
  const lists = await Promise.all(entities.map((e) => collect((after) => gql(`query P($e: String!, $after: String) {
    projects(entityName: $e, first: 100, after: $after) { pageInfo { hasNextPage endCursor }
      edges { node { id name entityName totalRuns createdAt lastActive } } } }`, { e: e.name, after }, { signal }),
  (d) => d.projects, { max: 1000 }).catch(() => [])));
  return lists.flat();
}

// ------------------------------------------------------------------ async tasks (moveRuns / moveProjects)

/** Poll a W&B Task until it leaves PENDING/RUNNING. onProgress(task). */
export async function waitForTask(taskId, { onProgress, signal, timeoutMs = 10 * 60 * 1000 } = {}) {
  const started = Date.now();
  let delay = 700;
  for (;;) {
    const d = await gql('query T($id: ID!) { task(id: $id) { id name type state progress result updatedAt } }', { id: taskId }, { signal });
    const t = d.task;
    onProgress?.(t);
    if (!t || !['PENDING', 'RUNNING', 'QUEUED', 'STARTED'].includes(String(t.state).toUpperCase())) return t;
    if (Date.now() - started > timeoutMs) return t;
    await new Promise((r) => setTimeout(r, delay));
    delay = Math.min(delay * 1.3, 3000);
  }
}

// ------------------------------------------------------------------ dialogs

/**
 * Destructive confirm that requires typing a name (GitHub-style).
 * Resolves true only when the typed text matches exactly.
 */
export function typedConfirm({ title, message, expected, confirmLabel = 'Delete', details }) {
  return new Promise((resolve) => {
    let ok = false;
    const input = h('input.form-control.form-control-sm.ms-mono', { placeholder: expected, autocomplete: 'off', spellcheck: 'false', 'aria-label': `Type ${expected} to confirm` });
    const m = modal({
      title, icon: 'exclamation-triangle-fill', size: 'sm',
      body: h('div.msa-typed',
        h('p.mb-2', message),
        details,
        h('label.form-label.small.mt-2', 'Type ', h('code', expected), ' to confirm'),
        input),
      actions: [
        { label: 'Cancel' },
        { label: confirmLabel, variant: 'danger', icon: 'trash', onClick: (close) => { ok = true; resolve(true); close(); } },
      ],
      onClose: () => { if (!ok) resolve(false); },
    });
    const btn = m.el.querySelector('.af-modal-footer .btn-danger');
    btn.disabled = true;
    input.addEventListener('input', () => { btn.disabled = input.value.trim() !== expected; });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !btn.disabled) btn.click(); });
  });
}

/** Show a one-time secret (e.g. a new API key) with copy and an explicit acknowledgement. */
export function showSecretOnce({ title, secret, note }) {
  return new Promise((resolve) => {
    const field = h('input.form-control.ms-mono.msa-secret', { value: secret, readonly: true, onfocus: (e) => e.target.select() });
    const copy = h('button.btn.btn-sm.btn-outline-primary', {
      type: 'button',
      onclick: async () => {
        try { await navigator.clipboard.writeText(secret); } catch { field.select(); document.execCommand('copy'); }
        mount(copy, icon('check2', 'me-1'), 'Copied');
        copy.classList.replace('btn-outline-primary', 'btn-success');
      },
    }, icon('clipboard', 'me-1'), 'Copy');
    modal({
      title, icon: 'key-fill', persistent: true,
      body: h('div',
        h('div.af-callout.af-callout-warn.mb-3', icon('eye-slash'), h('span', note || 'This is the only time the full key is shown. Store it somewhere safe now.')),
        h('div.d-flex.gap-2', field, copy)),
      actions: [{ label: "I've stored it", variant: 'primary', onClick: (close) => { close(); resolve(); } }],
    });
    setTimeout(() => field.select(), 60);
  });
}

// ------------------------------------------------------------------ pickers

/** <select> of projects grouped by entity; onChange({entityName, name}). */
export function projectSelect(projects, value, onChange, { placeholder = 'Choose a project…', exclude } = {}) {
  const byEntity = new Map();
  for (const p of projects) {
    if (exclude && exclude(p)) continue;
    if (!byEntity.has(p.entityName)) byEntity.set(p.entityName, []);
    byEntity.get(p.entityName).push(p);
  }
  const key = (p) => `${p.entityName}/${p.name}`;
  const sel = h('select.form-select.form-select-sm.msa-project-select', { 'aria-label': 'Project' },
    h('option', { value: '' }, placeholder),
    [...byEntity.entries()].map(([ent, list]) => h('optgroup', { label: ent },
      list.sort((a, b) => a.name.localeCompare(b.name)).map((p) => h('option', { value: key(p), selected: value === key(p) }, `${p.name}  (${p.totalRuns ?? 0} runs)`)))));
  sel.addEventListener('change', () => {
    const p = projects.find((x) => key(x) === sel.value);
    onChange(p || null);
  });
  return sel;
}

/** Debounced user search input with a dropdown of matches; onPick(user). */
export function userSearch(onPick, { placeholder = 'Search users by name, username or email…', exclude = () => false } = {}) {
  const wrap = h('div.msa-usersearch');
  const input = h('input.form-control.form-control-sm', { type: 'search', placeholder, autocomplete: 'off' });
  const pop = h('div.msa-usersearch-pop');
  wrap.append(icon('person-plus', 'msa-usersearch-icon'), input, pop);
  let seq = 0;
  const run = debounce(async () => {
    const q = input.value.trim();
    const mine = ++seq;
    if (q.length < 1) { pop.classList.remove('open'); return; }
    try {
      const d = await gql('query U($q: String) { users(first: 8, query: $q) { edges { node { id username name email photoUrl deletedAt } } } }', { q });
      if (mine !== seq) return;
      const users = (d.users?.edges || []).map((e) => e.node).filter((u) => !u.deletedAt && !exclude(u));
      mount(pop, users.length ? users.map((u) => h('button.msa-usersearch-item', {
        type: 'button',
        onclick: () => { pop.classList.remove('open'); input.value = ''; onPick(u); },
      }, avatar(u, 22), h('span.msa-us-main', h('b', u.username || u.email), h('span.text-muted', u.username ? ` ${u.name || ''}` : ' · hasn’t signed in yet')),
      u.username && h('span.msa-us-email', u.email || ''))) : h('div.msa-usersearch-empty', 'No matching users'));
      pop.classList.add('open');
    } catch (err) { mount(pop, h('div.msa-usersearch-empty', err.message)); pop.classList.add('open'); }
  }, 200);
  input.addEventListener('input', run);
  input.addEventListener('blur', () => setTimeout(() => pop.classList.remove('open'), 180));
  return { el: wrap, focus: () => input.focus() };
}

// ------------------------------------------------------------------ bits

export function avatar(user, size = 28) {
  const name = user?.name || user?.username || '?';
  if (user?.photoUrl && !/gravatar\.com\/avatar\/[0-9a-f]*\?d=blank/.test(user.photoUrl)) {
    return h('img.msa-avatar', { src: user.photoUrl, alt: '', width: size, height: size, style: { width: `${size}px`, height: `${size}px` }, onerror: (e) => e.target.replaceWith(initials(name, size)) });
  }
  return initials(name, size);
}

function initials(name, size) {
  const parts = String(name).trim().split(/[\s._-]+/).filter(Boolean);
  const text = ((parts[0]?.[0] || '?') + (parts[1]?.[0] || '')).toUpperCase();
  let hsh = 0;
  for (const ch of name) hsh = (hsh * 31 + ch.charCodeAt(0)) | 0;
  const hue = Math.abs(hsh) % 360;
  return h('span.msa-avatar.msa-initials', { style: { width: `${size}px`, height: `${size}px`, fontSize: `${Math.round(size * 0.4)}px`, background: `hsl(${hue} 65% 92%)`, color: `hsl(${hue} 55% 32%)` } }, text);
}

export function pill(text, tone = 'muted', ic) {
  return h(`span.ms-pill.ms-pill-${tone}`, ic && icon(ic), text);
}

/** Small "W&B said no" helper: turns proxy/GraphQL errors into a friendly toast message. */
export function friendly(err) {
  const msg = err?.message || String(err);
  if (err?.status === 409) return msg;
  if (/not available through Model Studio/.test(msg)) return `${msg} (blocked by the dashboard for safety).`;
  if (/permission|not authorized|unauthor/i.test(msg)) return 'W&B refused: your account lacks permission for this.';
  return msg;
}

export function done(message) { toast(message, 'success', { timeout: 3000 }); }
