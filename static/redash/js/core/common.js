// Shared Studio widgets and cached lookups.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { toast, toastError, openMenu } from '../../../ui/js/ui.js';
import { rd, enc } from './api.js';

// ------------------------------------------------------------------ cached lookups

const cache = new Map();
/** Memoise a loader for ttl ms (shared across pages). */
export function cached(key, loader, ttl = 60000) {
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < ttl) return hit.promise;
  const promise = loader().catch((err) => { cache.delete(key); throw err; });
  cache.set(key, { at: Date.now(), promise });
  return promise;
}
export const invalidate = (prefix) => { for (const k of cache.keys()) if (k.startsWith(prefix)) cache.delete(k); };

export const getDataSources = () => cached('ds', () => rd.get('/api/data_sources'));
export const getSession = () => cached('session', () => rd.get('/api/session'), 300000);
export const getOrgSettings = () => cached('orgsettings', () => rd.get('/api/settings/organization').then((r) => r.settings), 300000);
export const getSnippets = () => cached('snippets', () => rd.get('/api/query_snippets'));
export const getQueryTags = () => cached('qtags', () => rd.get('/api/queries/tags').then((r) => r.tags));
export const getDashboardTags = () => cached('dtags', () => rd.get('/api/dashboards/tags').then((r) => r.tags));

/** All dashboards with widgets, for "used in" lookups. */
export const getDashboardsFull = () => cached('dashfull', async () => {
  const list = await rd.all('/api/dashboards');
  return Promise.all(list.map((d) => rd.get(`/api/dashboards/${enc(d.id)}`).catch(() => ({ ...d, widgets: [] }))));
}, 120000);

/** Dashboards whose widgets use a query (or one visualization of it). */
export async function dashboardsUsing({ queryId, vizId }) {
  const dashes = await getDashboardsFull();
  return dashes.filter((d) => (d.widgets || []).some((w) => w.visualization
    && (vizId ? w.visualization.id === vizId : w.visualization.query?.id === queryId)));
}

// ------------------------------------------------------------------ widgets

/** Star toggle for queries ('queries') or dashboards ('dashboards'). */
export function favoriteStar(kind, obj, { onChange } = {}) {
  const btn = h('button.rd-fav', { type: 'button', 'aria-pressed': String(!!obj.is_favorite) });
  const draw = () => {
    mount(btn, icon(obj.is_favorite ? 'star-fill' : 'star'));
    btn.classList.toggle('on', !!obj.is_favorite);
    btn.title = obj.is_favorite ? 'Remove from favourites' : 'Add to favourites';
  };
  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    e.preventDefault();
    const next = !obj.is_favorite;
    obj.is_favorite = next;
    draw();
    try {
      if (next) await rd.post(`/api/${kind}/${enc(obj.id)}/favorite`, {});
      else await rd.del(`/api/${kind}/${enc(obj.id)}/favorite`);
      invalidate('favs');
      onChange?.(next);
      window.dispatchEvent(new CustomEvent('rd:favorites-changed'));
    } catch (err) {
      obj.is_favorite = !next;
      draw();
      toastError(err, 'Could not update favourite');
    }
  });
  draw();
  return btn;
}

/**
 * Click-to-edit text. onSave(newValue) must return a promise; reverts on failure.
 * Enter saves, Escape cancels.
 */
export function inlineEdit(value, onSave, { placeholder = 'Untitled', multiline = false, className = '', disabled = false } = {}) {
  const wrap = h('span.rd-inline', { class: className });
  let current = value;
  const view = () => {
    mount(wrap, h('span.rd-inline-text', { class: !current && 'empty', tabindex: disabled ? undefined : 0, title: disabled ? '' : 'Click to edit' }, current || placeholder),
      !disabled && icon('pencil', 'rd-inline-pencil'));
  };
  const edit = () => {
    if (disabled) return;
    const input = multiline
      ? h('textarea.form-control.form-control-sm.rd-inline-input', { rows: 3 }, current || '')
      : h('input.form-control.form-control-sm.rd-inline-input', { value: current || '', placeholder });
    mount(wrap, input);
    input.focus();
    input.select?.();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const next = input.value.trim();
      if (!save || next === (current || '')) { view(); return; }
      const prev = current;
      current = next;
      view();
      try { await onSave(next); } catch (err) { current = prev; view(); toastError(err, 'Could not save'); }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (!multiline || e.metaKey || e.ctrlKey)) { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
  };
  wrap.addEventListener('click', (e) => { if (!e.target.closest('input,textarea')) edit(); });
  wrap.addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target.classList.contains('rd-inline-text')) edit(); });
  view();
  return wrap;
}

/** Tag chips with add/remove; onSave(tags[]) persists. suggestions: string[] */
export function tagEditor(tags, onSave, { suggestions = [], readOnly = false } = {}) {
  let current = [...(tags || [])];
  const el = h('div.rd-tags');
  const persist = async (next) => {
    const prev = current;
    current = next;
    render();
    try { await onSave(next); } catch (err) { current = prev; render(); toastError(err, 'Could not save tags'); }
  };
  const render = () => {
    mount(el,
      current.map((t) => h('span.af-tag.rd-tag', icon('tag'), t,
        !readOnly && h('button.rd-tag-x', { type: 'button', title: `Remove ${t}`, onclick: () => persist(current.filter((x) => x !== t)) }, icon('x')))),
      !readOnly && h('button.rd-tag-add', {
        type: 'button',
        onclick: (e) => {
          const anchor = e.currentTarget;
          const input = h('input.form-control.form-control-sm.rd-tag-input', { placeholder: 'New tag…', list: 'rd-tag-suggest' });
          const dl = h('datalist#rd-tag-suggest', suggestions.filter((s) => !current.includes(s)).map((s) => h('option', { value: s })));
          anchor.replaceWith(input, dl);
          input.focus();
          // Re-rendering removes the focused input, which fires 'blur' synchronously;
          // without this guard that blur re-enters commit() mid-render.
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
  };
  render();
  return el;
}

export function userChip(user) {
  if (!user) return h('span.text-muted', '—');
  return h('span.rd-user', user.profile_image_url && h('img.rd-avatar', { src: user.profile_image_url, alt: '' }), user.name || user.email);
}

const DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
/** Human description of a Redash schedule {interval, time, day_of_week, until}. */
export function scheduleText(schedule) {
  if (!schedule || !schedule.interval) return 'Never';
  const s = Number(schedule.interval);
  let text;
  if (s < 3600) text = `Every ${s / 60} minute${s === 60 ? '' : 's'}`;
  else if (s < 86400) text = `Every ${s / 3600} hour${s === 3600 ? '' : 's'}`;
  else if (s < 604800) text = s === 86400 ? 'Every day' : `Every ${s / 86400} days`;
  else text = s === 604800 ? 'Every week' : `Every ${s / 604800} weeks`;
  if (schedule.day_of_week) text += ` on ${schedule.day_of_week}`;
  if (schedule.time) text += ` at ${schedule.time} UTC`;
  if (schedule.until) text += ` until ${schedule.until}`;
  return text;
}
export { DAY_NAMES };

export const dataSourceIcon = (type) => ({
  snowflake: 'snow', pg: 'database', postgres: 'database', mysql: 'database', bigquery: 'google', results: 'stack',
  json: 'braces', url: 'link-45deg', mongodb: 'database', mssql: 'database', redshift: 'database', athena: 'cloud', python: 'filetype-py',
}[type] || 'database');

/** Small "⋯" actions menu helper. */
export function moreMenu(items) {
  return h('button.btn.btn-sm.btn-outline-secondary', {
    type: 'button', title: 'More actions', 'aria-haspopup': 'menu',
    onclick: (e) => { e.stopPropagation(); openMenu(e.currentTarget, typeof items === 'function' ? items() : items); },
  }, icon('three-dots'));
}

export function confirmToast(message) { toast(message, 'success', { timeout: 2500 }); }
