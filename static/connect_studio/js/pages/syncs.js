// Syncs: mapping + destination + mode + schedule (Fivetran-style connections).

import { h, icon, mount } from '../../../ui/js/dom.js';
import { href, navigate } from '../../../ui/js/router.js';
import { drawer, toast, toastError } from '../../../ui/js/ui.js';
import { api } from '../api.js';
import { ago, compact, empty, logo, n, scheduleLabel, seg, statusBadge, statusDot, themed } from '../kit.js';

export const MODES = [
  { value: 'full_refresh_overwrite', icon: 'arrow-clockwise', title: 'Full refresh · overwrite', sub: 'Rebuild every destination table from scratch each run. Simple and always consistent.' },
  { value: 'full_refresh_append', icon: 'stack', title: 'Full refresh · append', sub: 'Read everything and append — keeps a history of every run.' },
  { value: 'incremental_append', icon: 'skip-forward', title: 'Incremental · append', sub: 'Only rows past the saved cursor (e.g. updated_at). Fast; may duplicate updated rows.' },
  { value: 'incremental_dedupe', icon: 'intersect', title: 'Incremental · deduped', sub: 'Incremental, then replace rows with the same primary key. The usual choice for big tables.' },
];
export const SCHEDULES = [{ value: 0, label: 'Manual' }, { value: 15, label: '15 min' }, { value: 60, label: 'Hourly' }, { value: 360, label: '6 h' }, { value: 1440, label: 'Daily' }];

export async function newSyncDrawer({ mappingId, onCreated } = {}) {
  const [meta, { mappings }, { sources }] = await Promise.all([api.meta(), api.mappings(), api.sources()]);
  const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
  const dests = sources.filter((s) => (specs[s.kind]?.roles || []).includes('destination'));
  const st = { mapping: mappings.find((m) => m.id === Number(mappingId)) || mappings.find((m) => m.status === 'approved') || mappings[0] || null,
    destination: dests[0] || null, mode: 'full_refresh_overwrite', schedule: 0, cursors: {}, rowLimit: '', runNow: true, name: '' };
  const body = h('div.cs-col.cs-gap-lg');
  const cursorBox = h('div');
  let mappingDetail = null;

  const loadCursors = async () => {
    if (!st.mapping || !st.mode.startsWith('incremental')) { mount(cursorBox); return; }
    mappingDetail = mappingDetail?.id === st.mapping.id ? mappingDetail : (await api.mapping(st.mapping.id, { snapshots: 1 })).mapping;
    const srcIdx = Object.fromEntries((mappingDetail.source_schema?.tables || []).map((t) => [t.key, t]));
    const pairs = mappingDetail.document.tables.filter((p) => p.source);
    mount(cursorBox, h('div.cs-field', h('label', 'Cursor column per table'),
      h('div.cs-picklist', pairs.map((p) => {
        const cols = srcIdx[p.source]?.columns || [];
        const guess = cols.find((c) => /^(updated_at|modified_at|last_modified|date_updated|updated|modified|created_at|date_created|created)$/i.test(c.name))?.name;
        return h('div.cs-pick', h('span.cs-mono.cs-grow', `${p.source} → ${p.target}`),
          h('select.cs-input.cs-input-sm', { style: { width: '190px' }, onchange: (e) => { if (e.target.value) st.cursors[p.target] = e.target.value; else delete st.cursors[p.target]; } },
            h('option', { value: '' }, guess ? `auto (${guess})` : 'auto'), cols.filter((c) => ['integer', 'datetime', 'date', 'decimal'].includes(c.family) || /id$|_at$/.test(c.name))
              .map((c) => h('option', { value: c.name, selected: st.cursors[p.target] === c.name }, c.name))));
      })), h('div.help', 'Rows whose cursor is past the last value seen are read. Auto picks updated_at / created_at / id.')));
  };

  const optionCards = () => h('div.cs-grid.cs-grid-2', MODES.map((md) => h('button.cs-option', { type: 'button', class: st.mode === md.value && 'on',
    onclick: (e) => { st.mode = md.value; e.currentTarget.parentElement.querySelectorAll('.cs-option').forEach((x) => x.classList.remove('on')); e.currentTarget.classList.add('on'); loadCursors(); } },
  h('span.cs-logo.cs-logo-sm', { style: { '--c': '#0072ce' } }, icon(md.icon)), h('div', h('div.cs-option-title', md.title), h('div.cs-option-sub', md.sub)))));

  const pick = (items, cur, render, onPick) => h('div.cs-col.cs-gap-sm', items.map((it) => h('button.cs-sel-card', { type: 'button', class: cur?.id === it.id && 'on',
    onclick: (e) => { onPick(it); e.currentTarget.parentElement.querySelectorAll('.cs-sel-card').forEach((x) => x.classList.remove('on')); e.currentTarget.classList.add('on'); } }, ...render(it), h('span.check', icon('check')))));

  mount(body,
    h('div.cs-field', h('label', 'Mapping (what to sync)'),
      mappings.length ? pick(mappings, st.mapping, (m) => [h('span.cs-logo.cs-logo-sm', { style: { '--c': '#7c3aed' } }, icon('stars')),
        h('div.cs-grow', h('div', { style: { fontWeight: 650 } }, m.name), h('div.cs-muted', { style: { fontSize: '12px' } }, `${n(m.summary?.mapped)} columns · ${n(m.summary?.approved)} approved · ${m.status}`))],
      (m) => { st.mapping = m; mappingDetail = null; loadCursors(); })
        : h('div.cs-callout.cs-callout-info', icon('info-circle'), h('div', 'No mappings yet — ', h('a', { href: href('/mappings/new') }, 'create one'), ' first.'))),
    h('div.cs-field', h('label', 'Destination (where rows land)', h('span.cs-grow'), h('a.cs-btn.cs-btn-sm.cs-btn-ghost', { href: href('/sources/new') }, icon('plus-lg'), 'Add')),
      dests.length ? pick(dests, st.destination, (s) => [logo(specs[s.kind], { size: 'sm' }), h('div.cs-grow', h('div', { style: { fontWeight: 650 } }, s.name), h('div.cs-muted', { style: { fontSize: '12px' } }, specs[s.kind]?.label))],
        (s) => { st.destination = s; })
        : h('div.cs-callout.cs-callout-warn', icon('exclamation-triangle'), h('div', 'No SQL destination yet. ', h('a', { href: href('/sources/new?kind=duckdb') }, 'Add DuckDB'), ', PostgreSQL, MySQL, Snowflake…'))),
    h('div.cs-field', h('label', 'Sync mode'), optionCards()),
    cursorBox,
    h('div.cs-form-grid',
      h('div.cs-field', h('label', 'Schedule'), seg(SCHEDULES, 0, (v) => { st.schedule = v; })),
      h('div.cs-field', h('label', 'Row limit per table (testing)'), h('input.cs-input', { type: 'number', min: 0, placeholder: 'no limit', oninput: (e) => { st.rowLimit = e.target.value; } }))),
    h('div.cs-field', h('label', 'Name'), h('input.cs-input', { placeholder: 'auto: source → destination', oninput: (e) => { st.name = e.target.value; } })),
    h('label.cs-switch', h('input', { type: 'checkbox', checked: true, onchange: (e) => { st.runNow = e.target.checked; } }), h('span.track'), h('span', 'Run the first sync now')));

  const createBtn = h('button.cs-btn.cs-btn-brand', { type: 'button', onclick: async () => {
    if (!st.mapping || !st.destination) { toast('Pick a mapping and a destination', 'warning'); return; }
    createBtn.disabled = true;
    try {
      const res = await api.createConnection({ mapping: st.mapping.id, destination: st.destination.id, mode: st.mode, schedule_minutes: st.schedule,
        name: st.name || undefined, run_now: st.runNow, settings: { cursors: st.cursors, row_limit: st.rowLimit ? Number(st.rowLimit) : null } });
      d.close();
      toast(st.runNow ? 'First sync started' : 'Sync created', 'success', { title: res.connection.name });
      onCreated?.(res.connection);
    } catch (e) { toastError(e, 'Could not create the sync'); createBtn.disabled = false; }
  } }, icon('arrow-repeat'), 'Create sync');
  const d = drawer({ title: 'New sync', subtitle: 'Move rows through a mapping into a destination', width: 760,
    body: themed(body, h('div.cs-wizard-foot', { style: { marginTop: '18px' } }, h('span.cs-grow'), createBtn)) });
  loadCursors();
  return d;
}

export default async function render(ctx) {
  ctx.setTitle('Syncs');
  const [{ connections }, meta] = await Promise.all([api.connections({ signal: ctx.signal }), api.meta()]);
  const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
  const open = () => newSyncDrawer({ mappingId: ctx.query.get('mapping'), onCreated: (c) => navigate(`/syncs/${c.id}`) });
  if (ctx.query.get('new')) { ctx.setQuery({ new: null, mapping: null }); setTimeout(open, 50); }

  const runNow = async (c, btn) => {
    btn.disabled = true;
    try { await api.runConnection(c.id); toast('Sync started', 'success', { title: c.name }); navigate(`/syncs/${c.id}`); } catch (e) { toastError(e, 'Could not start'); btn.disabled = false; }
  };

  const card = (c) => {
    const last = c.last_job;
    return h('div.cs-card.hoverable.cs-mcard', { onclick: (e) => { if (!e.target.closest('button, label, input')) navigate(`/syncs/${c.id}`); } },
      h('div.cs-row', h('div.cs-mcard-name.cs-grow.cs-ellipsis', c.name),
        h('label.cs-switch', { title: c.enabled ? 'Enabled' : 'Paused' }, h('input', { type: 'checkbox', checked: c.enabled, onchange: async (e) => {
          await api.updateConnection(c.id, { enabled: e.target.checked }); toast(e.target.checked ? 'Sync enabled' : 'Sync paused', 'info');
        } }), h('span.track'))),
      h('div.cs-mcard-flow',
        h('span.cs-flowchip', logo(specs[c.mapping.source_kind], { size: 'xs' }), h('span.cs-ellipsis', c.mapping.source)),
        h('span.cs-flow-arrow', h('span.dash'), icon('chevron-right')),
        h('span.cs-flowchip', logo(specs[c.destination.kind], { size: 'xs' }), h('span.cs-ellipsis', c.destination.name))),
      h('div.cs-row.cs-wrap', { style: { gap: '6px' } },
        h('span.cs-badge.cs-badge-info', icon('arrow-repeat'), c.mode_label),
        h('span.cs-badge.cs-badge-outline', icon('clock'), scheduleLabel(c.schedule_minutes)),
        c.next_run_at && c.enabled && h('span.cs-badge.cs-badge-outline', `next ${ago(c.next_run_at)}`)),
      h('div.cs-source-foot',
        c.running_job ? [h('span.cs-dot.run'), 'running…'] : last ? [statusDot(last.status), `${last.status} ${ago(last.finished_at || last.created_at)}`] : [h('span.cs-dot'), 'never run'],
        c.last_rows ? h('span.cs-badge.cs-badge-outline', `${compact(c.last_rows)} rows`) : null,
        h('span.cs-grow'),
        h('button.cs-btn.cs-btn-sm', { type: 'button', disabled: !!c.running_job, onclick: (e) => runNow(c, e.currentTarget) }, icon('play-fill'), 'Run')));
  };

  mount(ctx.el,
    h('div.cs-head',
      h('div', h('h1', 'Syncs', connections.length ? h('span.cs-badge', String(connections.length)) : null),
        h('div.cs-head-sub', 'Each sync reads its mapping’s source, applies the reviewed rules and writes into a destination — on demand or on a schedule.')),
      h('div.cs-head-actions', h('button.cs-btn.cs-btn-brand', { type: 'button', onclick: open }, icon('plus-lg'), 'New sync'))),
    connections.length ? h('div.cs-grid.cs-grid-auto', connections.map(card))
      : h('div.cs-card', empty({ icon: 'arrow-repeat', title: 'No syncs yet', message: 'Pick a reviewed mapping and a destination database. Full refresh or incremental, manual or scheduled.',
        action: h('button.cs-btn.cs-btn-brand.cs-btn-lg', { type: 'button', onclick: open }, icon('plus-lg'), 'New sync') })));
  ctx.every(8000, async () => {
    if (connections.some((c) => c.running_job)) navigate('/syncs', { replace: true });
  });
}
