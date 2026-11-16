// Sync detail: live run, run history with per-table results, cursor state, settings.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { href, navigate } from '../../../ui/js/router.js';
import { confirmDialog, toast, toastError } from '../../../ui/js/ui.js';
import { api } from '../api.js';
import { ago, compact, fmtDuration, jobTracker, kv, logo, n, openJobDrawer, scheduleLabel, seg, statusBadge, statusDot, themed } from '../kit.js';
import { MODES, SCHEDULES } from './syncs.js';

export default async function render(ctx) {
  const id = Number(ctx.params.id);
  const meta = await api.meta();
  const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
  let { connection: c, jobs } = await api.connection(id, { signal: ctx.signal });
  ctx.setTitle(c.name);
  const head = h('div.cs-card');
  const live = h('div');
  const body = h('div.cs-grid.cs-grid-main');
  mount(ctx.el, head, live, body);

  async function reload() {
    ({ connection: c, jobs } = await api.connection(id));
    renderAll();
  }

  function renderHead() {
    mount(head, h('div.cs-card-body.cs-col.cs-gap-lg',
      h('div.cs-row.cs-wrap', { style: { gap: '14px' } },
        h('span.cs-logo.cs-logo-lg.cs-logo-solid', { style: { '--c': '#0bb99f' } }, icon('arrow-repeat')),
        h('div.cs-grow',
          h('div.cs-crumbs', h('a', { href: href('/syncs') }, 'Syncs'), icon('chevron-right'), 'Detail'),
          h('h1', { style: { fontFamily: 'var(--cs-display)', fontSize: '22px', fontWeight: 700, margin: 0 } }, c.name),
          h('div.cs-row.cs-wrap', { style: { marginTop: '8px', gap: '8px' } },
            h('a.cs-flowchip', { href: href(`/mappings/${c.mapping.id}`) }, logo(specs[c.mapping.source_kind], { size: 'xs' }), h('span.cs-ellipsis', c.mapping.source)),
            h('span.cs-flow-arrow', h('span.dash'), icon('chevron-right')),
            h('span.cs-flowchip', h('span.cs-logo.cs-logo-xs', { style: { '--c': '#7c3aed' } }, icon('stars')), h('span.cs-ellipsis', c.mapping.name)),
            h('span.cs-flow-arrow', h('span.dash'), icon('chevron-right')),
            h('a.cs-flowchip', { href: href(`/sources/${c.destination.id}`) }, logo(specs[c.destination.kind], { size: 'xs' }), h('span.cs-ellipsis', c.destination.name)))),
        h('div.cs-head-actions',
          h('label.cs-switch', h('input', { type: 'checkbox', checked: c.enabled, onchange: async (e) => { await api.updateConnection(id, { enabled: e.target.checked }); reload(); } }), h('span.track'), h('span', c.enabled ? 'Enabled' : 'Paused')),
          h('button.cs-btn.cs-btn-brand', { type: 'button', disabled: !!c.running_job, onclick: run }, icon('play-fill'), c.running_job ? 'Running…' : 'Run now'),
          h('button.cs-icon-btn', { type: 'button', title: 'Delete sync', onclick: remove }, icon('trash3')))),
      kv([['Mode', c.mode_label], ['Schedule', scheduleLabel(c.schedule_minutes)], ['Next run', c.enabled && c.next_run_at ? ago(c.next_run_at) : '—'],
        ['Last run', c.last_run_at ? ago(c.last_run_at) : 'never'], ['Last status', c.last_status || '—'], ['Rows last run', n(c.last_rows)]])));
  }

  function renderLive() {
    if (!c.running_job) { mount(live); return; }
    const t = jobTracker(c.running_job, { title: 'Sync running' });
    mount(live, h('div.cs-card.cs-card-pad', t.el));
    t.done.then((final) => {
      if (!ctx.active) return;
      toast(final.message || final.status, final.status === 'succeeded' ? 'success' : 'error', { title: `Sync ${final.status}` });
      reload();
    });
    ctx.cleanup(() => t.stop());
  }

  function renderBody() {
    const state = Object.entries(c.state || {});
    mount(body,
      h('div.cs-card',
        h('div.cs-card-head', h('div.cs-card-title', icon('clock-history'), 'Runs'), h('span.cs-grow'), h('span.cs-muted', { style: { fontSize: '12px' } }, `${jobs.length} shown`)),
        jobs.length ? h('div.cs-list', jobs.map((j) => runRow(j))) : h('div.cs-card-body.cs-muted', 'No runs yet — press Run now.')),
      h('div.cs-col.cs-gap-lg',
        h('div.cs-card',
          h('div.cs-card-head', h('div.cs-card-title', icon('sliders'), 'Settings')),
          h('div.cs-card-body.cs-col.cs-gap-lg',
            h('div.cs-field', h('label', 'Mode'), h('select.cs-input', { onchange: async (e) => { await api.updateConnection(id, { mode: e.target.value }); toast('Mode updated', 'success'); reload(); } },
              MODES.map((md) => h('option', { value: md.value, selected: md.value === c.mode }, md.title))),
            h('div.help', MODES.find((md) => md.value === c.mode)?.sub)),
            h('div.cs-field', h('label', 'Schedule'), seg(SCHEDULES, SCHEDULES.some((x) => x.value === c.schedule_minutes) ? c.schedule_minutes : null,
              async (v) => { await api.updateConnection(id, { schedule_minutes: v }); toast(`Schedule: ${scheduleLabel(v)}`, 'success'); reload(); })),
            h('div.cs-field', h('label', 'Row limit per table'), h('input.cs-input', { type: 'number', min: 0, value: c.settings?.row_limit ?? '', placeholder: 'no limit',
              onchange: async (e) => { await api.updateConnection(id, { settings: { row_limit: e.target.value ? Number(e.target.value) : null } }); toast('Saved', 'success'); } })))),
        h('div.cs-card',
          h('div.cs-card-head', h('div.cs-card-title', icon('bookmark'), 'Cursor state'), h('span.cs-grow'),
            state.length ? h('button.cs-btn.cs-btn-sm.cs-btn-danger', { type: 'button', onclick: async () => {
              if (!(await confirmDialog({ title: 'Reset cursors?', message: 'The next incremental run re-reads every row.', confirmLabel: 'Reset', danger: true }))) return;
              await api.updateConnection(id, { reset_state: true }); toast('Cursors reset', 'success'); reload();
            } }, icon('arrow-counterclockwise'), 'Reset') : null),
          state.length ? h('div.cs-list', state.map(([t, v]) => h('div.cs-li', icon('table', 'cs-muted'), h('div.cs-grow', h('div.cs-li-title.cs-mono', t), h('div.cs-li-sub', `cursor ${v.cursor_column}`)),
            h('span.cs-chip.mono', String(v.cursor ?? '—')))))
            : h('div.cs-card-body.cs-muted', { style: { fontSize: '13px' } }, c.mode.startsWith('incremental') ? 'Saved after the first incremental run.' : 'Full-refresh syncs keep no cursor.'))));
  }

  function runRow(j) {
    const detailBox = h('div');
    let open = false;
    const row = h('div', h('div.cs-li.click', { onclick: async () => {
      open = !open;
      if (!open) { mount(detailBox); return; }
      const { job } = await api.job(j.id, 0);
      const tables = job.result?.tables || [];
      mount(detailBox, h('div', { style: { padding: '0 18px 14px' } },
        tables.length ? h('table.cs-coltable', h('thead', h('tr', h('th', 'Target'), h('th', 'Source'), h('th', 'Rows'), h('th', 'Status'), h('th', 'Time'), h('th', 'Cursor'))),
          h('tbody', tables.map((t) => h('tr', h('td.cs-colname', t.target), h('td.cs-mono.cs-muted', t.source), h('td.cs-num', n(t.rows)),
            h('td', statusBadge(t.status === 'ok' ? 'succeeded' : t.status), t.error && h('div', { style: { color: 'var(--cs-bad)', fontSize: '11.5px', maxWidth: '320px' } }, t.error)),
            h('td.cs-num', t.seconds != null ? `${t.seconds}s` : '—'), h('td.cs-mono', { style: { fontSize: '11.5px' } }, t.cursor ?? '—')))))
          : h('div.cs-muted', job.error || job.message),
        h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', style: { marginTop: '8px' }, onclick: () => openJobDrawer(j) }, icon('terminal'), 'Full log')));
    } },
    statusDot(j.status),
    h('div.cs-grow', h('div.cs-li-title', `Run #${j.id}`, h('span.cs-muted', { style: { fontWeight: 400, marginLeft: '8px', fontSize: '12px' } }, j.trigger === 'schedule' ? 'scheduled' : 'manual')),
      h('div.cs-li-sub.cs-ellipsis', j.message || j.error || '')),
    statusBadge(j.status),
    h('span.cs-muted.cs-nowrap', { style: { fontSize: '12px', minWidth: '110px', textAlign: 'right' } }, `${ago(j.created_at)}${j.duration ? ` · ${fmtDuration(j.duration)}` : ''}`),
    icon('chevron-down', 'cs-faint')), detailBox);
    return row;
  }

  async function run() {
    try { await api.runConnection(id); toast('Sync started', 'success'); reload(); } catch (e) { toastError(e, 'Could not start'); }
  }
  async function remove() {
    if (!(await confirmDialog({ title: `Delete “${c.name}”?`, message: 'The destination tables are left as they are.', confirmLabel: 'Delete', danger: true }))) return;
    await api.deleteConnection(id); toast('Sync deleted', 'success'); navigate('/syncs');
  }

  function renderAll() { renderHead(); renderLive(); renderBody(); }
  renderAll();
}
