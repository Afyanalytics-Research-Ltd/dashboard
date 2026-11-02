// DAG runs across every DAG: URL-synced filters, state counts, sorting, paging and row actions.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../core/router.js';
import {
  pageHeader, card, stateBadge, stateInfo, timeAgo, refreshControl, chips, searchInput, dataTable, pager,
  errorState, skeleton, menuButton, emptyState, RUN_STATES, copyButton,
} from '../core/ui.js';
import { fmtDate, fmtDuration, liveDuration, truncate, fromLocalInput, toLocalInput } from '../core/format.js';
import { clearRun, setRunState, editRunNote, deleteRun, triggerDag } from '../core/actions.js';

export const RANGES = [
  { value: '1h', label: '1h', hours: 1 },
  { value: '24h', label: '24h', hours: 24 },
  { value: '7d', label: '7d', hours: 24 * 7 },
  { value: '30d', label: '30d', hours: 24 * 30 },
  { value: '', label: 'All time' },
];
const RUN_TYPES = ['manual', 'scheduled', 'backfill', 'asset_triggered'];
const TYPE_ICONS = { manual: 'hand-index', scheduled: 'calendar-event', backfill: 'rewind', asset_triggered: 'database' };

export function rangeStart(since) {
  const r = RANGES.find((x) => x.value === since && x.hours);
  return r ? new Date(Date.now() - r.hours * 3600e3).toISOString() : null;
}

export function runTypeLabel(run) {
  return h('span.af-run-type', { title: run.triggered_by ? `triggered by ${run.triggered_by}${run.triggering_user_name ? ` (${run.triggering_user_name})` : ''}` : '' },
    icon(TYPE_ICONS[run.run_type] || 'play'), ' ', String(run.run_type || '').replace('_', ' '));
}

/** Row action menu shared with other run tables. */
export function runMenu(run, reload) {
  return menuButton(null, () => [
    { label: 'Open run', icon: 'box-arrow-in-right', href: urls.run(run.dag_id, run.dag_run_id) },
    { label: 'Open DAG', icon: 'diagram-3', href: urls.dag(run.dag_id) },
    { divider: true },
    { label: 'Clear (re-run)…', icon: 'arrow-counterclockwise', onClick: async () => { if (await clearRun(run.dag_id, run.dag_run_id)) reload(); } },
    { label: 'Mark success', icon: 'check-circle', disabled: run.state === 'success', onClick: async () => { if (await setRunState(run.dag_id, run.dag_run_id, 'success')) reload(); } },
    { label: 'Mark failed', icon: 'x-circle', disabled: run.state === 'failed', onClick: async () => { if (await setRunState(run.dag_id, run.dag_run_id, 'failed')) reload(); } },
    { label: 'Mark queued', icon: 'hourglass', disabled: run.state === 'queued', onClick: async () => { if (await setRunState(run.dag_id, run.dag_run_id, 'queued')) reload(); } },
    { label: 'Edit note…', icon: 'sticky', onClick: async () => { if (await editRunNote(run.dag_id, run)) reload(); } },
    { label: 'Trigger again with this conf…', icon: 'play', onClick: () => triggerDag(run.dag_id, { prefillConf: run.conf || {} }) },
    { divider: true },
    { label: 'Delete run…', icon: 'trash', danger: true, onClick: async () => { if (await deleteRun(run.dag_id, run.dag_run_id)) reload(); } },
  ], { title: 'Run actions' });
}

export default async function render(ctx) {
  const { el, signal, query } = ctx;
  const f = {
    dag: query.get('dag') || '',
    q: query.get('q') || '',
    state: query.getAll('state'),
    type: query.getAll('type'),
    // Deep links like ?state=running mean "all of them"; otherwise default to the last week.
    since: query.has('since') ? query.get('since') : (query.get('from') || query.get('to') || query.has('state') ? '' : '7d'),
    from: query.get('from') || '',
    to: query.get('to') || '',
    order: query.get('order') || '-run_after',
    limit: Number(query.get('limit')) || 50,
    offset: Number(query.get('offset')) || 0,
  };
  const sync = () => ctx.setQuery({
    dag: f.dag, q: f.q, state: f.state, type: f.type, since: f.since === '7d' ? null : (f.since || 'all'),
    from: f.from, to: f.to, order: f.order === '-run_after' ? null : f.order,
    limit: f.limit === 50 ? null : f.limit, offset: f.offset || null,
  });
  if (f.since === 'all') f.since = '';

  const countsEl = h('div');
  const tableEl = h('div', skeleton(10));
  const pagerEl = h('div');
  const refresh = refreshControl(() => load(), { key: 'runs', initial: 10 });
  ctx.cleanup(refresh.stop);

  const dagList = h('datalist#af-runs-dags');
  api.all('/dags', 'dags', { order_by: 'dag_id' }, { signal })
    .then((dags) => mount(dagList, dags.map((d) => h('option', { value: d.dag_id }))))
    .catch(() => {});

  const dagInput = h('input.form-control.form-control-sm', {
    list: 'af-runs-dags', placeholder: 'All DAGs', value: f.dag, 'aria-label': 'DAG',
    onchange: (e) => { f.dag = e.target.value.trim(); f.offset = 0; sync(); load(); },
  });
  const customRange = h('div.af-runs-custom', { class: f.since ? 'd-none' : '' },
    h('input.form-control.form-control-sm', {
      type: 'datetime-local', value: toLocalInput(f.from), title: 'From', 'aria-label': 'From',
      onchange: (e) => { f.from = fromLocalInput(e.target.value) || ''; f.offset = 0; sync(); load(); },
    }),
    h('span.text-muted.small', '→'),
    h('input.form-control.form-control-sm', {
      type: 'datetime-local', value: toLocalInput(f.to), title: 'To', 'aria-label': 'To',
      onchange: (e) => { f.to = fromLocalInput(e.target.value) || ''; f.offset = 0; sync(); load(); },
    }));

  mount(el,
    pageHeader({
      title: 'DAG runs', icon: 'play-circle',
      subtitle: 'Every run across every DAG. Filters are kept in the URL so you can share a view.',
      actions: [refresh.el],
    }),
    card({
      body: h('div.af-runs-filters',
        h('div.af-toolbar',
          searchInput({ placeholder: 'Run id contains…', value: f.q, onInput: (v) => { f.q = v; f.offset = 0; sync(); load(); } }),
          h('div.af-runs-dag', icon('diagram-3', 'text-muted'), dagInput, dagList),
          h('div.af-grow'),
          h('div.af-filter-group',
            h('span.small.text-muted', 'Run after'),
            chips(RANGES, f.since, (v) => {
              f.since = v; f.from = ''; f.to = ''; f.offset = 0;
              customRange.classList.toggle('d-none', !!v);
              sync(); load();
            })),
          customRange),
        h('div.af-toolbar',
          h('span.small.text-muted.fw-semibold', 'Type'),
          chips(RUN_TYPES.map((t) => ({ value: t, label: t.replace('_', ' ') })), f.type, (v) => { f.type = v; f.offset = 0; sync(); load(); }, { multi: true }),
          h('div.af-grow'),
          h('button.af-link-btn', {
            onclick: () => navigate(urls.runs()),
          }, icon('x-circle'), ' Reset filters'))),
    }),
    countsEl,
    card({ body: h('div', tableEl, pagerEl), flush: true }));

  function baseParams() {
    const p = {
      run_id_pattern: f.q ? `%${f.q}%` : null,
      run_type: f.type,
      run_after_gte: f.since ? rangeStart(f.since) : f.from || null,
      run_after_lte: f.since ? null : f.to || null,
    };
    return p;
  }
  const listPath = () => `/dags/${enc(f.dag || '~')}/dagRuns`;

  async function loadCounts() {
    const base = baseParams();
    const results = await Promise.all(RUN_STATES.map((st) =>
      api.get(listPath(), { ...base, state: st, limit: 1 }, { signal }).then((r) => r.total_entries).catch(() => null)));
    if (!ctx.active) return;
    const total = results.reduce((a, b) => a + (b || 0), 0);
    mount(countsEl, h('div.af-toolbar',
      h('span.small.text-muted.fw-semibold', 'State'),
      chips([
        ...RUN_STATES.map((st, i) => ({ value: st, label: stateInfo(st).label, color: stateInfo(st).color, count: results[i] ?? '?' })),
      ], f.state, (v) => { f.state = v; f.offset = 0; sync(); load(); }, { multi: true }),
      h('span.small.text-muted.ms-auto', `${total.toLocaleString()} runs match the other filters`)));
  }

  async function loadTable() {
    let page;
    try {
      page = await api.get(listPath(), { ...baseParams(), state: f.state, order_by: f.order, limit: f.limit, offset: f.offset }, { signal });
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(tableEl, errorState(err, load));
      mount(pagerEl);
      return;
    }
    if (!ctx.active) return;
    const sortKey = f.order.replace(/^-/, '');
    mount(tableEl, dataTable({
      rows: page.dag_runs,
      rowKey: (r) => `${r.dag_id}|${r.dag_run_id}`,
      sort: { key: sortKey, dir: f.order.startsWith('-') ? 'desc' : 'asc' },
      onSort: (key, dir) => { f.order = `${dir === 'desc' ? '-' : ''}${key}`; f.offset = 0; sync(); load(); },
      onRowClick: (r, e) => {
        const url = urls.run(r.dag_id, r.dag_run_id);
        if (e.metaKey || e.ctrlKey) window.open(url, '_blank'); else navigate(url);
      },
      rowClass: (r) => r.state === 'failed' && 'af-row-failed',
      empty: emptyState({ icon: 'play-circle', title: 'No runs match', message: 'Widen the time range or clear some filters.' }),
      columns: [
        { key: 'state', label: 'State', sortable: true, render: (r) => stateBadge(r.state), width: '120px' },
        {
          key: 'dag_id', label: 'DAG / run id', sortable: true,
          render: (r) => h('div.af-runs-cell',
            h('a.fw-semibold.af-runs-dagname', { href: urls.dag(r.dag_id) }, r.dag_display_name || r.dag_id),
            h('div.af-runs-id', h('a.af-mono.small', { href: urls.run(r.dag_id, r.dag_run_id), title: r.dag_run_id }, truncate(r.dag_run_id, 48)), copyButton(r.dag_run_id)),
            r.note ? h('div.af-runs-note', { title: r.note }, icon('sticky', 'me-1'), truncate(r.note, 60)) : null),
        },
        { key: 'run_type', label: 'Type', render: runTypeLabel },
        {
          key: 'run_after', label: 'Run after', sortable: true,
          render: (r) => h('span.af-nowrap', { title: `Run after ${fmtDate(r.run_after)}\nStarted ${fmtDate(r.start_date)}\nEnded ${fmtDate(r.end_date)}` }, timeAgo(r.run_after)),
        },
        {
          key: 'duration', label: 'Duration', sortable: true, align: 'right',
          render: (r) => h('span.af-nowrap', { class: r.state === 'running' && 'text-info fw-semibold' }, fmtDuration(liveDuration(r.start_date, r.end_date, r.duration))),
        },
        {
          key: 'conf', label: 'Conf', sortable: true,
          render: (r) => (r.conf && Object.keys(r.conf).length
            ? h('code.af-inline-json.af-runs-conf', { title: truncate(JSON.stringify(r.conf, null, 2), 1500) }, truncate(JSON.stringify(r.conf), 24))
            : h('span.text-muted', '—')),
        },
        { key: 'actions', label: '', align: 'right', render: (r) => runMenu(r, load), width: '48px' },
      ],
    }));
    mount(pagerEl, page.total_entries ? pager({
      total: page.total_entries, limit: f.limit, offset: f.offset,
      onChange: (offset, limit) => { f.offset = offset; f.limit = limit; sync(); load(); },
    }) : null);
  }

  async function load() {
    await Promise.all([loadCounts(), loadTable()]);
  }

  await load();
}
