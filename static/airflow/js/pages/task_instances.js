// Task instances across every DAG and run, with filters, state counts and bulk state changes.

import { h, icon, mount, debounce } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../core/router.js';
import {
  pageHeader, card, stateBadge, stateInfo, timeAgo, refreshControl, chips, searchInput, dataTable, pager,
  errorState, skeleton, emptyState, selectionBar, confirmDialog, toast, toastError, menuButton,
} from '../core/ui.js';
import { fmtDate, fmtDuration, liveDuration, truncate } from '../core/format.js';
import { markTask, clearTasks, editTaskNote } from '../core/actions.js';
import { RANGES, rangeStart } from './runs.js';

const COUNT_STATES = ['success', 'failed', 'running', 'queued', 'scheduled', 'up_for_retry', 'upstream_failed', 'skipped', 'deferred'];
const tiKey = (t) => `${t.dag_id}|${t.dag_run_id}|${t.task_id}|${t.map_index}`;

export default async function render(ctx) {
  const { el, signal, query } = ctx;
  const f = {
    dag: query.get('dag') || '',
    q: query.get('q') || '',
    state: query.getAll('state'),
    // Deep links like ?state=queued mean "all of them", not just the last day.
    since: query.has('since') ? (query.get('since') === 'all' ? '' : query.get('since')) : (query.has('state') ? '' : '24h'),
    pool: query.get('pool') || '',
    queue: query.get('queue') || '',
    executor: query.get('executor') || '',
    operator: query.get('operator') || '',
    dmin: query.get('dmin') || '',
    dmax: query.get('dmax') || '',
    order: query.get('order') || '-start_date',
    limit: Number(query.get('limit')) || 50,
    offset: Number(query.get('offset')) || 0,
  };
  const selected = new Set();
  let rowsByKey = new Map();
  const sync = () => ctx.setQuery({
    dag: f.dag, q: f.q, state: f.state, since: f.since === '24h' ? null : (f.since || 'all'),
    pool: f.pool, queue: f.queue, executor: f.executor, operator: f.operator, dmin: f.dmin, dmax: f.dmax,
    order: f.order === '-start_date' ? null : f.order, limit: f.limit === 50 ? null : f.limit, offset: f.offset || null,
  });
  const changed = () => { f.offset = 0; selected.clear(); sync(); load(); };

  const countsEl = h('div');
  const tableEl = h('div', skeleton(10));
  const pagerEl = h('div');
  const selEl = h('div');
  const refresh = refreshControl(() => load(), { key: 'task-instances', initial: 10 });
  ctx.cleanup(refresh.stop);

  const dagList = h('datalist#af-ti-dags');
  api.all('/dags', 'dags', { order_by: 'dag_id' }, { signal })
    .then((dags) => mount(dagList, dags.map((d) => h('option', { value: d.dag_id })))).catch(() => {});

  const textFilter = (key, label, ic) => h('div.af-ti-filter',
    h('label.small.text-muted', icon(ic, 'me-1'), label),
    h('input.form-control.form-control-sm', {
      value: f[key], placeholder: 'any',
      oninput: debounce((e) => { f[key] = e.target.value.trim(); changed(); }, 400),
    }));
  const advancedCount = () => ['pool', 'queue', 'executor', 'operator', 'dmin', 'dmax'].filter((k) => f[k]).length;
  const advanced = h('div.af-ti-advanced', { class: advancedCount() ? '' : 'd-none' },
    textFilter('pool', 'Pool', 'water'),
    textFilter('queue', 'Queue', 'inboxes'),
    textFilter('executor', 'Executor', 'cpu'),
    textFilter('operator', 'Operator', 'gear'),
    textFilter('dmin', 'Min duration (s)', 'stopwatch'),
    textFilter('dmax', 'Max duration (s)', 'stopwatch-fill'));

  mount(el,
    pageHeader({
      title: 'Task instances', icon: 'list-task',
      subtitle: 'Every task attempt across all DAGs and runs.',
      actions: [refresh.el],
    }),
    card({
      body: h('div.af-runs-filters',
        h('div.af-toolbar',
          searchInput({ placeholder: 'Task name contains…', value: f.q, onInput: (v) => { f.q = v; changed(); } }),
          h('div.af-runs-dag', icon('diagram-3', 'text-muted'), h('input.form-control.form-control-sm', {
            list: 'af-ti-dags', placeholder: 'All DAGs', value: f.dag, 'aria-label': 'DAG',
            onchange: (e) => { f.dag = e.target.value.trim(); changed(); },
          }), dagList),
          h('button.btn.btn-sm.btn-outline-secondary', {
            onclick: (e) => { advanced.classList.toggle('d-none'); e.currentTarget.classList.toggle('active'); },
          }, icon('funnel', 'me-1'), 'More filters', advancedCount() ? h('span.badge.text-bg-primary.ms-1', String(advancedCount())) : null),
          h('div.af-grow'),
          h('div.af-filter-group',
            h('span.small.text-muted', 'Run after'),
            chips(RANGES, f.since, (v) => { f.since = v; changed(); }))),
        advanced),
    }),
    countsEl,
    card({ body: h('div', tableEl, pagerEl), flush: true }),
    selEl);

  function params() {
    return {
      task_display_name_pattern: f.q ? `%${f.q}%` : null,
      run_after_gte: rangeStart(f.since),
      pool: f.pool ? [f.pool] : null,
      queue: f.queue ? [f.queue] : null,
      executor: f.executor ? [f.executor] : null,
      operator: f.operator ? [f.operator] : null,
      duration_gte: f.dmin || null,
      duration_lte: f.dmax || null,
    };
  }
  const listPath = () => `/dags/${enc(f.dag || '~')}/dagRuns/~/taskInstances`;

  async function loadCounts() {
    const base = params();
    const results = await Promise.all(COUNT_STATES.map((st) =>
      api.get(listPath(), { ...base, state: [st], limit: 1 }, { signal }).then((r) => r.total_entries).catch(() => null)));
    if (!ctx.active) return;
    // Always offer the main states; rarer ones only when they occur or are selected.
    const main = new Set(['success', 'failed', 'running', 'queued']);
    const opts = COUNT_STATES.map((st, i) => ({ value: st, label: stateInfo(st).label, color: stateInfo(st).color, count: results[i] ?? '?' }))
      .filter((o) => main.has(o.value) || o.count || f.state.includes(o.value));
    mount(countsEl, h('div.af-toolbar',
      h('span.small.text-muted.fw-semibold', 'State'),
      chips(opts, f.state, (v) => { f.state = v; changed(); }, { multi: true })));
  }

  function renderSelection() {
    const tis = [...selected].map((k) => rowsByKey.get(k)).filter(Boolean);
    const groups = groupByRun(tis);
    mount(selEl, selectionBar(tis.length, [
      ...['success', 'failed', 'skipped'].map((st) => h('button.btn.btn-sm.btn-light', { onclick: () => bulkMark(tis, st) },
        h('span.af-dot.me-1', { style: { background: stateInfo(st).color } }), `Mark ${st}`)),
      h('button.btn.btn-sm.btn-light', {
        disabled: groups.length !== 1,
        title: groups.length !== 1 ? 'Clearing works on one run at a time' : '',
        onclick: async () => {
          const [g] = groups;
          const res = await clearTasks(g.dag_id, { runId: g.dag_run_id, taskIds: g.tis.map((t) => t.task_id), mapIndexes: g.tis.map((t) => t.map_index) });
          if (res) { selected.clear(); load(); }
        },
      }, icon('arrow-counterclockwise', 'me-1'), 'Clear'),
    ], () => { selected.clear(); loadTable(); }));
  }

  async function bulkMark(tis, state) {
    const groups = groupByRun(tis);
    const ok = await confirmDialog({
      title: `Mark ${tis.length} task instance(s) ${state}?`, confirmLabel: `Mark ${state}`, danger: state === 'failed',
      message: `Across ${groups.length} run(s). Downstream tasks are not affected.`,
    });
    if (!ok) return;
    let failed = 0;
    for (const g of groups) {
      try {
        const res = await api.patch(`/dags/${enc(g.dag_id)}/dagRuns/${enc(g.dag_run_id)}/taskInstances`, {
          actions: [{
            action: 'update', update_mask: ['new_state'], action_on_non_existence: 'skip',
            entities: g.tis.map((t) => ({ task_id: t.task_id, map_index: t.map_index, new_state: state })),
          }],
        });
        failed += res?.update?.errors?.length || 0;
      } catch (err) {
        failed += g.tis.length;
        toastError(err, `${g.dag_id} / ${g.dag_run_id}`);
      }
    }
    toast(failed ? `${tis.length - failed} updated, ${failed} failed` : `${tis.length} task instance(s) marked ${state}`, failed ? 'warning' : 'success');
    selected.clear();
    load();
  }

  async function loadTable() {
    let page;
    try {
      page = await api.get(listPath(), { ...params(), state: f.state, order_by: f.order, limit: f.limit, offset: f.offset }, { signal });
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(tableEl, errorState(err, load));
      mount(pagerEl);
      return;
    }
    if (!ctx.active) return;
    rowsByKey = new Map(page.task_instances.map((t) => [tiKey(t), t]));
    for (const k of [...selected]) if (!rowsByKey.has(k)) selected.delete(k);
    const sortKey = f.order.replace(/^-/, '');
    mount(tableEl, dataTable({
      rows: page.task_instances,
      rowKey: tiKey,
      selectable: true,
      selected,
      onSelect: renderSelection,
      sort: { key: sortKey, dir: f.order.startsWith('-') ? 'desc' : 'asc' },
      onSort: (key, dir) => { f.order = `${dir === 'desc' ? '-' : ''}${key}`; f.offset = 0; sync(); load(); },
      onRowClick: (t, e) => {
        const url = urls.task(t.dag_id, t.dag_run_id, t.task_id, t.map_index);
        if (e.metaKey || e.ctrlKey) window.open(url, '_blank'); else navigate(url);
      },
      rowClass: (t) => t.state === 'failed' && 'af-row-failed',
      empty: emptyState({ icon: 'list-task', title: 'No task instances match', message: 'Widen the time range or clear some filters.' }),
      columns: [
        { key: 'state', label: 'State', sortable: true, render: (t) => stateBadge(t.state), width: '130px' },
        {
          key: 'task_id', label: 'Task',
          render: (t) => h('div',
            h('a.fw-semibold.af-mono', { href: urls.task(t.dag_id, t.dag_run_id, t.task_id, t.map_index) },
              t.task_display_name || t.task_id, t.map_index >= 0 ? h('span.text-muted', ` [${t.rendered_map_index || t.map_index}]`) : null),
            h('div.small.text-muted', t.operator_name || t.operator || '')),
        },
        {
          key: 'dag_id', label: 'DAG / run',
          render: (t) => h('div.af-runs-cell',
            h('a.af-runs-dagname', { href: urls.dag(t.dag_id) }, t.dag_display_name || t.dag_id),
            h('div', h('a.af-mono.small', { href: urls.run(t.dag_id, t.dag_run_id), title: t.dag_run_id }, truncate(t.dag_run_id, 40)))),
        },
        { key: 'try_number', label: 'Try', sortable: true, align: 'center', render: (t) => `${t.try_number}${t.max_tries ? `/${t.max_tries + 1}` : ''}` },
        { key: 'start_date', label: 'Started', sortable: true, render: (t) => h('span.af-nowrap', { title: fmtDate(t.start_date) }, timeAgo(t.start_date)) },
        {
          key: 'duration', label: 'Duration', sortable: true, align: 'right',
          render: (t) => h('span.af-nowrap', { class: t.state === 'running' && 'text-info fw-semibold' }, fmtDuration(liveDuration(t.start_date, t.end_date, t.duration))),
        },
        {
          key: 'actions', label: '', align: 'right', width: '48px',
          render: (t) => menuButton(null, () => [
            { label: 'Open task instance', icon: 'box-arrow-in-right', href: urls.task(t.dag_id, t.dag_run_id, t.task_id, t.map_index) },
            { label: 'Logs', icon: 'terminal', href: urls.task(t.dag_id, t.dag_run_id, t.task_id, t.map_index, { tab: 'logs' }) },
            { divider: true },
            { label: 'Mark state…', icon: 'check2-square', onClick: async () => { if (await markTask(t)) load(); } },
            { label: 'Clear…', icon: 'arrow-counterclockwise', onClick: async () => { if (await clearTasks(t.dag_id, { runId: t.dag_run_id, taskIds: [t.task_id], mapIndexes: [t.map_index] })) load(); } },
            { label: 'Edit note…', icon: 'sticky', onClick: async () => { if (await editTaskNote(t)) load(); } },
          ], { title: 'Task actions' }),
        },
      ],
    }));
    mount(pagerEl, page.total_entries ? pager({
      total: page.total_entries, limit: f.limit, offset: f.offset,
      onChange: (offset, limit) => { f.offset = offset; f.limit = limit; sync(); loadTable(); },
    }) : null);
    renderSelection();
  }

  async function load() {
    await Promise.all([loadCounts(), loadTable()]);
  }

  await load();
}

function groupByRun(tis) {
  const map = new Map();
  for (const t of tis) {
    const k = `${t.dag_id}|${t.dag_run_id}`;
    if (!map.has(k)) map.set(k, { dag_id: t.dag_id, dag_run_id: t.dag_run_id, tis: [] });
    map.get(k).tis.push(t);
  }
  return [...map.values()];
}
