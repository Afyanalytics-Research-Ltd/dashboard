// One DAG run: live header, task table with bulk actions, graph, gantt, details,
// upstream asset events, required actions (HITL) and the audit trail.

import { h, icon, mount } from '../core/dom.js';
import { api, enc, airflowLink } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../core/router.js';
import {
  pageHeader, card, stateBadge, stateInfo, timeAgo, refreshControl, chips, searchInput, dataTable, tabs,
  errorState, skeleton, emptyState, selectionBar, menuButton, kv, jsonView, copyButton, toast, loadInto,
  TASK_STATES,
} from '../core/ui.js';
import { dagGraph, gantt, stateBar } from '../core/charts.js';
import { fmtDate, fmtShortDate, fmtDuration, liveDuration, truncate } from '../core/format.js';
import {
  clearRun, setRunState, editRunNote, deleteRun, triggerDag, bulkTaskState, clearTasks, markTask, editTaskNote, runPath,
} from '../core/actions.js';
import { runTypeLabel } from './runs.js';

const ACTIVE = new Set(['queued', 'running']);
const tiKey = (t) => `${t.task_id}|${t.map_index}`;

export default async function render(ctx) {
  const { el, signal, params } = ctx;
  const { dagId, runId } = params;
  const base = runPath(dagId, runId);

  let run;
  let tis = [];
  let tasks = null;
  let graph = null;
  let tab = ctx.query.get('tab') || 'tasks';
  const taskFilter = { q: '', states: [], sort: { key: 'start_date', dir: 'asc' } };
  const selected = new Set();

  const headerEl = h('div');
  const summaryEl = h('div');
  const tabsHost = h('div');
  const bodyEl = h('div');
  const selEl = h('div');

  const refresh = refreshControl(() => load(), { key: 'run', initial: 5 });
  ctx.cleanup(refresh.stop);
  ctx.setTitle(`${runId} · ${dagId}`);

  mount(el, headerEl, summaryEl, card({ body: h('div', tabsHost, bodyEl) }), selEl);
  mount(bodyEl, skeleton(8));

  async function load() {
    const prevState = run?.state;
    const [r, tiList] = await Promise.all([
      api.get(base, null, { signal }),
      api.all(`${base}/taskInstances`, 'task_instances', {}, { signal, max: 5000 }),
    ]);
    if (!ctx.active) return;
    run = r;
    tis = tiList;
    if (prevState && ACTIVE.has(prevState) && !ACTIVE.has(run.state)) {
      toast(`Run finished: ${run.state}`, run.state === 'success' ? 'success' : 'error', { title: dagId });
    }
    ctx.setTitle(`${run.state} · ${runId} · ${dagId}`);
    renderHeader();
    renderSummary();
    renderTab(true);
  }

  function renderHeader() {
    const af = airflowLink(`/dags/${enc(dagId)}/runs/${enc(runId)}`);
    mount(headerEl, pageHeader({
      crumbs: [
        { label: 'DAGs', href: urls.dags() },
        { label: run.dag_display_name || dagId, href: urls.dag(dagId) },
        { label: 'Runs', href: urls.dag(dagId, 'runs') },
        { label: truncate(runId, 40) },
      ],
      title: h('span.af-mono.af-run-title', runId),
      icon: 'play-circle',
      badges: [stateBadge(run.state, { size: 'lg' }), copyButton(runId)],
      subtitle: h('span.af-run-sub',
        runTypeLabel(run),
        h('span', icon('diagram-3', 'me-1'), h('a', { href: urls.dag(dagId) }, run.dag_display_name || dagId)),
        h('span', icon('clock', 'me-1'), 'run after ', timeAgo(run.run_after)),
        run.dag_versions?.length ? h('span', icon('git', 'me-1'), `v${run.dag_versions.map((v) => v.version_number).join(', v')}`) : null),
      actions: [
        h('button.btn.btn-sm.btn-outline-primary', {
          onclick: async () => { if (await clearRun(dagId, runId)) load(); },
        }, icon('arrow-counterclockwise', 'me-1'), 'Clear'),
        menuButton('Mark as', [
          { label: 'Success', icon: 'check-circle', disabled: run.state === 'success', onClick: async () => { if (await setRunState(dagId, runId, 'success')) load(); } },
          { label: 'Failed', icon: 'x-circle', disabled: run.state === 'failed', onClick: async () => { if (await setRunState(dagId, runId, 'failed')) load(); } },
          { label: 'Queued', icon: 'hourglass', disabled: run.state === 'queued', onClick: async () => { if (await setRunState(dagId, runId, 'queued')) load(); } },
        ], { icon: 'check2-square' }),
        menuButton(null, [
          { label: 'Edit note…', icon: 'sticky', onClick: async () => { if (await editRunNote(dagId, run)) load(); } },
          { label: 'Trigger again with this conf…', icon: 'play', onClick: () => triggerDag(dagId, { prefillConf: run.conf || {}, openRun: true }) },
          af && { label: 'Open in Airflow', icon: 'wind', href: af, external: true },
          { divider: true },
          { label: 'Delete run…', icon: 'trash', danger: true, onClick: async () => { if (await deleteRun(dagId, runId)) navigate(urls.dag(dagId, 'runs')); } },
        ].filter(Boolean)),
        refresh.el,
      ],
    }));
  }

  function renderSummary() {
    const counts = {};
    tis.forEach((t) => { const st = t.state || 'none'; counts[st] = (counts[st] || 0) + 1; });
    const done = tis.filter((t) => t.state && !['running', 'queued', 'scheduled', 'up_for_retry', 'up_for_reschedule', 'deferred', 'restarting'].includes(t.state)).length;
    const item = (label, value, cls) => h('div.af-run-fact', { class: cls }, h('div.af-run-fact-label', label), h('div.af-run-fact-value', value));
    mount(summaryEl, h('div.af-run-facts',
      item('Duration', h('span', { class: run.state === 'running' && 'text-info' }, fmtDuration(liveDuration(run.start_date, run.end_date, run.duration)))),
      item('Queued', timeAgo(run.queued_at)),
      item('Started', run.start_date ? h('span', { title: fmtDate(run.start_date) }, fmtShortDate(run.start_date)) : '—'),
      item('Ended', run.end_date ? h('span', { title: fmtDate(run.end_date) }, fmtShortDate(run.end_date)) : '—'),
      item('Triggered by', run.triggered_by ? `${run.triggered_by}${run.triggering_user_name ? ` · ${run.triggering_user_name}` : ''}` : '—'),
      item(`Tasks · ${done}/${tis.length} done`, h('div.af-run-progress', stateBar(counts, { height: 10 })), 'af-run-fact-wide'),
      run.note ? item('Note', h('span', icon('sticky', 'me-1'), truncate(run.note, 120)), 'af-run-fact-wide') : null));
  }

  // ---------------------------------------------------------------- tabs

  const TABS = () => [
    { id: 'tasks', label: 'Tasks', icon: 'list-task', count: tis.length },
    { id: 'graph', label: 'Graph', icon: 'diagram-3' },
    { id: 'gantt', label: 'Gantt', icon: 'bar-chart-steps' },
    { id: 'details', label: 'Details', icon: 'info-circle' },
    { id: 'assets', label: 'Asset events', icon: 'database' },
    { id: 'hitl', label: 'Required actions', icon: 'person-check' },
    { id: 'audit', label: 'Audit log', icon: 'journal-text' },
  ];
  let tabBar = null;

  function renderTab(isRefresh = false) {
    if (!tabBar) {
      tabBar = tabs(TABS(), tab, (id) => {
        tab = id;
        ctx.setQuery({ tab: id === 'tasks' ? null : id });
        selected.clear();
        renderTab();
      });
      mount(tabsHost, tabBar);
    } else if (isRefresh) {
      // keep the counts fresh without rebuilding the active tab state
      const fresh = tabs(TABS(), tab, (id) => { tab = id; ctx.setQuery({ tab: id === 'tasks' ? null : id }); selected.clear(); renderTab(); });
      tabBar.replaceWith(fresh);
      tabBar = fresh;
    }
    if (tab !== 'tasks') mount(selEl);
    if (tab !== 'graph') graph = null;
    // Static tabs only load once per visit; live tabs re-render on every refresh.
    if (isRefresh && ['assets', 'hitl', 'audit'].includes(tab) && bodyEl.dataset.tab === tab) return;
    bodyEl.dataset.tab = tab;
    ({ tasks: renderTasks, graph: renderGraph, gantt: renderGantt, details: renderDetails, assets: renderAssets, hitl: renderHitl, audit: renderAudit }[tab] || renderTasks)();
  }

  function renderTasks() {
    const counts = {};
    tis.forEach((t) => { counts[t.state || 'none'] = (counts[t.state || 'none'] || 0) + 1; });
    const q = taskFilter.q.toLowerCase();
    let rows = tis.filter((t) => (!taskFilter.states.length || taskFilter.states.includes(t.state || 'none'))
      && (!q || t.task_id.toLowerCase().includes(q) || (t.operator_name || '').toLowerCase().includes(q)));
    const { key, dir } = taskFilter.sort;
    const val = (t) => (key === 'duration' ? liveDuration(t.start_date, t.end_date, t.duration) ?? -1 : t[key] ?? '');
    rows = [...rows].sort((a, b) => {
      const va = val(a); const vb = val(b);
      const c = va < vb ? -1 : va > vb ? 1 : (a.task_id.localeCompare(b.task_id) || a.map_index - b.map_index);
      return dir === 'asc' ? c : -c;
    });
    const tableHost = h('div');
    const stateOpts = [...TASK_STATES, 'none'].filter((st) => counts[st])
      .map((st) => ({ value: st, label: stateInfo(st).label, color: stateInfo(st).color, count: counts[st] }));
    mount(bodyEl,
      h('div.af-toolbar.mb-3',
        searchInput({ placeholder: 'Filter tasks…', value: taskFilter.q, delay: 120, onInput: (v) => { taskFilter.q = v; renderTasks(); } }),
        stateOpts.length > 1 ? chips(stateOpts, taskFilter.states, (v) => { taskFilter.states = v; renderTasks(); }, { multi: true }) : null),
      tableHost);
    mount(tableHost, dataTable({
      rows,
      rowKey: tiKey,
      selectable: true,
      selected,
      onSelect: renderSelection,
      sort: taskFilter.sort,
      onSort: (k, d) => { taskFilter.sort = { key: k, dir: d }; renderTasks(); },
      onRowClick: (t, e) => {
        const url = urls.task(dagId, runId, t.task_id, t.map_index);
        if (e.metaKey || e.ctrlKey) window.open(url, '_blank'); else navigate(url);
      },
      rowClass: (t) => t.state === 'failed' && 'af-row-failed',
      empty: emptyState({ icon: 'list-task', title: tis.length ? 'No tasks match the filter' : 'No task instances yet', message: tis.length ? null : 'The scheduler has not created task instances for this run.' }),
      columns: [
        { key: 'state', label: 'State', sortable: true, render: (t) => stateBadge(t.state), width: '130px' },
        {
          key: 'task_id', label: 'Task', sortable: true,
          render: (t) => h('div',
            h('a.fw-semibold.af-mono', { href: urls.task(dagId, runId, t.task_id, t.map_index) }, t.task_display_name || t.task_id,
              t.map_index >= 0 ? h('span.text-muted', ` [${t.rendered_map_index || t.map_index}]`) : null),
            t.note ? h('div.small.text-muted', icon('sticky', 'me-1'), truncate(t.note, 60)) : null),
        },
        { key: 'operator_name', label: 'Operator', sortable: true, render: (t) => h('span.small', t.operator_name || t.operator || '—') },
        { key: 'try_number', label: 'Try', sortable: true, align: 'center', render: (t) => `${t.try_number}${t.max_tries ? `/${t.max_tries + 1}` : ''}` },
        {
          key: 'start_date', label: 'Started', sortable: true,
          render: (t) => h('span.af-nowrap.small', { title: `Queued ${fmtDate(t.queued_when)}\nStarted ${fmtDate(t.start_date)}\nEnded ${fmtDate(t.end_date)}` }, fmtDate(t.start_date)),
        },
        {
          key: 'duration', label: 'Duration', sortable: true, align: 'right',
          render: (t) => h('span.af-nowrap', { class: t.state === 'running' && 'text-info fw-semibold' }, fmtDuration(liveDuration(t.start_date, t.end_date, t.duration))),
        },
        { key: 'pool', label: 'Pool', render: (t) => h('span.small.text-muted.af-nowrap', { title: `queue: ${t.queue}` }, t.pool) },
        {
          key: 'actions', label: '', align: 'right', width: '48px',
          render: (t) => menuButton(null, () => [
            { label: 'Open', icon: 'box-arrow-in-right', href: urls.task(dagId, runId, t.task_id, t.map_index) },
            { label: 'Logs', icon: 'terminal', href: urls.task(dagId, runId, t.task_id, t.map_index, { tab: 'logs' }) },
            { divider: true },
            { label: 'Mark state…', icon: 'check2-square', onClick: async () => { if (await markTask(t)) load(); } },
            { label: 'Clear…', icon: 'arrow-counterclockwise', onClick: async () => { if (await clearTasks(dagId, { runId, taskIds: [t.task_id], mapIndexes: [t.map_index] })) load(); } },
            { label: 'Edit note…', icon: 'sticky', onClick: async () => { if (await editTaskNote(t)) load(); } },
          ], { title: 'Task actions' }),
        },
      ],
    }));
    renderSelection();
  }

  function renderSelection() {
    if (tab !== 'tasks') return;
    const byKey = new Map(tis.map((t) => [tiKey(t), t]));
    const chosen = [...selected].map((k) => byKey.get(k)).filter(Boolean);
    mount(selEl, selectionBar(chosen.length, [
      ...['success', 'failed', 'skipped'].map((st) => h('button.btn.btn-sm.btn-light', {
        onclick: async () => { if (await bulkTaskState(dagId, runId, chosen, st)) { selected.clear(); load(); } },
      }, h('span.af-dot.me-1', { style: { background: stateInfo(st).color } }), `Mark ${st}`)),
      h('button.btn.btn-sm.btn-light', {
        onclick: async () => {
          const res = await clearTasks(dagId, { runId, taskIds: chosen.map((t) => t.task_id), mapIndexes: chosen.map((t) => t.map_index) });
          if (res) { selected.clear(); load(); }
        },
      }, icon('arrow-counterclockwise', 'me-1'), 'Clear'),
    ], () => { selected.clear(); renderTasks(); }));
  }

  /** State per task id; mapped tasks show their "worst" instance state. */
  function stateMap() {
    const rank = ['failed', 'upstream_failed', 'up_for_retry', 'running', 'restarting', 'deferred', 'queued', 'scheduled', 'up_for_reschedule', 'skipped', 'removed', 'success'];
    const map = {};
    for (const t of tis) {
      const cur = map[t.task_id];
      if (!cur || rank.indexOf(t.state) < rank.indexOf(cur)) map[t.task_id] = t.state;
    }
    return map;
  }

  async function renderGraph() {
    if (graph) { graph.setStates(stateMap()); return; }
    if (!tasks) {
      mount(bodyEl, skeleton(8));
      try {
        tasks = (await api.get(`/dags/${enc(dagId)}/tasks`, null, { signal })).tasks;
      } catch (err) {
        if (err.name !== 'AbortError' && ctx.active) mount(bodyEl, errorState(err, renderGraph));
        return;
      }
      if (!ctx.active || tab !== 'graph') return;
    }
    const search = searchInput({ placeholder: 'Highlight task…', delay: 100, onInput: (v) => graph?.highlight(v) });
    graph = dagGraph(tasks, {
      states: stateMap(),
      height: 560,
      onNodeClick: (t) => {
        const inst = tis.filter((x) => x.task_id === t.task_id);
        navigate(urls.task(dagId, runId, t.task_id, inst.length === 1 ? inst[0].map_index : -1, inst.length > 1 ? { tab: 'mapped' } : undefined));
      },
    });
    const legendStates = [...new Set(Object.values(stateMap()))];
    mount(bodyEl,
      h('div.af-toolbar.mb-2', search, h('div.af-grow'),
        h('div.af-legend', legendStates.map((st) => h('span.af-legend-item', h('span.af-dot', { style: { background: stateInfo(st).color } }), stateInfo(st).label))),
        h('span.small.text-muted', 'Scroll to zoom · drag to pan · hover to trace dependencies')),
      graph.el);
  }

  function renderGantt() {
    mount(bodyEl,
      h('div.small.text-muted.mb-2', 'Grey = waiting in queue, colour = running. Times relative to the first task queued.'),
      gantt(tis, { onClick: (t) => navigate(urls.task(dagId, runId, t.task_id, t.map_index)) }));
  }

  function renderDetails() {
    const conf = run.conf || {};
    mount(bodyEl, h('div.af-grid-main-side',
      h('div.d-flex.flex-column.gap-3',
        kv([
          ['Run id', h('span.af-mono', run.dag_run_id, ' ', copyButton(run.dag_run_id))],
          ['State', stateBadge(run.state)],
          ['Run type', run.run_type],
          ['Triggered by', run.triggered_by],
          ['Triggering user', run.triggering_user_name],
          ['Logical date', fmtDate(run.logical_date)],
          ['Run after', fmtDate(run.run_after)],
          ['Queued at', fmtDate(run.queued_at)],
          ['Started', fmtDate(run.start_date)],
          ['Ended', fmtDate(run.end_date)],
          ['Duration', fmtDuration(liveDuration(run.start_date, run.end_date, run.duration))],
          ['Data interval', run.data_interval_start ? `${fmtDate(run.data_interval_start)} → ${fmtDate(run.data_interval_end)}` : null],
          ['Last scheduling decision', fmtDate(run.last_scheduling_decision)],
          ['Bundle version', run.bundle_version],
        ]),
        h('div',
          h('div.af-section-title', 'Conf', h('span.text-muted.small.ms-2', `${Object.keys(conf).length} key(s)`)),
          Object.keys(conf).length ? jsonView(conf, { open: 2 }) : h('div.text-muted.small', 'This run was triggered without conf.'))),
      h('div.d-flex.flex-column.gap-3',
        h('div',
          h('div.af-section-title', 'Note', h('button.af-link-btn.ms-auto', { onclick: async () => { if (await editRunNote(dagId, run)) load(); } }, icon('pencil'), ' Edit')),
          run.note ? h('div.af-note', run.note) : h('div.text-muted.small', 'No note.')),
        h('div',
          h('div.af-section-title', 'DAG versions'),
          (run.dag_versions || []).length ? h('div.af-list.af-bordered', run.dag_versions.map((v) => h('div.af-list-item',
            h('span.af-tag', `v${v.version_number}`),
            h('div.af-list-main', h('div.af-list-title', v.bundle_name), h('div.af-list-sub', `created ${fmtDate(v.created_at)}${v.bundle_version ? ` · ${v.bundle_version}` : ''}`)),
            v.bundle_url ? h('a', { href: v.bundle_url, target: '_blank', rel: 'noopener' }, icon('box-arrow-up-right')) : null))) : h('div.text-muted.small', '—')),
        h('div',
          h('div.af-section-title', 'Raw JSON'),
          jsonView(run, { open: 0 })))));
  }

  function renderAssets() {
    loadInto(bodyEl, async () => {
      const res = await api.get(`${base}/upstreamAssetEvents`, null, { signal });
      const events = res.asset_events || [];
      if (!events.length) return emptyState({ icon: 'database', title: 'No upstream asset events', message: 'This run was not triggered by asset updates.' });
      return dataTable({
        rows: events, rowKey: (e) => e.id,
        columns: [
          { key: 'asset', label: 'Asset', render: (e) => h('a', { href: urls.asset(e.asset_id) }, e.name || e.uri || `#${e.asset_id}`, h('div.small.text-muted.af-mono', e.uri)) },
          { key: 'source', label: 'Source', render: (e) => (e.source_dag_id ? h('a.small', { href: urls.run(e.source_dag_id, e.source_run_id) }, `${e.source_dag_id} / ${e.source_task_id || ''}`) : h('span.text-muted', 'external')) },
          { key: 'timestamp', label: 'When', render: (e) => timeAgo(e.timestamp) },
          { key: 'extra', label: 'Extra', render: (e) => (e.extra && Object.keys(e.extra).length ? h('code.af-inline-json', JSON.stringify(e.extra)) : '—') },
        ],
      });
    });
  }

  function renderHitl() {
    loadInto(bodyEl, async () => {
      const res = await api.get(`${base}/hitlDetails`, { limit: 100 }, { signal });
      const items = res.hitl_details || [];
      if (!items.length) return emptyState({ icon: 'person-check', title: 'No required actions', message: 'No human-in-the-loop task asked for input in this run.' });
      return dataTable({
        rows: items, rowKey: (x) => x.task_instance?.id || `${x.task_instance?.task_id}|${x.task_instance?.map_index}`,
        onRowClick: (x) => navigate(urls.task(dagId, runId, x.task_instance.task_id, x.task_instance.map_index, { tab: 'hitl' })),
        columns: [
          { key: 'response', label: 'Status', render: (x) => (x.response_received ? h('span.badge.text-bg-success', 'answered') : h('span.badge.text-bg-warning', 'waiting')) },
          { key: 'subject', label: 'Subject', render: (x) => h('div', h('div.fw-semibold', x.subject), x.body ? h('div.small.text-muted', truncate(x.body, 100)) : null) },
          { key: 'task', label: 'Task', render: (x) => h('span.af-mono.small', x.task_instance?.task_id, x.task_instance?.map_index >= 0 ? `[${x.task_instance.map_index}]` : '') },
          { key: 'options', label: 'Options', render: (x) => h('div.af-tags', (x.options || []).map((o) => h('span.af-tag', { class: (x.chosen_options || []).includes(o) && 'active' }, o))) },
          { key: 'created_at', label: 'Asked', render: (x) => timeAgo(x.created_at) },
          { key: 'responded', label: 'Responded', render: (x) => (x.responded_at ? h('span.small', timeAgo(x.responded_at), x.responded_by_user ? ` by ${x.responded_by_user.name || x.responded_by_user.id}` : '') : '—') },
        ],
      });
    });
  }

  function renderAudit() {
    loadInto(bodyEl, async () => {
      const res = await api.get('/eventLogs', { dag_id: dagId, run_id: runId, order_by: '-when', limit: 100 }, { signal });
      const logs = res.event_logs || [];
      if (!logs.length) return emptyState({ icon: 'journal-text', title: 'No audit events for this run' });
      return dataTable({
        rows: logs, rowKey: (x) => x.event_log_id, dense: true,
        columns: [
          { key: 'when', label: 'When', render: (x) => h('span.af-nowrap', { title: fmtDate(x.when) }, fmtDate(x.when)) },
          { key: 'event', label: 'Event', render: (x) => h('span.af-tag', x.event) },
          { key: 'task_id', label: 'Task', render: (x) => (x.task_id ? h('a.af-mono.small', { href: urls.task(dagId, runId, x.task_id, x.map_index ?? -1) }, x.task_id) : '—') },
          { key: 'try_number', label: 'Try', align: 'center' },
          { key: 'owner', label: 'Owner' },
          { key: 'extra', label: 'Extra', render: (x) => auditExtra(x.extra) },
        ],
      });
    });
  }

  try {
    await load();
  } catch (err) {
    if (err.name === 'AbortError' || !ctx.active) return;
    refresh.stop();
    mount(el, pageHeader({
      crumbs: [{ label: 'DAGs', href: urls.dags() }, { label: dagId, href: urls.dag(dagId) }, { label: runId }],
      title: runId, icon: 'play-circle',
    }), errorState(err, () => navigate(location.pathname + location.search, { replace: true })));
  }
}

/** Event-log "extra" is a JSON string; show it compactly with an expander. */
export function auditExtra(extra) {
  if (!extra) return h('span.text-muted', '—');
  let parsed = null;
  try { parsed = JSON.parse(extra); } catch { /* plain text */ }
  const short = truncate(parsed ? JSON.stringify(parsed) : extra, 70);
  if (!parsed || short.length < 70) return h('code.af-inline-json', { title: extra }, short);
  const box = h('div.af-audit-extra.d-none', jsonView(parsed, { open: 1 }));
  return h('div.af-no-row-click',
    h('button.af-link-btn.af-mono.small', { onclick: () => box.classList.toggle('d-none') }, short),
    box);
}
