// DAG detail: header with controls + lazily loaded tabs
// (overview, grid, graph, runs, tasks, code, backfills, versions, audit, asset events).

import { h, icon, mount, prefs } from '../core/dom.js';
import { api, enc, airflowLink } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../core/router.js';
import {
  pageHeader, card, stateBadge, stateInfo, timeAgo, refreshControl, emptyState, errorState, chips, tabs,
  skeleton, dataTable, pager, selectionBar, searchInput, menuButton, confirmDialog, toast, toastError,
  kv, jsonView, codeBlock, copyButton, drawer, modal, buildForm, jsonEditor, RUN_STATES,
} from '../core/ui.js';
import { runBars, donut, legend, calendarHeatmap, gridMatrix, dagGraph, layoutGraph } from '../core/charts.js';
import { fmtDate, fmtDuration, fmtNumber, fmtPercent, fmtDay, liveDuration, truncate, toLocalInput } from '../core/format.js';
import {
  setPaused, setFavorite, triggerDag, reparseDag, deleteDag, clearRun, setRunState, editRunNote, deleteRun,
  clearTasks, runPath,
} from '../core/actions.js';

const TABS = [
  { id: 'overview', label: 'Overview', icon: 'info-circle' },
  { id: 'grid', label: 'Grid', icon: 'grid-3x3' },
  { id: 'graph', label: 'Graph', icon: 'diagram-2' },
  { id: 'runs', label: 'Runs', icon: 'play-circle' },
  { id: 'tasks', label: 'Tasks', icon: 'list-task' },
  { id: 'code', label: 'Code', icon: 'file-earmark-code' },
  { id: 'backfills', label: 'Backfills', icon: 'rewind-circle' },
  { id: 'versions', label: 'Versions', icon: 'clock-history' },
  { id: 'audit', label: 'Audit log', icon: 'journal-text' },
  { id: 'assets', label: 'Assets', icon: 'database' },
];

// Worst-first priority used to summarise mapped task instances.
const PRIORITY = ['failed', 'upstream_failed', 'up_for_retry', 'running', 'restarting', 'deferred', 'up_for_reschedule',
  'queued', 'scheduled', 'removed', 'skipped', 'success'];
const worst = (states) => states.reduce((a, b) => (PRIORITY.indexOf(b ?? 'removed') < PRIORITY.indexOf(a ?? 'removed') ? b : a));

/** {task_id -> {state, count, duration, try_number, map_index}} from a list of task instances. */
function summarise(tis) {
  const out = {};
  for (const t of tis) {
    const cur = out[t.task_id];
    if (!cur) out[t.task_id] = { state: t.state, states: [t.state], count: 1, duration: t.duration, try_number: t.try_number, map_index: t.map_index };
    else { cur.states.push(t.state); cur.count++; cur.state = worst(cur.states); cur.map_index = -1; }
  }
  return out;
}

/** Tasks ordered as the graph lays them out (layer, then position). */
function topoOrder(tasks) {
  const { nodes } = layoutGraph(tasks);
  return [...tasks].sort((a, b) => {
    const na = nodes.get(a.task_id); const nb = nodes.get(b.task_id);
    return na.layer - nb.layer || na.y - nb.y;
  }).map((t) => t.task_id);
}

const runTypeIcon = (t) => ({ manual: 'person', scheduled: 'calendar-event', backfill: 'rewind', asset_triggered: 'database' }[t] || 'question');

export default async function render(ctx) {
  const { el, signal, query, params } = ctx;
  const dagId = params.dagId;
  const base = `/dags/${enc(dagId)}`;
  let tab = TABS.some((t) => t.id === query.get('tab')) ? query.get('tab') : 'overview';
  let details;
  let tasksCache = null;
  let tabToken = 0;
  let tabRefresh = null;

  try {
    details = await api.get(`${base}/details`, null, { signal });
  } catch (err) {
    if (err.name === 'AbortError') return;
    mount(el, pageHeader({ title: dagId, icon: 'diagram-3', crumbs: [{ label: 'DAGs', href: urls.dags() }, { label: dagId }] }), errorState(err, () => navigate(urls.dag(dagId, tab))));
    return;
  }
  if (!ctx.active) return;
  ctx.setTitle(dagId);

  const getTasks = async () => {
    if (!tasksCache) tasksCache = (await api.get(`${base}/tasks`, null, { signal })).tasks;
    return tasksCache;
  };

  const headerEl = h('div');
  const bodyEl = h('div');
  const refresh = refreshControl(async () => {
    await reloadDetails();
    await tabRefresh?.();
  }, { key: 'dag', initial: 10 });
  ctx.cleanup(refresh.stop);

  const tabBar = tabs(TABS, tab, (id) => { tab = id; ctx.setQuery({ tab: id === 'overview' ? null : id }); showTab(); });
  mount(el, headerEl, card({ body: h('div', tabBar, bodyEl), cls: 'af-dag-tabs-card' }));
  renderHeader();
  showTab();

  async function reloadDetails() {
    try {
      details = await api.get(`${base}/details`, null, { signal });
      if (ctx.active) renderHeader();
    } catch { /* keep last header */ }
  }

  // ---------------------------------------------------------------- header

  function renderHeader() {
    const d = details;
    const pauseSwitch = h('label.af-dag-switch.form-switch', { class: d.is_paused ? 'paused' : 'active', title: d.is_paused ? 'Paused: the scheduler will not create runs' : 'Active: runs are scheduled' },
      h('input.form-check-input.m-0', {
        type: 'checkbox', role: 'switch', checked: !d.is_paused,
        onchange: async (e) => {
          const res = await setPaused(d, !e.target.checked);
          if (res) { details.is_paused = res.is_paused; renderHeader(); } else e.target.checked = !d.is_paused;
        },
      }),
      d.is_paused ? 'Paused' : 'Active');
    const fav = h('button.af-fav', {
      type: 'button', class: d.is_favorite && 'on', title: d.is_favorite ? 'Remove from favourites' : 'Add to favourites',
      style: { fontSize: '20px' },
      onclick: async () => { if (await setFavorite(dagId, !d.is_favorite)) { details.is_favorite = !d.is_favorite; renderHeader(); } },
    }, icon(d.is_favorite ? 'star-fill' : 'star'));
    const nativeLink = airflowLink(`/dags/${enc(dagId)}`);
    const menu = menuButton(null, () => [
      { label: 'Re-parse DAG file', icon: 'arrow-repeat', onClick: () => reparseDag(d) },
      { label: 'Copy DAG id', icon: 'clipboard', onClick: () => navigator.clipboard.writeText(dagId).then(() => toast('Copied', 'success', { timeout: 1500 })) },
      nativeLink && { label: 'Open in Airflow', icon: 'box-arrow-up-right', href: nativeLink, external: true },
      { label: 'Audit log', icon: 'journal-text', onClick: () => tabBar.querySelectorAll('.af-tab')[8].click() },
      { divider: true },
      { label: 'Delete DAG…', icon: 'trash', danger: true, onClick: async () => { if (await deleteDag(dagId)) navigate(urls.dags()); } },
    ].filter(Boolean), { title: 'More actions' });

    mount(headerEl, pageHeader({
      title: d.dag_display_name || dagId,
      icon: 'diagram-3',
      crumbs: [{ label: 'Pipelines', href: urls.home() }, { label: 'DAGs', href: urls.dags() }, { label: dagId }],
      badges: [
        fav,
        d.has_import_errors && h('a.badge.text-bg-danger', { href: urls.importErrors() }, icon('bug'), ' import error'),
        d.is_stale && h('span.badge.text-bg-secondary', 'stale'),
      ],
      subtitle: h('div',
        d.description && h('div', d.description),
        h('div.af-dag-meta',
          d.dag_display_name && d.dag_display_name !== dagId && h('span', icon('hash'), h('code', dagId), copyButton(dagId)),
          h('span', { title: d.timetable_description || '' }, icon('calendar-event'), d.timetable_summary ? h('code', d.timetable_summary) : 'No schedule',
            d.timetable_description && h('span', `· ${d.timetable_description}`)),
          d.next_dagrun_run_after && !d.is_paused && h('span', icon('skip-end'), 'Next ', timeAgo(d.next_dagrun_run_after)),
          d.owners?.length && h('span', icon('person'), d.owners.join(', ')),
          h('span', icon('file-earmark-code'), h('span.af-mono', d.relative_fileloc || d.fileloc)),
          d.last_parsed_time && h('span', icon('arrow-repeat'), 'Parsed ', timeAgo(d.last_parsed_time)),
          d.tags?.length ? h('span.af-tags', d.tags.map((t) => h('a.af-tag', { href: urls.dags({ tags: t.name }) }, t.name))) : null)),
      actions: [
        pauseSwitch,
        h('button.btn.btn-sm.btn-primary', { type: 'button', onclick: () => triggerDag(dagId, { details }) }, icon('play-fill', 'me-1'), 'Trigger'),
        refresh.el,
        menu,
      ],
    }));
  }

  // ---------------------------------------------------------------- tabs

  async function showTab() {
    const token = ++tabToken;
    tabRefresh = null;
    const pane = h('div');
    mount(bodyEl, pane);
    const live = () => ctx.active && token === tabToken;
    const renderers = { overview, grid, graph, runs, tasks: tasksTab, code, backfills, versions, audit, assets };
    try {
      tabRefresh = await renderers[tab](pane, live) || null;
    } catch (err) {
      if (err.name === 'AbortError' || !live()) return;
      mount(pane, errorState(err, showTab));
    }
  }

  // ---------------------------------------------------------------- overview

  async function overview(pane, live) {
    const chartEl = h('div', skeleton(4));
    const calEl = h('div', skeleton(3));
    const donutEl = h('div', skeleton(4));
    const d = details;
    const paramRows = Object.entries(d.params || {}).map(([name, p]) => ({
      name, value: p && typeof p === 'object' && 'value' in p ? p.value : p,
      description: p?.description, type: p?.schema?.type,
    }));
    mount(pane, h('div.d-flex.flex-column.gap-3',
      h('div.af-grid-main-side',
        card({ title: 'Recent run durations', icon: 'bar-chart', body: chartEl }),
        card({ title: 'All-time run states', icon: 'pie-chart', body: donutEl })),
      card({ title: 'Activity — last 12 months', icon: 'calendar3', body: calEl }),
      h('div.af-grid-main-side',
        card({
          title: 'Details', icon: 'card-list',
          body: kv([
            ['DAG id', h('span.af-mono', dagId)],
            ['Owners', d.owners?.join(', ')],
            ['Schedule', d.timetable_summary ? h('span', h('code', d.timetable_summary), ' — ', d.timetable_description) : d.timetable_description],
            ['Start date', fmtDate(d.start_date)],
            ['End date', d.end_date ? fmtDate(d.end_date) : null],
            ['Catchup', d.catchup ? 'Yes' : 'No'],
            ['Max active runs', d.max_active_runs],
            ['Max active tasks', d.max_active_tasks],
            ['Max consecutive failed runs', d.max_consecutive_failed_dag_runs || 'unlimited'],
            ['Run timeout', d.dag_run_timeout ? fmtDuration(typeof d.dag_run_timeout === 'object' ? d.dag_run_timeout.days * 86400 + d.dag_run_timeout.seconds : d.dag_run_timeout) : null],
            ['Timezone', d.timezone],
            ['Bundle', d.bundle_name && `${d.bundle_name}${d.bundle_version ? ` @ ${d.bundle_version}` : ''}`],
            ['Latest version', d.latest_dag_version ? `v${d.latest_dag_version.version_number} · ${fmtDate(d.latest_dag_version.created_at)}` : null],
            ['File', h('span.af-mono.small', d.fileloc), { wide: true }],
            ['Last parsed', d.last_parsed_time && h('span', fmtDate(d.last_parsed_time), d.last_parse_duration != null && h('span.text-muted', ` (${fmtDuration(d.last_parse_duration)})`))],
            ['Render templates as native', d.render_template_as_native_obj ? 'Yes' : 'No'],
            d.asset_expression && ['Asset schedule', jsonView(d.asset_expression, { open: 1 }), { wide: true }],
            d.default_args && Object.keys(d.default_args).length && ['Default args', jsonView(d.default_args, { open: 1 }), { wide: true }],
          ]),
        }),
        h('div.d-flex.flex-column.gap-3',
          card({
            title: 'Next run', icon: 'skip-end',
            body: d.is_paused
              ? emptyState({ icon: 'pause-circle', title: 'Paused', message: 'Unpause the DAG to resume scheduling.' })
              : d.next_dagrun_run_after
                ? kv([
                  ['Runs after', h('span', fmtDate(d.next_dagrun_run_after), ' · ', timeAgo(d.next_dagrun_run_after))],
                  ['Logical date', fmtDate(d.next_dagrun_logical_date)],
                  ['Data interval', d.next_dagrun_data_interval_start ? `${fmtDate(d.next_dagrun_data_interval_start)} → ${fmtDate(d.next_dagrun_data_interval_end)}` : null],
                ], { cols: 1 })
                : emptyState({ icon: 'calendar-x', title: 'Not scheduled', message: d.timetable_description || 'This DAG only runs when triggered.' }),
          }),
          paramRows.length ? card({
            title: 'Params', icon: 'sliders', flush: true,
            body: dataTable({
              dense: true, rows: paramRows, rowKey: (r) => r.name,
              columns: [
                { key: 'name', label: 'Param', render: (r) => h('div', h('span.af-mono', r.name), r.description && h('div.small.text-muted', truncate(r.description, 120))) },
                { key: 'type', label: 'Type', render: (r) => h('span.small', r.type ? String(r.type) : '—') },
                { key: 'value', label: 'Default', render: (r) => h('code.af-inline-json', JSON.stringify(r.value)) },
              ],
            }),
          }) : null)),
      d.doc_md && card({ title: 'Documentation', icon: 'book', body: h('div.af-doc', d.doc_md) })));

    const yearAgo = new Date(Date.now() - 365 * 86400e3).toISOString();
    const [runsRes, statsRes] = await Promise.allSettled([
      api.all(`${base}/dagRuns`, 'dag_runs', { run_after_gte: yearAgo, order_by: '-run_after' }, { max: 1000, signal }),
      api.get('/dagStats', { dag_ids: dagId }, { signal }),
    ]);
    if (!live()) return null;

    if (runsRes.status === 'fulfilled') {
      const runsList = runsRes.value;
      mount(chartEl, runsList.length
        ? h('div', runBars(runsList.slice(0, 50), { height: 170, onClick: (r) => navigate(urls.run(dagId, r.dag_run_id)) }),
          h('div.small.text-muted.mt-2', `Last ${Math.min(50, runsList.length)} runs, oldest → newest. Click a bar to open the run.`))
        : emptyState({ icon: 'bar-chart', title: 'No runs in the last year' }));
      const days = new Map();
      for (const r of runsList) {
        const k = fmtDay(r.run_after);
        const v = days.get(k) || { success: 0, failed: 0, other: 0 };
        if (r.state === 'success') v.success++; else if (r.state === 'failed') v.failed++; else v.other++;
        days.set(k, v);
      }
      mount(calEl, h('div', calendarHeatmap(days, {
        onDay: (day) => {
          const start = new Date(`${day}T00:00:00`);
          const end = new Date(start.getTime() + 86400e3);
          ctx.setQuery({ tab: 'runs', from: day, to: fmtDay(end) });
          tab = 'runs';
          tabBar.setActive('runs');
          showTab();
        },
      }), h('div.small.text-muted.mt-2', `${fmtNumber(runsList.length)} runs${runsList.length >= 1000 ? ' (latest 1,000)' : ''}. Green = all succeeded, orange/red = failures. Click a day to filter runs.`)));
    } else {
      mount(chartEl, errorState(runsRes.reason));
      mount(calEl, errorState(runsRes.reason));
    }

    if (statsRes.status === 'fulfilled') {
      const stats = statsRes.value.dags?.[0]?.stats || [];
      const segs = stats.map((x) => ({ label: stateInfo(x.state).label, value: x.count, color: stateInfo(x.state).color, state: x.state }));
      const total = segs.reduce((a, b) => a + b.value, 0);
      const succ = segs.find((x) => x.state === 'success')?.value || 0;
      const go = (seg) => { ctx.setQuery({ tab: 'runs', run_state: seg.state }); tab = 'runs'; tabBar.setActive('runs'); showTab(); };
      mount(donutEl, total
        ? h('div.af-overview-donut', donut(segs, { size: 160, centerLabel: fmtPercent(succ, total), centerSub: `of ${fmtNumber(total)} runs`, onClick: go }), legend(segs, { onClick: go }))
        : emptyState({ icon: 'pie-chart', title: 'No runs yet' }));
    } else mount(donutEl, errorState(statsRes.reason));

    return () => overview(pane, live);
  }

  // ---------------------------------------------------------------- grid

  async function grid(pane, live) {
    let n = prefs.get('dag:gridRuns', 25);
    const gridEl = h('div', skeleton(8));
    const legendEl = legend(['success', 'failed', 'running', 'queued', 'up_for_retry', 'upstream_failed', 'skipped', 'scheduled']
      .map((st) => ({ label: stateInfo(st).label, color: stateInfo(st).color })));
    mount(pane,
      h('div.af-tab-toolbar',
        h('span.small.text-muted.fw-semibold', 'Runs'),
        chips([10, 25, 50, 100].map((v) => ({ value: v, label: String(v) })), n, (v) => { n = v || 25; prefs.set('dag:gridRuns', n); load(); }),
        h('div.af-grow'),
        legendEl),
      gridEl);

    async function load() {
      const [runsPage, tasks] = await Promise.all([
        api.get(`${base}/dagRuns`, { limit: n, order_by: '-run_after' }, { signal }),
        getTasks(),
      ]);
      const runList = runsPage.dag_runs;
      if (!runList.length) { if (live()) mount(gridEl, emptyState({ icon: 'grid-3x3', title: 'No runs yet', message: 'Trigger the DAG to see its grid.' })); return; }
      const tis = [];
      let offset = 0;
      let total = Infinity;
      while (offset < total) {
        const page = await api.post('/dags/~/dagRuns/~/taskInstances/list', {
          dag_ids: [dagId], dag_run_ids: runList.map((r) => r.dag_run_id), page_limit: 100, page_offset: offset,
        }, { signal });
        tis.push(...page.task_instances);
        total = page.total_entries;
        offset += page.task_instances.length;
        if (!page.task_instances.length) break;
      }
      if (!live()) return;
      const byRun = new Map();
      for (const t of tis) {
        if (!byRun.has(t.dag_run_id)) byRun.set(t.dag_run_id, []);
        byRun.get(t.dag_run_id).push(t);
      }
      const summaries = new Map([...byRun].map(([r, list]) => [r, summarise(list)]));
      const known = new Set(tasks.map((t) => t.task_id));
      const extra = [...new Set(tis.map((t) => t.task_id))].filter((id) => !known.has(id)).sort();
      mount(gridEl, gridMatrix({
        runs: runList,
        taskIds: [...topoOrder(tasks), ...extra],
        cell: (runId, taskId) => summaries.get(runId)?.[taskId],
        onCell: (run, taskId) => {
          const c = summaries.get(run.dag_run_id)?.[taskId];
          navigate(urls.task(dagId, run.dag_run_id, taskId, c?.count > 1 ? -1 : c?.map_index));
        },
        onRun: (run) => navigate(urls.run(dagId, run.dag_run_id)),
      }), h('div.small.text-muted.mt-2', `${runList.length} most recent runs (oldest → newest) × ${tasks.length} tasks. Numbers mark mapped tasks.`));
    }
    await load();
    return load;
  }

  // ---------------------------------------------------------------- graph

  async function graph(pane, live) {
    const [tasks, runsPage] = await Promise.all([
      getTasks(),
      api.get(`${base}/dagRuns`, { limit: 25, order_by: '-run_after' }, { signal }),
    ]);
    if (!live()) return null;
    const runList = runsPage.dag_runs;
    let runId = query.get('run') && runList.some((r) => r.dag_run_id === query.get('run')) ? query.get('run') : runList[0]?.dag_run_id || '';
    // Size the canvas to the layout: small DAGs get a compact canvas, big ones fill the viewport.
    const natural = tasks.length ? layoutGraph(tasks).height + 140 : 300;
    const height = Math.round(Math.min(Math.max(300, natural), 720, Math.max(420, window.innerHeight - 320)));
    const g = dagGraph(tasks, { height, onNodeClick: (t) => openTaskDrawer(t.task_id, runId) });
    const runSelect = h('select.form-select.form-select-sm.af-select-inline', {
      'aria-label': 'Color by run',
      onchange: (e) => { runId = e.target.value; ctx.setQuery({ run: runId || null }); loadStates(); },
    },
    h('option', { value: '' }, 'No run (structure only)'),
    runList.map((r) => h('option', { value: r.dag_run_id, selected: r.dag_run_id === runId }, `${stateInfo(r.state).label} · ${fmtDate(r.run_after, { seconds: false })} · ${truncate(r.dag_run_id, 40)}`)));
    const runLink = h('span');
    mount(pane,
      h('div.af-tab-toolbar.af-graph-toolbar',
        searchInput({ placeholder: 'Highlight tasks…', onInput: (v) => g.highlight(v), shortcut: false }),
        h('span.small.text-muted.fw-semibold', 'Color by run'),
        runSelect,
        runLink,
        h('div.af-grow'),
        h('span.small.text-muted', `${tasks.length} tasks · scroll to zoom · drag to pan · hover to trace dependencies`)),
      g.el);

    async function loadStates() {
      mount(runLink, runId ? h('a.af-link-btn', { href: urls.run(dagId, runId) }, 'Open run →') : null);
      if (!runId) { g.setStates({}); return; }
      const tis = await api.all(`${runPath(dagId, runId)}/taskInstances`, 'task_instances', {}, { signal });
      if (!live()) return;
      const sum = summarise(tis);
      g.setStates(Object.fromEntries(Object.entries(sum).map(([k, v]) => [k, v.state])));
    }
    await loadStates();
    return loadStates;
  }

  async function openTaskDrawer(taskId, runId) {
    const dr = drawer({ title: h('span', icon('box', 'me-2'), taskId), subtitle: dagId, body: skeleton(8), width: 720 });
    try {
      const t = await api.get(`${base}/tasks/${enc(taskId)}`, null, { signal });
      if (!dr.isOpen()) return;
      const delay = t.retry_delay && typeof t.retry_delay === 'object' ? (t.retry_delay.days || 0) * 86400 + (t.retry_delay.seconds || 0) : t.retry_delay;
      mount(dr.body, h('div.d-flex.flex-column.gap-3',
        runId && h('div.d-flex.gap-2.flex-wrap',
          h('a.btn.btn-sm.btn-primary', { href: urls.task(dagId, runId, taskId) }, icon('box-arrow-in-right', 'me-1'), 'Open task instance in selected run'),
          h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => clearTasks(dagId, { runId, taskIds: [taskId] }) }, icon('arrow-counterclockwise', 'me-1'), 'Clear in this run…')),
        kv([
          ['Operator', h('span', t.operator_name, t.class_ref && h('div.small.text-muted.af-mono', `${t.class_ref.module_path}.${t.class_ref.class_name}`))],
          ['Owner', t.owner],
          ['Trigger rule', h('code', t.trigger_rule)],
          ['Retries', t.retries != null ? `${t.retries} (every ${fmtDuration(delay)}${t.retry_exponential_backoff ? ', exponential' : ''})` : null],
          ['Pool', t.pool_slots > 1 ? `${t.pool} (${t.pool_slots} slots)` : t.pool],
          ['Queue', t.queue],
          ['Priority weight', `${t.priority_weight} (${t.weight_rule})`],
          ['Execution timeout', t.execution_timeout ? JSON.stringify(t.execution_timeout) : null],
          ['Depends on past', t.depends_on_past ? 'Yes' : 'No'],
          ['Mapped', t.is_mapped ? 'Yes' : 'No'],
          ['Downstream', t.downstream_task_ids?.length ? h('div.af-tags', t.downstream_task_ids.map((x) => h('button.af-tag', { type: 'button', onclick: () => { dr.close(); openTaskDrawer(x, runId); } }, x))) : 'none', { wide: true }],
          ['Template fields', t.template_fields?.length ? t.template_fields.join(', ') : null, { wide: true }],
        ]),
        t.doc_md && h('div', h('div.fw-semibold.small.mb-1', 'Documentation'), h('div.af-doc', t.doc_md)),
        h('div', h('div.fw-semibold.small.mb-1', 'Full definition'), jsonView(t, { open: 1 }))));
    } catch (err) {
      if (err.name !== 'AbortError' && dr.isOpen()) mount(dr.body, errorState(err));
    }
  }

  // ---------------------------------------------------------------- runs

  async function runs(pane, live) {
    const q = new URLSearchParams(location.search);
    const st = {
      states: q.getAll('run_state'),
      type: q.get('run_type') || '',
      from: q.get('from') || '',
      to: q.get('to') || '',
      search: q.get('rq') || '',
      sort: q.get('rsort') || '-run_after',
      limit: Number(q.get('rlimit')) || 25,
      offset: Number(q.get('roffset')) || 0,
    };
    const selected = new Set();
    let rows = [];
    let total = 0;
    const tableEl = h('div', skeleton(8));
    const pagerEl = h('div');
    const selEl = h('div');
    const dateInput = (key, label) => h('label.d-flex.align-items-center.gap-1.small.text-muted.mb-0', label,
      h('input.form-control.form-control-sm', { type: 'date', value: st[key], onchange: (e) => set({ [key]: e.target.value }) }));

    mount(pane,
      h('div.af-tab-toolbar',
        searchInput({ placeholder: 'Run id contains…', value: st.search, onInput: (v) => set({ search: v }), shortcut: false }),
        chips(RUN_STATES.map((s) => ({ value: s, label: stateInfo(s).label, color: stateInfo(s).color })), st.states, (v) => set({ states: v }), { multi: true }),
        h('select.form-select.form-select-sm.af-select-inline', { 'aria-label': 'Run type', onchange: (e) => set({ type: e.target.value }) },
          [['', 'All run types'], ['manual', 'Manual'], ['scheduled', 'Scheduled'], ['backfill', 'Backfill'], ['asset_triggered', 'Asset triggered']]
            .map(([v, l]) => h('option', { value: v, selected: st.type === v }, l))),
        h('div.af-run-filters', dateInput('from', 'From'), dateInput('to', 'To')),
        (st.states.length || st.type || st.from || st.to || st.search) && h('button.af-link-btn', { onclick: () => set({ states: [], type: '', from: '', to: '', search: '' }) }, 'Reset')),
      h('div.af-card.overflow-hidden', tableEl, pagerEl),
      selEl);

    function set(patch, keepOffset = false) {
      Object.assign(st, patch);
      if (!keepOffset) st.offset = 0;
      ctx.setQuery({
        run_state: st.states, run_type: st.type, from: st.from, to: st.to, rq: st.search,
        rsort: st.sort === '-run_after' ? null : st.sort, rlimit: st.limit === 25 ? null : st.limit, roffset: st.offset || null,
      });
      selected.clear();
      // Filter changes rebuild the tab from the URL so the toolbar and refresh hook stay in sync.
      if ('states' in patch || 'type' in patch || 'from' in patch || 'to' in patch) { showTab(); return; }
      load();
    }

    async function load(quiet) {
      if (!quiet) mount(tableEl, skeleton(8));
      const page = await api.get(`${base}/dagRuns`, {
        limit: st.limit, offset: st.offset, order_by: st.sort,
        state: st.states, run_type: st.type || null,
        run_after_gte: st.from ? new Date(`${st.from}T00:00:00`).toISOString() : null,
        run_after_lte: st.to ? new Date(`${st.to}T23:59:59`).toISOString() : null,
        run_id_pattern: st.search || null,
      }, { signal });
      if (!live()) return;
      rows = page.dag_runs;
      total = page.total_entries;
      draw();
    }

    function rowMenu(r) {
      return menuButton(null, () => [
        { label: 'Open run', icon: 'box-arrow-in-right', href: urls.run(dagId, r.dag_run_id) },
        { label: 'Clear (re-run)…', icon: 'arrow-counterclockwise', onClick: async () => { if (await clearRun(dagId, r.dag_run_id)) load(true); } },
        { label: 'Mark success', icon: 'check-circle', onClick: async () => { if (await setRunState(dagId, r.dag_run_id, 'success')) load(true); } },
        { label: 'Mark failed', icon: 'x-circle', onClick: async () => { if (await setRunState(dagId, r.dag_run_id, 'failed')) load(true); } },
        { label: 'Mark queued', icon: 'hourglass-split', onClick: async () => { if (await setRunState(dagId, r.dag_run_id, 'queued')) load(true); } },
        { label: r.note ? 'Edit note…' : 'Add note…', icon: 'sticky', onClick: async () => { if (await editRunNote(dagId, r)) load(true); } },
        { label: 'Re-trigger with this conf…', icon: 'play', onClick: () => triggerDag(dagId, { details, prefillConf: r.conf || {} }) },
        { divider: true },
        { label: 'Delete run…', icon: 'trash', danger: true, onClick: async () => { if (await deleteRun(dagId, r.dag_run_id)) load(true); } },
      ], { title: 'Run actions' });
    }

    function draw() {
      const sortKey = st.sort.replace(/^-/, '');
      mount(tableEl, dataTable({
        rows, rowKey: (r) => r.dag_run_id, selectable: true, selected, onSelect: drawSelection,
        sort: { key: sortKey, dir: st.sort.startsWith('-') ? 'desc' : 'asc' },
        onSort: (key, dir) => set({ sort: dir === 'desc' ? `-${key}` : key }, true),
        onRowClick: (r) => navigate(urls.run(dagId, r.dag_run_id)),
        rowClass: (r) => r.state === 'failed' && 'af-row-failed',
        columns: [
          { key: 'state', label: 'State', sortable: 'state', render: (r) => stateBadge(r.state) },
          {
            key: 'dag_run_id', label: 'Run', sortable: 'dag_run_id',
            render: (r) => h('div.af-run-id-cell',
              h('a.af-mono.small.af-ellipsis', { href: urls.run(dagId, r.dag_run_id), title: r.dag_run_id }, r.dag_run_id),
              r.note && h('div.small.text-muted.af-ellipsis', { title: r.note }, icon('sticky', 'me-1'), truncate(r.note, 60))),
          },
          { key: 'run_type', label: 'Type', render: (r) => h('span.small.af-nowrap', { title: r.triggered_by ? `triggered by ${r.triggered_by}${r.triggering_user_name ? ` (${r.triggering_user_name})` : ''}` : '' }, icon(runTypeIcon(r.run_type), 'me-1'), r.run_type) },
          {
            key: 'run_after', label: 'Run after', sortable: 'run_after',
            render: (r) => h('span.small.af-nowrap', { title: `Run after ${fmtDate(r.run_after)}\nLogical date ${r.logical_date ? fmtDate(r.logical_date) : '—'}` }, fmtDate(r.run_after, { seconds: false })),
          },
          {
            key: 'duration', label: 'Duration', sortable: 'duration', align: 'right',
            render: (r) => h('div.small.af-nowrap',
              h('div.fw-semibold', fmtDuration(liveDuration(r.start_date, r.end_date, r.duration))),
              h('div.text-muted', r.start_date ? timeAgo(r.start_date) : 'not started')),
          },
          {
            key: 'conf', label: 'Conf',
            render: (r) => (r.conf && Object.keys(r.conf).length
              ? h('button.af-link-btn.af-mono.small.af-nowrap', {
                title: 'View conf', onclick: () => modal({ title: 'Run conf', subtitle: r.dag_run_id, size: 'lg', body: jsonView(r.conf, { open: 3 }) }),
              }, icon('braces', 'me-1'), `${Object.keys(r.conf).length} key${Object.keys(r.conf).length === 1 ? '' : 's'}`)
              : h('span.text-muted.small', '{}')),
          },
          { key: 'actions', label: '', align: 'right', render: rowMenu },
        ],
        empty: emptyState({ icon: 'play-circle', title: 'No runs match', message: 'Adjust the filters or trigger the DAG.' }),
      }));
      mount(pagerEl, total ? pager({ total, limit: st.limit, offset: st.offset, onChange: (offset, limit) => set({ offset, limit }, true) }) : null);
      drawSelection();
    }

    function drawSelection() {
      const ids = [...selected];
      mount(selEl, selectionBar(ids.length, [
        h('button.btn.btn-sm.btn-light', { onclick: () => bulk(ids, 'clear') }, icon('arrow-counterclockwise', 'me-1'), 'Clear failed'),
        h('button.btn.btn-sm.btn-light', { onclick: () => bulk(ids, 'success') }, icon('check-circle', 'me-1'), 'Mark success'),
        h('button.btn.btn-sm.btn-light', { onclick: () => bulk(ids, 'failed') }, icon('x-circle', 'me-1'), 'Mark failed'),
        h('button.btn.btn-sm.btn-danger', { onclick: () => bulk(ids, 'delete') }, icon('trash', 'me-1'), 'Delete'),
      ], () => { selected.clear(); draw(); }));
    }

    async function bulk(ids, action) {
      const labels = { clear: 'Clear failed tasks of', success: 'Mark success', failed: 'Mark failed', delete: 'Delete' };
      const ok = await confirmDialog({
        title: `${labels[action]} ${ids.length} run(s)?`, danger: action === 'delete' || action === 'failed',
        message: h('div.af-mono.small', ids.slice(0, 10).join('\n'), ids.length > 10 ? ` …and ${ids.length - 10} more` : ''),
        confirmLabel: labels[action],
      });
      if (!ok) return;
      const call = (id) => {
        if (action === 'delete') return api.del(runPath(dagId, id));
        if (action === 'clear') return api.post(`${runPath(dagId, id)}/clear`, { dry_run: false, only_failed: true });
        return api.patch(runPath(dagId, id), { state: action }, { params: { update_mask: 'state' } });
      };
      const results = await Promise.allSettled(ids.map(call));
      const failed = results.filter((r) => r.status === 'rejected');
      if (failed.length) toastError(failed[0].reason, `${failed.length} of ${ids.length} failed`);
      else toast(`${ids.length} run(s) updated`, 'success');
      selected.clear();
      load(true);
    }

    await load();
    return () => load(true);
  }

  // ---------------------------------------------------------------- tasks

  async function tasksTab(pane, live) {
    const tasks = await getTasks();
    if (!live()) return null;
    const { up, down } = layoutGraph(tasks);
    let filter = '';
    const tableEl = h('div');
    const draw = () => {
      const rows = tasks.filter((t) => !filter || t.task_id.toLowerCase().includes(filter) || (t.operator_name || '').toLowerCase().includes(filter));
      mount(tableEl, dataTable({
        rows, rowKey: (t) => t.task_id, dense: true,
        onRowClick: (t) => openTaskDrawer(t.task_id, query.get('run') || null),
        columns: [
          { key: 'task_id', label: 'Task', render: (t) => h('div', h('span.af-mono.fw-semibold', t.task_id), t.task_display_name !== t.task_id && h('div.small.text-muted', t.task_display_name)) },
          { key: 'operator_name', label: 'Operator', render: (t) => h('span.small', t.operator_name, t.is_mapped && h('span.badge.text-bg-info.ms-1', 'mapped')) },
          { key: 'trigger_rule', label: 'Trigger rule', render: (t) => h('code.small', t.trigger_rule) },
          { key: 'retries', label: 'Retries', align: 'right' },
          { key: 'pool', label: 'Pool', render: (t) => h('span.small', t.pool) },
          { key: 'queue', label: 'Queue', render: (t) => h('span.small', t.queue) },
          { key: 'up', label: 'Upstream', align: 'right', render: (t) => String(up.get(t.task_id)?.length || 0) },
          { key: 'down', label: 'Downstream', align: 'right', render: (t) => String(down.get(t.task_id)?.length || 0) },
          { key: 'depends_on_past', label: 'Depends on past', align: 'center' },
        ],
        empty: emptyState({ icon: 'list-task', title: 'No tasks match' }),
      }));
    };
    mount(pane,
      h('div.af-tab-toolbar',
        searchInput({ placeholder: 'Filter tasks or operators…', onInput: (v) => { filter = v.toLowerCase(); draw(); }, shortcut: false }),
        h('div.af-grow'),
        h('span.small.text-muted', `${tasks.length} tasks · click a row for the full definition`)),
      h('div.af-card.overflow-hidden', tableEl));
    draw();
    return null;
  }

  // ---------------------------------------------------------------- code

  async function code(pane, live) {
    const versionsRes = await api.get(`${base}/dagVersions`, { order_by: '-version_number', limit: 100 }, { signal });
    if (!live()) return null;
    const versionsList = versionsRes.dag_versions;
    let version = Number(query.get('version')) || versionsList[0]?.version_number || null;
    const srcEl = h('div', skeleton(12));
    const select = h('select.form-select.form-select-sm.af-select-inline', {
      'aria-label': 'DAG version',
      onchange: (e) => { version = Number(e.target.value); ctx.setQuery({ version: version === versionsList[0]?.version_number ? null : version }); load(); },
    }, versionsList.map((v, i) => h('option', { value: v.version_number, selected: v.version_number === version },
      `v${v.version_number}${i === 0 ? ' (latest)' : ''} · ${fmtDate(v.created_at, { seconds: false })}`)));
    let text = '';
    mount(pane,
      h('div.af-tab-toolbar',
        versionsList.length > 0 && h('span.small.text-muted.fw-semibold', 'Version'),
        versionsList.length > 0 && select,
        h('span.small.af-mono.text-muted', details.fileloc),
        h('div.af-grow'),
        h('button.btn.btn-sm.btn-outline-secondary', {
          onclick: () => {
            const a = h('a', { href: URL.createObjectURL(new Blob([text], { type: 'text/x-python' })), download: (details.relative_fileloc || `${dagId}.py`).split('/').pop() });
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 1000);
          },
        }, icon('download', 'me-1'), 'Download')),
      srcEl);
    async function load() {
      mount(srcEl, skeleton(12));
      const src = await api.get(`/dagSources/${enc(dagId)}`, version ? { version_number: version } : null, { signal });
      if (!live()) return;
      text = src.content || '';
      const lang = (details.fileloc || '').endsWith('.py') ? 'python' : 'text';
      mount(srcEl, codeBlock(text, { lang }));
    }
    await load();
    return null;
  }

  // ---------------------------------------------------------------- backfills

  async function backfills(pane, live) {
    const listEl = h('div', skeleton(4));
    mount(pane,
      h('div.af-tab-toolbar',
        h('div.small.text-muted', 'Backfills create runs for past logical dates. Every new backfill is previewed with a dry run first.'),
        h('div.af-grow'),
        h('button.btn.btn-sm.btn-primary', { onclick: createBackfill }, icon('plus-lg', 'me-1'), 'New backfill')),
      h('div.af-card.overflow-hidden', listEl));

    async function load() {
      const res = await api.get('/backfills', { dag_id: dagId, limit: 100, order_by: '-id' }, { signal });
      if (!live()) return;
      const status = (b) => (b.completed_at ? 'success' : b.is_paused ? 'paused' : 'running');
      mount(listEl, dataTable({
        rows: res.backfills, rowKey: (b) => b.id,
        columns: [
          { key: 'id', label: '#', render: (b) => h('span.af-mono', `#${b.id}`) },
          { key: 'status', label: 'Status', render: (b) => stateBadge(status(b)) },
          { key: 'range', label: 'Logical date range', render: (b) => h('span.small', `${fmtDate(b.from_date, { seconds: false })} → ${fmtDate(b.to_date, { seconds: false })}`) },
          { key: 'reprocess_behavior', label: 'Reprocess', render: (b) => h('code.small', b.reprocess_behavior) },
          { key: 'max_active_runs', label: 'Max active', align: 'right' },
          { key: 'dag_run_conf', label: 'Conf', render: (b) => (b.dag_run_conf && Object.keys(b.dag_run_conf).length ? h('code.af-inline-json', JSON.stringify(b.dag_run_conf)) : h('span.text-muted.small', '{}')) },
          { key: 'created_at', label: 'Created', render: (b) => h('span.small', timeAgo(b.created_at)) },
          { key: 'completed_at', label: 'Completed', render: (b) => h('span.small', b.completed_at ? timeAgo(b.completed_at) : '—') },
          {
            key: 'actions', label: '', align: 'right',
            render: (b) => (b.completed_at ? h('a.af-link-btn', { href: urls.dag(dagId, 'runs', { run_type: 'backfill' }) }, 'Runs →') : h('div.d-flex.gap-1.justify-content-end',
              h('button.btn.btn-sm.btn-outline-secondary', { title: b.is_paused ? 'Unpause' : 'Pause', onclick: () => act(b, b.is_paused ? 'unpause' : 'pause') }, icon(b.is_paused ? 'play' : 'pause')),
              h('button.btn.btn-sm.btn-outline-danger', { title: 'Cancel', onclick: () => act(b, 'cancel') }, icon('x-octagon')))),
          },
        ],
        empty: emptyState({ icon: 'rewind-circle', title: 'No backfills', message: 'Use “New backfill” to re-run this DAG for a past date range.' }),
      }));
    }

    async function act(b, action) {
      if (action === 'cancel' && !(await confirmDialog({ title: `Cancel backfill #${b.id}?`, message: 'Runs not yet started will not be created.', confirmLabel: 'Cancel backfill', danger: true }))) return;
      try {
        await api.put(`/backfills/${b.id}/${action}`, undefined);
        toast(`Backfill #${b.id} ${action === 'cancel' ? 'cancelled' : `${action}d`}`, 'success');
        load();
      } catch (err) { toastError(err, `Could not ${action} backfill`); }
    }

    function createBackfill() {
      const weekAgo = new Date(Date.now() - 7 * 86400e3);
      const form = buildForm([
        { name: 'from_date', label: 'From (logical date)', type: 'datetime', value: toLocalInput(weekAgo), required: true },
        { name: 'to_date', label: 'To (logical date)', type: 'datetime', value: toLocalInput(new Date()), required: true },
        {
          name: 'reprocess_behavior', label: 'Reprocess existing runs', type: 'select', value: 'none',
          options: [{ value: 'none', label: 'None — skip dates that already have a run' }, { value: 'failed', label: 'Failed — re-run dates whose run failed' }, { value: 'completed', label: 'Completed — re-run every existing date' }],
        },
        { name: 'max_active_runs', label: 'Max active runs', type: 'number', value: 10, min: 1 },
        { name: 'run_backwards', type: 'checkbox', checkLabel: 'Run backwards (newest date first)', value: false },
        { name: 'run_on_latest_version', type: 'checkbox', checkLabel: 'Run on the latest DAG version', value: true },
      ]);
      const conf = jsonEditor({}, { rows: 5 });
      const preview = h('div.af-preview', h('div.small.text-muted', 'Preview shows which logical dates would get a run.'));
      const body = () => ({ dag_id: dagId, ...form.values(), dag_run_conf: conf.value() || {} });
      const doPreview = async () => {
        const res = await api.post('/backfills/dry_run', body());
        mount(preview,
          h('div.small.fw-semibold.mb-2', `${fmtNumber(res.total_entries)} run(s) would be created`),
          res.backfills.length
            ? h('div.af-tags', res.backfills.slice(0, 200).map((b) => h('span.af-tag', fmtDate(b.logical_date, { seconds: false }))), res.backfills.length > 200 && h('span.small.text-muted', `…+${res.backfills.length - 200}`))
            : h('div.small.text-muted', 'Nothing to do for this range.'));
        return res;
      };
      modal({
        title: `New backfill — ${dagId}`, icon: 'rewind-circle', size: 'lg',
        body: h('div', form.el, h('div.af-form-row.mt-3', h('label.form-label.small.fw-semibold', 'Run conf (JSON)'), conf.el), preview),
        actions: [
          { label: 'Cancel' },
          { label: 'Preview (dry run)', icon: 'eye', onClick: doPreview },
          {
            label: 'Create backfill', variant: 'primary', icon: 'rewind',
            onClick: async (close) => {
              const dry = await doPreview();
              if (!dry.total_entries) throw new Error('The dry run found no dates to run — adjust the range or reprocess behaviour.');
              if (!(await confirmDialog({ title: `Create ${dry.total_entries} run(s)?`, message: `Backfill ${dagId} from ${fmtDate(body().from_date)} to ${fmtDate(body().to_date)}.`, confirmLabel: 'Create backfill' }))) return;
              const res = await api.post('/backfills', body());
              toast(`Backfill #${res.id} created`, 'success');
              close();
              load();
            },
          },
        ],
      });
    }

    await load();
    return load;
  }

  // ---------------------------------------------------------------- versions

  async function versions(pane, live) {
    const res = await api.get(`${base}/dagVersions`, { order_by: '-version_number', limit: 100 }, { signal });
    if (!live()) return null;
    mount(pane, h('div.af-card.overflow-hidden', dataTable({
      rows: res.dag_versions, rowKey: (v) => v.id,
      columns: [
        { key: 'version_number', label: 'Version', render: (v, i) => h('span.fw-semibold', `v${v.version_number}`) },
        { key: 'created_at', label: 'Created', render: (v) => h('span.small', fmtDate(v.created_at), ' · ', timeAgo(v.created_at)) },
        { key: 'bundle_name', label: 'Bundle' },
        { key: 'bundle_version', label: 'Bundle version', render: (v) => (v.bundle_version ? h('code.small', v.bundle_version) : h('span.text-muted', '—')) },
        { key: 'bundle_url', label: 'Source', render: (v) => (v.bundle_url ? h('a.small', { href: v.bundle_url, target: '_blank', rel: 'noopener' }, 'Open ', icon('box-arrow-up-right')) : h('span.text-muted', '—')) },
        { key: 'id', label: 'Id', render: (v) => h('span.af-mono.small.text-muted', v.id) },
        {
          key: 'actions', label: '', align: 'right',
          render: (v) => h('button.btn.btn-sm.btn-outline-secondary', {
            onclick: () => { ctx.setQuery({ tab: 'code', version: v.version_number }); tab = 'code'; tabBar.setActive('code'); showTab(); },
          }, icon('file-earmark-code', 'me-1'), 'View code'),
        },
      ],
      empty: emptyState({ icon: 'clock-history', title: 'No versions recorded' }),
    })), h('div.small.text-muted.mt-2', `${fmtNumber(res.total_entries)} version(s). A new version is recorded whenever the parsed DAG structure changes.`));
    return null;
  }

  // ---------------------------------------------------------------- audit

  async function audit(pane, live) {
    let limit = 25;
    let offset = 0;
    let event = '';
    const tableEl = h('div', skeleton(6));
    const pagerEl = h('div');
    mount(pane,
      h('div.af-tab-toolbar',
        searchInput({ placeholder: 'Event contains…', onInput: (v) => { event = v; offset = 0; load(); }, shortcut: false }),
        h('div.af-grow'),
        h('a.af-link-btn', { href: urls.audit({ dag_id: dagId }) }, 'Open full audit log →')),
      h('div.af-card.overflow-hidden', tableEl, pagerEl));
    async function load() {
      const res = await api.get('/eventLogs', { dag_id: dagId, limit, offset, order_by: '-when', event_pattern: event || null }, { signal });
      if (!live()) return;
      mount(tableEl, dataTable({
        rows: res.event_logs, rowKey: (e) => e.event_log_id, dense: true,
        columns: [
          { key: 'when', label: 'When', render: (e) => h('span.small.af-nowrap', { title: fmtDate(e.when) }, fmtDate(e.when)) },
          { key: 'event', label: 'Event', render: (e) => h('code.small', e.event) },
          { key: 'owner', label: 'User', render: (e) => h('span.small', e.owner || '—') },
          { key: 'task_id', label: 'Task', render: (e) => (e.task_id ? h('span.af-mono.small', e.task_id) : h('span.text-muted', '—')) },
          { key: 'run_id', label: 'Run', render: (e) => (e.run_id ? h('a.af-mono.small', { href: urls.run(dagId, e.run_id) }, truncate(e.run_id, 36)) : h('span.text-muted', '—')) },
          {
            key: 'extra', label: 'Details',
            render: (e) => {
              if (!e.extra) return h('span.text-muted', '—');
              let parsed = e.extra;
              try { parsed = JSON.parse(e.extra); } catch { /* plain text */ }
              return h('button.af-link-btn.af-mono.small', {
                onclick: () => modal({ title: e.event, subtitle: fmtDate(e.when), size: 'lg', body: typeof parsed === 'object' ? jsonView(parsed, { open: 2 }) : h('pre.af-doc', String(parsed)) }),
              }, truncate(e.extra, 60));
            },
          },
        ],
        empty: emptyState({ icon: 'journal-text', title: 'No events' }),
      }));
      mount(pagerEl, res.total_entries ? pager({ total: res.total_entries, limit, offset, onChange: (o, l) => { offset = o; limit = l; load(); } }) : null);
    }
    await load();
    return load;
  }

  // ---------------------------------------------------------------- asset events

  async function assets(pane, live) {
    const listEl = h('div', skeleton(3));
    mount(pane,
      details.asset_expression
        ? h('div.mb-3', h('div.small.fw-semibold.mb-1', 'This DAG is scheduled on assets:'), jsonView(details.asset_expression, { open: 3 }))
        : h('div.af-callout.af-callout-info.mb-3', icon('info-circle'), h('div', 'This DAG is not asset-scheduled. Queued events appear here when an upstream asset this DAG depends on is updated.')),
      h('div.af-tab-toolbar',
        h('div.fw-semibold.small', 'Queued asset events'),
        h('div.af-grow'),
        h('button.btn.btn-sm.btn-outline-danger', {
          onclick: async () => {
            if (!(await confirmDialog({ title: 'Clear all queued asset events?', message: 'The DAG will no longer be triggered by these pending events.', danger: true, confirmLabel: 'Clear all' }))) return;
            try { await api.del(`${base}/assets/queuedEvents`); toast('Queued events cleared', 'success'); load(); } catch (err) { toastError(err, 'Could not clear events'); }
          },
        }, icon('trash', 'me-1'), 'Clear all')),
      h('div.af-card.overflow-hidden', listEl));
    async function load() {
      let events = [];
      try {
        events = (await api.get(`${base}/assets/queuedEvents`, null, { signal })).queued_events || [];
      } catch (err) {
        if (err.status !== 404) throw err; // Airflow answers 404 when nothing is queued
      }
      if (!live()) return;
      mount(listEl, dataTable({
        rows: events, rowKey: (e) => `${e.asset_id}`,
        columns: [
          { key: 'asset_id', label: 'Asset', render: (e) => h('a', { href: urls.asset(e.asset_id) }, `Asset #${e.asset_id}`) },
          { key: 'created_at', label: 'Queued', render: (e) => h('span.small', fmtDate(e.created_at), ' · ', timeAgo(e.created_at)) },
          {
            key: 'actions', label: '', align: 'right',
            render: (e) => h('button.btn.btn-sm.btn-outline-danger', {
              title: 'Remove this queued event',
              onclick: async () => {
                try { await api.del(`${base}/assets/${enc(e.asset_id)}/queuedEvents`); toast('Event removed', 'success'); load(); } catch (err) { toastError(err, 'Could not remove event'); }
              },
            }, icon('trash')),
          },
        ],
        empty: emptyState({ icon: 'database', title: 'No queued asset events' }),
      }));
    }
    await load();
    return load;
  }
}
