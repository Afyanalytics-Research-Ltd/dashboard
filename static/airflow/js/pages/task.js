// One task instance: header with try switcher and actions, a structured live log
// viewer, details, XComs, dependencies, tries, mapped instances, extra links,
// the task definition and human-in-the-loop responses.

import { h, icon, mount, debounce, prefs } from '../core/dom.js';
import { api, enc, airflowLink, ApiError } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../core/router.js';
import {
  pageHeader, stateBadge, stateInfo, timeAgo, refreshControl, chips, dataTable, tabs, card, pager,
  errorState, skeleton, emptyState, menuButton, kv, jsonView, jsonEditor, copyButton, toast, toastError,
  loadInto, formModal, codeBlock, TASK_STATES,
} from '../core/ui.js';
import { fmtDate, fmtDuration, liveDuration, truncate } from '../core/format.js';
import { markTask, clearTasks, editTaskNote, deleteTaskInstance, runPath } from '../core/actions.js';

const LIVE = new Set(['running', 'queued', 'scheduled', 'deferred', 'up_for_retry', 'up_for_reschedule', 'restarting']);
const LEVELS = ['debug', 'info', 'warning', 'error', 'critical'];
let externalLogsSupported = (() => {
  try { return sessionStorage.getItem('af:extlogs') === 'no' ? false : null; } catch { return null; }
})();

export default async function render(ctx) {
  const { el, signal, params, query } = ctx;
  const { dagId, runId, taskId } = params;
  const mapIndex = params.mapIndex != null ? Number(params.mapIndex) : -1;
  const mapped = mapIndex >= 0;
  const tiBase = `${runPath(dagId, runId)}/taskInstances/${enc(taskId)}`;
  const tiPathM = mapped ? `${tiBase}/${mapIndex}` : tiBase;
  const mapParam = mapped ? { map_index: mapIndex } : {};
  const label = `${taskId}${mapped ? ` [${mapIndex}]` : ''}`;

  let ti = null;
  let tries = [];
  let taskDef = null;
  let tab = query.get('tab') || 'logs';
  let tryNumber = Number(query.get('try')) || null;
  let logViewer = null;

  const headerEl = h('div');
  const tryEl = h('div');
  const tabsHost = h('div');
  const bodyEl = h('div');
  const refresh = refreshControl(() => load(), { key: 'task', initial: 5 });
  ctx.cleanup(refresh.stop);
  ctx.cleanup(() => logViewer?.stop());
  ctx.setTitle(`${label} · ${runId}`);

  mount(el, headerEl, card({ body: h('div', tryEl, tabsHost, bodyEl) }));
  mount(bodyEl, skeleton(10));

  // ---------------------------------------------------------------- loading

  async function load() {
    const prev = ti?.state;
    const [t, tr] = await Promise.all([
      api.get(tiPathM, null, { signal }),
      api.get(`${tiPathM}/tries`, null, { signal }).catch(() => ({ task_instances: [] })),
    ]);
    if (!ctx.active) return;
    ti = t;
    tries = (tr.task_instances || []).sort((a, b) => a.try_number - b.try_number);
    if (!tryNumber || !tries.some((x) => x.try_number === tryNumber)) tryNumber = ti.try_number || tries.at(-1)?.try_number || 1;
    if (prev && LIVE.has(prev) && !LIVE.has(ti.state)) {
      toast(`${label} finished: ${ti.state}`, ti.state === 'success' ? 'success' : 'error', { title: 'Task finished' });
      logViewer?.refreshNow();
    }
    ctx.setTitle(`${ti.state || 'no status'} · ${label} · ${runId}`);
    renderHeader();
    renderTries();
    renderTabs();
    if (!bodyEl.dataset.tab || ['details', 'tries'].includes(tab)) renderTab();
  }

  function renderHeader() {
    const af = airflowLink(`/dags/${enc(dagId)}/runs/${enc(runId)}/tasks/${enc(taskId)}${mapped ? `/mapped/${mapIndex}` : ''}`);
    const reload = () => load().catch(toastError);
    mount(headerEl, pageHeader({
      crumbs: [
        { label: 'DAGs', href: urls.dags() },
        { label: ti.dag_display_name || dagId, href: urls.dag(dagId) },
        { label: truncate(runId, 36), href: urls.run(dagId, runId) },
        mapped ? { label: taskId, href: urls.task(dagId, runId, taskId, -1, { tab: 'mapped' }) } : null,
        { label: mapped ? `[${ti.rendered_map_index || mapIndex}]` : taskId },
      ].filter(Boolean),
      title: h('span.af-mono', ti.task_display_name || taskId, mapped ? h('span.text-muted', ` [${ti.rendered_map_index || mapIndex}]`) : null),
      icon: 'box',
      badges: [stateBadge(ti.state, { size: 'lg' })],
      subtitle: h('span.af-run-sub',
        h('span', icon('gear', 'me-1'), ti.operator_name || ti.operator || 'operator'),
        h('span', icon('arrow-repeat', 'me-1'), `try ${ti.try_number}${ti.max_tries != null ? ` of ${ti.max_tries + 1}` : ''}`),
        h('span', icon('stopwatch', 'me-1'), fmtDuration(liveDuration(ti.start_date, ti.end_date, ti.duration))),
        ti.hostname ? h('span', icon('hdd-network', 'me-1'), ti.hostname) : null,
        h('span', icon('play-circle', 'me-1'), h('a.af-mono', { href: urls.run(dagId, runId) }, truncate(runId, 40)))),
      actions: [
        menuButton('Mark as', ['success', 'failed', 'skipped'].map((st) => ({
          label: stateInfo(st).label, icon: stateInfo(st).icon, disabled: ti.state === st,
          onClick: async () => { if (await markTask(ti, st)) reload(); },
        })), { icon: 'check2-square' }),
        h('button.btn.btn-sm.btn-outline-primary', {
          onclick: async () => { if (await clearTasks(dagId, { runId, taskIds: [taskId], mapIndexes: [mapIndex] })) reload(); },
        }, icon('arrow-counterclockwise', 'me-1'), 'Clear'),
        menuButton(null, [
          { label: 'Edit note…', icon: 'sticky', onClick: async () => { if (await editTaskNote(ti)) reload(); } },
          { label: 'View run', icon: 'play-circle', href: urls.run(dagId, runId) },
          { label: 'View in graph', icon: 'diagram-3', href: urls.run(dagId, runId, 'graph') },
          af && { label: 'Open in Airflow', icon: 'wind', href: af, external: true },
          { divider: true },
          { label: 'Delete task instance…', icon: 'trash', danger: true, onClick: async () => { if (await deleteTaskInstance(ti)) navigate(urls.run(dagId, runId)); } },
        ].filter(Boolean)),
        refresh.el,
      ],
    }));
  }

  function renderTries() {
    if (tries.length < 2) { mount(tryEl); return; }
    mount(tryEl, h('div.af-tries-bar',
      h('span.small.text-muted.fw-semibold', 'Attempt'),
      tries.map((t) => {
        const info = stateInfo(t.state);
        return h('button.af-try-btn', {
          type: 'button', class: t.try_number === tryNumber && 'active',
          title: `${info.label} · ${fmtDuration(t.duration)} · ${fmtDate(t.start_date)}`,
          onclick: () => { tryNumber = t.try_number; ctx.setQuery({ try: tryNumber === ti.try_number ? null : tryNumber }); renderTries(); renderTab(); },
        }, h('span.af-dot', { style: { background: info.color } }), `#${t.try_number}`);
      })));
  }

  // ---------------------------------------------------------------- tabs

  let tabBar = null;
  function tabList() {
    return [
      { id: 'logs', label: 'Logs', icon: 'terminal' },
      { id: 'details', label: 'Details', icon: 'info-circle' },
      { id: 'xcom', label: 'XCom', icon: 'arrow-left-right' },
      { id: 'deps', label: 'Dependencies', icon: 'signpost-split' },
      { id: 'tries', label: 'Tries', icon: 'arrow-repeat', count: tries.length || null },
      (mapped || taskDef?.is_mapped) && { id: 'mapped', label: 'Mapped', icon: 'grid-3x3' },
      { id: 'links', label: 'Links', icon: 'link-45deg' },
      { id: 'definition', label: 'Task definition', icon: 'code-square' },
      { id: 'hitl', label: 'Required action', icon: 'person-check' },
    ].filter(Boolean);
  }
  function onTab(id) {
    tab = id;
    ctx.setQuery({ tab: id === 'logs' ? null : id });
    renderTab();
  }
  function renderTabs() {
    const fresh = tabs(tabList(), tab, onTab);
    if (tabBar) tabBar.replaceWith(fresh); else mount(tabsHost, fresh);
    tabBar = fresh;
  }

  function renderTab() {
    if (tab !== 'logs') { logViewer?.stop(); logViewer = null; }
    bodyEl.dataset.tab = tab;
    const fn = { logs: renderLogs, details: renderDetails, xcom: renderXcom, deps: renderDeps, tries: renderTriesTab, mapped: renderMapped, links: renderLinks, definition: renderDefinition, hitl: renderHitl }[tab] || renderLogs;
    fn();
  }

  // ---------------------------------------------------------------- logs

  function renderLogs() {
    if (logViewer && logViewer.tryNumber === tryNumber) return;
    logViewer?.stop();
    if (!ti.try_number && !tries.length) {
      mount(bodyEl, emptyState({ icon: 'terminal', title: 'No logs yet', message: 'This task instance has not started an attempt.' }));
      return;
    }
    logViewer = createLogViewer({
      path: `${tiBase}/logs/${tryNumber}`,
      externalPath: `${tiBase}/externalLogUrl/${tryNumber}`,
      params: mapParam,
      live: () => LIVE.has(tries.find((t) => t.try_number === tryNumber)?.state ?? ti.state) && tryNumber === ti.try_number,
      fileName: `${dagId}__${runId}__${taskId}${mapped ? `__${mapIndex}` : ''}__try${tryNumber}.log`.replace(/[^\w.+-]+/g, '_'),
      signal,
    });
    logViewer.tryNumber = tryNumber;
    mount(bodyEl, logViewer.el);
  }

  // ---------------------------------------------------------------- details

  function renderDetails() {
    const t = tries.find((x) => x.try_number === tryNumber) && tryNumber !== ti.try_number ? { ...ti, ...tries.find((x) => x.try_number === tryNumber) } : ti;
    const hasRendered = t.rendered_fields && Object.keys(t.rendered_fields).length;
    mount(bodyEl, h('div.af-grid-main-side',
      h('div.d-flex.flex-column.gap-3',
        tryNumber !== ti.try_number ? h('div.af-callout.af-callout-info', icon('info-circle'), `Showing attempt #${tryNumber}. The latest attempt is #${ti.try_number}.`) : null,
        kv([
          ['Task id', h('span.af-mono', t.task_id, ' ', copyButton(t.task_id))],
          ['State', stateBadge(t.state)],
          ['Map index', t.map_index >= 0 ? `${t.map_index}${t.rendered_map_index ? ` (${t.rendered_map_index})` : ''}` : 'not mapped'],
          ['Try', `${t.try_number}${t.max_tries != null ? ` of ${t.max_tries + 1}` : ''}`],
          ['Operator', t.operator_name || t.operator],
          ['Duration', fmtDuration(liveDuration(t.start_date, t.end_date, t.duration))],
          ['Scheduled', fmtDate(t.scheduled_when)],
          ['Queued', fmtDate(t.queued_when)],
          ['Started', fmtDate(t.start_date)],
          ['Ended', fmtDate(t.end_date)],
          ['Pool', t.pool ? `${t.pool} (${t.pool_slots} slot${t.pool_slots === 1 ? '' : 's'})` : null],
          ['Queue', t.queue],
          ['Priority weight', t.priority_weight],
          ['Executor', t.executor],
          ['Hostname', t.hostname],
          ['Unix name', t.unixname],
          ['PID', t.pid],
          ['Logical date', fmtDate(t.logical_date)],
          ['Run after', fmtDate(t.run_after)],
          ['DAG version', t.dag_version ? `v${t.dag_version.version_number} · ${t.dag_version.bundle_name}` : null],
          ['Trigger', t.trigger ? h('code.af-inline-json', JSON.stringify(t.trigger)) : null, { wide: true }],
          ['Triggerer job', t.triggerer_job ? `#${t.triggerer_job.id} on ${t.triggerer_job.hostname}` : null],
          ['Note', t.note, { wide: true }],
        ]),
        h('div',
          h('div.af-section-title', 'Rendered template fields'),
          hasRendered ? renderedFields(t.rendered_fields) : h('div.text-muted.small', 'No rendered fields recorded for this attempt.'))),
      h('div.d-flex.flex-column.gap-3',
        h('div', h('div.af-section-title', 'Executor config'), executorConfig(t.executor_config)),
        h('div', h('div.af-section-title', 'Raw JSON'), jsonView(t, { open: 0 })))));
  }

  // ---------------------------------------------------------------- xcom

  function renderXcom() {
    const add = h('button.btn.btn-sm.btn-outline-primary', { onclick: () => xcomForm(null) }, icon('plus-lg', 'me-1'), 'Add XCom');
    const host = h('div');
    mount(bodyEl, h('div.af-toolbar.mb-3', h('span.small.text-muted', 'Values this task pushed to (or that were written for) cross-communication.'), h('div.af-grow'), add), host);
    loadInto(host, async () => {
      const res = await api.get(`${tiBase}/xcomEntries`, { ...mapParam, limit: 100 }, { signal });
      const entries = res.xcom_entries || [];
      if (!entries.length) return emptyState({ icon: 'arrow-left-right', title: 'No XCom entries', message: 'This task did not push any XCom values.' });
      return h('div.af-xcom-list', entries.map((x) => xcomRow(x)));
    });
  }

  function xcomRow(x) {
    const valueHost = h('div.af-xcom-value.d-none');
    let loaded = false;
    const toggle = async () => {
      valueHost.classList.toggle('d-none');
      if (loaded) return;
      loaded = true;
      mount(valueHost, skeleton(3));
      try {
        const v = await fetchXcom(x.key, x.map_index);
        mount(valueHost, xcomValue(v.value));
      } catch (err) {
        loaded = false;
        mount(valueHost, errorState(err));
      }
    };
    return h('div.af-xcom-item',
      h('div.af-xcom-head',
        h('button.af-link-btn.af-mono.fw-semibold', { onclick: toggle }, icon('chevron-right', 'me-1'), x.key),
        x.map_index >= 0 ? h('span.af-tag', `map ${x.map_index}`) : null,
        h('span.small.text-muted', timeAgo(x.timestamp)),
        h('div.af-grow'),
        h('button.btn.btn-sm.btn-outline-secondary', {
          onclick: async () => {
            try {
              const v = await fetchXcom(x.key, x.map_index);
              xcomForm({ ...x, value: v.value });
            } catch (err) { toastError(err, 'Could not load XCom'); }
          },
        }, icon('pencil', 'me-1'), 'Edit')),
      valueHost);
  }

  async function fetchXcom(key, mi) {
    const path = `${tiBase}/xcomEntries/${enc(key)}`;
    const p = mi >= 0 ? { map_index: mi } : {};
    try {
      return await api.get(path, { ...p, deserialize: true }, { signal });
    } catch (err) {
      // deserialize can be disabled server-side; fall back to the stored form.
      if (err instanceof ApiError && err.status === 400) return api.get(path, { ...p, stringify: false }, { signal });
      throw err;
    }
  }

  function xcomForm(existing) {
    formModal({
      title: existing ? `Edit XCom ${existing.key}` : 'Add XCom', icon: 'arrow-left-right', size: 'lg',
      fields: [
        { name: 'key', label: 'Key', required: true, value: existing?.key || '', readonly: !!existing },
        { name: 'map_index', label: 'Map index', type: 'number', value: existing ? existing.map_index : mapIndex, help: '-1 for unmapped tasks' },
        { name: 'value', label: 'Value (JSON)', type: 'json', required: true, value: existing ? existing.value : {}, rows: 10 },
      ],
      submitLabel: existing ? 'Save' : 'Create',
      onSubmit: async (v, close) => {
        const mi = v.map_index ?? -1;
        if (existing) await api.patch(`${tiBase}/xcomEntries/${enc(existing.key)}`, { value: v.value, map_index: mi });
        else await api.post(`${tiBase}/xcomEntries`, { key: v.key, value: v.value, map_index: mi });
        toast(`XCom ${v.key} saved`, 'success');
        close();
        renderXcom();
      },
    });
  }

  // ---------------------------------------------------------------- dependencies

  function renderDeps() {
    loadInto(bodyEl, async () => {
      const res = await api.get(`${tiPathM}/dependencies`, null, { signal });
      const deps = res.dependencies || [];
      if (!deps.length) {
        return h('div.af-callout.af-callout-info', icon('check2-circle'),
          h('div', h('div.fw-semibold', 'No blocking dependencies'),
            h('div.small', LIVE.has(ti.state) || !ti.state
              ? 'All scheduling dependencies are met — the task is waiting only on executor capacity.'
              : `The task is ${stateInfo(ti.state).label}; nothing is holding it back.`)));
      }
      return h('div.d-flex.flex-column.gap-2',
        h('div.small.text-muted', 'Why this task is not running yet:'),
        deps.map((d) => h('div.af-callout.af-callout-warn', icon('signpost-split'),
          h('div', h('div.fw-semibold', d.name), h('div.small', d.reason)))));
    });
  }

  // ---------------------------------------------------------------- tries

  function renderTriesTab() {
    if (!tries.length) { mount(bodyEl, emptyState({ icon: 'arrow-repeat', title: 'No attempts recorded' })); return; }
    mount(bodyEl, dataTable({
      rows: [...tries].reverse(),
      rowKey: (t) => t.try_number,
      onRowClick: (t) => { tryNumber = t.try_number; renderTries(); onTab('logs'); renderTabs(); },
      rowClass: (t) => t.try_number === tryNumber && 'af-row-selected',
      columns: [
        { key: 'try_number', label: 'Try', render: (t) => h('span.fw-semibold', `#${t.try_number}`) },
        { key: 'state', label: 'State', render: (t) => stateBadge(t.state) },
        { key: 'start_date', label: 'Started', render: (t) => fmtDate(t.start_date) },
        { key: 'end_date', label: 'Ended', render: (t) => fmtDate(t.end_date) },
        { key: 'duration', label: 'Duration', align: 'right', render: (t) => fmtDuration(liveDuration(t.start_date, t.end_date, t.duration)) },
        { key: 'hostname', label: 'Host', render: (t) => h('span.small.af-mono', t.hostname || '—') },
        { key: 'executor', label: 'Executor' },
        { key: 'logs', label: '', align: 'right', render: () => h('span.af-link-btn', icon('terminal'), ' Logs') },
      ],
    }));
  }

  // ---------------------------------------------------------------- mapped

  function renderMapped() {
    const state = { states: [], offset: 0, limit: 50 };
    const host = h('div');
    const chipHost = h('div.mb-3');
    mount(bodyEl, chipHost, host);
    mount(chipHost, chips(TASK_STATES.map((st) => ({ value: st, label: stateInfo(st).label, color: stateInfo(st).color })), state.states,
      (v) => { state.states = v; state.offset = 0; draw(); }, { multi: true }));
    const draw = () => loadInto(host, async () => {
      const res = await api.get(`${tiBase}/listMapped`, { state: state.states, limit: state.limit, offset: state.offset, order_by: 'map_index' }, { signal });
      const rows = res.task_instances || [];
      if (!rows.length) return emptyState({ icon: 'grid-3x3', title: 'No mapped instances match' });
      return h('div',
        dataTable({
          rows, rowKey: (t) => t.map_index,
          onRowClick: (t) => navigate(urls.task(dagId, runId, taskId, t.map_index)),
          rowClass: (t) => t.map_index === mapIndex && 'af-row-selected',
          columns: [
            { key: 'map_index', label: 'Index', render: (t) => h('span.af-mono.fw-semibold', t.rendered_map_index ? `${t.map_index} · ${t.rendered_map_index}` : String(t.map_index)) },
            { key: 'state', label: 'State', render: (t) => stateBadge(t.state) },
            { key: 'try_number', label: 'Try', align: 'center' },
            { key: 'start_date', label: 'Started', render: (t) => fmtDate(t.start_date) },
            { key: 'duration', label: 'Duration', align: 'right', render: (t) => fmtDuration(liveDuration(t.start_date, t.end_date, t.duration)) },
          ],
        }),
        pager({ total: res.total_entries, limit: state.limit, offset: state.offset, onChange: (o, l) => { state.offset = o; state.limit = l; draw(); } }));
    });
    draw();
  }

  // ---------------------------------------------------------------- links

  function renderLinks() {
    loadInto(bodyEl, async () => {
      const res = await api.get(`${tiBase}/links`, mapParam, { signal });
      const links = Object.entries(res.extra_links || {});
      if (!links.length) return emptyState({ icon: 'link-45deg', title: 'No extra links', message: 'This operator does not define any operator extra links.' });
      return h('div.af-list.af-bordered', links.map(([name, url]) => (url
        ? h('a.af-list-item', { href: url, target: '_blank', rel: 'noopener' }, icon('box-arrow-up-right'), h('div.af-list-main', h('div.af-list-title', name), h('div.af-list-sub', url)))
        : h('div.af-list-item', icon('link-45deg', 'text-muted'), h('div.af-list-main', h('div.af-list-title', name), h('div.af-list-sub', 'No URL for this attempt'))))));
    });
  }

  // ---------------------------------------------------------------- definition

  function renderDefinition() {
    loadInto(bodyEl, async () => {
      taskDef = taskDef || await api.get(`/dags/${enc(dagId)}/tasks/${enc(taskId)}`, null, { signal });
      const t = taskDef;
      const rd = t.retry_delay;
      const delay = rd && typeof rd === 'object' ? (rd.days || 0) * 86400 + (rd.seconds || 0) + (rd.microseconds || 0) / 1e6 : null;
      const et = t.execution_timeout;
      const timeout = et && typeof et === 'object' ? (et.days || 0) * 86400 + (et.seconds || 0) : null;
      return h('div.af-grid-main-side',
        h('div.d-flex.flex-column.gap-3',
          kv([
            ['Operator', h('span', t.operator_name, t.class_ref ? h('div.small.text-muted.af-mono', `${t.class_ref.module_path}.${t.class_ref.class_name}`) : null)],
            ['Owner', t.owner],
            ['Trigger rule', h('code', t.trigger_rule)],
            ['Retries', t.retries],
            ['Retry delay', delay != null ? fmtDuration(delay) : null],
            ['Exponential backoff', t.retry_exponential_backoff ? 'yes' : 'no'],
            ['Execution timeout', timeout != null ? fmtDuration(timeout) : 'none'],
            ['Depends on past', t.depends_on_past ? 'yes' : 'no'],
            ['Wait for downstream', t.wait_for_downstream ? 'yes' : 'no'],
            ['Pool', `${t.pool} (${t.pool_slots} slot${t.pool_slots === 1 ? '' : 's'})`],
            ['Queue', t.queue],
            ['Priority', `${t.priority_weight} (${t.weight_rule})`],
            ['Mapped', t.is_mapped ? 'yes' : 'no'],
            ['Start / end', `${fmtDate(t.start_date)} → ${t.end_date ? fmtDate(t.end_date) : '∞'}`],
            ['Template fields', (t.template_fields || []).length ? h('div.af-tags', t.template_fields.map((f) => h('span.af-tag', f))) : null, { wide: true }],
            ['Downstream', (t.downstream_task_ids || []).length
              ? h('div.af-tags', t.downstream_task_ids.map((d) => h('a.af-tag', { href: urls.task(dagId, runId, d) }, d)))
              : 'none (leaf task)', { wide: true }],
          ]),
          t.doc_md ? h('div', h('div.af-section-title', 'Documentation'), h('div.af-doc-md', t.doc_md)) : null),
        h('div.d-flex.flex-column.gap-3',
          t.params && Object.keys(t.params).length ? h('div', h('div.af-section-title', 'Params'), jsonView(t.params, { open: 1 })) : null,
          h('div', h('div.af-section-title', 'Raw JSON'), jsonView(t, { open: 0 }))));
    });
  }

  // ---------------------------------------------------------------- HITL

  function renderHitl() {
    loadInto(bodyEl, async () => {
      let detail;
      try {
        detail = await api.get(`${tiBase}/${mapIndex}/hitlDetails`, null, { signal });
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          return emptyState({ icon: 'person-check', title: 'No input requested', message: 'This task is not a human-in-the-loop task, or it has not asked for a response.' });
        }
        throw err;
      }
      return hitlPanel(detail);
    });
  }

  function hitlPanel(d) {
    const chosen = new Set(d.chosen_options || d.defaults || []);
    const inputType = d.multiple ? 'checkbox' : 'radio';
    const name = `af-hitl-${Math.random().toString(36).slice(2, 7)}`;
    const options = h('div.af-hitl-options', (d.options || []).map((o) => h('label.af-hitl-option',
      h('input.form-check-input', {
        type: inputType, name, value: o, checked: chosen.has(o), disabled: d.response_received,
        onchange: (e) => { if (!d.multiple) chosen.clear(); e.target.checked ? chosen.add(o) : chosen.delete(o); },
      }), h('span', o))));
    const hasParams = d.params && Object.keys(d.params).length;
    const defaults = hasParams ? Object.fromEntries(Object.entries(d.params).map(([k, v]) => [k, v && typeof v === 'object' && 'value' in v ? v.value : v])) : {};
    const paramsEditor = hasParams ? jsonEditor(d.params_input && Object.keys(d.params_input).length ? d.params_input : defaults, { rows: 8 }) : null;
    return h('div.af-hitl',
      h('div.d-flex.align-items-center.gap-2.mb-2',
        d.response_received ? h('span.badge.text-bg-success', 'answered') : h('span.badge.text-bg-warning', 'waiting for response'),
        h('span.small.text-muted', 'asked ', timeAgo(d.created_at)),
        d.assigned_users?.length ? h('span.small.text-muted', `· assigned to ${d.assigned_users.map((u) => u.name || u.id).join(', ')}`) : null),
      h('h5.mb-1', d.subject),
      d.body ? h('div.af-doc-md.mb-3', d.body) : null,
      h('div.af-section-title', d.multiple ? 'Choose one or more' : 'Choose one'),
      options,
      paramsEditor ? h('div.mt-3', h('div.af-section-title', 'Parameters'), paramsEditor.el) : null,
      d.response_received
        ? h('div.af-callout.af-callout-info.mt-3', icon('check2-circle'),
          h('div', 'Responded ', timeAgo(d.responded_at), d.responded_by_user ? ` by ${d.responded_by_user.name || d.responded_by_user.id}` : '',
            ': ', h('b', (d.chosen_options || []).join(', ') || '—')))
        : h('div.mt-3', h('button.btn.btn-primary.btn-sm', {
          onclick: async (e) => {
            if (!chosen.size) { toast('Choose at least one option', 'warning'); return; }
            const btn = e.currentTarget;
            btn.disabled = true;
            try {
              await api.patch(`${tiBase}/${mapIndex}/hitlDetails`, { chosen_options: [...chosen], params_input: paramsEditor?.value() || {} });
              toast('Response submitted', 'success');
              renderHitl();
              load().catch(() => {});
            } catch (err) { toastError(err, 'Response failed'); btn.disabled = false; }
          },
        }, icon('send', 'me-1'), 'Submit response')));
  }

  // ---------------------------------------------------------------- boot

  try {
    await load();
  } catch (err) {
    if (err.name === 'AbortError' || !ctx.active) return;
    // An unmapped URL for a mapped task: show its mapped instances instead.
    if (!mapped && err instanceof ApiError && (err.status === 400 || err.status === 404 || err.status === 422)) {
      const list = await api.get(`${tiBase}/listMapped`, { limit: 1 }, { signal }).catch(() => null);
      if (list?.total_entries) return renderMappedParent(list.total_entries);
    }
    refresh.stop();
    mount(el, pageHeader({
      crumbs: [{ label: 'DAGs', href: urls.dags() }, { label: dagId, href: urls.dag(dagId) }, { label: truncate(runId, 36), href: urls.run(dagId, runId) }, { label }],
      title: label, icon: 'box',
    }), errorState(err, () => navigate(location.pathname + location.search, { replace: true })));
  }

  function renderMappedParent(total) {
    refresh.stop();
    mount(el,
      pageHeader({
        crumbs: [{ label: 'DAGs', href: urls.dags() }, { label: dagId, href: urls.dag(dagId) }, { label: truncate(runId, 36), href: urls.run(dagId, runId) }, { label: taskId }],
        title: h('span.af-mono', taskId), icon: 'grid-3x3',
        badges: [h('span.af-tag', `${total} mapped instance${total === 1 ? '' : 's'}`)],
        subtitle: 'This task is mapped — pick an instance to see its logs and details.',
      }),
      card({ body: bodyEl }));
    ti = { state: null };
    tab = 'mapped';
    renderMapped();
  }
}

// ======================================================================== log viewer

function normalizeEntry(raw) {
  if (typeof raw === 'string') {
    const m = raw.match(/^\[(.+?)\]\s+\{(.+?)\}\s+(\w+)\s+-\s+(.*)$/s);
    return m ? { timestamp: m[1], filename: m[2], level: m[3].toLowerCase(), event: m[4] } : { event: raw };
  }
  return raw || { event: '' };
}

const BASE_KEYS = new Set(['timestamp', 'event', 'level', 'logger', 'filename', 'lineno', 'sources', 'error_detail', 'chan']);

function extras(e) {
  return Object.entries(e).filter(([k]) => !BASE_KEYS.has(k));
}

function tracebackLines(detail) {
  if (!Array.isArray(detail)) return [];
  const out = [];
  for (const exc of detail) {
    out.push('Traceback (most recent call last):');
    for (const f of exc.frames || []) out.push(`  File "${f.filename}", line ${f.lineno}, in ${f.name}`);
    out.push(`${exc.exc_type}: ${exc.exc_value}`);
    if (exc.exc_notes?.length) out.push(...exc.exc_notes);
  }
  return out;
}

function entryText(e) {
  const ts = e.timestamp ? `[${e.timestamp}] ` : '';
  const lvl = e.level ? `${e.level.toUpperCase()} ` : '';
  const src = e.logger ? `${e.logger} ` : e.filename ? `${e.filename}:${e.lineno ?? ''} ` : '';
  const ex = extras(e).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(' ');
  return [`${ts}${lvl}${src}- ${e.event ?? ''}${ex ? ` ${ex}` : ''}`, ...tracebackLines(e.error_detail)].join('\n');
}

function highlightText(text, q) {
  if (!q) return [document.createTextNode(text)];
  const out = [];
  const lower = text.toLowerCase();
  let i = 0;
  let idx;
  while ((idx = lower.indexOf(q, i)) !== -1) {
    if (idx > i) out.push(document.createTextNode(text.slice(i, idx)));
    out.push(h('mark', text.slice(idx, idx + q.length)));
    i = idx + q.length;
  }
  if (i < text.length) out.push(document.createTextNode(text.slice(i)));
  return out;
}

function createLogViewer({ path, externalPath, params, live, fileName, signal }) {
  let entries = [];
  let token = null;
  let timer = null;
  let stopped = false;
  let follow = prefs.get('log:follow', true);
  let wrap = prefs.get('log:wrap', true);
  let showTs = prefs.get('log:ts', true);
  let showSrc = prefs.get('log:src', false);
  let levels = [];
  let q = '';
  let matchIdx = -1;
  const openGroups = new Set(); // ::group:: blocks start collapsed

  const logEl = h('div.af-log', { class: wrap && 'wrap', tabindex: 0 });
  const statusEl = h('span.af-log-status.small.text-muted');
  const matchEl = h('span.af-log-matches.small.text-muted');
  const levelHost = h('div');
  const extHost = h('span');

  const toggle = (labelText, ic, get, set) => {
    const btn = h('button.btn.btn-sm.btn-outline-secondary', {
      type: 'button', class: get() && 'active', title: labelText,
      onclick: () => { set(!get()); btn.classList.toggle('active', get()); draw(); },
    }, icon(ic), h('span.d-none.d-xl-inline.ms-1', labelText));
    return btn;
  };

  const searchBox = h('input.form-control.form-control-sm', { type: 'search', placeholder: 'Search logs…', dataset: { searchShortcut: '' } });
  searchBox.addEventListener('input', debounce(() => { q = searchBox.value.trim().toLowerCase(); matchIdx = -1; draw(); jump(1); }, 200));
  searchBox.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); jump(e.shiftKey ? -1 : 1); } });

  const toolbar = h('div.af-log-toolbar',
    h('div.af-search.af-log-search', icon('search', 'af-search-icon'), searchBox),
    matchEl,
    h('div.btn-group',
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Previous match (Shift+Enter)', onclick: () => jump(-1) }, icon('chevron-up')),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Next match (Enter)', onclick: () => jump(1) }, icon('chevron-down'))),
    levelHost,
    h('div.af-grow'),
    h('div.btn-group',
      toggle('Wrap', 'text-wrap', () => wrap, (v) => { wrap = v; prefs.set('log:wrap', v); logEl.classList.toggle('wrap', v); }),
      toggle('Time', 'clock', () => showTs, (v) => { showTs = v; prefs.set('log:ts', v); }),
      toggle('Source', 'file-earmark-code', () => showSrc, (v) => { showSrc = v; prefs.set('log:src', v); }),
      toggle('Follow', 'arrow-down-circle', () => follow, (v) => { follow = v; prefs.set('log:follow', v); if (v) scrollEnd(); })),
    h('div.btn-group',
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Jump to first error', onclick: jumpToError }, icon('exclamation-octagon')),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Expand / collapse all groups', onclick: toggleGroups }, icon('arrows-collapse')),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Copy all', onclick: copyAll }, icon('clipboard')),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Download .log', onclick: download }, icon('download')),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Reload', onclick: () => fetchAll() }, icon('arrow-clockwise'))),
    extHost);

  const el = h('div.af-logviewer', toolbar, logEl, h('div.af-log-foot', statusEl));
  mount(logEl, h('div.af-log-empty', 'Loading logs…'));

  function levelCounts() {
    const c = {};
    for (const e of entries) if (e.level) c[e.level] = (c[e.level] || 0) + 1;
    return c;
  }
  function drawLevels() {
    const c = levelCounts();
    const present = LEVELS.filter((l) => c[l]);
    const colors = { debug: '#64748b', info: '#0284c7', warning: '#d97706', error: '#dc2626', critical: '#991b1b' };
    mount(levelHost, present.length > 1 ? chips(present.map((l) => ({ value: l, label: l, color: colors[l], count: c[l] })), levels,
      (v) => { levels = v; draw(); }, { multi: true }) : null);
  }

  function lineEl(e, i) {
    const lvl = (e.level || '').toLowerCase();
    const msg = h('span.af-log-msg', highlightText(String(e.event ?? ''), q));
    const ex = extras(e);
    if (ex.length) msg.appendChild(h('span.af-log-extra', ex.map(([k, v]) => ` ${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join('')));
    const tb = tracebackLines(e.error_detail);
    if (tb.length) msg.appendChild(h('div.af-log-tb', highlightText(tb.join('\n'), q)));
    return h('div.af-log-line', { class: lvl && `lvl-${lvl}`, dataset: { i } },
      showTs && e.timestamp ? h('span.af-log-ts', { title: e.timestamp }, fmtLogTime(e.timestamp)) : null,
      lvl ? h(`span.af-log-lvl.af-log-lvl-${lvl}`, lvl) : (showTs ? h('span.af-log-lvl') : null),
      showSrc && (e.logger || e.filename) ? h('span.af-log-src', { title: e.logger || '' }, e.filename ? `${e.filename}:${e.lineno ?? ''}` : truncate(e.logger, 40)) : null,
      msg);
  }

  function matches(e) {
    if (levels.length && e.level && !levels.includes(e.level.toLowerCase())) return false;
    if (levels.length && !e.level) return false;
    return true;
  }

  function draw() {
    const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
    const frag = document.createDocumentFragment();
    let groupBody = null;
    let groupIdx = 0;
    let shown = 0;
    let hits = 0;
    entries.forEach((e, i) => {
      const ev = String(e.event ?? '');
      if (ev.startsWith('::group::')) {
        const gid = groupIdx++;
        const title = ev.slice(9) || 'Group';
        const isCollapsed = !openGroups.has(gid);
        const body = h('div.af-log-group-body', { class: isCollapsed && 'collapsed' });
        const header = h('div.af-log-group', {
          onclick: () => {
            const nowCollapsed = !body.classList.contains('collapsed');
            body.classList.toggle('collapsed', nowCollapsed);
            if (nowCollapsed) openGroups.delete(gid); else openGroups.add(gid);
            header.firstChild.className = `bi bi-caret-${nowCollapsed ? 'right' : 'down'}-fill me-1`;
          },
        }, icon(isCollapsed ? 'caret-right-fill' : 'caret-down-fill', 'me-1'), title);
        if (e.sources) body.appendChild(h('div.af-log-line', h('span.af-log-msg.text-muted', e.sources.map((src) => `source: ${src}`).join('\n'))));
        frag.append(header, body);
        groupBody = body;
        return;
      }
      if (ev.startsWith('::endgroup::')) { groupBody = null; return; }
      if (!matches(e)) return;
      const line = lineEl(e, i);
      if (q && entryText(e).toLowerCase().includes(q)) { line.classList.add('af-log-hit'); hits++; if (groupBody) groupBody.classList.remove('collapsed'); }
      (groupBody || frag).appendChild(line);
      shown++;
    });
    mount(logEl, shown || groupIdx ? frag : h('div.af-log-empty', entries.length ? 'No lines match the level filter.' : 'The log is empty.'));
    mount(matchEl, q ? `${hits.toLocaleString()} match${hits === 1 ? '' : 'es'}` : '');
    if (follow && atBottom && !q) scrollEnd();
  }

  function scrollEnd() { logEl.scrollTop = logEl.scrollHeight; }

  function jump(dir) {
    const hits = [...logEl.querySelectorAll('.af-log-hit')];
    if (!hits.length) return;
    hits.forEach((x) => x.classList.remove('af-log-current'));
    matchIdx = (matchIdx + dir + hits.length) % hits.length;
    const target = hits[matchIdx];
    target.classList.add('af-log-current');
    target.scrollIntoView({ block: 'center' });
    mount(matchEl, `${matchIdx + 1} / ${hits.length.toLocaleString()}`);
  }

  function jumpToError() {
    const target = logEl.querySelector('.lvl-error, .lvl-critical');
    if (!target) { toast('No error lines in this log', 'info', { timeout: 2000 }); return; }
    target.closest('.af-log-group-body')?.classList.remove('collapsed');
    target.scrollIntoView({ block: 'center' });
    target.classList.add('af-log-current');
    setTimeout(() => target.classList.remove('af-log-current'), 2000);
  }

  function toggleGroups() {
    const bodies = [...logEl.querySelectorAll('.af-log-group-body')];
    const anyCollapsed = bodies.some((b) => b.classList.contains('collapsed'));
    openGroups.clear();
    if (anyCollapsed) bodies.forEach((_, gid) => openGroups.add(gid));
    draw();
  }

  const plain = () => entries.map((e) => (String(e.event ?? '').startsWith('::') ? `${e.event}` : entryText(e))).join('\n');
  async function copyAll() {
    try { await navigator.clipboard.writeText(plain()); toast('Log copied', 'success', { timeout: 1500 }); } catch (err) { toastError(err, 'Copy failed'); }
  }
  function download() {
    const a = h('a', { href: URL.createObjectURL(new Blob([plain()], { type: 'text/plain' })), download: fileName });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  async function fetchAll() {
    try {
      const res = await api.get(path, params, { signal });
      if (stopped) return;
      entries = (res.content || []).map(normalizeEntry);
      token = res.continuation_token || null;
      drawLevels();
      draw();
      if (follow) scrollEnd();
      mount(statusEl, `${entries.length.toLocaleString()} lines`, live() ? h('span.af-log-live', ' · ', h('span.af-dot.af-pulse', { style: { background: '#16a34a' } }), ' live tail') : '');
    } catch (err) {
      if (err.name === 'AbortError' || stopped) return;
      mount(logEl, h('div.af-log-empty.text-danger', err.message));
    }
  }

  async function tail() {
    if (stopped || document.hidden) return;
    if (!live()) { mount(statusEl, `${entries.length.toLocaleString()} lines`); return; }
    try {
      if (token) {
        const res = await api.get(path, { ...params, token }, { signal });
        const more = (res.content || []).map(normalizeEntry);
        token = res.continuation_token || token;
        if (more.length) { entries.push(...more); drawLevels(); draw(); }
      } else {
        const res = await api.get(path, params, { signal });
        const all = (res.content || []).map(normalizeEntry);
        token = res.continuation_token || null;
        if (all.length !== entries.length) { entries = all; drawLevels(); draw(); }
      }
      mount(statusEl, `${entries.length.toLocaleString()} lines · `, h('span.af-dot.af-pulse', { style: { background: '#16a34a' } }), ' live tail');
    } catch { /* transient; next tick retries */ }
  }

  if (externalLogsSupported !== false) {
    api.get(externalPath, params, { signal }).then((res) => {
      externalLogsSupported = true;
      if (res?.url) mount(extHost, h('a.btn.btn-sm.btn-outline-secondary', { href: res.url, target: '_blank', rel: 'noopener' }, icon('box-arrow-up-right', 'me-1'), 'External logs'));
    }).catch((err) => {
      // 400 = the log handler has no external URL support; don't ask again this session.
      if (err instanceof ApiError && err.status === 400) {
        externalLogsSupported = false;
        try { sessionStorage.setItem('af:extlogs', 'no'); } catch { /* storage unavailable */ }
      }
    });
  }

  fetchAll().then(() => { if (!stopped) timer = setInterval(tail, 3000); });

  return {
    el,
    stop() { stopped = true; clearInterval(timer); },
    refreshNow: fetchAll,
  };
}

function fmtLogTime(ts) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
}

function renderedFields(fields) {
  return h('div.d-flex.flex-column.gap-2', Object.entries(fields).map(([k, v]) => h('div.af-rendered',
    h('div.af-rendered-key.af-mono', k),
    typeof v === 'string' && v.includes('\n')
      ? codeBlock(v, { lang: /select|insert|create|merge/i.test(v) ? 'sql' : 'text' })
      : (v && typeof v === 'object' ? jsonView(v, { open: 1 }) : h('code.af-rendered-val', v === '' ? '""' : String(v))))));
}

function executorConfig(cfg) {
  if (!cfg || cfg === '{}') return h('div.text-muted.small', 'Default executor config.');
  try { return jsonView(typeof cfg === 'string' ? JSON.parse(cfg) : cfg, { open: 1 }); } catch { return h('code', String(cfg)); }
}
