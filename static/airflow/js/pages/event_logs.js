// Audit log: GET /eventLogs with every filter the API offers, URL-synced and paginated.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  pageHeader, card, dataTable, pager, refreshControl, chips, timeAgo, drawer, kv, jsonView,
  errorState, skeleton, emptyState, loadInto,
} from '../core/ui.js';
import { fmtDate, truncate } from '../core/format.js';
import { readState, toParams, filterBar, parseMaybeJson } from './sys_util.js';

const FILTERS = [
  { name: 'event_pattern', label: 'Event contains', placeholder: 'e.g. trigger' },
  { name: 'dag_id_pattern', label: 'DAG contains', placeholder: 'dag id' },
  { name: 'owner_pattern', label: 'Owner contains', placeholder: 'user' },
  { name: 'after', label: 'After', type: 'datetime' },
  { name: 'before', label: 'Before', type: 'datetime' },
  { name: 'event', label: 'Event (exact)', advanced: true },
  { name: 'dag_id', label: 'DAG id (exact)', advanced: true },
  { name: 'task_id', label: 'Task id (exact)', advanced: true },
  { name: 'task_id_pattern', label: 'Task contains', advanced: true },
  { name: 'run_id', label: 'Run id (exact)', advanced: true },
  { name: 'run_id_pattern', label: 'Run contains', advanced: true },
  { name: 'owner', label: 'Owner (exact)', advanced: true },
  { name: 'map_index', label: 'Map index', type: 'number', advanced: true },
  { name: 'try_number', label: 'Try number', type: 'number', advanced: true },
  { name: 'included_events', label: 'Only events', type: 'list', advanced: true, placeholder: 'trigger_dag_run, cli_task_run' },
  { name: 'excluded_events', label: 'Exclude events', type: 'list', advanced: true, placeholder: 'get_dags, …' },
];

const EVENT_ICONS = [
  [/trigger/, 'play-circle', '#0072CE'],
  [/delete|remove/, 'trash', '#dc2626'],
  [/fail|error/, 'x-octagon', '#dc2626'],
  [/post|create|add/, 'plus-circle', '#16a34a'],
  [/patch|update|edit|put/, 'pencil-square', '#d97706'],
  [/clear/, 'arrow-counterclockwise', '#7c3aed'],
  [/pause/, 'pause-circle', '#64748b'],
  [/login|logout|auth/, 'person-badge', '#0891b2'],
  [/success/, 'check-circle', '#16a34a'],
];
const eventIcon = (ev = '') => {
  const m = EVENT_ICONS.find(([re]) => re.test(ev.toLowerCase()));
  return m ? icon(m[1], '', ) : icon('dot');
};
const eventColor = (ev = '') => EVENT_ICONS.find(([re]) => re.test(ev.toLowerCase()))?.[2] || '#64748b';

function eventTag(ev) {
  return h('span.sys-event', { style: { '--sys-ev': eventColor(ev) } }, eventIcon(ev), ev || '—');
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  const state = readState(ctx.query, FILTERS, { offset: '0', limit: '50', order_by: '-when' });
  const tableEl = h('div', skeleton(10));
  const topEl = h('div.sys-top-events');
  const countEl = h('span.af-muted.small');

  const refresh = refreshControl(() => load(), { key: 'audit', initial: 30 });
  ctx.cleanup(refresh.stop);

  const sync = () => ctx.setQuery({ ...state, offset: state.offset === '0' ? null : state.offset, limit: state.limit === '50' ? null : state.limit, order_by: state.order_by === '-when' ? null : state.order_by });

  mount(el,
    pageHeader({
      title: 'Audit log', icon: 'journal-text',
      subtitle: 'Every action recorded by Airflow: triggers, clears, edits, CLI and API calls.',
      actions: [refresh.el],
    }),
    card({
      body: h('div.d-flex.flex-column.gap-3',
        filterBar(FILTERS, state, () => { state.offset = '0'; sync(); load(); }),
        topEl),
    }),
    card({ title: 'Events', icon: 'list-ul', actions: countEl, body: tableEl, flush: true }));

  loadTopEvents();

  async function loadTopEvents() {
    try {
      const recent = await api.get('/eventLogs', { limit: 100, order_by: '-when' }, { signal });
      if (!ctx.active) return;
      const counts = {};
      recent.event_logs.forEach((e) => { counts[e.event] = (counts[e.event] || 0) + 1; });
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10);
      mount(topEl,
        h('div.small.text-muted.mb-1', 'Most frequent recent events'),
        chips(top.map(([ev, n]) => ({ value: ev, label: ev, count: n, color: eventColor(ev) })), state.event, (v) => {
          state.event = v || '';
          state.offset = '0';
          sync();
          load();
        }));
    } catch { mount(topEl); }
  }

  async function load() {
    try {
      const params = { ...toParams(state, FILTERS), limit: state.limit, offset: state.offset, order_by: state.order_by };
      const data = await api.get('/eventLogs', params, { signal });
      if (!ctx.active) return;
      mount(countEl, `${data.total_entries.toLocaleString()} events`);
      const sortKey = state.order_by.replace(/^-/, '');
      mount(tableEl,
        dataTable({
          rows: data.event_logs,
          rowKey: (r) => r.event_log_id,
          dense: true,
          sort: { key: sortKey, dir: state.order_by.startsWith('-') ? 'desc' : 'asc' },
          onSort: (key, dir) => { state.order_by = `${dir === 'desc' ? '-' : ''}${key}`; sync(); load(); },
          onRowClick: (r) => openEvent(r.event_log_id),
          empty: emptyState({ icon: 'journal-x', title: 'No events match', message: 'Try widening the time range or clearing filters.' }),
          columns: [
            { key: 'when', label: 'When', sortable: true, width: '150px', render: (r) => h('div', h('div.af-nowrap.small', fmtDate(r.when)), h('div.small', timeAgo(r.when))) },
            { key: 'event', label: 'Event', sortable: true, render: (r) => eventTag(r.event) },
            { key: 'dag_id', label: 'DAG', sortable: true, render: (r) => (r.dag_id ? h('a', { href: urls.dag(r.dag_id) }, r.dag_display_name || r.dag_id) : h('span.text-muted', '—')) },
            { key: 'task_id', label: 'Task / run', render: (r) => taskRunCell(r) },
            { key: 'owner', label: 'Owner', sortable: true, render: (r) => (r.owner ? h('span.sys-owner', icon('person'), r.owner) : h('span.text-muted', '—')) },
            { key: 'extra', label: 'Details', render: (r) => (r.extra ? h('code.af-inline-json', truncate(r.extra, 90)) : h('span.text-muted', '—')) },
          ],
        }),
        data.total_entries > Number(state.limit) || Number(state.offset) > 0 ? pager({
          total: data.total_entries, limit: Number(state.limit), offset: Number(state.offset),
          onChange: (offset, limit) => { state.offset = String(offset); state.limit = String(limit); sync(); load(); },
        }) : null);
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(tableEl, h('div.p-3', errorState(err, load)));
    }
  }

  function taskRunCell(r) {
    if (!r.task_id && !r.run_id) return h('span.text-muted', '—');
    const runLink = r.dag_id && r.run_id ? h('a.af-mono.small', { href: urls.run(r.dag_id, r.run_id) }, truncate(r.run_id, 42)) : r.run_id && h('span.af-mono.small', truncate(r.run_id, 42));
    const taskLink = r.task_id && (r.dag_id && r.run_id
      ? h('a.af-mono.small', { href: urls.task(r.dag_id, r.run_id, r.task_id, r.map_index) }, r.task_id, r.map_index >= 0 ? `[${r.map_index}]` : '')
      : h('span.af-mono.small', r.task_id));
    return h('div.d-flex.flex-column', taskLink, runLink);
  }

  function openEvent(id) {
    const d = drawer({ title: h('span', icon('journal-text'), ` Event #${id}`), width: 680 });
    loadInto(d.body, async () => {
      const e = await api.get(`/eventLogs/${enc(id)}`, null, { signal });
      d.setTitle(h('span', eventTag(e.event)), `#${e.event_log_id} · ${fmtDate(e.when)}`);
      const extra = parseMaybeJson(e.extra);
      return h('div.d-flex.flex-column.gap-3',
        kv([
          ['When', h('span', fmtDate(e.when), ' · ', timeAgo(e.when))],
          ['Owner', e.owner],
          e.dag_id && ['DAG', h('a', { href: urls.dag(e.dag_id) }, e.dag_display_name || e.dag_id)],
          e.task_id && ['Task', e.dag_id && e.run_id
            ? h('a.af-mono', { href: urls.task(e.dag_id, e.run_id, e.task_id, e.map_index) }, e.task_display_name || e.task_id)
            : h('span.af-mono', e.task_display_name || e.task_id)],
          e.run_id && ['Run', e.dag_id ? h('a.af-mono', { href: urls.run(e.dag_id, e.run_id) }, e.run_id) : e.run_id, { wide: true }],
          e.map_index != null && e.map_index >= 0 && ['Map index', e.map_index],
          e.try_number != null && ['Try', e.try_number],
          e.logical_date && ['Logical date', fmtDate(e.logical_date)],
        ]),
        h('div',
          h('div.fw-semibold.small.text-uppercase.text-muted.mb-2', 'Extra'),
          extra == null ? h('span.text-muted', 'No extra data')
            : typeof extra === 'object' ? jsonView(extra, { open: 3 }) : h('pre.sys-pre', extra)),
        h('div.d-flex.gap-2.flex-wrap',
          e.dag_id && h('a.btn.btn-sm.btn-outline-secondary', { href: urls.audit({ dag_id: e.dag_id }), onclick: () => d.close() }, icon('funnel', 'me-1'), 'All events for this DAG'),
          e.owner && h('a.btn.btn-sm.btn-outline-secondary', { href: urls.audit({ owner: e.owner }), onclick: () => d.close() }, icon('person', 'me-1'), `All events by ${e.owner}`)));
    });
  }

  await load();
}
