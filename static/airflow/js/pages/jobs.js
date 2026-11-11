// Jobs: scheduler / triggerer / DAG processor / local task jobs with heartbeat freshness.

import { h, icon, mount } from '../core/dom.js';
import { api } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  pageHeader, card, dataTable, pager, refreshControl, chips, timeAgo, stateBadge, statCard,
  errorState, skeleton, emptyState,
} from '../core/ui.js';
import { fmtDate, fmtDuration, liveDuration, toDate } from '../core/format.js';
import { readState, toParams, filterBar } from './sys_util.js';

const JOB_TYPES = ['SchedulerJob', 'TriggererJob', 'DagProcessorJob', 'LocalTaskJob', 'BackfillJob'];
const FILTERS = [
  { name: 'job_type', label: 'Type', type: 'select', options: JOB_TYPES },
  { name: 'job_state', label: 'State', type: 'select', options: ['running', 'success', 'failed', 'restarting'] },
  { name: 'is_alive', label: 'Alive', type: 'select', options: [{ value: 'true', label: 'Alive only' }, { value: 'false', label: 'Not alive' }] },
  { name: 'hostname', label: 'Hostname' },
  { name: 'executor_class', label: 'Executor class', advanced: true },
  { name: 'start_date_gte', label: 'Started after', type: 'datetime', advanced: true },
  { name: 'start_date_lte', label: 'Started before', type: 'datetime', advanced: true },
  { name: 'end_date_gte', label: 'Ended after', type: 'datetime', advanced: true },
  { name: 'end_date_lte', label: 'Ended before', type: 'datetime', advanced: true },
];
const STALE_SECONDS = 120;
const TYPE_ICONS = { SchedulerJob: 'calendar2-week', TriggererJob: 'lightning-charge', DagProcessorJob: 'file-earmark-code', LocalTaskJob: 'gear', BackfillJob: 'rewind-circle' };

const heartbeatAge = (job) => {
  const d = toDate(job.latest_heartbeat);
  return d ? (Date.now() - d.getTime()) / 1000 : null;
};
const isStale = (job) => job.state === 'running' && !job.end_date && (heartbeatAge(job) ?? Infinity) > STALE_SECONDS;

export default async function render(ctx) {
  const { el, signal } = ctx;
  const state = readState(ctx.query, FILTERS, { offset: '0', limit: '50', order_by: '-id' });
  const statsEl = h('div.af-stats', skeleton(1, { height: 60 }));
  const tableEl = h('div', skeleton(8));
  const countEl = h('span.af-muted.small');
  const refresh = refreshControl(() => load(), { key: 'jobs', initial: 10 });
  ctx.cleanup(refresh.stop);
  const sync = () => ctx.setQuery({ ...state, offset: state.offset === '0' ? null : state.offset, limit: state.limit === '50' ? null : state.limit, order_by: state.order_by === '-id' ? null : state.order_by });

  mount(el,
    pageHeader({
      title: 'Jobs', icon: 'cpu',
      subtitle: `Airflow's long-running components and task supervisors. A running job without a heartbeat for ${STALE_SECONDS}s is flagged stale.`,
      actions: [refresh.el],
    }),
    statsEl,
    card({ body: filterBar(FILTERS, state, () => { state.offset = '0'; sync(); load(); }) }),
    card({ title: 'Jobs', icon: 'list-ul', actions: countEl, body: tableEl, flush: true }));

  async function loadStats() {
    // `is_alive` does not narrow total_entries reliably, so count fresh heartbeats client-side.
    const [running, total] = await Promise.all([
      api.all('/jobs', 'jobs', { job_state: 'running', order_by: '-latest_heartbeat' }, { max: 500, signal }),
      api.get('/jobs', { limit: 1 }, { signal }).catch(() => null),
    ]);
    if (!ctx.active) return;
    const alive = {};
    let stale = 0;
    for (const j of running) {
      if (isStale(j)) stale++;
      else alive[j.job_type] = (alive[j.job_type] || 0) + 1;
    }
    const card = (t, label) => statCard({
      label, value: alive[t] || 0, icon: TYPE_ICONS[t], tone: alive[t] ? 'green' : 'red',
      sub: 'alive (fresh heartbeat)',
      onClick: () => { Object.assign(state, { job_type: t, job_state: 'running', offset: '0' }); sync(); render(ctx); },
    });
    mount(statsEl,
      card('SchedulerJob', 'Schedulers'),
      card('TriggererJob', 'Triggerers'),
      card('DagProcessorJob', 'DAG processors'),
      statCard({
        label: 'Stale jobs', value: stale, icon: 'exclamation-triangle', tone: stale ? 'amber' : 'green', sub: 'running, no heartbeat',
        onClick: () => { Object.assign(state, { job_type: '', job_state: 'running', offset: '0', order_by: 'latest_heartbeat' }); sync(); render(ctx); },
      }),
      statCard({ label: 'Jobs recorded', value: total ? total.total_entries.toLocaleString() : '—', icon: 'archive', tone: 'slate', sub: 'all time' }));
  }

  async function load() {
    loadStats().catch(() => {});
    try {
      const data = await api.get('/jobs', { ...toParams(state, FILTERS), limit: state.limit, offset: state.offset, order_by: state.order_by }, { signal });
      if (!ctx.active) return;
      const stale = data.jobs.filter(isStale).length;
      mount(countEl, `${data.total_entries.toLocaleString()} jobs`, stale ? h('span.ms-2.badge.text-bg-warning', `${stale} stale on this page`) : null);
      const sortKey = state.order_by.replace(/^-/, '');
      mount(tableEl,
        dataTable({
          rows: data.jobs,
          rowKey: (j) => j.id,
          dense: true,
          rowClass: (j) => isStale(j) && 'sys-row-stale',
          sort: { key: sortKey, dir: state.order_by.startsWith('-') ? 'desc' : 'asc' },
          onSort: (key, dir) => { state.order_by = `${dir === 'desc' ? '-' : ''}${key}`; sync(); load(); },
          empty: emptyState({ icon: 'cpu', title: 'No jobs match' }),
          columns: [
            { key: 'id', label: 'ID', sortable: true, width: '70px', render: (j) => h('span.af-mono', j.id) },
            { key: 'job_type', label: 'Type', sortable: true, render: (j) => h('span.sys-jobtype', icon(TYPE_ICONS[j.job_type] || 'gear'), j.job_type) },
            { key: 'state', label: 'State', sortable: true, render: (j) => h('div.d-flex.gap-1.align-items-center', stateBadge(j.state), isStale(j) && h('span.sys-stale', { title: 'Running but no recent heartbeat — probably a dead process that never recorded its end.' }, icon('exclamation-triangle-fill'), 'stale')) },
            { key: 'latest_heartbeat', label: 'Heartbeat', sortable: true, render: (j) => heartbeatCell(j) },
            { key: 'start_date', label: 'Started', sortable: true, render: (j) => h('span.small.af-nowrap', { title: fmtDate(j.start_date) }, fmtDate(j.start_date, { seconds: false })) },
            { key: 'duration', label: 'Uptime / duration', render: (j) => fmtDuration(liveDuration(j.start_date, j.end_date || (isStale(j) ? j.latest_heartbeat : null))) },
            { key: 'hostname', label: 'Host', sortable: true, render: (j) => h('span.af-mono.small.af-nowrap', { title: j.unixname ? `user: ${j.unixname}` : undefined }, j.hostname || '—') },
            { key: 'dag_id', label: 'DAG', render: (j) => (j.dag_id ? h('a', { href: urls.dag(j.dag_id) }, j.dag_display_name || j.dag_id) : h('span.text-muted', '—')) },
            { key: 'executor_class', label: 'Executor', render: (j) => (j.executor_class ? h('span.af-mono.small', j.executor_class) : h('span.text-muted', '—')) },
          ],
        }),
        pager({
          total: data.total_entries, limit: Number(state.limit), offset: Number(state.offset),
          onChange: (offset, limit) => { state.offset = String(offset); state.limit = String(limit); sync(); load(); },
        }));
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(tableEl, h('div.p-3', errorState(err, load)));
    }
  }

  function heartbeatCell(j) {
    if (!j.latest_heartbeat) return h('span.text-muted', '—');
    const age = heartbeatAge(j);
    const live = j.state === 'running' && !j.end_date;
    const tone = !live ? 'muted' : age > STALE_SECONDS ? 'bad' : age > 30 ? 'warn' : 'ok';
    return h(`span.sys-hb.sys-hb-${tone}`, h('span.sys-hb-dot'), timeAgo(j.latest_heartbeat));
  }

  await load();
}
