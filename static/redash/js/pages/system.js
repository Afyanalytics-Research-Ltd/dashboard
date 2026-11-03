// System status: server stats (/status.json), workers and queues (/api/admin/queries/rq_status),
// outdated scheduled queries, and client configuration / feature flags.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, statCard, dataTable, timeAgo, refreshControl, emptyState, errorState, skeleton,
  confirmDialog, toast, toastError, kv, jsonView,
} from '../../../ui/js/ui.js';
import { sparkline, tip } from '../../../ui/js/charts.js';
import { fmtBytes, fmtNumber, fmtDuration } from '../../../ui/js/format.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { scheduleText, getSession } from '../core/common.js';
import { countUp } from './home_shared.js';

// Queue backlog history for this browser session (survives page switches).
const backlog = [];
const STALE_HEARTBEAT_S = 120;

export default async function render(ctx) {
  const { el, signal } = ctx;
  const bannerEl = h('div', skeleton(1, { height: 64 }));
  const statsEl = h('div.af-stats', skeleton(1, { height: 70 }));
  const workersEl = h('div', skeleton(3));
  const queuesEl = h('div', skeleton(3));
  const outdatedEl = h('div', skeleton(3));
  const configEl = h('div', skeleton(5));
  const storageEl = h('div', skeleton(3));

  const refresh = refreshControl(() => load(), { key: 'studio-system', initial: 10 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({ title: 'System status', icon: 'heart-pulse', subtitle: 'Workers, queues, storage and configuration of your Redash server — refreshed live.', actions: refresh.el }),
    bannerEl,
    statsEl,
    card({ title: 'Workers', icon: 'cpu', body: workersEl, flush: true }),
    h('div.af-grid-main-side',
      card({ title: 'Queues', icon: 'stack', body: queuesEl }),
      card({ title: 'Storage', icon: 'hdd-stack', body: storageEl })),
    card({ title: 'Outdated scheduled queries', icon: 'calendar-x', body: outdatedEl, flush: true }),
    card({ title: 'Configuration', icon: 'sliders', body: configEl }));

  let configLoaded = false;

  async function load() {
    const [status, rq, outdated] = await Promise.allSettled([
      rd.get('/status.json', null, { signal }),
      rd.get('/api/admin/queries/rq_status', null, { signal }),
      rd.get('/api/admin/queries/outdated', null, { signal }),
    ]);
    if (!ctx.active) return;
    const st = status.status === 'fulfilled' ? status.value : null;
    const q = rq.status === 'fulfilled' ? rq.value : null;
    if (q) {
      const total = Object.values(q.queues || {}).reduce((a, x) => a + (x.queued || 0) + (x.started?.length || 0), 0);
      backlog.push(total);
      if (backlog.length > 90) backlog.shift();
    }
    renderBanner(st, q, status.reason || rq.reason);
    renderStats(st);
    renderWorkers(q, rq.reason);
    renderQueues(q, rq.reason);
    renderOutdated(outdated.status === 'fulfilled' ? outdated.value : null, outdated.reason);
    renderStorage(st, status.reason);
    if (!configLoaded) { configLoaded = true; loadConfig(); }
  }

  function renderBanner(st, q, err) {
    if (!st && !q) { mount(bannerEl, errorState(err)); return; }
    const workers = q?.workers || [];
    const alive = workers.filter((w) => heartbeatAge(w) < STALE_HEARTBEAT_S);
    const queued = Object.values(q?.queues || {}).reduce((a, x) => a + (x.queued || 0), 0);
    const ok = alive.length > 0;
    mount(bannerEl, h('div.rdh-banner', { class: ok ? 'ok' : 'bad' },
      h('span.rdh-banner-icon', icon(ok ? 'check-circle-fill' : 'exclamation-octagon-fill')),
      h('div.rdh-banner-text',
        h('div.rdh-banner-title', ok ? 'Redash is healthy' : alive.length === 0 && workers.length ? 'Workers stopped responding' : 'No workers are running'),
        h('div', ok
          ? `${alive.length} worker${alive.length === 1 ? '' : 's'} processing · ${fmtNumber(queued)} job${queued === 1 ? '' : 's'} waiting · Redash ${st?.version || ''}`
          : 'Queries, schema refreshes and scheduled jobs cannot run until a worker is started (docker: redash-worker).')),
      h('div.rdh-banner-spark', h('span.small.text-muted', 'Backlog this session'), sparkline(backlog.length > 1 ? backlog : [0, 0], { width: 160, height: 34, color: ok ? '#16a34a' : '#dc2626' }))));
  }

  function renderStats(st) {
    if (!st) { mount(statsEl); return; }
    const first = !statsEl.dataset.painted;
    const n = (x) => (first ? countUp(x ?? 0) : fmtNumber(x));
    mount(statsEl,
      statCard({ label: 'Version', value: st.version || '—', icon: 'tag', tone: 'blue' }),
      statCard({ label: 'Queries', value: n(st.queries_count), icon: 'code-square', tone: 'teal', href: urls.queries() }),
      statCard({ label: 'Dashboards', value: n(st.dashboards_count), icon: 'grid-1x2', tone: 'purple', href: urls.dashboards() }),
      statCard({ label: 'Widgets', value: n(st.widgets_count), icon: 'grid-3x2-gap', tone: 'slate' }),
      statCard({ label: 'Results', value: n(st.query_results_count), icon: 'archive', tone: 'amber', sub: `${fmtNumber(st.unused_query_results_count)} unused` }),
      statCard({ label: 'Redis', value: st.redis_used_memory_human || fmtBytes(st.redis_used_memory), icon: 'memory', tone: 'red', sub: 'memory used' }));
    statsEl.dataset.painted = '1';
  }

  function heartbeatAge(w) {
    const t = w.last_heartbeat ? new Date(`${w.last_heartbeat}Z`).getTime() : NaN;
    return Number.isFinite(t) ? (Date.now() - t) / 1000 : Infinity;
  }

  function renderWorkers(q, err) {
    if (!q) { mount(workersEl, errorState(err)); return; }
    const workers = q.workers || [];
    if (!workers.length) {
      mount(workersEl, emptyState({ icon: 'cpu', title: 'No workers', message: 'Start the redash-worker container so queries can execute.' }));
      return;
    }
    mount(workersEl, dataTable({
      rows: workers,
      rowKey: (w) => w.name,
      columns: [
        { key: 'state', label: 'State', render: (w) => {
          const age = heartbeatAge(w);
          const stale = age > STALE_HEARTBEAT_S;
          const busy = w.state === 'busy';
          return h('span.rdh-worker-state', { class: stale ? 'stale' : busy ? 'busy' : 'idle' }, h('span.rdh-pulse-dot'), stale ? 'not responding' : w.state);
        } },
        { key: 'name', label: 'Worker', render: (w) => h('div', h('div.af-mono', `${w.hostname}:${w.pid}`), h('div.small.text-muted', { title: w.name }, w.name.slice(0, 12))) },
        { key: 'queues', label: 'Queues', render: (w) => {
          const qs = String(w.queues || '').split(',').map((x) => x.trim()).filter(Boolean);
          const t = h('span.af-tag', { title: qs.join(', ') }, icon('stack'), ` ${qs.length} queue${qs.length === 1 ? '' : 's'}`);
          tip(t, () => h('div', h('div.af-tip-title', 'Listens on'), qs.map((x) => h('div', x))));
          return t;
        } },
        { key: 'current_job', label: 'Current job', render: (w) => {
          if (!w.current_job) return h('span.text-muted', 'waiting for work');
          const raw = typeof w.current_job === 'string' ? w.current_job : JSON.stringify(w.current_job);
          const m = raw.match(/^(\S+)\s*\(([^)]+)\)/);
          return m ? h('span', { title: raw }, h('b', m[2].replace(/_/g, ' ')), h('div.small.text-muted.af-mono', m[1].slice(0, 12))) : h('code.small', raw);
        } },
        { key: 'jobs', label: 'Jobs ok / failed', align: 'right', render: (w) => h('span.af-nowrap', h('b.text-success', fmtNumber(w.successful_jobs)), ' / ', h('b', { class: w.failed_jobs ? 'text-danger' : 'text-muted' }, fmtNumber(w.failed_jobs))) },
        { key: 'total_working_time', label: 'Busy time', align: 'right', render: (w) => fmtDuration(w.total_working_time) },
        { key: 'heartbeat', label: 'Heartbeat', render: (w) => h('span', { class: heartbeatAge(w) > STALE_HEARTBEAT_S ? 'text-danger fw-semibold' : '' }, timeAgo(`${w.last_heartbeat}Z`)) },
        { key: 'birth_date', label: 'Up since', render: (w) => timeAgo(`${w.birth_date}Z`) },
      ],
    }));
  }

  function renderQueues(q, err) {
    if (!q) { mount(queuesEl, errorState(err)); return; }
    const queues = Object.values(q.queues || {});
    const max = Math.max(1, ...queues.map((x) => (x.queued || 0) + (x.started?.length || 0)));
    mount(queuesEl, h('div.rdh-queues', queues.map((x) => {
      const started = x.started || [];
      const total = (x.queued || 0) + started.length;
      const bar = h('div.rdh-queue-bar',
        h('span.running', { style: { width: `${(started.length / max) * 100}%` } }),
        h('span.waiting', { style: { width: `${((x.queued || 0) / max) * 100}%` } }));
      tip(bar, `${started.length} running · ${x.queued || 0} waiting`);
      return h('div.rdh-queue',
        h('div.rdh-queue-head',
          h('span.fw-semibold', icon('stack', 'me-1'), x.name),
          h('span.rdh-queue-count', { class: total ? 'busy' : '' }, total ? `${started.length} running · ${x.queued || 0} waiting` : 'idle')),
        bar,
        started.length ? h('div.rdh-jobs', started.map((j) => jobRow(j))) : null);
    })));
  }

  function jobRow(j) {
    const meta = j.meta || {};
    const queryId = meta.query_id;
    const task = String(j.name || j.id).split('.').pop().replace(/_/g, ' ');
    return h('div.rdh-job',
      h('div.rdh-job-main',
        h('div.fw-semibold.small.text-capitalize', { title: j.name || j.id }, task),
        h('div.small.text-muted',
          'started ', timeAgo(`${j.started_at}Z`),
          queryId && h('span', ' · ', h('a', { href: urls.query(queryId) }, `query #${queryId}`)),
          meta.data_source_id && h('span', ' · ', h('a', { href: urls.dataSource(meta.data_source_id) }, `data source #${meta.data_source_id}`)),
          meta.username && h('span', ` · ${meta.username}`))),
      h('button.btn.btn-sm.btn-outline-danger', {
        title: 'Cancel this job',
        onclick: async () => {
          const ok = await confirmDialog({ title: 'Cancel this job?', danger: true, confirmLabel: 'Cancel job', message: h('span', 'The worker stops ', h('code', j.name || j.id), '. A query being executed will fail with "cancelled".') });
          if (!ok) return;
          try { await rd.del(`/api/jobs/${enc(j.id)}`); toast('Job cancelled', 'success'); load(); } catch (e) { toastError(e, 'Could not cancel the job'); }
        },
      }, icon('x-octagon', 'me-1'), 'Cancel'));
  }

  function renderOutdated(res, err) {
    if (!res) { mount(outdatedEl, errorState(err)); return; }
    const queries = res.queries || [];
    const updated = res.updated_at ? new Date(Number(res.updated_at) * 1000) : null;
    if (!queries.length) {
      mount(outdatedEl, emptyState({ icon: 'calendar-check', title: 'Every scheduled query is up to date', message: updated ? `Scheduler last checked ${updated.toLocaleTimeString()}.` : null }));
      return;
    }
    mount(outdatedEl,
      h('div.small.text-muted.px-3.pt-2', `Waiting for their next scheduled run · scheduler checked ${updated ? updated.toLocaleTimeString() : 'recently'}`),
      dataTable({
        rows: queries,
        rowKey: (x) => x.id,
        onRowClick: (x) => ctx.navigate(urls.query(x.id)),
        columns: [
          { key: 'name', label: 'Query', render: (x) => h('a', { href: urls.query(x.id) }, x.name || `Query #${x.id}`) },
          { key: 'schedule', label: 'Schedule', render: (x) => scheduleText(x.schedule) },
          { key: 'retrieved_at', label: 'Last run', render: (x) => timeAgo(x.retrieved_at) },
          { key: 'runtime', label: 'Runtime', align: 'right', render: (x) => fmtDuration(x.runtime) },
          { key: 'user', label: 'Owner', render: (x) => x.user?.name || '—' },
        ],
      }));
  }

  function renderStorage(st, err) {
    if (!st) { mount(storageEl, errorState(err)); return; }
    const metrics = st.database_metrics?.metrics || [];
    const max = Math.max(1, ...metrics.map((m) => m[1]));
    mount(storageEl, h('div.d-flex.flex-column.gap-3',
      metrics.map(([name, size]) => h('div',
        h('div.d-flex.justify-content-between.small.mb-1', h('span.fw-semibold', name), h('span', fmtBytes(size))),
        h('div.af-meter', h('span', { style: { width: `${(size / max) * 100}%`, background: 'var(--af-primary)' } })))),
      h('div',
        h('div.d-flex.justify-content-between.small.mb-1', h('span.fw-semibold', 'Redis memory'), h('span', st.redis_used_memory_human || fmtBytes(st.redis_used_memory))),
        h('div.small.text-muted', 'Queues: ', Object.entries(st.manager?.queues || {}).map(([k, v]) => `${k} ${v.size}`).join(' · ') || '—'))));
  }

  async function loadConfig() {
    try {
      const [cfg, session] = await Promise.all([rd.get('/api/config', null, { signal }), getSession()]);
      if (!ctx.active) return;
      const cc = cfg.client_config || {};
      const flags = Object.entries(cc).filter(([, v]) => typeof v === 'boolean');
      const interval = (s) => (s < 3600 ? `${s / 60}m` : s < 86400 ? `${s / 3600}h` : `${s / 86400}d`);
      mount(configEl, h('div.d-flex.flex-column.gap-3',
        kv([
          ['Organization', cfg.org_slug],
          ['Redash version', cc.version],
          ['Base URL', h('code', cc.basePath)],
          ['Date format', cc.dateFormat],
          ['Default page size', cc.pageSize],
          ['Signed in as', session?.user ? `${session.user.name} (${session.user.email})` : '—'],
        ], { cols: 3 }),
        h('div', h('div.af-form-section', 'Feature flags'), h('div.rdh-flags', flags.map(([k, v]) => h('span.rdh-flag', { class: v ? 'on' : 'off', title: k }, icon(v ? 'toggle-on' : 'toggle-off'), k.replace(/([A-Z])/g, ' $1').toLowerCase())))),
        h('div', h('div.af-form-section', 'Query refresh intervals'), h('div.af-tags', (cc.queryRefreshIntervals || []).map((s) => h('span.af-tag', interval(s))))),
        h('div', h('div.af-form-section', 'Dashboard refresh intervals'), h('div.af-tags', (cc.dashboardRefreshIntervals || []).map((s) => h('span.af-tag', interval(s))))),
        h('div', h('div.af-form-section', 'Your permissions'), h('div.af-tags', (session?.user?.permissions || []).map((p) => h('span.af-tag', p)))),
        h('details.rdh-raw', h('summary.small.text-muted', 'Raw client configuration'), jsonView(cc, { open: 1 }))));
    } catch (err) {
      if (err.name !== 'AbortError') mount(configEl, errorState(err));
    }
  }

  await load();
}
