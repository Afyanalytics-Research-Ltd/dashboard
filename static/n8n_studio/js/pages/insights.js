// Insights: n8n's summary (with deviations vs the previous period) for any range /
// project, plus Studio's per-workflow reliability & time analysis from executions.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, chips, dataTable, spinner, statCard } from '../../../ui/js/ui.js';
import { fmtDuration } from '../../../ui/js/format.js';
import { sparkline } from '../../../ui/js/charts.js';
import { n8n } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load, loadOptional } from '../core/store.js';
import { workflowStats } from '../core/execution.js';
import { licenseGate, workflowLink, emptyHint } from '../core/widgets.js';
import { fmtMinutes } from './home.js';

const RANGES = { '24h': 1, '7d': 7, '14d': 14, '30d': 30, '90d': 90 };

export default async function render(ctx) {
  let range = ctx.query.get('range') || '7d';
  let project = ctx.query.get('project') || '';
  const projects = await loadOptional('projects');
  const kpiEl = h('div.n8s-kpis');
  const tableEl = h('div');
  const head = h('div.d-flex.gap-2.align-items-center.flex-wrap');
  mount(ctx.el, pageHeader({ title: 'Insights', icon: 'graph-up-arrow', subtitle: 'Production runs, failures and time saved — with per-workflow reliability.', actions: head }), kpiEl,
    card({ title: 'Per-workflow reliability', icon: 'clipboard-data', body: tableEl }));

  const draw = () => mount(head,
    chips(Object.keys(RANGES).map((k) => ({ value: k, label: k })), range, (v) => { range = v || '7d'; ctx.setQuery({ range }); go(); }),
    projects.length > 1 && h('select.form-select.form-select-sm', { style: 'width:200px', onchange: (e) => { project = e.target.value; ctx.setQuery({ project: project || null }); go(); } },
      h('option', { value: '' }, 'All projects'), projects.map((p) => h('option', { value: p.id, selected: p.id === project }, p.name))));

  async function go() {
    draw();
    mount(kpiEl, spinner());
    const end = new Date();
    const start = new Date(Date.now() - RANGES[range] * 86400e3);
    try {
      const s = await n8n.get('/insights/summary', { startDate: start.toISOString(), endDate: end.toISOString(), projectId: project || undefined });
      const dev = (m, invert = false) => (m.deviation == null ? 'no prior data' : h('span', { class: (m.deviation > 0) !== invert ? 'text-success' : 'text-danger' }, `${m.deviation > 0 ? '▲' : '▼'} ${Math.abs(m.unit === 'ratio' ? Math.round(m.deviation * 1000) / 10 : Math.round(m.deviation))}${m.unit === 'ratio' ? ' pp' : m.unit === 'millisecond' ? ' ms' : ''} vs previous ${range}`));
      mount(kpiEl,
        statCard({ label: 'Production executions', value: s.total.value.toLocaleString(), sub: dev(s.total), icon: 'lightning-charge', tone: 'blue' }),
        statCard({ label: 'Failed', value: s.failed.value.toLocaleString(), sub: dev(s.failed, true), icon: 'x-octagon', tone: 'red' }),
        statCard({ label: 'Failure rate', value: `${(s.failureRate.value * 100).toFixed(2)}%`, sub: dev(s.failureRate, true), icon: 'percent', tone: 'amber' }),
        statCard({ label: 'Time saved', value: fmtMinutes(s.timeSaved.value), sub: dev(s.timeSaved), icon: 'hourglass-split', tone: 'green' }),
        statCard({ label: 'Average run time', value: fmtDuration(s.averageRunTime.value / 1000), sub: dev(s.averageRunTime, true), icon: 'stopwatch', tone: 'purple' }));
    } catch (err) {
      mount(kpiEl, err.license ? licenseGate(err) : h('div.text-danger', err.message));
    }
    mount(tableEl, spinner());
    const [wfs, execs] = await Promise.all([load('workflows').catch(() => []), n8n.all('/executions', { projectId: project || undefined }, { max: 2500 }).catch(() => [])]);
    const inRange = execs.filter((e) => new Date(e.startedAt || e.createdAt) >= start);
    const stats = workflowStats(inRange).sort((a, b) => b.total - a.total);
    const byWf = new Map(wfs.map((w) => [String(w.id), w]));
    const days = RANGES[range];
    const series = (wid) => {
      const arr = Array(Math.max(days, 2)).fill(0);
      for (const e of inRange) if (e.workflowId === wid) arr[Math.min(arr.length - 1, Math.floor((new Date(e.startedAt) - start) / 86400e3))]++;
      return arr;
    };
    mount(tableEl, stats.length ? h('div',
      execs.length >= 2500 && h('div.small.text-muted.mb-2', icon('info-circle', 'me-1'), 'Based on the most recent 2,500 executions.'),
      dataTable({
        rows: stats, dense: true, rowKey: (s) => s.workflowId,
        columns: [
          { key: 'wf', label: 'Workflow', render: (s) => workflowLink(byWf.get(String(s.workflowId)), { fallbackId: s.workflowId }) },
          { key: 'total', label: 'Runs', align: 'right', render: (s) => h('a', { href: urls.executions({ workflowId: s.workflowId, since: range === '90d' ? 'all' : range }) }, s.total) },
          { key: 'trend', label: 'Trend', render: (s) => sparkline(series(s.workflowId), { width: 110, height: 22 }) },
          { key: 'failed', label: 'Failed', align: 'right', render: (s) => (s.failed ? h('a.text-danger', { href: urls.executions({ workflowId: s.workflowId, status: 'error' }) }, s.failed) : '0') },
          { key: 'rate', label: 'Success', render: (s) => h('div.n8s-ratebar', h('span', { style: { width: `${(1 - s.failureRate) * 100}%` } }), h('b', `${Math.round((1 - s.failureRate) * 100)}%`)) },
          { key: 'p50', label: 'p50', align: 'right', render: (s) => (s.p50 != null ? fmtDuration(s.p50 / 1000) : '—') },
          { key: 'p95', label: 'p95', align: 'right', render: (s) => (s.p95 != null ? fmtDuration(s.p95 / 1000) : '—') },
          { key: 'saved', label: 'Time saved', align: 'right', render: (s) => { const per = byWf.get(String(s.workflowId))?.settings?.timeSavedPerExecution; return per ? fmtMinutes(per * (s.total - s.failed)) : h('span.text-muted.small', 'set in settings'); } },
          { key: 'last', label: 'Last', render: (s) => h('span.n8s-badge', { class: `n8s-st-${s.lastStatus}` }, s.lastStatus) },
        ],
      })) : emptyHint('graph-up', 'No executions in this range'));
  }

  go();
}

