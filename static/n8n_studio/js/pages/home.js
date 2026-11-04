// Overview: KPIs (n8n insights + live execution stats), 24h activity, failing
// workflows, running now, recently edited, and instance health.

import { h, s, icon, mount } from '../../../ui/js/dom.js';
import { card, timeAgo, spinner, statCard } from '../../../ui/js/ui.js';
import { fmtDuration } from '../../../ui/js/format.js';
import { tip, tipRows, donut, legend } from '../../../ui/js/charts.js';
import { n8n, studio, config } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load } from '../core/store.js';
import { loadCatalog, nodeIcon } from '../core/catalog.js';
import { FAILED, ACTIVE, timeBuckets, workflowStats, durationMs, statusInfo } from '../core/execution.js';
import { execBadge, activeSwitch, emptyHint } from '../core/widgets.js';
import { isSticky } from '../core/workflow.js';

export default async function render(ctx) {
  await loadCatalog().catch(() => null);
  const kpis = h('div.n8s-kpis', spinner());
  const activity = h('div');
  const failing = h('div');
  const running = h('div');
  const recent = h('div');
  const mix = h('div');
  const healthEl = h('div');
  const hour = new Date().getHours();
  mount(ctx.el,
    h('div.n8s-hero',
      h('div', h('div.n8s-hero-hi', `${hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'}, ${config.user || 'there'}`),
        h('div.n8s-hero-sub', 'Here is what your automations have been up to.')),
      h('div.d-flex.gap-2',
        h('a.btn.btn-primary', { href: urls.newWorkflow() }, icon('plus-lg', 'me-1'), 'New workflow'),
        h('a.btn.btn-outline-light', { href: urls.failures() }, icon('inbox', 'me-1'), 'Failure inbox'))),
    kpis,
    h('div.n8s-home-grid',
      card({ title: 'Last 24 hours', icon: 'activity', body: activity, cls: 'n8s-span-2' }),
      card({ title: 'Failing workflows', icon: 'x-octagon', body: failing, actions: h('a.btn.btn-sm.btn-link', { href: urls.failures() }, 'Inbox →') }),
      card({ title: 'Running now', icon: 'arrow-repeat', body: running }),
      card({ title: 'Recently edited', icon: 'pencil-square', body: recent }),
      card({ title: 'Execution mix', icon: 'pie-chart', body: mix }),
      card({ title: 'Instance', icon: 'heart-pulse', body: healthEl, actions: h('a.btn.btn-sm.btn-link', { href: urls.system() }, 'Details →') })));

  const [wfs, page, insights, health] = await Promise.all([
    load('workflows', { signal: ctx.signal }).catch(() => []),
    n8n.get('/executions', { limit: 250 }, { signal: ctx.signal }).catch(() => ({ data: [] })),
    n8n.get('/insights/summary', { startDate: new Date(Date.now() - 7 * 86400e3).toISOString(), endDate: new Date().toISOString() }).catch((err) => ({ error: err })),
    studio.get('/health/').catch(() => null),
  ]);
  const execs = page.data || [];
  const day = execs.filter((e) => Date.now() - new Date(e.startedAt || e.createdAt).getTime() < 86400e3);
  const fails = day.filter((e) => FAILED.has(e.status));
  const name = (id) => wfs.find((w) => String(w.id) === String(id))?.name || id;

  const ins = insights && !insights.error ? insights : null;
  const dev = (m) => (m?.deviation == null ? null : h('span', { class: m.deviation > 0 ? 'up' : 'down' }, `${m.deviation > 0 ? '▲' : '▼'} ${Math.abs(Math.round(m.unit === 'ratio' ? m.deviation * 100 : m.deviation))}${m.unit === 'ratio' ? 'pp' : ''}`));
  mount(kpis,
    statCard({ label: 'Workflows', value: String(wfs.filter((w) => !w.isArchived).length), sub: `${wfs.filter((w) => w.active).length} active`, icon: 'diagram-3', tone: 'blue', href: urls.workflows() }),
    statCard({ label: 'Runs · 24h', value: day.length.toLocaleString() + (execs.length >= 250 && day.length === execs.length ? '+' : ''), sub: `${fails.length} failed`, icon: 'lightning-charge', tone: fails.length ? 'red' : 'teal', href: urls.executions({ since: '24h' }) }),
    statCard({ label: 'Runs · 7d (insights)', value: ins ? ins.total.value.toLocaleString() : '—', sub: ins ? h('span.n8s-dev', dev(ins.total), ' vs prior week') : 'insights unavailable', icon: 'graph-up', tone: 'purple', href: urls.insights() }),
    statCard({ label: 'Failure rate · 7d', value: ins ? `${(ins.failureRate.value * 100).toFixed(1)}%` : '—', sub: ins && h('span.n8s-dev', dev(ins.failureRate)), icon: 'shield-exclamation', tone: ins && ins.failureRate.value > 0.05 ? 'red' : 'green', href: urls.insights() }),
    statCard({ label: 'Time saved · 7d', value: ins ? fmtMinutes(ins.timeSaved.value) : '—', sub: ins && h('span.n8s-dev', dev(ins.timeSaved)), icon: 'hourglass-split', tone: 'amber', href: urls.insights() }),
    statCard({ label: 'Avg run time · 7d', value: ins ? fmtDuration(ins.averageRunTime.value / 1000) : '—', sub: ins && h('span.n8s-dev', dev(ins.averageRunTime)), icon: 'stopwatch', tone: 'blue', href: urls.insights() }));

  // activity chart (hourly, 24h)
  const buckets = timeBuckets(day, { buckets: 24, span: 86400e3 });
  const max = Math.max(1, ...buckets.map((b) => b.success + b.error + b.other));
  const W = 720;
  const H = 150;
  const bw = W / 24;
  mount(activity, day.length ? h('div',
    s('svg', { viewBox: `0 0 ${W} ${H + 18}`, class: 'n8s-tl n8s-tl-big' },
      [0.25, 0.5, 0.75, 1].map((p) => s('line', { x1: 0, x2: W, y1: H - p * (H - 8), y2: H - p * (H - 8), class: 'grid' })),
      buckets.map((b, i) => {
        let y = H;
        const seg = (n, cls) => { if (!n) return null; const hh = (n / max) * (H - 8); y -= hh; return s('rect', { x: i * bw + 3, y, width: bw - 6, height: hh, class: cls, rx: 3 }); };
        const g = s('g', {}, seg(b.success, 'ok'), seg(b.error, 'err'), seg(b.other, 'other'), s('rect', { x: i * bw, y: 0, width: bw, height: H, fill: 'transparent' }),
          i % 3 === 0 && s('text', { x: i * bw + bw / 2, y: H + 14, class: 'lbl' }, new Date(b.t).getHours().toString().padStart(2, '0')));
        tip(g, () => tipRows(`${new Date(b.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} – +1h`, [['Succeeded', b.success], ['Failed', b.error], ['Other', b.other]]));
        return g;
      })),
    h('div.small.text-muted', `${day.length} runs · ${fails.length} failed · median ${fmtDuration(median(day.filter((e) => e.stoppedAt).map(durationMs)) / 1000)}`))
    : emptyHint('moon-stars', 'Quiet day', 'No executions in the last 24 hours.'));

  const stats = workflowStats(day).filter((x) => x.failed).sort((a, b) => b.failed - a.failed).slice(0, 6);
  mount(failing, stats.length ? h('div.n8s-list', stats.map((x) => h('a.n8s-list-row', { href: urls.executions({ workflowId: x.workflowId, status: 'error', since: '24h' }) },
    h('span.n8s-fail-pill', `${x.failed}✕`), h('span.flex-grow-1.text-truncate', name(x.workflowId)), h('span.small.text-muted', `${Math.round(x.failureRate * 100)}% of ${x.total}`))))
    : emptyHint('emoji-smile', 'No failures', 'Nothing failed in the last 24 hours.'));

  const live = execs.filter((e) => ACTIVE.has(e.status)).slice(0, 8);
  mount(running, live.length ? h('div.n8s-list', live.map((e) => h('a.n8s-list-row', { href: urls.execution(e.id) }, execBadge(e.status, { size: 'sm' }), h('span.flex-grow-1.text-truncate', name(e.workflowId)), timeAgo(e.startedAt))))
    : emptyHint('pause-circle', 'Nothing running', null));

  const edited = [...wfs].filter((w) => !w.isArchived).sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || '')).slice(0, 7);
  mount(recent, edited.length ? h('div.n8s-list', edited.map((w) => h('div.n8s-list-row',
    h('span.n8s-strip', [...new Set((w.nodes || []).filter((n) => !isSticky(n)).map((n) => n.type))].slice(0, 3).map((t) => nodeIcon(t, { size: 18 }))),
    h('a.flex-grow-1.text-truncate', { href: urls.workflow(w.id) }, w.name), h('span.small.text-muted', timeAgo(w.updatedAt)), activeSwitch(w))))
    : emptyHint('diagram-3', 'No workflows', h('a', { href: urls.newWorkflow() }, 'Create your first one')));

  const byMode = {};
  for (const e of day) byMode[e.mode] = (byMode[e.mode] || 0) + 1;
  const palette = ['#0072CE', '#0BB99F', '#ff6d5a', '#8b5cf6', '#f59e0b', '#64748b', '#ec4899', '#22c55e'];
  const segs = Object.entries(byMode).sort((a, b) => b[1] - a[1]).map(([k, v], i) => ({ label: k, value: v, color: palette[i % palette.length] }));
  const byStatus = {};
  for (const e of day) byStatus[e.status] = (byStatus[e.status] || 0) + 1;
  mount(mix, segs.length ? h('div.d-flex.gap-3.align-items-center.flex-wrap',
    donut(segs, { size: 130, thickness: 18, centerLabel: String(day.length), centerSub: 'runs' }),
    h('div', legend(segs), h('div.mt-2', Object.entries(byStatus).map(([k, v]) => h('div.small', h('span.n8s-dot', { class: `n8s-st-${k}` }), ` ${statusInfo(k).label}: ${v}`)))))
    : emptyHint('pie-chart', 'No data', null));

  mount(healthEl, health ? h('div.n8s-health-card',
    row('Instance', health.instance?.ok ? 'healthy' : health.instance?.reachable ? 'unhealthy' : 'unreachable', health.instance?.ok),
    row('API key', health.key?.ok ? `valid · ${health.key.ms} ms` : health.key?.error || 'not configured', health.key?.ok),
    row('Manual runs', health.login?.configured ? (health.login.ok ? 'enabled' : health.login.error || 'login failed') : 'off (set N8N_STUDIO_EMAIL)', health.login?.configured && health.login.ok),
    row('Node catalog', `${health.catalog?.nodes} types · ${health.catalog?.source}`, true),
    row('API coverage', `${health.operations} operations · spec ${health.specVersion}`, true))
    : h('div.text-muted.small', 'Health endpoint unavailable.'));
}

function row(label, value, ok) {
  return h('div.n8s-health-row', h('span.n8s-dot', { class: ok ? 'n8s-st-success' : 'n8s-st-error' }), h('span.n8s-health-k', label), h('span.n8s-health-v', value));
}

function median(v) {
  const a = v.filter((x) => x != null).sort((x, y) => x - y);
  return a.length ? a[Math.floor(a.length / 2)] : null;
}

export function fmtMinutes(min) {
  if (min == null) return '—';
  if (min < 60) return `${Math.round(min)} min`;
  if (min < 60 * 24) return `${(min / 60).toFixed(1)} h`;
  return `${(min / 60 / 24).toFixed(1)} d`;
}

