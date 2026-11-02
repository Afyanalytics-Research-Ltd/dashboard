// Health: component status with live heartbeat ages, version and client-measured API latency.

import { h, icon, mount } from '../core/dom.js';
import { api, config } from '../core/api.js';
import { pageHeader, card, refreshControl, kv, errorState, skeleton, copyButton, statCard } from '../core/ui.js';
import { sparkline } from '../core/charts.js';
import { fmtDate } from '../core/format.js';

// Survives navigation within the session so the latency chart keeps its history.
const latencies = [];
const MAX_POINTS = 60;

const COMPONENTS = [
  { key: 'metadatabase', label: 'Metadata database', icon: 'database', desc: 'Postgres holding DAG runs, task state and config.' },
  { key: 'scheduler', label: 'Scheduler', icon: 'calendar2-week', hb: 'latest_scheduler_heartbeat', desc: 'Creates DAG runs and queues tasks.' },
  { key: 'triggerer', label: 'Triggerer', icon: 'lightning-charge', hb: 'latest_triggerer_heartbeat', desc: 'Runs async triggers for deferred tasks.' },
  { key: 'dag_processor', label: 'DAG processor', icon: 'file-earmark-code', hb: 'latest_dag_processor_heartbeat', desc: 'Parses DAG files into the metadata DB.' },
];

function ageText(iso) {
  const d = iso ? new Date(iso) : null;
  if (!d || Number.isNaN(d.getTime())) return '—';
  const s = Math.max(0, Math.round((Date.now() - d.getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s ago`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m ago`;
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  const overallEl = h('div', skeleton(1, { height: 64 }));
  const compEl = h('div.af-health-grid', skeleton(1, { height: 100 }));
  const latencyEl = h('div', skeleton(2));
  const versionEl = h('div', skeleton(3));

  const refresh = refreshControl(() => load(), { key: 'health', initial: 10 });
  ctx.cleanup(refresh.stop);
  // Tick heartbeat ages every second.
  const ticker = setInterval(() => {
    el.querySelectorAll('[data-hb]').forEach((n) => { n.textContent = ageText(n.dataset.hb); });
  }, 1000);
  ctx.cleanup(() => clearInterval(ticker));

  mount(el,
    pageHeader({ title: 'Health', icon: 'heart-pulse', subtitle: 'Status of Airflow components, checked through the dashboard proxy.', actions: [refresh.el] }),
    overallEl,
    compEl,
    h('div.af-grid-2',
      card({ title: 'API latency (this session)', icon: 'speedometer', body: latencyEl }),
      card({ title: 'Server', icon: 'hdd-network', body: versionEl })));

  api.get('/version', null, { signal }).then((v) => {
    if (!ctx.active) return;
    mount(versionEl, kv([
      ['Airflow version', h('span.fw-semibold', v.version)],
      ['Git version', v.git_version ? h('span.af-mono.small.d-inline-flex.gap-1.align-items-center', v.git_version, copyButton(v.git_version)) : null],
      ['Airflow URL', config.airflowUrl ? h('a', { href: config.airflowUrl, target: '_blank', rel: 'noopener' }, config.airflowUrl, icon('box-arrow-up-right', 'ms-1 small')) : null, { wide: true }],
      ['Proxy endpoint', h('code', config.api || '/pipelines/api/v2/'), { wide: true }],
      ['API docs', config.airflowUrl ? h('a', { href: `${config.airflowUrl}/docs`, target: '_blank', rel: 'noopener' }, 'Swagger UI', icon('box-arrow-up-right', 'ms-1 small')) : null],
      ['Signed in as', config.user],
    ]));
  }).catch((err) => { if (ctx.active && err.name !== 'AbortError') mount(versionEl, errorState(err)); });

  async function load() {
    const t0 = performance.now();
    let health;
    try {
      health = await api.get('/monitor/health', null, { signal });
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      latencies.push({ ms: null, at: Date.now() });
      mount(overallEl, banner(false, 'Airflow is unreachable', err.message));
      mount(compEl, COMPONENTS.map((c) => component(c, null)));
      drawLatency();
      return;
    }
    const ms = performance.now() - t0;
    latencies.push({ ms, at: Date.now() });
    if (latencies.length > MAX_POINTS) latencies.shift();
    if (!ctx.active) return;
    const bad = COMPONENTS.filter((c) => health[c.key]?.status !== 'healthy');
    mount(overallEl, bad.length
      ? banner(false, `${bad.length} component${bad.length > 1 ? 's' : ''} unhealthy`, bad.map((c) => c.label).join(', '))
      : banner(true, 'All systems operational', `Checked ${new Date().toLocaleTimeString()} · ${Math.round(ms)} ms round-trip`));
    mount(compEl, COMPONENTS.map((c) => component(c, health[c.key])));
    drawLatency();
  }

  function banner(ok, title, sub) {
    return h('div.sys-banner', { class: ok ? 'ok' : 'bad' },
      h('div.sys-banner-icon', icon(ok ? 'check-circle-fill' : 'exclamation-octagon-fill')),
      h('div', h('div.sys-banner-title', title), h('div.small', sub)));
  }

  function component(c, info) {
    const ok = info?.status === 'healthy';
    const hb = c.hb && info?.[c.hb];
    return h('div.af-health-item.sys-comp', { class: ok ? 'af-health-ok' : 'af-health-bad' },
      h('div.af-health-icon', icon(ok ? c.icon : 'exclamation-triangle-fill')),
      h('div.sys-comp-body',
        h('div.fw-semibold', c.label),
        h('span.sys-status', { class: ok ? 'ok' : 'bad' }, info?.status || 'unknown'),
        h('div.small.text-muted', c.desc),
        c.hb && h('div.small.mt-1', icon('activity', 'me-1'), 'heartbeat ',
          hb ? h('span.fw-semibold', { dataset: { hb }, title: fmtDate(hb) }, ageText(hb)) : h('span.text-muted', 'none'))));
  }

  function drawLatency() {
    const values = latencies.filter((l) => l.ms != null).map((l) => l.ms);
    if (!values.length) { mount(latencyEl, h('div.small.text-muted', 'No successful checks yet.')); return; }
    const sorted = [...values].sort((a, b) => a - b);
    const p = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
    const failures = latencies.filter((l) => l.ms == null).length;
    mount(latencyEl,
      values.length > 1
        ? h('div.sys-latency-chart', sparkline(values, { width: 520, height: 70, color: '#0072CE' }))
        : h('div.sys-latency-chart.sys-latency-wait', icon('hourglass-split', 'me-1'), 'The trend appears after the next check…'),
      h('div.af-stats.mt-2',
        statCard({ label: 'Last', value: `${Math.round(values[values.length - 1])} ms`, icon: 'stopwatch', tone: 'blue' }),
        statCard({ label: 'Median', value: `${Math.round(p(0.5))} ms`, icon: 'bar-chart', tone: 'teal' }),
        statCard({ label: 'p95', value: `${Math.round(p(0.95))} ms`, icon: 'graph-up', tone: 'purple' }),
        statCard({ label: 'Failed checks', value: failures, icon: 'x-octagon', tone: failures ? 'red' : 'green' })),
      h('div.small.text-muted.mt-2', `${latencies.length} check${latencies.length === 1 ? '' : 's'} since this tab opened (browser → dashboard → Airflow).`));
  }

  await load();
}
