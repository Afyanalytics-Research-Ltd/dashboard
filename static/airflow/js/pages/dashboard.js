// Overview: health, headline numbers, run outcomes, activity, failures, running now.

import { h, s, icon, mount } from '../core/dom.js';
import { api } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../core/router.js';
import {
  pageHeader, card, statCard, stateBadge, stateInfo, timeAgo, refreshControl, emptyState, errorState,
  chips, skeleton, openMenu, RUN_STATES,
} from '../core/ui.js';
import { donut, legend, tip, tipRows, stateBar } from '../core/charts.js';
import { fmtDuration, liveDuration, fmtNumber, fmtPercent, toDate } from '../core/format.js';
import { triggerDag } from '../core/actions.js';

const WINDOWS = [
  { value: '24h', label: '24 hours', hours: 24 },
  { value: '7d', label: '7 days', hours: 24 * 7 },
  { value: '30d', label: '30 days', hours: 24 * 30 },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  let windowKey = ctx.query.get('window') || '24h';
  const since = () => new Date(Date.now() - WINDOWS.find((w) => w.value === windowKey).hours * 3600e3).toISOString();

  const healthEl = h('div.af-health-grid', skeleton(1, { height: 60 }));
  const statsEl = h('div.af-stats', skeleton(1, { height: 70 }));
  const outcomeEl = h('div', skeleton(4));
  const activityEl = h('div', skeleton(4));
  const failuresEl = h('div', skeleton(5));
  const runningEl = h('div', skeleton(4));
  const topEl = h('div', skeleton(6));
  const poolsEl = h('div', skeleton(3));

  const refresh = refreshControl(() => load(), { key: 'dashboard', initial: 30 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Pipelines overview', icon: 'speedometer2',
      subtitle: 'Live state of every Airflow pipeline, refreshed automatically.',
      actions: [
        chips(WINDOWS, windowKey, (v) => { windowKey = v || '24h'; ctx.setQuery({ window: windowKey === '24h' ? null : windowKey }); load(); }),
        h('button.btn.btn-sm.btn-primary', { onclick: pickAndTrigger }, icon('play-fill', 'me-1'), 'Trigger DAG'),
        refresh.el,
      ],
    }),
    healthEl,
    statsEl,
    h('div.af-grid-main-side',
      card({ title: 'Run activity', icon: 'bar-chart', body: activityEl }),
      card({ title: 'Run outcomes', icon: 'pie-chart', body: outcomeEl })),
    h('div.af-grid-2',
      card({ title: 'Running now', icon: 'arrow-repeat', body: runningEl, flush: true,
        actions: h('a.af-link-btn', { href: urls.runs({ state: 'running' }) }, 'All running →') }),
      card({ title: 'Recent failures', icon: 'x-octagon', body: failuresEl, flush: true,
        actions: h('a.af-link-btn', { href: urls.runs({ state: 'failed' }) }, 'All failures →') })),
    h('div.af-grid-main-side',
      card({ title: 'DAG reliability (all-time runs)', icon: 'trophy', body: topEl, flush: true }),
      card({ title: 'Pool usage', icon: 'water', body: poolsEl,
        actions: h('a.af-link-btn', { href: urls.pools() }, 'Manage →') })));

  async function pickAndTrigger(e) {
    const anchor = e.currentTarget;
    const dags = await api.all('/dags', 'dags', { order_by: 'dag_id', exclude_stale: true }, { signal });
    openMenu(anchor, [
      { header: 'Choose a DAG to trigger' },
      ...dags.slice(0, 60).map((d) => ({
        label: d.dag_id, icon: d.is_paused ? 'pause-circle' : 'play-circle',
        onClick: () => triggerDag(d.dag_id),
      })),
      dags.length > 60 && { label: `…${dags.length - 60} more — use Ctrl+K`, disabled: true },
    ]);
  }

  async function load() {
    const sinceIso = since();
    const [health, dags, active, importErrs, stateCounts, recentRuns, failures, running, dagStats, pools, queuedTis] = await Promise.allSettled([
      api.get('/monitor/health', null, { signal }),
      api.get('/dags', { limit: 1 }, { signal }),
      api.get('/dags', { limit: 1, paused: false }, { signal }),
      api.get('/importErrors', { limit: 1 }, { signal }),
      Promise.all(RUN_STATES.map((st) => api.get('/dags/~/dagRuns', { limit: 1, state: st, run_after_gte: sinceIso }, { signal }).then((r) => [st, r.total_entries]))),
      api.all('/dags/~/dagRuns', 'dag_runs', { run_after_gte: sinceIso, order_by: '-run_after' }, { max: 1000, signal }),
      api.get('/dags/~/dagRuns', { limit: 8, state: 'failed', order_by: '-run_after' }, { signal }),
      api.get('/dags/~/dagRuns', { limit: 20, state: ['running', 'queued'], order_by: '-run_after' }, { signal }),
      api.all('/dagStats', 'dags', {}, { signal }),
      api.get('/pools', { limit: 100 }, { signal }),
      api.get('/dags/~/dagRuns/~/taskInstances', { limit: 1, state: ['queued', 'scheduled'] }, { signal }),
    ]);
    if (!ctx.active) return;
    const val = (r) => (r.status === 'fulfilled' ? r.value : null);

    renderHealth(val(health), health.reason);
    const counts = Object.fromEntries(val(stateCounts) || []);
    const totalRuns = Object.values(counts).reduce((a, b) => a + b, 0);
    const totalDags = val(dags)?.total_entries;
    const activeDags = val(active)?.total_entries;
    const windowLabel = WINDOWS.find((w) => w.value === windowKey).label;
    mount(statsEl,
      statCard({ label: 'DAGs', value: fmtNumber(totalDags), icon: 'diagram-3', tone: 'blue', href: urls.dags(), sub: activeDags != null ? `${fmtNumber(activeDags)} active · ${fmtNumber(totalDags - activeDags)} paused` : null }),
      statCard({ label: 'Running now', value: fmtNumber(val(running)?.dag_runs.filter((r) => r.state === 'running').length), icon: 'arrow-repeat', tone: 'cyan', href: urls.runs({ state: 'running' }) }),
      statCard({ label: `Runs · ${windowLabel}`, value: fmtNumber(totalRuns), icon: 'play-circle', tone: 'teal', href: urls.runs({ since: windowKey }) }),
      statCard({ label: `Failed · ${windowLabel}`, value: fmtNumber(counts.failed), icon: 'x-octagon', tone: counts.failed ? 'red' : 'green', href: urls.runs({ state: 'failed', since: windowKey }), sub: totalRuns ? `${fmtPercent(counts.success || 0, totalRuns, 1)} success rate` : null }),
      statCard({ label: 'Queued tasks', value: fmtNumber(val(queuedTis)?.total_entries), icon: 'hourglass-split', tone: 'slate', href: urls.taskInstances({ state: 'queued' }) }),
      statCard({ label: 'Import errors', value: fmtNumber(val(importErrs)?.total_entries), icon: 'bug', tone: val(importErrs)?.total_entries ? 'red' : 'green', href: urls.importErrors() }));

    renderOutcome(counts, totalRuns);
    renderActivity(val(recentRuns) || [], WINDOWS.find((w) => w.value === windowKey).hours);
    renderRunList(failuresEl, val(failures)?.dag_runs, failures.reason, { title: 'No failed runs', icon: 'emoji-smile', message: 'Nothing has failed recently.' });
    renderRunList(runningEl, val(running)?.dag_runs, running.reason, { title: 'Nothing running', icon: 'cup-hot', message: 'All pipelines are idle right now.' });
    renderTop(val(dagStats), dagStats.reason);
    renderPools(val(pools)?.pools, pools.reason);
  }

  function renderHealth(health, err) {
    if (!health) { mount(healthEl, errorState(err || new Error('Health endpoint unavailable'))); return; }
    const parts = [
      ['metadatabase', 'Metadata DB', 'database'],
      ['scheduler', 'Scheduler', 'calendar2-week', 'latest_scheduler_heartbeat'],
      ['triggerer', 'Triggerer', 'lightning-charge', 'latest_triggerer_heartbeat'],
      ['dag_processor', 'DAG processor', 'file-earmark-code', 'latest_dag_processor_heartbeat'],
    ];
    mount(healthEl, parts.map(([key, label, ic, hb]) => {
      const st = health[key]?.status;
      const ok = st === 'healthy';
      return h('a.af-health-item', { class: ok ? 'af-health-ok' : 'af-health-bad', href: urls.health(), style: { color: 'inherit' } },
        h('div.af-health-icon', icon(ok ? ic : 'exclamation-triangle-fill')),
        h('div', h('div.fw-semibold', label),
          h('div.small', { class: ok ? 'text-success' : 'text-danger' }, st || 'unknown'),
          hb && health[key]?.[hb] && h('div.small.text-muted', 'heartbeat ', timeAgo(health[key][hb]))));
    }));
  }

  function renderOutcome(counts, total) {
    if (!total) { mount(outcomeEl, emptyState({ icon: 'pie-chart', title: 'No runs in this window' })); return; }
    const segs = RUN_STATES.map((st) => ({ label: stateInfo(st).label, value: counts[st] || 0, color: stateInfo(st).color, state: st }));
    const go = (seg) => navigate(urls.runs({ state: seg.state, since: windowKey }));
    mount(outcomeEl, h('div.d-flex.flex-column.align-items-center.gap-3',
      donut(segs, { size: 170, centerLabel: fmtPercent(counts.success || 0, total), centerSub: 'success', onClick: go }),
      legend(segs, { onClick: go })));
  }

  function renderActivity(runs, hours) {
    if (!runs.length) { mount(activityEl, emptyState({ icon: 'bar-chart', title: 'No runs in this window' })); return; }
    const buckets = hours <= 24 ? 24 : hours <= 168 ? 28 : 30;
    const span = (hours * 3600e3) / buckets;
    const start = Date.now() - hours * 3600e3;
    const bins = Array.from({ length: buckets }, (_, i) => ({ t: start + i * span, counts: {} }));
    for (const r of runs) {
      const t = toDate(r.run_after)?.getTime();
      if (t == null) continue;
      const i = Math.min(buckets - 1, Math.max(0, Math.floor((t - start) / span)));
      bins[i].counts[r.state] = (bins[i].counts[r.state] || 0) + 1;
    }
    const max = Math.max(1, ...bins.map((b) => Object.values(b.counts).reduce((a, c) => a + c, 0)));
    const W = 560; const H = 170; const pad = 28; const bw = (W - pad) / buckets;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H + 20}`, width: '100%', class: 'af-activity' });
    const ticks = [...new Set([0, Math.round(max / 2), max])];
    ticks.forEach((v) => {
      const y = H - (v / max) * (H - 10);
      svg.appendChild(s('line', { x1: pad, x2: W, y1: y, y2: y, stroke: '#eef2f7' }));
      svg.appendChild(s('text', { x: pad - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#94a3b8' }, String(v)));
    });
    bins.forEach((b, i) => {
      let y = H;
      const g = s('g', { style: 'cursor:pointer' });
      for (const st of ['success', 'failed', 'running', 'queued']) {
        const n = b.counts[st] || 0;
        if (!n) continue;
        const hh = (n / max) * (H - 10);
        y -= hh;
        g.appendChild(s('rect', { x: pad + i * bw + 2, y, width: Math.max(bw - 4, 2), height: hh, rx: 2, fill: stateInfo(st).color, opacity: 0.9 }));
      }
      g.appendChild(s('rect', { x: pad + i * bw, y: 0, width: bw, height: H, fill: 'transparent' }));
      const d = new Date(b.t);
      const label = hours <= 24 ? d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      tip(g, () => tipRows(label, ['success', 'failed', 'running', 'queued'].map((st) => [stateInfo(st).label, b.counts[st] || 0])));
      if (i % Math.ceil(buckets / 6) === 0) svg.appendChild(s('text', { x: pad + i * bw + bw / 2, y: H + 15, 'text-anchor': 'middle', 'font-size': 11, fill: '#94a3b8' }, label));
      svg.appendChild(g);
    });
    mount(activityEl, svg, h('div.small.text-muted.mt-1', `${fmtNumber(runs.length)} runs${runs.length >= 1000 ? ' (first 1,000 shown)' : ''}`));
  }

  function renderRunList(target, runs, err, empty) {
    if (!runs) { mount(target, errorState(err)); return; }
    if (!runs.length) { mount(target, emptyState(empty)); return; }
    mount(target, h('div.af-list', runs.map((r) => h('a.af-list-item', { href: urls.run(r.dag_id, r.dag_run_id) },
      stateBadge(r.state),
      h('div.af-list-main',
        h('div.af-list-title', r.dag_display_name || r.dag_id),
        h('div.af-list-sub.af-mono', r.dag_run_id)),
      h('div.text-end.small.af-nowrap',
        h('div', fmtDuration(liveDuration(r.start_date, r.end_date, r.duration))),
        h('div.text-muted', timeAgo(r.start_date || r.queued_at)))))));
  }

  function renderTop(stats, err) {
    if (!stats) { mount(topEl, errorState(err)); return; }
    const rows = stats.map((d) => {
      const c = Object.fromEntries((d.stats || []).map((x) => [x.state, x.count]));
      const total = Object.values(c).reduce((a, b) => a + b, 0);
      return { ...d, c, total, rate: total ? (c.success || 0) / total : 1 };
    }).filter((d) => d.total).sort((a, b) => (b.c.failed || 0) - (a.c.failed || 0) || a.rate - b.rate).slice(0, 10);
    if (!rows.length) { mount(topEl, emptyState({ title: 'No run history yet' })); return; }
    mount(topEl, h('div.af-list', rows.map((d) => h('a.af-list-item', { href: urls.dag(d.dag_id) },
      h('div.af-list-main',
        h('div.af-list-title', d.dag_display_name || d.dag_id),
        h('div.mt-1', stateBar(d.c))),
      h('div.text-end.small.af-nowrap', { style: { minWidth: '92px' } },
        h('div.fw-semibold', { class: d.rate < 0.8 ? 'text-danger' : d.rate < 0.95 ? 'text-warning' : 'text-success' }, fmtPercent(d.c.success || 0, d.total, 1)),
        h('div.text-muted', `${fmtNumber(d.c.failed || 0)} failed / ${fmtNumber(d.total)}`))))));
  }

  function renderPools(pools, err) {
    if (!pools) { mount(poolsEl, errorState(err)); return; }
    mount(poolsEl, h('div.d-flex.flex-column.gap-3', pools.map((p) => {
      const slots = p.slots < 0 ? Infinity : p.slots;
      const used = (p.running_slots || 0) + (p.queued_slots || 0) + (p.deferred_slots || 0) + (p.scheduled_slots || 0);
      const seg = (n, color, label) => {
        if (!n || slots === Infinity) return null;
        const e = h('span', { style: { width: `${(n / slots) * 100}%`, background: color } });
        tip(e, `${label}: ${n}`);
        return e;
      };
      return h('div',
        h('div.d-flex.justify-content-between.small.mb-1',
          h('span.fw-semibold', p.name),
          h('span.text-muted', `${used} / ${slots === Infinity ? '∞' : slots} slots`)),
        h('div.af-meter',
          seg(p.running_slots, stateInfo('running').color, 'Running'),
          seg(p.queued_slots, stateInfo('queued').color, 'Queued'),
          seg(p.scheduled_slots, stateInfo('scheduled').color, 'Scheduled'),
          seg(p.deferred_slots, stateInfo('deferred').color, 'Deferred')));
    })));
  }

  await load();
}
