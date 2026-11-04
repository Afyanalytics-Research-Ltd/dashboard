// Performance & Optimizer — the cost & performance stored procedures, live.
//
// Every detector card runs its procedure in HOSPITALS.ANALYTICS (via the
// Studio API, asynchronously), animates while Snowflake works, and feeds the
// findings explorer.  Findings can be triaged (acknowledge / in progress /
// resolve / ignore) — Snowsight has nothing like it.

import { h, icon, mount, card, ring, seg, tabs, chip, sevBadge, banner, toolkitBanner, errorView, empty, copyButton,
  modal, confirm, kv, skeleton, skeletonCards, toast, spinner, animatedValue, healthLabel, popover, table } from '../lib/ui.js';
import { api, waitForQuery, sleep } from '../lib/api.js';
import { state } from '../lib/store.js';
import * as fmt from '../lib/fmt.js';
import { fixesToReach } from '../lib/metrics.js';
import { chart, axisStyle, tipRow, escapeHtml, STATUS, treemap, scatter, tableToggle } from '../lib/charts.js';
import { openQuery } from '../lib/query.js';

const SEVS = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
const TRIAGE = [
  { key: 'open', label: 'Open', icon: 'circle' },
  { key: 'acknowledged', label: 'Acknowledged', icon: 'eye' },
  { key: 'in_progress', label: 'In progress', icon: 'tools' },
  { key: 'resolved', label: 'Resolved', icon: 'check2-circle' },
  { key: 'ignored', label: 'Ignored', icon: 'slash-circle' },
];
const CATS = [
  { value: 'all', label: 'All' }, { value: 'performance', label: 'Performance', icon: 'speedometer' },
  { value: 'cost', label: 'Cost', icon: 'cash-coin' }, { value: 'storage', label: 'Storage', icon: 'hdd-stack' },
  { value: 'concurrency', label: 'Concurrency', icon: 'stoplights' },
];
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const list = (v) => (v ? String(v).split(',').filter(Boolean) : []);

export default async function render(ctx) {
  ctx.setTitle('Performance & Optimizer');
  const q = ctx.query;
  const view = {
    days: q.get('days') || '7',
    tab: q.get('tab') || 'findings',
    cat: q.get('cat') || 'all',
    severity: list(q.get('severity')),
    detector: list(q.get('detector')),
    triage: list(q.get('triage')),
    q: q.get('q') || '',
    sort: q.get('sort') || 'severity',
    limit: 40,
  };
  const sync = () => ctx.setQuery({ days: view.days !== '7' ? view.days : '', tab: view.tab !== 'findings' ? view.tab : '', cat: view.cat !== 'all' ? view.cat : '',
    severity: view.severity.join(','), detector: view.detector.join(','), triage: view.triage.join(','), q: view.q, sort: view.sort !== 'severity' ? view.sort : '' });

  mount(ctx.el, skeletonCards(3, 180), h('div.ws-mt'), skeletonCards(4, 150));
  let catalog;
  try {
    catalog = await api.optimizer(false, { signal: ctx.signal });
  } catch (err) {
    if (err.name === 'AbortError') return;
    mount(ctx.el, errorView(err, () => render(ctx)));
    return;
  }
  const installed = !!catalog.toolkit?.toolkit_installed;
  const running = new Map();            // detector key -> {runId}
  const cardEls = new Map();
  let summary = null;
  let pageRows = [];
  let total = 0;
  const selected = new Set();
  const openRows = new Set();

  // ---------------------------------------------------------------- layout skeleton
  const heroEl = h('div');
  const progressEl = h('div');
  const catSeg = seg(CATS, view.cat, (v) => { view.cat = v; sync(); renderDetectors(); });
  const detectorsEl = h('div.wo-detectors');
  const analyticsEl = h('div.ws-grid.ws-g-12');
  const tabBodyEl = h('div');
  const tabBar = tabs([
    { key: 'findings', label: 'Findings', icon: 'list-check' },
    { key: 'reports', label: 'Reports', icon: 'clipboard2-pulse' },
    { key: 'runs', label: 'Runs & schedule', icon: 'calendar2-week' },
    { key: 'about', label: 'How it works', icon: 'info-circle' },
  ], view.tab, (t) => { view.tab = t; sync(); renderTab(); });

  mount(ctx.el,
    !installed && toolkitBanner(catalog.toolkit?.error ? `Could not check the toolkit: ${catalog.toolkit.error}` : null),
    heroEl,
    h('div.ws-mt'),
    progressEl,
    h('div.ws-row.ws-between.ws-mb', { style: { marginTop: '22px' } },
      h('div', h('h2.ws-card-title', { style: { fontSize: '18px' } }, icon('cpu'), 'Detectors'),
        h('div.ws-card-sub', `${catalog.detectors.length} stored procedures in ${catalog.toolkit?.toolkit || 'the toolkit schema'} — each scans account history and records findings`)),
      h('div.ws-row', catSeg)),
    detectorsEl,
    h('div', { style: { height: '22px' } }),
    analyticsEl,
    h('div', { style: { height: '22px' } }),
    tabBar,
    tabBodyEl);

  renderHero();
  renderDetectors();
  renderTab();
  await reload();
  if (q.get('run') === 'all' && installed) {
    ctx.setQuery({ run: '' });
    runAll();
  }

  // ---------------------------------------------------------------- data
  async function reload({ findingsOnly = false } = {}) {
    if (!installed) { renderHero(); renderAnalytics(); return; }
    const filters = { days: view.days };
    try {
      const [sum] = await Promise.all([
        findingsOnly ? Promise.resolve(summary) : api.findingsSummary(filters, { signal: ctx.signal }),
        loadFindings(true),
      ]);
      summary = sum;
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(heroEl, errorView(err, () => reload()));
      return;
    }
    if (!ctx.active) return;
    renderHero();
    renderDetectors();
    renderAnalytics();
    if (view.tab === 'findings') renderTab();
  }

  async function loadFindings(reset) {
    if (!installed) return;
    const params = {
      days: view.days, severity: view.severity.join(','), detector: view.detector.join(','), triage: view.triage.join(','),
      q: view.q, sort: view.sort, limit: view.limit, offset: reset ? 0 : pageRows.length,
    };
    const res = await api.findings(params, { signal: ctx.signal });
    pageRows = reset ? res.findings : [...pageRows, ...res.findings];
    total = res.total;
  }

  // ---------------------------------------------------------------- hero
  function renderHero() {
    if (installed && !summary) {
      mount(heroEl, h('div.ws-card.ws-card-glow.wo-hero',
        h('div.ws-skel', { style: { width: '168px', height: '168px', borderRadius: '50%' } }),
        h('div.ws-col', h('div.ws-skel', { style: { height: '14px', width: '220px' } }), h('div.ws-skel', { style: { height: '34px', width: '70%' } }),
          h('div.ws-skel', { style: { height: '14px', width: '90%' } }), h('div.ws-skel', { style: { height: '40px', width: '320px' } }))));
      return;
    }
    const bySev = summary?.by_severity || { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    const score = installed ? (summary?.health_score ?? 100) : null;
    const plan = installed ? fixesToReach(bySev, 80) : null;
    const lastRun = Object.values(catalog.last_runs || {}).sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))[0];
    const daysSeg = seg([{ value: '1', label: '24h' }, { value: '7', label: '7d' }, { value: '30', label: '30d' }], view.days, (v) => { view.days = v; sync(); reload(); });
    mount(heroEl, h('div.ws-card.ws-card-glow.wo-hero',
      installed ? ring(score, { caption: healthLabel(score).toLowerCase() }) : ring(0, { caption: 'not installed', color: '#64748b' }),
      h('div',
        h('div.wo-hero-k', icon('lightning-charge-fill'), 'Performance & cost optimizer', h('span.ws-muted', '·'), h('span.ws-muted', `${catalog.toolkit?.toolkit || ''}`)),
        h('h1', installed
          ? [summary?.total ? [fmt.fmtInt(summary.total), ' findings, ', h('em', `${fmt.usd(summary.est_usd)} of waste`), ' detected'] : ['All clear — ', h('em', 'nothing to fix'), ' right now']]
          : ['Turn query history into ', h('em', 'savings')]),
        h('p', installed
          ? `Twelve detectors read Snowflake's account usage — spill, pruning, compilation, queueing, idle warehouses, storage overhead and more — and record what to fix, in priority order. Window: last ${view.days === '1' ? '24 hours' : `${view.days} days`}.`
          : 'Install the toolkit once and every detector below becomes runnable from here, on demand or on a daily schedule.'),
        h('div.wo-hero-actions',
          h('button.ws-btn.ws-btn-primary.ws-btn-lg', { type: 'button', disabled: !installed, onclick: () => runAll() }, icon('lightning-charge-fill'), 'Run all detectors'),
          h('button.ws-btn.ws-btn-lg', { type: 'button', disabled: !installed, onclick: () => { view.tab = 'runs'; tabBar.set('runs'); sync(); renderTab(); tabBar.scrollIntoView({ behavior: 'smooth' }); } }, icon('calendar2-week'), 'Schedule'),
          daysSeg),
        plan && plan.plan.length ? h('div.wo-plan', icon('signpost-split'), 'Fastest path to healthy (80+):',
          plan.plan.map((p) => h('span.ws-tag', `fix ${p.count} ${p.severity.toLowerCase()}`)), plan.reachable ? null : h('span.ws-muted', '(and more)')) : null,
        lastRun && h('div.wo-plan', icon('clock-history'), `Last run from the Studio: ${fmt.detectorLabel(lastRun.key)} · ${fmt.relTime(lastRun.created_at)} by ${lastRun.user}`)),
      h('div.wo-hero-metrics',
        metric(installed ? animatedValue(summary?.est_usd || 0, (x) => fmt.usd(x)) : '—', 'Estimated waste', 'cash-coin'),
        metric(installed ? animatedValue((summary?.wasted_seconds || 0) / 3600, (x) => `${fmt.fmtNum(x, 1)} h`) : '—', 'Compute time lost', 'hourglass-split'),
        metric(installed ? animatedValue(bySev.CRITICAL || 0, (x) => fmt.fmtInt(x)) : '—', 'Critical', 'exclamation-octagon-fill', STATUS.CRITICAL),
        metric(installed ? animatedValue(bySev.HIGH || 0, (x) => fmt.fmtInt(x)) : '—', 'High', 'exclamation-triangle-fill', STATUS.HIGH))));
  }

  function metric(value, label, ic, color) {
    return h('div.wo-metric', h('div.wo-metric-v', value), h('div.wo-metric-l', icon(ic, '', ), label, color && h('span', { style: { width: '8px', height: '8px', borderRadius: '50%', background: color, display: 'inline-block' } })));
  }

  // ---------------------------------------------------------------- detectors
  function renderDetectors() {
    const byDet = new Map((summary?.by_detector || []).map((d) => [d.detector, d]));
    const shown = catalog.detectors.filter((d) => view.cat === 'all' || d.category === view.cat);
    mount(detectorsEl, shown.map((d) => {
      const stats = byDet.get(d.procedure.toLowerCase());
      const el = detectorCard(d, stats);
      cardEls.set(d.key, el);
      return el;
    }));
  }

  function detectorCard(d, stats) {
    const sev = stats?.severity || {};
    const n = stats?.findings || 0;
    const isRunning = running.has(d.key);
    const last = catalog.last_runs?.[d.key];
    const sevTotal = SEVS.reduce((a, s) => a + (sev[s] || 0), 0) || 1;
    const cls = ['wo-det', `wo-cat-${d.category}`, sev.CRITICAL ? 'has-critical' : sev.HIGH ? 'has-high' : '', isRunning ? 'running' : '', (!installed || !d.installed) ? 'disabled' : ''].filter(Boolean).join('.');
    const statusEl = h('span.wo-det-status',
      isRunning ? [spinner(), h('span', running.get(d.key).label || 'Running in Snowflake…')]
        : last ? [icon(last.status === 'SUCCESS' ? 'check-circle' : last.status === 'FAILED' ? 'x-circle' : 'clock'), h('span.ws-ellipsis', `${last.status === 'FAILED' ? 'Failed' : 'Ran'} ${fmt.relTime(last.created_at)}`)]
          : [icon('dash-circle'), 'Not run from the Studio yet']);
    return h(`div.${cls}`, { dataset: { key: d.key } },
      h('div.wo-det-top',
        h('div.wo-det-icon', icon(d.icon)),
        h('div', { style: { minWidth: 0 } }, h('div.wo-det-title', d.title), h('div.wo-det-sum', d.summary)),
        h('div.wo-det-count', h('b', installed ? fmt.fmtInt(n) : '–'), h('small', 'findings'))),
      h('div.wo-sevbar', { title: SEVS.map((s) => `${s.toLowerCase()}: ${sev[s] || 0}`).join(' · ') },
        n ? SEVS.filter((s) => sev[s]).map((s) => h('span', { style: { width: `${((sev[s] / sevTotal) * 100).toFixed(1)}%`, background: STATUS[s] } })) : null),
      h('div.wo-det-meta',
        h('span', icon('hourglass-split', 'me-1'), h('b', fmt.seconds(stats?.wasted_seconds || 0)), ' lost'),
        h('span', icon('cash', 'me-1'), h('b', fmt.usd(stats?.est_usd || 0))),
        d.heavy && h('span.ws-tag', { title: 'Inspects operator profiles; takes longer' }, 'deep scan')),
      h('div.wo-det-foot', statusEl,
        h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', title: 'Show findings', disabled: !n,
          onclick: () => { view.detector = [d.procedure.toLowerCase()]; view.tab = 'findings'; tabBar.set('findings'); sync(); refreshFindings(); tabBar.scrollIntoView({ behavior: 'smooth' }); } }, icon('funnel')),
        d.params.length ? h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', title: 'Run with custom parameters', disabled: isRunning, onclick: () => paramsDialog(d) }, icon('sliders')) : null,
        h('button.ws-btn.ws-btn-sm.ws-btn-primary', { type: 'button', disabled: isRunning || !d.installed, onclick: () => runOne(d, {}) },
          isRunning ? spinner() : icon('play-fill'), isRunning ? 'Running' : 'Run')));
  }

  function setCard(d) {
    const old = cardEls.get(d.key);
    if (!old || !old.isConnected) return null;
    const byDet = new Map((summary?.by_detector || []).map((x) => [x.detector, x]));
    const el = detectorCard(d, byDet.get(d.procedure.toLowerCase()));
    old.replaceWith(el);
    cardEls.set(d.key, el);
    return el;
  }

  function paramsDialog(proc, { onDone } = {}) {
    const inputs = {};
    const body = h('div.ws-col', { style: { gap: '14px' } },
      h('p.ws-text-2', { style: { margin: 0, fontSize: '13px' } }, proc.detects),
      proc.params.map((p) => {
        const input = h('input.ws-input', { type: p.type === 'int' || p.type === 'float' ? 'number' : 'text', value: p.default ?? '',
          step: p.type === 'float' ? 'any' : '1', min: p.min ?? undefined, max: p.max ?? undefined, placeholder: p.help || '' });
        inputs[p.name] = input;
        return h('div.ws-field', h('label', p.label, p.required && h('span', { style: { color: STATUS.CRITICAL } }, ' *')), input,
          h('div.ws-help', [p.help, p.min != null && `min ${p.min}`, p.max != null && `max ${p.max}`].filter(Boolean).join(' · ')));
      }),
      h('div.ws-help', h('span.ws-mono', `CALL ${proc.fq}(${proc.params.map((p) => p.name).join(', ')})`)));
    modal({ title: proc.title, sub: proc.summary, body, width: 520,
      actions: [{ label: 'Cancel' }, { label: 'Run', primary: true, icon: 'play-fill', onClick: async (close) => {
        const params = Object.fromEntries(Object.entries(inputs).map(([k, el]) => [k, el.value]));
        close();
        if (proc.kind === 'detector') await runOne(proc, params);
        else onDone?.(params);
      } }] });
  }

  async function execute(proc, params, onStatus) {
    const { run } = await api.runDetector(proc.key, params);
    let delay = 600;
    for (;;) {
      await sleep(delay, ctx.signal);
      const r = await api.detectorRun(run.id, { signal: ctx.signal });
      if (r.status !== 'RUNNING') return r;
      onStatus?.(r.live_status);
      delay = Math.min(2500, delay * 1.3);
    }
  }

  async function runOne(d, params, { quiet = false } = {}) {
    if (running.has(d.key)) return null;
    running.set(d.key, { label: 'Submitting…' });
    setCard(d);
    let result = null;
    try {
      result = await execute(d, params, (st) => {
        running.set(d.key, { label: st === 'QUEUED' ? 'Queued' : st === 'RESUMING_WAREHOUSE' ? 'Waking warehouse…' : 'Scanning history…' });
        const el = cardEls.get(d.key);
        const lbl = el?.querySelector('.wo-det-status span:last-child');
        if (lbl) lbl.textContent = running.get(d.key).label;
      });
    } catch (err) {
      if (err.name === 'AbortError') return null;
      running.delete(d.key);
      const el = setCard(d);
      el?.classList.add('done-bad');
      if (!quiet) toast(err.message, 'error', { title: `${d.title} failed` });
      return { status: 'FAILED', error: err.message };
    }
    running.delete(d.key);
    catalog.last_runs = { ...catalog.last_runs, [d.key]: result };
    const el = setCard(d);
    const found = result.outcome?.value?.findings ?? result.outcome?.value?.new_findings;
    if (result.status === 'SUCCESS') {
      el?.classList.add('done-ok');
      if (found) el?.appendChild(h('span.wo-det-new', `+${found} NEW`));
      if (!quiet) toast(found ? `${fmt.plural(found, 'new finding')} recorded.` : 'No new findings — nothing changed since the last scan.', 'success', { title: d.title });
    } else {
      el?.classList.add('done-bad');
      if (!quiet) toast(result.error || 'The procedure failed.', 'error', { title: d.title });
    }
    if (!quiet) { await reload(); window.dispatchEvent(new Event('ws:findings-changed')); }
    return result;
  }

  async function runAll() {
    const todo = catalog.detectors.filter((d) => d.installed && !running.has(d.key));
    if (!todo.length) return;
    if (!await confirm({ title: 'Run every detector?', text: `${todo.length} procedures will scan the last ${view.days === '1' ? 'day' : `${view.days} days`} of account history on the Studio's warehouse. Typical cost: a few cents.`, confirmLabel: 'Run all' })) return;
    let done = 0;
    let newFindings = 0;
    const failed = [];
    const t0 = Date.now();
    const draw = (current) => mount(progressEl, h('div.wo-progress',
      h('span.ws-spinner'),
      h('div', { style: { minWidth: '210px' } }, h('b', `Running ${done + 1 > todo.length ? todo.length : done + 1} of ${todo.length}`), h('div.ws-muted', { style: { fontSize: '12px' } }, current ? current.title : 'Finishing…')),
      h('div.wo-progress-bar', h('span', { style: { width: `${(done / todo.length) * 100}%` } })),
      h('span.ws-mono.ws-muted', fmt.ms(Date.now() - t0))));
    for (const d of todo) {
      if (!ctx.active) return;
      draw(d);
      const lookback = d.params.find((p) => p.name === 'P_LOOKBACK_DAYS');
      const params = lookback ? { P_LOOKBACK_DAYS: Math.min(Number(view.days), lookback.max || 365) } : {};
      const res = await runOne(d, params, { quiet: true });
      if (!res || res.status !== 'SUCCESS') failed.push(d.title);
      newFindings += Number(res?.outcome?.value?.findings || 0);
      done += 1;
    }
    mount(progressEl);
    toast(`${fmt.plural(newFindings, 'new finding')} in ${fmt.ms(Date.now() - t0)}${failed.length ? ` — ${failed.length} failed: ${failed.join(', ')}` : ''}.`,
      failed.length ? 'warning' : 'success', { title: 'All detectors finished', timeout: 8000 });
    await reload();
    window.dispatchEvent(new Event('ws:findings-changed'));
  }

  // ---------------------------------------------------------------- analytics
  function renderAnalytics() {
    if (!installed || !summary) { mount(analyticsEl); return; }
    if (!summary.total) {
      mount(analyticsEl, h('div.ws-span-12', card({ body: empty({ icon: 'stars', title: 'No findings in this window', text: 'Run the detectors to scan recent history, or widen the window to 30 days.' }) })));
      return;
    }
    const timelineEl = h('div.ws-chart');
    const treeEl = h('div.ws-chart');
    const impactEl = h('div.ws-chart');
    const days = summary.by_day || [];
    const tlTable = tableToggle([{ key: 'day', label: 'Day' }, ...SEVS.map((s) => ({ key: s, label: s.toLowerCase(), num: true }))], days);
    const dets = summary.by_detector.filter((d) => d.wasted_seconds > 0 || d.findings > 0);
    mount(analyticsEl,
      h('div.ws-span-5', card({ title: 'Findings over time', icon: 'bar-chart-line', sub: 'Detected per day, by severity', tools: tlTable.button, body: [timelineEl, tlTable.table] })),
      h('div.ws-span-4', card({ title: 'Where time is lost', icon: 'grid-1x2', sub: 'Wasted compute by detector — click to filter', body: treeEl })),
      h('div.ws-span-3', card({ title: 'Severity mix', icon: 'pie-chart', body: severityMix(summary.by_severity) })),
      h('div.ws-span-12', card({ title: 'Impact map', icon: 'bullseye', sub: 'Each dot is a finding on this page: when it happened × time wasted; size = estimated $. Click a dot to open it.', body: impactEl })));
    chart(timelineEl, (t) => ({
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(128,160,255,.06)' } }, backgroundColor: t.surface2, borderColor: t.border, textStyle: { color: t.text },
        formatter: (ps) => `<b>${escapeHtml(fmt.day(ps[0].axisValue))}</b>${ps.filter((p) => p.value).map((p) => tipRow(p.color, p.seriesName.toLowerCase(), fmt.fmtInt(p.value))).join('')}` },
      legend: { top: 0, left: 0, icon: 'roundRect', itemWidth: 10, itemHeight: 10, textStyle: { color: t.text2, fontSize: 11 } },
      grid: { left: 8, right: 10, top: 32, bottom: 6, containLabel: true },
      xAxis: { type: 'category', data: days.map((d) => d.day), ...axisStyle(t, { splitLine: { show: false }, axisLabel: { color: t.muted, fontSize: 11, formatter: (v) => fmt.day(v) } }) },
      yAxis: { type: 'value', minInterval: 1, ...axisStyle(t) },
      series: SEVS.map((s, i) => ({ name: s, type: 'bar', stack: 's', barMaxWidth: 22, data: days.map((d) => d[s] || 0),
        itemStyle: { color: STATUS[s], borderColor: t.surface, borderWidth: 1, borderRadius: i === SEVS.length - 1 ? [4, 4, 0, 0] : 0 } })),
    }), ctx);
    treemap(treeEl, { ctx, valueFmt: (v) => fmt.seconds(v),
      items: dets.map((d) => ({ name: fmt.detectorLabel(d.detector), value: Math.max(1, Math.round(d.wasted_seconds)), detector: d.detector })),
      onClick: (node) => { if (node?.detector) { view.detector = [node.detector]; view.tab = 'findings'; tabBar.set('findings'); sync(); refreshFindings(); } } });
    const pts = pageRows.filter((r) => r.occurred_at && r.wasted_seconds > 0).map((r) => ({
      x: new Date(r.occurred_at).getTime(), y: Number(r.wasted_seconds), size: 8 + Math.sqrt(Number(r.est_usd) || 0) * 12,
      label: `${fmt.detectorLabel(r.detector)} — ${fmt.truncate(r.object_name, 60)}`, row: r }));
    if (pts.length) {
      scatter(impactEl, { points: pts, xName: 'When', yName: 'Time wasted', ctx, logY: true, timeX: true,
        xFmt: (v) => fmt.dateTime(v), yFmt: (v) => fmt.seconds(v), colorOf: (p) => STATUS[String(p.row.severity).toUpperCase()] || STATUS.LOW,
        onClick: (p) => { openRows.add(p.row.finding_id); view.tab = 'findings'; tabBar.set('findings'); renderTab(); setTimeout(() => document.querySelector(`[data-fid="${p.row.finding_id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 60); } });
      const xs = impactEl.parentElement;
      xs.insertBefore(h('div.ws-legend', { style: { marginBottom: '6px' } }, SEVS.map((s) => h('span', h('i', { style: { background: STATUS[s] } }), s.toLowerCase()))), impactEl);
    } else {
      mount(impactEl, empty({ icon: 'bullseye', title: 'No time-based findings on this page' }));
      impactEl.style.height = 'auto';
    }
  }

  function severityMix(bySev) {
    const tot = SEVS.reduce((a, s) => a + (bySev?.[s] || 0), 0) || 1;
    const rows = SEVS.map((s) => {
      const n = bySev?.[s] || 0;
      return h('div.ws-col', {
        style: { gap: '5px', cursor: 'pointer' },
        onclick: () => { view.severity = [s]; view.tab = 'findings'; tabBar.set('findings'); sync(); refreshFindings(); },
      },
      h('div.ws-row.ws-between', sevBadge(s), h('span.ws-mono', h('b', fmt.fmtInt(n)), h('span.ws-muted', ` · ${fmt.pct(n / tot)}`))),
      h('div.ws-bar', h('span', { style: { width: `${(n / tot) * 100}%`, background: STATUS[s] } })));
    });
    const triaged = summary?.triage ? TRIAGE.filter((t) => summary.triage[t.key]) : [];
    return h('div.ws-col', { style: { gap: '12px', paddingTop: '6px' } },
      rows,
      triaged.length ? h('div.ws-divider') : null,
      triaged.length ? h('div.ws-row.ws-wrap.ws-gap-sm', triaged.map((t) => h(`span.wo-triage.${t.key}`, icon(t.icon), `${summary.triage[t.key]} ${t.label.toLowerCase()}`))) : null);
  }

  // ---------------------------------------------------------------- tabs
  function renderTab() {
    if (view.tab === 'reports') return renderReports();
    if (view.tab === 'runs') return renderRuns();
    if (view.tab === 'about') return renderAbout();
    return renderFindings();
  }

  async function refreshFindings() {
    renderTab();
    try { await loadFindings(true); } catch (err) { if (err.name !== 'AbortError') toast(err.message, 'error'); return; }
    renderTab();
    renderAnalytics();
  }

  function renderFindings() {
    if (!installed) {
      mount(tabBodyEl, card({ body: empty({ icon: 'list-check', title: 'Findings appear here', text: 'Install the toolkit, then run any detector.' }) }));
      return;
    }
    const counts = summary?.by_severity || {};
    const detOptions = catalog.detectors.map((d) => ({ label: d.title, value: d.procedure.toLowerCase(), icon: d.icon, on: view.detector.includes(d.procedure.toLowerCase()) }));
    const search = h('input.ws-input', { placeholder: 'Search object, recommendation, SQL…', value: view.q,
      oninput: debounceLocal(() => { view.q = search.value.trim(); sync(); refreshFindings(); }, 350) });
    const detBtn = h('button.ws-chip', { type: 'button', class: view.detector.length ? 'on' : '' }, icon('cpu'),
      view.detector.length ? (view.detector.length === 1 ? fmt.detectorLabel(view.detector[0]) : `${view.detector.length} detectors`) : 'All detectors', icon('chevron-down'));
    detBtn.addEventListener('click', () => popover(detBtn, { items: [{ label: 'All detectors', value: '', icon: 'asterisk', on: !view.detector.length }, ...detOptions], search: true, width: 300,
      onPick: (i) => { view.detector = i.value ? [i.value] : []; sync(); refreshFindings(); } }));
    const sortSel = h('select.ws-select', { onchange: () => { view.sort = sortSel.value; sync(); refreshFindings(); } },
      [['severity', 'Sort: severity'], ['cost', 'Sort: est. $'], ['wasted', 'Sort: time wasted'], ['recent', 'Sort: newest detected'], ['occurred', 'Sort: newest occurrence'], ['metric', 'Sort: metric']]
        .map(([v, l]) => h('option', { value: v, selected: view.sort === v }, l)));
    const listEl = h('div');
    const bulkEl = h('div');

    mount(tabBodyEl,
      h('div.wo-filters.ws-mb',
        SEVS.map((s) => chip(s.toLowerCase(), { on: view.severity.includes(s), count: counts[s] || 0, color: STATUS[s],
          onClick: () => { view.severity = view.severity.includes(s) ? view.severity.filter((x) => x !== s) : [...view.severity, s]; sync(); refreshFindings(); } })),
        h('span', { style: { width: '8px' } }),
        detBtn,
        TRIAGE.map((t) => chip(t.label, { on: view.triage.includes(t.key),
          onClick: () => { view.triage = view.triage.includes(t.key) ? view.triage.filter((x) => x !== t.key) : [...view.triage, t.key]; sync(); refreshFindings(); } })),
        h('div.ws-search', { style: { flex: '1 1 220px' } }, icon('search'), search),
        sortSel,
        (view.severity.length || view.detector.length || view.triage.length || view.q) && h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button',
          onclick: () => { view.severity = []; view.detector = []; view.triage = []; view.q = ''; sync(); refreshFindings(); } }, icon('x-circle'), 'Clear')),
      h('div.wo-f-head', h('span', h('input', { type: 'checkbox', title: 'Select page', 'aria-label': 'Select all on page',
        checked: pageRows.length && pageRows.every((r) => selected.has(r.finding_id)),
        onchange: (e) => { pageRows.forEach((r) => (e.target.checked ? selected.add(r.finding_id) : selected.delete(r.finding_id))); renderFindings(); } })),
      h('span', 'Severity'), h('span', 'Finding'), h('span', 'Detector'), h('span.r', 'Wasted'), h('span.r', 'Est. $'), h('span.r', 'When'), h('span')),
      listEl,
      bulkEl,
      h('div.ws-row.ws-between', { style: { marginTop: '12px' } },
        h('span.ws-muted', { style: { fontSize: '12.5px' } }, `Showing ${fmt.fmtInt(pageRows.length)} of ${fmt.fmtInt(total)}`),
        pageRows.length < total && h('button.ws-btn', { type: 'button', onclick: async (e) => {
          e.currentTarget.disabled = true;
          try { await loadFindings(false); } catch (err) { toast(err.message, 'error'); }
          renderFindings();
        } }, icon('chevron-double-down'), 'Load more')));

    if (!pageRows.length) {
      mount(listEl, card({ body: empty({ icon: 'search', title: 'No findings match', text: 'Clear a filter or run more detectors.' }) }));
    } else {
      mount(listEl, pageRows.map((r) => findingRow(r)));
    }
    if (selected.size) {
      mount(bulkEl, h('div.wo-bulk', h('b', `${selected.size} selected`), h('span.ws-muted', '→'),
        TRIAGE.filter((t) => t.key !== 'open').map((t) => h('button.ws-btn.ws-btn-sm', { type: 'button', onclick: () => bulkTriage(t.key) }, icon(t.icon), t.label)),
        h('button.ws-btn.ws-btn-sm', { type: 'button', onclick: () => bulkTriage('open') }, icon('arrow-counterclockwise'), 'Reopen'),
        h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', style: { marginLeft: 'auto' }, onclick: () => { selected.clear(); renderFindings(); } }, 'Clear selection')));
    }
  }

  async function bulkTriage(status) {
    const ids = [...selected];
    try {
      await api.triageBulk(ids, status);
      pageRows.forEach((r) => { if (selected.has(r.finding_id)) r.triage = { ...(r.triage || {}), status, updated_by: window.APP_UI?.user }; });
      selected.clear();
      toast(`${fmt.plural(ids.length, 'finding')} marked ${status.replace('_', ' ')}.`, 'success');
      renderFindings();
      window.dispatchEvent(new Event('ws:findings-changed'));
    } catch (err) { toast(err.message, 'error'); }
  }

  function findingRow(r) {
    const tri = r.triage?.status || 'open';
    const isOpen = openRows.has(r.finding_id);
    const d = r.detail || {};
    const objLabel = r.object_type === 'QUERY' && !UUID.test(r.object_name) ? 'Query shape' : r.object_type;
    const sql = d.sql_preview;
    const bodyEl = isOpen ? findingBody(r) : null;
    const el = h(`div.wo-finding${isOpen ? '.open' : ''}.triaged-${tri}`, { dataset: { fid: r.finding_id } },
      h('div.wo-f-row', { onclick: (e) => {
        if (e.target.closest('input,button,a')) return;
        if (openRows.has(r.finding_id)) openRows.delete(r.finding_id); else openRows.add(r.finding_id);
        el.replaceWith(findingRow(r));
      } },
      h('span', h('input', { type: 'checkbox', checked: selected.has(r.finding_id), 'aria-label': 'Select finding',
        onchange: (e) => { if (e.target.checked) selected.add(r.finding_id); else selected.delete(r.finding_id); renderFindings(); } })),
      sevBadge(r.severity),
      h('div.wo-f-obj', h('b', sql ? fmt.oneLine(sql, 90) : `${objLabel} ${r.object_name}`),
        h('span', [`${objLabel.toLowerCase()} ${fmt.truncate(r.object_name, 44)}`, d.warehouse && `· ${d.warehouse}`, d.user && `· ${d.user}`].filter(Boolean).join(' '))),
      h('div.wo-f-obj', h('b', { style: { fontWeight: 500, color: 'var(--ws-text-2)' } }, fmt.detectorLabel(r.detector)),
        h('span', `${r.metric_name?.replace(/_/g, ' ') || ''} ${r.metric_value != null ? fmt.fmtNum(r.metric_value, 2) : ''}`)),
      h('div.wo-f-num', h('b', r.wasted_seconds ? fmt.seconds(r.wasted_seconds) : '—')),
      h('div.wo-f-num', h('b', r.est_usd ? fmt.usd(r.est_usd) : (d.est_monthly_usd ? `${fmt.usd(d.est_monthly_usd)}/mo` : '—'))),
      h('div.wo-f-num', { title: fmt.dateTime(r.occurred_at || r.detected_at) }, fmt.relTime(r.occurred_at || r.detected_at),
        tri !== 'open' && h('div', h(`span.wo-triage.${tri}`, TRIAGE.find((t) => t.key === tri)?.label))),
      icon('chevron-right', 'wo-f-chev')),
      bodyEl);
    return el;
  }

  function findingBody(r) {
    const d = r.detail || {};
    const isQuery = r.object_type === 'QUERY' && UUID.test(r.object_name);
    const sample = d.sample_query_id && UUID.test(d.sample_query_id) ? d.sample_query_id : null;
    const fix = d.fix_sql || (r.detector === 'sp_detect_warehouse_waste' && d.warehouse ? `ALTER WAREHOUSE "${d.warehouse}" SET AUTO_SUSPEND = 60;` : null)
      || (r.detector === 'sp_detect_storage_overhead' ? `ALTER TABLE ${r.object_name} SET DATA_RETENTION_TIME_IN_DAYS = 1;  -- or recreate as TRANSIENT if reloadable` : null);
    const pairs = Object.entries(d).filter(([k]) => !['sql_preview', 'fix_sql', 'operator_attributes', 'roles'].includes(k))
      .map(([k, v]) => [k.replace(/_/g, ' '), formatDetail(k, v)]);
    const noteEl = h('textarea.ws-textarea', { rows: 2, placeholder: 'Note for the team (optional)', style: { width: '100%' } }, r.triage?.note || '');
    const triageBtns = h('div.ws-row.ws-wrap.ws-gap-sm', TRIAGE.map((t) => h(`button.ws-btn.ws-btn-sm${(r.triage?.status || 'open') === t.key ? '.ws-btn-primary' : ''}`, { type: 'button',
      onclick: async () => {
        try {
          r.triage = await api.triage(r.finding_id, { status: t.key, note: noteEl.value });
          toast(`Marked ${t.label.toLowerCase()}.`, 'success');
          document.querySelector(`[data-fid="${r.finding_id}"]`)?.replaceWith(findingRow(r));
          window.dispatchEvent(new Event('ws:findings-changed'));
        } catch (err) { toast(err.message, 'error'); }
      } }, icon(t.icon), t.label)));
    return h('div.wo-f-body',
      h('div.ws-col',
        h('div.wo-f-rec', icon('lightbulb-fill', 'me-1'), r.recommendation || 'No recommendation recorded.'),
        d.sql_preview && h('pre.ws-code', { style: { maxHeight: '160px' } }, d.sql_preview),
        fix && h('div.ws-col', { style: { gap: '6px' } }, h('div.ws-row.ws-between', h('b', { style: { fontSize: '12.5px' } }, icon('wrench-adjustable', 'me-1'), 'Suggested fix (copy and run deliberately)'), copyButton(fix, 'Copy')),
          h('pre.ws-code', fix)),
        h('div.ws-row.ws-wrap.ws-gap-sm',
          isQuery && h('button.ws-btn.ws-btn-sm', { type: 'button', onclick: () => openQuery(r.object_name, { title: fmt.detectorLabel(r.detector) }) }, icon('search'), 'Inspect query'),
          sample && h('button.ws-btn.ws-btn-sm', { type: 'button', onclick: () => openQuery(sample, { title: 'Sample execution' }) }, icon('search'), 'Inspect a sample run'),
          (isQuery || sample) && h('a.ws-btn.ws-btn-sm', { href: `${window.APP_UI.root}worksheets?query=${isQuery ? r.object_name : sample}` }, icon('code-square'), 'Open SQL in worksheet'),
          r.object_type === 'WAREHOUSE' && h('a.ws-btn.ws-btn-sm', { href: `${window.APP_UI.root}warehouses?focus=${encodeURIComponent(r.object_name)}` }, icon('hdd-rack'), 'Warehouse'),
          copyButton(String(r.finding_id), `#${r.finding_id}`))),
      h('div.ws-col',
        kv([['Finding', `#${r.finding_id}`], ['Detected', fmt.dateTime(r.detected_at)], ['Occurred', fmt.dateTime(r.occurred_at)],
          ['Metric', `${r.metric_name || '—'} = ${fmt.fmtNum(r.metric_value, 3)}`], ...pairs]),
        h('div.ws-divider'),
        h('b', { style: { fontSize: '12.5px' } }, 'Triage'),
        r.triage?.updated_by && h('div.ws-muted', { style: { fontSize: '12px' } }, `Last change by ${r.triage.updated_by} ${fmt.relTime(r.triage.updated_at)}`),
        noteEl,
        triageBtns));
  }

  function formatDetail(k, v) {
    if (v == null || v === '') return '—';
    if (typeof v === 'object') return h('span.ws-mono', JSON.stringify(v).slice(0, 300));
    if (/bytes|spill_local|spill_remote|network/.test(k)) return fmt.bytes(v);
    if (/_ms$/.test(k)) return fmt.ms(v);
    if (/_secs$|seconds/.test(k)) return fmt.seconds(v);
    if (/usd/.test(k)) return fmt.usd(v);
    if (typeof v === 'number') return fmt.fmtNum(v, 3);
    if (UUID.test(String(v))) return h('a', { href: '#', onclick: (e) => { e.preventDefault(); openQuery(String(v)); } }, String(v));
    return String(v);
  }

  // ---------------------------------------------------------------- reports
  function renderReports() {
    mount(tabBodyEl, h('div.wo-reports', catalog.reports.map((rep) => reportCard(rep))));
  }

  function reportCard(rep) {
    const resultEl = h('div.wo-report-result');
    const btn = h('button.ws-btn.ws-btn-sm.ws-btn-primary', { type: 'button', disabled: !rep.installed }, icon('play-fill'), 'Run');
    const go = async (params) => {
      btn.disabled = true;
      mount(resultEl, h('div.ws-row', spinner(), h('span.ws-muted', 'Running in Snowflake…')));
      try {
        const res = await execute(rep, params);
        if (res.status !== 'SUCCESS') { mount(resultEl, banner('error', 'Report failed', res.error)); return; }
        mount(resultEl, reportOutcome(rep, res.outcome));
        if (rep.kind === 'batch') { await reload(); window.dispatchEvent(new Event('ws:findings-changed')); }
      } catch (err) {
        if (err.name !== 'AbortError') mount(resultEl, errorView(err));
      } finally { btn.disabled = false; }
    };
    btn.addEventListener('click', () => (rep.params.length ? paramsDialog(rep, { onDone: go }) : go({})));
    return card({ title: rep.title, icon: rep.icon, sub: rep.summary, tools: btn,
      body: h('div', h('div.ws-muted', { style: { fontSize: '12.5px' } }, rep.detects), h('div.ws-help.ws-mono', { style: { fontSize: '11px', marginTop: '6px', color: 'var(--ws-muted)' } }, `CALL ${rep.fq}(…)`), resultEl) });
  }

  function reportOutcome(rep, outcome) {
    if (!outcome) return h('div.ws-muted', 'No output.');
    if (outcome.type === 'table') {
      const cols = outcome.columns || [];
      if (!outcome.rows?.length) return h('div.ws-muted', { style: { marginTop: '8px' } }, icon('check2-circle', 'me-1'), 'No rows — nothing to report.');
      return h('div.ws-table-wrap', { style: { maxHeight: '320px' } }, h('table.ws-table',
        h('thead', h('tr', cols.map((c) => h(`th${c.family === 'number' ? '.num' : ''}`, c.name)))),
        h('tbody', outcome.rows.map((row) => h('tr', row.map((v, i) => {
          const c = cols[i] || {};
          const text = v == null ? '—' : c.family === 'number' ? fmt.fmtNum(v, 3) : String(v);
          const m = /^Query ([0-9a-f-]{36})$/i.exec(text);
          return h(`td${c.family === 'number' ? '.num' : ''}`, m ? h('a', { href: '#', onclick: (e) => { e.preventDefault(); openQuery(m[1]); } }, text) : text);
        }))))));
    }
    const v = outcome.value;
    if (v && typeof v === 'object' && v.error) return banner('error', 'The procedure reported an error', String(v.error));
    if (rep.key === 'clustering_report' && v && typeof v === 'object') {
      const hist = v.partition_depth_histogram || {};
      const items = Object.entries(hist).map(([k, n]) => ({ name: `depth ${k.replace(/^0+/, '') || '0'}`, value: Number(n) })).filter((x) => x.value);
      const histEl = h('div.ws-chart-sm');
      setTimeout(() => import('../lib/charts.js').then((c) => c.hbars(histEl, { items, valueFmt: fmt.fmtInt, ctx })), 0);
      return h('div.ws-col', kv([['Table', v.table], ['Key', v.key], ['Partitions', fmt.fmtInt(v.total_partition_count)],
        ['Constant ratio', fmt.pct(v.constant_ratio, 1)], ['Average depth', fmt.fmtNum(v.average_depth, 2)], ['Average overlaps', fmt.fmtNum(v.average_overlaps, 2)],
        v.notes && ['Notes', String(v.notes)]]), items.length ? histEl : null);
    }
    if (rep.kind === 'batch' && v && typeof v === 'object') {
      return banner('info', `${fmt.plural(v.new_findings || 0, 'new finding')}`, `Batch ${v.batch_id || ''} · ${v.failed ? `${v.failed} detector(s) failed — see Runs & schedule` : 'every detector succeeded'}.`);
    }
    return h('pre.ws-code', JSON.stringify(v, null, 2));
  }

  // ---------------------------------------------------------------- runs & schedule
  async function renderRuns() {
    if (!installed) { mount(tabBodyEl, card({ body: empty({ icon: 'calendar2-week', title: 'Scheduling needs the toolkit' }) })); return; }
    const schedEl = h('div', skeleton(2));
    const batchesEl = h('div', skeleton(4));
    const studioEl = h('div', skeleton(4));
    mount(tabBodyEl, h('div.ws-grid.ws-g-12',
      h('div.ws-span-4', card({ title: 'Daily schedule', icon: 'alarm', sub: 'Snowflake task T_WS_TUNING_DETECTORS', body: schedEl })),
      h('div.ws-span-8', card({ title: 'Studio runs', icon: 'person-workspace', sub: 'Detectors and reports started from this page', body: studioEl })),
      h('div.ws-span-12', card({ title: 'Batch log (WS_DETECTOR_RUNS)', icon: 'journal-text', sub: 'Every detector inside SP_RUN_ALL_DETECTORS batches, including the scheduled task', body: batchesEl }))));
    loadSchedule(schedEl);
    api.detectorRuns({ limit: 50 }, { signal: ctx.signal }).then(({ runs }) => mount(studioEl, runsTable(runs))).catch((e) => e.name !== 'AbortError' && mount(studioEl, errorView(e)));
    api.batches({ signal: ctx.signal }).then(({ runs }) => mount(batchesEl, batchTable(runs))).catch((e) => e.name !== 'AbortError' && mount(batchesEl, errorView(e)));
  }

  async function loadSchedule(el) {
    try {
      const s = await api.schedule({ signal: ctx.signal });
      const on = String(s.state || '').toLowerCase() === 'started';
      const sw = h(`button.wo-switch${on ? '.on' : ''}`, { type: 'button', role: 'switch', 'aria-checked': String(on), 'aria-label': 'Daily detector schedule',
        onclick: async () => {
          if (!on && !await confirm({ title: 'Turn on the daily scan?', text: `Snowflake will run every detector daily (${s.schedule}) on ${s.warehouse}. Expect about a minute of warehouse time per day.`, confirmLabel: 'Turn on' })) return;
          sw.setAttribute('aria-busy', 'true');
          try { await api.setSchedule(on ? 'SUSPEND' : 'RESUME'); toast(on ? 'Daily scan paused.' : 'Daily scan scheduled.', 'success'); } catch (err) { toast(err.message, 'error'); }
          loadSchedule(el);
        } });
      mount(el, h('div.ws-col',
        h('div.ws-row', sw, h('div', h('b', on ? 'Running daily' : 'Paused'), h('div.ws-muted', { style: { fontSize: '12px' } }, on ? 'Findings refresh automatically.' : 'Run detectors on demand, or switch on.'))),
        kv([['Schedule', h('span.ws-mono', s.schedule || '—')], ['Warehouse', s.warehouse || '—'], ['Task', h('span.ws-mono', s.task || '—')], ['Calls', h('span.ws-mono', 'SP_RUN_ALL_DETECTORS(2)')]])));
    } catch (err) {
      if (err.name !== 'AbortError') mount(el, errorView(err));
    }
  }

  function statusPill(status) {
    if (status === 'SUCCESS') return h('span.wo-triage.resolved', icon('check2'), 'success');
    if (status === 'RUNNING') return h('span.wo-triage.acknowledged', spinner(), 'running');
    return h('span.wo-triage', { style: { color: STATUS.CRITICAL } }, icon('x'), String(status || 'error').toLowerCase());
  }

  function runResult(r) {
    if (r.outcome?.value?.findings != null) return `${r.outcome.value.findings} new`;
    if (r.outcome?.type === 'table') return `${r.outcome.rows?.length || 0} rows`;
    return r.error ? fmt.truncate(r.error, 40) : '—';
  }

  function runsTable(runs) {
    if (!runs?.length) return empty({ icon: 'clock-history', title: 'No runs yet' });
    return table([
      { key: 'created_at', label: 'When', render: (r) => h('span', { title: fmt.dateTime(r.created_at) }, fmt.relTime(r.created_at)) },
      { key: 'key', label: 'Procedure', cls: 'strong', render: (r) => fmt.detectorLabel(r.key) },
      { key: 'user', label: 'By' },
      { key: 'status', label: 'Status', render: (r) => statusPill(r.status) },
      { key: 'result', label: 'Result', num: true, sort: false, render: runResult },
      { key: 'query_id', label: 'Query', sort: false, render: (r) => h('a.ws-mono', { href: '#', onclick: (e) => { e.preventDefault(); openQuery(r.query_id); } }, r.query_id.slice(0, 8)) },
    ], runs, { sortKey: 'created_at', maxHeight: 340 }).el;
  }

  function batchTable(runs) {
    if (!runs?.length) return empty({ icon: 'journal-text', title: 'No batches yet', text: 'Run "Run every detector" in Reports, or turn on the daily schedule.' });
    return table([
      { key: 'started_at', label: 'Started', render: (r) => h('span', { title: fmt.dateTime(r.started_at) }, fmt.relTime(r.started_at)) },
      { key: 'batch_id', label: 'Batch', cls: 'ws-mono', render: (r) => String(r.batch_id || '').slice(0, 8) },
      { key: 'detector', label: 'Detector', cls: 'strong', render: (r) => fmt.detectorLabel(r.detector) },
      { key: 'status', label: 'Status', render: (r) => statusPill(r.status) },
      { key: 'duration_ms', label: 'Duration', num: true, render: (r) => fmt.ms(r.duration_ms) },
      { key: 'result', label: 'Result / error', sort: false, cls: 'ws-mono',
        render: (r) => (r.error_message ? fmt.truncate(r.error_message, 90) : r.result ? JSON.stringify(r.result).slice(0, 90) : '—') },
    ], runs, { sortKey: 'started_at', maxHeight: 380 }).el;
  }

  function renderAbout() {
    const tk = catalog.toolkit?.toolkit || 'HOSPITALS.ANALYTICS';
    mount(tabBodyEl, h('div.ws-grid.ws-g-2',
      card({ title: 'What runs where', icon: 'diagram-2', body: h('div.ws-col', { style: { gap: '10px', fontSize: '13px' } },
        h('p.ws-text-2', { style: { margin: 0 } }, `Every detector is a SQL stored procedure in ${tk}, created with EXECUTE AS OWNER so it can read SNOWFLAKE.ACCOUNT_USAGE while the Studio's own role only holds USAGE on the procedures. Findings land in ${tk}.TUNING_FINDINGS; detectors de-duplicate, so re-running never double counts.`),
        h('p.ws-text-2', { style: { margin: 0 } }, 'Nothing here changes a warehouse or table. Fixes are shown as SQL for a human to review and run.'),
        h('p.ws-text-2', { style: { margin: 0 } }, 'Dollar figures are estimates: wasted seconds × the warehouse\'s credits/hour × your credit price ',
          h('b', fmt.usd(state.session?.credit_price_usd)), ' (WAREHOUSE_STUDIO_CREDIT_PRICE_USD).'),
        h('div.ws-copy-row', h('code.ws-tag', catalog.install_command), copyButton(catalog.install_command))) }),
      card({ title: 'Toolkit objects', icon: 'boxes', sub: installed ? 'All present' : `Missing: ${(catalog.toolkit?.toolkit_missing || []).length}`,
        body: h('div.ws-table-wrap', { style: { maxHeight: '360px' } }, h('table.ws-table', h('tbody',
          Object.entries({ ...(catalog.toolkit?.toolkit_procedures || {}), ...(catalog.toolkit?.toolkit_relations || {}) }).map(([name, ok]) => h('tr',
            h('td.ws-mono', name), h('td', ok ? h('span.wo-triage.resolved', icon('check2'), 'installed') : h('span.wo-triage', icon('dash'), 'missing'))))))) })));
  }
}

function debounceLocal(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
