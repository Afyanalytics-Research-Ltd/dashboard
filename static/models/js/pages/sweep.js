// One sweep: overview + controls, best-so-far, runs ranked by the metric, parallel coordinates,
// parameter importance (correlation + split importance), and parameter-vs-metric scatter.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, statCard, tabs, emptyState, errorState, skeleton, timeAgo, toast, toastError, confirmDialog, codeBlock, copyButton, refreshControl,
} from '../../../ui/js/ui.js';
import { fmtNumber, fmtDuration } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { urls } from '../core/urls.js';
import { lineChart, parallelCoords, scatterRuns, barsH, downloadText } from '../core/charts.js';
import { runStateBadge, runColor, displayName, fmtMetric } from '../core/wb.js';
import { parameterImportance } from './artifacts/shared.js';
import { sweepConfig, tunedParams, describeParam, isLogParam, loadSweepRuns, metricOf, bestSoFar, sweepStateBadge, setSweepState } from './artifacts/sweep_data.js';

const TABS = [
  { id: 'overview', label: 'Overview', icon: 'speedometer2' },
  { id: 'runs', label: 'Runs', icon: 'table' },
  { id: 'parallel', label: 'Parallel coordinates', icon: 'distribute-horizontal' },
  { id: 'importance', label: 'Parameter importance', icon: 'bar-chart-steps' },
  { id: 'scatter', label: 'Scatter', icon: 'dot' },
];

export default async function render(ctx) {
  const { entity, project, sweep: sweepName } = ctx.params;
  const { signal } = ctx;
  let tab = ctx.query.get('tab') || 'overview';
  ctx.setTitle(`Sweep ${sweepName}`);

  const headEl = h('div', skeleton(3));
  const bodyEl = h('div');
  mount(ctx.el, headEl, bodyEl);
  let sweep;
  let runs = [];
  let cfg;
  let params = [];
  let lastDrawnTab = null;

  const refresh = refreshControl(() => load(true), { key: 'sweep', initial: 30 });
  ctx.cleanup(refresh.stop);

  async function load(isRefresh = false) {
    try {
      ({ sweep, runs } = await loadSweepRuns(entity, project, sweepName, { signal }));
      if (!ctx.active) return;
      cfg = sweepConfig(sweep);
      params = tunedParams(cfg.parameters);
      // Parameters can also come from run configs when the YAML is unavailable.
      if (!params.length) params = [...new Set(runs.flatMap((r) => Object.keys(r.config)))].slice(0, 12);
      drawHead();
      // Charts re-render only when their tab is open (no off-screen work).
      if (!isRefresh || ['overview', 'runs'].includes(tab) || lastDrawnTab !== tab) drawBody();
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(headEl, pageHeader({ title: sweepName, icon: 'shuffle', crumbs: [{ label: project, href: urls.workspace(entity, project) }, { label: 'Sweeps', href: urls.sweeps(entity, project) }, { label: sweepName }] }));
      mount(bodyEl, errorState(err, () => load()));
    }
  }

  const scored = () => runs.map((r) => ({ run: r, metric: metricOf(r, cfg.metricName) })).filter((x) => x.metric != null);
  const sortScored = (rows) => rows.sort((a, b) => (cfg.goal === 'maximize' ? b.metric - a.metric : a.metric - b.metric));
  const bestRun = () => sortScored(scored())[0];

  function drawHead() {
    const st = String(sweep.state || '').toUpperCase();
    const control = (label, ic, target, opts = {}) => h(`button.btn.btn-sm.btn-${opts.danger ? 'outline-danger' : 'outline-secondary'}`, {
      type: 'button', onclick: () => changeState(target, label),
    }, icon(ic, 'me-1'), label);
    const best = bestRun();
    const finished = runs.filter((r) => r.state === 'finished').length;
    const crashed = runs.filter((r) => ['crashed', 'failed'].includes(r.state)).length;
    const totalTime = runs.reduce((a, r) => a + (r.runtime || 0), 0);
    mount(headEl,
      pageHeader({
        title: cfg.name || sweep.displayName || sweep.name, icon: 'shuffle',
        crumbs: [{ label: 'Projects', href: urls.projects() }, { label: project, href: urls.workspace(entity, project) }, { label: 'Sweeps', href: urls.sweeps(entity, project) }, { label: sweep.name }],
        badges: [sweepStateBadge(sweep.state), h('span.ms-pill.ms-pill-muted', cfg.method || sweep.method?.toLowerCase())],
        subtitle: cfg.metricName ? h('span', 'Optimising ', h('b.ms-mono', cfg.metricName), ` — ${cfg.goal}`) : 'No metric configured for this sweep.',
        actions: [
          h('div.ma-controls',
            st === 'RUNNING' && control('Pause', 'pause-fill', 'PAUSED'),
            st === 'PAUSED' && control('Resume', 'play-fill', 'RUNNING'),
            ['RUNNING', 'PAUSED', 'PENDING'].includes(st) && control('Finish', 'stop-fill', 'FINISHED'),
            ['RUNNING', 'PAUSED', 'PENDING'].includes(st) && control('Cancel', 'x-octagon', 'CANCELED', { danger: true })),
          refresh.el,
        ],
      }),
      h('div.af-stats',
        statCard({ label: `Best ${cfg.metricName || 'metric'}`, value: best ? fmtMetric(best.metric) : '—', icon: 'trophy', tone: 'amber',
          sub: best ? displayName(best.run) : null, href: best ? urls.run(entity, project, best.run.name) : undefined }),
        statCard({ label: 'Runs', value: `${fmtNumber(runs.length)}${cfg.runCap ? ` / ${cfg.runCap}` : ''}`, icon: 'play-circle', tone: 'blue', sub: `${finished} finished${crashed ? ` · ${crashed} crashed` : ''}` }),
        statCard({ label: 'Tuned parameters', value: fmtNumber(params.length), icon: 'sliders', tone: 'purple' }),
        statCard({ label: 'Compute', value: fmtDuration(totalTime), icon: 'stopwatch', tone: 'teal', sub: runs.length ? `${fmtDuration(totalTime / runs.length)} per run` : null }),
        statCard({ label: 'Last activity', value: timeAgo(sweep.heartbeatAt || sweep.updatedAt || sweep.createdAt), icon: 'clock', tone: 'slate' })),
      tabs(TABS, tab, (id) => { tab = id; ctx.setQuery({ tab: id === 'overview' ? null : id }); drawBody(); }));
  }

  async function changeState(target, label) {
    const danger = target === 'CANCELED';
    const ok = await confirmDialog({
      title: `${label} sweep?`, confirmLabel: label, danger,
      message: { PAUSED: 'Agents stop receiving new trials until you resume. Running trials finish normally.', RUNNING: 'Agents start receiving new trials again.', FINISHED: 'No new trials are scheduled; running trials finish normally.', CANCELED: 'No new trials are scheduled and running agents are told to stop.' }[target],
    });
    if (!ok) return;
    try {
      await setSweepState(sweep.id, target);
      toast(`Sweep ${label.toLowerCase()}${label.endsWith('e') ? 'd' : 'ed'}`, 'success');
      load();
    } catch (err) { toastError(err, `Could not ${label.toLowerCase()} the sweep`); }
  }

  function drawBody() {
    lastDrawnTab = tab;
    if (!runs.length && tab !== 'overview') { mount(bodyEl, emptyState({ icon: 'hourglass', title: 'No runs yet', message: 'Start an agent (see Overview) and trials appear here live.' })); return; }
    ({ overview: drawOverview, runs: drawRuns, parallel: drawParallel, importance: drawImportance, scatter: drawScatter }[tab] || drawOverview)();
  }

  // ---------------------------------------------------------------- overview

  function drawOverview() {
    const chartEl = h('div');
    const cmd = `wandb agent ${entity}/${project}/${sweep.name}`;
    mount(bodyEl, h('div.ma-sweep-grid',
      h('div.d-flex.flex-column.gap-3',
        card({ title: 'Best so far', icon: 'graph-up-arrow', body: cfg.metricName ? chartEl : emptyState({ title: 'No metric', message: 'Add metric: {name, goal} to the sweep config to track progress.' }) }),
        card({ title: 'Top runs', icon: 'trophy', flush: true, body: topRuns(5) })),
      h('div.d-flex.flex-column.gap-3',
        card({
          title: 'Search space', icon: 'sliders',
          body: h('div.ma-param-list', Object.entries(cfg.parameters).map(([k, spec]) => h('div.ma-param',
            h('span.ma-param-name', k, isLogParam(spec) && h('span.ms-pill.ms-pill-info.ms-1', 'log')),
            h('span.ma-param-dist', describeParam(spec))))),
        }),
        card({ title: 'Add more agents', icon: 'terminal', body: h('div', h('div.d-flex.gap-2.align-items-center.mb-2', h('code.ma-ref', cmd), copyButton(cmd)),
          h('div.small.text-muted', 'Run this on any machine with WANDB_BASE_URL pointing at this server.')) }),
        card({ title: 'Configuration', icon: 'filetype-yml', body: codeBlock(sweep.config || '', { lang: 'yaml', maxHeight: '360px' }) }))));
    if (cfg.metricName) {
      const series = bestSoFar(runs, cfg.metricName, cfg.goal);
      lineChart(chartEl, [
        { name: `best ${cfg.metricName}`, runName: '__best', color: '#0BB99F', points: series.filter((p) => p.best != null).map((p) => ({ x: p.i + 1, y: p.best })) },
        { name: 'trial', runName: '__trial', color: '#94a3b8', dash: 'dot', points: series.filter((p) => p.value != null).map((p) => ({ x: p.i + 1, y: p.value })) },
      ], { height: 260, xLabel: 'trial #', smoothing: { type: 'none', weight: 0 }, bands: false,
        onClick: ({ x }) => { const r = series[x - 1]?.run; if (r) navigate(urls.run(entity, project, r.name)); } });
    }
  }

  function topRuns(n) {
    const rows = sortScored(scored()).slice(0, n);
    if (!rows.length) return emptyState({ icon: 'hourglass', title: 'No finished trials yet' });
    return h('div.af-list', rows.map((x, i) => h('a.af-list-item', { href: urls.run(entity, project, x.run.name) },
      h('span.ma-version', `#${i + 1}`),
      h('div.af-list-main', h('div.af-list-title', displayName(x.run)),
        h('div.af-list-sub', params.slice(0, 4).map((p) => `${p}=${fmtMetric(x.run.config[p])}`).join(' · '))),
      h('b.ms-metric', fmtMetric(x.metric)))));
  }

  // ---------------------------------------------------------------- runs table

  function drawRuns() {
    let sortKey = 'metric';
    let dir = cfg.goal === 'maximize' ? 'desc' : 'asc';
    const tableEl = h('div');
    const best = bestRun();
    const draw = () => {
      const rows = runs.map((r) => ({ run: r, metric: metricOf(r, cfg.metricName) })).sort((a, b) => {
        const va = sortKey === 'metric' ? a.metric : sortKey === 'created' ? new Date(a.run.createdAt) : a.run.config[sortKey];
        const vb = sortKey === 'metric' ? b.metric : sortKey === 'created' ? new Date(b.run.createdAt) : b.run.config[sortKey];
        if (va == null) return 1;
        if (vb == null) return -1;
        return (va > vb ? 1 : va < vb ? -1 : 0) * (dir === 'asc' ? 1 : -1);
      });
      const th = (key, label) => h('th.af-sortable', { class: sortKey === key && 'af-sorted', onclick: () => { dir = sortKey === key && dir === 'desc' ? 'asc' : 'desc'; sortKey = key; draw(); } },
        label, icon(sortKey === key ? (dir === 'asc' ? 'sort-up' : 'sort-down') : 'chevron-expand', 'af-sort-icon'));
      mount(tableEl, h('div.af-table-wrap', h('table.af-table.ma-table',
        h('thead', h('tr', h('th', 'Run'), h('th', 'State'), params.map((p) => th(p, p)), th('metric', cfg.metricName || 'metric'), h('th.text-end', 'Runtime'), th('created', 'Started'))),
        h('tbody', rows.map((x, i) => h('tr.af-row-click', { onclick: (e) => { if (!e.target.closest('a')) navigate(urls.run(entity, project, x.run.name)); } },
          h('td', h('span.d-flex.align-items-center.gap-2', h('span.ms-swatch', { style: { background: runColor(x.run, i) } }), h('a.ms-run-name', { href: urls.run(entity, project, x.run.name) }, displayName(x.run)))),
          h('td', runStateBadge(x.run.state)),
          params.map((p) => h('td.ms-metric', fmtMetric(x.run.config[p]))),
          h('td.ms-metric', h('span', { class: best && x.run.name === best.run.name ? 'ms-best' : '' }, fmtMetric(x.metric))),
          h('td.text-end.ms-metric', fmtDuration(x.run.runtime)),
          h('td.af-nowrap', timeAgo(x.run.createdAt))))))));
    };
    const exportBtn = h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => {
      const header = ['run', 'name', 'state', ...params, cfg.metricName || 'metric'];
      const lines = [header.join(','), ...runs.map((r) => [r.name, displayName(r), r.state, ...params.map((p) => r.config[p]), metricOf(r, cfg.metricName)].map((v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v)).join(','))];
      downloadText(lines.join('\n'), `sweep-${sweep.name}.csv`);
    } }, icon('download', 'me-1'), 'CSV');
    mount(bodyEl, card({ title: `${runs.length} trials`, icon: 'table', actions: exportBtn, body: tableEl, flush: true }));
    draw();
  }

  // ---------------------------------------------------------------- parallel coordinates

  function drawParallel() {
    const el = h('div');
    const rows = runs.filter((r) => metricOf(r, cfg.metricName) != null);
    mount(bodyEl, card({
      title: 'Parallel coordinates', icon: 'distribute-horizontal',
      actions: h('span.small.text-muted', 'Drag along an axis to filter trials; colour = metric'),
      body: rows.length ? el : emptyState({ title: 'No finished trials with the metric yet' }),
    }));
    if (!rows.length) return;
    const dims = params.map((p) => {
      const spec = cfg.parameters[p];
      const values = rows.map((r) => r.config[p]);
      return { label: isLogParam(spec) ? `log10 ${p}` : p, values: isLogParam(spec) ? values.map((v) => (typeof v === 'number' && v > 0 ? Math.log10(v) : v)) : values };
    });
    const metric = rows.map((r) => metricOf(r, cfg.metricName));
    dims.push({ label: cfg.metricName, values: metric });
    parallelCoords(el, dims, { colorValues: cfg.goal === 'maximize' ? metric : metric.map((v) => -v), colorLabel: cfg.goal === 'maximize' ? cfg.metricName : `−${cfg.metricName}`, height: 420 });
  }

  // ---------------------------------------------------------------- importance

  function drawImportance() {
    const rows = runs.map((r) => ({ params: r.config, metric: metricOf(r, cfg.metricName) })).filter((r) => r.metric != null);
    const imp = parameterImportance(rows, params);
    const chartEl = h('div');
    mount(bodyEl, card({
      title: `What drives ${cfg.metricName || 'the metric'}?`, icon: 'bar-chart-steps',
      actions: h('span.small.text-muted', `${rows.length} trials · computed in your browser`),
      body: imp.length ? h('div.d-flex.flex-column.gap-3',
        h('div.af-callout.af-callout-info', icon('info-circle'), h('span',
          h('b', 'Importance'), ' is the share of the metric’s variance explained by the best single split on that parameter (works for non-linear effects and categories). ',
          h('b', 'Correlation'), ' is Spearman’s rank correlation: positive means higher values go with a higher metric.')),
        chartEl,
        h('div.af-table-wrap', h('table.af-table.ma-imp-table',
          h('thead', h('tr', h('th', 'Parameter'), h('th', 'Importance'), h('th.text-end', 'Correlation'), h('th.text-end', 'Trials'), h('th', 'Reading'))),
          h('tbody', imp.map((x) => h('tr',
            h('td.ms-mono', x.param),
            h('td', h('div.d-flex.align-items-center.gap-2', h('div.ma-imp-bar', h('span', { style: { width: `${Math.round(x.importance * 100)}%`, background: '#7C3AED' } })), h('span.ms-metric', `${(x.importance * 100).toFixed(0)}%`))),
            h('td.text-end.ms-metric', x.correlation == null ? h('span.text-muted', 'categorical') : h('span', { class: x.correlation >= 0 ? 'ma-corr-pos' : 'ma-corr-neg' }, x.correlation.toFixed(2))),
            h('td.text-end.ms-metric', x.n),
            h('td.small', reading(x)))))))) : emptyState({ icon: 'bar-chart-steps', title: 'Not enough trials yet', message: 'Importance needs at least 3 finished trials where the parameter varied.' }),
    }));
    if (imp.length) barsH(chartEl, imp.map((x) => ({ label: x.param, value: x.importance, color: '#7C3AED' })), { valueLabel: 'importance (variance explained)', format: '.0%' });
  }

  function reading(x) {
    const better = cfg.goal === 'maximize' ? 'higher' : 'lower';
    if (x.importance < 0.1) return h('span.text-muted', 'little effect so far');
    if (x.correlation == null) return `strong effect — compare categories in Scatter`;
    const dirWord = (x.correlation > 0) === (cfg.goal === 'maximize') ? 'larger' : 'smaller';
    return `${dirWord} values tend to give ${better} ${cfg.metricName}`;
  }

  // ---------------------------------------------------------------- scatter

  function drawScatter() {
    let px = params[0];
    const el = h('div');
    const sel = h('select.form-select.form-select-sm', { 'aria-label': 'Parameter', onchange: (e) => { px = e.target.value; draw(); } }, params.map((p) => h('option', { value: p }, p)));
    const draw = () => {
      const pts = runs.map((r, i) => ({ r, i, x: r.config[px], y: metricOf(r, cfg.metricName) })).filter((p) => p.y != null && p.x != null);
      const cats = [...new Set(pts.filter((p) => typeof p.x !== 'number').map((p) => String(p.x)))];
      scatterRuns(el, pts.map((p) => ({ x: typeof p.x === 'number' ? p.x : String(p.x), y: p.y, name: displayName(p.r), runName: p.r.name, color: runColor(p.r, p.i) })),
        { xLabel: px, yLabel: cfg.metricName, height: 380, logX: isLogParam(cfg.parameters[px]) && !cats.length, onClick: (name) => navigate(urls.run(entity, project, name)) });
    };
    mount(bodyEl, card({ title: 'Parameter vs metric', icon: 'dot', actions: h('div.ma-axis-pick', h('span', 'X axis'), sel), body: el }));
    draw();
  }

  await load();
}
