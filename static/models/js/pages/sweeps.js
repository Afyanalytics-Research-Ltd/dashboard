// Sweeps of a project: state, method, progress, best metric and a best-so-far sparkline per sweep.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, emptyState, errorState, skeleton, timeAgo, refreshControl, searchInput, chips, codeBlock, card } from '../../../ui/js/ui.js';
import { sparkline } from '../../../ui/js/charts.js';
import { fmtNumber } from '../../../ui/js/format.js';
import { urls } from '../core/urls.js';
import { loadSweeps } from '../core/data.js';
import { fmtMetric } from '../core/wb.js';
import { sweepConfig, loadSweepRuns, bestSoFar, sweepStateBadge, tunedParams } from './artifacts/sweep_data.js';

export default async function render(ctx) {
  const { entity, project } = ctx.params;
  const { signal } = ctx;
  ctx.setTitle(`Sweeps · ${project}`);
  let q = ctx.query.get('q') || '';
  let state = ctx.query.get('state') || '';
  const gridEl = h('div', skeleton(6));
  const chipsEl = h('div');
  let rows = [];

  const refresh = refreshControl(() => load(), { key: 'sweeps', initial: 30 });
  ctx.cleanup(refresh.stop);

  mount(ctx.el,
    pageHeader({
      title: 'Sweeps', icon: 'shuffle',
      crumbs: [{ label: 'Projects', href: urls.projects() }, { label: project, href: urls.workspace(entity, project) }, { label: 'Sweeps' }],
      subtitle: 'Hyperparameter searches: what was tried, what won, and which parameters mattered.',
      actions: [refresh.el],
    }),
    h('div.ma-reg-filters', searchInput({ placeholder: 'Search sweeps…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } }), chipsEl),
    gridEl);

  async function load() {
    try {
      const sweeps = await loadSweeps(entity, project, { signal });
      if (!ctx.active) return;
      // Runs of up to 6 sweeps at a time for the best-so-far sparklines.
      const out = [];
      for (let i = 0; i < sweeps.length; i += 6) {
        out.push(...await Promise.all(sweeps.slice(i, i + 6).map(async (s) => {
          const cfg = sweepConfig(s);
          try {
            const { runs } = await loadSweepRuns(entity, project, s.name, { signal, max: 300 });
            return { s, cfg, runs, series: bestSoFar(runs, cfg.metricName, cfg.goal) };
          } catch { return { s, cfg, runs: [], series: [] }; }
        })));
      }
      if (!ctx.active) return;
      rows = out;
      draw();
    } catch (err) {
      if (err.name !== 'AbortError') mount(gridEl, errorState(err, load));
    }
  }

  function draw() {
    const counts = {};
    rows.forEach((r) => { const st = String(r.s.state).toLowerCase(); counts[st] = (counts[st] || 0) + 1; });
    mount(chipsEl, chips([{ value: '', label: 'All', count: rows.length }, ...Object.entries(counts).map(([k, n]) => ({ value: k, label: k, count: n }))], state,
      (v) => { state = v; ctx.setQuery({ state: v }); draw(); }));
    if (!rows.length) {
      mount(gridEl, h('div.d-flex.flex-column.gap-3',
        emptyState({ icon: 'shuffle', title: 'No sweeps yet', message: 'Define a search space, start agents, and every trial shows up here with live best-so-far tracking.' }),
        card({ title: 'Start a sweep', icon: 'terminal', body: codeBlock(`import wandb

sweep = {"method": "bayes", "metric": {"name": "val/loss", "goal": "minimize"},
         "parameters": {"learning_rate": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"}}}
sweep_id = wandb.sweep(sweep, project="${project}", entity="${entity}")
wandb.agent(sweep_id, function=train, count=20)`, { lang: 'python' }) })));
      return;
    }
    const needle = q.toLowerCase();
    const visible = rows.filter((r) => (!state || String(r.s.state).toLowerCase() === state)
      && (!needle || `${r.s.name} ${r.cfg.name || ''} ${r.cfg.metricName || ''}`.toLowerCase().includes(needle)));
    if (!visible.length) { mount(gridEl, emptyState({ icon: 'search', title: 'No sweep matches' })); return; }
    mount(gridEl, h('div.ma-sweep-cards', visible.map(({ s, cfg, runs, series }) => {
      const best = series.length ? series[series.length - 1].best : null;
      const done = runs.filter((r) => r.state !== 'running').length;
      const cap = cfg.runCap;
      const spark = series.filter((p) => p.best != null).map((p) => p.best);
      const params = tunedParams(cfg.parameters);
      return h('a.ma-sweep-card', { href: urls.sweep(entity, project, s.name) },
        h('div.ma-sweep-head',
          h('div.min-w-0', h('div.ma-sweep-name', cfg.name || s.displayName || s.name), h('div.ma-sweep-id', s.name, ' · ', cfg.method || s.method?.toLowerCase())),
          sweepStateBadge(s.state)),
        h('div.ma-sweep-best',
          h('b', best == null ? '—' : fmtMetric(best)),
          h('span', cfg.metricName ? `best ${cfg.metricName} (${cfg.goal === 'maximize' ? 'max' : 'min'})` : 'no metric configured')),
        spark.length > 1 ? sparkline(spark, { width: 300, height: 34, color: '#0072CE' }) : h('div.small.text-muted', 'Needs 2+ finished runs for a trend'),
        cap ? h('div.ma-sweep-progress', { title: `${done} of ${cap} runs` }, h('span', { style: { width: `${Math.min(100, (done / cap) * 100)}%` } })) : null,
        h('div.ma-sweep-foot',
          h('span', icon('play-circle', 'me-1'), `${fmtNumber(runs.length || s.runCount)} run${(runs.length || s.runCount) === 1 ? '' : 's'}${cap ? ` / ${cap}` : ''}`),
          h('span', icon('sliders', 'me-1'), `${params.length} tuned parameter${params.length === 1 ? '' : 's'}`),
          h('span', timeAgo(s.updatedAt || s.createdAt))));
    })));
  }

  await load();
}
