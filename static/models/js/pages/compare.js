// Compare runs side by side: unlimited runs (W&B's comparer stops at 10), "differences only",
// best-value highlighting per metric, overlaid charts, add/remove runs, CSV export.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { card, emptyState, errorState, toast, openMenu, spinner } from '../../../ui/js/ui.js';
import { fmtDuration } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { gql } from '../core/api.js';
import { urls } from '../core/urls.js';
import { RUN_FIELDS, normRun, loadHistories, loadMetricKeys, loadAllRuns } from '../core/data.js';
import { runColor, displayName, fmtMetric, runStateBadge } from '../core/wb.js';
import { lineChart, lazy, downloadText } from '../core/charts.js';
import { projectHeader, lowerIsBetter, toCsv } from './workspace/common.js';

export default async function render(ctx) {
  const { entity, project } = ctx.params;
  const { el, signal } = ctx;
  let names = (ctx.query.get('runs') || '').split(',').map((s) => s.trim()).filter(Boolean);
  let diffOnly = ctx.query.get('diff') === '1';
  let chartKeys = (ctx.query.get('metrics') || '').split(',').filter(Boolean);
  let runs = [];
  let allRuns = null;
  let metricKeys = [];

  const headerEl = h('div');
  const pickerEl = h('div.mp-cmp-picker');
  const chartsEl = h('div.ms-panels.mp-cmp-charts');
  const tablesEl = h('div.mp-cmp-tables');
  mount(el, headerEl, pickerEl, chartsEl, tablesEl);

  const sync = () => ctx.setQuery({ runs: names.join(',') || null, diff: diffOnly ? '1' : null, metrics: chartKeys.join(',') || null });

  async function load() {
    if (!names.length) { runs = []; renderAll(); return; }
    mount(tablesEl, h('div.mp-loading', spinner(), ' Loading runs…'));
    try {
      const d = await gql(`query Cmp($e: String!, $p: String!, $f: JSONString) { project(entityName: $e, name: $p) {
        runs(filters: $f, first: 200) { edges { node { ${RUN_FIELDS} } } } } }`,
      { e: entity, p: project, f: JSON.stringify({ name: { $in: names } }) }, { signal });
      const byName = new Map((d.project?.runs?.edges || []).map((e) => [e.node.name, normRun(e.node)]));
      runs = names.map((n) => byName.get(n)).filter(Boolean);
      const missing = names.filter((n) => !byName.has(n));
      if (missing.length) toast(`Not found: ${missing.join(', ')}`, 'warning');
      metricKeys = (await loadMetricKeys(entity, project, runs.map((r) => r.name), { signal }))
        .filter((k) => k.numeric && !k.key.startsWith('_')).map((k) => k.key).sort();
      if (!chartKeys.length) {
        const preferred = ['val/accuracy', 'val/loss', 'train/loss', 'train/accuracy', 'loss', 'accuracy'].filter((k) => metricKeys.includes(k));
        chartKeys = [...preferred, ...metricKeys.filter((k) => !preferred.includes(k) && /val.*(acc|loss)|loss|accuracy/i.test(k))].slice(0, 4);
      }
      if (!chartKeys.length) chartKeys = metricKeys.slice(0, 2);
      if (ctx.active) renderAll();
    } catch (err) {
      if (err.name !== 'AbortError') mount(tablesEl, errorState(err, load));
    }
  }

  function renderAll() {
    mount(headerEl, projectHeader({
      entity, project, active: 'table', icon: 'layout-split', title: 'Compare runs',
      subtitle: `${runs.length} run${runs.length === 1 ? '' : 's'} side by side — no limit on how many`,
      actions: [
        h('label.form-check.form-switch.mp-switch', h('input.form-check-input', { type: 'checkbox', checked: diffOnly, onchange: (e) => { diffOnly = e.target.checked; sync(); renderTables(); } }),
          h('span.form-check-label.small', 'Differences only')),
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => downloadText(csv(), `${project}-compare.csv`) }, icon('filetype-csv', 'me-1'), 'CSV'),
      ],
    }));
    renderPicker();
    if (!runs.length) {
      mount(chartsEl);
      mount(tablesEl, emptyState({ icon: 'layout-split', title: 'Pick runs to compare', message: 'Select runs in the Runs table, or add them here.' }));
      return;
    }
    renderCharts();
    renderTables();
  }

  function renderPicker() {
    mount(pickerEl,
      runs.map((r, i) => h('span.mp-cmp-chip', { style: { '--c': runColor(r, i) } },
        h('span.ms-swatch', { style: { background: runColor(r, i) } }),
        h('a', { href: urls.run(entity, project, r.name) }, displayName(r)),
        h('button.mp-fchip-x', { title: 'Remove', onclick: () => { names = names.filter((n) => n !== r.name); sync(); load(); } }, icon('x')))),
      h('button.btn.btn-sm.btn-outline-primary', { onclick: (e) => addRunMenu(e.currentTarget) }, icon('plus-lg', 'me-1'), 'Add run'),
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: (e) => metricMenu(e.currentTarget) }, icon('graph-up', 'me-1'), `Charts (${chartKeys.length})`));
  }

  async function addRunMenu(anchor) {
    if (!allRuns) allRuns = (await loadAllRuns(entity, project, { max: 500, signal })).runs;
    const input = h('input.form-control.form-control-sm', { placeholder: 'Search runs…', autofocus: true });
    const list = h('div.mp-pick-list');
    const draw = () => {
      const q = input.value.toLowerCase();
      mount(list, allRuns.filter((r) => !names.includes(r.name) && (!q || displayName(r).toLowerCase().includes(q) || r.name.includes(q))).slice(0, 40)
        .map((r) => h('button.mp-pick', { type: 'button', onclick: () => { names.push(r.name); sync(); closePop(); load(); } }, runStateBadge(r.state), h('span', displayName(r)), h('span.text-muted.small.ms-auto', r.group || ''))));
    };
    input.addEventListener('input', debounce(draw, 100));
    const pop = h('div.mp-pop', input, list);
    const closePop = () => pop.remove();
    const rect = anchor.getBoundingClientRect();
    Object.assign(pop.style, { left: `${Math.max(8, rect.left)}px`, top: `${rect.bottom + 6}px` });
    document.body.appendChild(pop);
    setTimeout(() => document.addEventListener('pointerdown', function off(e) { if (!pop.contains(e.target)) { closePop(); document.removeEventListener('pointerdown', off); } }), 0);
    draw();
    input.focus();
  }

  function metricMenu(anchor) {
    openMenu(anchor, [{ header: 'Charts to overlay' }, ...metricKeys.map((k) => ({
      label: k, icon: chartKeys.includes(k) ? 'check-square' : 'square',
      onClick: () => { chartKeys = chartKeys.includes(k) ? chartKeys.filter((x) => x !== k) : [...chartKeys, k]; sync(); renderPicker(); renderCharts(); },
    }))]);
  }

  async function renderCharts() {
    if (!chartKeys.length) { mount(chartsEl); return; }
    const panels = chartKeys.map((k) => {
      const body = h('div.ms-panel-body', h('div.ms-panel-loading'));
      const p = h('div.ms-panel', h('div.ms-panel-head', h('span.ms-panel-title', k)), body);
      p.body = body;
      p.metric = k;
      return p;
    });
    mount(chartsEl, panels);
    let hist;
    const get = async () => { hist ||= loadHistories(entity, project, runs.map((r) => r.name), chartKeys, { samples: 400, signal }); return hist; };
    panels.forEach((p) => lazy(p, async () => {
      try {
        const data = await get();
        const series = runs.map((r, i) => ({ name: displayName(r), runName: r.name, color: runColor(r, i), points: data.get(r.name)?.get(p.metric) || [] })).filter((s) => s.points.length);
        if (!series.length) { mount(p.body, h('div.mp-panel-empty', 'Not logged by these runs')); return; }
        await lineChart(p.body, series, { xLabel: 'step', smoothing: { type: 'ema', weight: 0.5 }, onClick: ({ runName }) => runName && navigate(urls.run(entity, project, runName)) });
      } catch (err) { mount(p.body, h('div.mp-panel-empty.text-danger', err.message)); }
    }));
  }

  function rowsFor(section) {
    const keys = new Set();
    runs.forEach((r) => Object.keys(section === 'config' ? r.config : r.summary).forEach((k) => keys.add(k)));
    return [...keys].sort().map((k) => {
      const vals = runs.map((r) => (section === 'config' ? r.config : r.summary)[k]);
      const same = vals.every((v) => JSON.stringify(v) === JSON.stringify(vals[0]));
      return { key: k, vals, same };
    });
  }

  function table(title, ic, rows, { highlightBest = false } = {}) {
    const shown = diffOnly ? rows.filter((r) => !r.same) : rows;
    const head = h('tr', h('th.mp-cmp-key', 'Key'), runs.map((r, i) => h('th', h('span.ms-swatch.me-1', { style: { background: runColor(r, i) } }),
      h('a', { href: urls.run(entity, project, r.name) }, displayName(r)))));
    const body = shown.map((row) => {
      let bestIdx = -1;
      const nums = row.vals.map((v) => (typeof v === 'number' ? v : null));
      if (highlightBest && !row.same && nums.filter((v) => v != null).length > 1) {
        const lower = lowerIsBetter(row.key);
        let best = null;
        nums.forEach((v, i) => { if (v != null && (best == null || (lower ? v < best : v > best))) { best = v; bestIdx = i; } });
      }
      return h('tr', { class: !row.same && 'mp-cmp-diff' },
        h('td.mp-cmp-key', { title: row.key }, row.key, highlightBest && bestIdx >= 0 && h('span.mp-dir', lowerIsBetter(row.key) ? ' ↓' : ' ↑')),
        row.vals.map((v, i) => h('td', { class: [typeof v === 'number' && 'num', i === bestIdx && 'mp-cmp-best'] },
          v == null ? h('span.text-muted', '—') : typeof v === 'object' ? h('code.mp-json', JSON.stringify(v)) : fmtMetric(v))));
    });
    return card({
      title: h('span', title, h('span.ms-count.ms-2', diffOnly ? `${shown.length} differ of ${rows.length}` : rows.length)), icon: ic, flush: true,
      body: shown.length ? h('div.mp-cmp-scroll', h('table.mp-cmp', h('thead', head), h('tbody', body)))
        : emptyState({ icon: 'check2-all', title: 'Identical across these runs' }),
    });
  }

  function renderTables() {
    const meta = [
      { key: 'state', vals: runs.map((r) => r.state) },
      { key: 'runtime', vals: runs.map((r) => (r.runtime != null ? fmtDuration(r.runtime) : null)) },
      { key: 'group', vals: runs.map((r) => r.group) },
      { key: 'jobType', vals: runs.map((r) => r.jobType) },
      { key: 'tags', vals: runs.map((r) => (r.tags || []).join(', ') || null) },
      { key: 'created', vals: runs.map((r) => r.createdAt?.slice(0, 16).replace('T', ' ')) },
    ].map((m) => ({ ...m, same: m.vals.every((v) => v === m.vals[0]) }));
    mount(tablesEl,
      table('Summary metrics', 'speedometer2', rowsFor('summary'), { highlightBest: true }),
      table('Config', 'sliders', rowsFor('config')),
      table('Run info', 'info-circle', meta));
  }

  function csv() {
    const rows = [...rowsFor('summary').map((r) => ({ ...r, key: `summary.${r.key}` })), ...rowsFor('config').map((r) => ({ ...r, key: `config.${r.key}` }))]
      .filter((r) => !diffOnly || !r.same);
    const cols = [{ label: 'key', value: (r) => r.key }, ...runs.map((run, i) => ({ label: displayName(run), value: (r) => r.vals[i] }))];
    return toCsv(cols, rows);
  }

  await load();
}
