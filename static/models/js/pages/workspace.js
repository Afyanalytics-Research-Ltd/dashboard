// Project workspace: run selector + auto-generated metric panels.
//
// Fixes for W&B complaints: panels render lazily and fetch server-side bucketed history for
// all visible runs × pending metrics in one batched request; WebGL traces; instant panel
// search; grouping by any field or expression with mean ± min/max; named saved views stored
// in Django (?view=<id>); PNG/CSV export per panel; refresh pauses while you interact.

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import {
  card, emptyState, errorState, skeleton, toast, toastError, formModal, confirmDialog, openMenu,
  refreshControl, drawer, copyButton, timeAgo, spinner,
} from '../../../ui/js/ui.js';
import { fmtNumber } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { savedViews } from '../core/api.js';
import { urls } from '../core/urls.js';
import { loadAllRuns, loadMetricKeys, loadHistories } from '../core/data.js';
import { lineChart, lazy, scatterRuns, parallelCoords, barsH, downloadText } from '../core/charts.js';
import { groupMetrics, runColor, displayName, runStateInfo, RUN_STATES, fmtMetric } from '../core/wb.js';
import {
  projectHeader, groupRuns, aggregateSeries, robustRange, catalog, runValue, columnLabel,
  notesDialog, runMenu, inlineText,
} from './workspace/common.js';

const X_AXES = [
  { value: '_step', label: 'Step' },
  { value: '_runtime', label: 'Relative time (s)' },
  { value: '_timestamp', label: 'Wall time' },
];
const SMOOTHERS = [
  { value: 'none', label: 'No smoothing' },
  { value: 'ema', label: 'Exponential (EMA)' },
  { value: 'twema', label: 'Time-weighted EMA' },
  { value: 'gaussian', label: 'Gaussian' },
  { value: 'average', label: 'Running average' },
];
const SIZES = { s: 240, m: 310, l: 460 };
const DEFAULT_STATE = {
  search: '', states: [], groupBy: '', showN: 10, hidden: [], extra: [],
  smoothing: { type: 'ema', weight: 0.6 }, x: '_step', logY: false, showOriginal: true, bands: true,
  clip: false, panelSize: 'm', panelFilter: '', collapsed: [], customPanels: [], expanded: [],
};

export default async function render(ctx) {
  const { entity, project } = ctx.params;
  const { el, signal } = ctx;
  const prefKey = `ms:ws:${entity}/${project}`;
  let state = { ...structuredClone(DEFAULT_STATE), ...prefs.get(prefKey, {}) };
  if (ctx.query.get('group') != null) state.groupBy = ctx.query.get('group');
  if (ctx.query.get('q') != null) state.search = ctx.query.get('q');
  if (ctx.query.get('x')) state.x = ctx.query.get('x');

  let runs = [];
  let total = 0;
  let metricKeys = [];
  let views = [];
  let activeView = null;
  let lastLoaded = null;
  let highlight = null;
  let lastInteraction = 0;
  const panels = new Map(); // id -> { kind, metric?, el, chart, drawn, cfg? }
  const cache = new Map(); // `${x}|${run}|${metric}` -> points

  // ---------------------------------------------------------------- skeleton
  const headerEl = h('div');
  const sideEl = h('aside.mp-ws-side', skeleton(10));
  const barEl = h('div.ms-chartbar.ms-sticky.mp-ws-bar');
  const panelsEl = h('div.mp-ws-panels', skeleton(6, { height: 40 }));
  const metaEl = h('span.mp-ws-meta');
  const refresh = refreshControl(() => reload({ quiet: true }), { key: 'ms-ws', initial: 30 });
  ctx.cleanup(refresh.stop);
  ['pointerdown', 'wheel', 'keydown'].forEach((ev) => el.addEventListener(ev, () => { lastInteraction = Date.now(); }, { passive: true }));

  mount(el, headerEl, h('div.mp-ws', sideEl, h('div.mp-ws-main', barEl, panelsEl)));

  const persist = debounce(() => {
    prefs.set(prefKey, state);
    ctx.setQuery({ group: state.groupBy || null, q: state.search || null, x: state.x !== '_step' ? state.x : null, view: activeView?.id || null });
  }, 250);

  // ---------------------------------------------------------------- data
  async function reload({ quiet = false } = {}) {
    if (quiet && Date.now() - lastInteraction < 8000) return; // pause refresh while the user is working
    try {
      const res = await loadAllRuns(entity, project, { order: '-createdAt', max: 1000, signal });
      if (!ctx.active) return;
      const prevRunning = new Set(runs.filter((r) => r.state === 'running').map((r) => r.name));
      runs = res.runs;
      total = res.total;
      lastLoaded = new Date();
      // Running runs gain new history: drop their cached points.
      for (const r of runs) if (r.state === 'running' || prevRunning.has(r.name)) invalidateRun(r.name);
      if (!metricKeys.length || !quiet) {
        metricKeys = (await loadMetricKeys(entity, project, runs.slice(0, 50).map((r) => r.name), { signal }))
          .filter((k) => k.numeric && !['_step', '_runtime', '_timestamp'].includes(k.key)).map((k) => k.key);
      }
      renderHeader();
      renderSide();
      renderBar();
      if (!quiet) renderPanels(); else redrawAll();
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (!quiet) mount(panelsEl, errorState(err, () => reload()));
      else toastError(err, 'Refresh failed');
    }
  }

  function invalidateRun(name) {
    for (const k of [...cache.keys()]) if (k.split('|')[1] === name) cache.delete(k);
  }

  function filteredRuns() {
    const q = state.search.toLowerCase();
    return runs.filter((r) => (!state.states.length || state.states.includes(r.state))
      && (!q || displayName(r).toLowerCase().includes(q) || r.name.toLowerCase().includes(q) || (r.tags || []).some((t) => t.toLowerCase().includes(q))
        || (r.group || '').toLowerCase().includes(q)));
  }

  function visibleRuns() {
    const f = filteredRuns();
    const top = f.slice(0, state.showN);
    const hidden = new Set(state.hidden);
    const extra = new Set(state.extra);
    return f.filter((r) => (top.includes(r) && !hidden.has(r.name)) || extra.has(r.name));
  }

  const isVisible = (r, vis = visibleRuns()) => vis.includes(r);
  const colorOf = (r) => runColor(r, runs.indexOf(r));

  // Batched history loader: panels ask for a metric; requests within 40ms merge into one call.
  let pending = new Set();
  let flushTimer = null;
  let waiters = [];
  function ensure(metric) {
    return new Promise((resolve, reject) => {
      pending.add(metric);
      waiters.push({ resolve, reject });
      clearTimeout(flushTimer);
      flushTimer = setTimeout(flush, 40);
    });
  }
  async function flush() {
    const metrics = [...pending];
    const ws = waiters;
    pending = new Set();
    waiters = [];
    const vis = visibleRuns();
    const need = vis.map((r) => r.name).filter((n) => metrics.some((m) => !cache.has(`${state.x}|${n}|${m}`)));
    try {
      if (need.length) {
        const res = await loadHistories(entity, project, need, metrics, { xKey: state.x, samples: 400, signal });
        for (const name of need) for (const m of metrics) cache.set(`${state.x}|${name}|${m}`, res.get(name)?.get(m) || []);
      }
      ws.forEach((w) => w.resolve());
    } catch (err) {
      ws.forEach((w) => w.reject(err));
    }
  }

  function seriesFor(metric) {
    const vis = visibleRuns();
    const pts = (r) => cache.get(`${state.x}|${r.name}|${metric}`) || [];
    if (state.groupBy) {
      return groupRuns(vis, state.groupBy).map((g, i) => ({
        name: `${g.key} (${g.runs.length})`, runName: `group:${g.key}`, color: runColor(null, i),
        points: aggregateSeries(g.runs.map((r) => ({ points: pts(r) }))),
      })).filter((s) => s.points.length);
    }
    return vis.map((r) => ({ name: displayName(r), runName: r.name, color: colorOf(r), points: pts(r) })).filter((s) => s.points.length);
  }

  function chartOpts(metric, panel) {
    const series = seriesFor(metric);
    return {
      series,
      opts: {
        height: state.panelSize === 'l' ? 340 : state.panelSize === 's' ? 220 : 270,
        smoothing: state.smoothing, showOriginal: state.showOriginal, bands: state.bands || !!state.groupBy,
        logY: panel?.logY ?? state.logY, xLabel: X_AXES.find((a) => a.value === state.x)?.label || state.x,
        yRange: state.clip ? robustRange(series) : undefined, highlight,
        onClick: ({ runName }) => { if (runName && !runName.startsWith('group:')) navigate(urls.run(entity, project, runName)); },
      },
    };
  }

  // ---------------------------------------------------------------- header
  function renderHeader() {
    const vis = visibleRuns();
    mount(headerEl, projectHeader({
      entity, project, active: 'workspace', icon: 'graph-up',
      subtitle: h('span', `${fmtNumber(total)} runs · ${metricKeys.length} metrics · `, metaEl),
      actions: [viewsControl(), h('button.btn.btn-sm.btn-outline-secondary', { onclick: openDecisionLog, title: 'All run notes in one place' }, icon('journal-text', 'me-1'), 'Decision log'),
        h('a.btn.btn-sm.btn-outline-primary', { href: urls.compare(entity, project, vis.slice(0, 20).map((r) => r.name)), title: 'Compare visible runs side by side' }, icon('layout-split', 'me-1'), 'Compare'),
        refresh.el],
    }));
    mount(metaEl, 'updated ', lastLoaded ? timeAgo(lastLoaded.toISOString()) : '—');
  }

  function viewsControl() {
    const label = activeView ? activeView.name : 'Default view';
    return h('button.btn.btn-sm.btn-outline-secondary.mp-views-btn', {
      type: 'button', title: 'Saved views',
      onclick: (e) => openMenu(e.currentTarget, [
        { header: 'Saved views' },
        { label: 'Default view', icon: activeView ? 'blank' : 'check2', onClick: () => applyView(null) },
        ...views.map((v) => ({ label: `${v.name}${v.shared ? '' : ' (private)'}`, icon: activeView?.id === v.id ? 'check2' : 'eye', onClick: () => applyView(v) })),
        { divider: true },
        { label: 'Save as new view…', icon: 'plus-lg', onClick: saveViewDialog },
        activeView && { label: `Update “${activeView.name}”`, icon: 'cloud-arrow-up', onClick: updateView },
        activeView && { label: 'Copy link to this view', icon: 'link-45deg', onClick: copyViewLink },
        activeView && { label: 'Delete view', icon: 'trash', danger: true, onClick: deleteView },
      ], 'end'),
    }, icon('bookmark-star', 'me-1'), h('span.mp-views-label', label), icon('chevron-down', 'ms-1 small'));
  }

  async function loadViews() {
    try { views = await savedViews.list(entity, project); } catch { views = []; }
    const id = Number(ctx.query.get('view'));
    if (id) {
      const v = views.find((x) => x.id === id);
      if (v) applyView(v, { quiet: true });
      else toast('That saved view no longer exists.', 'warning');
    }
  }
  function applyView(v, { quiet = false } = {}) {
    activeView = v;
    state = { ...structuredClone(DEFAULT_STATE), ...(v ? v.state : prefs.get(prefKey, {})) };
    if (!v) state = { ...structuredClone(DEFAULT_STATE) };
    persist();
    if (!quiet) toast(v ? `Applied view “${v.name}”` : 'Back to the default view', 'info', { timeout: 2000 });
    if (runs.length) { renderHeader(); renderSide(); renderBar(); renderPanels(); }
  }
  function saveViewDialog() {
    formModal({
      title: 'Save view', icon: 'bookmark-star',
      subtitle: 'Saves filters, visible runs, grouping, smoothing, axes and custom panels. Anyone with the link can open it.',
      fields: [
        { name: 'name', label: 'Name', required: true, placeholder: 'e.g. ResNet sweep — best runs' },
        { name: 'shared', type: 'checkbox', checkLabel: 'Share with all Model Studio users', value: true },
      ],
      onSubmit: async ({ name, shared }, close) => {
        const v = await savedViews.create(entity, project, { name, shared, state });
        views.push(v);
        activeView = v;
        persist();
        renderHeader();
        toast(`Saved view “${name}”`, 'success', { action: { label: 'Copy link', onClick: copyViewLink } });
        close();
      },
    });
  }
  async function updateView() {
    try {
      const v = await savedViews.update(activeView.id, { name: activeView.name, shared: activeView.shared, state });
      Object.assign(activeView, v);
      toast('View updated', 'success');
    } catch (err) { toastError(err, 'Could not update view'); }
  }
  async function deleteView() {
    if (!await confirmDialog({ title: `Delete view “${activeView.name}”?`, danger: true, confirmLabel: 'Delete', message: 'Links to this view will stop working.' })) return;
    try {
      await savedViews.remove(activeView.id);
      views = views.filter((v) => v.id !== activeView.id);
      activeView = null;
      persist();
      renderHeader();
      toast('View deleted', 'success');
    } catch (err) { toastError(err, 'Could not delete view'); }
  }
  function copyViewLink() {
    const url = `${location.origin}${urls.workspace(entity, project, { view: activeView.id })}`;
    navigator.clipboard?.writeText(url).then(() => toast('Link copied', 'success', { timeout: 1800 }), () => toast(url, 'info'));
  }

  // ---------------------------------------------------------------- run selector
  function renderSide() {
    const f = filteredRuns();
    const vis = visibleRuns();
    const groups = state.groupBy ? groupRuns(vis, state.groupBy) : null;
    const stateCounts = Object.fromEntries(Object.keys(RUN_STATES).map((s) => [s, runs.filter((r) => r.state === s).length]));
    const search = h('div.af-search.mp-side-search', icon('search', 'af-search-icon'),
      h('input.form-control.form-control-sm', {
        type: 'search', placeholder: 'Filter runs, tags, groups…', value: state.search, dataset: { searchShortcut: '' },
        oninput: debounce((e) => { state.search = e.target.value.trim(); persist(); renderSide(); redrawAll(); }, 200),
      }));
    const stateChips = h('div.mp-state-chips', Object.entries(stateCounts).filter(([, n]) => n).map(([s, n]) => {
      const info = runStateInfo(s);
      const on = state.states.includes(s);
      return h('button.mp-state-chip', {
        type: 'button', class: on && 'active', style: { '--c': info.color },
        onclick: () => { state.states = on ? state.states.filter((x) => x !== s) : [...state.states, s]; persist(); renderSide(); redrawAll(); },
      }, h('span.af-dot', { style: { background: info.color } }), info.label, h('b', n));
    }));
    const groupSel = groupSelect();
    const showN = h('select.form-select.form-select-sm.mp-shown', {
      'aria-label': 'Runs shown in charts',
      onchange: (e) => { state.showN = Number(e.target.value); state.hidden = []; state.extra = []; persist(); renderSide(); redrawAll(); },
    }, [5, 10, 20, 50, 100, 1000].map((n) => h('option', { value: n, selected: n === state.showN }, n === 1000 ? 'Show all' : `Show top ${n}`)));
    const list = h('div.mp-run-list');
    const LIMIT = 400;
    const rows = f.slice(0, LIMIT).map((r) => {
      const on = isVisible(r, vis);
      const row = h('div.mp-run-row', {
        class: [on && 'on', highlight === r.name && 'hl'],
        onmouseenter: () => setHighlight(r.name),
        onmouseleave: () => setHighlight(null),
      },
      h('button.mp-eye', { type: 'button', title: on ? 'Hide from charts' : 'Show in charts', 'aria-pressed': String(on), onclick: () => toggleRun(r, on) },
        icon(on ? 'eye-fill' : 'eye-slash')),
      h('span.ms-swatch', { style: { background: on ? colorOf(r) : '#e2e8f0' } }),
      h('a.mp-run-link', { href: urls.run(entity, project, r.name), title: `${displayName(r)} · ${r.name}` },
        h('span.mp-run-label', displayName(r)),
        h('span.mp-run-sub', [r.group, r.jobType].filter(Boolean).join(' · ') || r.name)),
      h('span.mp-run-state', { style: { color: runStateInfo(r.state).color }, title: runStateInfo(r.state).label }, icon(runStateInfo(r.state).icon)),
      h('button.af-icon-btn.mp-run-more', { type: 'button', title: 'Run actions', onclick: (e) => runMenu(e.currentTarget, r, { entity, project, onChanged: () => reload() }) }, icon('three-dots')));
      return row;
    });
    mount(list, rows.length ? rows : emptyState({ icon: 'search', title: 'No runs match', message: 'Clear the filter or state chips.' }),
      f.length > LIMIT && h('div.mp-more', `+${fmtNumber(f.length - LIMIT)} more — refine the filter`));
    mount(sideEl,
      h('div.mp-side-head', h('b', 'Runs'), h('span.text-muted.small', `${vis.length} of ${f.length} shown`),
        h('button.af-link-btn.ms-auto', { onclick: () => { state.hidden = []; state.extra = []; persist(); renderSide(); redrawAll(); } }, 'Reset')),
      search, stateChips, h('div.mp-side-ctrls', showN, groupSel),
      groups && h('div.mp-groups', groups.map((g, i) => h('div.mp-group-chip', h('span.ms-swatch', { style: { background: runColor(null, i) } }), h('span', g.key), h('b', g.runs.length)))),
      list);
  }

  function groupSelect() {
    const cat = catalog(runs);
    const opts = [
      ['', 'No grouping'], ['group', 'Group'], ['jobType', 'Job type'], ['state', 'State'], ['sweepName', 'Sweep'], ['tags', 'Tags'],
      ...cat.config.filter((c) => c.values.length <= 20).map((c) => [c.key, `config: ${c.name}`]),
    ];
    const known = opts.some(([v]) => v === state.groupBy);
    const sel = h('select.form-select.form-select-sm.mp-group-sel', {
      'aria-label': 'Group runs by',
      onchange: (e) => {
        if (e.target.value === '__expr') { expressionDialog(); e.target.value = state.groupBy; return; }
        state.groupBy = e.target.value; persist(); renderSide(); redrawAll();
      },
    },
    opts.map(([v, l]) => h('option', { value: v, selected: v === state.groupBy }, l)),
    !known && state.groupBy && h('option', { value: state.groupBy, selected: true }, `expr: ${state.groupBy}`),
    h('option', { value: '__expr' }, 'Expression…'));
    return sel;
  }

  function expressionDialog() {
    formModal({
      title: 'Group by expression', icon: 'braces',
      subtitle: 'Split runs into two groups by any condition. W&B disables grouping on expressions — here it just works.',
      fields: [{ name: 'expr', label: 'Expression', required: true, value: state.groupBy || 'summary.val/accuracy > 0.85',
        help: 'field operator value — fields: config.<name>, summary.<name>, state, group, jobType, tags. Operators: > >= < <= == != contains' }],
      submitLabel: 'Group',
      onSubmit: ({ expr }, close) => { state.groupBy = expr.trim(); persist(); renderSide(); redrawAll(); close(); },
    });
  }

  function toggleRun(r, on) {
    const top = filteredRuns().slice(0, state.showN);
    const inTop = top.includes(r);
    if (on) {
      if (inTop) state.hidden = [...new Set([...state.hidden, r.name])];
      else state.extra = state.extra.filter((n) => n !== r.name);
    } else if (inTop) state.hidden = state.hidden.filter((n) => n !== r.name);
    else state.extra = [...new Set([...state.extra, r.name])];
    persist();
    renderSide();
    redrawAll();
  }

  const setHighlight = (name) => {
    if (highlight === name) return;
    highlight = name;
    applyHighlight();
  };
  const applyHighlight = debounce(() => {
    for (const p of panels.values()) if (p.drawn && p.chart?.update && p.kind === 'line') p.chart.update(undefined, { highlight });
  }, 60);

  // ---------------------------------------------------------------- chart controls
  function renderBar() {
    const sm = state.smoothing;
    const weight = h('input', { type: 'range', min: 0, max: 0.99, step: 0.01, value: sm.weight, disabled: sm.type === 'none', 'aria-label': 'Smoothing weight' });
    const wLabel = h('span.ms-metric.mp-w', sm.type === 'none' ? '—' : sm.weight.toFixed(2));
    weight.addEventListener('input', debounce(() => { state.smoothing.weight = Number(weight.value); mount(wLabel, Number(weight.value).toFixed(2)); persist(); updateAll(); }, 80));
    const numericX = metricKeys.filter((k) => /epoch|step|iter/i.test(k)).slice(0, 8);
    const toggle = (key, label, ic) => h('button.btn.btn-sm.mp-toggle', {
      type: 'button', class: state[key] && 'active', 'aria-pressed': String(!!state[key]),
      onclick: (e) => { state[key] = !state[key]; e.currentTarget.classList.toggle('active', state[key]); persist(); updateAll(); },
    }, icon(ic, 'me-1'), label);
    mount(barEl,
      h('label', icon('activity'), h('select.form-select.form-select-sm', {
        'aria-label': 'Smoothing',
        onchange: (e) => { state.smoothing.type = e.target.value; weight.disabled = e.target.value === 'none'; mount(wLabel, e.target.value === 'none' ? '—' : state.smoothing.weight.toFixed(2)); persist(); updateAll(); },
      }, SMOOTHERS.map((s) => h('option', { value: s.value, selected: s.value === sm.type }, s.label)))),
      h('label.mp-weight', weight, wLabel),
      h('label', icon('arrows'), h('select.form-select.form-select-sm', {
        'aria-label': 'X axis',
        onchange: (e) => { state.x = e.target.value; persist(); redrawAll(); },
      }, [...X_AXES, ...numericX.map((k) => ({ value: k, label: k }))].map((a) => h('option', { value: a.value, selected: a.value === state.x }, a.label)))),
      h('div.mp-toggles', toggle('logY', 'Log Y', 'graph-up'), toggle('showOriginal', 'Raw', 'bezier2'), toggle('bands', 'Min/max', 'distribute-vertical'), toggle('clip', 'Clip outliers', 'scissors')),
      h('div.mp-size', ['s', 'm', 'l'].map((s) => h('button.mp-size-btn', {
        type: 'button', class: state.panelSize === s && 'active', title: { s: 'Small panels', m: 'Medium panels', l: 'Large panels' }[s],
        onclick: () => { state.panelSize = s; persist(); renderBar(); renderPanels(); },
      }, s.toUpperCase()))),
      h('div.af-search.mp-panel-search', icon('funnel', 'af-search-icon'), h('input.form-control.form-control-sm', {
        type: 'search', placeholder: 'Filter panels (e.g. val/ or acc)', value: state.panelFilter,
        oninput: debounce((e) => { state.panelFilter = e.target.value.trim(); persist(); renderPanels(); }, 150),
      })),
      h('button.btn.btn-sm.btn-primary.mp-add-panel', { type: 'button', onclick: (e) => addPanelMenu(e.currentTarget) }, icon('plus-lg', 'me-1'), 'Add panel'));
  }

  // ---------------------------------------------------------------- panels
  function renderPanels() {
    for (const p of panels.values()) { p.dispose?.(); p.chart?.destroy?.(); }
    panels.clear();
    panelsEl.style.setProperty('--ms-panel-min', `${SIZES[state.panelSize]}px`);
    if (!runs.length) {
      mount(panelsEl, emptyState({ icon: 'graph-up', title: 'No runs in this project yet', message: 'Log a run with the wandb SDK (see Home → Getting started) and it appears here live.' }));
      return;
    }
    const filter = state.panelFilter.toLowerCase();
    const keys = metricKeys.filter((k) => !filter || k.toLowerCase().includes(filter));
    // Biggest sections (usually train/ and val/) first; tiny one-off prefixes and "Charts" last.
    const sections = groupMetrics(keys).sort(([a, la], [b, lb]) => (a === 'Charts') - (b === 'Charts') || lb.length - la.length || a.localeCompare(b));
    const custom = state.customPanels.length ? [['Custom panels', null]] : [];
    if (!keys.length && !state.customPanels.length) {
      mount(panelsEl, emptyState({ icon: 'funnel', title: filter ? 'No panels match' : 'No numeric metrics logged', message: filter ? 'Clear the panel filter.' : 'Runs have not logged numeric history yet.' }));
      return;
    }
    mount(panelsEl,
      [...custom, ...sections].map(([name, list]) => {
        const collapsed = state.collapsed.includes(name);
        const grid = h('div.ms-panels');
        const sec = h('section.ms-section', { class: collapsed && 'collapsed' },
          h('div.ms-section-head', {
            role: 'button', tabindex: 0,
            onclick: () => {
              state.collapsed = collapsed ? state.collapsed.filter((x) => x !== name) : [...state.collapsed, name];
              persist();
              sec.classList.toggle('collapsed');
              if (collapsed) grid.querySelectorAll('.ms-panel').forEach((p) => panels.get(p.dataset.id)?.wake?.());
            },
          }, icon('chevron-down'), h('h3', name), h('span.ms-count', list ? list.length : state.customPanels.length)),
          grid);
        const items = list ? list.map((metric) => linePanel(metric)) : state.customPanels.map((cfg) => customPanel(cfg));
        grid.append(...items);
        return sec;
      }));
  }

  function panelShell(id, title, tools, { wide = false } = {}) {
    const body = h('div.ms-panel-body', h('div.ms-panel-loading'));
    const expanded = state.expanded.includes(id) || (wide && !state.expanded.includes(`-${id}`));
    const el2 = h('div.ms-panel', { dataset: { id }, class: expanded && 'expanded' },
      h('div.ms-panel-head', h('span.ms-panel-title', { title }, title),
        h('div.ms-panel-tools', tools,
          h('button.af-icon-btn', {
            type: 'button', title: 'Expand / shrink',
            onclick: () => {
              el2.classList.toggle('expanded');
              const on = el2.classList.contains('expanded');
              // Wide-by-default panels remember an explicit shrink as "-<id>".
              state.expanded = state.expanded.filter((x) => x !== id && x !== `-${id}`);
              if (on && !wide) state.expanded.push(id);
              if (!on && wide) state.expanded.push(`-${id}`);
              persist();
              setTimeout(() => panels.get(id)?.chart?.resize?.(), 220);
            },
          }, icon('arrows-angle-expand')))),
      body);
    return { el: el2, body };
  }

  function linePanel(metric) {
    const id = `m:${metric}`;
    const p = { kind: 'line', metric, drawn: false, logY: null };
    const tools = [
      h('button.af-icon-btn', { type: 'button', title: 'Toggle log scale', onclick: () => { p.logY = !(p.logY ?? state.logY); draw(); } }, icon('graph-up-arrow')),
      h('button.af-icon-btn', { type: 'button', title: 'Download PNG', onclick: () => p.chart?.toPng(metric.replace(/\W+/g, '_')) }, icon('image')),
      h('button.af-icon-btn', { type: 'button', title: 'Download CSV', onclick: () => p.chart && downloadText(p.chart.toCsv(), `${metric.replace(/\W+/g, '_')}.csv`) }, icon('filetype-csv')),
    ];
    const shell = panelShell(id, metric, tools);
    p.el = shell.el;
    async function draw() {
      try {
        await ensure(metric);
        if (!ctx.active) return;
        const { series, opts } = chartOpts(metric, p);
        if (!series.length) {
          p.chart?.destroy?.();
          p.chart = null;
          mount(shell.body, h('div.mp-panel-empty', icon('slash-circle'), 'No visible run logged this metric'));
          return;
        }
        if (p.chart) await p.chart.update(series, opts);
        else p.chart = await lineChart(shell.body, series, opts);
        p.drawn = true;
      } catch (err) {
        if (err.name === 'AbortError') return;
        mount(shell.body, h('div.mp-panel-empty.text-danger', icon('exclamation-triangle'), err.message));
      }
    }
    p.redraw = draw;
    p.update = () => { if (p.drawn && p.chart) { const { series, opts } = chartOpts(metric, p); p.chart.update(series, opts); } };
    p.dispose = lazy(shell.el, draw);
    p.wake = () => { if (!p.drawn) draw(); };
    panels.set(id, p);
    return shell.el;
  }

  // Custom panels: scatter / parallel coordinates / bar of a summary metric.
  function customPanel(cfg) {
    const id = `c:${cfg.id}`;
    const p = { kind: cfg.type, cfg, drawn: false };
    const title = cfg.type === 'scatter' ? `${columnLabel(cfg.y)} vs ${columnLabel(cfg.x)}`
      : cfg.type === 'bar' ? `${columnLabel(cfg.metric)} by run` : `Parallel coordinates → ${columnLabel(cfg.metric)}`;
    const tools = [
      h('button.af-icon-btn', { type: 'button', title: 'Download CSV', onclick: () => downloadText(customCsv(cfg), `${title.replace(/\W+/g, '_')}.csv`) }, icon('filetype-csv')),
      h('button.af-icon-btn', {
        type: 'button', title: 'Remove panel',
        onclick: () => { state.customPanels = state.customPanels.filter((c) => c.id !== cfg.id); persist(); renderPanels(); },
      }, icon('x-lg')),
    ];
    const shell = panelShell(id, title, tools, { wide: cfg.type === 'parcoords' });
    p.el = shell.el;
    const draw = async () => {
      const vis = visibleRuns();
      try {
        p.chart?.destroy?.();
        if (cfg.type === 'scatter') {
          const pts = vis.map((r) => ({ x: runValue(r, cfg.x), y: runValue(r, cfg.y), name: displayName(r), color: colorOf(r), runName: r.name }))
            .filter((q) => typeof q.x === 'number' && typeof q.y === 'number');
          if (!pts.length) { mount(shell.body, h('div.mp-panel-empty', 'No visible run has both values')); return; }
          p.chart = await scatterRuns(shell.body, pts, { xLabel: columnLabel(cfg.x), yLabel: columnLabel(cfg.y), logX: cfg.logX, onClick: (name) => navigate(urls.run(entity, project, name)) });
        } else if (cfg.type === 'bar') {
          const items = vis.map((r) => ({ label: displayName(r), value: runValue(r, cfg.metric), color: colorOf(r) })).filter((i) => typeof i.value === 'number')
            .sort((a, b) => b.value - a.value);
          if (!items.length) { mount(shell.body, h('div.mp-panel-empty', 'No visible run has this value')); return; }
          p.chart = await barsH(shell.body, items, { valueLabel: columnLabel(cfg.metric), format: '.4~g' });
        } else {
          const rs = vis.filter((r) => typeof runValue(r, cfg.metric) === 'number');
          if (!rs.length) { mount(shell.body, h('div.mp-panel-empty', 'No visible run has this metric')); return; }
          const dims = [...cfg.dims.map((d) => ({ label: columnLabel(d), values: rs.map((r) => runValue(r, d)) })),
            { label: columnLabel(cfg.metric), values: rs.map((r) => runValue(r, cfg.metric)) }];
          p.chart = await parallelCoords(shell.body, dims, { colorValues: rs.map((r) => runValue(r, cfg.metric)), colorLabel: columnLabel(cfg.metric), height: 320 });
        }
        p.drawn = true;
      } catch (err) {
        mount(shell.body, h('div.mp-panel-empty.text-danger', err.message));
      }
    };
    p.redraw = draw;
    p.update = draw;
    p.dispose = lazy(shell.el, draw);
    p.wake = () => { if (!p.drawn) draw(); };
    panels.set(id, p);
    return shell.el;
  }

  function customCsv(cfg) {
    const keys = cfg.type === 'scatter' ? [cfg.x, cfg.y] : cfg.type === 'bar' ? [cfg.metric] : [...cfg.dims, cfg.metric];
    const rows = [['run', ...keys.map(columnLabel)], ...visibleRuns().map((r) => [displayName(r), ...keys.map((k) => runValue(r, k))])];
    return rows.map((r) => r.map((v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v)).join(',')).join('\n');
  }

  function addPanelMenu(anchor) {
    openMenu(anchor, [
      { header: 'Add a panel' },
      { label: 'Scatter plot (run values)', icon: 'dot', onClick: () => customDialog('scatter') },
      { label: 'Parallel coordinates', icon: 'distribute-horizontal', onClick: () => customDialog('parcoords') },
      { label: 'Bar chart of a summary metric', icon: 'bar-chart-steps', onClick: () => customDialog('bar') },
    ]);
  }

  function customDialog(type) {
    const cat = catalog(runs);
    const numeric = [...cat.config.filter((c) => c.numeric), ...cat.summary.filter((s) => s.numeric)].map((c) => ({ value: c.key, label: c.key }));
    const summaryNum = cat.summary.filter((s) => s.numeric).map((s) => ({ value: s.key, label: s.name }));
    const configAll = cat.config.map((c) => ({ value: c.key, label: c.name }));
    const best = summaryNum.find((s) => /val.*acc|accuracy/i.test(s.label))?.value || summaryNum[0]?.value;
    const fields = type === 'scatter'
      ? [{ name: 'x', label: 'X axis', type: 'select', options: numeric, value: cat.config.find((c) => c.numeric)?.key },
        { name: 'y', label: 'Y axis', type: 'select', options: numeric, value: best },
        { name: 'logX', type: 'checkbox', checkLabel: 'Log scale X (good for learning rates)', value: false }]
      : type === 'bar'
        ? [{ name: 'metric', label: 'Summary metric', type: 'select', options: summaryNum, value: best }]
        : [{ name: 'dims', label: 'Hyperparameters (comma separated)', type: 'tags', value: configAll.slice(0, 5).map((c) => c.label),
          help: `Available: ${configAll.map((c) => c.label).join(', ')}` },
        { name: 'metric', label: 'Colour by metric', type: 'select', options: summaryNum, value: best }];
    formModal({
      title: { scatter: 'Add scatter plot', bar: 'Add bar chart', parcoords: 'Add parallel coordinates' }[type], icon: 'plus-square',
      fields,
      submitLabel: 'Add panel',
      onSubmit: (v, close) => {
        const cfg = { id: Math.random().toString(36).slice(2, 9), type, ...v };
        if (type === 'parcoords') {
          cfg.dims = v.dims.map((d) => (d.startsWith('config.') ? d : `config.${d}`)).filter((d) => cat.config.some((c) => c.key === d));
          if (!cfg.dims.length) throw new Error('Pick at least one hyperparameter from the list.');
        }
        state.customPanels = [...state.customPanels, cfg];
        state.collapsed = state.collapsed.filter((x) => x !== 'Custom panels');
        persist();
        renderPanels();
        close();
        setTimeout(() => panelsEl.querySelector('.ms-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
      },
    });
  }

  /** Settings that only restyle (smoothing, log, clip…) re-use cached data. */
  function updateAll() {
    for (const p of panels.values()) if (p.drawn) p.update?.();
  }
  /** Visible runs or x-axis changed: fetch what's missing, then redraw panels already on screen. */
  function redrawAll() {
    renderHeader();
    for (const p of panels.values()) if (p.drawn) p.redraw?.();
  }

  // ---------------------------------------------------------------- decision log
  function openDecisionLog() {
    const body = h('div.mp-log');
    const d = drawer({ title: 'Decision log', subtitle: `${entity}/${project} · run notes, newest first`, width: 640, body });
    const draw = () => {
      const withNotes = runs.filter((r) => r.notes);
      mount(body,
        h('p.text-muted.small', 'Teams keep run conclusions in spreadsheets because W&B buries notes. Here every note is one click away — and saved back to the run.'),
        withNotes.length ? withNotes.map((r) => h('div.mp-log-item',
          h('div.mp-log-head', h('span.ms-swatch', { style: { background: colorOf(r) } }),
            h('a.ms-run-name', { href: urls.run(entity, project, r.name), onclick: () => d.close() }, displayName(r)),
            h('span.text-muted.small.ms-auto', timeAgo(r.createdAt)),
            h('button.af-icon-btn', { title: 'Edit note', onclick: () => notesDialog(r, draw) }, icon('pencil'))),
          h('div.mp-log-note', r.notes),
          h('div.mp-log-meta', Object.entries(r.summary).filter(([k, v]) => typeof v === 'number' && /acc|loss/i.test(k)).slice(0, 4)
            .map(([k, v]) => h('span', h('b', k), ' ', fmtMetric(v))))))
          : h('div', h('div.af-empty', icon('journal', 'af-empty-icon'), h('div.af-empty-title', 'No notes yet'), h('div.af-empty-msg', 'Add one to a run below.'))),
        h('div.mt-3', h('label.form-label.small.fw-semibold', 'Add a note to a run'),
          h('select.form-select.form-select-sm', {
            onchange: (e) => { const r = runs.find((x) => x.name === e.target.value); if (r) notesDialog(r, draw); e.target.value = ''; },
          }, h('option', { value: '' }, '— pick a run —'), runs.filter((r) => !r.notes).slice(0, 300).map((r) => h('option', { value: r.name }, displayName(r))))));
    };
    draw();
  }

  // ---------------------------------------------------------------- go
  mount(panelsEl, h('div.mp-loading', spinner(), ' Loading runs and metrics…'));
  await Promise.all([reload(), loadViews()]);
  if (activeView && runs.length) renderPanels();
  void card; void copyButton; void inlineText;
}
