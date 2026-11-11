// Run page — Charts and System tabs.
//
// Panels render lazily (IntersectionObserver). Panels that become visible in the same
// tick are fetched together in one bucketedHistory request, so a 500-metric run never
// fires 500 requests or freezes the browser.

import { h, icon, mount, debounce, prefs } from '../../../../ui/js/dom.js';
import { emptyState, errorState, toastError } from '../../../../ui/js/ui.js';
import { gql, parseJSON } from '../../core/api.js';
import { loadHistories } from '../../core/data.js';
import { lineChart, lazy, downloadText } from '../../core/charts.js';
import { groupMetrics, runColor, displayName } from '../../core/wb.js';
import { highlight } from './common.js';

const SMOOTHING = [
  { value: 'none', label: 'No smoothing' },
  { value: 'ema', label: 'Exponential (EMA)' },
  { value: 'twema', label: 'Time-weighted EMA' },
  { value: 'gaussian', label: 'Gaussian' },
  { value: 'average', label: 'Running average' },
];

/** Numeric history keys of a run (from historyKeys), excluding internals. */
export function numericKeys(run) {
  const keys = run.historyKeys?.keys || {};
  return Object.entries(keys)
    .filter(([, info]) => (info.typeCounts || []).some((t) => t.type === 'number'))
    .map(([k, info]) => ({ key: k, monotonic: !!info.monotonic }));
}

/** Batches history fetches issued within one frame into a single request per x-axis. */
function historyBatcher(entity, project, runName, signal) {
  let pending = new Map(); // xKey -> Map(key -> [resolve, reject][])
  let timer = null;
  const flush = async () => {
    const batch = pending;
    pending = new Map();
    timer = null;
    for (const [xKey, waiters] of batch) {
      const keys = [...waiters.keys()];
      for (let i = 0; i < keys.length; i += 40) {
        const chunk = keys.slice(i, i + 40);
        try {
          const res = await loadHistories(entity, project, [runName], chunk, { xKey, samples: 600, signal });
          const perKey = res.get(runName) || new Map();
          chunk.forEach((k) => waiters.get(k).forEach(([ok]) => ok(perKey.get(k) || [])));
        } catch (err) {
          chunk.forEach((k) => waiters.get(k).forEach(([, fail]) => fail(err)));
        }
      }
    }
  };
  return (key, xKey) => new Promise((resolve, reject) => {
    if (!pending.has(xKey)) pending.set(xKey, new Map());
    const m = pending.get(xKey);
    if (!m.has(key)) m.set(key, []);
    m.get(key).push([resolve, reject]);
    if (!timer) timer = setTimeout(flush, 40);
  });
}

function chartControls(state, onChange, { xOptions }) {
  const smoothSel = h('select.form-select.form-select-sm', { 'aria-label': 'Smoothing' },
    SMOOTHING.map((s) => h('option', { value: s.value, selected: s.value === state.smoothing.type }, s.label)));
  const weight = h('input', { type: 'range', min: 0, max: 0.99, step: 0.01, value: state.smoothing.weight, 'aria-label': 'Smoothing amount' });
  const wLabel = h('span.ms-metric.msr-w', state.smoothing.weight.toFixed(2));
  const xSel = h('select.form-select.form-select-sm', { 'aria-label': 'X axis' }, xOptions.map((x) => h('option', { value: x.value, selected: x.value === state.xKey }, x.label)));
  const logY = h('input.form-check-input', { type: 'checkbox', checked: state.logY });
  const bands = h('input.form-check-input', { type: 'checkbox', checked: state.bands });
  const fire = debounce(() => onChange({ ...state }), 60);
  smoothSel.addEventListener('change', () => { state.smoothing.type = smoothSel.value; weight.disabled = smoothSel.value === 'none'; fire(); });
  weight.addEventListener('input', () => { state.smoothing.weight = Number(weight.value); wLabel.textContent = Number(weight.value).toFixed(2); fire(); });
  weight.disabled = state.smoothing.type === 'none';
  xSel.addEventListener('change', () => { state.xKey = xSel.value; onChange({ ...state }, { xChanged: true }); });
  logY.addEventListener('change', () => { state.logY = logY.checked; fire(); });
  bands.addEventListener('change', () => { state.bands = bands.checked; fire(); });
  return h('div.ms-chartbar.msr-chartbar',
    h('label', icon('activity'), smoothSel),
    h('label', weight, wLabel),
    h('label', icon('arrows'), 'X', xSel),
    h('label.form-check.form-switch.m-0', logY, h('span.ms-1', 'Log Y')),
    h('label.form-check.form-switch.m-0', bands, h('span.ms-1', 'Min/max bands')));
}

function panel(key, { load, getState, color, name, onExpand }) {
  const body = h('div.ms-panel-body', h('div.ms-panel-loading'));
  let chart = null;
  let points = null;
  const draw = async () => {
    try {
      points = await load(key);
      if (!points.length) { mount(body, h('div.msr-panel-empty', icon('slash-circle'), ' No numeric values logged')); return; }
      const st = getState();
      chart = await lineChart(body, [{ name, runName: name, color, points }], {
        height: 230, smoothing: st.smoothing, logY: st.logY, bands: st.bands, xLabel: st.xKey.replace(/^_/, ''),
      });
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(body, h('div.msr-panel-empty.text-danger', icon('exclamation-triangle'), ` ${err.message}`));
    }
  };
  const el = h('div.ms-panel', { dataset: { key } },
    h('div.ms-panel-head',
      h('span.ms-panel-title', { title: key }, key),
      h('div.ms-panel-tools',
        h('button.af-icon-btn', { type: 'button', title: 'Expand', onclick: () => { el.classList.toggle('expanded'); setTimeout(() => chart?.resize(), 220); onExpand?.(); } }, icon('arrows-angle-expand')),
        h('button.af-icon-btn', { type: 'button', title: 'Download CSV', onclick: () => chart && downloadText(chart.toCsv(), `${key.replace(/\W+/g, '_')}.csv`) }, icon('filetype-csv')),
        h('button.af-icon-btn', { type: 'button', title: 'Download PNG', onclick: () => chart?.toPng(key.replace(/\W+/g, '_')) }, icon('image')))),
    body);
  lazy(el, draw);
  return {
    el,
    async restyle() {
      if (!chart || !points?.length) return;
      const st = getState();
      await chart.update(undefined, { smoothing: st.smoothing, logY: st.logY, bands: st.bands, xLabel: st.xKey.replace(/^_/, '') });
    },
    async reload() { chart?.destroy(); chart = null; mount(body, h('div.ms-panel-loading')); await draw(); },
    destroy() { chart?.destroy(); },
  };
}

/** Metric charts tab. */
export async function chartsTab(el, ctx, run) {
  const { entity, project } = ctx.params;
  const all = numericKeys(run).filter((k) => !k.key.startsWith('_') && !k.key.startsWith('system/'));
  if (!all.length) {
    mount(el, emptyState({ icon: 'graph-up', title: 'No metrics logged', message: 'Call wandb.log({...}) with numbers and they appear here as charts.' }));
    return;
  }
  const monotonic = numericKeys(run).filter((k) => k.monotonic && !k.key.startsWith('system/'));
  const xOptions = [{ value: '_step', label: 'Step' }, { value: '_runtime', label: 'Runtime (s)' }, { value: '_timestamp', label: 'Wall time' },
    ...monotonic.filter((k) => !k.key.startsWith('_')).map((k) => ({ value: k.key, label: k.key }))];
  const state = prefs.get('ms:run:charts', { smoothing: { type: 'ema', weight: 0.6 }, xKey: '_step', logY: false, bands: true });
  if (!xOptions.some((x) => x.value === state.xKey)) state.xKey = '_step';
  const signal = ctx.signal;
  let fetcher = historyBatcher(entity, project, run.name, signal);
  const color = runColor(run, 0);
  const name = displayName(run);
  const panels = [];
  const groupsEl = h('div.msr-groups');
  let q = '';

  const controls = chartControls(state, async (st, { xChanged } = {}) => {
    Object.assign(state, st);
    prefs.set('ms:run:charts', state);
    if (xChanged) { fetcher = historyBatcher(entity, project, run.name, signal); await Promise.all(panels.map((p) => p.reload())); } else await Promise.all(panels.map((p) => p.restyle()));
  }, { xOptions });

  const searchEl = h('input.form-control.form-control-sm.msr-metric-search', {
    type: 'search', placeholder: `Filter ${all.length} metrics…`, dataset: { searchShortcut: '' }, 'aria-label': 'Filter metrics',
    oninput: debounce((e) => { q = e.target.value.trim().toLowerCase(); build(); }, 150),
  });

  function build() {
    panels.splice(0).forEach((p) => p.destroy());
    const keys = all.map((k) => k.key).filter((k) => !q || k.toLowerCase().includes(q));
    if (!keys.length) { mount(groupsEl, h('div.msr-kv-empty', `No metrics match “${q}”.`)); return; }
    const collapsed = new Set(prefs.get('ms:run:collapsed', []));
    mount(groupsEl, groupMetrics(keys).map(([group, gkeys]) => {
      const sec = h('section.ms-section', { class: collapsed.has(group) && !q && 'collapsed' });
      const grid = h('div.ms-panels');
      const head = h('div.ms-section-head', { role: 'button', tabindex: 0, 'aria-expanded': String(!sec.classList.contains('collapsed')) },
        icon('chevron-down'), h('h3', highlight(group, q)), h('span.ms-count', String(gkeys.length)));
      const toggle = () => {
        sec.classList.toggle('collapsed');
        const set = new Set(prefs.get('ms:run:collapsed', []));
        sec.classList.contains('collapsed') ? set.add(group) : set.delete(group);
        prefs.set('ms:run:collapsed', [...set]);
        head.setAttribute('aria-expanded', String(!sec.classList.contains('collapsed')));
      };
      head.addEventListener('click', toggle);
      head.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
      gkeys.forEach((k) => {
        const p = panel(k, { load: (key) => fetcher(key, state.xKey), getState: () => state, color, name });
        panels.push(p);
        grid.appendChild(p.el);
      });
      sec.append(head, grid);
      return sec;
    }));
  }

  mount(el, h('div.msr-charts', h('div.msr-charts-bar.ms-sticky', controls, searchEl), groupsEl));
  build();
  ctx.cleanup(() => panels.forEach((p) => p.destroy()));
}

// ------------------------------------------------------------------ system metrics

const SYS_GROUPS = [
  { id: 'cpu', label: 'CPU', test: /cpu/i },
  { id: 'gpu', label: 'GPU', test: /gpu/i },
  { id: 'memory', label: 'Memory', test: /memory|mem/i },
  { id: 'disk', label: 'Disk', test: /disk/i },
  { id: 'network', label: 'Network', test: /network/i },
  { id: 'other', label: 'Other', test: /./ },
];

/** System metrics come from the run's event stream (events(samples)). */
export async function systemTab(el, ctx, run) {
  mount(el, h('div.ms-panel-loading'));
  const { entity, project } = ctx.params;
  let rows;
  try {
    const d = await gql(`query RunEvents($e: String!, $p: String!, $r: String!) {
      project(entityName: $e, name: $p) { run(name: $r) { eventsLineCount events(samples: 1000) } } }`,
    { e: entity, p: project, r: run.name }, { signal: ctx.signal });
    rows = (d.project?.run?.events || []).map((line) => parseJSON(line, null)).filter(Boolean);
  } catch (err) {
    if (err.name !== 'AbortError') mount(el, errorState(err, () => systemTab(el, ctx, run)));
    return;
  }
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r).filter((k) => k.startsWith('system.') && typeof r[k] === 'number')))].sort();
  if (!keys.length) {
    mount(el, emptyState({ icon: 'cpu', title: 'No system metrics', message: 'W&B samples CPU, memory, GPU, disk and network while a run is active. This run has none recorded.' }));
    return;
  }
  if (rows.length < 3) {
    // Too few samples for a line chart (short runs): show the readings as a snapshot instead.
    const last = rows[rows.length - 1] || {};
    mount(el, h('div.msr-sys',
      h('div.af-callout.af-callout-info', icon('info-circle'),
        h('span', `This run was short, so W&B recorded only ${rows.length} system sample${rows.length === 1 ? '' : 's'}. Showing the latest readings.`)),
      h('div.msr-sys-grid', keys.map((k) => h('div.msr-sys-card',
        h('div.msr-sys-label', k.replace(/^system\./, '')),
        h('div.msr-sys-value.ms-metric', Number.isInteger(last[k]) ? last[k].toLocaleString() : (last[k] ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 })),
        /percent/i.test(k) && h('div.af-meter', h('span', { style: { width: `${Math.min(100, last[k] || 0)}%`, background: (last[k] || 0) > 85 ? '#E5484D' : '#0072CE' } })))))));
    return;
  }
  const t0 = rows[0]?._timestamp;
  const xOf = (r) => r._runtime ?? (r._timestamp != null && t0 != null ? r._timestamp - t0 : null);
  const charts = [];
  const assigned = new Set();
  const sections = SYS_GROUPS.map((g) => {
    const gk = keys.filter((k) => !assigned.has(k) && g.test.test(k.replace(/^system\./, '')));
    gk.forEach((k) => assigned.add(k));
    return { ...g, keys: gk };
  }).filter((g) => g.keys.length);
  mount(el, h('div.msr-sys',
    h('div.msr-sys-summary', sections.map((s) => h('span.ms-pill.ms-pill-info', `${s.label} · ${s.keys.length}`)),
      h('span.small.text-muted', `${rows.length} samples · x-axis is runtime in seconds`)),
    sections.map((s) => {
      const grid = h('div.ms-panels', { style: { '--ms-panel-min': '320px' } });
      for (const k of s.keys) {
        const pts = rows.map((r) => ({ x: xOf(r), y: r[k] })).filter((p) => p.x != null && typeof p.y === 'number');
        const body = h('div.ms-panel-body', h('div.ms-panel-loading'));
        const label = k.replace(/^system\./, '');
        const pnl = h('div.ms-panel', h('div.ms-panel-head', h('span.ms-panel-title', { title: k }, label),
          h('div.ms-panel-tools', h('button.af-icon-btn', {
            type: 'button', title: 'Download CSV',
            onclick: () => downloadText(['runtime_s,value', ...pts.map((p) => `${p.x},${p.y}`)].join('\n'), `${label.replace(/\W+/g, '_')}.csv`),
          }, icon('filetype-csv')))), body);
        lazy(pnl, async () => {
          try {
            charts.push(await lineChart(body, [{ name: label, runName: run.name, color: '#7C3AED', points: pts }],
              { height: 190, smoothing: { type: 'none', weight: 0 }, xLabel: 'runtime (s)', bands: false }));
          } catch (err) { toastError(err); }
        });
        grid.appendChild(pnl);
      }
      return h('section.ms-section', h('div.ms-section-head', h('h3', s.label), h('span.ms-count', String(s.keys.length))), grid);
    })));
  ctx.cleanup(() => charts.forEach((c) => c.destroy()));
}
