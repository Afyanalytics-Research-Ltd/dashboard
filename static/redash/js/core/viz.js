// Renders Redash visualizations from their saved options.
//
//   const handle = await renderViz(el, visualization, queryResult, { height, onPointClick })
//   handle.resize(); handle.destroy();
//
// CHART (line/column/area/pie/scatter/bubble/heatmap/box) uses Plotly, which is what Redash
// itself uses, so saved options keep their meaning. COUNTER, TABLE, PIVOT, DETAILS and FUNNEL
// are native. Other types render as a table with a notice.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { emptyState } from '../../../ui/js/ui.js';
import { loadPlotly } from './lib.js';
import { resultsGrid } from './results.js';
import { fmtNumeral, numeralToD3, inferType, isNumericType, isDateType, fmtCell } from './format.js';

export const VIZ_TYPES = [
  { type: 'CHART', label: 'Chart', icon: 'bar-chart-line' },
  { type: 'TABLE', label: 'Table', icon: 'table' },
  { type: 'COUNTER', label: 'Counter', icon: '123' },
  { type: 'PIVOT', label: 'Pivot table', icon: 'grid-3x3' },
  { type: 'DETAILS', label: 'Details view', icon: 'card-list' },
  { type: 'FUNNEL', label: 'Funnel', icon: 'funnel' },
];

export const CHART_TYPES = [
  { value: 'column', label: 'Bar', icon: 'bar-chart' },
  { value: 'line', label: 'Line', icon: 'graph-up' },
  { value: 'area', label: 'Area', icon: 'graph-up-arrow' },
  { value: 'pie', label: 'Pie', icon: 'pie-chart' },
  { value: 'scatter', label: 'Scatter', icon: 'dot' },
  { value: 'bubble', label: 'Bubble', icon: 'circle' },
  { value: 'heatmap', label: 'Heatmap', icon: 'grid-3x3-gap' },
  { value: 'box', label: 'Box plot', icon: 'distribute-vertical' },
];

// Brand-led categorical palette (Afya blue/teal first), colour-blind friendly order.
export const PALETTE = ['#0072CE', '#0BB99F', '#F5A623', '#7C3AED', '#E5484D', '#3E63DD', '#12A594', '#D6409F', '#8E4EC6', '#978365', '#5B5BD6', '#E54D2E'];

export const vizTypeInfo = (type) => VIZ_TYPES.find((v) => v.type === type) || { type, label: type, icon: 'question-square' };

const columnsWithTypes = (data) => (data.columns || []).map((c) => ({ ...c, type: c.type && c.type !== 'string' ? c.type : inferType(data.rows || [], c.name) }));

/** Suggest the best visualization for a result: {type, name, options}. */
export function suggestViz(data) {
  const cols = columnsWithTypes(data);
  const rows = data.rows || [];
  const nums = cols.filter((c) => isNumericType(c.type));
  const dates = cols.filter((c) => isDateType(c.type));
  const cats = cols.filter((c) => !isNumericType(c.type) && !isDateType(c.type));
  if (rows.length === 1 && nums.length >= 1) {
    return { type: 'COUNTER', name: nums[0].friendly_name || nums[0].name, options: { counterColName: nums[0].name, rowNumber: 1, targetRowNumber: 1, stringDecimal: 0, stringDecChar: '.', stringThouSep: ',' } };
  }
  if (dates.length && nums.length) {
    return { type: 'CHART', name: `${nums.slice(0, 2).map((n) => n.name).join(', ')} over time`, options: defaultChartOptions('line', dates[0].name, nums.slice(0, 3).map((n) => n.name)) };
  }
  if (cats.length && nums.length) {
    const distinct = new Set(rows.map((r) => r[cats[0].name])).size;
    const type = distinct <= 6 && nums.length === 1 ? 'pie' : 'column';
    return { type: 'CHART', name: `${nums[0].name} by ${cats[0].name}`, options: defaultChartOptions(type, cats[0].name, nums.slice(0, type === 'pie' ? 1 : 3).map((n) => n.name)) };
  }
  if (nums.length >= 2) return { type: 'CHART', name: `${nums[1].name} vs ${nums[0].name}`, options: defaultChartOptions('scatter', nums[0].name, [nums[1].name]) };
  return { type: 'TABLE', name: 'Table', options: {} };
}

export function defaultChartOptions(seriesType, x, ys, series) {
  const columnMapping = { [x]: 'x' };
  ys.forEach((y) => { columnMapping[y] = 'y'; });
  if (series) columnMapping[series] = 'series';
  return {
    globalSeriesType: seriesType,
    sortX: true,
    legend: { enabled: true, placement: 'auto', traceorder: 'normal' },
    xAxis: { type: '-', labels: { enabled: true } },
    yAxis: [{ type: 'linear' }, { type: 'linear', opposite: true }],
    alignYAxesAtZero: false,
    error_y: { type: 'data', visible: true },
    series: { stacking: null, error_y: { type: 'data', visible: true } },
    seriesOptions: {},
    valuesOptions: {},
    columnMapping,
    direction: { type: 'counterclockwise' },
    sizemode: 'diameter',
    coefficient: 1,
    numberFormat: '0,0[.]00000',
    percentFormat: '0[.]00%',
    textFormat: '',
    missingValuesAsZero: true,
    showDataLabels: seriesType === 'pie',
    dateTimeFormat: 'DD/MM/YY HH:mm',
  };
}

// ------------------------------------------------------------------ CHART

function mapping(options) {
  const m = { x: null, y: [], series: null, size: null, zVal: null, yError: null };
  for (const [col, role] of Object.entries(options.columnMapping || {})) {
    if (role === 'y') m.y.push(col);
    else if (role in m) m[role] = col;
  }
  return m;
}

const plotlyType = {
  column: (o) => ({ type: 'bar', orientation: o.swappedAxes ? 'h' : 'v' }),
  line: () => ({ type: 'scatter', mode: 'lines+markers' }),
  area: () => ({ type: 'scatter', mode: 'lines', fill: 'tozeroy' }),
  scatter: () => ({ type: 'scatter', mode: 'markers' }),
  bubble: () => ({ type: 'scatter', mode: 'markers' }),
  box: () => ({ type: 'box' }),
};

/** Build Plotly traces + layout from Redash chart options. Exported for tests/preview. */
export function chartFigure(options, data, { colors = PALETTE } = {}) {
  const rows = data.rows || [];
  const m = mapping(options);
  const globalType = options.globalSeriesType || 'column';
  if (!m.x || (!m.y.length && globalType !== 'heatmap')) {
    return { error: 'Pick an X column and at least one Y column to draw this chart.' };
  }
  const xType = (data.columns || []).find((c) => c.name === m.x)?.type || inferType(rows, m.x);
  const stacking = options.series?.stacking;
  const fmt = numeralToD3(options.numberFormat);

  if (globalType === 'pie') {
    const yCol = m.y[0];
    const groups = new Map();
    for (const r of rows) groups.set(r[m.x], (groups.get(r[m.x]) || 0) + (Number(r[yCol]) || 0));
    return {
      traces: [{
        type: 'pie', labels: [...groups.keys()].map(String), values: [...groups.values()], hole: 0.45,
        marker: { colors }, textinfo: options.showDataLabels === false ? 'none' : 'percent',
        direction: options.direction?.type === 'clockwise' ? 'clockwise' : 'counterclockwise', sort: false,
        hovertemplate: `%{label}<br>%{value${fmt ? `:${fmt}` : ''}} (%{percent})<extra></extra>`,
      }],
      layout: { showlegend: options.legend?.enabled !== false },
    };
  }

  if (globalType === 'heatmap') {
    const zCol = m.zVal || m.y[0];
    const xs = [...new Set(rows.map((r) => r[m.x]))];
    const ys = [...new Set(rows.map((r) => r[m.series || m.y[1] || m.y[0]]))];
    const yKey = m.series || m.y[1] || m.y[0];
    const z = ys.map((yv) => xs.map((xv) => {
      const row = rows.find((r) => r[m.x] === xv && r[yKey] === yv);
      return row ? row[zCol] : null;
    }));
    return { traces: [{ type: 'heatmap', x: xs, y: ys, z, colorscale: options.colorScheme || 'Blues' }], layout: {} };
  }

  // Series: either one per Y column, or split by the "series" column.
  const series = [];
  if (m.series) {
    const yCol = m.y[0];
    const bySeries = new Map();
    for (const r of rows) {
      const key = String(r[m.series]);
      if (!bySeries.has(key)) bySeries.set(key, []);
      bySeries.get(key).push(r);
    }
    for (const [name, rs] of bySeries) series.push({ key: name, name, yCol, rows: rs });
  } else {
    for (const yCol of m.y) series.push({ key: yCol, name: yCol, yCol, rows });
  }

  const traces = series.map((s, i) => {
    const so = options.seriesOptions?.[s.key] || {};
    const type = so.type || globalType;
    let pts = s.rows.map((r) => ({ x: r[m.x], y: r[s.yCol], size: m.size ? r[m.size] : null, err: m.yError ? r[m.yError] : null }));
    if (options.missingValuesAsZero) pts = pts.map((p) => ({ ...p, y: p.y == null ? 0 : p.y }));
    if (options.sortX !== false) pts.sort((a, b) => (a.x > b.x ? 1 : a.x < b.x ? -1 : 0));
    const base = (plotlyType[type] || plotlyType.column)(options);
    const color = so.color || colors[i % colors.length];
    const horizontal = base.orientation === 'h';
    const t = {
      ...base,
      name: so.name || s.name,
      x: horizontal ? pts.map((p) => p.y) : pts.map((p) => p.x),
      y: horizontal ? pts.map((p) => p.x) : pts.map((p) => p.y),
      yaxis: so.yAxis === 1 ? 'y2' : 'y',
      marker: { color },
      line: { color, width: 2 },
      hovertemplate: `%{${horizontal ? 'y' : 'x'}}<br>${(so.name || s.name).replace(/%/g, '%%')}: %{${horizontal ? 'x' : 'y'}${fmt ? `:${fmt}` : ''}}<extra></extra>`,
    };
    if (type === 'bubble') {
      const sizes = pts.map((p) => Number(p.size) || 0);
      const max = Math.max(...sizes, 1);
      t.marker = { color, size: sizes, sizemode: options.sizemode || 'diameter', sizeref: (2 * max) / (40 ** 2) * (options.coefficient || 1), sizemin: 3 };
    }
    if (m.yError) t.error_y = { type: 'data', array: pts.map((p) => p.err), visible: true };
    if (stacking && (type === 'area' || type === 'line')) { t.stackgroup = 'one'; if (type === 'line') delete t.fill; }
    if (options.showDataLabels) {
      t.text = pts.map((p) => fmtNumeral(p.y, options.numberFormat));
      if (t.type === 'bar') t.textposition = 'auto';
      else t.mode = `${t.mode || 'markers'}+text`, t.textposition = 'top center';
    }
    t.customdata = pts.map((p) => p.x);
    return t;
  });

  const yAxis0 = options.yAxis?.[0] || {};
  const yAxis1 = options.yAxis?.[1] || {};
  const xAxis = options.xAxis || {};
  const horizontal = globalType === 'column' && options.swappedAxes;
  const axisType = (t) => (t === '-' || !t ? undefined : t === 'datetime' ? 'date' : t === 'logarithmic' ? 'log' : t);
  const xa = {
    title: { text: xAxis.title?.text || '' },
    type: axisType(xAxis.type) || (isDateType(xType) ? 'date' : undefined),
    showticklabels: xAxis.labels?.enabled !== false,
    automargin: true,
    tickmode: options.studioShowAllXLabels ? 'linear' : undefined,
  };
  const ya = {
    title: { text: yAxis0.title?.text || '' },
    type: axisType(yAxis0.type),
    range: yAxis0.rangeMin != null || yAxis0.rangeMax != null ? [yAxis0.rangeMin ?? null, yAxis0.rangeMax ?? null] : undefined,
    tickformat: fmt || undefined,
    automargin: true,
    zeroline: true,
  };
  const layout = {
    barmode: stacking ? 'stack' : 'group',
    xaxis: horizontal ? ya : xa,
    yaxis: horizontal ? xa : ya,
    showlegend: options.legend?.enabled !== false && traces.length > 1,
  };
  if (traces.some((t) => t.yaxis === 'y2')) {
    layout.yaxis2 = { overlaying: 'y', side: 'right', type: axisType(yAxis1.type), title: { text: yAxis1.title?.text || '' }, automargin: true };
  }
  // Studio extension: horizontal reference lines (e.g. target / average).
  // Reference lines sit on the value axis: horizontal normally, vertical for horizontal bars.
  if (options.studioRefLines?.length) {
    const line = (l) => ({ color: l.color || '#E5484D', dash: 'dash', width: 1.5 });
    layout.shapes = options.studioRefLines.map((l) => (horizontal
      ? { type: 'line', yref: 'paper', y0: 0, y1: 1, x0: l.value, x1: l.value, line: line(l) }
      : { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: l.value, y1: l.value, line: line(l) }));
    layout.annotations = options.studioRefLines.filter((l) => l.label).map((l) => (horizontal
      ? { yref: 'paper', y: 1, x: l.value, text: l.label, showarrow: false, xanchor: 'left', yanchor: 'top', font: { size: 11, color: l.color || '#E5484D' } }
      : { xref: 'paper', x: 1, y: l.value, text: l.label, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 11, color: l.color || '#E5484D' } }));
  }
  return { traces, layout };
}

async function renderChart(el, viz, data, { height, onPointClick }) {
  const fig = chartFigure(viz.options || {}, data);
  if (fig.error) { mount(el, emptyState({ icon: 'bar-chart', title: 'Chart not configured', message: fig.error })); return {}; }
  const Plotly = await loadPlotly();
  const plotEl = h('div.rd-plot', { style: { height: `${height}px` } });
  mount(el, plotEl);
  const layout = {
    margin: { l: 50, r: 20, t: 10, b: 40 },
    font: { family: 'Inter, system-ui, sans-serif', size: 12, color: '#334155' },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    legend: { orientation: 'h', y: -0.18 },
    hoverlabel: { bgcolor: '#0f172a', bordercolor: '#0f172a', font: { color: '#e2e8f0' } },
    colorway: PALETTE,
    ...fig.layout,
    xaxis: { gridcolor: '#eef2f7', linecolor: '#cbd5e1', ...fig.layout.xaxis },
    yaxis: { gridcolor: '#eef2f7', linecolor: '#cbd5e1', ...fig.layout.yaxis },
  };
  await Plotly.newPlot(plotEl, fig.traces, layout, {
    responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
  });
  if (onPointClick) {
    plotEl.on('plotly_click', (ev) => {
      const p = ev.points?.[0];
      if (p) onPointClick({ x: p.customdata ?? p.label ?? p.x, y: p.value ?? p.y, series: p.data.name });
    });
  }
  return {
    resize: () => Plotly.Plots.resize(plotEl),
    destroy: () => Plotly.purge(plotEl),
    toImage: (opts) => Plotly.toImage(plotEl, { format: 'png', width: 1200, height: 700, ...opts }),
  };
}

// ------------------------------------------------------------------ COUNTER

export function counterValues(options, data) {
  const rows = data.rows || [];
  const o = options || {};
  const pick = (col, rowNumber) => {
    if (!col) return null;
    const r = rows[Math.max(0, (Number(rowNumber) || 1) - 1)];
    return r ? r[col] : null;
  };
  let value = o.countRow ? rows.length : pick(o.counterColName, o.rowNumber);
  const target = o.targetColName ? pick(o.targetColName, o.targetRowNumber) : null;
  const fmt = (v) => {
    if (v == null || v === '') return '—';
    if (typeof v !== 'number' && Number.isNaN(Number(v))) return String(v);
    const n = Number(v);
    const dec = Number(o.stringDecimal) || 0;
    let s = n.toFixed(dec);
    const [int, frac] = s.split('.');
    s = int.replace(/\B(?=(\d{3})+(?!\d))/g, o.stringThouSep ?? ',') + (frac ? (o.stringDecChar || '.') + frac : '');
    return `${o.stringPrefix || ''}${s}${o.stringSuffix || ''}`;
  };
  const delta = target != null && value != null && Number(target) !== 0 ? (Number(value) - Number(target)) / Math.abs(Number(target)) : null;
  return { value, target, display: fmt(value), targetDisplay: o.formatTargetValue ? fmt(target) : target, delta, format: fmt };
}

/** Count a number up from 0 in el (eased, ~700ms); skips when reduced motion is preferred. */
function countUp(el, value, format) {
  const n = Number(value);
  if (!Number.isFinite(n) || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const t0 = performance.now();
  const step = (t) => {
    const p = Math.min(1, (t - t0) / 700);
    const eased = 1 - (1 - p) ** 3;
    el.textContent = p < 1 ? format(Number.isInteger(n) ? Math.round(n * eased) : n * eased) : format(n);
    if (p < 1 && el.isConnected) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderCounter(el, viz, data) {
  const c = counterValues(viz.options, data);
  const trend = c.delta == null ? null : c.delta > 0 ? 'up' : c.delta < 0 ? 'down' : 'flat';
  const valueEl = h('div.rd-counter-value', { class: trend && `trend-${trend}` }, c.display);
  if (!viz.options?.countRow && typeof c.value === 'number') countUp(valueEl, c.value, c.format);
  mount(el, h('div.rd-counter',
    valueEl,
    c.target != null && h('div.rd-counter-target',
      trend && h('span.rd-counter-delta', { class: `trend-${trend}` }, icon(trend === 'up' ? 'arrow-up-right' : trend === 'down' ? 'arrow-down-right' : 'dash'), ` ${(c.delta * 100).toFixed(1)}%`),
      h('span.text-muted', ` vs ${c.targetDisplay ?? '—'}`)),
    viz.options?.counterLabel && h('div.rd-counter-label', viz.options.counterLabel)));
  return {};
}

// ------------------------------------------------------------------ PIVOT

const AGGREGATORS = {
  Count: (vals) => vals.length,
  Sum: (vals) => vals.reduce((a, v) => a + (Number(v) || 0), 0),
  Average: (vals) => (vals.length ? vals.reduce((a, v) => a + (Number(v) || 0), 0) / vals.length : null),
  Minimum: (vals) => (vals.length ? Math.min(...vals.map(Number)) : null),
  Maximum: (vals) => (vals.length ? Math.max(...vals.map(Number)) : null),
  'Count Unique Values': (vals) => new Set(vals).size,
};
export const PIVOT_AGGREGATORS = Object.keys(AGGREGATORS);

export function pivotData(options, data) {
  const o = options || {};
  const rowsBy = o.rows || [];
  const colsBy = o.cols || [];
  const val = (o.vals || [])[0];
  const agg = AGGREGATORS[o.aggregatorName] || AGGREGATORS.Count;
  const key = (r, cols) => cols.map((c) => String(r[c] ?? 'null')).join(' · ');
  const cells = new Map();
  const rowKeys = new Set();
  const colKeys = new Set();
  for (const r of data.rows || []) {
    const rk = key(r, rowsBy);
    const ck = key(r, colsBy);
    rowKeys.add(rk); colKeys.add(ck);
    const k = `${rk}\u0000${ck}`;
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(val ? r[val] : 1);
  }
  const sortKeys = (s) => [...s].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return { rowKeys: sortKeys(rowKeys), colKeys: sortKeys(colKeys), value: (rk, ck) => { const v = cells.get(`${rk}\u0000${ck}`); return v ? agg(v) : null; }, rowsBy, colsBy, agg: o.aggregatorName || 'Count', val };
}

function renderPivot(el, viz, data) {
  const p = pivotData(viz.options, data);
  const all = p.rowKeys.flatMap((rk) => p.colKeys.map((ck) => p.value(rk, ck))).filter((v) => v != null);
  const max = Math.max(...all.map(Number), 1);
  const heat = viz.options?.rendererName?.includes('Heatmap');
  const fmt = (v) => (v == null ? '' : fmtNumeral(v, '0,0[.]00'));
  mount(el, h('div.rd-pivot-wrap', h('table.rd-table.rd-pivot',
    h('thead', h('tr',
      h('th', `${p.rowsBy.join(' · ') || '—'} \\ ${p.colsBy.join(' · ') || ''}`),
      p.colKeys.map((ck) => h('th.align-right', ck || 'Total')),
      p.colKeys.length > 1 && h('th.align-right', 'Total'))),
    h('tbody', p.rowKeys.map((rk) => {
      const vals = p.colKeys.map((ck) => p.value(rk, ck));
      return h('tr',
        h('th', rk || 'Total'),
        vals.map((v) => h('td.align-right', { style: heat && v != null ? { background: `rgba(0,114,206,${0.08 + 0.6 * (Number(v) / max)})` } : null }, fmt(v))),
        p.colKeys.length > 1 && h('td.align-right.fw-semibold', fmt(vals.reduce((a, v) => a + (Number(v) || 0), 0))));
    })))),
  h('div.rd-pivot-caption', `${p.agg}${p.val ? ` of ${p.val}` : ''}`));
  return {};
}

// ------------------------------------------------------------------ DETAILS / FUNNEL

function renderDetails(el, viz, data) {
  const rows = data.rows || [];
  let i = 0;
  const body = h('div');
  const cols = columnsWithTypes(data);
  const draw = () => mount(body,
    h('dl.rd-details', cols.map((c) => [h('dt', c.friendly_name || c.name), h('dd', fmtCell(rows[i]?.[c.name], c.type) || h('span.rd-null', 'null'))])),
    rows.length > 1 && h('div.rd-details-nav',
      h('button.btn.btn-sm.btn-outline-secondary', { disabled: i === 0, onclick: () => { i--; draw(); } }, icon('chevron-left')),
      h('span', `${i + 1} / ${rows.length}`),
      h('button.btn.btn-sm.btn-outline-secondary', { disabled: i >= rows.length - 1, onclick: () => { i++; draw(); } }, icon('chevron-right'))));
  draw();
  mount(el, body);
  return {};
}

function renderFunnel(el, viz, data) {
  const o = viz.options || {};
  const step = o.stepCol?.colName || o.stepCol;
  const value = o.valueCol?.colName || o.valueCol;
  if (!step || !value) { mount(el, emptyState({ icon: 'funnel', title: 'Funnel not configured', message: 'Choose a step column and a value column.' })); return {}; }
  const sortKey = o.sortKeyCol?.colName;
  let rows = (data.rows || []).map((r) => ({ step: r[step], value: Number(r[value]) || 0, key: sortKey ? r[sortKey] : null }));
  // Redash: autoSort orders steps by value (largest first); otherwise an explicit sort column, else query order.
  if (o.autoSort) rows.sort((a, b) => b.value - a.value);
  else if (sortKey) rows.sort((a, b) => (a.key > b.key ? 1 : a.key < b.key ? -1 : 0));
  const first = rows[0]?.value || 1;
  mount(el, h('div.rd-funnel', rows.map((r, i) => {
    const pctFirst = r.value / first;
    const prev = i ? rows[i - 1].value : r.value;
    return h('div.rd-funnel-row',
      h('div.rd-funnel-step', String(r.step)),
      h('div.rd-funnel-bar-wrap', h('div.rd-funnel-bar', { style: { width: `${Math.max(pctFirst * 100, 1)}%` } }, fmtNumeral(r.value, '0,0'))),
      h('div.rd-funnel-pct', `${(pctFirst * 100).toFixed(1)}%`),
      h('div.rd-funnel-pct.text-muted', i ? `${((r.value / (prev || 1)) * 100).toFixed(1)}% of prev` : ''));
  })));
  return {};
}

// ------------------------------------------------------------------ entry point

/**
 * renderViz(el, viz, queryResult|data, opts) — renders and returns a handle {resize, destroy, toImage?}.
 * opts: { height = 320, onPointClick, settings, gridMaxHeight }
 */
export async function renderViz(el, viz, result, opts = {}) {
  const data = result?.data || result || { columns: [], rows: [] };
  const height = opts.height || 320;
  if (!data.columns?.length) {
    mount(el, emptyState({ icon: 'hourglass', title: 'No results yet', message: 'Run the query to see this visualization.' }));
    return {};
  }
  if (!data.rows?.length && viz.type !== 'TABLE') {
    mount(el, emptyState({ icon: 'inbox', title: 'No data', message: 'The query returned no rows for these parameters.' }));
    return {};
  }
  try {
    switch (viz.type) {
      case 'CHART': return await renderChart(el, viz, data, { height, onPointClick: opts.onPointClick });
      case 'COUNTER': return renderCounter(el, viz, data);
      case 'PIVOT': return renderPivot(el, viz, data);
      case 'DETAILS': return renderDetails(el, viz, data);
      case 'FUNNEL': return renderFunnel(el, viz, data);
      case 'TABLE': {
        const grid = resultsGrid(data, { tableOptions: viz.options, settings: opts.settings, compact: opts.compact, maxHeight: opts.gridMaxHeight });
        mount(el, grid.el);
        return { grid };
      }
      default: {
        const grid = resultsGrid(data, { settings: opts.settings, compact: true, maxHeight: opts.gridMaxHeight });
        mount(el, h('div.rd-viz-unsupported', h('div.af-callout.af-callout-info.mb-2', icon('info-circle'),
          h('span', `${vizTypeInfo(viz.type).label} visualizations are shown as a table here.`)), grid.el));
        return { grid };
      }
    }
  } catch (err) {
    console.error(err);
    mount(el, h('div.af-error', icon('exclamation-octagon', 'af-error-icon'), h('div', h('div.af-error-title', 'Could not draw this visualization'), h('div.af-error-msg', err.message))));
    return {};
  }
}
