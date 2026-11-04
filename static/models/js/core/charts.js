// Metric charts for Model Studio.
//
//   const chart = await lineChart(el, series, { title, smoothing, xLabel, logY, height })
//   chart.update(series, opts); chart.destroy(); chart.toCsv(); chart.toPng()
//   lazy(el, () => draw())   // render only when the panel scrolls into view
//
// series: [{ name, color, points: [{x, y, min?, max?}], runName, dash? }]
// Downsampling happens server-side (bucketedHistory); smoothing is computed here.

import { h } from '../../../ui/js/dom.js';
import { loadPlotly } from '../../../ui/js/lib.js';

// ------------------------------------------------------------------ smoothing

/** W&B-compatible smoothing. type: none | ema | twema | gaussian | average. weight 0..0.999 (or window). */
export function smooth(points, type = 'ema', weight = 0.6) {
  if (!points.length || type === 'none' || !weight) return points;
  const ys = points.map((p) => p.y);
  let out;
  if (type === 'ema') {
    // Debiased exponential moving average (same as TensorBoard / W&B default).
    let last = 0;
    let n = 0;
    out = ys.map((y) => {
      if (y == null || !Number.isFinite(y)) return y;
      last = last * weight + (1 - weight) * y;
      n++;
      return last / (1 - weight ** n);
    });
  } else if (type === 'twema') {
    // Time-weighted EMA: weight decays with the x-gap, so uneven logging smooths consistently.
    const xs = points.map((p) => p.x);
    const span = (xs[xs.length - 1] - xs[0]) || 1;
    const halfLife = span * (weight / 10 + 0.001);
    let acc = null;
    out = ys.map((y, i) => {
      if (y == null || !Number.isFinite(y)) return y;
      if (acc == null) { acc = y; return y; }
      const dt = xs[i] - xs[i - 1];
      const a = Math.exp(-Math.LN2 * dt / halfLife);
      acc = a * acc + (1 - a) * y;
      return acc;
    });
  } else if (type === 'gaussian') {
    const sigma = Math.max(1, Math.round(weight * 20));
    const k = Array.from({ length: sigma * 6 + 1 }, (_, i) => Math.exp(-((i - sigma * 3) ** 2) / (2 * sigma * sigma)));
    out = ys.map((_, i) => {
      let s = 0;
      let w = 0;
      for (let j = -sigma * 3; j <= sigma * 3; j++) {
        const v = ys[i + j];
        if (v == null || !Number.isFinite(v)) continue;
        s += v * k[j + sigma * 3];
        w += k[j + sigma * 3];
      }
      return w ? s / w : ys[i];
    });
  } else {
    const win = Math.max(1, Math.round(weight * 50));
    let sum = 0;
    const q = [];
    out = ys.map((y) => {
      if (y == null || !Number.isFinite(y)) return y;
      q.push(y); sum += y;
      if (q.length > win) sum -= q.shift();
      return sum / q.length;
    });
  }
  return points.map((p, i) => ({ ...p, y: out[i] }));
}

/** Largest-Triangle-Three-Buckets downsampling (keeps visual shape). */
export function lttb(points, threshold = 1500) {
  if (points.length <= threshold) return points;
  const sampled = [points[0]];
  const every = (points.length - 2) / (threshold - 2);
  let a = 0;
  for (let i = 0; i < threshold - 2; i++) {
    const start = Math.floor((i + 1) * every) + 1;
    const end = Math.min(Math.floor((i + 2) * every) + 1, points.length);
    let ax = 0;
    let ay = 0;
    for (let j = start; j < end; j++) { ax += points[j].x; ay += points[j].y; }
    ax /= (end - start) || 1;
    ay /= (end - start) || 1;
    const rs = Math.floor(i * every) + 1;
    const re = Math.floor((i + 1) * every) + 1;
    let max = -1;
    let pick = rs;
    for (let j = rs; j < re; j++) {
      const area = Math.abs((points[a].x - ax) * (points[j].y - points[a].y) - (points[a].x - points[j].x) * (ay - points[a].y));
      if (area > max) { max = area; pick = j; }
    }
    sampled.push(points[pick]);
    a = pick;
  }
  sampled.push(points[points.length - 1]);
  return sampled;
}

// ------------------------------------------------------------------ history helpers

/**
 * Convert bucketedHistory rows for one metric into points with min/max bands.
 * rows: [{ bucketSize, _stepAvg|<x>Avg, <key>Avg, <key>Min, <key>Max, <key>Last, ... }]
 */
export function bucketsToPoints(rows, key, xKey = '_step') {
  const pts = [];
  for (const r of rows || []) {
    const x = r[`${xKey}Avg`] ?? r[`${xKey}Last`] ?? r[xKey];
    const y = r[`${key}Avg`] ?? r[`${key}Last`] ?? r[key];
    if (x == null || y == null || r[`${key}HasNaN`] && !Number.isFinite(y)) continue;
    pts.push({ x, y, min: r[`${key}Min`] ?? y, max: r[`${key}Max`] ?? y });
  }
  return pts.sort((a, b) => a.x - b.x);
}

/** sampledHistory rows -> points. */
export function rowsToPoints(rows, key, xKey = '_step') {
  return (rows || []).filter((r) => r[key] != null && r[xKey] != null && typeof r[key] === 'number')
    .map((r) => ({ x: r[xKey], y: r[key] })).sort((a, b) => a.x - b.x);
}

// ------------------------------------------------------------------ lazy rendering

const observers = new WeakMap();
/** Call draw() the first time el scrolls near the viewport. Returns a disposer. */
export function lazy(el, draw, { rootMargin = '300px' } = {}) {
  if (!('IntersectionObserver' in window)) { draw(); return () => {}; }
  let done = false;
  const io = new IntersectionObserver((entries) => {
    if (done || !entries.some((e) => e.isIntersecting)) return;
    done = true;
    io.disconnect();
    draw();
  }, { rootMargin });
  io.observe(el);
  observers.set(el, io);
  return () => io.disconnect();
}

// ------------------------------------------------------------------ line chart

const BASE_LAYOUT = {
  margin: { l: 52, r: 14, t: 8, b: 36 },
  font: { family: 'Inter, system-ui, sans-serif', size: 11.5, color: '#334155' },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  hovermode: 'x unified',
  hoverlabel: { bgcolor: '#0f172a', bordercolor: '#0f172a', font: { color: '#e2e8f0', size: 11.5 }, namelength: 40 },
  showlegend: false,
  xaxis: { gridcolor: '#eef2f7', linecolor: '#cbd5e1', zeroline: false, automargin: true },
  yaxis: { gridcolor: '#eef2f7', linecolor: '#cbd5e1', zeroline: false, automargin: true },
};

/**
 * opts: { height = 260, smoothing: {type, weight}, showOriginal = true, bands = true, logY, logX, xLabel,
 *         yRange: [min,max], highlight: runName, onHover(points), onClick(point), webgl = auto }
 */
export async function lineChart(el, series, opts = {}) {
  const Plotly = await loadPlotly();
  const plot = h('div.ms-plot', { style: { height: `${opts.height || 260}px` } });
  el.replaceChildren(plot);
  let current = { series, opts };

  const build = (ser, o) => {
    const total = ser.reduce((a, s) => a + s.points.length, 0);
    const gl = o.webgl ?? total > 4000;
    const type = gl ? 'scattergl' : 'scatter';
    const traces = [];
    const sm = o.smoothing || { type: 'ema', weight: 0.6 };
    for (const s of ser) {
      const pts = lttb(s.points, 2000);
      const dim = o.highlight && o.highlight !== s.runName;
      const smoothed = smooth(pts, sm.type, sm.weight);
      const isSmoothed = sm.type !== 'none' && sm.weight > 0;
      if (o.bands !== false && pts.some((p) => p.min !== p.max && p.min != null)) {
        traces.push({
          type: 'scatter', mode: 'lines', x: [...pts.map((p) => p.x), ...pts.map((p) => p.x).reverse()],
          y: [...pts.map((p) => p.max), ...pts.map((p) => p.min).reverse()], fill: 'toself', fillcolor: hexA(s.color, dim ? 0.03 : 0.1),
          line: { width: 0 }, hoverinfo: 'skip', showlegend: false,
        });
      }
      if (isSmoothed && o.showOriginal !== false) {
        traces.push({ type, mode: 'lines', x: pts.map((p) => p.x), y: pts.map((p) => p.y), line: { color: hexA(s.color, dim ? 0.08 : 0.25), width: 1 }, hoverinfo: 'skip', showlegend: false });
      }
      traces.push({
        type, mode: pts.length < 3 ? 'lines+markers' : 'lines', name: s.name,
        x: smoothed.map((p) => p.x), y: smoothed.map((p) => p.y),
        line: { color: dim ? hexA(s.color, 0.25) : s.color, width: o.highlight === s.runName ? 3 : 1.8, dash: s.dash },
        hovertemplate: `<b>${escapeHtml(s.name)}</b>: %{y:.4~g}<extra></extra>`,
        meta: s.runName,
      });
    }
    const layout = structuredClone(BASE_LAYOUT);
    layout.xaxis.title = { text: o.xLabel || '', standoff: 6, font: { size: 11 } };
    if (o.logX) layout.xaxis.type = 'log';
    if (o.logY) layout.yaxis.type = 'log';
    if (o.yRange) layout.yaxis.range = o.yRange;
    if (o.xRange) layout.xaxis.range = o.xRange;
    return { traces, layout };
  };

  const { traces, layout } = build(series, opts);
  await Plotly.newPlot(plot, traces, layout, {
    responsive: true, displaylogo: false, displayModeBar: 'hover',
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian'],
  });
  if (opts.onClick) plot.on('plotly_click', (ev) => { const p = ev.points?.[0]; if (p) opts.onClick({ x: p.x, y: p.y, runName: p.data.meta }); });
  if (opts.onHover) plot.on('plotly_hover', (ev) => opts.onHover(ev.points.map((p) => ({ runName: p.data.meta, x: p.x, y: p.y }))));
  if (opts.onZoom) plot.on('plotly_relayout', (ev) => {
    if ('xaxis.range[0]' in ev) opts.onZoom([ev['xaxis.range[0]'], ev['xaxis.range[1]']]);
    else if (ev['xaxis.autorange']) opts.onZoom(null);
  });

  return {
    el: plot,
    async update(nextSeries = current.series, nextOpts = current.opts) {
      current = { series: nextSeries, opts: { ...current.opts, ...nextOpts } };
      const b = build(current.series, current.opts);
      await Plotly.react(plot, b.traces, b.layout);
    },
    resize: () => Plotly.Plots.resize(plot),
    destroy: () => Plotly.purge(plot),
    toPng: (name = 'chart') => Plotly.downloadImage(plot, { format: 'png', filename: name, width: 1400, height: 700, scale: 2 }),
    toCsv: () => {
      const rows = [['run', 'x', 'y']];
      for (const s of current.series) for (const p of s.points) rows.push([s.name, p.x, p.y]);
      return rows.map((r) => r.map((v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v)).join(',')).join('\n');
    },
  };
}

function hexA(hex, a) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** Download text as a file (CSV export from any chart/table). */
export function downloadText(text, filename, type = 'text/csv') {
  const a = h('a', { href: URL.createObjectURL(new Blob([text], { type })), download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ------------------------------------------------------------------ parallel coordinates / scatter / bar

/** Parallel coordinates for sweeps/comparisons. dims: [{label, values, range?}], colorValues + colorLabel. */
export async function parallelCoords(el, dims, { colorValues, colorLabel, height = 360, onSelect } = {}) {
  const Plotly = await loadPlotly();
  const plot = h('div.ms-plot', { style: { height: `${height}px` } });
  el.replaceChildren(plot);
  const toDim = (d) => {
    const numeric = d.values.every((v) => typeof v === 'number' || v == null);
    if (numeric) return { label: d.label, values: d.values.map((v) => v ?? NaN), range: d.range };
    const cats = [...new Set(d.values.map(String))];
    return { label: d.label, values: d.values.map((v) => cats.indexOf(String(v))), tickvals: cats.map((_, i) => i), ticktext: cats };
  };
  await Plotly.newPlot(plot, [{
    type: 'parcoords',
    line: colorValues ? { color: colorValues, colorscale: [[0, '#E5484D'], [0.5, '#F5A623'], [1, '#0BB99F']], showscale: true, colorbar: { title: { text: colorLabel || '' }, thickness: 10 } } : { color: '#0072CE' },
    dimensions: dims.map(toDim),
    labelfont: { size: 11 }, tickfont: { size: 10 },
  }], { ...BASE_LAYOUT, margin: { l: 60, r: 60, t: 36, b: 20 }, hovermode: false }, { responsive: true, displaylogo: false });
  if (onSelect) plot.on('plotly_restyle', () => onSelect(plot.data[0].dimensions.map((d) => d.constraintrange || null)));
  return { el: plot, destroy: () => Plotly.purge(plot) };
}

/** Scatter of runs: points [{x, y, name, color, runName}]. */
export async function scatterRuns(el, points, { xLabel, yLabel, height = 300, onClick, logX, logY } = {}) {
  const Plotly = await loadPlotly();
  const plot = h('div.ms-plot', { style: { height: `${height}px` } });
  el.replaceChildren(plot);
  const layout = structuredClone(BASE_LAYOUT);
  layout.hovermode = 'closest';
  layout.xaxis.title = { text: xLabel || '' };
  layout.yaxis.title = { text: yLabel || '' };
  if (logX) layout.xaxis.type = 'log';
  if (logY) layout.yaxis.type = 'log';
  await Plotly.newPlot(plot, [{
    type: 'scatter', mode: 'markers', x: points.map((p) => p.x), y: points.map((p) => p.y),
    text: points.map((p) => p.name), customdata: points.map((p) => p.runName),
    marker: { color: points.map((p) => p.color || '#0072CE'), size: 10, line: { color: '#fff', width: 1 }, opacity: 0.9 },
    hovertemplate: '<b>%{text}</b><br>%{x:.4~g}, %{y:.4~g}<extra></extra>',
  }], layout, { responsive: true, displaylogo: false });
  if (onClick) plot.on('plotly_click', (ev) => ev.points?.[0] && onClick(ev.points[0].customdata));
  return { el: plot, destroy: () => Plotly.purge(plot) };
}

/** Horizontal bars, e.g. parameter importance: items [{label, value, color?}]. */
export async function barsH(el, items, { height, valueLabel = '', format = '.3f' } = {}) {
  const Plotly = await loadPlotly();
  const plot = h('div.ms-plot', { style: { height: `${height || Math.max(140, items.length * 28 + 50)}px` } });
  el.replaceChildren(plot);
  const layout = structuredClone(BASE_LAYOUT);
  layout.hovermode = 'closest';
  layout.margin.l = 140;
  layout.yaxis.autorange = 'reversed';
  layout.xaxis.title = { text: valueLabel };
  await Plotly.newPlot(plot, [{
    type: 'bar', orientation: 'h', y: items.map((i) => i.label), x: items.map((i) => i.value),
    marker: { color: items.map((i) => i.color || (i.value >= 0 ? '#0BB99F' : '#E5484D')) },
    hovertemplate: `%{y}: %{x:${format}}<extra></extra>`,
  }], layout, { responsive: true, displaylogo: false, displayModeBar: false });
  return { el: plot, destroy: () => Plotly.purge(plot) };
}

/** Histogram-over-time heatmap from W&B histogram rows: [{step, bins:[edges], values:[counts]}]. */
export async function histogramHeatmap(el, rows, { height = 260 } = {}) {
  const Plotly = await loadPlotly();
  const plot = h('div.ms-plot', { style: { height: `${height}px` } });
  el.replaceChildren(plot);
  if (!rows.length) return { el: plot, destroy() {} };
  const all = rows.flatMap((r) => r.bins);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const nb = 48;
  const edges = Array.from({ length: nb }, (_, i) => lo + ((hi - lo) * (i + 0.5)) / nb);
  const z = edges.map(() => rows.map(() => 0));
  rows.forEach((r, ci) => r.values.forEach((count, bi) => {
    const mid = (r.bins[bi] + r.bins[bi + 1]) / 2;
    const idx = Math.min(nb - 1, Math.max(0, Math.floor(((mid - lo) / ((hi - lo) || 1)) * nb)));
    z[idx][ci] += count;
  }));
  const layout = structuredClone(BASE_LAYOUT);
  layout.hovermode = 'closest';
  layout.xaxis.title = { text: 'step' };
  // Zero counts stay white so the distribution reads clearly (Plotly's 'Blues' paints them dark).
  const colorscale = [[0, '#ffffff'], [0.0001, '#e8f4ff'], [0.5, '#5aa9e6'], [1, '#003467']];
  await Plotly.newPlot(plot, [{ type: 'heatmap', x: rows.map((r) => r.step), y: edges, z, colorscale, zmin: 0, showscale: false,
    hovertemplate: 'step %{x}<br>value %{y:.3g}<br>count %{z}<extra></extra>' }], layout, { responsive: true, displaylogo: false });
  return { el: plot, destroy: () => Plotly.purge(plot) };
}
