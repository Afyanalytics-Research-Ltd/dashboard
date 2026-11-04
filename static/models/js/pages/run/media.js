// Run page — Media: image galleries (step slider + playback, resizable thumbnails, mask and
// box overlays with opacity / score threshold, caption filter, zoom-and-pan lightbox),
// logged tables (sortable, searchable, CSV) and histograms over time.

import { h, icon, mount, debounce, prefs } from '../../../../ui/js/dom.js';
import { modal, emptyState, errorState, skeleton, toast } from '../../../../ui/js/ui.js';
import { gql, fileUrl, fetchFile } from '../../core/api.js';
import { loadPlotly } from '../../../../ui/js/lib.js';
import { fileIndex, CLASS_COLORS, BOX_COLORS } from './common.js';
import { tableFromJson } from './files.js';

const IMAGE_TYPES = new Set(['images/separated', 'image-file', 'images']);

/** Media keys of a run from historyKeys: [{key, kind: 'images'|'table'|'histogram'}]. */
export function mediaKeys(run) {
  const out = [];
  for (const [key, info] of Object.entries(run.historyKeys?.keys || {})) {
    const types = (info.typeCounts || []).map((t) => t.type);
    if (types.some((t) => IMAGE_TYPES.has(t))) out.push({ key, kind: 'images' });
    else if (types.includes('table-file') || types.includes('partitioned-table')) out.push({ key, kind: 'table' });
    else if (types.includes('histogram')) out.push({ key, kind: 'histogram' });
  }
  return out.sort((a, b) => a.kind.localeCompare(b.kind) || a.key.localeCompare(b.key));
}

async function sampleKey(ctx, run, key, samples) {
  const { entity, project } = ctx.params;
  const d = await gql(`query MediaHist($e: String!, $p: String!, $r: String!, $specs: [JSONString!]!) {
    project(entityName: $e, name: $p) { run(name: $r) { sampledHistory(specs: $specs) } } }`,
  { e: entity, p: project, r: run.name, specs: [JSON.stringify({ keys: ['_step', key], samples })] }, { signal: ctx.signal });
  return (d.project?.run?.sampledHistory?.[0] || []).filter((r) => r[key] != null).sort((a, b) => a._step - b._step);
}

// ------------------------------------------------------------------ image model

/** Normalise one history value into [{path, caption, masks: {name: path}, boxes: {name: path}}]. */
export function imageItems(value) {
  if (!value) return [];
  if (value._type === 'images/separated' || Array.isArray(value.filenames)) {
    return (value.filenames || []).map((path, i) => ({
      path,
      caption: value.captions?.[i] ?? '',
      masks: Object.fromEntries(Object.entries(value.all_masks?.[i] || {}).map(([n, m]) => [n, m.path])),
      boxes: Object.fromEntries(Object.entries(value.all_boxes?.[i] || {}).map(([n, b]) => [n, b.path])),
      width: value.width, height: value.height,
    }));
  }
  if (value._type === 'image-file' && value.path) {
    return [{
      path: value.path, caption: value.caption ?? '',
      masks: Object.fromEntries(Object.entries(value.masks || {}).map(([n, m]) => [n, m.path])),
      boxes: Object.fromEntries(Object.entries(value.boxes || {}).map(([n, b]) => [n, b.path])),
      width: value.width, height: value.height,
    }];
  }
  return [];
}

const imgCache = new Map();
function loadImage(url) {
  if (!imgCache.has(url)) {
    imgCache.set(url, new Promise((resolve, reject) => {
      const im = new Image();
      im.decoding = 'async';
      im.onload = () => resolve(im);
      im.onerror = () => { imgCache.delete(url); reject(new Error('Image failed to load')); };
      im.src = url;
    }));
  }
  return imgCache.get(url);
}

const maskCache = new Map();
/** Colourised mask canvas (class id -> palette colour, alpha 255) for a mask PNG. */
async function maskCanvas(url) {
  if (!maskCache.has(url)) {
    maskCache.set(url, loadImage(url).then((im) => {
      const c = document.createElement('canvas');
      c.width = im.naturalWidth;
      c.height = im.naturalHeight;
      const g = c.getContext('2d', { willReadFrequently: true });
      g.drawImage(im, 0, 0);
      const W = c.width;
      const H = c.height;
      const data = g.getImageData(0, 0, W, H);
      const px = data.data;
      const cls = new Uint8Array(W * H);
      for (let i = 0, p = 0; i < px.length; i += 4, p++) cls[p] = px[i + 3] === 0 ? 0 : px[i];
      // Fill (drawn with the opacity slider) and a crisp contour (always opaque) so masks stay
      // legible even when the fill colour blends into the object underneath.
      const edge = document.createElement('canvas');
      edge.width = W;
      edge.height = H;
      const eg = edge.getContext('2d');
      const edata = eg.createImageData(W, H);
      const ep = edata.data;
      const classes = new Set();
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const p = y * W + x;
          const v = cls[p];
          const i = p * 4;
          if (!v) { px[i + 3] = 0; continue; }
          classes.add(v);
          const [r, gg, b] = CLASS_COLORS[v % CLASS_COLORS.length] || CLASS_COLORS[1];
          px[i] = r; px[i + 1] = gg; px[i + 2] = b; px[i + 3] = 255;
          const border = x === 0 || y === 0 || x === W - 1 || y === H - 1
            || cls[p - 1] !== v || cls[p + 1] !== v || cls[p - W] !== v || cls[p + W] !== v;
          if (border) { ep[i] = r; ep[i + 1] = gg; ep[i + 2] = b; ep[i + 3] = 255; }
        }
      }
      g.putImageData(data, 0, 0);
      eg.putImageData(edata, 0, 0);
      return { canvas: c, edge, classes: [...classes] };
    }));
  }
  return maskCache.get(url);
}

const jsonCache = new Map();
function loadJson(url) {
  if (!jsonCache.has(url)) jsonCache.set(url, fetchFile(url).then((t) => JSON.parse(t)).catch((e) => { jsonCache.delete(url); throw e; }));
  return jsonCache.get(url);
}

/** Box position in image pixels, supporting pixel {minX..} and fractional {middle,width,height} forms. */
function boxRect(box, w, hgt) {
  const p = box.position || {};
  let x0;
  let y0;
  let x1;
  let y1;
  if ('minX' in p) { x0 = p.minX; x1 = p.maxX; y0 = p.minY; y1 = p.maxY; } else if (p.middle) {
    x0 = p.middle[0] - p.width / 2; x1 = p.middle[0] + p.width / 2; y0 = p.middle[1] - p.height / 2; y1 = p.middle[1] + p.height / 2;
  } else return null;
  if (box.domain !== 'pixel') { x0 *= w; x1 *= w; y0 *= hgt; y1 *= hgt; }
  return { x0, y0, x1, y1 };
}

/**
 * Draw image + overlays onto canvas at display size `size` (CSS px).
 * opts: { masks: bool, maskOpacity, boxes: bool, minScore, labels: bool, zoom, panX, panY }
 */
async function composite(canvas, item, files, opts, size) {
  const file = files.get(item.path);
  if (!file) throw new Error(`Missing file ${item.path}`);
  const im = await loadImage(fileUrl(file.directUrl || file.url));
  const W = im.naturalWidth;
  const H = im.naturalHeight;
  const dpr = window.devicePixelRatio || 1;
  const scale = Math.min(size.w / W, size.h / H);
  const cw = Math.round(size.w * dpr);
  const ch = Math.round(size.h * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  const g = canvas.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, cw, ch);
  const zoom = opts.zoom || 1;
  const s = scale * zoom * dpr;
  const ox = (cw - W * s) / 2 + (opts.panX || 0) * dpr;
  const oy = (ch - H * s) / 2 + (opts.panY || 0) * dpr;
  g.setTransform(s, 0, 0, s, ox, oy);
  g.imageSmoothingEnabled = W * s / dpr > W * 2 ? false : true;
  g.drawImage(im, 0, 0);
  const found = { maskClasses: new Set(), boxes: 0, hiddenBoxes: 0 };
  if (opts.masks) {
    for (const path of Object.values(item.masks || {})) {
      const mf = files.get(path);
      if (!mf) continue;
      const m = await maskCanvas(fileUrl(mf.directUrl || mf.url));
      m.classes.forEach((c) => found.maskClasses.add(c));
      g.globalAlpha = opts.maskOpacity ?? 0.45;
      g.drawImage(m.canvas, 0, 0, W, H);
      g.globalAlpha = 1;
      if (opts.maskEdges !== false) g.drawImage(m.edge, 0, 0, W, H);
    }
  }
  if (opts.boxes) {
    for (const path of Object.values(item.boxes || {})) {
      const bf = files.get(path);
      if (!bf) continue;
      const meta = await loadJson(bf.directUrl || bf.url);
      for (const box of meta.box_data || []) {
        const score = Object.values(box.scores || {})[0];
        if (score != null && score < (opts.minScore || 0)) { found.hiddenBoxes++; continue; }
        const r = boxRect(box, W, H);
        if (!r) continue;
        found.boxes++;
        const [cr, cg, cb] = BOX_COLORS[(box.class_id ?? 1) % BOX_COLORS.length];
        const color = `rgb(${cr},${cg},${cb})`;
        g.lineWidth = (2 * dpr) / s; // ~2 CSS px regardless of zoom (transform is in image space)
        g.strokeStyle = color;
        g.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
        if (opts.labels !== false) {
          const label = `${box.box_caption || meta.class_labels?.[box.class_id] || box.class_id}${score != null ? ` ${score.toFixed(2)}` : ''}`;
          const fs = (11 * dpr) / s;
          g.font = `600 ${fs}px Inter, system-ui, sans-serif`;
          const tw = g.measureText(label).width + fs * 0.6;
          g.fillStyle = color;
          g.fillRect(r.x0, Math.max(0, r.y0 - fs * 1.35), tw, fs * 1.35);
          g.fillStyle = '#0f172a';
          g.fillText(label, r.x0 + fs * 0.3, Math.max(fs, r.y0 - fs * 0.35));
        }
      }
    }
  }
  return { W, H, ...found };
}

// ------------------------------------------------------------------ image section

function imageSection(ctx, run, key, files, rows) {
  const st = { ...{ masks: true, maskOpacity: 0.45, maskEdges: true, boxes: true, minScore: 0, labels: true, thumb: 150 }, ...prefs.get('ms:run:media', {}) };
  const save = () => prefs.set('ms:run:media', { masks: st.masks, maskOpacity: st.maskOpacity, maskEdges: st.maskEdges, boxes: st.boxes, minScore: st.minScore, labels: st.labels, thumb: st.thumb });
  let stepIdx = rows.length - 1;
  let filter = '';
  let playing = null;
  const grid = h('div.ms-gallery', { style: { '--ms-thumb': `${st.thumb}px` } });
  const stepLabel = h('span.msr-step-label.ms-metric');
  const slider = h('input.msr-step-slider', { type: 'range', min: 0, max: Math.max(0, rows.length - 1), value: stepIdx, 'aria-label': 'Step', disabled: rows.length < 2 });
  const info = h('span.small.text-muted');
  const hasMasks = rows.some((r) => imageItems(r[key]).some((i) => Object.keys(i.masks).length));
  const hasBoxes = rows.some((r) => imageItems(r[key]).some((i) => Object.keys(i.boxes).length));

  const items = () => imageItems(rows[stepIdx]?.[key]).filter((i) => !filter || String(i.caption).toLowerCase().includes(filter));
  const redraw = () => {
    stepLabel.textContent = `step ${rows[stepIdx]?._step ?? '—'}  ·  ${stepIdx + 1}/${rows.length}`;
    const list = items();
    info.textContent = `${list.length} image${list.length === 1 ? '' : 's'}`;
    if (!list.length) { mount(grid, h('div.msr-kv-empty', filter ? `No captions match “${filter}”.` : 'No images at this step.')); return; }
    mount(grid, list.map((item, i) => {
      const canvas = h('canvas', { 'aria-label': item.caption || item.path });
      const cell = h('button.ms-media.msr-media', { type: 'button', title: item.caption || item.path, onclick: () => lightbox(list, i) },
        canvas, item.caption && h('span.ms-media-cap', item.caption));
      requestAnimationFrame(() => composite(canvas, item, files, st, { w: st.thumb, h: st.thumb })
        .catch(() => cell.classList.add('msr-media-broken')));
      return cell;
    }));
  };
  slider.addEventListener('input', () => { stepIdx = Number(slider.value); redraw(); });
  const stepBy = (d) => { stepIdx = Math.min(rows.length - 1, Math.max(0, stepIdx + d)); slider.value = stepIdx; redraw(); };
  const playBtn = h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Play through steps', disabled: rows.length < 2 }, icon('play-fill'));
  playBtn.addEventListener('click', () => {
    if (playing) { clearInterval(playing); playing = null; mount(playBtn, icon('play-fill')); return; }
    if (stepIdx >= rows.length - 1) stepIdx = -1;
    mount(playBtn, icon('pause-fill'));
    playing = setInterval(() => { if (stepIdx >= rows.length - 1) { playBtn.click(); return; } stepBy(1); }, 700);
  });
  ctx.cleanup(() => clearInterval(playing));

  const sw = (k, label) => {
    const c = h('input.form-check-input', { type: 'checkbox', checked: st[k] });
    c.addEventListener('change', () => { st[k] = c.checked; save(); redraw(); });
    return h('label.form-check.form-switch.m-0', c, h('span.ms-1', label));
  };
  const range = (k, min, max, step, label, fmt = (v) => v) => {
    const r = h('input', { type: 'range', min, max, step, value: st[k], 'aria-label': label });
    const out = h('span.ms-metric.msr-w', fmt(st[k]));
    r.addEventListener('input', debounce(() => { st[k] = Number(r.value); out.textContent = fmt(st[k]); save(); if (k === 'thumb') grid.style.setProperty('--ms-thumb', `${st.thumb}px`); redraw(); }, 60));
    return h('label', label, r, out);
  };
  const capFilter = h('input.form-control.form-control-sm.msr-cap-filter', {
    type: 'search', placeholder: 'Filter captions…', 'aria-label': 'Filter captions',
    oninput: debounce((e) => { filter = e.target.value.trim().toLowerCase(); redraw(); }, 150),
  });

  function lightbox(list, start) {
    let i = start;
    let zoom = 1;
    let panX = 0;
    let panY = 0;
    const canvas = h('canvas.msr-lb-canvas');
    const stage = h('div.msr-lb-stage', canvas);
    const cap = h('div.msr-lb-cap');
    const meta = h('div.msr-lb-meta');
    const zoomLbl = h('span.ms-metric', '100%');
    const size = () => ({ w: stage.clientWidth || 800, h: stage.clientHeight || 520 });
    const draw = async () => {
      const item = list[i];
      mount(cap, h('b', `${i + 1} / ${list.length}`), ' ', item.caption || h('span.text-muted', 'no caption'));
      zoomLbl.textContent = `${Math.round(zoom * 100)}%`;
      try {
        const r = await composite(canvas, item, files, { ...st, zoom, panX, panY }, size());
        mount(meta, `${r.W}×${r.H}px`, r.maskClasses.size ? ` · mask classes: ${[...r.maskClasses].join(', ')}` : '',
          st.boxes && (r.boxes || r.hiddenBoxes) ? ` · ${r.boxes} box${r.boxes === 1 ? '' : 'es'}${r.hiddenBoxes ? ` (${r.hiddenBoxes} below threshold)` : ''}` : '');
      } catch (err) { mount(meta, h('span.text-danger', err.message)); }
    };
    const go = (d) => { i = (i + d + list.length) % list.length; zoom = 1; panX = 0; panY = 0; draw(); };
    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const nz = Math.min(20, Math.max(0.5, zoom * f));
      const rect = stage.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      panX = cx - (cx - panX) * (nz / zoom);
      panY = cy - (cy - panY) * (nz / zoom);
      zoom = nz;
      draw();
    }, { passive: false });
    let drag = null;
    stage.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, px: panX, py: panY }; stage.setPointerCapture(e.pointerId); stage.classList.add('dragging'); });
    stage.addEventListener('pointermove', (e) => { if (!drag) return; panX = drag.px + e.clientX - drag.x; panY = drag.py + e.clientY - drag.y; draw(); });
    const end = () => { drag = null; stage.classList.remove('dragging'); };
    stage.addEventListener('pointerup', end);
    stage.addEventListener('pointercancel', end);
    stage.addEventListener('dblclick', () => { zoom = zoom > 1 ? 1 : 3; panX = 0; panY = 0; draw(); });
    const keys = (e) => {
      if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
      if (e.key === '+' || e.key === '=') { zoom = Math.min(20, zoom * 1.25); draw(); }
      if (e.key === '-') { zoom = Math.max(0.5, zoom / 1.25); draw(); }
      if (e.key === '0') { zoom = 1; panX = 0; panY = 0; draw(); }
    };
    document.addEventListener('keydown', keys);
    const file = () => files.get(list[i].path);
    const m = modal({
      title: key, subtitle: `step ${rows[stepIdx]._step}`, icon: 'image', size: 'xl',
      body: h('div.msr-lb',
        h('div.msr-lb-bar',
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Previous (←)', onclick: () => go(-1) }, icon('chevron-left')),
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Next (→)', onclick: () => go(1) }, icon('chevron-right')),
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Zoom out (-)', onclick: () => { zoom = Math.max(0.5, zoom / 1.25); draw(); } }, icon('zoom-out')),
          zoomLbl,
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Zoom in (+)', onclick: () => { zoom = Math.min(20, zoom * 1.25); draw(); } }, icon('zoom-in')),
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Reset (0)', onclick: () => { zoom = 1; panX = 0; panY = 0; draw(); } }, icon('aspect-ratio')),
          hasMasks && sw('masks', 'Masks'), hasBoxes && sw('boxes', 'Boxes'),
          h('a.btn.btn-sm.btn-outline-secondary.ms-auto', { href: '#', title: 'Download original', onclick: (e) => { e.preventDefault(); const f = file(); if (f) window.open(fileUrl(f.directUrl || f.url, { download: true }), '_blank'); } }, icon('download'))),
        stage, cap, meta,
        h('div.small.text-muted', 'Scroll to zoom · drag to pan · double-click to toggle 3× · ← → to browse')),
      onClose: () => { document.removeEventListener('keydown', keys); ro.disconnect(); },
    });
    m.el.classList.add('msr-lb-modal');
    // The toggles inside the lightbox share state with the gallery (which redraws itself).
    m.el.addEventListener('change', () => draw());
    const ro = new ResizeObserver(() => draw());
    ro.observe(stage);
  }

  const head = h('div.msr-media-bar',
    h('div.msr-stepper',
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Previous step', onclick: () => stepBy(-1), disabled: rows.length < 2 }, icon('skip-backward-fill')),
      playBtn,
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Next step', onclick: () => stepBy(1), disabled: rows.length < 2 }, icon('skip-forward-fill')),
      slider, stepLabel),
    h('div.msr-media-opts',
      range('thumb', 80, 420, 10, 'Size', (v) => `${v}px`),
      hasMasks && sw('masks', 'Masks'),
      hasMasks && range('maskOpacity', 0.05, 1, 0.05, 'Opacity', (v) => `${Math.round(v * 100)}%`),
      hasMasks && sw('maskEdges', 'Outline'),
      hasBoxes && sw('boxes', 'Boxes'),
      hasBoxes && sw('labels', 'Labels'),
      hasBoxes && range('minScore', 0, 1, 0.05, 'Min score', (v) => v.toFixed(2)),
      capFilter, info));
  redraw();
  return h('section.af-card.msr-media-card',
    h('header.af-card-header', h('h2.af-card-title', icon('images', 'me-2'), h('span.ms-mono', key)), h('span.ms-pill.ms-pill-muted', `${rows.length} step${rows.length === 1 ? '' : 's'}`)),
    h('div.af-card-body', head, grid));
}

/**
 * Histogram-over-time heatmap (local version of core charts.histogramHeatmap with an empty-is-white
 * colour scale: Plotly's "Blues" paints zero counts dark, which hides the distribution).
 */
async function histogramOverTime(el, rows, { height = 260 } = {}) {
  const Plotly = await loadPlotly();
  const plot = h('div.ms-plot', { style: { height: `${height}px` } });
  el.replaceChildren(plot);
  const all = rows.flatMap((r) => r.bins);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const nb = 48;
  const centers = Array.from({ length: nb }, (_, i) => lo + ((hi - lo) * (i + 0.5)) / nb);
  const z = centers.map(() => rows.map(() => 0));
  rows.forEach((r, ci) => r.values.forEach((count, bi) => {
    const mid = (r.bins[bi] + r.bins[bi + 1]) / 2;
    const idx = Math.min(nb - 1, Math.max(0, Math.floor(((mid - lo) / ((hi - lo) || 1)) * nb)));
    z[idx][ci] += count;
  }));
  const zz = z.map((row) => row.map((v) => (v ? v : null))); // empty bins stay transparent
  await Plotly.newPlot(plot, [{
    type: 'heatmap', x: rows.map((r) => r.step), y: centers, z: zz, showscale: false, hoverongaps: false,
    colorscale: [[0, '#dbeafe'], [0.35, '#60a5fa'], [0.7, '#2563eb'], [1, '#1e3a8a']],
    hovertemplate: 'step %{x}<br>value %{y:.3g}<br>count %{z}<extra></extra>',
  }], {
    margin: { l: 44, r: 10, t: 6, b: 34 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#f8fafc',
    font: { family: 'Inter, system-ui, sans-serif', size: 11, color: '#334155' },
    xaxis: { title: { text: 'step' }, gridcolor: '#eef2f7' }, yaxis: { gridcolor: '#eef2f7', zeroline: false },
  }, { responsive: true, displaylogo: false, displayModeBar: false });
  return { el: plot, destroy: () => Plotly.purge(plot) };
}

// ------------------------------------------------------------------ table + histogram sections

function tableSection(ctx, run, key, files, rows) {
  let idx = rows.length - 1;
  const body = h('div');
  const sel = h('select.form-select.form-select-sm.msr-step-select', { 'aria-label': 'Step' },
    rows.map((r, i) => h('option', { value: i, selected: i === idx }, `step ${r._step}`)));
  const load = async () => {
    mount(body, skeleton(4));
    const path = rows[idx][key]?.path;
    const f = path && files.get(path);
    if (!f) { mount(body, h('div.msr-kv-empty', 'The table file for this step is not available.')); return; }
    try { mount(body, tableFromJson(await loadJson(f.directUrl || f.url))); } catch (err) { mount(body, errorState(err)); }
  };
  sel.addEventListener('change', () => { idx = Number(sel.value); load(); });
  load();
  return h('section.af-card.msr-media-card',
    h('header.af-card-header', h('h2.af-card-title', icon('table', 'me-2'), h('span.ms-mono', key)), h('div.af-card-actions', rows.length > 1 && sel, h('span.ms-pill.ms-pill-muted', `${rows.length} version${rows.length === 1 ? '' : 's'}`))),
    h('div.af-card-body', body));
}

function histogramSection(ctx, key, rows) {
  const body = h('div', h('div.ms-panel-loading'));
  const data = rows.map((r) => ({ step: r._step, bins: r[key].bins || [], values: r[key].values || [] })).filter((r) => r.bins.length > 1 && r.values.length);
  requestAnimationFrame(async () => {
    if (!data.length) { mount(body, h('div.msr-kv-empty', 'No histogram data.')); return; }
    const c = await histogramOverTime(body, data, { height: 260 });
    ctx.cleanup(() => c.destroy());
  });
  return h('section.af-card.msr-media-card',
    h('header.af-card-header', h('h2.af-card-title', icon('bar-chart-steps', 'me-2'), h('span.ms-mono', key)), h('span.ms-pill.ms-pill-muted', `${data.length} snapshots`)),
    h('div.af-card-body', h('div.small.text-muted.mb-1', 'Distribution over time — darker cells hold more values.'), body));
}

// ------------------------------------------------------------------ tab

export async function mediaTab(el, ctx, run) {
  const keys = mediaKeys(run);
  if (!keys.length) {
    mount(el, emptyState({ icon: 'images', title: 'No media logged', message: 'Log wandb.Image, wandb.Table or wandb.Histogram values and they appear here.' }));
    return;
  }
  const { entity, project } = ctx.params;
  mount(el, skeleton(6));
  let files;
  try { files = await fileIndex(entity, project, run.name); } catch (err) { if (err.name !== 'AbortError') mount(el, errorState(err)); return; }
  const jump = h('div.msr-media-jump', keys.map((k) => h('a.ms-tag', { href: `#msr-media-${k.key.replace(/\W+/g, '-')}`, onclick: (e) => { e.preventDefault(); document.getElementById(`msr-media-${k.key.replace(/\W+/g, '-')}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); } },
    icon(k.kind === 'images' ? 'images' : k.kind === 'table' ? 'table' : 'bar-chart-steps'), k.key)));
  const sections = h('div.msr-media-sections');
  mount(el, h('div.msr-media', jump, sections));
  for (const k of keys) {
    const holder = h('div', { id: `msr-media-${k.key.replace(/\W+/g, '-')}` }, h('div.ms-panel-loading'));
    sections.appendChild(holder);
    try {
      const rows = await sampleKey(ctx, run, k.key, k.kind === 'histogram' ? 300 : 1000);
      if (!ctx.active) return;
      if (!rows.length) { mount(holder, h('div.msr-kv-empty', `${k.key}: no values.`)); continue; }
      holder.replaceChildren(k.kind === 'images' ? imageSection(ctx, run, k.key, files, rows)
        : k.kind === 'table' ? tableSection(ctx, run, k.key, files, rows) : histogramSection(ctx, k.key, rows));
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(holder, errorState(err));
      toast(`Could not load ${k.key}`, 'warning');
    }
  }
}
