// One dashboard widget: card chrome, data loading (cached result, parameterized runs with
// mappings, refresh with live timer + cancel), widget-level parameters and filters,
// rendering, and its action menu.

import { h, icon, mount, debounce } from '../../../../ui/js/dom.js';
import { timeAgo, openMenu, toast, toastError, modal, confirmDialog, spinner } from '../../../../ui/js/ui.js';
import { fmtDuration } from '../../../../ui/js/format.js';
import { rd, enc, runQuery, cancelJob, fileUrl } from '../../core/api.js';
import { urls } from '../../core/urls.js';
import { renderViz, vizTypeInfo } from '../../core/viz.js';
import { paramBar, serializeParams, missingParams } from '../../core/params.js';
import { loadMarkdown } from '../../core/lib.js';
import { invalidate } from '../../core/common.js';

/** Column names Redash treats as filters: "name::filter" / "name::multi-filter" (or __filter). */
export function filterColumns(data) {
  return (data?.columns || []).filter((c) => /(::|__)(multi-)?filter$/i.test(c.name)).map((c) => ({
    name: c.name,
    label: c.friendly_name?.replace(/(::|__)(multi-)?filter$/i, '') || c.name.replace(/(::|__)(multi-)?filter$/i, ''),
    multi: /(::|__)multi-filter$/i.test(c.name),
    values: [...new Set((data.rows || []).map((r) => r[c.name]))].filter((v) => v != null).sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true })),
  }));
}

export function applyFilters(data, filters = {}) {
  const active = Object.entries(filters).filter(([, v]) => v != null && v !== '' && !(Array.isArray(v) && !v.length));
  if (!data || !active.length) return data;
  const keyOf = (name) => name.replace(/(::|__)(multi-)?filter$/i, '');
  const relevant = active.filter(([k]) => data.columns.some((c) => keyOf(c.name) === k));
  if (!relevant.length) return data;
  const rows = data.rows.filter((r) => relevant.every(([k, v]) => {
    const col = data.columns.find((c) => keyOf(c.name) === k).name;
    return Array.isArray(v) ? v.map(String).includes(String(r[col])) : String(r[col]) === String(v);
  }));
  return { ...data, rows };
}

/** Values a widget should run with, given its mappings and the dashboard-level values. */
export function widgetParamValues(widget, dashValues, localValues) {
  const params = widget.visualization?.query?.options?.parameters || [];
  const mappings = widget.options?.parameterMappings || {};
  const out = {};
  for (const p of params) {
    const m = mappings[p.name] || { type: 'widget-level' };
    if (m.type === 'dashboard-level') out[p.name] = dashValues[m.mapTo || p.name] ?? p.value;
    else if (m.type === 'static-value') out[p.name] = m.value ?? p.value;
    else out[p.name] = localValues[p.name] ?? p.value;
  }
  return out;
}

/**
 * widgetCard(widget, env)
 * env: { dashboard, dashValues, filterValues, filtersEnabled, settings, isEditing(), onRemove(w), onChanged(w), onFilters() }
 * Returns { el, load(opts), refresh(), rerender(), setEditing(on), filterInfo(), destroy(), headings }
 */
export function widgetCard(widget, env) {
  const viz = widget.visualization;
  const query = viz?.query;
  const localValues = {};
  const localFilters = {};
  let result = null;
  let handle = null;
  let jobId = null;
  let controller = null;
  let state = 'idle';
  let startedAt = 0;
  let timer = 0;

  const titleEl = h('div.rdd-w-title');
  const metaEl = h('div.rdd-w-meta');
  const actionsEl = h('div.rdd-w-actions');
  const paramsEl = h('div.rdd-w-params');
  const filtersEl = h('div.rdd-w-filters');
  const bodyEl = h('div.rdd-w-body');
  const statusEl = h('div.rdd-w-status');
  const el = h('section.rdd-w', { class: [!viz && 'rdd-w-text', viz && `rdd-w-${String(viz.type).toLowerCase()}`], dataset: { widget: widget.id } });

  // ---------------------------------------------------------------- text widgets

  if (!viz) {
    const md = h('div.rdd-text.af-markdown');
    mount(el,
      h('div.rdd-w-texthead.rdd-drag', { tabindex: 0, title: 'Drag to move (edit mode)' }, icon('grip-horizontal'), h('span', 'Text box'), actionsEl),
      md);
    loadMarkdown().then((render) => {
      md.innerHTML = render(widget.text || '');
      md.querySelectorAll('h1, h2, h3').forEach((hd, i) => { hd.id = `rdd-sec-${widget.id}-${i}`; });
      md.querySelectorAll('a[href^="http"]').forEach((a) => { a.target = '_blank'; a.rel = 'noopener'; });
    }).catch(() => { md.textContent = widget.text || ''; });
    drawActions();
    return {
      el,
      load: async () => {},
      refresh: async () => {},
      rerender: () => {},
      setEditing: () => drawActions(),
      filterInfo: () => [],
      destroy: () => {},
      headings: () => [...md.querySelectorAll('h1, h2, h3')].map((hd) => ({ id: hd.id, text: hd.textContent, level: Number(hd.tagName[1]) })),
    };
  }

  // ---------------------------------------------------------------- viz widgets

  const params = query.options?.parameters || [];
  const mappings = () => widget.options?.parameterMappings || {};
  const widgetLevel = () => params.filter((p) => (mappings()[p.name]?.type || 'widget-level') === 'widget-level');

  mount(el,
    h('header.rdd-w-head.rdd-drag', { tabindex: 0 },
      h('div.rdd-w-heading', titleEl, metaEl),
      actionsEl),
    paramsEl,
    filtersEl,
    h('div.rdd-w-bodywrap', bodyEl, statusEl));

  function drawTitle() {
    const info = vizTypeInfo(viz.type);
    mount(titleEl,
      h('span.rdd-w-icon', { title: info.label }, icon(info.icon)),
      h('span.rdd-w-name', { title: viz.name }, viz.name),
      h('a.rdd-w-query', { href: urls.query(query.id), title: `Open query #${query.id}`, onclick: (e) => e.stopPropagation() }, query.name));
  }

  function drawMeta() {
    if (state === 'running') {
      mount(metaEl, h('span.rdd-w-running', spinner('sm'), h('span.rdd-w-timer', fmtDuration((Date.now() - startedAt) / 1000))));
      return;
    }
    if (!result) { mount(metaEl, h('span.text-muted', state === 'error' ? 'failed' : 'not run yet')); return; }
    mount(metaEl,
      h('span', { title: `Data retrieved ${result.retrieved_at}` }, icon('clock-history'), ' ', timeAgo(result.retrieved_at)),
      result.runtime != null && h('span', icon('stopwatch'), ` ${fmtDuration(result.runtime)}`),
      h('span', icon('list-ol'), ` ${(result.data?.rows || []).length.toLocaleString()} rows`));
  }

  function drawActions() {
    const editing = env.isEditing();
    mount(actionsEl,
      viz && h('button.af-icon-btn.rdd-w-btn', {
        type: 'button', title: state === 'running' ? 'Cancel run' : 'Refresh (re-run the query)',
        onclick: (e) => { e.stopPropagation(); state === 'running' ? cancel() : refresh(); },
      }, icon(state === 'running' ? 'stop-circle' : 'arrow-clockwise')),
      viz && h('button.af-icon-btn.rdd-w-btn', { type: 'button', title: 'Expand', onclick: (e) => { e.stopPropagation(); expand(); } }, icon('arrows-angle-expand')),
      h('button.af-icon-btn.rdd-w-btn', { type: 'button', title: 'Widget actions', onclick: (e) => { e.stopPropagation(); openMenu(e.currentTarget, menuItems(editing)); } }, icon('three-dots-vertical')));
  }

  function menuItems(editing) {
    if (!viz) {
      return [
        { label: 'Edit text', icon: 'pencil', onClick: () => env.onEditText?.(widget) },
        { divider: true },
        { label: 'Remove from dashboard', icon: 'trash', danger: true, onClick: () => remove() },
      ];
    }
    const rid = result?.id;
    return [
      { label: 'Refresh', icon: 'arrow-clockwise', onClick: () => refresh() },
      { label: 'Open query', icon: 'code-square', href: urls.query(query.id) },
      { label: 'Edit visualization', icon: 'sliders', onClick: () => editViz() },
      { label: 'Rename', icon: 'input-cursor-text', onClick: () => rename() },
      params.length && { label: 'Parameters…', icon: 'ui-radios', onClick: () => env.onMappings?.(widget) },
      { divider: true },
      { header: 'Download' },
      { label: 'CSV', icon: 'filetype-csv', disabled: !rid, href: rid ? fileUrl(`/api/query_results/${rid}.csv`) : undefined, external: true },
      { label: 'Excel', icon: 'filetype-xlsx', disabled: !rid, href: rid ? fileUrl(`/api/query_results/${rid}.xlsx`) : undefined, external: true },
      { label: 'TSV', icon: 'filetype-txt', disabled: !rid, href: rid ? fileUrl(`/api/query_results/${rid}.tsv`) : undefined, external: true },
      handle?.toImage && { label: 'Chart as PNG', icon: 'image', onClick: () => downloadPng() },
      { divider: true },
      { label: 'Remove from dashboard', icon: 'trash', danger: true, onClick: () => remove(), disabled: false },
    ].filter(Boolean);
  }

  function drawParams() {
    const local = widgetLevel();
    if (!local.length) { mount(paramsEl); return; }
    for (const p of local) if (!(p.name in localValues)) localValues[p.name] = p.value;
    const bar = paramBar(local, localValues, { compact: true, ownerQueryId: query.id, applyLabel: 'Run', onApply: () => { bar.reset(); load({ maxAge: -1 }); } });
    mount(paramsEl, bar.el);
  }

  function drawFilters() {
    const info = filterColumns(result?.data);
    if (!info.length || env.filtersEnabled()) { mount(filtersEl); return; }
    mount(filtersEl, h('div.rdd-filterbar', info.map((f) => {
      const sel = h('select.form-select.form-select-sm', { multiple: f.multi, size: f.multi ? 3 : undefined, 'aria-label': f.label },
        !f.multi && h('option', { value: '' }, `All ${f.label}`),
        f.values.map((v) => h('option', { value: v, selected: f.multi ? (localFilters[f.label] || []).includes(String(v)) : String(localFilters[f.label] ?? '') === String(v) }, String(v))));
      sel.addEventListener('change', () => { localFilters[f.label] = f.multi ? [...sel.selectedOptions].map((o) => o.value) : sel.value; rerender(); });
      return h('label.rdd-filter', h('span', f.label), sel);
    })));
  }

  const filtered = () => {
    const data = result?.data;
    if (!data) return null;
    return { ...result, data: applyFilters(data, env.filtersEnabled() ? env.filterValues : localFilters) };
  };

  let renderedSize = '';
  async function rerender() {
    if (!result) return;
    const height = Math.max(bodyEl.clientHeight - 4, 120);
    renderedSize = `${bodyEl.clientWidth}x${bodyEl.clientHeight}`;
    try { handle?.destroy?.(); } catch { /* ignore */ }
    const target = h('div.rdd-w-viz');
    mount(bodyEl, target);
    // Tables: leave room for the grid's own search toolbar and pager inside the card.
    handle = await renderViz(target, viz, filtered(), { height, settings: env.settings, compact: true, gridMaxHeight: `${Math.max(height - 96, 80)}px` });
  }

  function setRunning(on) {
    state = on ? 'running' : state;
    el.classList.toggle('is-running', on);
    clearInterval(timer);
    if (on) {
      startedAt = Date.now();
      timer = setInterval(drawMeta, 1000);
      mount(statusEl, h('div.rd-progress-bar'));
    } else mount(statusEl);
    drawMeta();
    drawActions();
  }

  /** opts.maxAge: undefined = cached result if any; 0 = force re-run; -1 = any cached for these params. */
  async function load({ maxAge } = {}) {
    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    const values = widgetParamValues(widget, env.dashValues, localValues);
    const missing = missingParams(params, values);
    if (missing.length) {
      state = 'idle';
      mount(bodyEl, h('div.rdd-w-empty', icon('ui-radios'), h('div.fw-semibold', 'Waiting for parameters'), h('div.small', `Set ${missing.join(', ')} to run this widget.`)));
      drawMeta();
      return;
    }
    try {
      if (!params.length && maxAge === undefined && query.latest_query_data_id) {
        if (!result) mount(bodyEl, h('div.rdd-w-loading', spinner()));
        result = (await rd.get(`/api/query_results/${enc(query.latest_query_data_id)}`, null, { signal })).query_result;
      } else {
        // No stored result (or parameters / forced refresh): like Redash, ask for any cached
        // result for this text + parameters (-1), which only executes when nothing is cached.
        setRunning(true);
        result = await runQuery({
          queryId: query.id, parameters: serializeParams(params, values), maxAge: maxAge ?? -1, signal,
          onJob: (job) => { jobId = job?.id; },
        });
        jobId = null;
        if (!params.length) query.latest_query_data_id = result.id;
      }
      state = 'ready';
      setRunning(false);
      drawFilters();
      env.onFilters?.();
      await rerender();
      drawMeta();
      el.classList.remove('flash');
      void el.offsetWidth;
      el.classList.add('flash');
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (result) {
        // Keep showing the last good data; flag the failed refresh above it.
        state = 'ready';
        setRunning(false);
        mount(statusEl, h('div.rdd-w-stale', icon('exclamation-triangle-fill'),
          h('span', { title: err.message }, `Refresh failed — showing data from ${new Date(result.retrieved_at).toLocaleString()}. ${err.message}`),
          h('button.btn.btn-sm.btn-link', { type: 'button', onclick: () => refresh() }, 'Retry'),
          h('button.af-icon-btn', { type: 'button', title: 'Dismiss', onclick: () => mount(statusEl) }, icon('x'))));
        drawMeta();
        return;
      }
      state = 'error';
      setRunning(false);
      mount(bodyEl, h('div.rdd-w-error', icon('exclamation-octagon'), h('div', h('div.fw-semibold', 'Query failed'), h('div.small', err.message)),
        h('button.btn.btn-sm.btn-outline-danger', { type: 'button', onclick: () => refresh() }, icon('arrow-clockwise', 'me-1'), 'Retry')));
      drawMeta();
    }
  }

  const refresh = () => load({ maxAge: 0 });

  async function cancel() {
    const id = jobId;
    controller?.abort();
    if (id) { try { await cancelJob(id); } catch { /* job may be finished */ } }
    state = result ? 'ready' : 'idle';
    setRunning(false);
    toast('Run cancelled', 'info', { timeout: 2000 });
    if (result) rerender(); else mount(bodyEl, h('div.rdd-w-empty', icon('stop-circle'), h('div', 'Cancelled')));
  }

  async function editViz() {
    try {
      const { openVizEditor } = await import('../../core/viz_editor.js');
      const fullQuery = await rd.get(`/api/queries/${enc(query.id)}`);
      const saved = await openVizEditor({ query: fullQuery, result, viz });
      if (saved) {
        Object.assign(viz, { name: saved.name, type: saved.type, options: saved.options, description: saved.description });
        invalidate('dashfull');
        drawTitle();
        rerender();
        env.onChanged?.(widget);
      }
    } catch (err) { toastError(err, 'Could not open the editor'); }
  }

  async function rename() {
    const input = h('input.form-control', { value: viz.name });
    modal({
      title: 'Rename widget', icon: 'input-cursor-text', size: 'sm',
      body: h('div', input, h('div.rdv-help.mt-2', 'Renames the visualization, so the new title also shows on the query page and other dashboards using it.')),
      actions: [{ label: 'Cancel' }, {
        label: 'Rename', variant: 'primary', onClick: async (close) => {
          const name = input.value.trim();
          if (!name) throw new Error('Name cannot be empty.');
          const saved = await rd.post(`/api/visualizations/${enc(viz.id)}`, { name });
          viz.name = saved.name;
          drawTitle();
          toast('Widget renamed', 'success', { timeout: 2000 });
          close();
        },
      }],
    });
    setTimeout(() => { input.focus(); input.select(); }, 50);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') input.closest('.af-modal').querySelector('.btn-primary').click(); });
  }

  async function remove() {
    const ok = await confirmDialog({ title: 'Remove widget?', message: `“${viz ? viz.name : 'Text box'}” is removed from this dashboard. The query and visualization are kept.`, confirmLabel: 'Remove', danger: true });
    if (!ok) return;
    try {
      await rd.del(`/api/widgets/${enc(widget.id)}`);
      invalidate('dashfull');
      env.onRemove?.(widget);
      toast('Widget removed', 'success', { timeout: 2000 });
    } catch (err) { toastError(err, 'Remove failed'); }
  }

  async function expand() {
    const target = h('div.rdd-expand');
    const m = modal({ title: viz.name, subtitle: query.name, icon: vizTypeInfo(viz.type).icon, size: 'xl', body: target });
    m.el.classList.add('rdd-expand-modal');
    const hdl = await renderViz(target, viz, filtered(), { height: Math.round(window.innerHeight * 0.62), settings: env.settings, gridMaxHeight: '62vh' });
    const origClose = m.close;
    m.close = () => { hdl?.destroy?.(); origClose(); };
  }

  async function downloadPng() {
    try {
      const url = await handle.toImage();
      const a = h('a', { href: url, download: `${viz.name.replace(/[^\w-]+/g, '_')}.png` });
      a.click();
    } catch (err) { toastError(err, 'Export failed'); }
  }

  const onResize = debounce(() => {
    if (state !== 'ready') return;
    // Re-render only when the box really changed (resize, reflow), never in a loop.
    if (`${bodyEl.clientWidth}x${bodyEl.clientHeight}` === renderedSize) return;
    if (viz.type === 'CHART' && handle?.resize && bodyEl.clientHeight === Number(renderedSize.split('x')[1])) { handle.resize(); renderedSize = `${bodyEl.clientWidth}x${bodyEl.clientHeight}`; return; }
    rerender();
  }, 200);
  el.addEventListener('rdd:resized', onResize);

  drawTitle();
  drawParams();
  drawMeta();
  drawActions();

  return {
    el,
    load,
    refresh,
    rerender,
    setEditing: () => drawActions(),
    filterInfo: () => filterColumns(result?.data),
    destroy: () => { controller?.abort(); clearInterval(timer); try { handle?.destroy?.(); } catch { /* ignore */ } },
    headings: () => [],
    get state() { return state; },
    get queryId() { return query.id; },
    get usesDashboardParams() { return params.some((p) => mappings()[p.name]?.type === 'dashboard-level'); },
    redrawParams: () => drawParams(),
  };
}
