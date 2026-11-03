// Dashboard view & editor.
//
// View: 6-column Redash grid, sticky parameter/filter bar, section jump links, per-widget
// "updated X ago" + refresh, refresh-all with progress, auto-refresh, TV mode, PDF export.
// Edit (?edit=1): drag/resize with snapping + push-down, add visualization or text widgets,
// undo / cancel / save layout. Header: inline rename, favourite, tags, publish, share,
// duplicate, filters toggle, archive.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import {
  pageHeader, toast, toastError, emptyState, errorState, skeleton, openMenu, confirmDialog, modal, spinner, timeAgo,
} from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc, redashLink } from '../core/api.js';
import { urls } from '../core/urls.js';
import { favoriteStar, inlineEdit, tagEditor, getDashboardTags, getOrgSettings, getSession, invalidate } from '../core/common.js';
import { paramBar } from '../core/params.js';
import { vizTypeInfo } from '../core/viz.js';
import { createVizWidget, defaultMappings } from '../core/add_to_dashboard.js';
import { gridView, initialLayout } from './dashboard/grid.js';
import { widgetCard } from './dashboard/widget.js';
import {
  duplicateDashboard, shareDialog, editTextDialog, addTextWidget, saveWidget, parameterMappingDialog,
} from './dashboard/actions.js';

const DEFAULT_INTERVALS = [60, 300, 600, 1800, 3600, 43200, 86400];
const intervalLabel = (s) => (s < 3600 ? `${s / 60} min` : s < 86400 ? `${s / 3600} hour${s === 3600 ? '' : 's'}` : `${s / 86400} day`);

async function fetchDashboard(idOrSlug, signal) {
  if (/^\d+$/.test(String(idOrSlug))) return rd.get(`/api/dashboards/${enc(idOrSlug)}`, null, { signal });
  // Redash still resolves slugs with ?legacy; also accept "12-my-slug" URLs copied from Redash.
  const m = String(idOrSlug).match(/^(\d+)-/);
  if (m) return rd.get(`/api/dashboards/${enc(m[1])}`, null, { signal });
  return rd.get(`/api/dashboards/${enc(idOrSlug)}`, { legacy: 1 }, { signal });
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  let dashboard;
  try {
    dashboard = await fetchDashboard(ctx.params.dashboardId, signal);
  } catch (err) {
    if (err.name === 'AbortError') return;
    mount(el, pageHeader({ title: 'Dashboard', icon: 'grid-1x2', crumbs: [{ label: 'Dashboards', href: urls.dashboards() }] }),
      errorState(err, () => navigate(location.pathname + location.search)));
    return;
  }
  if (!ctx.active) return;
  if (String(dashboard.id) !== String(ctx.params.dashboardId)) history.replaceState(null, '', urls.dashboard(dashboard.id, Object.fromEntries(ctx.query)));
  ctx.setTitle(dashboard.name);

  const [settings, session] = await Promise.all([getOrgSettings().catch(() => ({})), getSession().catch(() => null)]);
  const intervals = session?.client_config?.dashboardRefreshIntervals || DEFAULT_INTERVALS;

  // ---------------------------------------------------------------- state

  let editing = ctx.query.get('edit') === '1';
  let autoRefresh = Number(ctx.query.get('refresh')) || 0;
  let refreshTimer = 0;
  let savedLayout = null;
  let undoStack = [];
  let dirty = false;
  const dashValues = {};
  const filterValues = {};
  const cards = new Map();
  let grid = null;

  for (const [k, v] of ctx.query) if (k.startsWith('p_')) dashValues[k.slice(2)] = parseParamValue(v);

  const env = {
    dashboard,
    dashValues,
    filterValues,
    settings,
    filtersEnabled: () => !!dashboard.dashboard_filters_enabled,
    isEditing: () => editing,
    onRemove: (w) => { dashboard.widgets = dashboard.widgets.filter((x) => x.id !== w.id); cards.get(w.id)?.destroy(); cards.delete(w.id); grid.remove(w.id); afterWidgetsChanged(); },
    onChanged: () => buildSections(),
    onEditText: (w) => editText(w),
    onMappings: (w) => editMappings(w),
    onFilters: debounce(() => drawFilterBar(), 100),
  };

  // ---------------------------------------------------------------- header

  const headerEl = h('div');
  const toolbarEl = h('div.rdd-toolbar-sticky');
  const sectionsEl = h('nav.rdd-sections', { 'aria-label': 'Dashboard sections' });
  const dashParamsEl = h('div.rdd-dash-params');
  const filterBarEl = h('div.rdd-dash-filters');
  const gridHost = h('div.rdd-grid-host');
  const editBar = h('div.rdd-editbar');
  const progressEl = h('div.rdd-progress');

  function drawHeader() {
    const drafted = dashboard.is_draft;
    mount(headerEl, pageHeader({
      crumbs: [{ label: 'Dashboards', href: urls.dashboards() }, { label: dashboard.name }],
      title: inlineEdit(dashboard.name, (name) => update({ name }), { className: 'rdd-title-edit', disabled: !dashboard.can_edit }),
      icon: 'grid-1x2',
      badges: [
        favoriteStar('dashboards', dashboard),
        drafted && h('span.rd-draft-pill', { title: 'Drafts are only visible to you and admins in lists' }, 'draft'),
        dashboard.public_url && h('span.rd-type-pill', { title: 'Shared with a public link' }, icon('globe2'), 'public'),
      ],
      subtitle: h('div.rdd-subtitle',
        tagEditor(dashboard.tags || [], (tags) => update({ tags }), { suggestions: [], readOnly: !dashboard.can_edit }),
        h('span.text-muted.small', icon('person', 'me-1'), dashboard.user?.name || '—'),
        h('span.text-muted.small', icon('pencil', 'me-1'), 'updated ', timeAgo(dashboard.updated_at))),
      actions: actionButtons(),
    }));
    getDashboardTags().then((tags) => {
      const editor = headerEl.querySelector('.rd-tags');
      if (editor && dashboard.can_edit) editor.replaceWith(tagEditor(dashboard.tags || [], (t) => update({ tags: t }), { suggestions: tags.map((t) => t.name) }));
    }).catch(() => {});
  }

  function actionButtons() {
    if (editing) {
      return [h('span.rdd-editing-pill', icon('pencil-square', 'me-1'), 'Editing layout')];
    }
    const autoBtn = h('button.btn.btn-sm.btn-outline-secondary', {
      type: 'button', title: 'Auto refresh', class: autoRefresh && 'active',
      onclick: (e) => openMenu(e.currentTarget, [
        { header: 'Refresh automatically' },
        { label: 'Off', icon: autoRefresh ? 'blank' : 'check2', onClick: () => setAutoRefresh(0) },
        ...intervals.map((s) => ({ label: `Every ${intervalLabel(s)}`, icon: s === autoRefresh ? 'check2' : 'blank', onClick: () => setAutoRefresh(s) })),
      ]),
    }, icon('stopwatch'), autoRefresh ? ` ${intervalLabel(autoRefresh)}` : '');
    return [
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Re-run every query on this dashboard', onclick: () => refreshAll() }, icon('arrow-clockwise', 'me-1'), 'Refresh'),
      h('div.btn-group', autoBtn),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'TV mode (full screen, Esc to exit)', onclick: () => tvMode(true) }, icon('fullscreen')),
      dashboard.can_edit && h('button.btn.btn-sm.btn-primary', { type: 'button', onclick: () => setEditing(true) }, icon('pencil-square', 'me-1'), 'Edit'),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'More', onclick: (e) => openMenu(e.currentTarget, moreItems()) }, icon('three-dots-vertical')),
    ];
  }

  function moreItems() {
    const link = redashLink(`/dashboards/${dashboard.id}-${dashboard.slug}`);
    return [
      dashboard.can_edit && { label: dashboard.is_draft ? 'Publish' : 'Unpublish (back to draft)', icon: dashboard.is_draft ? 'send-check' : 'file-earmark', onClick: () => update({ is_draft: !dashboard.is_draft }, dashboard.is_draft ? 'Dashboard published' : 'Moved back to drafts') },
      { label: 'Share…', icon: 'share', onClick: () => shareDialog(dashboard, { onChange: () => drawHeader() }) },
      { label: 'Duplicate', icon: 'copy', onClick: async () => { const copy = await duplicateDashboard(dashboard); if (copy) navigate(urls.dashboard(copy.id)); } },
      { label: 'Export to PDF / print', icon: 'file-earmark-pdf', onClick: () => exportPdf() },
      dashboard.can_edit && { label: dashboard.dashboard_filters_enabled ? 'Use per-widget filters' : 'Use dashboard-wide filters', icon: 'funnel', onClick: () => update({ dashboard_filters_enabled: !dashboard.dashboard_filters_enabled }, 'Filter mode updated').then(() => { drawFilterBar(); cards.forEach((c) => c.rerender()); }) },
      link && { label: 'Open in Redash', icon: 'box-arrow-up-right', href: link, external: true },
      { divider: true },
      dashboard.can_edit && { label: 'Archive', icon: 'archive', danger: true, onClick: () => archive() },
    ].filter(Boolean);
  }

  async function update(changes, message) {
    try {
      const saved = await rd.post(`/api/dashboards/${enc(dashboard.id)}`, { ...changes, version: dashboard.version });
      Object.assign(dashboard, { ...saved, widgets: dashboard.widgets, public_url: dashboard.public_url, api_key: dashboard.api_key, can_edit: dashboard.can_edit });
      invalidate('dash');
      if (changes.name) ctx.setTitle(dashboard.name);
      if (message) toast(message, 'success', { timeout: 2500 });
      drawHeader();
      return saved;
    } catch (err) {
      if (err.status === 409) {
        toast('Someone else changed this dashboard. Reload to get the latest version.', 'warning', { title: 'Conflict', timeout: 0, action: { label: 'Reload', onClick: () => navigate(location.pathname + location.search, { replace: true }) } });
      }
      throw err;
    }
  }

  async function archive() {
    const ok = await confirmDialog({ title: `Archive “${dashboard.name}”?`, message: 'It disappears from lists and search. Widgets and queries are kept.', confirmLabel: 'Archive', danger: true });
    if (!ok) return;
    try {
      await rd.del(`/api/dashboards/${enc(dashboard.id)}`);
      invalidate('dash');
      const id = dashboard.id;
      toast(`“${dashboard.name}” archived`, 'success', {
        timeout: 8000,
        action: { label: 'Undo', onClick: async () => { await rd.post(`/api/dashboards/${enc(id)}`, { is_archived: false }); invalidate('dash'); navigate(urls.dashboard(id)); } },
      });
      navigate(urls.dashboards());
    } catch (err) { toastError(err, 'Archive failed'); }
  }

  // ---------------------------------------------------------------- parameters & filters

  function parseParamValue(v) {
    if (/^[{[]/.test(v)) { try { return JSON.parse(v); } catch { /* text */ } }
    return v;
  }

  /** Dashboard-level parameter definitions collected from widget mappings. */
  function dashboardParams() {
    const out = new Map();
    for (const w of dashboard.widgets || []) {
      const q = w.visualization?.query;
      for (const p of q?.options?.parameters || []) {
        const m = w.options?.parameterMappings?.[p.name];
        if (m?.type !== 'dashboard-level') continue;
        const name = m.mapTo || p.name;
        if (!out.has(name)) out.set(name, { ...p, name, title: m.title || p.title || name });
      }
    }
    return [...out.values()];
  }

  function drawParams() {
    const params = dashboardParams();
    if (!params.length) { mount(dashParamsEl); return; }
    for (const p of params) if (!(p.name in dashValues)) dashValues[p.name] = p.value;
    const bar = paramBar(params, dashValues, {
      applyLabel: 'Apply to dashboard',
      onApply: () => {
        bar.reset();
        const qs = {};
        for (const p of params) { const v = dashValues[p.name]; qs[`p_${p.name}`] = v == null ? null : typeof v === 'object' ? JSON.stringify(v) : v; }
        ctx.setQuery(qs);
        const affected = [...cards.values()].filter((c) => c.usesDashboardParams);
        runAll(affected, { maxAge: -1 }, 'Applying parameters');
      },
    });
    mount(dashParamsEl, bar.el);
  }

  function drawFilterBar() {
    if (!dashboard.dashboard_filters_enabled) { mount(filterBarEl); return; }
    const merged = new Map();
    for (const c of cards.values()) {
      for (const f of c.filterInfo()) {
        const cur = merged.get(f.label) || { ...f, values: [] };
        cur.values = [...new Set([...cur.values, ...f.values])];
        cur.multi = cur.multi || f.multi;
        merged.set(f.label, cur);
      }
    }
    if (!merged.size) { mount(filterBarEl); return; }
    mount(filterBarEl, h('div.rdd-filterbar.rdd-filterbar-dash', h('span.rdd-filterbar-label', icon('funnel', 'me-1'), 'Filters'),
      [...merged.values()].map((f) => {
        const sel = h('select.form-select.form-select-sm', { multiple: f.multi, size: f.multi ? 3 : undefined, 'aria-label': f.label },
          !f.multi && h('option', { value: '' }, `All ${f.label}`),
          f.values.map((v) => h('option', { value: v, selected: f.multi ? (filterValues[f.label] || []).includes(String(v)) : String(filterValues[f.label] ?? '') === String(v) }, String(v))));
        sel.addEventListener('change', () => {
          filterValues[f.label] = f.multi ? [...sel.selectedOptions].map((o) => o.value) : sel.value;
          cards.forEach((c) => c.rerender());
        });
        return h('label.rdd-filter', h('span', f.label), sel);
      }),
      Object.values(filterValues).some((v) => v && v.length) && h('button.btn.btn-sm.btn-link', { type: 'button', onclick: () => { Object.keys(filterValues).forEach((k) => delete filterValues[k]); drawFilterBar(); cards.forEach((c) => c.rerender()); } }, 'Clear')));
  }

  // ---------------------------------------------------------------- sections nav

  function buildSections() {
    const heads = [...cards.values()].flatMap((c) => c.headings());
    if (!heads.length) { mount(sectionsEl); sectionsEl.hidden = true; return; }
    sectionsEl.hidden = false;
    mount(sectionsEl, h('span.rdd-sections-label', icon('signpost-split', 'me-1'), 'Jump to'),
      heads.filter((x) => x.level <= 2).map((x) => h('button.rdd-section-link', {
        type: 'button', class: `lvl-${x.level}`,
        onclick: () => document.getElementById(x.id)?.closest('.rdd-cell')?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      }, x.text)));
  }

  // ---------------------------------------------------------------- refresh

  async function runAll(list, opts, label) {
    const items = list.filter((c) => c.queryId);
    if (!items.length) return;
    let done = 0;
    const draw = () => mount(progressEl, done < items.length
      ? h('div.rdd-progress-inner', spinner('sm'), h('span', `${label} — ${done} of ${items.length}`), h('div.af-meter', h('span', { style: { width: `${(done / items.length) * 100}%`, background: 'var(--af-primary)' } })))
      : null);
    draw();
    const queue = [...items];
    const worker = async () => {
      while (queue.length && ctx.active) {
        const c = queue.shift();
        await c.load(opts).catch(() => {});
        done++;
        draw();
      }
    };
    await Promise.all([worker(), worker(), worker()]);
    if (ctx.active) {
      mount(progressEl);
      drawFilterBar();
    }
  }

  const refreshAll = () => runAll([...cards.values()], { maxAge: 0 }, 'Refreshing');

  function setAutoRefresh(s) {
    autoRefresh = s;
    ctx.setQuery({ refresh: s || null });
    clearInterval(refreshTimer);
    if (s) {
      refreshTimer = setInterval(() => { if (!document.hidden && !editing) refreshAll(); }, s * 1000);
      toast(`Refreshing every ${intervalLabel(s)}`, 'info', { timeout: 2000 });
    }
    drawHeader();
  }
  ctx.cleanup(() => clearInterval(refreshTimer));

  // ---------------------------------------------------------------- TV mode & print

  function tvMode(on) {
    document.body.classList.toggle('rdd-tv', on);
    if (on) {
      el.requestFullscreen?.().catch(() => {});
      toast('TV mode — press Esc to exit', 'info', { timeout: 2500 });
    } else if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
    setTimeout(() => grid?.relayout(), 120);
  }
  const onFsChange = () => { if (!document.fullscreenElement && document.body.classList.contains('rdd-tv')) tvMode(false); };
  const onKey = (e) => {
    if (e.key === 'Escape' && document.body.classList.contains('rdd-tv')) tvMode(false);
    if (editing && (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); saveLayout(); }
    if (editing && (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z' && !e.target.closest('input,textarea')) { e.preventDefault(); undo(); }
  };
  document.addEventListener('fullscreenchange', onFsChange);
  document.addEventListener('keydown', onKey);
  ctx.cleanup(() => {
    document.removeEventListener('fullscreenchange', onFsChange);
    document.removeEventListener('keydown', onKey);
    document.body.classList.remove('rdd-tv', 'rdd-printing');
  });

  function exportPdf() {
    document.body.classList.add('rdd-printing');
    const done = () => { document.body.classList.remove('rdd-printing'); window.removeEventListener('afterprint', done); setTimeout(() => grid?.relayout(), 50); };
    window.addEventListener('afterprint', done);
    toast('Choose “Save as PDF” in the print dialog', 'info', { timeout: 3000 });
    setTimeout(() => window.print(), 300);
  }

  // ---------------------------------------------------------------- edit mode

  const beforeUnload = (e) => { if (dirty) { e.preventDefault(); e.returnValue = ''; } };
  window.addEventListener('beforeunload', beforeUnload);
  ctx.cleanup(() => window.removeEventListener('beforeunload', beforeUnload));

  function setEditing(on) {
    if (on && !dashboard.can_edit) return;
    editing = on;
    ctx.setQuery({ edit: on ? 1 : null });
    el.classList.toggle('rdd-is-editing', on);
    if (on) {
      savedLayout = grid.getLayout();
      undoStack = [];
      dirty = false;
    }
    grid.setEditing(on);
    cards.forEach((c) => c.setEditing(on));
    drawEditBar();
    drawHeader();
  }

  function drawEditBar() {
    if (!editing) { mount(editBar); editBar.hidden = true; return; }
    editBar.hidden = false;
    mount(editBar,
      h('div.rdd-editbar-hint', icon('arrows-move', 'me-2'), h('span', h('b', 'Drag'), ' a widget header to move it, ', h('b', 'drag the corner'), ' to resize. Arrow keys nudge a focused header (Shift = resize).')),
      h('div.rdd-editbar-actions',
        h('button.btn.btn-sm.btn-outline-primary', { type: 'button', onclick: () => addVizDialog() }, icon('bar-chart-line', 'me-1'), 'Add visualization'),
        h('button.btn.btn-sm.btn-outline-primary', { type: 'button', onclick: () => addText() }, icon('fonts', 'me-1'), 'Add text'),
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', disabled: !undoStack.length, title: 'Undo (Ctrl+Z)', onclick: () => undo() }, icon('arrow-counterclockwise', 'me-1'), 'Undo'),
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => cancelEdit() }, 'Cancel'),
        h('button.btn.btn-sm.btn-primary', { type: 'button', disabled: !dirty, title: 'Save layout (Ctrl+S)', onclick: () => saveLayout() }, icon('check2', 'me-1'), dirty ? 'Save layout' : 'Saved')));
  }

  function onLayoutChange(layout) {
    undoStack.push(lastLayout);
    lastLayout = layout;
    dirty = layoutDiffers(layout, savedLayout);
    drawEditBar();
  }
  let lastLayout = null;

  function layoutDiffers(a, b) {
    if (!a || !b) return true;
    for (const [id, p] of a) {
      const q = b.get(id);
      if (!q || q.col !== p.col || q.row !== p.row || q.sizeX !== p.sizeX || q.sizeY !== p.sizeY) return true;
    }
    return false;
  }

  function undo() {
    if (!undoStack.length) return;
    const prev = undoStack.pop();
    lastLayout = prev;
    grid.setLayout(prev);
    dirty = layoutDiffers(prev, savedLayout);
    drawEditBar();
    cards.forEach((c) => c.el.dispatchEvent(new CustomEvent('rdd:resized', { bubbles: true })));
  }

  async function cancelEdit() {
    if (dirty) {
      const ok = await confirmDialog({ title: 'Discard layout changes?', message: 'Widgets go back to where they were when you started editing.', confirmLabel: 'Discard', danger: true });
      if (!ok) return;
      grid.setLayout(savedLayout);
    }
    dirty = false;
    setEditing(false);
  }

  async function saveLayout() {
    if (!dirty) { setEditing(false); return; }
    const layout = grid.getLayout();
    const changed = (dashboard.widgets || []).filter((w) => {
      const p = layout.get(w.id);
      const o = w.options?.position;
      return p && (!o || o.col !== p.col || o.row !== p.row || o.sizeX !== p.sizeX || o.sizeY !== p.sizeY);
    });
    const btn = editBar.querySelector('.btn-primary');
    if (btn) { btn.disabled = true; mount(btn, spinner('sm'), ` Saving ${changed.length}…`); }
    try {
      for (const w of changed) {
        const p = layout.get(w.id);
        const options = { ...(w.options || {}), position: { ...(w.options?.position || {}), col: p.col, row: p.row, sizeX: p.sizeX, sizeY: p.sizeY, minSizeX: p.minSizeX ?? 1, maxSizeX: p.maxSizeX ?? 6, minSizeY: p.minSizeY ?? 1, maxSizeY: p.maxSizeY ?? 1000, autoHeight: false } };
        const saved = await saveWidget(w, { options });
        w.options = saved.options || options;
      }
      invalidate('dashfull');
      invalidate('dashpreview');
      savedLayout = layout;
      dirty = false;
      toast(changed.length ? `Layout saved (${changed.length} widget${changed.length === 1 ? '' : 's'})` : 'Layout saved', 'success', { timeout: 2500 });
      setEditing(false);
    } catch (err) {
      toastError(err, 'Could not save the layout');
      drawEditBar();
    }
  }

  // ---------------------------------------------------------------- adding widgets

  function mountCard(w, pos) {
    const card = widgetCard(w, env);
    cards.set(w.id, card);
    grid.add({ id: w.id, el: card.el }, pos);
    return card;
  }

  function afterWidgetsChanged() {
    drawParams();
    drawFilterBar();
    buildSections();
    drawEmpty();
    invalidate('dashfull');
    invalidate('dashpreview');
  }

  async function addText() {
    const text = await editTextDialog('');
    if (!text) return;
    try {
      const w = await addTextWidget(dashboard, text);
      dashboard.widgets.push(w);
      mountCard(w, w.options.position);
      setTimeout(buildSections, 300);
      afterWidgetsChanged();
      toast('Text box added', 'success', { timeout: 2000 });
    } catch (err) { toastError(err, 'Could not add the text box'); }
  }

  async function editText(w) {
    const text = await editTextDialog(w.text || '');
    if (text == null) return;
    try {
      const saved = await saveWidget(w, { text });
      w.text = saved.text ?? text;
      const old = cards.get(w.id);
      const card = widgetCard(w, env);
      cards.set(w.id, card);
      old.el.replaceWith(card.el);
      old.destroy();
      setTimeout(buildSections, 300);
      toast('Text updated', 'success', { timeout: 2000 });
    } catch (err) { toastError(err, 'Could not save the text'); }
  }

  async function editMappings(w) {
    const next = await parameterMappingDialog(w, w.visualization.query);
    if (!next) return;
    try {
      const saved = await saveWidget(w, { options: { ...w.options, parameterMappings: next } });
      w.options = saved.options || { ...w.options, parameterMappings: next };
      drawParams();
      const card = cards.get(w.id);
      card.redrawParams();
      card.load({ maxAge: -1 });
      toast('Parameter mapping saved', 'success', { timeout: 2000 });
    } catch (err) { toastError(err, 'Could not save the mapping'); }
  }

  function addVizDialog() {
    const search = h('input.form-control', { type: 'search', placeholder: 'Search queries by name…', autofocus: true, 'aria-label': 'Search queries' });
    const list = h('div.rdd-addviz-queries', h('div.text-center.p-3', spinner()));
    const vizList = h('div.rdd-addviz-vizzes', h('div.rdv-help.p-3', 'Pick a query on the left to see its visualizations.'));
    let chosenQuery = null;
    let seq = 0;
    const loadQueries = async () => {
      const mine = ++seq;
      try {
        const res = await rd.get('/api/queries', { q: search.value.trim() || undefined, page_size: 30, order: '-created_at' });
        if (mine !== seq) return;
        const items = res.results || [];
        mount(list, items.length ? items.map((q) => h('button.rdd-pick', {
          type: 'button', dataset: { id: q.id },
          onclick: () => { list.querySelectorAll('.rdd-pick').forEach((b) => b.classList.toggle('active', b.dataset.id === String(q.id))); pickQuery(q); },
        }, icon('code-square'), h('span.rdd-pick-name', q.name), h('span.text-muted.small', `#${q.id}`))) : emptyState({ icon: 'search', title: 'No queries found' }));
      } catch (err) { mount(list, h('div.text-danger.small.p-2', err.message)); }
    };
    const pickQuery = async (q) => {
      chosenQuery = null;
      mount(vizList, h('div.text-center.p-3', spinner()));
      try {
        const full = await rd.get(`/api/queries/${enc(q.id)}`);
        chosenQuery = full;
        const vis = full.visualizations || [];
        mount(vizList,
          h('div.rdv-section-title', `Visualizations of “${full.name}”`),
          vis.map((v) => h('button.rdd-viz-choice', {
            type: 'button',
            onclick: async () => {
              try {
                const w = await createVizWidget(dashboard, v, full);
                w.visualization ||= { ...v, query: full };
                if (!w.visualization.query) w.visualization.query = full;
                dashboard.widgets.push(w);
                const card = mountCard(w, w.options.position);
                card.load();
                afterWidgetsChanged();
                m.close();
                toast(`Added “${v.name}”`, 'success', { timeout: 2000 });
                setTimeout(() => card.el.scrollIntoView({ behavior: 'smooth', block: 'center' }), 300);
              } catch (err) { toastError(err, 'Could not add the widget'); }
            },
          }, icon(vizTypeInfo(v.type).icon), h('div', h('div.fw-semibold', v.name), h('div.small.text-muted', vizTypeInfo(v.type).label + (v.type === 'CHART' ? ` · ${v.options?.globalSeriesType || ''}` : ''))), icon('plus-circle', 'ms-auto'))),
          h('button.btn.btn-sm.btn-outline-secondary.mt-2', {
            type: 'button',
            onclick: async () => {
              const { openVizEditor } = await import('../core/viz_editor.js');
              const saved = await openVizEditor({ query: full, viz: null });
              if (saved) pickQuery(full);
            },
          }, icon('plus-lg', 'me-1'), 'Create a new visualization for this query'));
      } catch (err) { mount(vizList, h('div.text-danger.small.p-2', err.message)); }
    };
    search.addEventListener('input', debounce(loadQueries, 200));
    const m = modal({
      title: 'Add a visualization', icon: 'bar-chart-line', size: 'lg',
      body: h('div.rdd-addviz', h('div.rdd-addviz-left', search, list), vizList),
      actions: [{ label: 'Close' }],
    });
    loadQueries();
    void chosenQuery;
  }

  // ---------------------------------------------------------------- build

  function drawEmpty() {
    const empty = !(dashboard.widgets || []).length;
    gridHost.classList.toggle('is-empty', empty);
    const existing = gridHost.querySelector('.rdd-empty');
    if (!empty) { existing?.remove(); return; }
    if (existing) return;
    gridHost.prepend(h('div.rdd-empty', emptyState({
      icon: 'grid-1x2', title: 'This dashboard is empty',
      message: 'Add visualizations from your queries, or a text box with notes and section headings.',
      action: dashboard.can_edit && h('div.d-flex.gap-2.justify-content-center',
        h('button.btn.btn-sm.btn-primary', { type: 'button', onclick: () => { if (!editing) setEditing(true); addVizDialog(); } }, icon('bar-chart-line', 'me-1'), 'Add visualization'),
        h('button.btn.btn-sm.btn-outline-primary', { type: 'button', onclick: () => { if (!editing) setEditing(true); addText(); } }, icon('fonts', 'me-1'), 'Add text')),
    })));
  }

  mount(el,
    headerEl,
    h('div.rdd-sticky', toolbarEl),
    progressEl,
    gridHost,
    editBar);
  mount(toolbarEl, dashParamsEl, filterBarEl, sectionsEl);
  sectionsEl.hidden = true;
  el.classList.add('rdd-page');
  drawHeader();
  drawParams();

  dashboard.widgets ||= [];
  const layout = initialLayout(dashboard.widgets);
  grid = gridView(gridHost, [], { layout, editing: false, onChange: onLayoutChange });
  lastLayout = layout;
  for (const w of dashboard.widgets) {
    const card = widgetCard(w, env);
    cards.set(w.id, card);
    grid.add({ id: w.id, el: card.el }, layout.get(w.id));
  }
  grid.setLayout(layout);
  lastLayout = grid.getLayout();
  ctx.cleanup(() => { cards.forEach((c) => c.destroy()); grid.destroy(); });
  drawEmpty();
  drawEditBar();
  if (editing) setEditing(true);
  if (autoRefresh) setAutoRefresh(autoRefresh);

  // Stagger initial loads (cached results are cheap; parameterized ones run).
  runAll([...cards.values()], {}, 'Loading').then(() => { buildSections(); });
  setTimeout(buildSections, 600);
  void defaultMappings;
}
