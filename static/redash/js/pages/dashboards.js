// Dashboards: All / My / Favourites, tag folders, instant search, sort, card grid with
// mini layout previews (or a compact list), create (?new=1), duplicate and archive with undo.

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import {
  pageHeader, card, chips, searchInput, emptyState, errorState, skeleton, timeAgo, formModal, toast, toastError,
  confirmDialog, openMenu, dataTable, refreshControl,
} from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc, redashLink } from '../core/api.js';
import { urls } from '../core/urls.js';
import { favoriteStar, getDashboardTags, invalidate, userChip, cached } from '../core/common.js';
import { vizTypeInfo } from '../core/viz.js';
import { duplicateDashboard } from './dashboard/actions.js';

const VIEWS = [
  { value: 'all', label: 'All dashboards', path: '/api/dashboards' },
  { value: 'my', label: 'My dashboards', path: '/api/dashboards/my' },
  { value: 'favorites', label: 'Favourites', path: '/api/dashboards/favorites' },
];
const SORTS = [
  { value: '-created_at', label: 'Newest first' },
  { value: 'created_at', label: 'Oldest first' },
  { value: 'name', label: 'Name A–Z' },
  { value: '-name', label: 'Name Z–A' },
];
const PAGE_SIZE = 24;

export const dashboardUrl = (d) => urls.dashboard(d.id);
export const redashDashboardUrl = (d) => redashLink(`/dashboards/${d.id}-${d.slug}`);

export default async function render(ctx) {
  const { el, signal } = ctx;
  const q = ctx.query;
  const state = {
    view: VIEWS.some((v) => v.value === q.get('view')) ? q.get('view') : 'all',
    q: q.get('q') || '',
    tags: q.getAll('tags'),
    order: SORTS.some((s) => s.value === q.get('order')) ? q.get('order') : '-created_at',
    page: Math.max(1, Number(q.get('page')) || 1),
    layout: prefs.get('rd:dashboards:layout', 'grid'),
  };
  const sync = () => ctx.setQuery({
    view: state.view === 'all' ? null : state.view, q: state.q, tags: state.tags,
    order: state.order === '-created_at' ? null : state.order, page: state.page > 1 ? state.page : null,
  });

  const tagsEl = h('div.rdd-tagfolders', skeleton(5, { height: 14 }));
  const resultsEl = h('div.rdd-results', skeleton(6, { height: 90 }));
  const countEl = h('span.rdd-count');
  const layoutBtns = h('div.btn-group.btn-group-sm', { role: 'group', 'aria-label': 'Layout' });
  const drawLayoutBtns = () => mount(layoutBtns,
    h('button.btn.btn-outline-secondary', { type: 'button', class: state.layout === 'grid' && 'active', title: 'Cards', onclick: () => setLayout('grid') }, icon('grid-3x2-gap')),
    h('button.btn.btn-outline-secondary', { type: 'button', class: state.layout === 'list' && 'active', title: 'List', onclick: () => setLayout('list') }, icon('list-ul')));
  const setLayout = (l) => { state.layout = l; prefs.set('rd:dashboards:layout', l); drawLayoutBtns(); load(); };
  drawLayoutBtns();

  const refresh = refreshControl(() => load(), { key: 'rd-dashboards', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Dashboards', icon: 'grid-1x2',
      subtitle: 'Every dashboard in Redash. Star the ones you use most — they appear in the sidebar.',
      actions: [
        h('button.btn.btn-sm.btn-primary', { type: 'button', onclick: () => createDialog() }, icon('plus-lg', 'me-1'), 'New dashboard'),
        refresh.el,
      ],
    }),
    h('div.rdd-layout',
      h('aside.rdd-side',
        h('div.rdd-side-title', icon('folder2', 'me-1'), 'Folders (tags)'),
        tagsEl),
      h('div.rdd-main',
        h('div.af-toolbar.rdd-toolbar',
          chips(VIEWS, state.view, (v) => { state.view = v || 'all'; state.page = 1; sync(); load(); }),
          searchInput({ placeholder: 'Search dashboards…', value: state.q, onInput: (v) => { state.q = v; state.page = 1; sync(); load(); } }),
          h('select.form-select.form-select-sm.rdd-sort', {
            'aria-label': 'Sort', onchange: (e) => { state.order = e.target.value; state.page = 1; sync(); load(); },
          }, SORTS.map((s) => h('option', { value: s.value, selected: s.value === state.order }, s.label))),
          layoutBtns,
          countEl),
        resultsEl)));

  if (q.get('new')) {
    ctx.setQuery({ new: null });
    createDialog();
  }

  async function loadTags() {
    try {
      const tags = await getDashboardTags();
      if (!ctx.active) return;
      mount(tagsEl,
        h('button.rdd-folder', { type: 'button', class: !state.tags.length && 'active', onclick: () => { state.tags = []; state.page = 1; sync(); loadTags(); load(); } },
          icon('collection'), h('span', 'Everything')),
        tags.length ? tags.map((t) => {
          const on = state.tags.includes(t.name);
          return h('button.rdd-folder', {
            type: 'button', class: on && 'active',
            onclick: () => { state.tags = on ? state.tags.filter((x) => x !== t.name) : [...state.tags, t.name]; state.page = 1; sync(); loadTags(); load(); },
          }, icon(on ? 'folder2-open' : 'folder2'), h('span', t.name), h('span.rdd-folder-count', String(t.count)));
        }) : h('div.rdd-side-empty', 'No tags yet. Add tags on a dashboard to group them into folders here.'));
    } catch (err) {
      mount(tagsEl, h('div.small.text-danger', err.message));
    }
  }

  let seq = 0;
  async function load() {
    const mine = ++seq;
    const view = VIEWS.find((v) => v.value === state.view);
    try {
      let res;
      if (state.order.endsWith('name')) {
        // Redash 10 returns 500 for order=name on dashboard lists; sort the full list client-side.
        const all = await rd.all(view.path, { q: state.q || undefined, tags: state.tags }, { signal });
        const dir = state.order.startsWith('-') ? -1 : 1;
        all.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base', numeric: true }) * dir);
        res = { count: all.length, results: all.slice((state.page - 1) * PAGE_SIZE, state.page * PAGE_SIZE) };
      } else {
        res = await rd.page(view.path, {
          page: state.page, pageSize: PAGE_SIZE, q: state.q || undefined, tags: state.tags, order: state.q ? undefined : state.order,
        }, { signal });
      }
      if (mine !== seq || !ctx.active) return;
      const items = res.results || [];
      mount(countEl, `${res.count.toLocaleString()} dashboard${res.count === 1 ? '' : 's'}`);
      if (!items.length) {
        mount(resultsEl, emptyState(state.q || state.tags.length
          ? { icon: 'search', title: 'No dashboards match', message: 'Try another search or clear the folder filter.', action: h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => { state.q = ''; state.tags = []; sync(); navigate(urls.dashboards()); } }, 'Clear filters') }
          : state.view === 'favorites'
            ? { icon: 'star', title: 'No favourites yet', message: 'Click the ☆ on any dashboard to keep it here and in the sidebar.' }
            : { icon: 'grid-1x2', title: 'No dashboards yet', message: 'Dashboards combine visualizations from many queries on one page.', action: h('button.btn.btn-sm.btn-primary', { onclick: () => createDialog() }, icon('plus-lg', 'me-1'), 'Create your first dashboard') }));
        return;
      }
      const pages = Math.max(1, Math.ceil(res.count / PAGE_SIZE));
      const pagerEl = pages > 1 && h('div.rdd-pager',
        h('button.af-page-btn', { disabled: state.page <= 1, onclick: () => { state.page--; sync(); load(); } }, icon('chevron-left')),
        h('span.rd-page-label', `Page ${state.page} of ${pages}`),
        h('button.af-page-btn', { disabled: state.page >= pages, onclick: () => { state.page++; sync(); load(); } }, icon('chevron-right')));
      if (state.layout === 'list') mount(resultsEl, card({ body: listTable(items), flush: true }), pagerEl);
      else {
        const grid = h('div.rdd-grid', items.map(dashCard));
        mount(resultsEl, grid, pagerEl);
        loadPreviews(items, grid);
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(resultsEl, errorState(err, load));
    }
  }

  function cardMenu(d) {
    return [
      { label: 'Open', icon: 'box-arrow-in-right', onClick: () => navigate(dashboardUrl(d)) },
      { label: 'Edit layout', icon: 'pencil-square', onClick: () => navigate(urls.dashboard(d.id, { edit: 1 })) },
      { label: 'Duplicate', icon: 'copy', onClick: async () => { const copy = await duplicateDashboard(d); if (copy) navigate(dashboardUrl(copy)); } },
      redashDashboardUrl(d) && { label: 'Open in Redash', icon: 'box-arrow-up-right', href: redashDashboardUrl(d), external: true },
      { divider: true },
      { label: 'Archive', icon: 'archive', danger: true, onClick: () => archive(d) },
    ];
  }

  function dashCard(d) {
    const preview = h('div.rdd-preview', { dataset: { id: d.id } }, h('div.rdd-preview-loading'));
    return h('article.rdd-card', { tabindex: 0, onclick: (e) => { if (!e.target.closest('button,a')) navigate(dashboardUrl(d)); }, onkeydown: (e) => { if (e.key === 'Enter' && e.target === e.currentTarget) navigate(dashboardUrl(d)); } },
      preview,
      h('div.rdd-card-body',
        h('div.rdd-card-title',
          favoriteStar('dashboards', d),
          h('a.rdd-card-name', { href: dashboardUrl(d), title: d.name }, d.name),
          h('button.af-icon-btn.rdd-card-menu', { type: 'button', title: 'More', onclick: (e) => { e.stopPropagation(); openMenu(e.currentTarget, cardMenu(d)); } }, icon('three-dots-vertical'))),
        h('div.rdd-card-meta',
          d.is_draft && h('span.rd-draft-pill', 'draft'),
          (d.tags || []).slice(0, 4).map((t) => h('span.af-tag', t)),
          h('span.rdd-card-time', icon('clock', 'me-1'), timeAgo(d.updated_at || d.created_at)))));
  }

  /** Mini layout thumbnails: fetch dashboards (max 4 in flight) and draw their widget grid. */
  async function loadPreviews(items, grid) {
    const queue = [...items];
    const worker = async () => {
      while (queue.length && ctx.active) {
        const d = queue.shift();
        const target = grid.querySelector(`.rdd-preview[data-id="${d.id}"]`);
        try {
          const full = await cached(`dashpreview:${d.id}:${d.version}`, () => rd.get(`/api/dashboards/${enc(d.id)}`, null, { signal }), 300000);
          if (target) mount(target, miniLayout(full));
        } catch {
          if (target) mount(target, h('div.rdd-preview-empty', icon('grid-1x2')));
        }
      }
    };
    await Promise.all([worker(), worker(), worker(), worker()]);
  }

  function listTable(items) {
    return dataTable({
      rows: items, rowKey: (d) => d.id, onRowClick: (d) => navigate(dashboardUrl(d)),
      columns: [
        { key: 'fav', label: '', width: '34px', render: (d) => favoriteStar('dashboards', d) },
        { key: 'name', label: 'Name', render: (d) => h('div', h('a.fw-semibold', { href: dashboardUrl(d) }, d.name), d.is_draft && h('span.rd-draft-pill.ms-2', 'draft')) },
        { key: 'tags', label: 'Tags', render: (d) => h('div.af-tags', (d.tags || []).map((t) => h('span.af-tag', t))) },
        { key: 'user', label: 'Owner', render: (d) => userChip(d.user) },
        { key: 'created_at', label: 'Created', render: (d) => timeAgo(d.created_at) },
        { key: 'updated_at', label: 'Updated', render: (d) => timeAgo(d.updated_at) },
        { key: 'menu', label: '', width: '44px', render: (d) => h('button.af-icon-btn', { type: 'button', title: 'More', onclick: (e) => { e.stopPropagation(); openMenu(e.currentTarget, cardMenu(d)); } }, icon('three-dots-vertical')) },
      ],
    });
  }

  async function archive(d) {
    const ok = await confirmDialog({ title: `Archive “${d.name}”?`, message: 'It disappears from lists and search. You can undo right after.', confirmLabel: 'Archive', danger: true });
    if (!ok) return;
    try {
      await rd.del(`/api/dashboards/${enc(d.id)}`);
      invalidate('dash');
      toast(`“${d.name}” archived`, 'success', {
        timeout: 8000,
        action: { label: 'Undo', onClick: async () => { await rd.post(`/api/dashboards/${enc(d.id)}`, { is_archived: false }); invalidate('dash'); load(); } },
      });
      load();
    } catch (err) { toastError(err, 'Archive failed'); }
  }

  function createDialog() {
    getDashboardTags().then((tags) => {
      formModal({
        title: 'New dashboard', icon: 'grid-1x2', submitLabel: 'Create dashboard',
        fields: [
          { name: 'name', label: 'Name', required: true, placeholder: 'e.g. Pharmacy weekly overview' },
          { name: 'tags', label: 'Folders (tags)', type: 'tags', value: state.tags, help: tags.length ? `Existing: ${tags.map((t) => t.name).join(', ')}` : 'Comma separated' },
        ],
        onSubmit: async ({ name, tags: chosen }, close) => {
          let d = await rd.post('/api/dashboards', { name: name.trim() });
          if (chosen?.length) d = await rd.post(`/api/dashboards/${enc(d.id)}`, { tags: chosen });
          invalidate('dash');
          close();
          toast(`“${d.name}” created — add your first widget`, 'success');
          navigate(urls.dashboard(d.id, { edit: 1 }));
        },
      });
    }).catch((err) => toastError(err));
  }

  loadTags();
  await load();
}

/** Tiny 6-column thumbnail of a dashboard's widgets, coloured by visualization type. */
export function miniLayout(dashboard) {
  const widgets = dashboard.widgets || [];
  if (!widgets.length) return h('div.rdd-preview-empty', icon('plus-square-dotted'), h('span', 'Empty dashboard'));
  let autoRow = 0;
  const placed = widgets.map((w, i) => {
    const p = w.options?.position;
    if (p && p.sizeX) return { w, p };
    const pos = { col: (i % 2) * 3, row: autoRow, sizeX: 3, sizeY: w.visualization ? 8 : 3 };
    if (i % 2) autoRow += pos.sizeY;
    return { w, p: pos };
  });
  const rows = Math.max(...placed.map(({ p }) => (p.row || 0) + (p.sizeY || 1)), 1);
  return h('div.rdd-mini', { style: { gridTemplateRows: `repeat(${Math.min(rows, 60)}, 1fr)` } },
    placed.map(({ w, p }) => {
      const type = w.visualization ? (w.visualization.type === 'CHART' ? w.visualization.options?.globalSeriesType || 'column' : w.visualization.type) : 'TEXT';
      const ic = w.visualization ? vizTypeInfo(w.visualization.type).icon : 'fonts';
      return h('div.rdd-mini-w', {
        class: `t-${String(type).toLowerCase()}`,
        style: { gridColumn: `${(p.col || 0) + 1} / span ${Math.min(p.sizeX || 3, 6)}`, gridRow: `${(p.row || 0) + 1} / span ${Math.max(p.sizeY || 1, 1)}` },
        title: w.visualization ? `${w.visualization.name} · ${w.visualization.query?.name || ''}` : 'Text',
      }, icon(type === 'pie' ? 'pie-chart' : type === 'line' || type === 'area' ? 'graph-up' : ic));
    }),
    h('span.rdd-mini-count', `${widgets.length} widget${widgets.length === 1 ? '' : 's'}`));
}
