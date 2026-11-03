// Queries list: views (All / My / Favourites / Recent / Archived), tags as folders,
// instant search, sortable columns, bulk tag/archive, unarchive. All state is in the URL.

import { h, icon, mount, prefs } from '../../../ui/js/dom.js';
import {
  pageHeader, card, dataTable, pager, searchInput, selectionBar, emptyState, errorState, skeleton,
  timeAgo, toast, toastError, confirmDialog, formModal, refreshControl,
} from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { fmtDuration } from '../../../ui/js/format.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  favoriteStar, userChip, scheduleText, getDataSources, getQueryTags, moreMenu, invalidate, dataSourceIcon,
} from '../core/common.js';

const VIEWS = [
  { id: 'all', label: 'All queries', icon: 'collection', path: '/api/queries' },
  { id: 'my', label: 'My queries', icon: 'person', path: '/api/queries/my' },
  { id: 'favorites', label: 'Favourites', icon: 'star', path: '/api/queries/favorites' },
  { id: 'recent', label: 'Recently edited', icon: 'clock-history', path: '/api/queries/recent' },
  { id: 'archive', label: 'Archived', icon: 'archive', path: '/api/queries/archive' },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  const state = {
    view: VIEWS.some((v) => v.id === ctx.query.get('view')) ? ctx.query.get('view') : 'all',
    q: ctx.query.get('q') || '',
    tags: ctx.query.getAll('tags'),
    order: ctx.query.get('order') || '-created_at',
    page: Number(ctx.query.get('page')) || 1,
    pageSize: prefs.get('rd:queries:pageSize', 25),
  };
  const selected = new Set();
  let lastRows = [];

  const foldersEl = h('div.rd-q-strip');
  const tableEl = h('div', skeleton(8));
  const pagerEl = h('div');
  const barEl = h('div');
  const titleEl = h('span');

  const SORTS = [
    ['-created_at', 'Newest first'], ['created_at', 'Oldest first'], ['name', 'Name A → Z'], ['-name', 'Name Z → A'],
    ['-executed_at', 'Recently run'], ['-runtime', 'Slowest'], ['-schedule', 'Most frequent refresh'],
  ];
  const sortSelect = h('select.form-select.form-select-sm.rd-q-sort', {
    'aria-label': 'Sort queries',
    onchange: (e) => { state.order = e.target.value; state.page = 1; sync(); load(); },
  }, SORTS.map(([v, l]) => h('option', { value: v, selected: v === state.order }, l)));
  const refresh = refreshControl(() => load(), { key: 'queries', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Queries', icon: 'code-square',
      subtitle: 'Write SQL, explore schemas and turn results into charts. Tags work as folders.',
      actions: [
        h('a.btn.btn-sm.btn-primary', { href: urls.newQuery() }, icon('plus-lg', 'me-1'), 'New query'),
        refresh.el,
      ],
    }),
    h('section.af-card.rd-q-main',
      h('div.rd-q-toolbar',
        h('h2.af-card-title', titleEl),
        searchInput({
          placeholder: 'Search names, SQL and descriptions…', value: state.q,
          onInput: (v) => { state.q = v; state.page = 1; sync(); load(); },
        }),
        sortSelect),
      foldersEl,
      h('div.rd-q-active-tags'),
      tableEl,
      pagerEl),
    barEl);

  function sync() {
    ctx.setQuery({
      view: state.view === 'all' ? null : state.view,
      q: state.q || null,
      tags: state.tags,
      order: state.order === '-created_at' ? null : state.order,
      page: state.page > 1 ? state.page : null,
    });
  }

  async function renderFolders() {
    let tags = [];
    try { tags = await getQueryTags(); } catch { /* tags are optional */ }
    if (!ctx.active) return;
    mount(foldersEl,
      h('div.rd-q-views', { role: 'tablist' }, VIEWS.map((v) => h('button.rd-q-view', {
        type: 'button', role: 'tab', class: state.view === v.id && 'active', 'aria-selected': state.view === v.id,
        onclick: () => { state.view = v.id; state.page = 1; selected.clear(); sync(); renderFolders(); load(); },
      }, icon(v.icon), h('span', v.label)))),
      h('div.rd-q-folders-row',
        h('span.rd-q-folders-label', icon('folder2', 'me-1'), 'Folders'),
        tags.length
          ? tags.map((t) => {
            const on = state.tags.includes(t.name);
            return h('button.rd-q-folder', {
              type: 'button', class: on && 'active', title: on ? 'Remove filter' : `Show only queries tagged “${t.name}”`,
              onclick: () => {
                state.tags = on ? state.tags.filter((x) => x !== t.name) : [...state.tags, t.name];
                state.page = 1; sync(); renderFolders(); load();
              },
            }, icon(on ? 'folder2-open' : 'folder2'), h('span', t.name), h('span.rd-q-count', String(t.count)));
          })
          : h('span.rd-q-side-empty', 'No tags yet — tag queries to group them into folders.')));
  }

  const sortable = (key) => (state.view === 'recent' ? false : key);

  async function load() {
    const view = VIEWS.find((v) => v.id === state.view);
    mount(titleEl, view.label, state.tags.length ? h('span.rd-q-title-tags', ` · ${state.tags.join(', ')}`) : null);
    mount(el.querySelector('.rd-q-active-tags'), state.tags.length ? h('div.rd-q-chips',
      state.tags.map((t) => h('button.af-tag.active', {
        type: 'button', onclick: () => { state.tags = state.tags.filter((x) => x !== t); sync(); renderFolders(); load(); },
      }, icon('folder2-open'), t, icon('x'))),
      h('button.af-link-btn', { onclick: () => { state.tags = []; sync(); renderFolders(); load(); } }, 'Clear')) : null);
    if (!lastRows.length) mount(tableEl, skeleton(8));
    let rows;
    let total;
    try {
      const [dsList, res] = await Promise.all([
        getDataSources().catch(() => []),
        state.view === 'recent'
          ? rd.get(view.path, null, { signal })
          : rd.get(view.path, { page: state.page, page_size: state.pageSize, q: state.q, tags: state.tags, order: state.order }, { signal }),
      ]);
      dataSources = new Map(dsList.map((d) => [d.id, d]));
      if (Array.isArray(res)) {
        const q = state.q.toLowerCase();
        rows = res.filter((r) => (!q || `${r.name} ${r.query || ''}`.toLowerCase().includes(q))
          && state.tags.every((t) => (r.tags || []).includes(t)));
        total = rows.length;
      } else {
        rows = res.results || [];
        total = res.count ?? rows.length;
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(tableEl, errorState(err, load));
      mount(pagerEl);
      return;
    }
    if (!ctx.active) return;
    lastRows = rows;
    renderTable(rows, total);
  }

  let dataSources = new Map();

  function renderTable(rows, total) {
    const empty = state.q || state.tags.length
      ? emptyState({ icon: 'search', title: 'No queries match', message: 'Try a different search or clear the tag filter.' })
      : {
        favorites: emptyState({ icon: 'star', title: 'No favourites yet', message: 'Star a query to keep it one click away — it also appears in the sidebar.' }),
        archive: emptyState({ icon: 'archive', title: 'Nothing archived', message: 'Archived queries show up here, and you can restore them any time.' }),
        my: emptyState({ icon: 'person', title: 'You have no queries yet', action: h('a.btn.btn-sm.btn-primary.mt-2', { href: urls.newQuery() }, icon('plus-lg', 'me-1'), 'Write your first query') }),
      }[state.view] || emptyState({ icon: 'code-square', title: 'No queries yet', action: h('a.btn.btn-sm.btn-primary.mt-2', { href: urls.newQuery() }, icon('plus-lg', 'me-1'), 'New query') });

    const [key, dir] = state.order.startsWith('-') ? [state.order.slice(1), 'desc'] : [state.order, 'asc'];
    mount(tableEl, dataTable({
      rows,
      rowKey: (r) => r.id,
      selectable: true,
      selected,
      onSelect: renderBar,
      sort: { key, dir },
      onSort: (k, d) => {
        state.order = d === 'desc' ? `-${k}` : k;
        if ([...sortSelect.options].some((o) => o.value === state.order)) sortSelect.value = state.order;
        state.page = 1; sync(); load();
      },
      onRowClick: (r) => navigate(urls.query(r.id)),
      rowClass: (r) => r.is_archived && 'rd-q-row-archived',
      empty,
      columns: [
        { key: 'fav', label: '', width: '34px', render: (r) => favoriteStar('queries', r) },
        {
          key: 'name', label: 'Name', sortable: sortable('name'),
          render: (r) => h('div.rd-q-name-cell',
            h('a.rd-q-name', { href: urls.query(r.id) }, r.name || 'Untitled query'),
            h('div.rd-q-meta',
              h('span.af-mono', `#${r.id}`),
              r.is_draft && h('span.rd-draft-pill', 'draft'),
              r.is_archived && h('span.rd-archived-pill', 'archived'),
              (r.tags || []).map((t) => h('button.af-tag', {
                type: 'button', title: `Show only "${t}"`,
                onclick: (e) => { e.stopPropagation(); if (!state.tags.includes(t)) state.tags = [...state.tags, t]; state.page = 1; sync(); renderFolders(); load(); },
              }, t))),
            r.description && h('div.rd-q-desc', r.description)),
        },
        {
          key: 'ds', label: 'Data source',
          render: (r) => {
            const ds = dataSources.get(r.data_source_id);
            return ds ? h('span.rd-type-pill', icon(dataSourceIcon(ds.type)), ds.name) : h('span.text-muted', `#${r.data_source_id}`);
          },
        },
        {
          key: 'created_by', label: 'Owner', sortable: sortable('created_by'),
          render: (r) => h('div.rd-q-owner', userChip(r.user), h('div.small.text-muted', 'created ', timeAgo(r.created_at))),
        },
        { key: 'schedule', label: 'Refresh', sortable: sortable('schedule'), render: (r) => (r.schedule?.interval ? h('span.rd-q-sched', icon('arrow-repeat'), scheduleText(r.schedule)) : h('span.text-muted', 'Manual')) },
        {
          key: 'executed_at', label: 'Last run', sortable: sortable('executed_at'),
          render: (r) => (r.retrieved_at ? h('div', timeAgo(r.retrieved_at), r.runtime != null && h('div.small.text-muted', fmtDuration(r.runtime))) : h('span.text-muted', 'Never')),
        },
        {
          key: 'actions', label: '', width: '44px', cls: 'af-no-row-click',
          render: (r) => moreMenu(() => [
            { label: 'Open', icon: 'box-arrow-up-right', href: urls.query(r.id) },
            { label: 'Fork (make a copy)', icon: 'diagram-2', onClick: () => forkQuery(r) },
            { label: 'Add tag…', icon: 'tag', onClick: () => tagDialog([r], 'add') },
            { divider: true },
            r.is_archived
              ? { label: 'Unarchive', icon: 'arrow-counterclockwise', onClick: () => setArchived([r], false) }
              : { label: 'Archive', icon: 'archive', danger: true, onClick: () => setArchived([r], true) },
          ]),
        },
      ],
    }));
    if (state.view === 'recent') mount(pagerEl);
    else {
      mount(pagerEl, pager({
        total, limit: state.pageSize, offset: (state.page - 1) * state.pageSize,
        onChange: (offset, limit) => {
          state.pageSize = limit;
          prefs.set('rd:queries:pageSize', limit);
          state.page = Math.floor(offset / limit) + 1;
          sync(); load();
        },
      }));
    }
    renderBar();
  }

  function selectedRows() { return lastRows.filter((r) => selected.has(r.id)); }

  function renderBar() {
    const rows = selectedRows();
    const anyArchived = rows.some((r) => r.is_archived);
    mount(barEl, selectionBar(rows.length, [
      h('button.btn.btn-sm.btn-light', { onclick: () => tagDialog(rows, 'add') }, icon('tag', 'me-1'), 'Add tag'),
      h('button.btn.btn-sm.btn-light', { onclick: () => tagDialog(rows, 'remove') }, icon('tag', 'me-1'), 'Remove tag'),
      anyArchived
        ? h('button.btn.btn-sm.btn-light', { onclick: () => setArchived(rows, false) }, icon('arrow-counterclockwise', 'me-1'), 'Unarchive')
        : h('button.btn.btn-sm.btn-warning', { onclick: () => setArchived(rows, true) }, icon('archive', 'me-1'), 'Archive'),
    ], () => { selected.clear(); load(); }));
  }

  /** Run an async op for each row sequentially, with a progress toast. */
  async function eachRow(rows, label, fn) {
    let ok = 0;
    const errors = [];
    for (const r of rows) {
      try { await fn(r); ok++; } catch (err) { errors.push(`#${r.id}: ${err.message}`); }
    }
    if (errors.length) toast(errors.slice(0, 3).join('\n'), 'warning', { title: `${label}: ${ok} done, ${errors.length} failed` });
    else toast(`${label}: ${ok} quer${ok === 1 ? 'y' : 'ies'}`, 'success');
    selected.clear();
    invalidate('qtags');
    renderFolders();
    load();
  }

  async function setArchived(rows, archive) {
    const ok = await confirmDialog({
      title: archive ? `Archive ${rows.length} quer${rows.length === 1 ? 'y' : 'ies'}?` : `Restore ${rows.length} quer${rows.length === 1 ? 'y' : 'ies'}?`,
      message: archive
        ? 'Archived queries disappear from lists and search, and their widgets are removed from dashboards. You can restore them from the Archived view.'
        : 'They will be visible in lists and search again. Dashboard widgets that were removed on archive are not restored.',
      details: h('ul.small.mb-0', rows.slice(0, 8).map((r) => h('li', r.name)), rows.length > 8 && h('li', `…and ${rows.length - 8} more`)),
      confirmLabel: archive ? 'Archive' : 'Restore', danger: archive,
    });
    if (!ok) return;
    await eachRow(rows, archive ? 'Archived' : 'Restored', (r) => (archive
      ? rd.del(`/api/queries/${enc(r.id)}`)
      : rd.post(`/api/queries/${enc(r.id)}`, { is_archived: false })));
  }

  async function tagDialog(rows, mode) {
    let suggestions = [];
    try { suggestions = (await getQueryTags()).map((t) => t.name); } catch { /* optional */ }
    const listId = `rd-tag-list-${Math.random().toString(36).slice(2, 7)}`;
    formModal({
      title: mode === 'add' ? 'Add tag' : 'Remove tag', icon: 'tag',
      subtitle: `${rows.length} quer${rows.length === 1 ? 'y' : 'ies'}`,
      fields: [{ name: 'tag', label: 'Tag', required: true, placeholder: 'e.g. finance', help: 'Tags act as folders in the sidebar.' }],
      extra: h('datalist', { id: listId }, suggestions.map((s) => h('option', { value: s }))),
      submitLabel: mode === 'add' ? 'Add tag' : 'Remove tag',
      onSubmit: async ({ tag }, close) => {
        close();
        const t = tag.trim();
        await eachRow(rows, mode === 'add' ? `Tagged "${t}"` : `Removed "${t}"`, (r) => {
          const tags = new Set(r.tags || []);
          if (mode === 'add') tags.add(t); else tags.delete(t);
          return rd.post(`/api/queries/${enc(r.id)}`, { tags: [...tags] });
        });
      },
    }).el.querySelector('input')?.setAttribute('list', listId);
  }

  async function forkQuery(r) {
    try {
      const copy = await rd.post(`/api/queries/${enc(r.id)}/fork`, {});
      toast(`Forked into #${copy.id}`, 'success');
      navigate(urls.query(copy.id));
    } catch (err) { toastError(err, 'Fork failed'); }
  }

  renderFolders();
  await load();
}
