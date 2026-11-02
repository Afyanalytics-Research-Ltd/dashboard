// DAG list: server-side filtering/sorting/paging (all state in the URL),
// favourites, pause switches, recent-run strips and bulk actions.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../core/router.js';
import {
  pageHeader, card, stateBadge, stateInfo, timeAgo, refreshControl, emptyState, errorState, chips,
  skeleton, dataTable, pager, selectionBar, searchInput, menuButton, confirmDialog, toast, toastError,
  RUN_STATES,
} from '../core/ui.js';
import { tip, tipRows } from '../core/charts.js';
import { fmtDate, fmtDuration, fmtNumber, truncate } from '../core/format.js';
import { setPaused, setPausedBulk, setFavorite, triggerDag, reparseDag, deleteDag } from '../core/actions.js';

const STRIP = 14;

/** Run `fn` over items with at most `limit` in flight. */
async function pool(items, limit, fn) {
  const queue = [...items];
  const workers = Array.from({ length: Math.min(limit, queue.length) }, async () => {
    while (queue.length) await fn(queue.shift());
  });
  await Promise.all(workers);
}

export default async function render(ctx) {
  const { el, signal, query } = ctx;
  const state = {
    q: query.get('q') || '',
    tags: query.getAll('tags'),
    match: query.get('match') || 'any',
    owners: query.getAll('owners'),
    paused: query.get('paused') || '',
    fav: query.get('fav') === '1',
    errors: query.get('errors') === '1',
    last: query.get('last') || '',
    stale: query.get('stale') === '1',
    sort: query.get('sort') || 'dag_id',
    limit: Number(query.get('limit')) || 25,
    offset: Number(query.get('offset')) || 0,
  };
  const selected = new Set();
  let rows = [];
  let total = 0;
  let favourites = new Set();
  const runsByDag = new Map();
  let filtersOpen = !!(state.tags.length || state.owners.length);

  const countsEl = h('div');
  const filterPanel = h('div.af-dags-filters');
  const activeEl = h('div.af-dags-active');
  const tableEl = h('div', skeleton(8));
  const pagerEl = h('div');
  const selEl = h('div');

  const refresh = refreshControl(() => load({ quiet: true }), { key: 'dags', initial: 30 });
  ctx.cleanup(refresh.stop);

  const filterBtn = h('button.btn.btn-sm.btn-outline-secondary.af-filter-toggle', {
    type: 'button',
    onclick: () => { filtersOpen = !filtersOpen; renderFilterPanel(); },
  }, icon('funnel'), ' Tags & owners');

  mount(el,
    pageHeader({
      title: 'DAGs', icon: 'diagram-3',
      subtitle: 'Every pipeline Airflow knows about. Filters, sorting and paging are kept in the URL so views can be shared.',
      actions: [refresh.el],
    }),
    card({
      body: h('div.d-flex.flex-column.gap-3',
        h('div.af-toolbar',
          searchInput({ placeholder: 'Search DAGs…', value: state.q, onInput: (v) => update({ q: v }) }),
          countsEl,
          h('div.af-grow'),
          filterBtn,
          h('label.form-check.form-switch.small.mb-0.af-nowrap', { title: 'Include DAGs whose file no longer exists' },
            h('input.form-check-input', { type: 'checkbox', checked: state.stale, onchange: (e) => update({ stale: e.target.checked }) }),
            h('span.form-check-label', 'Show stale'))),
        h('div.af-toolbar',
          h('span.small.text-muted.fw-semibold', 'Last run'),
          chips([{ value: '', label: 'Any' }, ...RUN_STATES.map((s) => ({ value: s, label: stateInfo(s).label, color: stateInfo(s).color }))],
            state.last, (v) => update({ last: v }))),
        filterPanel,
        activeEl),
    }),
    card({ body: h('div', tableEl, pagerEl), flush: true }),
    selEl);

  // ---------------------------------------------------------------- state & URL

  function update(patch, { resetOffset = true } = {}) {
    Object.assign(state, patch);
    if (resetOffset && !('offset' in patch)) state.offset = 0;
    ctx.setQuery({
      q: state.q, tags: state.tags, match: state.match === 'any' ? null : state.match, owners: state.owners,
      paused: state.paused, fav: state.fav ? '1' : null, errors: state.errors ? '1' : null, last: state.last,
      stale: state.stale ? '1' : null, sort: state.sort === 'dag_id' ? null : state.sort,
      limit: state.limit === 25 ? null : state.limit, offset: state.offset || null,
    });
    selected.clear();
    renderActive();
    load();
  }

  function apiParams() {
    const p = {
      limit: state.limit, offset: state.offset, order_by: state.sort,
      dag_id_pattern: state.q ? (state.q.includes('%') ? state.q : `%${state.q}%`) : null,
      tags: state.tags, tags_match_mode: state.tags.length > 1 ? state.match : null,
      owners: state.owners, exclude_stale: !state.stale,
      last_dag_run_state: state.last || null,
    };
    if (state.paused) p.paused = state.paused;
    if (state.fav) p.is_favorite = true;
    if (state.errors) p.has_import_errors = true;
    return p;
  }

  // ---------------------------------------------------------------- loading

  async function load({ quiet = false } = {}) {
    if (!quiet) mount(tableEl, skeleton(8));
    try {
      const [page, favs] = await Promise.all([
        api.get('/dags', apiParams(), { signal }),
        api.get('/dags', { is_favorite: true, limit: 100, exclude_stale: false }, { signal }).catch(() => ({ dags: [] })),
      ]);
      if (!ctx.active) return;
      rows = page.dags;
      total = page.total_entries;
      favourites = new Set(favs.dags.map((d) => d.dag_id));
      if (state.offset && state.offset >= total && total) { update({ offset: Math.max(0, total - state.limit) }, { resetOffset: false }); return; }
      renderTable();
      loadCounts();
      loadRuns(rows.map((d) => d.dag_id));
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(tableEl, errorState(err, () => load()));
      mount(pagerEl);
    }
  }

  async function loadCounts() {
    const base = { limit: 1, exclude_stale: !state.stale };
    const q = (extra) => api.get('/dags', { ...base, ...extra }, { signal }).then((r) => r.total_entries).catch(() => null);
    const [all, active, paused, errs] = await Promise.all([q({}), q({ paused: false }), q({ paused: true }), q({ has_import_errors: true })]);
    if (!ctx.active) return;
    const opts = [
      { value: 'all', label: 'All', count: all },
      { value: 'active', label: 'Active', count: active, color: stateInfo('success').color },
      { value: 'paused', label: 'Paused', count: paused, color: stateInfo('paused').color },
      { value: 'fav', label: '★ Favourites', count: favourites.size },
      errs ? { value: 'errors', label: 'Import errors', count: errs, color: stateInfo('failed').color } : null,
    ].filter(Boolean);
    const current = state.fav ? 'fav' : state.errors ? 'errors' : state.paused === 'false' ? 'active' : state.paused === 'true' ? 'paused' : 'all';
    mount(countsEl, chips(opts, current, (v) => {
      update({
        paused: v === 'active' ? 'false' : v === 'paused' ? 'true' : '',
        fav: v === 'fav', errors: v === 'errors',
      });
    }));
  }

  async function loadRuns(dagIds) {
    await pool(dagIds, 6, async (id) => {
      try {
        const res = await api.get(`/dags/${enc(id)}/dagRuns`, { limit: STRIP, order_by: '-run_after' }, { signal });
        runsByDag.set(id, res.dag_runs);
      } catch (err) {
        if (err.name === 'AbortError') return;
        runsByDag.set(id, null);
      }
      if (!ctx.active) return;
      const strip = tableEl.querySelector(`[data-strip="${CSS.escape(id)}"]`);
      const last = tableEl.querySelector(`[data-last="${CSS.escape(id)}"]`);
      if (strip) mount(strip, runStrip(id));
      if (last) mount(last, lastRun(id));
    });
  }

  // ---------------------------------------------------------------- filters

  async function renderFilterPanel() {
    filterBtn.classList.toggle('active', filtersOpen);
    if (!filtersOpen) { mount(filterPanel); return; }
    mount(filterPanel, skeleton(2));
    try {
      const [tagRes, allDags] = await Promise.all([
        api.get('/dagTags', { limit: 100 }, { signal }),
        api.all('/dags', 'dags', { exclude_stale: true }, { max: 1000, signal }),
      ]);
      if (!ctx.active || !filtersOpen) return;
      const tagCounts = new Map();
      const ownerCounts = new Map();
      for (const d of allDags) {
        (d.tags || []).forEach((t) => tagCounts.set(t.name, (tagCounts.get(t.name) || 0) + 1));
        (d.owners || []).forEach((o) => ownerCounts.set(o, (ownerCounts.get(o) || 0) + 1));
      }
      const tagNames = tagRes.tags.length ? tagRes.tags : [...tagCounts.keys()].sort();
      mount(filterPanel,
        h('div.af-dags-filter-row',
          h('div.af-dags-filter-label', icon('tags'), ' Tags',
            h('div.btn-group.btn-group-sm.ms-2', { role: 'group', title: 'How multiple tags combine' },
              ['any', 'all'].map((m) => h('button.btn', {
                type: 'button', class: state.match === m ? 'btn-secondary' : 'btn-outline-secondary',
                onclick: () => { update({ match: m }); renderFilterPanel(); },
              }, m === 'any' ? 'Any' : 'All')))),
          chips(tagNames.map((t) => ({ value: t, label: t, count: tagCounts.get(t) })), state.tags,
            (v) => update({ tags: v }), { multi: true })),
        h('div.af-dags-filter-row',
          h('div.af-dags-filter-label', icon('person'), ' Owners'),
          chips([...ownerCounts.entries()].sort((a, b) => b[1] - a[1]).map(([o, n]) => ({ value: o, label: o, count: n })),
            state.owners, (v) => update({ owners: v }), { multi: true })));
    } catch (err) {
      if (err.name !== 'AbortError') mount(filterPanel, errorState(err));
    }
  }

  function renderActive() {
    const pills = [
      ...state.tags.map((t) => ({ label: `tag: ${t}`, clear: () => update({ tags: state.tags.filter((x) => x !== t) }) })),
      ...state.owners.map((o) => ({ label: `owner: ${o}`, clear: () => update({ owners: state.owners.filter((x) => x !== o) }) })),
    ];
    if (!pills.length) { mount(activeEl); return; }
    mount(activeEl, h('div.af-toolbar',
      pills.map((p) => h('button.af-tag.active', { type: 'button', onclick: () => { p.clear(); if (filtersOpen) renderFilterPanel(); } }, p.label, icon('x'))),
      h('button.af-link-btn', { onclick: () => { update({ tags: [], owners: [] }); if (filtersOpen) renderFilterPanel(); } }, 'Clear all')));
  }

  // ---------------------------------------------------------------- table

  function runStrip(dagId) {
    const runs = runsByDag.get(dagId);
    if (runs === undefined) return h('span.af-strip-loading');
    if (runs === null) return h('span.text-muted.small', 'n/a');
    if (!runs.length) return h('span.text-muted.small', 'no runs');
    const cells = [...runs].reverse();
    while (cells.length < STRIP) cells.unshift(null);
    return h('div.af-strip', cells.map((r) => {
      if (!r) return h('span.af-strip-cell.empty');
      const info = stateInfo(r.state);
      const a = h('a.af-strip-cell', {
        href: urls.run(dagId, r.dag_run_id), class: r.state === 'running' && 'af-pulse',
        style: { background: info.color }, 'aria-label': `${r.dag_run_id} ${r.state}`,
      });
      tip(a, () => tipRows(r.dag_run_id, [
        ['State', info.label], ['Type', r.run_type], ['Run after', fmtDate(r.run_after)],
        ['Duration', fmtDuration(r.duration)],
      ]));
      return a;
    }));
  }

  function lastRun(dagId) {
    const runs = runsByDag.get(dagId);
    if (!runs?.length) return h('span.text-muted', '—');
    const r = runs[0];
    return h('a.d-flex.flex-column.gap-1.align-items-start', { href: urls.run(dagId, r.dag_run_id) },
      stateBadge(r.state), h('span.small.text-muted', timeAgo(r.start_date || r.run_after)));
  }

  function rowMenu(d) {
    return menuButton(null, () => [
      { label: 'Open', icon: 'box-arrow-in-right', href: urls.dag(d.dag_id) },
      { label: 'Runs', icon: 'play-circle', href: urls.dag(d.dag_id, 'runs') },
      { label: 'Graph', icon: 'diagram-2', href: urls.dag(d.dag_id, 'graph') },
      { label: 'Code', icon: 'file-earmark-code', href: urls.dag(d.dag_id, 'code') },
      { divider: true },
      { label: 'Trigger…', icon: 'play-fill', onClick: () => triggerDag(d.dag_id) },
      { label: d.is_paused ? 'Unpause' : 'Pause', icon: d.is_paused ? 'play' : 'pause', onClick: () => togglePause(d, !d.is_paused) },
      { label: 'Re-parse DAG file', icon: 'arrow-repeat', onClick: () => reparseDag(d) },
      { divider: true },
      { label: 'Delete DAG…', icon: 'trash', danger: true, onClick: async () => { if (await deleteDag(d.dag_id)) load(); } },
    ], { title: 'Actions' });
  }

  async function togglePause(d, paused) {
    const res = await setPaused(d, paused);
    if (res) { d.is_paused = res.is_paused; renderTable(); loadCounts(); }
    else renderTable();
  }

  async function toggleFav(d) {
    const on = !favourites.has(d.dag_id);
    if (await setFavorite(d.dag_id, on)) {
      on ? favourites.add(d.dag_id) : favourites.delete(d.dag_id);
      if (state.fav && !on) load(); else { renderTable(); loadCounts(); }
    }
  }

  function renderTable() {
    const sortKey = state.sort.replace(/^-/, '');
    const columns = [
      {
        key: 'fav', label: '', width: '28px',
        render: (d) => h('button.af-fav', {
          type: 'button', class: favourites.has(d.dag_id) && 'on',
          title: favourites.has(d.dag_id) ? 'Remove from favourites' : 'Add to favourites',
          onclick: () => toggleFav(d),
        }, icon(favourites.has(d.dag_id) ? 'star-fill' : 'star')),
      },
      {
        key: 'paused', label: 'On', width: '56px', title: 'Unpaused DAGs are scheduled',
        render: (d) => h('div.form-check.form-switch.af-switch', { title: d.is_paused ? 'Paused — click to unpause' : 'Active — click to pause' },
          h('input.form-check-input', { type: 'checkbox', role: 'switch', checked: !d.is_paused, 'aria-label': `Toggle ${d.dag_id}`, onchange: (e) => togglePause(d, !e.target.checked) })),
      },
      {
        key: 'dag_id', label: 'DAG', sortable: 'dag_id',
        render: (d) => h('div.af-dag-cell',
          h('div.d-flex.align-items-center.gap-2.flex-wrap',
            h('a.af-dag-name', { href: urls.dag(d.dag_id) }, d.dag_display_name || d.dag_id),
            d.has_import_errors && h('a.badge.text-bg-danger', { href: urls.importErrors(), title: 'This DAG file failed to import' }, icon('bug'), ' import error'),
            d.is_stale && h('span.badge.text-bg-secondary', 'stale')),
          d.dag_display_name && d.dag_display_name !== d.dag_id && h('div.af-mono.small.text-muted', d.dag_id),
          d.owners?.length ? h('div.af-dag-owner', icon('person', 'me-1'), d.owners.join(', ')) : null,
          d.description && h('div.af-dag-desc', { title: d.description }, truncate(d.description, 140)),
          d.tags?.length ? h('div.af-tags.mt-1', d.tags.map((t) => h('button.af-tag', {
            type: 'button', class: state.tags.includes(t.name) && 'active',
            onclick: () => update({ tags: state.tags.includes(t.name) ? state.tags.filter((x) => x !== t.name) : [...state.tags, t.name] }),
          }, t.name))) : null),
      },
      {
        key: 'schedule', label: 'Schedule',
        render: (d) => d.timetable_summary
          ? h('code.af-schedule', { title: d.timetable_description || '' }, d.timetable_summary)
          : h('span.small.text-muted.af-nowrap', { title: d.timetable_description || 'No schedule' }, icon('hand-index', 'me-1'), 'Manual'),
      },
      { key: 'runs', label: `Last ${STRIP} runs`, render: (d) => h('div', { dataset: { strip: d.dag_id } }, runStrip(d.dag_id)) },
      { key: 'last', label: 'Last run', sortable: 'last_run_start_date', render: (d) => h('div', { dataset: { last: d.dag_id } }, lastRun(d.dag_id)) },
      {
        key: 'next', label: 'Next run', sortable: 'next_dagrun',
        render: (d) => (d.next_dagrun_run_after && !d.is_paused
          ? h('span.small', timeAgo(d.next_dagrun_run_after))
          : h('span.small.text-muted', d.is_paused && d.next_dagrun_run_after ? 'paused' : '—')),
      },
      {
        key: 'actions', label: '', align: 'right', width: '96px',
        render: (d) => h('div.d-flex.gap-1.justify-content-end',
          h('button.btn.btn-sm.btn-outline-primary', { type: 'button', title: 'Trigger', onclick: () => triggerDag(d.dag_id) }, icon('play-fill')),
          rowMenu(d)),
      },
    ];
    mount(tableEl, dataTable({
      columns, rows, rowKey: (d) => d.dag_id, selectable: true, selected,
      sort: { key: sortKey, dir: state.sort.startsWith('-') ? 'desc' : 'asc' },
      onSort: (key, dir) => update({ sort: dir === 'desc' ? `-${key}` : key }),
      onSelect: renderSelection,
      onRowClick: (d) => navigate(urls.dag(d.dag_id)),
      rowClass: (d) => (d.is_paused ? 'af-row-paused' : null),
      empty: emptyState({
        icon: 'diagram-3', title: 'No DAGs match',
        message: 'Try clearing the search or filters.',
        action: h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => update({ q: '', tags: [], owners: [], paused: '', fav: false, errors: false, last: '' }) }, 'Reset filters'),
      }),
    }));
    mount(pagerEl, total ? pager({
      total, limit: state.limit, offset: state.offset, sizes: [25, 50, 100],
      onChange: (offset, limit) => update({ offset, limit }, { resetOffset: false }),
    }) : null);
    renderSelection();
  }

  function renderSelection() {
    const ids = [...selected];
    mount(selEl, selectionBar(ids.length, [
      h('button.btn.btn-sm.btn-light', { onclick: () => bulkPause(ids, false) }, icon('play', 'me-1'), 'Unpause'),
      h('button.btn.btn-sm.btn-light', { onclick: () => bulkPause(ids, true) }, icon('pause', 'me-1'), 'Pause'),
      h('button.btn.btn-sm.btn-warning', { onclick: () => bulkTrigger(ids) }, icon('play-fill', 'me-1'), 'Trigger'),
    ], () => { selected.clear(); renderTable(); }));
  }

  async function bulkPause(ids, paused) {
    const ok = await confirmDialog({
      title: `${paused ? 'Pause' : 'Unpause'} ${ids.length} DAG(s)?`,
      message: ids.slice(0, 15).join(', ') + (ids.length > 15 ? '…' : ''),
      confirmLabel: paused ? 'Pause' : 'Unpause',
    });
    if (!ok) return;
    await setPausedBulk(ids, paused);
    selected.clear();
    load({ quiet: true });
  }

  async function bulkTrigger(ids) {
    if (ids.length === 1) { triggerDag(ids[0]); return; }
    const ok = await confirmDialog({
      title: `Trigger ${ids.length} DAGs?`,
      message: 'Each DAG gets one run now with its default parameters (empty conf). Paused DAGs stay queued until unpaused.',
      details: h('div.af-mono.small', ids.join(', ')),
      confirmLabel: `Trigger ${ids.length}`,
    });
    if (!ok) return;
    const results = await Promise.allSettled(ids.map((id) => api.post(`/dags/${enc(id)}/dagRuns`, {
      logical_date: null, conf: {}, note: 'Bulk-triggered from Afya DataHub',
    })));
    const failed = results.filter((r) => r.status === 'rejected');
    if (failed.length) toastError(failed[0].reason, `${failed.length} of ${ids.length} triggers failed`);
    else toast(`${fmtNumber(ids.length)} runs created`, 'success');
    selected.clear();
    load({ quiet: true });
  }

  renderActive();
  if (filtersOpen) renderFilterPanel();
  await load();
}
