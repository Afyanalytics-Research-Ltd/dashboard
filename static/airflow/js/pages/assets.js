// Assets: assets, aliases and the global asset-event stream.

import { h, icon, mount, debounce } from '../core/dom.js';
import { api } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  pageHeader, card, dataTable, pager, tabs, searchInput, chips, refreshControl, emptyState, errorState, skeleton,
  timeAgo, jsonView, drawer, kv,
} from '../core/ui.js';
import { fmtNumber, fmtDate } from '../core/format.js';
import { typeBadge, linkList, segmented } from './admin_shared.js';

const SINCE = [
  { value: '', label: 'Any time' },
  { value: '24h', label: '24h', hours: 24 },
  { value: '7d', label: '7 days', hours: 168 },
  { value: '30d', label: '30 days', hours: 720 },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  let tab = ctx.query.get('tab') || 'assets';
  const body = h('div');
  const counts = {};
  const tabBar = h('div');
  const views = {};
  const refresh = refreshControl(() => views[tab]?.reload(), { key: 'assets', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Assets', icon: 'database',
      subtitle: 'Data assets produced and consumed by DAGs, and the events that update them.',
      actions: [refresh.el],
    }),
    card({ body: h('div', tabBar, body) }));

  const drawTabs = () => mount(tabBar, tabs([
    { id: 'assets', label: 'Assets', icon: 'database', count: counts.assets },
    { id: 'events', label: 'Events', icon: 'lightning-charge', count: counts.events },
    { id: 'aliases', label: 'Aliases', icon: 'tags', count: counts.aliases },
  ], tab, (id) => { tab = id; ctx.setQuery({ tab: id === 'assets' ? null : id }); show(); }));

  // Background counts for the tab badges.
  Promise.allSettled([
    api.get('/assets', { limit: 1 }, { signal }),
    api.get('/assets/events', { limit: 1 }, { signal }),
    api.get('/assets/aliases', { limit: 1 }, { signal }),
  ]).then(([a, e, al]) => {
    if (!ctx.active) return;
    counts.assets = a.value?.total_entries;
    counts.events = e.value?.total_entries;
    counts.aliases = al.value?.total_entries;
    drawTabs();
  });

  function show() {
    drawTabs();
    views[tab] ||= { assets: assetsView, events: eventsView, aliases: aliasesView }[tab]();
    mount(body, views[tab].el);
    views[tab].reload();
  }

  // ---------------------------------------------------------------- assets

  function assetsView() {
    const st = {
      q: ctx.query.get('q') || '', field: ctx.query.get('field') || 'name',
      active: ctx.query.get('active') !== 'false', dag: ctx.query.get('dag') || '',
      order: 'name', limit: 50, offset: 0,
    };
    const tableEl = h('div', skeleton(6));
    const pagerEl = h('div');
    const dagSelect = h('select.form-select.form-select-sm.af-select-auto', {
      'aria-label': 'Filter by DAG',
      onchange: () => { st.dag = dagSelect.value; st.offset = 0; ctx.setQuery({ dag: st.dag || null }); reload(); },
    }, h('option', { value: '' }, 'All DAGs'));
    api.all('/dags', 'dags', { order_by: 'dag_id' }, { signal }).then((dags) => {
      dags.forEach((d) => dagSelect.appendChild(h('option', { value: d.dag_id, selected: d.dag_id === st.dag }, d.dag_id)));
    }).catch(() => {});
    const view = h('div',
      h('div.af-toolbar.mb-3',
        searchInput({ placeholder: 'Search assets…', value: st.q, onInput: (q) => { st.q = q; st.offset = 0; ctx.setQuery({ q: q || null }); reload(); } }),
        segmented([{ value: 'name', label: 'Name' }, { value: 'uri', label: 'URI' }], st.field, (v) => { st.field = v; ctx.setQuery({ field: v === 'name' ? null : v }); if (st.q) reload(); }),
        dagSelect,
        h('div.form-check.form-switch.mb-0',
          h('input.form-check-input#af-only-active', {
            type: 'checkbox', checked: st.active,
            onchange: (e) => { st.active = e.target.checked; st.offset = 0; ctx.setQuery({ active: st.active ? null : 'false' }); reload(); },
          }),
          h('label.form-check-label.small', { for: 'af-only-active' }, 'Only active'))),
      tableEl, pagerEl);

    async function reload() {
      try {
        const res = await api.get('/assets', {
          limit: st.limit, offset: st.offset, order_by: st.order, only_active: st.active,
          name_pattern: st.q && st.field === 'name' ? `%${st.q}%` : null,
          uri_pattern: st.q && st.field === 'uri' ? `%${st.q}%` : null,
          dag_ids: st.dag ? [st.dag] : null,
        }, { signal });
        if (!ctx.active) return;
        const [key, dir] = st.order.startsWith('-') ? [st.order.slice(1), 'desc'] : [st.order, 'asc'];
        mount(tableEl, dataTable({
          rows: res.assets || [],
          rowKey: (a) => a.id,
          sort: { key, dir },
          onSort: (k, d) => { st.order = d === 'desc' ? `-${k}` : k; reload(); },
          onRowClick: (a) => ctx.navigate(urls.asset(a.id)),
          empty: emptyState({
            icon: 'database', title: 'No assets',
            message: st.q || st.dag ? 'Nothing matches these filters.' : 'DAGs declare assets with outlets=[Asset(...)] or schedule=[Asset(...)]. None are registered on this server yet.',
          }),
          columns: [
            { key: 'name', label: 'Asset', sortable: true, render: (a) => h('div', h('a.fw-semibold', { href: urls.asset(a.id) }, a.name), h('div.af-mono.small.text-muted.af-ellipsis', { style: { maxWidth: '360px' }, title: a.uri }, a.uri)) },
            { key: 'group', label: 'Group', render: (a) => typeBadge(a.group) },
            { key: 'producing', label: 'Produced by', render: (a) => linkList((a.producing_tasks || []).map((t) => h('a.af-tag', { href: urls.dag(t.dag_id) }, icon('box-arrow-right'), `${t.dag_id}.${t.task_id}`)), '—') },
            { key: 'scheduled', label: 'Triggers DAGs', render: (a) => linkList((a.scheduled_dags || []).map((d) => h('a.af-tag', { href: urls.dag(d.dag_id) }, icon('diagram-3'), d.dag_id)), '—') },
            { key: 'updated_at', label: 'Last event', sortable: 'updated_at', render: (a) => (a.last_asset_event?.timestamp ? timeAgo(a.last_asset_event.timestamp) : h('span.text-muted', 'never')) },
          ],
        }));
        mount(pagerEl, res.total_entries > st.limit || st.offset ? pager({
          total: res.total_entries, limit: st.limit, offset: st.offset,
          onChange: (o, l) => { st.offset = o; st.limit = l; reload(); },
        }) : null);
      } catch (err) {
        if (err.name !== 'AbortError') mount(tableEl, errorState(err, reload));
      }
    }
    return { el: view, reload };
  }

  // ---------------------------------------------------------------- events

  function eventsView() {
    const st = { dag: '', task: '', run: '', since: '', limit: 50, offset: 0, order: '-timestamp' };
    const tableEl = h('div', skeleton(6));
    const pagerEl = h('div');
    const onInput = debounce(() => { st.offset = 0; reload(); }, 300);
    const input = (key, placeholder) => h('input.form-control.form-control-sm.af-event-filter', {
      type: 'search', placeholder, 'aria-label': placeholder, oninput: (e) => { st[key] = e.target.value.trim(); onInput(); },
    });
    const view = h('div',
      h('div.af-toolbar.mb-3',
        input('dag', 'Source DAG id'), input('task', 'Source task id'), input('run', 'Source run id'),
        chips(SINCE, st.since, (v) => { st.since = v; st.offset = 0; reload(); })),
      tableEl, pagerEl);

    async function reload() {
      const since = SINCE.find((s) => s.value === st.since);
      try {
        const res = await api.get('/assets/events', {
          limit: st.limit, offset: st.offset, order_by: st.order,
          source_dag_id: st.dag || null, source_task_id: st.task || null, source_run_id: st.run || null,
          timestamp_gte: since?.hours ? new Date(Date.now() - since.hours * 3600e3).toISOString() : null,
        }, { signal });
        if (!ctx.active) return;
        mount(tableEl, eventsTable(res.asset_events || [], { sort: st.order, onSort: (o) => { st.order = o; reload(); } }));
        mount(pagerEl, res.total_entries > st.limit || st.offset ? pager({
          total: res.total_entries, limit: st.limit, offset: st.offset,
          onChange: (o, l) => { st.offset = o; st.limit = l; reload(); },
        }) : null);
      } catch (err) {
        if (err.name !== 'AbortError') mount(tableEl, errorState(err, reload));
      }
    }
    return { el: view, reload };
  }

  // ---------------------------------------------------------------- aliases

  function aliasesView() {
    const st = { q: '', limit: 50, offset: 0, order: 'name' };
    const tableEl = h('div', skeleton(4));
    const pagerEl = h('div');
    const view = h('div',
      h('div.af-toolbar.mb-3', searchInput({ placeholder: 'Search aliases…', shortcut: false, onInput: (q) => { st.q = q; st.offset = 0; reload(); } })),
      tableEl, pagerEl);
    async function reload() {
      try {
        const res = await api.get('/assets/aliases', {
          limit: st.limit, offset: st.offset, order_by: st.order, name_pattern: st.q ? `%${st.q}%` : null,
        }, { signal });
        if (!ctx.active) return;
        const [key, dir] = st.order.startsWith('-') ? [st.order.slice(1), 'desc'] : [st.order, 'asc'];
        mount(tableEl, dataTable({
          rows: res.asset_aliases || [],
          rowKey: (a) => a.id,
          sort: { key, dir },
          onSort: (k, d) => { st.order = d === 'desc' ? `-${k}` : k; reload(); },
          empty: emptyState({ icon: 'tags', title: 'No asset aliases' }),
          columns: [
            { key: 'id', label: 'Id', sortable: true, width: '80px', render: (a) => h('span.af-mono', a.id) },
            { key: 'name', label: 'Alias', sortable: true, render: (a) => h('span.af-mono.fw-semibold', a.name) },
            { key: 'group', label: 'Group', render: (a) => typeBadge(a.group) },
          ],
        }));
        mount(pagerEl, res.total_entries > st.limit || st.offset ? pager({
          total: res.total_entries, limit: st.limit, offset: st.offset,
          onChange: (o, l) => { st.offset = o; st.limit = l; reload(); },
        }) : null);
      } catch (err) {
        if (err.name !== 'AbortError') mount(tableEl, errorState(err, reload));
      }
    }
    return { el: view, reload };
  }

  show();
}

/** Asset events table, shared with the asset detail page. */
export function eventsTable(events, { sort = '-timestamp', onSort, showAsset = true } = {}) {
  const [key, dir] = sort.startsWith('-') ? [sort.slice(1), 'desc'] : [sort, 'asc'];
  return dataTable({
    rows: events,
    rowKey: (e) => e.id,
    sort: { key, dir },
    onSort: onSort ? (k, d) => onSort(d === 'desc' ? `-${k}` : k) : undefined,
    onRowClick: (e) => showEvent(e),
    empty: emptyState({ icon: 'lightning-charge', title: 'No asset events', message: 'Events appear when a task with this asset as an outlet succeeds, or when one is created manually.' }),
    columns: [
      { key: 'timestamp', label: 'When', sortable: true, render: (e) => timeAgo(e.timestamp) },
      showAsset && { key: 'asset', label: 'Asset', render: (e) => h('a.fw-semibold', { href: urls.asset(e.asset_id) }, e.name || e.uri || `#${e.asset_id}`) },
      {
        key: 'source_dag_id', label: 'Source', sortable: true,
        render: (e) => (e.source_dag_id
          ? h('div.small',
            h('a', { href: urls.dag(e.source_dag_id) }, e.source_dag_id),
            e.source_task_id && e.source_run_id ? [' · ', h('a.af-mono', { href: urls.task(e.source_dag_id, e.source_run_id, e.source_task_id, e.source_map_index) }, e.source_task_id)] : null)
          : h('span.af-tag', icon('person'), 'manual / API')),
      },
      { key: 'source_run_id', label: 'Run', sortable: true, render: (e) => (e.source_run_id ? h('a.af-mono.small.af-ellipsis', { href: urls.run(e.source_dag_id, e.source_run_id), style: { maxWidth: '240px' }, title: e.source_run_id }, e.source_run_id) : h('span.text-muted', '—')) },
      { key: 'created_dagruns', label: 'Triggered runs', align: 'right', render: (e) => fmtNumber((e.created_dagruns || []).length) },
      { key: 'extra', label: 'Extra', render: (e) => (e.extra && Object.keys(e.extra).length ? h('code.af-inline-json', JSON.stringify(e.extra)) : h('span.text-muted', '—')) },
    ].filter(Boolean),
  });
}

function showEvent(e) {
  const d = drawer({
    title: `Asset event #${e.id}`,
    subtitle: e.uri,
    width: 640,
    body: h('div.d-flex.flex-column.gap-3',
      kv([
        ['Asset', h('a', { href: urls.asset(e.asset_id) }, e.name || `#${e.asset_id}`)],
        ['Group', e.group],
        ['When', fmtDate(e.timestamp)],
        ['Source DAG', e.source_dag_id && h('a', { href: urls.dag(e.source_dag_id) }, e.source_dag_id)],
        ['Source task', e.source_task_id],
        ['Map index', e.source_map_index >= 0 ? e.source_map_index : null],
        ['Source run', e.source_run_id && h('a.af-mono.small', { href: urls.run(e.source_dag_id, e.source_run_id) }, e.source_run_id), { wide: true }],
      ]),
      h('div', h('div.af-section-label', 'Extra'), jsonView(e.extra || {})),
      h('div', h('div.af-section-label', `Triggered DAG runs (${(e.created_dagruns || []).length})`),
        linkList((e.created_dagruns || []).map((r) => h('a.af-tag', { href: urls.run(r.dag_id, r.run_id) }, `${r.dag_id} · ${r.run_id}`)), 'None'))),
  });
  d.el.addEventListener('click', (ev) => { if (ev.target.closest('a[href]')) d.close(); });
}
