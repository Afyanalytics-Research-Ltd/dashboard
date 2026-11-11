// Query History: real-time (INFORMATION_SCHEMA) or 30-day account-wide history,
// with an interactive timeline, KPI strip, badges, multi-select compare + SQL diff.

import { h, icon, mount, card, empty, banner, errorView, toolkitBanner, chip, seg, popover, table, modal,
  skeleton, sevBadge, copyButton, toast, bar } from '../lib/ui.js';
import { api } from '../lib/api.js';
import { chart, ColorScale, STATUS, axisStyle, tipRow, escapeHtml, tableToggle } from '../lib/charts.js';
import { state, hasAccountUsage } from '../lib/store.js';
import { openQuery } from '../lib/query.js';
import * as fmt from '../lib/fmt.js';
import * as x from '../lib/explorercore.js';
import { prefs, debounce } from '../../../ui/js/dom.js';
import { href } from '../../../ui/js/router.js';

const HOURS = [
  { value: '1', label: '1h' }, { value: '6', label: '6h' }, { value: '24', label: '24h' },
  { value: '72', label: '3d' }, { value: '168', label: '7d' }, { value: '720', label: '30d' },
];
const SORT_OPTS = [
  { value: 'start', label: 'Newest' }, { value: 'duration', label: 'Slowest' }, { value: 'bytes', label: 'Most scanned' },
  { value: 'spill', label: 'Most spilled' }, { value: 'queued', label: 'Most queued' }, { value: 'compile', label: 'Longest compile' },
];
const QUERY_TYPES = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE_TABLE_AS_SELECT', 'COPY', 'CALL', 'UNLOAD', 'CREATE_TABLE'];
const PRESET_KEY = 'ws:history:presets';

export default async function render(ctx) {
  ctx.setTitle(ctx.params.queryId ? 'Query' : 'Query History');
  let filters = x.parseHistoryFilters(ctx.query);
  const accountOk = hasAccountUsage();
  if (filters.scope === 'account' && !accountOk) filters.scope = 'live';
  let rows = [];
  let selected = new Set();
  let timeRange = null;            // [msStart, msEnd] from the timeline zoom
  let auto = false;
  let loading = false;
  let reqId = 0;

  const filtersEl = h('div.ws-col', { style: { gap: '10px' } });
  const kpiEl = h('div.wsx-kpis');
  const timelineEl = h('div.ws-chart', { style: { height: '250px' } });
  const timelineFoot = h('div.ws-row.ws-between.ws-wrap', { style: { marginTop: '6px' } });
  const tableHost = h('div');
  const tableTools = h('div.ws-row.ws-wrap', { style: { gap: '6px' } });
  const moreEl = h('div', { style: { textAlign: 'center', marginTop: '12px' } });
  let timeline = null;

  mount(ctx.el,
    h('div.ws-card.ws-card-pad.ws-mb', filtersEl),
    kpiEl,
    card({ title: 'Timeline', icon: 'graph-up', sub: 'Each dot is a query: when it started, how long it took (log scale), dot size = data scanned. Drag the slider or scroll to zoom — the table follows.',
      body: h('div', timelineEl, timelineFoot) }),
    h('div.ws-mt'),
    card({ title: 'Queries', icon: 'list-ul', tools: tableTools, body: h('div', tableHost, moreEl) }));

  renderFilters();
  await load();
  if (ctx.params.queryId) openQuery(ctx.params.queryId);

  ctx.every(15000, () => { if (auto && filters.scope === 'live' && !loading) load({ quiet: true }); });

  // ---------------------------------------------------------------- filters

  function update(patch, { reload = true } = {}) {
    filters = x.parseHistoryFilters({ ...filters, ...patch });
    ctx.setQuery(x.historyQueryPatch(filters));
    renderFilters();
    if (reload) { timeRange = null; load(); }
  }

  function picker(label, key, options, ic) {
    const value = filters[key];
    return h(`button.ws-pill${value ? '.wsx-pill-on' : ''}`, { type: 'button', title: `Filter by ${label.toLowerCase()}`,
      onclick: (e) => popover(e.currentTarget, { search: options.length > 7, width: 300,
        items: [{ label: `Any ${label.toLowerCase()}`, value: '', icon: 'asterisk', on: !value },
          ...options.map((o) => ({ label: o, value: o, icon: ic, on: o === value }))],
        onPick: (i) => update({ [key]: i.value }) }) },
    icon(ic), h('span.ws-pill-k', label), h('b', value || 'any'), icon('chevron-down'));
  }

  function distinct(key, extra = []) {
    return [...new Set([...extra, ...rows.map((r) => r[key]).filter(Boolean)])].sort();
  }

  function renderFilters() {
    const scopeSeg = h('div.ws-seg', [
      { value: 'live', label: 'Live · 7 days', icon: 'broadcast', title: 'INFORMATION_SCHEMA: real time, queries this role can see' },
      { value: 'account', label: 'Account · 30 days', icon: 'globe2', title: accountOk ? 'Every user, complete statistics, ~45 min latency' : 'Needs the toolkit or ACCOUNT_USAGE access' },
    ].map((o) => h(`button${filters.scope === o.value ? '.on' : ''}`, { type: 'button', title: o.title, disabled: o.value === 'account' && !accountOk,
      onclick: () => update({ scope: o.value, hours: o.value === 'live' && Number(filters.hours) > 168 ? '24' : filters.hours }) }, icon(o.icon, 'me-1'), o.label)));
    const hoursSeg = seg(HOURS.filter((o) => filters.scope === 'account' || Number(o.value) <= 168), filters.hours, (v) => update({ hours: v }));
    const search = h('input.ws-input', { type: 'search', placeholder: 'Search SQL text or paste a query id…', value: filters.q,
      oninput: debounce((e) => update({ q: e.target.value }), 400) });
    const minMs = h('input.ws-input', { type: 'number', min: 0, step: 100, placeholder: 'min ms', value: filters.min_ms, style: { width: '110px' },
      title: 'Minimum duration (ms)', onchange: (e) => update({ min_ms: e.target.value }) });
    const statusChips = [['', 'All'], ['SUCCESS', 'Succeeded'], ['FAILED', 'Failed'], ['RUNNING', 'Running']].map(([v, label]) => chip(label, {
      on: filters.status === v, onClick: () => update({ status: v }),
      color: v === 'FAILED' ? STATUS.CRITICAL : v === 'SUCCESS' ? STATUS.good : v === 'RUNNING' ? STATUS.INFO : null }));
    const whs = distinct('warehouse_name', (state.session?.warehouses || []).map((w) => w.name));
    mount(filtersEl,
      h('div.ws-row.ws-wrap', scopeSeg, hoursSeg,
        h('div.ws-search.ws-grow', { style: { flex: '1 1 260px' } }, icon('search'), search), minMs,
        h('select.ws-select', { onchange: (e) => update({ sort: e.target.value }), title: 'Sort' },
          SORT_OPTS.filter((o) => filters.scope === 'account' || o.value !== 'spill').map((o) => h('option', { value: o.value, selected: o.value === filters.sort }, o.label)))),
      h('div.ws-row.ws-wrap', { style: { gap: '6px' } }, statusChips, h('span.wsx-sep'),
        picker('Warehouse', 'warehouse', whs, 'cpu'), picker('User', 'user', distinct('user_name'), 'person'),
        picker('Database', 'database', distinct('database_name'), 'database'), picker('Type', 'type', distinct('query_type', QUERY_TYPES), 'tag'),
        filters.scope === 'account' && chip('Spilled only', { on: filters.spilled === '1', onClick: () => update({ spilled: filters.spilled === '1' ? '' : '1' }) }),
        chip('Hide SHOW/DESCRIBE', { on: filters.hide_system === '1', onClick: () => update({ hide_system: filters.hide_system === '1' ? '0' : '1' }) }),
        h('span.wsx-sep'), presetsButton(),
        filters.scope === 'live' && h('button.ws-chip', { type: 'button', class: auto ? 'on' : '', title: 'Refresh every 15 s',
          onclick: () => { auto = !auto; renderFilters(); if (auto) load({ quiet: true }); } },
        h('span.ws-dot', { class: auto ? 'on' : '' }), auto ? 'Live' : 'Auto-refresh'),
        h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', onclick: () => load(), title: 'Reload' }, icon('arrow-clockwise'), 'Refresh')));
  }

  function presetsButton() {
    return h('button.ws-pill', { type: 'button', title: 'Saved filters', onclick: (e) => {
      const presets = prefs.get(PRESET_KEY, []);
      popover(e.currentTarget, { width: 300, items: [
        { label: 'Save current filters…', value: '__save', icon: 'bookmark-plus' },
        ...(presets.length ? [{ head: 'Saved' }] : []),
        ...presets.map((p, i) => ({ label: p.name, value: i, icon: 'bookmark' })),
        ...(presets.length ? [{ label: 'Delete all saved filters', value: '__clear', icon: 'trash' }] : []),
      ], onPick: (item) => {
        if (item.value === '__save') {
          const name = window.prompt('Name this filter preset', `${filters.scope} · ${filters.status || 'all'} · ${filters.hours}h`);
          if (!name) return;
          prefs.set(PRESET_KEY, [{ name: name.slice(0, 60), filters }, ...presets.filter((p) => p.name !== name)].slice(0, 20));
          toast('Filter preset saved', 'success');
        } else if (item.value === '__clear') {
          prefs.set(PRESET_KEY, []);
        } else {
          update(presets[item.value].filters);
        }
      } });
    } }, icon('bookmark'), h('b', 'Presets'), icon('chevron-down'));
  }

  // ---------------------------------------------------------------- data

  async function load({ quiet = false } = {}) {
    const my = ++reqId;
    loading = true;
    if (!quiet) { mount(tableHost, skeleton(8, 26)); mount(kpiEl, h('div.ws-skel', { style: { height: '92px', borderRadius: '16px', marginBottom: '16px' } })); }
    try {
      const res = await api.history(x.historyApiParams(filters), { signal: ctx.signal });
      if (my !== reqId || !ctx.active) return;
      rows = res.rows || [];
      selected = new Set([...selected].filter((id) => rows.some((r) => r.query_id === id)));
      renderFilters();
      renderAll();
    } catch (err) {
      if (err.name === 'AbortError' || my !== reqId) return;
      mount(kpiEl);
      mount(tableHost, err.code === 'toolkit' ? toolkitBanner(err.message) : errorView(err, () => load()));
    } finally {
      if (my === reqId) loading = false;
    }
  }

  function visibleRows() {
    if (!timeRange) return rows;
    return rows.filter((r) => { const t = Date.parse(r.start_time); return t >= timeRange[0] && t <= timeRange[1]; });
  }

  function renderAll() {
    renderTimeline();
    renderTable();
  }

  // ---------------------------------------------------------------- KPIs

  function renderKpis(list) {
    const k = x.historyKpis(list);
    const tile = (label, value, ic, foot, cls = '') => h(`div.ws-card.ws-stat${cls}`, h('div.ws-stat-label', icon(ic), label), h('div.ws-stat-value', { style: { fontSize: '24px' } }, value), foot && h('div.ws-stat-foot', foot));
    mount(kpiEl, h('div.ws-grid.wsx-kpi-grid.ws-mb',
      tile('Queries', fmt.fmtInt(k.count), 'list-ol', timeRange ? 'in the zoomed window' : `${filters.scope === 'live' ? 'real time' : 'account-wide'} · last ${fmt.fmtInt(filters.hours)}h`),
      tile('Failed', fmt.pct(k.failedRatio, 1), 'x-octagon', `${fmt.fmtInt(k.failed)} queries`, k.failed ? '.wsx-kpi-bad' : ''),
      tile('Median', fmt.ms(k.p50), 'stopwatch', `p95 ${fmt.ms(k.p95)}`),
      tile('Est. cost', fmt.usd(k.usd), 'cash-coin', `${fmt.credits(k.credits)} credits`),
      tile('Scanned', fmt.bytes(k.bytes), 'hdd', filters.scope === 'live' ? 'spill stats: account scope' : `${fmt.fmtInt(k.spilled)} spilled`)));
  }

  // ---------------------------------------------------------------- timeline

  function renderTimeline() {
    timeline?.dispose();
    timeline = null;
    const pts = rows.filter((r) => r.start_time && Number(r.total_elapsed_time ?? r.elapsed_ms) > 0);
    renderKpis(visibleRows());
    if (!pts.length) {
      mount(timelineEl, empty({ icon: 'graph-up', title: 'Nothing to plot', text: 'No queries match these filters.' }));
      mount(timelineFoot);
      return;
    }
    mount(timelineEl);
    const whKeys = [...new Set(pts.filter((r) => !x.isFailed(r)).map((r) => r.warehouse_name || '(no warehouse)'))];
    // Order warehouses by volume so the most used ones get the first (most separable) colours.
    const counts = new Map();
    pts.forEach((r) => counts.set(r.warehouse_name || '(no warehouse)', (counts.get(r.warehouse_name || '(no warehouse)') || 0) + 1));
    whKeys.sort((a, b) => counts.get(b) - counts.get(a));
    const shown = whKeys.slice(0, 7);
    const scale = new ColorScale(shown);
    const maxBytes = Math.max(1, ...pts.map((r) => Number(r.bytes_scanned) || 0));
    const size = (r) => 6 + 18 * Math.sqrt((Number(r.bytes_scanned) || 0) / maxBytes);
    const groups = [...shown.map((k) => ({ name: k, rows: pts.filter((r) => !x.isFailed(r) && (r.warehouse_name || '(no warehouse)') === k) })),
      ...(whKeys.length > 7 ? [{ name: 'Other warehouses', rows: pts.filter((r) => !x.isFailed(r) && !shown.includes(r.warehouse_name || '(no warehouse)')), other: true }] : []),
      { name: 'Failed', rows: pts.filter(x.isFailed), failed: true }].filter((g) => g.rows.length);
    timeline = chart(timelineEl, (t) => ({
      tooltip: { trigger: 'item', confine: true, backgroundColor: t.surface2, borderColor: t.border, textStyle: { color: t.text, fontSize: 12 },
        extraCssText: 'border-radius:12px;box-shadow:0 12px 32px rgba(0,0,0,.35);padding:10px 12px;max-width:420px;white-space:normal',
        formatter: (p) => {
          const r = p.data.row;
          return `<div style="font-weight:600;margin-bottom:4px">${escapeHtml(fmt.dateTime(r.start_time))}</div>`
            + `<div style="font-family:JetBrains Mono,monospace;font-size:11px;opacity:.85;margin-bottom:6px">${escapeHtml(fmt.oneLine(r.query_text, 160))}</div>`
            + tipRow(p.color, 'Duration', fmt.ms(r.total_elapsed_time ?? r.elapsed_ms)) + tipRow(p.color, 'Warehouse', r.warehouse_name || '—')
            + tipRow(p.color, 'Scanned', fmt.bytes(r.bytes_scanned)) + tipRow(p.color, 'Est. cost', fmt.usd(r.usd))
            + `<div style="margin-top:6px;opacity:.7;font-size:11px">Click to inspect</div>`;
        } },
      legend: { top: 0, left: 0, icon: 'circle', itemWidth: 9, itemHeight: 9, textStyle: { color: t.text2, fontSize: 11 }, type: 'scroll' },
      grid: { left: 8, right: 18, top: 30, bottom: 44, containLabel: true },
      xAxis: { type: 'time', ...axisStyle(t, { splitLine: { show: false } }) },
      yAxis: { type: 'log', logBase: 10, name: 'duration', nameTextStyle: { color: t.muted, fontSize: 11 },
        ...axisStyle(t, { axisLabel: { color: t.muted, fontSize: 11, formatter: (v) => fmt.ms(v) } }) },
      dataZoom: [{ type: 'inside', xAxisIndex: 0, filterMode: 'none' },
        { type: 'slider', xAxisIndex: 0, height: 18, bottom: 6, filterMode: 'none', borderColor: t.grid, backgroundColor: 'transparent',
          fillerColor: 'rgba(41,181,232,.15)', dataBackground: { lineStyle: { color: t.axis }, areaStyle: { color: t.grid } },
          handleStyle: { color: t.accent }, textStyle: { color: t.muted, fontSize: 10 }, labelFormatter: (v) => fmt.dateTime(v) }],
      series: groups.map((g) => ({
        name: g.name, type: 'scatter',
        data: g.rows.map((r) => ({ value: [Date.parse(r.start_time), Math.max(1, Number(r.total_elapsed_time ?? r.elapsed_ms))], row: r, symbolSize: size(r) })),
        itemStyle: { color: g.failed ? STATUS.CRITICAL : g.other ? t.muted : scale.get(g.name), opacity: 0.8, borderColor: t.surface, borderWidth: 1.5 },
        emphasis: { scale: 1.4, itemStyle: { opacity: 1 } },
        ...(g.failed ? { symbol: 'diamond' } : {}),
      })),
    }), ctx);
    if (!timeline) return;
    timeline.inst.on('click', (p) => p.data?.row && openQuery(p.data.row.query_id));
    timeline.inst.on('datazoom', debounce(() => {
      const opt = timeline.inst.getOption();
      const dz = opt.dataZoom?.[0];
      const xs = pts.map((r) => Date.parse(r.start_time));
      const lo = Math.min(...xs);
      const hi = Math.max(...xs);
      const start = dz.startValue ?? lo + ((dz.start ?? 0) / 100) * (hi - lo);
      const end = dz.endValue ?? lo + ((dz.end ?? 100) / 100) * (hi - lo);
      timeRange = (dz.start ?? 0) <= 0.01 && (dz.end ?? 100) >= 99.99 ? null : [start, end];
      renderKpis(visibleRows());
      renderTable();
      renderTimelineFoot(groups);
    }, 150));
    renderTimelineFoot(groups);
  }

  function renderTimelineFoot(groups) {
    const tt = tableToggle([
      { key: 'name', label: 'Series' }, { key: 'n', label: 'Queries', num: true, fmt: fmt.fmtInt },
      { key: 'p50', label: 'Median', num: true, fmt: fmt.ms }, { key: 'usd', label: 'Est. cost', num: true, fmt: fmt.usd },
    ], groups.map((g) => ({ name: g.name, n: g.rows.length, p50: x.percentile(g.rows.map((r) => Number(r.total_elapsed_time ?? r.elapsed_ms)), 0.5), usd: g.rows.reduce((a, r) => a + (Number(r.usd) || 0), 0) })));
    mount(timelineFoot,
      h('div.ws-muted', { style: { fontSize: '12px' } }, timeRange
        ? h('span', icon('zoom-in', 'me-1'), `Showing ${fmt.dateTime(timeRange[0])} → ${fmt.dateTime(timeRange[1])} `,
          h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', onclick: () => { timeline?.inst.dispatchAction({ type: 'dataZoom', start: 0, end: 100 }); } }, 'Reset zoom'))
        : 'Diamonds are failed queries. Colour = warehouse.'),
      tt.button, tt.table);
  }

  // ---------------------------------------------------------------- table

  function renderTable() {
    const list = visibleRows();
    const maxDur = Math.max(1, ...list.map((r) => Number(r.total_elapsed_time ?? r.elapsed_ms) || 0));
    const toggle = (id, on) => { if (on) selected.add(id); else selected.delete(id); renderTools(); };
    const cols = [
      { key: 'sel', label: '', sort: false, width: '34px', render: (r) => h('input', { type: 'checkbox', checked: selected.has(r.query_id), 'aria-label': 'Select for compare',
        onclick: (e) => { e.stopPropagation(); toggle(r.query_id, e.target.checked); } }) },
      { key: 'execution_status', label: '', width: '28px', render: (r) => statusIcon(r) },
      { key: 'start_time', label: 'Started', sort: (r) => Date.parse(r.start_time) || 0, render: (r) => h('span', { title: fmt.dateTime(r.start_time) }, fmt.relTime(r.start_time)) },
      { key: 'total_elapsed_time', label: 'Duration', num: true, sort: (r) => Number(r.total_elapsed_time ?? r.elapsed_ms) || 0, render: (r) => h('div.wsx-dur',
        h('span', fmt.ms(r.total_elapsed_time ?? r.elapsed_ms)), bar(x.durationBar(r.total_elapsed_time ?? r.elapsed_ms, maxDur), x.isFailed(r) ? STATUS.CRITICAL : null)) },
      { key: 'query_text', label: 'SQL', render: (r) => h('div.ws-col', { style: { gap: '4px', minWidth: 0 } },
        h('span.ws-sql-snippet', { title: fmt.truncate(r.query_text, 1500) }, fmt.oneLine(r.query_text, 220) || '—'),
        badgesEl(r)) },
      { key: 'user_name', label: 'User', render: (r) => h('span.ws-ellipsis', { style: { maxWidth: '140px' } }, r.user_name || '—') },
      { key: 'warehouse_name', label: 'Warehouse', render: (r) => (r.warehouse_name ? h('span', r.warehouse_name, r.warehouse_size && h('span.ws-tag', { style: { marginLeft: '6px' } }, r.warehouse_size)) : h('span.ws-muted', '—')) },
      { key: 'rows_produced', label: 'Rows', num: true, render: (r) => fmt.compact(r.rows_produced) },
      { key: 'bytes_scanned', label: 'Scanned', num: true, render: (r) => fmt.bytes(r.bytes_scanned) },
      { key: 'usd', label: 'Est. $', num: true, render: (r) => h('span', { title: `${fmt.credits(r.credits ?? r.est_credits)} credits${r.credits == null ? ' (estimated from execution time × warehouse rate)' : ' (attributed by Snowflake)'}` }, fmt.usd(r.usd)) },
    ];
    const t = table(cols, list, { onRow: (r) => openQuery(r.query_id), maxHeight: '70vh', rowKey: (r) => r.query_id,
      rowCls: (r) => (selected.has(r.query_id) ? 'wsx-row-sel' : ''),
      empty: empty({ icon: 'search', title: 'No queries match', text: 'Widen the time window or clear some filters.' }) });
    mount(tableHost, t.el);
    renderTools();
    const lim = Number(filters.limit);
    mount(moreEl, rows.length >= lim && lim < 1000
      ? h('button.ws-btn', { type: 'button', onclick: () => update({ limit: String(Math.min(1000, lim + 200)) }) }, icon('chevron-double-down'), `Load more (showing ${fmt.fmtInt(rows.length)})`)
      : rows.length ? h('span.ws-muted', { style: { fontSize: '12px' } }, `${fmt.fmtInt(rows.length)} queries loaded`) : null);
  }

  function renderTools() {
    const n = selected.size;
    mount(tableTools,
      n ? h('span.ws-muted', { style: { fontSize: '12px' } }, `${n} selected`) : h('span.ws-muted', { style: { fontSize: '12px' } }, 'Tick two queries to compare them'),
      h('button.ws-btn.ws-btn-sm', { type: 'button', disabled: n !== 2, title: 'Side-by-side metrics and SQL diff', onclick: () => compare([...selected]) }, icon('layout-split'), 'Compare'),
      n ? h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', onclick: () => { selected.clear(); renderTable(); } }, 'Clear') : null,
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', title: 'Download the loaded rows as CSV', onclick: downloadCsv }, icon('download'), 'CSV'));
  }

  function downloadCsv() {
    const cols = ['query_id', 'start_time', 'execution_status', 'total_elapsed_time', 'user_name', 'role_name', 'warehouse_name', 'warehouse_size',
      'database_name', 'query_type', 'rows_produced', 'bytes_scanned', 'credits', 'est_credits', 'usd', 'query_text'];
    const esc = (v) => { const s = v == null ? '' : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const csv = [cols.join(','), ...visibleRows().map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n');
    const a = h('a', { href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })), download: `query-history-${filters.scope}.csv` });
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  // ---------------------------------------------------------------- compare

  async function compare([idA, idB]) {
    const rowOf = (id) => rows.find((r) => r.query_id === id) || {};
    const ordered = [rowOf(idA), rowOf(idB)].sort((a, b) => (Date.parse(a.start_time) || 0) - (Date.parse(b.start_time) || 0));
    const body = h('div', skeleton(6, 22));
    modal({ title: 'Compare queries', sub: 'B relative to A (A = the earlier run)', width: 1100, body, actions: [{ label: 'Close' }] });
    let detA = null;
    let detB = null;
    try {
      [detA, detB] = await Promise.all(ordered.map((r) => api.details(r.query_id).catch(() => null)));
    } catch { /* fall back to the history rows */ }
    const qa = { ...ordered[0], ...(detA?.query || {}), usd: detA?.cost?.usd ?? ordered[0].usd };
    const qb = { ...ordered[1], ...(detB?.query || {}), usd: detB?.cost?.usd ?? ordered[1].usd };
    let normalize = true;
    const diffHost = h('div');
    const renderDiff = () => {
      const d = x.diffLines(normalize ? x.normalizeSql(qa.query_text) : qa.query_text, normalize ? x.normalizeSql(qb.query_text) : qb.query_text);
      const st = x.diffStats(d);
      mount(diffHost,
        h('div.ws-row.ws-between.ws-wrap', { style: { marginBottom: '8px' } },
          h('div.ws-row', { style: { gap: '8px' } }, st.identical ? h('span.ws-badge.ws-sev-INFO', icon('check2'), 'identical SQL')
            : [h('span.ws-tag.wsx-add', `+${st.add}`), h('span.ws-tag.wsx-del', `−${st.del}`), h('span.ws-muted', `${fmt.pct(st.similarity)} similar`)]),
          chip('Ignore whitespace', { on: normalize, onClick: () => { normalize = !normalize; renderDiff(); } })),
        h('div.wsx-diff', d.map((line) => h(`div.wsx-diff-line.wsx-${line.op}`,
          h('span.wsx-ln', line.a ?? ''), h('span.wsx-ln', line.b ?? ''), h('span.wsx-op', line.op === 'add' ? '+' : line.op === 'del' ? '−' : ' '), h('code', line.text || ' ')))));
    };
    renderDiff();
    const metrics = x.compareMetrics(qa, qb);
    const fmtKind = (kind, v) => (v == null ? '—' : kind === 'ms' ? fmt.ms(v) : kind === 'bytes' ? fmt.bytes(v) : kind === 'usd' ? fmt.usd(v) : fmt.fmtInt(v));
    const header = (q, label) => h('div.wsx-cmp-head', h('div.ws-row', h('span.ws-badge.ws-sev-INFO', label), statusIcon(q), h('a.ws-mono', { href: '#', onclick: (e) => { e.preventDefault(); openQuery(q.query_id); } }, q.query_id)),
      h('div.ws-muted', { style: { fontSize: '12px' } }, `${fmt.dateTime(q.start_time)} · ${q.warehouse_name || 'no warehouse'} ${q.warehouse_size || ''} · ${q.user_name || ''}`));
    mount(body,
      h('div.ws-grid.ws-g-2.ws-mb', header(qa, 'A'), header(qb, 'B')),
      h('div.ws-table-wrap', h('table.ws-table',
        h('thead', h('tr', h('th', 'Metric'), h('th.num', 'A'), h('th.num', 'B'), h('th.num', 'Δ'), h('th', ''))),
        h('tbody', metrics.map((m) => h('tr', h('td.strong', m.label), h('td.num', fmtKind(m.kind, m.a)), h('td.num', fmtKind(m.kind, m.b)),
          h('td.num', m.delta == null ? '—' : `${m.delta > 0 ? '+' : m.delta < 0 ? '−' : ''}${fmtKind(m.kind, Math.abs(m.delta))}${Number.isFinite(m.ratio) && m.ratio ? ` (${m.ratio > 0 ? '+' : '−'}${fmt.pct(Math.abs(m.ratio))})` : ''}`),
          h('td', verdictBadge(m.verdict))))))),
      h('div.ws-mt'),
      (detA?.insights || detB?.insights) && h('div.ws-grid.ws-g-2.ws-mb',
        h('div', h('div.ws-card-sub.ws-mb', 'A — diagnosis'), insightsMini(detA?.insights)),
        h('div', h('div.ws-card-sub.ws-mb', 'B — diagnosis'), insightsMini(detB?.insights))),
      h('h4.ws-card-title', { style: { marginBottom: '8px' } }, icon('file-diff'), 'SQL diff'),
      diffHost);
  }
}

function insightsMini(list) {
  if (!list?.length) return h('div.ws-muted', '—');
  return h('div.ws-col', { style: { gap: '6px' } }, list.slice(0, 4).map((i) => h('div.ws-row', { style: { gap: '8px' } }, sevBadge(i.severity), h('span', i.title), i.metric && h('span.ws-tag', i.metric))));
}

function verdictBadge(v) {
  if (v === 'better') return h('span.ws-badge.wsx-better', icon('arrow-down-right'), 'better');
  if (v === 'worse') return h('span.ws-badge.ws-sev-HIGH', icon('arrow-up-right'), 'worse');
  if (v === 'changed') return h('span.ws-badge.ws-sev-LOW', 'changed');
  if (v === 'same') return h('span.ws-muted', { style: { fontSize: '12px' } }, 'same');
  return h('span.ws-muted', { style: { fontSize: '12px' } }, 'n/a');
}

function statusIcon(r) {
  if (x.isFailed(r)) return h('span', { title: `${r.execution_status}: ${r.error_message || ''}` }, icon('x-octagon-fill', 'wsx-st-fail'));
  if (x.isRunning(r)) return h('span', { title: r.execution_status }, h('span.ws-spinner', { style: { width: '13px', height: '13px' } }));
  return h('span', { title: r.execution_status || 'SUCCESS' }, icon('check-circle-fill', 'wsx-st-ok'));
}

function badgesEl(r) {
  const b = x.queryBadges(r);
  if (!b.length && !r.error_message) return null;
  return h('div.ws-row.ws-wrap', { style: { gap: '4px' } },
    b.map((v) => h(`span.ws-badge.ws-sev-${v.severity}.wsx-mini`, { title: v.title }, v.label)),
    r.error_message && h('span.wsx-err', { title: r.error_message }, fmt.truncate(r.error_message, 90)));
}

export { href };
