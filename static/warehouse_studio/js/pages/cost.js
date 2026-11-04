// Cost Center: spend, forecast, burn-up, heat-map, monitors, warehouses,
// attribution, expensive query shapes and storage — every $ figure is
// recomputed client-side from credits at the viewer's credit price.

import { h, icon, mount, card, stat, delta, seg, tabs, table, modal, banner, errorView, toolkitBanner, skeletonCards,
  skeleton, empty, copyButton, bar, ring, toast } from '../lib/ui.js';
import * as charts from '../lib/charts.js';
import { api } from '../lib/api.js';
import { creditPrice } from '../lib/store.js';
import * as fmt from '../lib/fmt.js';
import * as cc from '../lib/costcore.js';
import { openQuery } from '../lib/query.js';
import { openWarehouse, svgSpark, stateDot } from '../lib/warehousepanel.js';
import { prefs } from '../../../ui/js/dom.js';
import { href } from '../../../ui/js/router.js';

const PERIODS = [7, 30, 90];
const TABS = [
  { key: 'overview', label: 'Overview', icon: 'grid-1x2' },
  { key: 'warehouses', label: 'Warehouses', icon: 'hdd-rack' },
  { key: 'attribution', label: 'Attribution', icon: 'people' },
  { key: 'queries', label: 'Expensive queries', icon: 'lightning' },
  { key: 'storage', label: 'Storage', icon: 'hdd-stack' },
];
const DIMS = [
  { value: 'user', label: 'Users', icon: 'person' }, { value: 'role', label: 'Roles', icon: 'person-badge' },
  { value: 'warehouse', label: 'Warehouses', icon: 'cpu' }, { value: 'database', label: 'Databases', icon: 'database' },
  { value: 'query_tag', label: 'Query tags', icon: 'tag' },
];
const BY = [
  { value: 'credits', label: 'Credits' }, { value: 'elapsed', label: 'Total time' }, { value: 'executions', label: 'Executions' },
  { value: 'bytes', label: 'Bytes scanned' }, { value: 'spill', label: 'Spill' },
];

/** Viewer's credit price: local override, else the server's. */
export const priceNow = () => cc.effectivePrice(prefs.get(cc.PRICE_KEY, null), creditPrice());

/** Pill showing the credit price; click to override it locally. */
export function priceButton(onChange) {
  const btn = h('button.ws-pill', { type: 'button', title: 'Price per credit used for every $ figure' });
  const paint = () => mount(btn, icon('tag'), h('span.ws-pill-k', 'Credit'), h('b', fmt.usd(priceNow())),
    prefs.get(cc.PRICE_KEY, null) != null ? h('span.ws-tag', 'custom') : null);
  paint();
  btn.addEventListener('click', () => {
    const input = h('input.ws-input', { type: 'number', min: '0.01', step: '0.01', value: String(priceNow()), style: { width: '100%' } });
    modal({ title: 'Price per credit', sub: 'Your contract price — saved in this browser only. Snowflake bills credits; this converts them to dollars.',
      width: 420,
      body: h('div.ws-field', h('label', 'USD per credit'), input, h('div.ws-help', `Server default: ${fmt.usd(creditPrice())} (WAREHOUSE_STUDIO_CREDIT_PRICE_USD).`)),
      actions: [
        { label: 'Use server price', onClick: (close) => { prefs.set(cc.PRICE_KEY, null); paint(); onChange(); close(); } },
        { label: 'Save', primary: true, onClick: (close) => {
          const v = Number(input.value);
          if (!(v > 0)) throw new Error('Enter a price above zero.');
          prefs.set(cc.PRICE_KEY, v); paint(); onChange(); close();
        } },
      ] });
  });
  btn.repaint = paint;
  return btn;
}

/** Disposable chart scope so re-rendered tabs don't leak ECharts instances. */
function makeScope() {
  const fns = [];
  return { cleanup: (fn) => fns.push(fn), dispose: () => fns.splice(0).forEach((f) => { try { f(); } catch (e) { console.error(e); } }) };
}

export function panel(x, build) {
  if (cc.isError(x)) return x.unavailable ? toolkitBanner(x.error) : banner('error', 'Could not load this panel', x.error);
  return build(x);
}

export default async function render(ctx) {
  ctx.setTitle('Cost Center');
  let days = PERIODS.includes(Number(ctx.query.get('days'))) ? Number(ctx.query.get('days')) : 30;
  let tab = TABS.some((t) => t.key === ctx.query.get('tab')) ? ctx.query.get('tab') : 'overview';
  let dim = DIMS.some((d) => d.value === ctx.query.get('dim')) ? ctx.query.get('dim') : 'user';
  let by = BY.some((b) => b.value === ctx.query.get('by')) ? ctx.query.get('by') : 'credits';
  const whColors = new charts.ColorScale();
  const svcColors = new charts.ColorScale();
  const cache = new Map();
  let kpiScope = makeScope();
  let bodyScope = makeScope();
  ctx.cleanup(() => { kpiScope.dispose(); bodyScope.dispose(); });
  let kgen = 0;
  let gen = 0;

  const load = (key, fn) => {
    const k = `${key}:${days}`;
    if (!cache.has(k)) {
      const p = fn().catch((err) => { cache.delete(k); throw err; });
      cache.set(k, p);
    }
    return cache.get(k);
  };
  const overview = () => load('overview', () => api.costOverview(days, { signal: ctx.signal }));

  const priceBtn = priceButton(() => { paintKpis(); paintBody(); });
  const periodSeg = seg(PERIODS.map((d) => ({ value: d, label: `${d} days` })), days, (v) => {
    days = Number(v); ctx.setQuery({ days }); paintKpis(); paintBody();
  });
  const refreshBtn = h('button.ws-btn.ws-btn-sm', { type: 'button', title: 'Clear server caches and reload', onclick: async () => {
    refreshBtn.disabled = true;
    try { await api.costRefresh(); cache.clear(); paintKpis(); paintBody(); toast('Fresh numbers loading from Snowflake.', 'success'); } catch (e) { toast(e.message, 'error'); } finally { refreshBtn.disabled = false; }
  } }, icon('arrow-clockwise'), 'Refresh');
  const tabsEl = tabs(TABS, tab, (t) => { tab = t; ctx.setQuery({ tab: t === 'overview' ? null : t }); paintBody(); });
  const kpiEl = h('div');
  const bodyEl = h('div');

  mount(ctx.el,
    h('div.ws-toolbar.wc-toolbar', periodSeg, h('div.ws-grow'), priceBtn, refreshBtn),
    kpiEl, h('div.ws-mt'), tabsEl, bodyEl);

  paintKpis();
  paintBody();

  // ------------------------------------------------------------------ KPIs
  async function paintKpis() {
    kpiScope.dispose(); kpiScope = makeScope();
    const my = ++kgen;
    mount(kpiEl, h('div.wc-kpis', [...Array(5)].map(() => h('div.ws-skel', { style: { height: '128px', borderRadius: '16px' } }))));
    let o;
    try { o = await overview(); } catch (err) {
      if (err.name === 'AbortError') return;
      mount(kpiEl, errorView(err, () => { cache.clear(); paintKpis(); }));
      return;
    }
    if (my !== kgen || !ctx.active) return;
    const price = priceNow();
    const daily = cc.dailyTotals(cc.listOr(o.daily_by_service));
    const f = o.forecast || {};
    const q = cc.listOr(o.queries);
    const totalQ = q.reduce((a, r) => a + fmt.num(r.queries), 0);
    const failedQ = q.reduce((a, r) => a + fmt.num(r.failed), 0);
    const sparkSpend = h('div.ws-stat-spark');
    const sparkQ = h('div.ws-stat-spark');
    const proj = fmt.num(f.projected_month);
    const last = fmt.num(f.last_month);
    mount(kpiEl, h('div.wc-kpis',
      stat({ label: `Spend · last ${days} days`, icon: 'cash-stack', value: fmt.usd(cc.toUsd(o.total_credits, price)),
        foot: [h('span.ws-mono', `${fmt.credits(o.total_credits)} credits`), h('span.ws-muted', `@ ${fmt.usd(price)}/credit`)], sparkEl: sparkSpend }),
      stat({ label: 'Month to date', icon: 'calendar3', value: fmt.usd(cc.toUsd(f.month_to_date, price)),
        foot: [delta(f.pace_vs_last_month, { goodWhen: 'down', suffix: 'vs same day last month' }) || h('span.ws-muted', 'no comparison yet'),
          h('span.ws-muted', `day ${f.day_of_month || '—'} of ${f.days_in_month || '—'}`)] }),
      stat({ label: 'Projected month-end', icon: 'graph-up-arrow', value: fmt.usd(cc.toUsd(proj, price)),
        foot: h('div.ws-col', { style: { gap: '6px', width: '100%' } },
          h('div.ws-row.ws-between', h('span.ws-muted', `last month ${fmt.usd(cc.toUsd(last, price))}`),
            last ? h('b.ws-mono', fmt.pct(proj / last)) : null),
          last ? h('div.wc-progress', { class: proj > last ? 'over' : '' }, h('span', { style: { width: `${Math.min(100, (proj / last) * 100).toFixed(1)}%` } })) : null,
          h('span.ws-muted', `run-rate ${fmt.usd(cc.toUsd(f.run_rate_per_day, price))}/day`)) }),
      stat({ label: 'Storage', icon: 'hdd-stack', value: fmt.usd(o.storage_monthly_usd), unit: '/mo',
        foot: [h('span.ws-mono', `${fmt.fmtNum(o.storage_tb * 1024, 1)} GB`), h('span.ws-muted', `@ ${fmt.usd(o.storage_price_per_tb_month_usd)}/TB-mo`)] }),
      cc.isError(o.queries)
        ? stat({ label: 'Queries', icon: 'braces', value: '—', foot: h('span.ws-muted', o.queries.unavailable ? 'Needs the toolkit or ACCOUNT_USAGE access' : o.queries.error) })
        : stat({ label: `Queries · last ${Math.min(days, 30)} days`, icon: 'braces', value: fmt.compact(totalQ),
          foot: [h('span', { class: failedQ / (totalQ || 1) > 0.05 ? 'wc-bad' : '' }, `${fmt.pct(failedQ / (totalQ || 1), 1)} failed`),
            h('span.ws-muted', `${fmt.compact(failedQ)} errors`)], sparkEl: sparkQ })));
    if (daily.length > 1) {
      charts.spark(sparkSpend, { values: daily.map((d) => d.credits * price), labels: daily.map((d) => d.day), valueFmt: fmt.usd, ctx: kpiScope });
    }
    if (q.length > 1) {
      charts.spark(sparkQ, { values: q.map((r) => fmt.num(r.queries)), labels: q.map((r) => r.day), valueFmt: fmt.fmtInt, ctx: kpiScope, color: 'var(--ws-c7)' });
    }
  }

  // ------------------------------------------------------------------ body
  async function paintBody() {
    bodyScope.dispose(); bodyScope = makeScope();
    const my = ++gen;
    tabsEl.set(tab);
    mount(bodyEl, skeletonCards(4, 260));
    try {
      const builders = { overview: overviewTab, warehouses: warehousesTab, attribution: attributionTab, queries: queriesTab, storage: storageTab };
      await builders[tab](my);
    } catch (err) {
      if (err.name === 'AbortError' || my !== gen) return;
      mount(bodyEl, errorView(err, () => { cache.clear(); paintBody(); }));
    }
  }
  const stale = (my) => my !== gen || !ctx.active;

  // ---------------------------------------------------------------- overview
  async function overviewTab(my) {
    const [o, heat, mon] = await Promise.all([
      overview(),
      load('heatmap', () => api.costHeatmap(Math.min(days, 90), { signal: ctx.signal })).catch((e) => ({ error: e.message, unavailable: e.code === 'toolkit' })),
      load('monitors', () => api.costMonitors({ signal: ctx.signal })).catch((e) => ({ error: e.message, unavailable: e.code === 'toolkit' })),
    ]);
    if (stale(my)) return;
    const price = priceNow();
    const usdFmt = (v) => `${fmt.usd(v * price)} · ${fmt.credits(v)} cr`;
    const els = { svc: h('div.ws-chart.ws-chart-lg'), mix: h('div.ws-chart.ws-chart-lg'), burn: h('div.ws-chart.ws-chart-lg'),
      wh: h('div.ws-chart.ws-chart-lg'), heat: h('div.ws-chart', { style: { height: '300px' } }), qv: h('div.ws-chart.ws-chart-sm') };
    const later = [];

    const svc = panel(o.daily_by_service, (rows) => {
      const p = cc.pivotDaily(rows, (r) => r.service_type, 'credits', cc.serviceLabel);
      if (!p.rows.length) return empty({ icon: 'graph-up', title: 'No metered usage in this window' });
      p.keys.forEach((k) => svcColors.get(k));
      const tt = charts.tableToggle([{ key: 'name', label: 'Service' }, { key: 'credits', label: 'Credits', num: true, fmt: fmt.credits }, { key: 'usd', label: 'USD', num: true, fmt: fmt.usd }],
        p.keys.map((k) => ({ name: k, credits: p.totals.get(k), usd: p.totals.get(k) * price })));
      later.push(() => charts.stackedTime(els.svc, { rows: p.rows, keys: p.keys, kind: 'bar', colors: svcColors, valueFmt: usdFmt, ctx: bodyScope }));
      return h('div', els.svc, tt.button, tt.table);
    });

    const mix = panel(o.daily_by_service, (rows) => {
      const p = cc.pivotDaily(rows, (r) => r.service_type, 'credits', cc.serviceLabel);
      const items = cc.foldTopN(p.keys.map((k) => ({ name: k, value: p.totals.get(k) })), 5);
      if (!items.length) return empty({ icon: 'pie-chart', title: 'Nothing billed' });
      later.push(() => charts.donut(els.mix, { items: items.map((i) => ({ ...i, color: i.other ? charts.tokens().muted : svcColors.get(i.name) })),
        valueFmt: (v) => fmt.usd(v * price), center: fmt.usd(cc.toUsd(o.total_credits, price)), ctx: bodyScope }));
      return els.mix;
    });

    const burnInfo = panel(o.month_daily, (rows) => {
      const b = cc.monthBurnup(rows, new Date(), { projected: o.forecast?.projected_month });
      later.push(() => burnupChart(els.burn, b, price, bodyScope));
      return h('div',
        h('div.wc-burn-stats',
          miniFig('Month to date', fmt.usd(b.mtd * price)),
          miniFig('Run-rate', `${fmt.usd((o.forecast?.run_rate_per_day ?? b.rate) * price)}/day`),
          miniFig('Projected', fmt.usd(b.projected * price), b.lastMonthTotal && b.projected > b.lastMonthTotal ? 'bad' : ''),
          miniFig('Last month', fmt.usd(b.lastMonthTotal * price))),
        els.burn,
        (() => {
          const tt = charts.tableToggle([{ key: 'x', label: 'Day' }, { key: 'y', label: 'Cumulative USD', num: true, fmt: (v) => fmt.usd(v * price) }], b.thisMonth);
          return h('div', tt.button, tt.table);
        })());
    });

    const wh = panel(o.daily_by_warehouse, (rows) => {
      const p = cc.pivotDaily(rows, (r) => r.warehouse_name, 'credits');
      if (!p.rows.length) return empty({ icon: 'cpu', title: 'No warehouse usage' });
      p.keys.forEach((k) => whColors.get(k));
      const tt = charts.tableToggle([{ key: 'name', label: 'Warehouse' }, { key: 'credits', label: 'Credits', num: true, fmt: fmt.credits }, { key: 'usd', label: 'USD', num: true, fmt: fmt.usd }],
        p.keys.map((k) => ({ name: k, credits: p.totals.get(k), usd: p.totals.get(k) * price })));
      later.push(() => charts.stackedTime(els.wh, { rows: p.rows, keys: p.keys, kind: 'area', colors: whColors, valueFmt: usdFmt, ctx: bodyScope }));
      return h('div', els.wh, tt.button, tt.table);
    });

    const heatEl = panel(heat, (hm) => {
      const cells = cc.listOr(hm.cells).map((c) => ({ dow: c.dow, hour: c.hour, value: fmt.num(c.credits) }));
      if (!cells.length) return empty({ icon: 'grid-3x3', title: 'No hourly usage' });
      const ins = cc.heatmapInsights(cells);
      later.push(() => charts.weekHeatmap(els.heat, { cells, valueFmt: usdFmt, ctx: bodyScope }));
      const tt = charts.tableToggle([{ key: 'd', label: 'Day' }, { key: 'v', label: 'Credits', num: true, fmt: fmt.credits }],
        ins.byDow.map((v, i) => ({ d: cc.DOW[i], v })));
      return h('div',
        h('div.wc-insights',
          ins.peak && insightTag('fire', `Peak: ${cc.DOW[ins.peak.dow - 1]} ${String(ins.peak.hour).padStart(2, '0')}:00 UTC`, `${fmt.usd(ins.peak.value * price)}`),
          insightTag('briefcase', 'Business hours', fmt.pct(ins.businessShare)),
          insightTag('moon-stars', 'Nights (20–06)', fmt.pct(ins.nightShare)),
          insightTag('calendar-week', 'Weekends', fmt.pct(ins.weekendShare))),
        els.heat, tt.button, tt.table,
        ins.nightShare + ins.weekendShare > 0.5 ? h('p.wc-note', icon('lightbulb'), ' Over half the spend happens outside business hours — check scheduled jobs and tasks; shorter auto-suspend helps most here.') : null);
    });

    const monEl = panel(mon, (m) => monitorsView(cc.listOr(m.monitors), o, price));

    const qv = panel(o.queries, (rows) => {
      if (!rows.length) return empty({ icon: 'braces', title: 'No queries' });
      const t = charts.tokens();
      const colors = { get: (k) => (k === 'Failed' ? charts.STATUS.CRITICAL : t.series[0]) };
      const flat = rows.flatMap((r) => [{ x: String(r.day).slice(0, 10), key: 'Succeeded', value: fmt.num(r.queries) - fmt.num(r.failed) },
        { x: String(r.day).slice(0, 10), key: 'Failed', value: fmt.num(r.failed) }]);
      later.push(() => charts.stackedTime(els.qv, { rows: flat, keys: ['Succeeded', 'Failed'], kind: 'bar', colors, valueFmt: fmt.fmtInt, ctx: bodyScope }));
      const tt = charts.tableToggle([{ key: 'day', label: 'Day', fmt: fmt.day }, { key: 'queries', label: 'Queries', num: true, fmt: fmt.fmtInt },
        { key: 'failed', label: 'Failed', num: true, fmt: fmt.fmtInt }, { key: 'p95_ms', label: 'p95', num: true, fmt: fmt.ms }], rows);
      return h('div', els.qv, tt.button, tt.table);
    });

    mount(bodyEl, h('div.ws-grid.ws-g-12',
      card({ title: 'Daily credits by service', icon: 'bar-chart', sub: 'What Snowflake billed, per day', cls: 'ws-span-8', body: svc }),
      card({ title: 'Spend mix', icon: 'pie-chart', sub: `Last ${days} days`, cls: 'ws-span-4', body: mix }),
      card({ title: 'Month burn-up', icon: 'graph-up-arrow', sub: 'Cumulative spend vs last month, dashed = forecast at the 7-day run-rate', cls: 'ws-span-7', glow: true, body: burnInfo }),
      card({ title: 'Daily credits by warehouse', icon: 'cpu', sub: 'Colour follows the warehouse on every tab', cls: 'ws-span-5', body: wh }),
      card({ title: 'When the account spends', icon: 'grid-3x3-gap', sub: `Credits by weekday × hour (UTC), last ${Math.min(days, 90)} days`, cls: 'ws-span-7', body: heatEl }),
      card({ title: 'Resource monitors', icon: 'shield-check', sub: 'Quota used vs notify / suspend thresholds', cls: 'ws-span-5', body: monEl }),
      card({ title: 'Query volume', icon: 'braces', sub: `Succeeded vs failed per day (last ${Math.min(days, 30)} days)`, cls: 'ws-span-12', body: qv })));
    later.forEach((fn) => fn());
  }

  // ---------------------------------------------------------------- warehouses
  async function warehousesTab(my) {
    const data = await load('warehouses', () => api.costWarehouses(Math.min(days, 30), { signal: ctx.signal }));
    if (stale(my)) return;
    const price = priceNow();
    const span = data.days;
    const list = (data.warehouses || []).map((w) => {
      const recs = cc.recommendations(w, { price, days: span });
      return { ...w, recs, potential_usd_month: recs.reduce((a, r) => a + r.saveUsdMonth, 0), total_credits: fmt.num(w.compute_credits) + fmt.num(w.cloud_credits) };
    });
    list.forEach((w) => whColors.get(w.name));
    const totalCredits = list.reduce((a, w) => a + w.total_credits, 0);
    const idle = list.reduce((a, w) => a + fmt.num(w.idle_hour_credits), 0);
    const savings = list.reduce((a, w) => a + w.potential_usd_month, 0);
    const queued = list.reduce((a, w) => a + fmt.num(w.queued_seconds), 0);
    const splitEl = h('div.ws-chart', { style: { height: `${Math.max(160, list.length * 34 + 40)}px` } });
    const errs = Object.entries(data.errors || {});
    const tbl = table([
      { key: 'name', label: 'Warehouse', render: (w) => h('span.ws-row', { style: { gap: '8px' } }, h('i.wc-swatch', { style: { background: whColors.get(w.name) } }), stateDot(w.settings?.state), h('b', w.name)) },
      { key: 'size', label: 'Size', sort: (w) => fmt.creditsPerHour(w.settings?.size), render: (w) => h('span.ws-tag', w.settings?.size || '—') },
      { key: 'total_credits', label: 'Credits', num: true, render: (w) => fmt.credits(w.total_credits) },
      { key: 'usd', label: 'Spend', num: true, sort: (w) => w.total_credits, render: (w) => h('b', fmt.usd(cc.toUsd(w.total_credits, price))) },
      { key: 'trend', label: 'Trend', sort: false, render: (w) => svgSpark((w.daily || []).map((d) => d.credits), { color: whColors.get(w.name), width: 110, label: `${w.name} daily credits` }) },
      { key: 'active_hours', label: 'Active h', num: true, render: (w) => fmt.fmtInt(w.active_hours) },
      { key: 'utilization', label: 'Busy', num: true, render: (w) => (w.utilization == null ? '—' : h('div.wc-util', bar(Math.min(1, w.utilization)), h('span', fmt.pct(w.utilization)))) },
      { key: 'idle_hour_credits', label: 'Idle burn', num: true, render: (w) => h('span', { class: fmt.num(w.idle_hour_credits) > w.total_credits * 0.2 ? 'wc-hot' : '' }, fmt.usd(cc.toUsd(w.idle_hour_credits, price))) },
      { key: 'queued_seconds', label: 'Queued', num: true, render: (w) => fmt.seconds(w.queued_seconds) },
      { key: 'remote_spill_bytes', label: 'Remote spill', num: true, render: (w) => (fmt.num(w.remote_spill_bytes) ? h('span.wc-hot', fmt.bytes(w.remote_spill_bytes)) : '—') },
      { key: 'auto', label: 'Auto-suspend', sort: (w) => w.settings?.auto_suspend ?? 1e9, render: (w) => (w.settings && 'auto_suspend' in w.settings ? h('span.ws-tag', { class: w.settings.auto_suspend == null || w.settings.auto_suspend > 60 ? 'wc-tag-warn' : '' }, w.settings.auto_suspend == null ? 'never' : `${w.settings.auto_suspend}s`) : '—') },
      { key: 'potential_usd_month', label: 'Could save', num: true, render: (w) => (w.potential_usd_month > 0.005 ? h('span.wc-save', `${fmt.usd(w.potential_usd_month)}/mo`) : '—') },
    ], list, { sortKey: 'total_credits', onRow: (w) => openWarehouse(w, { days: span, price, color: whColors.get(w.name) }), empty: empty({ icon: 'hdd-rack', title: 'No warehouses visible to this role' }) });
    mount(bodyEl,
      days > 30 ? banner('info', 'Warehouse analytics cover the last 30 days', 'Hourly utilisation comes from 30-day query history; the other tabs use the full period.') : null,
      errs.length ? banner('warn', 'Some warehouse data is missing', errs.map(([k, v]) => `${k}: ${v}`).join(' · ')) : null,
      h('div.ws-grid.ws-g-4.ws-mb',
        stat({ label: 'Warehouse spend', icon: 'cpu', value: fmt.usd(cc.toUsd(totalCredits, price)), foot: `${fmt.credits(totalCredits)} credits · ${span} days` }),
        stat({ label: 'Idle burn', icon: 'fire', value: fmt.usd(cc.toUsd(idle, price)), foot: `${fmt.pct(idle / (totalCredits || 1))} of warehouse credits in near-idle hours` }),
        stat({ label: 'Could save', icon: 'piggy-bank', value: fmt.usd(savings), unit: '/mo', foot: 'Sum of the recommendations below', cls: 'wc-stat-good' }),
        stat({ label: 'Queued time', icon: 'stoplights', value: fmt.seconds(queued), foot: 'Waiting for a slot (concurrency)' })),
      card({ title: 'Busy vs idle credits', icon: 'battery-half', sub: 'Idle = credits billed in hours with under a minute of execution', body: list.length ? splitEl : empty({ title: 'No usage' }) }),
      h('div.ws-mt'),
      card({ title: 'Warehouses', icon: 'hdd-rack', sub: 'Click a warehouse for daily spend, load, recommendations and the what-if simulator', body: tbl.el }));
    if (list.length) {
      charts.chart(splitEl, (t) => ({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(128,160,255,.06)' } }, backgroundColor: t.surface2, borderColor: t.border, textStyle: { color: t.text, fontSize: 12 },
          formatter: (ps) => `<b>${charts.escapeHtml(ps[0].name)}</b>${ps.map((p) => charts.tipRow(p.color, p.seriesName, `${fmt.usd(p.value * price)} · ${fmt.credits(p.value)} cr`)).join('')}` },
        legend: { top: 0, left: 0, icon: 'roundRect', itemWidth: 10, itemHeight: 10, textStyle: { color: t.text2, fontSize: 11 } },
        grid: { left: 8, right: 24, top: 30, bottom: 6, containLabel: true },
        xAxis: { type: 'value', ...charts.axisStyle(t, { axisLabel: { color: t.muted, fontSize: 11, formatter: (v) => fmt.usd(v * price, { cents: false }) } }) },
        yAxis: { type: 'category', inverse: true, data: list.map((w) => w.name), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.text2, fontSize: 12 } },
        series: [
          { name: 'Busy', type: 'bar', stack: 'c', barMaxWidth: 18, data: list.map((w) => Math.max(0, fmt.num(w.compute_credits) - fmt.num(w.idle_hour_credits))), itemStyle: { color: t.series[0], borderColor: t.surface, borderWidth: 1 } },
          { name: 'Idle burn', type: 'bar', stack: 'c', barMaxWidth: 18, data: list.map((w) => fmt.num(w.idle_hour_credits)), itemStyle: { color: t.series[1], borderColor: t.surface, borderWidth: 1, borderRadius: [0, 4, 4, 0] } },
        ],
      }), bodyScope);
    }
  }

  // ---------------------------------------------------------------- attribution
  async function attributionTab(my) {
    const data = await load('attribution', () => api.costAttribution(Math.min(days, 30), { signal: ctx.signal }));
    if (stale(my)) return;
    const price = priceNow();
    const span = data.days;
    const out = h('div');
    const dimSeg = seg(DIMS, dim, (v) => { dim = v; ctx.setQuery({ dim: v === 'user' ? null : v }); paint(); });
    mount(bodyEl,
      days > 30 ? banner('info', 'Attribution covers the last 30 days', 'Per-query credit attribution is read from 30 days of history.') : null,
      h('div.ws-toolbar', dimSeg, h('span.ws-muted', `Compute credits attributed to individual queries · last ${span} days`)), out);
    const scope = { list: [] };
    paint();

    function paint() {
      bodyScope.dispose(); bodyScope = makeScope();
      const items = cc.listOr(data[dim]);
      scope.list = items;
      const total = items.reduce((a, i) => a + fmt.num(i.credits), 0);
      const label = DIMS.find((d) => d.value === dim).label;
      if (!items.length) { mount(out, empty({ icon: 'people', title: `No attributed spend by ${label.toLowerCase()}` })); return; }
      const top = items.slice(0, 15);
      const isWh = dim === 'warehouse';
      const barsEl = h('div.ws-chart', { style: { height: `${Math.max(180, top.length * 30 + 20)}px` } });
      const donutEl = h('div.ws-chart.ws-chart-lg');
      const filterKey = { user: 'user', role: 'role', warehouse: 'warehouse', database: 'database' }[dim];
      const go = (name) => { if (filterKey && name !== '(none)') ctx.navigate(`/history?scope=account&hours=${span * 24}&${filterKey}=${encodeURIComponent(name)}`); };
      const tbl = table([
        { key: 'name', label: label.replace(/s$/, ''), render: (r) => h('span.ws-row', { style: { gap: '8px' } }, isWh && h('i.wc-swatch', { style: { background: whColors.get(r.name) } }), h('b', r.name)) },
        { key: 'credits', label: 'Credits', num: true, render: (r) => fmt.credits(r.credits) },
        { key: 'usd', label: 'Spend', num: true, sort: (r) => r.credits, render: (r) => h('b', fmt.usd(cc.toUsd(r.credits, price))) },
        { key: 'share', label: 'Share', sort: (r) => r.credits, render: (r) => h('div.wc-util', bar(fmt.num(r.credits) / (total || 1)), h('span', fmt.pct(fmt.num(r.credits) / (total || 1), 1))) },
        { key: 'queries', label: 'Queries', num: true, render: (r) => fmt.fmtInt(r.queries) },
        { key: 'exec_seconds', label: 'Exec time', num: true, render: (r) => fmt.seconds(r.exec_seconds) },
        { key: 'per_query', label: 'Per query', num: true, sort: (r) => fmt.num(r.credits) / (r.queries || 1), render: (r) => fmt.usd(cc.toUsd(fmt.num(r.credits) / (r.queries || 1), price)) },
      ], items, { sortKey: 'credits', onRow: filterKey ? (r) => go(r.name) : null });
      mount(out, h('div.ws-grid.ws-g-12',
        card({ title: `Top ${label.toLowerCase()} by spend`, icon: 'bar-chart-steps', cls: 'ws-span-7', sub: filterKey ? 'Click a bar to see their queries' : null, body: barsEl }),
        card({ title: 'Share of attributed spend', icon: 'pie-chart', cls: 'ws-span-5', sub: `${fmt.usd(cc.toUsd(total, price))} attributed`, body: donutEl }),
        card({ title: `All ${label.toLowerCase()}`, icon: 'table', cls: 'ws-span-12', body: tbl.el })));
      charts.hbars(barsEl, { items: top.map((i) => ({ name: i.name, value: fmt.num(i.credits) * price, color: isWh ? whColors.get(i.name) : undefined })),
        valueFmt: fmt.usd, ctx: bodyScope, onClick: (i) => go(i.name), labelWidth: 170 });
      const folded = cc.foldTopN(items.map((i) => ({ name: i.name, value: fmt.num(i.credits) })), 6);
      const t = charts.tokens();
      charts.donut(donutEl, { items: folded.map((i, k) => ({ ...i, color: i.other ? t.muted : isWh ? whColors.get(i.name) : t.series[k % 8] })),
        valueFmt: (v) => fmt.usd(v * price), center: fmt.usd(cc.toUsd(total, price)), ctx: bodyScope });
    }
  }

  // ---------------------------------------------------------------- expensive queries
  async function queriesTab(my) {
    const out = h('div');
    const bySeg = seg(BY, by, (v) => { by = v; ctx.setQuery({ by: v === 'credits' ? null : v }); paint(); });
    mount(bodyEl,
      h('div.ws-toolbar', h('span.ws-text-2', 'Rank query shapes by'), bySeg, h('span.ws-muted', 'A shape = the same SQL with different literals; one fix pays off on every run.')),
      out);
    await paint();

    async function paint() {
      bodyScope.dispose(); bodyScope = makeScope();
      mount(out, skeleton(6, 26));
      const span = Math.min(days, 30);
      let data;
      try {
        data = await load(`top:${by}`, () => api.costTopQueries(span, by, 60, { signal: ctx.signal }));
      } catch (err) {
        if (err.name === 'AbortError') return;
        mount(out, errorView(err, paint));
        return;
      }
      if (!ctx.active) return;
      const price = priceNow();
      const rows = cc.listOr(data.queries);
      if (!rows.length) { mount(out, empty({ icon: 'lightning', title: 'No query shapes in this window' })); return; }
      const scatterEl = h('div.ws-chart.ws-chart-lg');
      const tbl = table([
        { key: 'sample_text', label: 'Query shape', sort: false, render: (r) => h('div.ws-col', { style: { gap: '3px', minWidth: 0, maxWidth: '520px' } },
          h('span.ws-sql-snippet', { title: r.sample_text }, fmt.oneLine(r.sample_text, 160)),
          h('span.ws-row', { style: { gap: '6px' } }, h('span.ws-tag', r.query_type || '—'), r.warehouse_name && h('span.ws-tag', r.warehouse_name),
            fmt.num(r.failed) ? h('span.ws-badge.ws-sev-HIGH', `${fmt.fmtInt(r.failed)} failed`) : null)) },
        { key: 'executions', label: 'Runs', num: true, render: (r) => fmt.fmtInt(r.executions) },
        { key: 'credits', label: 'Spend', num: true, render: (r) => h('b', fmt.usd(cc.toUsd(r.credits, price))) },
        { key: 'avg_elapsed_ms', label: 'Avg', num: true, render: (r) => fmt.ms(r.avg_elapsed_ms) },
        { key: 'p95_elapsed_ms', label: 'p95', num: true, render: (r) => fmt.ms(r.p95_elapsed_ms) },
        { key: 'total_elapsed_ms', label: 'Total time', num: true, render: (r) => fmt.ms(r.total_elapsed_ms) },
        { key: 'bytes_scanned', label: 'Scanned', num: true, render: (r) => fmt.bytes(r.bytes_scanned) },
        { key: 'spilled_bytes', label: 'Spill', num: true, render: (r) => (fmt.num(r.spilled_bytes) ? h('span.wc-hot', fmt.bytes(r.spilled_bytes)) : '—') },
        { key: 'users', label: 'Users', sort: (r) => (r.users || []).length, render: (r) => h('span.ws-ellipsis', { style: { maxWidth: '160px' }, title: (r.users || []).join(', ') }, (r.users || []).slice(0, 2).join(', ') + ((r.users || []).length > 2 ? ` +${r.users.length - 2}` : '')) },
        { key: 'last_run', label: 'Last run', render: (r) => fmt.relTime(r.last_run) },
        { key: 'actions', label: '', sort: false, render: (r) => h('span.ws-row', { style: { gap: '4px' } },
          h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', title: 'Inspect the slowest run', onclick: (e) => { e.stopPropagation(); openQuery(r.slowest_query_id); } }, icon('search')),
          h('a.ws-btn.ws-btn-sm.ws-btn-ghost', { href: href('/worksheets', { query: r.slowest_query_id }), title: 'Open in a worksheet', onclick: (e) => e.stopPropagation() }, icon('code-square'))) },
      ], rows, { sortKey: { credits: 'credits', elapsed: 'total_elapsed_ms', executions: 'executions', bytes: 'bytes_scanned', spill: 'spilled_bytes' }[by], onRow: (r) => openQuery(r.slowest_query_id) });
      const totalUsd = rows.reduce((a, r) => a + cc.toUsd(r.credits, price), 0);
      mount(out, h('div.ws-grid.ws-g-12',
        card({ title: 'Shape map', icon: 'bullseye', cls: 'ws-span-12', glow: true,
          sub: 'Runs × average duration (log scales); bubble size = spend. Top-right bubbles are the best targets — click one to inspect it.',
          tools: h('span.ws-tag', `top ${rows.length} shapes · ${fmt.usd(totalUsd)}`), body: scatterEl }),
        card({ title: 'Most expensive query shapes', icon: 'list-ol', cls: 'ws-span-12', sub: `Last ${span} days`, body: tbl.el })));
      const maxUsd = Math.max(0.0001, ...rows.map((r) => cc.toUsd(r.credits, price)));
      const points = rows.filter((r) => fmt.num(r.executions) > 0 && fmt.num(r.avg_elapsed_ms) > 0).map((r) => ({
        x: fmt.num(r.executions), y: fmt.num(r.avg_elapsed_ms), size: 8 + 26 * Math.sqrt(cc.toUsd(r.credits, price) / maxUsd),
        label: `${fmt.oneLine(r.sample_text, 90)} — ${fmt.usd(cc.toUsd(r.credits, price))}`, row: r }));
      charts.scatter(scatterEl, { points, xName: 'Runs', yName: 'Avg duration', xFmt: fmt.compact, yFmt: fmt.ms, logX: true, logY: true,
        ctx: bodyScope, onClick: (p) => openQuery(p.row.slowest_query_id) });
    }
  }

  // ---------------------------------------------------------------- storage
  async function storageTab(my) {
    const data = await load('storage', () => api.costStorage(100, { signal: ctx.signal }));
    if (stale(my)) return;
    const price = data.storage_price_per_tb_month_usd;
    const GB = 1024 ** 3;
    const trendEl = h('div.ws-chart.ws-chart-lg');
    const dbEl = h('div.ws-chart.ws-chart-lg');
    const later = [];
    const keys = ['Tables', 'Stages', 'Fail-safe'];
    const colors = new charts.ColorScale(keys);
    const daily = cc.listOr(data.daily);
    const latest = daily[daily.length - 1];
    const latestTotal = latest ? fmt.num(latest.storage_bytes) + fmt.num(latest.stage_bytes) + fmt.num(latest.failsafe_bytes) : 0;
    const first = daily[0];
    const firstTotal = first ? fmt.num(first.storage_bytes) + fmt.num(first.stage_bytes) + fmt.num(first.failsafe_bytes) : 0;
    const tables = cc.listOr(data.tables);
    const flagged = tables.filter((t) => cc.storageFix(t).flagged);
    const overheadUsd = flagged.reduce((a, t) => a + fmt.num(t.overhead_monthly_usd), 0);

    const trend = panel(data.daily, (rows) => {
      if (!rows.length) return empty({ icon: 'hdd-stack', title: 'No storage history' });
      const flat = rows.flatMap((r) => [
        { x: String(r.day).slice(0, 10), key: 'Tables', value: fmt.num(r.storage_bytes) / GB },
        { x: String(r.day).slice(0, 10), key: 'Stages', value: fmt.num(r.stage_bytes) / GB },
        { x: String(r.day).slice(0, 10), key: 'Fail-safe', value: fmt.num(r.failsafe_bytes) / GB }]);
      later.push(() => charts.stackedTime(trendEl, { rows: flat, keys, kind: 'area', colors, valueFmt: (v) => `${fmt.fmtNum(v, 2)} GB`, ctx: bodyScope }));
      const tt = charts.tableToggle([{ key: 'day', label: 'Day', fmt: fmt.day }, { key: 'storage_bytes', label: 'Tables', num: true, fmt: fmt.bytes },
        { key: 'stage_bytes', label: 'Stages', num: true, fmt: fmt.bytes }, { key: 'failsafe_bytes', label: 'Fail-safe', num: true, fmt: fmt.bytes }], rows);
      return h('div', trendEl, tt.button, tt.table);
    });
    const dbs = panel(data.databases, (rows) => {
      if (!rows.length) return empty({ icon: 'database', title: 'No databases' });
      const items = rows.slice(0, 14).map((r) => ({ name: r.database_name, value: fmt.num(r.bytes) + fmt.num(r.failsafe_bytes) }));
      dbEl.style.height = `${Math.max(180, items.length * 30 + 20)}px`;
      later.push(() => charts.hbars(dbEl, { items, valueFmt: fmt.bytes, ctx: bodyScope, labelWidth: 170,
        onClick: (i) => ctx.navigate(`/data?database=${encodeURIComponent(i.name)}`) }));
      return dbEl;
    });
    const tbl = panel(data.tables, (rows) => table([
      { key: 'path_to_object', label: 'Table', render: (t) => h('div.ws-col', { style: { gap: '2px' } }, h('b.ws-mono', { style: { fontSize: '12px' } }, t.path_to_object),
        h('span.ws-row', { style: { gap: '6px' } }, t.is_transient === 'YES' || t.is_transient === true ? h('span.ws-tag', 'transient') : null)) },
      { key: 'active_bytes', label: 'Active', num: true, render: (t) => fmt.bytes(t.active_bytes) },
      { key: 'time_travel_bytes', label: 'Time travel', num: true, render: (t) => fmt.bytes(t.time_travel_bytes) },
      { key: 'failsafe_bytes', label: 'Fail-safe', num: true, render: (t) => fmt.bytes(t.failsafe_bytes) },
      { key: 'total_bytes', label: 'Total', num: true, render: (t) => h('b', fmt.bytes(t.total_bytes)) },
      { key: 'overhead_multiple', label: 'Overhead ×', num: true, render: (t) => (fmt.num(t.overhead_multiple) > 1
        ? h('span.ws-badge.ws-sev-MEDIUM', icon('exclamation-triangle-fill'), `${fmt.fmtNum(t.overhead_multiple, 1)}×`) : fmt.fmtNum(t.overhead_multiple, 2)) },
      { key: 'est_monthly_usd', label: '$/month', num: true, render: (t) => fmt.usd(t.est_monthly_usd) },
      { key: 'overhead_monthly_usd', label: 'Overhead $/mo', num: true, render: (t) => (fmt.num(t.overhead_monthly_usd) > 0.005 ? h('span.wc-hot', fmt.usd(t.overhead_monthly_usd)) : '—') },
      { key: 'fix', label: '', sort: false, render: (t) => (cc.storageFix(t).flagged ? h('button.ws-btn.ws-btn-sm', { type: 'button', onclick: (e) => { e.stopPropagation(); showFix(t); } }, icon('wrench'), 'Fix') : '') },
    ], rows, { sortKey: 'total_bytes', maxHeight: 560, rowCls: (t) => (cc.storageFix(t).flagged ? 'wc-row-flag' : '') }).el);

    mount(bodyEl,
      h('div.ws-grid.ws-g-4.ws-mb',
        stat({ label: 'Stored now', icon: 'hdd-stack', value: fmt.bytes(latestTotal), foot: firstTotal ? delta((latestTotal - firstTotal) / firstTotal, { goodWhen: 'down', suffix: 'over 90 days' }) : null }),
        stat({ label: 'Storage cost', icon: 'cash-coin', value: fmt.usd((latestTotal / 1024 ** 4) * price), unit: '/mo', foot: `@ ${fmt.usd(price)} per TB-month` }),
        stat({ label: 'Fail-safe share', icon: 'shield-lock', value: latest ? fmt.pct(fmt.num(latest.failsafe_bytes) / (latestTotal || 1)) : '—', foot: 'Not queryable, still billed (7 days)' }),
        stat({ label: 'Retention overhead', icon: 'exclamation-triangle', value: fmt.usd(overheadUsd), unit: '/mo', foot: `${fmt.plural(flagged.length, 'table')} where history outweighs the data` })),
      h('div.ws-grid.ws-g-12',
        card({ title: 'Storage over time', icon: 'graph-up', sub: 'Account storage, last 90 days', cls: 'ws-span-7', body: trend }),
        card({ title: 'By database', icon: 'database', sub: 'Including fail-safe · click to explore', cls: 'ws-span-5', body: dbs }),
        card({ title: 'Heaviest tables', icon: 'table', cls: 'ws-span-12', sub: 'Flagged rows carry more time-travel + fail-safe than live data — Fix shows the SQL (never run for you)', body: tbl })));
    later.forEach((fn) => fn());
  }

  function showFix(t) {
    const fix = cc.storageFix(t);
    modal({ title: `Reduce retention overhead`, sub: t.path_to_object, width: 680,
      body: h('div.ws-col', { style: { gap: '12px' } },
        h('p.ws-text-2', { style: { margin: 0 } }, `Active ${fmt.bytes(t.active_bytes)}, time travel ${fmt.bytes(t.time_travel_bytes)}, fail-safe ${fmt.bytes(t.failsafe_bytes)} — ${fmt.fmtNum(t.overhead_multiple, 1)}× overhead costing about ${fmt.usd(t.overhead_monthly_usd)}/month.`),
        h('div.ws-field', h('label', '1 · Shorten time travel (safe, immediate)'), h('div.wc-codebox', h('pre.ws-code', fix.retention), copyButton(fix.retention))),
        h('div.ws-field', h('label', '2 · Reloadable data only: switch to TRANSIENT (no fail-safe at all)'), h('div.wc-codebox', h('pre.ws-code', fix.transient), copyButton(fix.transient))),
        h('p.wc-note', icon('info-circle'), ' High churn (frequent full reloads, row-by-row DML) is what creates the overhead; batching writes reduces it too.')),
      actions: [{ label: 'Close' }] });
  }
}

// ------------------------------------------------------------------ helpers

function miniFig(label, value, cls = '') {
  return h(`div.wc-fig${cls ? `.${cls}` : ''}`, h('span', label), h('b', value));
}

function insightTag(ic, label, value) {
  return h('span.wc-insight', icon(ic), h('span', label), h('b', value));
}

function burnupChart(el, b, price, ctx) {
  const n = Math.max(b.daysInMonth, b.lastMonth.length);
  const xs = [...Array(n)].map((_, i) => i + 1);
  const at = (series) => { const m = new Map(series.map((p) => [p.x, p.y * price])); return xs.map((x) => (m.has(x) ? m.get(x) : null)); };
  return charts.chart(el, (t) => ({
    tooltip: { trigger: 'axis', backgroundColor: t.surface2, borderColor: t.border, textStyle: { color: t.text, fontSize: 12 },
      extraCssText: 'border-radius:12px;box-shadow:0 12px 32px rgba(0,0,0,.35);padding:10px 12px;',
      formatter: (ps) => `<div style="font-weight:600;margin-bottom:4px">Day ${ps[0].axisValue}</div>`
        + ps.filter((p) => p.value != null).map((p) => charts.tipRow(p.color, p.seriesName, fmt.usd(p.value))).join('') },
    legend: { top: 0, left: 0, icon: 'roundRect', itemWidth: 10, itemHeight: 10, textStyle: { color: t.text2, fontSize: 11 } },
    grid: { left: 8, right: 16, top: 34, bottom: 8, containLabel: true },
    xAxis: { type: 'category', data: xs, boundaryGap: false, ...charts.axisStyle(t, { splitLine: { show: false } }) },
    yAxis: { type: 'value', ...charts.axisStyle(t, { axisLabel: { color: t.muted, fontSize: 11, formatter: (v) => fmt.usd(v, { cents: false }) } }) },
    series: [
      { name: 'Last month', type: 'line', data: at(b.lastMonth), symbol: 'none', smooth: 0.2, lineStyle: { width: 2, color: t.muted }, itemStyle: { color: t.muted } },
      { name: 'This month', type: 'line', data: at(b.thisMonth), symbol: 'circle', symbolSize: 8, showSymbol: false, smooth: 0.2,
        lineStyle: { width: 2, color: t.series[0] }, itemStyle: { color: t.series[0], borderColor: t.surface, borderWidth: 2 },
        areaStyle: { color: new window.echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: charts.hexA(t.series[0], 0.3) }, { offset: 1, color: charts.hexA(t.series[0], 0) }]) } },
      { name: 'Forecast', type: 'line', data: at(b.forecast), symbol: 'none', lineStyle: { width: 2, type: [6, 5], color: t.series[0] }, itemStyle: { color: t.series[0] },
        markPoint: b.forecast.length > 1 ? { symbol: 'pin', symbolSize: 44, data: [{ coord: [b.forecast.at(-1).x - 1, b.forecast.at(-1).y * price], value: fmt.usd(b.projected * price, { cents: false }) }],
          label: { color: '#fff', fontSize: 10 }, itemStyle: { color: t.series[0] } } : undefined },
    ],
  }), ctx);
}

function monitorsView(monitors, overview, price) {
  if (!monitors.length) {
    const last = fmt.num(overview?.forecast?.last_month) || fmt.num(overview?.total_credits);
    const quota = Math.max(10, Math.ceil(last * 1.2 / 10) * 10);
    const sql = `CREATE RESOURCE MONITOR STUDIO_MONTHLY WITH CREDIT_QUOTA = ${quota}\n  FREQUENCY = MONTHLY START_TIMESTAMP = IMMEDIATELY\n  TRIGGERS ON 75 PERCENT DO NOTIFY\n           ON 100 PERCENT DO SUSPEND\n           ON 110 PERCENT DO SUSPEND_IMMEDIATE;\nALTER ACCOUNT SET RESOURCE_MONITOR = STUDIO_MONTHLY;`;
    return h('div.ws-col', { style: { gap: '10px' } },
      empty({ icon: 'shield-exclamation', title: 'No resource monitors', text: 'Nothing stops a runaway month. A monitor at ~120% of last month\'s credits is a sensible guard-rail:' }),
      h('div.wc-codebox', h('pre.ws-code', sql), copyButton(sql)));
  }
  const colors = { ok: '#0ca30c', warning: '#fab219', critical: '#d03b3b' };
  return h('div.wc-monitors', monitors.map((m) => {
    const st = cc.monitorStatus(m);
    return h('div.wc-monitor',
      ring(Math.min(100, st.usedPct), { size: 104, stroke: 10, caption: 'used', color: colors[st.level] }),
      h('div.ws-col', { style: { gap: '4px', minWidth: 0 } },
        h('b', m.name),
        h('span.ws-text-2', `${fmt.credits(st.used)} / ${fmt.credits(st.quota)} credits · ${fmt.usd(st.used * price)}`),
        h('span.ws-muted', `${String(m.frequency || '').toLowerCase()} · ${String(m.level || 'account').toLowerCase()} · resets ${m.end_time ? fmt.day(m.end_time) : 'per period'}`),
        h('span.ws-row.ws-wrap', { style: { gap: '4px' } },
          h('span.ws-badge', { class: `ws-sev-${st.level === 'critical' ? 'CRITICAL' : st.level === 'warning' ? 'MEDIUM' : 'INFO'}` },
            icon(st.level === 'ok' ? 'check2-circle' : 'exclamation-triangle-fill'), st.level === 'ok' ? 'within quota' : st.level),
          st.notify.map((p) => h('span.ws-tag', `notify ${p}%`)), st.suspend.map((p) => h('span.ws-tag', `suspend ${p}%`)))));
  }));
}
