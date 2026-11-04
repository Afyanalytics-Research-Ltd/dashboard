// Warehouses: the whole fleet at a glance — live state, spend, busy share,
// idle burn and ranked fixes per warehouse; click for the drawer + simulator.

import { h, icon, mount, seg, stat, ring, empty, banner, errorView, toolkitBanner, skeletonCards, sevBadge, toast } from '../lib/ui.js';
import * as charts from '../lib/charts.js';
import { api } from '../lib/api.js';
import { state } from '../lib/store.js';
import * as fmt from '../lib/fmt.js';
import * as cc from '../lib/costcore.js';
import { openWarehouse, svgSpark, stateDot } from '../lib/warehousepanel.js';
import { priceButton, priceNow } from './cost.js';

const PERIODS = [7, 14, 30];
const SORTS = [
  { value: 'credits', label: 'Spend' }, { value: 'idle', label: 'Idle burn' }, { value: 'savings', label: 'Savings' },
  { value: 'utilization', label: 'Busy' }, { value: 'name', label: 'Name' },
];
const UTIL_TEXT = { idle: 'mostly idle', low: 'lightly used', healthy: 'well used', saturated: 'saturated', unknown: 'no data' };

export default async function render(ctx) {
  ctx.setTitle('Warehouses');
  let days = PERIODS.includes(Number(ctx.query.get('days'))) ? Number(ctx.query.get('days')) : 30;
  const f = { state: ctx.query.get('state') || '', size: ctx.query.get('size') || '', q: ctx.query.get('q') || '' };
  let sort = SORTS.some((x) => x.value === ctx.query.get('sort')) ? ctx.query.get('sort') : 'credits';
  const colors = new charts.ColorScale();
  let data = null;
  let unavailable = null;

  const periodSeg = seg(PERIODS.map((d) => ({ value: d, label: `${d}d` })), days, (v) => { days = Number(v); ctx.setQuery({ days }); load(); });
  const stateSeg = seg([{ value: '', label: 'All' }, { value: 'STARTED', label: 'Running' }, { value: 'SUSPENDED', label: 'Suspended' }], f.state,
    (v) => { f.state = v; ctx.setQuery({ state: v }); paint(); });
  const sizeSel = h('select.ws-select', { 'aria-label': 'Size', onchange: () => { f.size = sizeSel.value; ctx.setQuery({ size: f.size }); paint(); } },
    h('option', { value: '' }, 'All sizes'), fmt.SIZE_ORDER.map((sz) => h('option', { value: sz, selected: sz === f.size }, sz)));
  const search = h('input.ws-input', { placeholder: 'Find a warehouse…', value: f.q, oninput: () => { f.q = search.value; paint(); } });
  const sortSeg = seg(SORTS, sort, (v) => { sort = v; ctx.setQuery({ sort: v === 'credits' ? null : v }); paint(); });
  const refreshBtn = h('button.ws-btn.ws-btn-sm', { type: 'button', onclick: async () => {
    try { await api.costRefresh(); toast('Reloading live warehouse state…', 'info'); } catch { /* still reload */ }
    load();
  } }, icon('arrow-clockwise'), 'Refresh');
  const summaryEl = h('div');
  const gridEl = h('div');
  mount(ctx.el,
    h('div.ws-toolbar', periodSeg, stateSeg, sizeSel, h('div.ws-search.ws-grow', icon('search'), search), h('span.ws-muted', 'Sort'), sortSeg,
      priceButton(() => paint()), refreshBtn),
    summaryEl, gridEl);
  load();

  async function load() {
    mount(summaryEl, '');
    mount(gridEl, skeletonCards(8, 260));
    unavailable = null;
    try {
      data = await api.costWarehouses(days, { signal: ctx.signal });
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (err.code !== 'toolkit') { mount(gridEl, errorView(err, load)); return; }
      // No telemetry access: still show the fleet's settings from the session.
      unavailable = err.message;
      data = { days, warehouses: (state.session?.warehouses || []).map((w) => ({ name: w.name, settings: { ...w }, daily: [] })), errors: {} };
    }
    if (!ctx.active) return;
    paint();
  }

  function paint() {
    if (!data) return;
    const price = priceNow();
    const span = data.days || days;
    const all = (data.warehouses || []).map((w) => {
      const recs = cc.recommendations(w, { price, days: span });
      return { ...w, recs, potential_usd_month: recs.reduce((a, r) => a + r.saveUsdMonth, 0) };
    });
    all.forEach((w) => colors.get(w.name));
    const list = cc.sortWarehouses(cc.filterWarehouses(all, f), sort);
    const credits = all.reduce((a, w) => a + fmt.num(w.compute_credits) + fmt.num(w.cloud_credits), 0);
    const idle = all.reduce((a, w) => a + fmt.num(w.idle_hour_credits), 0);
    const savings = all.reduce((a, w) => a + w.potential_usd_month, 0);
    const running = all.filter((w) => String(w.settings?.state).toUpperCase() === 'STARTED').length;
    mount(summaryEl,
      unavailable ? toolkitBanner(`${unavailable} Showing warehouse settings only.`) : null,
      Object.keys(data.errors || {}).length ? banner('warn', 'Some warehouse data is missing', Object.entries(data.errors).map(([k, v]) => `${k}: ${v}`).join(' · ')) : null,
      h('div.ws-grid.ws-g-4.ws-mb',
        stat({ label: 'Fleet', icon: 'hdd-rack', value: fmt.fmtInt(all.length), unit: 'warehouses', foot: [h('span.ws-dot.on'), `${running} running now`] }),
        stat({ label: `Spend · ${span} days`, icon: 'cash-stack', value: fmt.usd(cc.toUsd(credits, price)), foot: `${fmt.credits(credits)} credits` }),
        stat({ label: 'Idle burn', icon: 'fire', value: fmt.usd(cc.toUsd(idle, price)), foot: `${fmt.pct(idle / (credits || 1))} of credits billed while near-idle` }),
        stat({ label: 'Could save', icon: 'piggy-bank', value: fmt.usd(savings), unit: '/mo', cls: 'wc-stat-good', foot: 'If every recommendation below is applied' })));
    if (!list.length) {
      mount(gridEl, empty({ icon: 'hdd-rack', title: all.length ? 'No warehouse matches these filters' : 'No warehouses visible to this role' }));
      return;
    }
    mount(gridEl, h('div.wc-fleet', list.map((w) => fleetCard(w, { price, span }))));
  }

  function fleetCard(w, { price, span }) {
    const s = w.settings || {};
    const st = String(s.state || '').toUpperCase();
    const color = colors.get(w.name);
    const total = fmt.num(w.compute_credits) + fmt.num(w.cloud_credits);
    const util = w.utilization;
    const cls = cc.classifyUtilization(util);
    const daily = (w.daily || []).map((d) => fmt.num(d.credits));
    const open = () => openWarehouse(w, { days: span, price, color });
    return h('article.ws-card.wc-wh', { tabindex: '0', role: 'button', 'aria-label': `Open ${w.name}`, style: { '--wc-color': color },
      onclick: open, onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } } },
    h('div.wc-wh-head',
      h('div.ws-col', { style: { gap: '4px', minWidth: 0 } },
        h('div.wc-wh-name', stateDot(s.state), h('span.ws-ellipsis', w.name)),
        h('div.ws-row.ws-wrap', { style: { gap: '5px' } },
          h('span.ws-tag', s.size || 'size n/a'),
          h('span.ws-tag', st ? st.toLowerCase() : 'unknown'),
          fmt.num(s.max_cluster_count) > 1 ? h('span.ws-tag', `${s.min_cluster_count}–${s.max_cluster_count} clusters`) : null,
          s.type && String(s.type).toUpperCase() !== 'STANDARD' ? h('span.ws-tag', String(s.type).toLowerCase()) : null)),
      h('div.wc-wh-spend', h('b', fmt.usd(cc.toUsd(total, price))), h('span.ws-muted', `${fmt.credits(total)} cr · ${span}d`))),
    h('div.wc-wh-body',
      ring(util == null ? 0 : Math.min(100, util * 100), { size: 92, stroke: 9, caption: 'busy', color }),
      h('div.ws-col', { style: { gap: '6px', minWidth: 0, flex: 1 } },
        daily.length > 1 ? svgSpark(daily, { width: 220, height: 40, color, label: `${w.name} daily credits` }) : h('div.wc-nospark', 'no metered usage'),
        h('div.wc-wh-facts',
          fact('Auto-suspend', s.auto_suspend == null ? (Object.keys(s).length ? 'never' : '—') : `${s.auto_suspend}s`, s.auto_suspend == null || fmt.num(s.auto_suspend) > 60),
          fact('Running / queued', `${fmt.fmtInt(s.running)} / ${fmt.fmtInt(s.queued)}`, fmt.num(s.queued) > 0),
          fact('Busy', util == null ? '—' : `${fmt.pct(util)} · ${UTIL_TEXT[cls]}`, cls === 'idle'),
          fact('Idle burn', fmt.usd(cc.toUsd(w.idle_hour_credits, price)), fmt.num(w.idle_hour_credits) > total * 0.2 && total > 0)))),
    h('div.wc-wh-recs', w.recs.length ? w.recs.slice(0, 2).map((r) => h('span.wc-rec', { title: r.detail },
      sevBadge(r.severity), h('span.ws-ellipsis', r.title), r.saveUsdMonth > 0.005 ? h('b.wc-save', `~${fmt.usd(r.saveUsdMonth)}/mo`) : null))
      : h('span.wc-allgood', icon('check2-circle'), 'Nothing to fix'),
    w.recs.length > 2 ? h('span.ws-muted', `+${w.recs.length - 2} more`) : null));
  }
}

function fact(label, value, warn = false) {
  return h('div.wc-fact', h('span', label), h('b', { class: warn ? 'wc-hot' : '' }, value));
}
