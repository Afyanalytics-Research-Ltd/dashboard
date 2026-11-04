// Overview — the Studio's landing page: account pulse, spend trajectory,
// health of the warehouse fleet, what to fix next and what just ran.

import { h, icon, mount, card, ring, sevBadge, empty, errorView, toolkitBanner, skeleton, delta, healthLabel, animatedValue, bar } from '../lib/ui.js';
import { api } from '../lib/api.js';
import { state } from '../lib/store.js';
import * as fmt from '../lib/fmt.js';
import { projectMonth } from '../lib/metrics.js';
import { trendLine, spark, tableToggle } from '../lib/charts.js';
import { openQuery } from '../lib/query.js';
import { href } from '../../../ui/js/router.js';

export default async function render(ctx) {
  ctx.setTitle('Overview');
  const sess = state.session || {};
  const tk = sess.toolkit || {};
  const costOk = tk.toolkit_installed || tk.account_usage;
  const hour = new Date().getHours();
  const greet = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';

  const healthEl = h('div', ring(0, { caption: tk.toolkit_installed ? 'loading' : 'no toolkit', color: '#64748b' }));
  const kpiEl = h('div.wo-hero-metrics', [...Array(4)].map(() => h('div.wo-metric', h('div.ws-skel', { style: { height: '28px', width: '70%' } }), h('div.ws-skel', { style: { height: '10px', width: '50%', marginTop: '8px' } }))));
  const spendEl = h('div', skeleton(5, 24));
  const findingsEl = h('div', skeleton(5, 22));
  const fleetEl = h('div', skeleton(3, 22));
  const recentEl = h('div', skeleton(6, 20));

  mount(ctx.el,
    h('div.ws-card.ws-card-glow.wo-hero',
      healthEl,
      h('div',
        h('div.wo-hero-k', icon('snow2'), `${sess.organization || ''}${sess.account_name ? ` · ${sess.account_name}` : ''}`, h('span.ws-muted', `· ${String(sess.region || '').toLowerCase()}`)),
        h('h1', `${greet}, `, h('em', window.APP_UI?.user || sess.user || 'there')),
        h('p', `Connected as ${sess.user || '—'} with role ${sess.role || '—'} on Snowflake ${sess.version || ''}. Everything below is live from your account.`),
        h('div.wo-hero-actions',
          h('a.ws-btn.ws-btn-primary.ws-btn-lg', { href: href('/worksheets', { new: 1 }) }, icon('plus-lg'), 'New worksheet'),
          h('a.ws-btn.ws-btn-lg', { href: href('/optimizer') }, icon('lightning-charge'), 'Optimizer'),
          h('a.ws-btn.ws-btn-lg', { href: href('/cost') }, icon('cash-coin'), 'Cost Center'))),
      kpiEl),
    !costOk && h('div.ws-mt', toolkitBanner('Cost and optimizer panels need the toolkit (or ACCOUNT_USAGE access for the Studio role). Worksheets, history and the data explorer work already.')),
    h('div.ws-grid.ws-g-12.ws-mt',
      h('div.ws-span-8', card({ title: 'Spend trajectory', icon: 'graph-up-arrow', sub: 'Daily credits this month and last, with the projected month-end', body: spendEl })),
      h('div.ws-span-4', card({ title: 'Fix next', icon: 'bullseye', sub: 'Highest-impact open findings (7 days)', tools: h('a.ws-btn.ws-btn-sm.ws-btn-ghost', { href: href('/optimizer') }, 'All', icon('arrow-right')), body: findingsEl })),
      h('div.ws-span-5', card({ title: 'Warehouse fleet', icon: 'hdd-rack', sub: 'Live state', tools: h('a.ws-btn.ws-btn-sm.ws-btn-ghost', { href: href('/warehouses') }, 'Details', icon('arrow-right')), body: fleetEl })),
      h('div.ws-span-7', card({ title: 'Just ran', icon: 'clock-history', sub: 'Last 24 hours, real time', tools: h('a.ws-btn.ws-btn-sm.ws-btn-ghost', { href: href('/history') }, 'History', icon('arrow-right')), body: recentEl })),
      h('div.ws-span-12', card({ title: 'Jump to', icon: 'compass', body: h('div.wov-quick',
        quick('code-square', 'Worksheets', 'Multi-tab SQL with cost preview', '/worksheets'),
        quick('lightning-charge', 'Run all detectors', '12 tuning procedures', '/optimizer?run=all'),
        quick('fire', 'Idle burn', 'Warehouses billing for nothing', '/optimizer?detector=sp_detect_warehouse_waste'),
        quick('cash-stack', 'Expensive queries', 'Costliest query shapes', '/cost?tab=queries'),
        quick('database', 'Data explorer', 'Browse every object', '/data'),
        quick('braces-asterisk', 'Procedures', 'Run stored procedures', '/procedures')) })),
    ));

  loadKpis();
  loadSpend();
  loadFindings();
  loadFleet();
  loadRecent();

  async function loadKpis() {
    const tiles = [];
    let summary = null;
    let overview = null;
    await Promise.all([
      tk.toolkit_installed ? api.findingsSummary({ days: 7 }, { signal: ctx.signal }).then((s) => { summary = s; }).catch(() => {}) : null,
      costOk ? api.costOverview(30, { signal: ctx.signal }).then((o) => { overview = o; }).catch(() => {}) : null,
    ]);
    if (!ctx.active) return;
    if (summary) mount(healthEl, ring(summary.health_score, { caption: healthLabel(summary.health_score).toLowerCase() }));
    const f = overview?.forecast;
    tiles.push(tile(f ? animatedValue(f.month_to_date * (overview.credit_price_usd || 0), (x) => fmt.usd(x)) : '—', 'Spend this month', 'calendar-month',
      f ? delta(f.pace_vs_last_month, { suffix: 'vs same day last month' }) : null));
    tiles.push(tile(f ? animatedValue(f.projected_month * (overview.credit_price_usd || 0), (x) => fmt.usd(x)) : '—', 'Projected month-end', 'graph-up-arrow',
      f ? h('span.ws-muted', `${fmt.credits(f.projected_month)} credits`) : null));
    const q = Array.isArray(overview?.queries) ? overview.queries : [];
    const last = q[q.length - 1];
    tiles.push(tile(last ? animatedValue(last.queries, (x) => fmt.compact(x)) : '—', 'Queries today', 'lightning',
      last ? h('span.ws-muted', `${fmt.pct((last.failed || 0) / Math.max(1, last.queries), 1)} failed`) : null));
    tiles.push(tile(summary ? animatedValue(summary.est_usd, (x) => fmt.usd(x)) : '—', 'Waste found (7d)', 'fire',
      summary ? h('span.ws-muted', `${fmt.plural(summary.total, 'finding')}`) : null));
    mount(kpiEl, tiles);
  }

  function tile(value, label, ic, foot) {
    return h('div.wo-metric', h('div.wo-metric-v', value), h('div.wo-metric-l', icon(ic), label), foot && h('div', { style: { fontSize: '11.5px', marginTop: '4px' } }, foot));
  }

  async function loadSpend() {
    if (!costOk) { mount(spendEl, empty({ icon: 'graph-up-arrow', title: 'Spend needs account usage data' })); return; }
    try {
      const o = await api.costOverview(30, { signal: ctx.signal });
      if (!Array.isArray(o.month_daily)) { mount(spendEl, errorView({ message: o.month_daily?.error || 'Unavailable', code: o.month_daily?.unavailable ? 'toolkit' : '' })); return; }
      const price = o.credit_price_usd || 0;
      const proj = projectMonth(o.month_daily, new Date());
      const now = new Date();
      const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1)).toISOString().slice(0, 10);
      // Cumulative actual for this month + dashed forecast to month end.
      let run = 0;
      const actual = o.month_daily.filter((r) => String(r.day).slice(0, 10) >= monthStart).map((r) => { run += Number(r.credits) || 0; return { x: String(r.day).slice(0, 10), y: run * price }; });
      let cum = run;
      const forecast = proj.points.map((p) => { cum += p.y; return { x: p.x, y: cum * price }; });
      const lastMonthTotal = (o.forecast?.last_month || 0) * price;
      const chartEl = h('div.ws-chart');
      const tbl = tableToggle([{ key: 'x', label: 'Day' }, { key: 'y', label: 'Cumulative $', num: true, fmt: (v) => fmt.usd(v) }], [...actual, ...forecast]);
      mount(spendEl,
        h('div.ws-row.ws-wrap', { style: { gap: '22px', marginBottom: '8px' } },
          bigNum(fmt.usd(proj.mtd * price), 'month to date'),
          bigNum(fmt.usd(proj.projected * price), 'projected month-end'),
          bigNum(fmt.usd(lastMonthTotal), 'last month'),
          bigNum(`${fmt.credits(proj.rate)}/day`, '7-day run-rate'),
          h('div', { style: { marginLeft: 'auto' } }, tbl.button)),
        chartEl, tbl.table);
      trendLine(chartEl, { points: actual, forecast, ctx, name: 'Month to date', valueFmt: (v) => fmt.usd(v),
        markLine: lastMonthTotal ? { symbol: 'none', label: { formatter: `last month ${fmt.usd(lastMonthTotal)}`, color: 'inherit', fontSize: 11, position: 'insideEndTop' },
          lineStyle: { type: 'dotted', color: '#7d8aa5' }, data: [{ yAxis: lastMonthTotal }] } : undefined });
    } catch (err) {
      if (err.name !== 'AbortError') mount(spendEl, errorView(err, loadSpend));
    }
  }

  function bigNum(v, label) {
    return h('div', h('div', { style: { fontFamily: 'var(--ws-display)', fontSize: '22px', fontWeight: 600, letterSpacing: '-.02em' } }, v), h('div.ws-muted', { style: { fontSize: '11.5px' } }, label));
  }

  async function loadFindings() {
    if (!tk.toolkit_installed) { mount(findingsEl, empty({ icon: 'bullseye', title: 'Install the toolkit', text: 'Then the top issues show up here.' })); return; }
    try {
      const { findings } = await api.findings({ days: 7, severity: 'CRITICAL,HIGH', sort: 'cost', limit: 6, triage: 'open,acknowledged,in_progress' }, { signal: ctx.signal });
      if (!findings.length) { mount(findingsEl, empty({ icon: 'stars', title: 'Nothing critical', text: 'No open critical or high findings this week.' })); return; }
      mount(findingsEl, h('div.ws-col', { style: { gap: '8px' } }, findings.map((f) => h('a.wov-wh', { href: href('/optimizer', { detector: f.detector }), style: { textDecoration: 'none', color: 'inherit' } },
        h('div.ws-row.ws-between', sevBadge(f.severity), h('span.ws-mono.ws-muted', { style: { fontSize: '11.5px' } }, f.est_usd ? fmt.usd(f.est_usd) : fmt.seconds(f.wasted_seconds))),
        h('b', fmt.detectorLabel(f.detector)),
        h('span.ws-muted.ws-ellipsis', { style: { fontSize: '12px' } }, fmt.oneLine(f.detail?.sql_preview || f.object_name, 80))))));
    } catch (err) {
      if (err.name !== 'AbortError') mount(findingsEl, errorView(err, loadFindings));
    }
  }

  async function loadFleet() {
    const whs = sess.warehouses || [];
    if (!whs.length) { mount(fleetEl, empty({ icon: 'hdd-rack', title: 'No warehouses visible to this role' })); return; }
    let credits = {};
    if (costOk) {
      try {
        const w = await api.costWarehouses(7, { signal: ctx.signal });
        credits = Object.fromEntries((w.warehouses || []).map((x) => [x.name, x]));
      } catch { /* fleet still renders without spend */ }
    }
    if (!ctx.active) return;
    const maxCr = Math.max(0.0001, ...Object.values(credits).map((x) => x.compute_credits || 0));
    mount(fleetEl, h('div.wov-live', whs.map((w) => {
      const c = credits[w.name];
      const st = String(w.state || '').toUpperCase();
      const sparkEl = h('div', { style: { height: '30px' } });
      if (c?.daily?.length > 1) setTimeout(() => spark(sparkEl, { values: c.daily.map((d) => d.credits), labels: c.daily.map((d) => d.day), ctx }), 0);
      return h('div.wov-wh',
        h('div.ws-row', h(`span.ws-dot${st === 'STARTED' ? '.on' : st === 'SUSPENDED' ? '' : '.warn'}`), h('b.ws-ellipsis', w.name), h('span.ws-tag', { style: { marginLeft: 'auto' } }, w.size || '—')),
        h('div.ws-row.ws-between', { style: { fontSize: '12px' } }, h('span.ws-muted', st.toLowerCase() || 'unknown'),
          h('span.ws-mono', c ? `${fmt.credits(c.compute_credits)} cr · 7d` : `suspend ${w.auto_suspend ?? '—'}s`)),
        c ? bar(c.compute_credits / maxCr) : null,
        sparkEl);
    })));
  }

  async function loadRecent() {
    try {
      const { rows } = await api.history({ scope: 'live', hours: 24, limit: 8 }, { signal: ctx.signal });
      if (!rows.length) { mount(recentEl, empty({ icon: 'clock-history', title: 'No queries in the last 24 hours' })); return; }
      const maxMs = Math.max(1, ...rows.map((r) => Number(r.total_elapsed_time) || 0));
      mount(recentEl, h('div.ws-table-wrap', h('table.ws-table', h('tbody', rows.map((r) => h('tr.clickable', { onclick: () => openQuery(r.query_id) },
        h('td', { style: { width: '22px' } }, icon(r.execution_status === 'SUCCESS' ? 'check-circle-fill' : r.execution_status === 'RUNNING' ? 'arrow-repeat' : 'x-octagon-fill', r.execution_status === 'SUCCESS' ? 'text-success' : r.execution_status === 'RUNNING' ? 'text-info' : 'text-danger')),
        h('td', h('span.ws-sql-snippet', { style: { maxWidth: '360px' } }, fmt.oneLine(r.query_text, 140))),
        h('td.ws-muted', { style: { fontSize: '12px', whiteSpace: 'nowrap' } }, r.user_name),
        h('td', { style: { width: '120px' } }, bar((Number(r.total_elapsed_time) || 0) / maxMs)),
        h('td.num', fmt.ms(r.total_elapsed_time)),
        h('td.num.ws-muted', { title: fmt.dateTime(r.start_time) }, fmt.relTime(r.start_time))))))));
    } catch (err) {
      if (err.name !== 'AbortError') mount(recentEl, errorView(err, loadRecent));
    }
  }
}

function quick(ic, title, sub, path) {
  return h('a', { href: href(path.split('?')[0], Object.fromEntries(new URLSearchParams(path.split('?')[1] || ''))) }, icon(ic), h('div', h('b', title), h('small', sub)));
}
