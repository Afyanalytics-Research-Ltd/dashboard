// Cleanup: find sprawling / orphaned content and tidy it in bulk.
// Rules run client-side over every query, dashboard and alert.

import { h, icon, mount, prefs } from '../../../ui/js/dom.js';
import {
  pageHeader, card, dataTable, selectionBar, tabs, emptyState, errorState, skeleton, timeAgo, pager,
  confirmDialog, formModal, toast, toastError, refreshControl, chips, spinner,
} from '../../../ui/js/ui.js';
import { donut } from '../../../ui/js/charts.js';
import { fmtNumber, fmtPercent, truncate } from '../../../ui/js/format.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { getDashboardsFull, invalidate, scheduleText, userChip } from '../core/common.js';
import { countUp } from './home_shared.js';

const DAY = 86400000;
const UNNAMED = /^(new query|untitled|query \d+)?$/i;

export default async function render(ctx) {
  const { el, signal } = ctx;
  let tab = ctx.query.get('tab') || 'orphans';
  let staleDays = Number(ctx.query.get('days')) || prefs.get('rd:cleanup:days', 90);
  const data = { queries: [], dashboards: [], alerts: [], archived: 0, vizCounts: new Map() };
  const selected = new Set();
  const paging = { offset: 0, limit: prefs.get('rd:cleanup:limit', 25) };

  const scoreEl = h('div.rdh-score', skeleton(1, { height: 120 }));
  const tabsHost = h('div');
  const bodyEl = h('div.rdh-cleanup-body', skeleton(8));

  const refresh = refreshControl(() => load(true), { key: 'studio-cleanup', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Cleanup', icon: 'recycle',
      subtitle: 'Find queries and dashboards nobody uses, duplicates and abandoned drafts — and tidy them up in bulk.',
      actions: refresh.el,
    }),
    scoreEl,
    card({ body: h('div', tabsHost, bodyEl) }));

  async function load(force = false) {
    mount(bodyEl, h('div.rdh-loading', spinner(), ' Scanning every query, dashboard and alert…'));
    try {
      if (force) invalidate('dashfull');
      const [queries, dashboards, alerts, archive] = await Promise.all([
        rd.all('/api/queries', {}, { signal }),
        getDashboardsFull(),
        rd.get('/api/alerts', null, { signal }),
        rd.get('/api/queries/archive', { page_size: 1 }, { signal }),
      ]);
      Object.assign(data, { queries, dashboards, alerts: alerts || [], archived: archive.count || 0 });
    } catch (err) {
      if (err.name !== 'AbortError') mount(bodyEl, errorState(err, () => load(true)));
      return;
    }
    if (!ctx.active) return;
    selected.clear();
    draw();
  }

  function analyse() {
    const onDash = new Map();
    for (const d of data.dashboards) {
      for (const w of d.widgets || []) {
        const qid = w.visualization?.query?.id;
        if (qid) (onDash.get(qid) || onDash.set(qid, new Set()).get(qid)).add(d.name);
      }
    }
    const inAlert = new Set(data.alerts.map((a) => a.query_id ?? a.query?.id));
    const now = Date.now();
    const lastRun = (q) => (q.retrieved_at ? new Date(q.retrieved_at).getTime() : null);
    const byHash = new Map();
    for (const q of data.queries) {
      const key = q.query_hash || (q.query || '').trim().toLowerCase();
      if (!key) continue;
      (byHash.get(key) || byHash.set(key, []).get(key)).push(q);
    }
    const dupIds = new Set([...byHash.values()].filter((g) => g.length > 1).flat().map((q) => q.id));
    const rules = {
      never: data.queries.filter((q) => !q.latest_query_data_id && !q.retrieved_at),
      stale: data.queries.filter((q) => { const t = lastRun(q); return t && now - t > staleDays * DAY && !q.schedule?.interval; }),
      orphans: data.queries.filter((q) => !onDash.has(q.id) && !q.schedule?.interval && !inAlert.has(q.id)),
      drafts: data.queries.filter((q) => q.is_draft),
      unnamed: data.queries.filter((q) => UNNAMED.test((q.name || '').trim())),
      duplicates: data.queries.filter((q) => dupIds.has(q.id)),
      tableonly: data.queries.filter((q) => data.vizCounts.has(q.id) && data.vizCounts.get(q.id) <= 1),
      emptydash: data.dashboards.filter((d) => !(d.widgets || []).length),
      draftdash: data.dashboards.filter((d) => d.is_draft),
    };
    const flagged = new Set([...rules.never, ...rules.stale, ...rules.orphans, ...rules.drafts, ...rules.unnamed, ...rules.duplicates].map((q) => q.id));
    return { rules, onDash, inAlert, byHash, flagged };
  }

  function draw() {
    const a = analyse();
    const total = data.queries.length;
    const healthy = total - a.flagged.size;
    const score = total ? healthy / total : 1;
    const tone = score >= 0.8 ? '#16a34a' : score >= 0.5 ? '#d97706' : '#dc2626';
    mount(scoreEl, h('div.rdh-score-card',
      donut([{ label: 'Healthy', value: healthy, color: tone }, { label: 'Needs attention', value: a.flagged.size, color: '#e2e8f0' }],
        { size: 120, thickness: 14, centerLabel: fmtPercent(healthy, total || 1), centerSub: 'tidy' }),
      h('div.rdh-score-text',
        h('div.rdh-score-title', score >= 0.8 ? 'Your workspace is in good shape' : score >= 0.5 ? 'Some tidying would help' : 'Lots of clutter to clear'),
        h('div.text-muted', `${fmtNumber(a.flagged.size)} of ${fmtNumber(total)} queries match at least one cleanup rule.`),
        h('div.rdh-score-facts',
          fact('code-square', countUp(total), 'active queries'),
          fact('archive', countUp(data.archived), 'archived', urls.queries({ view: 'archive' })),
          fact('grid-1x2', countUp(data.dashboards.length), 'dashboards'),
          fact('bell', countUp(data.alerts.length), 'alerts')))));

    const items = [
      { id: 'orphans', label: 'Not used anywhere', icon: 'link-45deg', count: a.rules.orphans.length },
      { id: 'never', label: 'Never run', icon: 'hourglass', count: a.rules.never.length },
      { id: 'stale', label: `Not run in ${staleDays}d`, icon: 'clock-history', count: a.rules.stale.length },
      { id: 'duplicates', label: 'Duplicate SQL', icon: 'files', count: a.rules.duplicates.length },
      { id: 'unnamed', label: 'Unnamed', icon: 'question-square', count: a.rules.unnamed.length },
      { id: 'drafts', label: 'Drafts', icon: 'pencil', count: a.rules.drafts.length },
      { id: 'tableonly', label: 'Table only', icon: 'table', count: data.vizCounts.size ? a.rules.tableonly.length : '?' },
      { id: 'emptydash', label: 'Empty dashboards', icon: 'grid-1x2', count: a.rules.emptydash.length },
      { id: 'draftdash', label: 'Draft dashboards', icon: 'grid-1x2', count: a.rules.draftdash.length },
    ];
    mount(tabsHost, tabs(items, tab, (id) => { tab = id; selected.clear(); paging.offset = 0; ctx.setQuery({ tab: id === 'orphans' ? null : id }); drawBody(a); }));
    drawBody(a);
  }

  const fact = (ic, value, label, href) => h(href ? 'a.rdh-fact' : 'span.rdh-fact', { href }, icon(ic), h('b', value), ` ${label}`);

  const DESCRIPTIONS = {
    orphans: 'Queries not on any dashboard, without a refresh schedule and not used by an alert. Usually safe to archive once you check nobody runs them by hand.',
    never: 'Queries that have never produced a result.',
    stale: 'Unscheduled queries whose last result is older than the chosen age.',
    duplicates: 'Queries with exactly the same SQL text. Keep one and archive the rest.',
    unnamed: 'Queries still called "New Query" (or with no name) — hard to find later.',
    drafts: 'Unpublished drafts. They are hidden from most lists.',
    tableonly: 'Queries whose only visualization is the default table. Scan to find candidates for a chart.',
    emptydash: 'Dashboards without a single widget.',
    draftdash: 'Dashboards that were never published.',
  };

  function drawBody(a) {
    const isDash = tab === 'emptydash' || tab === 'draftdash';
    let rows = a.rules[tab] || [];
    if (tab === 'duplicates') rows = [...rows].sort((x, y) => String(x.query_hash).localeCompare(String(y.query_hash)) || x.id - y.id);
    if (paging.offset >= rows.length) paging.offset = 0;
    const pageRows = rows.slice(paging.offset, paging.offset + paging.limit);
    const header = h('div.rdh-rule-head',
      h('div.text-muted.small', DESCRIPTIONS[tab]),
      tab === 'stale' && chips([30, 90, 180, 365].map((d) => ({ value: d, label: `${d} days` })), staleDays, (v) => {
        staleDays = Number(v) || 90; prefs.set('rd:cleanup:days', staleDays); ctx.setQuery({ days: staleDays }); draw();
      }),
      tab === 'tableonly' && h('button.btn.btn-sm.btn-outline-primary', { onclick: (e) => scanViz(e.currentTarget) }, icon('search', 'me-1'), data.vizCounts.size ? 'Rescan visualizations' : `Scan ${fmtNumber(data.queries.length)} queries`));
    const bar = selectionBar(selected.size, isDash
      ? [h('button.btn.btn-sm.btn-light', { onclick: () => archiveDashboards(rows.filter((d) => selected.has(d.id))) }, icon('archive', 'me-1'), 'Archive'),
        selected.size < rows.length && h('button.btn.btn-sm.btn-link.text-white', { onclick: () => { rows.forEach((r) => selected.add(r.id)); drawBody(a); } }, `Select all ${fmtNumber(rows.length)}`)]
      : [
        h('button.btn.btn-sm.btn-light', { onclick: () => tagQueries(rows.filter((q) => selected.has(q.id))) }, icon('tag', 'me-1'), 'Add tag'),
        h('button.btn.btn-sm.btn-warning', { onclick: () => archiveQueries(rows.filter((q) => selected.has(q.id))) }, icon('archive', 'me-1'), 'Archive'),
        selected.size < rows.length && h('button.btn.btn-sm.btn-link.text-white', {
          onclick: () => { rows.forEach((r) => selected.add(r.id)); drawBody(a); },
        }, `Select all ${fmtNumber(rows.length)} matching`),
      ], () => { selected.clear(); drawBody(a); });

    if (tab === 'tableonly' && !data.vizCounts.size) {
      mount(bodyEl, header, emptyState({ icon: 'table', title: 'Not scanned yet', message: 'Scanning opens every query to count its visualizations (runs in the background, nothing is executed).' }));
      return;
    }
    if (!rows.length) {
      mount(bodyEl, header, emptyState({ icon: 'stars', title: 'Nothing to clean here', message: 'No content matches this rule. Nice!' }));
      return;
    }
    const onSelect = () => drawBody(a);
    const table = isDash
      ? dataTable({
        rows: pageRows, rowKey: (d) => d.id, selectable: true, selected, onSelect,
        onRowClick: (d) => ctx.navigate(urls.dashboard(d.id)),
        columns: [
          { key: 'name', label: 'Dashboard', render: (d) => h('div', h('a.fw-semibold', { href: urls.dashboard(d.id) }, d.name), h('div.small.text-muted', (d.tags || []).join(' · '))) },
          { key: 'widgets', label: 'Widgets', align: 'right', render: (d) => fmtNumber((d.widgets || []).length) },
          { key: 'user', label: 'Owner', render: (d) => userChip(d.user) },
          { key: 'updated_at', label: 'Updated', render: (d) => timeAgo(d.updated_at) },
        ],
      })
      : dataTable({
        rows: pageRows,
        rowKey: (q) => q.id, selectable: true, selected, onSelect,
        rowClass: (q) => tab === 'duplicates' && `rdh-dup-${hashColor(q.query_hash)}`,
        onRowClick: (q) => ctx.navigate(urls.query(q.id)),
        columns: [
          { key: 'name', label: 'Query', render: (q) => h('div.rdh-q-cell',
            h('a.fw-semibold', { href: urls.query(q.id) }, q.name || h('i', 'Unnamed'), ' ', h('span.text-muted.small', `#${q.id}`)),
            h('div.rdh-q-sql', truncate((q.query || '').replace(/\s+/g, ' '), 110)),
            tab === 'duplicates' && h('div.small.text-muted', `${a.byHash.get(q.query_hash)?.length || 1} copies`)) },
          { key: 'used', label: 'Used in', render: (q) => {
            const ds = a.onDash.get(q.id);
            const parts = [];
            if (ds) parts.push(h('span.af-tag', { title: [...ds].join(', ') }, icon('grid-1x2'), ` ${ds.size} dashboard${ds.size > 1 ? 's' : ''}`));
            if (q.schedule?.interval) parts.push(h('span.af-tag', { title: scheduleText(q.schedule) }, icon('arrow-repeat'), ' scheduled'));
            if (a.inAlert.has(q.id)) parts.push(h('span.af-tag', icon('bell'), ' alert'));
            return parts.length ? h('div.af-tags', parts) : h('span.text-muted.small', 'nowhere');
          } },
          { key: 'retrieved_at', label: 'Last run', render: (q) => (q.retrieved_at ? timeAgo(q.retrieved_at) : h('span.text-muted', 'never')) },
          { key: 'user', label: 'Owner', render: (q) => userChip(q.user) },
          { key: 'created_at', label: 'Created', render: (q) => timeAgo(q.created_at) },
        ],
      });
    const pg = rows.length > paging.limit || paging.limit !== 25 ? pager({
      total: rows.length, limit: paging.limit, offset: paging.offset,
      onChange: (offset, limit) => {
        paging.offset = offset;
        if (limit !== paging.limit) { paging.limit = limit; paging.offset = 0; prefs.set('rd:cleanup:limit', limit); }
        drawBody(a);
        bodyEl.scrollIntoView({ block: 'start', behavior: 'smooth' });
      },
    }) : null;
    mount(bodyEl, header, table, pg, bar);
  }

  async function scanViz(btn) {
    btn.disabled = true;
    const total = data.queries.length;
    let done = 0;
    const queue = [...data.queries];
    const worker = async () => {
      while (queue.length && ctx.active) {
        const q = queue.shift();
        try {
          const full = await rd.get(`/api/queries/${enc(q.id)}`, null, { signal });
          data.vizCounts.set(q.id, (full.visualizations || []).length);
        } catch (err) { if (err.name === 'AbortError') return; }
        done++;
        if (done % 10 === 0 || done === total) mount(btn, spinner('sm'), ` Scanned ${done} / ${total}`);
      }
    };
    mount(btn, spinner('sm'), ' Scanning…');
    await Promise.all(Array.from({ length: 6 }, worker));
    if (!ctx.active) return;
    toast(`Scanned ${fmtNumber(done)} queries`, 'success');
    draw();
  }

  async function archiveQueries(list) {
    if (!list.length) return;
    const ok = await confirmDialog({
      title: `Archive ${list.length} quer${list.length === 1 ? 'y' : 'ies'}?`, danger: true, confirmLabel: 'Archive',
      message: 'Archived queries disappear from lists and search, and their widgets are removed from dashboards. You can unarchive them from Queries → Archived.',
      details: h('ul.small.mb-0.rdh-confirm-list', list.slice(0, 12).map((q) => h('li', `#${q.id} ${q.name}`)), list.length > 12 && h('li', `…and ${list.length - 12} more`)),
    });
    if (!ok) return;
    const results = await Promise.allSettled(list.map((q) => rd.del(`/api/queries/${enc(q.id)}`)));
    const failed = results.filter((r) => r.status === 'rejected');
    if (failed.length) toast(`${failed.length} could not be archived: ${failed[0].reason.message}`, 'warning');
    else toast(`Archived ${list.length} quer${list.length === 1 ? 'y' : 'ies'}`, 'success');
    invalidate('dashfull');
    load();
  }

  async function archiveDashboards(list) {
    if (!list.length) return;
    const ok = await confirmDialog({
      title: `Archive ${list.length} dashboard${list.length === 1 ? '' : 's'}?`, danger: true, confirmLabel: 'Archive',
      message: 'Archived dashboards are hidden from lists. Their queries are not affected.',
      details: h('ul.small.mb-0.rdh-confirm-list', list.slice(0, 12).map((d) => h('li', d.name))),
    });
    if (!ok) return;
    const results = await Promise.allSettled(list.map((d) => rd.del(`/api/dashboards/${enc(d.id)}`)));
    const failed = results.filter((r) => r.status === 'rejected');
    if (failed.length) toast(`${failed.length} could not be archived: ${failed[0].reason.message}`, 'warning');
    else toast(`Archived ${list.length} dashboard${list.length === 1 ? '' : 's'}`, 'success');
    invalidate('dashfull');
    load(true);
  }

  function tagQueries(list) {
    if (!list.length) return;
    formModal({
      title: `Tag ${list.length} quer${list.length === 1 ? 'y' : 'ies'}`, icon: 'tag', submitLabel: 'Add tag',
      fields: [{ name: 'tag', label: 'Tag', required: true, placeholder: 'e.g. to-review', value: 'to-review' }],
      onSubmit: async ({ tag }, close) => {
        const results = await Promise.allSettled(list.map((q) => rd.post(`/api/queries/${enc(q.id)}`, { tags: [...new Set([...(q.tags || []), tag.trim()])] })));
        const failed = results.filter((r) => r.status === 'rejected');
        close();
        if (failed.length) toast(`${failed.length} failed: ${failed[0].reason.message}`, 'warning');
        else toast(`Tagged ${list.length} quer${list.length === 1 ? 'y' : 'ies'} "${tag}"`, 'success');
        invalidate('qtags');
        load();
      },
    });
  }

  await load();
}

// Stable 1..4 colour band per duplicate group so copies are visually grouped.
function hashColor(hash) {
  let n = 0;
  for (const ch of String(hash || '')) n = (n * 31 + ch.charCodeAt(0)) >>> 0;
  return (n % 4) + 1;
}
