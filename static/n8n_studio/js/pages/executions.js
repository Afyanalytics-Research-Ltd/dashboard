// Executions explorer across all workflows: status / workflow / mode / time window /
// duration filters, sorting, a stacked timeline, p50/p95, bulk retry/stop/delete/tag.

import { h, s, icon, mount, prefs } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toast, toastError, modal, chips, selectionBar, refreshControl, spinner, timeAgo, confirmDialog, menuButton } from '../../../ui/js/ui.js';
import { fmtDuration, fmtDate } from '../../../ui/js/format.js';
import { tip, tipRows } from '../../../ui/js/charts.js';
import { n8n, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load } from '../core/store.js';
import { durationMs, FAILED, ACTIVE, statusInfo, timeBuckets, percentile } from '../core/execution.js';
import { execBadge, bulkRun, comboBox, emptyHint, workflowLink } from '../core/widgets.js';

const WINDOWS = { '1h': 3600e3, '24h': 86400e3, '7d': 7 * 86400e3, '30d': 30 * 86400e3, all: null };

export default async function render(ctx) {
  const f = {
    status: ctx.query.get('status') || '',
    workflowId: ctx.query.get('workflowId') || '',
    since: ctx.query.get('since') || prefs.get('n8s:exec-since', '24h'),
    mode: ctx.query.get('mode') || '',
    minDur: Number(ctx.query.get('minDur') || 0),
    q: ctx.query.get('q') || '',
    sort: { key: 'startedAt', dir: 'desc' },
  };
  const selected = new Set();
  let rows = [];
  let cursor = null;
  let exhausted = false;
  let workflows = [];
  const wfName = (id) => workflows.find((w) => String(w.id) === String(id))?.name;

  const statsEl = h('div.n8s-exec-stats');
  const chartEl = h('div.n8s-exec-chart');
  const listEl = h('div');
  const barEl = h('div');
  const moreEl = h('div.text-center.my-2');
  const wfCombo = comboBox({ options: [], value: f.workflowId, emptyLabel: 'All workflows', placeholder: 'Find workflow…', onChange: (v) => { f.workflowId = v; sync(); reload(); } });
  const durInput = h('input.form-control.form-control-sm', { type: 'number', min: 0, step: 1, placeholder: 'min s', value: f.minDur || '', style: 'width:90px', title: 'Only executions slower than N seconds' });
  durInput.addEventListener('change', () => { f.minDur = Number(durInput.value || 0); sync(); draw(); });
  const idInput = h('input.form-control.form-control-sm', { type: 'search', placeholder: 'Go to #id', style: 'width:110px' });
  idInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && idInput.value.trim()) ctx.navigate(urls.execution(idInput.value.trim().replace('#', ''))); });
  const refresher = refreshControl(() => reload(true), { key: 'n8s-exec', initial: 0 });
  ctx.cleanup(refresher.stop);

  mount(ctx.el,
    pageHeader({
      title: 'Executions', icon: 'lightning-charge', subtitle: 'Every run across every workflow',
      actions: [
        refresher.el,
        menuButton('Stop…', [
          { header: 'Stop executions in bulk' },
          { label: 'Stop all running', icon: 'stop-fill', onClick: () => stopMany(['running']) },
          { label: 'Stop all waiting', icon: 'pause-fill', onClick: () => stopMany(['waiting']) },
          { label: 'Stop all queued', icon: 'hourglass', onClick: () => stopMany(['queued']) },
          { label: 'Stop running + waiting + queued', icon: 'x-octagon', danger: true, onClick: () => stopMany(['running', 'waiting', 'queued']) },
        ], { icon: 'stop-circle', variant: 'outline-danger' }),
      ],
    }),
    h('div.n8s-filterbar',
      h('div.n8s-st-chips'),
      wfCombo.el,
      h('div.n8s-since-chips'),
      h('select.form-select.form-select-sm', { style: 'width:130px', onchange: (e) => { f.mode = e.target.value; sync(); draw(); } },
        ['', 'trigger', 'webhook', 'manual', 'retry', 'integrated', 'internal', 'error', 'evaluation', 'chat'].map((m) => h('option', { value: m, selected: f.mode === m }, m || 'Any mode'))),
      durInput, idInput),
    statsEl, chartEl, barEl, listEl, moreEl);

  const sync = () => {
    prefs.set('n8s:exec-since', f.since);
    ctx.setQuery({ status: f.status || null, workflowId: f.workflowId || null, since: f.since, mode: f.mode || null, minDur: f.minDur || null });
  };

  function drawFilters() {
    mount(ctx.el.querySelector('.n8s-st-chips'), chips([{ value: '', label: 'All' }, ...['success', 'error', 'running', 'waiting', 'canceled', 'crashed', 'new'].map((x) => ({ value: x, label: statusInfo(x).label, color: statusInfo(x).color }))], f.status, (v) => { f.status = v; sync(); reload(); }));
    mount(ctx.el.querySelector('.n8s-since-chips'), chips(Object.keys(WINDOWS).map((k) => ({ value: k, label: k === 'all' ? 'All time' : `Last ${k}` })), f.since, (v) => { f.since = v || 'all'; sync(); reload(); }));
  }

  const sinceTs = () => (WINDOWS[f.since] ? Date.now() - WINDOWS[f.since] : null);

  async function fetchMore(reset) {
    if (reset) { rows = []; cursor = null; exhausted = false; }
    const limit = 250;
    const page = await n8n.get('/executions', { limit, cursor: cursor || undefined, status: f.status || undefined, workflowId: f.workflowId || undefined }, { signal: ctx.signal });
    const since = sinceTs();
    const fresh = page.data || [];
    rows.push(...fresh.filter((e) => !since || new Date(e.startedAt || e.createdAt).getTime() >= since));
    cursor = page.nextCursor;
    const oldest = fresh[fresh.length - 1];
    exhausted = !cursor || (since && oldest && new Date(oldest.startedAt || oldest.createdAt).getTime() < since);
  }

  async function reload(keepSelection = false) {
    if (!keepSelection) selected.clear();
    if (!rows.length) mount(listEl, spinner());
    try {
      await fetchMore(true);
      // fill the window: keep paging while still inside the time range (bounded)
      for (let i = 0; i < 6 && !exhausted && sinceTs(); i++) await fetchMore(false); // eslint-disable-line no-await-in-loop
      draw();
    } catch (err) {
      mount(listEl, h('div.text-danger', err.message));
    }
  }

  function filtered() {
    let out = rows;
    if (f.mode) out = out.filter((e) => e.mode === f.mode);
    if (f.minDur) out = out.filter((e) => (durationMs(e) || 0) >= f.minDur * 1000);
    const { key, dir } = f.sort;
    const val = (e) => (key === 'duration' ? durationMs(e) || 0 : key === 'workflow' ? (wfName(e.workflowId) || '').toLowerCase() : key === 'id' ? Number(e.id) : e[key] || '');
    return [...out].sort((a, b) => (val(a) < val(b) ? -1 : val(a) > val(b) ? 1 : 0) * (dir === 'asc' ? 1 : -1));
  }

  function draw() {
    drawFilters();
    const list = filtered();
    const done = list.filter((e) => e.stoppedAt).map((e) => durationMs(e));
    const fails = list.filter((e) => FAILED.has(e.status)).length;
    const counts = {};
    for (const e of list) counts[e.status] = (counts[e.status] || 0) + 1;
    mount(statsEl,
      stat('Executions', list.length.toLocaleString(), exhausted ? '' : '+ more not loaded'),
      stat('Success rate', list.length ? `${(((list.length - fails) / list.length) * 100).toFixed(1)}%` : '—', `${fails} failed`, fails ? 'bad' : 'good'),
      stat('p50 duration', done.length ? fmtDuration(percentile(done, 50) / 1000) : '—'),
      stat('p95 duration', done.length ? fmtDuration(percentile(done, 95) / 1000) : '—'),
      stat('Running now', String(list.filter((e) => ACTIVE.has(e.status)).length)),
      h('div.n8s-status-bar', Object.entries(counts).map(([st, n]) => h('span', { style: { flex: n, background: statusInfo(st).color }, title: `${statusInfo(st).label}: ${n}` }))));
    drawChart(list);
    drawBar();
    mount(listEl, dataTable({
      rows: list, selectable: true, selected, onSelect: drawBar, dense: true,
      sort: f.sort, onSort: (key, dir) => { f.sort = { key, dir }; draw(); },
      empty: emptyHint('lightning', 'No executions', 'Nothing matches these filters in the selected time window.'),
      rowClass: (e) => FAILED.has(e.status) && 'n8s-row-failed',
      columns: [
        { key: 'status', label: 'Status', render: (e) => execBadge(e.status, { size: 'sm' }) },
        { key: 'id', label: 'ID', sortable: true, render: (e) => h('a.af-mono', { href: urls.execution(e.id) }, `#${e.id}`) },
        { key: 'workflow', label: 'Workflow', sortable: true, render: (e) => workflowLink(workflows.find((w) => String(w.id) === String(e.workflowId)), { fallbackId: e.workflowId }) },
        { key: 'mode', label: 'Mode', render: (e) => h('span.n8s-mode', e.mode) },
        { key: 'startedAt', label: 'Started', sortable: true, render: (e) => h('span', { title: fmtDate(e.startedAt) }, timeAgo(e.startedAt)) },
        { key: 'duration', label: 'Duration', sortable: true, align: 'right', render: (e) => (e.stoppedAt ? durCell(durationMs(e), percentile(done, 95)) : h('span.text-info', icon('arrow-repeat', 'n8s-spin me-1'), 'running')) },
        { key: 'retry', label: '', render: (e) => (e.retryOf ? h('a.small', { href: urls.execution(e.retryOf) }, `↻ of #${e.retryOf}`) : e.retrySuccessId ? h('a.small.text-success', { href: urls.execution(e.retrySuccessId) }, `fixed by #${e.retrySuccessId}`) : null) },
      ],
      onRowClick: (e, ev) => (ev.metaKey || ev.ctrlKey ? window.open(urls.execution(e.id), '_blank') : ctx.navigate(urls.execution(e.id))),
    }));
    mount(moreEl, !exhausted && h('button.btn.btn-sm.btn-outline-secondary', { onclick: async () => { mount(moreEl, spinner()); await fetchMore(false); draw(); } }, 'Load older'));
  }

  function durCell(ms, p95) {
    const slow = p95 && ms > p95;
    return h('span', { class: slow && 'text-warning fw-semibold', title: slow ? 'Slower than the 95th percentile' : '' }, fmtDuration(ms / 1000));
  }

  function drawChart(list) {
    const span = WINDOWS[f.since] || (list.length ? Date.now() - new Date(list[list.length - 1].startedAt).getTime() : 86400e3);
    const buckets = timeBuckets(list, { buckets: 48, span: Math.max(span, 3600e3) });
    const max = Math.max(1, ...buckets.map((b) => b.success + b.error + b.other));
    const W = 960;
    const H = 90;
    const bw = W / buckets.length;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'n8s-tl', preserveAspectRatio: 'none' },
      buckets.map((b, i) => {
        const tot = b.success + b.error + b.other;
        let y = H;
        const seg = (n, cls) => { if (!n) return null; const hh = (n / max) * (H - 6); y -= hh; return s('rect', { x: i * bw + 1, y, width: bw - 2, height: hh, class: cls, rx: 2 }); };
        const g = s('g', {}, seg(b.success, 'ok'), seg(b.error, 'err'), seg(b.other, 'other'), s('rect', { x: i * bw, y: 0, width: bw, height: H, fill: 'transparent' }));
        tip(g, () => tipRows(new Date(b.t).toLocaleString(), [['Total', tot], ['Succeeded', b.success], ['Failed', b.error], ['Other', b.other]]));
        return g;
      }));
    mount(chartEl, svg);
  }

  function drawBar() {
    const items = rows.filter((e) => selected.has(e.id));
    mount(barEl, selectionBar(items.length, [
      h('button.btn.btn-sm.btn-outline-primary', { onclick: () => retry(items, false) }, icon('arrow-clockwise', 'me-1'), 'Retry (saved version)'),
      h('button.btn.btn-sm.btn-outline-primary', { onclick: () => retry(items, true) }, icon('arrow-repeat', 'me-1'), 'Retry (current workflow)'),
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => stopSel(items) }, icon('stop-fill', 'me-1'), 'Stop'),
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => tagSel(items) }, icon('tags', 'me-1'), 'Tag'),
      h('button.btn.btn-sm.btn-outline-danger', { onclick: () => del(items) }, icon('trash', 'me-1'), 'Delete'),
    ], () => { selected.clear(); draw(); }));
  }

  async function retry(items, loadWorkflow) {
    const failed = items.filter((e) => FAILED.has(e.status) || e.status === 'canceled');
    if (!failed.length) { toast('Only failed or canceled executions can be retried.', 'info'); return; }
    await bulkRun({ title: `Retry ${failed.length} execution(s)`, items: failed, label: (e) => `#${e.id} ${wfName(e.workflowId) || ''}`, run: (e) => n8n.post(`/executions/${enc(e.id)}/retry`, { loadWorkflow }), concurrency: 2 });
    reload();
  }

  async function stopSel(items) {
    await bulkRun({ title: 'Stop executions', items: items.filter((e) => ACTIVE.has(e.status)), label: (e) => `#${e.id}`, run: (e) => n8n.post(`/executions/${enc(e.id)}/stop`, {}) });
    reload();
  }

  async function stopMany(status) {
    const ok = await confirmDialog({ title: 'Stop executions?', danger: true, confirmLabel: 'Stop', message: `Stops every ${status.join(' / ')} execution${f.workflowId ? ` of ${wfName(f.workflowId)}` : ' on the instance'}.` });
    if (!ok) return;
    try {
      const r = await n8n.post('/executions/stop', { status, ...(f.workflowId ? { workflowId: f.workflowId } : {}) });
      toast(`Stopped ${r.stopped ?? 0} execution(s)`, 'success');
      reload();
    } catch (err) { toastError(err, 'Stop failed'); }
  }

  async function del(items) {
    const ok = await confirmDialog({ title: `Delete ${items.length} execution(s)?`, danger: true, confirmLabel: 'Delete', message: 'Execution data is removed permanently.' });
    if (!ok) return;
    await bulkRun({ title: 'Deleting executions', items, label: (e) => `#${e.id}`, run: (e) => n8n.del(`/executions/${enc(e.id)}`), concurrency: 6 });
    reload();
  }

  async function tagSel(items) {
    const tags = await load('tags', { force: true });
    const sel = h('select.form-select.form-select-sm', tags.map((t) => h('option', { value: t.id }, t.name)));
    modal({
      title: 'Tag executions', size: 'sm', body: h('div', h('div.small.text-muted.mb-2', 'Adds the tag to each execution (annotations).'), sel),
      actions: [{ label: 'Cancel' }, {
        label: 'Tag', variant: 'primary',
        onClick: async (close) => {
          close();
          await bulkRun({
            title: 'Tagging executions', items, label: (e) => `#${e.id}`,
            run: async (e) => {
              const cur = await n8n.get(`/executions/${enc(e.id)}/tags`);
              return n8n.put(`/executions/${enc(e.id)}/tags`, [...new Set([...(cur || []).map((t) => String(t.id)), String(sel.value)])].map((id) => ({ id })));
            },
          });
        },
      }],
    });
  }

  load('workflows').then((w) => {
    workflows = w;
    wfCombo.el.replaceWith(comboBox({ options: w.map((x) => ({ value: x.id, label: x.name, sub: x.active ? 'active' : '' })), value: f.workflowId, emptyLabel: 'All workflows', onChange: (v) => { f.workflowId = v; sync(); reload(); } }).el);
    if (rows.length) draw();
  }).catch(() => null);
  await reload();
}

function stat(label, value, sub, tone) {
  return h('div.n8s-kpi', { class: tone }, h('div.n8s-kpi-label', label), h('div.n8s-kpi-value', value), sub && h('div.n8s-kpi-sub', sub));
}
