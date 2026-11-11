// Execution inspector: error panel, canvas with per-node status, waterfall timeline,
// node-by-node input/output, comparison with another run (default: last success),
// tags, parent/child links, retry/stop/delete and "debug in editor".

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, toast, toastError, tabs, card, jsonView, menuButton, modal, spinner, confirmDialog, copyButton } from '../../../ui/js/ui.js';
import { fmtDate, fmtDuration } from '../../../ui/js/format.js';
import { tip, tipRows } from '../../../ui/js/charts.js';
import { n8n, enc, saveText } from '../core/api.js';
import { urls } from '../core/urls.js';
import { loadCatalog, nodeIcon } from '../core/catalog.js';
import { nodeSummaries, nodeStatusMap, edgeCounts, mainError, durationMs, ACTIVE, FAILED, compareExecutions, toPinData, parentExecution, subExecutions, triggerNode, errorSignature } from '../core/execution.js';
import { pathString } from '../core/jsondiff.js';
import { saveBody } from '../core/workflow.js';
import { load, patchWorkflow } from '../core/store.js';
import { execBadge, tagPills, workflowLink, emptyHint } from '../core/widgets.js';
import { createCanvas } from '../editor/canvas.js';
import { dataView } from '../editor/dataview.js';
import { stripHtml } from '../editor/paramform.js';

export default async function render(ctx) {
  const id = ctx.params.executionId;
  await loadCatalog().catch(() => null);
  let e = await n8n.get(`/executions/${enc(id)}`, { includeData: true }, { signal: ctx.signal });
  const workflows = await load('workflows').catch(() => []);
  const wf = workflows.find((w) => String(w.id) === String(e.workflowId));
  const wfData = e.workflowData || wf || { nodes: [], connections: {} };
  ctx.setTitle(`Execution #${e.id}`);

  let tab = ctx.query.get('tab') || (FAILED.has(e.status) ? 'nodes' : 'canvas');
  const body = h('div.n8s-exec-body');
  const err = mainError(e);
  const parent = parentExecution(e);
  const subs = subExecutions(e);
  const tagsEl = h('span');

  const header = pageHeader({
    title: h('span', `Execution #${e.id}`),
    icon: 'lightning-charge',
    badges: execBadge(e.status),
    crumbs: [{ label: 'Executions', href: urls.executions() }, wf ? { label: wf.name, href: urls.executions({ workflowId: wf.id }) } : null].filter(Boolean),
    subtitle: h('span', workflowLink(wf || wfData, { fallbackId: e.workflowId }), ` · ${e.mode} · started ${fmtDate(e.startedAt)} · `, e.stoppedAt ? fmtDuration(durationMs(e) / 1000) : 'running', ' ', tagsEl),
    actions: [
      ACTIVE.has(e.status) && h('button.btn.btn-sm.btn-outline-danger', { onclick: () => act('stop') }, icon('stop-fill', 'me-1'), 'Stop'),
      (FAILED.has(e.status) || e.status === 'canceled') && h('div.btn-group',
        h('button.btn.btn-sm.btn-primary', { onclick: () => act('retry', true) }, icon('arrow-repeat', 'me-1'), 'Retry with current workflow'),
        h('button.btn.btn-sm.btn-outline-primary', { onclick: () => act('retry', false), title: 'Retry with the workflow version that originally ran' }, 'original')),
      wf && h('a.btn.btn-sm.btn-outline-secondary', { href: urls.workflow(wf.id, { execution: e.id }) }, icon('diagram-3', 'me-1'), 'Open in editor'),
      menuButton('', [
        wf && { label: 'Debug in editor (pin this run\'s trigger data)', icon: 'bug', onClick: () => debugInEditor() },
        { label: 'Edit tags…', icon: 'tags', onClick: () => editTags() },
        { label: 'Download JSON', icon: 'download', onClick: () => saveText(JSON.stringify(e, null, 2), `execution-${e.id}.json`) },
        { label: 'Similar failures', icon: 'inbox', href: urls.failures({ workflowId: e.workflowId }) },
        { divider: true },
        { label: 'Delete execution', icon: 'trash', danger: true, onClick: () => act('delete') },
      ]),
    ],
  });

  const alerts = h('div',
    err && h('div.n8s-errpanel',
      h('div.n8s-errpanel-icon', icon('exclamation-octagon-fill')),
      h('div.flex-grow-1',
        h('div.n8s-errpanel-title', err.message),
        err.node && h('div.small', 'in node ', h('button.af-link-btn', { onclick: () => { tab = 'nodes'; tabBar.setActive('nodes'); show(err.node); } }, err.node), err.httpCode ? ` · HTTP ${err.httpCode}` : '', err.type ? ` · ${err.type}` : ''),
        err.description && h('div.small.mt-1', stripHtml(err.description)),
        hintFor(err.message),
        err.stack && h('details.mt-1', h('summary.small', 'Stack trace'), h('pre.n8s-stack', err.stack))),
      h('div.d-flex.flex-column.gap-1', copyButton(`${err.node || ''}: ${err.message}\n${err.description || ''}`, 'Copy error'),
        h('a.btn.btn-sm.btn-link', { href: `https://community.n8n.io/search?q=${encodeURIComponent(errorSignature(err.message).replace(/<\w+>/g, ''))}`, target: '_blank', rel: 'noopener' }, icon('search', 'me-1'), 'Ask the forum'))),
    (parent || subs.length || e.retryOf || e.retrySuccessId) && h('div.n8s-chain',
      parent && h('span', icon('arrow-up-left', 'me-1'), 'Called by ', h('a', { href: urls.execution(parent.executionId) }, `#${parent.executionId}`), parent.workflowId && [' in ', workflowLink(workflows.find((w) => w.id === parent.workflowId), { fallbackId: parent.workflowId })]),
      subs.map((x) => h('span', icon('arrow-down-right', 'me-1'), x.node, ' → ', h('a', { href: urls.execution(x.executionId) }, `#${x.executionId}`))),
      e.retryOf && h('span', icon('arrow-counterclockwise', 'me-1'), 'Retry of ', h('a', { href: urls.execution(e.retryOf) }, `#${e.retryOf}`)),
      e.retrySuccessId && h('span.text-success', icon('check2-circle', 'me-1'), 'Fixed by retry ', h('a', { href: urls.execution(e.retrySuccessId) }, `#${e.retrySuccessId}`))));

  const tabBar = tabs([
    { id: 'canvas', label: 'Canvas', icon: 'diagram-3' },
    { id: 'nodes', label: 'Nodes', icon: 'list-nested', count: nodeSummaries(e).length },
    { id: 'timeline', label: 'Timeline', icon: 'bar-chart-steps' },
    { id: 'compare', label: 'Compare', icon: 'layout-split' },
    { id: 'data', label: 'Raw', icon: 'braces' },
  ], tab, (t) => { tab = t; ctx.setQuery({ tab: t }); show(); });

  mount(ctx.el, header, alerts, tabBar, body);
  drawTags();

  if (ACTIVE.has(e.status)) {
    ctx.every(2000, async () => {
      const fresh = await n8n.get(`/executions/${enc(id)}`, { includeData: true }).catch(() => null);
      if (fresh && (fresh.status !== e.status || fresh.stoppedAt !== e.stoppedAt || JSON.stringify(fresh.data?.resultData?.runData)?.length !== JSON.stringify(e.data?.resultData?.runData)?.length)) {
        e = fresh;
        if (!ACTIVE.has(e.status)) ctx.navigate(urls.execution(id), { replace: true });
        else show();
      }
    });
  }

  async function drawTags() {
    try { mount(tagsEl, tagPills(await n8n.get(`/executions/${enc(id)}/tags`))); } catch { /* tags unsupported */ }
  }

  function show(focusNode) {
    if (tab === 'canvas') return showCanvas();
    if (tab === 'nodes') return showNodes(focusNode);
    if (tab === 'timeline') return showTimeline();
    if (tab === 'compare') return showCompare();
    return mount(body, card({ body: jsonView(e, { open: 2 }) }), e.customData && Object.keys(e.customData).length ? card({ title: 'Custom data', icon: 'bookmark', body: jsonView(e.customData) }) : null);
  }

  function showCanvas() {
    const host = h('div.n8s-exec-canvas');
    mount(body, host);
    const cv = createCanvas(host, { readOnly: true, onOpenNode: (name) => { tab = 'nodes'; tabBar.setActive('nodes'); showNodes(name); } });
    ctx.cleanup(() => cv.destroy());
    cv.setWorkflow(wfData);
    cv.setStatus(nodeStatusMap(e), edgeCounts(e));
    requestAnimationFrame(() => cv.fitView());
  }

  function showNodes(focus) {
    const list = nodeSummaries(e);
    const listEl = h('div.n8s-runlist');
    const detail = h('div.n8s-rundetail');
    const pick = (s) => {
      listEl.querySelectorAll('.n8s-run').forEach((x) => x.classList.toggle('active', x.dataset.key === `${s.name}:${s.runIndex}`));
      const run = e.data.resultData.runData[s.name][s.runIndex];
      const node = (wfData.nodes || []).find((n) => n.name === s.name);
      const src = run.source?.[0];
      const inRun = src ? e.data.resultData.runData[src.previousNode]?.[src.previousNodeRun || 0] : null;
      const inItems = inRun?.data?.main?.[src.previousNodeOutput || 0] || [];
      mount(detail,
        h('div.n8s-rundetail-head', node && nodeIcon(node.type, { size: 28 }), h('div', h('b', s.name), h('div.small.text-muted', `${node?.type || ''} · run ${s.runIndex + 1}/${s.runs} · ${s.ms} ms · started ${new Date(s.start).toLocaleTimeString()}`)),
          h('div.flex-grow-1'), execBadge(s.status, { size: 'sm' })),
        s.error && h('div.n8s-ndv-error', icon('exclamation-octagon-fill', 'me-1'), h('div', h('b', s.error.message), s.error.description && h('div.small', stripHtml(s.error.description)), s.error.context?.data && h('pre.small', JSON.stringify(s.error.context.data, null, 2).slice(0, 2000)))),
        s.hints?.length ? h('div.n8s-hints', s.hints.map((x) => h('div', icon('lightbulb', 'me-1'), x.message))) : null,
        h('div.n8s-io',
          h('div', dataView({ items: inItems, title: h('span', icon('box-arrow-in-right', 'me-1'), 'Input', src && h('span.small.text-muted', ` from ${src.previousNode}`)), mappable: false, emptyText: 'No input (trigger or first node).' })),
          h('div', dataView({ outputs: (run.data?.main || [[]]).map((items, i) => ({ items: items || [], label: (run.data?.main?.length || 0) > 1 ? `Output ${i}` : null })), title: h('span', icon('box-arrow-right', 'me-1'), 'Output'), mappable: false, mode: 'table' }))),
        node && h('details.mt-2', h('summary.small', 'Node parameters at run time'), jsonView(node.parameters || {}, { open: 2 })));
    };
    mount(listEl, list.length ? list.map((s) => h('button.n8s-run', {
      type: 'button', dataset: { key: `${s.name}:${s.runIndex}` }, class: `st-${s.status}`, onclick: () => pick(s),
    }, h('span.n8s-dot', { class: `n8s-st-${s.status}` }), h('span.n8s-run-name', s.name, s.runs > 1 && h('span.text-muted', ` #${s.runIndex + 1}`)), h('span.n8s-run-meta', `${s.items} item${s.items === 1 ? '' : 's'} · ${s.ms}ms`))) : emptyHint('inbox', 'No node data', e.status === 'success' ? 'Execution data was not saved (check the workflow\'s save settings).' : 'This execution has no stored run data.'));
    mount(body, h('div.n8s-nodes-split', listEl, detail));
    const first = list.find((s) => s.name === focus) || list.find((s) => s.error) || list[list.length - 1];
    if (first) pick(first);
  }

  function showTimeline() {
    const list = nodeSummaries(e);
    if (!list.length) { mount(body, emptyHint('bar-chart-steps', 'No timing data')); return; }
    const t0 = Math.min(...list.map((s) => s.start));
    const t1 = Math.max(...list.map((s) => s.start + s.ms), t0 + 1);
    const span = t1 - t0;
    const trig = triggerNode(e);
    mount(body, card({
      title: `Waterfall · ${fmtDuration(span / 1000)} of node time`, icon: 'bar-chart-steps',
      body: h('div.n8s-gantt',
        h('div.n8s-gantt-axis', [0, 0.25, 0.5, 0.75, 1].map((p) => h('span', { style: { left: `${p * 100}%` } }, `${Math.round(span * p)} ms`))),
        list.map((s) => {
          const bar = h('div.n8s-gantt-bar', { class: `st-${s.status}`, style: { left: `${((s.start - t0) / span) * 100}%`, width: `${Math.max(0.4, (s.ms / span) * 100)}%` } });
          tip(bar, () => tipRows(s.name, [['Start', `+${s.start - t0} ms`], ['Duration', `${s.ms} ms`], ['Items', s.items], ['Run', `${s.runIndex + 1} of ${s.runs}`]]));
          return h('div.n8s-gantt-row', h('div.n8s-gantt-label', { title: s.name }, s.name === trig && icon('lightning-charge-fill', 'text-warning me-1'), s.name, s.runs > 1 && ` #${s.runIndex + 1}`), h('div.n8s-gantt-track', bar));
        }),
        h('div.small.text-muted.mt-2', 'Slowest: ', [...list].sort((a, b) => b.ms - a.ms).slice(0, 3).map((s) => `${s.name} (${s.ms} ms)`).join(', '))),
    }));
  }

  async function showCompare() {
    mount(body, spinner());
    const recent = await n8n.get('/executions', { workflowId: e.workflowId, limit: 50 }).catch(() => ({ data: [] }));
    const others = (recent.data || []).filter((x) => String(x.id) !== String(e.id));
    const defaultOther = others.find((x) => x.status === 'success') || others[0];
    if (!defaultOther) { mount(body, emptyHint('layout-split', 'Nothing to compare', 'This workflow has no other executions.')); return; }
    const sel = h('select.form-select.form-select-sm', { style: 'max-width:360px' }, others.map((x) => h('option', { value: x.id, selected: x === defaultOther }, `#${x.id} · ${x.status} · ${new Date(x.startedAt).toLocaleString()}`)));
    const out = h('div');
    const run = async () => {
      mount(out, spinner());
      const other = await n8n.get(`/executions/${enc(sel.value)}`, { includeData: true });
      const rows = compareExecutions(other, e);
      mount(out, h('table.af-table.af-table-dense.n8s-cmp',
        h('thead', h('tr', h('th', 'Node'), h('th', `#${other.id}`), h('th', `#${e.id} (this)`), h('th', 'Items'), h('th', 'Time'), h('th', 'First item differences'))),
        h('tbody', rows.map((r) => h('tr', { class: r.statusA !== r.statusB && 'n8s-cmp-diff' },
          h('td.fw-semibold', r.name),
          h('td', r.inA ? execBadge(r.statusA, { size: 'sm' }) : h('span.text-muted', 'did not run')),
          h('td', r.inB ? execBadge(r.statusB, { size: 'sm' }) : h('span.text-muted', 'did not run')),
          h('td', `${r.itemsA} → ${r.itemsB}`),
          h('td.small', r.msA != null && r.msB != null ? `${r.msA} → ${r.msB} ms` : '—'),
          h('td.small', r.diff.length ? h('details', h('summary', `${r.diff.length} field${r.diff.length === 1 ? '' : 's'}`), r.diff.slice(0, 40).map((d) => h('div.n8s-drow', h('span.n8s-dpath', pathString(d.path)), ' ', h('span.del', JSON.stringify(d.a)?.slice(0, 80)), ' → ', h('span.add', JSON.stringify(d.b)?.slice(0, 80))))) : h('span.text-muted', 'same')))))));
    };
    sel.addEventListener('change', run);
    mount(body, card({ title: 'Compare with another run', icon: 'layout-split', actions: sel, body: out }));
    run();
  }

  async function act(what, loadWorkflow) {
    try {
      if (what === 'retry') {
        const r = await n8n.post(`/executions/${enc(id)}/retry`, { loadWorkflow });
        toast(`Retried as #${r.id}`, 'success', { action: { label: 'Open', onClick: () => ctx.navigate(urls.execution(r.id)) } });
      } else if (what === 'stop') {
        await n8n.post(`/executions/${enc(id)}/stop`, {});
        toast('Stop requested', 'info');
        ctx.navigate(urls.execution(id), { replace: true });
      } else if (what === 'delete') {
        if (!(await confirmDialog({ title: 'Delete execution?', danger: true, confirmLabel: 'Delete', message: `Execution #${id} and its data will be removed.` }))) return;
        await n8n.del(`/executions/${enc(id)}`);
        toast('Execution deleted', 'success');
        ctx.navigate(urls.executions());
      }
    } catch (err2) { toastError(err2); }
  }

  async function debugInEditor() {
    const trig = triggerNode(e);
    const pins = toPinData(e, trig ? [trig] : undefined);
    if (!Object.keys(pins).length) { toast('This execution has no trigger output to pin.', 'warning'); return; }
    const ok = await confirmDialog({ title: 'Debug in editor', confirmLabel: 'Pin & open', message: h('span', 'Pins the trigger output of this run (', h('b', Object.keys(pins).join(', ')), ') onto the workflow and saves it, so the next manual run replays exactly this input.') });
    if (!ok) return;
    try {
      const current = await n8n.get(`/workflows/${enc(e.workflowId)}`);
      const saved = await n8n.put(`/workflows/${enc(e.workflowId)}`, saveBody({ ...current, pinData: { ...(current.pinData || {}), ...pins } }));
      patchWorkflow(saved);
      ctx.navigate(urls.workflow(e.workflowId, { execution: e.id }));
    } catch (err2) { toastError(err2, 'Could not pin data'); }
  }

  async function editTags() {
    const [all, cur] = await Promise.all([load('tags', { force: true }), n8n.get(`/executions/${enc(id)}/tags`).catch(() => [])]);
    const chosen = new Set((cur || []).map((t) => String(t.id)));
    modal({
      title: 'Execution tags', size: 'sm', icon: 'tags',
      body: h('div', all.length ? all.map((t) => h('label.d-flex.gap-2.align-items-center', h('input.form-check-input', { type: 'checkbox', checked: chosen.has(String(t.id)), onchange: (ev) => (ev.target.checked ? chosen.add(String(t.id)) : chosen.delete(String(t.id))) }), tagPills([t]))) : h('div.small.text-muted', 'Create tags on the Tags page first.')),
      actions: [{ label: 'Cancel' }, { label: 'Save', variant: 'primary', onClick: async (close) => { await n8n.put(`/executions/${enc(id)}/tags`, [...chosen].map((x) => ({ id: x }))); drawTags(); close(); } }],
    });
  }

  show();
}

/** Plain-language hints for the errors people hit most. */
function hintFor(message) {
  const m = String(message || '');
  const rules = [
    [/ECONNREFUSED|ENOTFOUND|EAI_AGAIN|getaddrinfo/i, 'The target host could not be reached from the n8n container — check the URL, DNS and docker network (localhost inside n8n is the n8n container itself).'],
    [/401|unauthori[sz]ed|invalid (api )?key|authentication/i, 'The service rejected the credentials. Re-check or re-connect the credential used by this node.'],
    [/403|forbidden|permission/i, 'The credential works but lacks permission/scopes for this operation.'],
    [/404|not found/i, 'The resource or endpoint does not exist — check ids and URLs (an expression may have resolved to an empty value).'],
    [/429|rate limit|too many requests/i, 'Rate-limited. Enable "Retry on fail" with a wait, or batch requests (Loop Over Items + Wait).'],
    [/timeout|ETIMEDOUT|ESOCKETTIMEDOUT/i, 'The call timed out. Increase the node timeout or the workflow timeout, or check the remote service.'],
    [/Cannot read propert(y|ies) of (undefined|null)|is not defined|undefined is not/i, 'An expression or Code node read a field that is missing on some items. Use optional chaining ($json.a?.b) or check the input on the Nodes tab.'],
    [/JSON|Unexpected token/i, 'A value is not valid JSON — check the body/expression; strings inside JSON need quotes.'],
    [/item.*paired|pairedItem|Can’?'?t get data for expression/i, 'Item linking broke (often after Merge/Code nodes). Reference the node with .first()/.all() or return pairedItem from Code nodes.'],
    [/out of memory|heap|crashed/i, 'The process ran out of memory. Split large datasets with Loop Over Items, or reduce binary data held in memory.'],
  ];
  const hit = rules.find(([re]) => re.test(m));
  return hit ? h('div.n8s-errhint', icon('lightbulb', 'me-1'), hit[1]) : null;
}

