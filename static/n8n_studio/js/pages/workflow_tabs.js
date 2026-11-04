// Secondary tabs of the workflow page: executions, versions (diff/restore), triggers &
// webhooks (tester, dependencies), settings, evaluations (test runs) and docs.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { toast, toastError, modal, card, dataTable, jsonEditor, spinner, copyButton, kv, timeAgo, chips, codeBlock, jsonView } from '../../../ui/js/ui.js';
import { fmtDate, fmtDuration, relTime } from '../../../ui/js/format.js';
import { sparkline } from '../../../ui/js/charts.js';
import { n8n, studio, enc, config, saveText } from '../core/api.js';
import { urls } from '../core/urls.js';
import * as W from '../core/workflow.js';
import { lineDiff, pathString } from '../core/jsondiff.js';
import { nextRuns, describeCron } from '../core/cron.js';
import { durationMs, FAILED, ACTIVE, statusInfo } from '../core/execution.js';
import { load } from '../core/store.js';
import { catalog, nodeIcon } from '../core/catalog.js';
import { execBadge, gated, comboBox, emptyHint } from '../core/widgets.js';
import { createCanvas } from '../editor/canvas.js';

export function renderTab(tab, el, api) {
  const fn = { executions, versions, triggers, settings, tests, docs }[tab];
  if (fn) fn(el, api); else mount(el, h('div', 'Unknown tab'));
}

// ------------------------------------------------------------------ executions

function executions(el, { state, ctx, loadExecution, showEditor }) {
  let status = '';
  let rows = [];
  let cursor = null;
  const listEl = h('div');
  const statsEl = h('div.n8s-wfx-stats');
  const fetchPage = async (reset) => {
    const page = await n8n.get('/executions', { workflowId: state.wf.id, limit: 50, status: status || undefined, cursor: reset ? undefined : cursor }, { signal: ctx.signal });
    rows = reset ? page.data : [...rows, ...page.data];
    cursor = page.nextCursor;
    draw();
  };
  const draw = () => {
    const done = rows.filter((r) => r.stoppedAt);
    const fails = rows.filter((r) => FAILED.has(r.status)).length;
    mount(statsEl,
      h('div.n8s-mini-stat', h('b', String(rows.length)), ' loaded'),
      h('div.n8s-mini-stat', h('b', rows.length ? `${Math.round(((rows.length - fails) / rows.length) * 100)}%` : '—'), ' success'),
      h('div.n8s-mini-stat', h('b', done.length ? fmtDuration(done.reduce((a, r) => a + durationMs(r), 0) / done.length / 1000) : '—'), ' avg'),
      done.length > 2 && h('div.n8s-mini-stat', sparkline(done.slice(0, 40).reverse().map((r) => durationMs(r) / 1000), { width: 140, height: 26 }), ' duration trend'));
    mount(listEl, dataTable({
      rows, dense: true,
      empty: emptyHint('lightning', 'No executions', status ? 'Nothing with this status.' : 'Run the workflow or wait for its trigger.'),
      columns: [
        { key: 'status', label: 'Status', render: (r) => execBadge(r.status, { size: 'sm' }) },
        { key: 'id', label: 'ID', render: (r) => h('a.af-mono', { href: urls.execution(r.id) }, `#${r.id}`) },
        { key: 'mode', label: 'Mode', render: (r) => h('span.n8s-mode', r.mode) },
        { key: 'startedAt', label: 'Started', render: (r) => timeAgo(r.startedAt) },
        { key: 'dur', label: 'Duration', align: 'right', render: (r) => (r.stoppedAt ? fmtDuration(durationMs(r) / 1000) : h('span.text-info', 'running')) },
        { key: 'retry', label: '', render: (r) => r.retryOf && h('span.small.text-muted', `retry of #${r.retryOf}`) },
        {
          key: 'act', label: '', align: 'right', render: (r) => h('div.btn-group.btn-group-sm',
            h('button.btn.btn-outline-secondary', { title: 'Show on canvas', onclick: async () => { try { await loadExecution(r.id); showEditor(); } catch (err) { toastError(err); } } }, icon('diagram-3')),
            FAILED.has(r.status) && h('button.btn.btn-outline-secondary', { title: 'Retry', onclick: async () => { try { const x = await n8n.post(`/executions/${enc(r.id)}/retry`, { loadWorkflow: true }); toast(`Retried as #${x.id}`, 'success'); fetchPage(true); } catch (err) { toastError(err, 'Retry failed'); } } }, icon('arrow-clockwise')),
            ACTIVE.has(r.status) && h('button.btn.btn-outline-danger', { title: 'Stop', onclick: async () => { await n8n.post(`/executions/${enc(r.id)}/stop`, {}).catch(toastError); fetchPage(true); } }, icon('stop-fill'))),
        },
      ],
      onRowClick: (r) => ctx.navigate(urls.execution(r.id)),
    }), cursor && h('div.text-center.mt-2', h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => fetchPage(false) }, 'Load more')));
  };
  mount(el, h('div.n8s-tabpad',
    h('div.d-flex.align-items-center.gap-2.mb-2.flex-wrap',
      chips([{ value: '', label: 'All' }, ...['success', 'error', 'running', 'waiting', 'canceled'].map((s) => ({ value: s, label: statusInfo(s).label, color: statusInfo(s).color }))], status, (v) => { status = v; fetchPage(true); }),
      h('div.flex-grow-1'), statsEl,
      h('a.btn.btn-sm.btn-outline-secondary', { href: urls.executions({ workflowId: state.wf.id }) }, icon('box-arrow-up-right', 'me-1'), 'Explorer')),
    listEl));
  mount(listEl, spinner());
  fetchPage(true).catch((err) => mount(listEl, h('div.text-danger', err.message)));
}

// ------------------------------------------------------------------ versions

async function versions(el, { state, ctx, apply, showEditor }) {
  mount(el, h('div.n8s-tabpad', spinner()));
  const [hist, snaps] = await Promise.all([
    n8n.all(`/workflows/${enc(state.wf.id)}/history`, {}, { signal: ctx.signal, max: 500 }).catch((err) => ({ error: err })),
    studio.get('/snapshots/', { workflow: state.wf.id, limit: 500 }).catch((err) => ({ error: err })),
  ]);
  const entries = [
    { key: 'current', label: 'Current editor state', at: new Date().toISOString(), source: 'editor', icon: 'pencil-square', load: async () => state.wf },
    ...(Array.isArray(hist) ? hist : []).map((v) => ({
      key: `n8n:${v.versionId}`, label: v.name || `Version ${v.versionId.slice(0, 8)}`, at: v.createdAt, who: v.authors, source: 'n8n', icon: 'clock-history',
      published: v.versionId === state.wf.activeVersionId,
      load: () => n8n.get(`/workflows/${enc(state.wf.id)}/${enc(v.versionId)}`),
    })),
    ...(snaps.data || []).map((s) => ({
      key: `studio:${s.id}`, label: `${s.name}`, at: s.createdAt, who: s.user, source: `studio · ${s.source}`, icon: 'camera', nodes: s.nodeCount, note: s.note,
      load: async () => (await studio.get(`/snapshots/${s.id}/`)).data,
    })),
  ].sort((a, b) => (a.key === 'current' ? -1 : b.key === 'current' ? 1 : b.at.localeCompare(a.at)));

  let left = entries[1] || null;
  let right = entries[0];
  const listEl = h('div.n8s-ver-list');
  const diffEl = h('div.n8s-ver-diff');

  const drawList = () => mount(listEl,
    hist.error && (hist.error.license ? h('div.small.text-muted.p-2', icon('info-circle', 'me-1'), 'n8n history is licence-limited here; Studio snapshots below keep everything saved through Studio.') : h('div.small.text-danger.p-2', hist.error.message)),
    entries.map((e) => h('button.n8s-ver-item', {
      type: 'button',
      class: [e === left && 'sel-a', e === right && 'sel-b'],
      title: 'Click = compare FROM this version · Shift+click = compare TO this version',
      onclick: (ev) => { if (ev.shiftKey) right = e; else left = e; drawList(); drawDiff(); },
    }, icon(e.icon),
    h('div.n8s-ver-text', h('div.n8s-ver-label', e.label, e.published && h('span.n8s-pill-live', 'published')),
      h('div.n8s-ver-sub', e.key === 'current' ? 'unsaved edits included' : `${relTime(e.at)} · ${e.source}${e.who ? ` · ${e.who}` : ''}${e.nodes != null ? ` · ${e.nodes} nodes` : ''}`)),
    e === left && h('span.n8s-ver-tag.a', 'A'), e === right && h('span.n8s-ver-tag.b', 'B'))));

  async function drawDiff() {
    if (!left || !right) { mount(diffEl, emptyHint('clock-history', 'No history yet', 'Save the workflow through Studio to start recording versions.')); return; }
    mount(diffEl, spinner());
    let a; let b;
    try { [a, b] = await Promise.all([left.load(), right.load()]); } catch (err) { mount(diffEl, h('div.text-danger', err.message)); return; }
    const d = W.diffWorkflows(a, b);
    const nodeChanges = d.changed.filter((c) => c.changes.length || c.renamed);
    mount(diffEl,
      h('div.n8s-ver-head',
        h('div', h('span.n8s-ver-tag.a', 'A'), ' ', left.label, ' ', icon('arrow-right'), ' ', h('span.n8s-ver-tag.b', 'B'), ' ', right.label),
        h('div.d-flex.gap-2',
          h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => previewCanvas(a, left.label) }, icon('eye', 'me-1'), 'Preview A'),
          left.key !== 'current' && h('button.btn.btn-sm.btn-primary', {
            onclick: () => {
              apply({ ...state.wf, name: a.name || state.wf.name, nodes: W.clone(a.nodes), connections: W.clone(a.connections), ...(a.settings ? { settings: W.clone(a.settings) } : {}), ...(a.pinData ? { pinData: W.clone(a.pinData) } : {}) }, { label: `Restore ${left.label}` });
              toast('Version loaded into the editor — review it, then Save to make it current (undo with Ctrl+Z).', 'success', { timeout: 6000 });
              showEditor();
            },
          }, icon('arrow-counterclockwise', 'me-1'), 'Restore A into editor'))),
      d.empty ? h('div.n8s-ver-same', icon('check2-circle', 'me-1'), 'No differences.') : h('div',
        h('div.n8s-diff-summary',
          d.name && h('span.n8s-dchip.mod', `renamed “${d.name.from}” → “${d.name.to}”`),
          h('span.n8s-dchip.add', `+${d.added.length} nodes`), h('span.n8s-dchip.del', `−${d.removed.length} nodes`), h('span.n8s-dchip.mod', `~${nodeChanges.length} changed`),
          h('span.n8s-dchip', `${d.connections.added.length + d.connections.removed.length} connection changes`),
          d.settings.length && h('span.n8s-dchip', `${d.settings.length} settings`)),
        d.added.map((n) => h('div.n8s-dnode.add', icon('plus-circle'), ' ', n)),
        d.removed.map((n) => h('div.n8s-dnode.del', icon('dash-circle'), ' ', n)),
        nodeChanges.map((c) => h('details.n8s-dnode.mod', { open: nodeChanges.length < 6 },
          h('summary', icon('pencil'), ' ', c.renamed ? `${c.from} → ${c.name}` : c.name, h('span.text-muted.small.ms-2', `${c.changes.length} change${c.changes.length === 1 ? '' : 's'}`)),
          c.changes.map((x) => diffRow(x)))),
        (d.connections.added.length || d.connections.removed.length) && h('div.n8s-dconn',
          d.connections.added.map((e) => h('div.add', '+ ', e.from, ' → ', e.to, e.fromType !== 'main' ? ` (${e.fromType})` : '')),
          d.connections.removed.map((e) => h('div.del', '− ', e.from, ' → ', e.to, e.fromType !== 'main' ? ` (${e.fromType})` : ''))),
        d.settings.length && h('div.n8s-dconn', h('b.small', 'Settings'), d.settings.map(diffRow))));
  }

  function diffRow(x) {
    const a = x.a;
    const b = x.b;
    if (typeof a === 'string' && typeof b === 'string' && (a.includes('\n') || b.includes('\n'))) {
      return h('div.n8s-drow', h('div.n8s-dpath', pathString(x.path)), h('pre.n8s-linediff', lineDiff(a, b).map((l) => h('div', { class: l.op === '+' ? 'add' : l.op === '-' ? 'del' : '' }, `${l.op} ${l.text}`))));
    }
    const show = (v) => (v === undefined ? h('i.text-muted', 'unset') : h('code', typeof v === 'string' ? v : JSON.stringify(v)));
    return h('div.n8s-drow', h('div.n8s-dpath', pathString(x.path)), h('div.n8s-dvals', x.kind !== 'added' && h('span.del', show(a)), x.kind !== 'removed' && h('span.add', show(b))));
  }

  mount(el, h('div.n8s-tabpad.n8s-ver', h('div.n8s-ver-side', h('div.n8s-ver-hint', icon('info-circle', 'me-1'), 'Click picks A, Shift+click picks B. Studio snapshots every save — no 24-hour limit.'), listEl), diffEl));
  drawList();
  drawDiff();
}

function previewCanvas(wf, title) {
  const host = h('div.n8s-preview-canvas');
  modal({ title: `Preview — ${title}`, size: 'xl', icon: 'eye', body: host });
  requestAnimationFrame(() => {
    const cv = createCanvas(host, { readOnly: true });
    cv.setWorkflow(wf, { fit: true });
  });
}

// ------------------------------------------------------------------ triggers & webhooks

async function triggers(el, { state }) {
  const wf = state.wf;
  const trig = W.triggers(wf, catalog.nodes);
  const hooks = W.webhooks(wf);
  const base = (config.webhookUrl || config.n8nUrl || '').replace(/\/$/, '');
  const all = await load('workflows').catch(() => []);
  const errorUsers = all.filter((w) => w.settings?.errorWorkflow === wf.id);
  const callers = all.filter((w) => (w.nodes || []).some((n) => /executeWorkflow$/.test(n.type) && JSON.stringify(n.parameters || {}).includes(wf.id)));
  const callees = (wf.nodes || []).filter((n) => /executeWorkflow$/.test(n.type)).map((n) => {
    const v = n.parameters?.workflowId;
    const id = typeof v === 'object' ? v?.value : v;
    return { node: n.name, id, wf: all.find((w) => String(w.id) === String(id)) };
  });

  const trigCards = trig.map((n) => {
    const sched = W.scheduleSummary(n);
    const crons = (n.parameters?.rule?.interval || []).filter((r) => r.field === 'cronExpression').map((r) => r.expression);
    return h('div.n8s-trig', nodeIcon(n.type, { size: 34 }),
      h('div.flex-grow-1', h('div.fw-semibold', n.name, n.disabled && h('span.badge.text-bg-secondary.ms-2', 'disabled')), h('div.small.text-muted', n.type),
        sched && h('div.small', icon('clock', 'me-1'), sched),
        crons.map((c) => {
          let runs = [];
          try { runs = nextRuns(c, 5); } catch { /* invalid cron */ }
          return h('div.small.mt-1', h('code', c), ' — ', describeCron(c), runs.length ? h('div.text-muted', 'Next (your time): ', runs.map((d) => fmtDate(d, { seconds: false })).join(' · ')) : h('span.text-danger', ' invalid'));
        })));
  });

  const hookCards = hooks.map((w) => {
    const prod = `${base}/${w.prefix}/${w.path}`;
    const test = `${base}/${w.prefix === 'webhook' ? 'webhook-test' : `${w.prefix}-test`}/${w.path}`;
    const method = h('select.form-select.form-select-sm', { style: 'width:110px' }, ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD'].map((m) => h('option', { selected: m === w.methods[0] }, m)));
    const mode = h('select.form-select.form-select-sm', { style: 'width:150px' }, h('option', { value: 'production' }, 'Production URL'), h('option', { value: 'test' }, 'Test URL'));
    const body = jsonEditor({ hello: 'world' }, { rows: 6 });
    const headers = jsonEditor({}, { rows: 3 });
    const query = h('input.form-control.form-control-sm', { placeholder: 'query string, e.g. a=1&b=2' });
    const out = h('div.n8s-hook-out');
    const curl = () => `curl -X ${method.value} '${mode.value === 'test' ? test : prod}${query.value ? `?${query.value}` : ''}' -H 'Content-Type: application/json' -d '${JSON.stringify(safeJson(body)).replace(/'/g, "'\\''")}'`;
    const send = async () => {
      mount(out, spinner());
      try {
        const r = await studio.post('/webhook/', {
          path: w.path, method: method.value, mode: mode.value, body: safeJson(body), headers: safeJson(headers) || {},
          query: Object.fromEntries(new URLSearchParams(query.value)),
        });
        mount(out, h('div.n8s-hook-res', { class: r.status < 400 ? 'ok' : 'bad' },
          h('div.d-flex.gap-3.align-items-center', h('b', `HTTP ${r.status}`), h('span.text-muted', `${r.ms} ms`), h('span.text-muted.small', r.contentType)),
          r.json !== null && r.json !== undefined ? jsonView(r.json, { open: 2 }) : h('pre.small.mb-0', r.text || '(empty body)')));
      } catch (err) { mount(out, h('div.text-danger', err.message)); }
    };
    return h('div.n8s-hook',
      h('div.n8s-hook-top', h('span.n8s-hook-method', w.methods.join(' / ')), h('b', w.node), h('span.text-muted.small', `auth: ${w.auth} · response: ${w.responseMode}`),
        !wf.active && h('span.n8s-pill-warn', icon('exclamation-triangle', 'me-1'), 'inactive — production URL answers 404 until activated')),
      h('div.n8s-hook-url', h('span.n8s-hook-k', 'Production'), h('code', prod), copyButton(prod)),
      h('div.n8s-hook-url', h('span.n8s-hook-k', 'Test'), h('code', test), copyButton(test)),
      h('details.n8s-hook-tester', h('summary', icon('send', 'me-1'), 'Send a test request'),
        h('div.d-flex.gap-2.my-2.flex-wrap', method, mode, query, h('button.btn.btn-sm.btn-primary', { onclick: send }, icon('send', 'me-1'), 'Send'), h('button.btn.btn-sm.btn-outline-secondary', { onclick: async () => { await navigator.clipboard.writeText(curl()); toast('curl copied', 'success', { timeout: 1200 }); } }, icon('terminal', 'me-1'), 'Copy as curl')),
        h('div.row.g-2', h('div.col-md-7', h('label.small.text-muted', 'JSON body'), body.el), h('div.col-md-5', h('label.small.text-muted', 'Headers'), headers.el)),
        h('div.small.text-muted.mt-1', 'Test URLs only listen while n8n waits for a test event (after clicking Run on a workflow with a webhook trigger).'),
        out));
  });

  mount(el, h('div.n8s-tabpad.n8s-grid-2',
    card({ title: `Triggers (${trig.length})`, icon: 'lightning-charge', body: trigCards.length ? trigCards : emptyHint('lightning', 'No trigger', 'Runs only as a sub-workflow or manually.') }),
    card({
      title: 'Dependencies', icon: 'diagram-2',
      body: h('div.n8s-deps',
        h('div.n8s-dep-k', 'Error workflow'), h('div', wf.settings?.errorWorkflow ? h('a', { href: urls.workflow(wf.settings.errorWorkflow) }, all.find((w) => w.id === wf.settings.errorWorkflow)?.name || wf.settings.errorWorkflow) : h('span.text-warning', 'none — failures are silent')),
        h('div.n8s-dep-k', 'Used as error workflow by'), h('div', errorUsers.length ? errorUsers.map((w) => h('a.d-block', { href: urls.workflow(w.id) }, w.name)) : '—'),
        h('div.n8s-dep-k', 'Calls sub-workflows'), h('div', callees.length ? callees.map((c) => h('div', c.node, ' → ', c.wf ? h('a', { href: urls.workflow(c.wf.id) }, c.wf.name) : h('span.text-danger', c.id ? `missing (${c.id})` : 'dynamic'))) : '—'),
        h('div.n8s-dep-k', 'Called by'), h('div', callers.length ? callers.map((w) => h('a.d-block', { href: urls.workflow(w.id) }, w.name)) : '—')),
    }),
    hooks.length ? h('div.n8s-span-2', card({ title: `Webhooks (${hooks.length})`, icon: 'broadcast', body: hookCards })) : null));
}

const safeJson = (editor) => { try { return editor.value(); } catch { return null; } };

// ------------------------------------------------------------------ settings

async function settings(el, { state, save, apply }) {
  const s = { ...(state.wf.settings || {}) };
  const all = await load('workflows').catch(() => []);
  const tzs = typeof Intl.supportedValuesOf === 'function' ? Intl.supportedValuesOf('timeZone') : ['UTC', 'Africa/Nairobi', 'Europe/London', 'America/New_York'];
  const field = (label, input, help) => h('div.n8s-set-row', h('div', h('div.fw-semibold.small', label), help && h('div.small.text-muted', help)), h('div', input));
  const select = (key, options, def = 'DEFAULT') => {
    const sel = h('select.form-select.form-select-sm', options.map(([v, l]) => h('option', { value: v, selected: String(s[key] ?? def) === String(v) }, l)));
    sel.addEventListener('change', () => { if (sel.value === 'DEFAULT') delete s[key]; else s[key] = sel.value === 'true' ? true : sel.value === 'false' ? false : sel.value; });
    return sel;
  };
  const errCombo = comboBox({
    options: all.filter((w) => w.id !== state.wf.id).map((w) => ({ value: w.id, label: w.name, sub: (w.nodes || []).some((n) => /errorTrigger$/.test(n.type)) ? 'has Error Trigger ✓' : 'no Error Trigger node' })),
    value: s.errorWorkflow || '', emptyLabel: '— none —', onChange: (v) => { if (v) s.errorWorkflow = v; else delete s.errorWorkflow; },
  });
  const timeout = h('input.form-control.form-control-sm', { type: 'number', value: s.executionTimeout ?? -1, min: -1, style: 'width:140px' });
  timeout.addEventListener('input', () => { s.executionTimeout = Number(timeout.value); });
  const saved = h('input.form-control.form-control-sm', { type: 'number', min: 0, value: s.timeSavedPerExecution ?? '', placeholder: 'minutes', style: 'width:140px' });
  saved.addEventListener('input', () => { if (saved.value === '') delete s.timeSavedPerExecution; else s.timeSavedPerExecution = Number(saved.value); });
  const callerIds = h('input.form-control.form-control-sm', { value: s.callerIds || '', placeholder: 'comma-separated workflow ids' });
  callerIds.addEventListener('input', () => { s.callerIds = callerIds.value; });
  const desc = h('textarea.form-control.form-control-sm', { rows: 3, placeholder: 'What does this workflow do? Shown in lists and docs.' }, state.wf.description || '');
  const tz = h('select.form-select.form-select-sm', h('option', { value: 'DEFAULT' }, 'Instance default'), tzs.map((t) => h('option', { value: t, selected: s.timezone === t }, t)));
  tz.addEventListener('change', () => { if (tz.value === 'DEFAULT') delete s.timezone; else s.timezone = tz.value; });

  const staticData = state.wf.staticData;
  mount(el, h('div.n8s-tabpad', card({
    title: 'Workflow settings', icon: 'gear',
    actions: h('button.btn.btn-sm.btn-primary', {
      onclick: async () => {
        apply({ ...state.wf, settings: s, description: desc.value || undefined }, { label: 'Settings' });
        try { await save(); } catch { /* toast shown */ }
      },
    }, icon('cloud-check', 'me-1'), 'Save settings'),
    body: h('div.n8s-set',
      field('Description', desc),
      field('Error workflow', errCombo.el, 'Runs when this workflow fails. It should start with an Error Trigger node.'),
      field('Timezone', tz, 'Used by Schedule triggers and date expressions.'),
      field('Save failed executions', select('saveDataErrorExecution', [['DEFAULT', 'Default'], ['all', 'Save'], ['none', 'Do not save']])),
      field('Save successful executions', select('saveDataSuccessExecution', [['DEFAULT', 'Default'], ['all', 'Save'], ['none', 'Do not save']]), 'Turning this off keeps the database small for chatty workflows.'),
      field('Save manual executions', select('saveManualExecutions', [['DEFAULT', 'Default'], ['true', 'Save'], ['false', 'Do not save']])),
      field('Save execution progress', select('saveExecutionProgress', [['DEFAULT', 'Default'], ['true', 'Save (allows resuming)'], ['false', 'Do not save']])),
      field('Timeout (seconds)', timeout, '-1 = no timeout.'),
      field('Execution order', select('executionOrder', [['v1', 'v1 (recommended)'], ['v0', 'v0 (legacy)']], 'v1')),
      field('Who can call this as a sub-workflow', select('callerPolicy', [['DEFAULT', 'Default'], ['any', 'Any workflow'], ['none', 'No workflows'], ['workflowsFromSameOwner', 'Workflows from the same owner'], ['workflowsFromAList', 'Selected workflows']])),
      field('Allowed caller ids', callerIds, 'Only for "Selected workflows".'),
      field('Time saved per execution', saved, 'Feeds the Insights "time saved" metric.'),
      field('Available in MCP', select('availableInMCP', [['DEFAULT', 'Default'], ['true', 'Yes'], ['false', 'No']]), 'Expose to MCP clients (instance-level MCP must be on).')),
  }),
  card({
    title: 'Static data', icon: 'database',
    actions: staticData && h('button.btn.btn-sm.btn-outline-danger', { onclick: () => { apply({ ...state.wf, staticData: null }, { label: 'Clear static data' }); toast('Static data will be cleared on save (polling triggers will re-fetch from scratch).', 'warning'); } }, icon('eraser', 'me-1'), 'Clear'),
    body: staticData ? jsonView(staticData, { open: 2 }) : h('div.small.text-muted', 'No static data (used by polling triggers to remember what they have seen).'),
  })));
}

// ------------------------------------------------------------------ evaluations (test runs)

function tests(el, { state, ctx }) {
  const body = h('div');
  const refresh = () => gated(body, async () => {
    const runs = await n8n.all(`/workflows/${enc(state.wf.id)}/test-runs`, {}, { signal: ctx.signal, max: 200 });
    if (runs.some((r) => ['new', 'running'].includes(r.status))) setTimeout(() => { if (ctx.active && state.tab === 'tests') refresh(); }, 3000);
    return dataTable({
      rows: runs,
      empty: emptyHint('clipboard-check', 'No test runs yet', h('span', 'Evaluations run this workflow over a dataset (an ', h('b', 'Evaluation Trigger'), ' node) and score the outputs with ', h('b', 'Evaluation'), ' nodes.')),
      columns: [
        { key: 'status', label: 'Status', render: (r) => h('span.n8s-badge', { class: `n8s-st-${r.status === 'completed' ? 'success' : r.status === 'error' ? 'error' : r.status === 'running' ? 'running' : 'unknown'}` }, r.status) },
        { key: 'runAt', label: 'Started', render: (r) => timeAgo(r.runAt || r.createdAt) },
        { key: 'cases', label: 'Cases', render: (r) => r.testCaseCount ?? '—' },
        { key: 'final', label: 'Result', render: (r) => r.finalResult || '—' },
        { key: 'metrics', label: 'Metrics', render: (r) => (r.metrics ? h('span.n8s-metrics', Object.entries(r.metrics).map(([k, v]) => h('span.n8s-metric', `${k}: ${typeof v === 'number' ? Math.round(v * 1000) / 1000 : v}`))) : '—') },
        { key: 'err', label: '', render: (r) => r.errorCode && h('span.text-danger.small', r.errorCode) },
        { key: 'act', label: '', align: 'right', render: (r) => ['new', 'running'].includes(r.status) && h('button.btn.btn-sm.btn-outline-danger', { onclick: async () => { await n8n.post(`/workflows/${enc(state.wf.id)}/test-runs/${enc(r.id)}/cancel`, {}).catch(toastError); refresh(); } }, 'Cancel') },
      ],
      onRowClick: (r) => showRun(r),
    });
  });
  async function showRun(r) {
    const m = modal({ title: `Test run ${r.id}`, size: 'xl', icon: 'clipboard-check', body: spinner() });
    try {
      const [run, cases] = await Promise.all([n8n.get(`/workflows/${enc(state.wf.id)}/test-runs/${enc(r.id)}`), n8n.all(`/workflows/${enc(state.wf.id)}/test-runs/${enc(r.id)}/test-cases`, {}, { max: 1000 })]);
      mount(m.body, kv([['Status', run.status], ['Result', run.finalResult || '—'], ['Cases', run.testCaseCount ?? cases.length], ['Completed', run.completedAt ? fmtDate(run.completedAt) : '—'], ['Error', run.errorCode || '—', { wide: true }]]),
        run.metrics && h('div.my-2', jsonView(run.metrics)),
        dataTable({
          rows: cases, dense: true,
          columns: [
            { key: 'status', label: 'Status', render: (c) => c.status },
            { key: 'metrics', label: 'Metrics', render: (c) => h('code.small', JSON.stringify(c.metrics || {})) },
            { key: 'exec', label: 'Execution', render: (c) => c.executionId && h('a', { href: urls.execution(c.executionId) }, `#${c.executionId}`) },
            { key: 'err', label: 'Error', render: (c) => c.errorCode || '' },
          ],
        }));
    } catch (err) { mount(m.body, h('div.text-danger', err.message)); }
  }
  mount(el, h('div.n8s-tabpad', card({
    title: 'Evaluations', icon: 'clipboard-check',
    actions: h('button.btn.btn-sm.btn-primary', { onclick: async () => { try { await n8n.post(`/workflows/${enc(state.wf.id)}/test-runs`, {}); toast('Test run started', 'success'); refresh(); } catch (err) { toastError(err, 'Could not start a test run'); } } }, icon('play-fill', 'me-1'), 'Start test run'),
    body,
  })));
  refresh();
}

// ------------------------------------------------------------------ docs

function docs(el, { state }) {
  const md = W.workflowDocs(state.wf, catalog.nodes);
  mount(el, h('div.n8s-tabpad', card({
    title: 'Generated documentation', icon: 'journal-text',
    actions: h('div.d-flex.gap-2',
      copyButton(md, 'Copy Markdown'),
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => saveText(md, `${(state.wf.name || 'workflow').replace(/\W+/g, '-')}.md`, 'text/markdown') }, icon('download', 'me-1'), '.md'),
      copyButton(W.toMermaid(state.wf), 'Copy Mermaid')),
    body: markdown(md),
  })));
}

/** Small, safe Markdown renderer (headings, lists, tables, code, bold, links). */
export function markdown(text) {
  const root = h('div.n8s-md');
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('```')) {
      const lang = line.slice(3).trim();
      const buf = [];
      for (i++; i < lines.length && !lines[i].startsWith('```'); i++) buf.push(lines[i]);
      root.appendChild(codeBlock(buf.join('\n'), { lang: lang || 'text' }));
    } else if (/^#{1,6} /.test(line)) {
      const n = line.match(/^#+/)[0].length;
      root.appendChild(h(`h${Math.min(6, n + 1)}`, inlineMd(line.slice(n + 1))));
    } else if (/^\|/.test(line)) {
      const rows = [];
      for (; i < lines.length && /^\|/.test(lines[i]); i++) rows.push(lines[i]);
      i--;
      const cells = rows.filter((r) => !/^\|[-|\s]+\|$/.test(r)).map((r) => r.split('|').slice(1, -1).map((c) => c.trim()));
      root.appendChild(h('table.af-table.af-table-dense', h('thead', h('tr', cells[0].map((c) => h('th', inlineMd(c))))), h('tbody', cells.slice(1).map((r) => h('tr', r.map((c) => h('td', inlineMd(c))))))));
    } else if (/^\s*(-|\d+\.) /.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      for (; i < lines.length && /^\s*(-|\d+\.) |^ {3}>/.test(lines[i]); i++) {
        if (/^ {3}>/.test(lines[i])) items[items.length - 1]?.appendChild(h('div.n8s-md-quote', inlineMd(lines[i].replace(/^ {3}> ?/, ''))));
        else items.push(h('li', inlineMd(lines[i].replace(/^\s*(-|\d+\.) /, ''))));
      }
      i--;
      root.appendChild(h(ordered ? 'ol' : 'ul', items));
    } else if (line.trim()) root.appendChild(h('p', inlineMd(line)));
  }
  return root;
}

function inlineMd(text) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|_[^_]+_|\[[^\]]+\]\((https?:[^)\s]+)\))/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith('**')) out.push(h('b', t.slice(2, -2)));
    else if (t.startsWith('`')) out.push(h('code', t.slice(1, -1)));
    else if (t.startsWith('[')) out.push(h('a', { href: m[2], target: '_blank', rel: 'noopener noreferrer' }, t.slice(1, t.indexOf(']'))));
    else out.push(h('i', t.slice(1, -1)));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

