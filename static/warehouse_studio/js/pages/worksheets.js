// Worksheets IDE — multi-tab SQL editor with object browser, async execution,
// virtualised results, chart builder, cost preview, per-run diagnosis and history.

import {
  h, icon, mount, card, banner, empty, drawer, modal, confirm, popover, copyButton, skeleton, spinner, toast, errorView, seg, sevBadge,
} from '../lib/ui.js';
import { api, waitForQuery, sleep } from '../lib/api.js';
import { state, setContext, on } from '../lib/store.js';
import { chart, axisStyle, ColorScale, tipRow, escapeHtml, hexA, hbars } from '../lib/charts.js';
import { waterfall, insightsList, renderProfile, openQuery } from '../lib/query.js';
import * as fmt from '../lib/fmt.js';
import { prefs } from '../../../ui/js/dom.js';
import { splitStatements, statementAt, errorPosInDoc, isExplainable } from '../lib/sqlsplit.js';
import { createGrid, inspectText } from '../lib/grid.js';
import { profileColumn, toCSV, toTSV, toMarkdown, toJSON, autoChart, chartSeries, cellText, cellValue } from '../lib/gridcore.js';
import { createEditor, formatSql, hintTablesFrom, fqName, ident, modKey } from '../lib/editor.js';

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `t${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`);
const TAB_KEY = 'ws:ide:tabs';
const ACTIVE_KEY = 'ws:ide:active';
const SPLIT_KEY = 'ws:ide:split';
const SIDE_KEY = 'ws:ide:side';
const EXPLAIN_WARN_BYTES = 10 * 1024 ** 3;

const WELCOME = `-- Welcome to Warehouse Studio worksheets.
-- ${'Ctrl/⌘'}+Enter runs the statement under the cursor, Shift+${'Ctrl/⌘'}+Enter runs them all.
-- ${'Ctrl/⌘'}+E previews what a query will cost before it runs.

SELECT CURRENT_ROLE() AS role, CURRENT_WAREHOUSE() AS warehouse, CURRENT_DATABASE() AS database;
`;

export default async function render(ctx) {
  ctx.setTitle('Worksheets');

  // ------------------------------------------------------------------ tab model
  const saved = prefs.get(TAB_KEY, null);
  const tabs = (Array.isArray(saved) && saved.length ? saved : [newTabData({ title: 'Worksheet 1', sql: WELCOME })])
    .map((t) => ({ ...t, results: [], activeResult: 0, running: null, view: 'results' }));
  let activeId = prefs.get(ACTIVE_KEY, tabs[0].id);
  if (!tabs.find((t) => t.id === activeId)) activeId = tabs[0].id;
  const active = () => tabs.find((t) => t.id === activeId);

  function newTabData({ title, sql = '', worksheetId = null, context = null }) {
    return { id: uid(), title: title || `Worksheet ${Date.now() % 1000}`, sql, worksheetId, dirty: !worksheetId && !!sql && sql !== WELCOME, context: context || { ...state.context } };
  }
  const persist = () => {
    prefs.set(TAB_KEY, tabs.map(({ id, title, sql, worksheetId, dirty, context }) => ({ id, title, sql: sql.length > 400_000 ? sql.slice(0, 400_000) : sql, worksheetId, dirty, context })));
    prefs.set(ACTIVE_KEY, activeId);
  };

  // ------------------------------------------------------------------ layout
  const sideEl = h('div.ws-card.ws-ide-side');
  const tabsEl = h('div.ws-ide-tabs');
  const toolbarEl = h('div.ws-ide-toolbar');
  const editorEl = h('div.ws-ide-editor');
  const resizer = h('div.ws-ide-resizer', { title: 'Drag to resize' });
  const resultsEl = h('div.ws-ide-results');
  const main = h('div.ws-card.ws-ide-main', tabsEl, toolbarEl, editorEl, resizer, resultsEl);
  const ide = h(`div.ws-ide${prefs.get(SIDE_KEY, true) ? '' : '.ws-side-hidden'}`, sideEl, main);
  mount(ctx.el, ide);

  const split = Number(prefs.get(SPLIT_KEY, 0.5)) || 0.5;
  const applySplit = (r) => { editorEl.style.flex = `${r} 1 0`; resultsEl.style.flex = `${1 - r} 1 0`; };
  applySplit(split);
  resizer.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const rect = main.getBoundingClientRect();
    const top = editorEl.getBoundingClientRect().top;
    const total = rect.bottom - top - resizer.offsetHeight;
    resizer.classList.add('drag');
    const move = (ev) => { const r = Math.max(0.15, Math.min(0.85, (ev.clientY - top) / total)); applySplit(r); prefs.set(SPLIT_KEY, r); editor.refresh(); };
    const up = () => { resizer.classList.remove('drag'); document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); editor.refresh(); };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  });

  // ------------------------------------------------------------------ editor
  const docs = new Map();
  const editor = createEditor(editorEl, {
    value: active().sql,
    onChange: (v) => {
      const t = active();
      if (!t || t.sql === v) return;
      t.sql = v;
      if (!t.dirty) { t.dirty = true; renderTabs(); }
      persist();
      scheduleAutosave(t);
    },
    onRun: () => runCurrent(),
    onRunAll: () => runAll(),
    onSave: () => saveTab(active()),
    onExplain: () => explainCurrent(),
    onFormat: () => formatCurrent(),
  });
  if (editor.cm && window.CodeMirror?.Doc) docs.set(activeId, editor.cm.getDoc());
  ctx.cleanup(() => editor.destroy());

  function switchTab(id) {
    if (id === activeId) return;
    activeId = id;
    const t = active();
    if (editor.cm && window.CodeMirror?.Doc) {
      if (!docs.has(id)) docs.set(id, window.CodeMirror.Doc(t.sql, 'text/x-snowflake'));
      editor.cm.swapDoc(docs.get(id));
    } else {
      editor.setValue(t.sql);
    }
    editor.clearErrors();
    editor.clearRunning();
    if (t.context) setContext({ ...t.context });
    persist();
    renderTabs();
    renderToolbar();
    renderResults();
    editor.focus();
  }

  function openTab({ title, sql = '', worksheetId = null, context = null, run = false }) {
    if (worksheetId) {
      const existing = tabs.find((t) => t.worksheetId === worksheetId);
      if (existing) { switchTab(existing.id); return existing; }
    }
    const t = { ...newTabData({ title, sql, worksheetId, context }), results: [], activeResult: 0, running: null, view: 'results' };
    tabs.push(t);
    switchTab(t.id);
    if (run) setTimeout(() => runAll(), 50);
    return t;
  }

  async function closeTab(id) {
    const t = tabs.find((x) => x.id === id);
    if (!t) return;
    if (t.dirty && t.sql.trim() && t.sql !== WELCOME) {
      const ok = await confirm({ title: `Close “${t.title}”?`, text: t.worksheetId ? 'Unsaved edits will be lost.' : 'This worksheet was never saved. Close it anyway?', confirmLabel: 'Close', danger: true });
      if (!ok) return;
    }
    t.running?.abort?.abort();
    const idx = tabs.indexOf(t);
    tabs.splice(idx, 1);
    docs.delete(id);
    if (!tabs.length) tabs.push({ ...newTabData({ title: 'Worksheet 1', sql: '' }), results: [], activeResult: 0, running: null, view: 'results' });
    if (activeId === id) {
      activeId = '';
      switchTab(tabs[Math.max(0, idx - 1)].id);
    } else { persist(); renderTabs(); }
  }

  function renderTabs() {
    mount(tabsEl, tabs.map((t) => {
      const titleEl = h('span.ws-ide-tab-title', t.title);
      const el = h(`div.ws-ide-tab${t.id === activeId ? '.on' : ''}`, { title: t.worksheetId ? `${t.title} (saved worksheet)` : `${t.title} (local draft)`,
        onclick: () => switchTab(t.id), onauxclick: (e) => { if (e.button === 1) closeTab(t.id); } },
      t.running ? h('span.ws-run-dot', { title: 'Running' }) : icon(t.worksheetId ? 'file-earmark-code' : 'file-earmark'),
      titleEl,
      t.dirty && h('span.ws-dirty', { title: 'Unsaved changes' }),
      h('button.ws-ide-tab-x', { title: 'Close', onclick: (e) => { e.stopPropagation(); closeTab(t.id); } }, icon('x-lg')));
      titleEl.addEventListener('dblclick', (e) => { e.stopPropagation(); renameInline(t, titleEl); });
      return el;
    }), h('button.ws-ide-newtab', { title: 'New worksheet', onclick: () => openTab({ title: `Worksheet ${tabs.length + 1}` }) }, icon('plus-lg')));
  }

  function renameInline(t, titleEl) {
    const input = h('input', { value: t.title });
    titleEl.replaceWith(input);
    input.focus();
    input.select();
    const done = (commit) => {
      if (commit && input.value.trim()) {
        t.title = input.value.trim().slice(0, 200);
        if (t.worksheetId) api.updateWorksheet(t.worksheetId, { title: t.title }).then(() => loadWorksheets()).catch((err) => toast(err.message, 'error'));
        persist();
      }
      renderTabs();
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') done(true); if (e.key === 'Escape') done(false); e.stopPropagation(); });
    input.addEventListener('blur', () => done(true));
  }

  // ------------------------------------------------------------------ toolbar
  const allowWrites = !!(state.session?.allow_writes ?? window.APP_UI?.allowWrites);

  function renderToolbar() {
    const t = active();
    const c = t?.context || state.context;
    const running = !!t?.running;
    mount(toolbarEl,
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost.ws-btn-icon', { title: 'Toggle side panel', onclick: () => { ide.classList.toggle('ws-side-hidden'); prefs.set(SIDE_KEY, !ide.classList.contains('ws-side-hidden')); editor.refresh(); } }, icon('layout-sidebar')),
      h('span.ws-sep'),
      running
        ? h('button.ws-btn.ws-btn-sm.ws-btn-danger', { onclick: () => cancelRun(t) }, icon('stop-fill'), 'Cancel')
        : h('div.ws-run-split',
          h('button.ws-btn.ws-btn-sm.ws-btn-primary', { title: `Run statement at cursor or selection (${modKey()}+Enter)`, onclick: () => runCurrent() }, icon('play-fill'), 'Run', h('kbd', `${modKey()}↵`)),
          h('button.ws-btn.ws-btn-sm.ws-btn-primary', { title: 'More run options', onclick: (e) => popover(e.currentTarget, { width: 260, items: [
            { label: 'Run statement at cursor', value: 'cur', icon: 'play', hint: `${modKey()}+↵` },
            { label: 'Run all statements', value: 'all', icon: 'collection-play', hint: `⇧${modKey()}+↵` },
            { label: 'Run selection', value: 'sel', icon: 'cursor-text' },
            { label: 'Preview cost (EXPLAIN)', value: 'explain', icon: 'calculator', hint: `${modKey()}+E` },
          ], onPick: (i) => ({ cur: runCurrent, all: runAll, sel: runCurrent, explain: explainCurrent })[i.value]() }) }, icon('chevron-down'))),
      h('button.ws-btn.ws-btn-sm', { title: `Run every statement in order (⇧${modKey()}+Enter)`, disabled: running, onclick: () => runAll() }, icon('collection-play'), 'Run all'),
      h('button.ws-btn.ws-btn-sm', { title: `Estimate partitions, bytes and credits before running (${modKey()}+E)`, disabled: running, onclick: () => explainCurrent() }, icon('calculator'), 'Cost preview'),
      h('span.ws-sep'),
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { title: `Format SQL (⇧${modKey()}+F)`, onclick: () => formatCurrent() }, icon('text-indent-left'), 'Format'),
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { title: `Save (${modKey()}+S)`, onclick: () => saveTab(t) }, icon(t?.worksheetId ? 'cloud-check' : 'cloud-upload'), t?.worksheetId ? 'Saved' : 'Save'),
      allowWrites && h(`button.ws-btn.ws-btn-sm.ws-write-toggle${state.writeMode ? '.on' : ''}`, { title: 'Write mode lets DML/DDL run (still subject to the role\'s privileges)',
        onclick: async () => {
          if (!state.writeMode && !(await confirm({ title: 'Unlock write mode?', text: 'Statements that change data or objects will run with the Studio\'s Snowflake role until you lock it again.', confirmLabel: 'Unlock', danger: true }))) return;
          state.writeMode = !state.writeMode; renderToolbar();
        } }, icon(state.writeMode ? 'unlock-fill' : 'lock-fill'), state.writeMode ? 'Write mode' : 'Read-only'),
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost.ws-btn-icon', { title: 'Keyboard shortcuts', onclick: showShortcuts }, icon('keyboard')),
      h('div.ws-ide-ctx', { title: 'Context for this worksheet — change it from the top bar' },
        c.role && h('span.ws-tag', icon('person-badge'), c.role),
        c.warehouse && h('span.ws-tag', icon('cpu'), c.warehouse),
        (c.database || c.schema) && h('span.ws-tag', icon('database'), [c.database, c.schema].filter(Boolean).join('.'))));
  }

  const offCtx = on('context', (c) => {
    const t = active();
    if (t) { t.context = { ...c }; persist(); }
    renderToolbar();
    loadHints();
  });
  ctx.cleanup(offCtx);

  // ------------------------------------------------------------------ autocomplete
  const hintCache = new Map();
  async function loadHints() {
    const c = active()?.context || state.context;
    if (!c.database || !c.schema) { editor.setHintTables({}); return; }
    const key = `${c.role}|${c.database}|${c.schema}`;
    if (!hintCache.has(key)) {
      hintCache.set(key, api.columns(c.database, c.schema, '', { role: c.role })
        .then(({ columns }) => hintTablesFrom(columns, { database: c.database, schema: c.schema }))
        .catch(() => ({})));
    }
    editor.setHintTables(await hintCache.get(key));
  }

  // ------------------------------------------------------------------ save / autosave
  const autosaveTimers = new Map();
  function scheduleAutosave(t) {
    if (!t.worksheetId) return;
    clearTimeout(autosaveTimers.get(t.id));
    autosaveTimers.set(t.id, setTimeout(() => saveTab(t, { quiet: true }), 1500));
  }
  ctx.cleanup(() => autosaveTimers.forEach((id) => clearTimeout(id)));

  async function saveTab(t, { quiet = false } = {}) {
    if (!t) return;
    try {
      if (t.worksheetId) {
        await api.updateWorksheet(t.worksheetId, { sql: t.sql, title: t.title, context: t.context });
      } else {
        const ws = await api.createWorksheet({ title: t.title, sql: t.sql, context: t.context });
        t.worksheetId = ws.id;
        loadWorksheets();
      }
      t.dirty = false;
      persist();
      renderTabs();
      renderToolbar();
      if (!quiet) toast(`Saved “${t.title}”`, 'success', { timeout: 1800 });
    } catch (err) {
      if (!quiet) toast(err.message, 'error', { title: 'Save failed' });
    }
  }

  // ------------------------------------------------------------------ running
  function pickStatements(all) {
    const sql = editor.getValue();
    if (!all && editor.somethingSelected()) {
      const selText = editor.getSelection();
      const base = editor.selectionStart();
      return splitStatements(selText).map((s) => ({ ...s, start: s.start + base, end: s.end + base }));
    }
    if (all) return splitStatements(sql);
    const st = statementAt(sql, editor.cursorOffset());
    return st ? [st] : [];
  }

  function runCurrent() { return execute(pickStatements(false)); }
  function runAll() { return execute(pickStatements(true)); }

  async function execute(statements, { mode } = {}) {
    const t = active();
    if (!t || t.running) return;
    if (!statements.length) { toast('Nothing to run — the editor is empty or only has comments.', 'warning'); return; }
    editor.clearErrors();
    const abort = new AbortController();
    t.running = { abort, qid: null };
    t.results = statements.map((st) => ({ id: uid(), statement: st, status: 'pending' }));
    t.activeResult = 0;
    t.view = 'results';
    renderTabs(); renderToolbar(); renderResults();
    try {
      for (let i = 0; i < t.results.length; i += 1) {
        const ok = await runOne(t, t.results[i], i, { mode, signal: abort.signal });
        if (!ok) {
          t.results.slice(i + 1).forEach((r) => { r.status = 'skipped'; });
          break;
        }
      }
    } finally {
      t.running = null;
      if (t.id === activeId) editor.clearRunning();
      renderTabs(); renderToolbar(); renderResults();
    }
  }

  async function runOne(t, entry, index, { mode, signal }) {
    entry.status = 'submitting';
    entry.startedAt = Date.now();
    t.activeResult = index;
    if (t.id === activeId) editor.markRunning(entry.statement.start, entry.statement.end);
    renderResults();
    const tick = setInterval(() => { if (t.id === activeId && t.activeResult === index) updateStatus(); }, 200);
    try {
      let sub;
      try {
        sub = await api.submit({ sql: entry.statement.text, context: t.context, mode: mode || (state.writeMode ? 'write' : 'read'), worksheet_id: t.worksheetId, kind: 'worksheet' }, { signal });
      } catch (err) {
        return failEntry(t, entry, err);
      }
      entry.qid = sub.query_id;
      entry.kind = sub.kind;
      entry.status = 'running';
      entry.sfStatus = 'RUNNING';
      t.running.qid = sub.query_id;
      renderResults();
      const st = await waitForQuery(sub.query_id, { signal, onTick: (s2) => { entry.sfStatus = s2.status; } });
      entry.elapsedMs = Date.now() - entry.startedAt;
      if (!st.ok) {
        entry.status = st.status === 'ABORTED' ? 'cancelled' : 'error';
        entry.error = st.error || { message: `Query ${st.status}` };
        markEditorError(t, entry);
        return false;
      }
      entry.status = 'fetching';
      renderResults();
      entry.data = await api.results(sub.query_id, 0, 10000, { signal });
      entry.status = 'success';
      return true;
    } catch (err) {
      if (err.name === 'AbortError') { entry.status = 'cancelled'; entry.elapsedMs = Date.now() - entry.startedAt; return false; }
      return failEntry(t, entry, err);
    } finally {
      clearInterval(tick);
      if (t.id === activeId) renderResults();
    }
  }

  function failEntry(t, entry, err) {
    entry.elapsedMs = Date.now() - (entry.startedAt || Date.now());
    entry.status = err.status === 400 ? 'blocked' : 'error';
    entry.error = { message: err.message, code: err.code, position: err.data?.position, status: err.status };
    markEditorError(t, entry);
    return false;
  }

  function markEditorError(t, entry) {
    if (t.id !== activeId || !entry.error?.position) return;
    const pos = errorPosInDoc(editor.getValue(), entry.statement.start, entry.error.position);
    editor.markError(pos, entry.error.message);
  }

  async function cancelRun(t) {
    const qid = t.running?.qid;
    t.running?.abort.abort();
    if (qid) {
      try { await api.cancel(qid); toast('Cancel requested', 'info', { timeout: 1500 }); } catch (err) { toast(err.message, 'error', { title: 'Cancel failed' }); }
    }
  }

  // ------------------------------------------------------------------ cost preview
  async function explainCurrent() {
    const sts = pickStatements(false);
    const st = sts[0];
    if (!st) { toast('Put the cursor in a SELECT statement first.', 'warning'); return; }
    if (!isExplainable(st)) { toast('Cost preview works for SELECT / WITH queries.', 'warning'); return; }
    const body = h('div', skeleton(4, 22));
    const m = modal({ title: 'Cost preview', sub: 'EXPLAIN USING JSON — nothing is executed', width: 720, body,
      actions: [{ label: 'Close' }, { label: 'Run it', primary: true, icon: 'play-fill', onClick: (close) => { close(); execute([st]); } }] });
    try {
      const p = await api.explain(st.text, active().context);
      const e = p.estimate || {};
      const pruned = p.prune_ratio;
      mount(body,
        p.bytes_assigned > EXPLAIN_WARN_BYTES && banner('warn', 'Large scan', `This query will read about ${fmt.bytes(p.bytes_assigned)}. Consider a tighter filter or a LIMIT while exploring.`),
        h('div.ws-explain-grid',
          xstat('Partitions to scan', `${fmt.fmtInt(p.partitions_assigned)}`, `of ${fmt.fmtInt(p.partitions_total)}`),
          xstat('Pruned away', pruned == null ? '—' : fmt.pct(pruned), pruned != null && pruned < 0.2 && p.partitions_total > 100 ? 'filters barely prune' : 'by filters & clustering'),
          xstat('Bytes to read', fmt.bytes(p.bytes_assigned), `${p.tables?.length || 0} table(s)`),
          xstat('Est. run time', fmt.seconds(e.seconds), `≈${fmt.bytes(e.throughput_bytes_per_second)}/s on ${e.warehouse_size || 'X-Small'}`),
          xstat('Est. credits', fmt.credits(e.credits), e.resume_minimum_applies ? '60 s resume minimum' : `${e.credits_per_hour || 1} cr/hour warehouse`),
          xstat('Est. cost', fmt.usd(e.usd), `at ${fmt.usd(state.session?.credit_price_usd ?? 2.6)}/credit`)),
        (p.warnings || []).length ? h('div.ws-mt', p.warnings.map((w) => h('div.ws-insight', sevBadge(w.severity), h('h6', w.text)))) : h('div.ws-mt.ws-muted', icon('check2-circle', 'me-1'), 'No plan warnings.'),
        p.tables?.length ? h('div.ws-mt', h('div.ws-muted', { style: { fontSize: '12px', marginBottom: '6px' } }, 'Tables'), h('div.ws-row.ws-wrap.ws-gap-sm', p.tables.map((tb) => h('span.ws-tag', tb)))) : null,
        h('details.ws-mt', h('summary.ws-muted', 'Plan operations'), h('pre.ws-code', (p.operations || []).map((o) => `${o.id}: ${o.operation}${o.objects?.length ? ` ${o.objects.join(', ')}` : ''}${o.partitions_assigned != null ? `  [${o.partitions_assigned}/${o.partitions_total} partitions]` : ''}`).join('\n'))),
        h('div.ws-muted.ws-mt', { style: { fontSize: '11.5px' } }, 'Estimates assume scan-bound work at typical throughput for the warehouse size; joins, sorts and spill take longer.'));
    } catch (err) {
      mount(body, errorView(err));
    }
    return m;
  }
  const xstat = (k, v, sub) => h('div.ws-card', h('div.ws-k', k), h('div.ws-v', v), sub && h('div.ws-muted', { style: { fontSize: '11.5px' } }, sub));

  function formatCurrent() {
    const t = active();
    if (editor.somethingSelected()) editor.insert(formatSql(editor.getSelection()));
    else editor.replaceAll(formatSql(editor.getValue()));
    if (t) { t.sql = editor.getValue(); persist(); }
  }

  function showShortcuts() {
    const k = modKey();
    const rows = [
      ['Run statement at cursor / selection', [k, 'Enter']], ['Run all statements', ['Shift', k, 'Enter']], ['Cost preview', [k, 'E']],
      ['Save worksheet', [k, 'S']], ['Format SQL', ['Shift', k, 'F']], ['Toggle comment', [k, '/']], ['Autocomplete', ['Ctrl', 'Space']],
      ['Find / replace', [k, 'F']], ['Copy selected cells', [k, 'C']], ['Select all cells', [k, 'A']], ['Command palette', [k, 'K']],
    ];
    modal({ title: 'Keyboard shortcuts', width: 480, body: h('div.ws-shortcuts', rows.flatMap(([label, keys]) => [h('span', label), h('span', keys.map((x) => h('kbd', x)))])) });
  }

  // ------------------------------------------------------------------ results
  let grid = null;
  let inspectorEl = null;
  const disposers = [];

  function disposeCharts() { while (disposers.length) disposers.pop()?.dispose?.(); }

  function renderResults() {
    const t = active();
    disposeCharts();
    if (grid) { grid.destroy(); grid = null; }
    inspectorEl = null;
    if (!t || !t.results.length) {
      mount(resultsEl, h('div.ws-res-tabs', h('button.ws-tab', { onclick: () => { t.view = 'history'; t.results = []; renderHistory(resultsEl.querySelector('.ws-res-body'), t); } }, icon('clock-history'), 'History')),
        h('div.ws-res-body', h('div.ws-res-pane', empty({ icon: 'lightning', title: 'Run a query', text: `${modKey()}+Enter runs the statement under the cursor. Results, a chart builder, the query profile and cost diagnosis appear here.` }))));
      return;
    }
    const entry = t.results[t.activeResult] || t.results[0];
    const statusEl = h('div.ws-ide-status');
    resultsEl._status = statusEl;
    const pills = t.results.length > 1 ? h('div.ws-stmt-pills', t.results.map((r, i) => h(`button.ws-stmt-pill${i === t.activeResult ? '.on' : ''}`, {
      title: fmt.oneLine(r.statement.text, 300), onclick: () => { t.activeResult = i; renderResults(); },
    }, h('span.ws-dot', { class: r.status === 'success' ? 'on' : ['error', 'blocked'].includes(r.status) ? 'bad' : ['running', 'submitting', 'fetching'].includes(r.status) ? 'warn' : '' }),
    `#${i + 1} ${r.statement.keyword || ''}`))) : null;
    const viewTabs = [
      { key: 'results', label: 'Results', icon: 'table' }, { key: 'chart', label: 'Chart', icon: 'bar-chart-line' },
      { key: 'details', label: 'Profile & cost', icon: 'activity' }, { key: 'history', label: 'History', icon: 'clock-history' },
    ];
    const tools = h('div.ws-res-tools');
    const tabBar = h('div.ws-res-tabs', viewTabs.map((v) => h(`button.ws-tab${t.view === v.key ? '.on' : ''}`, { onclick: () => { t.view = v.key; renderResults(); } }, icon(v.icon), v.label)), tools);
    const bodyEl = h('div.ws-res-body');
    mount(resultsEl, pills, statusEl, tabBar, bodyEl);
    updateStatus();

    if (t.view === 'history') { renderHistory(bodyEl, t); return; }
    if (['pending', 'submitting', 'running', 'fetching'].includes(entry.status)) {
      mount(bodyEl, h('div.ws-res-pane.pad', skeleton(6, 20)));
      return;
    }
    if (entry.status === 'error' || entry.status === 'blocked' || entry.status === 'cancelled' || entry.status === 'skipped') {
      mount(bodyEl, h('div.ws-res-pane', errorPanel(t, entry)));
      if (t.view === 'details' && entry.qid && entry.status === 'error') renderDetails(bodyEl, entry);
      return;
    }
    if (t.view === 'results') renderGrid(bodyEl, tools, entry);
    else if (t.view === 'chart') renderChart(bodyEl, entry);
    else if (t.view === 'details') renderDetails(bodyEl, entry);
  }

  function updateStatus() {
    const t = active();
    const statusEl = resultsEl._status;
    if (!t || !statusEl) return;
    const entry = t.results[t.activeResult];
    if (!entry) return;
    const elapsed = entry.elapsedMs ?? (entry.startedAt ? Date.now() - entry.startedAt : 0);
    const snippet = h('span.ws-ellipsis.ws-mono.ws-muted', { style: { maxWidth: '380px', fontSize: '11.5px' }, title: entry.statement.text }, fmt.oneLine(entry.statement.text, 120));
    if (['pending', 'submitting', 'running', 'fetching'].includes(entry.status)) {
      const label = entry.status === 'fetching' ? 'FETCHING' : entry.status === 'submitting' ? 'SUBMITTING' : (entry.sfStatus || 'RUNNING');
      mount(statusEl, spinner(), h('span.ws-status-state', label.replace(/_/g, ' ')), h('span.ws-mono', fmt.ms(elapsed)), snippet, h('span.ws-grow'),
        entry.qid && h('span.ws-tag', { title: entry.qid }, entry.qid.slice(0, 8)),
        h('button.ws-btn.ws-btn-sm.ws-btn-danger', { onclick: () => cancelRun(t) }, icon('stop-fill'), 'Cancel'));
      return;
    }
    if (entry.status === 'success') {
      const d = entry.data || {};
      mount(statusEl, h('span.ws-ok', icon('check-circle-fill'), `${fmt.fmtInt(d.total ?? d.fetched)} rows`),
        h('span.ws-muted', `in ${fmt.ms(elapsed)}`),
        d.truncated && h('span.ws-badge.ws-sev-MEDIUM', { title: 'The Studio keeps the first rows in memory; export streams everything.' }, icon('scissors'), `showing first ${fmt.fmtInt(d.fetched)}`),
        d.fetch_ms != null && h('span.ws-muted', { title: 'Time to fetch results from Snowflake' }, `fetch ${fmt.ms(d.fetch_ms)}`),
        snippet, h('span.ws-grow'),
        entry.qid && h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { title: 'Inspect query', onclick: () => openQuery(entry.qid) }, icon('search'), h('span.ws-mono', entry.qid.slice(0, 8))),
        entry.qid && copyButton(entry.qid));
      return;
    }
    const icons = { error: 'x-octagon-fill', blocked: 'shield-lock-fill', cancelled: 'stop-circle', skipped: 'skip-forward' };
    mount(statusEl, h('span.ws-bad', icon(icons[entry.status] || 'x-octagon-fill'), entry.status === 'skipped' ? 'Not run' : entry.status === 'blocked' ? 'Blocked' : entry.status === 'cancelled' ? 'Cancelled' : 'Failed'),
      entry.elapsedMs != null && h('span.ws-muted', `after ${fmt.ms(entry.elapsedMs)}`), snippet, h('span.ws-grow'),
      entry.qid && h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { onclick: () => openQuery(entry.qid) }, icon('search'), 'Inspect'));
  }

  function errorPanel(t, entry) {
    if (entry.status === 'skipped') return h('div.ws-res-pane.pad', empty({ icon: 'skip-forward', title: 'Not run', text: 'An earlier statement failed, so this one was skipped.' }));
    if (entry.status === 'cancelled') return h('div.ws-res-pane.pad', empty({ icon: 'stop-circle', title: 'Cancelled', text: 'The query was cancelled.' }));
    const e = entry.error || {};
    const guard = entry.status === 'blocked';
    const canWrite = guard && e.code === 'write' && allowWrites;
    const pos = e.position ? errorPosInDoc(editor.getValue(), entry.statement.start, e.position) : null;
    return h(`div.ws-err-panel${guard ? '.guard' : ''}`,
      h('h5', icon(guard ? 'shield-lock-fill' : 'x-octagon-fill'), guard ? 'The Studio refused this statement' : 'Snowflake returned an error'),
      h('pre', e.message || 'Unknown error'),
      h('div.ws-row.ws-mt.ws-wrap',
        pos && h('button.ws-btn.ws-btn-sm', { onclick: () => { editor.cm?.setCursor(pos); editor.focus(); } }, icon('crosshair'), `Go to line ${pos.line + 1}, col ${pos.ch + 1}`),
        e.code && h('span.ws-tag', e.code),
        canWrite && h('button.ws-btn.ws-btn-sm.ws-btn-danger', { onclick: async () => {
          if (await confirm({ title: 'Run with write mode?', text: 'This statement changes data or objects. It will run once with the Studio\'s Snowflake role.', confirmLabel: 'Run it', danger: true })) execute([entry.statement], { mode: 'write' });
        } }, icon('unlock'), 'Run with write mode'),
        guard && e.code === 'session' && h('span.ws-muted', 'Use the Role / Warehouse / Database pickers in the top bar.'),
        entry.qid && h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { onclick: () => openQuery(entry.qid) }, icon('search'), 'Inspect query')));
  }

  function renderGrid(bodyEl, tools, entry) {
    const d = entry.data || { columns: [], rows: [] };
    if (!d.columns?.length) {
      mount(bodyEl, h('div.ws-res-pane.pad', empty({ icon: 'check2-circle', title: 'Statement executed', text: 'It returned no result set.' })));
      return;
    }
    const gridHost = h('div.ws-res-pane', { style: { overflow: 'hidden' } });
    const insp = h('div.ws-inspector', { style: { display: 'none' } });
    inspectorEl = insp;
    mount(bodyEl, gridHost, insp);
    grid = createGrid(gridHost, {
      onCell: (ri, ci, v) => showInspector(d, ri, ci, v),
      onProfile: (ci) => showProfile(d, ci),
    });
    grid.setData(d.columns, d.rows);
    const copy = (text, what) => navigator.clipboard?.writeText(text).then(() => toast(`Copied ${what}`, 'success', { timeout: 1200 })).catch(() => toast('Clipboard unavailable', 'warning'));
    mount(tools,
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { title: 'Copy', onclick: (e) => popover(e.currentTarget, { width: 260, items: [
        { label: 'Selected cells (TSV)', value: 'sel', icon: 'grid-3x3' },
        { label: 'Selected row as JSON', value: 'row', icon: 'braces' },
        { label: 'Selected column values', value: 'col', icon: 'layout-three-columns' },
        { label: 'Visible rows as TSV', value: 'tsv', icon: 'table' },
        { label: 'Visible rows as Markdown', value: 'md', icon: 'markdown' },
        { label: 'Visible rows as CSV', value: 'csv', icon: 'filetype-csv' },
      ], onPick: (i) => {
        const view = grid.view();
        if (i.value === 'sel') { const s = grid.selectionTSV(); if (!s) toast('Select cells first (click, shift-click).', 'warning'); else copy(s, 'cells'); }
        if (i.value === 'row') { const r = grid.selectedRowIndex(); if (r == null) toast('Select a row first.', 'warning'); else copy(toJSON(d.columns, d.rows, { idx: [r] }).replace(/^\[\n|\n\]$/g, ''), 'row'); }
        if (i.value === 'col') { const c = grid.selectedColIndex(); if (c == null) toast('Select a cell in the column first.', 'warning'); else copy(view.map((r) => cellText(d.rows[r][c])).join('\n'), 'column'); }
        if (i.value === 'tsv') copy(toTSV(d.columns, d.rows, { idx: view }), `${view.length} rows`);
        if (i.value === 'md') copy(toMarkdown(d.columns, d.rows, { idx: view }), 'Markdown table');
        if (i.value === 'csv') copy(toCSV(d.columns, d.rows, { idx: view }), `${view.length} rows`);
      } }) }, icon('clipboard'), 'Copy'),
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { title: 'Download the full result (streamed from Snowflake)', onclick: (e) => popover(e.currentTarget, { width: 270, items: [
        { head: 'Full result · streamed' },
        { label: 'CSV', value: 'csv', icon: 'filetype-csv' }, { label: 'TSV', value: 'tsv', icon: 'filetype-txt' },
        { label: 'JSON', value: 'json', icon: 'filetype-json' }, { label: 'NDJSON', value: 'ndjson', icon: 'file-earmark-code' },
        { head: 'Loaded rows · this view' }, { label: 'Filtered view as CSV', value: 'view', icon: 'funnel' },
      ], onPick: (i) => {
        if (i.value === 'view') { download(`result-${(entry.qid || 'view').slice(0, 8)}.csv`, toCSV(d.columns, d.rows, { idx: grid.view() }), 'text/csv'); return; }
        const a = h('a', { href: api.exportUrl(entry.qid, i.value), download: '' });
        document.body.appendChild(a); a.click(); a.remove();
      } }) }, icon('download'), 'Export'));
  }

  function showInspector(d, ri, ci, v) {
    if (!inspectorEl) return;
    const col = d.columns[ci];
    const text = inspectText(v);
    inspectorEl.style.display = '';
    mount(inspectorEl,
      h('div.ws-inspector-head', icon('search'), h('b', col.name), h('span.ws-tag', col.label || col.type), h('span.ws-muted', `row ${ri + 1}`),
        h('span', { style: { marginLeft: 'auto', display: 'inline-flex', gap: '2px' } }, copyButton(text),
          h('button.ws-btn.ws-btn-ghost.ws-btn-sm.ws-btn-icon', { title: 'Close', onclick: () => { inspectorEl.style.display = 'none'; } }, icon('x-lg')))),
      h('pre', text));
  }

  function showProfile(d, ci) {
    const col = d.columns[ci];
    const p = profileColumn(d.rows, ci, col.family, { idx: grid?.view() });
    const histEl = h('div.ws-chart.ws-chart-sm');
    const topEl = h('div.ws-chart', { style: { height: `${Math.max(140, Math.min(10, p.top.length) * 28 + 20)}px` } });
    const dr = drawer({ title: h('span.ws-row', icon('bar-chart-line'), col.name), sub: `${col.label || col.type} · profile of ${fmt.fmtInt(p.count)} loaded rows`, width: 620,
      body: h('div.ws-col', { style: { gap: '14px' } },
        h('div.ws-prof-stats',
          pstat('Null', `${fmt.pct(p.nullPct, 1)}`, `${fmt.fmtInt(p.nulls)} rows`), pstat('Distinct', fmt.fmtInt(p.distinct), p.unique ? 'every value unique' : ''),
          pstat('Non-null', fmt.fmtInt(p.count - p.nulls)),
          pstat('Min', p.min ?? '—'), pstat('Max', p.max ?? '—'),
          col.family === 'number' ? pstat('Mean', p.mean != null ? fmt.fmtNum(p.mean, 3) : '—', p.stddev != null ? `σ ${fmt.fmtNum(p.stddev, 3)}` : '') : pstat('Length', p.minLength != null ? `${p.minLength}–${p.maxLength}` : '—'),
          col.family === 'number' && pstat('Median', p.median != null ? fmt.fmtNum(p.median, 3) : '—'),
          col.family === 'number' && pstat('p95', p.p95 != null ? fmt.fmtNum(p.p95, 3) : '—'),
          col.family === 'number' && pstat('Sum', p.sum != null ? fmt.compact(p.sum, 2) : '—')),
        col.family === 'number' && p.histogram?.length ? card({ title: 'Distribution', icon: 'bar-chart', body: histEl }) : null,
        card({ title: 'Most frequent values', icon: 'list-ol', body: p.top.length ? topEl : empty({ title: 'All values are NULL' }) })) });
    setTimeout(() => {
      if (col.family === 'number' && p.histogram?.length) {
        const c = chart(histEl, (t2) => ({
          tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: t2.surface2, borderColor: t2.border, textStyle: { color: t2.text },
            formatter: (ps) => `${escapeHtml(fmt.fmtNum(p.histogram[ps[0].dataIndex].lo, 3))} – ${escapeHtml(fmt.fmtNum(p.histogram[ps[0].dataIndex].hi, 3))}<br><b>${fmt.fmtInt(ps[0].value)}</b> rows` },
          grid: { left: 8, right: 8, top: 10, bottom: 8, containLabel: true },
          xAxis: { type: 'category', data: p.histogram.map((b) => fmt.compact(b.lo, 2)), ...axisStyle(t2, { splitLine: { show: false } }) },
          yAxis: { type: 'value', ...axisStyle(t2) },
          series: [{ type: 'bar', data: p.histogram.map((b) => b.count), barCategoryGap: '8%', itemStyle: { color: t2.series[0], borderRadius: [4, 4, 0, 0] } }],
        }));
        dr.cleanup(() => c?.dispose());
      }
      if (p.top.length) {
        const c2 = hbars(topEl, { items: p.top.map((x) => ({ name: x.value === '' ? '(empty)' : fmt.truncate(x.value, 40), value: x.count })), valueFmt: (v) => fmt.fmtInt(v), labelWidth: 180 });
        dr.cleanup(() => c2?.dispose());
      }
    }, 30);
  }
  const pstat = (k, v, sub) => h('div.ws-card', h('div.ws-k', k), h('div.ws-v', { title: String(v) }, v), sub && h('div.ws-muted', { style: { fontSize: '11px' } }, sub));

  // ------------------------------------------------------------------ chart builder
  function renderChart(bodyEl, entry) {
    const d = entry.data || {};
    const cols = d.columns || [];
    if (!cols.length || !d.rows?.length) { mount(bodyEl, h('div.ws-res-pane.pad', empty({ icon: 'bar-chart', title: 'Nothing to chart', text: 'Run a query that returns rows.' }))); return; }
    const cfg = entry.chart || (entry.chart = autoChart(cols, d.rows));
    if (!cfg.ok && !cfg.y.length) {
      mount(bodyEl, h('div.ws-res-pane.pad', empty({ icon: 'bar-chart', title: 'No numeric columns', text: 'Charts need at least one numeric column (try COUNT(*) … GROUP BY).' })));
      return;
    }
    const side = h('div.ws-chartb-side');
    const chartEl = h('div.ws-chart');
    const note = h('div.ws-muted', { style: { fontSize: '11.5px', padding: '4px 4px 0' } });
    const mainEl = h('div.ws-chartb-main', chartEl, note);
    mount(bodyEl, h('div.ws-res-pane', { style: { overflow: 'hidden' } }, h('div.ws-chartb', side, mainEl)));
    let inst = null;
    const types = [['bar', 'bar-chart'], ['line', 'graph-up'], ['area', 'graph-up-arrow'], ['scatter', 'diagram-2'], ['pie', 'pie-chart'], ['heatmap', 'grid-3x3']];
    const sel = (label, value, options, onChange, allowNone) => h('div.ws-field', h('label', label),
      h('select.ws-select', { onchange: (e) => onChange(e.target.value === '' ? null : Number(e.target.value)) },
        allowNone && h('option', { value: '', selected: value == null }, '— none —'),
        options.map(([i, c]) => h('option', { value: i, selected: i === value }, `${c.name}`))));
    const numCols = cols.map((c, i) => [i, c]).filter(([, c]) => c.family === 'number');
    const allCols = cols.map((c, i) => [i, c]);
    const draw = () => {
      inst?.dispose();
      inst = drawChart(chartEl, cfg, d, note);
      if (inst) disposers.push(inst);
    };
    const renderSide = () => mount(side,
      h('div.ws-field', h('label', 'Chart type'), h('div.ws-chart-types', types.map(([tname, ic]) => h(`button${cfg.type === tname ? '.on' : ''}`, { onclick: () => { cfg.type = tname; renderSide(); draw(); } }, icon(ic), tname)))),
      sel(cfg.type === 'scatter' ? 'X (numeric)' : 'X axis / category', cfg.x, cfg.type === 'scatter' ? numCols : allCols, (v) => { cfg.x = v ?? 0; draw(); }),
      h('div.ws-field', h('label', cfg.type === 'pie' || cfg.type === 'heatmap' || cfg.group != null ? 'Value' : 'Y values'),
        h('div.ws-chartb-ys', numCols.map(([i, c]) => h('label', h('input', { type: cfg.type === 'pie' || cfg.type === 'heatmap' || cfg.group != null ? 'radio' : 'checkbox', name: 'ws-y', checked: cfg.y.includes(i),
          onchange: (e) => {
            const single = cfg.type === 'pie' || cfg.type === 'heatmap' || cfg.group != null;
            cfg.y = single ? [i] : e.target.checked ? [...new Set([...cfg.y, i])] : cfg.y.filter((x) => x !== i);
            draw();
          } }), c.name)))),
      cfg.type !== 'pie' && cfg.type !== 'scatter' && sel(cfg.type === 'heatmap' ? 'Rows (Y category)' : 'Group / colour by', cfg.group, allCols.filter(([i]) => i !== cfg.x), (v) => { cfg.group = v; if (v != null) cfg.y = cfg.y.slice(0, 1); renderSide(); draw(); }, true),
      h('button.ws-btn.ws-btn-sm', { onclick: () => {
        const url = inst?.inst?.getDataURL({ pixelRatio: 2, backgroundColor: getComputedStyle(document.querySelector('.ws-root')).getPropertyValue('--ws-surface').trim() || '#111a2e' });
        if (url) { const a = h('a', { href: url, download: 'chart.png' }); document.body.appendChild(a); a.click(); a.remove(); }
      } }, icon('image'), 'Download PNG'));
    renderSide();
    setTimeout(draw, 20);
  }

  function drawChart(el, cfg, d, note) {
    const cols = d.columns;
    const ys = cfg.y.filter((i) => cols[i]);
    if (!ys.length) { mount(el, empty({ icon: 'bar-chart', title: 'Pick a value column' })); return null; }
    const xFam = cols[cfg.x]?.family;
    const nameOf = (i) => cols[i]?.name;
    mount(note, `${fmt.fmtInt(d.rows.length)} loaded rows${d.truncated ? ' (result truncated)' : ''} · values are summed per ${cols[cfg.x]?.name}${cfg.group != null ? ` and ${cols[cfg.group]?.name}` : ''}`);
    if (cfg.type === 'scatter') {
      const pts = d.rows.slice(0, 5000).map((r) => [Number(cellValue(r[cfg.x])), Number(cellValue(r[ys[0]]))]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
      mount(note, `${fmt.fmtInt(pts.length)} points${d.rows.length > 5000 ? ' (first 5,000 rows)' : ''}`);
      return chart(el, (t) => ({
        tooltip: { trigger: 'item', backgroundColor: t.surface2, borderColor: t.border, textStyle: { color: t.text }, formatter: (p) => `${escapeHtml(nameOf(cfg.x))}: <b>${fmt.fmtNum(p.value[0])}</b><br>${escapeHtml(nameOf(ys[0]))}: <b>${fmt.fmtNum(p.value[1])}</b>` },
        grid: { left: 8, right: 16, top: 16, bottom: 28, containLabel: true },
        xAxis: { type: 'value', name: nameOf(cfg.x), nameLocation: 'middle', nameGap: 24, scale: true, ...axisStyle(t) },
        yAxis: { type: 'value', name: nameOf(ys[0]), scale: true, ...axisStyle(t) },
        series: [{ type: 'scatter', data: pts, symbolSize: 8, itemStyle: { color: t.series[0], opacity: 0.8, borderColor: t.surface, borderWidth: 1 } }],
      }));
    }
    if (cfg.type === 'pie') {
      const s = chartSeries(d.rows, { x: cfg.x, y: [ys[0]], xFamily: xFam });
      let items = s.xs.map((x, i) => ({ name: x, value: s.series[0].data[i] })).filter((x) => x.value > 0).sort((a, b) => b.value - a.value);
      if (items.length > 7) items = [...items.slice(0, 6), { name: 'Other', value: items.slice(6).reduce((a, b) => a + b.value, 0) }];
      return chart(el, (t) => ({
        tooltip: { trigger: 'item', backgroundColor: t.surface2, borderColor: t.border, textStyle: { color: t.text }, formatter: (p) => tipRow(p.color, p.name, `${fmt.fmtNum(p.value)} · ${p.percent}%`) },
        legend: { orient: 'vertical', right: 0, top: 'middle', icon: 'roundRect', itemWidth: 10, itemHeight: 10, textStyle: { color: t.text2 }, type: 'scroll' },
        series: [{ type: 'pie', radius: ['50%', '78%'], center: ['38%', '50%'], padAngle: 2, itemStyle: { borderRadius: 6, borderColor: t.surface, borderWidth: 2 },
          label: { color: t.text2, formatter: '{b}: {d}%' }, data: items.map((x, i) => ({ ...x, itemStyle: { color: x.name === 'Other' ? t.muted : t.series[i % 8] } })) }],
      }));
    }
    if (cfg.type === 'heatmap') {
      if (cfg.group == null) { mount(el, empty({ icon: 'grid-3x3', title: 'Pick a row category', text: 'Heatmaps need X, a row category and a value.' })); return null; }
      const xs = [...new Set(d.rows.map((r) => cellText(r[cfg.x])))].slice(0, 200);
      const yv = [...new Set(d.rows.map((r) => cellText(r[cfg.group])))].slice(0, 100);
      const sums = new Map();
      for (const r of d.rows) { const k = `${cellText(r[cfg.x])}\u0000${cellText(r[cfg.group])}`; sums.set(k, (sums.get(k) || 0) + (Number(cellValue(r[ys[0]])) || 0)); }
      const data = []; let max = 0;
      xs.forEach((x, i) => yv.forEach((y, j) => { const v = sums.get(`${x}\u0000${y}`); if (v != null) { data.push([i, j, v]); max = Math.max(max, v); } }));
      return chart(el, (t) => ({
        tooltip: { backgroundColor: t.surface2, borderColor: t.border, textStyle: { color: t.text }, formatter: (p) => `${escapeHtml(xs[p.value[0]])} · ${escapeHtml(yv[p.value[1]])}<br><b>${fmt.fmtNum(p.value[2])}</b>` },
        grid: { left: 8, right: 8, top: 8, bottom: 44, containLabel: true },
        xAxis: { type: 'category', data: xs, ...axisStyle(t, { splitLine: { show: false } }) },
        yAxis: { type: 'category', data: yv, ...axisStyle(t, { splitLine: { show: false } }) },
        visualMap: { min: 0, max: max || 1, orient: 'horizontal', left: 'center', bottom: 0, itemHeight: 140, itemWidth: 12, textStyle: { color: t.muted },
          inRange: { color: t.dark ? ['#16233c', '#184f95', '#2a78d6', '#5598e7', '#9ec5f4'] : ['#eef4fd', '#b7d3f6', '#6da7ec', '#2a78d6', '#104281'] } },
        series: [{ type: 'heatmap', data, itemStyle: { borderColor: t.surface, borderWidth: 1, borderRadius: 3 } }],
      }));
    }
    const s = chartSeries(d.rows, { x: cfg.x, y: ys, group: cfg.group, xFamily: xFam });
    const names = s.series.map((x) => (cfg.group != null ? x.name : nameOf(Number(x.name))));
    const scale = new ColorScale(names);
    const kind = cfg.type;
    const stacked = cfg.group != null && kind !== 'line';
    return chart(el, (t) => ({
      tooltip: { trigger: 'axis', axisPointer: { type: kind === 'bar' ? 'shadow' : 'line' }, backgroundColor: t.surface2, borderColor: t.border, textStyle: { color: t.text },
        formatter: (ps) => `<div style="font-weight:600;margin-bottom:4px">${escapeHtml(ps[0]?.axisValue)}</div>${ps.filter((p) => p.value).sort((a, b) => b.value - a.value).slice(0, 12).map((p) => tipRow(p.color, p.seriesName, fmt.fmtNum(p.value))).join('')}` },
      legend: names.length > 1 ? { top: 0, left: 0, icon: 'roundRect', itemWidth: 10, itemHeight: 10, textStyle: { color: t.text2 }, type: 'scroll' } : undefined,
      grid: { left: 8, right: 16, top: names.length > 1 ? 34 : 14, bottom: 8, containLabel: true },
      xAxis: { type: 'category', data: s.xs, boundaryGap: kind === 'bar', ...axisStyle(t, { splitLine: { show: false }, axisLabel: { color: t.muted, fontSize: 11, hideOverlap: true,
        formatter: (v) => (xFam === 'date' || xFam === 'timestamp' ? fmt.day(v) : fmt.truncate(v, 18)) } }) },
      yAxis: { type: 'value', ...axisStyle(t, { axisLabel: { color: t.muted, fontSize: 11, formatter: (v) => fmt.compact(v) } }) },
      dataZoom: s.xs.length > 60 ? [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 0, borderColor: 'transparent', textStyle: { color: t.muted } }] : undefined,
      series: s.series.map((ser, i) => {
        const name = names[i];
        const color = name === 'Other' ? t.muted : scale.get(name);
        return kind === 'bar'
          ? { name, type: 'bar', stack: stacked ? 'all' : undefined, data: ser.data, barMaxWidth: 28, itemStyle: { color, borderRadius: stacked ? 0 : [4, 4, 0, 0], borderColor: t.surface, borderWidth: stacked ? 1 : 0 } }
          : { name, type: 'line', stack: stacked ? 'all' : undefined, data: ser.data, smooth: 0.3, symbol: 'none', lineStyle: { width: 2, color }, itemStyle: { color }, areaStyle: kind === 'area' ? { color: hexA(color, 0.22) } : undefined };
      }),
    }));
  }

  // ------------------------------------------------------------------ details (profile & cost)
  async function renderDetails(bodyEl, entry) {
    if (!entry.qid) { mount(bodyEl, h('div.ws-res-pane.pad', empty({ icon: 'activity', title: 'No query id', text: 'The statement never reached Snowflake.' }))); return; }
    const pane = h('div.ws-res-pane.pad', skeleton(5, 22));
    mount(bodyEl, pane);
    if (!entry.details) {
      for (let attempt = 0; attempt < 4 && !entry.details; attempt += 1) {
        try {
          entry.details = await api.details(entry.qid);
        } catch (err) {
          if (err.status === 404 && attempt < 3) { await sleep(1500 * (attempt + 1)); continue; }
          if (pane.isConnected) mount(pane, errorView(err, () => renderDetails(bodyEl, entry)));
          return;
        }
      }
    }
    if (!pane.isConnected) return;
    const { query: q, insights, cost } = entry.details;
    const profEl = h('div.ws-profile', { style: { height: '380px' } }, skeleton(4));
    mount(pane,
      h('div.ws-grid.ws-g-4.ws-mb',
        dstat('Elapsed', fmt.ms(q.total_elapsed_time), 'stopwatch'),
        dstat(cost.estimated ? 'Est. cost' : 'Attributed cost', fmt.usd(cost.usd), 'cash-coin', `${fmt.credits(cost.credits)} credits`),
        dstat('Scanned', fmt.bytes(q.bytes_scanned), 'hdd', q.partitions_total ? `${fmt.fmtInt(q.partitions_scanned)} / ${fmt.fmtInt(q.partitions_total)} partitions` : ''),
        dstat('Rows', fmt.compact(q.rows_produced), 'list-ol', `${q.warehouse_name || '—'} · ${q.warehouse_size || 'no warehouse'}`)),
      h('div.ws-grid.ws-g-2',
        h('div.ws-col', card({ title: 'Where the time went', icon: 'hourglass-split', body: waterfall(q) }),
          card({ title: 'Diagnosis', icon: 'activity', body: insightsList(insights) })),
        card({ title: 'Operator profile', icon: 'diagram-3', tools: h('button.ws-btn.ws-btn-sm', { onclick: () => openQuery(entry.qid) }, icon('arrows-fullscreen'), 'Full inspector'), body: profEl })));
    if (entry.details.profile_available) {
      try { renderProfile(profEl, entry.profile || (entry.profile = await api.profile(entry.qid))); } catch (err) { mount(profEl, errorView(err)); }
    } else {
      mount(profEl, empty({ icon: 'diagram-3', title: 'No operator profile', text: entry.details.profile_error || 'Metadata-only or result-cache queries have no profile.' }));
    }
  }
  const dstat = (label, value, ic, foot) => h('div.ws-card.ws-stat', h('div.ws-stat-label', icon(ic), label), h('div.ws-stat-value', { style: { fontSize: '22px' } }, value), foot && h('div.ws-stat-foot', foot));

  // ------------------------------------------------------------------ history
  async function renderHistory(bodyEl, t) {
    const scopeKey = t.historyScope || (t.worksheetId ? 'worksheet' : 'mine');
    const pane = h('div.ws-res-pane');
    const head = h('div.ws-row', { style: { padding: '8px 12px', borderBottom: '1px solid var(--ws-border)' } },
      seg([{ value: 'worksheet', label: 'This worksheet' }, { value: 'mine', label: 'All my runs' }], scopeKey, (v) => { t.historyScope = v; renderHistory(bodyEl, t); }),
      h('span.ws-muted', { style: { fontSize: '12px' } }, 'Results stay re-openable for 30 minutes after a run.'));
    const list = h('div', skeleton(5, 18));
    mount(bodyEl, h('div.ws-res-pane', head, list));
    if (scopeKey === 'worksheet' && !t.worksheetId) { mount(list, empty({ icon: 'cloud-upload', title: 'Save this worksheet to keep its history', text: 'Unsaved drafts only appear under “All my runs”.' })); return; }
    try {
      const { runs } = await api.runs(scopeKey === 'worksheet' ? { worksheet: t.worksheetId, limit: 100 } : { limit: 100 });
      if (!runs.length) { mount(list, empty({ icon: 'clock-history', title: 'No runs yet' })); return; }
      const st = { SUCCESS: ['check-circle-fill', 'var(--ws-good-text)'], FAILED: ['x-octagon-fill', 'var(--ws-critical-text)'], BLOCKED: ['shield-lock-fill', 'var(--ws-warning)'],
        CANCELLED: ['stop-circle', 'var(--ws-muted)'], RUNNING: ['hourglass-split', 'var(--ws-accent)'], SUBMITTED: ['hourglass', 'var(--ws-accent)'] };
      mount(list, runs.map((r) => {
        const [ic, color] = st[r.status] || ['dot', 'var(--ws-muted)'];
        return h('div.ws-hist-row', { title: r.sql, onclick: (e) => historyMenu(e.currentTarget, r) },
          h('span', { style: { color } }, icon(ic)),
          h('span.ws-muted', fmt.relTime(r.created_at)),
          h('span.ws-sql-snippet', fmt.oneLine(r.sql, 260)),
          h('span.ws-mono.ws-muted', r.rows != null ? `${fmt.compact(r.rows)} rows` : r.status.toLowerCase()),
          h('span.ws-mono.ws-muted', r.elapsed_ms != null ? fmt.ms(r.elapsed_ms) : ''),
          h('span.ws-tag', r.kind));
      }));
    } catch (err) { mount(list, errorView(err, () => renderHistory(bodyEl, t))); }
  }

  function historyMenu(anchor, r) {
    popover(anchor, { width: 280, items: [
      r.query_id && r.status === 'SUCCESS' && { label: 'Re-open results', value: 'results', icon: 'table' },
      { label: 'Insert SQL at cursor', value: 'insert', icon: 'box-arrow-in-down' },
      { label: 'Open SQL in a new tab', value: 'tab', icon: 'plus-square' },
      r.query_id && { label: 'Inspect query', value: 'inspect', icon: 'search' },
      { label: 'Copy SQL', value: 'copy', icon: 'clipboard' },
    ].filter(Boolean), onPick: async (i) => {
      if (i.value === 'insert') editor.insert(`${r.sql}\n`);
      if (i.value === 'tab') openTab({ title: `Run ${String(r.query_id || r.id).slice(0, 8)}`, sql: r.sql, context: r.context });
      if (i.value === 'inspect') openQuery(r.query_id);
      if (i.value === 'copy') navigator.clipboard?.writeText(r.sql).then(() => toast('SQL copied', 'success', { timeout: 1200 }));
      if (i.value === 'results') {
        const t = active();
        const st = splitStatements(r.sql)[0] || { text: r.sql, start: 0, end: 0, keyword: '' };
        const entry = { id: uid(), statement: { ...st, start: -1 }, status: 'fetching', qid: r.query_id, startedAt: Date.now() };
        t.results = [entry]; t.activeResult = 0; t.view = 'results';
        renderResults();
        try { entry.data = await api.results(r.query_id, 0, 10000); entry.status = 'success'; entry.elapsedMs = r.elapsed_ms; } catch (err) {
          entry.status = 'error'; entry.error = { message: `${err.message} (results are kept for 30 minutes — re-run the query instead)` };
        }
        renderResults();
      }
    } });
  }

  // ------------------------------------------------------------------ side panel
  let sideView = prefs.get('ws:ide:sideview', 'objects');
  const sideSearch = h('input.ws-input', { placeholder: 'Filter…', oninput: () => applySideFilter() });
  const sideBody = h('div.ws-ide-side-body');
  const objectsEl = h('div.ws-tree');
  const wsListEl = h('div');
  let worksheetsData = { worksheets: [], shared: [] };

  function renderSide() {
    mount(sideEl,
      h('div.ws-tabs', ['objects', 'worksheets'].map((v) => h(`button.ws-tab${sideView === v ? '.on' : ''}`, { onclick: () => { sideView = v; prefs.set('ws:ide:sideview', v); renderSide(); } },
        icon(v === 'objects' ? 'database' : 'journal-code'), v === 'objects' ? 'Objects' : 'Worksheets'))),
      h('div.ws-ide-side-tools', h('div.ws-search', icon('search'), sideSearch),
        sideView === 'objects'
          ? h('button.ws-btn.ws-btn-sm.ws-btn-icon', { title: 'Refresh', onclick: () => loadObjects(true) }, icon('arrow-clockwise'))
          : h('button.ws-btn.ws-btn-sm.ws-btn-icon', { title: 'New worksheet', onclick: () => openTab({ title: `Worksheet ${tabs.length + 1}` }) }, icon('plus-lg'))),
      sideBody);
    mount(sideBody, sideView === 'objects' ? objectsEl : wsListEl);
    applySideFilter();
  }

  function applySideFilter() {
    const q = sideSearch.value.trim().toLowerCase();
    if (sideView === 'worksheets') { renderWorksheetList(); return; }
    const walk = (item) => {
      const name = (item.dataset.name || '').toLowerCase();
      let any = !q || name.includes(q);
      for (const kid of item.querySelectorAll(':scope > .ws-tree-kids > .ws-tree-item, :scope > .ws-tree-kids > .ws-tree-groupwrap > .ws-tree-item')) any = walk(kid) || any;
      item.style.display = any ? '' : 'none';
      return any;
    };
    objectsEl.querySelectorAll(':scope > .ws-tree-item').forEach(walk);
  }

  function treeItem({ name, label, ic, icCls = '', meta, title, actions = [], load, onClick, onDbl, open = false }) {
    const kids = h('div.ws-tree-kids', { style: { display: 'none' } });
    const caret = load ? icon('chevron-right', 'ws-tree-caret') : h('span.ws-tree-caret');
    const row = h('div.ws-tree-row', { title: title || label, draggable: 'true' }, caret, icon(ic, `ws-tree-icon ${icCls}`), h('span.ws-tree-name', label),
      meta != null && h('span.ws-tree-meta', meta),
      actions.length ? h('span.ws-tree-actions', actions.map((a) => h('button', { title: a.title, onclick: (e) => { e.stopPropagation(); a.run(); } }, icon(a.icon)))) : null);
    const item = h('div.ws-tree-item', { dataset: { name } }, row, kids);
    let loaded = false;
    const toggle = async () => {
      if (!load) return;
      const opening = kids.style.display === 'none';
      kids.style.display = opening ? '' : 'none';
      row.classList.toggle('open', opening);
      if (opening && !loaded) {
        loaded = true;
        mount(kids, h('div.ws-tree-row', spinner(), h('span.ws-muted', 'Loading…')));
        try { mount(kids, await load()); applySideFilter(); } catch (err) { loaded = false; mount(kids, h('div.ws-tree-row.ws-muted', { title: err.message }, icon('exclamation-triangle'), fmt.truncate(err.message, 60))); }
      }
    };
    let clickTimer = null;
    row.addEventListener('click', (e) => {
      if (e.target.closest('.ws-tree-caret') || !onClick) { toggle(); return; }
      clearTimeout(clickTimer);
      clickTimer = setTimeout(() => { onClick(); }, onDbl ? 220 : 0);
    });
    row.addEventListener('dblclick', () => { clearTimeout(clickTimer); if (onDbl) onDbl(); else toggle(); });
    row.addEventListener('dragstart', (e) => e.dataTransfer?.setData('text/plain', title || label));
    if (open) setTimeout(toggle, 0);
    item.reload = () => { loaded = false; if (kids.style.display !== 'none') { kids.style.display = 'none'; toggle(); } };
    return item;
  }

  const insertName = (text) => { editor.insert(text); };
  const roleCtx = () => ({ role: (active()?.context || state.context).role });

  async function loadObjects(refresh = false) {
    mount(objectsEl, skeleton(8, 16));
    try {
      const { databases } = await api.databases(roleCtx(), refresh);
      const cur = (active()?.context || state.context).database;
      mount(objectsEl, databases.length ? databases.map((d) => dbNode(d, d.name === cur)) : empty({ icon: 'database', title: 'No databases', text: 'This role cannot see any database.' }));
      applySideFilter();
    } catch (err) { mount(objectsEl, errorView(err, () => loadObjects(true))); }
  }

  function dbNode(d, open) {
    return treeItem({ name: d.name, label: d.name, ic: d.kind === 'APPLICATION' ? 'app' : 'database', icCls: 'ws-ic-db', title: fqName(d.name), meta: d.kind && d.kind !== 'STANDARD' ? d.kind.toLowerCase() : '',
      open,
      actions: [{ icon: 'input-cursor', title: 'Insert name', run: () => insertName(fqName(d.name)) }, { icon: 'pin-angle', title: 'Use as context', run: () => setContext({ database: d.name }) }],
      load: async () => {
        const { schemas } = await api.schemas(d.name, roleCtx());
        const cur = (active()?.context || state.context);
        return schemas.length ? schemas.map((s) => schemaNode(d.name, s, cur.database === d.name && cur.schema === s.name)) : h('div.ws-tree-row.ws-muted', 'No schemas');
      } });
  }

  const KIND_META = {
    TABLE: ['table', 'ws-ic-table', 'Tables'], VIEW: ['eye', 'ws-ic-view', 'Views'], MATERIALIZED_VIEW: ['eye-fill', 'ws-ic-view', 'Materialized views'],
    DYNAMIC_TABLE: ['lightning', 'ws-ic-table', 'Dynamic tables'], EXTERNAL_TABLE: ['cloud', 'ws-ic-table', 'External tables'], EVENT_TABLE: ['calendar-event', 'ws-ic-table', 'Event tables'],
    PROCEDURE: ['braces-asterisk', 'ws-ic-proc', 'Procedures'], FUNCTION: ['function', 'ws-ic-proc', 'Functions'], STAGE: ['box-seam', 'ws-ic-other', 'Stages'],
    TASK: ['alarm', 'ws-ic-other', 'Tasks'], STREAM: ['water', 'ws-ic-other', 'Streams'], PIPE: ['funnel', 'ws-ic-other', 'Pipes'],
  };

  function schemaNode(db, s, open) {
    return treeItem({ name: s.name, label: s.name, ic: 'folder2', icCls: 'ws-ic-schema', title: fqName(db, s.name), open,
      actions: [{ icon: 'input-cursor', title: 'Insert name', run: () => insertName(fqName(db, s.name)) }, { icon: 'pin-angle', title: 'Use as context', run: () => setContext({ database: db, schema: s.name }) }],
      load: async () => {
        const { objects } = await api.objects(db, s.name, roleCtx());
        const groups = Object.entries(objects).filter(([, items]) => items?.length);
        if (!groups.length) return h('div.ws-tree-row.ws-muted', 'Empty schema');
        return groups.map(([kind, items]) => {
          const [ic, cls, label] = KIND_META[kind] || ['dot', 'ws-ic-other', kind.toLowerCase()];
          return h('div.ws-tree-groupwrap', h('div.ws-tree-group', `${label} · ${items.length}`), items.map((o) => objectNode(db, s.name, kind, o, ic, cls)));
        });
      } });
  }

  function objectNode(db, schema, kind, o, ic, cls) {
    const fq = fqName(db, schema, o.name);
    const relational = ['TABLE', 'VIEW', 'MATERIALIZED_VIEW', 'DYNAMIC_TABLE', 'EXTERNAL_TABLE', 'EVENT_TABLE'].includes(kind);
    const routine = kind === 'PROCEDURE' || kind === 'FUNCTION';
    const sigTypes = routine ? (o.params || []).map((p) => p.type.split('(')[0]).join(', ') : '';
    const meta = relational && o.rows != null ? `${fmt.compact(o.rows)}${o.bytes ? ` · ${fmt.bytes(o.bytes)}` : ''}` : routine ? `(${(o.params || []).length})` : (o.state || '');
    const title = routine ? `${fq}${o.argument_signature || '()'} → ${o.data_type || ''}` : `${fq}${o.comment ? ` — ${o.comment}` : ''}`;
    const actions = [];
    if (relational) {
      actions.push({ icon: 'play-fill', title: 'Preview 100 rows', run: () => preview(fq, o.name) });
      actions.push({ icon: 'input-cursor-text', title: 'Insert SELECT with all columns', run: () => insertSelect(db, schema, o.name) });
    }
    actions.push({ icon: 'code', title: 'Show DDL', run: () => showDdl(routine ? kind : kind, routine ? `${fq}(${sigTypes})` : fq, o.name) });
    return treeItem({ name: o.name, label: o.name, ic, icCls: cls, meta, title, actions,
      onClick: () => insertName(routine ? `${kind === 'PROCEDURE' ? 'CALL ' : ''}${fq}(${(o.params || []).map((p) => p.name.toLowerCase()).join(', ')})` : fq),
      onDbl: relational ? () => preview(fq, o.name) : null,
      load: relational ? async () => {
        const { columns } = await api.columns(db, schema, o.name, roleCtx());
        return columns.map((c) => treeItem({ name: c.column_name, label: c.column_name, ic: 'dot', icCls: 'ws-tree-col', meta: String(c.data_type || '').toLowerCase(),
          title: `${c.column_name} ${c.data_type}${c.is_nullable === 'NO' ? ' NOT NULL' : ''}${c.comment ? ` — ${c.comment}` : ''}`, onClick: () => insertName(ident(c.column_name)) }));
      } : null });
  }

  function preview(fq, name) {
    openTab({ title: `${name} preview`, sql: `SELECT *\nFROM ${fq}\nLIMIT 100;\n`, run: true });
  }

  async function insertSelect(db, schema, table) {
    try {
      const { columns } = await api.columns(db, schema, table, roleCtx());
      editor.insert(`SELECT\n  ${columns.map((c) => ident(c.column_name)).join(',\n  ')}\nFROM ${fqName(db, schema, table)}\nLIMIT 100;\n`);
    } catch (err) { toast(err.message, 'error'); }
  }

  async function showDdl(kind, name, short) {
    try {
      const { ddl } = await api.ddl(kind, name, roleCtx());
      openTab({ title: `${short} DDL`, sql: `${ddl.trim()}\n` });
    } catch (err) { toast(err.message, 'error', { title: 'Could not load DDL' }); }
  }

  async function loadWorksheets() {
    try {
      worksheetsData = await api.worksheets();
    } catch (err) {
      mount(wsListEl, errorView(err, loadWorksheets));
      return;
    }
    renderWorksheetList();
  }

  function renderWorksheetList() {
    const q = sideSearch.value.trim().toLowerCase();
    const match = (w) => !q || `${w.title} ${w.folder} ${w.preview || ''}`.toLowerCase().includes(q);
    const mine = worksheetsData.worksheets.filter(match);
    const shared = (worksheetsData.shared || []).filter(match);
    const folders = new Map();
    for (const w of mine) { const f = w.folder || ''; if (!folders.has(f)) folders.set(f, []); folders.get(f).push(w); }
    const openIds = new Set(tabs.map((t) => t.worksheetId).filter(Boolean));
    const item = (w, owned) => h(`div.ws-wslist-item${openIds.has(w.id) ? '.active' : ''}`, { onclick: () => openWorksheet(w) },
      owned ? h(`button.ws-star${w.starred ? '.on' : ''}`, { title: w.starred ? 'Unstar' : 'Star', onclick: async (e) => { e.stopPropagation(); await api.updateWorksheet(w.id, { starred: !w.starred }); loadWorksheets(); } }, icon(w.starred ? 'star-fill' : 'star'))
        : icon('people', 'ws-muted'),
      h('div', { style: { minWidth: 0, flex: 1 } }, h('div.ws-wslist-title', w.title), h('div.ws-wslist-sub', owned ? `${fmt.relTime(w.updated_at)} · ${w.preview || 'empty'}` : `by ${w.owner} · ${fmt.relTime(w.updated_at)}`)),
      owned && h('span.ws-tree-actions',
        h('button', { title: 'Rename', onclick: (e) => { e.stopPropagation(); promptText('Rename worksheet', w.title, async (v) => { await api.updateWorksheet(w.id, { title: v }); const t = tabs.find((x) => x.worksheetId === w.id); if (t) { t.title = v; renderTabs(); persist(); } loadWorksheets(); }); } }, icon('pencil')),
        h('button', { title: 'Move to folder', onclick: (e) => { e.stopPropagation(); promptText('Folder', w.folder || '', async (v) => { await api.updateWorksheet(w.id, { folder: v }); loadWorksheets(); }, true); } }, icon('folder-symlink')),
        h('button', { title: w.shared ? 'Stop sharing' : 'Share read-only with Studio users', onclick: async (e) => { e.stopPropagation(); await api.updateWorksheet(w.id, { shared: !w.shared }); toast(w.shared ? 'No longer shared' : 'Shared with Studio users', 'success', { timeout: 1500 }); loadWorksheets(); } }, icon(w.shared ? 'people-fill' : 'share')),
        h('button', { title: 'Delete', onclick: async (e) => {
          e.stopPropagation();
          if (!(await confirm({ title: `Delete “${w.title}”?`, text: 'The worksheet is removed for good. Query history is kept.', confirmLabel: 'Delete', danger: true }))) return;
          await api.deleteWorksheet(w.id);
          const t = tabs.find((x) => x.worksheetId === w.id);
          if (t) { t.worksheetId = null; t.dirty = true; renderTabs(); persist(); }
          loadWorksheets();
        } }, icon('trash'))));
    const sortFn = (a, b) => (b.starred - a.starred) || String(b.updated_at).localeCompare(String(a.updated_at));
    mount(wsListEl,
      !mine.length && !shared.length ? empty({ icon: 'journal-code', title: q ? 'No matches' : 'No saved worksheets', text: q ? '' : `${modKey()}+S saves the current tab.` }) : null,
      [...folders.entries()].sort(([a], [b]) => (a === '' ? -1 : b === '' ? 1 : a.localeCompare(b))).map(([folder, list]) => [
        folder && h('div.ws-tree-group', icon('folder2', 'me-1'), folder),
        list.sort(sortFn).map((w) => item(w, true)),
      ]),
      shared.length ? [h('div.ws-tree-group', icon('people', 'me-1'), 'Shared with me'), shared.map((w) => item(w, false))] : null);
  }

  async function openWorksheet(w) {
    const existing = tabs.find((t) => t.worksheetId === w.id);
    if (existing) { switchTab(existing.id); return; }
    try {
      const full = await api.worksheet(w.id);
      const owned = full.owner === (window.APP_UI?.user || state.session?.user);
      const t = openTab({ title: full.title, sql: full.sql, worksheetId: owned ? full.id : null, context: Object.keys(full.context || {}).length ? { ...state.context, ...full.context } : null });
      t.dirty = !owned;
      if (!owned) t.title = `${full.title} (copy)`;
      renderTabs();
      persist();
    } catch (err) { toast(err.message, 'error', { title: 'Could not open worksheet' }); }
  }

  function promptText(title, value, onSave, allowEmpty = false) {
    const input = h('input.ws-input', { value, style: { width: '100%' } });
    const m = modal({ title, width: 420, body: input, actions: [{ label: 'Cancel' }, { label: 'Save', primary: true, onClick: async (close) => {
      const v = input.value.trim();
      if (!v && !allowEmpty) return;
      await onSave(v);
      close();
    } }] });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') m.el.querySelector('.ws-btn-primary')?.click(); });
  }

  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = h('a', { href: url, download: name });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  // ------------------------------------------------------------------ deep links
  const q = ctx.query;
  if (q.get('new') === '1') openTab({ title: `Worksheet ${tabs.length + 1}` });
  if (q.get('sql')) openTab({ title: 'From link', sql: q.get('sql') });
  if (q.get('worksheet')) openWorksheet({ id: q.get('worksheet') });
  if (q.get('query')) {
    const qid = q.get('query');
    const t = openTab({ title: `Query ${qid.slice(0, 8)}`, sql: `-- Loading SQL of query ${qid}…\n` });
    api.details(qid).then((dt) => {
      const text = dt.query?.query_text || '';
      t.sql = `-- From query ${qid}\n${text}${text.trim().endsWith(';') ? '' : ';'}\n`;
      if (t.id === activeId) editor.setValue(t.sql); else docs.delete(t.id);
      persist();
    }).catch((err) => toast(err.message, 'error', { title: 'Could not load the query' }));
  }
  if ([...q.keys()].length) ctx.setQuery({ new: null, sql: null, query: null, worksheet: null });

  // ------------------------------------------------------------------ go
  if (active()?.context && (active().context.database || active().context.warehouse)) setContext({ ...active().context }, { silent: true });
  renderTabs();
  renderToolbar();
  renderSide();
  renderResults();
  loadObjects();
  loadWorksheets();
  loadHints();
  persist();
  setTimeout(() => { editor.refresh(); editor.focus(); }, 30);

  const beforeUnload = (e) => {
    if (tabs.some((t) => t.running)) { e.preventDefault(); e.returnValue = ''; }
  };
  window.addEventListener('beforeunload', beforeUnload);
  ctx.cleanup(() => {
    window.removeEventListener('beforeunload', beforeUnload);
    tabs.forEach((t) => t.running?.abort.abort());
    disposeCharts();
    grid?.destroy();
    persist();
  });
}
