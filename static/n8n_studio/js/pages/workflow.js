// Workflow editor page (/workflows/:id and /workflows/new).

import { h, icon, mount, prefs, debounce } from '../../../ui/js/dom.js';
import { toast, toastError, modal, menuButton, openMenu, confirmDialog, tabs, spinner, jsonEditor } from '../../../ui/js/ui.js';
import { n8n, studio, enc, config, n8nLink, saveText, download } from '../core/api.js';
import { urls } from '../core/urls.js';
import { loadCatalog, catalog } from '../core/catalog.js';
import * as W from '../core/workflow.js';
import { UndoStack } from '../core/history.js';
import { nodeStatusMap, edgeCounts, ACTIVE, FAILED, toPinData, statusInfo } from '../core/execution.js';
import { load, invalidate, patchWorkflow, removeWorkflow, loadOptional } from '../core/store.js';
import { activeSwitch, inlineEdit, saveIndicator, execBadge, tagPills, MOD, promptText, licenseGate } from '../core/widgets.js';
import { createCanvas } from '../editor/canvas.js';
import { openNodePanel } from '../editor/ndv.js';
import { openPalette, insertNode } from '../editor/palette.js';
import { fmtDuration } from '../../../ui/js/format.js';
import { renderTab } from './workflow_tabs.js';

const DRAFT_KEY = (id) => `n8s:draft:${id}`;

export default async function render(ctx) {
  const isNew = !ctx.params.workflowId;
  await loadCatalog().catch(() => null);
  let remote;
  if (isNew) {
    remote = W.newWorkflow(ctx.query.get('name') || 'My workflow');
    const template = sessionStorage.getItem('n8s:import');
    if (template) {
      sessionStorage.removeItem('n8s:import');
      try { Object.assign(remote, JSON.parse(template)); } catch { /* ignore */ }
    }
  } else {
    remote = await n8n.get(`/workflows/${enc(ctx.params.workflowId)}`, undefined, { signal: ctx.signal });
  }

  const state = {
    wf: W.clone(remote),
    savedJson: JSON.stringify(W.saveBody(remote)),
    undo: new UndoStack(),
    execution: null,
    running: null,
    tab: ctx.query.get('tab') || 'editor',
    issues: [],
    credentials: [],
  };
  ctx.setTitle(state.wf.name);
  const dirty = () => JSON.stringify(W.saveBody(state.wf)) !== state.savedJson;

  // ---------------------------------------------------------------- header

  const indicator = saveIndicator();
  const nameEdit = inlineEdit(state.wf.name, async (v) => { state.wf.name = v; ctx.setTitle(v); changed(); }, { cls: 'n8s-wf-title' });
  const tagsEl = h('span.n8s-wf-tags');
  const activeEl = h('span');
  const runBtn = h('button.btn.btn-sm.btn-success.n8s-run-btn', { type: 'button', onclick: () => runWorkflow() }, icon('play-fill', 'me-1'), 'Run');
  const saveBtn = h('button.btn.btn-sm.btn-primary', { type: 'button', title: `Save (${MOD}+S)`, onclick: () => save() }, icon('cloud-check', 'me-1'), 'Save');
  const undoBtn = h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: `Undo (${MOD}+Z)`, onclick: () => undo() }, icon('arrow-counterclockwise'));
  const redoBtn = h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: `Redo (${MOD}+Shift+Z)`, onclick: () => redo() }, icon('arrow-clockwise'));

  const header = h('div.n8s-wf-header',
    h('a.n8s-back', { href: urls.workflows(), title: 'All workflows' }, icon('arrow-left')),
    h('div.n8s-wf-heading', h('div.d-flex.align-items-center.gap-2.flex-wrap', nameEdit, activeEl, state.wf.isArchived && h('span.badge.text-bg-secondary', icon('archive', 'me-1'), 'archived')), h('div.n8s-wf-meta', tagsEl, indicator.el)),
    h('div.n8s-wf-actions', h('div.btn-group', undoBtn, redoBtn), runBtn, saveBtn,
      menuButton('', () => moreMenu(), { title: 'More actions' })));

  const tabBar = tabs([
    { id: 'editor', label: 'Editor', icon: 'diagram-3' },
    { id: 'executions', label: 'Executions', icon: 'lightning-charge' },
    { id: 'versions', label: 'Versions', icon: 'clock-history' },
    { id: 'triggers', label: 'Triggers & webhooks', icon: 'broadcast' },
    { id: 'settings', label: 'Settings', icon: 'gear' },
    { id: 'tests', label: 'Evaluations', icon: 'clipboard-check' },
    { id: 'docs', label: 'Docs', icon: 'journal-text' },
  ], state.tab, (t) => { state.tab = t; ctx.setQuery({ tab: t === 'editor' ? null : t }); showTab(); });
  const tabBody = h('div.n8s-wf-tabbody');

  // ---------------------------------------------------------------- editor surface

  const canvasHost = h('div.n8s-canvas-host');
  const issuesEl = h('div.n8s-issues.d-none');
  const execBar = h('div.n8s-execbar');
  const findInput = h('input.form-control.form-control-sm.n8s-find', { type: 'search', placeholder: 'Find nodes…', value: ctx.query.get('find') || '' });
  const lintBtn = h('button.n8s-tool', { type: 'button', title: 'Problems (lint)', onclick: () => issuesEl.classList.toggle('d-none') }, icon('exclamation-triangle'), h('span.n8s-tool-count'));
  const tools = h('div.n8s-toolbar',
    h('button.n8s-tool.primary', { type: 'button', title: 'Add node (Tab)', onclick: () => addNode({ at: cv.viewCenter() }) }, icon('plus-lg')),
    h('button.n8s-tool', { type: 'button', title: 'Add sticky note (Shift+S)', onclick: () => addSticky() }, icon('sticky')),
    h('button.n8s-tool', { type: 'button', title: 'Tidy up — auto-layout (Shift+L)', onclick: () => tidy() }, icon('magic')),
    h('button.n8s-tool', { type: 'button', title: 'Toggle dark canvas', onclick: () => { const v = !prefs.get('n8s:dark', false); prefs.set('n8s:dark', v); cv.setDark(v); } }, icon('moon-stars')),
    lintBtn,
    findInput);
  const editorPane = h('div.n8s-editor', canvasHost, tools, issuesEl, execBar);

  const cv = createCanvas(canvasHost, {
    dark: prefs.get('n8s:dark', false),
    onChange: (next, meta) => apply(next, meta),
    onOpenNode: (name) => openNode(name),
    onAddRequest: (req) => addNode(req),
    onContextMenu: (e, target) => contextMenu(e, target),
  });
  ctx.cleanup(() => cv.destroy());

  mount(ctx.el, h('div.n8s-wf-page', header, tabBar, tabBody));

  function showTab() {
    if (state.tab === 'editor') {
      mount(tabBody, editorPane);
      requestAnimationFrame(() => { cv.setWorkflow(state.wf); if (!cv._fitted) { cv.fitView(); cv._fitted = true; } });
      return;
    }
    if (isNew && !state.wf.id) {
      mount(tabBody, h('div.n8s-empty', h('div.n8s-empty-title', 'Save the workflow first'), h('button.btn.btn-primary.btn-sm.mt-2', { onclick: () => save() }, 'Save now')));
      return;
    }
    renderTab(state.tab, tabBody, {
      ctx, state, wf: () => state.wf, cv,
      apply, save, reload, loadExecution, showEditor: () => { state.tab = 'editor'; tabBar.setActive('editor'); ctx.setQuery({ tab: null }); showTab(); },
    });
  }

  // ---------------------------------------------------------------- state changes

  function apply(next, meta = {}) {
    state.undo.push(state.wf, { group: meta.group || null });
    state.wf = next;
    changed();
  }

  const persistDraft = debounce(() => {
    const id = state.wf.id || 'new';
    try {
      if (dirty()) localStorage.setItem(DRAFT_KEY(id), JSON.stringify({ at: Date.now(), wf: W.saveBody(state.wf), base: state.savedJson.length }));
      else localStorage.removeItem(DRAFT_KEY(id));
    } catch { /* storage full or blocked */ }
  }, 600);

  function changed() {
    if (state.tab === 'editor') cv.setWorkflow(state.wf);
    indicator.set(dirty() ? 'dirty' : 'saved');
    undoBtn.disabled = !state.undo.canUndo;
    redoBtn.disabled = !state.undo.canRedo;
    relint();
    persistDraft();
  }

  function undo() {
    const prev = state.undo.undo(state.wf);
    if (prev) { state.wf = prev; changed(); }
  }

  function redo() {
    const next = state.undo.redo(state.wf);
    if (next) { state.wf = next; changed(); }
  }

  const relint = debounce(async () => {
    const credentials = new Map(state.credentials.map((c) => [String(c.id), c]));
    const workflows = await load('workflows').catch(() => []);
    state.issues = W.lint(state.wf, { catalog: catalog.nodes, credentials, workflowsById: new Map(workflows.map((w) => [String(w.id), w])) });
    const errors = state.issues.filter((i) => i.level === 'error').length;
    const warns = state.issues.filter((i) => i.level === 'warning').length;
    mount(lintBtn.querySelector('.n8s-tool-count'), errors + warns ? String(errors + warns) : '');
    lintBtn.classList.toggle('bad', errors > 0);
    lintBtn.classList.toggle('warn', !errors && warns > 0);
    mount(issuesEl,
      h('div.n8s-issues-head', icon('clipboard-pulse', 'me-1'), `Problems (${state.issues.length})`, h('button.af-icon-btn.ms-auto', { onclick: () => issuesEl.classList.add('d-none') }, icon('x'))),
      state.issues.length ? state.issues.map((i) => h('button.n8s-issue', {
        type: 'button', class: `lvl-${i.level}`,
        onclick: () => { if (i.node) { cv.select([i.node]); cv.center(i.node); cv.flash(i.node); } },
      }, icon(i.level === 'error' ? 'x-octagon-fill' : i.level === 'warning' ? 'exclamation-triangle-fill' : 'info-circle'), h('div', i.node && h('b', i.node, ': '), i.message)))
        : h('div.n8s-issues-ok', icon('check2-circle', 'me-1'), 'No problems found'));
  }, 400);

  // ---------------------------------------------------------------- header widgets

  function renderHeaderBits() {
    mount(activeEl, state.wf.id ? activeSwitch(state.wf, { onChange: (u) => { state.wf.active = u.active; state.wf.activeVersionId = u.activeVersionId; renderHeaderBits(); } }) : null,
      state.wf.id && state.wf.active && state.wf.activeVersionId && state.wf.versionId && state.wf.activeVersionId !== state.wf.versionId
        && h('button.btn.btn-sm.btn-warning.n8s-publish', { type: 'button', title: 'The active (published) version differs from the latest saved version', onclick: () => publish() }, icon('rocket-takeoff', 'me-1'), 'Publish changes'));
    mount(tagsEl, tagPills(state.wf.tags || []), state.wf.id && h('button.n8s-tag-add', { type: 'button', onclick: () => editTags() }, icon('tag'), state.wf.tags?.length ? '' : ' Add tags'));
  }

  async function editTags() {
    const all = await load('tags', { force: true });
    const chosen = new Set((state.wf.tags || []).map((t) => String(t.id)));
    const newInput = h('input.form-control.form-control-sm', { placeholder: 'New tag name, Enter to create' });
    const box = h('div.n8s-tagpick');
    const draw = () => mount(box, all.map((t) => h('label.n8s-tagpick-item', h('input.form-check-input', { type: 'checkbox', checked: chosen.has(String(t.id)), onchange: (e) => { e.target.checked ? chosen.add(String(t.id)) : chosen.delete(String(t.id)); } }), tagPills([t]))));
    newInput.addEventListener('keydown', async (e) => {
      if (e.key !== 'Enter' || !newInput.value.trim()) return;
      e.preventDefault();
      try {
        const t = await n8n.post('/tags', { name: newInput.value.trim() });
        all.push(t); chosen.add(String(t.id)); newInput.value = ''; draw(); invalidate('tags');
      } catch (err) { toastError(err, 'Could not create tag'); }
    });
    draw();
    modal({
      title: 'Workflow tags', icon: 'tags', size: 'sm', body: h('div', newInput, box),
      actions: [{ label: 'Cancel' }, {
        label: 'Save tags', variant: 'primary',
        onClick: async (close) => {
          const tags = await n8n.put(`/workflows/${enc(state.wf.id)}/tags`, [...chosen].map((id) => ({ id })));
          state.wf.tags = tags;
          patchWorkflow({ id: state.wf.id, tags });
          renderHeaderBits();
          close();
        },
      }],
    });
  }

  // ---------------------------------------------------------------- save / publish

  async function save({ quiet = false } = {}) {
    indicator.set('saving');
    try {
      const body = W.saveBody(state.wf);
      let saved;
      if (!state.wf.id) {
        saved = await n8n.post('/workflows', body);
        toast('Workflow created', 'success', { timeout: 2000 });
        try { localStorage.removeItem(DRAFT_KEY('new')); } catch { /* ignore */ }
        Object.assign(state.wf, { id: saved.id, versionId: saved.versionId, active: saved.active });
        // Compare against what we sent: n8n's echo adds defaults that would look "dirty".
        state.savedJson = JSON.stringify(W.saveBody(state.wf));
        patchWorkflow(saved);
        ctx.navigate(urls.workflow(saved.id), { replace: true });
        return saved;
      }
      saved = await n8n.put(`/workflows/${enc(state.wf.id)}`, body);
      Object.assign(state.wf, { versionId: saved.versionId, activeVersionId: saved.activeVersionId, active: saved.active, updatedAt: saved.updatedAt });
      state.savedJson = JSON.stringify(W.saveBody({ ...state.wf }));
      patchWorkflow(saved);
      indicator.set('saved', `Saved ${new Date().toLocaleTimeString()}`);
      if (!quiet) toast('Saved', 'success', { timeout: 1500 });
      renderHeaderBits();
      relint();
      try { localStorage.removeItem(DRAFT_KEY(state.wf.id)); } catch { /* ignore */ }
      return saved;
    } catch (err) {
      indicator.set('error');
      toastError(err, 'Save failed');
      throw err;
    }
  }

  async function publish() {
    if (dirty()) await save({ quiet: true });
    try {
      const out = await n8n.post(`/workflows/${enc(state.wf.id)}/publish`, {});
      Object.assign(state.wf, { active: out.active, activeVersionId: out.activeVersionId, versionId: out.versionId });
      patchWorkflow(out);
      renderHeaderBits();
      toast('Published — production triggers now use this version.', 'success');
    } catch (err) { toastError(err, 'Publish failed'); }
  }

  async function reload() {
    const fresh = await n8n.get(`/workflows/${enc(state.wf.id)}`);
    state.wf = fresh;
    state.savedJson = JSON.stringify(W.saveBody(fresh));
    state.undo.clear();
    changed();
    renderHeaderBits();
  }

  // ---------------------------------------------------------------- nodes

  async function addNode(req = {}) {
    if (!catalog.ready) await loadCatalog();
    const type = await openPalette(state.wf, req);
    if (!type) return;
    const { wf, node } = insertNode(state.wf, type, req);
    apply(wf, { label: 'Add node' });
    cv.select([node.name]);
    if (type !== W.STICKY) setTimeout(() => openNode(node.name), 60);
  }

  function addSticky() {
    const { wf, node } = insertNode(state.wf, W.STICKY, { at: cv.viewCenter() });
    apply(wf, { label: 'Add note' });
    cv.select([node.name]);
  }

  function openNode(name) {
    openNodePanel({
      wf: state.wf, name, execution: state.execution, credentials: state.credentials,
      onChange: (next, meta) => apply(next, meta),
      onRun: config.canRun ? (dest) => runWorkflow({ destinationNode: dest }) : null,
    });
  }

  function tidy() {
    apply(W.autoLayout(state.wf), { label: 'Tidy up' });
    requestAnimationFrame(() => cv.fitView());
  }

  function deleteSelection() {
    const sel = cv.selection();
    const edge = cv.selectedEdge();
    if (sel.length) apply(W.removeNodes(state.wf, sel), { label: 'Delete' });
    else if (edge) apply(W.removeConnection(state.wf, edge), { label: 'Delete connection' });
  }

  function toggleDisable(names = cv.selection()) {
    if (!names.length) return;
    const allDisabled = names.every((n) => state.wf.nodes.find((x) => x.name === n)?.disabled);
    apply({ ...state.wf, nodes: state.wf.nodes.map((n) => {
      if (!names.includes(n.name) || W.isSticky(n)) return n;
      const c = { ...n };
      if (allDisabled) delete c.disabled; else c.disabled = true;
      return c;
    }) }, { label: allDisabled ? 'Enable' : 'Disable' });
  }

  function togglePin(name = cv.selection()[0]) {
    if (!name) return;
    const pd = { ...(state.wf.pinData || {}) };
    if (pd[name]) delete pd[name];
    else {
      const pins = toPinData(state.execution, [name]);
      if (!pins[name]) { toast('No output to pin — run the workflow first, or edit pinned data in the node panel.', 'info'); return; }
      pd[name] = pins[name];
    }
    apply({ ...state.wf, pinData: pd }, { label: 'Toggle pin' });
  }

  async function renameSelected() {
    const name = cv.selection()[0];
    if (!name) return;
    const v = await promptText({ title: 'Rename node', value: name, validate: (x) => (!x ? 'Name is required' : x !== name && state.wf.nodes.some((n) => n.name === x) ? 'That name is taken' : null) });
    if (v && v !== name) { apply(W.renameNode(state.wf, name, v), { label: 'Rename' }); cv.select([v]); }
  }

  async function copySelection(cut = false) {
    const sel = cv.selection();
    if (!sel.length) return;
    const text = W.toClipboard(state.wf, sel);
    try { await navigator.clipboard.writeText(text); } catch { prefs.set('n8s:clipboard', text); }
    prefs.set('n8s:clipboard', text);
    toast(`${cut ? 'Cut' : 'Copied'} ${sel.length} node${sel.length > 1 ? 's' : ''} — paste here or in the n8n editor`, 'info', { timeout: 1800 });
    if (cut) apply(W.removeNodes(state.wf, sel), { label: 'Cut' });
  }

  async function paste(text) {
    let raw = text;
    if (raw == null) { try { raw = await navigator.clipboard.readText(); } catch { raw = prefs.get('n8s:clipboard', ''); } }
    const clip = W.parseClipboard(raw || '');
    if (!clip) { toast('Clipboard does not contain n8n nodes.', 'warning'); return; }
    const at = cv.viewCenter();
    const { wf, names } = W.pasteInto(state.wf, clip, { at });
    apply(wf, { label: 'Paste' });
    cv.select(names);
  }

  function duplicateSelection() {
    const sel = cv.selection();
    if (!sel.length) return;
    const { wf, names } = W.pasteInto(state.wf, W.parseClipboard(W.toClipboard(state.wf, sel)), { offset: [40, 60] });
    apply(wf, { label: 'Duplicate' });
    cv.select(names);
  }

  function contextMenu(e, target) {
    const sel = cv.selection();
    const anchor = h('span', { style: `position:fixed;left:${e.clientX}px;top:${e.clientY}px;width:1px;height:1px` });
    document.body.appendChild(anchor);
    setTimeout(() => anchor.remove(), 0);
    const items = target.node ? [
      { header: sel.length > 1 ? `${sel.length} nodes` : target.node },
      sel.length === 1 && { label: 'Open', icon: 'box-arrow-up-right', onClick: () => openNode(target.node) },
      sel.length === 1 && config.canRun && { label: 'Execute up to here', icon: 'play', onClick: () => runWorkflow({ destinationNode: target.node }) },
      sel.length === 1 && { label: 'Rename', icon: 'input-cursor-text', onClick: () => renameSelected() },
      { label: 'Copy', icon: 'clipboard', onClick: () => copySelection() },
      { label: 'Duplicate', icon: 'copy', onClick: () => duplicateSelection() },
      { label: 'Disable / enable', icon: 'slash-circle', onClick: () => toggleDisable() },
      sel.length === 1 && { label: state.wf.pinData?.[target.node] ? 'Unpin data' : 'Pin output', icon: 'pin-angle', onClick: () => togglePin(target.node) },
      { label: 'Select upstream', icon: 'arrow-left-square', onClick: () => cv.select([target.node, ...W.ancestors(state.wf, target.node)]) },
      { divider: true },
      { label: 'Delete', icon: 'trash', danger: true, onClick: () => deleteSelection() },
    ] : [
      { label: 'Add node', icon: 'plus-lg', onClick: () => addNode({ at: target.world.map((v) => W.snap(v - 50)) }) },
      { label: 'Add sticky note', icon: 'sticky', onClick: () => { const { wf } = insertNode(state.wf, W.STICKY, { at: target.world }); apply(wf, { label: 'Add note' }); } },
      { label: 'Paste', icon: 'clipboard-plus', onClick: () => paste() },
      { label: 'Select all', icon: 'check2-all', onClick: () => cv.select(state.wf.nodes.map((n) => n.name)) },
      { label: 'Tidy up', icon: 'magic', onClick: () => tidy() },
      { label: 'Fit view', icon: 'arrows-fullscreen', onClick: () => cv.fitView() },
    ];
    openMenu(anchor, items.filter(Boolean), 'start');
  }

  // ---------------------------------------------------------------- run & execution overlay

  async function runWorkflow({ destinationNode, triggerName } = {}) {
    if (!config.canRun) {
      modal({
        title: 'Manual runs need an n8n login', icon: 'play-circle',
        body: h('div', h('p', 'n8n\'s public API cannot start executions. Set ', h('code', 'N8N_STUDIO_EMAIL'), ' and ', h('code', 'N8N_STUDIO_PASSWORD'), ' in the dashboard .env (a non-MFA n8n user) and restart, and the Run button will work exactly like the editor\'s.'),
          h('p.mb-0', 'Meanwhile: webhook-triggered workflows can be fired from the ', h('b', 'Triggers & webhooks'), ' tab, and failed runs can be retried from Executions.')),
        actions: [{ label: 'Close' }, { label: 'Open webhook tester', variant: 'primary', onClick: (close) => { close(); state.tab = 'triggers'; tabBar.setActive('triggers'); showTab(); } }],
      });
      return;
    }
    const trig = W.triggers(state.wf, catalog.nodes).filter((n) => !n.disabled);
    if (!destinationNode && !triggerName && trig.length > 1) {
      const pick = await new Promise((resolve) => {
        const anchor = runBtn;
        openMenu(anchor, [{ header: 'Start from trigger' }, ...trig.map((t) => ({ label: t.name, icon: 'lightning-charge', onClick: () => resolve(t.name) }))]);
        setTimeout(() => resolve(null), 15000);
      });
      if (!pick) return;
      triggerName = pick;
    }
    if (!state.wf.id) {
      toast('Saving the workflow before its first run…', 'info', { timeout: 1500 });
      await save({ quiet: true });
      return;
    }
    runBtn.disabled = true;
    mount(runBtn, spinner('sm'), ' Running');
    try {
      const res = await studio.post(`/run/${enc(state.wf.id)}/`, { workflowData: { ...W.saveBody(state.wf), id: state.wf.id }, destinationNode, triggerName });
      if (res.waitingForWebhook) {
        toast('Waiting for a test webhook call — send a request to the test URL (Triggers & webhooks tab).', 'info', { timeout: 10000 });
      }
      if (res.executionId) await followExecution(res.executionId);
    } catch (err) {
      toastError(err, 'Run failed');
    } finally {
      runBtn.disabled = false;
      mount(runBtn, icon('play-fill', 'me-1'), 'Run');
    }
  }

  async function followExecution(id) {
    state.running = id;
    const started = Date.now();
    while (ctx.active && state.running === id) {
      let e;
      try { e = await n8n.get(`/executions/${enc(id)}`, { includeData: true }); } catch (err) {
        if (err.status === 404 && Date.now() - started < 15000) { await sleep(600); continue; }
        throw err;
      }
      showExecution(e);
      if (!ACTIVE.has(e.status) && e.stoppedAt) {
        state.running = null;
        const info = statusInfo(e.status);
        toast(`Execution #${e.id} ${info.label.toLowerCase()} in ${fmtDuration((new Date(e.stoppedAt) - new Date(e.startedAt)) / 1000)}`, FAILED.has(e.status) ? 'error' : 'success', { action: { label: 'Open', onClick: () => ctx.navigate(urls.execution(e.id)) } });
        return e;
      }
      await sleep(Date.now() - started < 10000 ? 700 : 2000);
    }
    return null;
  }

  function showExecution(e) {
    state.execution = e;
    cv.setStatus(e ? nodeStatusMap(e) : new Map(), e ? edgeCounts(e) : new Map());
    renderExecBar();
  }

  async function loadExecution(id) {
    const e = await n8n.get(`/executions/${enc(id)}`, { includeData: true });
    showExecution(e);
    return e;
  }

  function renderExecBar() {
    const e = state.execution;
    if (!e) {
      mount(execBar, h('span.text-muted.small', icon('info-circle', 'me-1'), state.wf.id ? 'No execution loaded. ' : 'Unsaved workflow. '),
        state.wf.id && h('button.af-link-btn.small', { type: 'button', onclick: () => loadLatest(true) }, 'Load latest execution'));
      return;
    }
    mount(execBar,
      execBadge(e.status, { size: 'sm' }),
      h('span.small', h('a', { href: urls.execution(e.id) }, `#${e.id}`), ` · ${e.mode} · `, e.stoppedAt ? fmtDuration((new Date(e.stoppedAt) - new Date(e.startedAt)) / 1000) : 'running…'),
      ACTIVE.has(e.status) && h('button.btn.btn-sm.btn-outline-danger', { type: 'button', onclick: async () => { try { await n8n.post(`/executions/${enc(e.id)}/stop`, {}); toast('Stop requested', 'info'); } catch (err) { toastError(err, 'Stop failed'); } } }, icon('stop-fill'), ' Stop'),
      h('div.flex-grow-1'),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Pin all node outputs from this execution (n8n\'s "Debug in editor")', onclick: () => debugInEditor(e) }, icon('bug', 'me-1'), 'Debug in editor'),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Clear execution overlay', onclick: () => { state.running = null; showExecution(null); } }, icon('x-lg')));
  }

  function debugInEditor(e) {
    const pins = toPinData(e);
    const trig = new Set(W.triggers(state.wf, catalog.nodes).map((n) => n.name));
    const onlyTriggers = Object.fromEntries(Object.entries(pins).filter(([k]) => trig.has(k)));
    const use = Object.keys(onlyTriggers).length ? onlyTriggers : pins;
    apply({ ...state.wf, pinData: { ...(state.wf.pinData || {}), ...use } }, { label: 'Debug in editor' });
    toast(`Pinned ${Object.keys(use).join(', ')} from execution #${e.id}. Run again to replay with the same data; save to keep the pins.`, 'success', { timeout: 6000 });
  }

  async function loadLatest(explicit = false) {
    if (!state.wf.id) return;
    try {
      const list = await n8n.get('/executions', { workflowId: state.wf.id, limit: 1 });
      const latest = list.data?.[0];
      if (latest) await loadExecution(latest.id);
      else if (explicit) toast('This workflow has no executions yet.', 'info');
    } catch (err) { if (explicit) toastError(err); }
  }

  // ---------------------------------------------------------------- more menu

  function moreMenu() {
    const id = state.wf.id;
    return [
      { header: 'Workflow' },
      id && { label: 'Duplicate', icon: 'copy', onClick: () => duplicateWorkflow() },
      { label: 'Download JSON', icon: 'download', onClick: () => saveText(JSON.stringify({ ...W.saveBody(state.wf), tags: state.wf.tags || [] }, null, 2), `${slug(state.wf.name)}.json`) },
      { label: 'Copy JSON', icon: 'clipboard', onClick: async () => { await navigator.clipboard.writeText(JSON.stringify(W.saveBody(state.wf), null, 2)); toast('Workflow JSON copied', 'success', { timeout: 1500 }); } },
      { label: 'Edit JSON…', icon: 'braces', onClick: () => editJson() },
      id && { label: 'Export n8n package (.n8np)', icon: 'box-seam', onClick: () => exportPackage() },
      { divider: true },
      id && { label: 'Transfer to project…', icon: 'arrow-left-right', onClick: () => transfer() },
      id && (state.wf.isArchived
        ? { label: 'Unarchive', icon: 'archive', onClick: () => archive(false) }
        : { label: 'Archive', icon: 'archive', onClick: () => archive(true) }),
      id && { label: 'Delete…', icon: 'trash', danger: true, onClick: () => deleteWorkflow() },
      { divider: true },
      id && { label: 'Open in n8n', icon: 'box-arrow-up-right', href: n8nLink(`/workflow/${id}`), external: true },
      { label: 'Keyboard shortcuts', icon: 'keyboard', onClick: () => import('../app.js').then((m) => m.showShortcuts()) },
    ];
  }

  async function duplicateWorkflow() {
    const name = await promptText({ title: 'Duplicate workflow', label: 'Name', value: `${state.wf.name} (copy)` });
    if (!name) return;
    try {
      const copy = await n8n.post('/workflows', W.importBody(state.wf, { name }));
      if (state.wf.tags?.length) await n8n.put(`/workflows/${enc(copy.id)}/tags`, state.wf.tags.map((t) => ({ id: t.id }))).catch(() => null);
      invalidate('workflows');
      toast('Duplicated (webhook paths got new ids so the copy does not clash)', 'success', { action: { label: 'Open', onClick: () => ctx.navigate(urls.workflow(copy.id)) } });
    } catch (err) { toastError(err, 'Duplicate failed'); }
  }

  function editJson() {
    const editor = jsonEditor(W.saveBody(state.wf), { rows: 26, allowEmpty: false });
    modal({
      title: 'Workflow JSON', icon: 'braces', size: 'xl', body: editor.el,
      actions: [{ label: 'Cancel' }, {
        label: 'Apply', variant: 'primary',
        onClick: (close) => {
          const v = editor.value();
          if (!Array.isArray(v.nodes) || typeof v.connections !== 'object') throw new Error('JSON needs "nodes" and "connections".');
          apply({ ...state.wf, ...v }, { label: 'Edit JSON' });
          nameEdit.setValue(state.wf.name);
          close();
        },
      }],
    });
  }

  async function exportPackage() {
    try {
      const out = await download('POST', '/n8n-packages/export', { workflowIds: [state.wf.id], includeTags: true, missingWorkflowDependencyPolicy: 'reference-only' }, `${slug(state.wf.name)}.n8np`);
      toast(`Downloaded ${out.name}`, 'success');
    } catch (err) { toastError(err, 'Export failed'); }
  }

  async function transfer() {
    let projects;
    try { projects = await load('projects'); } catch (err) {
      if (err.license) { modal({ title: 'Transfer', body: licenseGate(err) }); return; }
      throw err;
    }
    const sel = h('select.form-select.form-select-sm', projects.map((p) => h('option', { value: p.id }, `${p.name} (${p.type})`)));
    modal({
      title: 'Transfer to project', icon: 'arrow-left-right', size: 'sm', body: h('div', h('label.form-label.small', 'Destination project'), sel),
      actions: [{ label: 'Cancel' }, {
        label: 'Transfer', variant: 'primary',
        onClick: async (close) => { await n8n.put(`/workflows/${enc(state.wf.id)}/transfer`, { destinationProjectId: sel.value }); toast('Transferred', 'success'); invalidate('workflows'); close(); },
      }],
    });
  }

  async function archive(on) {
    try {
      const out = await n8n.post(`/workflows/${enc(state.wf.id)}/${on ? 'archive' : 'unarchive'}`, {});
      state.wf.isArchived = out.isArchived ?? on;
      state.wf.active = out.active;
      patchWorkflow(out);
      toast(on ? 'Archived' : 'Restored from archive', 'success');
      ctx.navigate(urls.workflow(state.wf.id), { replace: true });
    } catch (err) { toastError(err); }
  }

  async function deleteWorkflow() {
    const ok = await confirmDialog({
      title: 'Delete workflow?', danger: true, confirmLabel: 'Delete',
      message: h('span', 'Delete ', h('b', state.wf.name), '? Studio keeps a snapshot, so you can restore it from ', h('b', 'Backups → Recycle bin'), '.'),
    });
    if (!ok) return;
    try {
      await n8n.del(`/workflows/${enc(state.wf.id)}`);
      removeWorkflow(state.wf.id);
      toast('Workflow deleted (recoverable from the recycle bin)', 'success');
      ctx.navigate(urls.workflows());
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  // ---------------------------------------------------------------- keyboard

  const onKey = (e) => {
    if (state.tab !== 'editor' || document.querySelector('.af-backdrop')) return;
    const typing = e.target.closest?.('input, textarea, select, [contenteditable=true]');
    const mod = e.metaKey || e.ctrlKey;
    const k = e.key.toLowerCase();
    if (mod && k === 's') { e.preventDefault(); save(); return; }
    if (typing) return;
    if (mod && k === 'enter') { e.preventDefault(); runWorkflow(); return; }
    if (mod && k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (mod && k === 'y') { e.preventDefault(); redo(); return; }
    if (mod && k === 'a') { e.preventDefault(); cv.select(state.wf.nodes.map((n) => n.name)); return; }
    if (mod && k === 'c') { copySelection(); return; }
    if (mod && k === 'x') { copySelection(true); return; }
    if (mod && k === 'd') { e.preventDefault(); duplicateSelection(); return; }
    if (mod && k === 'f') { e.preventDefault(); findInput.focus(); return; }
    if (mod) return;
    if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); deleteSelection(); return; }
    if (e.key === 'Tab') { e.preventDefault(); const s = cv.selection()[0]; addNode(s ? { from: { node: s, type: 'main', index: 0 } } : { at: cv.viewCenter() }); return; }
    if (e.key === 'Enter') { const s = cv.selection()[0]; if (s) { e.preventDefault(); openNode(s); } return; }
    if (e.key === 'F2') { e.preventDefault(); renameSelected(); return; }
    if (k === 'd') { toggleDisable(); return; }
    if (k === 'p') { togglePin(); return; }
    if (e.key === '1') { cv.fitView(); return; }
    if (e.key === '0') { cv.zoomTo(1); return; }
    if (e.key === '+' || e.key === '=') { cv.zoomBy(1.2); return; }
    if (e.key === '-') { cv.zoomBy(1 / 1.2); return; }
    if (e.shiftKey && k === 'l') { tidy(); return; }
    if (e.shiftKey && k === 's') { addSticky(); return; }
    if (e.key === 'Escape') cv.select([]);
  };
  document.addEventListener('keydown', onKey);
  ctx.cleanup(() => document.removeEventListener('keydown', onKey));

  const onPaste = (e) => {
    if (state.tab !== 'editor' || e.target.closest?.('input, textarea, [contenteditable=true]') || document.querySelector('.af-backdrop')) return;
    const text = e.clipboardData?.getData('text/plain');
    if (text && W.parseClipboard(text)) { e.preventDefault(); paste(text); }
  };
  document.addEventListener('paste', onPaste);
  ctx.cleanup(() => document.removeEventListener('paste', onPaste));

  const beforeUnload = (e) => { if (dirty()) { e.preventDefault(); e.returnValue = ''; } };
  window.addEventListener('beforeunload', beforeUnload);
  ctx.cleanup(() => window.removeEventListener('beforeunload', beforeUnload));

  findInput.addEventListener('input', debounce(() => {
    const q = findInput.value.trim();
    if (!q) { cv.setHighlight(null); return; }
    const terms = W.parseQuery(q);
    const hits = state.wf.nodes.filter((n) => W.matchWorkflow({ nodes: [n], name: '', tags: [] }, terms)).map((n) => n.name);
    cv.setHighlight(hits);
    if (hits.length) { cv.fitView(hits); }
  }, 200));
  findInput.addEventListener('keydown', (e) => { if (e.key === 'Escape') { findInput.value = ''; cv.setHighlight(null); cv.el.focus(); } });

  // ---------------------------------------------------------------- boot

  renderHeaderBits();
  showTab();
  changed();
  state.undo.clear();
  undoBtn.disabled = true;
  redoBtn.disabled = true;
  if (ctx.query.get('find')) findInput.dispatchEvent(new Event('input'));

  loadOptional('credentials').then((c) => { state.credentials = c; relint(); });
  if (!isNew) {
    if (ctx.query.get('execution')) loadExecution(ctx.query.get('execution')).catch(toastError);
    else loadLatest();
  } else renderExecBar();

  // Offer to restore a local draft that is newer than the server copy.
  try {
    const raw = localStorage.getItem(DRAFT_KEY(state.wf.id || 'new'));
    if (raw) {
      const draft = JSON.parse(raw);
      const serverAt = state.wf.updatedAt ? new Date(state.wf.updatedAt).getTime() : 0;
      if (draft.wf && JSON.stringify(draft.wf) !== state.savedJson && draft.at > serverAt) {
        toast(`You have unsaved changes from ${new Date(draft.at).toLocaleString()} (browser draft).`, 'warning', {
          title: 'Draft found', timeout: 0,
          action: { label: 'Restore draft', onClick: () => { apply({ ...state.wf, ...draft.wf }, { label: 'Restore draft' }); nameEdit.setValue(state.wf.name); indicator.set('draft'); } },
        });
      }
    }
  } catch { /* ignore */ }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const slug = (s) => String(s || 'workflow').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'workflow';

export { slug };
