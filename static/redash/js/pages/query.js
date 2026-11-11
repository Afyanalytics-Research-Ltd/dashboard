// Query editor: schema browser | SQL editor + parameters | results & visualizations.
// Saves through the proxy (which records version history), autosaves local drafts,
// detects concurrent edits (409), runs selection or whole query with live job status.

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import {
  tabs, toast, toastError, confirmDialog, modal, drawer, emptyState, errorState, skeleton,
  spinner, timeAgo, openMenu, formModal,
} from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { fmtDuration } from '../../../ui/js/format.js';
import { rd, enc, runQuery, cancelJob, fileUrl, redashLink } from '../core/api.js';
import { urls } from '../core/urls.js';
import { syncParams, paramBar, paramDefinitionForm, serializeParams, missingParams } from '../core/params.js';
import { renderViz, suggestViz, vizTypeInfo } from '../core/viz.js';
import { resultsGrid, downloadCsv } from '../core/results.js';
import { sqlEditor } from '../core/editor.js';
import {
  favoriteStar, inlineEdit, tagEditor, userChip, scheduleText, getDataSources, getSnippets, getQueryTags,
  getOrgSettings, dashboardsUsing, invalidate, dataSourceIcon,

} from '../core/common.js';
import { schemaBrowser, errorText } from './query/schema.js';
import { openBuilder } from './query/builder.js';
import { openHistory } from './query/history.js';
import { scheduleDialog, apiKeyDialog, permissionsDialog, editorSettingsDialog, conflictDialog } from './query/dialogs.js';

const NEW_NAME = 'New Query';
const draftKey = (id) => `rd:draft:${id || 'new'}`;

// ------------------------------------------------------------------ recent query tabs

const RECENT_KEY = 'rd:recentQueries';
function rememberRecent(q) {
  const list = prefs.get(RECENT_KEY, []).filter((x) => x.id !== q.id);
  list.unshift({ id: q.id, name: q.name });
  prefs.set(RECENT_KEY, list.slice(0, 8));
}
function forgetRecent(id) { prefs.set(RECENT_KEY, prefs.get(RECENT_KEY, []).filter((x) => x.id !== id)); }

// Optional modules owned by the dashboards fork.
async function optional(path, name) {
  try {
    const mod = await import(path);
    if (typeof mod[name] === 'function') return mod[name];
  } catch { /* not built yet */ }
  toast('This action is not available yet.', 'warning');
  return null;
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  const queryId = ctx.params.queryId ? Number(ctx.params.queryId) : null;
  mount(el, skeleton(10));

  const [dataSources, snippets, settings, tagList, query] = await Promise.all([
    getDataSources().catch(() => []),
    getSnippets().catch(() => []),
    getOrgSettings().catch(() => ({})),
    getQueryTags().catch(() => []),
    queryId ? rd.get(`/api/queries/${enc(queryId)}`, null, { signal }) : Promise.resolve(null),
  ]);
  if (!ctx.active) return;

  // ---------------------------------------------------------------- state

  const preferredDs = Number(ctx.query.get('ds')) || prefs.get('rd:lastDs', null);
  const firstDs = dataSources.find((d) => d.id === preferredDs) || dataSources[0];
  let server = query || {
    id: null, name: NEW_NAME, query: '', description: '', data_source_id: firstDs?.id, options: { parameters: [] },
    tags: [], visualizations: [], is_draft: true, is_archived: false, schedule: null, can_edit: true,
  };
  const s = {
    text: server.query || '',
    name: server.name || NEW_NAME,
    dsId: server.data_source_id,
    params: structuredClone(server.options?.parameters || []),
    values: {},
    result: null,
    running: null, // {jobId, started, label}
    activeViz: Number(ctx.query.get('viz')) || null,
    controller: null,
  };
  s.params.forEach((p) => { s.values[p.name] = p.value; });
  const canEdit = server.can_edit !== false;
  if (server.id) { rememberRecent(server); ctx.setTitle(server.name); } else ctx.setTitle('New query');

  const isDirty = () => s.text !== (server.query || '') || s.name !== (server.name || NEW_NAME)
    || s.dsId !== server.data_source_id
    || JSON.stringify(s.params.map(({ value, ...rest }) => rest)) !== JSON.stringify((server.options?.parameters || []).map(({ value, ...rest }) => rest));

  // ---------------------------------------------------------------- layout

  const recentEl = h('div.rd-qe-recent');
  const headEl = h('div.rd-qe-head');
  const bannerEl = h('div.rd-qe-banners');
  const schemaWrap = h('aside.rd-qe-schema');
  const editorHost = h('div.rd-qe-editor');
  const paramsEl = h('div.rd-qe-params');
  const toolbarEl = h('div.rd-qe-toolbar');
  const resultsEl = h('section.rd-qe-results.af-card');
  const vsplit = h('div.rd-qe-vsplit', { title: 'Drag to resize the editor' });
  const hsplit = h('div.rd-qe-hsplit', { title: 'Drag to resize the schema browser' });
  const workEl = h('div.rd-qe-work', toolbarEl, editorHost, vsplit, paramsEl);
  const topEl = h('div.rd-qe-top', schemaWrap, hsplit, workEl);
  mount(el, recentEl, headEl, bannerEl, topEl, resultsEl);

  let schemaW = prefs.get('rd:schemaW', 270);
  let editorH = prefs.get('rd:editorH', 300);
  // On phones the schema starts collapsed (one tap on the sidebar icon opens it).
  let schemaOpen = window.innerWidth < 768 ? false : prefs.get('rd:schemaOpen', true);
  const applySizes = () => {
    topEl.style.setProperty('--rd-schema-w', `${schemaOpen ? schemaW : 0}px`);
    editorHost.style.height = `${editorH}px`;
    topEl.classList.toggle('schema-closed', !schemaOpen);
    editor?.refresh();
  };
  dragResize(hsplit, 'x', (dx, start) => { schemaW = Math.min(Math.max(start + dx, 180), 560); applySizes(); }, () => schemaW, () => prefs.set('rd:schemaW', schemaW));
  dragResize(vsplit, 'y', (dy, start) => { editorH = Math.min(Math.max(start + dy, 140), 900); applySizes(); }, () => editorH, () => prefs.set('rd:editorH', editorH));

  // ---------------------------------------------------------------- recent tabs

  function drawRecent() {
    const list = prefs.get(RECENT_KEY, []);
    if (list.length < 2 && server.id) { mount(recentEl); return; }
    mount(recentEl, list.map((r) => h('a.rd-qe-rtab', { href: urls.query(r.id), class: r.id === server.id && 'active', title: r.name },
      icon('code-square'), h('span', r.name),
      h('button.rd-qe-rtab-x', {
        type: 'button', title: 'Close tab',
        onclick: (e) => { e.preventDefault(); e.stopPropagation(); forgetRecent(r.id); drawRecent(); },
      }, icon('x')))),
    !server.id && h('span.rd-qe-rtab.active', icon('plus-square'), h('span', 'New query')),
    h('a.rd-qe-rtab.rd-qe-rtab-new', { href: urls.newQuery({ ds: s.dsId }), title: 'New query' }, icon('plus-lg')));
  }

  // ---------------------------------------------------------------- header

  function drawHead() {
    const ds = dataSources.find((d) => d.id === s.dsId);
    const saveBtn = h('button.btn.btn-sm.btn-primary.rd-qe-save', { type: 'button', onclick: () => save(), disabled: !canEdit, title: 'Save (Ctrl+S)' },
      icon('cloud-check', 'me-1'), server.id ? 'Save' : 'Save query');
    saveBtn.classList.toggle('dirty', isDirty());
    mount(headEl,
      h('div.rd-qe-title-row',
        server.id && favoriteStar('queries', server),
        h('h1.rd-qe-title', inlineEdit(s.name, async (v) => {
          s.name = v || NEW_NAME;
          if (server.id) await patch({ name: s.name, ...(server.is_draft && s.name !== NEW_NAME ? { is_draft: false } : {}) }, 'Renamed');
          else drawHead();
          ctx.setTitle(s.name);
        }, { placeholder: 'Name this query', disabled: !canEdit })),
        server.is_draft && server.id && h('span.rd-draft-pill', { title: 'Drafts are hidden from other users’ lists' }, 'draft'),
        server.is_archived && h('span.rd-archived-pill', 'archived'),
        h('div.rd-qe-head-actions',
          h('span.rd-qe-dirty', { class: isDirty() && 'show' }, icon('circle-fill'), 'Unsaved changes'),
          saveBtn,
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'More actions', 'aria-haspopup': 'menu', onclick: (e) => { e.stopPropagation(); openMenu(e.currentTarget, moreItems()); } }, icon('three-dots')))),
      h('div.rd-qe-meta',
        server.id && h('span.af-mono', `#${server.id}`),
        ds && h('span.rd-type-pill', icon(dataSourceIcon(ds.type)), ds.name),
        server.user && h('span', userChip(server.user)),
        server.updated_at && h('span', icon('pencil', 'me-1'), 'Updated ', timeAgo(server.updated_at)),
        server.id && h('button.rd-qe-sched', { type: 'button', onclick: editSchedule, disabled: !canEdit, title: 'Change refresh schedule' },
          icon(server.schedule?.interval ? 'arrow-repeat' : 'hand-index', 'me-1'), server.schedule?.interval ? scheduleText(server.schedule) : 'Manual refresh'),
        server.id && tagEditor(server.tags, async (tags) => { await patch({ tags }); invalidate('qtags'); }, { readOnly: !canEdit, suggestions: tagList.map((t) => t.name) })),
      server.id && h('div.rd-qe-desc', inlineEdit(server.description || '', async (v) => { await patch({ description: v }, 'Description saved'); }, {
        placeholder: 'Add a description — what does this query answer?', multiline: true, disabled: !canEdit,
      })));
  }

  function moreItems() {
    const id = server.id;
    return [
      { header: 'Query' },
      { label: 'Version history', icon: 'clock-history', disabled: !id, onClick: () => openHistory({ queryId: id, current: () => ({ query: s.text, name: s.name }), onRestore: restoreRevision }) },
      { label: 'Fork (make a copy)', icon: 'diagram-2', disabled: !id, onClick: fork },
      { label: 'Refresh schedule…', icon: 'calendar2-week', disabled: !id || !canEdit, onClick: editSchedule },
      { label: server.is_draft ? 'Publish' : 'Unpublish (make draft)', icon: server.is_draft ? 'megaphone' : 'eye-slash', disabled: !id || !canEdit, onClick: () => patch({ is_draft: !server.is_draft }, server.is_draft ? 'Published' : 'Moved to drafts') },
      { label: 'Used in dashboards…', icon: 'grid-1x2', disabled: !id, onClick: showUsage },
      { divider: true },
      { header: 'Share' },
      { label: 'API key & result links…', icon: 'key', disabled: !id, onClick: () => apiKeyDialog(server) },
      { label: 'Who can edit…', icon: 'shield-lock', disabled: !id, onClick: () => permissionsDialog(server) },
      { label: 'Copy link', icon: 'link-45deg', disabled: !id, onClick: () => { navigator.clipboard.writeText(location.origin + urls.query(id)); toast('Link copied', 'success', { timeout: 1500 }); } },
      id && redashLink() && { label: 'Open in Redash', icon: 'box-arrow-up-right', href: redashLink(`/queries/${id}/source`), external: true },
      { divider: true },
      { label: 'Editor settings…', icon: 'gear', onClick: () => editorSettingsDialog(() => reloadEditor()) },
      { divider: true },
      server.is_archived
        ? { label: 'Unarchive', icon: 'arrow-counterclockwise', disabled: !id, onClick: unarchive }
        : { label: 'Archive', icon: 'archive', danger: true, disabled: !id || !canEdit, onClick: archive },
    ].filter(Boolean);
  }

  // ---------------------------------------------------------------- banners (draft restore, archived)

  function drawBanners() {
    const items = [];
    if (server.is_archived) {
      items.push(h('div.af-callout.af-callout-warn', icon('archive'), h('span', 'This query is archived — it is hidden from lists and search.'),
        h('button.btn.btn-sm.btn-warning.ms-auto', { onclick: unarchive }, icon('arrow-counterclockwise', 'me-1'), 'Unarchive')));
    }
    const draft = readDraft();
    if (draft && (draft.text !== (server.query || '') || draft.name !== (server.name || NEW_NAME)) && draft.text !== s.text) {
      items.push(h('div.af-callout.af-callout-info.rd-qe-draft', icon('life-preserver'),
        h('span', 'You have unsaved changes from ', h('b', timeAgo(new Date(draft.at).toISOString())), '.'),
        h('div.ms-auto.d-flex.gap-2',
          h('button.btn.btn-sm.btn-primary', { onclick: () => { applyDraft(draft); drawBanners(); } }, icon('arrow-counterclockwise', 'me-1'), 'Restore'),
          h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => { clearDraft(); drawBanners(); } }, 'Discard'))));
    }
    if (!canEdit) items.push(h('div.af-callout.af-callout-info', icon('eye'), h('span', 'You can view and run this query, but not change it. Fork it to make your own copy.')));
    mount(bannerEl, items);
  }

  const readDraft = () => { try { return JSON.parse(localStorage.getItem(draftKey(server.id)) || 'null'); } catch { return null; } };
  const writeDraft = debounce(() => {
    try {
      if (!isDirty()) { localStorage.removeItem(draftKey(server.id)); return; }
      localStorage.setItem(draftKey(server.id), JSON.stringify({ text: s.text, name: s.name, dsId: s.dsId, params: s.params, at: Date.now(), baseVersion: server.version }));
    } catch { /* storage unavailable */ }
  }, 600);
  const clearDraft = () => { try { localStorage.removeItem(draftKey(server.id)); } catch { /* ignore */ } };
  function applyDraft(d) {
    s.name = d.name || s.name;
    if (d.dsId) s.dsId = d.dsId;
    if (Array.isArray(d.params)) s.params = d.params;
    editor.replaceAll(d.text || '');
    s.text = d.text || '';
    drawHead(); drawParams(); drawToolbar();
    toast('Draft restored — press Save to keep it', 'success');
  }

  // ---------------------------------------------------------------- toolbar

  function drawToolbar() {
    const ds = dataSources.find((d) => d.id === s.dsId);
    const running = !!s.running;
    mount(toolbarEl,
      h('button.btn.btn-sm.btn-outline-secondary.rd-qe-schema-toggle', {
        type: 'button', title: schemaOpen ? 'Hide schema' : 'Show schema',
        onclick: () => { schemaOpen = !schemaOpen; prefs.set('rd:schemaOpen', schemaOpen); applySizes(); drawToolbar(); },
      }, icon(schemaOpen ? 'layout-sidebar-inset' : 'layout-sidebar')),
      h('select.form-select.form-select-sm.rd-qe-ds', {
        'aria-label': 'Data source', disabled: !canEdit,
        onchange: (e) => {
          s.dsId = Number(e.target.value);
          if (!server.id) prefs.set('rd:lastDs', s.dsId);
          schema.load(s.dsId);
          changed();
        },
      }, dataSources.map((d) => h('option', { value: d.id, selected: d.id === s.dsId }, `${d.name}${d.paused ? ' (paused)' : ''}${d.view_only ? ' · view only' : ''}`))),
      running
        ? h('button.btn.btn-sm.btn-danger.rd-qe-run', { type: 'button', onclick: cancelRun }, icon('stop-fill', 'me-1'), 'Cancel')
        : h('button.btn.btn-sm.btn-primary.rd-qe-run', { type: 'button', onclick: () => run(editor.getSelection() || null), title: 'Run (Ctrl+Enter) — runs only the selected text if there is a selection' },
          icon('play-fill', 'me-1'), 'Run', h('kbd.rd-kbd-inv', navigator.platform.includes('Mac') ? '⌘↵' : 'Ctrl↵')),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', disabled: running, title: 'Run only the selected text', onclick: () => { const sel = editor.getSelection(); if (!sel) toast('Select some SQL first', 'info', { timeout: 1800 }); else run(sel); } }, icon('cursor-text', 'me-1'), 'Run selection'),
      h('div.rd-qe-tool-sep'),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Format SQL (Ctrl+Shift+F)', onclick: formatSql, disabled: !canEdit }, icon('magic', 'me-1'), 'Format'),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Build a query without writing SQL', disabled: !canEdit, onclick: () => openBuilder({ schema: schema.getSchema(), onInsert: insertBuilt, onRun: () => run(null) }) }, icon('ui-checks-grid', 'me-1'), 'Builder'),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Insert a saved snippet', disabled: !canEdit, 'aria-haspopup': 'menu', onclick: (e) => { e.stopPropagation(); snippetMenu(e.currentTarget); } }, icon('scissors', 'me-1'), 'Snippets'),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Add a {{ parameter }}', disabled: !canEdit, onclick: addParam }, icon('braces', 'me-1'), 'Parameter'),
      h('div.rd-qe-tool-grow'),
      ds?.supports_auto_limit && h('label.form-check.form-switch.small.mb-0.rd-qe-autolimit', { title: 'Let Redash add LIMIT 1000 when the query has none' },
        h('input.form-check-input', { type: 'checkbox', checked: prefs.get('rd:autoLimit', true), onchange: (e) => prefs.set('rd:autoLimit', e.target.checked) }),
        h('span', 'Limit 1000')),
      h('button.af-icon-btn', { type: 'button', title: 'Editor settings', onclick: () => editorSettingsDialog(() => reloadEditor()) }, icon('gear')));
  }

  function snippetMenu(anchor) {
    if (!snippets.length) {
      openMenu(anchor, [{ header: 'No snippets yet' }, { label: 'Create snippets in Data → Query snippets', icon: 'scissors', href: urls.snippets() }]);
      return;
    }
    openMenu(anchor, [
      { header: 'Insert snippet (or type its trigger + Tab)' },
      ...snippets.map((sn) => ({ label: `${sn.trigger}${sn.description ? ` — ${sn.description}` : ''}`, icon: 'scissors', onClick: () => editor.insert(sn.snippet) })),
      { divider: true },
      { label: 'Manage snippets', icon: 'gear', href: urls.snippets() },
    ]);
  }

  function insertBuilt(sql, { replace }) {
    if (replace) editor.replaceAll(sql);
    else editor.insert(`${s.text.trim() ? '\n\n' : ''}${sql}`);
    s.text = editor.getValue();
    changed();
  }

  function addParam() {
    formModal({
      title: 'Add a parameter', icon: 'braces', size: 'sm',
      fields: [{ name: 'name', label: 'Keyword', required: true, placeholder: 'e.g. start_date', help: 'Inserted as {{ keyword }} at the cursor. Configure its type with the ⚙ next to it.' }],
      submitLabel: 'Insert',
      onSubmit: ({ name }, close) => {
        const key = name.trim().replace(/[^\w]+/g, '_');
        if (!key) throw new Error('Use letters, numbers or underscores.');
        editor.insert(`{{ ${key} }}`);
        close();
      },
    });
  }

  // ---------------------------------------------------------------- schema + editor

  const schema = schemaBrowser({
    onInsert: (text) => editor.insert(text),
    onSchema: (sch) => editor?.setSchema(sch),
    onQuickSelect: (t) => editor.insert(`SELECT *\nFROM ${t.name}\nLIMIT 100`),
    onBuilder: (t) => openBuilder({ schema: schema.getSchema(), table: t, onInsert: insertBuilt, onRun: () => run(null) }),
  });
  mount(schemaWrap, schema.el);

  let editor = null;
  async function reloadEditor() {
    const text = editor ? editor.getValue() : s.text;
    mount(editorHost);
    editor = await sqlEditor(editorHost, {
      value: text,
      readOnly: !canEdit,
      schema: schema.getSchema(),
      snippets,
      onChange: (v) => { s.text = v; changed(); },
      onRun: (sel) => run(sel),
      onSave: () => save(),
      onFormat: formatSql,
    });
    applySizes();
  }
  await reloadEditor();
  if (!ctx.active) return;

  const changed = debounce(() => {
    const next = syncParams(s.text, s.params);
    const namesChanged = next.map((p) => p.name).join('|') !== s.params.map((p) => p.name).join('|');
    s.params = next;
    s.params.forEach((p) => { if (!(p.name in s.values)) s.values[p.name] = p.value; });
    if (namesChanged) drawParams();
    drawHead();
    drawToolbar();
    writeDraft();
  }, 250);

  // ---------------------------------------------------------------- parameters

  let bar = null;
  function drawParams() {
    if (!s.params.length) { mount(paramsEl); bar = null; return; }
    bar = paramBar(s.params, s.values, { onApply: () => run(null), applyLabel: 'Apply & run', ownerQueryId: server.id });
    mount(paramsEl,
      h('div.rd-qe-params-head',
        h('span.rd-qe-params-title', icon('braces', 'me-1'), `Parameters (${s.params.length})`),
        s.params.map((p) => h('button.rd-qe-param-gear', {
          type: 'button', title: `Configure {{ ${p.name} }}`, disabled: !canEdit,
          onclick: () => configureParam(p),
        }, icon('gear'), p.title || p.name))),
      bar.el);
  }

  async function configureParam(p) {
    let queries = [];
    try { queries = (await rd.get('/api/queries', { page_size: 250, order: 'name' })).results.filter((q) => q.id !== server.id); } catch { /* optional */ }
    const form = paramDefinitionForm(p, { queries });
    modal({
      title: `Parameter {{ ${p.name} }}`, icon: 'braces', body: form.el,
      actions: [
        { label: 'Cancel' },
        {
          label: 'Apply', variant: 'primary',
          onClick: (close) => {
            const next = form.value();
            s.params = s.params.map((x) => (x.name === p.name ? next : x));
            if (next.type !== p.type) s.values[p.name] = next.value;
            close();
            drawParams();
            drawHead();
            writeDraft();
          },
        },
      ],
    });
  }

  // ---------------------------------------------------------------- run

  async function run(selection) {
    if (s.running) return;
    // Parameter detection is debounced while typing; sync now so Ctrl+Enter right after typing is accurate.
    s.text = editor.getValue();
    const synced = syncParams(s.text, s.params);
    if (synced.map((p) => p.name).join('|') !== s.params.map((p) => p.name).join('|')) {
      s.params = synced;
      s.params.forEach((p) => { if (!(p.name in s.values)) s.values[p.name] = p.value; });
      drawParams();
    }
    const missing = missingParams(s.params.filter((p) => !selection || selection.includes(p.name)), s.values);
    const textToRun = selection || s.text;
    if (!textToRun.trim()) { toast('Write some SQL first', 'info', { timeout: 1800 }); return; }
    if (missing.length) { toast(`Fill in: ${missing.join(', ')}`, 'warning', { title: 'Missing parameter values' }); return; }
    const ds = dataSources.find((d) => d.id === s.dsId);
    if (ds?.paused) { toast(`${ds.name} is paused${ds.pause_reason ? `: ${ds.pause_reason}` : ''}`, 'warning'); return; }
    s.controller = new AbortController();
    ctx.signal.addEventListener('abort', () => s.controller?.abort(), { once: true });
    s.running = { started: Date.now(), label: 'Queued', jobId: null };
    clearErrorMark?.();
    drawToolbar();
    drawResults();
    const tick = setInterval(() => drawRunning(), 250);
    const parameters = serializeParams(s.params, s.values);
    // Saved, unchanged text: execute the saved query (applies saved parameter schema safely).
    const useSaved = server.id && !selection && s.text === (server.query || '') && s.dsId === server.data_source_id;
    try {
      const result = await runQuery({
        ...(useSaved ? { queryId: server.id } : { query: textToRun, dataSourceId: s.dsId }),
        parameters, maxAge: 0, applyAutoLimit: !!ds?.supports_auto_limit && prefs.get('rd:autoLimit', true),
        signal: s.controller.signal,
        onJob: (job) => { s.running.jobId = job.id; },
        onProgress: ({ label }) => { if (s.running) s.running.label = label; },
      });
      s.result = result;
      s.error = null;
      bar?.reset();
      if (useSaved) server.latest_query_data_id = result.id;
      toast(`${result.data.rows.length.toLocaleString()} rows in ${fmtDuration(result.runtime)}`, 'success', { timeout: 2200 });
    } catch (err) {
      if (err.name === 'AbortError') { s.error = null; } else {
        s.error = new Error(errorText(err));
        const line = Number((s.error.message.match(/line (\d+)/i) || [])[1]);
        if (line && !selection) clearErrorMark = editor.markError(line);
      }
    } finally {
      clearInterval(tick);
      s.running = null;
      s.controller = null;
      if (ctx.active) { drawToolbar(); drawResults(); }
    }
  }
  let clearErrorMark = null;

  async function cancelRun() {
    const jobId = s.running?.jobId;
    s.controller?.abort();
    if (jobId) {
      try { await cancelJob(jobId); toast('Query cancelled', 'info', { timeout: 1800 }); } catch (err) { toastError(err, 'Could not cancel'); }
    }
  }

  // ---------------------------------------------------------------- results

  let runningEl = null;
  function drawRunning() {
    if (!runningEl || !s.running) return;
    const secs = (Date.now() - s.running.started) / 1000;
    runningEl.querySelector('.rd-running-time').textContent = fmtDuration(secs);
    runningEl.querySelector('.rd-running-label').textContent = s.running.label === 'running' ? 'Running on the data source…' : s.running.label === 'pending' ? 'Waiting for a worker…' : `${s.running.label}…`;
  }

  let grid = null;
  let vizHandle = null;
  async function drawResults() {
    vizHandle?.destroy?.();
    vizHandle = null;
    const vizs = [...(server.visualizations || [])].sort((a, b) => a.id - b.id);
    const status = [];
    if (s.result && !s.error) {
      status.push(h('span', icon('clock', 'me-1'), 'Ran ', timeAgo(s.result.retrieved_at)),
        h('span', icon('stopwatch', 'me-1'), fmtDuration(s.result.runtime)),
        h('span', icon('list-ol', 'me-1'), `${s.result.data.rows.length.toLocaleString()} rows`));
    }
    const head = h('div.rd-qe-res-head',
      h('div.rd-qe-res-status', status),
      s.result && !s.error && h('div.rd-qe-res-actions',
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', disabled: !!s.running, onclick: () => run(null), title: 'Run again' }, icon('arrow-clockwise', 'me-1'), 'Refresh'),
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', 'aria-haspopup': 'menu', onclick: (e) => { e.stopPropagation(); downloadMenu(e.currentTarget); } }, icon('download', 'me-1'), 'Download')));

    if (s.running) {
      runningEl = h('div.rd-running', spinner(), h('div', h('div.fw-semibold.rd-running-label', 'Queued…'), h('div.small', 'Elapsed ', h('span.rd-running-time', '0s'))),
        h('button.btn.btn-sm.btn-outline-danger', { onclick: cancelRun }, icon('stop-fill', 'me-1'), 'Cancel'));
      mount(resultsEl, h('div.rd-progress-bar'), head, h('div.af-card-body', runningEl));
      drawRunning();
      return;
    }
    if (s.error) {
      mount(resultsEl, head, h('div.af-card-body', h('div.af-error.rd-qe-error', icon('exclamation-octagon', 'af-error-icon'),
        h('div', h('div.af-error-title', 'The query failed'), h('pre.rd-qe-error-msg', s.error.message),
          h('div.af-error-hint', 'Fix the SQL and press Run again. Errors from the database are shown exactly as returned.')))));
      return;
    }
    if (!s.result && s.loadingCached) {
      mount(resultsEl, head, h('div.af-card-body', h('div.d-flex.align-items-center.gap-2.small.text-muted.mb-2', spinner('sm'), 'Loading the last results…'), skeleton(6)));
      return;
    }
    if (!s.result) {
      mount(resultsEl, head, h('div.af-card-body', emptyState({
        icon: 'play-circle', title: server.id ? 'No results yet' : 'Write a query and press Run',
        message: h('span', 'Press ', h('kbd', navigator.platform.includes('Mac') ? '⌘ Enter' : 'Ctrl Enter'), ' to run. Select part of the query to run only that part.'),
      })));
      return;
    }

    const tabItems = vizs.length
      ? vizs.map((v) => ({ id: v.id, label: v.name || vizTypeInfo(v.type).label, icon: vizTypeInfo(v.type).icon }))
      : [{ id: 'results', label: 'Results', icon: 'table' }];
    if (!tabItems.some((t) => t.id === s.activeViz)) s.activeViz = tabItems[0].id;
    const body = h('div.rd-qe-viz-body');
    const tabBar = tabs(tabItems, s.activeViz, (id) => { s.activeViz = id; ctx.setQuery({ viz: typeof id === 'number' ? id : null }); showViz(); });
    const vizTools = h('div.rd-qe-viz-tools');
    mount(resultsEl, head, h('div.rd-qe-tabs-row', tabBar,
      h('button.btn.btn-sm.btn-outline-primary.rd-qe-newviz', { type: 'button', onclick: () => newViz(), title: 'Create a chart, counter, pivot…' }, icon('plus-lg', 'me-1'), 'Add visualization')),
    vizTools, h('div.af-card-body.rd-qe-viz-card', body));

    async function showViz() {
      vizHandle?.destroy?.();
      const viz = vizs.find((v) => v.id === s.activeViz);
      if (!viz) {
        grid = resultsGrid(s.result.data, { settings });
        mount(body, grid.el);
        mount(vizTools, !server.id && h('div.small.text-muted', 'Save the query to add charts and dashboards.'));
        return;
      }
      mount(vizTools,
        h('span.rd-qe-viz-type', icon(vizTypeInfo(viz.type).icon, 'me-1'), vizTypeInfo(viz.type).label),
        viz.description && h('span.small.text-muted', viz.description),
        h('div.rd-qe-viz-actions',
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', disabled: !canEdit, onclick: () => editViz(viz) }, icon('pencil', 'me-1'), 'Edit'),
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => addVizToDashboard(viz) }, icon('grid-1x2', 'me-1'), 'Add to dashboard'),
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Dashboards that show this visualization', onclick: () => showUsage(viz) }, icon('diagram-3')),
          h('button.btn.btn-sm.btn-outline-danger', { type: 'button', disabled: !canEdit, title: 'Delete visualization', onclick: () => removeViz(viz) }, icon('trash'))));
      grid = null;
      mount(body, skeleton(4));
      const handle = await renderViz(body, viz, s.result, {
        height: 380, settings, gridMaxHeight: '520px',
        onPointClick: viz.type === 'CHART' ? (pt) => drillDown(viz, pt) : undefined,
      });
      vizHandle = handle;
      if (handle?.grid) grid = handle.grid;
    }
    showViz();
  }

  function drillDown(viz, pt) {
    const xCol = Object.entries(viz.options?.columnMapping || {}).find(([, role]) => role === 'x')?.[0];
    if (!xCol) return;
    const rows = s.result.data.rows.filter((r) => String(r[xCol]) === String(pt.x) || (r[xCol] && new Date(r[xCol]).getTime() === new Date(pt.x).getTime()));
    const g = resultsGrid({ columns: s.result.data.columns, rows }, { settings });
    drawer({ title: `${xCol} = ${pt.x}`, subtitle: `${rows.length} row${rows.length === 1 ? '' : 's'} behind this point${pt.series ? ` · ${pt.series}` : ''}`, width: 900, body: g.el });
  }

  function downloadMenu(anchor) {
    const r = s.result;
    const base = server.id && server.latest_query_data_id === r.id ? `/api/queries/${server.id}/results/${r.id}` : `/api/query_results/${r.id}`;
    const safe = (s.name || 'query').replace(/[^\w-]+/g, '_').slice(0, 60);
    const file = (ext) => ({ label: `${ext.toUpperCase()} file`, icon: { csv: 'filetype-csv', tsv: 'file-earmark-spreadsheet', json: 'filetype-json', xlsx: 'file-earmark-excel' }[ext], onClick: () => {
      const a = h('a', { href: fileUrl(`${base}.${ext}`), download: `${safe}.${ext}` });
      document.body.appendChild(a); a.click(); a.remove();
    } });
    openMenu(anchor, [
      { header: 'All rows' }, file('csv'), file('tsv'), file('xlsx'), file('json'),
      { divider: true },
      { header: 'What you see' },
      { label: grid ? 'Current view as CSV (filters & sort)' : 'Open the table tab to export the current view', icon: 'funnel', disabled: !grid, onClick: () => downloadCsv(grid.getView(), `${safe}-view.csv`) },
    ]);
  }

  async function ensureSaved() {
    if (server.id) return true;
    const ok = await confirmDialog({ title: 'Save the query first?', message: 'Visualizations belong to a saved query.', confirmLabel: 'Save now' });
    if (!ok) return false;
    return save();
  }

  async function newViz() {
    if (!(await ensureSaved())) return;
    const open = await optional('../core/viz_editor.js', 'openVizEditor');
    if (!open) return;
    const suggestion = s.result ? suggestViz(s.result.data) : null;
    const viz = await open({
      query: server, result: s.result, viz: null, suggestion,
      onSaved: (v) => upsertViz(v),
    });
    if (viz) upsertViz(viz);
  }
  async function editViz(viz) {
    const open = await optional('../core/viz_editor.js', 'openVizEditor');
    if (!open) return;
    const saved = await open({ query: server, result: s.result, viz, onSaved: (v) => upsertViz(v), onDeleted: (id) => dropViz(id) });
    if (saved) upsertViz(saved);
  }
  async function removeViz(viz) {
    const del = await optional('../core/viz_editor.js', 'deleteViz');
    if (del && await del(viz)) dropViz(viz.id);
  }
  async function addVizToDashboard(viz) {
    const add = await optional('../core/add_to_dashboard.js', 'addToDashboard');
    if (add) await add({ viz, query: server });
  }
  function upsertViz(v) {
    const list = server.visualizations || [];
    const i = list.findIndex((x) => x.id === v.id);
    if (i >= 0) list[i] = v; else list.push(v);
    server.visualizations = list;
    s.activeViz = v.id;
    drawResults();
  }
  function dropViz(id) {
    server.visualizations = (server.visualizations || []).filter((v) => v.id !== id);
    if (s.activeViz === id) s.activeViz = null;
    drawResults();
  }

  async function showUsage(viz) {
    const body = h('div', spinner('sm'), ' Looking through dashboards…');
    modal({ title: viz?.id ? `Dashboards showing “${viz.name}”` : 'Dashboards using this query', icon: 'grid-1x2', body, actions: [{ label: 'Close' }] });
    try {
      const dashes = await dashboardsUsing(viz?.id ? { vizId: viz.id } : { queryId: server.id });
      mount(body, dashes.length
        ? h('div.af-list', dashes.map((d) => h('a.af-list-item', { href: urls.dashboard(d.id) }, icon('grid-1x2'),
          h('div.af-list-main', h('div.af-list-title', d.name), h('div.af-list-sub', `${(d.widgets || []).length} widgets${d.is_draft ? ' · draft' : ''}`)), icon('chevron-right'))))
        : emptyState({ icon: 'grid-1x2', title: 'Not on any dashboard yet', message: 'Use “Add to dashboard” on a visualization tab.' }));
    } catch (err) { mount(body, errorState(err)); }
  }

  // ---------------------------------------------------------------- save & metadata

  /** POST a partial update for an already-saved query (metadata changes). */
  async function patch(fields, message) {
    try {
      const res = await rd.post(`/api/queries/${enc(server.id)}`, { ...fields, version: server.version });
      Object.assign(server, res);
      if (message) toast(message, 'success', { timeout: 1800 });
      drawHead(); drawBanners();
      return res;
    } catch (err) {
      if (err.status === 409) { await handleConflict(fields); return null; }
      throw err;
    }
  }
  async function save({ overwrite = false } = {}) {
    if (!canEdit) return false;
    if (!s.dsId) { toast('Choose a data source first', 'warning'); return false; }
    // Remember current parameter values as defaults, like Redash does.
    const params = s.params.map((p) => ({ ...p, value: s.values[p.name] ?? p.value }));
    const body = {
      name: s.name || NEW_NAME,
      query: s.text,
      data_source_id: s.dsId,
      options: { ...(server.options || {}), parameters: params },
    };
    const saveBtn = headEl.querySelector('.rd-qe-save');
    if (saveBtn) { saveBtn.disabled = true; mount(saveBtn, spinner('sm'), ' Saving…'); }
    try {
      if (!server.id) {
        let created = await rd.post('/api/queries', { ...body, schedule: null });
        if (body.name !== NEW_NAME) created = await rd.post(`/api/queries/${enc(created.id)}`, { is_draft: false, version: created.version });
        clearDraft();
        prefs.set('rd:lastDs', s.dsId);
        toast(`Saved as query #${created.id}`, 'success');
        server = created;
        rememberRecent(created);
        navigate(urls.query(created.id), { replace: true });
        return true;
      }
      if (!overwrite) {
        body.version = server.version;
        // Redash 10 never bumps `version`, so its 409 check can't fire. Detect concurrent
        // edits ourselves: someone changed the SQL or name since we loaded it.
        const latest = await rd.get(`/api/queries/${enc(server.id)}`);
        if (latest.updated_at !== server.updated_at && (latest.query !== server.query || latest.name !== server.name)) {
          return await handleConflict(null, latest);
        }
      }
      if (server.is_draft && body.name !== NEW_NAME) body.is_draft = false;
      const res = await rd.post(`/api/queries/${enc(server.id)}`, body);
      server = { ...server, ...res };
      s.params = structuredClone(res.options?.parameters || params);
      clearDraft();
      rememberRecent(server);
      toast('Saved', 'success', { timeout: 1500 });
      drawHead(); drawBanners(); drawRecent(); drawToolbar();
      return true;
    } catch (err) {
      if (err.status === 409) return await handleConflict(null);
      toastError(err, 'Save failed');
      return false;
    } finally {
      drawHead();
    }
  }

  async function handleConflict(pendingFields, known = null) {
    let theirs = known;
    if (!theirs) { try { theirs = await rd.get(`/api/queries/${enc(server.id)}`); } catch { /* still offer choices */ } }
    const choice = await conflictDialog(theirs);
    if (choice === 'reload' && theirs) {
      writeDraftNow();
      server = theirs;
      s.name = theirs.name;
      s.dsId = theirs.data_source_id;
      s.params = structuredClone(theirs.options?.parameters || []);
      editor.replaceAll(theirs.query || '');
      s.text = theirs.query || '';
      drawHead(); drawBanners(); drawParams(); drawToolbar(); drawResults();
      toast('Loaded the latest version. Your edits are kept as a draft above.', 'info');
      return false;
    }
    if (choice === 'overwrite') {
      if (theirs) Object.assign(server, { version: theirs.version, updated_at: theirs.updated_at, query: theirs.query, name: theirs.name });
      if (pendingFields) return patch(pendingFields);
      return save({ overwrite: true });
    }
    return false;
  }
  function writeDraftNow() {
    try { localStorage.setItem(draftKey(server.id), JSON.stringify({ text: s.text, name: s.name, dsId: s.dsId, params: s.params, at: Date.now() })); } catch { /* ignore */ }
  }

  async function formatSql() {
    const sel = editor.getSelection();
    try {
      const res = await rd.post('/api/queries/format', { query: sel || s.text });
      if (sel) editor.insert(res.query);
      else editor.replaceAll(res.query);
      s.text = editor.getValue();
      changed();
      toast('Formatted — Ctrl+Z to undo', 'success', { timeout: 1500 });
    } catch (err) { toastError(err, 'Format failed'); }
  }

  function restoreRevision(rev) {
    editor.replaceAll(rev.query || '');
    s.text = rev.query || '';
    if (rev.name) s.name = rev.name;
    if (rev.data_source_id) s.dsId = rev.data_source_id;
    changed();
    drawHead();
    toast('Version restored into the editor — press Save to keep it', 'success');
  }

  async function editSchedule() {
    const next = await scheduleDialog(server.schedule);
    if (next === undefined) return;
    await patch({ schedule: next }, next ? `Refreshes ${scheduleText(next).toLowerCase()}` : 'Schedule removed');
  }

  async function fork() {
    if (isDirty()) {
      const ok = await confirmDialog({ title: 'Fork without your unsaved changes?', message: 'A fork copies the saved version. Save first if you want your edits in the copy.', confirmLabel: 'Fork saved version' });
      if (!ok) return;
    }
    try {
      const copy = await rd.post(`/api/queries/${enc(server.id)}/fork`, {});
      toast(`Forked into #${copy.id}`, 'success');
      navigate(urls.query(copy.id));
    } catch (err) { toastError(err, 'Fork failed'); }
  }

  async function archive() {
    const ok = await confirmDialog({
      title: 'Archive this query?', danger: true, confirmLabel: 'Archive',
      message: 'It disappears from lists and search, and its widgets are removed from dashboards. You can unarchive it later from Queries → Archived.',
    });
    if (!ok) return;
    try {
      await rd.del(`/api/queries/${enc(server.id)}`);
      server = await rd.get(`/api/queries/${enc(server.id)}`);
      forgetRecent(server.id);
      toast('Query archived', 'success');
      drawHead(); drawBanners(); drawRecent();
    } catch (err) { toastError(err, 'Archive failed'); }
  }

  async function unarchive() {
    try {
      await patch({ is_archived: false }, 'Query restored');
      rememberRecent(server);
      drawRecent();
    } catch (err) { toastError(err, 'Unarchive failed'); }
  }

  // ---------------------------------------------------------------- page-level keys & guards

  const onKey = (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === 's' && !e.target.closest?.('.CodeMirror')) { e.preventDefault(); save(); }
    if (mod && e.key === 'Enter' && !e.target.closest?.('.CodeMirror, textarea')) { e.preventDefault(); run(null); }
  };
  const onBeforeUnload = (e) => { if (isDirty()) { e.preventDefault(); e.returnValue = ''; } };
  document.addEventListener('keydown', onKey);
  window.addEventListener('beforeunload', onBeforeUnload);
  ctx.cleanup(() => {
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('beforeunload', onBeforeUnload);
    s.controller?.abort();
    vizHandle?.destroy?.();
    if (isDirty()) {
      writeDraftNow();
      toast('Your unsaved changes were kept as a draft — reopen the query to restore them.', 'info', { title: 'Draft saved', timeout: 5000 });
    }
  });

  // ---------------------------------------------------------------- initial render

  drawRecent();
  drawHead();
  drawBanners();
  drawToolbar();
  drawParams();
  applySizes();
  schema.load(s.dsId);
  s.loadingCached = !!server.latest_query_data_id;
  drawResults();

  // Show the cached latest result without executing anything.
  if (server.latest_query_data_id) {
    try {
      const res = await rd.get(`/api/query_results/${enc(server.latest_query_data_id)}`, null, { signal });
      if (ctx.active && !s.result && !s.running) s.result = res.query_result;
    } catch { /* stale result id: user can just run */ }
    s.loadingCached = false;
    if (ctx.active && !s.running) drawResults();
  }
  if (!server.id) editor.focus();
}

/** Pointer-drag resizing helper. axis 'x' | 'y'. */
function dragResize(handle, axis, onMove, getStart, onEnd) {
  handle.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    const origin = axis === 'x' ? e.clientX : e.clientY;
    const start = getStart();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add('dragging');
    document.body.classList.add(axis === 'x' ? 'rd-resizing-x' : 'rd-resizing-y');
    const move = (ev) => onMove((axis === 'x' ? ev.clientX : ev.clientY) - origin, start);
    const up = () => {
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      handle.classList.remove('dragging');
      document.body.classList.remove('rd-resizing-x', 'rd-resizing-y');
      onEnd();
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
  });
  handle.addEventListener('dblclick', () => onEnd());
}
