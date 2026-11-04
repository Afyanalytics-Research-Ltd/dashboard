// Workflows: faceted list with deep content search, saved views, last-run health,
// bulk actions, card/table views, and an import wizard (JSON / file / .n8np package)
// that remaps credentials by type and name.

import { h, icon, mount, prefs } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toast, toastError, modal, menuButton, chips, searchInput, selectionBar, jsonEditor, spinner, timeAgo, confirmDialog } from '../../../ui/js/ui.js';
import { n8n, enc, download, saveText, upload, studio } from '../core/api.js';
import { urls } from '../core/urls.js';
import * as W from '../core/workflow.js';
import { load, invalidate, loadOptional } from '../core/store.js';
import { loadCatalog, catalog, nodeIcon, displayName, credentialDisplayName } from '../core/catalog.js';
import { activeSwitch, tagPills, bulkRun, execDot, emptyHint, pickFile, readFileText, promptText, licenseGate } from '../core/widgets.js';
import { FAILED } from '../core/execution.js';

const VIEW_KEY = 'n8s:wf-view';
const SAVED_KEY = 'n8s:wf-saved-views';
const TRIGGER_KINDS = { webhook: 'broadcast', schedule: 'clock', manual: 'hand-index', form: 'ui-checks', chat: 'chat-dots', app: 'app-indicator', error: 'bug', 'sub-workflow': 'box-arrow-in-right' };

export default async function render(ctx) {
  await loadCatalog().catch(() => null);
  const saved = prefs.get(VIEW_KEY, {});
  const f = {
    q: ctx.query.get('q') ?? saved.q ?? '',
    status: ctx.query.get('status') ?? saved.status ?? 'live',
    triggers: ctx.query.getAll('trigger').length ? ctx.query.getAll('trigger') : saved.triggers || [],
    tags: ctx.query.getAll('tag').length ? ctx.query.getAll('tag') : saved.tags || [],
    nodeType: ctx.query.get('type') ?? saved.nodeType ?? '',
    cred: ctx.query.get('cred') ?? saved.cred ?? '',
    project: ctx.query.get('project') ?? saved.project ?? '',
    health: ctx.query.get('health') ?? saved.health ?? '',
    sort: saved.sort || { key: 'updatedAt', dir: 'desc' },
    layout: saved.layout || 'table',
  };
  const selected = new Set();

  const listEl = h('div');
  const facetsEl = h('div.n8s-facets');
  const barEl = h('div');
  const countEl = h('span.n8s-count');
  mount(ctx.el,
    pageHeader({
      title: 'Workflows', icon: 'diagram-3',
      subtitle: 'Search reaches inside nodes, parameters, notes and credentials — try type:slack, cred:prod, tag:billing, is:active, -type:code',
      actions: [
        h('a.btn.btn-sm.btn-primary', { href: urls.newWorkflow() }, icon('plus-lg', 'me-1'), 'New'),
        h('button.btn.btn-sm.btn-outline-primary', { onclick: () => openImport() }, icon('upload', 'me-1'), 'Import'),
        menuButton('', [
          { label: 'Export all as JSON', icon: 'filetype-json', onClick: () => exportJson(null) },
          { label: 'Back up all now (Studio snapshots)', icon: 'cloud-arrow-down', onClick: () => ctx.navigate(urls.backups({ run: 1 })) },
          { label: 'Refresh', icon: 'arrow-clockwise', onClick: () => { invalidate('workflows'); refresh(); } },
        ]),
      ],
    }),
    h('div.n8s-wf-toolbar',
      searchInput({ placeholder: 'Search workflows and their contents…', value: f.q, onInput: (v) => { f.q = v; persist(); draw(); }, autofocus: !f.q }),
      h('div.n8s-status-chips'),
      h('div.flex-grow-1'),
      countEl,
      h('div.btn-group.btn-group-sm',
        h('button.btn', { class: f.layout === 'table' ? 'btn-secondary' : 'btn-outline-secondary', title: 'Table', onclick: (e) => { f.layout = 'table'; persist(); syncLayoutBtns(e.currentTarget.parentElement); draw(); } }, icon('list-ul')),
        h('button.btn', { class: f.layout === 'cards' ? 'btn-secondary' : 'btn-outline-secondary', title: 'Cards', onclick: (e) => { f.layout = 'cards'; persist(); syncLayoutBtns(e.currentTarget.parentElement); draw(); } }, icon('grid-3x2-gap')))),
    h('div.n8s-wf-layout', facetsEl, h('div.n8s-wf-main', barEl, listEl)));

  const syncLayoutBtns = (group) => [...group.children].forEach((b, i) => { b.className = `btn ${(i === 0 ? 'table' : 'cards') === f.layout ? 'btn-secondary' : 'btn-outline-secondary'}`; });
  const persist = () => {
    prefs.set(VIEW_KEY, { ...f });
    ctx.setQuery({ q: f.q || null, status: f.status === 'live' ? null : f.status, trigger: f.triggers, tag: f.tags, type: f.nodeType || null, cred: f.cred || null, project: f.project || null, health: f.health || null });
  };

  let all = [];
  let lastRun = new Map();
  let lintCache = new Map();
  let credentials = [];
  let projects = [];

  async function refresh() {
    mount(listEl, spinner());
    try {
      [all, credentials, projects] = await Promise.all([load('workflows', { signal: ctx.signal }), loadOptional('credentials'), loadOptional('projects')]);
    } catch (err) { mount(listEl, h('div.text-danger', err.message)); return; }
    lintCache = new Map();
    draw();
    // last-run health from the most recent executions across all workflows
    n8n.get('/executions', { limit: 250 }, { signal: ctx.signal }).then((page) => {
      lastRun = new Map();
      const since = Date.now() - 24 * 3600e3;
      for (const e of page.data || []) {
        const s = lastRun.get(e.workflowId) || { last: null, fails24: 0, runs24: 0 };
        if (!s.last) s.last = e;
        if (new Date(e.startedAt || e.createdAt).getTime() > since) { s.runs24++; if (FAILED.has(e.status)) s.fails24++; }
        lastRun.set(e.workflowId, s);
      }
      draw();
    }).catch(() => null);
  }

  const lintOf = (w) => {
    if (!lintCache.has(w.id)) {
      lintCache.set(w.id, W.lint(w, { catalog: catalog.nodes, credentials: new Map(credentials.map((c) => [String(c.id), c])), workflowsById: new Map(all.map((x) => [String(x.id), x])) }));
    }
    return lintCache.get(w.id);
  };

  function filtered() {
    const terms = W.parseQuery(f.q);
    return all.filter((w) => {
      if (f.status === 'live' && w.isArchived) return false;
      if (f.status === 'active' && (!w.active || w.isArchived)) return false;
      if (f.status === 'inactive' && (w.active || w.isArchived)) return false;
      if (f.status === 'archived' && !w.isArchived) return false;
      if (f.triggers.length && !W.triggerKinds(w, catalog.nodes).some((k) => f.triggers.includes(k))) return false;
      if (f.tags.length && !f.tags.every((t) => (w.tags || []).some((x) => x.name === t))) return false;
      if (f.nodeType && !(w.nodes || []).some((n) => n.type === f.nodeType)) return false;
      if (f.cred && !W.credentialRefs(w).some((c) => c.id === f.cred)) return false;
      if (f.project && !(w.shared || []).some((s) => s.projectId === f.project)) return false;
      if (f.health === 'failing' && !(lastRun.get(w.id)?.fails24)) return false;
      if (f.health === 'issues' && !lintOf(w).some((i) => i.level === 'error')) return false;
      if (f.health === 'never' && lastRun.has(w.id)) return false;
      return W.matchWorkflow(w, terms);
    });
  }

  function sorted(rows) {
    const { key, dir } = f.sort;
    const val = (w) => (key === 'name' ? (w.name || '').toLowerCase() : key === 'nodes' ? (w.nodes || []).length : key === 'lastRun' ? (lastRun.get(w.id)?.last?.startedAt || '') : w[key] || '');
    return [...rows].sort((a, b) => (val(a) < val(b) ? -1 : val(a) > val(b) ? 1 : 0) * (dir === 'asc' ? 1 : -1));
  }

  function facet(title, items, active, onPick, { multi = false } = {}) {
    if (!items.length) return null;
    return h('div.n8s-facet', h('div.n8s-facet-title', title), items.map((it) => {
      const on = multi ? active.includes(it.value) : active === it.value;
      return h('button.n8s-facet-item', { type: 'button', class: on && 'active', title: it.title || it.label, onclick: () => onPick(it.value, on) },
        it.icon || null, h('span.n8s-facet-label', it.label), h('span.n8s-facet-count', String(it.count)));
    }));
  }

  function drawFacets(base) {
    const count = (fn) => { const m = new Map(); for (const w of base) for (const k of fn(w)) m.set(k, (m.get(k) || 0) + 1); return m; };
    const trig = count((w) => W.triggerKinds(w, catalog.nodes));
    const tags = count((w) => (w.tags || []).map((t) => t.name));
    const types = count((w) => [...new Set((w.nodes || []).filter((n) => !W.isSticky(n)).map((n) => n.type))]);
    const creds = count((w) => [...new Set(W.credentialRefs(w).map((c) => c.id).filter(Boolean))]);
    const projs = count((w) => (w.shared || []).map((s) => s.projectId).filter(Boolean));
    const savedViews = prefs.get(SAVED_KEY, []);
    const sortDesc = (m) => [...m.entries()].sort((a, b) => b[1] - a[1]);
    mount(facetsEl,
      h('div.n8s-facet', h('div.n8s-facet-title', 'Saved views', h('button.af-link-btn.ms-auto', { title: 'Save current filters as a view', onclick: saveView }, icon('plus'))),
        savedViews.length ? savedViews.map((v, i) => h('div.n8s-facet-row',
          h('button.n8s-facet-item', { type: 'button', onclick: () => { Object.assign(f, v.filters); persist(); render(ctx); } }, icon('bookmark'), h('span.n8s-facet-label', v.name)),
          h('button.af-icon-btn', { title: 'Delete view', onclick: () => { savedViews.splice(i, 1); prefs.set(SAVED_KEY, savedViews); drawFacets(base); } }, icon('x')))) : h('div.n8s-facet-empty', 'Save filter combinations as smart folders.')),
      facet('Health', [
        { value: 'failing', label: 'Failed in last 24h', icon: icon('x-octagon', 'text-danger'), count: base.filter((w) => lastRun.get(w.id)?.fails24).length },
        { value: 'issues', label: 'Has problems', icon: icon('exclamation-triangle', 'text-warning'), count: base.filter((w) => lintOf(w).some((i) => i.level === 'error')).length },
        { value: 'never', label: 'No recent runs', icon: icon('moon', 'text-muted'), count: base.filter((w) => !lastRun.has(w.id)).length },
      ].filter((x) => x.count), f.health, (v, on) => { f.health = on ? '' : v; persist(); draw(); }),
      facet('Trigger', sortDesc(trig).map(([k, n]) => ({ value: k, label: k, icon: icon(TRIGGER_KINDS[k] || 'lightning'), count: n })), f.triggers, (v, on) => { f.triggers = on ? f.triggers.filter((x) => x !== v) : [...f.triggers, v]; persist(); draw(); }, { multi: true }),
      facet('Tags', sortDesc(tags).slice(0, 30).map(([k, n]) => ({ value: k, label: k, icon: h('span.n8s-tag-dot', { style: { '--tag-hue': hueOf(k) } }), count: n })), f.tags, (v, on) => { f.tags = on ? f.tags.filter((x) => x !== v) : [...f.tags, v]; persist(); draw(); }, { multi: true }),
      projects.length > 1 && facet('Project', sortDesc(projs).map(([k, n]) => ({ value: k, label: projects.find((p) => p.id === k)?.name || k, icon: icon('kanban'), count: n })), f.project, (v, on) => { f.project = on ? '' : v; persist(); draw(); }),
      facet('Uses node', sortDesc(types).slice(0, 25).map(([k, n]) => ({ value: k, label: displayName(k), icon: nodeIcon(k, { size: 16 }), count: n })), f.nodeType, (v, on) => { f.nodeType = on ? '' : v; persist(); draw(); }),
      facet('Uses credential', sortDesc(creds).slice(0, 20).map(([k, n]) => ({ value: k, label: credentials.find((c) => String(c.id) === k)?.name || `missing ${k}`, icon: icon('key'), count: n })), f.cred, (v, on) => { f.cred = on ? '' : v; persist(); draw(); }));
  }

  async function saveView() {
    const name = await promptText({ title: 'Save view', label: 'Name', placeholder: 'e.g. Failing webhooks' });
    if (!name) return;
    const views = prefs.get(SAVED_KEY, []);
    views.push({ name, filters: { q: f.q, status: f.status, triggers: f.triggers, tags: f.tags, nodeType: f.nodeType, cred: f.cred, project: f.project, health: f.health } });
    prefs.set(SAVED_KEY, views);
    draw();
  }

  function draw() {
    const statusBase = all.filter((w) => W.matchWorkflow(w, W.parseQuery(f.q)));
    mount(ctx.el.querySelector('.n8s-status-chips'), chips([
      { value: 'live', label: 'All', count: statusBase.filter((w) => !w.isArchived).length },
      { value: 'active', label: 'Active', color: '#16a34a', count: statusBase.filter((w) => w.active && !w.isArchived).length },
      { value: 'inactive', label: 'Inactive', color: '#94a3b8', count: statusBase.filter((w) => !w.active && !w.isArchived).length },
      { value: 'archived', label: 'Archived', color: '#64748b', count: statusBase.filter((w) => w.isArchived).length },
    ], f.status, (v) => { f.status = v || 'live'; persist(); draw(); }));
    const rows = sorted(filtered());
    drawFacets(all.filter((w) => (f.status === 'archived' ? w.isArchived : !w.isArchived)));
    mount(countEl, `${rows.length} of ${all.length}`);
    drawBar();
    if (!all.length) {
      mount(listEl, emptyHint('diagram-3', 'No workflows yet', 'Create one, import JSON from anywhere, or restore a package.', h('div.d-flex.gap-2.justify-content-center', h('a.btn.btn-primary.btn-sm', { href: urls.newWorkflow() }, 'New workflow'), h('button.btn.btn-outline-primary.btn-sm', { onclick: () => openImport() }, 'Import'))));
      return;
    }
    mount(listEl, f.layout === 'cards' ? cards(rows) : table(rows));
  }

  function nodeStrip(w) {
    const types = [...new Set((w.nodes || []).filter((n) => !W.isSticky(n)).map((n) => n.type))];
    return h('span.n8s-strip', types.slice(0, 6).map((t) => nodeIcon(t, { size: 20 })), types.length > 6 && h('span.n8s-strip-more', `+${types.length - 6}`));
  }

  function health(w) {
    const r = lastRun.get(w.id);
    const errs = lintOf(w).filter((i) => i.level === 'error');
    return h('span.n8s-health',
      r?.last ? h('a', { href: urls.execution(r.last.id), title: `Last run #${r.last.id}: ${r.last.status}` }, execDot(r.last.status), ' ', timeAgo(r.last.startedAt)) : h('span.text-muted.small', '—'),
      r?.fails24 ? h('span.n8s-fail-pill', { title: `${r.fails24} of ${r.runs24} runs failed in 24h` }, `${r.fails24}✕`) : null,
      errs.length ? h('span.n8s-issue-pill', { title: errs.map((i) => `${i.node ? `${i.node}: ` : ''}${i.message}`).join('\n') }, icon('exclamation-triangle-fill'), errs.length) : null);
  }

  function table(rows) {
    return dataTable({
      rows, selectable: true, selected,
      onSelect: () => drawBar(),
      sort: f.sort, onSort: (key, dir) => { f.sort = { key, dir }; persist(); draw(); },
      empty: emptyHint('search', 'No workflows match', 'Clear some filters or broaden the search.', h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => { Object.assign(f, { q: '', triggers: [], tags: [], nodeType: '', cred: '', project: '', health: '', status: 'live' }); persist(); render(ctx); } }, 'Reset filters')),
      columns: [
        {
          key: 'name', label: 'Name', sortable: true, render: (w) => h('div.n8s-wf-cell',
            h('a.n8s-wf-name', { href: urls.workflow(w.id) }, w.name), tagPills(w.tags, { onClick: (t) => { if (!f.tags.includes(t.name)) f.tags.push(t.name); persist(); draw(); } }),
            w.description && h('div.n8s-wf-desc', w.description)),
        },
        { key: 'trig', label: 'Triggers', render: (w) => h('span.n8s-trig-icons', W.triggerKinds(w, catalog.nodes).map((k) => h('span', { title: k }, icon(TRIGGER_KINDS[k] || 'lightning')))) },
        { key: 'nodes', label: 'Nodes', sortable: true, render: (w) => h('span.d-flex.align-items-center.gap-2', nodeStrip(w), h('span.small.text-muted', String((w.nodes || []).filter((n) => !W.isSticky(n)).length))) },
        { key: 'lastRun', label: 'Last run', sortable: true, render: health },
        { key: 'updatedAt', label: 'Updated', sortable: true, render: (w) => timeAgo(w.updatedAt) },
        { key: 'active', label: 'Active', align: 'center', render: (w) => activeSwitch(w) },
        { key: 'menu', label: '', align: 'right', render: (w) => menuButton('', () => rowMenu(w)) },
      ],
      onRowClick: (w, e) => (e.metaKey || e.ctrlKey ? window.open(urls.workflow(w.id), '_blank') : ctx.navigate(urls.workflow(w.id))),
    });
  }

  function cards(rows) {
    if (!rows.length) return emptyHint('search', 'No workflows match', 'Clear some filters.');
    return h('div.n8s-wf-cards', rows.map((w) => h('a.n8s-wf-card', { href: urls.workflow(w.id), class: w.active && 'active' },
      h('div.n8s-wf-card-top', nodeStrip(w), h('div.flex-grow-1'), activeSwitch(w)),
      h('div.n8s-wf-card-name', w.name),
      w.description && h('div.n8s-wf-desc', w.description),
      tagPills(w.tags),
      h('div.n8s-wf-card-foot', health(w), h('span.small.text-muted', 'updated ', timeAgo(w.updatedAt))))));
  }

  function rowMenu(w) {
    return [
      { label: 'Open', icon: 'box-arrow-up-right', href: urls.workflow(w.id) },
      { label: 'Executions', icon: 'lightning-charge', href: urls.executions({ workflowId: w.id }) },
      { label: 'Versions', icon: 'clock-history', href: urls.workflow(w.id, { tab: 'versions' }) },
      { label: 'Duplicate', icon: 'copy', onClick: () => duplicate(w) },
      { label: 'Download JSON', icon: 'download', onClick: () => saveText(JSON.stringify(W.saveBody(w), null, 2), `${w.name}.json`) },
      { divider: true },
      w.isArchived ? { label: 'Unarchive', icon: 'archive', onClick: () => bulk('unarchive', [w]) } : { label: 'Archive', icon: 'archive', onClick: () => bulk('archive', [w]) },
      { label: 'Delete', icon: 'trash', danger: true, onClick: () => bulk('delete', [w]) },
    ];
  }

  async function duplicate(w) {
    const name = await promptText({ title: 'Duplicate', label: 'Name', value: `${w.name} (copy)` });
    if (!name) return;
    try {
      const c = await n8n.post('/workflows', W.importBody(w, { name }));
      invalidate('workflows');
      toast('Duplicated', 'success', { action: { label: 'Open', onClick: () => ctx.navigate(urls.workflow(c.id)) } });
      refresh();
    } catch (err) { toastError(err, 'Duplicate failed'); }
  }

  // ---------------------------------------------------------------- bulk

  function drawBar() {
    const items = all.filter((w) => selected.has(w.id));
    mount(barEl, selectionBar(items.length, [
      h('button.btn.btn-sm.btn-success', { onclick: () => bulk('activate', items) }, icon('play-fill', 'me-1'), 'Activate'),
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => bulk('deactivate', items) }, icon('pause-fill', 'me-1'), 'Deactivate'),
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => bulkTag(items) }, icon('tags', 'me-1'), 'Tag'),
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => exportJson(items) }, icon('filetype-json', 'me-1'), 'JSON'),
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => exportPackage(items) }, icon('box-seam', 'me-1'), '.n8np'),
      menuButton('More', [
        { label: 'Archive', icon: 'archive', onClick: () => bulk('archive', items) },
        { label: 'Unarchive', icon: 'archive', onClick: () => bulk('unarchive', items) },
        { label: 'Transfer to project…', icon: 'arrow-left-right', onClick: () => bulkTransfer(items) },
        { label: 'Snapshot now', icon: 'camera', onClick: () => studio.post('/snapshots/capture/', {}).then((r) => toast(`${r.captured} snapshot(s) taken`, 'success')).catch(toastError) },
        { divider: true },
        { label: 'Delete…', icon: 'trash', danger: true, onClick: () => bulk('delete', items) },
      ], { icon: 'three-dots' }),
    ], () => { selected.clear(); draw(); }));
  }

  async function bulk(action, items) {
    if (!items.length) return;
    if (action === 'delete') {
      const ok = await confirmDialog({ title: `Delete ${items.length} workflow${items.length > 1 ? 's' : ''}?`, danger: true, confirmLabel: 'Delete', message: 'Studio snapshots each one first — restore from Backups → Recycle bin.', details: h('ul.small', items.slice(0, 10).map((w) => h('li', w.name)), items.length > 10 && h('li', `…and ${items.length - 10} more`)) });
      if (!ok) return;
    }
    const run = {
      activate: (w) => n8n.post(`/workflows/${enc(w.id)}/activate`, {}),
      deactivate: (w) => n8n.post(`/workflows/${enc(w.id)}/deactivate`, {}),
      archive: (w) => n8n.post(`/workflows/${enc(w.id)}/archive`, {}),
      unarchive: (w) => n8n.post(`/workflows/${enc(w.id)}/unarchive`, {}),
      delete: (w) => n8n.del(`/workflows/${enc(w.id)}`),
    }[action];
    await bulkRun({ title: `${action[0].toUpperCase()}${action.slice(1)} ${items.length} workflow(s)`, items, label: (w) => w.name, run });
    selected.clear();
    invalidate('workflows');
    refresh();
  }

  async function bulkTag(items) {
    const tags = await load('tags', { force: true });
    const sel = h('select.form-select.form-select-sm', h('option', { value: '' }, '— choose tag —'), tags.map((t) => h('option', { value: t.id }, t.name)));
    const fresh = h('input.form-control.form-control-sm.mt-2', { placeholder: '…or create a new tag' });
    const mode = h('select.form-select.form-select-sm.mt-2', h('option', { value: 'add' }, 'Add tag'), h('option', { value: 'remove' }, 'Remove tag'));
    modal({
      title: `Tag ${items.length} workflow(s)`, icon: 'tags', size: 'sm', body: h('div', sel, fresh, mode),
      actions: [{ label: 'Cancel' }, {
        label: 'Apply', variant: 'primary',
        onClick: async (close) => {
          let tagId = sel.value;
          if (fresh.value.trim()) tagId = (await n8n.post('/tags', { name: fresh.value.trim() })).id;
          if (!tagId) throw new Error('Choose or create a tag.');
          close();
          await bulkRun({
            title: `${mode.value === 'add' ? 'Tagging' : 'Untagging'} workflows`, items, label: (w) => w.name,
            run: async (w) => {
              const current = (await n8n.get(`/workflows/${enc(w.id)}/tags`)).map((t) => String(t.id));
              const next = mode.value === 'add' ? [...new Set([...current, String(tagId)])] : current.filter((id) => id !== String(tagId));
              return n8n.put(`/workflows/${enc(w.id)}/tags`, next.map((id) => ({ id })));
            },
          });
          invalidate('workflows', 'tags');
          refresh();
        },
      }],
    });
  }

  async function bulkTransfer(items) {
    let projs;
    try { projs = await load('projects'); } catch (err) { if (err.license) { modal({ title: 'Transfer', body: licenseGate(err) }); return; } throw err; }
    const sel = h('select.form-select.form-select-sm', projs.map((p) => h('option', { value: p.id }, p.name)));
    modal({
      title: 'Transfer workflows', size: 'sm', body: sel,
      actions: [{ label: 'Cancel' }, { label: 'Transfer', variant: 'primary', onClick: async (close) => { close(); await bulkRun({ title: 'Transferring', items, label: (w) => w.name, run: (w) => n8n.put(`/workflows/${enc(w.id)}/transfer`, { destinationProjectId: sel.value }) }); invalidate('workflows'); refresh(); } }],
    });
  }

  function exportJson(items) {
    const list = (items || all).map((w) => ({ id: w.id, ...W.saveBody(w), tags: w.tags || [], active: w.active }));
    saveText(JSON.stringify(list, null, 2), `n8n-workflows-${new Date().toISOString().slice(0, 10)}.json`);
    toast(`Exported ${list.length} workflow(s)`, 'success');
  }

  async function exportPackage(items) {
    try {
      const r = await download('POST', '/n8n-packages/export', { workflowIds: items.map((w) => w.id), includeTags: true, missingWorkflowDependencyPolicy: 'include-in-package' }, 'workflows.n8np');
      toast(`Downloaded ${r.name}`, 'success');
    } catch (err) { toastError(err, 'Package export failed'); }
  }

  // ---------------------------------------------------------------- import

  function openImport() {
    const editor = jsonEditor('', { rows: 12 });
    const status = h('div.small.mt-2');
    const m = modal({
      title: 'Import workflows', icon: 'upload', size: 'lg',
      body: h('div',
        h('p.small.text-muted', 'Paste workflow JSON (one workflow, an array, or nodes copied from any n8n editor), drop .json files, or import an n8n package (.n8np). Credentials are matched to yours by type and name.'),
        h('div.d-flex.gap-2.mb-2',
          h('button.btn.btn-sm.btn-outline-secondary', { onclick: async () => { const file = await pickFile('.json,application/json'); if (file) { editor.set(await readFileText(file)); } } }, icon('filetype-json', 'me-1'), 'Choose .json'),
          h('button.btn.btn-sm.btn-outline-secondary', { onclick: async () => { const file = await pickFile('.n8np,.gz,.tgz,application/gzip'); if (file) { m.close(); importPackage(file); } } }, icon('box-seam', 'me-1'), 'Import .n8np package')),
        editor.el, status),
      actions: [{ label: 'Cancel' }, { label: 'Next', variant: 'primary', onClick: async (close) => { const parsed = parseImport(editor.textarea.value); close(); await importWizard(parsed); } }],
    });
    m.el.addEventListener('dragover', (e) => e.preventDefault());
    m.el.addEventListener('drop', async (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (!file) return;
      if (/\.n8np$|\.gz$/.test(file.name)) { m.close(); importPackage(file); } else editor.set(await readFileText(file));
    });
  }

  function parseImport(text) {
    let data;
    try { data = JSON.parse(text); } catch (err) { throw new Error(`Not valid JSON: ${err.message}`); }
    const list = Array.isArray(data) ? data : data.workflows && Array.isArray(data.workflows) ? data.workflows : [data];
    const wfs = list.map((w) => (w.nodes ? w : W.parseClipboard(JSON.stringify(w)) ? { name: 'Imported nodes', ...W.parseClipboard(JSON.stringify(w)) } : null)).filter(Boolean);
    if (!wfs.length) throw new Error('No workflows found in that JSON (expected objects with "nodes" and "connections").');
    return wfs;
  }

  async function importWizard(wfs) {
    const creds = await loadOptional('credentials', { force: true });
    const byId = new Map(creds.map((c) => [String(c.id), c]));
    const refs = new Map();
    for (const w of wfs) for (const r of W.credentialRefs(w)) if (!r.id || !byId.has(r.id)) refs.set(`${r.type}\u0000${r.id}\u0000${r.name}`, r);
    const mapping = new Map();
    const rows = [...refs.values()].map((r) => {
      const candidates = creds.filter((c) => c.type === r.type);
      const auto = candidates.find((c) => c.name === r.name) || (candidates.length === 1 ? candidates[0] : null);
      const sel = h('select.form-select.form-select-sm', h('option', { value: '' }, candidates.length ? '— leave unset —' : `no ${credentialDisplayName(r.type)} credentials`), candidates.map((c) => h('option', { value: c.id, selected: auto?.id === c.id }, c.name)));
      const key = `${r.type}\u0000${r.id}\u0000${r.name}`;
      if (auto) mapping.set(key, auto);
      sel.addEventListener('change', () => { const c = byId.get(sel.value); if (c) mapping.set(key, c); else mapping.delete(key); });
      return h('tr', h('td', h('b', r.name || r.id || '(unnamed)'), h('div.small.text-muted', credentialDisplayName(r.type))), h('td', sel));
    });
    const activate = h('input.form-check-input', { type: 'checkbox' });
    modal({
      title: `Import ${wfs.length} workflow(s)`, icon: 'upload', size: 'lg',
      body: h('div',
        h('ul.small', wfs.slice(0, 12).map((w) => h('li', h('b', w.name || 'Untitled'), ` — ${(w.nodes || []).length} nodes`)), wfs.length > 12 && h('li', `…${wfs.length - 12} more`)),
        rows.length ? h('div', h('div.fw-semibold.small.mb-1', icon('key', 'me-1'), 'Map credentials'), h('table.af-table.af-table-dense', h('tbody', rows))) : h('div.small.text-success', icon('check2-circle', 'me-1'), 'All credential references resolve on this instance.'),
        h('label.form-check.mt-3', activate, h('span.form-check-label.small', 'Activate imported workflows that have triggers'))),
      actions: [{ label: 'Cancel' }, {
        label: 'Import', variant: 'primary',
        onClick: async (close) => {
          close();
          const results = await bulkRun({
            title: 'Importing workflows', items: wfs, label: (w) => w.name || 'Untitled',
            run: async (w) => {
              const remapped = {
                ...w,
                nodes: (w.nodes || []).map((n) => {
                  if (!n.credentials) return n;
                  const credentials = {};
                  for (const [type, ref] of Object.entries(n.credentials)) {
                    const c = mapping.get(`${type}\u0000${ref?.id ?? null}\u0000${ref?.name ?? null}`) || (ref?.id && byId.get(String(ref.id)));
                    credentials[type] = c ? { id: String(c.id), name: c.name } : ref;
                  }
                  return { ...n, credentials };
                }),
              };
              const created = await n8n.post('/workflows', W.importBody(remapped));
              if (activate.checked && W.triggers(remapped, catalog.nodes).length) await n8n.post(`/workflows/${enc(created.id)}/activate`, {}).catch(() => null);
              return created;
            },
          });
          invalidate('workflows');
          if (results.ok.length) toast(`Imported ${results.ok.length} workflow(s)`, 'success');
          refresh();
        },
      }],
    });
  }

  async function importPackage(file) {
    const opt = (name, values, def, help) => {
      const sel = h('select.form-select.form-select-sm', { name }, values.map((v) => h('option', { value: v, selected: v === def }, v)));
      return h('div.col-md-6', h('label.small.fw-semibold', name.replace(/([A-Z])/g, ' $1').toLowerCase()), sel, help && h('div.form-text.small', help));
    };
    const projs = await loadOptional('projects');
    const body = h('div.row.g-2',
      h('div.col-12.small', icon('box-seam', 'me-1'), h('b', file.name), ` · ${(file.size / 1024).toFixed(1)} KB`),
      projs.length ? h('div.col-md-6', h('label.small.fw-semibold', 'target project'), h('select.form-select.form-select-sm', { name: 'projectId' }, h('option', { value: '' }, 'My personal project'), projs.map((p) => h('option', { value: p.id }, p.name)))) : null,
      opt('workflowConflictPolicy', ['new-version', 'skip', 'fail'], 'new-version', 'When a workflow already exists'),
      opt('workflowIdPolicy', ['new', 'source'], 'new'),
      opt('credentialMatchingMode', ['name-and-type', 'type-only', 'id-only'], 'name-and-type'),
      opt('credentialMissingMode', ['create-stub', 'must-preexist'], 'create-stub'),
      opt('missingNodeTypeMode', ['fail', 'import-anyway'], 'fail'),
      opt('workflowPublishingPolicy', ['preserve-published-state', 'match-source', 'publish-all', 'unpublish-all'], 'preserve-published-state'),
      opt('folderConflictPolicy', ['merge', 'fail'], 'merge'),
      opt('dataTableMissingMode', ['create', 'must-preexist', 'do-nothing'], 'create'),
      opt('variableMissingMode', ['do-nothing', 'must-preexist', 'create-stub', 'create-with-value'], 'do-nothing'));
    modal({
      title: 'Import n8n package', icon: 'box-seam', size: 'lg', body,
      actions: [{ label: 'Cancel' }, {
        label: 'Import package', variant: 'primary',
        onClick: async (close) => {
          const fd = new FormData();
          fd.append('package', file, file.name);
          body.querySelectorAll('select[name]').forEach((s) => { if (s.value) fd.append(s.name, s.value); });
          const res = await upload('/n8n-packages/import', fd);
          close();
          invalidate('workflows', 'credentials', 'tags');
          modal({
            title: 'Package imported', icon: 'check2-circle', size: 'lg',
            body: h('div', h('div.small.mb-2', `From n8n ${res.package?.sourceN8nVersion} · exported ${res.package?.exportedAt}`),
              dataTable({ rows: res.workflows || [], dense: true, columns: [{ key: 'name', label: 'Workflow', render: (w) => (w.localId ? h('a', { href: urls.workflow(w.localId) }, w.name) : w.name) }, { key: 'status', label: 'Result' }, { key: 'publishing', label: 'Publishing' }] }),
              h('pre.small.mt-2', JSON.stringify({ credentials: res.credentials, variables: res.variables, tags: res.tags }, null, 2))),
          });
          refresh();
        },
      }],
    });
  }

  if (ctx.query.get('import')) { ctx.setQuery({ import: null }); setTimeout(openImport, 50); }
  await refresh();
}

function hueOf(s) { let x = 0; for (const c of String(s)) x = (x * 31 + c.charCodeAt(0)) >>> 0; return x % 360; }

