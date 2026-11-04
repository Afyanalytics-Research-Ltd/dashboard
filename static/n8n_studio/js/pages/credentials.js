// Credentials: usage map (which workflows use each credential), unused / missing
// detection, typed create & edit forms (catalog definition + API JSON schema), test,
// transfer between projects, bulk clean-up.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toast, toastError, modal, chips, searchInput, selectionBar, spinner, timeAgo, menuButton, card } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load, invalidate, loadOptional } from '../core/store.js';
import { loadCatalog, catalog, credentialIcon, credentialDisplayName, loadCredentialType } from '../core/catalog.js';
import { credentialRefs } from '../core/workflow.js';
import { rank } from '../core/fuzzy.js';
import { isVisible } from '../core/params.js';
import { bulkRun, emptyHint, confirmDelete, licenseGate, workflowLink } from '../core/widgets.js';
import { stripHtml } from '../editor/paramform.js';

export default async function render(ctx) {
  await loadCatalog().catch(() => null);
  let filter = ctx.query.get('filter') || 'all';
  let q = '';
  const selected = new Set();
  const listEl = h('div');
  const barEl = h('div');
  const missingEl = h('div');
  mount(ctx.el,
    pageHeader({
      title: 'Credentials', icon: 'key', subtitle: 'See exactly which workflows use each credential before you rotate or delete it.',
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => createFlow() }, icon('plus-lg', 'me-1'), 'New credential')],
    }),
    h('div.n8s-wf-toolbar', searchInput({ placeholder: 'Search credentials by name or type…', onInput: (v) => { q = v; draw(); } }), h('div.n8s-cred-chips')),
    missingEl, barEl, listEl);

  let creds = [];
  let workflows = [];
  let usage = new Map();

  async function refresh() {
    mount(listEl, spinner());
    try {
      [creds, workflows] = await Promise.all([load('credentials', { force: true, signal: ctx.signal }), load('workflows', { signal: ctx.signal })]);
    } catch (err) {
      mount(listEl, err.license ? licenseGate(err) : h('div.text-danger', err.message));
      return;
    }
    usage = new Map();
    const missing = new Map();
    const ids = new Set(creds.map((c) => String(c.id)));
    for (const w of workflows) {
      for (const r of credentialRefs(w)) {
        if (!r.id) continue;
        if (!ids.has(r.id)) { const m = missing.get(r.id) || { ...r, workflows: new Set() }; m.workflows.add(w); missing.set(r.id, m); continue; }
        const u = usage.get(r.id) || { workflows: new Map(), nodes: 0 };
        u.workflows.set(w.id, w);
        u.nodes++;
        usage.set(r.id, u);
      }
    }
    mount(missingEl, missing.size ? card({
      title: `${missing.size} missing credential${missing.size > 1 ? 's' : ''} referenced by workflows`, icon: 'exclamation-triangle', cls: 'n8s-warn-card',
      body: h('div', [...missing.values()].map((m) => h('div.small', h('b', m.name || m.id), ` (${credentialDisplayName(m.type)}) — used in `, [...m.workflows].map((w, i) => [i ? ', ' : '', h('a', { href: urls.workflow(w.id) }, w.name)])))),
    }) : null);
    draw();
    const focus = ctx.query.get('focus');
    if (focus) { const c = creds.find((x) => String(x.id) === focus); if (c) showUsage(c); }
    const create = ctx.query.get('create');
    if (create) { ctx.setQuery({ create: null }); createFlow(create === '1' ? null : create); }
  }

  function draw() {
    const unused = creds.filter((c) => !usage.has(String(c.id)));
    mount(ctx.el.querySelector('.n8s-cred-chips'), chips([
      { value: 'all', label: 'All', count: creds.length },
      { value: 'used', label: 'In use', count: creds.length - unused.length },
      { value: 'unused', label: 'Unused', count: unused.length, color: '#f59e0b' },
    ], filter, (v) => { filter = v || 'all'; ctx.setQuery({ filter }); draw(); }));
    let rows = creds;
    if (filter === 'used') rows = rows.filter((c) => usage.has(String(c.id)));
    if (filter === 'unused') rows = unused;
    if (q) rows = rank(rows, q, (c) => [c.name, c.type, credentialDisplayName(c.type)], 500);
    drawBar();
    mount(listEl, dataTable({
      rows, selectable: true, selected, onSelect: drawBar,
      empty: emptyHint('key', creds.length ? 'No credentials match' : 'No credentials yet', creds.length ? null : 'Create one, then select it in any node.'),
      columns: [
        { key: 'name', label: 'Name', render: (c) => h('div.d-flex.align-items-center.gap-2', credentialIcon(c.type, 26), h('div', h('div.fw-semibold', c.name), h('div.small.text-muted', credentialDisplayName(c.type)))) },
        {
          key: 'usage', label: 'Used by', render: (c) => {
            const u = usage.get(String(c.id));
            if (!u) return h('span.n8s-pill-warn', 'unused');
            const list = [...u.workflows.values()];
            return h('button.af-link-btn', { onclick: () => showUsage(c) }, `${list.length} workflow${list.length > 1 ? 's' : ''}`, h('span.text-muted', ` · ${u.nodes} node${u.nodes > 1 ? 's' : ''}`));
          },
        },
        { key: 'updatedAt', label: 'Updated', render: (c) => timeAgo(c.updatedAt) },
        { key: 'createdAt', label: 'Created', render: (c) => timeAgo(c.createdAt) },
        {
          key: 'act', label: '', align: 'right', render: (c) => h('div.d-flex.gap-1.justify-content-end',
            h('button.btn.btn-sm.btn-outline-secondary', { title: 'Test connection', onclick: (e) => test(c, e.currentTarget) }, icon('plug')),
            menuButton('', [
              { label: 'Edit…', icon: 'pencil', onClick: () => editFlow(c) },
              { label: 'Where is it used?', icon: 'diagram-3', onClick: () => showUsage(c) },
              { label: 'Transfer to project…', icon: 'arrow-left-right', onClick: () => transfer(c) },
              { divider: true },
              { label: 'Delete…', icon: 'trash', danger: true, onClick: () => del([c]) },
            ])),
        },
      ],
      onRowClick: (c) => editFlow(c),
    }));
  }

  function drawBar() {
    const items = creds.filter((c) => selected.has(c.id));
    mount(barEl, selectionBar(items.length, [
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => bulkRun({ title: 'Testing credentials', items, label: (c) => c.name, run: async (c) => { const r = await n8n.post(`/credentials/${enc(c.id)}/test`, {}); if (r?.status && r.status !== 'OK') throw new Error(r.message || 'Test failed'); return r; } }) }, icon('plug', 'me-1'), 'Test'),
      h('button.btn.btn-sm.btn-outline-danger', { onclick: () => del(items) }, icon('trash', 'me-1'), 'Delete'),
    ], () => { selected.clear(); draw(); }));
  }

  function showUsage(c) {
    const u = usage.get(String(c.id));
    const list = u ? [...u.workflows.values()] : [];
    modal({
      title: c.name, subtitle: credentialDisplayName(c.type), icon: 'diagram-3', size: 'lg',
      body: list.length ? h('div', h('p.small.text-muted', `Rotating or deleting this credential affects ${list.length} workflow(s):`),
        dataTable({
          rows: list, dense: true,
          columns: [
            { key: 'name', label: 'Workflow', render: (w) => workflowLink(w) },
            { key: 'nodes', label: 'Nodes using it', render: (w) => credentialRefs(w).filter((r) => r.id === String(c.id)).map((r) => r.node).join(', ') },
            { key: 'active', label: 'Active', render: (w) => (w.active ? h('span.text-success', 'active') : h('span.text-muted', 'inactive')) },
          ],
        })) : emptyHint('check2-circle', 'Not used by any workflow', 'It is safe to delete.'),
    });
  }

  async function test(c, btn) {
    const old = [...btn.childNodes];
    mount(btn, spinner('sm'));
    try {
      const r = await n8n.post(`/credentials/${enc(c.id)}/test`, {});
      if (!r || r.status === 'OK') toast(`${c.name}: connection OK`, 'success');
      else toast(`${c.name}: ${r.message || 'test failed'}`, 'error');
    } catch (err) { toastError(err, `${c.name}: test failed`); } finally { mount(btn, ...old); }
  }

  async function del(items) {
    const inUse = items.filter((c) => usage.has(String(c.id)));
    if (items.length === 1 && !(await confirmDelete('credential', items[0].name, { details: inUse.length ? h('div.text-danger.small', icon('exclamation-triangle', 'me-1'), `Used by ${usage.get(String(items[0].id)).workflows.size} workflow(s) — they will fail.`) : null }))) return;
    if (items.length > 1 && !(await confirmDelete(`${items.length} credentials`, `${items.length} credentials`, { details: inUse.length ? h('div.text-danger.small', `${inUse.length} of them are in use.`) : null }))) return;
    await bulkRun({ title: 'Deleting credentials', items, label: (c) => c.name, run: (c) => n8n.del(`/credentials/${enc(c.id)}`) });
    selected.clear();
    invalidate('credentials');
    refresh();
  }

  async function transfer(c) {
    const projects = await load('projects').catch((err) => err);
    if (projects instanceof Error) { modal({ title: 'Transfer credential', body: projects.license ? licenseGate(projects) : projects.message }); return; }
    const sel = h('select.form-select.form-select-sm', projects.map((p) => h('option', { value: p.id }, p.name)));
    modal({
      title: `Transfer ${c.name}`, size: 'sm', body: sel,
      actions: [{ label: 'Cancel' }, { label: 'Transfer', variant: 'primary', onClick: async (close) => { await n8n.put(`/credentials/${enc(c.id)}/transfer`, { destinationProjectId: sel.value }); toast('Transferred', 'success'); close(); refresh(); } }],
    });
  }

  // ---------------------------------------------------------------- create / edit

  function createFlow(type) {
    if (type) { credentialForm({ type }); return; }
    const all = [...catalog.credentials.values()];
    const search = h('input.form-control', { placeholder: `Search ${all.length} credential types…`, autofocus: true });
    const list = h('div.n8s-pal-list');
    const draw2 = () => mount(list, rank(all, search.value, (c) => [c.displayName, c.name], 80).map((c) => h('button.n8s-pal-item', { type: 'button', onclick: () => { m.close(); credentialForm({ type: c.name }); } },
      credentialIcon(c.name, 28), h('div.n8s-pal-text', h('div.n8s-pal-name', c.displayName), h('div.n8s-pal-desc', c.name, c.testable ? ' · testable' : '')))));
    search.addEventListener('input', draw2);
    search.addEventListener('keydown', (e) => { if (e.key === 'Enter') list.querySelector('button')?.click(); });
    const m = modal({ title: 'New credential', icon: 'key', size: 'lg', body: h('div.n8s-pal', search, list) });
    draw2();
    setTimeout(() => search.focus(), 30);
  }

  function editFlow(c) { credentialForm({ type: c.type, existing: c }); }

  async function credentialForm({ type, existing }) {
    const m = modal({ title: existing ? `Edit ${existing.name}` : `New ${credentialDisplayName(type)}`, icon: 'key', size: 'lg', body: spinner() });
    let def = null;
    let schema = null;
    try {
      [def, schema] = await Promise.all([loadCredentialType(type).catch(() => null), n8n.get(`/credentials/schema/${enc(type)}`)]);
    } catch (err) { mount(m.body, h('div.text-danger', err.message)); return; }
    const props = def?.properties || Object.entries(schema.properties || {}).map(([name, p]) => ({ name, displayName: name, type: p.type === 'boolean' ? 'boolean' : p.type === 'number' ? 'number' : p.enum ? 'options' : 'string', options: (p.enum || []).map((v) => ({ name: v, value: v })) }));
    const required = new Set(schema.required || []);
    const values = {};
    const nameInput = h('input.form-control.form-control-sm', { value: existing?.name || `${credentialDisplayName(type)} account`, autofocus: true });
    const projects = existing ? [] : await loadOptional('projects');
    const projectSel = projects.length > 1 && h('select.form-select.form-select-sm', h('option', { value: '' }, 'My personal project'), projects.filter((p) => p.type !== 'personal').map((p) => h('option', { value: p.id }, p.name)));
    const fieldsEl = h('div');
    const errorBox = h('div.alert.alert-danger.small.py-2.d-none');

    const drawFields = () => mount(fieldsEl, props.filter((p) => p.type !== 'hidden' && isVisible(p, values, values, {})).map((p) => {
      const id = `cf-${p.name}`;
      const v = values[p.name] ?? (existing ? undefined : p.default);
      let input;
      if (p.type === 'boolean') {
        input = h('div.form-check.form-switch', h('input.form-check-input', { id, type: 'checkbox', checked: !!v, onchange: (e) => { values[p.name] = e.target.checked; drawFields(); } }));
      } else if (p.type === 'options') {
        input = h('select.form-select.form-select-sm', { id, onchange: (e) => { values[p.name] = JSON.parse(e.target.value); drawFields(); } },
          (p.options || []).map((o) => h('option', { value: JSON.stringify(o.value), selected: o.value === v }, o.name)));
        if (values[p.name] === undefined && v !== undefined) values[p.name] = v;
      } else if (p.type === 'notice') {
        return h('div.n8s-notice.small', stripHtml(p.displayName));
      } else {
        const secret = p.typeOptions?.password;
        input = h(p.typeOptions?.rows ? 'textarea.form-control.form-control-sm' : 'input.form-control.form-control-sm', {
          id, type: secret ? 'password' : p.type === 'number' ? 'number' : 'text', rows: p.typeOptions?.rows, autocomplete: 'off',
          value: v ?? '', placeholder: existing && secret ? '•••••• unchanged — type to replace' : p.placeholder || '',
          oninput: (e) => { values[p.name] = p.type === 'number' ? Number(e.target.value) : e.target.value; },
        }, p.typeOptions?.rows ? (v ?? '') : null);
        if (values[p.name] === undefined && v !== undefined && v !== '' && !existing) values[p.name] = v;
      }
      return h('div.af-form-row', h('label.form-label.small.fw-semibold', { for: id }, p.displayName || p.name, required.has(p.name) && h('span.text-danger', ' *')), input, p.description && h('div.form-text.small', stripHtml(p.description)));
    }));
    drawFields();

    mount(m.body, h('div',
      errorBox,
      h('div.d-flex.align-items-center.gap-2.mb-3', credentialIcon(type, 32), h('div', h('b', credentialDisplayName(type)), def?.documentationUrl && h('a.small.ms-2', { href: /^https?:/.test(def.documentationUrl) ? def.documentationUrl : `https://docs.n8n.io/integrations/builtin/credentials/${def.documentationUrl}/`, target: '_blank', rel: 'noopener' }, icon('book', 'me-1'), 'docs'))),
      h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Name'), nameInput),
      projectSel && h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Project'), projectSel),
      existing && h('div.small.text-muted.mb-2', icon('shield-lock', 'me-1'), 'n8n never returns secret values. Leave fields blank to keep them; only filled fields are updated.'),
      fieldsEl));
    const footer = h('div.af-modal-footer',
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => m.close() }, 'Cancel'),
      h('button.btn.btn-sm.btn-primary', {
        onclick: async (ev) => {
          const btn = ev.currentTarget;
          errorBox.classList.add('d-none');
          btn.disabled = true;
          try {
            const data = Object.fromEntries(Object.entries(values).filter(([, v]) => v !== '' && v !== undefined));
            let saved;
            if (existing) {
              saved = await n8n.patch(`/credentials/${enc(existing.id)}`, { name: nameInput.value.trim(), ...(Object.keys(data).length ? { data, isPartialData: true } : {}) });
            } else {
              const missing = [...required].filter((k) => data[k] === undefined && props.some((p) => p.name === k && isVisible(p, values, values, {})));
              if (missing.length) throw new Error(`Required: ${missing.join(', ')}`);
              saved = await n8n.post('/credentials', { name: nameInput.value.trim(), type, data, ...(projectSel?.value ? { projectId: projectSel.value } : {}) });
            }
            invalidate('credentials');
            toast(existing ? 'Credential updated' : 'Credential created', 'success');
            if (def?.test || catalog.credentials.get(type)?.testable) {
              n8n.post(`/credentials/${enc(saved.id || existing.id)}/test`, {}).then((r) => {
                if (!r || r.status === 'OK') toast('Connection test passed', 'success', { timeout: 2500 });
                else toast(`Connection test: ${r.message}`, 'warning', { timeout: 8000 });
              }).catch(() => null);
            }
            m.close();
            refresh();
          } catch (err) {
            mount(errorBox, err.message);
            errorBox.classList.remove('d-none');
          } finally { btn.disabled = false; }
        },
      }, icon('check2', 'me-1'), existing ? 'Save' : 'Create'));
    m.el.appendChild(footer);
  }

  await refresh();
}
