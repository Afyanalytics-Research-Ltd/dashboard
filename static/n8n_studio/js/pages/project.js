// Project detail: members (add, change role, remove), folder tree (create, rename,
// move, delete with transfer), and the project's workflows.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toastError, modal, formModal, card, spinner, timeAgo } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load, loadOptional } from '../core/store.js';
import { gated, emptyHint, confirmDelete, avatar, userLabel, workflowLink, activeSwitch } from '../core/widgets.js';

const PROJECT_ROLES = ['project:admin', 'project:editor', 'project:viewer'];

export default async function render(ctx) {
  const pid = ctx.params.projectId;
  const projects = await load('projects').catch(() => []);
  const project = projects.find((p) => p.id === pid) || { id: pid, name: pid };
  ctx.setTitle(project.name);
  const membersEl = h('div');
  const foldersEl = h('div');
  const wfEl = h('div');
  mount(ctx.el,
    pageHeader({ title: project.name, icon: 'kanban', crumbs: [{ label: 'Projects', href: urls.projects() }], subtitle: `${project.type || ''} project · ${pid}` }),
    h('div.n8s-grid-2',
      card({ title: 'Members', icon: 'people', body: membersEl, actions: h('button.btn.btn-sm.btn-primary', { onclick: () => addMembers() }, icon('person-plus', 'me-1'), 'Add') }),
      card({ title: 'Folders', icon: 'folder2', body: foldersEl, actions: h('button.btn.btn-sm.btn-outline-primary', { onclick: () => folderForm() }, icon('folder-plus', 'me-1'), 'New folder') }),
      h('div.n8s-span-2', card({ title: 'Workflows', icon: 'diagram-3', body: wfEl }))));

  let roles = PROJECT_ROLES;
  loadRoles();
  async function loadRoles() {
    try { const r = await n8n.get('/roles'); roles = (r.project || []).map((x) => x.slug); } catch { /* keep defaults */ }
  }

  const refreshMembers = () => gated(membersEl, async () => {
    const members = await n8n.all(`/projects/${enc(pid)}/users`, {}, { signal: ctx.signal });
    return dataTable({
      rows: members, dense: true, rowKey: (m) => m.id,
      empty: emptyHint('people', 'No members'),
      columns: [
        { key: 'user', label: 'User', render: (m) => h('span.d-flex.gap-2.align-items-center', avatar(m), h('span', userLabel(m), h('div.small.text-muted', m.email))) },
        {
          key: 'role', label: 'Role', render: (m) => {
            const sel = h('select.form-select.form-select-sm', roles.map((r) => h('option', { value: r, selected: r === m.role }, r.replace('project:', ''))));
            sel.addEventListener('change', async () => { try { await n8n.patch(`/projects/${enc(pid)}/users/${enc(m.id)}`, { role: sel.value }); } catch (err) { toastError(err); sel.value = m.role; } });
            return sel;
          },
        },
        { key: 'x', label: '', align: 'right', render: (m) => h('button.btn.btn-sm.btn-outline-danger', { onclick: async () => { if (await confirmDelete('member', userLabel(m))) { await n8n.del(`/projects/${enc(pid)}/users/${enc(m.id)}`).catch(toastError); refreshMembers(); } } }, icon('person-dash')) },
      ],
    });
  });

  async function addMembers() {
    const users = await loadOptional('users', { force: true });
    const picks = new Set();
    const role = h('select.form-select.form-select-sm', roles.map((r) => h('option', { value: r }, r.replace('project:', ''))));
    modal({
      title: 'Add members', icon: 'person-plus',
      body: h('div', h('div.n8s-userpick', users.map((u) => h('label.n8s-userpick-row', h('input.form-check-input', { type: 'checkbox', onchange: (e) => (e.target.checked ? picks.add(u.id) : picks.delete(u.id)) }), avatar(u, 22), h('span', userLabel(u)), h('span.small.text-muted.ms-auto', u.email)))), h('label.small.fw-semibold.mt-2', 'Role'), role),
      actions: [{ label: 'Cancel' }, { label: 'Add', variant: 'primary', onClick: async (close) => { await n8n.post(`/projects/${enc(pid)}/users`, { relations: [...picks].map((userId) => ({ userId, role: role.value })) }); close(); refreshMembers(); } }],
    });
  }

  let folders = [];
  const refreshFolders = () => gated(foldersEl, async () => {
    const res = await n8n.get(`/projects/${enc(pid)}/folders`, { take: '500', sortBy: 'name:asc' }, { signal: ctx.signal });
    folders = res.data || [];
    const children = (parent) => folders.filter((f) => (f.parentFolderId || null) === parent);
    const node = (f, depth) => h('div',
      h('div.n8s-folder', { style: { paddingLeft: `${depth * 18 + 4}px` } }, icon('folder2'), h('span.flex-grow-1', f.name), f.totalWorkflows != null && h('span.small.text-muted', `${f.totalWorkflows} wf`),
        h('button.af-icon-btn', { title: 'Subfolder', onclick: () => folderForm({ parentFolderId: f.id }) }, icon('folder-plus')),
        h('button.af-icon-btn', { title: 'Rename / move', onclick: () => folderForm(f) }, icon('pencil')),
        h('button.af-icon-btn', { title: 'Delete', onclick: () => deleteFolder(f) }, icon('trash'))),
      children(f.id).map((c) => node(c, depth + 1)));
    return folders.length ? h('div.n8s-folders', children(null).map((f) => node(f, 0))) : emptyHint('folder2', 'No folders');
  });

  function folderForm(f = {}) {
    const existing = !!f.id;
    formModal({
      title: existing ? 'Edit folder' : 'New folder', icon: 'folder2', size: 'sm',
      fields: [
        { name: 'name', label: 'Name', value: f.name, required: true },
        { name: 'parentFolderId', label: 'Parent', type: 'select', value: f.parentFolderId || '', options: [{ value: '', label: '(project root)' }, ...folders.filter((x) => x.id !== f.id).map((x) => ({ value: x.id, label: x.name }))] },
      ],
      onSubmit: async (v, close) => {
        const body = { name: v.name, ...(v.parentFolderId ? { parentFolderId: v.parentFolderId } : {}) };
        if (existing) await n8n.patch(`/projects/${enc(pid)}/folders/${enc(f.id)}`, body); else await n8n.post(`/projects/${enc(pid)}/folders`, body);
        close();
        refreshFolders();
      },
    });
  }

  function deleteFolder(f) {
    const sel = h('select.form-select.form-select-sm', h('option', { value: '' }, 'Delete its workflows too'), folders.filter((x) => x.id !== f.id).map((x) => h('option', { value: x.id }, `Move contents to “${x.name}”`)));
    modal({
      title: `Delete folder ${f.name}?`, icon: 'trash', size: 'sm', body: sel,
      actions: [{ label: 'Cancel' }, { label: 'Delete', variant: 'danger', onClick: async (close) => { await n8n.del(`/projects/${enc(pid)}/folders/${enc(f.id)}`, { params: sel.value ? { transferToFolderId: sel.value } : undefined }); close(); refreshFolders(); } }],
    });
  }

  async function refreshWorkflows() {
    mount(wfEl, spinner());
    const all = await load('workflows').catch(() => []);
    const rows = all.filter((w) => (w.shared || []).some((s) => s.projectId === pid));
    mount(wfEl, dataTable({
      rows, dense: true, empty: emptyHint('diagram-3', 'No workflows in this project'),
      columns: [
        { key: 'name', label: 'Workflow', render: (w) => workflowLink(w) },
        { key: 'folder', label: 'Folder', render: (w) => w.parentFolder?.name || '—' },
        { key: 'updatedAt', label: 'Updated', render: (w) => timeAgo(w.updatedAt) },
        { key: 'active', label: 'Active', render: (w) => activeSwitch(w) },
      ],
    }));
  }

  refreshMembers();
  refreshFolders();
  refreshWorkflows();
}
