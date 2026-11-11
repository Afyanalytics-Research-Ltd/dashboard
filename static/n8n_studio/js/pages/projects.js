// Projects: list with member/workflow counts, create, rename, delete.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toastError, formModal } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load, invalidate } from '../core/store.js';
import { gated, emptyHint, confirmDelete } from '../core/widgets.js';

export default async function render(ctx) {
  const body = h('div');
  mount(ctx.el, pageHeader({
    title: 'Projects', icon: 'kanban', subtitle: 'Group workflows and credentials, and control who can touch them.',
    actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => edit() }, icon('plus-lg', 'me-1'), 'New project')],
  }), body);

  const refresh = () => gated(body, async () => {
    const [projects, workflows] = await Promise.all([load('projects', { force: true, signal: ctx.signal }), load('workflows').catch(() => [])]);
    return dataTable({
      rows: projects,
      empty: emptyHint('kanban', 'No projects'),
      columns: [
        { key: 'name', label: 'Project', render: (p) => h('a.fw-semibold', { href: urls.project(p.id) }, p.name) },
        { key: 'type', label: 'Type', render: (p) => h('span.n8s-mode', p.type) },
        { key: 'wf', label: 'Workflows', render: (p) => workflows.filter((w) => (w.shared || []).some((s) => s.projectId === p.id)).length },
        {
          key: 'act', label: '', align: 'right', render: (p) => p.type !== 'personal' && h('div.btn-group.btn-group-sm',
            h('button.btn.btn-outline-secondary', { onclick: () => edit(p) }, icon('pencil')),
            h('button.btn.btn-outline-danger', { onclick: async () => { if (await confirmDelete('project', p.name)) { await n8n.del(`/projects/${enc(p.id)}`).catch(toastError); invalidate('projects'); refresh(); } } }, icon('trash'))),
        },
      ],
      onRowClick: (p) => ctx.navigate(urls.project(p.id)),
    });
  }, { alternative: 'Projects need an n8n Pro/Enterprise licence. Saved views on the Workflows page work as personal smart folders meanwhile.' });

  function edit(p) {
    formModal({
      title: p ? 'Rename project' : 'New project', icon: 'kanban', size: 'sm',
      fields: [{ name: 'name', label: 'Name', value: p?.name, required: true }],
      onSubmit: async (v, close) => {
        if (p) await n8n.put(`/projects/${enc(p.id)}`, { name: v.name }); else await n8n.post('/projects', { name: v.name, type: 'team' });
        invalidate('projects');
        close();
        refresh();
      },
    });
  }

  refresh();
}
