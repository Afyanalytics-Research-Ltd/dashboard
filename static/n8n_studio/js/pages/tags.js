// Tags: usage counts, create, rename, delete, and merge (re-tag workflows, delete source).

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toast, toastError, modal, searchInput, timeAgo } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load, invalidate } from '../core/store.js';
import { gated, emptyHint, confirmDelete, promptText, tagPills, bulkRun } from '../core/widgets.js';

export default async function render(ctx) {
  let q = '';
  const body = h('div');
  mount(ctx.el, pageHeader({
    title: 'Tags', icon: 'tags', subtitle: 'Organise workflows and executions. Merge duplicates in one step.',
    actions: [h('button.btn.btn-sm.btn-primary', { onclick: async () => { const name = await promptText({ title: 'New tag', label: 'Name' }); if (name) { await n8n.post('/tags', { name }).catch(toastError); invalidate('tags'); refresh(); } } }, icon('plus-lg', 'me-1'), 'New tag')],
  }), h('div.n8s-wf-toolbar', searchInput({ placeholder: 'Search tags…', onInput: (v) => { q = v.toLowerCase(); refresh(); } })), body);

  const refresh = () => gated(body, async () => {
    const [tags, workflows] = await Promise.all([load('tags', { force: true, signal: ctx.signal }), load('workflows', { force: true }).catch(() => [])]);
    const usage = new Map(tags.map((t) => [String(t.id), workflows.filter((w) => (w.tags || []).some((x) => String(x.id) === String(t.id)))]));
    const rows = tags.filter((t) => !q || t.name.toLowerCase().includes(q)).sort((a, b) => usage.get(String(b.id)).length - usage.get(String(a.id)).length);
    return dataTable({
      rows,
      empty: emptyHint('tags', 'No tags'),
      columns: [
        { key: 'name', label: 'Tag', render: (t) => tagPills([t]) },
        { key: 'count', label: 'Workflows', render: (t) => { const n = usage.get(String(t.id)).length; return n ? h('a', { href: urls.workflows({ tag: t.name }) }, `${n} workflow${n > 1 ? 's' : ''}`) : h('span.n8s-pill-warn', 'unused'); } },
        { key: 'updatedAt', label: 'Updated', render: (t) => timeAgo(t.updatedAt) },
        {
          key: 'act', label: '', align: 'right', render: (t) => h('div.btn-group.btn-group-sm',
            h('button.btn.btn-outline-secondary', { title: 'Rename', onclick: async () => { const name = await promptText({ title: 'Rename tag', value: t.name }); if (name && name !== t.name) { await n8n.put(`/tags/${enc(t.id)}`, { name }).catch(toastError); invalidate('tags', 'workflows'); refresh(); } } }, icon('pencil')),
            h('button.btn.btn-outline-secondary', { title: 'Merge into another tag', onclick: () => merge(t, tags, usage.get(String(t.id))) }, icon('sign-merge-left')),
            h('button.btn.btn-outline-danger', { title: 'Delete', onclick: async () => { if (await confirmDelete('tag', t.name, { details: usage.get(String(t.id)).length ? h('div.small', `It is removed from ${usage.get(String(t.id)).length} workflow(s).`) : null })) { await n8n.del(`/tags/${enc(t.id)}`).catch(toastError); invalidate('tags', 'workflows'); refresh(); } } }, icon('trash'))),
        },
      ],
    });
  });

  function merge(source, tags, workflows) {
    const sel = h('select.form-select.form-select-sm', tags.filter((t) => t.id !== source.id).map((t) => h('option', { value: t.id }, t.name)));
    modal({
      title: `Merge “${source.name}” into…`, icon: 'sign-merge-left', size: 'sm',
      body: h('div', sel, h('div.small.text-muted.mt-2', `${workflows.length} workflow(s) get the target tag, then “${source.name}” is deleted.`)),
      actions: [{ label: 'Cancel' }, {
        label: 'Merge', variant: 'primary',
        onClick: async (close) => {
          close();
          const target = sel.value;
          await bulkRun({
            title: 'Merging tags', items: workflows, label: (w) => w.name,
            run: async (w) => {
              const ids = new Set((w.tags || []).map((x) => String(x.id)).filter((id) => id !== String(source.id)));
              ids.add(String(target));
              return n8n.put(`/workflows/${enc(w.id)}/tags`, [...ids].map((id) => ({ id })));
            },
          });
          await n8n.del(`/tags/${enc(source.id)}`).catch(toastError);
          invalidate('tags', 'workflows');
          toast('Tags merged', 'success');
          refresh();
        },
      }],
    });
  }

  refresh();
}
