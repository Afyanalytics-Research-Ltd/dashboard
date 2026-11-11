// Variables ($vars): CRUD, project scope, and where each key is referenced.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toastError, formModal, searchInput, copyButton } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load, loadOptional } from '../core/store.js';
import { mapStrings } from '../core/workflow.js';
import { gated, emptyHint, confirmDelete } from '../core/widgets.js';

export default async function render(ctx) {
  let q = '';
  const body = h('div');
  mount(ctx.el, pageHeader({
    title: 'Variables', icon: 'braces-asterisk', subtitle: h('span', 'Instance-wide constants available in expressions as ', h('code', '$vars.KEY'), '.'),
    actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => edit() }, icon('plus-lg', 'me-1'), 'New variable')],
  }), h('div.n8s-wf-toolbar', searchInput({ placeholder: 'Search keys and values…', onInput: (v) => { q = v.toLowerCase(); refresh(); } })), body);

  const refresh = () => gated(body, async () => {
    const [vars, workflows, projects] = await Promise.all([n8n.all('/variables', {}, { signal: ctx.signal }), load('workflows').catch(() => []), loadOptional('projects')]);
    const usage = new Map();
    for (const w of workflows) {
      const found = new Set();
      for (const n of w.nodes || []) mapStrings(n.parameters || {}, (s) => { for (const m of s.matchAll(/\$vars\.([A-Za-z_][\w]*)|\$vars\[\s*['"]([^'"]+)['"]\s*\]/g)) found.add(m[1] || m[2]); return s; });
      for (const k of found) { if (!usage.has(k)) usage.set(k, []); usage.get(k).push(w); }
    }
    const rows = vars.filter((v) => !q || `${v.key} ${v.value}`.toLowerCase().includes(q));
    const undefinedRefs = [...usage.keys()].filter((k) => !vars.some((v) => v.key === k));
    return h('div',
      undefinedRefs.length ? h('div.alert.alert-warning.small', icon('exclamation-triangle', 'me-1'), 'Referenced but not defined: ', undefinedRefs.map((k, i) => [i ? ', ' : '', h('code', k), ` (${usage.get(k).map((w) => w.name).join(', ')})`])) : null,
      dataTable({
        rows,
        empty: emptyHint('braces-asterisk', 'No variables', 'Store values like API base URLs or environment names once, use them everywhere.'),
        columns: [
          { key: 'key', label: 'Key', render: (v) => h('span', h('code.fw-semibold', v.key), ' ', copyButton(`{{ $vars.${v.key} }}`)) },
          { key: 'value', label: 'Value', render: (v) => h('code.small', String(v.value ?? '').slice(0, 120)) },
          { key: 'type', label: 'Type', render: (v) => v.type || 'string' },
          projects.length > 1 && { key: 'project', label: 'Scope', render: (v) => (v.project?.name || projects.find((p) => p.id === v.projectId)?.name || 'Global') },
          { key: 'used', label: 'Used in', render: (v) => { const u = usage.get(v.key) || []; return u.length ? h('span', u.slice(0, 3).map((w, i) => [i ? ', ' : '', h('a', { href: urls.workflow(w.id) }, w.name)]), u.length > 3 && ` +${u.length - 3}`) : h('span.n8s-pill-warn', 'unused'); } },
          {
            key: 'act', label: '', align: 'right', render: (v) => h('div.btn-group.btn-group-sm',
              h('button.btn.btn-outline-secondary', { onclick: () => edit(v, projects) }, icon('pencil')),
              h('button.btn.btn-outline-danger', { onclick: async () => { if (await confirmDelete('variable', v.key, { details: usage.get(v.key)?.length ? h('div.small.text-danger', `Used by ${usage.get(v.key).length} workflow(s).`) : null })) { await n8n.del(`/variables/${enc(v.id)}`).catch(toastError); refresh(); } } }, icon('trash'))),
          },
        ].filter(Boolean),
      }));
  }, { alternative: 'Without the licence, keep shared constants in a Data table or a Set node and reference them from there.' });

  async function edit(v, projects) {
    const ps = projects || await loadOptional('projects');
    formModal({
      title: v ? `Edit ${v.key}` : 'New variable', icon: 'braces-asterisk',
      fields: [
        { name: 'key', label: 'Key', value: v?.key, required: true, help: 'Letters, numbers and underscores.' },
        { name: 'value', label: 'Value', type: 'textarea', rows: 3, value: v?.value },
        ps.length > 1 && { name: 'projectId', label: 'Scope', type: 'select', value: v?.projectId || '', options: [{ value: '', label: 'Global' }, ...ps.map((p) => ({ value: p.id, label: p.name }))] },
      ].filter(Boolean),
      onSubmit: async (vals, close) => {
        if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(vals.key)) throw new Error('Key must start with a letter/underscore and contain only letters, numbers and underscores.');
        const body2 = { key: vals.key, value: vals.value ?? '', ...(vals.projectId ? { projectId: vals.projectId } : {}) };
        if (v) await n8n.put(`/variables/${enc(v.id)}`, body2); else await n8n.post('/variables', body2);
        close();
        refresh();
      },
    });
  }

  refresh();
}
