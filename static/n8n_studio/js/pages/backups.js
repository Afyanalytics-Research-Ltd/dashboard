// Backups & migration: Studio snapshots of every workflow, recycle bin for deleted
// workflows, n8n package export (projects / folders / workflows) & import, full JSON export.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, dataTable, toast, toastError, spinner, timeAgo, chips } from '../../../ui/js/ui.js';
import { n8n, studio, enc, download, saveText } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load, invalidate, loadOptional } from '../core/store.js';
import { importBody, saveBody } from '../core/workflow.js';
import { emptyHint } from '../core/widgets.js';

export default async function render(ctx) {
  const snapEl = h('div');
  const binEl = h('div');
  const exportEl = h('div');
  mount(ctx.el,
    pageHeader({ title: 'Backups & migration', icon: 'cloud-arrow-down', subtitle: 'Nothing gets lost: Studio snapshots every save and every delete, independent of n8n\'s history window.' }),
    h('div.n8s-grid-2',
      card({ title: 'Studio snapshots', icon: 'camera', body: snapEl, actions: h('button.btn.btn-sm.btn-primary', { onclick: () => capture() }, icon('camera', 'me-1'), 'Back up all now') }),
      card({ title: 'Recycle bin', icon: 'trash3', body: binEl }),
      h('div.n8s-span-2', card({ title: 'Export / migrate', icon: 'box-seam', body: exportEl }))));

  async function drawSnaps() {
    mount(snapEl, spinner());
    const res = await studio.get('/snapshots/', { limit: 8 }).catch((err) => ({ error: err }));
    if (res.error) { mount(snapEl, h('div.text-danger', res.error.message)); return; }
    mount(snapEl,
      h('div.n8s-mini-stat.mb-2', h('b', res.total.toLocaleString()), ' snapshots stored · schedule ', h('code', 'python manage.py n8n_snapshot'), ' for nightly backups'),
      res.data.length ? h('div.n8s-list', res.data.map((s) => h('a.n8s-list-row', { href: urls.workflow(s.workflowId, { tab: 'versions' }) }, icon('camera'), h('span.flex-grow-1.text-truncate', s.name), h('span.n8s-chip-sm', s.source), timeAgo(s.createdAt))))
        : emptyHint('camera', 'No snapshots yet', 'Save a workflow or run a backup.'),
      h('a.btn.btn-sm.btn-link.mt-1', { href: urls.history() }, 'Browse all history →'));
  }

  async function capture() {
    mount(snapEl, h('div.text-center.p-3', spinner(), h('div.small.text-muted', 'Snapshotting every workflow…')));
    try {
      const r = await studio.post('/snapshots/capture/', {});
      toast(`${r.workflows} workflows: ${r.captured} new snapshot(s), ${r.unchanged} unchanged`, 'success');
    } catch (err) { toastError(err, 'Backup failed'); }
    drawSnaps();
  }

  async function drawBin() {
    mount(binEl, spinner());
    const res = await studio.get('/snapshots/deleted/').catch((err) => ({ error: err }));
    if (res.error) { mount(binEl, h('div.text-danger', res.error.message)); return; }
    mount(binEl, res.data.length ? dataTable({
      rows: res.data, dense: true,
      columns: [
        { key: 'name', label: 'Deleted workflow', render: (s) => h('span', h('b', s.name), h('div.small.text-muted', `${s.nodeCount} nodes · id ${s.workflowId}`)) },
        { key: 'when', label: 'Deleted', render: (s) => h('span', timeAgo(s.createdAt), s.user && h('div.small.text-muted', `by ${s.user}`)) },
        { key: 'act', label: '', align: 'right', render: (s) => h('button.btn.btn-sm.btn-outline-primary', { onclick: () => restore(s) }, icon('arrow-counterclockwise', 'me-1'), 'Restore') },
      ],
    }) : emptyHint('trash3', 'Recycle bin is empty', 'Workflows deleted through Studio appear here and can be restored.'));
  }

  async function restore(s) {
    try {
      const snap = await studio.get(`/snapshots/${s.id}/`);
      const created = await n8n.post('/workflows', importBody({ ...snap.data, name: snap.data.name }, { freshWebhooks: false }));
      if (snap.data.tags?.length) await n8n.put(`/workflows/${enc(created.id)}/tags`, snap.data.tags.map((t) => ({ id: t.id }))).catch(() => null);
      invalidate('workflows');
      toast(`Restored “${created.name}” (new id ${created.id}, inactive)`, 'success', { action: { label: 'Open', onClick: () => ctx.navigate(urls.workflow(created.id)) } });
      drawBin();
    } catch (err) { toastError(err, 'Restore failed'); }
  }

  async function drawExport() {
    const [workflows, projects] = await Promise.all([load('workflows').catch(() => []), loadOptional('projects')]);
    let scope = 'all';
    const policy = h('select.form-select.form-select-sm', ['latest', 'prefer-published', 'published-strict', 'ignore-unpublished'].map((p) => h('option', p)));
    const deps = h('select.form-select.form-select-sm', ['include-in-package', 'reference-only', 'fail'].map((p) => h('option', p)));
    const vars = h('input.form-check-input', { type: 'checkbox' });
    const projSel = projects.length && h('select.form-select.form-select-sm', { multiple: true, size: Math.min(6, projects.length) }, projects.map((p) => h('option', { value: p.id }, p.name)));
    mount(exportEl, h('div.row.g-3',
      h('div.col-lg-7',
        h('div.fw-semibold.mb-1', icon('box-seam', 'me-1'), 'n8n package (.n8np)'),
        h('p.small.text-muted', 'The native migration format: workflows with their folders, tags, sub-workflow dependencies and credential references. Import it on any n8n ≥ 2.30 from Workflows → Import.'),
        chips([{ value: 'all', label: `All workflows (${workflows.length})` }, { value: 'active', label: `Active only (${workflows.filter((w) => w.active).length})` }, projects.length ? { value: 'projects', label: 'Selected projects' } : null].filter(Boolean), scope, (v) => { scope = v || 'all'; }),
        projSel && h('div.mt-2', projSel),
        h('div.row.g-2.mt-1', h('div.col', h('label.small', 'Version policy'), policy), h('div.col', h('label.small', 'Sub-workflow dependencies'), deps)),
        h('label.n8s-chk.mt-2', vars, 'Include variable values'),
        h('div.mt-2', h('button.btn.btn-sm.btn-primary', {
          onclick: async () => {
            const body = { includeTags: true, includeVariableValues: vars.checked, workflowVersionPolicy: policy.value, missingWorkflowDependencyPolicy: deps.value };
            if (scope === 'projects') body.projectIds = [...projSel.selectedOptions].map((o) => o.value);
            else body.workflowIds = workflows.filter((w) => scope === 'all' || w.active).map((w) => w.id);
            try { const r = await download('POST', '/n8n-packages/export', body, 'n8n-export.n8np'); toast(`Downloaded ${r.name} (${(r.size / 1024).toFixed(1)} KB)`, 'success'); } catch (err) { toastError(err, 'Export failed'); }
          },
        }, icon('download', 'me-1'), 'Export package'))),
      h('div.col-lg-5',
        h('div.fw-semibold.mb-1', icon('filetype-json', 'me-1'), 'Plain JSON'),
        h('p.small.text-muted', 'One file with every workflow (n8n editor format) — handy for git or grep.'),
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => { saveText(JSON.stringify(workflows.map((w) => ({ id: w.id, ...saveBody(w), tags: w.tags, active: w.active })), null, 2), `n8n-workflows-${new Date().toISOString().slice(0, 10)}.json`); } }, icon('download', 'me-1'), 'Download all JSON'),
        h('div.fw-semibold.mt-3.mb-1', icon('upload', 'me-1'), 'Import'),
        h('a.btn.btn-sm.btn-outline-primary', { href: urls.workflows({ import: 1 }) }, 'Import JSON or .n8np →'))));
  }

  drawSnaps();
  drawBin();
  drawExport();
  if (ctx.query.get('run')) { ctx.setQuery({ run: null }); capture(); }
}
