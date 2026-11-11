// Reports of a project (W&B "runs/report" views): search, open, create, duplicate, delete.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, emptyState, errorState, skeleton, timeAgo, searchInput, menuButton, confirmDialog, toast, toastError } from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { urls } from '../core/urls.js';
import { listReports, deleteReport, saveReport, reportSummary } from './artifacts/report_data.js';

export default async function render(ctx) {
  const { entity, project } = ctx.params;
  const { signal } = ctx;
  ctx.setTitle(`Reports · ${project}`);
  let q = ctx.query.get('q') || '';
  const gridEl = h('div', skeleton(6));
  let reports = [];

  mount(ctx.el,
    pageHeader({
      title: 'Reports', icon: 'file-earmark-richtext',
      crumbs: [{ label: 'Projects', href: urls.projects() }, { label: project, href: urls.workspace(entity, project) }, { label: 'Reports' }],
      subtitle: 'Write up findings next to the live charts they are about — no more conclusions lost in spreadsheets.',
      actions: [h('a.btn.btn-sm.btn-primary', { href: urls.report(entity, project, 'new') }, icon('plus-lg', 'me-1'), 'New report')],
    }),
    h('div.ma-reg-filters', searchInput({ placeholder: 'Search reports…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } })),
    gridEl);

  async function load() {
    try {
      reports = await listReports(entity, project, { signal });
      if (ctx.active) draw();
    } catch (err) { if (err.name !== 'AbortError') mount(gridEl, errorState(err, load)); }
  }

  function draw() {
    if (!reports.length) {
      mount(gridEl, emptyState({
        icon: 'file-earmark-richtext', title: 'No reports yet',
        message: 'A report mixes text, decisions and live charts of your runs. Start one for your next experiment write-up.',
        action: h('a.btn.btn-sm.btn-primary', { href: urls.report(entity, project, 'new') }, icon('plus-lg', 'me-1'), 'Create the first report'),
      }));
      return;
    }
    const needle = q.toLowerCase();
    const rows = reports.filter((r) => !needle || `${r.displayName} ${r.description || ''} ${r.user?.username || ''}`.toLowerCase().includes(needle));
    if (!rows.length) { mount(gridEl, emptyState({ icon: 'search', title: 'No report matches' })); return; }
    mount(gridEl, h('div.ma-report-list', rows.map((r) => {
      const sum = reportSummary(r);
      const link = urls.report(entity, project, r.id);
      return h('a.ma-report-card', { href: link },
        h('div.d-flex.justify-content-between.gap-2',
          h('div.ma-report-title', r.displayName || 'Untitled report'),
          h('span.af-no-row-click', { onclick: (e) => e.preventDefault() }, menuButton(null, () => [
            { label: 'Open', icon: 'box-arrow-up-right', onClick: () => navigate(link) },
            { label: 'Edit', icon: 'pencil', onClick: () => navigate(`${link}?edit=1`) },
            { label: 'Duplicate', icon: 'copy', onClick: () => duplicate(r) },
            { label: 'Copy link', icon: 'link-45deg', onClick: () => navigator.clipboard.writeText(location.origin + link).then(() => toast('Link copied', 'success', { timeout: 1500 })) },
            { divider: true },
            { label: 'Delete…', icon: 'trash', danger: true, onClick: () => remove(r) },
          ]))),
        r.description ? h('div.ma-desc', r.description) : h('div.ma-desc', sum.preview || h('span.fst-italic', 'Empty report')),
        h('div.ma-col-facts', h('span', icon('text-paragraph', 'me-1'), `${sum.blocks} block${sum.blocks === 1 ? '' : 's'}`), sum.charts ? h('span', icon('graph-up', 'me-1'), `${sum.charts} chart${sum.charts === 1 ? '' : 's'}`) : null),
        h('div.ma-report-meta', h('span', icon('person', 'me-1'), r.user?.username || '—'), h('span', 'updated ', timeAgo(r.updatedAt || r.createdAt))));
    })));
  }

  async function duplicate(r) {
    try {
      const copy = await saveReport({ entity, project, displayName: `${r.displayName || 'Untitled'} (copy)`, description: r.description, spec: r.specObj });
      toast('Report duplicated', 'success');
      navigate(urls.report(entity, project, copy.id));
    } catch (err) { toastError(err, 'Could not duplicate'); }
  }

  async function remove(r) {
    const ok = await confirmDialog({ title: `Delete “${r.displayName || 'Untitled report'}”?`, danger: true, confirmLabel: 'Delete report', message: 'Anyone with the link loses access. The runs it charts are not affected.' });
    if (!ok) return;
    try {
      await deleteReport(r.id);
      reports = reports.filter((x) => x.id !== r.id);
      draw();
      toast('Report deleted', 'success');
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  await load();
}
