// A report: read view (rich text + live charts of current runs) and a block editor.
// /reports/new opens the editor for a new report; ?edit=1 edits an existing one.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import {
  pageHeader, emptyState, errorState, skeleton, timeAgo, toast, toastError, confirmDialog, codeBlock, copyButton, menuButton,
} from '../../../ui/js/ui.js';
import { fmtDate } from '../../../ui/js/format.js';
import { loadMarkdown } from '../../../ui/js/lib.js';
import { navigate } from '../../../ui/js/router.js';
import { wandbLink } from '../core/api.js';
import { urls } from '../core/urls.js';
import { loadRunsPage, loadHistories, loadMetricKeys } from '../core/data.js';
import { lineChart, lazy } from '../core/charts.js';
import { runColor, displayName } from '../core/wb.js';
import {
  loadReport, saveReport, deleteReport, emptySpec, specToBlocks, blocksToSpec, panelGridInfo,
} from './artifacts/report_data.js';

const BLOCK_KINDS = [
  { kind: 'heading', label: 'Heading', icon: 'type-h1' },
  { kind: 'paragraph', label: 'Text', icon: 'text-paragraph' },
  { kind: 'markdown', label: 'Markdown', icon: 'markdown' },
  { kind: 'chart', label: 'Live chart', icon: 'graph-up' },
  { kind: 'quote', label: 'Quote / callout', icon: 'quote' },
  { kind: 'list', label: 'List', icon: 'list-ul' },
  { kind: 'code', label: 'Code', icon: 'code-slash' },
  { kind: 'divider', label: 'Divider', icon: 'hr' },
];

export default async function render(ctx) {
  const { entity, project, viewId } = ctx.params;
  const { signal } = ctx;
  const isNew = viewId === 'new';
  let editing = isNew || ctx.query.get('edit') === '1';
  let report = null;
  let blocks = [];
  let title = '';
  let description = '';
  let dirty = false;
  let metricKeys = null;

  const root = h('div.d-flex.flex-column.gap-3');
  mount(ctx.el, root, skeleton(6));

  try {
    if (!isNew) {
      report = await loadReport(viewId, { signal });
      title = report.displayName || '';
      description = report.description || '';
      blocks = specToBlocks(report.specObj);
    } else {
      title = 'Untitled report';
      blocks = [
        { kind: 'heading', level: 1, text: 'What we tried' },
        { kind: 'paragraph', text: 'Summarise the question, the runs you compared and what you decided.' },
      ];
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    mount(ctx.el, pageHeader({ title: 'Report', icon: 'file-earmark-richtext', crumbs: [{ label: project, href: urls.workspace(entity, project) }, { label: 'Reports', href: urls.reports(entity, project) }] }), errorState(err));
    return;
  }
  if (!ctx.active) return;
  mount(ctx.el, root);

  const beforeUnload = (e) => { if (dirty) { e.preventDefault(); e.returnValue = ''; } };
  window.addEventListener('beforeunload', beforeUnload);
  ctx.cleanup(() => window.removeEventListener('beforeunload', beforeUnload));
  const onKey = (e) => { if (editing && (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); save(); } };
  document.addEventListener('keydown', onKey);
  ctx.cleanup(() => document.removeEventListener('keydown', onKey));

  const markDirty = () => { dirty = true; saveBtn?.classList.add('ma-dirty'); };
  let saveBtn = null;

  function header() {
    ctx.setTitle(title || 'Report');
    const shareUrl = report ? `${location.origin}${urls.report(entity, project, report.id)}` : '';
    const wb = report ? wandbLink(`/${entity}/${project}/reports/${encodeURIComponent((title || 'report').replace(/\s+/g, '-'))}--${report.id}`) : null;
    saveBtn = editing ? h('button.btn.btn-sm.btn-primary', { type: 'button', onclick: save, title: 'Save (Ctrl/⌘+S)' }, icon('cloud-arrow-up', 'me-1'), isNew ? 'Create report' : 'Save') : null;
    return pageHeader({
      title: editing ? (isNew ? 'New report' : 'Editing report') : (title || 'Untitled report'), icon: 'file-earmark-richtext',
      crumbs: [{ label: 'Projects', href: urls.projects() }, { label: project, href: urls.workspace(entity, project) }, { label: 'Reports', href: urls.reports(entity, project) }, { label: title || 'Report' }],
      actions: editing
        ? [h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: cancelEdit }, 'Cancel'), saveBtn]
        : [
          shareUrl && h('span.d-inline-flex.align-items-center.gap-1', copyButton(shareUrl, 'Copy link')),
          h('button.btn.btn-sm.btn-outline-primary', { type: 'button', onclick: () => { editing = true; ctx.setQuery({ edit: 1 }); draw(); } }, icon('pencil', 'me-1'), 'Edit'),
          h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => window.print(), title: 'Print or save as PDF' }, icon('printer')),
          menuButton(null, () => [
            wb && { label: 'Open in W&B', icon: 'box-arrow-up-right', href: wb, external: true },
            { label: 'Delete report…', icon: 'trash', danger: true, onClick: remove },
          ]),
        ],
    });
  }

  function draw() {
    mount(root, header(), editing ? editor() : viewer());
  }

  // ---------------------------------------------------------------- viewer

  function viewer() {
    const doc = h('article.ma-doc');
    const parts = [
      h('h1', title || 'Untitled report'),
      h('div.ma-doc-byline',
        report?.user?.username && h('span', icon('person', 'me-1'), report.user.username),
        report && h('span', icon('clock', 'me-1'), 'updated ', timeAgo(report.updatedAt || report.createdAt), ` · ${fmtDate(report.updatedAt || report.createdAt, { seconds: false })}`)),
      description && h('p.fst-italic', description),
    ];
    mount(doc, parts);
    if (!blocks.length) doc.appendChild(emptyState({ icon: 'file-earmark', title: 'This report is empty', message: 'Click Edit to add text and live charts.' }));
    for (const b of blocks) doc.appendChild(renderBlock(b));
    return doc;
  }

  function renderBlock(b) {
    switch (b.kind) {
      // The report title is the page's h1, so block headings render one level down (h2–h3).
      case 'heading': return h(`h${Math.min(3, (b.level || 1) + 1)}`, b.text);
      case 'paragraph': return b.text ? h('p', b.text) : h('div', { style: { height: '8px' } });
      case 'quote': return h('blockquote', b.text);
      case 'divider': return h('hr');
      case 'list': return h(b.ordered ? 'ol' : 'ul', String(b.text || '').split('\n').filter(Boolean).map((l) => h('li', l)));
      case 'code': return codeBlock(b.text || '', { lang: b.language || 'python' });
      case 'markdown': {
        const el = h('div.af-markdown');
        loadMarkdown().then((md) => { el.innerHTML = md(b.text || ''); }).catch(() => { el.textContent = b.text || ''; });
        return el;
      }
      case 'chart': return chartBlock(b.chart);
      case 'wandb-panels': {
        const info = panelGridInfo(b.raw);
        const lines = info.charts.filter((c) => c.metrics?.length);
        if (!lines.length) return h('div.ma-doc-note', icon('info-circle', 'me-1'), 'This report section has W&B panels Model Studio does not render yet — open the report in W&B to see them.');
        const proj = info.project || { entityName: entity, name: project };
        return chartBlock({ metrics: [...new Set(lines.flatMap((c) => c.metrics))].slice(0, 8), limit: 20, project: proj, title: lines[0].title });
      }
      default: return b.text ? h('p', b.text) : h('div');
    }
  }

  function chartBlock(chart = {}) {
    const metrics = chart.metrics || [];
    const proj = chart.project || { entityName: entity, name: project };
    const wrap = h('div.ma-doc-panels');
    if (!metrics.length) return h('div.ma-doc-note', 'Chart block with no metrics selected.');
    lazy(wrap, async () => {
      try {
        const filters = JSON.stringify(chart.state ? { state: chart.state } : {});
        const page = await loadRunsPage(proj.entityName, proj.name, { first: Math.min(chart.limit || 10, 50), filters, order: '-createdAt' });
        const names = page.runs.map((r) => r.name);
        const hist = await loadHistories(proj.entityName, proj.name, names, metrics, { samples: 300 });
        mount(wrap, metrics.map((m) => {
          const el = h('div');
          const panel = h('div.ma-doc-panel', h('div.ma-doc-panel-title', m), el);
          lineChart(el, page.runs.map((r, i) => ({ name: displayName(r), runName: r.name, color: runColor(r, i), points: hist.get(r.name)?.get(m) || [] })),
            { height: 230, xLabel: 'step', smoothing: { type: 'ema', weight: 0.5 }, onClick: ({ runName }) => runName && navigate(urls.run(proj.entityName, proj.name, runName)) });
          return panel;
        }), h('div.small.text-muted', { style: { gridColumn: '1 / -1' } }, `Live: ${page.runs.length} most recent ${chart.state ? `${chart.state} ` : ''}runs of ${proj.name}`));
      } catch (err) { mount(wrap, errorState(err)); }
    });
    mount(wrap, skeleton(4, { height: 30 }));
    return wrap;
  }

  // ---------------------------------------------------------------- editor

  function editor() {
    const list = h('div.ma-block-editor');
    const previewEl = h('div');
    let preview = false;
    const titleInput = h('input.form-control.form-control-lg', { value: title, placeholder: 'Report title', 'aria-label': 'Title', oninput: (e) => { title = e.target.value; markDirty(); } });
    const descInput = h('input.form-control.form-control-sm', { value: description, placeholder: 'One-line summary (shown on the reports list)', 'aria-label': 'Description', oninput: (e) => { description = e.target.value; markDirty(); } });
    const toggle = h('div.btn-group.btn-group-sm',
      h('button.btn.btn-outline-secondary.active', { type: 'button', onclick: (e) => { preview = false; e.currentTarget.classList.add('active'); e.currentTarget.nextSibling.classList.remove('active'); redraw(); } }, icon('pencil', 'me-1'), 'Write'),
      h('button.btn.btn-outline-secondary', { type: 'button', onclick: (e) => { preview = true; e.currentTarget.classList.add('active'); e.currentTarget.previousSibling.classList.remove('active'); redraw(); } }, icon('eye', 'me-1'), 'Preview'));
    const addBar = h('div.d-flex.flex-wrap.gap-2', BLOCK_KINDS.map((k) => h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => {
      blocks.push(k.kind === 'chart' ? { kind: 'chart', chart: { metrics: [], limit: 10 } } : { kind: k.kind, text: '', level: 2 });
      markDirty();
      redraw();
      list.lastElementChild?.querySelector('textarea, input')?.focus();
    } }, icon(k.icon, 'me-1'), k.label)));

    function redraw() {
      if (preview) { mount(previewEl, viewer()); list.style.display = 'none'; previewEl.style.display = ''; return; }
      list.style.display = '';
      previewEl.style.display = 'none';
      mount(list, blocks.map((b, i) => blockEditor(b, i, redraw)), !blocks.length && emptyState({ icon: 'plus-square', title: 'Add your first block below' }));
    }
    redraw();
    return h('div.d-flex.flex-column.gap-3',
      h('div.af-card.p-3.d-flex.flex-column.gap-2', titleInput, descInput, h('div.d-flex.justify-content-between.align-items-center.flex-wrap.gap-2', toggle, h('span.small.text-muted', 'Ctrl/⌘+S saves'))),
      list, previewEl,
      h('div.af-card.p-3', h('div.small.fw-semibold.text-muted.mb-2', 'Add a block'), addBar));
  }

  function blockEditor(b, i, redraw) {
    const kindSel = h('select.form-select.form-select-sm', { 'aria-label': 'Block type', onchange: (e) => {
      b.kind = e.target.value;
      if (b.kind === 'chart' && !b.chart) b.chart = { metrics: [], limit: 10 };
      markDirty();
      redraw();
    } }, BLOCK_KINDS.map((k) => h('option', { value: k.kind, selected: k.kind === b.kind }, k.label)),
    ['wandb-panels', 'unknown'].includes(b.kind) && h('option', { value: b.kind, selected: true }, 'W&B block'));
    let input;
    if (b.kind === 'divider') input = h('div.small.text-muted.pt-1', '— horizontal rule —');
    else if (b.kind === 'chart') input = chartEditor(b);
    else if (['wandb-panels', 'unknown'].includes(b.kind)) input = h('div.small.text-muted.pt-1', 'Created in W&B — kept as-is when you save.');
    else if (b.kind === 'heading') {
      input = h('div.d-flex.gap-2',
        h('select.form-select.form-select-sm', { style: { width: '80px' }, 'aria-label': 'Level', onchange: (e) => { b.level = Number(e.target.value); markDirty(); } }, [1, 2, 3].map((l) => h('option', { value: l, selected: (b.level || 1) === l }, `H${l}`))),
        h('input.form-control.form-control-sm.fw-semibold', { value: b.text || '', placeholder: 'Heading', oninput: (e) => { b.text = e.target.value; markDirty(); } }));
    } else {
      input = h('textarea.form-control.form-control-sm', { rows: b.kind === 'code' || b.kind === 'markdown' ? 6 : 3, class: (b.kind === 'code' || b.kind === 'markdown') && 'af-mono',
        placeholder: { paragraph: 'Write…', markdown: '**Markdown** with lists, links, tables…', quote: 'A note or decision to highlight', list: 'One item per line', code: 'code' }[b.kind] || '',
        oninput: (e) => { b.text = e.target.value; markDirty(); } }, b.text || '');
    }
    const move = (d) => { const j = i + d; if (j < 0 || j >= blocks.length) return; [blocks[i], blocks[j]] = [blocks[j], blocks[i]]; markDirty(); redraw(); };
    return h('div.ma-block', kindSel, input,
      h('div.d-flex.flex-column.gap-1',
        h('button.af-icon-btn', { title: 'Move up', disabled: i === 0, onclick: () => move(-1) }, icon('arrow-up')),
        h('button.af-icon-btn', { title: 'Move down', disabled: i === blocks.length - 1, onclick: () => move(1) }, icon('arrow-down')),
        h('button.af-icon-btn', { title: 'Remove block', onclick: () => { blocks.splice(i, 1); markDirty(); redraw(); } }, icon('trash'))));
  }

  function chartEditor(b) {
    const chart = b.chart;
    const picks = h('div.ma-metric-picks', h('span.small.text-muted', 'Loading metrics…'));
    const drawPicks = (keys) => mount(picks, keys.length ? keys.map((k) => h('button.ma-metric-pick', {
      type: 'button', class: chart.metrics.includes(k) && 'on',
      onclick: (e) => {
        chart.metrics = chart.metrics.includes(k) ? chart.metrics.filter((x) => x !== k) : [...chart.metrics, k];
        e.currentTarget.classList.toggle('on');
        markDirty();
      },
    }, k)) : h('span.small.text-muted', 'No numeric metrics logged in this project yet.'));
    const ensureKeys = async () => {
      if (!metricKeys) {
        const page = await loadRunsPage(entity, project, { first: 30 });
        metricKeys = (await loadMetricKeys(entity, project, page.runs.map((r) => r.name)))
          .filter((k) => k.numeric && !k.key.startsWith('_') && !k.key.startsWith('system/')).map((k) => k.key).sort();
      }
      return metricKeys;
    };
    ensureKeys().then(drawPicks).catch((err) => mount(picks, h('span.text-danger.small', err.message)));
    return h('div.d-flex.flex-column.gap-2', picks,
      h('div.d-flex.gap-2.flex-wrap.align-items-center.small',
        h('label.d-inline-flex.align-items-center.gap-1', 'Latest', h('input.form-control.form-control-sm', { type: 'number', min: 1, max: 50, value: chart.limit || 10, style: { width: '70px' }, oninput: debounce((e) => { chart.limit = Number(e.target.value) || 10; markDirty(); }, 200) }), 'runs'),
        h('label.d-inline-flex.align-items-center.gap-1', 'State', h('select.form-select.form-select-sm', { style: { width: 'auto' }, onchange: (e) => { chart.state = e.target.value || undefined; markDirty(); } },
          ['', 'finished', 'running', 'crashed'].map((s) => h('option', { value: s, selected: (chart.state || '') === s }, s || 'any'))))));
  }

  // ---------------------------------------------------------------- actions

  async function save() {
    if (!title.trim()) { toast('Give the report a title first.', 'warning'); return; }
    saveBtn && (saveBtn.disabled = true);
    try {
      const spec = blocksToSpec(blocks, report?.specObj || emptySpec());
      const saved = await saveReport({ id: report?.id, entity, project, displayName: title.trim(), description, spec });
      dirty = false;
      toast(isNew ? 'Report created' : 'Report saved', 'success', { timeout: 2000 });
      navigate(urls.report(entity, project, saved.id), { replace: true });
    } catch (err) {
      toastError(err, 'Could not save the report');
      saveBtn && (saveBtn.disabled = false);
    }
  }

  async function cancelEdit() {
    if (dirty && !(await confirmDialog({ title: 'Discard changes?', message: 'Your edits to this report will be lost.', confirmLabel: 'Discard', danger: true }))) return;
    dirty = false;
    if (isNew) { navigate(urls.reports(entity, project)); return; }
    blocks = specToBlocks(report.specObj);
    title = report.displayName || '';
    description = report.description || '';
    editing = false;
    ctx.setQuery({ edit: null });
    draw();
  }

  async function remove() {
    const ok = await confirmDialog({ title: `Delete “${title || 'Untitled report'}”?`, danger: true, confirmLabel: 'Delete report', message: 'Anyone with the link loses access.' });
    if (!ok) return;
    try {
      await deleteReport(report.id);
      toast('Report deleted', 'success');
      navigate(urls.reports(entity, project));
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  draw();
}
