// Runs table: virtualized rows, server-side filter/sort, any config/summary columns,
// client-side grouping with aggregates, bulk tag/delete/move/compare, inline rename & notes,
// CSV export, and auto-refresh that pauses while you scroll or edit.

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import {
  emptyState, errorState, toast, toastError, formModal, openMenu, refreshControl, drawer, timeAgo, spinner,
} from '../../../ui/js/ui.js';
import { fmtNumber } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { urls } from '../core/urls.js';
import { loadAllRuns } from '../core/data.js';
import { buildFilter, orderString, runColor, displayName, fmtMetric, RUN_STATES } from '../core/wb.js';
import { mutations } from '../core/data.js';
import { downloadText } from '../core/charts.js';
import {
  projectHeader, META_COLUMNS, runValue, columnLabel, cellValue, catalog, groupRuns, toCsv,
  inlineText, bulkTagDialog, bulkDelete, bulkMoveDialog, notesDialog, runMenu, lowerIsBetter,
} from './workspace/common.js';

const ROW_H = 40;
const SERVER_SORTABLE = /^(createdAt|heartbeatAt|config\.|summary\.)/;
const OPS = ['=', '!=', '>', '>=', '<', '<=', 'contains', 'in', 'exists'];

export default async function render(ctx) {
  const { entity, project } = ctx.params;
  const { el, signal } = ctx;
  const prefKey = `ms:table:${entity}/${project}`;
  const saved = prefs.get(prefKey, {});
  const state = {
    columns: saved.columns || null,
    sort: saved.sort || { key: 'createdAt', dir: 'desc' },
    conditions: saved.conditions || [],
    groupBy: ctx.query.get('group') ?? saved.groupBy ?? '',
    search: ctx.query.get('q') || '',
  };
  let runs = [];
  let total = 0;
  let selected = new Set();
  let lastLoaded = null;
  let lastInteraction = 0;
  let collapsedGroups = new Set();
  let loading = false;

  const headerEl = h('div');
  const toolbarEl = h('div.mp-tb-toolbar');
  const filtersEl = h('div.mp-tb-filters');
  const bulkEl = h('div');
  const tableEl = h('div.mp-vt-wrap', h('div.mp-loading', spinner(), ' Loading runs…'));
  const statusEl = h('span.mp-tb-status');
  const refresh = refreshControl(() => load({ quiet: true }), { key: 'ms-table', initial: 30 });
  ctx.cleanup(refresh.stop);
  mount(el, headerEl, h('section.af-card.mp-tb-card', toolbarEl, filtersEl, tableEl), bulkEl);

  const persist = debounce(() => {
    prefs.set(prefKey, { columns: state.columns, sort: state.sort, conditions: state.conditions, groupBy: state.groupBy });
    ctx.setQuery({ group: state.groupBy || null, q: state.search || null });
  }, 200);
  const touch = () => { lastInteraction = Date.now(); };
  ['wheel', 'pointerdown', 'keydown'].forEach((ev) => el.addEventListener(ev, touch, { passive: true }));

  // ---------------------------------------------------------------- data
  async function load({ quiet = false } = {}) {
    if (quiet) {
      const editing = el.querySelector('.mp-inline input, .mp-inline textarea');
      if (editing || Date.now() - lastInteraction < 8000) { mount(statusEl, icon('pause-circle', 'me-1'), 'auto-refresh paused while you work'); return; }
    }
    if (loading) return;
    loading = true;
    const serverSort = SERVER_SORTABLE.test(state.sort.key);
    try {
      const res = await loadAllRuns(entity, project, {
        filters: buildFilter(state.conditions, { search: state.search }),
        order: serverSort ? orderString(state.sort.key, state.sort.dir) : '-createdAt',
        max: 2000, signal,
        onProgress: (n, t) => { if (!quiet) mount(statusEl, spinner('sm'), ` loading ${fmtNumber(n)} of ${fmtNumber(t)}…`); },
      });
      if (!ctx.active) return;
      runs = res.runs;
      total = res.total;
      lastLoaded = new Date();
      const names = new Set(runs.map((r) => r.name));
      selected = new Set([...selected].filter((n) => names.has(n)));
      if (!state.columns) state.columns = defaultColumns();
      renderAll();
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (quiet) toastError(err, 'Refresh failed');
      else mount(tableEl, errorState(err, () => load()));
    } finally { loading = false; }
  }

  function defaultColumns() {
    const cat = catalog(runs);
    const pref = cat.summary.filter((s) => s.numeric && /val|acc|loss|f1|auc|best/i.test(s.name)).slice(0, 4).map((s) => s.key);
    const cfg = cat.config.filter((c) => c.values.length > 1).slice(0, 4).map((c) => c.key);
    return ['state', 'createdAt', 'runtime', 'group', 'tags', ...cfg, ...pref];
  }

  // ---------------------------------------------------------------- render
  function renderAll() {
    mount(headerEl, projectHeader({
      entity, project, active: 'table', icon: 'table',
      subtitle: h('span', `${fmtNumber(total)} run${total === 1 ? '' : 's'}${state.conditions.length || state.search ? ' match' : ''} · `, statusEl),
      actions: [refresh.el],
    }));
    mount(statusEl, 'updated ', lastLoaded ? timeAgo(lastLoaded.toISOString()) : '—');
    renderToolbar();
    renderFilters();
    renderTable();
    renderBulk();
  }

  function renderToolbar() {
    const cat = catalog(runs);
    const groupOpts = [['', 'No grouping'], ['group', 'Group'], ['jobType', 'Job type'], ['state', 'State'], ['sweepName', 'Sweep'], ['tags', 'Tags'],
      ...cat.config.filter((c) => c.values.length <= 25).map((c) => [c.key, `config: ${c.name}`])];
    mount(toolbarEl,
      h('div.af-search.mp-tb-search', icon('search', 'af-search-icon'), h('input.form-control.form-control-sm', {
        type: 'search', placeholder: 'Search names and tags (server-side)…', value: state.search, dataset: { searchShortcut: '' },
        oninput: debounce((e) => { state.search = e.target.value.trim(); persist(); load(); }, 350),
      })),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => conditionDialog() }, icon('funnel', 'me-1'), 'Add filter'),
      h('label.mp-tb-group', icon('collection'), h('select.form-select.form-select-sm', {
        'aria-label': 'Group rows by',
        onchange: (e) => { state.groupBy = e.target.value; collapsedGroups = new Set(); persist(); renderTable(); },
      }, groupOpts.map(([v, l]) => h('option', { value: v, selected: v === state.groupBy }, l)),
      state.groupBy && !groupOpts.some(([v]) => v === state.groupBy) && h('option', { value: state.groupBy, selected: true }, state.groupBy))),
      h('div.ms-auto.d-flex.gap-2',
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: columnsDrawer }, icon('layout-three-columns', 'me-1'), `Columns (${state.columns.length})`),
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Download the current view as CSV', onclick: () => exportCsv(viewRuns()) }, icon('filetype-csv', 'me-1'), 'CSV')));
  }

  function renderFilters() {
    if (!state.conditions.length) { mount(filtersEl); return; }
    mount(filtersEl,
      h('span.mp-tb-flabel', icon('funnel-fill', 'me-1'), 'Filters'),
      state.conditions.map((c, i) => h('span.mp-fchip',
        h('button.mp-fchip-body', { type: 'button', onclick: () => conditionDialog(i) }, h('b', columnLabel(c.field)), ` ${c.op} `, c.op !== 'exists' && h('code', String(c.value))),
        h('button.mp-fchip-x', { type: 'button', title: 'Remove filter', onclick: () => { state.conditions.splice(i, 1); persist(); load(); } }, icon('x')))),
      h('button.af-link-btn', { onclick: () => { state.conditions = []; persist(); load(); } }, 'Clear all'));
  }

  function conditionDialog(index) {
    const cat = catalog(runs);
    const existing = index != null ? state.conditions[index] : null;
    const fields = [
      { value: 'state', label: 'State' }, { value: 'displayName', label: 'Name' }, { value: 'tags', label: 'Tags' },
      { value: 'group', label: 'Group' }, { value: 'jobType', label: 'Job type' }, { value: 'sweep', label: 'Sweep' },
      ...cat.config.map((c) => ({ value: c.key, label: `config: ${c.name}` })),
      ...cat.summary.map((s) => ({ value: s.key, label: `summary: ${s.name}` })),
    ];
    formModal({
      title: existing ? 'Edit filter' : 'Add filter', icon: 'funnel',
      subtitle: 'Filters run on the W&B server, so they scale to any number of runs.',
      fields: [
        { name: 'field', label: 'Field', type: 'select', options: fields, value: existing?.field || 'state' },
        { name: 'op', label: 'Condition', type: 'select', options: OPS, value: existing?.op || '=' },
        { name: 'value', label: 'Value', value: existing?.value ?? '', placeholder: 'e.g. finished, 0.001, adam — for "in": a, b, c',
          help: `States: ${Object.keys(RUN_STATES).join(', ')}` },
      ],
      submitLabel: existing ? 'Update filter' : 'Add filter',
      onSubmit: (v, close) => {
        if (v.op !== 'exists' && v.value === '') throw new Error('Enter a value.');
        if (existing) state.conditions[index] = v; else state.conditions.push(v);
        persist();
        close();
        load();
      },
    });
  }

  function columnsDrawer() {
    const cat = catalog(runs);
    const all = [
      ...META_COLUMNS.map((c) => ({ key: c.key, label: c.label, group: 'Run' })),
      ...cat.config.map((c) => ({ key: c.key, label: c.name, group: 'Config' })),
      ...cat.summary.map((s) => ({ key: s.key, label: s.name, group: 'Summary' })),
    ];
    const body = h('div.mp-cols');
    const d = drawer({ title: 'Columns', subtitle: 'Pick and order columns. Saved per project in this browser.', width: 460, body });
    let filter = '';
    const draw = () => {
      const chosen = state.columns;
      mount(body,
        h('div.af-search.mb-2', icon('search', 'af-search-icon'), h('input.form-control.form-control-sm', {
          type: 'search', placeholder: 'Find a column…', value: filter, autofocus: true,
          oninput: debounce((e) => { filter = e.target.value.toLowerCase(); draw(); body.querySelector('input').focus(); }, 120),
        })),
        h('div.mp-cols-chosen', h('div.mp-cols-h', 'Shown (drag order with arrows)'),
          chosen.map((k, i) => h('div.mp-col-row.on',
            h('span.mp-col-name', columnLabel(k), h('span.mp-col-kind', k.split('.')[0] === k ? 'run' : k.split('.')[0])),
            h('button.af-icon-btn', { disabled: i === 0, title: 'Move up', onclick: () => { [chosen[i - 1], chosen[i]] = [chosen[i], chosen[i - 1]]; persist(); draw(); renderTable(); } }, icon('arrow-up')),
            h('button.af-icon-btn', { disabled: i === chosen.length - 1, title: 'Move down', onclick: () => { [chosen[i + 1], chosen[i]] = [chosen[i], chosen[i + 1]]; persist(); draw(); renderTable(); } }, icon('arrow-down')),
            h('button.af-icon-btn', { title: 'Hide', onclick: () => { state.columns = chosen.filter((x) => x !== k); persist(); draw(); renderTable(); renderToolbar(); } }, icon('eye-slash'))))),
        ['Run', 'Config', 'Summary'].map((g) => {
          const items = all.filter((c) => c.group === g && !chosen.includes(c.key) && (!filter || c.label.toLowerCase().includes(filter)));
          return items.length ? h('div.mp-cols-group', h('div.mp-cols-h', `${g} (${items.length})`),
            items.map((c) => h('button.mp-col-row', { type: 'button', onclick: () => { state.columns = [...chosen, c.key]; persist(); draw(); renderTable(); renderToolbar(); } },
              icon('plus-circle', 'text-primary'), h('span.mp-col-name', c.label)))) : null;
        }),
        h('div.mt-3.d-flex.gap-2',
          h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => { state.columns = defaultColumns(); persist(); draw(); renderTable(); renderToolbar(); } }, 'Reset to defaults'),
          h('button.btn.btn-sm.btn-primary.ms-auto', { onclick: () => d.close() }, 'Done')));
    };
    draw();
  }

  /** Runs in display order: client-side sort for non-server-sortable keys. */
  function viewRuns() {
    if (SERVER_SORTABLE.test(state.sort.key)) return runs;
    const dir = state.sort.dir === 'asc' ? 1 : -1;
    return [...runs].sort((a, b) => cmp(runValue(a, state.sort.key), runValue(b, state.sort.key)) * dir);
  }

  /** Flattened rows: group headers + runs. */
  function rowModel() {
    const list = viewRuns();
    if (!state.groupBy) return list.map((r) => ({ kind: 'run', run: r }));
    const out = [];
    for (const g of groupRuns(list, state.groupBy)) {
      out.push({ kind: 'group', key: g.key, runs: g.runs });
      if (!collapsedGroups.has(g.key)) g.runs.forEach((r) => out.push({ kind: 'run', run: r, grouped: true }));
    }
    return out;
  }

  function colWidths() {
    return ['36px', 'minmax(220px, 2.2fr)', ...state.columns.map((k) => (k === 'tags' || k === 'notes' ? 'minmax(150px, 1.4fr)' : k === 'state' ? '110px' : 'minmax(110px, 1fr)')), '44px'].join(' ');
  }

  function renderTable() {
    if (!runs.length) {
      mount(tableEl, emptyState({
        icon: state.conditions.length || state.search ? 'funnel' : 'table',
        title: state.conditions.length || state.search ? 'No runs match these filters' : 'No runs yet',
        message: state.conditions.length || state.search ? 'Remove a filter chip above.' : 'Runs logged with the wandb SDK show up here.',
      }));
      return;
    }
    const rows = rowModel();
    const grid = colWidths();
    const allNames = runs.map((r) => r.name);
    const headCheck = h('input.form-check-input', {
      type: 'checkbox', 'aria-label': 'Select all', checked: allNames.length && allNames.every((n) => selected.has(n)),
      onclick: (e) => { allNames.forEach((n) => (e.target.checked ? selected.add(n) : selected.delete(n))); paint(true); renderBulk(); },
    });
    const head = h('div.mp-vt-head', { style: { gridTemplateColumns: grid } },
      h('div.mp-vt-cell', headCheck),
      ['name', ...state.columns].map((k) => {
        const sorted = state.sort.key === k;
        return h('button.mp-vt-cell.mp-vt-th', {
          type: 'button', class: sorted && 'sorted', title: SERVER_SORTABLE.test(k) ? 'Sorted by the W&B server' : 'Sorted in your browser',
          onclick: () => {
            state.sort = { key: k, dir: sorted && state.sort.dir === 'desc' ? 'asc' : 'desc' };
            persist();
            if (SERVER_SORTABLE.test(k)) load(); else paint(true);
            head.querySelectorAll('.mp-vt-th').forEach((th) => th.classList.remove('sorted'));
          },
        }, h('span', columnLabel(k)), icon(sorted ? (state.sort.dir === 'asc' ? 'sort-up' : 'sort-down') : 'chevron-expand', 'mp-sort'));
      }),
      h('div.mp-vt-cell'));
    const spacer = h('div.mp-vt-spacer');
    const layer = h('div.mp-vt-layer');
    const scroller = h('div.mp-vt-scroll', { role: 'grid', 'aria-rowcount': rows.length }, spacer, layer);
    mount(tableEl, h('div.mp-vt', head, scroller), h('div.mp-vt-foot', `${fmtNumber(runs.length)} of ${fmtNumber(total)} runs loaded`,
      total > runs.length && h('span.text-muted', ' · first 2,000 shown — add a filter to narrow down')));

    let model = rows;
    let lastRange = '';
    function paint(force = false) {
      if (force) { model = rowModel(); spacer.style.height = `${model.length * ROW_H}px`; }
      const top = scroller.scrollTop;
      const vh = scroller.clientHeight || 600;
      const start = Math.max(0, Math.floor(top / ROW_H) - 8);
      const end = Math.min(model.length, Math.ceil((top + vh) / ROW_H) + 8);
      const range = `${start}:${end}`;
      if (!force && range === lastRange) return;
      lastRange = range;
      mount(layer, model.slice(start, end).map((row, i) => (row.kind === 'group' ? groupRow(row, start + i, grid) : runRow(row, start + i, grid))));
      headCheck.checked = allNames.length > 0 && allNames.every((n) => selected.has(n));
    }
    spacer.style.height = `${rows.length * ROW_H}px`;
    scroller.addEventListener('scroll', () => { touch(); requestAnimationFrame(() => paint()); }, { passive: true });
    tableEl.paint = paint;
    // Rows share the header's full width so horizontal scrolling keeps columns aligned.
    const syncWidth = () => { scroller.style.width = `${head.scrollWidth}px`; };
    requestAnimationFrame(syncWidth);
    const ro = new ResizeObserver(syncWidth);
    ro.observe(head);
    ctx.cleanup(() => ro.disconnect());
    paint(true);
  }

  function groupRow(row, index, grid) {
    const collapsed = collapsedGroups.has(row.key);
    const aggs = state.columns.map((k) => {
      const vals = row.runs.map((r) => runValue(r, k)).filter((v) => typeof v === 'number');
      if (!vals.length) return h('div.mp-vt-cell');
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const best = lowerIsBetter(k) ? Math.min(...vals) : Math.max(...vals);
      return h('div.mp-vt-cell.ms-metric', { title: `mean ${fmtMetric(mean)} · best ${fmtMetric(best)} · n=${vals.length}` }, h('span', `μ ${fmtMetric(mean)}`), h('span.mp-agg-best', ` · best ${fmtMetric(best)}`));
    });
    return h('div.mp-vt-row.mp-vt-group', {
      style: { transform: `translateY(${index * ROW_H}px)`, gridTemplateColumns: grid },
      onclick: () => { collapsed ? collapsedGroups.delete(row.key) : collapsedGroups.add(row.key); tableEl.paint(true); },
    },
    h('div.mp-vt-cell', icon(collapsed ? 'chevron-right' : 'chevron-down')),
    h('div.mp-vt-cell', h('b', row.key), h('span.ms-count.ms-2', row.runs.length)),
    aggs, h('div.mp-vt-cell'));
  }

  function runRow(row, index, grid) {
    const r = row.run;
    const on = selected.has(r.name);
    const color = runColor(r, runs.indexOf(r));
    return h('div.mp-vt-row', {
      class: [on && 'selected', row.grouped && 'grouped', r.state === 'running' && 'running'],
      style: { transform: `translateY(${index * ROW_H}px)`, gridTemplateColumns: grid },
      role: 'row',
      ondblclick: (e) => { if (!e.target.closest('input,button,a,.mp-inline')) navigate(urls.run(entity, project, r.name)); },
    },
    h('div.mp-vt-cell', h('input.form-check-input', {
      type: 'checkbox', checked: on, 'aria-label': `Select ${displayName(r)}`,
      onclick: (e) => { e.target.checked ? selected.add(r.name) : selected.delete(r.name); e.target.closest('.mp-vt-row').classList.toggle('selected', e.target.checked); renderBulk(); },
    })),
    h('div.mp-vt-cell.mp-name-cell',
      h('span.ms-swatch', { style: { background: color } }),
      h('div.mp-name-stack',
        inlineText(displayName(r), async (v) => { await mutations.updateRun(r.id, { displayName: v }); r.displayName = v; toast('Renamed', 'success', { timeout: 1500 }); }, { className: 'ms-run-name' }),
        h('a.ms-run-id', { href: urls.run(entity, project, r.name) }, r.name)),
      h('button.af-icon-btn.mp-note-btn', { type: 'button', class: r.notes && 'has', title: r.notes ? `Notes: ${r.notes}` : 'Add notes', onclick: () => notesDialog(r, () => tableEl.paint(true)) }, icon(r.notes ? 'journal-text' : 'journal-plus'))),
    state.columns.map((k) => h('div.mp-vt-cell', { class: typeof runValue(r, k) === 'number' && 'num' }, cellValue(r, k))),
    h('div.mp-vt-cell', h('button.af-icon-btn', { type: 'button', title: 'Run actions', onclick: (e) => runMenu(e.currentTarget, r, { entity, project, onChanged: () => load() }) }, icon('three-dots'))));
  }

  function renderBulk() {
    const chosen = runs.filter((r) => selected.has(r.name));
    if (!chosen.length) { mount(bulkEl); return; }
    mount(bulkEl, h('div.af-selection-bar.mp-bulk',
      h('span.fw-semibold', `${chosen.length} selected`),
      h('div.d-flex.gap-2.flex-wrap',
        h('a.btn.btn-sm.btn-light', { href: urls.compare(entity, project, chosen.slice(0, 50).map((r) => r.name)) }, icon('layout-split', 'me-1'), 'Compare'),
        h('button.btn.btn-sm.btn-light', { onclick: () => bulkTagDialog(entity, project, chosen, () => load()) }, icon('tags', 'me-1'), 'Tags'),
        h('button.btn.btn-sm.btn-light', { onclick: () => bulkMoveDialog(entity, project, chosen, () => { selected.clear(); load(); }) }, icon('box-arrow-right', 'me-1'), 'Move'),
        h('button.btn.btn-sm.btn-light', { onclick: () => exportCsv(chosen) }, icon('filetype-csv', 'me-1'), 'Export'),
        h('button.btn.btn-sm.btn-danger', { onclick: async () => { if (await bulkDelete(entity, project, chosen)) { selected.clear(); load(); } } }, icon('trash', 'me-1'), 'Delete')),
      h('button.af-link-btn', { onclick: () => { selected.clear(); renderBulk(); tableEl.paint?.(true); } }, 'Clear')));
  }

  function exportCsv(list) {
    const cols = [{ label: 'name', value: (r) => r.name }, { label: 'display_name', value: (r) => displayName(r) },
      ...state.columns.map((k) => ({ label: k, value: (r) => runValue(r, k) }))];
    downloadText(toCsv(cols, list), `${project}-runs.csv`);
    toast(`Exported ${list.length} runs`, 'success', { timeout: 1800 });
  }

  void openMenu;
  await load();
}

function cmp(a, b) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}
