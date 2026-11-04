// Data table grid: inline cell editing, filter builder (and/or), sort, search, paging,
// add/delete rows, bulk update with dry-run preview, upsert, column management,
// BOM-safe CSV/JSON import & export.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { pageHeader, toast, toastError, modal, spinner, menuButton, confirmDialog, jsonEditor, timeAgo } from '../../../ui/js/ui.js';
import { n8n, enc, saveText } from '../core/api.js';
import { urls } from '../core/urls.js';
import { csvToRecords, toCSV, coerce, inferType, RESERVED_COLUMNS } from '../core/csv.js';
import { emptyHint, promptText, pickFile, readFileText, inlineEdit, licenseGate } from '../core/widgets.js';
import { COLUMN_TYPES } from './datatables.js';

const CONDITIONS = [['eq', '='], ['neq', '≠'], ['like', 'like'], ['ilike', 'ilike (case-insensitive)'], ['gt', '>'], ['gte', '≥'], ['lt', '<'], ['lte', '≤']];
const SYSTEM = ['id', 'createdAt', 'updatedAt'];
const PAGE = 100;

export default async function render(ctx) {
  const id = ctx.params.tableId;
  let table;
  try { table = await n8n.get(`/data-tables/${enc(id)}`, undefined, { signal: ctx.signal }); } catch (err) {
    if (err.license) { mount(ctx.el, licenseGate(err)); return; }
    throw err;
  }
  ctx.setTitle(table.name);
  const st = { rows: [], cursor: null, search: '', filters: [], combinator: 'and', sort: null, selected: new Set(), loading: false };

  const gridEl = h('div.n8s-grid-wrap');
  const filterEl = h('div.n8s-dtfilters');
  const infoEl = h('span.small.text-muted');
  const search = h('input.form-control.form-control-sm', { type: 'search', placeholder: 'Search all columns…', style: 'max-width:260px', dataset: { searchShortcut: '' } });
  search.addEventListener('input', debounce(() => { st.search = search.value.trim(); reload(); }, 300));

  mount(ctx.el,
    pageHeader({
      title: inlineEdit(table.name, async (v) => { table = await n8n.patch(`/data-tables/${enc(id)}`, { name: v }); }),
      icon: 'table', crumbs: [{ label: 'Data tables', href: urls.dataTables() }],
      subtitle: h('span', `${table.columns.length} columns · updated `, timeAgo(table.updatedAt), ' · id ', h('code', table.id)),
      actions: [
        h('button.btn.btn-sm.btn-primary', { onclick: () => addRow() }, icon('plus-lg', 'me-1'), 'Row'),
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => addColumn() }, icon('layout-three-columns', 'me-1'), 'Column'),
        menuButton('Import / export', [
          { label: 'Import CSV…', icon: 'filetype-csv', onClick: () => importCsv() },
          { label: 'Import JSON…', icon: 'filetype-json', onClick: () => importJson() },
          { divider: true },
          { label: 'Export CSV', icon: 'download', onClick: () => exportAll('csv') },
          { label: 'Export CSV (Excel, with BOM)', icon: 'file-earmark-spreadsheet', onClick: () => exportAll('csv-bom') },
          { label: 'Export JSON', icon: 'braces', onClick: () => exportAll('json') },
        ], { icon: 'arrow-down-up', variant: 'outline-secondary' }),
        menuButton('', [
          { label: 'Bulk update…', icon: 'pencil-square', onClick: () => bulkUpdate() },
          { label: 'Upsert row…', icon: 'arrow-repeat', onClick: () => upsert() },
          { label: 'Delete matching rows…', icon: 'funnel', onClick: () => deleteMatching() },
          { divider: true },
          { label: 'Clear all rows…', icon: 'eraser', danger: true, onClick: () => clearAll() },
        ]),
      ],
    }),
    h('div.n8s-dt-toolbar', search, h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => { st.filters.push({ columnName: userCols()[0]?.name || 'id', condition: 'eq', value: '' }); drawFilters(); } }, icon('funnel', 'me-1'), 'Filter'), filterEl, h('div.flex-grow-1'), infoEl),
    gridEl);

  const userCols = () => [...(table.columns || [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  const filterParam = () => {
    const valid = st.filters.filter((f) => f.columnName && f.value !== '');
    return valid.length ? JSON.stringify({ type: st.combinator, filters: valid.map((f) => ({ ...f, value: coerceFor(f.columnName, f.value) })) }) : undefined;
  };
  const coerceFor = (col, v) => {
    const c = userCols().find((x) => x.name === col);
    if (col === 'id') return Number(v);
    return c ? coerce(v, c.type) ?? v : v;
  };

  function drawFilters() {
    mount(filterEl, st.filters.map((f, i) => h('div.n8s-dtfilter',
      i > 0 && h('select.form-select.form-select-sm.n8s-comb', { onchange: (e) => { st.combinator = e.target.value; reload(); } }, ['and', 'or'].map((x) => h('option', { selected: st.combinator === x }, x))),
      h('select.form-select.form-select-sm', { onchange: (e) => { f.columnName = e.target.value; reload(); } }, [...SYSTEM, ...userCols().map((c) => c.name)].map((c) => h('option', { selected: c === f.columnName }, c))),
      h('select.form-select.form-select-sm', { onchange: (e) => { f.condition = e.target.value; reload(); } }, CONDITIONS.map(([v, l]) => h('option', { value: v, selected: v === f.condition }, l))),
      h('input.form-control.form-control-sm', { value: f.value, placeholder: f.condition.includes('like') ? '%text%' : 'value', oninput: debounce((e) => { f.value = e.target.value; reload(); }, 400) }),
      h('button.af-icon-btn', { onclick: () => { st.filters.splice(i, 1); drawFilters(); reload(); } }, icon('x-lg')))));
  }

  async function fetchPage(reset) {
    st.loading = true;
    const page = await n8n.get(`/data-tables/${enc(id)}/rows`, {
      limit: PAGE, cursor: reset ? undefined : st.cursor || undefined, search: st.search || undefined,
      filter: filterParam(), sortBy: st.sort ? `${st.sort.key}:${st.sort.dir}` : undefined,
    }, { signal: ctx.signal });
    st.rows = reset ? page.data : [...st.rows, ...page.data];
    st.cursor = page.nextCursor;
    st.loading = false;
  }

  async function reload() {
    try { await fetchPage(true); draw(); } catch (err) { mount(gridEl, h('div.text-danger.p-3', err.message)); }
  }

  function draw() {
    const cols = userCols();
    mount(infoEl, `${st.rows.length}${st.cursor ? '+' : ''} row${st.rows.length === 1 ? '' : 's'} loaded${st.selected.size ? ` · ${st.selected.size} selected` : ''}`);
    if (!st.rows.length) {
      mount(gridEl, emptyHint('table', st.filters.length || st.search ? 'No matching rows' : 'Empty table', st.filters.length || st.search ? null : 'Add rows, import a CSV, or let a workflow insert them.', h('button.btn.btn-sm.btn-primary', { onclick: () => addRow() }, 'Add a row')));
      return;
    }
    const sortIcon = (k) => (st.sort?.key === k ? icon(st.sort.dir === 'asc' ? 'sort-up' : 'sort-down', 'ms-1') : null);
    const head = h('thead', h('tr',
      h('th.n8s-grid-check', h('input.form-check-input', { type: 'checkbox', checked: st.rows.every((r) => st.selected.has(r.id)), onchange: (e) => { st.rows.forEach((r) => (e.target.checked ? st.selected.add(r.id) : st.selected.delete(r.id))); draw(); } })),
      ['id', ...cols.map((c) => c.name), 'updatedAt'].map((name) => {
        const col = cols.find((c) => c.name === name);
        return h('th', { class: !col && 'sys' },
          h('div.n8s-grid-th',
            h('button.n8s-grid-sort', { onclick: () => { st.sort = st.sort?.key === name && st.sort.dir === 'asc' ? { key: name, dir: 'desc' } : st.sort?.key === name ? null : { key: name, dir: 'asc' }; reload(); } }, name, col && h('span.n8s-coltype', col.type), sortIcon(name)),
            col && menuButton('', [
              { label: 'Rename…', icon: 'pencil', onClick: () => renameColumn(col) },
              { label: 'Move left', icon: 'arrow-left', onClick: () => moveColumn(col, -1) },
              { label: 'Move right', icon: 'arrow-right', onClick: () => moveColumn(col, 1) },
              { divider: true },
              { label: 'Delete column…', icon: 'trash', danger: true, onClick: () => deleteColumn(col) },
            ], { icon: 'chevron-down' })));
      }),
      h('th')));
    const bodyRows = st.rows.map((r) => h('tr', { class: st.selected.has(r.id) && 'sel' },
      h('td.n8s-grid-check', h('input.form-check-input', { type: 'checkbox', checked: st.selected.has(r.id), onchange: (e) => { e.target.checked ? st.selected.add(r.id) : st.selected.delete(r.id); draw(); } })),
      h('td.sys.af-mono', String(r.id)),
      cols.map((c) => cell(r, c)),
      h('td.sys.small', { title: r.updatedAt }, timeAgo(r.updatedAt)),
      h('td', h('button.af-icon-btn', { title: 'Delete row', onclick: () => deleteRows([r.id]) }, icon('trash')))));
    mount(gridEl,
      st.selected.size ? h('div.n8s-grid-selbar', `${st.selected.size} selected`, h('button.btn.btn-sm.btn-outline-danger', { onclick: () => deleteRows([...st.selected]) }, icon('trash', 'me-1'), 'Delete'), h('button.af-link-btn', { onclick: () => { st.selected.clear(); draw(); } }, 'Clear')) : null,
      h('div.n8s-grid-scroll', h('table.n8s-grid', head, h('tbody', bodyRows))),
      st.cursor && h('div.text-center.my-2', h('button.btn.btn-sm.btn-outline-secondary', { onclick: async () => { await fetchPage(false); draw(); } }, 'Load more')));
  }

  function cell(row, col) {
    const v = row[col.name];
    const td = h('td', { class: [`t-${col.type}`, v == null && 'null'], tabindex: 0, title: 'Double-click or Enter to edit' }, display(v, col.type));
    const edit = () => {
      if (td.querySelector('input,select')) return;
      const input = col.type === 'boolean'
        ? h('select.form-select.form-select-sm', ['', 'true', 'false'].map((x) => h('option', { value: x, selected: String(v ?? '') === x }, x || 'null')))
        : h('input.form-control.form-control-sm', { type: col.type === 'number' ? 'number' : col.type === 'date' ? 'datetime-local' : 'text', value: col.type === 'date' && v ? String(v).slice(0, 16) : v ?? '' });
      mount(td, input);
      input.focus();
      let done = false;
      const commit = async (save) => {
        if (done) return;
        done = true;
        const raw = input.value;
        if (!save || raw === String(v ?? '')) { mount(td, display(v, col.type)); return; }
        const value = coerce(raw, col.type);
        mount(td, spinner('sm'));
        try {
          await n8n.patch(`/data-tables/${enc(id)}/rows/update`, { filter: { type: 'and', filters: [{ columnName: 'id', condition: 'eq', value: row.id }] }, data: { [col.name]: value } });
          row[col.name] = value;
          td.classList.add('saved');
          setTimeout(() => td.classList.remove('saved'), 900);
        } catch (err) { toastError(err, 'Update failed'); }
        mount(td, display(row[col.name], col.type));
        td.classList.toggle('null', row[col.name] == null);
      };
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); commit(true); }
        if (e.key === 'Escape') commit(false);
        if (e.key === 'Tab') { commit(true); }
      });
      input.addEventListener('blur', () => commit(true));
      if (input.tagName === 'SELECT') input.addEventListener('change', () => commit(true));
    };
    td.addEventListener('dblclick', edit);
    td.addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target === td) { e.preventDefault(); edit(); } });
    return td;
  }

  const display = (v, type) => {
    if (v == null) return h('span.n8s-null', 'null');
    if (type === 'boolean') return h('span', { class: v ? 'text-success' : 'text-danger' }, String(v));
    if (type === 'date') return h('span', { title: v }, new Date(v).toLocaleString());
    return String(v);
  };

  // ---------------------------------------------------------------- rows

  async function addRow() {
    const cols = userCols();
    const inputs = cols.map((c) => ({ c, el: c.type === 'boolean' ? h('select.form-select.form-select-sm', h('option', { value: '' }, 'null'), h('option', 'true'), h('option', 'false')) : h('input.form-control.form-control-sm', { type: c.type === 'number' ? 'number' : c.type === 'date' ? 'datetime-local' : 'text' }) }));
    modal({
      title: 'Add row', icon: 'plus-lg',
      body: h('div', inputs.map(({ c, el }) => h('div.af-form-row', h('label.form-label.small.fw-semibold', c.name, h('span.n8s-coltype.ms-1', c.type)), el))),
      actions: [{ label: 'Cancel' }, {
        label: 'Insert', variant: 'primary',
        onClick: async (close) => {
          const data = Object.fromEntries(inputs.map(({ c, el }) => [c.name, coerce(el.value, c.type)]));
          await n8n.post(`/data-tables/${enc(id)}/rows`, { data: [data], returnType: 'all' });
          close();
          reload();
        },
      }],
    });
  }

  async function deleteRows(ids) {
    if (!(await confirmDialog({ title: `Delete ${ids.length} row(s)?`, danger: true, confirmLabel: 'Delete', message: 'Rows are removed permanently.' }))) return;
    try {
      for (let i = 0; i < ids.length; i += 100) {
        const chunk = ids.slice(i, i + 100);
        // eslint-disable-next-line no-await-in-loop
        await n8n.del(`/data-tables/${enc(id)}/rows/delete`, { params: { filter: JSON.stringify({ type: 'or', filters: chunk.map((x) => ({ columnName: 'id', condition: 'eq', value: x })) }) } });
      }
      st.selected.clear();
      toast(`Deleted ${ids.length} row(s)`, 'success');
      reload();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  function filterEditor(initial) {
    return jsonEditor(initial || { type: 'and', filters: [{ columnName: userCols()[0]?.name || 'id', condition: 'eq', value: '' }] }, { rows: 7, allowEmpty: false });
  }

  function bulkUpdate() {
    const f = filterEditor(st.filters.length ? JSON.parse(filterParam() || 'null') : null);
    const d = jsonEditor(Object.fromEntries(userCols().slice(0, 2).map((c) => [c.name, null])), { rows: 5, allowEmpty: false });
    const preview = h('div.small');
    modal({
      title: 'Bulk update rows', icon: 'pencil-square', size: 'lg',
      body: h('div', h('label.small.fw-semibold', 'Filter (which rows)'), f.el, h('label.small.fw-semibold.mt-2', 'Set values'), d.el, preview),
      actions: [{ label: 'Cancel' },
        { label: 'Preview (dry run)', onClick: async () => { const r = await n8n.patch(`/data-tables/${enc(id)}/rows/update`, { filter: f.value(), data: d.value(), dryRun: true, returnData: true }); mount(preview, h('div.mt-2', h('b', `${Array.isArray(r) ? r.length : '?'} row(s) would change`), Array.isArray(r) && h('pre.n8s-pre', JSON.stringify(r.slice(0, 20), null, 2)))); } },
        { label: 'Apply', variant: 'primary', onClick: async (close) => { const r = await n8n.patch(`/data-tables/${enc(id)}/rows/update`, { filter: f.value(), data: d.value(), returnData: true }); toast(`Updated ${Array.isArray(r) ? r.length : ''} row(s)`, 'success'); close(); reload(); } }],
    });
  }

  function upsert() {
    const f = filterEditor();
    const d = jsonEditor(Object.fromEntries(userCols().map((c) => [c.name, null])), { rows: 6, allowEmpty: false });
    modal({
      title: 'Upsert a row', icon: 'arrow-repeat', size: 'lg',
      body: h('div', h('p.small.text-muted', 'Updates the row matching the filter, or inserts it when none matches.'), h('label.small.fw-semibold', 'Match filter'), f.el, h('label.small.fw-semibold.mt-2', 'Data'), d.el),
      actions: [{ label: 'Cancel' }, { label: 'Upsert', variant: 'primary', onClick: async (close) => { await n8n.post(`/data-tables/${enc(id)}/rows/upsert`, { filter: f.value(), data: d.value(), returnData: true }); close(); reload(); } }],
    });
  }

  function deleteMatching() {
    const f = filterEditor(st.filters.length ? JSON.parse(filterParam() || 'null') : null);
    const preview = h('div.small');
    modal({
      title: 'Delete matching rows', icon: 'funnel', size: 'lg', body: h('div', f.el, preview),
      actions: [{ label: 'Cancel' },
        { label: 'Preview', onClick: async () => { const r = await n8n.del(`/data-tables/${enc(id)}/rows/delete`, { params: { filter: JSON.stringify(f.value()), dryRun: true, returnData: true } }); mount(preview, h('b.d-block.mt-2', `${Array.isArray(r) ? r.length : '?'} row(s) would be deleted`)); } },
        { label: 'Delete', variant: 'danger', onClick: async (close) => { await n8n.del(`/data-tables/${enc(id)}/rows/delete`, { params: { filter: JSON.stringify(f.value()) } }); close(); reload(); } }],
    });
  }

  async function clearAll() {
    if (!(await confirmDialog({ title: 'Clear all rows?', danger: true, confirmLabel: 'Clear table', message: h('span', 'Deletes every row of ', h('b', table.name), '. Columns stay.') }))) return;
    const r = await n8n.del(`/data-tables/${enc(id)}/rows/clear`).catch(toastError);
    if (r) toast(`Deleted ${r.deletedCount ?? 'all'} rows`, 'success');
    reload();
  }

  // ---------------------------------------------------------------- columns

  async function refreshTable() { table = await n8n.get(`/data-tables/${enc(id)}`); }

  function addColumn() {
    const name = h('input.form-control.form-control-sm', { placeholder: 'column name', autofocus: true });
    const type = h('select.form-select.form-select-sm', COLUMN_TYPES.map((t) => h('option', t)));
    modal({
      title: 'Add column', size: 'sm', body: h('div', name, h('div.mt-2', type)),
      actions: [{ label: 'Cancel' }, {
        label: 'Add', variant: 'primary',
        onClick: async (close) => {
          if (RESERVED_COLUMNS.includes(name.value.trim())) throw new Error(`"${name.value}" is reserved by n8n.`);
          await n8n.post(`/data-tables/${enc(id)}/columns`, { name: name.value.trim(), type: type.value });
          await refreshTable();
          close();
          reload();
        },
      }],
    });
  }

  async function renameColumn(col) {
    const v = await promptText({ title: 'Rename column', value: col.name, help: 'Workflows referencing the old name must be updated.' });
    if (!v || v === col.name) return;
    try { await n8n.patch(`/data-tables/${enc(id)}/columns/${enc(col.id)}`, { name: v }); await refreshTable(); reload(); } catch (err) { toastError(err); }
  }

  async function moveColumn(col, dir) {
    const cols = userCols();
    const i = cols.indexOf(col);
    const j = i + dir;
    if (j < 0 || j >= cols.length) return;
    try { await n8n.patch(`/data-tables/${enc(id)}/columns/${enc(col.id)}`, { index: cols[j].index ?? j }); await refreshTable(); draw(); } catch (err) { toastError(err); }
  }

  async function deleteColumn(col) {
    if (!(await confirmDialog({ title: `Delete column ${col.name}?`, danger: true, confirmLabel: 'Delete', message: 'Its values are lost in every row.' }))) return;
    try { await n8n.del(`/data-tables/${enc(id)}/columns/${enc(col.id)}`); await refreshTable(); reload(); } catch (err) { toastError(err); }
  }

  // ---------------------------------------------------------------- import / export

  async function insertRecords(records, headers) {
    let cols = userCols();
    const missing = headers.filter((x) => !RESERVED_COLUMNS.includes(x) && !cols.some((c) => c.name === x));
    if (missing.length) {
      const ok = await confirmDialog({ title: 'New columns', confirmLabel: 'Create columns', message: h('span', 'The file has columns the table does not: ', h('b', missing.join(', ')), '. Create them (types inferred from the data)?') });
      if (ok) {
        for (const m of missing) await n8n.post(`/data-tables/${enc(id)}/columns`, { name: m, type: inferType(records.slice(0, 200).map((r) => r[m])) }); // eslint-disable-line no-await-in-loop
        await refreshTable();
        cols = userCols();
      }
    }
    const typed = records.map((r) => Object.fromEntries(cols.filter((c) => c.name in r).map((c) => [c.name, coerce(r[c.name], c.type)])));
    const bar = h('div.progress-bar', { style: 'width:0%' });
    const m = modal({ title: `Importing ${typed.length} rows`, persistent: true, body: h('div.progress', { style: 'height:8px' }, bar) });
    try {
      for (let i = 0; i < typed.length; i += 500) {
        // eslint-disable-next-line no-await-in-loop
        await n8n.post(`/data-tables/${enc(id)}/rows`, { data: typed.slice(i, i + 500), returnType: 'count' });
        bar.style.width = `${Math.min(100, ((i + 500) / typed.length) * 100)}%`;
      }
      toast(`Imported ${typed.length} rows`, 'success');
    } catch (err) { toastError(err, 'Import failed'); } finally { m.close(); reload(); }
  }

  async function importCsv() {
    const file = await pickFile('.csv,text/csv');
    if (!file) return;
    const { headers, records } = csvToRecords(await readFileText(file));
    const reserved = headers.filter((x) => RESERVED_COLUMNS.includes(x));
    if (reserved.length) toast(`Ignoring reserved column(s) ${reserved.join(', ')} — n8n generates them.`, 'info', { timeout: 6000 });
    insertRecords(records, headers);
  }

  async function importJson() {
    const file = await pickFile('.json,application/json');
    if (!file) return;
    let data = JSON.parse(await readFileText(file));
    if (!Array.isArray(data)) data = data.data || [data];
    insertRecords(data, [...new Set(data.flatMap((r) => Object.keys(r)))]);
  }

  async function exportAll(kind) {
    const rows = await n8n.all(`/data-tables/${enc(id)}/rows`, { filter: filterParam(), search: st.search || undefined }, { max: 200000 });
    const stamp = new Date().toISOString().slice(0, 10);
    if (kind === 'json') saveText(JSON.stringify(rows, null, 2), `${table.name}-${stamp}.json`);
    else saveText(toCSV(rows, ['id', ...userCols().map((c) => c.name), 'createdAt', 'updatedAt'], { bom: kind === 'csv-bom' }), `${table.name}-${stamp}.csv`, 'text/csv');
    toast(`Exported ${rows.length} rows`, 'success');
  }

  mount(gridEl, spinner());
  drawFilters();
  await reload();
}
