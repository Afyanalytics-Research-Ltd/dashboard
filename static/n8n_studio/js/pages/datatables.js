// Data tables: list, create (with columns or straight from a CSV), rename, delete.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toast, toastError, modal, spinner, timeAgo, menuButton } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { invalidate, loadOptional } from '../core/store.js';
import { csvToRecords, inferType, coerce, RESERVED_COLUMNS } from '../core/csv.js';
import { emptyHint, gated, confirmDelete, promptText, pickFile, readFileText } from '../core/widgets.js';

export const COLUMN_TYPES = ['string', 'number', 'boolean', 'date'];

export default async function render(ctx) {
  const body = h('div');
  mount(ctx.el, pageHeader({
    title: 'Data tables', icon: 'table', subtitle: 'Persistent tables your workflows read and write — edit them like a spreadsheet.',
    actions: [
      h('button.btn.btn-sm.btn-primary', { onclick: () => create() }, icon('plus-lg', 'me-1'), 'New table'),
      h('button.btn.btn-sm.btn-outline-primary', { onclick: () => createFromCsv() }, icon('filetype-csv', 'me-1'), 'From CSV'),
    ],
  }), body);

  const refresh = () => gated(body, async () => {
    const tables = await n8n.all('/data-tables', { sortBy: 'updatedAt:desc' }, { signal: ctx.signal });
    return dataTable({
      rows: tables,
      empty: emptyHint('table', 'No data tables yet', 'Create one, or upload a CSV to create and fill it in one go.'),
      columns: [
        { key: 'name', label: 'Name', render: (t) => h('a.fw-semibold', { href: urls.dataTable(t.id) }, t.name) },
        { key: 'columns', label: 'Columns', render: (t) => h('span.small', (t.columns || []).map((c) => c.name).join(', ') || '—') },
        { key: 'updatedAt', label: 'Updated', render: (t) => timeAgo(t.updatedAt) },
        {
          key: 'act', label: '', align: 'right', render: (t) => menuButton('', [
            { label: 'Open', icon: 'table', href: urls.dataTable(t.id) },
            { label: 'Rename…', icon: 'pencil', onClick: async () => { const name = await promptText({ title: 'Rename table', value: t.name }); if (name) { await n8n.patch(`/data-tables/${enc(t.id)}`, { name }).catch(toastError); refresh(); } } },
            { divider: true },
            { label: 'Delete…', icon: 'trash', danger: true, onClick: async () => { if (await confirmDelete('data table', t.name)) { await n8n.del(`/data-tables/${enc(t.id)}`).catch(toastError); invalidate('dataTables'); refresh(); } } },
          ]),
        },
      ],
      onRowClick: (t) => ctx.navigate(urls.dataTable(t.id)),
    });
  }, { alternative: 'Data tables need n8n 1.113+ with the data-table module enabled.' });

  async function projectPicker() {
    const projects = await loadOptional('projects');
    return projects.length > 1 ? h('select.form-select.form-select-sm', h('option', { value: '' }, 'Personal project'), projects.map((p) => h('option', { value: p.id }, p.name))) : null;
  }

  async function create(prefill = { name: '', columns: [{ name: 'name', type: 'string' }] }, rows = null) {
    const name = h('input.form-control.form-control-sm', { value: prefill.name, placeholder: 'e.g. customers', autofocus: true });
    const cols = prefill.columns.map((c) => ({ ...c }));
    const colEl = h('div');
    const project = await projectPicker();
    const drawCols = () => mount(colEl, cols.map((c, i) => h('div.n8s-colrow',
      h('input.form-control.form-control-sm', { value: c.name, placeholder: 'column name', oninput: (e) => { c.name = e.target.value; } }),
      h('select.form-select.form-select-sm', { onchange: (e) => { c.type = e.target.value; } }, COLUMN_TYPES.map((t) => h('option', { selected: t === c.type }, t))),
      h('button.af-icon-btn', { type: 'button', onclick: () => { cols.splice(i, 1); drawCols(); } }, icon('x-lg')))),
    h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => { cols.push({ name: '', type: 'string' }); drawCols(); } }, icon('plus', 'me-1'), 'Column'));
    drawCols();
    modal({
      title: 'New data table', icon: 'table',
      body: h('div', h('label.form-label.small.fw-semibold', 'Name'), name, project && [h('label.form-label.small.fw-semibold.mt-2', 'Project'), project],
        h('label.form-label.small.fw-semibold.mt-3', 'Columns'), h('div.small.text-muted.mb-1', 'id, createdAt and updatedAt are added automatically.'), colEl,
        rows && h('div.small.text-success.mt-2', icon('check2', 'me-1'), `${rows.length} rows will be inserted.`)),
      actions: [{ label: 'Cancel' }, {
        label: 'Create', variant: 'primary',
        onClick: async (close) => {
          const columns = cols.filter((c) => c.name.trim() && !RESERVED_COLUMNS.includes(c.name.trim()));
          if (!name.value.trim()) throw new Error('Name is required.');
          const t = await n8n.post('/data-tables', { name: name.value.trim(), columns, ...(project?.value ? { projectId: project.value } : {}) });
          if (rows?.length) {
            const typed = rows.map((r) => Object.fromEntries(columns.map((c) => [c.name, coerce(r[c.name], c.type)])));
            for (let i = 0; i < typed.length; i += 500) await n8n.post(`/data-tables/${enc(t.id)}/rows`, { data: typed.slice(i, i + 500), returnType: 'count' }); // eslint-disable-line no-await-in-loop
          }
          invalidate('dataTables');
          close();
          toast('Table created', 'success');
          ctx.navigate(urls.dataTable(t.id));
        },
      }],
    });
  }

  async function createFromCsv() {
    const file = await pickFile('.csv,text/csv');
    if (!file) return;
    try {
      const { headers, records } = csvToRecords(await readFileText(file));
      const usable = headers.filter((x) => !RESERVED_COLUMNS.includes(x));
      const columns = usable.map((x) => ({ name: x, type: inferType(records.slice(0, 200).map((r) => r[x])) }));
      if (headers.length !== usable.length) toast(`Skipped reserved column(s): ${headers.filter((x) => RESERVED_COLUMNS.includes(x)).join(', ')} — n8n manages those itself.`, 'info', { timeout: 7000 });
      create({ name: file.name.replace(/\.csv$/i, ''), columns }, records);
    } catch (err) { toastError(err, 'Could not read CSV'); }
  }

  mount(body, spinner());
  refresh();
}
