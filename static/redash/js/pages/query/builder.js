// Visual query builder: table -> columns (+ aggregates) -> filters -> grouping -> order -> limit,
// with a live SQL preview. For people who don't write SQL every day.

import { h, icon, mount } from '../../../../ui/js/dom.js';
import { modal, codeBlock } from '../../../../ui/js/ui.js';
import { colName, colType } from './schema.js';

const AGGS = [
  { value: '', label: 'Value' },
  { value: 'COUNT', label: 'Count' },
  { value: 'COUNT_DISTINCT', label: 'Count distinct' },
  { value: 'SUM', label: 'Sum' },
  { value: 'AVG', label: 'Average' },
  { value: 'MIN', label: 'Minimum' },
  { value: 'MAX', label: 'Maximum' },
];
const OPS = ['=', '!=', '>', '>=', '<', '<=', 'LIKE', 'ILIKE', 'IN', 'NOT IN', 'BETWEEN', 'IS NULL', 'IS NOT NULL'];
const NUMERIC = /int|float|double|decimal|numeric|number|real/i;

const isParam = (v) => /^\s*\{\{.*\}\}\s*$/.test(v);
function literal(value, type) {
  const v = String(value ?? '').trim();
  if (isParam(v)) return v;
  if ((type && NUMERIC.test(type)) || /^-?\d+(\.\d+)?$/.test(v)) return /^-?\d+(\.\d+)?$/.test(v) ? v : `'${v.replace(/'/g, "''")}'`;
  if (/^(true|false|null)$/i.test(v)) return v.toUpperCase();
  return `'${v.replace(/'/g, "''")}'`;
}

const aggExpr = (agg, col) => (agg === 'COUNT_DISTINCT' ? `COUNT(DISTINCT ${col})` : agg === 'COUNT' && col === '*' ? 'COUNT(*)' : `${agg}(${col})`);
const aliasFor = (agg, col) => `${agg === 'COUNT_DISTINCT' ? 'distinct' : agg.toLowerCase()}_${col === '*' ? 'rows' : col}`.toLowerCase().replace(/[^a-z0-9_]/g, '_');

/** Build SQL from a builder state (exported for testing). */
export function buildSql(state) {
  if (!state.table) return '';
  const cols = state.columns.filter((c) => c.on);
  const hasAgg = cols.some((c) => c.agg) || state.countRows;
  const select = [];
  const groupBy = [];
  for (const c of cols) {
    if (c.agg) select.push(`${aggExpr(c.agg, c.name)} AS ${aliasFor(c.agg, c.name)}`);
    else { select.push(c.name); if (hasAgg) groupBy.push(c.name); }
  }
  if (state.countRows) select.push('COUNT(*) AS row_count');
  const where = state.filters.filter((f) => f.column && f.op).map((f) => {
    const type = state.columns.find((c) => c.name === f.column)?.type;
    switch (f.op) {
      case 'IS NULL': case 'IS NOT NULL': return `${f.column} ${f.op}`;
      case 'IN': case 'NOT IN': {
        const items = isParam(f.value) ? f.value : String(f.value || '').split(',').map((s) => literal(s, type)).join(', ');
        return `${f.column} ${f.op} (${items})`;
      }
      case 'BETWEEN': return `${f.column} BETWEEN ${literal(f.value, type)} AND ${literal(f.value2, type)}`;
      default: return `${f.column} ${f.op} ${literal(f.value, type)}`;
    }
  });
  const lines = [`SELECT ${select.length ? select.join(',\n       ') : '*'}`, `FROM ${state.table}`];
  if (where.length) lines.push(`WHERE ${where.join('\n  AND ')}`);
  if (groupBy.length) lines.push(`GROUP BY ${groupBy.join(', ')}`);
  if (state.orderBy) lines.push(`ORDER BY ${state.orderBy} ${state.orderDir || 'DESC'}`);
  if (state.limit) lines.push(`LIMIT ${Number(state.limit)}`);
  return lines.join('\n');
}

/**
 * openBuilder({ schema, table, onInsert(sql, {replace}), onRun(sql) })
 */
export function openBuilder({ schema = [], table, onInsert, onRun } = {}) {
  const state = { table: null, columns: [], filters: [], countRows: false, orderBy: '', orderDir: 'DESC', limit: 100 };
  const tableSearch = h('input.form-control.form-control-sm', { type: 'search', placeholder: `Search ${schema.length.toLocaleString()} tables…`, list: 'rd-builder-tables', value: table?.name || '' });
  const tableList = h('datalist#rd-builder-tables', schema.slice(0, 2000).map((t) => h('option', { value: t.name })));
  const colsEl = h('div.rd-b-cols');
  const filtersEl = h('div.rd-b-filters');
  const orderEl = h('div.rd-b-order');
  const previewEl = h('div.rd-b-preview');

  const setTable = (name) => {
    const t = schema.find((x) => x.name === name);
    if (!t) return;
    state.table = t.name;
    state.columns = (t.columns || []).map((c) => ({ name: colName(c), type: colType(c), on: false, agg: '' }));
    state.filters = [];
    state.orderBy = '';
    draw();
  };
  tableSearch.addEventListener('change', () => setTable(tableSearch.value));
  tableSearch.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); setTable(tableSearch.value); } });

  const colOptions = () => state.columns.map((c) => h('option', { value: c.name }, c.name));
  const orderOptions = () => {
    const out = state.columns.filter((c) => c.on).map((c) => (c.agg ? aliasFor(c.agg, c.name) : c.name));
    if (state.countRows) out.push('row_count');
    return out.length ? out : state.columns.map((c) => c.name);
  };

  function draw() {
    if (!state.table) {
      mount(colsEl, h('div.text-muted.small', 'Choose a table to see its columns.'));
      mount(filtersEl); mount(orderEl);
      preview();
      return;
    }
    mount(colsEl,
      h('div.rd-b-cols-tools',
        h('button.af-link-btn', { type: 'button', onclick: () => { state.columns.forEach((c) => { c.on = true; }); draw(); } }, 'Select all'),
        h('button.af-link-btn', { type: 'button', onclick: () => { state.columns.forEach((c) => { c.on = false; c.agg = ''; }); draw(); } }, 'Clear'),
        h('label.form-check.form-switch.small.ms-auto.mb-0',
          h('input.form-check-input', { type: 'checkbox', checked: state.countRows, onchange: (e) => { state.countRows = e.target.checked; draw(); } }),
          h('span.ms-1', 'Add a row count'))),
      h('div.rd-b-col-grid', state.columns.map((c) => h('div.rd-b-col', { class: c.on && 'on' },
        h('label.rd-b-col-name',
          h('input.form-check-input', { type: 'checkbox', checked: c.on, onchange: (e) => { c.on = e.target.checked; if (!c.on) c.agg = ''; draw(); } }),
          h('span', c.name), c.type && h('span.rd-schema-coltype', c.type)),
        c.on && h('select.form-select.form-select-sm', { 'aria-label': `How to show ${c.name}`, onchange: (e) => { c.agg = e.target.value; draw(); } },
          AGGS.filter((a) => !['SUM', 'AVG'].includes(a.value) || !c.type || NUMERIC.test(c.type))
            .map((a) => h('option', { value: a.value, selected: a.value === c.agg }, a.label)))))));

    mount(filtersEl,
      state.filters.map((f, i) => h('div.rd-b-filter',
        h('select.form-select.form-select-sm', { onchange: (e) => { f.column = e.target.value; preview(); } }, h('option', { value: '' }, 'Column…'), colOptions().map((o) => { o.selected = o.value === f.column; return o; })),
        h('select.form-select.form-select-sm.rd-b-op', { onchange: (e) => { f.op = e.target.value; draw(); } }, OPS.map((op) => h('option', { value: op, selected: op === f.op }, op))),
        !/NULL/.test(f.op) && h('input.form-control.form-control-sm', {
          value: f.value || '', placeholder: /IN/.test(f.op) ? 'a, b, c' : f.op.includes('LIKE') ? '%text%' : 'value or {{ param }}',
          oninput: (e) => { f.value = e.target.value; preview(); },
        }),
        f.op === 'BETWEEN' && h('span.text-muted.small', 'and'),
        f.op === 'BETWEEN' && h('input.form-control.form-control-sm', { value: f.value2 || '', oninput: (e) => { f.value2 = e.target.value; preview(); } }),
        h('button.af-icon-btn', { type: 'button', title: 'Remove filter', onclick: () => { state.filters.splice(i, 1); draw(); } }, icon('x-lg')))),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => { state.filters.push({ column: state.columns[0]?.name || '', op: '=', value: '' }); draw(); } }, icon('plus', 'me-1'), 'Add filter'));

    mount(orderEl,
      h('select.form-select.form-select-sm', { 'aria-label': 'Order by', onchange: (e) => { state.orderBy = e.target.value; preview(); } },
        h('option', { value: '' }, 'No ordering'), orderOptions().map((o) => h('option', { value: o, selected: o === state.orderBy }, o))),
      h('select.form-select.form-select-sm', { 'aria-label': 'Direction', onchange: (e) => { state.orderDir = e.target.value; preview(); } },
        ['DESC', 'ASC'].map((d) => h('option', { value: d, selected: d === state.orderDir }, d === 'DESC' ? 'Largest / newest first' : 'Smallest / oldest first'))),
      h('label.rd-b-limit', 'Limit', h('input.form-control.form-control-sm', { type: 'number', min: 1, value: state.limit, oninput: (e) => { state.limit = e.target.value; preview(); } })));
    preview();
  }

  const preview = () => {
    const sql = buildSql(state);
    mount(previewEl, sql ? codeBlock(sql, { lang: 'sql' }) : h('div.text-muted.small.p-2', 'The SQL appears here as you build.'));
    m.el.querySelectorAll('.rd-b-action').forEach((b) => { b.disabled = !sql; });
  };

  const m = modal({
    title: 'Visual query builder', icon: 'ui-checks-grid', size: 'xl',
    subtitle: 'Build a query without writing SQL — you can keep editing the generated SQL afterwards.',
    body: h('div.rd-builder',
      h('div.rd-builder-form',
        h('div.rd-b-step', h('div.rd-b-step-title', h('span.rd-b-num', '1'), 'Table'), tableSearch, tableList),
        h('div.rd-b-step', h('div.rd-b-step-title', h('span.rd-b-num', '2'), 'Columns'), colsEl),
        h('div.rd-b-step', h('div.rd-b-step-title', h('span.rd-b-num', '3'), 'Filters'), filtersEl),
        h('div.rd-b-step', h('div.rd-b-step-title', h('span.rd-b-num', '4'), 'Order & limit'), orderEl)),
      h('div.rd-builder-preview', h('div.rd-b-step-title', icon('code-slash', 'me-1'), 'SQL'), previewEl)),
    actions: [
      { label: 'Cancel' },
      { label: 'Append to editor', icon: 'plus-square', onClick: (close) => { onInsert(buildSql(state), { replace: false }); close(); } },
      { label: 'Replace editor', icon: 'arrow-left-right', onClick: (close) => { onInsert(buildSql(state), { replace: true }); close(); } },
      onRun && { label: 'Replace & run', variant: 'primary', icon: 'play-fill', onClick: (close) => { onInsert(buildSql(state), { replace: true }); close(); onRun(); } },
    ].filter(Boolean),
  });
  m.el.querySelectorAll('.af-modal-footer .btn:not(:first-child)').forEach((b) => b.classList.add('rd-b-action'));
  if (table) setTable(table.name); else draw();
  return m;
}
