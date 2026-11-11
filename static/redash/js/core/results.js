// Results grid: sticky headers, sort, global search with highlight, per-column filters,
// conditional colouring, column show/hide, row detail, copy, paging and large-result
// friendly rendering. Used for query results and TABLE visualizations.
//
//   const grid = resultsGrid(queryResult.data, { tableOptions, settings, onRowOpen })
//   el.appendChild(grid.el)

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import { copyButton, drawer, kv, emptyState, openMenu } from '../../../ui/js/ui.js';
import { fmtCell, fmtNumeral, fmtMoment, inferType, isNumericType } from './format.js';

const PAGE_SIZES = [25, 50, 100, 250, 500];

/** Normalise Redash TABLE options.columns to a full column list for this result. */
export function tableColumns(data, tableOptions = {}) {
  const cols = data.columns || [];
  const saved = new Map((tableOptions.columns || []).map((c) => [c.name, c]));
  const out = cols.map((c, i) => {
    const type = c.type && c.type !== 'string' ? c.type : inferType(data.rows || [], c.name);
    const s = saved.get(c.name) || {};
    return {
      name: c.name,
      title: s.title || c.friendly_name || c.name,
      type,
      visible: s.visible !== false,
      order: s.order ?? i,
      displayAs: s.displayAs || (isNumericType(type) ? 'number' : type === 'datetime' || type === 'date' ? 'datetime' : type === 'boolean' ? 'boolean' : 'string'),
      numberFormat: s.numberFormat,
      dateTimeFormat: s.dateTimeFormat,
      alignContent: s.alignContent || (isNumericType(type) ? 'right' : 'left'),
      linkUrlTemplate: s.linkUrlTemplate,
      linkTextTemplate: s.linkTextTemplate,
      imageUrlTemplate: s.imageUrlTemplate,
      highlightLinks: s.highlightLinks,
      // Studio extension (ignored by Redash's own renderer): conditional colours.
      rules: s.studioRules || [],
    };
  });
  return out.sort((a, b) => a.order - b.order);
}

const template = (tpl, row, col) => String(tpl || '').replace(/\{\{\s*@?\s*([^}\s]+)\s*\}\}/g, (_, k) => (k === '@' || k === 'value' ? row[col] : row[k] ?? ''));

function renderValue(col, row, settings) {
  const v = row[col.name];
  if (v == null) return h('span.rd-null', 'null');
  switch (col.displayAs) {
    case 'number': return col.numberFormat ? fmtNumeral(v, col.numberFormat) : fmtCell(v, col.type, settings);
    case 'datetime': return col.dateTimeFormat ? fmtMoment(v, col.dateTimeFormat) : fmtCell(v, col.type, settings);
    case 'boolean': return v ? icon('check-lg', 'text-success') : icon('x-lg', 'text-muted');
    case 'json': return h('code.rd-json-cell', typeof v === 'string' ? v : JSON.stringify(v));
    case 'link': {
      const url = template(col.linkUrlTemplate || '{{ @ }}', row, col.name);
      if (!/^(https?:|mailto:|\/)/.test(url)) return String(v);
      return h('a', { href: url, target: '_blank', rel: 'noopener' }, template(col.linkTextTemplate || '{{ @ }}', row, col.name));
    }
    case 'image': {
      const url = template(col.imageUrlTemplate || '{{ @ }}', row, col.name);
      return /^https?:/.test(url) ? h('img.rd-img-cell', { src: url, alt: '', loading: 'lazy' }) : String(v);
    }
    default: return typeof v === 'object' ? JSON.stringify(v) : String(v);
  }
}

function ruleMatch(rule, v) {
  const n = Number(v);
  const r = Number(rule.value);
  switch (rule.op) {
    case '>': return n > r;
    case '>=': return n >= r;
    case '<': return n < r;
    case '<=': return n <= r;
    case '=': return String(v) === String(rule.value);
    case '!=': return String(v) !== String(rule.value);
    case 'contains': return String(v ?? '').toLowerCase().includes(String(rule.value).toLowerCase());
    case 'empty': return v == null || v === '';
    default: return false;
  }
}

function highlightText(text, q) {
  if (!q) return text;
  const s = String(text);
  const i = s.toLowerCase().indexOf(q);
  if (i < 0) return s;
  return [s.slice(0, i), h('mark', s.slice(i, i + q.length)), highlightText(s.slice(i + q.length), q)];
}

function compare(a, b) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
}

/**
 * resultsGrid(data, opts)
 *   data: {columns, rows}
 *   opts: { tableOptions, settings, pageSize, compact, searchable = true, onColumnsChange(cols), maxHeight, title }
 * Returns { el, getView() -> {columns, rows} (filtered+sorted), setData(data) }
 */
export function resultsGrid(data, opts = {}) {
  const { settings = {}, searchable = true, compact = false, maxHeight } = opts;
  let columns = tableColumns(data, opts.tableOptions);
  let rows = data.rows || [];
  let sort = null; // {name, dir}
  let query = '';
  const colFilters = {}; // name -> lowercase substring
  let pageSize = opts.pageSize || opts.tableOptions?.itemsPerPage || prefs.get('rd:grid:pageSize', 50);
  let page = 0;

  const el = h('div.rd-grid', { class: compact && 'rd-grid-compact' });
  const toolbar = h('div.rd-grid-toolbar');
  const tableWrap = h('div.rd-grid-scroll', { style: maxHeight ? { maxHeight } : null });
  const footer = h('div.rd-grid-footer');
  el.append(toolbar, tableWrap, footer);

  const visible = () => columns.filter((c) => c.visible);

  function view() {
    let out = rows;
    const q = query.toLowerCase();
    const cf = Object.entries(colFilters).filter(([, v]) => v);
    if (q || cf.length) {
      const vis = visible();
      out = out.filter((r) => (!q || vis.some((c) => String(r[c.name] ?? '').toLowerCase().includes(q)))
        && cf.every(([name, v]) => String(r[name] ?? '').toLowerCase().includes(v)));
    }
    if (sort) {
      const dir = sort.dir === 'asc' ? 1 : -1;
      out = [...out].sort((a, b) => compare(a[sort.name], b[sort.name]) * dir);
    }
    return out;
  }

  function renderToolbar(count) {
    const search = searchable && h('div.af-search.rd-grid-search',
      icon('search', 'af-search-icon'),
      h('input.form-control.form-control-sm', {
        type: 'search', placeholder: `Search ${rows.length.toLocaleString()} rows…`, value: query,
        oninput: debounce((e) => { query = e.target.value.trim(); page = 0; renderBody(); }, 150),
      }));
    mount(toolbar,
      search,
      h('span.rd-grid-count', count === rows.length ? `${rows.length.toLocaleString()} rows` : `${count.toLocaleString()} of ${rows.length.toLocaleString()} rows`),
      h('div.rd-grid-tools',
        Object.values(colFilters).some(Boolean) && h('button.btn.btn-sm.btn-link', { onclick: () => { Object.keys(colFilters).forEach((k) => delete colFilters[k]); renderBody(); } }, icon('funnel', 'me-1'), 'Clear column filters'),
        h('button.btn.btn-sm.btn-outline-secondary', { title: 'Show / hide columns', onclick: (e) => columnsMenu(e.currentTarget) }, icon('layout-three-columns'), ` ${visible().length}/${columns.length}`)));
  }

  function columnsMenu(anchor) {
    openMenu(anchor, [
      { header: 'Columns' },
      ...columns.map((c) => ({
        label: c.title, icon: c.visible ? 'check-square' : 'square',
        onClick: () => { c.visible = !c.visible; opts.onColumnsChange?.(columns); renderBody(); setTimeout(() => columnsMenu(anchor), 0); },
      })),
      { divider: true },
      { label: 'Show all', icon: 'eye', onClick: () => { columns.forEach((c) => { c.visible = true; }); renderBody(); } },
    ]);
  }

  function header() {
    return h('thead', h('tr',
      h('th.rd-rownum', '#'),
      visible().map((c) => {
        const sorted = sort?.name === c.name;
        const filter = h('input.rd-col-filter', {
          type: 'text', placeholder: 'filter', value: colFilters[c.name] || '', 'aria-label': `Filter ${c.title}`,
          onclick: (e) => e.stopPropagation(),
          oninput: debounce((e) => { colFilters[c.name] = e.target.value.trim().toLowerCase(); page = 0; renderBody(true); }, 200),
        });
        return h('th', {
          class: [sorted && 'sorted', `align-${c.alignContent}`],
          title: `${c.name} · ${c.type}`,
          onclick: () => {
            sort = !sorted ? { name: c.name, dir: 'asc' } : sort.dir === 'asc' ? { name: c.name, dir: 'desc' } : null;
            renderBody();
          },
        },
        h('div.rd-th',
          h('span.rd-th-title', c.title),
          h('span.rd-th-type', c.type),
          icon(sorted ? (sort.dir === 'asc' ? 'sort-up' : 'sort-down') : 'chevron-expand', 'rd-sort-icon')),
        filter);
      })));
  }

  function renderBody(keepFocus = false) {
    const focused = keepFocus ? document.activeElement : null;
    const focusName = focused?.getAttribute?.('aria-label');
    const v = view();
    renderToolbar(v.length);
    const pages = Math.max(1, Math.ceil(v.length / pageSize));
    page = Math.min(page, pages - 1);
    const slice = v.slice(page * pageSize, (page + 1) * pageSize);
    const q = query.toLowerCase();
    const vis = visible();
    if (!rows.length) {
      mount(tableWrap, emptyState({ icon: 'inbox', title: 'The query returned no rows', message: 'It ran successfully, there is just no data for these filters or parameters.' }));
      mount(footer);
      return;
    }
    const tbody = h('tbody', slice.map((r, i) => h('tr', { ondblclick: () => openRow(r) },
      h('td.rd-rownum', h('button.rd-row-open', { title: 'Open row', onclick: () => openRow(r) }, String(page * pageSize + i + 1))),
      vis.map((c) => {
        const raw = r[c.name];
        const rule = c.rules.find((ru) => ruleMatch(ru, raw));
        const content = renderValue(c, r, settings);
        return h('td', {
          class: [`align-${c.alignContent}`, raw == null && 'is-null'],
          style: rule ? { background: rule.bg || undefined, color: rule.color || undefined, fontWeight: rule.bold ? 600 : undefined } : null,
        }, q && typeof content === 'string' ? highlightText(content, q) : content);
      }))));
    mount(tableWrap,
      v.length ? h('table.rd-table', header(), tbody)
        : h('div', h('table.rd-table', header()), emptyState({ icon: 'search', title: 'No rows match', message: 'Clear the search or column filters to see all rows.' })));
    if (focusName) tableWrap.querySelector(`[aria-label="${CSS.escape(focusName)}"]`)?.focus();
    mount(footer,
      h('div.rd-grid-pager',
        h('select.form-select.form-select-sm', {
          'aria-label': 'Rows per page',
          onchange: (e) => { pageSize = Number(e.target.value); prefs.set('rd:grid:pageSize', pageSize); page = 0; renderBody(); },
        }, PAGE_SIZES.map((n) => h('option', { value: n, selected: n === pageSize }, `${n} / page`))),
        h('button.af-page-btn', { disabled: page === 0, onclick: () => { page = 0; renderBody(); }, title: 'First page' }, icon('chevron-double-left')),
        h('button.af-page-btn', { disabled: page === 0, onclick: () => { page--; renderBody(); }, title: 'Previous page' }, icon('chevron-left')),
        h('span.rd-page-label', `Page ${page + 1} of ${pages}`),
        h('button.af-page-btn', { disabled: page >= pages - 1, onclick: () => { page++; renderBody(); }, title: 'Next page' }, icon('chevron-right')),
        h('button.af-page-btn', { disabled: page >= pages - 1, onclick: () => { page = pages - 1; renderBody(); }, title: 'Last page' }, icon('chevron-double-right'))));
  }

  function openRow(r) {
    drawer({
      title: 'Row details', width: 560,
      body: h('div',
        h('div.d-flex.justify-content-end.mb-2', copyButton(JSON.stringify(r, null, 2), 'Copy row as JSON')),
        kv(columns.map((c) => [c.title, h('div.rd-detail-val', renderValue(c, r, settings), r[c.name] != null && copyButton(String(r[c.name])))]), { cols: 1 })),
    });
  }

  renderBody();
  return {
    el,
    getView: () => ({ columns: visible(), rows: view() }),
    getColumns: () => columns,
    setData(next, tableOptions = opts.tableOptions) {
      rows = next.rows || [];
      columns = tableColumns(next, tableOptions);
      page = 0;
      renderBody();
    },
  };
}

/** Download the current view (after search/filters/sort) as CSV, client-side. */
export function downloadCsv(view, filename = 'results.csv') {
  const esc = (v) => {
    if (v == null) return '';
    const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [view.columns.map((c) => esc(c.title)).join(','), ...view.rows.map((r) => view.columns.map((c) => esc(r[c.name])).join(','))];
  const a = h('a', { href: URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' })), download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
