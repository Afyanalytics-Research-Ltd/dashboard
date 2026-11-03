// Schema browser: cached per data source, alphabetical, instant filter over tables and
// columns, column types when the data source reports them, click to insert.

import { h, icon, mount, debounce } from '../../../../ui/js/dom.js';
import { openMenu, spinner, toastError } from '../../../../ui/js/ui.js';
import { getSchema } from '../../core/api.js';

const PAGE = 200;
const cacheKey = (ds) => `rd:schema:${ds}`;

function readCache(ds) {
  try {
    const raw = sessionStorage.getItem(cacheKey(ds));
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function writeCache(ds, schema) {
  try { sessionStorage.setItem(cacheKey(ds), JSON.stringify({ at: Date.now(), schema })); } catch { /* too big / unavailable */ }
}

/** Redash job errors may be {code, message} objects rather than strings. */
export function errorText(err) {
  const m = err?.body?.message;
  if (m && typeof m === 'object') return m.message || JSON.stringify(m);
  return err?.message || String(err);
}

export const colName = (c) => (typeof c === 'string' ? c : c.name);
export const colType = (c) => (typeof c === 'string' ? null : c.type || null);

function normalise(schema) {
  return (schema || [])
    .map((t) => ({ ...t, columns: [...(t.columns || [])].sort((a, b) => colName(a).localeCompare(colName(b))) }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * schemaBrowser({ onInsert(text), onSchema(schema), onQuickSelect(table), onBuilder(table) })
 * -> { el, load(dataSourceId, {refresh}), getSchema() }
 */
export function schemaBrowser({ onInsert, onSchema, onQuickSelect, onBuilder } = {}) {
  let schema = [];
  let ds = null;
  let filter = '';
  let limit = PAGE;
  const expanded = new Set();
  let loadSeq = 0;

  const listEl = h('div.rd-schema-list');
  const metaEl = h('div.rd-schema-meta');
  const searchEl = h('input.form-control.form-control-sm', {
    type: 'search', placeholder: 'Filter tables and columns…', 'aria-label': 'Filter schema',
    oninput: debounce(() => { filter = searchEl.value.trim().toLowerCase(); limit = PAGE; draw(); }, 120),
  });
  const refreshBtn = h('button.af-icon-btn', { type: 'button', title: 'Refresh schema from the data source', onclick: () => load(ds, { refresh: true }) }, icon('arrow-clockwise'));
  const el = h('div.rd-schema',
    h('div.rd-schema-head', h('div.rd-schema-title', icon('diagram-3', 'me-1'), 'Schema'), refreshBtn),
    h('div.rd-schema-search', icon('search'), searchEl),
    metaEl,
    listEl);

  function matches(t) {
    if (!filter) return { table: true, cols: t.columns };
    const tHit = t.name.toLowerCase().includes(filter);
    const cols = t.columns.filter((c) => colName(c).toLowerCase().includes(filter));
    return tHit || cols.length ? { table: tHit, cols: tHit ? t.columns : cols } : null;
  }

  const mark = (text) => {
    if (!filter) return text;
    const i = text.toLowerCase().indexOf(filter);
    if (i < 0) return text;
    return [text.slice(0, i), h('mark', text.slice(i, i + filter.length)), text.slice(i + filter.length)];
  };

  function draw() {
    const hits = [];
    for (const t of schema) {
      const m = matches(t);
      if (m) hits.push([t, m]);
    }
    const totalCols = schema.reduce((a, t) => a + t.columns.length, 0);
    mount(metaEl, schema.length
      ? `${hits.length === schema.length ? schema.length.toLocaleString() : `${hits.length.toLocaleString()} of ${schema.length.toLocaleString()}`} tables · ${totalCols.toLocaleString()} columns`
      : '');
    if (!schema.length) {
      mount(listEl, h('div.rd-schema-empty', icon('inbox'), h('div', 'No tables found for this data source.'), h('button.af-link-btn', { onclick: () => load(ds, { refresh: true }) }, 'Refresh schema')));
      return;
    }
    if (!hits.length) { mount(listEl, h('div.rd-schema-empty', icon('search'), h('div', `Nothing matches “${filter}”.`))); return; }
    mount(listEl,
      hits.slice(0, limit).map(([t, m]) => {
        const open = expanded.has(t.name) || (!!filter && !m.table);
        const row = h('div.rd-schema-table', { class: open && 'open' },
          h('div.rd-schema-trow',
            h('button.rd-schema-toggle', { type: 'button', 'aria-label': open ? 'Collapse' : 'Expand', onclick: () => { open ? expanded.delete(t.name) : expanded.add(t.name); draw(); } }, icon(open ? 'chevron-down' : 'chevron-right')),
            h('button.rd-schema-name', { type: 'button', title: `Insert ${t.name}`, onclick: () => onInsert?.(t.name) }, icon('table', 'me-1'), mark(t.name)),
            h('span.rd-schema-ncols', String(t.columns.length)),
            h('button.rd-schema-more', {
              type: 'button', title: 'Table actions', 'aria-haspopup': 'menu',
              onclick: (e) => { e.stopPropagation(); openMenu(e.currentTarget, [
                { label: 'Insert table name', icon: 'input-cursor-text', onClick: () => onInsert?.(t.name) },
                { label: 'Insert SELECT … LIMIT 100', icon: 'code', onClick: () => onQuickSelect?.(t) },
                { label: 'Insert all column names', icon: 'list-columns', onClick: () => onInsert?.(t.columns.map(colName).join(', ')) },
                onBuilder && { label: 'Build a query visually…', icon: 'ui-checks-grid', onClick: () => onBuilder(t) },
              ].filter(Boolean)); },
            }, icon('three-dots'))),
          open && h('div.rd-schema-cols', m.cols.map((c) => h('button.rd-schema-col', { type: 'button', title: `Insert ${colName(c)}`, onclick: () => onInsert?.(colName(c)) },
            h('span.rd-schema-colname', mark(colName(c))),
            colType(c) && h('span.rd-schema-coltype', colType(c))))));
        return row;
      }),
      hits.length > limit && h('button.btn.btn-sm.btn-link.w-100', { onclick: () => { limit += PAGE; draw(); } }, `Show ${Math.min(PAGE, hits.length - limit)} more of ${hits.length - limit}`));
  }

  async function load(dataSourceId, { refresh = false } = {}) {
    ds = dataSourceId;
    const seq = ++loadSeq;
    if (!ds) { schema = []; draw(); return schema; }
    const cached = !refresh && readCache(ds);
    if (cached) {
      schema = cached.schema;
      draw();
      onSchema?.(schema);
      mount(metaEl, metaEl.textContent, h('span.rd-schema-cached', { title: new Date(cached.at).toLocaleString() }, ' · cached'));
      return schema;
    }
    mount(listEl, h('div.rd-schema-loading', spinner('sm'), h('span', refresh ? 'Refreshing schema…' : 'Loading schema…')));
    refreshBtn.disabled = true;
    try {
      const fresh = normalise(await getSchema(ds, { refresh }));
      if (seq !== loadSeq) return schema;
      schema = fresh;
      writeCache(ds, schema);
      draw();
      onSchema?.(schema);
    } catch (err) {
      if (err.name === 'AbortError' || seq !== loadSeq) return schema;
      mount(metaEl);
      mount(listEl, h('div.rd-schema-empty.rd-schema-error', icon('exclamation-triangle'),
        h('div.fw-semibold', errorText(err)),
        h('div.small', 'The data source could not list its tables. You can still write and run queries — autocomplete just won’t know table names.'),
        h('button.af-link-btn', { onclick: () => load(ds, { refresh: true }) }, icon('arrow-clockwise', 'me-1'), 'Try again')));
      if (refresh) toastError(new Error(errorText(err)), 'Could not refresh the schema');
    } finally {
      if (seq === loadSeq) refreshBtn.disabled = false;
    }
    return schema;
  }

  return { el, load, getSchema: () => schema, focus: () => searchEl.focus() };
}
