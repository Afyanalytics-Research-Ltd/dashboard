// Item viewer used by the node panel and execution inspector:
// Table / JSON / Schema modes, search, paging, binary summary, and drag-to-map —
// schema fields and table headers drag as {{ expression }} text into any input.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { jsonView, copyButton } from '../../../ui/js/ui.js';
import { fmtBytes } from '../../../ui/js/format.js';
import { schemaOf, itemsTable } from '../core/execution.js';

const PAGE = 25;

/** Expression that reads `path` (array of keys) from `node` (null = current input). */
export function exprFor(path, node) {
  const access = path.map((k) => (/^\d+$/.test(k) ? `[${k}]` : /^[A-Za-z_$][\w$]*$/.test(k) ? `.${k}` : `[${JSON.stringify(k)}]`)).join('');
  return node ? `{{ $('${node.replace(/'/g, "\\'")}').item.json${access} }}` : `{{ $json${access} }}`;
}

function typeIcon(types) {
  const t = types[0];
  const map = { string: 'fonts', number: 'hash', boolean: 'toggle-on', object: 'braces', array: 'list-ol', null: 'slash-circle' };
  return icon(map[t] || 'question', 'n8s-type-ic');
}

function dragAttrs(expr) {
  return {
    draggable: 'true',
    title: `Drag into a field to insert ${expr}`,
    ondragstart: (e) => {
      e.dataTransfer.setData('text/plain', expr);
      e.dataTransfer.effectAllowed = 'copy';
      document.body.classList.add('n8s-mapping');
    },
    ondragend: () => document.body.classList.remove('n8s-mapping'),
  };
}

const cellText = (v) => {
  if (v == null) return h('span.n8s-null', v === null ? 'null' : '—');
  if (typeof v === 'object') return h('code.n8s-cell-json', JSON.stringify(v).slice(0, 160));
  if (typeof v === 'boolean') return h('span', { class: v ? 'text-success' : 'text-danger' }, String(v));
  const s = String(v);
  return s.length > 200 ? `${s.slice(0, 200)}…` : s;
};

/**
 * dataView({ items, node, title, mode, emptyText, actions, mappable })
 * Returns an element; `node` is the source node name used for drag expressions.
 */
export function dataView({ items = [], node = null, title, mode, emptyText = 'No data', actions, mappable = true, outputs } = {}) {
  let current = mode || (items.length > 1 ? 'table' : 'json');
  let query = '';
  let page = 0;
  let outputIndex = 0;
  let list = outputs ? outputs[0]?.items || [] : items;
  const body = h('div.n8s-dv-body');
  const info = h('span.n8s-dv-info');
  const modes = h('div.btn-group.btn-group-sm.n8s-dv-modes');

  const filtered = () => {
    if (!query) return list;
    const q = query.toLowerCase();
    return list.filter((it) => JSON.stringify(it.json ?? it).toLowerCase().includes(q));
  };

  function renderTable(rows) {
    const { columns } = itemsTable(rows);
    if (!columns.length) return h('div.n8s-dv-empty', 'Items have no fields.');
    const slice = rows.slice(page * PAGE, page * PAGE + PAGE);
    return h('div.n8s-dv-table-wrap', h('table.n8s-dv-table',
      h('thead', h('tr', h('th.n8s-dv-idx', '#'), columns.map((c) => h('th', mappable ? dragAttrs(exprFor([c], node)) : {}, mappable && icon('grip-vertical', 'n8s-grip'), c)))),
      h('tbody', slice.map((it, i) => h('tr', h('td.n8s-dv-idx', String(page * PAGE + i)), columns.map((c) => h('td', cellText(it.json?.[c]))))))));
  }

  function renderSchema(rows) {
    const fields = schemaOf(rows);
    if (!fields.length) return h('div.n8s-dv-empty', 'Items have no fields.');
    return h('div.n8s-dv-schema', fields.map((f) => {
      const path = f.path.split('.');
      const expr = exprFor(path, node);
      return h('div.n8s-dv-field', { style: { paddingLeft: `${(f.depth - 1) * 16 + 6}px` }, ...(mappable ? dragAttrs(expr) : {}) },
        mappable && icon('grip-vertical', 'n8s-grip'), typeIcon(f.types), h('span.n8s-dv-key', path[path.length - 1]),
        f.sample !== undefined && h('span.n8s-dv-sample', String(f.sample).slice(0, 60)),
        f.count < rows.length && h('span.n8s-dv-partial', { title: `present in ${f.count} of ${rows.length} items` }, `${f.count}/${rows.length}`),
        copyButton(expr));
    }));
  }

  function renderJson(rows) {
    const slice = rows.slice(page * PAGE, page * PAGE + PAGE);
    return h('div', jsonView(slice.map((it) => (it.binary ? { json: it.json, binary: Object.fromEntries(Object.entries(it.binary).map(([k, b]) => [k, `${b.fileName || k} (${b.mimeType || '?'}, ${fmtBytes(b.fileSize ? Number(String(b.fileSize).replace(/\D/g, '')) : null)})`])) } : it.json ?? it)), { open: 3 }));
  }

  function renderBinary(rows) {
    const bins = rows.flatMap((it, i) => Object.entries(it.binary || {}).map(([k, b]) => ({ i, k, b })));
    if (!bins.length) return h('div.n8s-dv-empty', 'No binary data.');
    return h('div.n8s-dv-binary', bins.map(({ i, k, b }) => h('div.n8s-dv-bin', icon(/image/.test(b.mimeType) ? 'file-image' : /pdf/.test(b.mimeType) ? 'file-pdf' : 'file-earmark'),
      h('div', h('b', b.fileName || k), h('div.small.text-muted', `item ${i} · ${k} · ${b.mimeType || 'unknown type'} · ${b.fileSize || ''}`)))));
  }

  function render() {
    const rows = filtered();
    const hasBinary = list.some((it) => it.binary && Object.keys(it.binary).length);
    mount(modes, [['table', 'table', 'Table'], ['json', 'braces', 'JSON'], ['schema', 'diagram-2', 'Schema'], hasBinary && ['binary', 'file-earmark-binary', 'Binary']].filter(Boolean)
      .map(([m, ic, label]) => h('button.btn', { type: 'button', class: current === m ? 'btn-secondary' : 'btn-outline-secondary', title: label, onclick: () => { current = m; page = 0; render(); } }, icon(ic), h('span.d-none.d-xl-inline.ms-1', label))));
    mount(info, `${rows.length}${query ? ` of ${list.length}` : ''} item${rows.length === 1 ? '' : 's'}`);
    if (!list.length) { mount(body, h('div.n8s-dv-empty', icon('inbox', 'me-1'), emptyText)); return; }
    const content = current === 'table' ? renderTable(rows) : current === 'schema' ? renderSchema(rows) : current === 'binary' ? renderBinary(rows) : renderJson(rows);
    const pages = Math.ceil(rows.length / PAGE);
    mount(body, content, (current === 'table' || current === 'json') && pages > 1 && h('div.n8s-dv-pager',
      h('button.btn.btn-sm.btn-link', { disabled: page === 0, onclick: () => { page--; render(); } }, icon('chevron-left')),
      h('span', `${page * PAGE + 1}–${Math.min(rows.length, (page + 1) * PAGE)} of ${rows.length}`),
      h('button.btn.btn-sm.btn-link', { disabled: page >= pages - 1, onclick: () => { page++; render(); } }, icon('chevron-right'))));
  }

  const search = h('input.form-control.form-control-sm.n8s-dv-search', { type: 'search', placeholder: 'Filter items…' });
  search.addEventListener('input', debounce(() => { query = search.value.trim(); page = 0; render(); }, 150));
  const outputTabs = outputs && outputs.length > 1 && h('div.n8s-dv-outputs', outputs.map((o, i) => h('button.n8s-chip-sm', {
    type: 'button', class: i === outputIndex && 'active',
    onclick: (e) => { outputIndex = i; list = outputs[i].items || []; page = 0; e.currentTarget.parentElement.querySelectorAll('button').forEach((b, j) => b.classList.toggle('active', j === i)); render(); },
  }, o.label || `Output ${i}`, h('span.n8s-chip-count', String(o.items?.length || 0)))));

  render();
  return h('div.n8s-dv',
    h('div.n8s-dv-head', title && h('div.n8s-dv-title', title), outputTabs, h('div.flex-grow-1'), info, search, modes, actions),
    body);
}
