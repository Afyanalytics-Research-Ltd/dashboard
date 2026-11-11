// Data Explorer: databases → schemas → objects → object detail (columns, preview, DDL, signature, storage).

import { h, icon, mount, empty, errorView, tabs, skeleton, copyButton, kv, spinner, toast, chip } from '../lib/ui.js';
import { api, waitForQuery } from '../lib/api.js';
import { state, hasAccountUsage } from '../lib/store.js';
import * as fmt from '../lib/fmt.js';
import * as x from '../lib/explorercore.js';
import { href } from '../../../ui/js/router.js';
import { debounce } from '../../../ui/js/dom.js';

// Module-level caches survive navigation inside the SPA (cheap, per tab).
const cache = new Map();
const cached = (key, fn) => {
  if (!cache.has(key)) cache.set(key, fn().catch((err) => { cache.delete(key); throw err; }));
  return cache.get(key);
};
let storageIndex = null;

const roleCtx = () => ({ role: state.context.role });

export default async function render(ctx) {
  ctx.setTitle('Data Explorer');
  const sel = { database: ctx.query.get('database') || '', schema: ctx.query.get('schema') || '', object: ctx.query.get('object') || '', kind: ctx.query.get('kind') || '' };
  let dbFilter = '';
  let objFilter = '';
  let objSort = 'name';
  let objectsData = null;

  const dbPane = h('div.wsx-pane');
  const schemaPane = h('div.wsx-pane');
  const detailPane = h('div.wsx-pane.wsx-detail');
  mount(ctx.el, h('div.wsx-explorer', dbPane, schemaPane, detailPane));

  const syncUrl = () => ctx.setQuery({ database: sel.database || null, schema: sel.schema || null, object: sel.object || null, kind: sel.kind || null });

  // ---------------------------------------------------------------- databases

  let databases = [];
  async function loadDatabases(refresh = false) {
    mount(dbPane, paneHead('Databases', 'database'), skeleton(10, 22));
    try {
      if (refresh) cache.delete(`dbs:${state.context.role}`);
      const res = await cached(`dbs:${state.context.role}`, () => api.databases(roleCtx(), refresh));
      databases = res.databases || [];
      renderDatabases();
    } catch (err) {
      if (err.name !== 'AbortError') mount(dbPane, paneHead('Databases', 'database'), errorView(err, () => loadDatabases(true)));
    }
  }

  function renderDatabases() {
    const q = dbFilter.toLowerCase();
    const list = databases.filter((d) => !q || d.name.toLowerCase().includes(q));
    const input = h('input.ws-input', { type: 'search', placeholder: 'Filter databases', value: dbFilter, oninput: debounce((e) => { dbFilter = e.target.value; renderDatabases(); input.focus(); }, 120) });
    mount(dbPane,
      paneHead('Databases', 'database', h('button.ws-btn.ws-btn-sm.ws-btn-ghost.ws-btn-icon', { type: 'button', title: 'Refresh', onclick: () => loadDatabases(true) }, icon('arrow-clockwise'))),
      h('div.ws-search.wsx-pane-search', icon('search'), input),
      h('div.wsx-list', list.length ? list.map((d) => h(`button.wsx-item${d.name === sel.database ? '.on' : ''}`, { type: 'button', title: d.comment || d.name,
        onclick: () => { sel.database = d.name; sel.schema = ''; sel.object = ''; sel.kind = ''; syncUrl(); renderDatabases(); loadSchemas(); renderDetail(); } },
      icon(d.kind === 'APPLICATION' ? 'app' : d.kind === 'IMPORTED DATABASE' ? 'box-arrow-in-down' : d.kind === 'PERSONAL DATABASE' ? 'person' : 'database'),
      h('span.ws-ellipsis', d.name), d.kind && d.kind !== 'STANDARD' && h('span.ws-tag', d.kind.toLowerCase().replace(' database', '')))) : h('div.ws-muted.wsx-pad', 'No databases.')));
  }

  // ---------------------------------------------------------------- schemas + objects

  let schemas = [];
  async function loadSchemas() {
    if (!sel.database) { mount(schemaPane, paneHead('Schemas', 'folder2'), empty({ icon: 'arrow-left', title: 'Pick a database' })); return; }
    mount(schemaPane, paneHead(sel.database, 'folder2'), skeleton(8, 22));
    try {
      const res = await cached(`schemas:${state.context.role}:${sel.database}`, () => api.schemas(sel.database, roleCtx()));
      schemas = res.schemas || [];
      if (sel.schema && !schemas.some((s) => s.name === sel.schema)) sel.schema = '';
      if (sel.schema) await loadObjects(); else renderSchemas();
    } catch (err) {
      if (err.name !== 'AbortError') mount(schemaPane, paneHead(sel.database, 'folder2'), errorView(err, loadSchemas));
    }
  }

  function renderSchemas() {
    mount(schemaPane, paneHead(sel.database, 'folder2', null, `${schemas.length} schemas`),
      h('div.wsx-list', schemas.length ? schemas.map((s) => h('button.wsx-item', { type: 'button', title: s.comment || s.name,
        onclick: () => { sel.schema = s.name; sel.object = ''; syncUrl(); loadObjects(); renderDetail(); } },
      icon('folder2'), h('span.ws-ellipsis', s.name), icon('chevron-right', 'wsx-chev'))) : h('div.ws-muted.wsx-pad', 'No schemas visible to this role.')));
  }

  async function loadObjects() {
    mount(schemaPane, paneHead(`${sel.database}.${sel.schema}`, 'folder2-open', backBtn()), skeleton(10, 22));
    try {
      const res = await cached(`objects:${state.context.role}:${sel.database}:${sel.schema}`, () => api.objects(sel.database, sel.schema, roleCtx()));
      objectsData = res.objects || {};
      renderObjects();
      if (sel.object) renderDetail();
    } catch (err) {
      if (err.name !== 'AbortError') mount(schemaPane, paneHead(`${sel.database}.${sel.schema}`, 'folder2-open', backBtn()), errorView(err, loadObjects));
    }
  }

  function backBtn() {
    return h('button.ws-btn.ws-btn-sm.ws-btn-ghost.ws-btn-icon', { type: 'button', title: 'All schemas', onclick: () => { sel.schema = ''; sel.object = ''; syncUrl(); renderSchemas(); renderDetail(); } }, icon('arrow-left'));
  }

  function renderObjects() {
    const groups = x.groupObjects(objectsData, { q: objFilter, sort: objSort });
    const totals = x.objectTotals(objectsData);
    const input = h('input.ws-input', { type: 'search', placeholder: 'Filter objects', value: objFilter,
      oninput: debounce((e) => { objFilter = e.target.value; renderObjects(); const i = schemaPane.querySelector('input'); i?.focus(); i?.setSelectionRange(i.value.length, i.value.length); }, 120) });
    mount(schemaPane,
      paneHead(`${sel.database}.${sel.schema}`, 'folder2-open', backBtn(), `${fmt.plural(totals.count, 'object')} · ${fmt.compact(totals.rows)} rows · ${fmt.bytes(totals.bytes)}`),
      h('div.ws-row.wsx-pane-search', h('div.ws-search', { style: { flex: 1 } }, icon('search'), input),
        h('select.ws-select', { title: 'Sort', onchange: (e) => { objSort = e.target.value; renderObjects(); } },
          [['name', 'Name'], ['rows', 'Rows'], ['size', 'Size'], ['altered', 'Recently altered']].map(([v, l]) => h('option', { value: v, selected: v === objSort }, l)))),
      h('div.wsx-list', groups.length ? groups.map((g) => h('details.wsx-group', { open: true },
        h('summary', icon(g.icon), g.label, h('span.ws-tag', g.count)),
        g.items.map((o) => h(`button.wsx-item.wsx-obj${o.name === sel.object && g.kind === sel.kind ? '.on' : ''}`, { type: 'button', title: o.comment || o.name,
          onclick: () => { sel.object = o.name; sel.kind = g.kind; syncUrl(); renderObjects(); renderDetail(); } },
        h('span.ws-ellipsis', o.name),
        o.rows != null && h('span.wsx-meta', fmt.compact(o.rows)),
        o.bytes != null && h('span.wsx-meta', fmt.bytes(o.bytes)),
        o.params && h('span.wsx-meta', `(${o.params.length})`),
        o.state && h('span.wsx-meta', String(o.state).toLowerCase()))))) : h('div.ws-muted.wsx-pad', objFilter ? 'No objects match.' : 'This schema is empty (for this role).')));
  }

  // ---------------------------------------------------------------- detail

  function findObject() {
    if (!objectsData || !sel.object) return null;
    const pick = (kind) => (objectsData[kind] || []).find((o) => o.name === sel.object);
    if (sel.kind) { const o = pick(sel.kind); if (o) return { kind: sel.kind, obj: o }; }
    for (const kind of Object.keys(objectsData)) { const o = pick(kind); if (o) return { kind, obj: o }; }
    return null;
  }

  function renderDetail() {
    if (!sel.database) { mount(detailPane, overviewEmpty()); return; }
    if (!sel.schema) { mount(detailPane, empty({ icon: 'folder2', title: sel.database, text: 'Pick a schema to browse its tables, views, procedures, stages and tasks.' })); return; }
    const found = findObject();
    if (!found) { mount(detailPane, schemaSummary()); return; }
    renderObject(found.kind, found.obj);
  }

  function overviewEmpty() {
    return empty({ icon: 'database', title: 'Explore your Snowflake data', text: 'Browse databases and schemas on the left. Every table shows its columns, a live preview, DDL and storage footprint.' });
  }

  function schemaSummary() {
    const totals = x.objectTotals(objectsData);
    const groups = x.groupObjects(objectsData);
    const biggest = [...(objectsData?.TABLE || [])].sort((a, b) => (Number(b.bytes) || 0) - (Number(a.bytes) || 0)).slice(0, 8);
    const maxB = Math.max(1, ...biggest.map((t) => Number(t.bytes) || 0));
    return h('div.ws-col', { style: { gap: '16px' } },
      h('div', h('div.ws-card-sub', 'Schema'), h('h2.wsx-title', `${sel.database}.${sel.schema}`)),
      h('div.ws-grid.ws-g-3',
        statTile('Objects', fmt.fmtInt(totals.count), 'boxes'), statTile('Rows', fmt.compact(totals.rows), 'list-ol'), statTile('Storage', fmt.bytes(totals.bytes), 'hdd')),
      h('div.ws-row.ws-wrap', { style: { gap: '6px' } }, groups.map((g) => chip(g.label, { count: g.count }))),
      biggest.length ? h('div.ws-card.ws-card-pad', h('h3.ws-card-title', { style: { marginBottom: '10px' } }, icon('hdd-stack'), 'Largest tables'),
        h('div.ws-col', { style: { gap: '8px' } }, biggest.map((t) => h('button.wsx-bigrow', { type: 'button', onclick: () => { sel.object = t.name; sel.kind = 'TABLE'; syncUrl(); renderObjects(); renderDetail(); } },
          h('span.ws-ellipsis', t.name), h('div.ws-bar', { style: { flex: 1 } }, h('span', { style: { width: `${((Number(t.bytes) || 0) / maxB) * 100}%` } })), h('span.ws-mono.ws-muted', fmt.bytes(t.bytes)))))) : null);
  }

  function renderObject(kind, obj) {
    const meta = x.KIND_META[kind] || { label: kind, icon: 'box' };
    const fq = x.quoteIdent(sel.database, sel.schema, obj.name);
    const isRel = ['TABLE', 'VIEW', 'MATERIALIZED_VIEW', 'DYNAMIC_TABLE', 'EXTERNAL_TABLE', 'EVENT_TABLE'].includes(kind);
    const isRoutine = kind === 'PROCEDURE' || kind === 'FUNCTION';
    const tabList = [
      isRel && { key: 'columns', label: 'Columns', icon: 'layout-three-columns' },
      isRel && { key: 'preview', label: 'Preview', icon: 'table' },
      isRoutine && { key: 'signature', label: 'Signature', icon: 'braces' },
      { key: 'ddl', label: 'DDL', icon: 'code-slash' },
      kind === 'TABLE' && hasAccountUsage() && { key: 'storage', label: 'Storage', icon: 'hdd-stack' },
    ].filter(Boolean);
    const body = h('div');
    const ddlType = kind === 'EVENT_TABLE' ? 'TABLE' : kind;
    const ddlName = isRoutine ? x.procedureDdlName(sel.database, sel.schema, obj.name, obj.params) : fq;
    const show = (key) => {
      if (key === 'columns') return showColumns(body, obj);
      if (key === 'preview') return showPreview(body, fq);
      if (key === 'signature') return showSignature(body, kind, obj);
      if (key === 'ddl') return showDdl(body, ddlType, ddlName);
      if (key === 'storage') return showStorage(body, obj);
      return null;
    };
    const headFacts = [
      obj.rows != null && ['Rows', fmt.fmtInt(obj.rows)], obj.bytes != null && ['Size', fmt.bytes(obj.bytes)],
      obj.last_altered && ['Last altered', h('span', { title: fmt.dateTime(obj.last_altered) }, fmt.relTime(obj.last_altered))],
      obj.clustering_key && ['Clustering key', h('span.ws-mono', obj.clustering_key)],
      obj.retention_days != null && ['Time travel', `${obj.retention_days} day${Number(obj.retention_days) === 1 ? '' : 's'}`],
      obj.data_type && ['Returns', h('span.ws-mono', obj.data_type)], obj.language && ['Language', obj.language],
      obj.schedule && ['Schedule', obj.schedule], obj.warehouse && ['Warehouse', obj.warehouse], obj.state && ['State', obj.state],
      obj.url && ['URL', h('span.ws-mono', obj.url)], obj.comment && ['Comment', obj.comment],
    ].filter(Boolean);
    const sqlFor = isRel ? `SELECT *\nFROM ${fq}\nLIMIT 100;` : isRoutine && kind === 'PROCEDURE' ? `CALL ${fq}(${(obj.params || []).map((p) => `/* ${p.name} ${p.type} */ NULL`).join(', ')});` : `-- ${fq}`;
    mount(detailPane,
      h('div.wsx-objhead',
        h('div.ws-row.ws-wrap', { style: { gap: '8px' } }, h('span.ws-badge.ws-sev-INFO', icon(meta.icon), meta.label.replace(/s$/, '').toLowerCase()),
          obj.transient && h('span.ws-badge.ws-sev-LOW', 'transient'), h('span.ws-muted.ws-mono', `${sel.database}.${sel.schema}`)),
        h('h2.wsx-title', obj.name, copyButton(fq)),
        h('div.ws-row.ws-wrap', { style: { gap: '6px' } },
          h('a.ws-btn.ws-btn-sm.ws-btn-primary', { href: href('/worksheets', { sql: sqlFor }) }, icon('code-square'), isRel ? 'Query in worksheet' : 'Open in worksheet'),
          kind === 'PROCEDURE' && h('a.ws-btn.ws-btn-sm', { href: href('/procedures', { database: sel.database, schema: sel.schema, name: obj.name }) }, icon('play-fill'), 'Run procedure')),
        headFacts.length ? h('div.wsx-facts', kv(headFacts)) : null),
      tabs(tabList, tabList[0].key, show),
      body);
    show(tabList[0].key);
  }

  // ---------------------------------------------------------------- tabs

  async function showColumns(el, obj) {
    mount(el, skeleton(8, 20));
    try {
      const res = await cached(`cols:${state.context.role}:${sel.database}:${sel.schema}:${obj.name}`, () => api.columns(sel.database, sel.schema, obj.name, roleCtx()));
      const cols = res.columns || [];
      let q = '';
      const listEl = h('div');
      const draw = () => {
        const shown = cols.filter((c) => !q || `${c.column_name} ${c.data_type} ${c.comment || ''}`.toLowerCase().includes(q));
        mount(listEl, h('div.ws-table-wrap', { style: { maxHeight: '62vh' } }, h('table.ws-table',
          h('thead', h('tr', h('th.num', '#'), h('th', 'Column'), h('th', 'Type'), h('th', 'Null'), h('th', 'Default'), h('th', 'Comment'))),
          h('tbody', shown.map((c) => h('tr', h('td.num', c.ordinal_position), h('td.strong', h('span.ws-mono', c.column_name)),
            h('td', h('span.ws-tag', typeLabel(c))), h('td', c.is_nullable === 'YES' ? h('span.ws-muted', 'yes') : h('b', 'NOT NULL')),
            h('td', c.column_default ? h('span.ws-mono', fmt.truncate(c.column_default, 40)) : h('span.ws-muted', '—')),
            h('td', c.comment || h('span.ws-muted', '—'))))))));
      };
      mount(el, h('div.ws-row.ws-between.ws-wrap', { style: { marginBottom: '10px' } },
        h('div.ws-search', { style: { flex: '1 1 260px', maxWidth: '360px' } }, icon('search'), h('input.ws-input', { type: 'search', placeholder: `Filter ${cols.length} columns`, oninput: (e) => { q = e.target.value.toLowerCase(); draw(); } })),
        copyButton(() => cols.map((c) => c.column_name).join(',\n'), 'Column list')), listEl);
      draw();
    } catch (err) { if (err.name !== 'AbortError') mount(el, errorView(err, () => showColumns(el, obj))); }
  }

  async function showPreview(el, fq) {
    const ac = new AbortController();
    ctx.cleanup(() => ac.abort());
    const status = h('span.ws-live', spinner(), 'Running preview…');
    let qid = null;
    mount(el, h('div.ws-row', { style: { marginBottom: '10px' } }, status,
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', onclick: () => { ac.abort(); if (qid) api.cancel(qid).catch(() => {}); mount(status, 'Cancelled'); } }, 'Cancel')), skeleton(8, 20));
    const t0 = Date.now();
    try {
      const started = await api.submit({ sql: `SELECT * FROM ${fq} LIMIT 100`, context: state.context, kind: 'preview' }, { signal: ac.signal });
      qid = started.query_id;
      const st = await waitForQuery(qid, { signal: ac.signal, onTick: ({ status: s }) => mount(status, spinner(), `${String(s).toLowerCase().replace(/_/g, ' ')} · ${fmt.ms(Date.now() - t0)}`) });
      if (!st.ok) throw Object.assign(new Error(st.error?.message || 'Preview failed'), { code: 'query' });
      const res = await api.results(qid, 0, 100, { signal: ac.signal });
      mount(el, h('div.ws-row.ws-between.ws-wrap', { style: { marginBottom: '10px' } },
        h('span.ws-muted', { style: { fontSize: '12px' } }, `${fmt.plural(res.returned, 'row')} · ${fmt.ms(Date.now() - t0)} · `, h('a', { href: '#', onclick: (e) => { e.preventDefault(); import('../lib/query.js').then((m) => m.openQuery(qid)); } }, 'inspect query')),
        h('a.ws-btn.ws-btn-sm', { href: api.exportUrl(qid, 'csv') }, icon('download'), 'CSV')),
      resultGrid(res.columns, res.rows));
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(el, errorView(err, () => showPreview(el, fq)));
    }
  }

  function showSignature(el, kind, obj) {
    const params = obj.params?.length ? obj.params : x.parseSignature(obj.argument_signature);
    mount(el, h('div.ws-col', { style: { gap: '14px' } },
      h('pre.ws-code', `${obj.name}${obj.argument_signature || '()'}\nRETURNS ${obj.data_type || '?'}${obj.language ? `\nLANGUAGE ${obj.language}` : ''}`),
      params.length ? h('div.ws-table-wrap', h('table.ws-table', h('thead', h('tr', h('th', 'Parameter'), h('th', 'Type'), h('th', 'Default'))),
        h('tbody', params.map((p) => h('tr', h('td.strong.ws-mono', p.name), h('td', h('span.ws-tag', p.type)), h('td', p.has_default ? 'optional' : h('b', 'required'))))))) : h('div.ws-muted', 'No parameters.'),
      kind === 'PROCEDURE' && h('div', h('a.ws-btn.ws-btn-primary', { href: href('/procedures', { database: sel.database, schema: sel.schema, name: obj.name }) }, icon('play-fill'), 'Run with a form'))));
  }

  async function showDdl(el, type, name) {
    mount(el, skeleton(10, 16));
    try {
      const res = await cached(`ddl:${state.context.role}:${type}:${name}`, () => api.ddl(type, name, roleCtx()));
      const ddl = res.ddl || '';
      mount(el, h('div.ws-row', { style: { marginBottom: '8px', gap: '6px' } }, copyButton(ddl, 'Copy DDL'),
        h('a.ws-btn.ws-btn-sm', { href: href('/worksheets', { sql: ddl }) }, icon('code-square'), 'Open in worksheet')),
      h('pre.ws-code.wsx-ddl', { style: { maxHeight: '64vh' } }, highlightSql(ddl)));
    } catch (err) { if (err.name !== 'AbortError') mount(el, errorView(err, () => showDdl(el, type, name))); }
  }

  async function showStorage(el, obj) {
    mount(el, skeleton(4, 20));
    try {
      if (!storageIndex) {
        storageIndex = api.costStorage(500).then((res) => new Map((res.tables || []).map((t) => [String(t.path_to_object).toUpperCase(), t])))
          .catch((err) => { storageIndex = null; throw err; });
      }
      const idx = await storageIndex;
      const t = idx.get(`${sel.database}.${sel.schema}.${obj.name}`.toUpperCase());
      if (!t) { mount(el, empty({ icon: 'hdd-stack', title: 'Not among the 500 largest tables', text: 'Storage metrics are shown for the heaviest tables in the account.' })); return; }
      const parts = [['Active', t.active_bytes, 'var(--ws-c1)'], ['Time travel', t.time_travel_bytes, 'var(--ws-c7)'], ['Fail-safe', t.failsafe_bytes, 'var(--ws-c2)'], ['Clones', t.retained_for_clone_bytes, 'var(--ws-c3)']];
      const total = Math.max(1, Number(t.total_bytes) || parts.reduce((a, p) => a + (Number(p[1]) || 0), 0));
      mount(el, h('div.ws-col', { style: { gap: '14px' } },
        h('div.ws-grid.ws-g-3', statTile('Total footprint', fmt.bytes(t.total_bytes), 'hdd-stack'), statTile('Est. monthly', fmt.usd(t.est_monthly_usd), 'cash-coin'),
          statTile('Retention overhead', t.overhead_multiple != null ? `${fmt.fmtNum(t.overhead_multiple)}×` : '—', 'clock-history')),
        h('div.ws-waterfall', { style: { height: '16px' } }, parts.map(([l, v, c]) => (Number(v) ? h('span', { title: `${l}: ${fmt.bytes(v)}`, style: { width: `${(Number(v) / total) * 100}%`, background: c } }) : null))),
        h('div.ws-legend', parts.map(([l, v, c]) => h('span', h('i', { style: { background: c } }), `${l} `, h('b.ws-mono', fmt.bytes(v))))),
        Number(t.overhead_multiple) > 1 && h('div.ws-insight', h('span.ws-badge.ws-sev-MEDIUM', 'tip'), h('h6', 'Time travel and fail-safe outweigh the data'),
          h('p', 'Lower DATA_RETENTION_TIME_IN_DAYS, or make the table TRANSIENT if it can be reloaded — transient tables carry no fail-safe.'))));
    } catch (err) { if (err.name !== 'AbortError') mount(el, errorView(err, () => showStorage(el, obj))); }
  }

  // ---------------------------------------------------------------- boot

  mount(detailPane, overviewEmpty());
  await loadDatabases();
  if (sel.database) await loadSchemas(); else mount(schemaPane, paneHead('Schemas', 'folder2'), empty({ icon: 'arrow-left', title: 'Pick a database' }));
  renderDetail();
}

function paneHead(title, ic, tool, sub) {
  return h('div.wsx-pane-head', h('div.ws-col', { style: { gap: '1px', minWidth: 0, flex: 1 } },
    h('div.ws-row', { style: { gap: '7px' } }, icon(ic), h('b.ws-ellipsis', title)), sub && h('span.ws-muted', { style: { fontSize: '11.5px' } }, sub)), tool);
}

function statTile(label, value, ic) {
  return h('div.ws-card.ws-stat', h('div.ws-stat-label', icon(ic), label), h('div.ws-stat-value', { style: { fontSize: '22px' } }, value));
}

function typeLabel(c) {
  const t = c.data_type || '';
  if (t === 'NUMBER' && c.numeric_precision != null) return `NUMBER(${c.numeric_precision},${c.numeric_scale ?? 0})`;
  if (t === 'TEXT' && c.character_maximum_length) return c.character_maximum_length >= 16777216 ? 'VARCHAR' : `VARCHAR(${c.character_maximum_length})`;
  return t;
}

const KW = /\b(SELECT|FROM|WHERE|AND|OR|NOT|NULL|AS|ON|JOIN|LEFT|RIGHT|INNER|OUTER|GROUP|BY|ORDER|HAVING|LIMIT|QUALIFY|WITH|UNION|ALL|CREATE|OR|REPLACE|TABLE|VIEW|PROCEDURE|FUNCTION|RETURNS|LANGUAGE|EXECUTE|OWNER|CALLER|BEGIN|END|DECLARE|IF|THEN|ELSE|ELSEIF|CASE|WHEN|RETURN|INSERT|INTO|VALUES|UPDATE|SET|DELETE|MERGE|USING|DEFAULT|COMMENT|CLUSTER|TRANSIENT|TEMPORARY|VARCHAR|NUMBER|FLOAT|BOOLEAN|VARIANT|ARRAY|OBJECT|DATE|TIMESTAMP_LTZ|TIMESTAMP_NTZ|AUTOINCREMENT|START|INCREMENT|IS|IN|EXISTS|DISTINCT|COUNT|SUM|MIN|MAX|AVG|FOR|DO|LET|CURSOR|RESULTSET|EXCEPTION|OTHER)\b/gi;

/** Minimal, safe SQL highlighting (text nodes only — no innerHTML). */
export function highlightSql(sql) {
  const out = [];
  const re = /('(?:[^'\\]|\\.|'')*'|--[^\n]*|\/\*[\s\S]*?\*\/|\$\$[\s\S]*?\$\$)/g;
  let last = 0;
  let m;
  const pushCode = (text) => {
    let i = 0;
    let k;
    KW.lastIndex = 0;
    while ((k = KW.exec(text))) {
      if (k.index > i) out.push(text.slice(i, k.index));
      out.push(h('span.wsx-kw', k[0]));
      i = k.index + k[0].length;
    }
    if (i < text.length) out.push(text.slice(i));
  };
  while ((m = re.exec(sql))) {
    if (m.index > last) pushCode(sql.slice(last, m.index));
    const tok = m[0];
    out.push(h(tok.startsWith("'") ? 'span.wsx-str' : tok.startsWith('$$') ? 'span.wsx-body' : 'span.wsx-cm', tok));
    last = m.index + tok.length;
  }
  if (last < sql.length) pushCode(sql.slice(last));
  return out;
}

/** Compact result table for previews and procedure output. */
export function resultGrid(columns, rows, { maxHeight = '60vh' } = {}) {
  if (!columns?.length) return empty({ icon: 'table', title: 'No columns returned' });
  if (!rows?.length) return h('div', h('div.ws-muted', { style: { marginBottom: '8px' } }, '0 rows'),
    h('div.ws-row.ws-wrap', { style: { gap: '6px' } }, columns.map((c) => h('span.ws-tag', c.name))));
  return h('div.ws-table-wrap.wsx-grid', { style: { maxHeight } }, h('table.ws-table',
    h('thead', h('tr', h('th.num', '#'), columns.map((c) => h(`th${c.family === 'number' ? '.num' : ''}`, { title: c.label || c.type }, c.name, h('div.wsx-coltype', (c.label || c.type || '').toLowerCase()))))),
    h('tbody', rows.map((r, i) => h('tr', h('td.num.ws-muted', i + 1), columns.map((c, j) => cell(r[j], c)))))));
}

function cell(v, c) {
  const num = c.family === 'number';
  if (v === null || v === undefined) return h(`td${num ? '.num' : ''}`, h('span.wsx-null', 'NULL'));
  if (typeof v === 'object' && '$json' in v) {
    const text = JSON.stringify(v.$json);
    return h('td', h('span.ws-mono.wsx-json', { title: JSON.stringify(v.$json, null, 2).slice(0, 4000) }, fmt.truncate(text, 80)));
  }
  if (typeof v === 'object' && '$binary' in v) return h('td', h('span.ws-mono.ws-muted', fmt.truncate(`0x${v.$binary}`, 40)));
  if (typeof v === 'boolean') return h('td', h('span.ws-tag', String(v)));
  if (num) return h('td.num', typeof v === 'number' ? v.toLocaleString('en-US', { maximumFractionDigits: 6 }) : String(v));
  if (c.family === 'timestamp' || c.family === 'date') return h('td', h('span.ws-mono', String(v).replace('T', ' ')));
  const s = String(v);
  return h('td', { title: s.length > 80 ? s.slice(0, 2000) : null }, fmt.truncate(s, 120));
}

export { toast };
