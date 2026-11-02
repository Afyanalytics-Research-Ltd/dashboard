// API explorer: every Airflow REST API v2 endpoint from the live OpenAPI spec,
// callable through the dashboard proxy with typed inputs, generated request
// bodies, timing, snippets and history. Deep link: ?op=<operationId>.

import { h, icon, mount, debounce } from '../core/dom.js';
import { request, config, qs } from '../core/api.js';
import {
  pageHeader, card, jsonEditor, jsonView, codeBlock, copyButton, tabs, confirmDialog, errorState,
  skeleton, toast, chips, spinner,
} from '../core/ui.js';
import { fmtBytes, fmtDuration } from '../core/format.js';

const METHODS = ['get', 'post', 'put', 'patch', 'delete'];
const PREFIX = '/api/v2';

// Module scope: survives navigation inside the SPA.
let specCache = null;
const savedValues = new Map(); // opId -> {path, query, body, accept}
const history = [];

// ------------------------------------------------------------------ spec helpers

function makeResolver(spec) {
  const resolve = (schema, seen = new Set()) => {
    if (!schema || typeof schema !== 'object') return schema || {};
    if (schema.$ref) {
      if (seen.has(schema.$ref)) return {};
      const name = schema.$ref.split('/').pop();
      const target = spec.components?.schemas?.[name] || {};
      return resolve({ ...target, ...Object.fromEntries(Object.entries(schema).filter(([k]) => k !== '$ref')) }, new Set([...seen, schema.$ref]));
    }
    return schema;
  };
  return resolve;
}

/** Strip `anyOf: [X, {type: null}]` down to X; return {schema, nullable}. */
function unwrap(schema, resolve) {
  let s = resolve(schema);
  let nullable = false;
  const variants = s.anyOf || s.oneOf;
  if (variants) {
    const nonNull = variants.filter((v) => resolve(v).type !== 'null');
    nullable = nonNull.length < variants.length;
    if (nonNull.length === 1) s = { ...resolve(nonNull[0]), title: s.title, description: s.description, default: s.default };
  }
  return { schema: s, nullable };
}

function typeLabel(schema, resolve) {
  const { schema: s, nullable } = unwrap(schema, resolve);
  let t = s.type || (s.anyOf || s.oneOf ? (s.anyOf || s.oneOf).map((v) => resolve(v).type || resolve(v).title || 'object').filter((x) => x !== 'null').join(' | ') : 'any');
  if (t === 'array') {
    const items = unwrap(s.items || {}, resolve).schema;
    t = `${items.type || items.title || 'any'}[]`;
  }
  if (s.format) t += ` (${s.format})`;
  return nullable ? `${t}?` : t;
}

/**
 * Example JSON value generated from a schema: required fields first.
 * Nullable scalars without a default become null (for Airflow, null usually
 * means "let the server decide", e.g. logical_date: null = now).
 */
function example(schema, resolve, depth = 0) {
  const { schema: s, nullable } = unwrap(schema, resolve);
  if (depth > 5) return null;
  if (nullable && s.default === undefined && s.type !== 'object' && s.type !== 'array') return null;
  if (s.example !== undefined) return s.example;
  if (s.examples?.length) return s.examples[0];
  if (s.default !== undefined && s.type !== 'object') return s.default;
  if (s.const !== undefined) return s.const;
  if (s.enum) return s.enum[0];
  if (s.allOf) return Object.assign({}, ...s.allOf.map((x) => example(x, resolve, depth + 1)));
  if (s.anyOf || s.oneOf) {
    const first = (s.anyOf || s.oneOf).find((v) => resolve(v).type !== 'null');
    return first ? example(first, resolve, depth + 1) : null;
  }
  switch (s.type) {
    case 'object': {
      const props = s.properties || {};
      const req = new Set(s.required || []);
      const keys = [...Object.keys(props).filter((k) => req.has(k)), ...Object.keys(props).filter((k) => !req.has(k))];
      if (!keys.length) return s.additionalProperties ? {} : {};
      return Object.fromEntries(keys.map((k) => [k, example(props[k], resolve, depth + 1)]));
    }
    case 'array': return s.items ? [example(s.items, resolve, depth + 1)] : [];
    case 'string':
      if (s.format === 'date-time') return new Date().toISOString().replace(/\.\d+Z$/, 'Z');
      if (s.format === 'date') return new Date().toISOString().slice(0, 10);
      return '';
    case 'integer': return s.minimum ?? 0;
    case 'number': return s.minimum ?? 0;
    case 'boolean': return false;
    default: return s.properties ? example({ ...s, type: 'object' }, resolve, depth + 1) : {};
  }
}

function buildOps(spec) {
  const ops = [];
  for (const [path, item] of Object.entries(spec.paths || {})) {
    for (const method of METHODS) {
      const op = item[method];
      if (!op) continue;
      const params = [...(item.parameters || []), ...(op.parameters || [])];
      const body = op.requestBody?.content?.['application/json']?.schema;
      ops.push({
        id: op.operationId || `${method}_${path}`,
        method,
        path,
        apiPath: path.startsWith(PREFIX) ? path.slice(PREFIX.length) : path,
        tag: op.tags?.[0] || 'Other',
        summary: op.summary || op.operationId || path,
        description: op.description || '',
        deprecated: !!op.deprecated,
        pathParams: params.filter((p) => p.in === 'path'),
        queryParams: params.filter((p) => p.in === 'query'),
        headerParams: params.filter((p) => p.in === 'header'),
        body,
        bodyRequired: !!op.requestBody?.required,
        responses: op.responses || {},
        fields: {
          summary: (op.summary || '').toLowerCase(),
          id: (op.operationId || '').toLowerCase().replace(/_/g, ' '),
          path: path.slice(PREFIX.length).toLowerCase(),
          tag: `${method} ${op.tags?.[0] || ''}`.toLowerCase(),
        },
      });
    }
  }
  return ops;
}

const WEIGHTS = { summary: 100, id: 60, path: 40, tag: 25 };
const wordStart = (text, w) => new RegExp(`(^|[\\s/{_.-])${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`).test(text);

/**
 * Score an op against the query: summary hits beat id/path/tag hits.
 * Returns {score, loose} or null for no match; `loose` = some word only matched as a subsequence.
 */
function fuzzyScore(op, q) {
  if (!q) return { score: 1, loose: false };
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  const f = op.fields;
  let score = 0;
  let loose = false;
  for (const w of words) {
    let best = 0;
    for (const [key, weight] of Object.entries(WEIGHTS)) {
      if (f[key].includes(w)) best = Math.max(best, weight + (wordStart(f[key], w) ? 15 : 0));
    }
    if (!best) {
      const all = `${f.summary} ${f.path}`;
      let ti = 0;
      for (const ch of w) {
        ti = all.indexOf(ch, ti);
        if (ti === -1) return null;
        ti++;
      }
      best = 5;
      loose = true;
    }
    score += best;
  }
  const qn = q.toLowerCase().trim();
  if (f.summary === qn || f.id === qn) score += 500;
  else if (f.summary.startsWith(qn)) score += 120;
  return { score: score - f.summary.length * 0.3, loose };
}

const methodBadge = (m) => h(`span.af-method.af-method-${m}`, m.toUpperCase());

// ------------------------------------------------------------------ page

export default async function render(ctx) {
  const { el, signal } = ctx;
  mount(el, pageHeader({ title: 'API explorer', icon: 'terminal', subtitle: 'Loading the Airflow OpenAPI spec…' }), skeleton(10));

  if (!specCache) {
    try {
      const res = await fetch(config.spec || '/pipelines/api/spec/', { credentials: 'same-origin', signal });
      const data = await res.json();
      if (!res.ok) throw Object.assign(new Error(data.detail || `Spec request failed (${res.status})`), { status: res.status });
      specCache = { spec: data, ops: buildOps(data), resolve: makeResolver(data) };
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(el, pageHeader({ title: 'API explorer', icon: 'terminal' }), errorState(err, () => render(ctx)));
      return;
    }
  }
  if (!ctx.active) return;
  const { spec, ops, resolve } = specCache;
  const byId = new Map(ops.map((o) => [o.id, o]));
  const tags = [...new Set(ops.map((o) => o.tag))].sort();

  let q = ctx.query.get('q') || '';
  let methodFilter = ctx.query.get('method') || '';
  let current = byId.get(ctx.query.get('op')) || null;
  const collapsed = new Set();

  const listEl = h('div.sys-exp-list');
  const panelEl = h('div.sys-exp-panel');
  const historyEl = h('div.sys-exp-history');
  const countEl = h('span.sys-exp-count');
  const searchBox = h('input.form-control.form-control-sm', {
    type: 'search', placeholder: 'Search, e.g. "task logs"', value: q, dataset: { searchShortcut: '' },
    'aria-label': 'Search endpoints',
  });
  searchBox.addEventListener('input', debounce(() => { q = searchBox.value.trim(); ctx.setQuery({ q }); drawList(); }, 120));
  searchBox.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const first = listEl.querySelector('.sys-exp-op');
      first?.click();
    }
  });

  const counts = Object.fromEntries(METHODS.map((m) => [m, ops.filter((o) => o.method === m).length]));
  mount(el,
    pageHeader({
      title: 'API explorer', icon: 'terminal',
      subtitle: h('span', `Every endpoint of the Airflow REST API v2 — ${ops.length} operations across ${tags.length} resources. Calls go through the dashboard proxy using the server's credentials.`),
      actions: [
        config.airflowUrl && h('a.btn.btn-sm.btn-outline-secondary', { href: `${config.airflowUrl}/docs`, target: '_blank', rel: 'noopener' }, icon('book', 'me-1'), 'Swagger UI'),
      ],
    }),
    h('div.sys-exp',
      h('aside.sys-exp-side',
        h('div.sys-exp-search', icon('search', 'sys-exp-search-icon'), searchBox),
        chips([{ value: '', label: 'All' }, ...METHODS.map((m) => ({ value: m, label: m.toUpperCase(), count: counts[m] }))], methodFilter,
          (v) => { methodFilter = v; ctx.setQuery({ method: v }); drawList(); }),
        countEl,
        listEl,
        historyEl),
      panelEl));

  drawList();
  drawHistory();
  if (current) select(current, { scroll: true }); else drawWelcome();

  const onKey = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && current) {
      e.preventDefault();
      panelEl.querySelector('.sys-exp-run')?.click();
    }
  };
  document.addEventListener('keydown', onKey);
  ctx.cleanup(() => document.removeEventListener('keydown', onKey));

  // ---------------------------------------------------------------- list

  function drawList() {
    let matches = ops
      .filter((o) => !methodFilter || o.method === methodFilter)
      .map((o) => ({ o, m: fuzzyScore(o, q) }))
      .filter((x) => x.m)
      .map(({ o, m }) => ({ o, score: m.score, loose: m.loose }));
    // When some endpoints match every word literally, drop the loose subsequence matches.
    if (matches.some((x) => !x.loose)) matches = matches.filter((x) => !x.loose);
    mount(countEl, `${matches.length} of ${ops.length} endpoints`);
    if (!matches.length) { mount(listEl, h('div.small.text-muted.p-3', 'No endpoints match.')); return; }
    const groups = new Map();
    for (const { o, score } of matches.sort((a, b) => b.score - a.score)) {
      if (!groups.has(o.tag)) groups.set(o.tag, { items: [], top: score });
      groups.get(o.tag).items.push(o);
    }
    const searching = !!q || !!methodFilter;
    const ordered = [...groups.entries()].sort((a, b) => (q ? b[1].top - a[1].top : 0) || a[0].localeCompare(b[0]));
    mount(listEl, ordered.map(([tag, { items }]) => {
      const isCollapsed = !searching && collapsed.has(tag);
      return h('div.sys-exp-group', { class: isCollapsed && 'collapsed' },
        h('button.sys-exp-tag', {
          type: 'button',
          onclick: () => { collapsed.has(tag) ? collapsed.delete(tag) : collapsed.add(tag); drawList(); },
        }, icon('chevron-down', 'sys-exp-caret'), h('span', tag), h('span.af-tab-count', String(items.length))),
        isCollapsed ? null : items.map((o) => h('button.sys-exp-op', {
          type: 'button', class: current?.id === o.id && 'active', title: `${o.method.toUpperCase()} ${o.path}`,
          dataset: { op: o.id },
          onclick: () => select(o),
        }, methodBadge(o.method), h('span.sys-exp-op-text',
          h('span.sys-exp-op-summary', { class: o.deprecated && 'text-decoration-line-through' }, o.summary),
          h('span.sys-exp-op-path', o.apiPath)))));
    }));
  }

  function select(op, { scroll = false } = {}) {
    current = op;
    ctx.setQuery({ op: op.id });
    listEl.querySelectorAll('.sys-exp-op').forEach((b) => b.classList.toggle('active', b.dataset.op === op.id));
    if (scroll) listEl.querySelector(`[data-op="${CSS.escape(op.id)}"]`)?.scrollIntoView({ block: 'center' });
    drawPanel(op);
    if (window.innerWidth < 992) panelEl.scrollIntoView({ behavior: 'smooth' });
  }

  // ---------------------------------------------------------------- welcome

  function drawWelcome() {
    const picks = ['get_dags', 'get_dag_runs', 'get_health', 'get_event_logs', 'get_pools', 'get_task_instances', 'get_log', 'get_version']
      .map((id) => byId.get(id)).filter(Boolean);
    mount(panelEl, card({
      body: h('div.sys-exp-welcome',
        h('div.sys-exp-welcome-icon', icon('terminal-fill')),
        h('h3', 'Pick an endpoint to begin'),
        h('p.text-muted', 'Fill in parameters, press ', h('kbd', 'Ctrl'), ' + ', h('kbd', 'Enter'), ' to execute. Requests that change data ask for confirmation first.'),
        h('div.sys-exp-method-stats', METHODS.map((m) => h('div.sys-exp-mstat', methodBadge(m), h('b', String(counts[m]))))),
        h('div.small.fw-semibold.text-muted.mt-3.mb-2', 'Popular'),
        h('div.sys-exp-picks', picks.map((o) => h('button.sys-exp-pick', { type: 'button', onclick: () => select(o, { scroll: true }) },
          methodBadge(o.method), h('span', o.summary), h('code', o.apiPath))))),
    }));
  }

  // ---------------------------------------------------------------- panel

  function drawPanel(op) {
    const saved = savedValues.get(op.id) || { path: {}, query: {}, body: null, accept: null };
    savedValues.set(op.id, saved);
    const pathInputs = new Map();
    const queryGetters = new Map();

    // Offer formats both the spec and the proxy support (the proxy relays json + text/plain;
    // x-ndjson streaming is not proxied). Task logs on Airflow 3.1 are JSON-only.
    const acceptHeader = op.headerParams.find((p) => p.name.toLowerCase() === 'accept');
    const specFormats = acceptHeader ? (unwrap(acceptHeader.schema, resolve).schema.enum || []) : [];
    const acceptOptions = ['application/json', 'text/plain'].filter((a) => specFormats.includes(a));
    const acceptSelect = acceptOptions.length > 1 && h('select.form-select.form-select-sm.sys-exp-accept', {
      onchange: (e) => { saved.accept = e.target.value; updatePreview(); },
    }, acceptOptions.map((a) => h('option', { value: a, selected: (saved.accept || 'application/json') === a }, a)));

    const pathSection = op.pathParams.length ? section('Path parameters', 'signpost-split', op.pathParams.map((p) => {
      const input = h('input.form-control.form-control-sm', {
        value: saved.path[p.name] ?? '', placeholder: p.name === 'dag_id' ? 'e.g. flatten_jsons_schemas' : p.name === 'dag_run_id' ? 'e.g. manual__2026-…  or ~ for all' : '',
        oninput: (e) => { saved.path[p.name] = e.target.value; updatePreview(); },
      });
      pathInputs.set(p.name, input);
      return paramRow(p, input, true);
    })) : null;

    const querySection = op.queryParams.length ? section('Query parameters', 'question-circle', op.queryParams.map((p) => {
      const { input, get } = queryInput(p, saved);
      queryGetters.set(p.name, get);
      input.addEventListener('input', updatePreview);
      input.addEventListener('change', updatePreview);
      return paramRow(p, input, p.required);
    })) : null;

    let editor = null;
    let bodySection = null;
    let schemaBox = null;
    if (op.body) {
      const ex = example(op.body, resolve);
      editor = jsonEditor(saved.body ?? JSON.stringify(ex, null, 2), { rows: Math.min(18, Math.max(6, JSON.stringify(ex, null, 2).split('\n').length + 1)), allowEmpty: !op.bodyRequired, onChange: (v) => { saved.body = v; updatePreview(); } });
      const schemaName = op.body.$ref ? op.body.$ref.split('/').pop() : 'body';
      bodySection = section(`Request body · ${schemaName}`, 'braces', [
        editor.el,
        h('div.d-flex.gap-3.mt-2.flex-wrap',
          h('button.af-link-btn', { type: 'button', onclick: () => { editor.set(example(op.body, resolve)); saved.body = null; updatePreview(); } }, icon('arrow-counterclockwise'), ' Reset to example'),
          h('button.af-link-btn', { type: 'button', onclick: () => schemaBox.classList.toggle('d-none') }, icon('diagram-2'), ' Show schema')),
        h('div.d-none.mt-2', { ref: (n) => { schemaBox = n; } }, jsonView(schemaTree(op.body), { open: 2 })),
      ]);
    }

    const responseEl = h('div.sys-exp-response');
    const urlPreview = h('code.sys-exp-url');
    const snippetHost = h('div');
    let snippetTab = 'curl';
    const runBtn = h('button.btn.btn-primary.btn-sm.sys-exp-run', { type: 'button', onclick: () => execute() },
      icon('send-fill', 'me-1'), 'Execute', h('kbd.ms-2.sys-kbd-dark', 'Ctrl ↵'));

    function collect() {
      const missing = [];
      let path = op.apiPath;
      for (const p of op.pathParams) {
        const v = (pathInputs.get(p.name).value || '').trim();
        if (!v) missing.push(p.name);
        path = path.replace(`{${p.name}}`, encodeURIComponent(v || `{${p.name}}`));
      }
      const params = {};
      for (const p of op.queryParams) {
        const v = queryGetters.get(p.name)();
        if (v == null || v === '' || (Array.isArray(v) && !v.length)) {
          if (p.required) missing.push(p.name);
          continue;
        }
        params[p.name] = v;
      }
      let body;
      if (editor) {
        body = editor.value();
        if (body == null) body = op.bodyRequired ? {} : undefined;
      }
      const accept = acceptSelect ? acceptSelect.value : 'application/json';
      return { path, params, body, accept, missing };
    }

    function updatePreview() {
      let c;
      try { c = collect(); } catch { return; }
      mount(urlPreview, `${PREFIX}${c.path}${qs(c.params)}`);
      drawSnippet();
    }

    function drawSnippet() {
      let c;
      try { c = collect(); } catch (err) { mount(snippetHost, h('div.small.text-danger', err.message)); return; }
      const base = config.airflowUrl || 'https://<airflow-host>';
      const url = `${base}${PREFIX}${c.path}${qs(c.params)}`;
      const bodyStr = c.body !== undefined ? JSON.stringify(c.body, null, 2) : null;
      const M = op.method.toUpperCase();
      const text = {
        curl: [
          '# 1) get a token (valid 24h)',
          `TOKEN=$(curl -s -X POST "${base}/auth/token" -H "Content-Type: application/json" \\`,
          `  -d '{"username": "'"$AIRFLOW_USERNAME"'", "password": "'"$AIRFLOW_PASSWORD"'"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')`,
          '',
          '# 2) call the endpoint',
          `curl -X ${M} "${url}" \\`,
          `  -H "Authorization: Bearer $TOKEN" \\`,
          `  -H "Accept: ${c.accept}"${bodyStr ? ' \\' : ''}`,
          bodyStr ? `  -H "Content-Type: application/json" \\\n  -d '${bodyStr.replace(/'/g, "'\\''")}'` : null,
        ].filter((x) => x != null).join('\n'),
        fetch: [
          `const res = await fetch(${JSON.stringify(url)}, {`,
          `  method: '${M}',`,
          '  headers: {',
          '    Authorization: `Bearer ${token}`,',
          `    Accept: '${c.accept}',`,
          bodyStr ? "    'Content-Type': 'application/json'," : null,
          '  },',
          bodyStr ? `  body: JSON.stringify(${bodyStr.split('\n').join('\n  ')}),` : null,
          '});',
          `const data = await res.${c.accept === 'text/plain' ? 'text' : 'json'}();`,
        ].filter((x) => x != null).join('\n'),
        python: [
          'import requests',
          '',
          `BASE = ${JSON.stringify(base)}`,
          'token = requests.post(f"{BASE}/auth/token", json={"username": USER, "password": PASSWORD}).json()["access_token"]',
          '',
          `resp = requests.${op.method}(`,
          `    f"{BASE}${PREFIX}${c.path}",`,
          Object.keys(c.params).length ? `    params=${JSON.stringify(c.params)},` : null,
          bodyStr ? `    json=${bodyStr.replace(/\btrue\b/g, 'True').replace(/\bfalse\b/g, 'False').replace(/\bnull\b/g, 'None').split('\n').join('\n    ')},` : null,
          `    headers={"Authorization": f"Bearer {token}", "Accept": "${c.accept}"},`,
          ')',
          'resp.raise_for_status()',
          c.accept === 'text/plain' ? 'print(resp.text)' : 'print(resp.json())',
        ].filter((x) => x != null).join('\n'),
      }[snippetTab];
      mount(snippetHost, codeBlock(text, { lang: snippetTab === 'python' ? 'python' : snippetTab === 'curl' ? 'bash' : 'javascript', maxHeight: '320px' }));
    }

    async function execute() {
      let c;
      try { c = collect(); } catch (err) { toast(err.message, 'error', { title: 'Invalid request body' }); return; }
      if (c.missing.length) {
        toast(`Missing required: ${c.missing.join(', ')}`, 'warning', { title: 'Cannot execute yet' });
        const first = pathInputs.get(c.missing[0]);
        first?.focus();
        return;
      }
      if (op.method !== 'get') {
        const ok = await confirmDialog({
          title: `Send ${op.method.toUpperCase()} request?`,
          danger: op.method === 'delete',
          confirmLabel: `Send ${op.method.toUpperCase()}`,
          message: h('div',
            h('p.mb-2', 'This request goes to the live Airflow server and may change or delete data.'),
            h('div.sys-exp-confirm', methodBadge(op.method), h('code', `${PREFIX}${c.path}${qs(c.params)}`))),
        });
        if (!ok) return;
      }
      runBtn.disabled = true;
      mount(responseEl, h('div.sys-exp-waiting', spinner(), ' Waiting for Airflow…'));
      const started = performance.now();
      let res;
      try {
        res = await request(op.method.toUpperCase(), c.path, { params: c.params, body: c.body, accept: c.accept, raw: true, signal });
      } catch (err) {
        runBtn.disabled = false;
        if (err.name === 'AbortError') return;
        mount(responseEl, errorState(err));
        return;
      }
      const text = await res.text();
      const ms = performance.now() - started;
      runBtn.disabled = false;
      if (!ctx.active) return;
      const type = res.headers.get('Content-Type') || '';
      let parsed = null;
      if (type.includes('json') && text) { try { parsed = JSON.parse(text); } catch { /* text */ } }
      history.unshift({ op: op.id, method: op.method, path: c.path, status: res.status, ms, at: new Date(), values: JSON.parse(JSON.stringify(saved)) });
      if (history.length > 20) history.pop();
      drawHistory();
      showResponse({ status: res.status, statusText: res.statusText, ms, size: new Blob([text]).size, type, text, parsed });
    }

    function showResponse(r) {
      const tone = r.status < 300 ? 'ok' : r.status < 400 ? 'info' : r.status < 500 ? 'warn' : 'bad';
      // Collections look like {<items>: [...], total_entries: n}.
      const listKey = r.parsed && r.parsed.total_entries != null
        ? Object.keys(r.parsed).find((k) => Array.isArray(r.parsed[k])) : null;
      const summary = listKey
        ? `${listKey}: ${r.parsed[listKey].length.toLocaleString()} of ${r.parsed.total_entries.toLocaleString()}` : null;
      mount(responseEl,
        h('div.sys-exp-resmeta',
          h(`span.sys-exp-status.sys-exp-status-${tone}`, `${r.status} ${r.statusText || ''}`.trim()),
          h('span', icon('stopwatch', 'me-1'), r.ms < 1000 ? `${Math.round(r.ms)} ms` : fmtDuration(r.ms / 1000)),
          h('span', icon('file-earmark', 'me-1'), fmtBytes(r.size)),
          r.type && h('span.af-mono.small', r.type.split(';')[0]),
          summary && h('span.af-tag', summary),
          h('span.ms-auto.d-flex.gap-2',
            copyButton(r.text, 'Copy'),
            h('button.af-copy-btn', {
              type: 'button',
              onclick: () => {
                const blob = new Blob([r.parsed ? JSON.stringify(r.parsed, null, 2) : r.text], { type: r.parsed ? 'application/json' : 'text/plain' });
                const a = h('a', { href: URL.createObjectURL(blob), download: `${op.id}.${r.parsed ? 'json' : 'txt'}` });
                a.click();
                setTimeout(() => URL.revokeObjectURL(a.href), 1000);
              },
            }, icon('download'), ' Download'))),
        r.status >= 400 && r.parsed?.detail ? h('div.af-callout.af-callout-danger.mb-2', icon('exclamation-octagon'), h('div', typeof r.parsed.detail === 'string' ? r.parsed.detail : jsonView(r.parsed.detail, { open: 2, root: false }))) : null,
        !r.text ? h('div.small.text-muted', 'Empty response body.')
          : r.parsed != null ? jsonView(r.parsed, { open: 2 }) : codeBlock(r.text, { lang: 'text', maxHeight: '520px' }));
    }

    const respCodes = Object.entries(op.responses);

    mount(panelEl,
      card({
        body: h('div.d-flex.flex-column.gap-3',
          h('div.sys-exp-head',
            methodBadge(op.method),
            h('code.sys-exp-path', op.path),
            copyButton(op.path),
            h('span.af-tag.ms-auto', op.tag)),
          h('div',
            h('h2.sys-exp-title', op.summary, op.deprecated && h('span.badge.text-bg-warning.ms-2', 'deprecated')),
            op.description && op.description !== op.summary && h('div.sys-exp-desc', op.description),
            h('div.small.text-muted.mt-1', 'operationId ', h('code', op.id))),
          pathSection,
          querySection,
          acceptSelect && section('Headers', 'card-heading', [paramRow({ name: 'Accept', description: 'Response format.' }, acceptSelect, false)]),
          bodySection,
          h('div.sys-exp-runbar',
            runBtn,
            h('div.sys-exp-urlwrap', h('span.small.text-muted.me-1', 'Request'), methodBadge(op.method), urlPreview)),
          h('div', { ref: (n) => n.appendChild(responseEl) }),
          h('div',
            h('div.sys-exp-section-title', icon('code-slash'), 'Code snippet', h('span.small.text-muted.ms-2.fw-normal', 'calls Airflow directly with a JWT')),
            tabs([{ id: 'curl', label: 'cURL' }, { id: 'fetch', label: 'JavaScript' }, { id: 'python', label: 'Python' }], snippetTab, (t) => { snippetTab = t; drawSnippet(); }),
            snippetHost),
          h('details.sys-exp-responses',
            h('summary.sys-exp-section-title', icon('reply'), `Documented responses (${respCodes.length})`),
            h('div.d-flex.flex-column.gap-2.mt-2', respCodes.map(([code, r]) => {
              const sch = r.content?.['application/json']?.schema;
              return h('details.sys-exp-rescode',
                h('summary', h(`span.sys-exp-status.sys-exp-status-${code < 300 ? 'ok' : code < 500 ? 'warn' : 'bad'}`, code), ' ', r.description || ''),
                sch ? jsonView(example(sch, resolve), { open: 2 }) : h('div.small.text-muted.ps-2', 'No body'));
            })))),
      }));
    updatePreview();
  }

  // ---------------------------------------------------------------- inputs

  function section(title, ic, rows) {
    return h('div.sys-exp-section',
      h('div.sys-exp-section-title', icon(ic), title),
      h('div.sys-exp-params', rows));
  }

  function paramRow(p, input, required) {
    const schema = p.schema ? unwrap(p.schema, resolve).schema : {};
    return h('div.sys-exp-param',
      h('div.sys-exp-param-meta',
        h('div', h('code.sys-exp-pname', p.name), required ? h('span.text-danger.ms-1', '*') : null),
        p.schema && h('div.sys-exp-ptype', typeLabel(p.schema, resolve)),
        schema.default !== undefined && h('div.sys-exp-pdefault', `default: ${JSON.stringify(schema.default)}`)),
      h('div.sys-exp-param-input', input,
        (p.description || schema.description) && h('div.sys-exp-pdesc', p.description || schema.description)));
  }

  function queryInput(p, saved) {
    const { schema: s } = unwrap(p.schema || {}, resolve);
    const stored = saved.query[p.name];
    const store = (v) => { saved.query[p.name] = v; };
    if (s.type === 'array') {
      const items = unwrap(s.items || {}, resolve).schema;
      if (items.enum) {
        let current = Array.isArray(stored) ? stored : [];
        const c = chips(items.enum.map((v) => ({ value: v, label: v })), current, (vals) => { current = vals; store(vals); c.dispatchEvent(new Event('change')); }, { multi: true });
        return { input: c, get: () => current };
      }
      const input = h('input.form-control.form-control-sm', {
        value: Array.isArray(stored) ? stored.join(', ') : '',
        placeholder: p.name === 'order_by' ? `e.g. -${(s.default || ['id'])[0]}  (prefix - for descending)` : 'comma, separated, values',
        oninput: (e) => store(e.target.value.split(',').map((x) => x.trim()).filter(Boolean)),
      });
      return { input, get: () => input.value.split(',').map((x) => x.trim()).filter(Boolean) };
    }
    if (s.enum || s.type === 'boolean') {
      const opts = s.enum || [true, false];
      const sel = h('select.form-select.form-select-sm', { onchange: (e) => store(e.target.value) },
        h('option', { value: '' }, s.default !== undefined ? `default (${s.default})` : '— not set —'),
        opts.map((v) => h('option', { value: String(v), selected: stored === String(v) }, String(v))));
      return { input: sel, get: () => sel.value };
    }
    const isNum = s.type === 'integer' || s.type === 'number';
    const isDate = s.format === 'date-time';
    const input = h('input.form-control.form-control-sm', {
      type: isNum ? 'number' : 'text',
      value: stored ?? '',
      min: s.minimum, max: s.maximum,
      placeholder: isDate ? 'ISO-8601, e.g. 2026-11-01T00:00:00Z' : s.default !== undefined ? String(s.default) : /pattern$/.test(p.name) ? 'SQL LIKE, e.g. %snowflake%' : '',
      oninput: (e) => store(e.target.value),
    });
    if (isDate) {
      const wrap = h('div.input-group.input-group-sm', input,
        h('button.btn.btn-outline-secondary', { type: 'button', title: '24 hours ago', onclick: () => { input.value = new Date(Date.now() - 864e5).toISOString(); store(input.value); input.dispatchEvent(new Event('input')); } }, '-24h'),
        h('button.btn.btn-outline-secondary', { type: 'button', title: 'Now', onclick: () => { input.value = new Date().toISOString(); store(input.value); input.dispatchEvent(new Event('input')); } }, 'now'));
      return { input: wrap, get: () => input.value.trim() };
    }
    return { input, get: () => input.value.trim() };
  }

  function schemaTree(schema, depth = 0) {
    const { schema: s, nullable } = unwrap(schema, resolve);
    if (depth > 4) return typeLabel(schema, resolve);
    if (s.type === 'object' || s.properties) {
      const req = new Set(s.required || []);
      return Object.fromEntries(Object.entries(s.properties || {}).map(([k, v]) => [`${k}${req.has(k) ? ' *' : ''}`, schemaTree(v, depth + 1)]));
    }
    if (s.type === 'array') return [schemaTree(s.items || {}, depth + 1)];
    if (s.enum) return `enum: ${s.enum.join(' | ')}${nullable ? ' | null' : ''}`;
    return typeLabel(schema, resolve);
  }

  // ---------------------------------------------------------------- history

  function drawHistory() {
    if (!history.length) { mount(historyEl); return; }
    mount(historyEl,
      h('div.sys-exp-history-head', icon('clock-history'), 'History', h('button.af-link-btn.ms-auto', { type: 'button', onclick: () => { history.length = 0; drawHistory(); } }, 'Clear')),
      history.map((e) => h('button.sys-exp-hist', {
        type: 'button',
        title: `${e.method.toUpperCase()} ${e.path}`,
        onclick: () => {
          const o = byId.get(e.op);
          if (!o) return;
          savedValues.set(o.id, JSON.parse(JSON.stringify(e.values)));
          select(o, { scroll: true });
        },
      },
      methodBadge(e.method),
      h('span.sys-exp-hist-path', e.path),
      h(`span.sys-exp-status.sys-exp-status-${e.status < 300 ? 'ok' : e.status < 500 ? 'warn' : 'bad'}`, String(e.status)),
      h('span.sys-exp-hist-ms', `${Math.round(e.ms)}ms`))));
  }
}
