// API explorer: every Redash API operation (see core/endpoints.js) with typed inputs,
// id pickers, request body examples, live execution through the proxy, response viewer,
// code snippets and history.

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import {
  pageHeader, jsonEditor, jsonView, codeBlock, copyButton, confirmDialog, emptyState, spinner, chips, toastError, tabs,
} from '../../../ui/js/ui.js';
import { fmtBytes } from '../../../ui/js/format.js';
import { request, fileUrl, config, qs, rd } from '../core/api.js';
import { cached, getDataSources } from '../core/common.js';
import { ENDPOINTS, GROUPS, buildPath } from '../core/endpoints.js';

const METHODS = ['GET', 'POST', 'DELETE'];
const history = [];
const lastValues = new Map(); // endpoint id -> {path, query, body}

/** Options for an id picker, by parameter name. */
const PICKERS = {
  query_id: () => cached('xp:queries', () => rd.get('/api/queries', { page_size: 100, order: '-created_at' }).then((r) => r.results.map((q) => ({ value: q.id, label: `#${q.id} ${q.name}` })))),
  dropdown_query_id: () => PICKERS.query_id(),
  dashboard_id: () => cached('xp:dash', () => rd.get('/api/dashboards', { page_size: 100 }).then((r) => r.results.map((d) => ({ value: d.id, label: `#${d.id} ${d.name}` })))),
  object_id: () => PICKERS.dashboard_id(),
  data_source_id: () => getDataSources().then((l) => l.map((d) => ({ value: d.id, label: `#${d.id} ${d.name} (${d.type})` }))),
  user_id: () => cached('xp:users', () => rd.get('/api/users', { page_size: 100 }).then((r) => r.results.map((u) => ({ value: u.id, label: `#${u.id} ${u.name} <${u.email}>` })))),
  group_id: () => cached('xp:groups', () => rd.get('/api/groups').then((l) => l.map((g) => ({ value: g.id, label: `#${g.id} ${g.name}` })))),
  alert_id: () => cached('xp:alerts', () => rd.get('/api/alerts').then((l) => l.map((a) => ({ value: a.id, label: `#${a.id} ${a.name}` })))),
  destination_id: () => cached('xp:dest', () => rd.get('/api/destinations').then((l) => l.map((d) => ({ value: d.id, label: `#${d.id} ${d.name} (${d.type})` })))),
  snippet_id: () => cached('xp:snip', () => rd.get('/api/query_snippets').then((l) => l.map((s) => ({ value: s.id, label: `#${s.id} ${s.trigger}` })))),
  query_result_id: () => cached('xp:results', () => rd.get('/api/queries', { page_size: 100 }).then((r) => r.results.filter((q) => q.latest_query_data_id).map((q) => ({ value: q.latest_query_data_id, label: `#${q.latest_query_data_id} (latest of “${q.name}”)` })))),
};
const FIXED = {
  filetype: ['csv', 'tsv', 'json', 'xlsx'],
  object_type: ['queries', 'dashboards'],
  access_type: ['modify'],
};

function fuzzy(text, q) {
  text = text.toLowerCase();
  if (text.includes(q)) return 100 - text.indexOf(q);
  let i = 0;
  for (const ch of q) { i = text.indexOf(ch, i); if (i < 0) return -1; i++; }
  return 5;
}

export default async function render(ctx) {
  const { el } = ctx;
  let current = ENDPOINTS.find((e) => e.id === ctx.query.get('op')) || ENDPOINTS[0];
  let search = '';
  let method = '';
  const collapsed = new Set(prefs.get('rd:xp:collapsed', []));

  const listEl = h('div.rdx-list');
  const countEl = h('div.rdx-count');
  const detailEl = h('div.rdx-detail');
  const historyEl = h('div.rdx-history');

  mount(el,
    pageHeader({
      title: 'API explorer', icon: 'terminal',
      subtitle: `All ${ENDPOINTS.length} operations of the Redash API, called through the dashboard proxy with the server-side admin key.`,
    }),
    h('div.rdx',
      h('aside.rdx-side.af-card',
        h('div.rdx-side-head',
          h('div.af-search',
            icon('search', 'af-search-icon'),
            h('input.form-control.form-control-sm', {
              type: 'search', placeholder: 'Search, e.g. "favorite dashboards"', dataset: { searchShortcut: '' },
              oninput: debounce((e) => { search = e.target.value.trim().toLowerCase(); drawList(); }, 100),
            })),
          chips([{ value: '', label: 'All' }, ...METHODS.map((m) => ({ value: m, label: m, count: ENDPOINTS.filter((x) => x.method === m).length }))], method, (v) => { method = v; drawList(); }),
          countEl),
        listEl),
      h('div.rdx-main', detailEl, historyEl)));

  function drawList() {
    const items = ENDPOINTS
      .filter((e) => !method || e.method === method)
      .map((e) => ({ e, score: search ? Math.max(fuzzy(`${e.summary} ${e.group}`, search), fuzzy(e.path, search), fuzzy(`${e.method} ${e.path}`, search)) : 0 }))
      .filter((x) => x.score >= 0);
    mount(countEl, `${items.length} of ${ENDPOINTS.length} operations`);
    if (!items.length) { mount(listEl, h('div.rdx-empty', 'No operation matches.')); return; }
    if (search) {
      mount(listEl, items.sort((a, b) => b.score - a.score).map(({ e }) => item(e, true)));
      return;
    }
    mount(listEl, GROUPS.map((g) => {
      const ops = items.filter((x) => x.e.group === g).map((x) => x.e);
      if (!ops.length) return null;
      const isOpen = !collapsed.has(g) || ops.includes(current);
      const groupEl = h('div.rdx-group', { class: isOpen && 'open' },
        h('button.rdx-group-head', {
          type: 'button',
          onclick: () => {
            if (collapsed.has(g)) collapsed.delete(g); else collapsed.add(g);
            prefs.set('rd:xp:collapsed', [...collapsed]);
            drawList();
          },
        }, icon('chevron-right', 'rdx-caret'), h('span', g), h('span.rdx-group-count', String(ops.length))),
        h('div.rdx-group-body', ops.map((e) => item(e))));
      return groupEl;
    }));
    centerActive();
  }

  // Keep the selected operation visible in the list without scrolling the page.
  function centerActive() {
    requestAnimationFrame(() => {
      const a = listEl.querySelector('.rdx-item.active');
      if (!a) return;
      const top = a.offsetTop;
      if (top < listEl.scrollTop || top + a.offsetHeight > listEl.scrollTop + listEl.clientHeight) {
        listEl.scrollTop = Math.max(0, top - listEl.clientHeight / 2 + a.offsetHeight / 2);
      }
    });
  }

  function item(e, showGroup = false) {
    return h('button.rdx-item', {
      type: 'button', class: e === current && 'active', title: `${e.method} ${e.path}`,
      onclick: () => select(e),
    },
    h(`span.af-method.af-method-${e.method.toLowerCase()}`, e.method),
    h('span.rdx-item-text', h('span.rdx-item-sum', e.summary), h('span.rdx-item-path', showGroup ? `${e.group} · ${e.path}` : e.path)),
    e.danger && icon('exclamation-triangle', 'rdx-danger-icon'));
  }

  function select(e) {
    current = e;
    ctx.setQuery({ op: e.id });
    drawList();
    drawDetail();
  }

  // ---------------------------------------------------------------- detail

  let runRequest = null;

  function drawDetail() {
    const e = current;
    const saved = lastValues.get(e.id) || { path: {}, query: {}, body: undefined };
    const pathInputs = {};
    const queryInputs = {};
    const responseEl = h('div.rdx-response', emptyState({ icon: 'send', title: 'No request sent yet', message: 'Fill in the parameters and press Execute (Ctrl+Enter).' }));
    const bodyEditor = e.body !== undefined && e.method !== 'GET' ? jsonEditor(saved.body ?? e.body, { rows: 10 }) : null;
    const previewEl = h('code.rdx-url');
    const snippetsHost = h('div');

    const pathField = (name) => {
      const id = `rdx-p-${name}`;
      const value = saved.path[name] ?? '';
      let input;
      if (FIXED[name]) {
        input = h('select.form-select.form-select-sm', { id }, FIXED[name].map((v) => h('option', { value: v, selected: v === value }, v)));
      } else {
        input = h('input.form-control.form-control-sm', { id, value, placeholder: name, list: PICKERS[name] ? `${id}-list` : undefined, autocomplete: 'off' });
      }
      const dl = PICKERS[name] ? h(`datalist#${id}-list`) : null;
      if (dl) PICKERS[name]().then((opts) => mount(dl, opts.map((o) => h('option', { value: o.value }, o.label)))).catch(() => {});
      pathInputs[name] = input;
      input.addEventListener('input', updatePreview);
      input.addEventListener('change', updatePreview);
      return h('label.rdx-field', h('span.rdx-field-name', h('code', `{${name}}`), h('span.rdx-req', 'required')), input, dl,
        PICKERS[name] && h('span.rdx-hint', 'Start typing or pick from the suggestions'));
    };

    const queryField = (p) => {
      const id = `rdx-q-${p.name}`;
      const value = saved.query[p.name] ?? '';
      const input = p.type === 'boolean'
        ? h('select.form-select.form-select-sm', { id }, ['', 'true', 'false'].map((v) => h('option', { value: v, selected: v === String(value) }, v || '— not set —')))
        : h('input.form-control.form-control-sm', { id, value, placeholder: p.example != null ? `e.g. ${p.example}` : p.type === 'string[]' ? 'e.g. finance, weekly' : 'optional' });
      queryInputs[p.name] = { input, p };
      input.addEventListener('input', updatePreview);
      input.addEventListener('change', updatePreview);
      return h('label.rdx-field', h('span.rdx-field-name', h('code', p.name), h('span.rdx-type', p.type)), input, p.desc && h('span.rdx-hint', p.desc));
    };

    function collect() {
      const path = Object.fromEntries(Object.entries(pathInputs).map(([k, i]) => [k, i.value.trim()]));
      const query = {};
      for (const [k, { input, p }] of Object.entries(queryInputs)) {
        const v = input.value.trim();
        if (!v) continue;
        query[k] = p.type === 'string[]' ? v.split(',').map((s) => s.trim()).filter(Boolean) : v;
      }
      return { path, query };
    }

    function urlFor(c) {
      return `${buildPath(e, c.path)}${qs(c.query)}`;
    }

    function updatePreview() {
      const c = collect();
      const missing = e.params.filter((p) => !c.path[p]);
      previewEl.textContent = `${e.method} ${urlFor(c)}`;
      previewEl.classList.toggle('incomplete', missing.length > 0);
      lastValues.set(e.id, { path: c.path, query: Object.fromEntries(Object.entries(queryInputs).map(([k, { input }]) => [k, input.value])), body: bodyEditor ? bodyEditor.textarea.value : undefined });
      drawSnippets(c);
    }

    let snippetTab = prefs.get('rd:xp:snippet', 'curl');
    function drawSnippets(c) {
      const base = (config.redashUrl || 'http://localhost:5050').replace(/\/$/, '');
      const url = `${base}${urlFor(c)}`;
      let body = null;
      if (bodyEditor) { try { body = bodyEditor.value(); } catch { body = null; } }
      const bodyStr = body != null ? JSON.stringify(body) : null;
      const snippets = {
        curl: [`curl -X ${e.method} '${url}' \\`, `  -H "Authorization: Key $REDASH_API_KEY"${bodyStr ? ' \\' : ''}`,
          ...(bodyStr ? ['  -H "Content-Type: application/json" \\', `  -d '${bodyStr.replace(/'/g, "'\\''")}'`] : [])].join('\n'),
        fetch: [`// Through the Studio proxy (session auth + CSRF token):`,
          `const res = await fetch('${fileUrl(urlFor(c))}', {`, `  method: '${e.method}',`,
          `  headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrftoken },`,
          bodyStr ? `  body: JSON.stringify(${JSON.stringify(body, null, 2).split('\n').join('\n  ')}),` : null,
          `});`, `const data = await res.json();`].filter(Boolean).join('\n'),
        python: ['import os, requests', '', `resp = requests.${e.method.toLowerCase()}(`, `    "${url}",`,
          '    headers={"Authorization": f"Key {os.environ[\'REDASH_API_KEY\']}"},',
          bodyStr ? `    json=${JSON.stringify(body)},` : null, '    timeout=60,', ')', 'resp.raise_for_status()', 'print(resp.json())'].filter(Boolean).join('\n'),
      };
      const lang = { curl: 'bash', fetch: 'javascript', python: 'python' };
      mount(snippetsHost,
        tabs([{ id: 'curl', label: 'cURL' }, { id: 'fetch', label: 'JavaScript' }, { id: 'python', label: 'Python' }], snippetTab, (id) => { snippetTab = id; prefs.set('rd:xp:snippet', id); drawSnippets(collect()); }),
        codeBlock(snippets[snippetTab], { lang: lang[snippetTab] === 'python' ? 'python' : 'text' }));
    }

    async function execute() {
      const c = collect();
      const missing = e.params.filter((p) => !c.path[p]);
      if (missing.length) {
        mount(responseEl, h('div.af-callout.af-callout-warn', icon('exclamation-triangle'), h('span', 'Fill in ', missing.map((m, i) => [i ? ', ' : '', h('code', m)]), ' first.')));
        pathInputs[missing[0]]?.focus();
        return;
      }
      let body;
      if (bodyEditor) {
        try { body = bodyEditor.value() ?? {}; } catch (err) { mount(responseEl, h('div.af-callout.af-callout-danger', icon('braces'), err.message)); return; }
      } else if (e.method === 'POST') body = {};
      const path = urlFor({ path: c.path, query: {} });
      if (e.method !== 'GET') {
        const ok = await confirmDialog({
          title: `Send ${e.method} request?`, danger: e.danger || e.method === 'DELETE', confirmLabel: `Send ${e.method}`,
          message: h('span', 'This changes data in Redash: ', h('code', `${e.method} ${path}`)),
          details: e.danger ? h('div.af-callout.af-callout-danger.mt-2', icon('exclamation-octagon'), h('span', e.desc || 'This action is destructive.')) : null,
        });
        if (!ok) return;
      }
      mount(responseEl, h('div.rdx-sending', spinner(), ' Sending…'));
      const t0 = performance.now();
      let res;
      try {
        res = await request(e.method, path, { params: c.query, body, raw: true });
      } catch (err) {
        if (err.name === 'AbortError') return;
        mount(responseEl, h('div.af-callout.af-callout-danger', icon('wifi-off'), err.message));
        return;
      }
      const ms = performance.now() - t0;
      const type = res.headers.get('Content-Type') || '';
      const blob = await res.blob();
      const entry = { e, values: lastValues.get(e.id), status: res.status, ms, at: new Date(), size: blob.size };
      history.unshift(entry);
      if (history.length > 20) history.pop();
      drawHistory();
      await showResponse(res, type, blob, ms, c);
    }
    runRequest = execute;

    async function showResponse(res, type, blob, ms, c) {
      const ok = res.ok;
      const head = h('div.rdx-resp-head',
        h('span.rdx-status', { class: ok ? 'ok' : res.status >= 500 ? 'err' : 'warn' }, `${res.status} ${res.statusText || ''}`.trim()),
        h('span.rdx-meta', icon('stopwatch'), ` ${ms.toFixed(0)} ms`),
        h('span.rdx-meta', icon('file-earmark'), ` ${fmtBytes(blob.size)}`),
        h('span.rdx-meta.af-mono', type.split(';')[0] || '—'));
      if (/json/.test(type)) {
        const text = await blob.text();
        let data;
        try { data = JSON.parse(text); } catch { data = text; }
        const count = Array.isArray(data) ? data.length : Array.isArray(data?.results) ? `${data.results.length} of ${data.count}` : null;
        mount(responseEl, head, count != null && h('div.small.text-muted.mb-2', `${count} item(s)`),
          typeof data === 'object' ? jsonView(data, { open: 2 }) : codeBlock(String(data), { lang: 'text' }));
      } else if (/text|csv|tab-separated/.test(type)) {
        const text = await blob.text();
        const lines = text.split('\n');
        mount(responseEl, head,
          h('div.d-flex.gap-2.mb-2', downloadLink(blob, `result.${c.path.filetype || 'txt'}`), copyButton(text, 'Copy')),
          codeBlock(lines.slice(0, 200).join('\n') + (lines.length > 200 ? `\n… ${lines.length - 200} more lines` : ''), { lang: 'text', maxHeight: '420px' }));
      } else if (blob.size) {
        mount(responseEl, head, h('div.rdx-download', icon('file-earmark-spreadsheet', 'rdx-download-icon'),
          h('div', h('div.fw-semibold', 'Binary response'), h('div.small.text-muted', `${type || 'unknown type'} · ${fmtBytes(blob.size)}`),
            downloadLink(blob, `result.${c.path.filetype || 'bin'}`))));
      } else {
        mount(responseEl, head, h('div.text-muted.small', 'Empty response body.'));
      }
    }

    const groupLabel = h('span.rdx-group-label', current.group);
    mount(detailEl, h('section.af-card.rdx-card',
      h('div.rdx-card-head',
        h('div.rdx-title-row',
          h(`span.af-method.af-method-${e.method.toLowerCase()}.rdx-method-lg`, e.method),
          h('code.rdx-path', e.path),
          copyButton(e.path),
          groupLabel),
        h('h2.rdx-summary', e.summary),
        e.desc && h('p.rdx-desc', e.desc),
        e.danger && h('div.af-callout.af-callout-danger', icon('exclamation-octagon'), h('span', 'Destructive or security-sensitive operation — you will be asked to confirm.'))),
      h('div.rdx-card-body',
        e.params.length > 0 && h('div.rdx-section', h('div.af-form-section', 'Path parameters'), h('div.rdx-fields', e.params.map(pathField))),
        e.query.length > 0 && h('div.rdx-section', h('div.af-form-section', 'Query parameters'), h('div.rdx-fields', e.query.map(queryField))),
        bodyEditor && h('div.rdx-section',
          h('div.af-form-section.d-flex.justify-content-between', h('span', 'Request body (JSON)'),
            h('button.af-link-btn', { type: 'button', onclick: () => { bodyEditor.set(e.body); updatePreview(); } }, icon('arrow-counterclockwise'), ' Reset example')),
          bodyEditor.el),
        h('div.rdx-send',
          previewEl,
          h('button.btn.btn-primary', { type: 'button', onclick: execute },
            icon('send-fill', 'me-1'), 'Execute', h('kbd.rdx-kbd', navigator.platform.includes('Mac') ? '⌘↵' : 'Ctrl↵')))),
      h('div.rdx-card-section', h('div.af-form-section', 'Response'), responseEl),
      h('details.rdx-card-section.rdx-snippets', { open: prefs.get('rd:xp:snipopen', false) },
        h('summary.af-form-section', 'Code snippets'), snippetsHost)));
    detailEl.querySelector('details.rdx-snippets').addEventListener('toggle', (ev) => prefs.set('rd:xp:snipopen', ev.target.open));
    if (bodyEditor) bodyEditor.textarea.addEventListener('input', debounce(updatePreview, 250));
    updatePreview();
  }

  function downloadLink(blob, name) {
    const a = h('a.btn.btn-sm.btn-outline-primary', { download: name, href: URL.createObjectURL(blob) }, icon('download', 'me-1'), `Save ${name}`);
    ctx.cleanup(() => URL.revokeObjectURL(a.href));
    return a;
  }

  function drawHistory() {
    if (!history.length) { mount(historyEl); return; }
    mount(historyEl, h('section.af-card.rdx-hist',
      h('div.rdx-hist-head', h('span.fw-semibold', icon('clock-history', 'me-1'), 'History'), h('span.small.text-muted', 'this session · click to restore')),
      h('div', history.map((x) => h('button.rdx-hist-item', {
        type: 'button',
        onclick: () => { if (x.values) lastValues.set(x.e.id, x.values); select(x.e); },
      },
      h(`span.af-method.af-method-${x.e.method.toLowerCase()}`, x.e.method),
      h('span.af-mono.rdx-hist-path', x.values ? buildPath(x.e, x.values.path || {}) : x.e.path),
      h('span.rdx-status.sm', { class: x.status < 400 ? 'ok' : x.status >= 500 ? 'err' : 'warn' }, String(x.status)),
      h('span.small.text-muted', `${x.ms.toFixed(0)} ms · ${x.at.toLocaleTimeString()}`))))));
  }

  const onKey = (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter' && document.body.contains(detailEl) && !document.querySelector('.af-backdrop')) {
      ev.preventDefault();
      runRequest?.().catch?.((err) => toastError(err));
    }
  };
  document.addEventListener('keydown', onKey);
  ctx.cleanup(() => document.removeEventListener('keydown', onKey));

  drawList();
  drawDetail();
  drawHistory();
}
