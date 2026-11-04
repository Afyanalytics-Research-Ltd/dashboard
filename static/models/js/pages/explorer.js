// API explorer: every query and mutation of the live W&B GraphQL schema, a form-based query builder
// (typed arguments + selection tree), a raw editor, safe execution and copyable snippets.

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import {
  pageHeader, card, tabs, emptyState, errorState, skeleton, jsonView, codeBlock, copyButton, confirmDialog, toast, timeAgo,
} from '../../../ui/js/ui.js';
import { loadCodeMirror, loadCodeMirrorMode } from '../../../ui/js/lib.js';
import { config } from '../core/api.js';
import {
  BLOCKED, loadSchema, named, sig, isLeaf, isList, isRequired, defaultSelection, buildDocument, coerce, inputSkeleton,
} from './admin/gqlschema.js';
import { pill } from './admin/shared.js';

const HISTORY_KEY = 'ms:explorer:history';

function readHistory() { try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; } }
function pushHistory(entry) {
  const list = [entry, ...readHistory().filter((e) => e.text !== entry.text)].slice(0, 20);
  try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(list)); } catch { /* ignore */ }
}

function csrf() {
  const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

/** Raw POST to the proxy so we can show status, timing and partial data/errors exactly. */
async function execute(query, variables, signal) {
  const t0 = performance.now();
  const res = await fetch(config.graphql || '/models/graphql', {
    method: 'POST', credentials: 'same-origin', signal,
    headers: { 'Content-Type': 'application/json', Accept: 'application/json', 'X-CSRFToken': csrf() },
    body: JSON.stringify({ query, variables }),
  });
  const text = await res.text();
  let body = text;
  try { body = JSON.parse(text); } catch { /* keep text */ }
  return { status: res.status, ms: performance.now() - t0, size: text.length, body };
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  const side = h('aside.msa-exp-side', skeleton(10));
  const main = h('section.msa-exp-main', emptyState({ icon: 'terminal', title: 'Pick an operation', message: 'Choose a query or mutation on the left, or open a type to read its docs.' }));
  mount(el,
    pageHeader({
      title: 'API explorer', icon: 'terminal',
      subtitle: 'Every query and mutation of your W&B Server’s GraphQL API — build requests with forms, run them through the dashboard proxy, copy them as code.',
      actions: [h('button.btn.btn-sm.btn-outline-secondary', { title: 'Reload the schema from the server', onclick: async () => { await loadSchema({ fresh: true }); toast('Schema reloaded', 'success', { timeout: 2000 }); render(ctx); } }, icon('arrow-clockwise', 'me-1'), 'Reload schema')],
    }),
    h('div.msa-exp', side, main));

  let schema;
  try { schema = await loadSchema(); } catch (err) { mount(side, errorState(err, () => render(ctx))); mount(main); return; }
  if (!ctx.active) return;

  const ops = {
    query: [...schema.query.fields].sort((a, b) => a.name.localeCompare(b.name)),
    mutation: [...(schema.mutation?.fields || [])].sort((a, b) => a.name.localeCompare(b.name)),
  };
  const typeList = [...schema.types.values()].filter((t) => !t.name.startsWith('__')).sort((a, b) => a.name.localeCompare(b.name));
  let sideTab = ctx.query.get('side') || 'query';
  let search = '';
  let showBlocked = prefs.get('ms:exp:blocked', false);

  // ------------------------------------------------------------------ sidebar

  const listEl = h('div.msa-exp-list');
  const searchEl = h('input.form-control.form-control-sm', { type: 'search', placeholder: 'Search operations and types…', 'aria-label': 'Search', dataset: { searchShortcut: '' } });
  searchEl.addEventListener('input', debounce(() => { search = searchEl.value.trim().toLowerCase(); drawList(); }, 100));
  const blockedToggle = h('label.form-check.form-switch.small.msa-exp-blocked',
    h('input.form-check-input', { type: 'checkbox', checked: showBlocked, onchange: (e) => { showBlocked = e.target.checked; prefs.set('ms:exp:blocked', showBlocked); drawList(); } }),
    h('span', 'Show blocked'));
  const sideTabs = tabs([
    { id: 'query', label: 'Queries', count: ops.query.filter((f) => !BLOCKED.has(f.name)).length },
    { id: 'mutation', label: 'Mutations', count: ops.mutation.filter((f) => !BLOCKED.has(f.name)).length },
    { id: 'types', label: 'Types', count: typeList.length },
    { id: 'history', label: 'History' },
  ], sideTab, (t) => { sideTab = t; ctx.setQuery({ side: t === 'query' ? null : t }); drawList(); });
  mount(side, h('div.af-search.msa-exp-search', icon('search', 'af-search-icon'), searchEl), sideTabs, blockedToggle, listEl);

  const score = (name, desc) => {
    if (!search) return 1;
    const n = name.toLowerCase();
    if (n === search) return 100;
    if (n.startsWith(search)) return 50;
    if (n.includes(search)) return 20;
    if ((desc || '').toLowerCase().includes(search)) return 5;
    return 0;
  };

  function drawList() {
    if (sideTab === 'history') {
      const hist = readHistory();
      mount(listEl, hist.length ? hist.map((e) => h('button.msa-exp-item', { type: 'button', onclick: () => openRaw(e) },
        h('span.af-method', { class: e.op === 'mutation' ? 'af-method-post' : 'af-method-get' }, e.op === 'mutation' ? 'MUT' : 'QRY'),
        h('span.msa-exp-name', e.field || 'request'),
        h('span.msa-exp-meta', `${e.status} · ${timeAgo(e.at).textContent}`))) : h('div.msa-exp-empty', 'Requests you run appear here.'));
      return;
    }
    if (sideTab === 'types') {
      const hits = typeList.map((t) => ({ t, s: score(t.name, t.description) })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s).slice(0, 300);
      mount(listEl, hits.map(({ t }) => h('button.msa-exp-item', { type: 'button', onclick: () => openType(t.name) },
        h('span.msa-kind', { class: `k-${t.kind.toLowerCase()}` }, t.kind.replace('INPUT_OBJECT', 'INPUT').slice(0, 5)),
        h('span.msa-exp-name', t.name))));
      return;
    }
    const list = ops[sideTab].map((f) => ({ f, s: score(f.name, f.description) })).filter((x) => x.s > 0 && (showBlocked || !BLOCKED.has(x.f.name)))
      .sort((a, b) => b.s - a.s || a.f.name.localeCompare(b.f.name));
    mount(listEl, list.length ? list.map(({ f }) => h('button.msa-exp-item', {
      type: 'button', class: BLOCKED.has(f.name) && 'blocked', title: f.description || f.name,
      onclick: () => openOp(sideTab, f.name),
    },
    h('span.af-method', { class: sideTab === 'mutation' ? (f.name.startsWith('delete') ? 'af-method-delete' : 'af-method-post') : 'af-method-get' }, sideTab === 'mutation' ? 'MUT' : 'QRY'),
    h('span.msa-exp-name', f.name),
    BLOCKED.has(f.name) && icon('slash-circle', 'msa-exp-blockicon'))) : h('div.msa-exp-empty', 'No matches'));
    listEl.querySelector(`[data-active]`)?.scrollIntoView({ block: 'nearest' });
  }

  // ------------------------------------------------------------------ type docs

  function typeLink(t) {
    const n = named(t);
    return h('button.msa-typelink', { type: 'button', onclick: () => openType(n.name) }, sig(t));
  }

  function openType(name) {
    const t = schema.types.get(name);
    if (!t) return;
    ctx.setQuery({ type: name, op: null });
    const rows = (t.fields || t.inputFields || []).map((f) => h('div.msa-doc-field',
      h('div.msa-doc-sig', h('b', f.name),
        f.args?.length ? h('span.text-muted', '(', f.args.map((a, i) => [i ? ', ' : '', a.name, ': ', typeLink(a.type)]), ')') : null,
        ': ', typeLink(f.type), f.isDeprecated && pill('deprecated', 'warn')),
      f.description && h('div.msa-doc-desc', f.description)));
    mount(main, card({
      title: name, icon: 'diagram-2',
      actions: pill(t.kind.toLowerCase().replace('_', ' '), 'info'),
      body: h('div.msa-doc',
        t.description && h('p.msa-doc-desc', t.description),
        t.interfaces?.length ? h('div.small.mb-2', 'Implements ', t.interfaces.map((i) => h('button.msa-typelink', { type: 'button', onclick: () => openType(i.name) }, i.name))) : null,
        t.possibleTypes?.length ? h('div.small.mb-2', 'Possible types: ', t.possibleTypes.map((i) => h('button.msa-typelink', { type: 'button', onclick: () => openType(i.name) }, i.name))) : null,
        t.enumValues?.length ? h('div.d-flex.flex-wrap.gap-1', t.enumValues.map((v) => h('code.msa-enum', { title: v.description || '' }, v.name))) : null,
        rows.length ? h('div.msa-doc-fields', rows) : (!t.enumValues?.length && h('div.text-muted.small', 'Scalar type.'))),
    }));
  }

  // ------------------------------------------------------------------ operation builder

  async function openOp(op, name) {
    const field = ops[op].find((f) => f.name === name);
    if (!field) return;
    ctx.setQuery({ op: `${op}.${name}`, type: null });
    if (BLOCKED.has(name)) {
      mount(main, card({ title: name, icon: 'slash-circle', body: h('div',
        h('div.af-callout.af-callout-warn', icon('shield-lock'), h('span', h('b', 'Blocked by the dashboard. '),
          'This is a W&B cloud billing/sandbox/Slurm operation or one that destroys the organisation, so the Model Studio proxy refuses it.')),
        field.description && h('p.msa-doc-desc.mt-2', field.description)) }));
      return;
    }
    const returnType = named(field.type).name;
    const args = {};
    const argInputs = {};
    let selection = {};
    defaultSelection(schema, returnType).forEach((f) => { selection[f] = true; });
    if (!Object.keys(selection).length && schema.types.get(returnType)?.fields) {
      // Payload/connection types: pick their most useful child.
      const t = schema.types.get(returnType);
      const first = t.fields.find((f) => ['edges', 'success', 'user', 'project', 'entity', 'bucket', 'artifact'].includes(f.name)) || t.fields[0];
      selection[first.name] = isLeaf(schema, first.type) ? true : buildDefaultSub(named(first.type).name, 0);
    }

    const argsEl = h('div.msa-args', field.args.length ? field.args.map((a) => argInput(a)) : h('div.text-muted.small', 'No arguments.'));
    const treeEl = h('div.msa-tree');
    let treeFilter = '';
    const treeFilterEl = h('input.form-control.form-control-sm.msa-tree-filter', {
      type: 'search', placeholder: 'Filter fields…', 'aria-label': 'Filter fields',
      oninput: debounce((e) => { treeFilter = e.target.value.trim().toLowerCase(); applyTreeFilter(); }, 80),
    });
    // Hide unchecked fields whose name doesn't match; checked fields always stay visible.
    function applyTreeFilter() {
      treeEl.querySelectorAll('.msa-tree-item').forEach((li) => {
        const name = li.querySelector(':scope > .msa-tree-label .msa-tree-name')?.textContent.toLowerCase() || '';
        li.hidden = !!treeFilter && !li.classList.contains('on') && !name.includes(treeFilter);
      });
    }
    const editorHost = h('div.msa-exp-editor');
    const varsTa = h('textarea.form-control.form-control-sm.ms-mono.msa-vars', { rows: 6, spellcheck: 'false', 'aria-label': 'Variables JSON' });
    const resultEl = h('div.msa-exp-result', emptyState({ icon: 'play-circle', title: 'Run the request', message: 'Ctrl/⌘ + Enter runs it. Mutations ask before sending.' }));
    const detached = h('div.af-callout.af-callout-info.small.d-none', icon('pencil'), h('span', 'Edited by hand — the form no longer drives the text. '),
      h('button.af-link-btn', { onclick: () => { manual = false; detached.classList.add('d-none'); regenerate(); } }, 'Rebuild from form'));
    const snippetsEl = h('div.msa-snippets');
    let cm = null;
    let manual = false;

    function buildDefaultSub(typeName, depth) {
      const sub = {};
      const t = schema.types.get(typeName);
      if (!t) return { __typename: true };
      if (t.kind === 'UNION' || t.kind === 'INTERFACE') { sub.__typename = true; }
      defaultSelection(schema, typeName, 4).forEach((n) => { sub[n] = true; });
      if (!Object.keys(sub).length && t.fields && depth < 3) {
        const pref = t.fields.find((f) => ['edges', 'node', 'pageInfo'].includes(f.name)) || null;
        if (pref) sub[pref.name] = isLeaf(schema, pref.type) ? true : buildDefaultSub(named(pref.type).name, depth + 1);
      }
      return Object.keys(sub).length ? sub : { __typename: true };
    }

    function argInput(a) {
      const t = named(a.type);
      const kind = schema.types.get(t.name)?.kind;
      const id = `arg-${a.name}`;
      let input;
      if (kind === 'ENUM' && !isList(a.type)) {
        input = h('select.form-select.form-select-sm', { id },
          h('option', { value: '' }, isRequired(a.type) ? '— choose —' : '— not set —'),
          schema.types.get(t.name).enumValues.map((v) => h('option', { value: v.name }, v.name)));
      } else if (t.name === 'Boolean' && !isList(a.type)) {
        input = h('select.form-select.form-select-sm', { id }, h('option', { value: '' }, '— not set —'), h('option', { value: 'true' }, 'true'), h('option', { value: 'false' }, 'false'));
      } else if (kind === 'INPUT_OBJECT' || isList(a.type) || t.name === 'JSONString' || t.name === 'JSON') {
        const skeleton = kind === 'INPUT_OBJECT' && !isList(a.type) ? JSON.stringify(inputSkeleton(schema, t.name), null, 2) : '';
        input = h('textarea.form-control.form-control-sm.ms-mono', { id, rows: kind === 'INPUT_OBJECT' ? Math.min(12, skeleton.split('\n').length + 1) : 2, spellcheck: 'false', placeholder: isList(a.type) ? '["a", "b"]  or  a, b' : t.name === 'JSONString' ? '{"key": "value"}' : '' }, skeleton);
      } else {
        input = h('input.form-control.form-control-sm', { id, type: /Int|Float/.test(t.name) ? 'number' : 'text', placeholder: a.defaultValue ? `default ${a.defaultValue}` : t.name });
      }
      argInputs[a.name] = { input, arg: a };
      const status = h('span.msa-arg-err');
      input.addEventListener('input', debounce(() => {
        try { readArg(a); status.textContent = ''; input.classList.remove('is-invalid'); } catch (err) { status.textContent = err.message; input.classList.add('is-invalid'); }
        regenerate();
      }, 200));
      input.addEventListener('change', () => { try { readArg(a); } catch { /* shown on input */ } regenerate(); });
      return h('div.msa-arg',
        h('label.msa-arg-label', { for: id }, h('b', a.name), isRequired(a.type) && h('span.text-danger', ' *'), h('span.msa-arg-type', ' ', typeLink(a.type))),
        input, a.description && h('div.msa-arg-help', a.description), status);
    }

    function readArg(a) {
      const { input } = argInputs[a.name];
      const raw = input.value;
      const t = named(a.type);
      if (raw === '' || raw == null) { delete args[a.name]; return; }
      args[a.name] = t.name === 'JSONString' && !isList(a.type) ? (() => { JSON.parse(raw); return raw; })() : coerce(schema, a.type, raw);
    }

    // ---- selection tree
    function drawTree() {
      mount(treeEl, isLeaf(schema, field.type)
        ? h('div.text-muted.small', `Returns ${sig(field.type)} — nothing to select.`)
        : treeNode(returnType, selection, 0));
      applyTreeFilter();
    }

    function treeNode(typeName, sel, depth) {
      const t = schema.types.get(typeName);
      if (!t) return null;
      const items = [];
      if (t.kind === 'UNION' || t.kind === 'INTERFACE') {
        for (const pt of t.possibleTypes || []) {
          const key = `... on ${pt.name}`;
          items.push(treeItem({ name: key, fragment: pt.name }, sel, depth));
        }
      }
      for (const f of t.fields || []) items.push(treeItem(f, sel, depth));
      return h('ul.msa-tree-list', { class: depth && 'nested' }, items);
    }

    function treeItem(f, sel, depth) {
      const key = f.fragment ? f.name : f.name;
      const leaf = !f.fragment && isLeaf(schema, f.type);
      const needsArgs = !f.fragment && (f.args || []).some((a) => isRequired(a.type));
      const checked = key in sel;
      const box = h('input.form-check-input', { type: 'checkbox', checked, disabled: needsArgs, 'aria-label': key });
      const li = h('li.msa-tree-item', { class: [checked && 'on', needsArgs && 'disabled'] });
      const label = h('label.msa-tree-label', box, h('span.msa-tree-name', key),
        !f.fragment && h('span.msa-tree-type', sig(f.type)),
        needsArgs && h('span.msa-tree-note', { title: 'Has required arguments — add it in the raw editor' }, 'needs args'));
      box.addEventListener('change', () => {
        if (box.checked) sel[key] = leaf ? true : buildDefaultSub(f.fragment || named(f.type).name, depth + 1);
        else delete sel[key];
        drawTree();
        regenerate();
      });
      li.append(label);
      if (checked && !leaf && sel[key] !== true && depth < 8) li.append(treeNode(f.fragment || named(f.type).name, sel[key], depth + 1));
      if (f.description && !f.fragment) label.title = f.description;
      return li;
    }

    function regenerate() {
      if (manual) return;
      const { text, variables } = buildDocument(schema, op, field, args, selection);
      cm?.setValue(text);
      varsTa.value = JSON.stringify(variables, null, 2);
      drawSnippets();
    }

    function currentRequest() {
      const text = cm ? cm.getValue() : '';
      let variables = {};
      const v = varsTa.value.trim();
      if (v) variables = JSON.parse(v);
      return { text, variables };
    }

    function drawSnippets() {
      let req;
      try { req = currentRequest(); } catch { return; }
      const payload = JSON.stringify({ query: req.text, variables: req.variables });
      const base = (config.wandbUrl || 'http://localhost:8090').replace(/\/$/, '');
      const curl = `curl -s -u "api:$WANDB_API_KEY" -H 'Content-Type: application/json' \\\n  ${base}/graphql --data-binary @- <<'JSON'\n${payload}\nJSON`;
      const py = `import os, wandb\nfrom wandb_gql import gql\n\napi = wandb.Api(overrides={"base_url": "${base}"})  # uses WANDB_API_KEY\nresult = api.client.execute(gql('''\n${req.text}\n'''), variable_values=${JSON.stringify(req.variables, null, 2).replace(/\n/g, '\n')})\nprint(result)`;
      mount(snippetsEl,
        h('div.msa-snippet', h('div.msa-snippet-head', h('span', icon('terminal', 'me-1'), 'cURL'), copyButton(curl, 'Copy')), codeBlock(curl, { lang: 'bash' })),
        h('div.msa-snippet', h('div.msa-snippet-head', h('span', icon('filetype-py', 'me-1'), 'Python (wandb SDK)'), copyButton(py, 'Copy')), codeBlock(py, { lang: 'python' })));
    }

    async function run() {
      let req;
      try { req = currentRequest(); } catch (err) { toast(`Variables are not valid JSON: ${err.message}`, 'error'); return; }
      const isMutation = /^\s*mutation\b/.test(req.text);
      if (isMutation) {
        const ok = await confirmDialog({
          title: `Run mutation ${field.name}?`, danger: /delete|remove|revoke/i.test(field.name), confirmLabel: 'Run mutation',
          message: 'Mutations change data on your W&B Server. They are recorded in the dashboard audit log.',
          details: h('pre.msa-confirm-pre', JSON.stringify(req.variables, null, 2)),
        });
        if (!ok) return;
      }
      mount(resultEl, h('div.msa-running', h('span.spinner-border.spinner-border-sm'), ' Running…'));
      try {
        const r = await execute(req.text, req.variables, signal);
        pushHistory({ op: isMutation ? 'mutation' : 'query', field: field.name, text: req.text, variables: req.variables, status: r.status, at: new Date().toISOString() });
        const errs = r.body?.errors || [];
        mount(resultEl,
          h('div.msa-result-head',
            pill(`HTTP ${r.status}`, r.status < 300 && !errs.length ? 'ok' : r.status === 409 || r.status === 403 ? 'warn' : 'bad'),
            pill(`${r.ms.toFixed(0)} ms`, 'muted', 'stopwatch'), pill(`${(r.size / 1024).toFixed(1)} KB`, 'muted', 'file-earmark'),
            errs.length ? pill(`${errs.length} error${errs.length > 1 ? 's' : ''}`, 'bad', 'exclamation-triangle') : null,
            h('span.ms-auto'), copyButton(JSON.stringify(r.body, null, 2), 'Copy JSON')),
          errs.length ? h('div.af-callout.af-callout-danger.mb-2', icon('exclamation-octagon'), h('div', errs.map((e) => h('div', e.message)))) : null,
          typeof r.body === 'object' ? jsonView(r.body.data !== undefined ? r.body.data : r.body, { open: 3 }) : codeBlock(String(r.body), { lang: 'text' }));
        if (sideTab === 'history') drawList();
      } catch (err) {
        if (err.name !== 'AbortError') mount(resultEl, errorState(err));
      }
    }

    const runBtn = h('button.btn.btn-sm.btn-primary', { onclick: run }, icon('play-fill', 'me-1'), 'Run', h('kbd.ms-1', 'Ctrl ⏎'));
    mount(main,
      card({
        title: h('span', h('span.af-method', { class: op === 'mutation' ? 'af-method-post' : 'af-method-get' }, op === 'mutation' ? 'MUTATION' : 'QUERY'), ' ', field.name),
        icon: null,
        actions: [h('span.small.text-muted', 'returns ', typeLink(field.type)), runBtn],
        body: h('div',
          field.description && h('p.msa-doc-desc', field.description),
          h('div.msa-builder',
            h('div.msa-builder-col', h('div.msa-col-title', icon('sliders', 'me-1'), 'Arguments'), argsEl),
            h('div.msa-builder-col', h('div.msa-col-title', icon('diagram-3', 'me-1'), 'Fields to return'),
              !isLeaf(schema, field.type) && treeFilterEl, treeEl))),
      }),
      card({ title: 'Request', icon: 'code-slash', body: h('div', detached, editorHost, h('div.msa-col-title.mt-2', 'Variables (JSON)'), varsTa) }),
      card({ title: 'Response', icon: 'reply', body: resultEl }),
      card({ title: 'Use it in code', icon: 'clipboard-data', body: snippetsEl }));

    drawTree();
    await loadCodeMirror();
    await loadCodeMirrorMode('javascript');
    if (!ctx.active) return;
    cm = window.CodeMirror(editorHost, {
      value: '', mode: 'javascript', lineNumbers: true, matchBrackets: true, autoCloseBrackets: true, tabSize: 2,
      extraKeys: { 'Ctrl-Enter': run, 'Cmd-Enter': run },
    });
    cm.on('change', (_, change) => {
      if (change.origin !== 'setValue') { manual = true; detached.classList.remove('d-none'); drawSnippets(); }
    });
    varsTa.addEventListener('input', debounce(drawSnippets, 200));
    varsTa.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); run(); } });
    regenerate();
    setTimeout(() => cm.refresh(), 0);
  }

  async function openRaw(entry) {
    const [op, name] = [entry.op, entry.field];
    if (ops[op]?.some((f) => f.name === name)) {
      await openOp(op, name);
      const ed = main.querySelector('.CodeMirror')?.CodeMirror;
      if (ed) {
        ed.setValue(entry.text);
        main.querySelector('.msa-vars').value = JSON.stringify(entry.variables || {}, null, 2);
      }
    }
  }

  drawList();
  const deep = ctx.query.get('op');
  const deepType = ctx.query.get('type');
  if (deep) {
    const [op, name] = deep.split('.');
    if (ops[op]) { sideTab = op; sideTabs.setActive(op); drawList(); openOp(op, name); }
  } else if (deepType) openType(deepType);
}
