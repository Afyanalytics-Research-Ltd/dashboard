// Node detail view: input data | parameters & settings | output data.
//
//   openNodePanel({ wf, name, execution, credentials, readOnly, onChange(nextWf, meta), onRun(name) })

import { h, icon, mount } from '../../../ui/js/dom.js';
import { modal, toast, toastError, jsonEditor, spinner, tabs } from '../../../ui/js/ui.js';
import { loadDescriptions, nodeIcon, entryFor, displayName, credentialDisplayName } from '../core/catalog.js';
import { descriptionFor, isVisible, versionsOf } from '../core/params.js';
import { renameNode, parents, isSticky, ancestors } from '../core/workflow.js';
import { previewExpression } from '../core/sandbox.js';
import { paramForm, stripHtml } from './paramform.js';
import { dataView } from './dataview.js';
import { comboBox, inlineEdit } from '../core/widgets.js';
import { urls } from '../core/urls.js';

const runItems = (execution, name) => {
  const runs = execution?.data?.resultData?.runData?.[name];
  if (!runs?.length) return null;
  const last = runs[runs.length - 1];
  return { run: last, outputs: (last.data?.main || []).map((items, i) => ({ label: null, index: i, items: items || [] })) };
};

const STICKY_COLORS = ['#fff5d6', '#ffe0d2', '#ffd6e0', '#e6f7d9', '#d7ecff', '#ece0ff', '#eceff3'];

export function openNodePanel(opts) {
  const { name, readOnly } = opts;
  let wf = opts.wf;
  let node = wf.nodes.find((n) => n.name === name);
  if (!node) return null;
  const commit = (nextNode, meta = {}) => {
    wf = { ...wf, nodes: wf.nodes.map((n) => (n.name === node.name ? nextNode : n)) };
    node = nextNode;
    opts.onChange?.(wf, { label: meta.label || 'Edit parameters', group: meta.group || `params:${node.name}` });
  };

  if (isSticky(node)) return openStickyPanel(node, commit, readOnly);

  const entry = entryFor(node.type);
  const titleEl = h('span.n8s-ndv-title');
  const center = h('div.n8s-ndv-center', spinner());
  const left = h('div.n8s-ndv-side');
  const right = h('div.n8s-ndv-side');
  const runBtn = !readOnly && opts.onRun && h('button.btn.btn-sm.btn-primary', { type: 'button', title: 'Run the workflow up to and including this node', onclick: () => opts.onRun(node.name) }, icon('play-fill', 'me-1'), 'Execute step');

  const m = modal({
    title: h('span.n8s-ndv-heading', nodeIcon(node.type, { size: 26 }), titleEl, h('span.n8s-ndv-type', `${displayName(node.type)} · v${node.typeVersion}`)),
    size: 'xl',
    body: h('div.n8s-ndv', left, center, right),
    actions: [],
  });
  m.el.classList.add('n8s-ndv-modal');
  const headerActions = h('div.n8s-ndv-actions',
    entry?.documentationUrl && h('a.btn.btn-sm.btn-link', { href: entry.documentationUrl, target: '_blank', rel: 'noopener' }, icon('book', 'me-1'), 'Docs'),
    runBtn);
  m.el.querySelector('.af-modal-header').insertBefore(headerActions, m.el.querySelector('.af-modal-header .af-icon-btn'));

  mount(titleEl, readOnly ? node.name : inlineEdit(node.name, async (newName) => {
    if (wf.nodes.some((n) => n.name === newName)) throw new Error(`A node named "${newName}" already exists.`);
    const next = renameNode(wf, node.name, newName);
    wf = next;
    node = next.nodes.find((n) => n.name === newName);
    opts.onChange?.(next, { label: 'Rename node' });
    toast(`Renamed — expressions referencing it were updated too.`, 'success', { timeout: 2500 });
  }));

  // ---------------------------------------------------------------- input / output panels

  const execution = opts.execution;
  const nodeOutputs = () => {
    const outs = {};
    for (const n of wf.nodes) {
      const pinned = wf.pinData?.[n.name];
      if (pinned?.length) outs[n.name] = pinned;
      else {
        const r = runItems(execution, n.name);
        if (r) outs[n.name] = r.outputs[0]?.items || [];
      }
    }
    return outs;
  };

  let inputSource = parents(wf, node.name)[0] || null;
  function renderInput() {
    const ps = parents(wf, node.name);
    const ups = ancestors(wf, node.name);
    if (!ps.length) {
      mount(left, h('div.n8s-ndv-panel-title', icon('box-arrow-in-right', 'me-1'), 'Input'), h('div.n8s-dv-empty', entry?.trigger ? 'Trigger nodes start the workflow — they have no input.' : 'Connect a node to this one to see its input.'));
      return;
    }
    const pick = h('select.form-select.form-select-sm.n8s-ndv-src', { onchange: (e) => { inputSource = e.target.value; renderInput(); } },
      ups.map((u) => h('option', { value: u, selected: u === inputSource }, `${u}${ps.includes(u) ? '' : ' (earlier)'}`)));
    const pinned = wf.pinData?.[inputSource];
    const r = runItems(execution, inputSource);
    const items = pinned?.length ? pinned : r?.outputs?.[0]?.items || [];
    mount(left, h('div.n8s-ndv-panel-title', icon('box-arrow-in-right', 'me-1'), 'Input', pick),
      dataView({
        items, node: ps.length === 1 && inputSource === ps[0] ? null : inputSource, mode: 'schema',
        emptyText: execution ? `"${inputSource}" did not run in the loaded execution.` : 'No execution data yet — run the workflow, or pin data on the previous node.',
        title: pinned?.length ? h('span.n8s-pill-pin', icon('pin-angle-fill'), 'pinned') : null,
      }));
  }

  function renderOutput() {
    const pinned = wf.pinData?.[node.name];
    const r = runItems(execution, node.name);
    const actions = !readOnly && h('div.btn-group.btn-group-sm',
      pinned?.length
        ? h('button.btn.btn-outline-warning', { type: 'button', title: 'Unpin data', onclick: () => { const pd = { ...(wf.pinData || {}) }; delete pd[node.name]; wf = { ...wf, pinData: pd }; opts.onChange?.(wf, { label: 'Unpin data' }); renderOutput(); renderInput(); } }, icon('pin-angle'), ' Unpin')
        : r?.outputs?.[0]?.items?.length && h('button.btn.btn-outline-secondary', { type: 'button', title: 'Pin this output so later runs reuse it', onclick: () => pin(r.outputs[0].items) }, icon('pin-angle'), ' Pin'),
      h('button.btn.btn-outline-secondary', { type: 'button', title: 'Edit pinned data as JSON', onclick: editPinned }, icon('pencil-square')));
    const err = r?.run?.error;
    mount(right, h('div.n8s-ndv-panel-title', icon('box-arrow-right', 'me-1'), 'Output',
      r && h('span.n8s-ndv-meta', `${r.run.executionTime ?? 0} ms`)),
    err && h('div.n8s-ndv-error', icon('exclamation-octagon-fill', 'me-1'), h('div', h('b', err.message || 'Error'), err.description && h('div.small', stripHtml(err.description)), err.httpCode && h('div.small', `HTTP ${err.httpCode}`))),
    dataView({
      outputs: pinned?.length ? [{ items: pinned, label: 'Pinned' }] : r?.outputs?.length ? r.outputs.map((o, i) => ({ ...o, label: (Array.isArray(entry?.outputNames) && entry.outputNames[i]) || (r.outputs.length > 1 ? `Output ${i}` : null) })) : [{ items: [] }],
      node: node.name, mode: 'table', actions,
      title: pinned?.length ? h('span.n8s-pill-pin', icon('pin-angle-fill'), 'pinned') : null,
      emptyText: execution ? 'This node produced no items in the loaded execution.' : 'Run the workflow (or this step) to see output.',
    }));
  }

  function pin(items) {
    wf = { ...wf, pinData: { ...(wf.pinData || {}), [node.name]: items.map((it) => ({ json: it.json ?? it })) } };
    opts.onChange?.(wf, { label: 'Pin data' });
    toast('Output pinned — runs will reuse it instead of calling the node.', 'success', { timeout: 2500 });
    renderOutput();
  }

  function editPinned() {
    const current = wf.pinData?.[node.name] || runItems(execution, node.name)?.outputs?.[0]?.items || [{ json: {} }];
    const editor = jsonEditor(current.map((it) => it.json ?? it), { rows: 18, allowEmpty: false });
    modal({
      title: `Pinned data — ${node.name}`, size: 'lg', icon: 'pin-angle',
      body: h('div', h('p.small.text-muted', 'An array of JSON objects (one per item). Pinned data replaces this node\'s output during manual runs.'), editor.el),
      actions: [{ label: 'Cancel' }, {
        label: 'Pin', variant: 'primary',
        onClick: (close) => {
          let v = editor.value();
          if (!Array.isArray(v)) v = [v];
          pin(v.map((x) => ({ json: x })));
          close();
        },
      }],
    });
  }

  // ---------------------------------------------------------------- center: parameters / settings

  let tab = 'params';
  let descs = null;
  const centerTabs = tabs([
    { id: 'params', label: 'Parameters', icon: 'sliders' },
    { id: 'settings', label: 'Settings', icon: 'gear' },
    { id: 'json', label: 'JSON', icon: 'braces' },
  ], tab, (t) => { tab = t; renderCenter(); });
  const centerBody = h('div.n8s-ndv-form');

  const previewCtx = () => {
    const ps = parents(wf, node.name);
    const outs = nodeOutputs();
    const input = ps.length ? outs[inputSource || ps[0]] || [] : [];
    return { json: input[0]?.json || {}, input, nodes: outs, workflow: { id: wf.id, name: wf.name, active: !!wf.active }, prevNode: ps[0], parameters: node.parameters };
  };
  const hasData = () => !!execution || Object.keys(wf.pinData || {}).length;

  function credentialsSlot(desc) {
    const types = (desc?.credentials || []).filter((c) => isVisible(c, node.parameters || {}, node.parameters || {}, { typeVersion: node.typeVersion }));
    if (!types.length) return null;
    return h('div.n8s-cred-slot', types.map((ct) => {
      const options = (opts.credentials || []).filter((c) => c.type === ct.name).map((c) => ({ value: c.id, label: c.name, sub: `id ${c.id}` }));
      const current = node.credentials?.[ct.name];
      const combo = comboBox({
        options, value: current?.id || '', emptyLabel: options.length ? '— select credential —' : 'No credentials of this type yet',
        onChange: (id, o) => {
          const credentials = { ...(node.credentials || {}) };
          if (id) credentials[ct.name] = { id: String(id), name: o.label };
          else delete credentials[ct.name];
          commit({ ...node, credentials }, { label: 'Set credential' });
        },
      });
      return h('div.n8s-field',
        h('div.n8s-field-head', h('label.n8s-field-label', icon('key', 'me-1'), credentialDisplayName(ct.name), ct.required && h('span.text-danger', ' *')), h('div.flex-grow-1'),
          h('a.af-link-btn.small', { href: urls.credentials({ create: ct.name }), target: '_blank' }, icon('plus'), ' New')),
        readOnly ? h('div', current?.name || '—') : combo.el,
        current && !options.some((o) => String(o.value) === String(current.id)) && h('div.n8s-field-note.text-danger', icon('exclamation-triangle', 'me-1'), `Credential "${current.name}" (${current.id}) is missing on this instance.`));
    }));
  }

  async function renderCenter() {
    mount(center, centerTabs, centerBody);
    if (tab === 'json') {
      const editor = jsonEditor(node, { rows: 24, allowEmpty: false });
      mount(centerBody, editor.el, !readOnly && h('button.btn.btn-sm.btn-primary.mt-2', {
        type: 'button',
        onclick: () => {
          try {
            const v = editor.value();
            if (v.name !== node.name) throw new Error('Rename the node with the title field, not in JSON.');
            commit(v, { label: 'Edit node JSON' });
            toast('Node updated', 'success', { timeout: 1500 });
          } catch (err) { toastError(err, 'Invalid node JSON'); }
        },
      }, 'Apply JSON'));
      return;
    }
    if (tab === 'settings') { mount(centerBody, settingsForm()); return; }
    mount(centerBody, spinner());
    try {
      descs = descs || await loadDescriptions(node.type);
    } catch (err) {
      mount(centerBody, h('div.alert.alert-warning.small', icon('exclamation-triangle', 'me-1'), err.message, ' Edit the parameters as JSON instead:'),
        rawParams());
      return;
    }
    const desc = descriptionFor(descs, node.typeVersion);
    const form = paramForm({
      props: desc?.properties || [],
      values: node.parameters || {},
      typeVersion: node.typeVersion,
      readOnly,
      credentialsSlot: credentialsSlot(desc),
      preview: hasData() ? (expr) => previewExpression(expr, previewCtx()) : null,
      onChange: (params) => commit({ ...node, parameters: params }),
    });
    mount(centerBody, desc?.description && h('div.n8s-ndv-desc', stripHtml(desc.description)), form.el);
  }

  function rawParams() {
    const editor = jsonEditor(node.parameters || {}, { rows: 18, allowEmpty: false, onChange: () => { try { commit({ ...node, parameters: editor.value() }); } catch { /* invalid while typing */ } } });
    return editor.el;
  }

  function settingsForm() {
    const field = (labelText, input, hint) => h('div.n8s-field', h('div.n8s-field-head', h('label.n8s-field-label', labelText), hint && h('span.n8s-field-hint', { title: hint }, icon('question-circle'))), input);
    const toggle = (key, labelText, hint) => {
      const cb = h('input.form-check-input', { type: 'checkbox', role: 'switch', checked: !!node[key], disabled: readOnly });
      cb.addEventListener('change', () => { const next = { ...node }; if (cb.checked) next[key] = true; else delete next[key]; commit(next, { label: labelText }); renderCenter(); });
      return field(labelText, h('div.form-check.form-switch', cb), hint);
    };
    const num = (key, labelText, def) => {
      const input = h('input.form-control.form-control-sm', { type: 'number', value: node[key] ?? def, readonly: readOnly });
      input.addEventListener('change', () => commit({ ...node, [key]: Number(input.value) }, { label: labelText }));
      return field(labelText, input);
    };
    const onError = h('select.form-select.form-select-sm', { disabled: readOnly },
      [['stopWorkflow', 'Stop workflow'], ['continueRegularOutput', 'Continue (regular output)'], ['continueErrorOutput', 'Continue (using error output)']]
        .map(([v, l]) => h('option', { value: v, selected: (node.onError || (node.continueOnFail ? 'continueRegularOutput' : 'stopWorkflow')) === v }, l)));
    onError.addEventListener('change', () => { const next = { ...node, onError: onError.value }; delete next.continueOnFail; commit(next, { label: 'On error' }); });
    const versions = descs ? [...new Set(descs.flatMap(versionsOf))].sort((a, b) => b - a) : entry?.versions || [node.typeVersion];
    const ver = h('select.form-select.form-select-sm', { disabled: readOnly }, versions.map((v) => h('option', { value: v, selected: v === node.typeVersion }, `v${v}${v === Math.max(...versions) ? ' (latest)' : ''}`)));
    ver.addEventListener('change', () => { commit({ ...node, typeVersion: Number(ver.value) }, { label: 'Change version' }); toast('Node version changed — check its parameters.', 'warning'); renderCenter(); });
    const notes = h('textarea.form-control.form-control-sm', { rows: 3, readonly: readOnly, placeholder: 'Notes shown on hover (and on the canvas if enabled below)' }, node.notes || '');
    notes.addEventListener('change', () => commit({ ...node, notes: notes.value || undefined }, { label: 'Notes' }));
    return h('div',
      toggle('disabled', 'Disable node', 'A disabled node passes its input through unchanged.'),
      toggle('alwaysOutputData', 'Always output data', 'Output an empty item when the node returns nothing — keeps the branch running.'),
      toggle('executeOnce', 'Execute once', 'Run with the first input item only.'),
      toggle('retryOnFail', 'Retry on fail'),
      node.retryOnFail && num('maxTries', 'Max tries', 3),
      node.retryOnFail && num('waitBetweenTries', 'Wait between tries (ms)', 1000),
      field('On error', onError),
      field('Node version', ver, 'Newer versions can change behaviour and parameters.'),
      field('Notes', notes),
      toggle('notesInFlow', 'Show notes on canvas'),
      h('div.n8s-field-note.mt-3', 'Node id ', h('code', node.id || '—'), node.webhookId && [' · webhook id ', h('code', node.webhookId)]));
  }

  renderInput();
  renderCenter();
  renderOutput();
  return m;
}

function openStickyPanel(node, commit, readOnly) {
  const ta = h('textarea.form-control.n8s-sticky-edit', { rows: 12, readonly: readOnly, placeholder: '## Title\nMarkdown: **bold**, _italic_, `code`, [links](https://…), - lists' }, node.parameters?.content || '');
  let color = node.parameters?.color || 1;
  const swatches = h('div.n8s-swatches', STICKY_COLORS.map((c, i) => h('button.n8s-swatch', {
    type: 'button', style: { background: c }, class: color === i + 1 && 'on', disabled: readOnly,
    onclick: (e) => { color = i + 1; e.currentTarget.parentElement.querySelectorAll('.n8s-swatch').forEach((b, j) => b.classList.toggle('on', j === i)); save(); },
  })));
  const save = () => commit({ ...node, parameters: { ...node.parameters, content: ta.value, color } }, { label: 'Edit note', group: `sticky:${node.name}` });
  ta.addEventListener('input', save);
  return modal({ title: 'Sticky note', icon: 'sticky', body: h('div', ta, h('div.d-flex.align-items-center.gap-2.mt-2', h('span.small.text-muted', 'Colour'), swatches)), actions: [{ label: 'Done', variant: 'primary' }] });
}

