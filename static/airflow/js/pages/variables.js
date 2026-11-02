// Variables: search/sort/paginate, view, create/edit with text or JSON
// values, delete, bulk delete, JSON export and bulk import.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import {
  pageHeader, card, dataTable, pager, searchInput, selectionBar, refreshControl, modal, buildForm, jsonEditor,
  confirmDialog, menuButton, toast, toastError, emptyState, errorState, skeleton, jsonView, drawer, copyButton,
} from '../core/ui.js';
import { fmtNumber, fmtBytes } from '../core/format.js';
import {
  downloadJson, pickJsonFile, bulkToast, parseJsonish, isMasked, valuePreview, importDialog, segmented,
  containsMask, sameValue, MASK_ERROR,
} from './admin_shared.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  const state = {
    q: ctx.query.get('q') || '',
    order: ctx.query.get('order') || 'key',
    limit: Number(ctx.query.get('limit')) || 50,
    offset: Number(ctx.query.get('offset')) || 0,
  };
  const selected = new Set();
  let rows = [];
  let total = 0;

  const tableEl = h('div', skeleton(8));
  const pagerEl = h('div');
  const barEl = h('div');
  const countEl = h('span.text-muted.small');
  const refresh = refreshControl(() => load(), { key: 'variables', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Variables', icon: 'braces-asterisk',
      subtitle: 'Key/value settings shared by every DAG (Variable.get).',
      actions: [
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: importFile }, icon('upload', 'me-1'), 'Import'),
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: exportAll }, icon('download', 'me-1'), 'Export all'),
        h('button.btn.btn-sm.btn-primary', { onclick: () => openForm() }, icon('plus-lg', 'me-1'), 'New variable'),
        refresh.el,
      ],
    }),
    card({
      body: h('div',
        h('div.af-toolbar.af-card-toolbar',
          searchInput({ placeholder: 'Search key…', value: state.q, onInput: (q) => { state.q = q; state.offset = 0; sync(); load(); } }),
          h('div.af-grow'),
          countEl),
        tableEl, pagerEl),
      flush: true,
    }),
    barEl);

  function sync() {
    ctx.setQuery({
      q: state.q || null,
      order: state.order === 'key' ? null : state.order,
      limit: state.limit === 50 ? null : state.limit,
      offset: state.offset || null,
    });
  }

  async function load() {
    try {
      const res = await api.get('/variables', {
        limit: state.limit, offset: state.offset, order_by: state.order,
        variable_key_pattern: state.q ? `%${state.q}%` : null,
      }, { signal });
      if (!ctx.active) return;
      rows = res.variables || [];
      total = res.total_entries || 0;
      draw();
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(tableEl, h('div.p-3', errorState(err, load)));
    }
  }

  function draw() {
    mount(countEl, `${fmtNumber(total)} variable${total === 1 ? '' : 's'}`);
    const [key, dir] = state.order.startsWith('-') ? [state.order.slice(1), 'desc'] : [state.order, 'asc'];
    mount(tableEl, dataTable({
      rows,
      rowKey: (r) => r.key,
      selectable: true,
      selected,
      onSelect: drawBar,
      sort: { key, dir },
      onSort: (k, d) => { state.order = d === 'desc' ? `-${k}` : k; state.offset = 0; sync(); load(); },
      onRowClick: (r) => showDetail(r),
      empty: emptyState({
        icon: 'braces-asterisk', title: state.q ? 'No matching variables' : 'No variables yet',
        message: state.q ? `Nothing matches “${state.q}”.` : null,
        action: h('button.btn.btn-sm.btn-primary', { onclick: () => openForm() }, icon('plus-lg', 'me-1'), 'New variable'),
      }),
      columns: [
        { key: 'key', label: 'Key', sortable: true, render: (r) => h('span.af-mono.fw-semibold', r.key) },
        { key: '_val', label: 'Value', sortable: '_val', render: (r) => valuePreview(r.value, 70) },
        { key: 'description', label: 'Description', sortable: true, render: (r) => (r.description ? h('span.af-ellipsis', { style: { maxWidth: '260px' }, title: r.description }, r.description) : h('span.text-muted', '—')) },
        { key: 'is_encrypted', label: 'Encrypted', sortable: true, align: 'center', render: (r) => (r.is_encrypted ? icon('lock-fill', 'text-success') : icon('unlock', 'text-muted')) },
        {
          key: 'actions', label: '', width: '48px', align: 'right',
          render: (r) => menuButton(null, [
            { label: 'View', icon: 'eye', onClick: () => showDetail(r) },
            { label: 'Edit', icon: 'pencil', onClick: () => openForm(r) },
            { label: 'Copy value', icon: 'clipboard', onClick: () => copyText(r.value) },
            { divider: true },
            { label: 'Delete', icon: 'trash', danger: true, onClick: () => remove([r.key]) },
          ]),
        },
      ],
    }));
    mount(pagerEl, total > state.limit || state.offset ? pager({
      total, limit: state.limit, offset: state.offset,
      onChange: (offset, limit) => { state.offset = offset; state.limit = limit; sync(); load(); },
    }) : null);
    drawBar();
  }

  function drawBar() {
    mount(barEl, selectionBar(selected.size, [
      h('button.btn.btn-sm.btn-light', { onclick: () => downloadJson('variables.json', exportable(rows.filter((r) => selected.has(r.key)))) }, icon('download', 'me-1'), 'Export'),
      h('button.btn.btn-sm.btn-danger', { onclick: () => remove([...selected]) }, icon('trash', 'me-1'), 'Delete'),
    ], () => { selected.clear(); draw(); }));
  }

  async function copyText(text) {
    try { await navigator.clipboard.writeText(text ?? ''); toast('Value copied', 'success', { timeout: 1500 }); } catch { toast('Clipboard unavailable', 'warning'); }
  }

  // ---------------------------------------------------------------- detail

  function showDetail(v) {
    const parsed = parseJsonish(v.value);
    const d = drawer({
      title: h('span.af-mono', v.key),
      subtitle: v.description || null,
      width: 680,
      actions: [
        copyButton(v.value ?? '', 'Copy value'),
        h('button.btn.btn-sm.btn-outline-primary', { onclick: () => { d.close(); openForm(v); } }, icon('pencil', 'me-1'), 'Edit'),
      ],
      body: h('div.d-flex.flex-column.gap-3',
        h('div.d-flex.gap-2.flex-wrap.small',
          h('span.af-tag', parsed.ok ? 'JSON' : 'text'),
          h('span.af-tag', fmtBytes(new Blob([v.value ?? '']).size)),
          v.is_encrypted ? h('span.af-tag', icon('lock-fill'), 'encrypted at rest') : h('span.af-tag', icon('unlock'), 'not encrypted'),
          isMasked(v.value) && h('span.af-tag', icon('eye-slash'), 'masked by Airflow')),
        parsed.ok ? jsonView(parsed.value) : h('pre.af-pre', v.value ?? ''),
        h('div.small.text-muted', 'In a DAG: ', h('code', parsed.ok ? `Variable.get("${v.key}", deserialize_json=True)` : `Variable.get("${v.key}")`))),
    });
  }

  // ---------------------------------------------------------------- create / edit

  function openForm(v) {
    const editing = !!v;
    const masked = editing && isMasked(v.value);
    const parsed = parseJsonish(v?.value);
    let mode = parsed.ok ? 'json' : 'text';
    const meta = buildForm([
      editing
        ? { name: 'key', label: 'Key', type: 'static', value: v.key }
        : { name: 'key', label: 'Key', required: true, placeholder: 'e.g. MY_SETTING' },
      { name: 'description', label: 'Description', type: 'textarea', rows: 2, value: v?.description || '' },
    ]);
    const textArea = h('textarea.form-control.form-control-sm.af-mono', { rows: 8, spellcheck: 'false', placeholder: masked ? 'masked — type a new value to replace it' : '' },
      masked || parsed.ok ? '' : (v?.value ?? ''));
    const json = jsonEditor(parsed.ok ? parsed.value : {}, { rows: 12, allowEmpty: false });
    const host = h('div');
    const show = () => mount(host, mode === 'json' ? json.el : textArea);
    const switchMode = (m) => {
      if (m === 'json') {
        const p = parseJsonish(textArea.value);
        if (p.ok) json.set(p.value);
        else if (textArea.value.trim()) json.set(textArea.value.trim());
      } else {
        try { const val = json.value(); textArea.value = typeof val === 'string' ? val : JSON.stringify(val, null, 2); } catch { textArea.value = json.textarea.value; }
      }
      mode = m;
      show();
    };
    show();
    const errorBox = h('div.alert.alert-danger.py-2.small.d-none');
    modal({
      title: editing ? `Edit ${v.key}` : 'New variable', icon: 'braces-asterisk', size: 'lg',
      body: h('div', errorBox, meta.el,
        h('div.d-flex.justify-content-between.align-items-center.mt-3.mb-1',
          h('label.form-label.small.fw-semibold.mb-0', 'Value'),
          segmented([{ value: 'text', label: 'Text', icon: 'fonts' }, { value: 'json', label: 'JSON', icon: 'braces' }], mode, switchMode)),
        masked && h('div.af-callout.af-callout-warn.mb-2', icon('eye-slash'), 'Airflow masks this value. Leave it empty to keep the stored value and only update the description.'),
        !masked && containsMask(v?.value) && h('div.af-callout.af-callout-warn.mb-2', icon('eye-slash'),
          h('div', 'Parts of this value are masked by Airflow (', h('code', '***'), '). Saving works only if you replace every masked part, or leave the value untouched and change just the description.')),
        host),
      actions: [
        { label: 'Cancel' },
        {
          label: editing ? 'Save changes' : 'Create variable', variant: 'primary',
          onClick: async (close) => {
            errorBox.classList.add('d-none');
            try {
              const m = meta.values();
              let value;
              if (mode === 'json') {
                const parsedValue = json.value();
                value = typeof parsedValue === 'string' ? parsedValue : JSON.stringify(parsedValue, null, 2);
              } else {
                value = textArea.value;
              }
              const key = editing ? v.key : m.key.trim();
              const description = m.description || null;
              if (editing) {
                const mask = [];
                const valueChanged = !(masked && value === '') && !sameValue(value, v.value);
                if (valueChanged && containsMask(value)) throw new Error(MASK_ERROR);
                if (valueChanged) mask.push('value');
                if ((description ?? null) !== (v.description ?? null)) mask.push('description');
                if (!mask.length) { toast('Nothing changed', 'info'); close(); return; }
                await api.patch(`/variables/${enc(key)}`, { key, value: mask.includes('value') ? value : v.value, description }, { params: { update_mask: mask } });
                toast(`${key} updated`, 'success');
              } else {
                if (containsMask(value)) throw new Error(MASK_ERROR);
                await api.post('/variables', { key, value, description });
                toast(`${key} created`, 'success');
              }
              close();
              load();
            } catch (err) {
              mount(errorBox, icon('exclamation-octagon', 'me-1'), err.message);
              errorBox.classList.remove('d-none');
            }
          },
        },
      ],
    });
  }

  // ---------------------------------------------------------------- delete / export / import

  async function remove(keys) {
    const ok = await confirmDialog({
      title: keys.length === 1 ? `Delete ${keys[0]}?` : `Delete ${keys.length} variables?`,
      message: 'DAGs reading these variables will fail or fall back to defaults.',
      details: keys.length > 1 ? h('div.af-preview.small.af-mono', keys.join('\n')) : null,
      confirmLabel: 'Delete', danger: true,
    });
    if (!ok) return;
    try {
      if (keys.length === 1) {
        await api.del(`/variables/${enc(keys[0])}`);
        toast(`${keys[0]} deleted`, 'success');
      } else {
        const res = await api.patch('/variables', { actions: [{ action: 'delete', entities: keys, action_on_non_existence: 'skip' }] });
        bulkToast(res, 'deleted');
      }
      keys.forEach((k) => selected.delete(k));
      load();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  /** Airflow CLI format: { key: value }, JSON values unpacked. */
  function exportable(list) {
    return Object.fromEntries(list.map((v) => {
      const p = parseJsonish(v.value);
      return [v.key, p.ok ? p.value : v.value];
    }));
  }

  async function exportAll() {
    try {
      const all = await api.all('/variables', 'variables', { order_by: 'key' }, { signal });
      downloadJson('variables.json', exportable(all));
      const masked = all.filter((v) => containsMask(v.value)).length;
      toast(`${all.length} variables exported${masked ? ` — ${masked} contain values masked by Airflow (***), which cannot be re-imported as-is` : ''}`, masked ? 'warning' : 'success');
    } catch (err) { toastError(err, 'Export failed'); }
  }

  async function importFile() {
    let picked;
    try { picked = await pickJsonFile(); } catch (err) { toastError(err, 'Import failed'); return; }
    if (!picked) return;
    // Accept { key: value } (CLI format) or [ {key, value, description} ].
    const raw = Array.isArray(picked.data)
      ? picked.data
      : Object.entries(picked.data || {}).map(([key, value]) => ({ key, value }));
    const entities = raw.filter((x) => x && x.key).map((x) => ({
      key: String(x.key),
      value: typeof x.value === 'string' ? x.value : JSON.stringify(x.value, null, 2),
      description: x.description ?? null,
    }));
    if (!entities.length) { toast('No variables found in file', 'error', { title: 'Import failed' }); return; }
    const masked = entities.filter((e) => containsMask(e.value));
    if (masked.length) {
      toast(`${masked.map((e) => e.key).slice(0, 5).join(', ')}${masked.length > 5 ? '…' : ''} contain masked values (***), probably from an API export. Fill in the real values first.`, 'error', { title: 'Import blocked', timeout: 9000 });
      return;
    }
    importDialog({
      title: `Import variables from ${picked.name}`,
      entities,
      rowKey: (e) => e.key,
      columns: [
        { key: 'key', label: 'Key', render: (e) => h('span.af-mono', e.key) },
        { key: 'value', label: 'Value', render: (e) => valuePreview(e.value, 60) },
      ],
      onConfirm: async (list, onExistence) => {
        const res = await api.patch('/variables', { actions: [{ action: 'create', entities: list, action_on_existence: onExistence }] });
        bulkToast(res, 'imported');
        load();
      },
    });
  }

  await load();
}
