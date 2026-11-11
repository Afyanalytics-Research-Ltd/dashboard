// Connections: search/sort/paginate, create/edit/duplicate/test/delete,
// bulk delete, JSON export (without passwords) and bulk import.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import {
  pageHeader, card, dataTable, pager, searchInput, selectionBar, refreshControl, formModal,
  confirmDialog, menuButton, toast, toastError, emptyState, errorState, skeleton, jsonView, drawer, kv, copyButton,
} from '../core/ui.js';
import { fmtNumber } from '../core/format.js';
import {
  downloadJson, pickJsonFile, bulkToast, typeBadge, parseJsonish, isMasked, importDialog, fieldControl,
  containsMask, MASK_ERROR,
} from './admin_shared.js';

const FIELDS = ['conn_type', 'description', 'host', 'login', 'schema', 'port', 'password', 'extra'];

export default async function render(ctx) {
  const { el, signal } = ctx;
  const state = {
    q: ctx.query.get('q') || '',
    order: ctx.query.get('order') || 'connection_id',
    limit: Number(ctx.query.get('limit')) || 50,
    offset: Number(ctx.query.get('offset')) || 0,
  };
  const selected = new Set();
  let rows = [];
  let total = 0;
  let knownTypes = null;

  const tableEl = h('div', skeleton(8));
  const pagerEl = h('div');
  const barEl = h('div');
  const countEl = h('span.text-muted.small');
  const refresh = refreshControl(() => load(), { key: 'connections', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Connections', icon: 'plug',
      subtitle: 'Credentials and endpoints Airflow tasks use to reach external systems.',
      actions: [
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: importFile }, icon('upload', 'me-1'), 'Import'),
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: exportAll }, icon('download', 'me-1'), 'Export all'),
        h('button.btn.btn-sm.btn-primary', { onclick: () => openForm() }, icon('plus-lg', 'me-1'), 'New connection'),
        refresh.el,
      ],
    }),
    h('div.af-callout.af-callout-info', icon('shield-lock'),
      h('div', 'Airflow masks secrets in API responses: passwords and sensitive ', h('code', 'extra'), ' fields come back as ',
        h('code', '***'), '. Leaving the password blank while editing keeps the stored value.')),
    card({
      body: h('div',
        h('div.af-toolbar.af-card-toolbar',
          searchInput({ placeholder: 'Search connection id…', value: state.q, onInput: (q) => { state.q = q; state.offset = 0; sync(); load(); } }),
          h('div.af-grow'),
          countEl),
        tableEl, pagerEl),
      flush: true,
    }),
    barEl);

  function sync() {
    ctx.setQuery({
      q: state.q || null,
      order: state.order === 'connection_id' ? null : state.order,
      limit: state.limit === 50 ? null : state.limit,
      offset: state.offset || null,
    });
  }

  async function load() {
    try {
      const res = await api.get('/connections', {
        limit: state.limit, offset: state.offset, order_by: state.order,
        connection_id_pattern: state.q ? `%${state.q}%` : null,
      }, { signal });
      if (!ctx.active) return;
      rows = res.connections || [];
      total = res.total_entries || 0;
      draw();
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(tableEl, h('div.p-3', errorState(err, load)));
    }
  }

  function draw() {
    mount(countEl, `${fmtNumber(total)} connection${total === 1 ? '' : 's'}`);
    const [key, dir] = state.order.startsWith('-') ? [state.order.slice(1), 'desc'] : [state.order, 'asc'];
    mount(tableEl, dataTable({
      rows,
      rowKey: (r) => r.connection_id,
      selectable: true,
      selected,
      onSelect: drawBar,
      sort: { key, dir },
      onSort: (k, d) => { state.order = d === 'desc' ? `-${k}` : k; state.offset = 0; sync(); load(); },
      onRowClick: (r) => showDetail(r),
      empty: emptyState({
        icon: 'plug', title: state.q ? 'No matching connections' : 'No connections yet',
        message: state.q ? `Nothing matches “${state.q}”.` : 'Create one to let tasks reach external systems.',
        action: h('button.btn.btn-sm.btn-primary', { onclick: () => openForm() }, icon('plus-lg', 'me-1'), 'New connection'),
      }),
      columns: [
        {
          key: 'connection_id', label: 'Connection id', sortable: true,
          render: (r) => h('div', h('span.af-mono.fw-semibold', r.connection_id),
            r.description && h('div.small.text-muted.af-ellipsis', { style: { maxWidth: '280px', display: 'block' }, title: r.description }, r.description)),
        },
        { key: 'conn_type', label: 'Type', sortable: true, render: (r) => typeBadge(r.conn_type) },
        {
          key: 'host', label: 'Endpoint', sortable: true,
          render: (r) => (r.host || r.port || r.schema
            ? h('div.af-conn-endpoint',
              h('span.af-ellipsis.af-mono', { title: r.host || '' }, r.host || h('span.text-muted', 'no host'), r.port != null ? h('span.text-muted', `:${r.port}`) : null),
              r.schema && h('div.small.text-muted.af-ellipsis', { title: r.schema }, icon('database', 'me-1'), r.schema))
            : h('span.text-muted', '—')),
        },
        { key: 'login', label: 'Login', render: (r) => (r.login ? h('span.af-ellipsis', { style: { maxWidth: '160px' }, title: r.login }, r.login) : h('span.text-muted', '—')) },
        {
          key: 'actions', label: '', width: '48px', align: 'right',
          render: (r) => menuButton(null, [
            { label: 'Edit', icon: 'pencil', onClick: () => openForm(r) },
            { label: 'Duplicate', icon: 'copy', onClick: () => openForm(r, { duplicate: true }) },
            { label: 'Export JSON', icon: 'download', onClick: () => downloadJson(`connection-${r.connection_id}.json`, exportable([r])) },
            { divider: true },
            { label: 'Delete', icon: 'trash', danger: true, onClick: () => remove([r.connection_id]) },
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
      h('button.btn.btn-sm.btn-light', { onclick: () => downloadJson('connections.json', exportable(rows.filter((r) => selected.has(r.connection_id)))) }, icon('download', 'me-1'), 'Export'),
      h('button.btn.btn-sm.btn-danger', { onclick: () => remove([...selected]) }, icon('trash', 'me-1'), 'Delete'),
    ], () => { selected.clear(); draw(); }));
  }

  // ---------------------------------------------------------------- detail

  function showDetail(c) {
    const extra = parseJsonish(c.extra);
    const d = drawer({
      title: h('span.af-mono', c.connection_id),
      subtitle: c.description || null,
      width: 620,
      actions: [
        h('button.btn.btn-sm.btn-outline-primary', { onclick: () => { d.close(); openForm(c); } }, icon('pencil', 'me-1'), 'Edit'),
      ],
      body: h('div.d-flex.flex-column.gap-3',
        kv([
          ['Type', typeBadge(c.conn_type)],
          ['Port', c.port],
          ['Host', c.host && h('span.af-mono', c.host, ' ', copyButton(c.host)), { wide: true }],
          ['Login', c.login],
          ['Schema', c.schema],
          ['Password', c.password ? (isMasked(c.password) ? h('span.af-masked', icon('eye-slash'), ' set (masked)') : h('span.af-mono', c.password)) : h('span.text-muted', 'not set')],
        ]),
        h('div',
          h('div.af-section-label', 'Extra'),
          c.extra ? (extra.ok ? jsonView(extra.value) : h('pre.af-pre', c.extra)) : h('div.text-muted.small', 'No extra fields')),
        h('div.small.text-muted', 'Use the connection in tasks via ', h('code', `BaseHook.get_connection("${c.connection_id}")`), ' ', copyButton(`BaseHook.get_connection("${c.connection_id}")`))),
    });
  }

  // ---------------------------------------------------------------- create / edit

  async function loadTypes() {
    if (knownTypes) return knownTypes;
    try {
      const all = await api.all('/connections', 'connections', {}, { max: 1000, signal });
      knownTypes = [...new Set([...all.map((c) => c.conn_type), 'generic', 'http', 'postgres', 'mysql', 'snowflake', 'aws', 'google_cloud_platform', 'ssh', 'sftp', 'fs'])].filter(Boolean).sort();
    } catch { knownTypes = ['generic', 'http', 'postgres', 'mysql']; }
    return knownTypes;
  }

  async function openForm(conn, { duplicate = false } = {}) {
    const editing = !!conn && !duplicate;
    const extraParsed = parseJsonish(conn?.extra);
    const testOut = h('div.af-conn-test');
    let form;
    const m = formModal({
      title: editing ? `Edit ${conn.connection_id}` : duplicate ? `Duplicate ${conn.connection_id}` : 'New connection',
      icon: 'plug', size: 'lg',
      submitLabel: editing ? 'Save changes' : 'Create connection',
      fields: [
        editing
          ? { name: 'connection_id', label: 'Connection id', type: 'static', value: conn.connection_id }
          : { name: 'connection_id', label: 'Connection id', required: true, value: duplicate ? `${conn.connection_id}_copy` : '', placeholder: 'e.g. my_postgres' },
        { name: 'conn_type', label: 'Connection type', required: true, value: conn?.conn_type || '', placeholder: 'e.g. postgres, http, snowflake' },
        { name: 'description', label: 'Description', type: 'textarea', rows: 2, value: conn?.description || '' },
        { type: 'section', label: 'Endpoint' },
        { name: 'host', label: 'Host', value: conn?.host || '' },
        { name: 'port', label: 'Port', type: 'number', value: conn?.port ?? '', min: 0, max: 65535 },
        { name: 'schema', label: 'Schema / database', value: conn?.schema || '' },
        { type: 'section', label: 'Credentials' },
        { name: 'login', label: 'Login', value: conn?.login || '' },
        {
          name: 'password', label: 'Password', type: 'password', value: '',
          placeholder: editing || duplicate ? (conn.password ? 'unchanged — type to replace' : 'not set') : '',
          help: duplicate && conn.password ? 'Passwords are masked by Airflow and cannot be copied — enter it again.' : null,
        },
        { type: 'section', label: 'Extra' },
        {
          name: 'extra', label: 'Extra (JSON)', type: extraParsed.ok || !conn?.extra ? 'json' : 'textarea', rows: 6,
          value: extraParsed.ok ? extraParsed.value : (conn?.extra || {}),
          help: conn?.extra && isMasked(conn.extra) ? 'Extra is masked by Airflow.' : null,
        },
      ],
      extra: h('div.af-conn-test-wrap',
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: (e) => testConnection(e.currentTarget) }, icon('lightning', 'me-1'), 'Test connection'),
        testOut),
      onSubmit: async (values, close) => {
        const body = toBody(values, editing ? conn.connection_id : values.connection_id);
        if (editing) {
          const changed = FIELDS.filter((f) => {
            if (f === 'password') return !!values.password;
            if (f === 'extra') return (body.extra || null) !== (normalizeExtra(conn.extra) || null);
            return (body[f] ?? null) !== (conn[f] ?? null);
          });
          if (!changed.length) { toast('Nothing changed', 'info'); close(); return; }
          if (changed.includes('extra') && containsMask(body.extra)) throw new Error(`Extra: ${MASK_ERROR}`);
          if (!values.password) delete body.password;
          await api.patch(`/connections/${enc(conn.connection_id)}`, body, { params: { update_mask: changed } });
          toast(`${conn.connection_id} updated (${changed.join(', ')})`, 'success');
        } else {
          if (containsMask(body.extra)) throw new Error(`Extra: ${MASK_ERROR}`);
          await api.post('/connections', body);
          toast(`${body.connection_id} created`, 'success');
        }
        close();
        load();
      },
    });
    form = m.form;
    form.el.classList.add('af-conn-form');

    const typeInput = fieldControl(form.el, 'Connection type');
    if (typeInput) {
      const listId = `af-conn-types-${Math.random().toString(36).slice(2, 7)}`;
      typeInput.setAttribute('list', listId);
      const dl = h('datalist', { id: listId });
      typeInput.after(dl);
      loadTypes().then((types) => mount(dl, types.map((t) => h('option', { value: t }))));
    }

    async function testConnection(btn) {
      let values;
      try { values = form.values(); } catch (err) { mount(testOut, h('div.af-test-result.bad', icon('x-circle'), err.message)); return; }
      const body = toBody(values, editing ? conn.connection_id : values.connection_id || 'connection_test');
      if (editing && !values.password) delete body.password;
      btn.disabled = true;
      mount(testOut, h('span.small.text-muted', 'Testing…'));
      try {
        const res = await api.post('/connections/test', body);
        mount(testOut, h('div.af-test-result', { class: res.status ? 'ok' : 'bad' },
          icon(res.status ? 'check-circle-fill' : 'x-circle-fill'), h('span', res.message || (res.status ? 'Connection OK' : 'Connection failed'))));
      } catch (err) {
        const disabled = err.status === 403 || /disabled/i.test(err.message);
        mount(testOut, h('div.af-test-result', { class: disabled ? 'warn' : 'bad' },
          icon(disabled ? 'slash-circle' : 'x-circle-fill'),
          h('span', disabled ? 'Connection testing is disabled on this Airflow server (core.test_connection).' : err.message)));
      } finally {
        btn.disabled = false;
      }
    }
    return m;
  }

  function normalizeExtra(extra) {
    if (extra == null || extra === '') return null;
    if (typeof extra === 'object') return JSON.stringify(extra);
    const p = parseJsonish(extra);
    return p.ok ? JSON.stringify(p.value) : extra;
  }

  function toBody(v, connectionId) {
    const blank = (x) => (x === '' || x == null ? null : x);
    return {
      connection_id: connectionId,
      conn_type: v.conn_type.trim(),
      description: blank(v.description),
      host: blank(v.host),
      login: blank(v.login),
      schema: blank(v.schema),
      port: v.port == null || Number.isNaN(v.port) ? null : v.port,
      password: blank(v.password),
      extra: normalizeExtra(v.extra),
    };
  }

  // ---------------------------------------------------------------- delete / export / import

  async function remove(ids) {
    const ok = await confirmDialog({
      title: ids.length === 1 ? `Delete ${ids[0]}?` : `Delete ${ids.length} connections?`,
      message: 'Tasks that use these connections will fail until they are recreated.',
      details: ids.length > 1 ? h('div.af-preview.small.af-mono', ids.join('\n')) : null,
      confirmLabel: 'Delete', danger: true,
    });
    if (!ok) return;
    try {
      if (ids.length === 1) {
        await api.del(`/connections/${enc(ids[0])}`);
        toast(`${ids[0]} deleted`, 'success');
      } else {
        const res = await api.patch('/connections', { actions: [{ action: 'delete', entities: ids, action_on_non_existence: 'skip' }] });
        bulkToast(res, 'deleted');
      }
      ids.forEach((id) => selected.delete(id));
      load();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  function exportable(list) {
    return list.map(({ password, ...rest }) => {
      const p = parseJsonish(rest.extra);
      return { ...rest, extra: p.ok ? p.value : rest.extra };
    });
  }

  async function exportAll() {
    try {
      const all = await api.all('/connections', 'connections', { order_by: 'connection_id' }, { signal });
      downloadJson('connections.json', exportable(all));
      toast(`${all.length} connections exported (passwords excluded)`, 'success');
    } catch (err) { toastError(err, 'Export failed'); }
  }

  async function importFile() {
    let picked;
    try { picked = await pickJsonFile(); } catch (err) { toastError(err, 'Import failed'); return; }
    if (!picked) return;
    // Accept [ {connection_id, ...} ] or { conn_id: {...} } (Airflow CLI export format).
    const raw = Array.isArray(picked.data)
      ? picked.data
      : Object.entries(picked.data || {}).map(([id, c]) => ({ connection_id: id, ...c }));
    const entities = raw.map((c) => ({
      connection_id: c.connection_id || c.conn_id,
      conn_type: c.conn_type,
      description: c.description ?? null,
      host: c.host ?? null,
      login: c.login ?? null,
      schema: c.schema ?? null,
      port: c.port == null || c.port === '' ? null : Number(c.port),
      password: c.password ?? null,
      extra: normalizeExtra(c.extra),
    }));
    const invalid = entities.filter((e) => !e.connection_id || !e.conn_type);
    if (!entities.length || invalid.length) {
      toast(invalid.length ? `${invalid.length} item(s) lack connection_id or conn_type` : 'No connections found in file', 'error', { title: 'Import failed' });
      return;
    }
    const masked = entities.filter((e) => isMasked(e.password) || containsMask(e.extra));
    if (masked.length) {
      toast(`${masked.map((e) => e.connection_id).slice(0, 5).join(', ')}${masked.length > 5 ? '…' : ''} contain masked values (***), probably from an API export. Fill in the real secrets first.`, 'error', { title: 'Import blocked', timeout: 9000 });
      return;
    }
    importDialog({
      title: `Import connections from ${picked.name}`,
      entities,
      rowKey: (e) => e.connection_id,
      columns: [
        { key: 'connection_id', label: 'Connection id', render: (e) => h('span.af-mono', e.connection_id) },
        { key: 'conn_type', label: 'Type', render: (e) => typeBadge(e.conn_type) },
        { key: 'host', label: 'Host' },
        { key: 'password', label: 'Password', render: (e) => (e.password ? icon('key-fill', 'text-muted') : h('span.text-muted', '—')) },
      ],
      onConfirm: async (list, onExistence) => {
        const res = await api.patch('/connections', { actions: [{ action: 'create', entities: list, action_on_existence: onExistence }] });
        bulkToast(res, 'imported');
        load();
      },
    });
  }

  await load();
}
