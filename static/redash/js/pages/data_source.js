// Data source detail (/data-sources/:id, or /data-sources/new?type=pg).
// Tabs: Settings (schema-driven form, test connection), Schema (searchable browser),
// Access (groups + view-only), and pause / delete.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import {
  pageHeader, card, tabs, emptyState, errorState, skeleton, toast, toastError, confirmDialog,
  modal, buildForm, spinner, copyButton,
} from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc, getSchema } from '../core/api.js';
import { urls } from '../core/urls.js';
import { invalidate } from '../core/common.js';
import { configForm, typeLogo, pill, hl } from './admin_shared.js';

/** Ask for a reason and pause a data source. Resolves with the updated source or null. */
export function pauseDialog(ds) {
  return new Promise((resolve) => {
    let updated = null;
    const form = buildForm([{ name: 'reason', label: 'Reason (shown to users)', placeholder: 'e.g. Database maintenance until 14:00' }]);
    modal({
      title: `Pause ${ds.name}?`, icon: 'pause-circle', size: 'sm',
      body: h('div', h('p.small.text-muted', 'Queries against a paused source are rejected until it is resumed. Scheduled refreshes are skipped.'), form.el),
      actions: [
        { label: 'Cancel' },
        {
          label: 'Pause data source', variant: 'warning', icon: 'pause-fill',
          onClick: async (close) => {
            const { reason } = form.values();
            updated = await rd.post(`/api/data_sources/${enc(ds.id)}/pause`, { reason: reason || null });
            toast(`${ds.name} paused`, 'success');
            close();
          },
        },
      ],
      onClose: () => resolve(updated),
    });
  });
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  const id = ctx.params.dataSourceId;
  const isNew = id === 'new';
  let types;
  let ds;
  try {
    [types, ds] = await Promise.all([
      rd.get('/api/data_sources/types', null, { signal }),
      isNew ? null : rd.get(`/api/data_sources/${enc(id)}`, null, { signal }),
    ]);
  } catch (err) {
    mount(el, pageHeader({ title: 'Data source', icon: 'database', crumbs: [{ label: 'Data sources', href: urls.dataSources() }] }), errorState(err));
    return;
  }
  const typeKey = isNew ? ctx.query.get('type') : ds.type;
  const type = types.find((t) => t.type === typeKey);
  if (!type) {
    mount(el, pageHeader({ title: 'Unknown type', icon: 'database' }), errorState(new Error(`Redash has no data source type "${typeKey}".`)));
    return;
  }
  if (!isNew) ctx.setTitle(ds.name);

  let tab = isNew ? 'settings' : ctx.query.get('tab') || 'settings';
  const pane = h('div');
  const headerHost = h('div');

  const drawHeader = () => mount(headerHost, pageHeader({
    title: isNew ? `New ${type.name} data source` : ds.name,
    icon: null,
    crumbs: [{ label: 'Data sources', href: urls.dataSources() }, { label: isNew ? 'New' : ds.name }],
    badges: !isNew && (ds.paused ? pill(`Paused${ds.pause_reason ? `: ${ds.pause_reason}` : ''}`, 'warn', 'pause-circle-fill') : pill('Active', 'ok', 'circle-fill')),
    subtitle: h('span.d-inline-flex.align-items-center.gap-2', typeLogo(type.type, 'db', 22), type.name, !isNew && h('span.text-muted', `· id ${ds.id}`)),
    actions: !isNew && [
      h('a.btn.btn-sm.btn-outline-primary', { href: urls.newQuery({ data_source_id: ds.id }) }, icon('code-square', 'me-1'), 'New query'),
      ds.paused
        ? h('button.btn.btn-sm.btn-outline-success', { onclick: resume }, icon('play-fill', 'me-1'), 'Resume')
        : h('button.btn.btn-sm.btn-outline-warning', { onclick: pause }, icon('pause-fill', 'me-1'), 'Pause'),
      h('button.btn.btn-sm.btn-outline-danger', { onclick: remove }, icon('trash', 'me-1'), 'Delete'),
    ],
  }));

  const tabBar = !isNew && tabs([
    { id: 'settings', label: 'Connection', icon: 'sliders' },
    { id: 'schema', label: 'Schema', icon: 'diagram-3' },
    { id: 'access', label: 'Access', icon: 'people' },
  ], tab, (t) => { tab = t; ctx.setQuery({ tab: t === 'settings' ? null : t }); show(); });

  drawHeader();
  mount(el, headerHost, card({ body: h('div', tabBar, pane), cls: 'rd-ds-detail' }));

  function show() {
    if (tab === 'schema') return showSchema();
    if (tab === 'access') return showAccess();
    return showSettings();
  }

  // ---- settings
  function showSettings() {
    const name = h('input.form-control', { value: ds?.name || '', placeholder: `e.g. ${type.name} production`, autofocus: isNew });
    const form = configForm(type.configuration_schema, ds?.options || {}, { isNew });
    const errorBox = h('div.alert.alert-danger.small.py-2.d-none');
    const testOut = h('div.rd-ds-test-out');
    const saveBtn = h('button.btn.btn-primary', { type: 'button' }, icon('check2', 'me-1'), isNew ? 'Create data source' : 'Save');
    const testBtn = !isNew && h('button.btn.btn-outline-secondary', { type: 'button', onclick: () => test(testBtn) }, icon('plug', 'me-1'), 'Test connection');

    async function test(btn) {
      btn.disabled = true;
      mount(testOut, h('div.rd-ds-test.rd-ds-test-running', spinner('sm'), ' Testing the saved connection…'));
      try {
        const r = await rd.post(`/api/data_sources/${enc(ds.id)}/test`, {});
        mount(testOut, h(`div.rd-ds-test.rd-ds-test-${r.ok ? 'ok' : 'bad'}`, icon(r.ok ? 'check-circle-fill' : 'x-octagon-fill'), ' ', r.ok ? 'Connection successful' : r.message));
      } catch (err) {
        mount(testOut, h('div.rd-ds-test.rd-ds-test-bad', icon('x-octagon-fill'), ' ', err.message));
      } finally { btn.disabled = false; }
    }

    saveBtn.addEventListener('click', async () => {
      errorBox.classList.add('d-none');
      let options;
      try {
        if (!name.value.trim()) throw new Error('Give the data source a name.');
        options = form.values();
      } catch (err) {
        mount(errorBox, icon('exclamation-octagon', 'me-1'), err.message);
        errorBox.classList.remove('d-none');
        return;
      }
      saveBtn.disabled = true;
      try {
        const body = { name: name.value.trim(), type: type.type, options };
        const saved = isNew ? await rd.post('/api/data_sources', body) : await rd.post(`/api/data_sources/${enc(ds.id)}`, body);
        invalidate('ds');
        toast(isNew ? `${saved.name} connected` : 'Saved', 'success');
        if (isNew) navigate(urls.dataSource(saved.id));
        else { ds = saved; drawHeader(); showSettings(); }
      } catch (err) {
        mount(errorBox, icon('exclamation-octagon', 'me-1'), err.status === 400 ? `Redash rejected the settings: ${err.message}` : err.message);
        errorBox.classList.remove('d-none');
      } finally { saveBtn.disabled = false; }
    });

    mount(pane, h('div.rd-ds-settings',
      h('div.rd-ds-form',
        errorBox,
        h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Name', h('span.text-danger', ' *')), name),
        form.el,
        h('div.d-flex.gap-2.mt-3', saveBtn, testBtn),
        testOut),
      h('aside.rd-ds-help',
        h('div.rd-ds-help-head', typeLogo(type.type, 'db', 40), h('div', h('div.fw-semibold', type.name), h('div.small.text-muted', type.type))),
        h('ul.rd-howto',
          h('li', 'Secret fields (marked ', icon('lock-fill'), ') are stored encrypted and never shown again. Leave them blank to keep the saved value.'),
          h('li', 'Redash connects from its own server, so hosts must be reachable from there (e.g. ', h('code', 'host.docker.internal'), ').'),
          !isNew && h('li', 'Test connection checks the ', h('b', 'saved'), ' settings — save first after editing.'),
          h('li', 'Give read-only database credentials where possible.')))));
  }

  // ---- schema
  async function showSchema(refresh = false) {
    mount(pane, h('div.rda-schema-loading', spinner(), h('span', refresh ? ' Refreshing schema from the database…' : ' Loading schema…')));
    let schema;
    try {
      schema = await getSchema(ds.id, { refresh, signal });
    } catch (err) {
      if (err.name !== 'AbortError') mount(pane, errorState(err, () => showSchema(refresh)));
      return;
    }
    if (!ctx.active || tab !== 'schema') return;
    schema = [...schema].sort((a, b) => a.name.localeCompare(b.name));
    const totalCols = schema.reduce((n, t) => n + (t.columns?.length || 0), 0);
    let q = '';
    const list = h('div.rda-schema-list');
    const draw = () => {
      const ql = q.toLowerCase();
      const hits = schema.map((t) => {
        const cols = (t.columns || []).map((c) => (typeof c === 'string' ? { name: c } : c));
        if (!ql) return { t, cols, open: false };
        const tableHit = t.name.toLowerCase().includes(ql);
        const colHits = cols.filter((c) => c.name.toLowerCase().includes(ql));
        if (!tableHit && !colHits.length) return null;
        return { t, cols: tableHit ? cols : colHits, open: !!colHits.length };
      }).filter(Boolean);
      mount(list, hits.length ? hits.slice(0, 400).map(({ t, cols, open }) => h('details.rda-schema-table', { open },
        h('summary', icon('table', 'me-2'), h('span.af-mono', hl(t.name, q)), h('span.rda-schema-count', String(t.columns?.length || 0)), copyButton(t.name)),
        h('div.rda-schema-cols', cols.map((c) => h('div.rda-schema-col',
          icon('dot'), h('span.af-mono', hl(c.name, q)), c.type && h('span.rda-col-type', c.type))))))
        : emptyState({ icon: 'search', title: 'Nothing matches', message: 'Try another table or column name.' }),
      hits.length > 400 && h('div.small.text-muted.p-2', `Showing 400 of ${hits.length} tables — refine the search.`));
    };
    mount(pane, h('div.d-flex.flex-column.gap-3',
      h('div.af-toolbar',
        h('div.af-search', icon('search', 'af-search-icon'),
          h('input.form-control.form-control-sm', { type: 'search', placeholder: `Filter ${schema.length.toLocaleString()} tables and ${totalCols.toLocaleString()} columns…`, dataset: { searchShortcut: '' }, oninput: debounce((e) => { q = e.target.value.trim(); draw(); }, 120) })),
        h('span.small.text-muted', `${schema.length.toLocaleString()} tables · ${totalCols.toLocaleString()} columns`),
        h('button.btn.btn-sm.btn-outline-secondary.ms-auto', { onclick: () => showSchema(true) }, icon('arrow-clockwise', 'me-1'), 'Refresh from database')),
      schema.length ? list : emptyState({ icon: 'diagram-3', title: 'No tables found', message: 'The data source returned an empty schema. Check permissions or refresh.' })));
    if (schema.length) draw();
  }

  // ---- access
  async function showAccess() {
    mount(pane, skeleton(4));
    let groups;
    try {
      [groups, ds] = await Promise.all([rd.get('/api/groups', null, { signal }), rd.get(`/api/data_sources/${enc(ds.id)}`, null, { signal })]);
    } catch (err) {
      if (err.name !== 'AbortError') mount(pane, errorState(err, showAccess));
      return;
    }
    if (!ctx.active || tab !== 'access') return;
    const access = ds.groups || {}; // { group_id: view_only }
    const row = (g) => {
      const has = String(g.id) in access;
      const viewOnly = !!access[g.id];
      const toggle = h('input.form-check-input', { type: 'checkbox', checked: has, 'aria-label': `Give ${g.name} access` });
      const vo = h('input.form-check-input', { type: 'checkbox', checked: viewOnly, disabled: !has, 'aria-label': 'View only' });
      toggle.addEventListener('change', async () => {
        toggle.disabled = true;
        try {
          if (toggle.checked) await rd.post(`/api/groups/${enc(g.id)}/data_sources`, { data_source_id: ds.id });
          else await rd.del(`/api/groups/${enc(g.id)}/data_sources/${enc(ds.id)}`);
          toast(toggle.checked ? `${g.name} can now use ${ds.name}` : `${g.name} no longer has access`, 'success');
          showAccess();
        } catch (err) { toggle.checked = !toggle.checked; toastError(err, 'Could not change access'); toggle.disabled = false; }
      });
      vo.addEventListener('change', async () => {
        vo.disabled = true;
        try {
          await rd.post(`/api/groups/${enc(g.id)}/data_sources/${enc(ds.id)}`, { view_only: vo.checked });
          toast(vo.checked ? `${g.name}: view only` : `${g.name}: full access`, 'success');
        } catch (err) { vo.checked = !vo.checked; toastError(err, 'Could not change permission'); }
        vo.disabled = false;
      });
      return h('tr', { class: has && 'rd-access-on' },
        h('td', h('label.form-check.form-switch.m-0', toggle)),
        h('td', h('a.fw-semibold', { href: urls.group(g.id) }, g.name), g.type === 'builtin' && pill('built-in', 'muted')),
        h('td', h('label.form-check.form-switch.m-0.d-inline-flex.align-items-center.gap-2', vo, h('span.small', viewOnly ? 'View only' : 'Full access'))),
        h('td.small.text-muted', has ? (viewOnly ? 'Can view results, cannot write or run new queries' : 'Can write and run queries') : 'No access'));
    };
    mount(pane, h('div',
      h('p.small.text-muted', 'Users get a data source through the groups they belong to. "View only" lets a group see existing queries and dashboards but not write or run new SQL.'),
      h('div.af-table-wrap', h('table.af-table',
        h('thead', h('tr', h('th', 'Access'), h('th', 'Group'), h('th', 'Permission'), h('th', ''))),
        h('tbody', groups.map(row))))));
  }

  async function pause() {
    const updated = await pauseDialog(ds);
    if (updated) { ds = { ...ds, ...updated }; invalidate('ds'); drawHeader(); }
  }
  async function resume() {
    try {
      ds = { ...ds, ...(await rd.del(`/api/data_sources/${enc(ds.id)}/pause`)) };
      invalidate('ds');
      toast(`${ds.name} resumed`, 'success');
      drawHeader();
    } catch (err) { toastError(err, 'Could not resume'); }
  }
  async function remove() {
    const ok = await confirmDialog({
      title: `Delete ${ds.name}?`, danger: true, confirmLabel: 'Delete data source',
      message: 'Queries using this data source stop working and their results are deleted. This cannot be undone.',
    });
    if (!ok) return;
    try {
      await rd.del(`/api/data_sources/${enc(ds.id)}`);
      invalidate('ds');
      toast(`${ds.name} deleted`, 'success');
      navigate(urls.dataSources());
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  show();
}
