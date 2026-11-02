// XComs: every XCom entry across DAGs with filters; the drawer loads the
// full value (native / deserialized / stringified).

import { h, icon, mount, debounce } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  pageHeader, card, dataTable, pager, refreshControl, emptyState, errorState, skeleton, jsonView, drawer,
  copyButton, kv, timeAgo,
} from '../core/ui.js';
import { fmtDate, fmtNumber, fmtBytes } from '../core/format.js';
import { segmented, parseJsonish } from './admin_shared.js';

const FILTERS = [
  { key: 'xcom_key_pattern', label: 'Key', placeholder: 'e.g. return_value' },
  { key: 'task_id_pattern', label: 'Task', placeholder: 'task id contains…' },
  { key: 'run_id_pattern', label: 'Run', placeholder: 'run id contains…' },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  const state = {
    dag: ctx.query.get('dag') || '~',
    limit: Number(ctx.query.get('limit')) || 50,
    offset: Number(ctx.query.get('offset')) || 0,
    map_index: ctx.query.get('map_index') || '',
    ...Object.fromEntries(FILTERS.map((f) => [f.key, ctx.query.get(f.key) || ''])),
  };
  let rows = [];
  let total = 0;

  const tableEl = h('div', skeleton(8));
  const pagerEl = h('div');
  const countEl = h('span.text-muted.small');
  const dagSelect = h('select.form-select.form-select-sm', {
    'aria-label': 'DAG',
    onchange: () => { state.dag = dagSelect.value; state.offset = 0; sync(); load(); },
  }, h('option', { value: '~' }, 'All DAGs'));
  const refresh = refreshControl(() => load(), { key: 'xcoms', initial: 0 });
  ctx.cleanup(refresh.stop);

  const onFilter = debounce(() => { state.offset = 0; sync(); load(); }, 300);
  const filterInputs = FILTERS.map((f) => h('div.af-xcom-filter',
    h('label.form-label.small.fw-semibold.mb-1', f.label),
    h('input.form-control.form-control-sm', {
      type: 'search', placeholder: f.placeholder, value: state[f.key],
      oninput: (e) => { state[f.key] = e.target.value.trim(); onFilter(); },
    })));

  mount(el,
    pageHeader({
      title: 'XComs', icon: 'arrow-left-right',
      subtitle: 'Values tasks passed to each other (cross-communication).',
      actions: [refresh.el],
    }),
    card({
      body: h('div',
        h('div.af-xcom-filters',
          h('div.af-xcom-filter', h('label.form-label.small.fw-semibold.mb-1', 'DAG'), dagSelect),
          filterInputs,
          h('div.af-xcom-filter.af-xcom-filter-sm',
            h('label.form-label.small.fw-semibold.mb-1', 'Map index'),
            h('input.form-control.form-control-sm', {
              type: 'number', min: -1, placeholder: 'any', value: state.map_index,
              oninput: (e) => { state.map_index = e.target.value; onFilter(); },
            }))),
        h('div.af-toolbar.af-card-toolbar', h('div.af-grow'), countEl),
        tableEl, pagerEl),
      flush: true,
    }));

  api.all('/dags', 'dags', { order_by: 'dag_id' }, { signal }).then((dags) => {
    if (!ctx.active) return;
    dags.forEach((d) => dagSelect.appendChild(h('option', { value: d.dag_id, selected: d.dag_id === state.dag }, d.dag_display_name || d.dag_id)));
  }).catch(() => {});

  function sync() {
    ctx.setQuery({
      dag: state.dag === '~' ? null : state.dag,
      limit: state.limit === 50 ? null : state.limit,
      offset: state.offset || null,
      map_index: state.map_index || null,
      ...Object.fromEntries(FILTERS.map((f) => [f.key, state[f.key] || null])),
    });
  }

  async function load() {
    const params = { limit: state.limit, offset: state.offset };
    for (const f of FILTERS) if (state[f.key]) params[f.key] = `%${state[f.key]}%`;
    if (state.map_index !== '') params.map_index_filter = state.map_index;
    try {
      const res = await api.get(`/dags/${enc(state.dag)}/dagRuns/~/taskInstances/~/xcomEntries`, params, { signal });
      if (!ctx.active) return;
      rows = res.xcom_entries || [];
      total = res.total_entries || 0;
      draw();
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(tableEl, h('div.p-3', errorState(err, load)));
    }
  }

  function draw() {
    mount(countEl, `${fmtNumber(total)} entr${total === 1 ? 'y' : 'ies'}`);
    mount(tableEl, dataTable({
      rows,
      rowKey: (x) => `${x.dag_id}|${x.run_id}|${x.task_id}|${x.map_index}|${x.key}`,
      onRowClick: (x) => showEntry(x),
      empty: emptyState({ icon: 'arrow-left-right', title: 'No XComs found', message: 'Try clearing the filters.' }),
      columns: [
        { key: 'key', label: 'Key', render: (x) => h('span.af-mono.fw-semibold', x.key) },
        { key: 'dag_id', label: 'DAG', render: (x) => h('a', { href: urls.dag(x.dag_id) }, x.dag_display_name || x.dag_id) },
        { key: 'run_id', label: 'Run', render: (x) => h('a.af-mono.small.af-ellipsis', { href: urls.run(x.dag_id, x.run_id), style: { maxWidth: '280px' }, title: x.run_id }, x.run_id) },
        {
          key: 'task_id', label: 'Task',
          render: (x) => h('a.af-mono.small', { href: urls.task(x.dag_id, x.run_id, x.task_id, x.map_index) },
            x.task_display_name || x.task_id, x.map_index >= 0 ? h('span.af-map-idx', `[${x.map_index}]`) : null),
        },
        { key: 'timestamp', label: 'Pushed', render: (x) => timeAgo(x.timestamp) },
        { key: 'open', label: '', width: '36px', render: () => icon('chevron-right', 'text-muted') },
      ],
    }));
    mount(pagerEl, total > state.limit || state.offset ? pager({
      total, limit: state.limit, offset: state.offset,
      onChange: (offset, limit) => { state.offset = offset; state.limit = limit; sync(); load(); },
    }) : null);
  }

  function showEntry(x) {
    const path = `/dags/${enc(x.dag_id)}/dagRuns/${enc(x.run_id)}/taskInstances/${enc(x.task_id)}/xcomEntries/${enc(x.key)}`;
    const valueEl = h('div', skeleton(6));
    const metaEl = h('div.d-flex.gap-2.flex-wrap.align-items-center.small');
    let mode = 'native';
    const d = drawer({
      title: h('span.af-mono', x.key),
      subtitle: `${x.dag_id} · ${x.task_id}${x.map_index >= 0 ? ` [${x.map_index}]` : ''}`,
      width: 760,
      body: h('div.d-flex.flex-column.gap-3',
        kv([
          ['DAG', h('a', { href: urls.dag(x.dag_id) }, x.dag_id)],
          ['Task', h('a', { href: urls.task(x.dag_id, x.run_id, x.task_id, x.map_index) }, x.task_id, x.map_index >= 0 ? ` [${x.map_index}]` : '')],
          ['Run', h('a.af-mono.small', { href: urls.run(x.dag_id, x.run_id) }, x.run_id), { wide: true }],
          ['Logical date', fmtDate(x.logical_date)],
          ['Pushed', fmtDate(x.timestamp)],
        ]),
        h('div.d-flex.justify-content-between.align-items-center.flex-wrap.gap-2',
          h('div.af-section-label.mb-0', 'Value'),
          segmented([
            { value: 'native', label: 'Native JSON' },
            { value: 'deserialize', label: 'Deserialized' },
            { value: 'stringify', label: 'Python repr' },
          ], mode, (m) => { mode = m; loadValue(); })),
        metaEl,
        valueEl),
    });
    // Links inside the drawer navigate the app; close the drawer when they do.
    d.el.addEventListener('click', (e) => { if (e.target.closest('a[href]')) d.close(); });

    async function loadValue() {
      mount(valueEl, skeleton(6));
      const params = { map_index: x.map_index };
      if (mode === 'deserialize') params.deserialize = true;
      if (mode === 'stringify') params.stringify = true;
      try {
        const entry = await api.get(path, params);
        if (!d.isOpen()) return;
        const v = entry.value;
        const text = typeof v === 'string' ? v : JSON.stringify(v, null, 2);
        const parsed = typeof v === 'string'
          ? (mode === 'stringify' ? { ok: false } : parseJsonish(v))
          : { ok: v !== null && typeof v === 'object', value: v };
        mount(metaEl,
          h('span.af-tag', v === null ? 'null' : Array.isArray(v) ? `array · ${v.length} items` : typeof v === 'object' ? `object · ${Object.keys(v).length} keys` : typeof v),
          h('span.af-tag', fmtBytes(new Blob([text ?? '']).size)),
          copyButton(text ?? '', 'Copy value'));
        mount(valueEl, parsed.ok ? jsonView(parsed.value, { open: 2 }) : h('pre.af-pre', text ?? 'null'));
      } catch (err) {
        if (d.isOpen()) mount(valueEl, errorState(err, loadValue));
      }
    }
    loadValue();
  }

  await load();
}
