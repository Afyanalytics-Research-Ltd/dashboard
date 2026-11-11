// DAG warnings: non-fatal problems (missing pools, asset conflicts) reported by the DAG processor.

import { h, icon, mount } from '../core/dom.js';
import { api } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  pageHeader, card, dataTable, pager, refreshControl, chips, searchInput, timeAgo, errorState, skeleton,
} from '../core/ui.js';
import { fmtDate } from '../core/format.js';

const TYPES = [
  { value: '', label: 'All types' },
  { value: 'non-existent pool', label: 'Non-existent pool', color: '#d97706' },
  { value: 'asset conflict', label: 'Asset conflict', color: '#7c3aed' },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  const state = {
    dag_id: ctx.query.get('dag_id') || '',
    warning_type: ctx.query.get('warning_type') || '',
    offset: Number(ctx.query.get('offset') || 0),
    limit: 50,
    order_by: ctx.query.get('order_by') || '-timestamp',
  };
  const tableEl = h('div', skeleton(6));
  const countEl = h('span.af-muted.small');
  const refresh = refreshControl(() => load(), { key: 'warnings', initial: 30 });
  ctx.cleanup(refresh.stop);
  const sync = () => ctx.setQuery({ dag_id: state.dag_id, warning_type: state.warning_type, offset: state.offset || null, order_by: state.order_by === '-timestamp' ? null : state.order_by });

  mount(el,
    pageHeader({
      title: 'DAG warnings', icon: 'exclamation-diamond',
      subtitle: 'Non-fatal issues found while parsing DAGs. The DAG still runs, but may not behave as intended.',
      actions: [refresh.el],
    }),
    card({
      title: 'Warnings', icon: 'list-ul', actions: countEl, flush: true,
      body: h('div',
        h('div.af-toolbar.p-3.pb-2',
          searchInput({ placeholder: 'Exact DAG id…', value: state.dag_id, onInput: (v) => { state.dag_id = v; state.offset = 0; sync(); load(); } }),
          chips(TYPES, state.warning_type, (v) => { state.warning_type = v; state.offset = 0; sync(); load(); })),
        tableEl),
    }));

  async function load() {
    try {
      const params = { limit: state.limit, offset: state.offset, order_by: state.order_by };
      if (state.dag_id) params.dag_id = state.dag_id;
      if (state.warning_type) params.warning_type = state.warning_type;
      const data = await api.get('/dagWarnings', params, { signal });
      if (!ctx.active) return;
      mount(countEl, `${data.total_entries} warning${data.total_entries === 1 ? '' : 's'}`);
      const sortKey = state.order_by.replace(/^-/, '');
      mount(tableEl,
        dataTable({
          rows: data.dag_warnings,
          rowKey: (w) => `${w.dag_id}|${w.warning_type}`,
          sort: { key: sortKey, dir: state.order_by.startsWith('-') ? 'desc' : 'asc' },
          onSort: (key, dir) => { state.order_by = `${dir === 'desc' ? '-' : ''}${key}`; sync(); load(); },
          empty: h('div.sys-celebrate', h('div.sys-celebrate-icon', icon('shield-check')),
            h('div.af-empty-title', state.dag_id || state.warning_type ? 'No warnings match' : 'No DAG warnings'),
            h('div.af-empty-msg', 'Pools and assets referenced by your DAGs all resolve.')),
          columns: [
            { key: 'dag_id', label: 'DAG', sortable: true, render: (w) => h('a.fw-semibold', { href: urls.dag(w.dag_id) }, w.dag_display_name || w.dag_id) },
            { key: 'warning_type', label: 'Type', sortable: true, render: (w) => typeBadge(w.warning_type) },
            { key: 'message', label: 'Message', render: (w) => h('div.sys-warn-msg', w.message) },
            { key: 'timestamp', label: 'Seen', sortable: true, width: '140px', render: (w) => h('span', { title: fmtDate(w.timestamp) }, timeAgo(w.timestamp)) },
          ],
        }),
        data.total_entries > state.limit ? pager({
          total: data.total_entries, limit: state.limit, offset: state.offset,
          onChange: (offset, limit) => { state.offset = offset; state.limit = limit; sync(); load(); },
        }) : null);
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(tableEl, h('div.p-3', errorState(err, load)));
    }
  }

  await load();
}

function typeBadge(type) {
  const t = TYPES.find((x) => x.value === type);
  const color = t?.color || '#64748b';
  return h('span.af-badge', { style: { color, background: `${color}1a` } }, h('span.af-dot', { style: { background: color } }), type);
}
