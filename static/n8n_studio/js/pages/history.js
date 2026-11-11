// Version history across all workflows (Studio snapshots), filterable by source.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, chips, spinner, timeAgo } from '../../../ui/js/ui.js';
import { studio } from '../core/api.js';
import { urls } from '../core/urls.js';
import { emptyHint } from '../core/widgets.js';

export default async function render(ctx) {
  let source = ctx.query.get('source') || '';
  const body = h('div');
  const head = h('div');
  mount(ctx.el, pageHeader({ title: 'Version history', icon: 'clock-history', subtitle: 'Every snapshot Studio has taken — open a workflow\'s Versions tab to diff and restore.', actions: head }), body);

  async function go() {
    mount(head, chips([{ value: '', label: 'All' }, { value: 'save', label: 'Saves' }, { value: 'create', label: 'Creates' }, { value: 'restore', label: 'Restores' }, { value: 'deleted', label: 'Deletes' }, { value: 'backup', label: 'Backups' }, { value: 'manual', label: 'Manual' }], source, (v) => { source = v; ctx.setQuery({ source: v || null }); go(); }));
    mount(body, spinner());
    const res = await studio.get('/snapshots/', { source: source || undefined, limit: 1000 }, { signal: ctx.signal });
    mount(body, h('div.small.text-muted.mb-2', `${res.total.toLocaleString()} snapshot(s)${res.total > res.data.length ? ` · showing latest ${res.data.length}` : ''}`), dataTable({
      rows: res.data, dense: true,
      empty: emptyHint('clock-history', 'No snapshots'),
      columns: [
        { key: 'name', label: 'Workflow', render: (s) => h('a', { href: urls.workflow(s.workflowId, { tab: 'versions' }) }, s.name || s.workflowId) },
        { key: 'source', label: 'Event', render: (s) => h('span.n8s-chip-sm', s.source) },
        { key: 'nodes', label: 'Nodes', render: (s) => s.nodeCount },
        { key: 'active', label: 'Active', render: (s) => (s.active ? icon('check-lg', 'text-success') : '') },
        { key: 'user', label: 'By', render: (s) => s.user || h('span.text-muted', 'system') },
        { key: 'hash', label: 'Content', render: (s) => h('code.small', s.hash) },
        { key: 'when', label: 'When', render: (s) => timeAgo(s.createdAt) },
      ],
      onRowClick: (s) => ctx.navigate(urls.workflow(s.workflowId, { tab: 'versions' })),
    }));
  }
  go();
}
