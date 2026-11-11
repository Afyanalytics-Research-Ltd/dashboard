// Alerts list: instant search, state chips with counts, inline mute/unmute,
// human-readable conditions and notification behaviour.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, dataTable, searchInput, chips, emptyState, errorState, skeleton, timeAgo,
  refreshControl, confirmDialog, toast, toastError, openMenu,
} from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { userChip } from '../core/common.js';
import { alertStateBadge, opInfo, rearmText, matches, hl } from './admin_shared.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  let q = ctx.query.get('q') || '';
  let state = ctx.query.get('state') || '';
  let sort = { key: ctx.query.get('sort') || 'name', dir: ctx.query.get('dir') || 'asc' };
  let alerts = [];

  const chipsEl = h('div');
  const bodyEl = h('div', skeleton(6));
  const refresh = refreshControl(() => load(), { key: 'rd-alerts', initial: 60 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Alerts', icon: 'bell',
      subtitle: 'Get notified when a query result crosses a threshold. Alerts are checked every time their query runs.',
      actions: [refresh.el, h('a.btn.btn-sm.btn-primary', { href: urls.newAlert() }, icon('plus-lg', 'me-1'), 'New alert')],
    }),
    card({
      body: h('div.d-flex.flex-column.gap-3',
        h('div.af-toolbar',
          searchInput({ placeholder: 'Search alerts by name, query, column or owner…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } }),
          chipsEl),
        bodyEl),
    }));

  async function load() {
    try {
      alerts = await rd.get('/api/alerts', null, { signal });
      if (!ctx.active) return;
      draw();
    } catch (err) {
      if (err.name !== 'AbortError') mount(bodyEl, errorState(err, load));
    }
  }

  function stateOf(a) { return a.options?.muted ? 'muted' : (a.state || 'unknown'); }

  function draw() {
    const counts = { '': alerts.length, triggered: 0, ok: 0, unknown: 0, muted: 0 };
    alerts.forEach((a) => { counts[stateOf(a)] = (counts[stateOf(a)] || 0) + 1; });
    mount(chipsEl, chips([
      { value: '', label: 'All', count: counts[''] },
      { value: 'triggered', label: 'Triggered', count: counts.triggered, color: '#dc2626' },
      { value: 'ok', label: 'OK', count: counts.ok, color: '#16a34a' },
      { value: 'unknown', label: 'Unknown', count: counts.unknown, color: '#94a3b8' },
      { value: 'muted', label: 'Muted', count: counts.muted, color: '#d97706' },
    ], state, (v) => { state = v; ctx.setQuery({ state: v }); draw(); }));

    if (!alerts.length) {
      mount(bodyEl, emptyState({
        icon: 'bell', title: 'No alerts yet',
        message: 'An alert watches one value of a query (for example "failed payments > 10") and notifies you or a channel when it changes state.',
        action: h('a.btn.btn-sm.btn-primary', { href: urls.newAlert() }, icon('plus-lg', 'me-1'), 'Create your first alert'),
      }));
      return;
    }

    const rows = alerts.filter((a) => (!state || stateOf(a) === state)
      && matches(a, q, ['name', (x) => x.query?.name, (x) => x.options?.column, (x) => x.user?.name, (x) => x.user?.email]));
    const dir = sort.dir === 'asc' ? 1 : -1;
    const keyFn = {
      name: (a) => a.name?.toLowerCase(),
      state: (a) => stateOf(a),
      query: (a) => a.query?.name?.toLowerCase(),
      last_triggered_at: (a) => a.last_triggered_at || '',
      updated_at: (a) => a.updated_at || '',
    }[sort.key] || ((a) => a.name);
    rows.sort((a, b) => (keyFn(a) > keyFn(b) ? 1 : keyFn(a) < keyFn(b) ? -1 : 0) * dir);

    mount(bodyEl, dataTable({
      rows,
      rowKey: (a) => a.id,
      sort,
      onSort: (key, d) => { sort = { key, dir: d }; ctx.setQuery({ sort: key, dir: d }); draw(); },
      onRowClick: (a) => navigate(urls.alert(a.id)),
      rowClass: (a) => (a.state === 'triggered' && !a.options?.muted ? 'rd-row-triggered' : null),
      empty: emptyState({ icon: 'search', title: 'No alerts match', message: 'Try a different search or state filter.' }),
      columns: [
        { key: 'state', label: 'State', sortable: true, width: '150px', render: (a) => alertStateBadge(a.state, { muted: a.options?.muted }) },
        {
          key: 'name', label: 'Alert', sortable: true,
          render: (a) => h('div',
            h('a.fw-semibold', { href: urls.alert(a.id) }, hl(a.name, q)),
            h('div.small.text-muted', condition(a))),
        },
        { key: 'query', label: 'Query', sortable: true, render: (a) => (a.query ? h('a.small', { href: urls.query(a.query.id) }, icon('code-square', 'me-1'), hl(a.query.name, q)) : '—') },
        { key: 'rearm', label: 'Notifies', render: (a) => h('span.small', rearmText(a.rearm)) },
        { key: 'last_triggered_at', label: 'Last triggered', sortable: true, render: (a) => (a.last_triggered_at ? timeAgo(a.last_triggered_at) : h('span.text-muted.small', 'never')) },
        { key: 'user', label: 'Owner', render: (a) => userChip(a.user) },
        { key: 'updated_at', label: 'Updated', sortable: true, render: (a) => timeAgo(a.updated_at) },
        {
          key: 'actions', label: '', width: '90px', align: 'right',
          render: (a) => h('div.d-flex.gap-1.justify-content-end',
            h('button.btn.btn-sm.btn-outline-secondary', {
              title: a.options?.muted ? 'Unmute notifications' : 'Mute notifications',
              onclick: () => toggleMute(a),
            }, icon(a.options?.muted ? 'bell' : 'bell-slash')),
            h('button.btn.btn-sm.btn-outline-secondary', {
              title: 'More actions',
              onclick: (e) => openMenu(e.currentTarget, [
                { label: 'Open', icon: 'box-arrow-up-right', href: urls.alert(a.id) },
                a.query && { label: 'Open query', icon: 'code-square', href: urls.query(a.query.id) },
                { divider: true },
                { label: 'Delete alert', icon: 'trash', danger: true, onClick: () => remove(a) },
              ]),
            }, icon('three-dots'))),
        },
      ],
    }));
  }

  function condition(a) {
    const o = a.options || {};
    return h('span', h('code', o.column || '?'), ` ${opInfo(o.op).symbol} `, h('code', String(o.value ?? '?')));
  }

  async function toggleMute(a) {
    const muted = !a.options?.muted;
    try {
      if (muted) await rd.post(`/api/alerts/${enc(a.id)}/mute`, {});
      else await rd.del(`/api/alerts/${enc(a.id)}/mute`);
      a.options = { ...a.options, muted };
      toast(muted ? `"${a.name}" muted — it will still be evaluated, just not sent.` : `"${a.name}" unmuted`, 'success');
      draw();
    } catch (err) { toastError(err, 'Could not change mute state'); }
  }

  async function remove(a) {
    const ok = await confirmDialog({
      title: `Delete alert "${a.name}"?`, danger: true, confirmLabel: 'Delete alert',
      message: 'Its subscriptions are removed too. The query itself is not affected.',
    });
    if (!ok) return;
    try {
      await rd.del(`/api/alerts/${enc(a.id)}`);
      alerts = alerts.filter((x) => x.id !== a.id);
      toast('Alert deleted', 'success');
      draw();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  await load();
}
