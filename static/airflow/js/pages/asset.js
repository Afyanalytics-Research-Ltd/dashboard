// Asset detail: producers/consumers, events, queued events, manual event
// creation and materialization.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../core/router.js';
import {
  pageHeader, card, kv, dataTable, pager, refreshControl, emptyState, errorState, skeleton, jsonView, timeAgo,
  confirmDialog, formModal, toast, toastError, copyButton,
} from '../core/ui.js';
import { fmtDate } from '../core/format.js';
import { eventsTable } from './assets.js';
import { typeBadge, linkList } from './admin_shared.js';

export default async function render(ctx) {
  const { el, signal, params } = ctx;
  const assetId = params.assetId;
  let asset;
  try {
    asset = await api.get(`/assets/${enc(assetId)}`, null, { signal });
  } catch (err) {
    if (err.name === 'AbortError') return;
    mount(el, pageHeader({ title: `Asset ${assetId}`, icon: 'database', crumbs: [{ label: 'Assets', href: urls.assets() }, { label: assetId }] }), errorState(err));
    return;
  }
  if (!ctx.active) return;
  ctx.setTitle(asset.name);

  const evState = { limit: 25, offset: 0, order: '-timestamp' };
  const eventsEl = h('div', skeleton(5));
  const eventsPager = h('div');
  const queuedEl = h('div', skeleton(3));
  const clearAllBtn = h('button.btn.btn-sm.btn-outline-danger', { onclick: () => clearQueued(), disabled: true }, icon('trash', 'me-1'), 'Clear all');
  const refresh = refreshControl(() => Promise.all([loadEvents(), loadQueued()]), { key: 'asset', initial: 30 });
  ctx.cleanup(refresh.stop);

  const producing = asset.producing_tasks || [];
  const consuming = asset.consuming_tasks || [];
  const scheduled = asset.scheduled_dags || [];

  mount(el,
    pageHeader({
      title: asset.name, icon: 'database',
      crumbs: [{ label: 'Assets', href: urls.assets() }, { label: asset.name }],
      subtitle: h('span.af-mono', asset.uri, ' ', copyButton(asset.uri)),
      badges: typeBadge(asset.group),
      actions: [
        h('button.btn.btn-sm.btn-outline-primary', { onclick: createEvent }, icon('lightning-charge', 'me-1'), 'Create event'),
        h('button.btn.btn-sm.btn-primary', { onclick: materialize, disabled: !producing.length, title: producing.length ? 'Trigger the DAG that produces this asset' : 'No producing task' }, icon('play-fill', 'me-1'), 'Materialize'),
        refresh.el,
      ],
    }),
    h('div.af-grid-main-side',
      card({
        title: 'Details', icon: 'info-circle',
        body: h('div.d-flex.flex-column.gap-3',
          kv([
            ['Id', h('span.af-mono', asset.id)],
            ['Group', typeBadge(asset.group)],
            ['Created', fmtDate(asset.created_at)],
            ['Updated', fmtDate(asset.updated_at)],
            ['Last event', asset.last_asset_event?.timestamp ? timeAgo(asset.last_asset_event.timestamp) : 'never'],
            ['Aliases', linkList((asset.aliases || []).map((a) => h('span.af-tag', icon('tag'), a.name)), 'None')],
          ]),
          h('div', h('div.af-section-label', 'Extra'), jsonView(asset.extra || {}))),
      }),
      card({
        title: 'Lineage', icon: 'diagram-2',
        body: h('div.af-lineage',
          lineageBlock('Produced by', 'box-arrow-right', producing.map((t) => h('a.af-tag', { href: urls.dag(t.dag_id) }, `${t.dag_id} · ${t.task_id}`))),
          h('div.af-lineage-arrow', icon('arrow-down')),
          h('div.af-lineage-asset', icon('database'), h('span', asset.name)),
          h('div.af-lineage-arrow', icon('arrow-down')),
          lineageBlock('Triggers DAGs', 'diagram-3', scheduled.map((d) => h('a.af-tag', { href: urls.dag(d.dag_id) }, d.dag_id))),
          lineageBlock('Consumed by tasks (inlets)', 'box-arrow-in-right', consuming.map((t) => h('a.af-tag', { href: urls.dag(t.dag_id) }, `${t.dag_id} · ${t.task_id}`)))),
      })),
    card({
      title: 'Queued events', icon: 'hourglass-split', flush: true,
      actions: clearAllBtn,
      body: queuedEl,
    }),
    card({ title: 'Events', icon: 'lightning-charge', flush: true, body: h('div', eventsEl, eventsPager) }));

  function lineageBlock(label, ic, items) {
    return h('div.af-lineage-block', h('div.af-section-label', icon(ic, 'me-1'), label), linkList(items, 'None'));
  }

  async function loadEvents() {
    try {
      const res = await api.get('/assets/events', { asset_id: asset.id, limit: evState.limit, offset: evState.offset, order_by: evState.order }, { signal });
      if (!ctx.active) return;
      mount(eventsEl, eventsTable(res.asset_events || [], { sort: evState.order, onSort: (o) => { evState.order = o; loadEvents(); }, showAsset: false }));
      mount(eventsPager, res.total_entries > evState.limit || evState.offset ? pager({
        total: res.total_entries, limit: evState.limit, offset: evState.offset,
        onChange: (o, l) => { evState.offset = o; evState.limit = l; loadEvents(); },
      }) : null);
    } catch (err) {
      if (err.name !== 'AbortError') mount(eventsEl, h('div.p-3', errorState(err, loadEvents)));
    }
  }

  async function loadQueued() {
    try {
      const res = await api.get(`/assets/${enc(asset.id)}/queuedEvents`, null, { signal });
      if (!ctx.active) return;
      renderQueued(res.queued_events || []);
    } catch (err) {
      if (err.name === 'AbortError') return;
      // 404 = nothing queued for this asset.
      if (err.status === 404) renderQueued([]);
      else mount(queuedEl, h('div.p-3', errorState(err, loadQueued)));
    }
  }

  function renderQueued(rows) {
    clearAllBtn.disabled = !rows.length;
    mount(queuedEl, dataTable({
      rows,
      rowKey: (q) => q.dag_id,
      empty: emptyState({ icon: 'hourglass', title: 'Nothing queued', message: 'Queued events wait here until every asset a DAG is scheduled on has updated.' }),
      columns: [
        { key: 'dag_id', label: 'Waiting DAG', render: (q) => h('a.fw-semibold', { href: urls.dag(q.dag_id) }, q.dag_display_name || q.dag_id) },
        { key: 'created_at', label: 'Queued', render: (q) => timeAgo(q.created_at) },
        {
          key: 'actions', label: '', align: 'right', width: '120px',
          render: (q) => h('button.btn.btn-sm.btn-outline-danger', { onclick: () => clearQueued(q.dag_id) }, icon('x-lg', 'me-1'), 'Remove'),
        },
      ],
    }));
  }

  async function clearQueued(dagId) {
    const ok = await confirmDialog({
      title: dagId ? `Remove queued event for ${dagId}?` : 'Clear all queued events?',
      message: dagId
        ? 'The DAG will no longer count this asset update toward its next scheduled run.'
        : 'Every DAG waiting on this asset loses the pending update.',
      confirmLabel: dagId ? 'Remove' : 'Clear all', danger: true,
    });
    if (!ok) return;
    try {
      if (dagId) await api.del(`/dags/${enc(dagId)}/assets/${enc(asset.id)}/queuedEvents`);
      else await api.del(`/assets/${enc(asset.id)}/queuedEvents`);
      toast('Queued events removed', 'success');
      loadQueued();
    } catch (err) { toastError(err, 'Could not remove queued events'); }
  }

  function createEvent() {
    formModal({
      title: 'Create asset event', subtitle: asset.uri, icon: 'lightning-charge',
      submitLabel: 'Create event',
      fields: [
        { name: 'extra', label: 'Extra (JSON)', type: 'json', value: {}, rows: 8, help: 'Arbitrary metadata attached to the event. DAGs scheduled on this asset may be triggered.' },
      ],
      onSubmit: async ({ extra }, close) => {
        if (extra != null && (typeof extra !== 'object' || Array.isArray(extra))) throw new Error('Extra must be a JSON object.');
        const ev = await api.post('/assets/events', { asset_id: asset.id, extra: extra || {} });
        toast(`Event #${ev.id} created${ev.created_dagruns?.length ? ` — ${ev.created_dagruns.length} run(s) triggered` : ''}`, 'success');
        close();
        loadEvents();
        loadQueued();
      },
    });
  }

  async function materialize() {
    const ok = await confirmDialog({
      title: `Materialize ${asset.name}?`,
      message: h('span', 'This triggers a run of the DAG that produces the asset: ',
        producing.map((t, i) => [i ? ', ' : '', h('code', t.dag_id)]), '.'),
      confirmLabel: 'Materialize',
    });
    if (!ok) return;
    try {
      const run = await api.post(`/assets/${enc(asset.id)}/materialize`, undefined);
      toast(`Run ${run.dag_run_id} queued`, 'success', { title: `${run.dag_id} triggered` });
      navigate(urls.run(run.dag_id, run.dag_run_id));
    } catch (err) { toastError(err, 'Materialize failed'); }
  }

  await Promise.all([loadEvents(), loadQueued()]);
}
