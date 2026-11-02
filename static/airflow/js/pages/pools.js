// Pools: live slot usage cards, create/edit/delete.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  pageHeader, card, statCard, searchInput, refreshControl, formModal, confirmDialog, menuButton, toast, toastError,
  emptyState, errorState, skeleton, stateInfo,
} from '../core/ui.js';
import { tip, tipRows } from '../core/charts.js';
import { fmtNumber } from '../core/format.js';

const SEGMENTS = [
  { key: 'running_slots', label: 'Running', state: 'running' },
  { key: 'queued_slots', label: 'Queued', state: 'queued' },
  { key: 'scheduled_slots', label: 'Scheduled', state: 'scheduled' },
  { key: 'deferred_slots', label: 'Deferred', state: 'deferred' },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  let q = ctx.query.get('q') || '';
  let pools = [];

  const statsEl = h('div.af-stats', skeleton(1, { height: 60 }));
  const gridEl = h('div.af-pool-grid', skeleton(4, { height: 120 }));
  const refresh = refreshControl(() => load(), { key: 'pools', initial: 10 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Pools', icon: 'water',
      subtitle: 'Limit how many task instances run at once against shared resources.',
      actions: [
        searchInput({ placeholder: 'Search pools…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v || null }); load(); } }),
        h('button.btn.btn-sm.btn-primary', { onclick: () => openForm() }, icon('plus-lg', 'me-1'), 'New pool'),
        refresh.el,
      ],
    }),
    statsEl,
    gridEl);

  async function load() {
    try {
      const all = await api.all('/pools', 'pools', { order_by: 'name', pool_name_pattern: q ? `%${q}%` : null }, { signal });
      if (!ctx.active) return;
      pools = all;
      draw();
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(gridEl, errorState(err, load));
    }
  }

  function draw() {
    const finite = pools.filter((p) => p.slots >= 0);
    const sum = (k) => pools.reduce((n, p) => n + (p[k] || 0), 0);
    mount(statsEl,
      statCard({ label: 'Pools', value: fmtNumber(pools.length), icon: 'water', tone: 'blue' }),
      statCard({ label: 'Total slots', value: finite.length === pools.length ? fmtNumber(sum('slots')) : '∞', icon: 'grid-3x3-gap', tone: 'slate' }),
      statCard({ label: 'Running', value: fmtNumber(sum('running_slots')), icon: 'arrow-repeat', tone: 'cyan' }),
      statCard({ label: 'Queued', value: fmtNumber(sum('queued_slots') + sum('scheduled_slots')), icon: 'hourglass-split', tone: 'amber' }),
      statCard({ label: 'Open', value: finite.length === pools.length ? fmtNumber(sum('open_slots')) : '∞', icon: 'check2-circle', tone: 'green' }));

    if (!pools.length) {
      mount(gridEl, card({ body: emptyState({ icon: 'water', title: q ? 'No matching pools' : 'No pools', message: q ? `Nothing matches “${q}”.` : null }) }));
      return;
    }
    mount(gridEl, pools.map(poolCard));
  }

  function poolCard(p) {
    const unlimited = p.slots < 0;
    const occupied = p.occupied_slots ?? SEGMENTS.reduce((n, s) => n + (p[s.key] || 0), 0);
    const pct = unlimited || !p.slots ? 0 : Math.min(100, (occupied / p.slots) * 100);
    const meter = h('div.af-meter.af-pool-meter');
    if (!unlimited && p.slots) {
      for (const seg of SEGMENTS) {
        const n = p[seg.key] || 0;
        if (!n) continue;
        const span = h('span', { style: { width: `${(n / p.slots) * 100}%`, background: stateInfo(seg.state).color } });
        tip(span, () => tipRows(seg.label, [['Slots', n], ['Share', `${((n / p.slots) * 100).toFixed(1)}%`]]));
        meter.appendChild(span);
      }
    }
    const tone = pct >= 90 ? 'hot' : pct >= 60 ? 'warm' : 'cool';
    return h('section.af-card.af-pool-card',
      h('div.af-pool-head',
        h('div.af-pool-title-wrap',
          h('h3.af-pool-name', icon('water', 'me-2'), p.name),
          p.description && h('div.af-pool-desc', p.description)),
        menuButton(null, [
          { label: 'Edit', icon: 'pencil', onClick: () => openForm(p) },
          { label: 'Task instances in this pool', icon: 'list-task', href: urls.taskInstances({ pool: p.name }) },
          { divider: true },
          { label: p.name === 'default_pool' ? 'default_pool cannot be deleted' : 'Delete', icon: 'trash', danger: true, disabled: p.name === 'default_pool', onClick: () => remove(p) },
        ])),
      h('div.af-pool-figures',
        h('div.af-pool-big', { class: `af-pool-${tone}` }, fmtNumber(occupied), h('span.af-pool-of', ` / ${unlimited ? '∞' : fmtNumber(p.slots)}`)),
        h('div.af-pool-pct', unlimited ? 'unlimited slots' : `${pct.toFixed(0)}% occupied`)),
      meter,
      h('div.af-pool-legend',
        SEGMENTS.map((seg) => h('span.af-legend-item', h('span.af-dot', { style: { background: stateInfo(seg.state).color } }), seg.label, h('b', fmtNumber(p[seg.key] || 0)))),
        h('span.af-legend-item', h('span.af-dot', { style: { background: '#e2e8f0' } }), 'Open', h('b', unlimited ? '∞' : fmtNumber(p.open_slots)))),
      h('div.af-pool-foot',
        h('span.af-tag', { title: 'Whether deferred tasks count against the pool' }, icon(p.include_deferred ? 'check2' : 'dash'), ' deferred tasks ', p.include_deferred ? 'counted' : 'not counted'),
        h('a.af-link-btn', { href: urls.taskInstances({ pool: p.name }) }, 'View tasks →')));
  }

  function openForm(p) {
    const editing = !!p;
    formModal({
      title: editing ? `Edit ${p.name}` : 'New pool', icon: 'water',
      submitLabel: editing ? 'Save changes' : 'Create pool',
      fields: [
        editing
          ? { name: 'name', label: 'Name', type: 'static', value: p.name }
          : { name: 'name', label: 'Name', required: true, placeholder: 'e.g. snowflake_pool' },
        { name: 'slots', label: 'Slots', type: 'number', required: true, value: p?.slots ?? 16, min: -1, step: 1, help: 'Maximum concurrent task instances. -1 means unlimited.' },
        { name: 'description', label: 'Description', type: 'textarea', rows: 2, value: p?.description || '' },
        { name: 'include_deferred', type: 'checkbox', checkLabel: 'Count deferred tasks against this pool', value: p?.include_deferred ?? false },
      ],
      onSubmit: async (v, close) => {
        if (!Number.isInteger(v.slots) || v.slots < -1) throw new Error('Slots must be an integer ≥ -1.');
        const description = v.description || null;
        if (editing) {
          const mask = [];
          if (v.slots !== p.slots) mask.push('slots');
          if (description !== (p.description ?? null)) mask.push('description');
          if (v.include_deferred !== p.include_deferred) mask.push('include_deferred');
          if (!mask.length) { toast('Nothing changed', 'info'); close(); return; }
          await api.patch(`/pools/${enc(p.name)}`, { pool: p.name, slots: v.slots, description, include_deferred: v.include_deferred }, { params: { update_mask: mask } });
          toast(`${p.name} updated`, 'success');
        } else {
          await api.post('/pools', { name: v.name.trim(), slots: v.slots, description, include_deferred: v.include_deferred });
          toast(`${v.name} created`, 'success');
        }
        close();
        load();
      },
    });
  }

  async function remove(p) {
    const ok = await confirmDialog({
      title: `Delete pool ${p.name}?`, danger: true, confirmLabel: 'Delete pool',
      message: 'Tasks assigned to this pool will fail to schedule until it is recreated.',
    });
    if (!ok) return;
    try {
      await api.del(`/pools/${enc(p.name)}`);
      toast(`${p.name} deleted`, 'success');
      load();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  await load();
}
