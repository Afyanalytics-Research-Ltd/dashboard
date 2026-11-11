// Alert destinations (Slack, webhook, email, PagerDuty, …): logos, instant search,
// schema-driven create/edit forms with secret handling, delete.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, emptyState, errorState, skeleton, searchInput, modal, toast, toastError, confirmDialog } from '../../../ui/js/ui.js';
import { rd, enc } from '../core/api.js';
import { configForm, typeLogo, pickType, matches, hl } from './admin_shared.js';

const POPULAR = ['slack', 'email', 'webhook', 'mattermost', 'pagerduty', 'hangouts_chat'];

export default async function render(ctx) {
  const { el, signal } = ctx;
  let q = ctx.query.get('q') || '';
  let dests = [];
  let types = [];
  const grid = h('div.rd-dest-grid', skeleton(3, { height: 56 }));

  mount(el,
    pageHeader({
      title: 'Alert destinations', icon: 'send',
      subtitle: 'Where alerts can send notifications. Subscribe an alert to one of these from the alert page.',
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: create }, icon('plus-lg', 'me-1'), 'New destination')],
    }),
    h('div.af-toolbar', searchInput({ placeholder: 'Search destinations…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } })),
    grid);

  async function load() {
    try {
      [dests, types] = await Promise.all([rd.get('/api/destinations', null, { signal }), rd.get('/api/destinations/types', null, { signal })]);
      if (ctx.active) draw();
    } catch (err) {
      if (err.name !== 'AbortError') mount(grid, errorState(err, load));
    }
  }

  const typeName = (t) => types.find((x) => x.type === t)?.name || t;

  function draw() {
    if (!dests.length) {
      mount(grid, emptyState({
        icon: 'send', title: 'No destinations yet',
        message: 'Add Slack, a webhook, PagerDuty or email so alerts can notify a team channel instead of just one person.',
        action: h('button.btn.btn-sm.btn-primary', { onclick: create }, icon('plus-lg', 'me-1'), 'Add a destination'),
      }));
      return;
    }
    const rows = dests.filter((d) => matches(d, q, ['name', 'type', (x) => typeName(x.type)]));
    mount(grid, rows.length ? rows.map((d) => h('div.rd-dest-card',
      typeLogo(d.type, 'dest', 34),
      h('div.rd-dest-main', h('div.fw-semibold', hl(d.name, q)), h('div.small.text-muted', hl(typeName(d.type), q))),
      h('div.d-flex.gap-1',
        h('button.btn.btn-sm.btn-outline-secondary', { title: 'Edit', onclick: () => edit(d) }, icon('pencil')),
        h('button.btn.btn-sm.btn-outline-danger', { title: 'Delete', onclick: () => remove(d) }, icon('trash')))))
      : emptyState({ icon: 'search', title: 'No destinations match' }));
  }

  async function create() {
    if (!types.length) types = await rd.get('/api/destinations/types');
    const t = await pickType(types, { title: 'New alert destination', kind: 'dest', popular: POPULAR });
    if (t) openForm(t, null);
  }

  async function edit(d) {
    try {
      const full = await rd.get(`/api/destinations/${enc(d.id)}`);
      const t = types.find((x) => x.type === full.type);
      if (!t) throw new Error(`Unknown destination type ${full.type}`);
      openForm(t, full);
    } catch (err) { toastError(err, 'Could not open destination'); }
  }

  function openForm(t, existing) {
    const name = h('input.form-control.form-control-sm', { value: existing?.name || '', placeholder: `e.g. #data-alerts ${t.name}`, autofocus: true });
    const form = configForm(t.configuration_schema, existing?.options || {}, { isNew: !existing });
    const errorBox = h('div.alert.alert-danger.small.py-2.d-none');
    modal({
      title: existing ? `Edit ${existing.name}` : `New ${t.name} destination`, icon: null, size: 'lg',
      subtitle: h('span.d-inline-flex.align-items-center.gap-2', typeLogo(t.type, 'dest', 20), t.name),
      body: h('div.af-form', errorBox, h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Name', h('span.text-danger', ' *')), name), form.el),
      actions: [
        { label: 'Cancel' },
        {
          label: existing ? 'Save' : 'Create destination', variant: 'primary', icon: 'check2',
          onClick: async (close) => {
            errorBox.classList.add('d-none');
            try {
              if (!name.value.trim()) throw new Error('Give the destination a name.');
              const body = { name: name.value.trim(), type: t.type, options: form.values() };
              const saved = existing ? await rd.post(`/api/destinations/${enc(existing.id)}`, body) : await rd.post('/api/destinations', body);
              dests = existing ? dests.map((x) => (x.id === saved.id ? saved : x)) : [...dests, saved];
              toast(existing ? 'Destination saved' : `${saved.name} added`, 'success');
              close();
              draw();
            } catch (err) {
              mount(errorBox, icon('exclamation-octagon', 'me-1'), err.message);
              errorBox.classList.remove('d-none');
            }
          },
        },
      ],
    });
  }

  async function remove(d) {
    const ok = await confirmDialog({ title: `Delete ${d.name}?`, danger: true, confirmLabel: 'Delete destination', message: 'Alerts subscribed to it will stop notifying this destination.' });
    if (!ok) return;
    try {
      await rd.del(`/api/destinations/${enc(d.id)}`);
      dests = dests.filter((x) => x.id !== d.id);
      toast('Destination deleted', 'success');
      draw();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  await load();
}
