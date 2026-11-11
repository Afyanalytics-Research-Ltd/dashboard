// "Add to dashboard": pick (or create) a dashboard and append a widget for a visualization.
//
//   const widget = await addToDashboard({ viz, query })

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { modal, toast, toastError, spinner, emptyState } from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc } from './api.js';
import { urls } from './urls.js';
import { invalidate } from './common.js';

/** First free row below every widget of a dashboard (6-column grid). */
export function bottomRow(widgets = []) {
  return widgets.reduce((max, w) => {
    const p = w.options?.position;
    return p ? Math.max(max, (p.row || 0) + (p.sizeY || 0)) : max;
  }, 0);
}

/** Parameter mappings for a new widget: reuse dashboard-level names that already exist. */
export function defaultMappings(query, dashboardWidgets = []) {
  const existing = new Set();
  for (const w of dashboardWidgets) {
    for (const m of Object.values(w.options?.parameterMappings || {})) if (m.type === 'dashboard-level') existing.add(m.mapTo || m.name);
  }
  const out = {};
  for (const p of query?.options?.parameters || []) {
    const shared = existing.has(p.name);
    out[p.name] = { name: p.name, type: shared ? 'dashboard-level' : 'widget-level', mapTo: p.name, value: null, title: '' };
  }
  return out;
}

/** POST a visualization widget at the bottom of a dashboard. */
export async function createVizWidget(dashboard, viz, query, { sizeX = 3, sizeY = 8, col = 0 } = {}) {
  const full = dashboard.widgets ? dashboard : await rd.get(`/api/dashboards/${enc(dashboard.id)}`);
  const isTable = viz.type === 'TABLE';
  return rd.post('/api/widgets', {
    dashboard_id: full.id,
    visualization_id: viz.id,
    text: '',
    width: 1,
    options: {
      isHidden: false,
      position: { col, row: bottomRow(full.widgets || []), sizeX: isTable ? 6 : sizeX, sizeY: isTable ? 12 : sizeY, autoHeight: false, minSizeX: 1, maxSizeX: 6, minSizeY: 2, maxSizeY: 1000 },
      parameterMappings: defaultMappings(query, full.widgets || []),
    },
  });
}

export function addToDashboard({ viz, query } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const list = h('div.rdd-pick-list', h('div.text-center.p-3', spinner()));
    const search = h('input.form-control', { type: 'search', placeholder: 'Search dashboards…', autofocus: true, 'aria-label': 'Search dashboards' });
    const newName = h('input.form-control.form-control-sm', { placeholder: 'New dashboard name', value: query?.name ? `${query.name}` : '' });
    let selected = null;
    let seq = 0;

    const load = async () => {
      const mine = ++seq;
      try {
        // Note: Redash 10 returns 500 for order=name on dashboards, so sort client-side.
        const res = await rd.get('/api/dashboards', { q: search.value.trim() || undefined, page_size: 100 });
        if (mine !== seq) return;
        const items = (res.results || []).sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
        if (!items.length) { mount(list, emptyState({ icon: 'grid-1x2', title: 'No dashboards found', message: 'Create a new one below.' })); return; }
        mount(list, items.map((d) => h('button.rdd-pick', {
          type: 'button', class: selected?.id === d.id && 'active',
          onclick: () => { selected = d; list.querySelectorAll('.rdd-pick').forEach((b) => b.classList.toggle('active', b.dataset.id === String(d.id))); },
          ondblclick: () => { selected = d; submitBtn()?.click(); },
          dataset: { id: d.id },
        }, icon('grid-1x2'), h('span.rdd-pick-name', d.name),
        d.is_draft && h('span.rd-draft-pill', 'draft'),
        (d.tags || []).slice(0, 3).map((t) => h('span.af-tag', t)))));
      } catch (err) {
        mount(list, h('div.text-danger.small.p-2', err.message));
      }
    };
    search.addEventListener('input', debounce(load, 200));

    const m = modal({
      title: 'Add to dashboard', icon: 'grid-1x2',
      subtitle: viz?.name ? `${viz.name}${query?.name ? ` · ${query.name}` : ''}` : null,
      body: h('div.rdd-pick-wrap',
        search, list,
        h('div.rdd-pick-new', h('span.small.fw-semibold', icon('plus-circle', 'me-1'), 'Or create a new dashboard'), newName)),
      actions: [
        { label: 'Cancel' },
        {
          label: 'Add widget', variant: 'primary', icon: 'plus-lg',
          onClick: async (close) => {
            let dashboard = selected;
            if (!dashboard) {
              const name = newName.value.trim();
              if (!name) throw new Error('Pick a dashboard or type a name for a new one.');
              dashboard = await rd.post('/api/dashboards', { name });
              invalidate('dash');
            }
            const widget = await createVizWidget(dashboard, viz, query);
            done = true;
            invalidate('dashfull');
            toast(`Added to “${dashboard.name}”`, 'success', {
              action: { label: 'Open dashboard →', onClick: () => navigate(urls.dashboard(dashboard.id)) },
            });
            close();
            resolve(widget);
          },
        },
      ],
      onClose: () => { if (!done) resolve(null); },
    });
    const submitBtn = () => m.el.querySelector('.af-modal-footer .btn-primary');
    newName.addEventListener('focus', () => { selected = null; list.querySelectorAll('.rdd-pick').forEach((b) => b.classList.remove('active')); });
    load().catch((err) => toastError(err));
  });
}
