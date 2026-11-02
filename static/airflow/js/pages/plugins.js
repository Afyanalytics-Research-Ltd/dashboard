// Plugins: every loaded plugin with the components it contributes, plus plugin import errors.

import { h, icon, mount } from '../core/dom.js';
import { api } from '../core/api.js';
import {
  pageHeader, card, refreshControl, searchInput, emptyState, errorState, skeleton, jsonView, codeBlock,
} from '../core/ui.js';

const COMPONENTS = [
  ['macros', 'Macros', 'braces'],
  ['operator_extra_links', 'Operator extra links', 'link-45deg'],
  ['global_operator_extra_links', 'Global extra links', 'globe'],
  ['listeners', 'Listeners', 'broadcast'],
  ['timetables', 'Timetables', 'calendar-range'],
  ['fastapi_apps', 'FastAPI apps', 'lightning'],
  ['fastapi_root_middlewares', 'FastAPI middlewares', 'layers'],
  ['react_apps', 'React apps', 'window'],
  ['external_views', 'External views', 'box-arrow-up-right'],
  ['flask_blueprints', 'Flask blueprints', 'diagram-2'],
  ['appbuilder_views', 'AppBuilder views', 'layout-text-window'],
  ['appbuilder_menu_items', 'AppBuilder menu items', 'menu-button-wide'],
];

const itemLabel = (item) => {
  if (typeof item === 'string') return item;
  if (item && typeof item === 'object') return item.name || item.label || item.url_route || item.href || item.category || JSON.stringify(item);
  return String(item);
};

export default async function render(ctx) {
  const { el, signal } = ctx;
  let q = ctx.query.get('q') || '';
  let plugins = [];
  const listEl = h('div', skeleton(6));
  const errEl = h('div');
  const countEl = h('span.af-muted.small');
  const refresh = refreshControl(() => load(), { key: 'plugins', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({ title: 'Plugins', icon: 'puzzle', subtitle: 'Airflow plugins loaded by the API server and what each one contributes.', actions: [refresh.el] }),
    errEl,
    h('div.af-toolbar',
      searchInput({ placeholder: 'Filter plugins or components…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } }),
      countEl),
    listEl);

  async function load() {
    try {
      const [data, errs] = await Promise.all([
        api.all('/plugins', 'plugins', {}, { signal }),
        api.get('/plugins/importErrors', null, { signal }).catch(() => ({ import_errors: [] })),
      ]);
      if (!ctx.active) return;
      plugins = data;
      mount(errEl, errs.import_errors.length ? card({
        title: `${errs.import_errors.length} plugin(s) failed to import`, icon: 'exclamation-octagon', cls: 'sys-card-danger',
        body: h('div.d-flex.flex-column.gap-2', errs.import_errors.map((e) => h('div',
          h('div.af-mono.small.fw-semibold.mb-1', e.source), codeBlock(e.error || '', { lang: 'text', maxHeight: '220px' })))),
      }) : null);
      draw();
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(listEl, errorState(err, load));
    }
  }

  function draw() {
    const needle = q.toLowerCase();
    const shown = plugins.filter((p) => !needle || JSON.stringify(p).toLowerCase().includes(needle));
    mount(countEl, `${shown.length} of ${plugins.length} plugins`);
    if (!shown.length) {
      mount(listEl, emptyState({ icon: 'puzzle', title: plugins.length ? 'No plugins match' : 'No plugins installed' }));
      return;
    }
    mount(listEl, h('div.sys-plugin-grid', shown.map(pluginCard)));
  }

  function pluginCard(p) {
    const parts = COMPONENTS.filter(([key]) => Array.isArray(p[key]) && p[key].length);
    const total = parts.reduce((a, [key]) => a + p[key].length, 0);
    return card({
      title: p.name, icon: 'puzzle-fill',
      actions: h('span.af-tag', `${total} component${total === 1 ? '' : 's'}`),
      body: h('div.d-flex.flex-column.gap-3',
        p.source && h('div.small.text-muted.af-mono', icon('file-earmark-code', 'me-1'), p.source),
        parts.length ? parts.map(([key, label, ic]) => h('div',
          h('div.sys-comp-title', icon(ic), label, h('span.af-tab-count', String(p[key].length))),
          h('div.af-tags.mt-1', p[key].map((item) => h('span.af-tag.sys-comp-item', { title: typeof item === 'object' ? JSON.stringify(item) : item }, itemLabel(item)))))) : h('div.small.text-muted', 'This plugin registers no UI or API components.'),
        h('details.sys-raw', h('summary.small.text-muted', 'Raw definition'), jsonView(p, { open: 1 }))),
    });
  }

  await load();
}
