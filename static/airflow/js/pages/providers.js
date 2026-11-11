// Providers: installed provider packages with versions and docs links.

import { h, icon, mount } from '../core/dom.js';
import { api } from '../core/api.js';
import {
  pageHeader, card, dataTable, searchInput, emptyState, errorState, skeleton, statCard, copyButton,
} from '../core/ui.js';

const shortName = (pkg) => pkg.replace(/^apache-airflow-providers-/, '');

export default async function render(ctx) {
  const { el, signal } = ctx;
  let q = ctx.query.get('q') || '';
  let sort = { key: 'package_name', dir: 'asc' };
  let providers = [];
  const statsEl = h('div.af-stats', skeleton(1, { height: 60 }));
  const tableEl = h('div', skeleton(10));
  const countEl = h('span.af-muted.small');

  mount(el,
    pageHeader({ title: 'Providers', icon: 'box-seam', subtitle: 'Provider packages installed on the Airflow server — hooks, operators and connection types come from these.' }),
    statsEl,
    card({
      title: 'Installed packages', icon: 'boxes', actions: countEl, flush: true,
      body: h('div',
        h('div.af-toolbar.p-3.pb-2', searchInput({
          placeholder: 'Search packages or descriptions…', value: q,
          onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); },
        })),
        tableEl),
    }));

  try {
    providers = await api.all('/providers', 'providers', {}, { signal });
  } catch (err) {
    if (err.name === 'AbortError' || !ctx.active) return;
    mount(tableEl, h('div.p-3', errorState(err, () => render(ctx))));
    mount(statsEl);
    return;
  }
  if (!ctx.active) return;
  const official = providers.filter((p) => p.package_name.startsWith('apache-airflow-providers-')).length;
  mount(statsEl,
    statCard({ label: 'Provider packages', value: providers.length, icon: 'box-seam', tone: 'blue' }),
    statCard({ label: 'Official (apache-airflow-providers-*)', value: official, icon: 'patch-check', tone: 'teal' }),
    statCard({ label: 'Third-party', value: providers.length - official, icon: 'boxes', tone: 'purple' }));
  draw();

  function draw() {
    const needle = q.toLowerCase();
    const rows = providers
      .filter((p) => !needle || `${p.package_name} ${p.description}`.toLowerCase().includes(needle))
      .sort((a, b) => {
        const cmp = sort.key === 'version'
          ? String(a.version).localeCompare(String(b.version), undefined, { numeric: true })
          : String(a[sort.key]).localeCompare(String(b[sort.key]));
        return sort.dir === 'asc' ? cmp : -cmp;
      });
    mount(countEl, `${rows.length} of ${providers.length}`);
    mount(tableEl, dataTable({
      rows,
      rowKey: (p) => p.package_name,
      sort,
      onSort: (key, dir) => { sort = { key, dir }; draw(); },
      empty: emptyState({ icon: 'search', title: 'No providers match' }),
      columns: [
        {
          key: 'package_name', label: 'Package', sortable: true,
          render: (p) => h('div',
            h('div.fw-semibold', shortName(p.package_name)),
            h('div.af-mono.small.text-muted.d-flex.align-items-center.gap-1', p.package_name, copyButton(`${p.package_name}==${p.version}`))),
        },
        { key: 'version', label: 'Version', sortable: true, width: '110px', render: (p) => h('span.sys-version', p.version) },
        { key: 'description', label: 'Description', render: (p) => h('div.small.sys-provider-desc', p.description) },
        {
          key: 'docs', label: 'Docs', width: '80px', align: 'center',
          render: (p) => {
            const url = (p.documentation_url || '').trim();
            return url ? h('a.btn.btn-sm.btn-outline-secondary', { href: url, target: '_blank', rel: 'noopener', title: url }, icon('book')) : h('span.text-muted', '—');
          },
        },
      ],
    }));
  }
}
