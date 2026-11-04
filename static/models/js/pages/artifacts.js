// Artifacts of a project: type sidebar with counts, every collection searchable/sortable in one list.

import { h, icon, mount, prefs } from '../../../ui/js/dom.js';
import { pageHeader, card, statCard, searchInput, emptyState, errorState, skeleton, timeAgo, refreshControl } from '../../../ui/js/ui.js';
import { fmtNumber } from '../../../ui/js/format.js';
import { urls } from '../core/urls.js';
import { loadTypes, loadCollections, typeBadge, typeIcon, aliasChip, userAliases, fmtBytes, typeLook } from './artifacts/shared.js';

const SORTS = [
  { value: 'updated', label: 'Recently updated' },
  { value: 'name', label: 'Name (A–Z)' },
  { value: 'versions', label: 'Most versions' },
  { value: 'size', label: 'Largest latest version' },
];

export default async function render(ctx) {
  const { entity, project } = ctx.params;
  const { signal } = ctx;
  ctx.setTitle(`Artifacts · ${project}`);
  let type = ctx.query.get('type') || '';
  let q = ctx.query.get('q') || '';
  let sort = ctx.query.get('sort') || prefs.get('ma:sort', 'updated');
  let layout = prefs.get('ma:layout', 'list');

  const statsEl = h('div.af-stats');
  const sideEl = h('nav.ma-types', { 'aria-label': 'Artifact types' }, skeleton(5));
  const listEl = h('div', skeleton(8));
  const countEl = h('span.text-muted.small');
  let types = [];
  let all = [];

  const sortSel = h('select.form-select.form-select-sm.ma-sort', { 'aria-label': 'Sort', onchange: (e) => { sort = e.target.value; prefs.set('ma:sort', sort); ctx.setQuery({ sort }); drawList(); } },
    SORTS.map((s) => h('option', { value: s.value, selected: s.value === sort }, s.label)));
  const layoutBtns = h('div.btn-group.btn-group-sm', { role: 'group', 'aria-label': 'Layout' },
    ['list', 'cards'].map((l) => h('button.btn.btn-outline-secondary', {
      type: 'button', class: layout === l && 'active', title: l === 'list' ? 'List' : 'Cards',
      onclick: (e) => { layout = l; prefs.set('ma:layout', l); [...e.currentTarget.parentNode.children].forEach((b) => b.classList.toggle('active', b === e.currentTarget)); drawList(); },
    }, icon(l === 'list' ? 'list-ul' : 'grid-3x3-gap'))));

  const refresh = refreshControl(() => load(true), { key: 'artifacts', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(ctx.el,
    pageHeader({
      title: 'Artifacts', icon: 'boxes',
      crumbs: [{ label: 'Projects', href: urls.projects() }, { label: project, href: urls.workspace(entity, project) }, { label: 'Artifacts' }],
      subtitle: 'Datasets, models, tables and every other versioned file this project produced or used.',
      actions: [h('a.btn.btn-sm.btn-outline-primary', { href: urls.registry() }, icon('box-seam', 'me-1'), 'Model registry'), refresh.el],
    }),
    statsEl,
    h('div.ma-layout',
      h('aside.ma-side.af-card', sideEl),
      card({
        title: h('span', 'Collections ', countEl), icon: 'collection',
        actions: [searchInput({ placeholder: 'Search collections, aliases, tags…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); drawList(); } }), sortSel, layoutBtns],
        body: listEl, flush: true,
      })));

  async function load() {
    try {
      types = await loadTypes(entity, project, { signal });
      if (!ctx.active) return;
      const lists = await Promise.all(types.map((t) => loadCollections(entity, project, t.name, { signal }).then((cols) => cols.map((c) => ({ ...c, type: t.name })))));
      if (!ctx.active) return;
      all = lists.flat().map((c) => {
        const latest = c.artifacts?.edges?.[0];
        return {
          ...c,
          versions: c.artifacts?.totalCount ?? 0,
          latest: latest?.node,
          latestLabel: latest?.version,
          aliasList: userAliases((c.aliases?.edges || []).map((e) => e.node.alias)),
          tagList: (c.tags?.edges || []).map((e) => e.node.name),
        };
      });
      drawSide();
      drawStats();
      drawList();
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(listEl, errorState(err, () => load()));
      mount(sideEl);
    }
  }

  function drawStats() {
    const versions = all.reduce((a, c) => a + c.versions, 0);
    const bytes = all.reduce((a, c) => a + (c.latest?.size || 0), 0);
    const models = all.filter((c) => c.type === 'model').length;
    mount(statsEl,
      statCard({ label: 'Collections', value: fmtNumber(all.length), icon: 'collection', tone: 'blue' }),
      statCard({ label: 'Versions', value: fmtNumber(versions), icon: 'layers', tone: 'purple' }),
      statCard({ label: 'Types', value: fmtNumber(types.length), icon: 'tags', tone: 'teal' }),
      statCard({ label: 'Models', value: fmtNumber(models), icon: 'cpu', tone: 'amber', href: urls.artifacts(entity, project, { type: 'model' }) }),
      statCard({ label: 'Latest versions size', value: fmtBytes(bytes), icon: 'hdd', tone: 'slate' }));
  }

  function drawSide() {
    const item = (value, label, count, look) => h('button.ma-type-item', {
      type: 'button', class: type === value && 'active',
      onclick: () => { type = value; ctx.setQuery({ type: value || null }); drawSide(); drawList(); },
    }, look ? h('span.ma-type-dot', { style: { background: look.color } }) : icon('grid'), h('span.ma-type-name', label), h('span.ma-type-count', fmtNumber(count)));
    mount(sideEl,
      h('div.af-nav-section', 'Types'),
      item('', 'All types', all.length),
      types.map((t) => item(t.name, t.name, t.count, typeLook(t.name))),
      !types.length && h('div.small.text-muted.p-2', 'No artifacts logged yet.'));
  }

  function filtered() {
    const needle = q.toLowerCase();
    let rows = all.filter((c) => (!type || c.type === type)
      && (!needle || c.name.toLowerCase().includes(needle) || c.aliasList.some((a) => a.includes(needle)) || c.tagList.some((t) => t.toLowerCase().includes(needle)) || (c.description || '').toLowerCase().includes(needle)));
    const by = {
      updated: (a, b) => new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt),
      name: (a, b) => a.name.localeCompare(b.name),
      versions: (a, b) => b.versions - a.versions || a.name.localeCompare(b.name),
      size: (a, b) => (b.latest?.size || 0) - (a.latest?.size || 0),
    }[sort] || (() => 0);
    rows = [...rows].sort(by);
    return rows;
  }

  function drawList() {
    const rows = filtered();
    mount(countEl, `${rows.length.toLocaleString()}${rows.length !== all.length ? ` of ${all.length.toLocaleString()}` : ''}`);
    if (!all.length) {
      mount(listEl, emptyState({ icon: 'boxes', title: 'No artifacts yet', message: 'Log one from training code with wandb.log_artifact(…) and it appears here with its versions and lineage.' }));
      return;
    }
    if (!rows.length) { mount(listEl, emptyState({ icon: 'search', title: 'Nothing matches', message: 'Try another search or type.' })); return; }
    const link = (c) => urls.collection(entity, project, c.type, c.name);
    if (layout === 'cards') {
      mount(listEl, h('div.ma-cards', rows.slice(0, 300).map((c) => h('a.ma-card', { href: link(c) },
        h('div.ma-card-head', typeIcon(c.type), h('div.ma-card-title', h('div.ma-name', c.name), typeBadge(c.type))),
        c.description && h('div.ma-desc', c.description),
        h('div.ma-card-aliases', c.aliasList.slice(0, 5).map((a) => aliasChip(a))),
        h('div.ma-card-foot',
          h('span', icon('layers', 'me-1'), `${c.versions} version${c.versions === 1 ? '' : 's'}`),
          h('span', fmtBytes(c.latest?.size)),
          h('span', timeAgo(c.updatedAt || c.createdAt)))))));
      return;
    }
    mount(listEl, h('div.af-table-wrap', h('table.af-table.ma-table',
      h('thead', h('tr', h('th', 'Collection'), h('th', 'Type'), h('th.text-end', 'Versions'), h('th', 'Aliases'), h('th.text-end', 'Latest size'), h('th', 'Latest from'), h('th', 'Updated'))),
      h('tbody', rows.slice(0, 500).map((c) => h('tr.af-row-click', { onclick: (e) => { if (!e.target.closest('a')) ctx.navigate(link(c)); } },
        h('td.ma-name-col', h('div.d-flex.align-items-center.gap-2', typeIcon(c.type, 28), h('div.min-w-0', h('a.ma-name', { href: link(c) }, c.name),
          c.description && h('div.ma-desc.small', c.description), c.tagList.length ? h('div.ma-tags', c.tagList.map((t) => h('span.ms-tag', t))) : null))),
        h('td', typeBadge(c.type)),
        h('td.text-end.ms-metric', fmtNumber(c.versions)),
        h('td', h('div.ma-aliases.compact', c.aliasList.slice(0, 4).map((a) => aliasChip(a)), c.aliasList.length > 4 && h('span.small.text-muted', `+${c.aliasList.length - 4}`))),
        h('td.text-end.ms-metric', fmtBytes(c.latest?.size)),
        h('td', c.latest?.createdBy?.__typename === 'Run'
          ? h('a.ms-run-name', { href: urls.run(entity, project, c.latest.createdBy.name) }, c.latest.createdBy.displayName || c.latest.createdBy.name)
          : h('span.text-muted', '—')),
        h('td.af-nowrap', timeAgo(c.updatedAt || c.createdAt))))))),
    rows.length > 500 && h('div.small.text-muted.p-2', `Showing the first 500 of ${rows.length}. Narrow the search to see more.`));
  }

  await load();
}
