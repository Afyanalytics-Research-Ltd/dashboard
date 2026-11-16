// Sources: every configured connector, with live status and snapshot stats.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { href, navigate } from '../../../ui/js/router.js';
import { api } from '../api.js';
import { ago, compact, empty, logo, n, seg, statusBadge } from '../kit.js';

export default async function render(ctx) {
  ctx.setTitle('Sources');
  const [{ sources }, meta] = await Promise.all([api.sources({}, { signal: ctx.signal }), api.meta()]);
  const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
  let cat = ctx.query.get('category') || '';
  let q = '';
  const kindFilter = ctx.query.get('kind') || '';
  const grid = h('div.cs-grid.cs-grid-auto');
  const cats = [...new Set(sources.map((s) => specs[s.kind]?.category).filter(Boolean))];

  const card = (src) => {
    const spec = specs[src.kind] || {};
    const st = src.snapshot?.stats;
    const drift = src.snapshot?.diff;
    return h('a.cs-card.hoverable.cs-source-card', { href: href(`/sources/${src.id}`) },
      h('div.cs-source-top', logo(spec, { size: 'lg' }),
        h('div.cs-grow', h('div.cs-source-name', { title: src.name }, src.name), h('div.cs-source-kind', spec.label || src.kind, h('span.cs-faint', '·'), spec.category)),
        statusBadge(src.status)),
      st ? h('div.cs-statline',
        h('div.cs-stat', h('b', n(st.tables)), h('span', 'tables')),
        h('div.cs-stat', h('b', n(st.columns)), h('span', 'columns')),
        h('div.cs-stat', h('b', n(st.relationships)), h('span', 'relations')))
        : h('div.cs-callout.cs-callout-info', icon('info-circle'), h('div', src.status === 'error' ? src.last_error : 'No schema snapshot yet — open it and extract.')),
      h('div.cs-source-foot',
        src.snapshot ? [icon('camera'), `extracted ${ago(src.snapshot.created_at)}`] : [icon('clock'), `added ${ago(src.created_at)}`],
        h('span.cs-grow'),
        drift && !drift.first && drift.summary !== 'no changes' && h('span.cs-badge.cs-badge-warn', { title: 'Schema drift since the previous snapshot' }, icon('shuffle'), drift.summary),
        st?.rows ? h('span.cs-badge.cs-badge-outline', `${compact(st.rows)} rows`) : null,
        src.mappings ? h('span.cs-badge.cs-badge-ai', icon('stars'), String(src.mappings)) : null));
  };

  const renderGrid = () => {
    const list = sources.filter((s) => (!cat || specs[s.kind]?.category === cat) && (!kindFilter || s.kind === kindFilter)
      && (!q || `${s.name} ${specs[s.kind]?.label}`.toLowerCase().includes(q)));
    mount(grid, list.length ? list.map(card) : h('div', { style: { gridColumn: '1/-1' } },
      empty({ icon: 'search', title: 'No sources match', message: 'Try another filter.' })));
  };

  if (!sources.length) {
    const featured = meta.connectors.slice(0, 8);
    mount(ctx.el,
      h('div.cs-head', h('div', h('h1', 'Sources'), h('div.cs-head-sub', 'Databases, files, APIs and schema files — anything you map from, map to or sync into.'))),
      h('div.cs-card', empty({ icon: 'database-add', title: 'Connect your first source', message: 'Pick a connector below. dbatlas extracts tables, keys, profiles and relationships automatically.',
        action: h('a.cs-btn.cs-btn-brand.cs-btn-lg', { href: href('/sources/new') }, icon('plus-lg'), 'Add a source') }),
      h('div.cs-grid.cs-grid-tiles', { style: { padding: '0 18px 22px' } }, featured.map((c) => h('button.cs-tile', { type: 'button', style: { '--c': c.color },
        onclick: () => navigate(`/sources/new?kind=${c.key}`) }, h('div.cs-row', logo(c), h('div', h('div.cs-tile-name', c.label), h('div.cs-tile-cat', c.category))),
        h('div.cs-tile-desc', c.description))))));
    return;
  }

  mount(ctx.el,
    h('div.cs-head',
      h('div', h('h1', 'Sources', h('span.cs-badge', String(sources.length))), h('div.cs-head-sub', 'Every connection you can map from, map to, or sync into. Click one to explore its schema and relationships.')),
      h('div.cs-head-actions', h('a.cs-btn.cs-btn-brand', { href: href('/sources/new') }, icon('plus-lg'), 'Add source'))),
    h('div.cs-row.cs-wrap',
      seg([{ value: '', label: 'All', count: sources.length }, ...cats.map((c) => ({ value: c, label: c, count: sources.filter((s) => specs[s.kind]?.category === c).length }))],
        cat, (v) => { cat = v; ctx.setQuery({ category: v }); renderGrid(); }),
      h('span.cs-grow'),
      h('div.cs-input-wrap', { style: { width: '280px' } }, icon('search'),
        h('input.cs-input', { placeholder: 'Filter sources…', oninput: (e) => { q = e.target.value.trim().toLowerCase(); renderGrid(); } }))),
    grid);
  renderGrid();
}
