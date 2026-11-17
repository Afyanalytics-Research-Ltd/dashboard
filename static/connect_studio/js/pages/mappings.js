// Mappings: every source → target mapping with coverage and review progress.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { href } from '../../../ui/js/router.js';
import { api } from '../api.js';
import { ago, compact, empty, logo, n, ring, seg, statusBadge } from '../kit.js';

export default async function render(ctx) {
  ctx.setTitle('Mappings');
  const [{ mappings }, meta] = await Promise.all([api.mappings({ signal: ctx.signal }), api.meta()]);
  const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
  let status = ctx.query.get('status') || '';
  const grid = h('div.cs-grid.cs-grid-auto');

  const card = (m) => {
    const s = m.summary || {};
    return h('a.cs-card.hoverable.cs-mcard', { href: href(`/mappings/${m.id}`) },
      h('div.cs-row', h('div.cs-mcard-name.cs-grow.cs-ellipsis', m.name), statusBadge(m.status)),
      h('div.cs-mcard-flow',
        h('span.cs-flowchip', logo(specs[m.source.kind], { size: 'xs' }), h('span.cs-ellipsis', m.source.name)),
        h('span.cs-flow-arrow', h('span.dash'), icon('chevron-right')),
        h('span.cs-flowchip', logo(specs[m.target.kind], { size: 'xs' }), h('span.cs-ellipsis', m.target.name))),
      h('div.cs-row', { style: { gap: '16px' } },
        h('div.cs-row.cs-gap-sm', ring(s.coverage || 0, { size: 46 }), h('div', h('div.cs-mstat-v', `${n(s.mapped)}/${n(s.columns)}`), h('div.cs-mstat-k', 'columns mapped'))),
        h('div.cs-row.cs-gap-sm', ring(s.approved_pct || 0, { size: 46, color: '#16a34a' }), h('div', h('div.cs-mstat-v', n(s.approved)), h('div.cs-mstat-k', 'approved')))),
      h('div.cs-source-foot',
        s.review ? h('span.cs-badge.cs-badge-warn', icon('eye'), `${s.review} to review`) : null,
        s.unmapped ? h('span.cs-badge.cs-badge-outline', `${s.unmapped} unmapped`) : null,
        h('span.cs-grow'),
        m.usage?.total_tokens ? h('span.cs-badge.cs-badge-ai', { title: `${m.usage.calls} Groq calls` }, icon('stars'), `${compact(m.usage.total_tokens)} tok`) : null,
        h('span', ago(m.updated_at))));
  };

  const draw = () => {
    const list = mappings.filter((m) => !status || m.status === status);
    mount(grid, list.length ? list.map(card) : h('div', { style: { gridColumn: '1/-1' } }, empty({ icon: 'funnel', title: 'Nothing here', message: 'No mappings with this status.' })));
  };

  if (!mappings.length) {
    mount(ctx.el,
      h('div.cs-head', h('div', h('h1', 'Mappings'), h('div.cs-head-sub', 'A mapping turns a source schema into a target schema — table by table, column by column.'))),
      h('div.cs-card', empty({ icon: 'stars', title: 'Create your first AI mapping', message: 'Pick a source and a target schema. Groq proposes every table match and column rule; you review and export mappings.json.',
        action: h('a.cs-btn.cs-btn-ai.cs-btn-lg', { href: href('/mappings/new') }, icon('stars'), 'New AI mapping') })));
    return;
  }
  const counts = (st) => mappings.filter((m) => m.status === st).length;
  mount(ctx.el,
    h('div.cs-head',
      h('div', h('h1', 'Mappings', h('span.cs-badge', String(mappings.length))), h('div.cs-head-sub', 'Review AI suggestions, fix the rest by drag & drop, export mappings.json, sync.')),
      h('div.cs-head-actions', h('a.cs-btn.cs-btn-ai', { href: href('/mappings/new') }, icon('stars'), 'New AI mapping'))),
    seg([{ value: '', label: 'All', count: mappings.length }, { value: 'review', label: 'In review', count: counts('review') },
      { value: 'approved', label: 'Approved', count: counts('approved') }, { value: 'generating', label: 'Generating', count: counts('generating') },
      { value: 'failed', label: 'Failed', count: counts('failed') }].filter((o) => !o.value || o.count), status, (v) => { status = v; ctx.setQuery({ status: v }); draw(); }),
    grid);
  draw();
}
