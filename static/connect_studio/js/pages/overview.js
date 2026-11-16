// Overview: hero flow, onboarding, KPIs, activity, recent mappings & jobs.

import { h, s, icon, mount } from '../../../ui/js/dom.js';
import { href, navigate } from '../../../ui/js/router.js';
import { api } from '../api.js';
import { ago, compact, logo, n, openJobDrawer, ring, statusBadge, statusDot } from '../kit.js';

function heroFlow() {
  const nodes = [
    { x: 10, y: 22, w: 128, label: 'PostgreSQL · MySQL', sub: 'databases', c: '#60a5fa' },
    { x: 10, y: 92, w: 128, label: 'CSV · Excel · JSON', sub: 'files', c: '#34d399' },
    { x: 10, y: 162, w: 128, label: 'REST · Afya API', sub: 'APIs', c: '#a78bfa' },
    { x: 186, y: 92, w: 112, label: 'dbatlas', sub: 'schema + relations', c: '#5eead4' },
    { x: 340, y: 92, w: 112, label: 'Groq AI', sub: 'gpt-oss mapping', c: '#c4b5fd' },
    { x: 494, y: 42, w: 120, label: 'mappings.json', sub: 'reviewed', c: '#fcd34d' },
    { x: 494, y: 142, w: 120, label: 'Destinations', sub: 'synced on schedule', c: '#f9a8d4' },
  ];
  const H = 40;
  const mid = (nd) => [nd.x + nd.w, nd.y + H / 2];
  const left = (nd) => [nd.x, nd.y + H / 2];
  const curve = ([x1, y1], [x2, y2]) => `M${x1},${y1} C${x1 + 40},${y1} ${x2 - 40},${y2} ${x2},${y2}`;
  const edges = [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6]];
  return s('svg.cs-hero-flow', { viewBox: '0 0 624 220', role: 'img', 'aria-label': 'Sources flow through dbatlas and Groq into mappings and destinations' },
    s('defs', ...edges.map((_, i) => s('linearGradient', { id: `csg${i}`, x1: 0, x2: 1, y1: 0, y2: 0 },
      s('stop', { offset: '0%', 'stop-color': '#5eead4' }), s('stop', { offset: '100%', 'stop-color': '#c4b5fd' })))),
    edges.map(([a, b], i) => [
      s('path.path', { d: curve(mid(nodes[a]), left(nodes[b])) }),
      s('path.pulse', { d: curve(mid(nodes[a]), left(nodes[b])), stroke: `url(#csg${i})`, style: `animation-delay:${i * 0.45}s` }),
    ]),
    nodes.map((nd) => s('g.node',
      s('rect', { x: nd.x, y: nd.y, width: nd.w, height: H, rx: 11 }),
      s('circle', { cx: nd.x + 14, cy: nd.y + H / 2, r: 4, fill: nd.c }),
      s('text', { x: nd.x + 26, y: nd.y + 17 }, nd.label),
      s('text.sub', { x: nd.x + 26, y: nd.y + 31 }, nd.sub))));
}

function activityChart(daily) {
  const max = Math.max(1, ...daily.map((d) => d.total));
  return h('div',
    h('div.cs-bars', daily.map((d) => h('div.cs-bars-col', { title: `${d.day}: ${d.succeeded} succeeded · ${d.failed} failed · ${d.total} total` },
      h('span', { style: { height: `${(d.failed / max) * 100}%`, background: '#f87171' } }),
      h('span', { style: { height: `${((d.total - d.failed - d.succeeded) / max) * 100}%`, background: '#93c5fd' } }),
      h('span', { style: { height: `${(d.succeeded / max) * 100}%`, background: 'linear-gradient(#0bb99f, #0072ce)' } })))),
    h('div.cs-bars-axis', daily.map((d, i) => h('span', i % 2 ? '' : d.day.slice(5)))));
}

export default async function render(ctx) {
  ctx.setTitle('Overview');
  const [data, meta] = await Promise.all([api.overview({ signal: ctx.signal }), api.meta()]);
  const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
  const c = data.counts;

  const steps = [
    { done: c.sources > 0, title: 'Connect a source', sub: 'A database, files or an API — schema extracted by dbatlas.', to: '/sources/new' },
    { done: data.sources_by_kind.length > 1 || c.mappings > 0, title: 'Add a target schema', sub: 'Upload schema.json / DDL, or point at the target database.', to: '/sources/new?kind=schema_file' },
    { done: c.mappings > 0, title: 'Generate an AI mapping', sub: 'Groq maps every target column; you review.', to: '/mappings/new' },
    { done: c.connections > 0, title: 'Sync it', sub: 'Full refresh or incremental, on a schedule.', to: '/syncs?new=1' },
  ];
  const onboarding = steps.some((x) => !x.done);

  const kpi = (label, value, ic, color, sub, to, bad) => h(to ? 'a.cs-kpi' : 'div.cs-kpi', { href: to ? href(to) : undefined },
    h('span.cs-kpi-icon', { style: { '--c': color } }, icon(ic)),
    h('div', h('div.cs-kpi-label', label), h('div.cs-kpi-value', value), sub && h('div.cs-kpi-sub', { class: bad && 'bad' }, sub)));

  mount(ctx.el,
    h('section.cs-hero',
      h('div', { style: { position: 'relative', zIndex: 1 } },
        h('div.cs-row.cs-gap-sm', { style: { marginBottom: '10px' } }, h('span.cs-pill-ai', icon('stars'), meta.llm.available ? 'AI ON' : 'RULES ONLY'),
          h('span', { style: { color: '#8fa3c4', fontSize: '12.5px' } }, meta.llm.available ? `Groq · ${meta.llm.model}` : 'Set GROQ_API_KEY for AI mapping')),
        h('h2', 'Move data from ', h('em', 'anywhere to anywhere'), ' — mapped by AI.'),
        h('p', 'Connect a source, let dbatlas extract its schema and relationships, let Groq map it onto your target schema, review it visually, then sync it — and export the mappings.json whenever you need it.'),
        h('div.cs-row.cs-wrap',
          h('a.cs-btn.cs-btn-brand.cs-btn-lg', { href: href('/sources/new') }, icon('plus-lg'), 'Add a source'),
          h('a.cs-btn.cs-btn-ai.cs-btn-lg', { href: href('/mappings/new') }, icon('stars'), 'New AI mapping'),
          h('a.cs-btn.cs-btn-lg', { href: href('/syncs?new=1') }, icon('arrow-repeat'), 'New sync'))),
      heroFlow()),

    onboarding && h('section.cs-card.cs-card-pad',
      h('div.cs-row', { style: { marginBottom: '12px' } }, h('h2.cs-card-title', icon('rocket-takeoff'), 'Get started'),
        h('span.cs-muted', { style: { fontSize: '12.5px' } }, `${steps.filter((x) => x.done).length} of ${steps.length} done`)),
      h('div.cs-grid.cs-grid-4', steps.map((st, i) => h('a.cs-check', { href: href(st.to), class: st.done && 'done' },
        h('span.cs-check-n', st.done ? icon('check-lg') : String(i + 1)), h('div', h('div.cs-check-title', st.title), h('div.cs-check-sub', st.sub)))))),

    h('section.cs-kpis',
      kpi('Sources', n(c.sources), 'database', '#0072ce', c.sources_error ? `${c.sources_error} with errors` : `${n(c.snapshots)} snapshots`, '/sources', c.sources_error),
      kpi('Mappings', n(c.mappings), 'stars', '#7c3aed', c.mappings_review ? `${c.mappings_review} waiting for review` : 'all reviewed', '/mappings'),
      kpi('Syncs', n(c.connections), 'arrow-repeat', '#0bb99f', `${c.connections_scheduled} scheduled`, '/syncs'),
      kpi('Rows synced · 7d', compact(c.rows_week), 'arrow-left-right', '#0891b2', `${n(c.jobs_week)} jobs this week`, '/activity'),
      kpi('AI tokens used', compact(c.ai_tokens), 'cpu', '#db2777', c.jobs_failed_week ? `${c.jobs_failed_week} failed jobs · 7d` : 'no failed jobs this week', '/activity', c.jobs_failed_week)),

    h('section.cs-grid.cs-grid-main',
      h('div.cs-card',
        h('div.cs-card-head', h('h2.cs-card-title', icon('bar-chart-line'), 'Activity · last 14 days'), h('span.cs-grow'),
          h('span.cs-row.cs-gap-sm.cs-muted', { style: { fontSize: '12px' } }, h('span.cs-dot.ok'), 'succeeded', h('span.cs-dot.bad'), 'failed')),
        h('div.cs-card-body', activityChart(data.daily))),
      h('div.cs-card',
        h('div.cs-card-head', h('h2.cs-card-title', icon('diagram-3'), 'Sources by type'), h('span.cs-grow'), h('a.cs-btn.cs-btn-sm.cs-btn-ghost', { href: href('/sources') }, 'All', icon('arrow-right'))),
        h('div.cs-list', data.sources_by_kind.length ? data.sources_by_kind.map((k) => h('a.cs-li', { href: href(`/sources?kind=${k.kind}`) },
          logo(specs[k.kind] || k, { size: 'sm' }), h('div.cs-grow', h('div.cs-li-title', k.label)),
          h('span.cs-badge', String(k.count)))) : h('div.cs-li.cs-muted', 'No sources yet.')))),

    h('section.cs-grid.cs-grid-main',
      h('div.cs-card',
        h('div.cs-card-head', h('h2.cs-card-title', icon('stars'), 'Recent mappings'), h('span.cs-grow'),
          h('a.cs-btn.cs-btn-sm.cs-btn-ghost', { href: href('/mappings') }, 'All', icon('arrow-right'))),
        data.recent_mappings.length ? h('div.cs-list', data.recent_mappings.map((m) => h('a.cs-li', { href: href(`/mappings/${m.id}`) },
          ring(m.summary?.coverage || 0, { size: 40, stroke: 4 }),
          h('div.cs-grow', h('div.cs-li-title.cs-ellipsis', m.name),
            h('div.cs-li-sub', `${m.summary?.mapped ?? 0}/${m.summary?.columns ?? 0} columns · ${m.summary?.approved ?? 0} approved · ${ago(m.updated_at)}`)),
          statusBadge(m.status))))
          : h('div.cs-card-body', h('div.cs-callout.cs-callout-ai', icon('stars'), h('div', 'No mappings yet. ', h('a', { href: href('/mappings/new') }, 'Create one'), ' — Groq maps a source schema onto your target in seconds.')))),
      h('div.cs-card',
        h('div.cs-card-head', h('h2.cs-card-title', icon('activity'), 'Latest jobs'), h('span.cs-grow'),
          h('a.cs-btn.cs-btn-sm.cs-btn-ghost', { href: href('/activity') }, 'All', icon('arrow-right'))),
        h('div.cs-list', data.recent_jobs.length ? data.recent_jobs.slice(0, 8).map((j) => h('div.cs-li.click', { onclick: () => openJobDrawer(j) },
          statusDot(j.status), h('div.cs-grow', h('div.cs-li-title.cs-ellipsis', j.title || j.kind_label), h('div.cs-li-sub.cs-ellipsis', j.message || j.kind_label)),
          h('span.cs-muted.cs-nowrap', { style: { fontSize: '12px' } }, ago(j.created_at)))) : h('div.cs-li.cs-muted', 'Nothing has run yet.')))));
}
