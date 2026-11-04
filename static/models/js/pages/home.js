// Model Studio home: projects across your entities, live run activity, quick search,
// and a getting-started guide that shows the exact SDK settings for this server.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { card, statCard, emptyState, errorState, skeleton, timeAgo, refreshControl, copyButton, codeBlock } from '../../../ui/js/ui.js';
import { fmtNumber, fmtDuration } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { gql, config } from '../core/api.js';
import { urls } from '../core/urls.js';
import { loadViewer, loadProjects, RUN_FIELDS, normRun } from '../core/data.js';
import { runStateBadge, displayName, runStateInfo, fmtMetric } from '../core/wb.js';

const greeting = () => { const hr = new Date().getHours(); return hr < 12 ? 'Good morning' : hr < 18 ? 'Good afternoon' : 'Good evening'; };

export default async function render(ctx) {
  const { el, signal } = ctx;
  const statsEl = h('div.af-stats', skeleton(1, { height: 70 }));
  const projectsEl = h('div', skeleton(4));
  const feedEl = h('div', skeleton(6));
  const liveEl = h('div', skeleton(3));
  const startEl = h('div');
  const nameEl = h('span', config.user || '');
  const resultsEl = h('div.mh-results');
  let projects = [];

  const refresh = refreshControl(() => load(), { key: 'ms-home', initial: 30 });
  ctx.cleanup(refresh.stop);

  const searchInput = h('input.form-control.mh-search-input', {
    type: 'search', placeholder: 'Jump to a project…', 'aria-label': 'Search projects', dataset: { searchShortcut: '' },
    oninput: debounce(() => drawResults(), 80),
    onkeydown: (e) => {
      const items = [...resultsEl.querySelectorAll('a')];
      const i = items.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); (items[i + 1] || items[0])?.focus(); }
      if (e.key === 'Enter' && items[0] && i < 0) { e.preventDefault(); navigate(items[0].getAttribute('href')); }
    },
  });
  resultsEl.addEventListener('keydown', (e) => {
    const items = [...resultsEl.querySelectorAll('a')];
    const i = items.indexOf(document.activeElement);
    if (e.key === 'ArrowDown') { e.preventDefault(); items[i + 1]?.focus(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); (items[i - 1] || searchInput).focus(); }
  });

  mount(el,
    h('section.mh-hero',
      h('div.mh-hero-text', h('h1.mh-hello', `${greeting()}, `, nameEl),
        h('p.mh-sub', 'Track experiments, compare runs and ship the best model — fast, even with thousands of runs.')),
      h('div.mh-hero-actions',
        h('a.btn.btn-light', { href: urls.projects() }, icon('folder2', 'me-1'), 'All projects'),
        h('a.btn.btn-outline-light', { href: urls.registry() }, icon('box-seam', 'me-1'), 'Model registry'),
        refresh.el),
      h('div.mh-search', icon('search', 'mh-search-icon'), searchInput, resultsEl)),
    statsEl,
    h('div.af-grid-main-side',
      card({ title: 'Projects', icon: 'folder2', body: projectsEl, flush: true, actions: h('a.af-link-btn', { href: urls.projects() }, 'All projects →') }),
      card({ title: 'Running now', icon: 'activity', body: liveEl, flush: true })),
    card({ title: 'Recent runs', icon: 'clock-history', body: feedEl, flush: true }),
    startEl);

  function drawResults() {
    const q = searchInput.value.trim().toLowerCase();
    if (!q) { resultsEl.classList.remove('open'); mount(resultsEl); return; }
    const hits = projects.filter((p) => `${p.entityName}/${p.name}`.toLowerCase().includes(q)).slice(0, 8);
    resultsEl.classList.add('open');
    mount(resultsEl, hits.length ? hits.map((p) => h('a.mh-result', { href: urls.workspace(p.entityName, p.name) },
      icon('folder2'), h('span.mh-result-name', p.name), h('span.mh-result-sub', `${p.entityName} · ${fmtNumber(p.totalRuns || 0)} runs`)))
      : h('div.mh-result-empty', 'No project matches'));
  }

  async function load() {
    try {
      const { viewer, teams } = await loadViewer({ signal });
      if (!ctx.active) return;
      if (viewer?.name || viewer?.username) mount(nameEl, viewer.name || viewer.username);
      const entities = [...new Set([viewer?.username, viewer?.entity, ...teams.map((t) => t.name)].filter(Boolean))];
      projects = (await Promise.all(entities.map((e) => loadProjects(e, { max: 200, signal }).catch(() => []))))
        .flat().sort((a, b) => new Date(b.lastActive || b.updatedAt || 0) - new Date(a.lastActive || a.updatedAt || 0));
      const top = projects.slice(0, 6);
      // Recent + running runs for the most active projects, one query per project, in parallel.
      const per = await Promise.all(top.map((p) => gql(`query R($e: String!, $p: String!, $live: JSONString) { project(entityName: $e, name: $p) {
          running: runCount(filters: $live) crashed: runCount(filters: "{\\"state\\":{\\"$in\\":[\\"crashed\\",\\"failed\\"]}}")
          runs(first: 6, order: "-createdAt") { edges { node { ${RUN_FIELDS} } } }
          live: runs(first: 10, filters: $live, order: "-createdAt") { edges { node { ${RUN_FIELDS} } } } } }`,
      { e: p.entityName, p: p.name, live: JSON.stringify({ state: 'running' }) }, { signal }).then((d) => ({ p, d: d.project })).catch(() => ({ p, d: null }))));
      if (!ctx.active) return;
      renderStats(entities, per);
      renderProjects(per);
      renderFeed(per);
      renderLive(per);
      renderStart(viewer);
      drawResults();
    } catch (err) {
      if (err.name !== 'AbortError') mount(projectsEl, errorState(err, load));
    }
  }

  function renderStats(entities, per) {
    const totalRuns = projects.reduce((a, p) => a + (p.totalRuns || 0), 0);
    const running = per.reduce((a, x) => a + (x.d?.running || 0), 0);
    const crashed = per.reduce((a, x) => a + (x.d?.crashed || 0), 0);
    mount(statsEl,
      statCard({ label: 'Projects', value: fmtNumber(projects.length), icon: 'folder2', tone: 'blue', href: urls.projects() }),
      statCard({ label: 'Runs', value: fmtNumber(totalRuns), icon: 'play-circle', tone: 'teal', sub: 'across all projects' }),
      statCard({ label: 'Running now', value: fmtNumber(running), icon: 'activity', tone: running ? 'cyan' : 'slate', sub: 'in active projects' }),
      statCard({ label: 'Crashed / failed', value: fmtNumber(crashed), icon: 'x-octagon', tone: crashed ? 'red' : 'green', sub: 'in active projects' }),
      statCard({ label: 'Entities', value: fmtNumber(entities.length), icon: 'people', tone: 'purple', href: urls.teams(), sub: entities.join(', ') }));
  }

  function renderProjects(per) {
    if (!projects.length) {
      mount(projectsEl, emptyState({ icon: 'folder-plus', title: 'No projects yet', message: 'Log your first run (see Getting started below) — the project is created automatically.' }));
      return;
    }
    const byName = new Map(per.map((x) => [`${x.p.entityName}/${x.p.name}`, x.d]));
    mount(projectsEl, h('div.af-list', projects.slice(0, 8).map((p) => {
      const d = byName.get(`${p.entityName}/${p.name}`);
      return h('a.af-list-item.mh-proj', { href: urls.workspace(p.entityName, p.name) },
        h('span.mh-proj-icon', icon('folder2-open')),
        h('div.af-list-main', h('div.af-list-title', p.name), h('div.af-list-sub', `${p.entityName} · ${p.description || 'No description'}`)),
        d?.running ? h('span.ms-pill.ms-pill-info', h('span.af-dot.af-pulse', { style: { background: '#0891b2' } }), `${d.running} running`) : null,
        h('div.text-end.small.af-nowrap', h('div.fw-semibold', `${fmtNumber(p.totalRuns || 0)} runs`), h('div.text-muted', timeAgo(p.lastActive || p.updatedAt))));
    })));
  }

  function runItem(p, r) {
    const metric = Object.entries(r.summary).find(([k, v]) => typeof v === 'number' && /val.*acc|accuracy/i.test(k))
      || Object.entries(r.summary).find(([k, v]) => typeof v === 'number' && /loss/i.test(k));
    return h('a.af-list-item', { href: urls.run(p.entityName, p.name, r.name) },
      runStateBadge(r.state),
      h('div.af-list-main', h('div.af-list-title', displayName(r)), h('div.af-list-sub', `${p.name}${r.group ? ` · ${r.group}` : ''}${r.user ? ` · ${r.user.username}` : ''}`)),
      metric && h('span.mh-metric', h('span.text-muted', metric[0]), ' ', h('b', fmtMetric(metric[1]))),
      h('div.text-end.small.af-nowrap', h('div', r.runtime != null ? fmtDuration(r.runtime) : '—'), h('div.text-muted', timeAgo(r.createdAt))));
  }

  function renderFeed(per) {
    const items = per.flatMap((x) => (x.d?.runs?.edges || []).map((e) => ({ p: x.p, r: normRun(e.node) })))
      .sort((a, b) => new Date(b.r.createdAt) - new Date(a.r.createdAt)).slice(0, 12);
    mount(feedEl, items.length ? h('div.af-list', items.map(({ p, r }) => runItem(p, r)))
      : emptyState({ icon: 'clock-history', title: 'No runs yet' }));
  }

  function renderLive(per) {
    const items = per.flatMap((x) => (x.d?.live?.edges || []).map((e) => ({ p: x.p, r: normRun(e.node) })));
    mount(liveEl, items.length ? h('div.af-list', items.slice(0, 8).map(({ p, r }) => runItem(p, r)))
      : emptyState({ icon: 'cup-hot', title: 'Nothing running', message: 'Training runs appear here live, with their latest metrics.' }));
    void runStateInfo;
  }

  function renderStart(viewer) {
    const host = config.wandbUrl || 'http://localhost:8090';
    const project = projects[0]?.name || 'my-first-project';
    const snippet = `pip install wandb
wandb login --host=${host}

import wandb
run = wandb.init(entity="${viewer?.entity || 'your-entity'}", project="${project}", config={"lr": 1e-3, "epochs": 10})
for epoch in range(10):
    run.log({"train/loss": 1 / (epoch + 1), "val/accuracy": epoch / 10})
run.finish()`;
    mount(startEl, card({
      title: 'Getting started — log to this server', icon: 'rocket-takeoff',
      actions: copyButton(snippet, 'Copy'),
      body: h('div',
        h('p.text-muted.small.mb-2', 'Your training code talks to this self-hosted W&B Server. Point the SDK at it once; everything logged appears here live.'),
        codeBlock(snippet, { lang: 'python' })),
    }));
  }

  await load();
}
