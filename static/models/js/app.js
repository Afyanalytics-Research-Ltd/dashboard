// Model Studio entry point: layout, context-aware navigation (global + current project),
// command palette (Ctrl/⌘+K), keyboard shortcuts and server health.

import { h, icon, mount, debounce, prefs } from '../../ui/js/dom.js';
import { route, start, navigate, onRouteChange, href } from '../../ui/js/router.js';
import { modal, startClock, stateDot, toast } from '../../ui/js/ui.js';
import { gql, onRequest, config } from './core/api.js';
import { urls } from './core/urls.js';
import { loadViewer, loadProjects } from './core/data.js';

const page = (name) => () => import(`./pages/${name}.js`);
const P = '/p/:entity/:project';

route('/', page('home'), { title: 'Home' });
route('/projects', page('projects'), { title: 'Projects' });
route(P, page('workspace'), { title: 'Workspace' });
route(`${P}/table`, page('runs_table'), { title: 'Runs' });
route(`${P}/runs/:run`, page('run'), { title: 'Run' });
route(`${P}/compare`, page('compare'), { title: 'Compare runs' });
route(`${P}/sweeps`, page('sweeps'), { title: 'Sweeps' });
route(`${P}/sweeps/:sweep`, page('sweep'), { title: 'Sweep' });
route(`${P}/artifacts`, page('artifacts'), { title: 'Artifacts' });
route(`${P}/artifacts/:type/:collection`, page('collection'), { title: 'Artifact' });
route(`${P}/artifacts/:type/:collection/:version`, page('artifact'), { title: 'Artifact version' });
route(`${P}/reports`, page('reports'), { title: 'Reports' });
route(`${P}/reports/:viewId`, page('report'), { title: 'Report' });
route(`${P}/settings`, page('project_settings'), { title: 'Project settings' });
route('/registry', page('registry'), { title: 'Model registry' });
route('/teams', page('teams'), { title: 'Teams' });
route('/teams/:entity', page('team'), { title: 'Team' });
route('/users', page('users'), { title: 'Users' });
route('/system', page('system'), { title: 'System' });
route('/cleanup', page('cleanup'), { title: 'Cleanup' });
route('/explorer', page('explorer'), { title: 'API explorer' });

export const NAV = [
  { section: 'Models', items: [
    { label: 'Home', icon: 'house', path: '/', key: 'g h' },
    { label: 'Projects', icon: 'folder2', path: '/projects', key: 'g p' },
    { label: 'Model registry', icon: 'box-seam', path: '/registry', key: 'g m' },
  ] },
  { section: 'Admin', items: [
    { label: 'Teams', icon: 'people', path: '/teams', key: 'g t' },
    { label: 'Users', icon: 'person-badge', path: '/users', key: 'g u' },
    { label: 'System', icon: 'heart-pulse', path: '/system' },
    { label: 'Cleanup', icon: 'recycle', path: '/cleanup' },
    { label: 'API explorer', icon: 'terminal', path: '/explorer', key: 'g x' },
  ] },
];

const PROJECT_NAV = [
  { label: 'Workspace', icon: 'graph-up', sub: '', key: 'w' },
  { label: 'Runs', icon: 'table', sub: '/table', key: 't' },
  { label: 'Sweeps', icon: 'shuffle', sub: '/sweeps', key: 's' },
  { label: 'Artifacts', icon: 'boxes', sub: '/artifacts', key: 'a' },
  { label: 'Reports', icon: 'file-earmark-richtext', sub: '/reports', key: 'r' },
  { label: 'Settings', icon: 'gear', sub: '/settings' },
];

// ------------------------------------------------------------------ layout

const root = document.getElementById('af-app');
const navLinks = [];
const projectNav = h('div.af-nav-group.ms-project-nav');

const navEl = h('nav.af-nav', { 'aria-label': 'Model Studio navigation' },
  NAV.map((group, gi) => [
    h('div.af-nav-group',
      h('div.af-nav-section', group.section),
      group.items.map((item) => {
        const a = h('a.af-nav-link', { href: href(item.path) }, icon(item.icon), h('span', item.label));
        navLinks.push({ a, item });
        return a;
      })),
    gi === 0 ? projectNav : null,
  ]));

const healthPill = h('a.af-health-pill', { href: urls.system(), title: 'W&B Server status' }, stateDot('none'), h('span', 'Checking…'));
const versionEl = h('span.af-version');
const netEl = h('span.af-net', { title: 'Requests in flight' });
const outlet = h('main.af-main');

const shell = h('div.af-shell',
  h('aside.af-side',
    h('div.af-side-head',
      h('div.af-brand', h('span.af-brand-logo.ms-brand-logo', icon('cpu-fill')), h('div', h('div.af-brand-name', 'Model Studio'), versionEl))),
    h('button.af-palette-btn', { type: 'button', onclick: () => openPalette() },
      icon('search'), h('span', 'Search…'), h('kbd', navigator.platform.includes('Mac') ? '⌘K' : 'Ctrl K')),
    navEl,
    h('div.af-side-foot', healthPill, netEl)),
  outlet);

const mobileToggle = h('button.af-mobile-nav.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => shell.classList.toggle('nav-open') }, icon('list'), ' Menu');

if (!config.configured) {
  mount(root, h('div.af-error.m-3', icon('key', 'af-error-icon'), h('div',
    h('div.af-error-title', 'Weights & Biases is not connected'),
    h('div.af-error-msg', 'Set WANDB_API_KEY (W&B → /authorize) and WANDB_BASE_URL in .env, then restart the dashboard. See docker-compose.wandb.yaml.'))));
} else {
  mount(root, mobileToggle, shell);
}

let currentProject = null;
onRouteChange((found) => {
  shell.classList.remove('nav-open');
  const path = found?.path || '';
  const m = path.match(/^\/p\/([^/]+)\/([^/]+)(\/[^/]*)?/);
  const proj = m ? { entity: decodeURIComponent(m[1]), project: decodeURIComponent(m[2]), section: m[3] || '' } : null;
  renderProjectNav(proj);
  let best = null;
  for (const l of navLinks) {
    const p = l.item.path;
    const hit = p === '/' ? path === '/' : path === p || path.startsWith(`${p}/`);
    if (hit && (!best || p.length > best.item.path.length)) best = l;
  }
  if (!best && proj) best = navLinks.find((l) => l.item.path === '/projects');
  navLinks.forEach((l) => l.a.classList.toggle('active', l === best && !proj));
});

function renderProjectNav(proj) {
  if (!proj) { mount(projectNav); currentProject = null; return; }
  if (proj) prefs.set('ms:lastProject', { entity: proj.entity, project: proj.project });
  currentProject = proj;
  const base = `/p/${encodeURIComponent(proj.entity)}/${encodeURIComponent(proj.project)}`;
  mount(projectNav,
    h('div.af-nav-section.ms-project-title', { title: `${proj.entity}/${proj.project}` },
      icon('folder2-open', 'me-1'), h('span', proj.project)),
    PROJECT_NAV.map((it) => {
      const active = it.sub === '' ? (proj.section === '' || /^\/(runs|compare)/.test(proj.section)) : proj.section.startsWith(it.sub);
      return h('a.af-nav-link', { href: href(`${base}${it.sub}`), class: active && 'active' }, icon(it.icon), h('span', it.label), it.key && h('kbd.ms-nav-key', it.key));
    }),
    h('button.af-nav-link.ms-switch-project', { type: 'button', onclick: () => openPalette('', { projectsOnly: true }) }, icon('arrow-left-right'), h('span', 'Switch project…')));
}

// ------------------------------------------------------------------ health

async function refreshStatus() {
  try {
    const d = await gql('query Health { serverInfo { latestLocalVersionInfo { versionOnThisInstanceString } } viewer { id } }');
    mount(versionEl, `W&B Server v${d.serverInfo?.latestLocalVersionInfo?.versionOnThisInstanceString || '?'}`);
    mount(healthPill, stateDot(d.viewer ? 'success' : 'failed'), h('span', d.viewer ? 'Connected' : 'API key rejected'));
    healthPill.classList.toggle('bad', !d.viewer);
  } catch {
    mount(healthPill, stateDot('failed'), h('span', 'W&B unreachable'));
    healthPill.classList.add('bad');
  }
}

if (config.configured) {
  refreshStatus();
  setInterval(() => { if (!document.hidden) refreshStatus(); }, 60000);
}

let inflight = 0;
const origFetch = window.fetch;
window.fetch = (...args) => {
  inflight++;
  netEl.classList.add('busy');
  return origFetch(...args).finally(() => { inflight--; if (!inflight) netEl.classList.remove('busy'); });
};
let sessionToastShown = false;
onRequest(({ sessionExpired }) => {
  if (!sessionExpired || sessionToastShown) return;
  sessionToastShown = true;
  toast('Your session expired. Reload the page to sign in again.', 'warning', { title: 'Signed out', timeout: 0 });
});

// ------------------------------------------------------------------ command palette

let projectCache = null;
async function allProjects() {
  if (projectCache && Date.now() - projectCache.at < 60000) return projectCache.list;
  const { viewer, teams } = await loadViewer();
  const entities = [...new Set([viewer?.entity, ...teams.map((t) => t.name)].filter(Boolean))];
  const lists = await Promise.all(entities.map((e) => loadProjects(e, { max: 300 }).catch(() => [])));
  projectCache = { at: Date.now(), list: lists.flat() };
  return projectCache.list;
}

function openPalette(initial = '', { projectsOnly = false } = {}) {
  const input = h('input.af-palette-input', { type: 'text', placeholder: projectsOnly ? 'Switch to project…' : 'Jump to a project, run or page…', value: initial, autofocus: true, 'aria-label': 'Search' });
  const list = h('div.af-palette-list', { role: 'listbox' });
  let items = [];
  let active = 0;
  let seq = 0;
  const m = modal({ title: 'Go to…', body: h('div.af-palette', input, list, h('div.af-palette-foot', h('span', h('kbd', '↑↓'), ' navigate'), h('span', h('kbd', '↵'), ' open'), h('span', h('kbd', 'esc'), ' close'))), size: 'lg' });
  m.el.classList.add('af-palette-modal');
  const pages = [
    ...NAV.flatMap((g) => g.items.map((i) => ({ label: i.label, sub: g.section, icon: i.icon, run: () => navigate(i.path) }))),
    ...(currentProject ? PROJECT_NAV.map((it) => ({ label: `${it.label} · ${currentProject.project}`, sub: 'This project', icon: it.icon,
      run: () => navigate(`/p/${encodeURIComponent(currentProject.entity)}/${encodeURIComponent(currentProject.project)}${it.sub}`) })) : []),
    { label: 'Keyboard shortcuts', sub: 'Help', icon: 'keyboard', run: showShortcuts },
  ];
  const draw = () => {
    mount(list, items.length ? items.map((it, i) => h('button.af-palette-item', {
      type: 'button', role: 'option', class: i === active && 'active',
      onmouseenter: () => { active = i; draw(); },
      onclick: () => { m.close(); it.run(); },
    }, icon(it.icon), h('span.af-palette-label', it.label), h('span.af-palette-sub', it.sub))) : h('div.af-palette-empty', 'No matches'));
    list.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
  };
  const search = async () => {
    const q = input.value.trim().toLowerCase();
    const mine = ++seq;
    const pageHits = projectsOnly ? [] : pages.filter((p) => !q || p.label.toLowerCase().includes(q));
    items = pageHits.slice(0, 8);
    active = 0;
    draw();
    try {
      const projects = await allProjects();
      if (mine !== seq) return;
      const projHits = projects.filter((p) => !q || `${p.entityName}/${p.name}`.toLowerCase().includes(q)).slice(0, 15)
        .map((p) => ({ label: p.name, sub: `${p.entityName} · ${p.totalRuns ?? 0} runs`, icon: 'folder2', run: () => navigate(urls.workspace(p.entityName, p.name)) }));
      items = projectsOnly ? projHits : [...projHits.slice(0, 8), ...pageHits];
      draw();
    } catch { /* pages still listed */ }
  };
  input.addEventListener('input', debounce(search, 120));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(active + 1, items.length - 1); draw(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(active - 1, 0); draw(); }
    if (e.key === 'Enter' && items[active]) { e.preventDefault(); m.close(); items[active].run(); }
  });
  search();
}

function showShortcuts() {
  const rows = [
    ['Ctrl/⌘ + K', 'Search projects and pages'],
    ['/', 'Focus the search box on the page'],
    ['?', 'This help'],
    ['w t s a r', 'Inside a project: Workspace, Runs, Sweeps, Artifacts, Reports'],
    ...NAV.flatMap((g) => g.items.filter((i) => i.key).map((i) => [i.key, `Go to ${i.label}`])),
  ];
  modal({
    title: 'Keyboard shortcuts', icon: 'keyboard',
    body: h('table.af-table.af-table-dense', h('tbody', rows.map(([k, d]) => h('tr', h('td.af-nowrap', k.split(' ').map((p) => (p === '+' ? ' + ' : h('kbd.me-1', p)))), h('td', d))))),
  });
}

// ------------------------------------------------------------------ keyboard

let pendingG = 0;
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); return; }
  const t = e.target;
  if (t.closest?.('input, textarea, select, [contenteditable=true], .CodeMirror') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (document.querySelector('.af-backdrop')) return;
  if (e.key === '/') {
    const s = document.querySelector('.af-page [data-search-shortcut]');
    if (s) { e.preventDefault(); s.focus(); s.select(); }
    return;
  }
  if (e.key === '?') { showShortcuts(); return; }
  if (e.key === 'g') { pendingG = Date.now(); return; }
  if (Date.now() - pendingG < 1200) {
    const item = NAV.flatMap((g) => g.items).find((i) => i.key === `g ${e.key}`);
    pendingG = 0;
    if (item) navigate(item.path);
    return;
  }
  if (currentProject) {
    const it = PROJECT_NAV.find((x) => x.key === e.key);
    if (it) navigate(`/p/${encodeURIComponent(currentProject.entity)}/${encodeURIComponent(currentProject.project)}${it.sub}`);
  }
});

startClock();
if (config.configured) start(outlet);
window.MS = { navigate, gql, config };
