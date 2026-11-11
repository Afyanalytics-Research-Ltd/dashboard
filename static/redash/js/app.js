// Analytics Studio entry point: layout, navigation (with favourites), health,
// command palette (Ctrl/⌘+K) and keyboard shortcuts.

import { h, icon, mount, debounce } from '../../ui/js/dom.js';
import { route, start, navigate, onRouteChange, href } from '../../ui/js/router.js';
import { modal, startClock, stateDot, toast } from '../../ui/js/ui.js';
import { rd, onRequest, config } from './core/api.js';
import { urls } from './core/urls.js';

const page = (name) => () => import(`./pages/${name}.js`);

route('/', page('home'), { title: 'Home' });
route('/queries', page('queries'), { title: 'Queries' });
route('/queries/new', page('query'), { title: 'New query' });
route('/queries/:queryId', page('query'), { title: 'Query' });
route('/dashboards', page('dashboards'), { title: 'Dashboards' });
route('/dashboards/:dashboardId', page('dashboard'), { title: 'Dashboard' });
route('/alerts', page('alerts'), { title: 'Alerts' });
route('/alerts/new', page('alert'), { title: 'New alert' });
route('/alerts/:alertId', page('alert'), { title: 'Alert' });
route('/data-sources', page('data_sources'), { title: 'Data sources' });
route('/data-sources/:dataSourceId', page('data_source'), { title: 'Data source' });
route('/destinations', page('destinations'), { title: 'Alert destinations' });
route('/users', page('users'), { title: 'Users' });
route('/users/:userId', page('user'), { title: 'User' });
route('/groups', page('groups'), { title: 'Groups' });
route('/groups/:groupId', page('group'), { title: 'Group' });
route('/snippets', page('snippets'), { title: 'Query snippets' });
route('/settings', page('settings'), { title: 'Settings' });
route('/events', page('events'), { title: 'Activity' });
route('/system', page('system'), { title: 'System status' });
route('/cleanup', page('cleanup'), { title: 'Cleanup' });
route('/explorer', page('explorer'), { title: 'API explorer' });

export const NAV = [
  { section: 'Analyze', items: [
    { label: 'Home', icon: 'house', path: '/', key: 'g h' },
    { label: 'Queries', icon: 'code-square', path: '/queries', key: 'g q' },
    { label: 'Dashboards', icon: 'grid-1x2', path: '/dashboards', key: 'g d' },
    { label: 'Alerts', icon: 'bell', path: '/alerts', key: 'g a' },
  ] },
  { section: 'Data', items: [
    { label: 'Data sources', icon: 'database', path: '/data-sources', key: 'g s' },
    { label: 'Query snippets', icon: 'scissors', path: '/snippets' },
    { label: 'Alert destinations', icon: 'send', path: '/destinations' },
  ] },
  { section: 'Admin', items: [
    { label: 'Users', icon: 'people', path: '/users', key: 'g u' },
    { label: 'Groups', icon: 'diagram-2', path: '/groups', key: 'g g' },
    { label: 'Settings', icon: 'sliders', path: '/settings' },
    { label: 'Activity', icon: 'activity', path: '/events', key: 'g e' },
    { label: 'System status', icon: 'heart-pulse', path: '/system' },
    { label: 'Cleanup', icon: 'recycle', path: '/cleanup' },
    { label: 'API explorer', icon: 'terminal', path: '/explorer', key: 'g x' },
  ] },
];

// ------------------------------------------------------------------ layout

const root = document.getElementById('af-app');
const navLinks = [];
const favsEl = h('div.af-nav-group.rd-nav-favs');

const navEl = h('nav.af-nav', { 'aria-label': 'Analytics Studio navigation' },
  h('div.rd-nav-new',
    h('a.btn.btn-sm.btn-primary', { href: urls.newQuery() }, icon('plus-lg', 'me-1'), 'New query'),
    h('button.btn.btn-sm.btn-outline-primary', { type: 'button', onclick: () => navigate(urls.dashboards({ new: 1 })) }, icon('grid-1x2', 'me-1'), 'Dashboard')),
  NAV.map((group, gi) => [
    h('div.af-nav-group',
      h('div.af-nav-section', group.section),
      group.items.map((item) => {
        const a = h('a.af-nav-link', { href: href(item.path) }, icon(item.icon), h('span', item.label));
        navLinks.push({ a, item });
        return a;
      })),
    gi === 0 ? favsEl : null,
  ]));

const healthPill = h('a.af-health-pill', { href: urls.system(), title: 'Redash status' }, stateDot('none'), h('span', 'Checking…'));
const versionEl = h('span.af-version');
const netEl = h('span.af-net', { title: 'Requests in flight' });
const outlet = h('main.af-main');

const shell = h('div.af-shell',
  h('aside.af-side',
    h('div.af-side-head',
      h('div.af-brand', h('span.af-brand-logo.rd-brand-logo', icon('bar-chart-line-fill')), h('div', h('div.af-brand-name', 'Analytics Studio'), versionEl))),
    h('button.af-palette-btn', { type: 'button', onclick: () => openPalette() },
      icon('search'), h('span', 'Search…'), h('kbd', navigator.platform.includes('Mac') ? '⌘K' : 'Ctrl K')),
    navEl,
    h('div.af-side-foot', healthPill, netEl)),
  outlet);

const mobileToggle = h('button.af-mobile-nav.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => shell.classList.toggle('nav-open') }, icon('list'), ' Menu');

if (!config.configured) {
  mount(root, h('div.af-error.m-3', icon('key', 'af-error-icon'), h('div',
    h('div.af-error-title', 'Redash is not connected'),
    h('div.af-error-msg', 'Set REDASH_ADMIN_API_KEY (an admin user\'s API key from Redash → Profile) and REDASH_BASE_URL in .env, then restart the dashboard.'))));
} else {
  mount(root, mobileToggle, shell);
}

onRouteChange((found) => {
  shell.classList.remove('nav-open');
  const path = found?.path || '';
  let best = null;
  for (const l of navLinks) {
    const p = l.item.path;
    const hit = p === '/' ? path === '/' : path === p || path.startsWith(`${p}/`);
    if (hit && (!best || p.length > best.item.path.length)) best = l;
  }
  navLinks.forEach((l) => l.a.classList.toggle('active', l === best));
  favsEl.querySelectorAll('a').forEach((a) => a.classList.toggle('active', a.getAttribute('href') === location.pathname));
});

// ------------------------------------------------------------------ favourites in nav

async function loadFavorites() {
  try {
    const [qs, ds] = await Promise.all([
      rd.get('/api/queries/favorites', { page_size: 8 }),
      rd.get('/api/dashboards/favorites', { page_size: 8 }),
    ]);
    const items = [
      ...(ds.results || []).map((d) => ({ label: d.name, icon: 'grid-1x2', href: urls.dashboard(d.id) })),
      ...(qs.results || []).map((q) => ({ label: q.name, icon: 'code-square', href: urls.query(q.id) })),
    ];
    mount(favsEl, h('div.af-nav-section', icon('star-fill', 'me-1 text-warning'), 'Favourites'),
      items.length
        ? items.map((it) => h('a.af-nav-link.rd-nav-fav', { href: it.href, title: it.label }, icon(it.icon), h('span', it.label)))
        : h('div.rd-nav-empty', 'Star a query or dashboard to pin it here.'));
  } catch {
    mount(favsEl);
  }
}
window.addEventListener('rd:favorites-changed', debounce(loadFavorites, 300));

// ------------------------------------------------------------------ health

async function refreshStatus() {
  try {
    const st = await rd.get('/api/admin/queries/rq_status');
    const workers = st.workers?.length || 0;
    const queued = Object.values(st.queues || {}).reduce((a, q) => a + (q.queued || 0), 0);
    const ok = workers > 0;
    mount(healthPill, stateDot(ok ? 'success' : 'failed'), h('span', ok ? `${workers} worker${workers > 1 ? 's' : ''} · ${queued} queued` : 'No workers running'));
    healthPill.classList.toggle('bad', !ok);
  } catch {
    mount(healthPill, stateDot('failed'), h('span', 'Redash unreachable'));
    healthPill.classList.add('bad');
  }
}

if (config.configured) {
  rd.get('/api/session').then((s) => mount(versionEl, `Redash v${s.client_config?.version || '?'}`)).catch(() => {});
  loadFavorites();
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
onRequest(({ status }) => {
  if (status === 401) toast('Your session expired. Reload the page to sign in again.', 'warning', { title: 'Signed out', timeout: 0 });
});

// ------------------------------------------------------------------ command palette

function openPalette(initial = '') {
  const input = h('input.af-palette-input', { type: 'text', placeholder: 'Search queries and dashboards, or jump to a page…', value: initial, autofocus: true, 'aria-label': 'Search' });
  const list = h('div.af-palette-list', { role: 'listbox' });
  let items = [];
  let active = 0;
  let seq = 0;
  const m = modal({ title: 'Go to…', body: h('div.af-palette', input, list, h('div.af-palette-foot', h('span', h('kbd', '↑↓'), ' navigate'), h('span', h('kbd', '↵'), ' open'), h('span', h('kbd', 'esc'), ' close'))), size: 'lg' });
  m.el.classList.add('af-palette-modal');

  const pages = [
    { label: 'New query', sub: 'Create', icon: 'plus-square', run: () => navigate(urls.newQuery()) },
    { label: 'New dashboard', sub: 'Create', icon: 'grid-1x2', run: () => navigate(urls.dashboards({ new: 1 })) },
    { label: 'New alert', sub: 'Create', icon: 'bell', run: () => navigate(urls.newAlert()) },
    ...NAV.flatMap((g) => g.items.map((i) => ({ label: i.label, sub: g.section, icon: i.icon, run: () => navigate(i.path) }))),
    { label: 'My queries', sub: 'Queries', icon: 'person', run: () => navigate(urls.queries({ view: 'my' })) },
    { label: 'Favourite queries', sub: 'Queries', icon: 'star', run: () => navigate(urls.queries({ view: 'favorites' })) },
    { label: 'Archived queries', sub: 'Queries', icon: 'archive', run: () => navigate(urls.queries({ view: 'archive' })) },
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
    const pageHits = pages.filter((p) => !q || p.label.toLowerCase().includes(q));
    items = q ? pageHits.slice(0, 6) : pageHits.slice(0, 10);
    active = 0;
    draw();
    if (q.length < 2) return;
    try {
      const [qs, ds] = await Promise.all([
        rd.get('/api/queries', { q, page_size: 10 }),
        rd.get('/api/dashboards', { q, page_size: 6 }),
      ]);
      if (mine !== seq) return;
      items = [
        ...(ds.results || []).map((d) => ({ label: d.name, sub: 'Dashboard', icon: 'grid-1x2', run: () => navigate(urls.dashboard(d.id)) })),
        ...(qs.results || []).map((x) => ({ label: x.name, sub: `Query #${x.id}`, icon: 'code-square', run: () => navigate(urls.query(x.id)) })),
        ...pageHits.slice(0, 4),
      ];
      draw();
    } catch { /* pages still listed */ }
  };
  input.addEventListener('input', debounce(search, 160));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(active + 1, items.length - 1); draw(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(active - 1, 0); draw(); }
    if (e.key === 'Enter' && items[active]) { e.preventDefault(); m.close(); items[active].run(); }
  });
  search();
}

function showShortcuts() {
  const rows = [
    ['Ctrl/⌘ + K', 'Search queries, dashboards and pages'],
    ['c', 'Create a new query'],
    ['/', 'Focus the search box on the page'],
    ['?', 'This help'],
    ['Ctrl/⌘ + Enter', 'Run the query (or the selected text)'],
    ['Ctrl/⌘ + S', 'Save the query'],
    ['Ctrl/⌘ + Shift + F', 'Format SQL'],
    ['Ctrl/⌘ + /', 'Comment / uncomment lines'],
    ['Ctrl + Space', 'Autocomplete tables, columns and keywords'],
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
    const search = document.querySelector('.af-page [data-search-shortcut]');
    if (search) { e.preventDefault(); search.focus(); search.select(); }
    return;
  }
  if (e.key === '?') { showShortcuts(); return; }
  if (e.key === 'c') { navigate(urls.newQuery()); return; }
  if (e.key === 'g') { pendingG = Date.now(); return; }
  if (Date.now() - pendingG < 1200) {
    const item = NAV.flatMap((g) => g.items).find((i) => i.key === `g ${e.key}`);
    pendingG = 0;
    if (item) navigate(item.path);
  }
});

startClock();
if (config.configured) start(outlet);
window.RD = { navigate, rd, config };
