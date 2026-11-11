// Pipeline UI entry point: layout, navigation, health indicator,
// command palette (Ctrl/⌘+K) and keyboard shortcuts.

import { h, icon, mount, debounce, prefs } from './core/dom.js';
import { api, onRequest, config } from './core/api.js';
import { route, start, navigate, onRouteChange, href } from './core/router.js';
import { urls } from './core/urls.js';
import { modal, startClock, stateDot, toast } from './core/ui.js';

const page = (name) => () => import(`./pages/${name}.js`);

route('/', page('dashboard'), { title: 'Overview' });
route('/dags', page('dags'), { title: 'DAGs' });
route('/dags/:dagId', page('dag'), { title: 'DAG' });
route('/dags/:dagId/runs/:runId', page('run'), { title: 'DAG run' });
route('/dags/:dagId/runs/:runId/tasks/:taskId', page('task'), { title: 'Task instance' });
route('/dags/:dagId/runs/:runId/tasks/:taskId/:mapIndex', page('task'), { title: 'Task instance' });
route('/runs', page('runs'), { title: 'DAG runs' });
route('/task-instances', page('task_instances'), { title: 'Task instances' });
route('/backfills', page('backfills'), { title: 'Backfills' });
route('/assets', page('assets'), { title: 'Assets' });
route('/assets/:assetId', page('asset'), { title: 'Asset' });
route('/hitl', page('hitl'), { title: 'Required actions' });
route('/connections', page('connections'), { title: 'Connections' });
route('/variables', page('variables'), { title: 'Variables' });
route('/pools', page('pools'), { title: 'Pools' });
route('/xcoms', page('xcoms'), { title: 'XComs' });
route('/audit', page('event_logs'), { title: 'Audit log' });
route('/jobs', page('jobs'), { title: 'Jobs' });
route('/import-errors', page('import_errors'), { title: 'Import errors' });
route('/warnings', page('dag_warnings'), { title: 'DAG warnings' });
route('/plugins', page('plugins'), { title: 'Plugins' });
route('/providers', page('providers'), { title: 'Providers' });
route('/config', page('config'), { title: 'Configuration' });
route('/health', page('health'), { title: 'Health' });
route('/explorer', page('explorer'), { title: 'API explorer' });

export const NAV = [
  { section: 'Overview', items: [
    { label: 'Dashboard', icon: 'speedometer2', path: '/', key: 'g h' },
    { label: 'Health', icon: 'heart-pulse', path: '/health' },
  ] },
  { section: 'Pipelines', items: [
    { label: 'DAGs', icon: 'diagram-3', path: '/dags', key: 'g d' },
    { label: 'DAG runs', icon: 'play-circle', path: '/runs', key: 'g r' },
    { label: 'Task instances', icon: 'list-task', path: '/task-instances', key: 'g t' },
    { label: 'Backfills', icon: 'rewind-circle', path: '/backfills', key: 'g b' },
    { label: 'Assets', icon: 'database', path: '/assets', key: 'g a' },
    { label: 'Required actions', icon: 'person-check', path: '/hitl', badge: 'hitl' },
  ] },
  { section: 'Admin', items: [
    { label: 'Connections', icon: 'plug', path: '/connections', key: 'g c' },
    { label: 'Variables', icon: 'braces-asterisk', path: '/variables', key: 'g v' },
    { label: 'Pools', icon: 'water', path: '/pools', key: 'g p' },
    { label: 'XComs', icon: 'arrow-left-right', path: '/xcoms', key: 'g x' },
  ] },
  { section: 'Observe', items: [
    { label: 'Audit log', icon: 'journal-text', path: '/audit', key: 'g l' },
    { label: 'Jobs', icon: 'cpu', path: '/jobs', key: 'g j' },
    { label: 'Import errors', icon: 'bug', path: '/import-errors', badge: 'importErrors' },
    { label: 'DAG warnings', icon: 'exclamation-diamond', path: '/warnings', badge: 'warnings' },
  ] },
  { section: 'System', items: [
    { label: 'Plugins', icon: 'puzzle', path: '/plugins' },
    { label: 'Providers', icon: 'box-seam', path: '/providers' },
    { label: 'Configuration', icon: 'sliders', path: '/config' },
    { label: 'API explorer', icon: 'terminal', path: '/explorer', key: 'g e' },
  ] },
];

// ------------------------------------------------------------------ layout

const root = document.getElementById('af-app');
const badges = {};
const navLinks = [];

const navEl = h('nav.af-nav', { 'aria-label': 'Pipelines navigation' },
  NAV.map((group) => h('div.af-nav-group',
    h('div.af-nav-section', group.section),
    group.items.map((item) => {
      const badge = item.badge ? (badges[item.badge] = h('span.af-nav-badge.d-none')) : null;
      const a = h('a.af-nav-link', { href: href(item.path) }, icon(item.icon), h('span', item.label), badge);
      navLinks.push({ a, item });
      return a;
    }))));

const healthPill = h('a.af-health-pill', { href: urls.health(), title: 'Airflow component health' }, stateDot('none'), h('span', 'Checking…'));
const versionEl = h('span.af-version');
const netEl = h('span.af-net', { title: 'Requests in flight' });

const outlet = h('main.af-main');
const shell = h('div.af-shell',
  h('aside.af-side',
    h('div.af-side-head',
      h('div.af-brand', h('span.af-brand-logo', icon('wind')), h('div', h('div.af-brand-name', 'Airflow'), versionEl))),
    h('button.af-palette-btn', { type: 'button', onclick: () => openPalette() },
      icon('search'), h('span', 'Search or jump to…'), h('kbd', navigator.platform.includes('Mac') ? '⌘K' : 'Ctrl K')),
    navEl,
    h('div.af-side-foot', healthPill, netEl)),
  outlet);

const mobileToggle = h('button.af-mobile-nav.btn.btn-sm.btn-outline-secondary', {
  type: 'button', onclick: () => shell.classList.toggle('nav-open'),
}, icon('list'), ' Menu');
mount(root, mobileToggle, shell);

onRouteChange((found) => {
  shell.classList.remove('nav-open');
  const path = found?.path || '';
  let best = null;
  for (const l of navLinks) {
    const p = l.item.path;
    const hit = p === '/' ? path === '/' : path === p || path.startsWith(`${p}/`);
    if (hit && (!best || p.length > best.item.path.length)) best = l;
  }
  // DAG sub-pages belong to "DAGs"
  if (!best && path.startsWith('/dags')) best = navLinks.find((l) => l.item.path === '/dags');
  navLinks.forEach((l) => l.a.classList.toggle('active', l === best));
});

// ------------------------------------------------------------------ health + badges

async function refreshStatus() {
  try {
    const health = await api.get('/monitor/health');
    const parts = ['metadatabase', 'scheduler', 'triggerer', 'dag_processor'];
    const bad = parts.filter((p) => health[p] && health[p].status !== 'healthy');
    mount(healthPill, stateDot(bad.length ? 'failed' : 'success'), h('span', bad.length ? `${bad.length} unhealthy` : 'All systems healthy'));
    healthPill.classList.toggle('bad', !!bad.length);
  } catch {
    mount(healthPill, stateDot('failed'), h('span', 'Airflow unreachable'));
    healthPill.classList.add('bad');
  }
  const counts = await Promise.allSettled([
    api.get('/importErrors', { limit: 1 }),
    api.get('/dagWarnings', { limit: 1 }),
    api.get('/dags/~/dagRuns/~/hitlDetails', { limit: 1, response_received: false }),
  ]);
  const setBadge = (key, res, tone) => {
    const n = res.status === 'fulfilled' ? res.value.total_entries : 0;
    badges[key].textContent = n;
    badges[key].className = `af-nav-badge af-nav-badge-${tone}${n ? '' : ' d-none'}`;
  };
  setBadge('importErrors', counts[0], 'danger');
  setBadge('warnings', counts[1], 'warning');
  setBadge('hitl', counts[2], 'info');
}

api.get('/version').then((v) => mount(versionEl, `v${v.version}`)).catch(() => {});
refreshStatus();
setInterval(() => { if (!document.hidden) refreshStatus(); }, 60000);

let inflight = 0;
const origFetch = window.fetch;
window.fetch = (...args) => {
  inflight++;
  netEl.classList.add('busy');
  return origFetch(...args).finally(() => {
    inflight--;
    if (!inflight) netEl.classList.remove('busy');
  });
};
onRequest(({ status, method }) => {
  if (status === 401) toast('Your session expired. Reload the page to sign in again.', 'warning', { title: 'Signed out', timeout: 0 });
  if (method !== 'GET' && status < 400) refreshStatus();
});

// ------------------------------------------------------------------ command palette

let dagCache = null;
let dagCacheAt = 0;
async function loadDags() {
  if (dagCache && Date.now() - dagCacheAt < 60000) return dagCache;
  dagCache = await api.all('/dags', 'dags', { order_by: 'dag_id' });
  dagCacheAt = Date.now();
  return dagCache;
}

function fuzzy(text, q) {
  text = text.toLowerCase();
  q = q.toLowerCase();
  if (text.includes(q)) return 100 - text.indexOf(q);
  let ti = 0;
  for (const ch of q) {
    ti = text.indexOf(ch, ti);
    if (ti === -1) return -1;
    ti++;
  }
  return 10;
}

function openPalette(initial = '') {
  const input = h('input.af-palette-input', { type: 'text', placeholder: 'Jump to a DAG, page or action…', value: initial, autofocus: true, 'aria-label': 'Search' });
  const list = h('div.af-palette-list', { role: 'listbox' });
  let items = [];
  let active = 0;
  const m = modal({ title: 'Go to…', body: h('div.af-palette', input, list, h('div.af-palette-foot', h('span', h('kbd', '↑↓'), ' navigate'), h('span', h('kbd', '↵'), ' open'), h('span', h('kbd', 'esc'), ' close'))), size: 'lg' });
  m.el.classList.add('af-palette-modal');

  const staticItems = [
    ...NAV.flatMap((g) => g.items.map((i) => ({ label: i.label, sub: g.section, icon: i.icon, run: () => navigate(i.path) }))),
    { label: 'Failed runs (last 24h)', sub: 'Shortcut', icon: 'x-octagon', run: () => navigate(urls.runs({ state: 'failed', since: '24h' })) },
    { label: 'Running task instances', sub: 'Shortcut', icon: 'arrow-repeat', run: () => navigate(urls.taskInstances({ state: 'running' })) },
    { label: 'Paused DAGs', sub: 'Shortcut', icon: 'pause-circle', run: () => navigate(urls.dags({ paused: 'true' })) },
    { label: 'Keyboard shortcuts', sub: 'Help', icon: 'keyboard', run: showShortcuts },
  ];

  const draw = () => {
    mount(list, items.length ? items.map((it, i) => h('button.af-palette-item', {
      type: 'button', role: 'option', class: i === active && 'active',
      onmouseenter: () => { active = i; draw(); },
      onclick: () => { m.close(); it.run(); },
    }, icon(it.icon), h('span.af-palette-label', it.label), it.badge, h('span.af-palette-sub', it.sub))) : h('div.af-palette-empty', 'No matches'));
    list.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
  };
  const search = async () => {
    const q = input.value.trim();
    let dags = [];
    try { dags = await loadDags(); } catch { /* palette still works for pages */ }
    const dagItems = dags.map((d) => ({
      label: d.dag_display_name || d.dag_id, sub: d.is_paused ? 'DAG · paused' : 'DAG', icon: 'diagram-3',
      badge: d.has_import_errors ? h('span.badge.text-bg-danger.ms-2', 'import error') : null,
      run: () => navigate(urls.dag(d.dag_id)), text: `${d.dag_id} ${(d.tags || []).map((t) => t.name).join(' ')}`,
    }));
    const all = [...staticItems, ...dagItems];
    items = q
      ? all.map((it) => ({ it, score: fuzzy(it.text || it.label, q) })).filter((x) => x.score >= 0).sort((a, b) => b.score - a.score).slice(0, 40).map((x) => x.it)
      : [...staticItems.slice(0, 8), ...dagItems.slice(0, 12)];
    active = 0;
    draw();
  };
  input.addEventListener('input', debounce(search, 80));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(active + 1, items.length - 1); draw(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(active - 1, 0); draw(); }
    if (e.key === 'Enter' && items[active]) { e.preventDefault(); m.close(); items[active].run(); }
  });
  search();
}

function showShortcuts() {
  const rows = [
    ['Ctrl/⌘ + K', 'Command palette'],
    ['/', 'Focus the search box on the page'],
    ['r', 'Refresh the current page'],
    ['?', 'This help'],
    ...NAV.flatMap((g) => g.items.filter((i) => i.key).map((i) => [i.key, `Go to ${i.label}`])),
  ];
  modal({
    title: 'Keyboard shortcuts', icon: 'keyboard',
    body: h('table.af-table.af-table-dense', h('tbody', rows.map(([k, d]) => h('tr', h('td', k.split(' ').map((p) => h('kbd.me-1', p))), h('td', d))))),
  });
}

// ------------------------------------------------------------------ keyboard

let pendingG = 0;
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    openPalette();
    return;
  }
  const t = e.target;
  if (t.closest?.('input, textarea, select, [contenteditable=true]') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (document.querySelector('.af-backdrop')) return;
  if (e.key === '/') {
    const search = document.querySelector('.af-page [data-search-shortcut]');
    if (search) { e.preventDefault(); search.focus(); search.select(); }
    return;
  }
  if (e.key === '?') { showShortcuts(); return; }
  if (e.key === 'r') { document.querySelector('.af-page .af-refresh .btn')?.click(); return; }
  if (e.key === 'g') { pendingG = Date.now(); return; }
  if (Date.now() - pendingG < 1200) {
    const item = NAV.flatMap((g) => g.items).find((i) => i.key === `g ${e.key}`);
    pendingG = 0;
    if (item) navigate(item.path);
  }
});

window.addEventListener('af:session-expired', () => toast('Please reload and sign in again.', 'warning', { title: 'Session expired', timeout: 0 }));

if (prefs.get('nav:collapsed', false)) shell.classList.add('nav-collapsed');
startClock();
start(outlet);
window.AF = { navigate, api, config };
