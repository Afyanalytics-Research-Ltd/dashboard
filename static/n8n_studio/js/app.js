// Automation Studio (n8n) entry point: layout, navigation, health, command palette
// (Ctrl/⌘+K, searches inside workflow contents) and global keyboard shortcuts.

import { h, icon, mount, debounce } from '../../ui/js/dom.js';
import { route, start, navigate, onRouteChange, href } from '../../ui/js/router.js';
import { modal, startClock, toast } from '../../ui/js/ui.js';
import { n8n, studio, onRequest, config, n8nLink } from './core/api.js';
import { urls } from './core/urls.js';
import { load } from './core/store.js';
import { loadCatalog } from './core/catalog.js';
import { matchWorkflow, parseQuery } from './core/workflow.js';
import { rank } from './core/fuzzy.js';
import { MOD } from './core/widgets.js';

const page = (name) => () => import(`./pages/${name}.js`);

route('/', page('home'), { title: 'Overview' });
route('/workflows', page('workflows'), { title: 'Workflows' });
route('/workflows/new', page('workflow'), { title: 'New workflow' });
route('/workflows/:workflowId', page('workflow'), { title: 'Workflow' });
route('/executions', page('executions'), { title: 'Executions' });
route('/executions/:executionId', page('execution'), { title: 'Execution' });
route('/failures', page('failures'), { title: 'Failure inbox' });
route('/credentials', page('credentials'), { title: 'Credentials' });
route('/data-tables', page('datatables'), { title: 'Data tables' });
route('/data-tables/:tableId', page('datatable'), { title: 'Data table' });
route('/variables', page('variables'), { title: 'Variables' });
route('/tags', page('tags'), { title: 'Tags' });
route('/projects', page('projects'), { title: 'Projects' });
route('/projects/:projectId', page('project'), { title: 'Project' });
route('/users', page('users'), { title: 'Users' });
route('/roles', page('roles'), { title: 'Roles' });
route('/insights', page('insights'), { title: 'Insights' });
route('/audit', page('audit'), { title: 'Security audit' });
route('/packages', page('packages'), { title: 'Community packages' });
route('/log-streaming', page('logstreaming'), { title: 'Log streaming' });
route('/settings', page('settings'), { title: 'Instance settings' });
route('/backups', page('backups'), { title: 'Backups & migration' });
route('/history', page('history'), { title: 'Version history' });
route('/explorer', page('explorer'), { title: 'API explorer' });
route('/system', page('system'), { title: 'System status' });

export const NAV = [
  { section: 'Automate', items: [
    { label: 'Overview', icon: 'speedometer2', path: '/', key: 'g o' },
    { label: 'Workflows', icon: 'diagram-3', path: '/workflows', key: 'g w' },
    { label: 'Executions', icon: 'lightning-charge', path: '/executions', key: 'g e' },
    { label: 'Failure inbox', icon: 'inbox', path: '/failures', key: 'g f', badge: 'failures' },
  ] },
  { section: 'Resources', items: [
    { label: 'Credentials', icon: 'key', path: '/credentials', key: 'g c' },
    { label: 'Data tables', icon: 'table', path: '/data-tables', key: 'g d' },
    { label: 'Variables', icon: 'braces-asterisk', path: '/variables', key: 'g v' },
    { label: 'Tags', icon: 'tags', path: '/tags', key: 'g t' },
    { label: 'Projects', icon: 'kanban', path: '/projects', key: 'g p' },
  ] },
  { section: 'Insight', items: [
    { label: 'Insights', icon: 'graph-up-arrow', path: '/insights', key: 'g i' },
    { label: 'Security audit', icon: 'shield-check', path: '/audit', key: 'g a' },
    { label: 'Version history', icon: 'clock-history', path: '/history', key: 'g h' },
  ] },
  { section: 'Admin', items: [
    { label: 'Users', icon: 'people', path: '/users', key: 'g u' },
    { label: 'Roles', icon: 'person-badge', path: '/roles' },
    { label: 'Community packages', icon: 'box-seam', path: '/packages' },
    { label: 'Log streaming', icon: 'broadcast-pin', path: '/log-streaming' },
    { label: 'Instance settings', icon: 'sliders', path: '/settings', key: 'g s' },
    { label: 'Backups & migration', icon: 'cloud-arrow-down', path: '/backups', key: 'g b' },
  ] },
  { section: 'Developer', items: [
    { label: 'API explorer', icon: 'terminal', path: '/explorer', key: 'g x' },
    { label: 'System status', icon: 'heart-pulse', path: '/system' },
  ] },
];

// ------------------------------------------------------------------ layout

const root = document.getElementById('af-app');
const navLinks = [];
const failBadge = h('span.n8s-nav-badge.d-none');

const navEl = h('nav.af-nav', { 'aria-label': 'Automation Studio navigation' },
  h('div.n8s-nav-new',
    h('a.btn.btn-sm.btn-primary', { href: urls.newWorkflow() }, icon('plus-lg', 'me-1'), 'New workflow'),
    h('button.btn.btn-sm.btn-outline-primary', { type: 'button', title: 'Import a workflow (JSON, file, URL or package)', onclick: () => navigate(urls.workflows({ import: 1 })) }, icon('upload'))),
  NAV.map((group) => h('div.af-nav-group',
    h('div.af-nav-section', group.section),
    group.items.map((item) => {
      const a = h('a.af-nav-link', { href: href(item.path) }, icon(item.icon), h('span', item.label), item.badge === 'failures' ? failBadge : null);
      navLinks.push({ a, item });
      return a;
    }))));

const healthPill = h('a.af-health-pill', { href: urls.system(), title: 'n8n status' }, h('span.n8s-dot.n8s-st-unknown'), h('span', 'Checking…'));
const versionEl = h('span.af-version', `n8n API ${config.specVersion || ''} · ${config.operations || 0} ops`);
const netEl = h('span.af-net', { title: 'Requests in flight' });
const outlet = h('main.af-main');

const shell = h('div.af-shell.n8s-shell',
  h('aside.af-side',
    h('div.af-side-head',
      h('div.af-brand', h('span.af-brand-logo.n8s-brand-logo', icon('diagram-3-fill')), h('div', h('div.af-brand-name', 'Automation Studio'), versionEl))),
    h('button.af-palette-btn', { type: 'button', onclick: () => openPalette() }, icon('search'), h('span', 'Search everything…'), h('kbd', `${MOD} K`)),
    navEl,
    h('div.af-side-foot', healthPill, netEl)),
  outlet);

const mobileToggle = h('button.af-mobile-nav.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => shell.classList.toggle('nav-open') }, icon('list'), ' Menu');

if (!config.configured) {
  mount(root, h('div.n8s-setup',
    h('div.n8s-setup-art', icon('plug')),
    h('h2', 'Connect Automation Studio to n8n'),
    h('ol',
      h('li', 'In n8n open ', h('b', 'Settings → n8n API'), ' and create an API key (all scopes).'),
      h('li', 'Add it to the dashboard ', h('code', '.env'), ': ', h('code', 'N8N_API_KEY=…'), ' and point ', h('code', 'N8N_API_URL'), ' at n8n (inside docker: ', h('code', 'http://n8n-container:5678'), ').'),
      h('li', 'Optional: ', h('code', 'N8N_STUDIO_EMAIL'), ' / ', h('code', 'N8N_STUDIO_PASSWORD'), ' enable manual runs and the live node catalog.'),
      h('li', 'Restart the dashboard.')),
    config.n8nUrl && h('a.btn.btn-primary', { href: n8nLink('/settings/api'), target: '_blank', rel: 'noopener' }, icon('box-arrow-up-right', 'me-1'), 'Open n8n API settings')));
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
  document.body.classList.toggle('n8s-editor-open', /^\/workflows\/[^/]+$/.test(path));
});

// ------------------------------------------------------------------ health & failure badge

async function refreshStatus() {
  try {
    const st = await studio.get('/health/');
    const ok = st.instance?.ok && st.key?.ok;
    mount(healthPill, h('span.n8s-dot', { class: ok ? 'n8s-st-success' : 'n8s-st-error' }),
      h('span', ok ? `n8n online · ${st.key.ms ?? '?'} ms` : !st.instance?.reachable ? 'n8n unreachable' : st.key?.ok === false ? 'API key rejected' : 'n8n unhealthy'));
    healthPill.classList.toggle('bad', !ok);
  } catch {
    mount(healthPill, h('span.n8s-dot.n8s-st-error'), h('span', 'Status unavailable'));
    healthPill.classList.add('bad');
  }
}

async function refreshFailures() {
  try {
    const page1 = await n8n.get('/executions', { status: 'error', limit: 100 });
    const since = Date.now() - 24 * 3600 * 1000;
    const n = (page1.data || []).filter((e) => new Date(e.startedAt || e.createdAt).getTime() > since).length;
    mount(failBadge, n >= 100 ? '99+' : String(n));
    failBadge.classList.toggle('d-none', !n);
    failBadge.title = `${n} failed execution${n === 1 ? '' : 's'} in the last 24 hours`;
  } catch { /* badge stays hidden */ }
}

let inflight = 0;
onRequest((ev) => {
  if (ev.phase === 'start') inflight++;
  if (ev.phase === 'end') inflight = Math.max(0, inflight - 1);
  netEl.classList.toggle('busy', inflight > 0);
});
window.addEventListener('n8n:session-expired', debounce(() => toast('Your dashboard session expired. Reload the page to sign in again.', 'warning', { title: 'Signed out', timeout: 0 }), 500));

// ------------------------------------------------------------------ command palette

const ACTIONS = [
  { label: 'New workflow', sub: 'Create', icon: 'plus-square', run: () => navigate(urls.newWorkflow()) },
  { label: 'Import workflow', sub: 'Create', icon: 'upload', run: () => navigate(urls.workflows({ import: 1 })) },
  { label: 'New credential', sub: 'Create', icon: 'key', run: () => navigate(urls.credentials({ create: 1 })) },
  { label: 'New data table', sub: 'Create', icon: 'table', run: () => navigate(urls.dataTables()) },
  { label: 'Failed executions (24h)', sub: 'Executions', icon: 'x-octagon', run: () => navigate(urls.executions({ status: 'error', since: '24h' })) },
  { label: 'Running executions', sub: 'Executions', icon: 'arrow-repeat', run: () => navigate(urls.executions({ status: 'running' })) },
  { label: 'Back up all workflows now', sub: 'Backups', icon: 'cloud-arrow-down', run: () => navigate(urls.backups({ run: 1 })) },
  { label: 'Run security audit', sub: 'Audit', icon: 'shield-check', run: () => navigate(urls.audit()) },
  { label: 'Open n8n editor', sub: 'External', icon: 'box-arrow-up-right', run: () => window.open(n8nLink('/'), '_blank', 'noopener') },
  { label: 'Keyboard shortcuts', sub: 'Help', icon: 'keyboard', run: () => showShortcuts() },
];

function openPalette(initial = '') {
  const input = h('input.af-palette-input', { type: 'text', placeholder: 'Workflows (names and contents), credentials, pages, actions…  try  type:slack  or  cred:prod', value: initial, autofocus: true, 'aria-label': 'Search' });
  const list = h('div.af-palette-list', { role: 'listbox' });
  let items = [];
  let active = 0;
  let seq = 0;
  const m = modal({ title: 'Go to…', body: h('div.af-palette', input, list, h('div.af-palette-foot', h('span', h('kbd', '↑↓'), ' navigate'), h('span', h('kbd', '↵'), ' open'), h('span', h('kbd', 'esc'), ' close'), h('span.ms-auto', 'Searches inside nodes, parameters and notes'))), size: 'lg' });
  m.el.classList.add('af-palette-modal');
  const pages = NAV.flatMap((g) => g.items.map((i) => ({ label: i.label, sub: g.section, icon: i.icon, run: () => navigate(i.path) })));
  const draw = () => {
    mount(list, items.length ? items.map((it, i) => h('button.af-palette-item', {
      type: 'button', role: 'option', class: i === active && 'active',
      onmouseenter: () => { if (active !== i) { active = i; draw(); } },
      onclick: () => { m.close(); it.run(); },
    }, icon(it.icon), h('span.af-palette-label', it.label), it.hint && h('span.n8s-pal-hint', it.hint), h('span.af-palette-sub', it.sub))) : h('div.af-palette-empty', 'No matches'));
    list.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
  };
  const search = async () => {
    const q = input.value.trim();
    const mine = ++seq;
    const statics = rank([...ACTIONS, ...pages], q, (x) => [x.label, x.sub], q ? 6 : 14);
    items = statics;
    active = 0;
    draw();
    if (!q) return;
    try {
      const [wfs, creds] = await Promise.all([load('workflows'), load('credentials').catch(() => [])]);
      if (mine !== seq) return;
      const terms = parseQuery(q);
      const byName = rank(wfs, q, (w) => [w.name], 8);
      const deep = wfs.filter((w) => !byName.includes(w) && matchWorkflow(w, terms)).slice(0, 8);
      const credHits = rank(creds, q, (c) => [c.name, c.type], 5);
      items = [
        ...byName.map((w) => ({ label: w.name, sub: w.active ? 'Workflow · active' : 'Workflow', icon: 'diagram-3', run: () => navigate(urls.workflow(w.id)) })),
        ...deep.map((w) => ({ label: w.name, sub: 'Workflow · content match', hint: hitHint(w, q), icon: 'search', run: () => navigate(urls.workflow(w.id, { find: q })) })),
        ...credHits.map((c) => ({ label: c.name, sub: `Credential · ${c.type}`, icon: 'key', run: () => navigate(urls.credentials({ focus: c.id })) })),
        ...statics,
      ];
      if (/^\d+$/.test(q)) items.unshift({ label: `Execution #${q}`, sub: 'Open execution', icon: 'lightning-charge', run: () => navigate(urls.execution(q)) });
      draw();
    } catch { /* static results still listed */ }
  };
  input.addEventListener('input', debounce(search, 140));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(active + 1, items.length - 1); draw(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(active - 1, 0); draw(); }
    if (e.key === 'Enter' && items[active]) { e.preventDefault(); m.close(); items[active].run(); }
  });
  search();
}

function hitHint(wf, q) {
  const needle = q.replace(/^\w+:/, '').replace(/"/g, '').toLowerCase();
  const n = (wf.nodes || []).find((x) => JSON.stringify(x).toLowerCase().includes(needle));
  return n ? `in “${n.name}”` : '';
}

export function showShortcuts() {
  const rows = [
    [`${MOD} K`, 'Search workflows (incl. node contents), credentials, pages'],
    ['n', 'New workflow'],
    ['/', 'Focus the search box on the page'],
    ['?', 'This help'],
    ...NAV.flatMap((g) => g.items.filter((i) => i.key).map((i) => [i.key, `Go to ${i.label}`])),
    ['—', 'Workflow editor'],
    [`${MOD} S`, 'Save'],
    [`${MOD} Enter`, 'Run workflow'],
    [`${MOD} Z / ${MOD} Shift Z`, 'Undo / redo'],
    [`${MOD} C / ${MOD} V`, 'Copy / paste nodes (compatible with the n8n editor)'],
    [`${MOD} D`, 'Duplicate selection'],
    [`${MOD} A`, 'Select all'],
    ['Tab', 'Add node'],
    ['Enter', 'Open selected node'],
    ['F2', 'Rename selected node'],
    ['d', 'Disable / enable selected nodes'],
    ['p', 'Pin / unpin selected node output'],
    ['Delete', 'Delete selection'],
    ['1 / 0', 'Fit view / reset zoom'],
    ['+ / -', 'Zoom'],
    [`Shift L`, 'Tidy up (auto-layout)'],
    ['Space + drag', 'Pan'],
    ['Shift + drag', 'Box select'],
  ];
  modal({
    title: 'Keyboard shortcuts', icon: 'keyboard', size: 'lg',
    body: h('table.af-table.af-table-dense', h('tbody', rows.map(([k, d]) => (k === '—'
      ? h('tr', h('td.fw-semibold.pt-3', { colspan: 2 }, d))
      : h('tr', h('td.af-nowrap', k.split(' ').map((p) => h('kbd.me-1', p))), h('td', d)))))),
  });
}

// ------------------------------------------------------------------ keyboard

let pendingG = 0;
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); return; }
  const t = e.target;
  if (t.closest?.('input, textarea, select, [contenteditable=true]') || e.metaKey || e.ctrlKey || e.altKey) return;
  if (document.querySelector('.af-backdrop')) return;
  if (document.body.classList.contains('n8s-editor-open') && !['?', 'g'].includes(e.key) && Date.now() - pendingG > 1200) return; // editor owns its keys
  if (e.key === '/') {
    const search = document.querySelector('.af-page [data-search-shortcut]');
    if (search) { e.preventDefault(); search.focus(); search.select(); }
    return;
  }
  if (e.key === '?') { showShortcuts(); return; }
  if (e.key === 'n') { navigate(urls.newWorkflow()); return; }
  if (e.key === 'g') { pendingG = Date.now(); return; }
  if (Date.now() - pendingG < 1200) {
    const item = NAV.flatMap((g) => g.items).find((i) => i.key === `g ${e.key}`);
    pendingG = 0;
    if (item) { e.preventDefault(); navigate(item.path); }
  }
});

startClock();
if (config.configured) {
  loadCatalog().catch((err) => console.warn('Node catalog unavailable', err));
  refreshStatus();
  refreshFailures();
  setInterval(() => { if (!document.hidden) { refreshStatus(); refreshFailures(); } }, 60000);
  start(outlet);
}
window.N8S = { navigate, n8n, studio, config };
