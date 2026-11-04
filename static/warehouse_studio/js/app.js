// Warehouse Studio entry point: shell, navigation, context bar, command palette, shortcuts.

import { h, icon, mount, debounce } from '../../ui/js/dom.js';
import { route, start, navigate, onRouteChange, href, ROOT } from '../../ui/js/router.js';
import { api, onRequest } from './lib/api.js';
import { state, setSession, setContext, setTheme, on } from './lib/store.js';
import { popover, toast, errorView } from './lib/ui.js';
import * as fmt from './lib/fmt.js';

const page = (name) => () => import(`./pages/${name}.js`);

route('/', page('overview'), { title: 'Overview', icon: 'speedometer2' });
route('/worksheets', page('worksheets'), { title: 'Worksheets', icon: 'code-square', full: true });
route('/optimizer', page('optimizer'), { title: 'Performance & Optimizer', icon: 'lightning-charge' });
route('/cost', page('cost'), { title: 'Cost Center', icon: 'cash-coin' });
route('/warehouses', page('warehouses'), { title: 'Warehouses', icon: 'hdd-rack' });
route('/history', page('history'), { title: 'Query History', icon: 'clock-history' });
route('/queries/:queryId', page('history'), { title: 'Query', icon: 'search' });
route('/data', page('data'), { title: 'Data Explorer', icon: 'database' });
route('/procedures', page('procedures'), { title: 'Procedures', icon: 'braces-asterisk' });

export const NAV = [
  { section: 'Monitor', items: [
    { label: 'Overview', icon: 'speedometer2', path: '/', key: 'g o' },
    { label: 'Performance & Optimizer', icon: 'lightning-charge', path: '/optimizer', key: 'g p', badge: 'findings' },
    { label: 'Cost Center', icon: 'cash-coin', path: '/cost', key: 'g c' },
    { label: 'Warehouses', icon: 'hdd-rack', path: '/warehouses', key: 'g r' },
  ] },
  { section: 'Build', items: [
    { label: 'Worksheets', icon: 'code-square', path: '/worksheets', key: 'g w' },
    { label: 'Query History', icon: 'clock-history', path: '/history', key: 'g h' },
    { label: 'Data Explorer', icon: 'database', path: '/data', key: 'g d' },
    { label: 'Procedures', icon: 'braces-asterisk', path: '/procedures', key: 'g s' },
  ] },
];

// ------------------------------------------------------------------ layout

const root = document.getElementById('ws-root');
const applyTheme = () => {
  root.setAttribute('data-theme', state.theme === 'light' ? 'light' : 'dark');
  document.body.classList.toggle('ws-theme-dark', state.theme !== 'light');
};
applyTheme();
on('theme', applyTheme);

const navLinks = [];
const badgeEls = {};
const nav = h('nav.ws-nav', { 'aria-label': 'Warehouse Studio' }, NAV.map((group) => [
  h('div.ws-nav-section', group.section),
  group.items.map((item) => {
    const badge = item.badge ? h('span.ws-nav-badge', { style: { display: 'none' } }) : null;
    if (badge) badgeEls[item.badge] = badge;
    const a = h('a.ws-nav-link', { href: href(item.path), title: item.label }, icon(item.icon), h('span.ws-nav-label', item.label), badge || h('span.ws-nav-kbd', item.key));
    navLinks.push({ a, item });
    return a;
  }),
]));

const shell = h('div.ws-shell');
const themeBtn = h('button.ws-rail-btn', { type: 'button', onclick: () => setTheme(state.theme === 'light' ? 'dark' : 'light') });
const renderThemeBtn = () => mount(themeBtn, icon(state.theme === 'light' ? 'moon-stars' : 'sun'), h('span.ws-rail-foot-text', state.theme === 'light' ? 'Polar (dark)' : 'Glacier (light)'));
renderThemeBtn();
on('theme', renderThemeBtn);
const collapseBtn = h('button.ws-rail-btn', { type: 'button', onclick: () => shell.classList.toggle('ws-rail-collapsed') },
  icon('layout-sidebar-inset'), h('span.ws-rail-foot-text', 'Collapse'));
const connEl = h('div.ws-rail-btn', { style: { cursor: 'default' } }, h('span.ws-dot'), h('span.ws-rail-foot-text', 'Connecting…'));

const rail = h('aside.ws-rail',
  h('div.ws-brand', h('div.ws-brand-mark', icon('snow2')), h('div.ws-brand-text', h('div.ws-brand-name', 'Warehouse Studio'), h('div.ws-brand-sub', 'Snowflake · Afya DataHub'))),
  h('button.ws-rail-btn', { type: 'button', onclick: () => openPalette() }, icon('search'), h('span.ws-rail-foot-text', 'Search & commands'), h('span.ws-nav-kbd', '⌘K')),
  nav,
  h('div.ws-rail-foot', connEl, themeBtn, collapseBtn));

const titleEl = h('div.ws-topbar-title');
const ctxEl = h('div.ws-ctx');
const busyEl = h('span.ws-spinner', { style: { visibility: 'hidden' } });
const topbar = h('header.ws-topbar', titleEl, busyEl, ctxEl,
  h('button.ws-btn.ws-btn-sm', { type: 'button', onclick: () => openPalette(), title: 'Command palette' }, icon('command'), 'K'));
const outlet = h('main.ws-page');
shell.append(rail, h('div.ws-main', topbar, outlet));
mount(root, shell);

let inflight = 0;
onRequest(({ phase }) => {
  inflight += phase === 'start' ? 1 : -1;
  busyEl.style.visibility = inflight > 0 ? 'visible' : 'hidden';
});

onRouteChange((found) => {
  const path = found?.path || '/';
  for (const { a, item } of navLinks) {
    const active = item.path === '/' ? path === '/' : path.startsWith(item.path);
    a.classList.toggle('active', active);
  }
  const meta = found?.route?.meta || {};
  mount(titleEl, meta.icon && icon(meta.icon), meta.title || 'Warehouse Studio');
});

// ------------------------------------------------------------------ context bar

function renderContext() {
  const sess = state.session;
  const ctx = state.context;
  const wh = sess?.warehouses?.find((w) => w.name === ctx.warehouse);
  const whState = String(wh?.state || '').toUpperCase();
  const pill = (k, ic, value, onClick, extra) => h('button.ws-pill', { type: 'button', onclick: onClick, title: `${k}: ${value || 'default'}` },
    extra || icon(ic), h('span.ws-pill-k', k), h('b', value || 'default'), icon('chevron-down'));
  mount(ctxEl,
    pill('Role', 'person-badge', ctx.role, (e) => pickRole(e.currentTarget)),
    pill('Warehouse', 'cpu', ctx.warehouse ? `${ctx.warehouse}${wh?.size ? ` · ${wh.size}` : ''}` : '', (e) => pickWarehouse(e.currentTarget),
      h('span.ws-dot', { class: whState === 'STARTED' ? 'on' : whState === 'SUSPENDED' ? '' : whState ? 'warn' : '' })),
    pill('Database', 'database', [ctx.database, ctx.schema].filter(Boolean).join('.'), (e) => pickDatabase(e.currentTarget)));
}

function pickRole(anchor) {
  const roles = state.session?.available_roles || [];
  popover(anchor, { search: roles.length > 8, items: roles.map((r) => ({ label: r, value: r, icon: 'person-badge', on: r === state.context.role })),
    onPick: (i) => setContext({ role: i.value }) });
}

function pickWarehouse(anchor) {
  const whs = state.session?.warehouses || [];
  popover(anchor, { search: whs.length > 8, width: 320, items: whs.map((w) => ({ label: w.name, value: w.name, icon: 'cpu', hint: `${w.size || ''} · ${String(w.state || '').toLowerCase()}`, on: w.name === state.context.warehouse })),
    onPick: (i) => setContext({ warehouse: i.value }) });
}

async function pickDatabase(anchor) {
  try {
    const { databases } = await api.databases({ role: state.context.role });
    popover(anchor, { search: true, width: 320, items: databases.map((d) => ({ label: d.name, value: d.name, icon: 'database', hint: d.kind !== 'STANDARD' ? d.kind?.toLowerCase() : '', on: d.name === state.context.database })),
      onPick: async (i) => {
        setContext({ database: i.value, schema: '' });
        const { schemas } = await api.schemas(i.value, { role: state.context.role });
        popover(anchor, { search: true, width: 320, items: [{ head: `${i.value} · schema` }, ...schemas.map((sc) => ({ label: sc.name, value: sc.name, icon: 'folder2', on: sc.name === state.context.schema }))],
          onPick: (j) => setContext({ schema: j.value }) });
      } });
  } catch (err) { toast(err.message, 'error', { title: 'Could not list databases' }); }
}

on('context', renderContext);
on('session', renderContext);

// ------------------------------------------------------------------ session boot

async function loadSession(refresh = false) {
  try {
    const sess = await api.session({ role: state.context.role }, refresh);
    setSession(sess);
    mount(connEl, h('span.ws-dot.on'), h('span.ws-rail-foot-text', h('b', sess.account_name || sess.account_locator), h('span.ws-muted', ` · ${sess.region?.replace(/^AWS_|^AZURE_|^GCP_/, '').toLowerCase() || ''}`)));
    connEl.title = `${sess.user} @ ${sess.organization}-${sess.account_name} · Snowflake ${sess.version}`;
    refreshBadges();
    return sess;
  } catch (err) {
    mount(connEl, h('span.ws-dot.bad'), h('span.ws-rail-foot-text', 'Disconnected'));
    connEl.title = err.message;
    if (!state.session) {
      mount(outlet, errorView(err, () => { loadSession().then(() => start(outlet)); }));
    }
    throw err;
  }
}

async function refreshBadges() {
  if (!state.session?.toolkit?.toolkit_installed) return;
  try {
    const summary = await api.findingsSummary({ days: 7, severity: 'CRITICAL,HIGH' });
    const n = (summary.by_severity?.CRITICAL || 0) + (summary.by_severity?.HIGH || 0);
    const b = badgeEls.findings;
    if (b) { b.style.display = n ? '' : 'none'; b.textContent = n > 99 ? '99+' : String(n); b.title = `${n} critical/high findings in the last 7 days`; }
  } catch { /* badge is decoration */ }
}

on('context', debounce(() => { if (state.context.role !== state.session?.role) loadSession().catch(() => {}); }, 50));

// ------------------------------------------------------------------ command palette

const COMMANDS = () => [
  ...NAV.flatMap((g) => g.items.map((i) => ({ group: 'Go to', label: i.label, icon: i.icon, hint: i.key, run: () => navigate(i.path) }))),
  { group: 'Actions', label: 'New worksheet', icon: 'plus-lg', run: () => navigate('/worksheets?new=1') },
  { group: 'Actions', label: 'Run every optimizer detector', icon: 'lightning-charge', run: () => navigate('/optimizer?run=all') },
  { group: 'Actions', label: 'Show critical findings', icon: 'exclamation-octagon', run: () => navigate('/optimizer?severity=CRITICAL') },
  { group: 'Actions', label: 'Refresh cost data', icon: 'arrow-clockwise', run: async () => { await api.costRefresh(); toast('Cost caches cleared — pages reload fresh data.', 'success'); } },
  { group: 'Actions', label: `Switch to ${state.theme === 'light' ? 'dark' : 'light'} theme`, icon: 'circle-half', run: () => setTheme(state.theme === 'light' ? 'dark' : 'light') },
  { group: 'Actions', label: 'Reconnect to Snowflake', icon: 'plug', run: () => loadSession(true).then(() => toast('Session refreshed', 'success')) },
  ...(state.session?.warehouses || []).map((w) => ({ group: 'Use warehouse', label: w.name, icon: 'cpu', hint: w.size, run: () => setContext({ warehouse: w.name }) })),
];

let paletteOpen = false;
export function openPalette() {
  if (paletteOpen) return;
  paletteOpen = true;
  const input = h('input', { placeholder: 'Search pages, actions, warehouses, databases… or paste a query id', 'aria-label': 'Command' });
  const list = h('div.ws-palette-list');
  const back = h('div.ws-palette-back.ws-root', { 'data-theme': state.theme === 'light' ? 'light' : 'dark', style: { background: 'rgba(3,7,18,.55)', borderRadius: 0, minHeight: 0, overflow: 'visible', backgroundImage: 'none', margin: 0 },
    onmousedown: (e) => { if (e.target === back) close(); } },
  h('div.ws-palette', h('div.ws-palette-in', icon('search'), input), list,
    h('div.ws-palette-foot', h('span', h('kbd', '↑'), h('kbd', '↓'), ' navigate'), h('span', h('kbd', '↵'), ' run'), h('span', h('kbd', 'esc'), ' close'))));
  document.body.appendChild(back);
  let items = COMMANDS();
  let dbItems = [];
  let active = 0;
  let shown = [];
  api.databases({ role: state.context.role }).then(({ databases }) => {
    dbItems = databases.map((d) => ({ group: 'Explore database', label: d.name, icon: 'database', run: () => navigate(`/data?database=${encodeURIComponent(d.name)}`) }));
    render();
  }).catch(() => {});
  function render() {
    const q = input.value.trim().toLowerCase();
    const extra = [];
    if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(q)) {
      extra.push({ group: 'Query', label: `Inspect query ${q}`, icon: 'search', run: () => navigate(`/queries/${q}`) });
    }
    shown = [...extra, ...[...items, ...dbItems].filter((i) => !q || i.label.toLowerCase().includes(q) || i.group.toLowerCase().includes(q))].slice(0, 60);
    active = Math.min(active, Math.max(0, shown.length - 1));
    let group = null;
    mount(list, shown.length ? shown.flatMap((i, idx) => {
      const out = [];
      if (i.group !== group) { group = i.group; out.push(h('div.ws-palette-group', group)); }
      out.push(h(`div.ws-palette-item${idx === active ? '.on' : ''}`, { onclick: () => run(i), onmousemove: () => { if (active !== idx) { active = idx; render(); } } },
        icon(i.icon), h('span', i.label), i.hint && h('small', i.hint)));
      return out;
    }) : h('div.ws-palette-item', h('span.ws-muted', 'No matches')));
    list.querySelector('.on')?.scrollIntoView({ block: 'nearest' });
  }
  function run(i) { close(); Promise.resolve(i.run()).catch((e) => toast(e.message, 'error')); }
  function close() { paletteOpen = false; back.remove(); document.removeEventListener('keydown', onKey, true); }
  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(shown.length - 1, active + 1); render(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(0, active - 1); render(); }
    else if (e.key === 'Enter') { e.preventDefault(); if (shown[active]) run(shown[active]); }
  }
  input.addEventListener('input', () => { active = 0; render(); });
  document.addEventListener('keydown', onKey, true);
  render();
  setTimeout(() => input.focus(), 10);
}

// ------------------------------------------------------------------ shortcuts

let chord = null;
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); return; }
  const tag = (e.target.tagName || '').toLowerCase();
  if (['input', 'textarea', 'select'].includes(tag) || e.target.isContentEditable || e.target.closest?.('.CodeMirror')) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (chord === 'g') {
    const item = NAV.flatMap((g) => g.items).find((i) => i.key === `g ${e.key}`);
    chord = null;
    if (item) { e.preventDefault(); navigate(item.path); }
    return;
  }
  if (e.key === 'g') { chord = 'g'; setTimeout(() => { chord = null; }, 1200); }
  if (e.key === '?') { e.preventDefault(); openPalette(); }
});

// ------------------------------------------------------------------ go

renderContext();
loadSession().then(() => start(outlet)).catch(() => {});
setInterval(() => { if (!document.hidden) refreshBadges(); }, 120_000);
window.addEventListener('ws:findings-changed', refreshBadges);

export { ROOT, fmt };
