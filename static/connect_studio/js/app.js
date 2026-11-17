// Connect Studio entry point: shell, navigation, AI status, theme, command palette, shortcuts.

import { h, icon, mount, prefs } from '../../ui/js/dom.js';
import { route, start, navigate, onRouteChange, href } from '../../ui/js/router.js';
import { toast } from '../../ui/js/ui.js';
import { api } from './api.js';
import { logo } from './kit.js';

const page = (name) => () => import(`./pages/${name}.js`);

route('/', page('overview'), { title: 'Overview', nav: '/' });
route('/sources', page('sources'), { title: 'Sources', nav: '/sources' });
route('/sources/new', page('source_new'), { title: 'Add a source', nav: '/sources' });
route('/sources/:id', page('source_detail'), { title: 'Source', nav: '/sources' });
route('/mappings', page('mappings'), { title: 'Mappings', nav: '/mappings' });
route('/mappings/new', page('mapping_new'), { title: 'New mapping', nav: '/mappings' });
route('/mappings/:id', page('mapping_studio'), { title: 'Mapping studio', nav: '/mappings' });
route('/syncs', page('syncs'), { title: 'Syncs', nav: '/syncs' });
route('/syncs/:id', page('sync_detail'), { title: 'Sync', nav: '/syncs' });
route('/activity', page('activity'), { title: 'Activity', nav: '/activity' });

export const NAV = [
  { label: 'Overview', icon: 'grid-1x2', path: '/', key: 'g o' },
  { label: 'Sources', icon: 'database', path: '/sources', key: 'g s' },
  { label: 'Mappings', icon: 'stars', path: '/mappings', key: 'g m', badge: 'review' },
  { label: 'Syncs', icon: 'arrow-repeat', path: '/syncs', key: 'g y' },
  { label: 'Activity', icon: 'activity', path: '/activity', key: 'g a' },
];

// ------------------------------------------------------------------ theme

const root = document.getElementById('cs-root');
document.body.classList.add('cs-app');
let theme = prefs.get('cs:theme', 'light');
function applyTheme() {
  root.setAttribute('data-theme', theme);
  document.body.classList.toggle('cs-dark', theme === 'dark');
}
applyTheme();
export function toggleTheme() {
  theme = theme === 'dark' ? 'light' : 'dark';
  prefs.set('cs:theme', theme);
  applyTheme();
  mount(themeBtn, icon(theme === 'dark' ? 'sun' : 'moon-stars'));
}

// ------------------------------------------------------------------ shell

const navLinks = [];
const badges = {};
const nav = h('nav.cs-nav', { 'aria-label': 'Connect Studio' }, NAV.map((item) => {
  const badge = item.badge ? h('span.cs-nav-count', { style: { display: 'none' } }) : null;
  if (badge) badges[item.badge] = badge;
  const a = h('a', { href: href(item.path), title: `${item.label} (${item.key})` }, icon(item.icon), item.label, badge);
  navLinks.push({ a, item });
  return a;
}));
// "more" = tabs hidden past the right edge (narrow screens) → CSS fades that edge
const markMore = () => nav.classList.toggle('more', nav.scrollLeft + nav.clientWidth < nav.scrollWidth - 4);
nav.addEventListener('scroll', markMore, { passive: true });
window.addEventListener('resize', markMore);
requestAnimationFrame(markMore);
const aiChip = h('span.cs-ai-chip.off', { title: 'AI mapping provider' }, h('span.cs-ai-dot'), 'AI');
const themeBtn = h('button.cs-icon-btn', { type: 'button', title: 'Toggle dark mode', onclick: toggleTheme }, icon(theme === 'dark' ? 'sun' : 'moon-stars'));
const busy = h('span.spinner-border.spinner-border-sm.text-primary', { style: { visibility: 'hidden', width: '14px', height: '14px' } });
const appbar = h('header.cs-appbar',
  h('a.cs-brand', { href: href('/') }, h('span.cs-brand-mark', icon('bezier2')),
    h('span', h('div.cs-brand-name', 'Connect Studio'), h('div.cs-brand-sub', 'Connectors · dbatlas · Groq'))),
  nav,
  h('div.cs-appbar-tools', busy,
    h('button.cs-search-btn', { type: 'button', onclick: () => openPalette() }, icon('search'), h('span', 'Search or jump to…'), h('span.cs-kbd', '⌘K')),
    aiChip, themeBtn));
const outlet = h('main');
mount(root, appbar, outlet);

let inflight = 0;
import('./api.js').then(({ onRequest }) => onRequest((phase) => {
  inflight += phase === 'start' ? 1 : -1;
  busy.style.visibility = inflight > 0 ? 'visible' : 'hidden';
}));

onRouteChange((found) => {
  const nav = found?.route?.meta?.nav || '/';
  for (const { a, item } of navLinks) a.classList.toggle('active', item.path === nav);
  // on narrow screens the tab strip scrolls — keep the current tab visible
  const active = navLinks.find(({ a }) => a.classList.contains('active'))?.a;
  if (active && active.parentElement.scrollWidth > active.parentElement.clientWidth) {
    active.parentElement.scrollTo({ left: active.offsetLeft - 8, behavior: 'smooth' });
  }
  setTimeout(markMore, 400);
  outlet.firstElementChild?.classList.add('cs-page');
});

async function loadMeta() {
  try {
    const meta = await api.meta();
    if (meta.llm?.available) {
      aiChip.classList.remove('off');
      mount(aiChip, h('span.cs-ai-dot'), icon('stars'), meta.llm.model.replace(/^openai\//, ''));
      aiChip.title = `AI mapping on Groq · ${meta.llm.model}`;
    } else {
      mount(aiChip, h('span.cs-ai-dot'), 'AI off');
      aiChip.title = 'Set GROQ_API_KEY to enable AI mapping — rule-based matching still works.';
    }
  } catch { /* shown on pages */ }
}

export async function refreshBadges() {
  try {
    const { mappings } = await api.mappings();
    const n = mappings.filter((m) => m.status === 'review').length;
    const b = badges.review;
    b.style.display = n ? '' : 'none';
    b.textContent = String(n);
    b.title = `${n} mapping(s) in review`;
  } catch { /* decoration */ }
}
window.addEventListener('cs:changed', refreshBadges);

// ------------------------------------------------------------------ command palette

let paletteOpen = false;
export async function openPalette(initial = '') {
  if (paletteOpen) return;
  paletteOpen = true;
  const input = h('input', { placeholder: 'Jump to a source or mapping, add a connector, run an action…', value: initial, 'aria-label': 'Command' });
  const list = h('div.cs-palette-list');
  const back = h('div.cs-palette-back', { onmousedown: (e) => { if (e.target === back) close(); } },
    h('div.cs-palette', h('div.cs-palette-in', icon('search'), input), list,
      h('div.cs-palette-foot', h('span', h('span.cs-kbd', '↑'), ' ', h('span.cs-kbd', '↓'), ' navigate'), h('span', h('span.cs-kbd', '↵'), ' open'), h('span', h('span.cs-kbd', 'esc'), ' close'))));
  document.body.appendChild(back);
  let items = [
    ...NAV.map((i) => ({ group: 'Go to', label: i.label, icon: i.icon, hint: i.key, run: () => navigate(i.path) })),
    { group: 'Actions', label: 'Add a source', icon: 'plus-circle', run: () => navigate('/sources/new') },
    { group: 'Actions', label: 'New AI mapping', icon: 'stars', run: () => navigate('/mappings/new') },
    { group: 'Actions', label: 'New sync', icon: 'arrow-repeat', run: () => navigate('/syncs?new=1') },
    { group: 'Actions', label: 'Upload a target schema (schema.json / DDL)', icon: 'filetype-json', run: () => navigate('/sources/new?kind=schema_file') },
    { group: 'Actions', label: `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`, icon: 'circle-half', run: toggleTheme },
  ];
  let active = 0;
  let shown = [];
  const render = () => {
    const q = input.value.trim().toLowerCase();
    shown = items.filter((i) => !q || `${i.label} ${i.group} ${i.hint || ''}`.toLowerCase().includes(q)).slice(0, 80);
    active = Math.min(active, Math.max(0, shown.length - 1));
    let group = null;
    mount(list, shown.length ? shown.flatMap((i, idx) => {
      const out = [];
      if (i.group !== group) { group = i.group; out.push(h('div.cs-palette-group', group)); }
      out.push(h('div.cs-palette-item', { class: idx === active && 'on', onclick: () => run(i), onmousemove: () => { if (active !== idx) { active = idx; render(); } } },
        i.logo || icon(i.icon), h('span', i.label), i.hint && h('small', i.hint)));
      return out;
    }) : h('div.cs-palette-item', h('span.cs-muted', 'No matches')));
    list.querySelector('.on')?.scrollIntoView({ block: 'nearest' });
  };
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
  try {
    const [meta, { sources }, { mappings }] = await Promise.all([api.meta(), api.sources(), api.mappings()]);
    const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
    items = items.concat(
      mappings.map((m) => ({ group: 'Mappings', label: m.name, icon: 'stars', hint: m.status, run: () => navigate(`/mappings/${m.id}`) })),
      sources.map((s) => ({ group: 'Sources', label: s.name, logo: logo(specs[s.kind], { size: 'xs' }), hint: s.label, run: () => navigate(`/sources/${s.id}`) })),
      meta.connectors.map((c) => ({ group: 'Add connector', label: `Add ${c.label}`, logo: logo(c, { size: 'xs' }), hint: c.category, run: () => navigate(`/sources/new?kind=${c.key}`) })),
    );
    if (paletteOpen) render();
  } catch { /* static items still work */ }
}

// ------------------------------------------------------------------ shortcuts

let chord = null;
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); return; }
  const tag = (e.target.tagName || '').toLowerCase();
  if (['input', 'textarea', 'select'].includes(tag) || e.target.isContentEditable) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (chord === 'g') {
    chord = null;
    const item = NAV.find((i) => i.key === `g ${e.key}`);
    if (item) { e.preventDefault(); navigate(item.path); }
    return;
  }
  if (e.key === 'g') { chord = 'g'; setTimeout(() => { chord = null; }, 1200); }
});

// ------------------------------------------------------------------ go

loadMeta();
refreshBadges();
start(outlet);
