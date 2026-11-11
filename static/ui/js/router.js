// History-API router. Pages are lazily imported ES modules exporting
//   export default async function render(ctx) { ... }
// ctx = { el, params, query, signal, cleanup(fn), every(ms, fn), setQuery(obj), navigate(url), setTitle(t) }

import { mount } from './dom.js';
import { errorState, skeleton, closeMenu } from './ui.js';
import { hideTip } from './charts.js';
// Each app's template sets window.APP_UI (or the legacy window.AIRFLOW_UI) with at least { root }.
const config = window.APP_UI || window.AIRFLOW_UI || {};

export const ROOT = (config.root || '/pipelines/').replace(/\/$/, '');

/** App-internal path ('/dags/x') -> full URL ('/pipelines/dags/x'). */
export const href = (path = '/', query) => {
  const q = query ? new URLSearchParams(Object.entries(query).filter(([, v]) => v != null && v !== '')).toString() : '';
  return `${ROOT}${path === '/' ? '/' : path}${q ? `?${q}` : ''}`;
};

const routes = [];
let current = null;
let outlet = null;
let onChangeHandlers = [];

export function route(pattern, loader, meta = {}) {
  const keys = [];
  const regex = new RegExp(`^${pattern.replace(/\/:([a-zA-Z]+)/g, (_, k) => { keys.push(k); return '/([^/]+)'; })}/?$`);
  routes.push({ pattern, regex, keys, loader, meta });
}

export const onRouteChange = (fn) => onChangeHandlers.push(fn);

function match(pathname) {
  let path = pathname.startsWith(ROOT) ? pathname.slice(ROOT.length) : pathname;
  if (!path || path === '') path = '/';
  for (const r of routes) {
    const m = path.match(r.regex);
    if (m) {
      const params = {};
      r.keys.forEach((k, i) => { params[k] = decodeURIComponent(m[i + 1]); });
      return { route: r, params, path };
    }
  }
  return null;
}

export function navigate(url, { replace = false } = {}) {
  const target = url.startsWith(ROOT) || /^https?:/.test(url) ? url : href(url);
  if (replace) history.replaceState(null, '', target);
  else history.pushState(null, '', target);
  render();
}

/** Update the query string of the current page without re-rendering it. */
export function setQuery(values, { replace = true } = {}) {
  const sp = new URLSearchParams(location.search);
  for (const [k, v] of Object.entries(values)) {
    if (v == null || v === '' || (Array.isArray(v) && !v.length)) sp.delete(k);
    else if (Array.isArray(v)) { sp.delete(k); v.forEach((x) => sp.append(k, x)); } else sp.set(k, v);
  }
  const qs = sp.toString();
  const url = `${location.pathname}${qs ? `?${qs}` : ''}`;
  if (replace) history.replaceState(null, '', url); else history.pushState(null, '', url);
}

export async function render() {
  closeMenu();
  hideTip();
  if (current) {
    current.controller.abort();
    current.cleanups.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
  }
  const found = match(location.pathname);
  const controller = new AbortController();
  const cleanups = [];
  const page = { controller, cleanups };
  current = page;
  const el = document.createElement('div');
  el.className = 'af-page';
  mount(outlet, el);
  onChangeHandlers.forEach((fn) => fn(found));

  if (!found) {
    mount(el, errorState({ status: 404, message: `No page at ${location.pathname}` }));
    return;
  }
  mount(el, skeleton(8));
  const ctx = {
    el,
    params: found.params,
    query: new URLSearchParams(location.search),
    signal: controller.signal,
    cleanup: (fn) => cleanups.push(fn),
    every(ms, fn) {
      const id = setInterval(() => { if (!document.hidden) fn(); }, ms);
      cleanups.push(() => clearInterval(id));
    },
    setQuery,
    navigate,
    setTitle: (t) => { document.title = `${t} — ${config.appName || 'Pipelines'} — Afya DataHub`; },
    get active() { return current === page; },
  };
  ctx.setTitle(found.route.meta.title || 'Pipelines');
  try {
    const mod = await found.route.loader();
    if (current !== page) return;
    mount(el);
    await mod.default(ctx);
  } catch (err) {
    if (err?.name === 'AbortError' || current !== page) return;
    console.error(err);
    mount(el, errorState(err, () => render()));
  }
  if (!location.hash) window.scrollTo({ top: 0 });
}

export function start(outletEl) {
  outlet = outletEl;
  window.addEventListener('popstate', render);
  // Intercept internal links: <a href="/pipelines/..."> (not new-tab clicks, not downloads/external).
  document.addEventListener('click', (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest('a[href]');
    if (!a || a.target || a.hasAttribute('download')) return;
    const url = new URL(a.href, location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith(`${ROOT}/`) && url.pathname !== ROOT) return;
    if (url.pathname.startsWith(`${ROOT}/api/`)) return;
    e.preventDefault();
    if (url.pathname === location.pathname && url.search === location.search && url.hash) { location.hash = url.hash; return; }
    navigate(url.pathname + url.search);
  });
  render();
}
