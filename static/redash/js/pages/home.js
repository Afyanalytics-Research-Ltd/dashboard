// Home: greeting, instant search, org counters, health, favourites, recent work,
// activity feed, data sources and a getting-started checklist.

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import {
  card, statCard, timeAgo, refreshControl, emptyState, errorState, skeleton, stateDot,
} from '../../../ui/js/ui.js';
import { fmtNumber } from '../../../ui/js/format.js';
import { rd, config } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../../../ui/js/router.js';
import { favoriteStar, getDataSources, getSession, dataSourceIcon, scheduleText } from '../core/common.js';
import { eventSentence, actionInfo, greeting, countUp, NOISE_ACTIONS } from './home_shared.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  const statsEl = h('div.af-stats', skeleton(1, { height: 70 }));
  const healthEl = h('div.rdh-health', skeleton(1, { height: 44 }));
  const favDashEl = h('div', skeleton(4));
  const favQueryEl = h('div', skeleton(4));
  const recentEl = h('div', skeleton(5));
  const myDashEl = h('div', skeleton(4));
  const feedEl = h('div', skeleton(6));
  const dsEl = h('div', skeleton(3));
  const checklistEl = h('div');
  const nameEl = h('span', config.user || '');

  const refresh = refreshControl(() => load(), { key: 'studio-home', initial: 60 });
  ctx.cleanup(refresh.stop);

  mount(el,
    h('section.rdh-hero',
      h('div.rdh-hero-text',
        h('h1.rdh-hello', `${greeting()}, `, nameEl),
        h('p.rdh-sub', 'Ask a question of your data, build a dashboard, or pick up where you left off.')),
      h('div.rdh-hero-actions',
        h('a.btn.btn-primary', { href: urls.newQuery() }, icon('plus-lg', 'me-1'), 'New query'),
        h('a.btn.btn-outline-primary', { href: urls.dashboards({ new: 1 }) }, icon('grid-1x2', 'me-1'), 'New dashboard'),
        h('a.btn.btn-outline-secondary', { href: urls.newAlert() }, icon('bell', 'me-1'), 'New alert'),
        refresh.el),
      instantSearch(signal)),
    checklistEl,
    statsEl,
    healthEl,
    h('div.af-grid-2',
      card({ title: 'Favourite dashboards', icon: 'star', body: favDashEl, flush: true, actions: h('a.af-link-btn', { href: urls.dashboards({ view: 'favorites' }) }, 'All →') }),
      card({ title: 'Favourite queries', icon: 'star', body: favQueryEl, flush: true, actions: h('a.af-link-btn', { href: urls.queries({ view: 'favorites' }) }, 'All →') })),
    h('div.af-grid-2',
      card({ title: 'Recently edited queries', icon: 'clock-history', body: recentEl, flush: true, actions: h('a.af-link-btn', { href: urls.queries({ view: 'my' }) }, 'My queries →') }),
      card({ title: 'My dashboards', icon: 'grid-1x2', body: myDashEl, flush: true, actions: h('a.af-link-btn', { href: urls.dashboards() }, 'All dashboards →') })),
    h('div.af-grid-main-side',
      card({ title: 'Recent activity', icon: 'activity', body: feedEl, flush: true, actions: h('a.af-link-btn', { href: urls.events() }, 'Full activity log →') }),
      card({ title: 'Data sources', icon: 'database', body: dsEl, flush: true, actions: h('a.af-link-btn', { href: urls.dataSources() }, 'Manage →') })));

  getSession().then((s) => { if (ctx.active && s?.user?.name) mount(nameEl, s.user.name); }).catch(() => {});

  async function load() {
    const [status, favD, favQ, recent, myD, events, ds, rq, outdated] = await Promise.allSettled([
      rd.get('/api/organization/status', null, { signal }),
      rd.get('/api/dashboards/favorites', { page_size: 8 }, { signal }),
      rd.get('/api/queries/favorites', { page_size: 8 }, { signal }),
      rd.get('/api/queries/recent', null, { signal }),
      rd.get('/api/dashboards/my', { page_size: 8, order: '-created_at' }, { signal }),
      rd.get('/api/events', { page: 1, page_size: 60 }, { signal }),
      getDataSources(),
      rd.get('/api/admin/queries/rq_status', null, { signal }),
      rd.get('/api/admin/queries/outdated', null, { signal }),
    ]);
    if (!ctx.active) return;
    const val = (r) => (r.status === 'fulfilled' ? r.value : null);
    const counters = val(status)?.object_counters || {};
    renderStats(counters, status.reason);
    renderHealth(val(rq), val(outdated), val(ds));
    renderChecklist(counters);
    renderDashList(favDashEl, val(favD)?.results, favD.reason, { icon: 'star', title: 'No favourite dashboards yet', message: 'Click the ☆ next to any dashboard to keep it one click away.' });
    renderQueryList(favQueryEl, val(favQ)?.results, favQ.reason, { icon: 'star', title: 'No favourite queries yet', message: 'Star the queries you open most often.' });
    renderQueryList(recentEl, val(recent), recent.reason, { icon: 'clock-history', title: 'Nothing edited recently', message: 'Queries you change appear here.', action: h('a.btn.btn-sm.btn-primary', { href: urls.newQuery() }, 'Write a query') });
    renderDashList(myDashEl, val(myD)?.results, myD.reason, { icon: 'grid-1x2', title: 'You have no dashboards', message: 'Combine visualizations from several queries on one page.', action: h('a.btn.btn-sm.btn-primary', { href: urls.dashboards({ new: 1 }) }, 'Create a dashboard') });
    const feed = val(events)?.results;
    renderFeed(feed && feed.filter((e) => !NOISE_ACTIONS.has(e.action)).slice(0, 10), events.reason);
    renderDataSources(val(ds), ds.reason);
  }

  function renderStats(c, err) {
    if (err) { mount(statsEl, errorState(err)); return; }
    // Count up only on first paint; auto-refreshes just update the numbers.
    const v = (n) => (statsEl.dataset.painted ? fmtNumber(n) : countUp(n ?? 0));
    mount(statsEl,
      statCard({ label: 'Queries', value: v(c.queries), icon: 'code-square', tone: 'blue', href: urls.queries() }),
      statCard({ label: 'Dashboards', value: v(c.dashboards), icon: 'grid-1x2', tone: 'teal', href: urls.dashboards() }),
      statCard({ label: 'Alerts', value: v(c.alerts), icon: 'bell', tone: 'amber', href: urls.alerts() }),
      statCard({ label: 'Data sources', value: v(c.data_sources), icon: 'database', tone: 'purple', href: urls.dataSources() }),
      statCard({ label: 'Users', value: v(c.users), icon: 'people', tone: 'slate', href: urls.users() }));
    statsEl.dataset.painted = '1';
  }

  function renderHealth(rq, outdated, ds) {
    const workers = rq?.workers || [];
    const queues = Object.values(rq?.queues || {});
    const queued = queues.reduce((a, q) => a + (q.queued || 0), 0);
    const running = queues.reduce((a, q) => a + (q.started?.length || 0), 0);
    const paused = (ds || []).filter((d) => d.paused);
    const stale = outdated?.queries?.length || 0;
    const item = (ok, ic, label, sub, href) => h(href ? 'a.rdh-health-item' : 'div.rdh-health-item', { href, class: ok ? 'ok' : 'bad' },
      h('span.rdh-health-icon', icon(ic)), h('span', h('b', label), h('span.rdh-health-sub', sub)));
    mount(healthEl,
      item(workers.length > 0, workers.length ? 'cpu' : 'exclamation-triangle-fill', rq ? `${workers.length} worker${workers.length === 1 ? '' : 's'}` : 'Workers unknown',
        workers.length ? 'executing queries' : 'queries cannot run', urls.system()),
      item(queued < 20, 'hourglass-split', `${fmtNumber(queued)} queued`, `${running} running now`, urls.system()),
      item(stale === 0, 'calendar-x', `${stale} outdated`, 'scheduled refreshes waiting', urls.system()),
      item(!paused.length, paused.length ? 'pause-circle' : 'database-check', paused.length ? `${paused.length} paused` : 'All sources live',
        paused.length ? paused.map((d) => d.name).join(', ') : `${(ds || []).length} data source${(ds || []).length === 1 ? '' : 's'}`, urls.dataSources()));
  }

  function renderChecklist(c) {
    const steps = [
      { done: (c.data_sources || 0) > 0, label: 'Connect a data source', href: urls.dataSources(), icon: 'database' },
      { done: (c.queries || 0) > 0, label: 'Write your first query', href: urls.newQuery(), icon: 'code-square' },
      { done: (c.dashboards || 0) > 0, label: 'Build a dashboard', href: urls.dashboards({ new: 1 }), icon: 'grid-1x2' },
      { done: (c.alerts || 0) > 0, label: 'Set up an alert', href: urls.newAlert(), icon: 'bell' },
      { done: (c.users || 0) > 1, label: 'Invite your team', href: urls.users(), icon: 'people' },
    ];
    const done = steps.filter((s) => s.done).length;
    if (done === steps.length || prefs.get('studio:checklist-hidden', false)) { mount(checklistEl); return; }
    mount(checklistEl, h('section.rdh-checklist',
      h('div.rdh-checklist-head',
        h('div', h('div.fw-semibold', icon('rocket-takeoff', 'me-2'), 'Get the most out of Analytics Studio'),
          h('div.small.text-muted', `${done} of ${steps.length} done`)),
        h('div.rdh-progress', h('span', { style: { width: `${(done / steps.length) * 100}%` } })),
        h('button.af-icon-btn', { title: 'Hide', onclick: () => { prefs.set('studio:checklist-hidden', true); mount(checklistEl); } }, icon('x-lg'))),
      h('div.rdh-steps', steps.map((s) => h(s.done ? 'div.rdh-step.done' : 'a.rdh-step', { href: s.done ? undefined : s.href },
        icon(s.done ? 'check-circle-fill' : s.icon), h('span', s.label))))));
  }

  function renderQueryList(target, list, err, empty) {
    if (!list) { mount(target, errorState(err)); return; }
    if (!list.length) { mount(target, emptyState(empty)); return; }
    mount(target, h('div.af-list', list.slice(0, 8).map((q) => h('a.af-list-item', { href: urls.query(q.id) },
      favoriteStar('queries', q),
      h('div.af-list-main',
        h('div.af-list-title', q.name || `Query #${q.id}`, q.is_draft && h('span.rd-draft-pill.ms-2', 'draft')),
        h('div.af-list-sub', q.description || q.query?.replace(/\s+/g, ' ').slice(0, 90) || '—')),
      h('div.text-end.small.af-nowrap',
        h('div.text-muted', q.schedule?.interval ? h('span', { title: scheduleText(q.schedule) }, icon('arrow-repeat'), ' scheduled') : `#${q.id}`),
        h('div', timeAgo(q.updated_at)))))));
  }

  function renderDashList(target, list, err, empty) {
    if (!list) { mount(target, errorState(err)); return; }
    if (!list.length) { mount(target, emptyState(empty)); return; }
    mount(target, h('div.af-list', list.slice(0, 8).map((d) => h('a.af-list-item', { href: urls.dashboard(d.id) },
      favoriteStar('dashboards', d),
      h('span.rdh-dash-icon', icon('grid-1x2')),
      h('div.af-list-main',
        h('div.af-list-title', d.name, d.is_draft && h('span.rd-draft-pill.ms-2', 'draft')),
        h('div.af-list-sub', (d.tags || []).length ? d.tags.join(' · ') : `by ${d.user?.name || 'unknown'}`)),
      h('div.small.text-muted.af-nowrap', timeAgo(d.updated_at))))));
  }

  function renderFeed(events, err) {
    if (!events) { mount(feedEl, errorState(err)); return; }
    if (!events.length) { mount(feedEl, emptyState({ icon: 'activity', title: 'No activity yet' })); return; }
    mount(feedEl, h('div.rdh-feed', events.map((e) => {
      const info = actionInfo(e.action);
      return h('div.rdh-feed-item',
        h('span.rdh-feed-icon', { class: `af-tone-${info.tone}` }, icon(info.icon)),
        h('div.rdh-feed-main', eventSentence(e), h('div.rdh-feed-meta', timeAgo(e.created_at), e.browser && h('span', ` · ${e.browser.split(' / ').slice(-1)[0]}`))));
    })));
  }

  function renderDataSources(list, err) {
    if (!list) { mount(dsEl, errorState(err)); return; }
    if (!list.length) {
      mount(dsEl, emptyState({ icon: 'database-add', title: 'No data sources', message: 'Connect a database to start querying.', action: h('a.btn.btn-sm.btn-primary', { href: urls.dataSources() }, 'Add data source') }));
      return;
    }
    mount(dsEl, h('div.af-list', list.map((d) => h('a.af-list-item', { href: urls.dataSource(d.id) },
      h('span.rdh-ds-icon', icon(dataSourceIcon(d.type))),
      h('div.af-list-main', h('div.af-list-title', d.name), h('div.af-list-sub', d.type, d.view_only && ' · view only')),
      d.paused ? h('span.rd-draft-pill', { title: d.pause_reason || '' }, 'paused') : stateDot('success', 'Active')))));
  }

  await load();
}

/** Big search box with live results from queries and dashboards. */
function instantSearch(signal) {
  const input = h('input.form-control.rdh-search-input', { type: 'search', placeholder: 'Search queries and dashboards…', 'aria-label': 'Search queries and dashboards', dataset: { searchShortcut: '' } });
  const results = h('div.rdh-search-results', { role: 'listbox' });
  const wrap = h('div.rdh-search', icon('search', 'rdh-search-icon'), input, h('kbd.af-kbd-hint', '/'), results);
  let seq = 0;
  let active = -1;
  let items = [];
  const close = () => { results.classList.remove('open'); active = -1; };
  const draw = () => {
    mount(results, items.length
      ? items.map((it, i) => h('a.rdh-search-item', { href: it.href, class: i === active && 'active', role: 'option' },
        icon(it.icon), h('span.rdh-search-name', it.label), h('span.rdh-search-kind', it.kind)))
      : h('div.rdh-search-empty', 'No queries or dashboards match.'));
    results.classList.add('open');
  };
  input.addEventListener('input', debounce(async () => {
    const q = input.value.trim();
    const mine = ++seq;
    if (q.length < 2) { close(); return; }
    mount(results, h('div.rdh-search-empty', 'Searching…'));
    results.classList.add('open');
    try {
      const [qs, ds] = await Promise.all([
        rd.get('/api/queries', { q, page_size: 8 }, { signal }),
        rd.get('/api/dashboards', { q, page_size: 5 }, { signal }),
      ]);
      if (mine !== seq) return;
      items = [
        ...(ds.results || []).map((d) => ({ label: d.name, kind: 'Dashboard', icon: 'grid-1x2', href: urls.dashboard(d.id) })),
        ...(qs.results || []).map((x) => ({ label: x.name || `Query #${x.id}`, kind: `Query #${x.id}`, icon: 'code-square', href: urls.query(x.id) })),
      ];
      active = items.length ? 0 : -1;
      draw();
    } catch (err) {
      if (err.name !== 'AbortError') mount(results, h('div.rdh-search-empty.text-danger', err.message));
    }
  }, 180));
  input.addEventListener('keydown', (e) => {
    if (!results.classList.contains('open')) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(active + 1, items.length - 1); draw(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(active - 1, 0); draw(); }
    if (e.key === 'Enter' && items[active]) { e.preventDefault(); navigate(items[active].href); }
    if (e.key === 'Escape') close();
  });
  input.addEventListener('blur', () => setTimeout(close, 150));
  return wrap;
}
