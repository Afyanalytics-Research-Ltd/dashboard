// Activity: Redash's audit log (/api/events), humanized, grouped by day, filterable,
// with a per-day activity chart and expandable details.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, chips, searchInput, emptyState, errorState, skeleton, jsonView, timeAgo, refreshControl,
} from '../../../ui/js/ui.js';
import { sparkline } from '../../../ui/js/charts.js';
import { fmtDate, fmtNumber, fmtDay } from '../../../ui/js/format.js';
import { rd } from '../core/api.js';
import { actionInfo, eventSentence, objectLabel, dayLabel, NOISE_ACTIONS as NOISE } from './home_shared.js';

const PAGE = 100;
const SHOW_STEP = 100;

export default async function render(ctx) {
  const { el, signal } = ctx;
  const q = ctx.query;
  const state = {
    events: [],
    total: 0,
    page: 0,
    text: q.get('q') || '',
    action: q.get('action') || '',
    type: q.get('type') || '',
    user: q.get('user') || '',
    hideNoise: q.get('noise') !== 'show',
    shown: SHOW_STEP,
  };

  const summaryEl = h('div.rdh-ev-summary');
  const filtersEl = h('div.rdh-ev-filters');
  const listEl = h('div', skeleton(10));
  const moreEl = h('div.rdh-ev-more');

  const refresh = refreshControl(() => reload(), { key: 'studio-events', initial: 0 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Activity', icon: 'activity',
      subtitle: 'Everything people and API keys did in Redash: views, edits, executions, sharing and admin changes.',
      actions: refresh.el,
    }),
    summaryEl,
    card({ body: h('div', filtersEl, listEl, moreEl) }));

  const sync = () => { state.shown = SHOW_STEP; syncUrl(); };
  const syncUrl = () => ctx.setQuery({ q: state.text || null, action: state.action || null, type: state.type || null, user: state.user || null, noise: state.hideNoise ? null : 'show' });

  async function fetchPage() {
    const res = await rd.get('/api/events', { page: state.page + 1, page_size: PAGE }, { signal });
    state.page += 1;
    state.total = res.count;
    state.events.push(...(res.results || []));
  }

  async function reload() {
    state.events = [];
    state.page = 0;
    try {
      await fetchPage();
      // Load a little more history up front so the day chart and filters are meaningful.
      while (state.events.length < 300 && state.events.length < state.total) await fetchPage();
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(listEl, errorState(err, reload));
      return;
    }
    if (!ctx.active) return;
    draw();
  }

  function filtered() {
    const t = state.text.toLowerCase();
    return state.events.filter((e) => (!state.hideNoise || !NOISE.has(e.action))
      && (!state.action || e.action === state.action)
      && (!state.type || e.object_type === state.type)
      && (!state.user || (e.user_name || '') === state.user)
      && (!t || JSON.stringify(e).toLowerCase().includes(t)));
  }

  function draw() {
    const rows = filtered();
    // per-day counts over the loaded window
    const byDay = new Map();
    for (const e of state.events) byDay.set(fmtDay(e.created_at), (byDay.get(fmtDay(e.created_at)) || 0) + 1);
    const days = [];
    const oldest = state.events.length ? new Date(state.events[state.events.length - 1].created_at) : new Date();
    // Always chart at least two weeks so a single busy day still reads as a trend.
    const fortnight = new Date(Date.now() - 13 * 86400000);
    const from = oldest < fortnight ? oldest : fortnight;
    for (let d = new Date(from.getFullYear(), from.getMonth(), from.getDate()); d <= new Date(); d.setDate(d.getDate() + 1)) days.push(byDay.get(fmtDay(d)) || 0);
    const users = new Set(state.events.map((e) => e.user_name).filter(Boolean));
    const executions = state.events.filter((e) => e.action === 'execute_query' || e.action === 'execute').length;
    mount(summaryEl,
      h('div.af-stats',
        h('div.af-stat.af-tone-blue', h('div.af-stat-icon', icon('activity')), h('div.af-stat-text', h('div.af-stat-label', 'Events recorded'), h('div.af-stat-value', fmtNumber(state.total)), h('div.af-stat-sub', `${fmtNumber(state.events.length)} loaded`))),
        h('div.af-stat.af-tone-teal', h('div.af-stat-icon', icon('play-circle')), h('div.af-stat-text', h('div.af-stat-label', 'Query executions'), h('div.af-stat-value', fmtNumber(executions)), h('div.af-stat-sub', 'in loaded window'))),
        h('div.af-stat.af-tone-purple', h('div.af-stat-icon', icon('people')), h('div.af-stat-text', h('div.af-stat-label', 'Active users & keys'), h('div.af-stat-value', fmtNumber(users.size)), h('div.af-stat-sub', [...users].slice(0, 3).join(', ')))),
        h('div.af-stat.af-tone-slate.rdh-ev-spark', h('div.af-stat-text',
          h('div.af-stat-label', `Events per day · last ${Math.min(days.length, 60)} days`),
          sparkline(days.slice(-60), { width: 260, height: 40 }),
          h('div.af-stat-sub', `peak ${fmtNumber(Math.max(...days))} · today ${fmtNumber(days[days.length - 1] || 0)}`)))));

    const counts = (key) => {
      const m = new Map();
      for (const e of state.events) if (e[key]) m.set(e[key], (m.get(e[key]) || 0) + 1);
      return [...m.entries()].sort((a, b) => b[1] - a[1]);
    };
    const selectOf = (label, key, value, onChange, fmt = (v) => v) => h('label.rdh-ev-filter',
      h('span', label),
      h('select.form-select.form-select-sm', { onchange: (e) => onChange(e.target.value) },
        h('option', { value: '' }, 'Any'),
        counts(key).map(([v, n]) => h('option', { value: v, selected: v === value }, `${fmt(v)} (${n})`))));
    mount(filtersEl,
      h('div.af-toolbar',
        searchInput({ placeholder: 'Search events, names, SQL, browsers…', value: state.text, onInput: (v) => { state.text = v; sync(); draw(); } }),
        selectOf('Action', 'action', state.action, (v) => { state.action = v; sync(); draw(); }, (v) => actionInfo(v).verb),
        selectOf('Object', 'object_type', state.type, (v) => { state.type = v; sync(); draw(); }, objectLabel),
        selectOf('Who', 'user_name', state.user, (v) => { state.user = v; sync(); draw(); }),
        h('label.form-check.form-switch.small.mb-0.rdh-ev-noise',
          h('input.form-check-input', { type: 'checkbox', checked: state.hideNoise, onchange: (e) => { state.hideNoise = e.target.checked; sync(); draw(); } }),
          h('span.form-check-label', 'Hide list / favourites noise'))),
      h('div.rdh-ev-count', `${fmtNumber(rows.length)} matching of ${fmtNumber(state.events.length)} loaded events`));

    if (!rows.length) {
      mount(listEl, emptyState({ icon: 'search', title: 'No events match', message: state.events.length < state.total ? 'Load older events or relax the filters.' : 'Relax the filters to see more.' }));
    } else {
      const groups = new Map();
      for (const e of rows.slice(0, state.shown)) {
        const k = dayLabel(e.created_at);
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k).push(e);
      }
      mount(listEl, [...groups.entries()].map(([day, evs]) => h('section.rdh-ev-day',
        h('h3.rdh-ev-dayhead', day, h('span', fmtNumber(evs.length))),
        evs.map((e) => eventRow(e)))));
    }
    const hidden = rows.length - Math.min(rows.length, state.shown);
    mount(moreEl,
      hidden > 0 && h('button.btn.btn-sm.btn-outline-primary', {
        onclick: () => { state.shown += SHOW_STEP; draw(); },
      }, icon('chevron-down', 'me-1'), `Show ${fmtNumber(Math.min(hidden, SHOW_STEP))} more (${fmtNumber(hidden)} hidden)`),
      hidden <= 0 && state.events.length < state.total && h('button.btn.btn-sm.btn-outline-secondary', {
        onclick: async (ev) => {
          const b = ev.currentTarget;
          b.disabled = true;
          try { await fetchPage(); await fetchPage(); } catch { b.disabled = false; return; }
          state.shown += SHOW_STEP;
          if (ctx.active) draw();
        },
      }, icon('clock-history', 'me-1'), `Load older events (${fmtNumber(state.total - state.events.length)} more on the server)`));
  }

  function eventRow(e) {
    const info = actionInfo(e.action);
    const details = h('div.rdh-ev-details');
    let open = false;
    const toggle = h('button.af-icon-btn', { title: 'Details', 'aria-expanded': 'false' }, icon('chevron-down'));
    const row = h('div.rdh-ev-row',
      h('span.rdh-feed-icon', { class: `af-tone-${info.tone}` }, icon(info.icon)),
      h('div.rdh-ev-main',
        eventSentence(e),
        h('div.rdh-feed-meta',
          h('time', { title: fmtDate(e.created_at) }, new Date(e.created_at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })),
          ' · ', timeAgo(e.created_at),
          e.browser && h('span', ` · ${e.browser}`),
          e.location && e.location !== 'Unknown' && h('span', ` · ${e.location}`)),
        e.details?.query && h('code.rdh-ev-sql', e.details.query.replace(/\s+/g, ' ').slice(0, 220))),
      toggle);
    toggle.addEventListener('click', () => {
      open = !open;
      toggle.setAttribute('aria-expanded', String(open));
      toggle.classList.toggle('open', open);
      mount(details, open ? jsonView(e, { open: 2 }) : null);
    });
    return h('div', row, details);
  }

  await reload();
}
