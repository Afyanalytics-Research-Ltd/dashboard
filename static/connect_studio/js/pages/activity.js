// Activity: every background job (tests, extractions, AI mappings, syncs) with live logs.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { api } from '../api.js';
import { ago, empty, fmtDuration, openJobDrawer, seg, statusBadge, statusDot } from '../kit.js';

const KIND_ICON = { test: 'plug', extract: 'lightning-charge', map: 'stars', remap: 'stars', sync: 'arrow-repeat' };

export default async function render(ctx) {
  ctx.setTitle('Activity');
  let kind = ctx.query.get('kind') || '';
  let status = ctx.query.get('status') || '';
  const listEl = h('div.cs-card');

  async function load() {
    const { jobs, total } = await api.jobs({ kind, status, limit: 100 });
    mount(listEl, jobs.length ? h('div',
      h('div.cs-card-head', h('div.cs-card-title', icon('activity'), `${total} job${total === 1 ? '' : 's'}`), h('span.cs-grow'),
        jobs.some((j) => j.status === 'running') && h('span.cs-badge.cs-badge-info', h('span.cs-dot.run'), 'live')),
      h('div.cs-list', jobs.map((j) => h('div.cs-li.click', { onclick: () => openJobDrawer(j) },
        statusDot(j.status),
        h('span.cs-logo.cs-logo-sm', { style: { '--c': j.kind === 'map' || j.kind === 'remap' ? '#7c3aed' : j.kind === 'sync' ? '#0bb99f' : '#0072ce' } }, icon(KIND_ICON[j.kind] || 'cpu')),
        h('div.cs-grow', { style: { minWidth: 0 } }, h('div.cs-li-title.cs-ellipsis', j.title || j.kind_label),
          h('div.cs-li-sub.cs-ellipsis', j.status === 'failed' ? j.error || j.message : j.message || j.kind_label)),
        j.status === 'running' ? h('div', { style: { width: '120px' } }, h('div.cs-bar', h('span', { style: { width: `${j.progress}%` } }))) : null,
        h('span.cs-li-status', statusBadge(j.status)),
        h('span.cs-muted.cs-nowrap.cs-li-time', { style: { fontSize: '12px', textAlign: 'right' } },
          `${ago(j.created_at)}${j.duration ? ` · ${fmtDuration(j.duration)}` : ''}`),
        j.user && h('span.cs-badge.cs-badge-outline.cs-li-user', icon('person'), j.user)))))
      : empty({ icon: 'activity', title: 'No jobs', message: 'Tests, extractions, AI mappings and syncs show up here.' }));
  }

  mount(ctx.el,
    h('div.cs-head', h('div', h('h1', 'Activity'), h('div.cs-head-sub', 'Every test, schema extraction, AI mapping and sync — click one for its live log.'))),
    h('div.cs-row.cs-wrap',
      seg([{ value: '', label: 'All' }, { value: 'extract', label: 'Extractions', icon: 'lightning-charge' }, { value: 'map', label: 'AI mappings', icon: 'stars' },
        { value: 'sync', label: 'Syncs', icon: 'arrow-repeat' }, { value: 'test', label: 'Tests', icon: 'plug' }], kind, (v) => { kind = v; ctx.setQuery({ kind: v }); load(); }),
      h('span.cs-grow'),
      seg([{ value: '', label: 'Any status' }, { value: 'running', label: 'Running' }, { value: 'succeeded', label: 'Succeeded' }, { value: 'failed', label: 'Failed' }],
        status, (v) => { status = v; ctx.setQuery({ status: v }); load(); })),
    listEl);
  await load();
  ctx.every(5000, load);
}
