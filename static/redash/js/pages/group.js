// Group detail (/groups/:groupId): rename, members (add via search, remove),
// data sources (add, remove, view-only toggle) and permissions.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { pageHeader, card, tabs, emptyState, errorState, skeleton, timeAgo, toast, toastError, confirmDialog, spinner } from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { inlineEdit, invalidate } from '../core/common.js';
import { typeLogo, pill, matches, hl, groupDataSources } from './admin_shared.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  const id = ctx.params.groupId;
  let group;
  try {
    group = await rd.get(`/api/groups/${enc(id)}`, null, { signal });
  } catch (err) {
    // Redash answers 500 (not 404) for a group id that no longer exists.
    const missing = err.status === 404 || err.status === 500;
    mount(el, pageHeader({ title: 'Group', icon: 'diagram-2', crumbs: [{ label: 'Groups', href: urls.groups() }] }),
      missing ? emptyState({ icon: 'diagram-2', title: `Group #${id} not found`, message: 'It may have been deleted.', action: h('a.btn.btn-sm.btn-primary', { href: urls.groups() }, 'All groups') }) : errorState(err));
    return;
  }
  ctx.setTitle(group.name);
  const builtin = group.type === 'builtin';
  let tab = ctx.query.get('tab') || 'members';
  const pane = h('div');
  const counts = { members: null, sources: null };

  const tabBar = () => tabs([
    { id: 'members', label: 'Members', icon: 'person', count: counts.members },
    { id: 'sources', label: 'Data sources', icon: 'database', count: counts.sources },
    { id: 'permissions', label: 'Permissions', icon: 'shield-check', count: (group.permissions || []).length },
  ], tab, (t) => { tab = t; ctx.setQuery({ tab: t === 'members' ? null : t }); show(); });
  const tabHost = h('div', tabBar());
  const refreshTabs = () => mount(tabHost, tabBar());

  mount(el,
    pageHeader({
      title: builtin ? group.name : inlineEdit(group.name, async (v) => {
        group = await rd.post(`/api/groups/${enc(group.id)}`, { name: v });
        toast('Group renamed', 'success');
      }),
      icon: group.name === 'admin' ? 'shield-fill' : 'diagram-2',
      crumbs: [{ label: 'Groups', href: urls.groups() }, { label: group.name }],
      badges: builtin && pill('built-in', 'muted'),
      subtitle: builtin ? 'Built-in groups cannot be renamed or deleted.' : h('span', 'Created ', timeAgo(group.created_at)),
      actions: !builtin && [h('button.btn.btn-sm.btn-outline-danger', { onclick: remove }, icon('trash', 'me-1'), 'Delete group')],
    }),
    card({ body: h('div', tabHost, pane) }));

  function show() {
    if (tab === 'sources') return showSources();
    if (tab === 'permissions') return showPermissions();
    return showMembers();
  }

  // ---- members
  async function showMembers() {
    mount(pane, skeleton(4));
    let members;
    try { members = await rd.get(`/api/groups/${enc(group.id)}/members`, null, { signal }); } catch (err) {
      if (err.name !== 'AbortError') mount(pane, errorState(err, showMembers));
      return;
    }
    if (!ctx.active || tab !== 'members') return;
    counts.members = members.length;
    refreshTabs();
    const ids = new Set(members.map((m) => m.id));
    let q = '';
    const list = h('div.rd-member-list');
    const draw = () => mount(list, members.filter((m) => matches(m, q, ['name', 'email'])).map((m) => h('div.rd-member',
      h('img.rd-avatar-lg', { src: m.profile_image_url, alt: '' }),
      h('div.rd-member-main', h('a.fw-semibold', { href: urls.user(m.id) }, hl(m.name, q)), h('div.small.text-muted', hl(m.email, q))),
      m.is_disabled && pill('disabled', 'bad'),
      h('button.btn.btn-sm.btn-outline-danger', {
        title: builtin && group.name === 'default' ? 'Everyone is in the default group' : `Remove ${m.name}`,
        disabled: builtin && group.name === 'default',
        onclick: async () => {
          if (!await confirmDialog({ title: `Remove ${m.name} from ${group.name}?`, confirmLabel: 'Remove', danger: true })) return;
          try {
            await rd.del(`/api/groups/${enc(group.id)}/members/${enc(m.id)}`);
            toast(`${m.name} removed`, 'success');
            invalidate('session');
            showMembers();
          } catch (err) { toastError(err, 'Could not remove member'); }
        },
      }, icon('person-dash')))));

    // add-member picker
    const input = h('input.form-control.form-control-sm', { type: 'search', placeholder: 'Add people — search by name or email…' });
    const results = h('div.rd-picker-list.rd-picker-pop.d-none');
    let seq = 0;
    input.addEventListener('input', debounce(async () => {
      const term = input.value.trim();
      const mine = ++seq;
      if (!term) { results.classList.add('d-none'); return; }
      results.classList.remove('d-none');
      mount(results, h('div.p-2', spinner('sm')));
      try {
        const res = await rd.get('/api/users', { q: term, page_size: 8 }, { signal });
        if (mine !== seq) return;
        const cands = (res.results || []).filter((u) => !ids.has(u.id));
        mount(results, cands.length ? cands.map((u) => h('button.rd-picker-item', {
          type: 'button',
          onclick: async () => {
            try {
              await rd.post(`/api/groups/${enc(group.id)}/members`, { user_id: u.id });
              toast(`${u.name} added to ${group.name}`, 'success');
              invalidate('session');
              showMembers();
            } catch (err) { toastError(err, 'Could not add member'); }
          },
        }, h('img.rd-avatar', { src: u.profile_image_url, alt: '' }), h('span.rd-picker-name', u.name), h('span.small.text-muted', u.email)))
          : h('div.small.text-muted.p-2', 'No one else matches.'));
      } catch (err) { mount(results, h('div.text-danger.small.p-2', err.message)); }
    }, 200));
    input.addEventListener('blur', () => setTimeout(() => results.classList.add('d-none'), 200));

    mount(pane, h('div.d-flex.flex-column.gap-3',
      h('div.rd-member-tools',
        h('div.rd-picker-wrap', icon('person-plus', 'rd-picker-icon'), input, results),
        members.length > 6 && h('div.af-search', icon('search', 'af-search-icon'), h('input.form-control.form-control-sm', { type: 'search', placeholder: 'Filter members…', oninput: debounce((e) => { q = e.target.value.trim(); draw(); }, 100) }))),
      members.length ? list : emptyState({ icon: 'people', title: 'No members yet', message: 'Search above to add people to this group.' })));
    draw();
  }

  // ---- data sources
  async function showSources() {
    mount(pane, skeleton(3));
    let attached;
    let all;
    let degraded = false;
    try {
      let res;
      [res, all] = await Promise.all([
        groupDataSources(group.id, { signal }),
        rd.get('/api/data_sources', null, { signal }),
      ]);
      attached = res.sources;
      degraded = res.degraded;
    } catch (err) {
      if (err.name !== 'AbortError') mount(pane, errorState(err, showSources));
      return;
    }
    if (!ctx.active || tab !== 'sources') return;
    counts.sources = attached.length;
    refreshTabs();
    const ids = new Set(attached.map((d) => d.id));
    const available = all.filter((d) => !ids.has(d.id));
    const add = h('select.form-select.form-select-sm.rd-add-ds', h('option', { value: '' }, available.length ? `Give access to a data source (${available.length} available)…` : 'All data sources already added'),
      available.map((d) => h('option', { value: d.id }, `${d.name} (${d.type})`)));
    add.addEventListener('change', async () => {
      if (!add.value) return;
      try {
        await rd.post(`/api/groups/${enc(group.id)}/data_sources`, { data_source_id: Number(add.value) });
        toast('Access granted', 'success');
        invalidate('ds');
        showSources();
      } catch (err) { toastError(err, 'Could not add data source'); }
    });
    mount(pane, h('div.d-flex.flex-column.gap-3',
      degraded && h('div.af-callout.af-callout-warn', icon('exclamation-triangle'),
        h('span', 'Redash has duplicate permission records for this group, so its own data-sources endpoint fails. ',
          'Studio rebuilt the list from each data source instead; toggling view-only may report an error until the duplicates are removed in the Redash database.')),
      add,
      attached.length ? h('div.rd-member-list', attached.map((d) => {
        const vo = h('input.form-check-input', { type: 'checkbox', checked: !!d.view_only });
        vo.addEventListener('change', async () => {
          vo.disabled = true;
          try {
            await rd.post(`/api/groups/${enc(group.id)}/data_sources/${enc(d.id)}`, { view_only: vo.checked });
            toast(vo.checked ? 'Now view only' : 'Now full access', 'success');
            d.view_only = vo.checked;
            mount(label, vo.checked ? 'View only' : 'Full access');
          } catch (err) { vo.checked = !vo.checked; toastError(err); }
          vo.disabled = false;
        });
        const label = h('span.small', d.view_only ? 'View only' : 'Full access');
        return h('div.rd-member',
          typeLogo(d.type, 'db', 30),
          h('div.rd-member-main', h('a.fw-semibold', { href: urls.dataSource(d.id) }, d.name), h('div.small.text-muted', d.type)),
          h('label.form-check.form-switch.m-0.d-inline-flex.align-items-center.gap-2', { title: 'View only: can see results but not write or run new SQL' }, vo, label),
          h('button.btn.btn-sm.btn-outline-danger', {
            title: 'Remove access',
            onclick: async () => {
              if (!await confirmDialog({ title: `Remove ${d.name} from ${group.name}?`, danger: true, confirmLabel: 'Remove access', message: 'Members lose access unless another of their groups grants it.' })) return;
              try {
                await rd.del(`/api/groups/${enc(group.id)}/data_sources/${enc(d.id)}`);
                toast('Access removed', 'success');
                invalidate('ds');
                showSources();
              } catch (err) { toastError(err); }
            },
          }, icon('x-lg')));
      })) : emptyState({ icon: 'database', title: 'No data sources', message: 'Members of this group cannot query anything through it yet.' })));
  }

  // ---- permissions
  function showPermissions() {
    const desc = {
      admin: 'Manage users, groups, data sources and settings', super_admin: 'Everything, across the organization',
      execute_query: 'Run queries', create_query: 'Create queries', edit_query: 'Edit queries', view_query: 'See queries',
      view_source: 'See query SQL', create_dashboard: 'Create dashboards', edit_dashboard: 'Edit dashboards',
      list_dashboards: 'Browse dashboards', list_alerts: 'Browse alerts', list_users: 'See other users', list_data_sources: 'See data sources',
      schedule_query: 'Schedule queries',
    };
    mount(pane, h('div',
      h('p.small.text-muted', 'Permissions are fixed per group type in Redash. Use different groups to give different people different capabilities.'),
      (group.permissions || []).length ? h('div.rd-perm-grid', group.permissions.map((p) => h('div.rd-perm-card', icon('check-circle-fill', 'text-success'),
        h('div', h('div.fw-semibold.af-mono', p), h('div.small.text-muted', desc[p] || '')))))
        : emptyState({ icon: 'shield', title: 'No permissions' })));
  }

  async function remove() {
    const ok = await confirmDialog({ title: `Delete group ${group.name}?`, danger: true, confirmLabel: 'Delete group', message: 'Members lose any data source access that came only through this group.' });
    if (!ok) return;
    try {
      await rd.del(`/api/groups/${enc(group.id)}`);
      toast('Group deleted', 'success');
      navigate(urls.groups());
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  show();
}
