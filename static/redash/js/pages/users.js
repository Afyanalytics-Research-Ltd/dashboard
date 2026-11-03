// Users: Active / Pending / Disabled tabs, server-side search + sort + paging,
// group chips, invite flow that surfaces the invite link when mail is off,
// and row actions (disable/enable, resend invite, reset link, delete pending).

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, dataTable, pager, searchInput, tabs, emptyState, errorState, skeleton, timeAgo,
  formModal, modal, toast, toastError, confirmDialog, copyButton, openMenu,
} from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { getSession } from '../core/common.js';
import { pill, hl } from './admin_shared.js';

const VIEWS = {
  active: { label: 'Active', icon: 'person-check', params: { pending: false } },
  pending: { label: 'Pending invitations', icon: 'envelope-paper', params: { pending: true } },
  disabled: { label: 'Disabled', icon: 'person-slash', params: { disabled: true } },
};

/** Show a link that must be shared manually (mail not configured). */
export function showLinkDialog(title, link, explanation) {
  modal({
    title, icon: 'link-45deg',
    body: h('div',
      h('p.small', explanation),
      h('div.rd-link-box', h('code', link), copyButton(link, 'Copy link'))),
    actions: [{ label: 'Done', variant: 'primary' }],
  });
}

export async function inviteUser() {
  return new Promise((resolve) => {
    formModal({
      title: 'Invite a user', icon: 'person-plus', submitLabel: 'Create invitation',
      subtitle: 'They join the "default" group; add more groups from their profile.',
      fields: [
        { name: 'name', label: 'Full name', required: true, placeholder: 'Jane Wanjiru' },
        { name: 'email', label: 'Email', type: 'email', required: true, placeholder: 'jane@afya.ai' },
      ],
      onSubmit: async ({ name, email }, close) => {
        if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) throw new Error('That email address does not look right.');
        const user = await rd.post('/api/users', { name: name.trim(), email: email.trim() });
        close();
        if (user.invite_link) {
          showLinkDialog('Invitation created', user.invite_link, `Email is not configured on Redash, so send ${user.name} this link yourself. It lets them set a password and sign in.`);
        } else toast(`Invitation emailed to ${user.email}`, 'success');
        resolve(user);
      },
    });
  });
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  let view = VIEWS[ctx.query.get('view')] ? ctx.query.get('view') : 'active';
  let q = ctx.query.get('q') || '';
  let order = ctx.query.get('order') || 'name';
  let page = Number(ctx.query.get('page')) || 1;
  let pageSize = 25;
  const body = h('div', skeleton(6));
  let groups = {};
  let me = null;

  const tabBar = tabs(Object.entries(VIEWS).map(([id, v]) => ({ id, label: v.label, icon: v.icon })), view, (v) => {
    view = v; page = 1; ctx.setQuery({ view: v === 'active' ? null : v, page: null }); load();
  });

  mount(el,
    pageHeader({
      title: 'Users', icon: 'people',
      subtitle: 'Everyone who can sign in to Redash, and what they can reach through their groups.',
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => inviteUser().then((u) => { if (u) { view = 'pending'; tabBar.setActive('pending'); load(); } }) }, icon('person-plus', 'me-1'), 'Invite user')],
    }),
    card({
      body: h('div',
        tabBar,
        h('div.af-toolbar.mb-3', searchInput({ placeholder: 'Search by name or email…', value: q, onInput: (v) => { q = v; page = 1; ctx.setQuery({ q: v, page: null }); load(); } })),
        body),
    }));

  async function load() {
    mount(body, skeleton(6));
    try {
      const [res, gs, session] = await Promise.all([
        rd.get('/api/users', { ...VIEWS[view].params, q: q || undefined, order, page, page_size: pageSize }, { signal }),
        rd.get('/api/groups', null, { signal }),
        getSession(),
      ]);
      if (!ctx.active) return;
      groups = Object.fromEntries(gs.map((g) => [g.id, g]));
      me = session?.user;
      draw(res);
    } catch (err) {
      if (err.name !== 'AbortError') mount(body, errorState(err, load));
    }
  }

  function draw(res) {
    const users = res.results || [];
    const sortKey = order.replace(/^-/, '');
    mount(body,
      dataTable({
        rows: users,
        rowKey: (u) => u.id,
        sort: { key: sortKey, dir: order.startsWith('-') ? 'desc' : 'asc' },
        onSort: (key, dir) => { order = dir === 'desc' ? `-${key}` : key; ctx.setQuery({ order }); load(); },
        onRowClick: (u) => navigate(urls.user(u.id)),
        empty: emptyState({
          icon: view === 'pending' ? 'envelope-paper' : 'people',
          title: q ? 'No users match' : view === 'pending' ? 'No pending invitations' : view === 'disabled' ? 'No disabled users' : 'No users',
          message: q ? 'Try another name or email.' : null,
        }),
        columns: [
          {
            key: 'name', label: 'User', sortable: true,
            render: (u) => h('div.rd-user-cell',
              h('img.rd-avatar-lg', { src: u.profile_image_url, alt: '', loading: 'lazy' }),
              h('div',
                h('div.fw-semibold', hl(u.name, q), me?.id === u.id && pill('you', 'info')),
                h('div.small.text-muted', hl(u.email, q)))),
          },
          {
            key: 'groups', label: 'Groups', sortable: true,
            render: (u) => h('div.rd-chip-row', (u.groups || []).map((g) => {
              const grp = typeof g === 'object' ? g : groups[g] || { id: g, name: `#${g}` };
              return h('a.af-tag', { href: urls.group(grp.id), class: grp.name === 'admin' && 'rd-tag-admin' }, grp.name);
            })),
          },
          { key: 'auth_type', label: 'Sign-in', render: (u) => h('span.small', u.auth_type === 'password' ? 'Password' : u.auth_type || '—') },
          {
            key: 'status', label: 'Status',
            render: (u) => (u.is_disabled ? pill('Disabled', 'bad', 'slash-circle')
              : u.is_invitation_pending ? pill('Invite pending', 'warn', 'envelope')
                : pill('Active', 'ok', 'circle-fill')),
          },
          { key: 'active_at', label: 'Last active', sortable: true, render: (u) => (u.active_at ? timeAgo(u.active_at) : h('span.text-muted.small', 'never')) },
          { key: 'created_at', label: 'Joined', sortable: true, render: (u) => timeAgo(u.created_at) },
          {
            key: 'actions', label: '', align: 'right', width: '50px',
            render: (u) => h('button.btn.btn-sm.btn-outline-secondary', { title: 'Actions', onclick: (e) => openMenu(e.currentTarget, userMenu(u)) }, icon('three-dots')),
          },
        ],
      }),
      res.count > pageSize && pager({ total: res.count, limit: pageSize, offset: (page - 1) * pageSize, onChange: (off, lim) => { pageSize = lim; page = Math.floor(off / lim) + 1; ctx.setQuery({ page: page > 1 ? page : null }); load(); } }));
  }

  function userMenu(u) {
    const self = me?.id === u.id;
    return [
      { label: 'Open profile', icon: 'person', href: urls.user(u.id) },
      u.is_invitation_pending && { label: 'Get invite link', icon: 'link-45deg', onClick: () => act(u, 'invite') },
      !u.is_invitation_pending && !u.is_disabled && { label: 'Password reset link', icon: 'key', onClick: () => act(u, 'reset') },
      { divider: true },
      u.is_disabled
        ? { label: 'Enable user', icon: 'person-check', onClick: () => act(u, 'enable') }
        : { label: self ? 'Disable (not allowed for yourself)' : 'Disable user', icon: 'person-slash', danger: true, disabled: self, onClick: () => act(u, 'disable') },
      u.is_invitation_pending && { label: 'Delete invitation', icon: 'trash', danger: true, onClick: () => act(u, 'delete') },
    ];
  }

  async function act(u, what) {
    try {
      if (what === 'invite') {
        const r = await rd.post(`/api/users/${enc(u.id)}/invite`, {});
        if (r.invite_link) showLinkDialog(`Invite link for ${u.name}`, r.invite_link, 'Email is not configured on Redash, so share this link yourself.');
        else toast(`Invitation re-sent to ${u.email}`, 'success');
      } else if (what === 'reset') {
        const r = await rd.post(`/api/users/${enc(u.id)}/reset_password`, {});
        if (r.reset_link) showLinkDialog(`Password reset for ${u.name}`, r.reset_link, 'Email is not configured on Redash, so share this link yourself. It lets them choose a new password.');
        else toast(`Reset email sent to ${u.email}`, 'success');
      } else if (what === 'disable') {
        if (!await confirmDialog({ title: `Disable ${u.name}?`, danger: true, confirmLabel: 'Disable', message: 'They will not be able to sign in or use their API key. You can enable them again later.' })) return;
        await rd.post(`/api/users/${enc(u.id)}/disable`, {});
        toast(`${u.name} disabled`, 'success');
        load();
      } else if (what === 'enable') {
        await rd.del(`/api/users/${enc(u.id)}/disable`);
        toast(`${u.name} enabled`, 'success');
        load();
      } else if (what === 'delete') {
        if (!await confirmDialog({ title: `Delete invitation for ${u.name}?`, danger: true, confirmLabel: 'Delete', message: 'The invite link stops working.' })) return;
        await rd.del(`/api/users/${enc(u.id)}`);
        toast('Invitation deleted', 'success');
        load();
      }
    } catch (err) { toastError(err, 'Action failed'); }
  }

  await load();
}
