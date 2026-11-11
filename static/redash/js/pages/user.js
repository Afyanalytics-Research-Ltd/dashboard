// User profile (/users/:userId): inline name/email edit, group membership editor,
// API key (masked, reveal, copy, regenerate), status actions and invite/reset links.
// The account whose API key powers Analytics Studio is protected from being
// disabled or having its key regenerated (the proxy also refuses with 409).

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, kv, errorState, timeAgo, toast, toastError, confirmDialog, copyButton } from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { getSession, inlineEdit, invalidate } from '../core/common.js';
import { pill } from './admin_shared.js';
import { showLinkDialog } from './users.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  const userId = Number(ctx.params.userId);
  let user;
  let groups;
  let session;
  try {
    [user, groups, session] = await Promise.all([
      rd.get(`/api/users/${enc(userId)}`, null, { signal }),
      rd.get('/api/groups', null, { signal }),
      getSession(),
    ]);
  } catch (err) {
    mount(el, pageHeader({ title: 'User', icon: 'person', crumbs: [{ label: 'Users', href: urls.users() }] }), errorState(err));
    return;
  }
  ctx.setTitle(user.name);
  const keyOwner = session?.user?.id === user.id; // this user's key is what the Studio proxy uses
  const headerHost = h('div');
  const groupsEl = h('div');
  const keyEl = h('div');
  const actionsEl = h('div.d-flex.flex-column.gap-2');

  const status = () => (user.is_disabled ? pill('Disabled', 'bad', 'slash-circle')
    : user.is_invitation_pending ? pill('Invite pending', 'warn', 'envelope') : pill('Active', 'ok', 'circle-fill'));

  const save = async (patch) => {
    user = await rd.post(`/api/users/${enc(user.id)}`, patch);
    toast('Saved', 'success');
    drawHeader();
  };

  function drawHeader() {
    mount(headerHost, h('section.rd-profile',
      h('img.rd-profile-avatar', { src: (user.profile_image_url || '').replace('s=40', 's=160'), alt: '' }),
      h('div.rd-profile-main',
        h('nav.af-crumbs', h('a', { href: urls.users(), dataset: { link: '' } }, 'Users'), icon('chevron-right', 'af-crumb-sep'), h('span', user.name)),
        h('h1.rd-profile-name', inlineEdit(user.name, (v) => save({ name: v }), { placeholder: 'Name' }), status(), keyOwner && pill('Studio service account', 'info', 'plug-fill')),
        h('div.rd-profile-email', icon('envelope', 'me-1'), inlineEdit(user.email, (v) => {
          if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v)) return Promise.reject(new Error('Enter a valid email address.'));
          return save({ email: v });
        }, { placeholder: 'Email' })),
        h('div.rd-profile-meta',
          h('span', icon('calendar', 'me-1'), 'Joined ', timeAgo(user.created_at)),
          h('span', icon('activity', 'me-1'), 'Last active ', user.active_at ? timeAgo(user.active_at) : 'never'),
          h('span', icon('shield-lock', 'me-1'), user.auth_type === 'password' ? 'Password sign-in' : user.auth_type)))));
  }

  // ---- groups
  function drawGroups() {
    const member = new Set(user.groups.map((g) => (typeof g === 'object' ? g.id : g)));
    mount(groupsEl, h('div.rd-group-list', groups.map((g) => {
      const on = member.has(g.id);
      const builtinDefault = g.type === 'builtin' && g.name === 'default';
      const cb = h('input.form-check-input', { type: 'checkbox', checked: on, disabled: builtinDefault, 'aria-label': `Member of ${g.name}` });
      cb.addEventListener('change', async () => {
        const next = new Set(member);
        if (cb.checked) next.add(g.id); else next.delete(g.id);
        if (!next.size) { cb.checked = true; toast('A user must belong to at least one group.', 'warning'); return; }
        cb.disabled = true;
        try {
          user = await rd.post(`/api/users/${enc(user.id)}`, { group_ids: [...next] });
          toast(cb.checked ? `Added to ${g.name}` : `Removed from ${g.name}`, 'success');
          invalidate('session');
        } catch (err) { cb.checked = !cb.checked; toastError(err, 'Could not change groups'); }
        drawGroups();
      });
      return h('label.rd-group-item', { class: on && 'on' },
        cb,
        h('div.rd-group-item-main',
          h('div.fw-semibold', g.name, g.type === 'builtin' && pill('built-in', 'muted'), g.name === 'admin' && pill('full control', 'warn', 'shield-fill')),
          h('div.small.text-muted', (g.permissions || []).slice(0, 6).join(', ') + ((g.permissions || []).length > 6 ? '…' : ''))),
        h('a.small', { href: urls.group(g.id), onclick: (e) => e.stopPropagation() }, 'View'),
        builtinDefault && h('span.small.text-muted', { title: 'Every user is in the default group' }, icon('lock')));
    })));
  }

  // ---- API key
  function drawKey() {
    const key = user.api_key;
    if (!key) { mount(keyEl, h('div.small.text-muted', 'The API key is only visible to admins and the user.')); return; }
    let shown = false;
    const code = h('code.rda-key');
    const draw = () => { code.textContent = shown ? key : `${key.slice(0, 4)}${'•'.repeat(28)}${key.slice(-4)}`; };
    draw();
    const regen = h('button.btn.btn-sm.btn-outline-danger', {
      disabled: keyOwner,
      title: keyOwner ? 'Analytics Studio uses this key — regenerating it would disconnect the Studio until .env is updated.' : 'Issue a new key; the old one stops working',
      onclick: async () => {
        const ok = await confirmDialog({
          title: 'Regenerate API key?', danger: true, confirmLabel: 'Regenerate',
          message: `Scripts and integrations using ${user.name}'s current key will stop working immediately.`,
        });
        if (!ok) return;
        try {
          user = await rd.post(`/api/users/${enc(user.id)}/regenerate_api_key`, {});
          toast('New API key issued', 'success');
          drawKey();
        } catch (err) { toastError(err, err.status === 409 ? 'Protected account' : 'Could not regenerate'); }
      },
    }, icon('arrow-repeat', 'me-1'), 'Regenerate');
    mount(keyEl,
      h('div.rda-key-row', code,
        h('button.btn.btn-sm.btn-outline-secondary', { title: 'Show / hide', onclick: (e) => { shown = !shown; draw(); mount(e.currentTarget, icon(shown ? 'eye-slash' : 'eye')); } }, icon('eye')),
        copyButton(key, 'Copy'), regen),
      keyOwner && h('div.af-callout.af-callout-info.mt-2', icon('info-circle'),
        h('span', 'This is the account Analytics Studio connects with, so its key can\'t be regenerated or the account disabled from here.')),
      h('div.small.text-muted.mt-2', 'Use it as ', h('code', 'Authorization: Key …'), ' when calling the Redash API. It grants the same access as this user.'));
  }

  // ---- actions
  function drawActions() {
    const btn = (label, ic, variant, onClick, opts = {}) => h(`button.btn.btn-sm.btn-${variant}.text-start`, { onclick: onClick, disabled: opts.disabled, title: opts.title }, icon(ic, 'me-2'), label);
    mount(actionsEl,
      user.is_invitation_pending && btn('Get invite link', 'link-45deg', 'outline-primary', async () => {
        try {
          const r = await rd.post(`/api/users/${enc(user.id)}/invite`, {});
          if (r.invite_link) showLinkDialog(`Invite link for ${user.name}`, r.invite_link, 'Email is not configured on Redash, so share this link yourself.');
          else toast('Invitation re-sent', 'success');
        } catch (err) { toastError(err); }
      }),
      !user.is_invitation_pending && !user.is_disabled && btn('Send password reset link', 'key', 'outline-secondary', async () => {
        try {
          const r = await rd.post(`/api/users/${enc(user.id)}/reset_password`, {});
          if (r.reset_link) showLinkDialog(`Password reset for ${user.name}`, r.reset_link, 'Email is not configured on Redash, so share this link yourself.');
          else toast('Reset email sent', 'success');
        } catch (err) { toastError(err); }
      }),
      user.is_disabled
        ? btn('Enable user', 'person-check', 'outline-success', async () => {
          try { user = await rd.del(`/api/users/${enc(user.id)}/disable`); toast('User enabled', 'success'); refresh(); } catch (err) { toastError(err); }
        })
        : btn('Disable user', 'person-slash', 'outline-danger', async () => {
          if (!await confirmDialog({ title: `Disable ${user.name}?`, danger: true, confirmLabel: 'Disable', message: 'They will not be able to sign in or use their API key until enabled again.' })) return;
          try { user = await rd.post(`/api/users/${enc(user.id)}/disable`, {}); toast('User disabled', 'success'); refresh(); } catch (err) { toastError(err, err.status === 409 || err.status === 403 ? 'Not allowed' : 'Could not disable'); }
        }, { disabled: keyOwner, title: keyOwner ? 'The Studio service account cannot be disabled here.' : '' }),
      user.is_invitation_pending && btn('Delete invitation', 'trash', 'outline-danger', async () => {
        if (!await confirmDialog({ title: `Delete invitation for ${user.name}?`, danger: true, confirmLabel: 'Delete' })) return;
        try { await rd.del(`/api/users/${enc(user.id)}`); toast('Invitation deleted', 'success'); navigate(urls.users({ view: 'pending' })); } catch (err) { toastError(err); }
      }));
  }

  function refresh() { drawHeader(); drawGroups(); drawKey(); drawActions(); }

  mount(el,
    headerHost,
    h('div.af-grid-main-side',
      h('div.d-flex.flex-column.gap-3',
        card({ title: 'Groups', icon: 'diagram-2', body: h('div', h('p.small.text-muted', 'Groups decide which data sources the user can query and what they may do.'), groupsEl) }),
        card({ title: 'API key', icon: 'key', body: keyEl })),
      h('div.d-flex.flex-column.gap-3',
        card({ title: 'Actions', icon: 'lightning', body: actionsEl }),
        card({ title: 'Details', icon: 'info-circle', body: kv([
          ['User id', String(user.id)],
          ['Email verified', user.is_email_verified ? 'Yes' : 'No'],
          ['Disabled at', user.disabled_at ? timeAgo(user.disabled_at) : '—'],
          ['Updated', timeAgo(user.updated_at)],
        ], { cols: 1 }) }))));
  refresh();
}
