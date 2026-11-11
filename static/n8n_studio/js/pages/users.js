// Users: invite (with copyable invite links), change global role, delete with
// workflow/credential transfer, pending/MFA status.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, dataTable, toast, toastError, modal, searchInput, timeAgo, copyButton, chips } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { invalidate } from '../core/store.js';
import { gated, emptyHint, avatar, userLabel, confirmDelete } from '../core/widgets.js';

export default async function render(ctx) {
  let q = '';
  let status = '';
  const body = h('div');
  mount(ctx.el, pageHeader({
    title: 'Users', icon: 'people', subtitle: 'Everyone with an n8n login.',
    actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => invite() }, icon('person-plus', 'me-1'), 'Invite')],
  }), h('div.n8s-wf-toolbar', searchInput({ placeholder: 'Search name or email…', onInput: (v) => { q = v.toLowerCase(); refresh(); } }), h('div.n8s-uchips')), body);

  let roles = ['global:admin', 'global:member'];
  n8n.get('/roles').then((r) => { roles = (r.global || []).map((x) => x.slug).filter((s) => s !== 'global:owner'); }).catch(() => null);

  const refresh = () => gated(body, async () => {
    const users = await n8n.all('/users', { includeRole: true }, { signal: ctx.signal });
    mount(ctx.el.querySelector('.n8s-uchips'), chips([{ value: '', label: 'All', count: users.length }, { value: 'pending', label: 'Pending invite', count: users.filter((u) => u.isPending).length }, { value: 'nomfa', label: 'No MFA', count: users.filter((u) => !u.mfaEnabled && !u.isPending).length }], status, (v) => { status = v; refresh(); }));
    const rows = users.filter((u) => (!q || `${u.email} ${u.firstName} ${u.lastName}`.toLowerCase().includes(q))
      && (!status || (status === 'pending' ? u.isPending : !u.mfaEnabled && !u.isPending)));
    return dataTable({
      rows, empty: emptyHint('people', 'No users match'),
      columns: [
        { key: 'name', label: 'User', render: (u) => h('span.d-flex.gap-2.align-items-center', avatar(u, 30), h('span', h('div.fw-semibold', userLabel(u)), h('div.small.text-muted', u.email))) },
        {
          key: 'role', label: 'Role', render: (u) => {
            if (u.role === 'global:owner') return h('span.badge.text-bg-dark', 'owner');
            const sel = h('select.form-select.form-select-sm', { style: 'width:170px' }, [...new Set([u.role, ...roles])].filter(Boolean).map((r) => h('option', { value: r, selected: r === u.role }, r.replace('global:', ''))));
            sel.addEventListener('change', async () => { try { await n8n.patch(`/users/${enc(u.id)}/role`, { newRoleName: sel.value }); toast('Role updated', 'success', { timeout: 1500 }); } catch (err) { toastError(err); sel.value = u.role; } });
            return sel;
          },
        },
        { key: 'status', label: 'Status', render: (u) => (u.isPending ? h('span.n8s-pill-warn', 'pending') : h('span', u.mfaEnabled ? h('span.text-success', icon('shield-check', 'me-1'), 'MFA') : h('span.text-muted', icon('shield', 'me-1'), 'no MFA'))) },
        { key: 'createdAt', label: 'Joined', render: (u) => timeAgo(u.createdAt) },
        { key: 'act', label: '', align: 'right', render: (u) => u.role !== 'global:owner' && h('button.btn.btn-sm.btn-outline-danger', { onclick: () => remove(u, users) }, icon('person-x')) },
      ],
    });
  });

  function invite() {
    const emails = h('textarea.form-control.form-control-sm', { rows: 4, placeholder: 'one@example.com, two@example.com\n…one per line or comma separated' });
    const role = h('select.form-select.form-select-sm', roles.map((r) => h('option', { value: r }, r.replace('global:', ''))));
    modal({
      title: 'Invite users', icon: 'person-plus',
      body: h('div', emails, h('label.small.fw-semibold.mt-2', 'Role'), role, h('div.small.text-muted.mt-1', 'Without SMTP configured in n8n, share the invite links shown next.')),
      actions: [{ label: 'Cancel' }, {
        label: 'Invite', variant: 'primary',
        onClick: async (close) => {
          const list = emails.value.split(/[\s,;]+/).map((x) => x.trim()).filter(Boolean);
          if (!list.length) throw new Error('Enter at least one email address.');
          const res = await n8n.post('/users', list.map((email) => ({ email, role: role.value })));
          close();
          invalidate('users');
          const arr = Array.isArray(res) ? res : [res];
          modal({
            title: 'Invitations', icon: 'envelope-check', size: 'lg',
            body: h('div', arr.map((r) => h('div.n8s-invite', h('b', r.user?.email || '?'), r.error ? h('span.text-danger', ` — ${r.error}`) : [
              h('span.small.text-muted', r.user?.emailSent ? ' — email sent' : ' — no email sent, share this link:'),
              r.user?.inviteAcceptUrl && h('div.d-flex.gap-2.align-items-center', h('code.small.text-break', r.user.inviteAcceptUrl), copyButton(r.user.inviteAcceptUrl))]))),
          });
          refresh();
        },
      }],
    });
  }

  async function remove(u, users) {
    const transfer = h('select.form-select.form-select-sm', h('option', { value: '' }, 'Delete their workflows & credentials'), users.filter((x) => x.id !== u.id && !x.isPending).map((x) => h('option', { value: x.id }, `Transfer to ${userLabel(x)}`)));
    if (!(await confirmDelete('user', userLabel(u), { details: h('div', h('label.small.fw-semibold', 'Their workflows and credentials'), transfer) }))) return;
    try {
      await n8n.del(`/users/${enc(u.id)}`, { params: transfer.value ? { transferId: transfer.value } : undefined });
      toast('User deleted', 'success');
      invalidate('users');
      refresh();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  refresh();
}
