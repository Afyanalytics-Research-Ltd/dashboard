// One team (or personal entity): members & roles, invites, projects, settings, danger zone.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, tabs, dataTable, buildForm, confirmDialog, emptyState, errorState, skeleton, timeAgo, toastError,
  statCard, kv,
} from '../../../ui/js/ui.js';
import { fmtBytes, fmtNumber } from '../../../ui/js/format.js';
import { gql, collect } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../../../ui/js/router.js';
import { adminContext, invalidateAdminContext, avatar, pill, friendly, done, typedConfirm, userSearch } from './admin/shared.js';

const ROLES = [
  { value: 'admin', label: 'Admin', help: 'Manage members, settings and delete projects' },
  { value: 'member', label: 'Member', help: 'Create projects and log runs' },
  { value: 'viewer', label: 'Viewer', help: 'Read-only access' },
];

export default async function render(ctx) {
  const { el, signal, params } = ctx;
  const name = params.entity;
  let tab = ctx.query.get('tab') || 'members';
  const body = h('div');
  const head = h('div');
  const stats = h('div.af-stats');
  mount(el, head, stats, body);

  let entity;
  const c = await adminContext({ signal });
  async function load() {
    const d = await gql(`query Team($n: String!) { entity(name: $n) { id name isTeam photoUrl defaultAccess privateOnly codeSavingEnabled
      createdAt storageBytes projectCount memberCount readOnly
      members { id username name email photoUrl role accountType pending apiKeyCount }
      invites(first: 100) { edges { node { id email name role createdAt toUser { username email } fromUser { username } } } } } }`, { n: name }, { signal });
    // W&B answers unknown names with a placeholder entity whose id decodes to "Entity:-1".
    let placeholder = false;
    try { placeholder = atob(d.entity?.id || '').endsWith(':-1'); } catch { /* not base64 */ }
    if (!d.entity || placeholder) throw Object.assign(new Error(`No team or user called “${name}”.`), { status: 404 });
    entity = d.entity;
  }

  function header() {
    mount(head, pageHeader({
      title: entity.name, icon: entity.isTeam ? 'people-fill' : 'person-fill',
      crumbs: [{ label: 'Teams', href: urls.teams() }, { label: entity.name }],
      subtitle: entity.isTeam ? `Team · created ${new Date(entity.createdAt).toLocaleDateString()}` : 'Personal space',
      badges: [pill((entity.defaultAccess || 'PRIVATE').toLowerCase(), 'muted', 'lock')],
    }));
    mount(stats,
      statCard({ label: 'Members', value: fmtNumber(entity.members.filter((m) => !m.pending).length), icon: 'people', tone: 'blue' }),
      statCard({ label: 'Pending invites', value: fmtNumber((entity.invites?.edges || []).length + entity.members.filter((m) => m.pending).length), icon: 'envelope', tone: 'amber' }),
      statCard({ label: 'Projects', value: fmtNumber(entity.projectCount), icon: 'folder2', tone: 'teal' }),
      statCard({ label: 'Storage', value: fmtBytes(entity.storageBytes || 0), icon: 'hdd', tone: 'purple' }));
  }

  const TABS = () => [
    { id: 'members', label: 'Members', icon: 'people', count: entity.members.length },
    { id: 'projects', label: 'Projects', icon: 'folder2', count: entity.projectCount },
    { id: 'settings', label: 'Settings', icon: 'gear' },
  ];

  function draw() {
    header();
    mount(body, tabs(TABS(), tab, (t) => { tab = t; ctx.setQuery({ tab: t === 'members' ? null : t }); draw(); }), h('div.msa-tab-body', panel()));
  }
  function panel() {
    if (tab === 'projects') return projectsPanel();
    if (tab === 'settings') return settingsPanel();
    return membersPanel();
  }

  // ------------------------------------------------------------------ members

  function membersPanel() {
    const me = c.viewer;
    const mine = entity.members.find((m) => m.username === me.username);
    const canManage = me.admin || mine?.role === 'admin';
    const add = entity.isTeam && canManage ? userSearch((u) => addMember(u), { exclude: (u) => entity.members.some((m) => m.username === u.username) }) : null;
    const invites = (entity.invites?.edges || []).map((e) => e.node);
    return h('div.d-flex.flex-column.gap-3',
      card({
        title: 'Members', icon: 'people', flush: true,
        actions: add && h('div.msa-add-member', add.el),
        body: dataTable({
          rows: entity.members, rowKey: (m) => m.id,
          empty: emptyState({ icon: 'people', title: 'No members' }),
          columns: [
            { key: 'user', label: 'Member', render: (m) => h('div.msa-user', avatar(m, 30), h('div',
              h('div.fw-semibold', m.name || m.username || m.email, m.username === me.username && pill('you', 'info'), m.pending && pill('invite pending', 'warn'),
                !m.username && pill('hasn’t signed in yet', 'muted')),
              h('div.small.text-muted', [m.username && `@${m.username}`, m.email].filter(Boolean).join(' · ')))) },
            { key: 'role', label: 'Role', render: (m) => (entity.isTeam && canManage && m.username !== me.username
              ? roleSelect(m) : pill(m.role || 'member', m.role === 'admin' ? 'warn' : 'muted')) },
            { key: 'keys', label: 'API keys', render: (m) => String(m.apiKeyCount ?? '—') },
            { key: 'actions', label: '', align: 'right', render: (m) => (entity.isTeam && canManage && m.username !== me.username
              ? h('button.btn.btn-sm.btn-outline-danger', { title: 'Remove from team', onclick: () => removeMember(m) }, icon('person-dash'))
              : null) },
          ],
        }),
      }),
      invites.length ? card({
        title: 'Pending invitations', icon: 'envelope', flush: true,
        body: dataTable({
          rows: invites, rowKey: (i) => i.id,
          columns: [
            { key: 'who', label: 'Invitee', render: (i) => (i.toUser?.username ? `@${i.toUser.username}` : i.email || i.toUser?.email || i.name) },
            { key: 'role', label: 'Role', render: (i) => pill(i.role, 'muted') },
            { key: 'from', label: 'Invited by', render: (i) => i.fromUser?.username || '—' },
            { key: 'createdAt', label: 'Sent', render: (i) => timeAgo(i.createdAt) },
            { key: 'x', label: '', align: 'right', render: (i) => (canManage ? h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => cancelInvite(i) }, icon('x-lg'), ' Cancel') : null) },
          ],
        }),
      }) : null,
      !entity.isTeam && h('div.af-callout.af-callout-info', icon('info-circle'), h('span', 'This is a personal space — it has a single owner. Create a team to collaborate.')));
  }

  function roleSelect(m) {
    const sel = h('select.form-select.form-select-sm.msa-role', { 'aria-label': `Role for ${m.username}` },
      ROLES.map((r) => h('option', { value: r.value, selected: r.value === (m.role || 'member'), title: r.help }, r.label)));
    sel.addEventListener('change', async () => {
      const prev = m.role;
      try {
        await gql('mutation R($input: UpdateMemberInput!) { updateMember(input: $input) { member { id role } } }',
          { input: { entityName: entity.name, user: m.id, role: sel.value, admin: sel.value === 'admin' } });
        m.role = sel.value;
        done(`@${m.username} is now ${sel.value}`);
        sel.classList.add('msa-flash');
        setTimeout(() => sel.classList.remove('msa-flash'), 900);
      } catch (err) { sel.value = prev || 'member'; toastError({ message: friendly(err) }, 'Could not change role'); }
    });
    return sel;
  }

  async function addMember(u) {
    // Admin-created users have no username until their first sign-in; invite those by email.
    const who = u.username ? { username: u.username } : { email: u.email };
    try {
      const d = await gql('mutation I($input: CreateInviteInput!) { createInvite(input: $input) { success invite { id } } }',
        { input: { entityName: entity.name, role: 'member', admin: false, ...who } });
      if (d.createInvite?.success === false) throw new Error('W&B did not accept the invitation.');
      done(u.username ? `Added @${u.username}` : `Invited ${u.email} — they join the team when they first sign in`);
      await reload();
    } catch (err) { toastError({ message: friendly(err) }, 'Could not add member'); }
  }

  async function removeMember(m) {
    const ok = await confirmDialog({
      title: `Remove ${m.username ? `@${m.username}` : m.email} from ${entity.name}?`, danger: true, confirmLabel: 'Remove',
      message: 'They lose access to the team’s private projects. Their runs stay in the projects.',
    });
    if (!ok) return;
    const del = (id) => gql('mutation D($input: DeleteInviteInput!) { deleteInvite(input: $input) { success } }', { input: { id, entityName: entity.name } });
    try {
      // An invitee who never signed in shows up both as a member and as a pending invitation;
      // cancel the invitation too or they would still join on first sign-in.
      const pendingInvites = (entity.invites?.edges || []).map((e) => e.node)
        .filter((i) => m.email && (i.email === m.email || i.toUser?.email === m.email));
      for (const i of pendingInvites) await del(i.id);
      // W&B removes a team member through deleteInvite with the member id.
      await del(m.id).catch((err) => { if (!pendingInvites.length) throw err; });
      done(`Removed ${m.username ? `@${m.username}` : m.email}`);
      await reload();
    } catch (err) { toastError({ message: friendly(err) }, 'Could not remove member'); }
  }

  async function cancelInvite(i) {
    try {
      await gql('mutation D($input: DeleteInviteInput!) { deleteInvite(input: $input) { success } }', { input: { id: i.id, entityName: entity.name } });
      done('Invitation cancelled');
      await reload();
    } catch (err) { toastError({ message: friendly(err) }, 'Could not cancel'); }
  }

  // ------------------------------------------------------------------ projects

  function projectsPanel() {
    const host = h('div', skeleton(5));
    collect((after) => gql(`query P($e: String!, $after: String) { projects(entityName: $e, first: 100, after: $after, order: "-lastActive") {
      pageInfo { hasNextPage endCursor } edges { node { id name description totalRuns lastActive createdAt access } } } }`,
    { e: entity.name, after }, { signal }), (d) => d.projects).then((projects) => {
      if (!ctx.active) return;
      mount(host, card({
        title: 'Projects', icon: 'folder2', flush: true,
        body: dataTable({
          rows: projects, rowKey: (p) => p.id,
          onRowClick: (p) => navigate(urls.workspace(entity.name, p.name)),
          empty: emptyState({ icon: 'folder2', title: 'No projects yet', message: h('span', 'Log a run with ', h('code', `wandb.init(entity="${entity.name}", project="…")`)) }),
          columns: [
            { key: 'name', label: 'Project', render: (p) => h('div', h('a.fw-semibold', { href: urls.workspace(entity.name, p.name) }, p.name), p.description && h('div.small.text-muted', p.description)) },
            { key: 'totalRuns', label: 'Runs', align: 'right', render: (p) => fmtNumber(p.totalRuns || 0) },
            { key: 'access', label: 'Visibility', render: (p) => pill((p.access || 'private').toLowerCase(), 'muted') },
            { key: 'lastActive', label: 'Last active', render: (p) => timeAgo(p.lastActive) },
            { key: 'settings', label: '', align: 'right', render: (p) => h('a.btn.btn-sm.btn-outline-secondary', { href: urls.projectSettings(entity.name, p.name), title: 'Project settings' }, icon('gear')) },
          ],
        }),
      }));
    }).catch((err) => mount(host, errorState(err)));
    return host;
  }

  // ------------------------------------------------------------------ settings

  function settingsPanel() {
    const form = buildForm([
      { name: 'privateOnly', type: 'checkbox', checkLabel: 'Force all projects to stay private', value: !!entity.privateOnly },
      { name: 'defaultAccess', label: 'Default visibility for new projects', type: 'select', value: entity.defaultAccess || 'PRIVATE',
        options: [{ value: 'PRIVATE', label: 'Private — members only' }, { value: 'USER_READ', label: 'Readable by every user on this server' }],
        help: 'Only applies while “Force all projects to stay private” is off — W&B keeps every project private while it is on.' },
      { name: 'codeSavingEnabled', type: 'checkbox', checkLabel: 'Save code (git diff, notebooks) with runs by default', value: !!entity.codeSavingEnabled },
    ]);
    const privateBox = form.el.querySelector('input[type=checkbox]');
    const accessSel = form.el.querySelector('select');
    const syncAccess = () => {
      accessSel.disabled = privateBox.checked;
      if (privateBox.checked) accessSel.value = 'PRIVATE';
    };
    privateBox.addEventListener('change', syncAccess);
    syncAccess();
    const save = h('button.btn.btn-sm.btn-primary', {
      onclick: async () => {
        const v = form.values();
        if (v.privateOnly) v.defaultAccess = 'PRIVATE';
        save.disabled = true;
        try {
          const d = await gql('mutation U($input: UpdateEntityInput!) { updateEntity(input: $input) { entity { id defaultAccess privateOnly codeSavingEnabled } } }',
            { input: { entity: entity.name, defaultAccess: v.defaultAccess, privateOnly: v.privateOnly, codeSavingEnabled: v.codeSavingEnabled } });
          Object.assign(entity, d.updateEntity?.entity || v);
          done('Settings saved');
          header();
        } catch (err) { toastError({ message: friendly(err) }, 'Save failed'); } finally { save.disabled = false; }
      },
    }, icon('check2', 'me-1'), 'Save settings');
    return h('div.d-flex.flex-column.gap-3',
      card({ title: 'Settings', icon: 'gear', body: h('div', form.el, h('div.mt-3', save)) }),
      card({ title: 'Details', icon: 'info-circle', cls: 'msa-kv-wrap', body: kv([
        ['Entity name', h('code', entity.name)], ['Type', entity.isTeam ? 'Team' : 'Personal'],
        ['Created', new Date(entity.createdAt).toLocaleString()], ['Storage', fmtBytes(entity.storageBytes || 0)],
        ['SDK usage', h('code', `wandb.init(entity="${entity.name}", project="my-project")`), { wide: true }],
      ]) }),
      entity.isTeam && card({
        title: 'Danger zone', icon: 'exclamation-octagon', cls: 'msa-danger',
        body: h('div.msa-danger-row',
          h('div', h('div.fw-semibold', 'Delete this team'), h('div.small.text-muted', 'Removes the team. Its projects must be moved or deleted first.')),
          h('button.btn.btn-sm.btn-danger', { onclick: deleteTeam }, icon('trash', 'me-1'), 'Delete team')),
      }));
  }

  async function deleteTeam() {
    if (entity.projectCount > 0) {
      const ok = await confirmDialog({ title: 'This team still has projects', message: `Move or delete its ${entity.projectCount} project(s) first (Project settings → Move / Delete). W&B refuses to delete a team that owns projects.`, confirmLabel: 'Show projects' });
      if (ok) { tab = 'projects'; draw(); }
      return;
    }
    const ok = await typedConfirm({ title: `Delete team ${entity.name}?`, expected: entity.name, confirmLabel: 'Delete team', message: 'This cannot be undone. Members lose access immediately.' });
    if (!ok) return;
    try {
      await gql('mutation D($input: DeleteTeamInput!) { deleteTeam(input: $input) { success } }', { input: { teamName: entity.name } });
      invalidateAdminContext();
      done(`Team ${entity.name} deleted`);
      navigate(urls.teams());
    } catch (err) { toastError({ message: friendly(err) }, 'Delete failed'); }
  }

  async function reload() { await load(); if (ctx.active) draw(); }

  mount(body, skeleton(6));
  try { await load(); } catch (err) { mount(body, errorState(err, () => render(ctx))); return; }
  if (!ctx.active) return;
  draw();
}
