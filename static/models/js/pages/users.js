// Users: people on this W&B Server, the viewer's API keys and service accounts.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, tabs, chips, searchInput, dataTable, formModal, confirmDialog, emptyState, errorState,
  skeleton, timeAgo, toastError, menuButton, statCard,
} from '../../../ui/js/ui.js';
import { gql, wandbLink, config } from '../core/api.js';
import { adminContext, avatar, pill, friendly, done, showSecretOnce, invalidateAdminContext } from './admin/shared.js';

const TABS = [
  { id: 'people', label: 'People', icon: 'people' },
  { id: 'keys', label: 'API keys', icon: 'key' },
  { id: 'service', label: 'Service accounts', icon: 'robot' },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  let tab = ctx.query.get('tab') || 'people';
  const ctxData = await adminContext({ signal });
  const me = ctxData.viewer;
  const body = h('div');
  const statsEl = h('div.af-stats');

  mount(el,
    pageHeader({
      title: 'Users', icon: 'person-badge',
      subtitle: 'Everyone with an account on this W&B Server, your API keys, and machine accounts.',
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => createUser() }, icon('person-plus', 'me-1'), 'Add user')],
    }),
    statsEl,
    tabs(TABS, tab, (t) => { tab = t; ctx.setQuery({ tab: t === 'people' ? null : t }); draw(); }),
    body);

  // ------------------------------------------------------------------ people

  let people = [];
  let filter = ctx.query.get('show') || 'active';
  let search = ctx.query.get('q') || '';

  async function loadPeople() {
    const d = await gql(`query Users($q: String) { users(first: 500, query: $q) { edges { node {
      id username name email admin accountType createdAt loggedInAt deletedAt photoUrl } } } }`, { q: search || null }, { signal });
    // Internal bootstrap accounts W&B creates on first start.
    people = (d.users?.edges || []).map((e) => e.node).filter((u) => !(u.deletedAt && ['restore', 'local'].includes(u.username) && /@wandb\.com$/.test(u.email || '')));
    return people;
  }

  function renderStats() {
    const active = people.filter((u) => !u.deletedAt);
    mount(statsEl,
      statCard({ label: 'Active users', value: active.length, icon: 'people', tone: 'blue' }),
      statCard({ label: 'Admins', value: active.filter((u) => u.admin).length, icon: 'shield-lock', tone: 'purple' }),
      statCard({ label: 'Signed in (30 days)', value: active.filter((u) => u.loggedInAt && Date.now() - new Date(u.loggedInAt) < 30 * 864e5).length, icon: 'box-arrow-in-right', tone: 'teal' }),
      statCard({ label: 'Deleted', value: people.filter((u) => u.deletedAt).length, icon: 'person-x', tone: 'slate' }));
  }

  async function drawPeople() {
    mount(body, card({ body: skeleton(6) }));
    try { await loadPeople(); } catch (err) { mount(body, errorState(err, drawPeople)); return; }
    if (!ctx.active) return;
    renderStats();
    const list = people.filter((u) => (filter === 'active' ? !u.deletedAt : filter === 'deleted' ? u.deletedAt : filter === 'admins' ? u.admin && !u.deletedAt : true));
    const table = dataTable({
      rows: list,
      rowKey: (u) => u.id,
      rowClass: (u) => (u.deletedAt ? 'msa-row-deleted' : u.id === me.id ? 'msa-row-me' : null),
      empty: emptyState({ icon: 'person', title: search ? 'No users match' : 'No users here', message: search ? 'Try a different name or email.' : null }),
      columns: [
        { key: 'username', label: 'User', render: (u) => h('div.msa-user', avatar(u, 32), h('div',
          h('div.fw-semibold', u.name || u.username || u.email, u.id === me.id && pill('you', 'info'), !u.username && !u.deletedAt && pill('hasn’t signed in yet', 'muted')),
          h('div.small.text-muted', [u.username && `@${u.username}`, u.email].filter(Boolean).join(' · ')))) },
        { key: 'role', label: 'Role', render: (u) => (u.deletedAt ? pill('deleted', 'bad', 'person-x') : u.admin ? pill('Instance admin', 'warn', 'shield-lock') : pill(String(u.accountType || 'user').toLowerCase(), 'muted')) },
        { key: 'createdAt', label: 'Joined', render: (u) => timeAgo(u.createdAt) },
        { key: 'loggedInAt', label: 'Last sign-in', render: (u) => (u.loggedInAt ? timeAgo(u.loggedInAt) : h('span.text-muted', 'never')) },
        { key: 'actions', label: '', align: 'right', render: (u) => userMenu(u) },
      ],
    });
    mount(body, card({
      title: `${list.length} ${list.length === 1 ? 'user' : 'users'}`, icon: 'people', flush: true,
      actions: [
        searchInput({ placeholder: 'Search name, username, email…', value: search, onInput: (v) => { search = v; ctx.setQuery({ q: v || null }); drawPeople(); } }),
        chips([{ value: 'active', label: 'Active' }, { value: 'admins', label: 'Admins' }, { value: 'deleted', label: 'Deleted' }, { value: 'all', label: 'All' }],
          filter, (v) => { filter = v || 'active'; ctx.setQuery({ show: filter === 'active' ? null : filter }); drawPeople(); }),
      ],
      body: table,
    }));
  }

  function userMenu(u) {
    const self = u.id === me.id;
    return menuButton('', () => [
      { header: `@${u.username}` },
      !u.deletedAt && {
        label: u.admin ? 'Remove instance admin' : 'Make instance admin', icon: u.admin ? 'shield-x' : 'shield-plus',
        disabled: self, onClick: () => setAdmin(u, !u.admin),
      },
      !u.deletedAt && { label: 'Copy username', icon: 'clipboard', onClick: () => navigator.clipboard.writeText(u.username).then(() => done('Copied')) },
      { divider: true },
      u.deletedAt
        ? { label: 'Restore user', icon: 'arrow-counterclockwise', onClick: () => restoreUser(u) }
        : { label: self ? 'Delete user (not yourself)' : 'Delete user', icon: 'person-x', danger: true, disabled: self, onClick: () => deleteUser(u) },
    ].filter(Boolean), { title: 'User actions' });
  }

  async function setAdmin(u, admin) {
    const ok = await confirmDialog({
      title: admin ? `Make @${u.username} an instance admin?` : `Remove admin from @${u.username}?`,
      message: admin ? 'Instance admins can manage every user, team and project on this server.' : 'They will keep their account and projects but lose server-wide admin rights.',
      confirmLabel: admin ? 'Make admin' : 'Remove admin', danger: !admin,
    });
    if (!ok) return;
    try {
      await gql('mutation A($input: UpdateUserInput!) { updateUser(input: $input) { user { id admin } } }', { input: { id: u.id, admin } });
      done(admin ? `@${u.username} is now an admin` : `@${u.username} is no longer an admin`);
      drawPeople();
    } catch (err) { toastError({ message: friendly(err) }, 'Could not change admin rights'); }
  }

  async function deleteUser(u) {
    const ok = await confirmDialog({
      title: `Delete @${u.username}?`, danger: true, confirmLabel: 'Delete user',
      message: 'They can no longer sign in. Their runs and projects stay, and you can restore the account later from the “Deleted” filter.',
    });
    if (!ok) return;
    try {
      await gql('mutation D($input: DeleteUserInput!) { deleteUser(input: $input) { user { id deletedAt } } }', { input: { id: u.id } });
      done(`@${u.username} deleted`);
      drawPeople();
    } catch (err) { toastError({ message: friendly(err) }, 'Delete failed'); }
  }

  async function restoreUser(u) {
    try {
      await gql('mutation U($input: UndeleteUserInput!) { undeleteUser(input: $input) { user { id deletedAt } } }', { input: { id: u.id } });
      done(`@${u.username} restored`);
      drawPeople();
    } catch (err) { toastError({ message: friendly(err) }, 'Restore failed'); }
  }

  function createUser() {
    formModal({
      title: 'Add a user', icon: 'person-plus',
      subtitle: 'Creates the account directly — this server has no email delivery, so nobody is emailed.',
      fields: [
        { name: 'email', label: 'Email', type: 'email', required: true, placeholder: 'person@afyaanalytics.com' },
        { name: 'password', label: 'Initial password', type: 'password', help: 'Optional. Leave empty and the user sets one via “Forgot password” / an admin reset.' },
        { name: 'admin', type: 'checkbox', checkLabel: 'Instance admin', value: false },
      ],
      submitLabel: 'Create user',
      onSubmit: async (v, close) => {
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v.email)) throw new Error('Enter a valid email address.');
        const input = { email: v.email.trim(), admin: !!v.admin };
        if (v.password) input.password = v.password;
        if (ctxData.org?.id) input.organizationID = ctxData.org.id;
        const d = await gql('mutation C($input: CreateUserInput!) { createUser(input: $input) { user { id username email } } }', { input });
        close();
        done(`Created ${d.createUser?.user?.username || v.email}`);
        invalidateAdminContext();
        if (tab === 'people') drawPeople();
      },
    });
  }

  // ------------------------------------------------------------------ API keys

  async function drawKeys() {
    mount(body, card({ body: skeleton(4) }));
    let keys;
    try {
      const d = await gql('query K { viewer { id apiKeys { edges { node { id apiKeyId name description createdAt lastUsedAt expiresAt } } } } }', {}, { signal });
      keys = (d.viewer?.apiKeys?.edges || []).map((e) => e.node);
    } catch (err) {
      if (/relogin required/i.test(err.message)) {
        // W&B only lists API keys when the account signed in to the W&B web UI recently.
        mount(statsEl);
        mount(body, card({
          title: `API keys for @${me.username}`, icon: 'key',
          body: h('div.d-flex.flex-column.gap-3',
            h('div.af-callout.af-callout-warn', icon('shield-lock'), h('span',
              h('b', 'W&B wants a fresh sign-in before showing API keys. '),
              'For security, W&B Server only lists keys when ', h('code', `@${me.username}`), ' has signed in to the W&B web app recently. ',
              'Sign in there once, then come back and reload this tab.')),
            h('div.d-flex.gap-2.flex-wrap',
              wandbLink('/login') && h('a.btn.btn-sm.btn-primary', { href: wandbLink('/login'), target: '_blank', rel: 'noopener' }, icon('box-arrow-up-right', 'me-1'), 'Sign in to W&B'),
              h('button.btn.btn-sm.btn-outline-secondary', { onclick: drawKeys }, icon('arrow-clockwise', 'me-1'), 'Reload keys'),
              h('button.btn.btn-sm.btn-outline-primary', { onclick: generateKey }, icon('plus-lg', 'me-1'), 'Generate key anyway'))),
        }));
        return;
      }
      mount(body, errorState(err, drawKeys));
      return;
    }
    if (!ctx.active) return;
    // The page config names the key Model Studio connects with (its non-secret id). Classic keys
    // aren't exposed, so fall back to the most recently used one. The proxy also refuses (409).
    const known = config.apiKeyId;
    const newest = keys.reduce((a, k) => (k.lastUsedAt && (!a || k.lastUsedAt > a.lastUsedAt) ? k : a), null);
    const inUse = new Set(keys.filter((k) => (known ? (k.apiKeyId || k.name) === known : k === newest)).map((k) => k.id));
    mount(statsEl,
      statCard({ label: 'Your API keys', value: keys.length, icon: 'key', tone: 'blue' }),
      statCard({ label: 'Used in last 7 days', value: keys.filter((k) => k.lastUsedAt && Date.now() - new Date(k.lastUsedAt) < 7 * 864e5).length, icon: 'activity', tone: 'teal' }),
      statCard({ label: 'Never used', value: keys.filter((k) => !k.lastUsedAt).length, icon: 'hourglass', tone: 'amber' }));
    mount(body, card({
      title: `API keys for @${me.username}`, icon: 'key', flush: true,
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: generateKey }, icon('plus-lg', 'me-1'), 'Generate key')],
      body: h('div',
        h('div.af-callout.af-callout-info.m-3', icon('info-circle'), h('span',
          'Model Studio talks to W&B with one of these keys. It is marked ', pill('In use by Model Studio', 'ok', 'plug-fill'),
          ' and cannot be deleted here — rotate it by generating a new key, updating WANDB_API_KEY in .env, restarting, then deleting the old one.')),
        dataTable({
          rows: keys, rowKey: (k) => k.id,
          empty: emptyState({ icon: 'key', title: 'No API keys', message: 'Generate one to log runs from scripts and notebooks.' }),
          columns: [
            { key: 'name', label: 'Key', render: (k) => h('div', h('div.ms-mono.fw-semibold', `${k.name}…`), h('div.small.text-muted', k.description || 'No description')) },
            { key: 'status', label: '', render: (k) => (inUse.has(k.id) ? pill('In use by Model Studio', 'ok', 'plug-fill') : null) },
            { key: 'createdAt', label: 'Created', render: (k) => timeAgo(k.createdAt) },
            { key: 'lastUsedAt', label: 'Last used', render: (k) => (k.lastUsedAt ? timeAgo(k.lastUsedAt) : h('span.text-muted', 'never')) },
            { key: 'actions', label: '', align: 'right', render: (k) => h('button.btn.btn-sm.btn-outline-danger', {
              disabled: inUse.has(k.id), title: inUse.has(k.id) ? 'This key is how Model Studio connects to W&B' : 'Delete key',
              onclick: () => deleteKey(k),
            }, icon('trash'), ' Delete') },
          ],
        })),
    }));
  }

  function generateKey() {
    formModal({
      title: 'Generate an API key', icon: 'key',
      fields: [{ name: 'description', label: 'What is it for?', placeholder: 'e.g. training server, CI pipeline', required: true }],
      submitLabel: 'Generate',
      onSubmit: async (v, close) => {
        const input = { description: v.description.trim() };
        const d = await gql('mutation G($input: GenerateApiKeyInput!) { generateApiKey(input: $input) { secretApiKey apiKey { id name } } }', { input });
        close();
        const secret = d.generateApiKey?.secretApiKey || d.generateApiKey?.apiKey?.name;
        await showSecretOnce({
          title: 'Your new API key', secret,
          note: 'Copy it now — W&B will only ever show the first part again. Use it as WANDB_API_KEY with WANDB_BASE_URL pointing at this server.',
        });
        drawKeys();
      },
    });
  }

  async function deleteKey(k) {
    const ok = await confirmDialog({
      title: 'Delete this API key?', danger: true, confirmLabel: 'Delete key',
      message: h('span', 'Anything still using ', h('code', `${k.name}…`), ' (scripts, CI, notebooks) will stop being able to log to W&B.'),
    });
    if (!ok) return;
    try {
      await gql('mutation D($input: DeleteApiKeyInput!) { deleteApiKey(input: $input) { success } }', { input: { id: k.id } });
      done('API key deleted');
      drawKeys();
    } catch (err) { toastError({ message: friendly(err) }, 'Could not delete key'); }
  }

  // ------------------------------------------------------------------ service accounts

  async function drawService() {
    mount(body, card({ body: skeleton(4) }));
    let accounts = [];
    try {
      const d = await gql(`query S { viewer { organizations { id name serviceAccounts { id name createdAt apiKeyCount accountType
        defaultEntity { name } teams(first: 20) { edges { node { name } } } } } } }`, {}, { signal });
      accounts = (d.viewer?.organizations || []).flatMap((o) => o.serviceAccounts || []);
    } catch (err) { mount(body, errorState(err, drawService)); return; }
    if (!ctx.active) return;
    mount(statsEl,
      statCard({ label: 'Service accounts', value: accounts.length, icon: 'robot', tone: 'purple' }),
      statCard({ label: 'Their API keys', value: accounts.reduce((a, s) => a + (s.apiKeyCount || 0), 0), icon: 'key', tone: 'blue' }));
    mount(body, card({
      title: 'Service accounts', icon: 'robot', flush: true,
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: createService }, icon('plus-lg', 'me-1'), 'New service account')],
      body: h('div',
        h('div.af-callout.af-callout-info.m-3', icon('lightbulb'), h('span', 'Use a service account for pipelines (Airflow, CI) so runs are not tied to a person who might leave.')),
        dataTable({
          rows: accounts, rowKey: (s) => s.id,
          empty: emptyState({ icon: 'robot', title: 'No service accounts yet', message: 'Create one for automated training jobs.' }),
          columns: [
            { key: 'name', label: 'Account', render: (s) => h('div.msa-user', h('span.msa-avatar.msa-initials.msa-bot', icon('robot')), h('div', h('div.fw-semibold', s.name), h('div.small.text-muted', s.defaultEntity?.name ? `default team: ${s.defaultEntity.name}` : ''))) },
            { key: 'teams', label: 'Teams', render: (s) => h('div.d-flex.gap-1.flex-wrap', (s.teams?.edges || []).map((e) => pill(e.node.name, 'info'))) },
            { key: 'apiKeyCount', label: 'Keys', render: (s) => String(s.apiKeyCount ?? 0) },
            { key: 'createdAt', label: 'Created', render: (s) => timeAgo(s.createdAt) },
            { key: 'actions', label: '', align: 'right', render: (s) => h('button.btn.btn-sm.btn-outline-danger', { onclick: () => deleteService(s) }, icon('trash')) },
          ],
        })),
    }));
  }

  function createService() {
    const teams = ctxData.entities.filter((e) => e.isTeam);
    formModal({
      title: 'New service account', icon: 'robot',
      fields: [
        { name: 'description', label: 'Name / purpose', required: true, placeholder: 'airflow-training' },
        { name: 'entityName', label: 'Team', type: 'select', options: teams.length ? teams.map((t) => t.name) : [{ value: '', label: '— no teams yet (create one first) —' }] },
      ],
      submitLabel: 'Create',
      onSubmit: async (v, close) => {
        if (!v.entityName) throw new Error('Service accounts belong to a team. Create a team first on the Teams page.');
        const input = { description: v.description.trim(), entityName: v.entityName };
        if (ctxData.org?.id) input.organizationId = ctxData.org.id;
        await gql('mutation C($input: CreateServiceAccountInput!) { createServiceAccount(input: $input) { user { id name } } }', { input });
        close();
        done('Service account created');
        drawService();
      },
    });
  }

  async function deleteService(s) {
    const ok = await confirmDialog({ title: `Delete service account “${s.name}”?`, danger: true, confirmLabel: 'Delete', message: 'Its API keys stop working immediately.' });
    if (!ok) return;
    try {
      await gql('mutation D($input: DeleteServiceAccountInput!) { deleteServiceAccount(input: $input) { user { id } } }', { input: { id: s.id } });
      done('Service account deleted');
      drawService();
    } catch (err) { toastError({ message: friendly(err) }, 'Delete failed'); }
  }

  function draw() {
    mount(statsEl);
    if (tab === 'keys') drawKeys();
    else if (tab === 'service') drawService();
    else drawPeople();
  }
  draw();
}
