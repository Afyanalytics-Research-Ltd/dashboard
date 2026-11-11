// Teams: every team on the server (admins) plus your personal entity, as cards.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, formModal, emptyState, errorState, skeleton, searchInput, statCard } from '../../../ui/js/ui.js';
import { fmtBytes } from '../../../ui/js/format.js';
import { gql } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../../../ui/js/router.js';
import { adminContext, invalidateAdminContext, pill, done } from './admin/shared.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  const grid = h('div.msa-team-grid', skeleton(3, { height: 120 }));
  const stats = h('div.af-stats');
  let q = ctx.query.get('q') || '';
  let teams = [];

  mount(el,
    pageHeader({
      title: 'Teams', icon: 'people',
      subtitle: 'Teams own projects and artifacts together. Your personal space is listed too.',
      actions: [
        searchInput({ placeholder: 'Filter teams…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v || null }); draw(); } }),
        h('button.btn.btn-sm.btn-primary', { onclick: createTeam }, icon('plus-lg', 'me-1'), 'New team'),
      ],
    }),
    stats, grid);

  async function load() {
    const c = await adminContext({ fresh: true, signal });
    // Instance admins see every team through the organization; others see their own.
    let all = c.entities;
    try {
      const d = await gql(`query OrgTeams { viewer { organizations { id teams { id name isTeam photoUrl memberCount projectCount storageBytes defaultAccess createdAt } } } }`, {}, { signal });
      const orgTeams = (d.viewer?.organizations || []).flatMap((o) => o.teams || []);
      const byName = new Map(all.map((t) => [t.name, t]));
      orgTeams.forEach((t) => byName.set(t.name, { ...byName.get(t.name), ...t }));
      all = [...byName.values()];
    } catch { /* fall back to viewer teams */ }
    teams = all.sort((a, b) => (a.isTeam === b.isTeam ? a.name.localeCompare(b.name) : a.isTeam ? 1 : -1));
    return c;
  }

  function draw() {
    const list = teams.filter((t) => !q || t.name.toLowerCase().includes(q.toLowerCase()));
    const real = teams.filter((t) => t.isTeam);
    mount(stats,
      statCard({ label: 'Teams', value: real.length, icon: 'people', tone: 'blue' }),
      statCard({ label: 'Projects', value: teams.reduce((a, t) => a + (t.projectCount || 0), 0), icon: 'folder2', tone: 'teal' }),
      statCard({ label: 'Storage used', value: fmtBytes(teams.reduce((a, t) => a + (t.storageBytes || 0), 0)), icon: 'hdd', tone: 'purple' }));
    if (!list.length) {
      mount(grid, emptyState({ icon: 'people', title: q ? 'No teams match' : 'No teams yet', message: 'Create a team to share projects, artifacts and the model registry.',
        action: h('button.btn.btn-sm.btn-primary', { onclick: createTeam }, icon('plus-lg', 'me-1'), 'New team') }));
      return;
    }
    mount(grid, list.map((t, i) => h('a.msa-team-card', { href: urls.team(t.name), style: { animationDelay: `${Math.min(i, 10) * 0.04}s` } },
      h('div.msa-team-head',
        h('span.msa-team-icon', { class: !t.isTeam && 'personal' }, icon(t.isTeam ? 'people-fill' : 'person-fill')),
        h('div.msa-team-titles', h('div.msa-team-name', t.name), h('div.small.text-muted', t.isTeam ? 'Team' : 'Personal space')),
        pill((t.defaultAccess || 'private').toLowerCase().replace('_', ' '), t.defaultAccess === 'PUBLIC' ? 'warn' : 'muted', t.defaultAccess === 'PUBLIC' ? 'globe' : 'lock')),
      h('div.msa-team-stats',
        h('div', h('b', String(t.memberCount ?? '—')), h('span', 'members')),
        h('div', h('b', String(t.projectCount ?? 0)), h('span', 'projects')),
        h('div', h('b', fmtBytes(t.storageBytes || 0)), h('span', 'storage'))))));
  }

  function createTeam() {
    formModal({
      title: 'Create a team', icon: 'people',
      fields: [
        { name: 'teamName', label: 'Team name', required: true, placeholder: 'e.g. kisumu-ml', help: 'Lowercase letters, numbers and dashes. This becomes the entity name in URLs and wandb.init(entity=…).' },
        { name: 'defaultAccess', label: 'Default project visibility', type: 'select', value: 'PRIVATE', options: [{ value: 'PRIVATE', label: 'Private — team members only' }, { value: 'USER_READ', label: 'Visible to all users on this server' }] },
      ],
      submitLabel: 'Create team',
      onSubmit: async (v, close) => {
        const name = v.teamName.trim();
        if (!/^[a-z0-9][a-z0-9-_]{1,62}$/.test(name)) throw new Error('Use 2–63 lowercase letters, numbers, dashes or underscores.');
        const c = await adminContext({ signal });
        const input = { teamName: name, defaultAccess: v.defaultAccess, teamAdminUserName: c.viewer.username };
        if (c.org?.id) input.organizationId = c.org.id;
        await gql('mutation C($input: CreateTeamInput!) { createTeam(input: $input) { entity { id name } } }', { input });
        close();
        invalidateAdminContext();
        done(`Team ${name} created`);
        navigate(urls.team(name));
      },
    });
  }

  try { await load(); } catch (err) { mount(grid, errorState(err, () => render(ctx))); return; }
  if (!ctx.active) return;
  draw();
  if (ctx.query.get('new')) createTeam();
}
