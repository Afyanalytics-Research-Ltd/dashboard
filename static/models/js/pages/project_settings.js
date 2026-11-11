// Project settings: details & visibility, rename, move between personal space and teams, delete.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, kv, buildForm, errorState, skeleton, statCard, toastError, confirmDialog, modal } from '../../../ui/js/ui.js';
import { fmtBytes, fmtNumber, fmtDuration } from '../../../ui/js/format.js';
import { gql } from '../core/api.js';
import { urls } from '../core/urls.js';
import { navigate } from '../../../ui/js/router.js';
import { adminContext, friendly, done, typedConfirm, waitForTask, pill } from './admin/shared.js';

export const ACCESS = [
  { value: 'PRIVATE', label: 'Private', help: 'Only you (or team members of the owning team).', icon: 'lock' },
  { value: 'ENTITY_READ', label: 'Team read', help: 'Team members can view; only writers can change.', icon: 'people' },
  { value: 'USER_READ', label: 'Open (read)', help: 'Every user on this server can view.', icon: 'eye' },
  { value: 'USER_WRITE', label: 'Open (write)', help: 'Every user on this server can view and log runs.', icon: 'pencil-square' },
];

const NAME_RE = /^[A-Za-z0-9_.-]{1,128}$/;

export default async function render(ctx) {
  const { el, signal, params } = ctx;
  const { entity, project } = params;
  const body = h('div.d-flex.flex-column.gap-3', skeleton(8));
  const stats = h('div.af-stats');
  mount(el,
    pageHeader({
      title: 'Project settings', icon: 'gear',
      crumbs: [{ label: 'Projects', href: urls.projects() }, { label: entity }, { label: project, href: urls.workspace(entity, project) }, { label: 'Settings' }],
    }),
    stats, body);

  let p;
  try {
    const d = await gql(`query PS($e: String!, $p: String!) { project(entityName: $e, name: $p) { id name entityName description access createdAt updatedAt lastActive
      totalRuns storageBytes computeHours readOnly user { username } } }`, { e: entity, p: project }, { signal });
    p = d.project;
    if (!p) throw Object.assign(new Error(`Project ${entity}/${project} not found`), { status: 404 });
  } catch (err) { mount(body, errorState(err, () => render(ctx))); return; }
  if (!ctx.active) return;
  const c = await adminContext({ signal });

  mount(stats,
    statCard({ label: 'Runs', value: fmtNumber(p.totalRuns || 0), icon: 'play-circle', tone: 'blue', href: urls.table(entity, project) }),
    statCard({ label: 'Storage', value: fmtBytes(p.storageBytes || 0), icon: 'hdd', tone: 'purple' }),
    statCard({ label: 'Compute', value: p.computeHours != null ? fmtDuration(Number(p.computeHours) * (Number(p.computeHours) > 1e5 ? 1 : 3600)) : '—', icon: 'cpu', tone: 'teal' }),
    statCard({ label: 'Last active', value: p.lastActive ? new Date(p.lastActive).toLocaleDateString() : '—', icon: 'clock-history', tone: 'slate' }));

  // ------------------------------------------------------------------ details & visibility

  const details = buildForm([
    { name: 'description', label: 'Description', type: 'textarea', rows: 3, value: p.description || '', placeholder: 'What is this project about? Shown on the projects list.' },
    { name: 'access', label: 'Visibility', type: 'select', value: p.access || 'PRIVATE', options: ACCESS.map((a) => ({ value: a.value, label: `${a.label} — ${a.help}` })) },
  ]);
  const saveDetails = h('button.btn.btn-sm.btn-primary', {
    onclick: async () => {
      const v = details.values();
      saveDetails.disabled = true;
      try {
        await gql('mutation U($input: UpsertModelInput!) { upsertModel(input: $input) { project { id description access } } }',
          { input: { entityName: entity, name: project, description: v.description, access: v.access } });
        Object.assign(p, v);
        done('Project details saved');
      } catch (err) { toastError({ message: friendly(err) }, 'Save failed'); } finally { saveDetails.disabled = false; }
    },
  }, icon('check2', 'me-1'), 'Save');

  // ------------------------------------------------------------------ rename

  const renameInput = h('input.form-control.form-control-sm.ms-mono', { value: p.name, 'aria-label': 'New project name' });
  const renameBtn = h('button.btn.btn-sm.btn-outline-primary', {
    onclick: async () => {
      const next = renameInput.value.trim();
      if (next === p.name) return;
      if (!NAME_RE.test(next)) { toastError({ message: 'Use letters, numbers, dots, dashes or underscores (max 128).' }, 'Invalid name'); return; }
      const ok = await confirmDialog({
        title: `Rename to “${next}”?`, confirmLabel: 'Rename',
        message: h('span', 'Training scripts must use the new name: ', h('code', `wandb.init(project="${next}")`), '. Old links stop working.'),
      });
      if (!ok) return;
      try {
        await gql('mutation R($input: RenameProjectInput!) { renameProject(input: $input) { project { id name } } }',
          { input: { entityName: entity, oldProjectName: p.name, newProjectName: next } });
        done(`Renamed to ${next}`);
        navigate(urls.projectSettings(entity, next), { replace: true });
      } catch (err) { toastError({ message: friendly(err) }, 'Rename failed'); }
    },
  }, icon('input-cursor-text', 'me-1'), 'Rename');

  // ------------------------------------------------------------------ move

  const destinations = c.entities.filter((e) => e.name !== entity);
  const moveSel = h('select.form-select.form-select-sm', { 'aria-label': 'Destination' },
    destinations.length ? destinations.map((e) => h('option', { value: e.name }, `${e.name}${e.isTeam ? ' (team)' : ' (personal)'}`)) : h('option', { value: '' }, 'No other team or personal space available'));
  const moveBtn = h('button.btn.btn-sm.btn-outline-primary', {
    disabled: !destinations.length,
    onclick: async () => {
      const dest = moveSel.value;
      if (!dest) return;
      const ok = await confirmDialog({
        title: `Move ${project} to ${dest}?`, confirmLabel: 'Move project',
        message: h('span', 'All runs, artifacts and reports move with it. Afterwards log to ', h('code', `wandb.init(entity="${dest}", project="${project}")`), '.'),
      });
      if (!ok) return;
      const prog = h('div.msa-task', h('div.msa-task-bar', h('span')), h('div.small.text-muted.msa-task-label', 'Starting…'));
      const m = modal({ title: 'Moving project', icon: 'arrow-left-right', body: prog, persistent: true });
      try {
        const d = await gql('mutation M($input: MoveProjectsInput!) { moveProjects(input: $input) { tasks { id state progress } } }',
          { input: { sourceEntityName: entity, destinationEntityName: dest, projectNames: [project], ownerUsername: c.viewer.username } });
        const task = d.moveProjects?.tasks?.[0];
        const final = task ? await waitForTask(task.id, {
          signal,
          onProgress: (t) => {
            prog.querySelector('.msa-task-bar span').style.width = `${Math.max(5, Number(t?.progress) || 0)}%`;
            prog.querySelector('.msa-task-label').textContent = `${String(t?.state || '').toLowerCase()} · ${t?.progress ?? 0}%`;
          },
        }) : null;
        m.close();
        if (final && /FAIL|ERROR/i.test(final.state)) throw new Error(final.result || 'W&B reported the move as failed.');
        done(`Moved to ${dest}`);
        navigate(urls.projectSettings(dest, project), { replace: true });
      } catch (err) { m.close(); toastError({ message: friendly(err) }, 'Move failed'); }
    },
  }, icon('arrow-left-right', 'me-1'), 'Move');

  async function deleteProject() {
    const ok = await typedConfirm({
      title: `Delete ${entity}/${project}?`, expected: project, confirmLabel: 'Delete project',
      message: `Permanently deletes ${fmtNumber(p.totalRuns || 0)} runs, their files, artifacts and reports. This cannot be undone.`,
      details: h('div.af-callout.af-callout-danger.small', icon('exclamation-octagon'), h('span', 'Tip: to keep the history but speed up a crowded workspace, use Cleanup → “Archive to project” instead.')),
    });
    if (!ok) return;
    try {
      await gql('mutation D($input: DeleteModelInput!) { deleteModel(input: $input) { success } }', { input: { id: p.id } });
      done(`Deleted ${project}`);
      navigate(urls.projects());
    } catch (err) { toastError({ message: friendly(err) }, 'Delete failed'); }
  }

  mount(body,
    card({ title: 'Details & visibility', icon: 'card-text', body: h('div', details.el, h('div.mt-3', saveDetails)) }),
    h('div.af-grid-2',
      card({ title: 'Rename', icon: 'input-cursor-text', body: h('div.d-flex.flex-column.gap-2',
        h('div.small.text-muted', 'Changes the project’s URL and the name used by wandb.init.'),
        h('div.d-flex.gap-2', renameInput, renameBtn)) }),
      card({ title: 'Move to another team', icon: 'arrow-left-right', body: h('div.d-flex.flex-column.gap-2',
        h('div.small.text-muted', 'Moves the whole project between your personal space and a team.'),
        h('div.d-flex.gap-2', moveSel, moveBtn)) })),
    card({ title: 'About', icon: 'info-circle', cls: 'msa-kv-wrap', body: kv([
      ['Owner', p.user?.username ? `@${p.user.username}` : '—'], ['Entity', h('code', p.entityName)],
      ['Created', new Date(p.createdAt).toLocaleString()], ['Updated', p.updatedAt ? new Date(p.updatedAt).toLocaleString() : '—'],
      ['Visibility', pill((ACCESS.find((a) => a.value === p.access) || { label: p.access || 'Private' }).label, 'muted', 'lock')],
      ['Log to it', h('code', `wandb.init(entity="${entity}", project="${project}")`), { wide: true }],
    ]) }),
    card({ title: 'Danger zone', icon: 'exclamation-octagon', cls: 'msa-danger', body: h('div.msa-danger-row',
      h('div', h('div.fw-semibold', 'Delete this project'), h('div.small.text-muted', 'Removes every run, file and artifact in it.')),
      h('button.btn.btn-sm.btn-danger', { onclick: deleteProject }, icon('trash', 'me-1'), 'Delete project')) }));
}
