// Run page: header (rename, state, actions) + tabs Overview / Charts / System / Logs / Files /
// Media / Artifacts. Tabs load lazily and stay in the URL (?tab=).

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  tabs, menuButton, confirmDialog, formModal, toast, toastError, copyButton, errorState, refreshControl, timeAgo,
} from '../../../ui/js/ui.js';
import { fmtDuration } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { wandbLink } from '../core/api.js';
import { loadRun, loadProjects, mutations } from '../core/data.js';
import { urls } from '../core/urls.js';
import { runStateBadge, runColor, displayName } from '../core/wb.js';
import overviewTab from './run/overview.js';
import { chartsTab, systemTab, numericKeys } from './run/charts.js';
import { logsTab } from './run/logs.js';
import { filesTab } from './run/files.js';
import { mediaTab, mediaKeys } from './run/media.js';
import { artifactsTab } from './run/artifacts.js';
import { inlineText } from './run/common.js';

const TABS = {
  overview: { label: 'Overview', icon: 'info-circle', render: overviewTab },
  charts: { label: 'Charts', icon: 'graph-up', render: chartsTab },
  system: { label: 'System', icon: 'cpu', render: systemTab },
  logs: { label: 'Logs', icon: 'terminal', render: logsTab },
  files: { label: 'Files', icon: 'folder2-open', render: filesTab },
  media: { label: 'Media', icon: 'images', render: mediaTab },
  artifacts: { label: 'Artifacts', icon: 'boxes', render: artifactsTab },
};

export default async function render(ctx) {
  const { entity, project } = ctx.params;
  const runName = ctx.params.run;
  let run;
  try {
    run = await loadRun(entity, project, runName, { signal: ctx.signal });
  } catch (err) {
    if (err.name === 'AbortError') return;
    mount(ctx.el, errorState(err.status === 404 ? Object.assign(err, { message: `Run “${runName}” was not found in ${entity}/${project}. It may have been deleted or moved.` }) : err));
    return;
  }
  if (!ctx.active) return;
  ctx.setTitle(`${displayName(run)} · ${project}`);
  // Summary _runtime can lag (or be ~0 for very short runs); the heartbeat gap is a better floor.
  const span = run.heartbeatAt && run.createdAt ? (new Date(run.heartbeatAt) - new Date(run.createdAt)) / 1000 : 0;
  run.runtime = Math.max(Number(run.runtime) || 0, span || 0);

  const metricCount = numericKeys(run).filter((k) => !k.key.startsWith('_') && !k.key.startsWith('system/')).length;
  const mediaCount = mediaKeys(run).length;
  let active = TABS[ctx.query.get('tab')] ? ctx.query.get('tab') : 'overview';
  const host = h('div.msr-tab-host');
  const loaded = new Map();

  const counts = { charts: metricCount, media: mediaCount, logs: run.logLineCount || null, files: run.fileCount || null };
  const tabBar = tabs(Object.entries(TABS).map(([id, t]) => ({ id, label: t.label, icon: t.icon, count: counts[id] ?? undefined })), active, (id) => show(id));

  function show(id) {
    active = id;
    ctx.setQuery({ tab: id === 'overview' ? null : id, path: id === 'files' ? ctx.query.get('path') : null });
    for (const [tid, pane] of loaded) pane.hidden = tid !== id;
    if (!loaded.has(id)) {
      const pane = h('div.msr-pane', { dataset: { tab: id } });
      loaded.set(id, pane);
      host.appendChild(pane);
      Promise.resolve(TABS[id].render(pane, ctx, run)).catch((err) => { if (err?.name !== 'AbortError') mount(pane, errorState(err)); });
    }
  }

  // ---- header
  const color = runColor(run, 0);
  const titleEl = h('h1.af-page-title.msr-title',
    h('span.msr-run-swatch', { style: { background: color } }),
    inlineText(run.displayName || run.name, async (v) => {
      await mutations.updateRun(run.id, { displayName: v });
      run.displayName = v;
      ctx.setTitle(`${v} · ${project}`);
      toast('Run renamed', 'success', { timeout: 1500 });
    }, { className: 'msr-title-edit', title: 'Click to rename' }),
    runStateBadge(run.state),
    run.state === 'running' && h('span.ms-pill.ms-pill-warn.msr-stop-pill', { class: !run.stopped && 'd-none', title: 'A stop was requested; the training script has not exited yet' },
      icon('hourglass-split'), 'Stop requested'));

  const actions = [
    run.state === 'running' && h('button.btn.btn-sm.btn-outline-danger', { type: 'button', onclick: () => stopRun(run) }, icon('stop-circle', 'me-1'), 'Stop'),
    h('a.btn.btn-sm.btn-outline-secondary', { href: urls.compare(entity, project, [run.name]), title: 'Compare with other runs' }, icon('layout-split', 'me-1'), 'Compare'),
    run.state === 'running' && refreshControl(() => reload(), { key: 'ms-run', initial: 10 }).el,
    menuButton('', [
      { label: 'Copy run path', icon: 'clipboard', onClick: () => copy(`${entity}/${project}/${run.name}`) },
      { label: 'Copy run id', icon: 'hash', onClick: () => copy(run.name) },
      { label: 'Copy SDK snippet', icon: 'code-slash', onClick: () => copy(`import wandb\nrun = wandb.Api().run("${entity}/${project}/${run.name}")`) },
      wandbLink() && { label: 'Open in W&B', icon: 'box-arrow-up-right', href: wandbLink(`/${entity}/${project}/runs/${run.name}`), external: true },
      { divider: true },
      { label: 'Move to project…', icon: 'folder-symlink', onClick: () => moveRun(run, ctx) },
      { label: 'Delete run…', icon: 'trash', danger: true, onClick: () => deleteRun(run, ctx) },
    ].filter(Boolean), { icon: 'three-dots', title: 'More actions' }),
  ].filter(Boolean);

  mount(ctx.el,
    h('div.af-page-header.msr-header',
      h('nav.af-crumbs',
        h('a', { href: urls.projects() }, 'Projects'), icon('chevron-right', 'af-crumb-sep'),
        h('a', { href: urls.workspace(entity, project) }, project), icon('chevron-right', 'af-crumb-sep'),
        h('a', { href: urls.table(entity, project) }, 'Runs'), icon('chevron-right', 'af-crumb-sep'),
        h('span.ms-mono', run.name)),
      h('div.af-page-header-row',
        h('div.af-page-heading', titleEl,
          h('div.msr-sub',
            h('span', icon('hash'), h('code.ms-mono', run.name), copyButton(run.name)),
            run.group && h('span', icon('collection'), run.group),
            run.jobType && h('span', icon('briefcase'), run.jobType),
            h('span', icon('clock'), fmtDuration(run.runtime)),
            h('span', icon('calendar3'), timeAgo(run.createdAt)),
            run.user && h('span', icon('person'), run.user.username),
            run.sweep && h('a', { href: urls.sweep(entity, project, run.sweep.name) }, icon('shuffle'), run.sweep.displayName || run.sweep.name),
            (run.tags || []).slice(0, 5).map((t) => h('span.ms-tag', t)))),
        h('div.af-page-actions', actions))),
    h('div.msr-tabs-wrap', tabBar),
    host);
  show(active);

  async function reload() {
    try {
      const fresh = await loadRun(entity, project, runName, { signal: ctx.signal });
      if (fresh.state !== run.state) { toast(`Run is now ${fresh.state}`, fresh.state === 'finished' ? 'success' : 'warning'); navigate(location.pathname + location.search, { replace: true }); }
    } catch { /* keep current view */ }
  }
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied', 'success', { timeout: 1200 }); } catch { toast(text, 'info', { title: 'Copy this' }); }
}

async function stopRun(run) {
  const ok = await confirmDialog({
    title: 'Stop this run?', confirmLabel: 'Stop run', danger: true,
    message: 'W&B flags the run to stop. The training script stops when its W&B SDK next checks for the flag (usually within a minute); runs whose SDK has stop-polling disabled keep going until they finish.',
  });
  if (!ok) return;
  try {
    await mutations.stopRun(run.id);
    run.stopped = true;
    document.querySelector('.msr-stop-pill')?.classList.remove('d-none');
    toast('Stop requested — waiting for the training script to acknowledge it', 'success');
  } catch (err) { toastError(err, 'Could not stop run'); }
}

async function deleteRun(run, ctx) {
  const { entity, project } = ctx.params;
  const withArtifacts = h('input.form-check-input', { type: 'checkbox', id: 'msr-del-art' });
  const ok = await confirmDialog({
    title: `Delete ${displayName(run)}?`, danger: true, confirmLabel: 'Delete run',
    message: 'The run, its history, logs and files are removed from W&B. This cannot be undone.',
    details: h('label.form-check.mt-2', withArtifacts, h('span.form-check-label.ms-2', 'Also delete artifacts this run logged')),
  });
  if (!ok) return;
  try {
    await mutations.deleteRun(run.id, withArtifacts.checked);
    toast(`Deleted ${displayName(run)}`, 'success');
    navigate(urls.table(entity, project));
  } catch (err) { toastError(err, 'Could not delete run'); }
}

async function moveRun(run, ctx) {
  const { entity, project } = ctx.params;
  let projects = [];
  try { projects = await loadProjects(entity, { max: 300 }); } catch { /* free text still works */ }
  formModal({
    title: 'Move run to another project', icon: 'folder-symlink',
    subtitle: 'Handy for archiving old runs so workspaces stay fast.',
    fields: [
      { name: 'entity', label: 'Destination entity', value: entity, required: true },
      { name: 'project', label: 'Destination project', type: projects.length ? 'select' : 'text', required: true,
        options: projects.filter((p) => p.name !== project).map((p) => ({ value: p.name, label: p.name })), placeholder: 'project name' },
    ],
    submitLabel: 'Move run',
    onSubmit: async (v, close) => {
      await mutations.moveRuns(entity, project, [run.name], v.entity, v.project);
      close();
      toast(`Moved to ${v.entity}/${v.project}`, 'success');
      navigate(urls.run(v.entity, v.project, run.name));
    },
  });
}
