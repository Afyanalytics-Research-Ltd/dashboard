// Cleanup: keep projects fast and tidy. Rules find crashed, tiny, old, untriaged and duplicate runs;
// the preview lets you untick runs; then delete them or archive them to another project (moveRuns).

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import {
  pageHeader, card, dataTable, emptyState, errorState, skeleton, statCard, timeAgo, toastError, confirmDialog, modal, selectionBar,
} from '../../../ui/js/ui.js';
import { fmtNumber, fmtDuration } from '../../../ui/js/format.js';
import { gql } from '../core/api.js';
import { urls } from '../core/urls.js';
import { loadAllRuns } from '../core/data.js';
import { runStateBadge, displayName } from '../core/wb.js';
import { allProjects, projectSelect, friendly, done, typedConfirm, waitForTask, pill } from './admin/shared.js';

const AUTO_NAME = /^[a-z]+-[a-z]+-\d+$/;
const PROTECT_TAGS = ['keep', 'baseline', 'best', 'production', 'prod'];

const RULES = [
  { id: 'crashed', label: 'Crashed or failed', icon: 'x-octagon', tone: 'red', help: 'Runs that ended in crashed, failed or killed.',
    match: (r) => ['crashed', 'failed', 'killed'].includes(r.state) },
  { id: 'short', label: 'Too short', icon: 'stopwatch', tone: 'amber', help: 'Fewer logged steps or seconds than the thresholds.',
    params: { steps: 20, seconds: 60 },
    match: (r, p) => (r.historyLineCount ?? 0) < p.steps || (r.runtime != null && r.runtime < p.seconds) },
  { id: 'old', label: 'Older than', icon: 'calendar-x', tone: 'slate', help: 'Created more than N days ago.',
    params: { days: 90 },
    match: (r, p) => Date.now() - new Date(r.createdAt) > p.days * 864e5 },
  { id: 'untriaged', label: 'Untriaged', icon: 'question-circle', tone: 'purple', help: 'Auto-generated name, no tags and no notes — nobody looked at it.',
    match: (r) => AUTO_NAME.test(r.displayName || '') && !(r.tags || []).length && !r.notes },
  { id: 'duplicate', label: 'Duplicate config', icon: 'files', tone: 'blue', help: 'Same configuration as a newer run; the newest of each group is kept.',
    match: (r, p, ctx) => ctx.dupes.has(r.name) },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  const pickerHost = h('div.msa-cleanup-picker');
  const rulesEl = h('div.msa-rules');
  const previewEl = h('div');
  const statsEl = h('div.af-stats');
  const barHost = h('div');
  mount(el,
    pageHeader({
      title: 'Cleanup', icon: 'recycle',
      subtitle: 'Big projects slow every workspace. Find runs you no longer need, then delete them or move them to an archive project.',
    }),
    card({ title: 'Project', icon: 'folder2', body: pickerHost }),
    statsEl, rulesEl, previewEl, barHost);

  const state = {
    project: null, runs: [], enabled: new Set(['crashed']), params: Object.fromEntries(RULES.filter((r) => r.params).map((r) => [r.id, { ...r.params }])),
    protectTags: true, excluded: new Set(), dupes: new Set(),
  };

  let projects = [];
  try { projects = await allProjects({ signal }); } catch (err) { mount(pickerHost, errorState(err)); return; }
  if (!ctx.active) return;
  const initial = ctx.query.get('project');
  mount(pickerHost, h('div.d-flex.gap-2.align-items-center.flex-wrap',
    projectSelect(projects, initial, (p) => { ctx.setQuery({ project: p ? `${p.entityName}/${p.name}` : null }); choose(p); }),
    h('span.small.text-muted', 'Only runs you can see are analysed. Running runs are never touched.')));
  if (!projects.length) mount(rulesEl, emptyState({ icon: 'folder2', title: 'No projects to clean up' }));
  const pre = initial && projects.find((p) => `${p.entityName}/${p.name}` === initial);
  if (pre) choose(pre); else mount(rulesEl, emptyState({ icon: 'arrow-up', title: 'Pick a project to analyse' }));

  async function choose(p) {
    state.project = p;
    state.excluded.clear();
    mount(statsEl);
    mount(previewEl);
    mount(barHost);
    if (!p) { mount(rulesEl, emptyState({ icon: 'arrow-up', title: 'Pick a project to analyse' })); return; }
    const prog = h('div.small.text-muted', 'Loading runs…');
    mount(rulesEl, card({ body: h('div', skeleton(3), prog) }));
    try {
      const { runs, total } = await loadAllRuns(p.entityName, p.name, { max: 5000, signal, onProgress: (n, t) => { prog.textContent = `Loaded ${fmtNumber(n)} of ${fmtNumber(t)} runs…`; } });
      if (!ctx.active || state.project !== p) return;
      state.runs = runs;
      state.total = total;
      computeDupes();
      drawRules();
      drawPreview();
    } catch (err) { mount(rulesEl, errorState(err, () => choose(p))); }
  }

  function computeDupes() {
    const groups = new Map();
    for (const r of state.runs) {
      const key = JSON.stringify(Object.keys(r.config).sort().filter((k) => !/seed|^_|wandb/i.test(k)).map((k) => [k, r.config[k]]));
      if (key === '[]') continue;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    state.dupes = new Set();
    for (const list of groups.values()) {
      if (list.length < 2) continue;
      list.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)).slice(1).forEach((r) => state.dupes.add(r.name));
    }
  }

  const isProtected = (r) => r.state === 'running' || (state.protectTags && (r.tags || []).some((t) => PROTECT_TAGS.includes(String(t).toLowerCase())));
  const matchesRule = (rule, r) => rule.match(r, state.params[rule.id] || {}, state);
  const candidates = () => state.runs.filter((r) => !isProtected(r) && RULES.some((rule) => state.enabled.has(rule.id) && matchesRule(rule, r)));
  const selected = () => candidates().filter((r) => !state.excluded.has(r.name));

  function drawRules() {
    const protectedCount = state.runs.filter(isProtected).length;
    mount(statsEl,
      statCard({ label: 'Runs in project', value: fmtNumber(state.total), icon: 'play-circle', tone: 'blue', href: urls.table(state.project.entityName, state.project.name) }),
      statCard({ label: 'Analysed', value: fmtNumber(state.runs.length), icon: 'search', tone: 'slate', sub: state.runs.length < state.total ? 'first 5,000' : 'all runs' }),
      statCard({ label: 'Protected', value: fmtNumber(protectedCount), icon: 'shield-check', tone: 'green', sub: 'running or tagged keep/baseline/best' }),
      statCard({ label: 'Selected', value: fmtNumber(selected().length), icon: 'check2-square', tone: selected().length ? 'red' : 'slate' }));
    mount(rulesEl, h('div.msa-rule-grid', RULES.map((rule) => {
      const count = state.runs.filter((r) => !isProtected(r) && matchesRule(rule, r)).length;
      const on = state.enabled.has(rule.id);
      const paramInputs = rule.params && Object.entries(state.params[rule.id]).map(([k, v]) => h('label.msa-rule-param',
        h('input.form-control.form-control-sm', {
          type: 'number', min: 0, value: v, 'aria-label': `${rule.label} ${k}`,
          onclick: (e) => e.stopPropagation(),
          oninput: debounce((e) => { state.params[rule.id][k] = Number(e.target.value) || 0; drawRules(); drawPreview(); }, 250),
        }), h('span', k)));
      return h('button.msa-rule', {
        type: 'button', class: [on && 'on', `af-tone-${rule.tone}`], 'aria-pressed': String(on),
        onclick: (e) => {
          if (e.target.closest('input')) return;
          on ? state.enabled.delete(rule.id) : state.enabled.add(rule.id);
          drawRules(); drawPreview();
        },
      },
      h('div.msa-rule-top', h('span.msa-rule-icon', icon(rule.icon)), h('span.msa-rule-count', fmtNumber(count)), h('span.msa-rule-check', icon(on ? 'check-circle-fill' : 'circle'))),
      h('div.msa-rule-label', rule.label),
      h('div.msa-rule-help', rule.help),
      paramInputs && h('div.msa-rule-params', paramInputs));
    })),
    h('label.form-check.form-switch.mt-2.msa-protect',
      h('input.form-check-input', { type: 'checkbox', checked: state.protectTags, onchange: (e) => { state.protectTags = e.target.checked; drawRules(); drawPreview(); } }),
      h('span.form-check-label.small', 'Protect runs tagged ', PROTECT_TAGS.map((t) => h('code.me-1', t)))));
  }

  function drawPreview() {
    const list = candidates();
    const sel = selected();
    const reasons = (r) => RULES.filter((rule) => state.enabled.has(rule.id) && matchesRule(rule, r));
    const chosen = new Set(sel.map((r) => r.name));
    mount(previewEl, card({
      title: `Preview · ${fmtNumber(sel.length)} of ${fmtNumber(list.length)} matching runs selected`, icon: 'list-check', flush: true,
      actions: list.length ? [
        h('button.btn.btn-sm.btn-link', { onclick: () => { state.excluded.clear(); drawRules(); drawPreview(); } }, 'Select all'),
        h('button.btn.btn-sm.btn-link', { onclick: () => { list.forEach((r) => state.excluded.add(r.name)); drawRules(); drawPreview(); } }, 'Select none'),
      ] : null,
      body: list.length ? dataTable({
        rows: list.slice(0, 500), rowKey: (r) => r.name, selectable: true, selected: chosen, dense: true,
        onSelect: (set) => { list.forEach((r) => (set.has(r.name) ? state.excluded.delete(r.name) : state.excluded.add(r.name))); drawRules(); drawBar(); },
        rowClass: (r) => (state.excluded.has(r.name) ? 'msa-row-excluded' : null),
        columns: [
          { key: 'name', label: 'Run', render: (r) => h('div', h('a.ms-run-name', { href: urls.run(state.project.entityName, state.project.name, r.name) }, displayName(r)), h('div.ms-run-id', r.name)) },
          { key: 'state', label: 'State', render: (r) => runStateBadge(r.state) },
          { key: 'why', label: 'Why', render: (r) => h('div.d-flex.gap-1.flex-wrap', reasons(r).map((x) => pill(x.label, 'muted', x.icon))) },
          { key: 'steps', label: 'Steps', align: 'right', render: (r) => fmtNumber(r.historyLineCount ?? 0) },
          { key: 'runtime', label: 'Runtime', align: 'right', render: (r) => fmtDuration(r.runtime) },
          { key: 'createdAt', label: 'Created', render: (r) => timeAgo(r.createdAt) },
          { key: 'tags', label: 'Tags', render: (r) => h('div.d-flex.gap-1.flex-wrap', (r.tags || []).map((t) => h('span.ms-tag', t))) },
        ],
      }) : emptyState({ icon: 'stars', title: 'Nothing to clean up', message: 'No unprotected runs match the selected rules. Nice and tidy.' }),
    }));
    if (list.length > 500) previewEl.appendChild(h('div.small.text-muted.mt-1', `Showing the first 500 of ${fmtNumber(list.length)}; actions apply to all ${fmtNumber(sel.length)} selected.`));
    drawBar();
  }

  function drawBar() {
    const sel = selected();
    mount(barHost, selectionBar(sel.length, [
      h('button.btn.btn-sm.btn-light', { onclick: () => archive(sel) }, icon('archive', 'me-1'), 'Archive to project…'),
      h('button.btn.btn-sm.btn-danger', { onclick: () => remove(sel) }, icon('trash', 'me-1'), `Delete ${fmtNumber(sel.length)}`),
    ], () => { candidates().forEach((r) => state.excluded.add(r.name)); drawRules(); drawPreview(); }));
  }

  // ------------------------------------------------------------------ actions

  async function inBatches(items, size, fn, label) {
    const bar = h('div.msa-task', h('div.msa-task-bar', h('span')), h('div.small.text-muted.msa-task-label', 'Starting…'));
    const m = modal({ title: label, icon: 'hourglass-split', body: bar, persistent: true });
    try {
      for (let i = 0; i < items.length; i += size) {
        await fn(items.slice(i, i + size));
        const pct = Math.round(((i + size) / items.length) * 100);
        bar.querySelector('span').style.width = `${Math.min(100, pct)}%`;
        bar.querySelector('.msa-task-label').textContent = `${fmtNumber(Math.min(items.length, i + size))} of ${fmtNumber(items.length)}`;
      }
    } finally { m.close(); }
  }

  async function remove(sel) {
    if (!sel.length) return;
    const { entityName: e, name: p } = state.project;
    const artifacts = h('input.form-check-input', { type: 'checkbox' });
    const ok = await typedConfirm({
      title: `Delete ${fmtNumber(sel.length)} runs from ${p}?`, expected: `delete ${sel.length}`, confirmLabel: 'Delete runs',
      message: 'Runs, their history, logs and media are removed permanently.',
      details: h('label.form-check.small', artifacts, h('span.ms-2', 'Also delete artifacts these runs logged (models, tables). Artifacts used by other runs are kept.')),
    });
    if (!ok) return;
    try {
      await inBatches(sel.map((r) => r.name), 100, (names) => gql('mutation D($input: DeleteRunsInput!) { deleteRuns(input: $input) { errors } }',
        { input: { entityName: e, projectName: p, filters: JSON.stringify({ name: { $in: names } }), deleteArtifacts: artifacts.checked } }), 'Deleting runs');
      done(`Deleted ${fmtNumber(sel.length)} runs`);
      choose(state.project);
    } catch (err) { toastError({ message: friendly(err) }, 'Delete failed'); choose(state.project); }
  }

  async function archive(sel) {
    if (!sel.length) return;
    const { entityName: e, name: p } = state.project;
    const dest = h('input.form-control.form-control-sm.ms-mono', { value: `${p}-archive`, 'aria-label': 'Archive project name' });
    let go = false;
    await new Promise((resolve) => {
      modal({
        title: `Archive ${fmtNumber(sel.length)} runs`, icon: 'archive',
        body: h('div.d-flex.flex-column.gap-2',
          h('p.mb-1.small', 'Moves the runs (with history, files and artifacts) to another project in ', h('code', e), '. The current project gets lighter; nothing is lost.'),
          h('label.form-label.small.mb-0', 'Archive project (created if it does not exist)'), dest),
        actions: [{ label: 'Cancel' }, { label: 'Move runs', variant: 'primary', icon: 'archive', onClick: (close) => { go = true; close(); } }],
        onClose: resolve,
      });
    });
    const target = dest.value.trim();
    if (!go || !target) return;
    if (target === p) { toastError({ message: 'Choose a different project to archive into.' }, 'Same project'); return; }
    try {
      const exists = await gql('query E($e: String!, $p: String!) { project(entityName: $e, name: $p) { id } }', { e, p: target });
      if (!exists.project) {
        await gql('mutation C($input: UpsertModelInput!) { upsertModel(input: $input) { project { id name } } }',
          { input: { entityName: e, name: target, description: `Archived runs from ${p}`, access: 'PRIVATE' } });
      }
      const tasks = [];
      await inBatches(sel.map((r) => r.name), 100, async (names) => {
        const d = await gql('mutation M($input: MoveRunsInput!) { moveRuns(input: $input) { task { id state } } }', { input: {
          sourceEntityName: e, sourceProjectName: p, destinationEntityName: e, destinationProjectName: target, filters: JSON.stringify({ name: { $in: names } }),
        } });
        if (d.moveRuns?.task?.id) tasks.push(d.moveRuns.task.id);
      }, 'Scheduling move');
      const results = await Promise.all(tasks.map((id) => waitForTask(id, { signal })));
      const failed = results.filter((t) => t && /FAIL|ERROR/i.test(t.state));
      if (failed.length) throw new Error(failed[0].result || 'W&B reported the move as failed.');
      done(`Moved ${fmtNumber(sel.length)} runs to ${target}`);
      const view = await confirmDialog({ title: 'Runs archived', message: h('span', `${fmtNumber(sel.length)} runs now live in `, h('code', `${e}/${target}`), '.'), confirmLabel: 'Open archive' });
      if (view) location.assign(urls.workspace(e, target));
      else choose(state.project);
    } catch (err) { toastError({ message: friendly(err) }, 'Archive failed'); choose(state.project); }
  }
}
