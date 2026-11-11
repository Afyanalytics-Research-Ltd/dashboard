// Helpers shared by the "projects" page group: home, projects, workspace, runs table, compare.

import { h, icon, mount } from '../../../../ui/js/dom.js';
import { pageHeader, toast, toastError, openMenu, formModal, confirmDialog } from '../../../../ui/js/ui.js';
import { fmtDuration, fmtDate } from '../../../../ui/js/format.js';
import { gql, collect } from '../../core/api.js';
import { urls } from '../../core/urls.js';
import { fmtMetric, runStateBadge, displayName } from '../../core/wb.js';
import { mutations } from '../../core/data.js';

// ------------------------------------------------------------------ values & columns

export const META_COLUMNS = [
  { key: 'state', label: 'State' },
  { key: 'createdAt', label: 'Created' },
  { key: 'runtime', label: 'Runtime' },
  { key: 'user', label: 'User' },
  { key: 'group', label: 'Group' },
  { key: 'jobType', label: 'Job type' },
  { key: 'tags', label: 'Tags' },
  { key: 'sweepName', label: 'Sweep' },
  { key: 'notes', label: 'Notes' },
  { key: 'host', label: 'Host' },
  { key: 'historyLineCount', label: 'Steps' },
];

/** Value of a column key for a run: meta keys, "config.<k>", "summary.<k>". */
export function runValue(run, key) {
  if (key.startsWith('config.')) return run.config?.[key.slice(7)];
  if (key.startsWith('summary.')) return run.summary?.[key.slice(8)];
  switch (key) {
    case 'name': return displayName(run);
    case 'user': return run.user?.username;
    case 'tags': return run.tags || [];
    default: return run[key];
  }
}

export function columnLabel(key) {
  const meta = META_COLUMNS.find((c) => c.key === key);
  if (meta) return meta.label;
  if (key === 'name') return 'Name';
  return key.replace(/^(config|summary)\./, '');
}

/** Render a cell value. */
export function cellValue(run, key) {
  const v = runValue(run, key);
  if (key === 'state') return runStateBadge(v);
  if (key === 'createdAt') return h('span.af-nowrap', { title: fmtDate(v) }, fmtDate(v, { seconds: false }));
  if (key === 'runtime') return h('span.ms-metric', v == null ? '—' : fmtDuration(v));
  if (key === 'tags') return v.length ? h('span.mp-tags', v.map((t) => h('span.ms-tag', t))) : h('span.text-muted', '—');
  if (key === 'notes') return v ? h('span.mp-note-cell', { title: v }, v) : h('span.text-muted', '—');
  if (v == null || v === '') return h('span.text-muted', '—');
  if (typeof v === 'number') return h('span.ms-metric', fmtMetric(v));
  if (typeof v === 'object') return h('code.mp-json', JSON.stringify(v));
  return String(v);
}

/** All config/summary keys present across runs (numeric-first ordering for summary). */
export function catalog(runs) {
  const config = new Map();
  const summary = new Map();
  for (const r of runs) {
    for (const [k, v] of Object.entries(r.config || {})) {
      if (!config.has(k)) config.set(k, new Set());
      if (config.get(k).size < 30) config.get(k).add(typeof v === 'object' ? JSON.stringify(v) : v);
    }
    for (const [k, v] of Object.entries(r.summary || {})) {
      if (!summary.has(k)) summary.set(k, { numeric: true });
      if (typeof v !== 'number') summary.get(k).numeric = false;
    }
  }
  return {
    config: [...config.entries()].map(([k, vals]) => ({ key: `config.${k}`, name: k, values: [...vals], numeric: [...vals].every((x) => typeof x === 'number') }))
      .sort((a, b) => a.name.localeCompare(b.name)),
    summary: [...summary.entries()].map(([k, info]) => ({ key: `summary.${k}`, name: k, numeric: info.numeric }))
      .filter((s) => !s.name.startsWith('_'))
      .sort((a, b) => a.name.localeCompare(b.name)),
  };
}

// ------------------------------------------------------------------ grouping (any field or expression)

const EXPR = /^\s*([\w.\/-]+)\s*(>=|<=|>|<|==|!=|contains)\s*(.+?)\s*$/;

/** Parse a group-by spec: a field key ("config.optimizer") or an expression ("summary.val/accuracy > 0.85"). */
export function parseGroupSpec(spec) {
  if (!spec) return null;
  const m = spec.match(EXPR);
  if (!m) return { key: spec, label: columnLabel(spec) };
  const raw = m[3].replace(/^["']|["']$/g, '');
  const num = Number(raw);
  return { key: m[1], op: m[2], value: Number.isNaN(num) ? raw : num, label: spec };
}

export function groupValue(run, spec) {
  const g = typeof spec === 'string' ? parseGroupSpec(spec) : spec;
  if (!g) return null;
  const v = runValue(run, g.key);
  if (!g.op) {
    if (Array.isArray(v)) return v.length ? v.join(', ') : '(no tags)';
    return v == null || v === '' ? '(none)' : typeof v === 'number' ? fmtMetric(v) : String(v);
  }
  const n = Number(v);
  let ok;
  switch (g.op) {
    case '>': ok = n > g.value; break;
    case '>=': ok = n >= g.value; break;
    case '<': ok = n < g.value; break;
    case '<=': ok = n <= g.value; break;
    case '==': ok = String(v) === String(g.value); break;
    case '!=': ok = String(v) !== String(g.value); break;
    case 'contains': ok = String(v ?? '').toLowerCase().includes(String(g.value).toLowerCase()); break;
    default: ok = false;
  }
  return v == null ? '(none)' : ok ? `${g.label}` : `not (${g.label})`;
}

/** Group runs: [{ key, runs }] in first-seen order. */
export function groupRuns(runs, spec) {
  const parsed = parseGroupSpec(spec);
  if (!parsed) return [{ key: null, runs }];
  const map = new Map();
  for (const r of runs) {
    const k = groupValue(r, parsed);
    if (!map.has(k)) map.set(k, []);
    map.get(k).push(r);
  }
  return [...map.entries()].map(([key, rs]) => ({ key, runs: rs }));
}

/**
 * Aggregate several runs' series into mean ± min/max (binned on x).
 * series: [{points:[{x,y}]}] -> points [{x, y(mean), min, max}]
 */
export function aggregateSeries(seriesList, bins = 200) {
  const all = seriesList.flatMap((s) => s.points);
  if (!all.length) return [];
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of all) { if (p.x < lo) lo = p.x; if (p.x > hi) hi = p.x; }
  const width = (hi - lo) / bins || 1;
  const acc = Array.from({ length: bins + 1 }, () => ({ sum: 0, n: 0, min: Infinity, max: -Infinity, xs: 0 }));
  for (const s of seriesList) {
    // One value per run per bin (the run's mean within the bin), so long runs don't dominate.
    const perBin = new Map();
    for (const p of s.points) {
      if (!Number.isFinite(p.y)) continue;
      const b = Math.min(bins, Math.floor((p.x - lo) / width));
      const cur = perBin.get(b) || { sum: 0, n: 0, x: 0 };
      cur.sum += p.y; cur.n++; cur.x += p.x;
      perBin.set(b, cur);
    }
    for (const [b, v] of perBin) {
      const y = v.sum / v.n;
      const a = acc[b];
      a.sum += y; a.n++; a.xs += v.x / v.n;
      if (y < a.min) a.min = y;
      if (y > a.max) a.max = y;
    }
  }
  return acc.filter((a) => a.n).map((a) => ({ x: a.xs / a.n, y: a.sum / a.n, min: a.min, max: a.max }));
}

/** 5th–95th percentile range of y values (for "clip outliers"). */
export function robustRange(seriesList) {
  const ys = seriesList.flatMap((s) => s.points.map((p) => p.y)).filter(Number.isFinite).sort((a, b) => a - b);
  if (ys.length < 10) return null;
  const q = (p) => ys[Math.min(ys.length - 1, Math.max(0, Math.floor(p * (ys.length - 1))))];
  const lo = q(0.05);
  const hi = q(0.95);
  const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;
  return [lo - pad, hi + pad];
}

/** Is a lower value better for this metric? (loss, error, perplexity, time…) */
export const lowerIsBetter = (name) => /loss|err|perplex|mse|mae|rmse|wer|cer|latency|time|_runtime|distance/i.test(name);

// ------------------------------------------------------------------ CSV

export function toCsv(columns, rows) {
  const esc = (v) => {
    if (v == null) return '';
    const s = Array.isArray(v) ? v.join('|') : typeof v === 'object' ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns.map((c) => esc(c.label)).join(','), ...rows.map((r) => columns.map((c) => esc(c.value(r))).join(','))].join('\n');
}

// ------------------------------------------------------------------ project header & chrome

/** Project page header with breadcrumbs and project-level tabs. */
export function projectHeader({ entity, project, title, icon: ic = 'folder2-open', subtitle, actions, active }) {
  const tabs = [
    ['workspace', 'Workspace', 'graph-up', urls.workspace(entity, project)],
    ['table', 'Runs', 'table', urls.table(entity, project)],
    ['sweeps', 'Sweeps', 'shuffle', urls.sweeps(entity, project)],
    ['artifacts', 'Artifacts', 'boxes', urls.artifacts(entity, project)],
    ['reports', 'Reports', 'file-earmark-richtext', urls.reports(entity, project)],
  ];
  return h('div.mp-head',
    pageHeader({
      title: title || project, icon: ic, subtitle,
      crumbs: [{ label: 'Projects', href: urls.projects() }, { label: entity, href: urls.projects({ entity }) }, { label: project }],
      actions,
    }),
    h('nav.mp-tabs', tabs.map(([id, label, i, href]) => h('a.mp-tab', { href, class: id === active && 'active' }, icon(i), label))));
}

/** Small click-to-edit text (Enter saves, Esc cancels). onSave returns a promise. */
export function inlineText(value, onSave, { placeholder = '—', className = '', multiline = false } = {}) {
  const wrap = h('span.mp-inline', { class: className });
  let current = value || '';
  const view = () => mount(wrap, h('span.mp-inline-text', { class: !current && 'empty', tabindex: 0, title: 'Click to edit' }, current || placeholder), icon('pencil', 'mp-inline-pencil'));
  const edit = () => {
    const input = multiline
      ? h('textarea.form-control.form-control-sm', { rows: 3 }, current)
      : h('input.form-control.form-control-sm', { value: current, placeholder });
    mount(wrap, input);
    input.focus();
    input.select?.();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const next = input.value.trim();
      if (!save || next === current) { view(); return; }
      const prev = current;
      current = next;
      view();
      try { await onSave(next); } catch (err) { current = prev; view(); toastError(err, 'Could not save'); }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (!multiline || e.ctrlKey || e.metaKey)) { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
  };
  wrap.addEventListener('click', (e) => { e.stopPropagation(); if (!wrap.querySelector('input,textarea')) edit(); });
  wrap.addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target.classList.contains('mp-inline-text')) edit(); });
  view();
  return wrap;
}

// ------------------------------------------------------------------ bulk actions (table + workspace)

/** Tag/untag dialog → modifyRuns in one call. */
export function bulkTagDialog(entity, project, runs, onDone) {
  const existing = [...new Set(runs.flatMap((r) => r.tags || []))].sort();
  formModal({
    title: `Tags for ${runs.length} run${runs.length === 1 ? '' : 's'}`, icon: 'tags',
    fields: [
      { name: 'add', label: 'Add tags', type: 'tags', placeholder: 'e.g. best, ready-for-review' },
      { name: 'remove', label: 'Remove tags', type: 'tags', value: [], placeholder: existing.join(', ') || 'none on these runs',
        help: existing.length ? `On these runs: ${existing.join(', ')}` : null },
    ],
    submitLabel: 'Apply',
    onSubmit: async ({ add, remove }, close) => {
      if (!add.length && !remove.length) throw new Error('Enter at least one tag to add or remove.');
      // W&B's modifyRuns wants exactly one of addTags/removeTags per call and rejects empty arrays,
      // so the core helper (which always sends both) is bypassed here.
      const filters = JSON.stringify({ name: { $in: runs.map((r) => r.name) } });
      const modify = (key, tags) => gql(`mutation TagRuns($input: ModifyRunsInput!) { modifyRuns(input: $input) { clientMutationId } }`,
        { input: { entityName: entity, projectName: project, filters, [key]: tags } });
      if (add.length) await modify('addTags', add);
      if (remove.length) await modify('removeTags', remove);
      toast(`Updated tags on ${runs.length} run(s)`, 'success');
      close();
      onDone?.();
    },
  });
}

export async function bulkDelete(entity, project, runs, onDone) {
  const box = h('input.form-check-input', { type: 'checkbox', id: 'mp-del-art' });
  const ok = await confirmDialog({
    title: `Delete ${runs.length} run${runs.length === 1 ? '' : 's'}?`, danger: true, confirmLabel: 'Delete',
    message: h('div', h('p.mb-2', 'This permanently deletes the runs, their history and logs from W&B.'),
      h('ul.mp-del-list', runs.slice(0, 8).map((r) => h('li', h('code', displayName(r)))), runs.length > 8 && h('li.text-muted', `…and ${runs.length - 8} more`)),
      h('label.form-check.small.mt-2', box, h('span.ms-2', 'Also delete artifacts these runs produced'))),
  });
  if (!ok) return false;
  try {
    await mutations.deleteRuns(entity, project, runs.map((r) => r.name), box.checked);
    toast(`Deleted ${runs.length} run(s)`, 'success');
    onDone?.();
    return true;
  } catch (err) { toastError(err, 'Delete failed'); return false; }
}

export async function listEntityProjects(entity) {
  return collect((after) => gql(`query P($e: String!, $after: String) { projects(entityName: $e, first: 100, after: $after, order: "-lastActive") {
    pageInfo { hasNextPage endCursor } edges { node { id name entityName } } } }`, { e: entity, after }), (d) => d.projects, { max: 500 });
}

/** Move runs to another (possibly new) project of the same entity — e.g. an archive project. */
export async function bulkMoveDialog(entity, project, runs, onDone) {
  let projects = [];
  try { projects = (await listEntityProjects(entity)).filter((p) => p.name !== project); } catch { /* typed name still works */ }
  formModal({
    title: `Move ${runs.length} run${runs.length === 1 ? '' : 's'}`, icon: 'box-arrow-right',
    subtitle: 'Keep this workspace fast by moving old runs into an archive project.',
    fields: [
      { name: 'dest', label: 'Destination project', required: true, placeholder: `${project}-archive`, value: projects[0]?.name || `${project}-archive`,
        help: projects.length ? `Existing: ${projects.slice(0, 8).map((p) => p.name).join(', ')}${projects.length > 8 ? '…' : ''}. A new name creates the project.` : 'A new name creates the project.' },
    ],
    submitLabel: 'Move runs',
    onSubmit: async ({ dest }, close) => {
      const name = dest.trim();
      if (name === project) throw new Error('Pick a different project.');
      if (!/^[\w.-]+$/.test(name)) throw new Error('Project names may only use letters, numbers, ".", "_" and "-".');
      if (!projects.some((p) => p.name === name)) {
        await gql(`mutation C($input: UpsertModelInput!) { upsertModel(input: $input) { project { id name } } }`, { input: { name, entityName: entity } });
      }
      await mutations.moveRuns(entity, project, runs.map((r) => r.name), entity, name);
      toast(`Moved ${runs.length} run(s) to ${name}`, 'success', { action: { label: 'Open →', onClick: () => { location.href = urls.workspace(entity, name); } } });
      close();
      onDone?.();
    },
  });
}

/** Edit a run's notes (a lightweight decision log lives in run notes). */
export function notesDialog(run, onSaved) {
  formModal({
    title: `Notes · ${displayName(run)}`, icon: 'journal-text',
    subtitle: 'Why did you run this, what did you learn? Notes are saved to the W&B run.',
    fields: [{ name: 'notes', label: 'Notes (Markdown)', type: 'textarea', rows: 7, value: run.notes || '' }],
    onSubmit: async ({ notes }, close) => {
      await mutations.updateRun(run.id, { notes });
      run.notes = notes;
      toast('Notes saved', 'success');
      close();
      onSaved?.(run);
    },
  });
}

/** Per-run "⋯" menu used in lists. */
export function runMenu(anchor, run, { entity, project, onChanged }) {
  openMenu(anchor, [
    { label: 'Open run', icon: 'box-arrow-in-right', href: urls.run(entity, project, run.name) },
    { label: 'Rename…', icon: 'pencil', onClick: () => formModal({
      title: 'Rename run', icon: 'pencil', fields: [{ name: 'displayName', label: 'Display name', value: displayName(run), required: true }],
      onSubmit: async ({ displayName: dn }, close) => { await mutations.updateRun(run.id, { displayName: dn }); run.displayName = dn; toast('Renamed', 'success'); close(); onChanged?.(); },
    }) },
    { label: 'Notes…', icon: 'journal-text', onClick: () => notesDialog(run, onChanged) },
    { label: 'Tags…', icon: 'tags', onClick: () => bulkTagDialog(entity, project, [run], onChanged) },
    { label: 'Compare with…', icon: 'layout-split', href: urls.compare(entity, project, [run.name]) },
    { divider: true },
    { label: 'Move to project…', icon: 'box-arrow-right', onClick: () => bulkMoveDialog(entity, project, [run], onChanged) },
    { label: 'Delete run', icon: 'trash', danger: true, onClick: () => bulkDelete(entity, project, [run], onChanged) },
  ]);
}
