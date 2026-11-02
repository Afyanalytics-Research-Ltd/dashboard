// Reusable Airflow actions with their dialogs. Every function resolves to the
// API result (or null when the user cancels) and shows its own toasts.

import { h, icon, mount } from './dom.js';
import { api, enc } from './api.js';
import { navigate } from './router.js';
import { urls } from './urls.js';
import {
  buildForm, confirmDialog, dataTable, formModal, jsonEditor, modal, stateBadge, tabs, toast, toastError,
} from './ui.js';

export const runPath = (dagId, runId) => `/dags/${enc(dagId)}/dagRuns/${enc(runId)}`;
export const tiPath = (dagId, runId, taskId, mapIndex) =>
  `${runPath(dagId, runId)}/taskInstances/${enc(taskId)}${mapIndex != null && mapIndex >= 0 ? `/${mapIndex}` : ''}`;

// ------------------------------------------------------------------ DAG

export async function setPaused(dag, paused) {
  try {
    const res = await api.patch(`/dags/${enc(dag.dag_id)}`, { is_paused: paused });
    toast(`${dag.dag_id} ${paused ? 'paused' : 'unpaused'}`, 'success');
    return res;
  } catch (err) {
    toastError(err, `Could not ${paused ? 'pause' : 'unpause'} ${dag.dag_id}`);
    return null;
  }
}

export async function setPausedBulk(dagIds, paused) {
  const results = await Promise.allSettled(dagIds.map((id) => api.patch(`/dags/${enc(id)}`, { is_paused: paused })));
  const failed = results.filter((r) => r.status === 'rejected').length;
  if (failed) toast(`${failed} of ${dagIds.length} failed`, 'warning', { title: `Bulk ${paused ? 'pause' : 'unpause'}` });
  else toast(`${dagIds.length} DAG(s) ${paused ? 'paused' : 'unpaused'}`, 'success');
  return results;
}

export async function setFavorite(dagId, favorite) {
  try {
    await api.post(`/dags/${enc(dagId)}/${favorite ? 'favorite' : 'unfavorite'}`, undefined);
    toast(favorite ? 'Added to favourites' : 'Removed from favourites', 'success', { timeout: 2000 });
    return true;
  } catch (err) {
    toastError(err, 'Favourite failed');
    return false;
  }
}

export async function deleteDag(dagId) {
  const ok = await confirmDialog({
    title: `Delete DAG ${dagId}?`,
    message: 'This deletes all metadata for this DAG (runs, task instances, log references). The DAG file is not removed, so the DAG reappears on the next parse.',
    confirmLabel: 'Delete DAG', danger: true,
  });
  if (!ok) return null;
  try {
    await api.del(`/dags/${enc(dagId)}`);
    toast(`DAG ${dagId} deleted`, 'success');
    return true;
  } catch (err) { toastError(err, 'Delete failed'); return null; }
}

export async function reparseDag(dag) {
  if (!dag.file_token) { toast('No file token for this DAG', 'warning'); return null; }
  try {
    await api.put(`/parseDagFile/${enc(dag.file_token)}`, undefined);
    toast(`Re-parse requested for ${dag.relative_fileloc || dag.dag_id}`, 'success');
    return true;
  } catch (err) { toastError(err, 'Re-parse failed'); return null; }
}

// ------------------------------------------------------------------ trigger

/** Build a form field from an Airflow Param ({value, description, schema}). */
function paramField(name, param) {
  const p = param && typeof param === 'object' && 'schema' in param ? param : { value: param, schema: {} };
  const schema = p.schema || {};
  let type = Array.isArray(schema.type) ? schema.type.find((t) => t !== 'null') : schema.type;
  if (!type) type = Array.isArray(p.value) ? 'array' : p.value === null ? 'string' : typeof p.value === 'object' ? 'object' : typeof p.value;
  const label = schema.title || name;
  const help = p.description || schema.description;
  const base = { name, label: label === name ? name : `${label} (${name})`, help };
  if (schema.enum) return { ...base, type: 'select', options: schema.enum.map((v) => ({ value: JSON.stringify(v), label: String(v) })), value: JSON.stringify(p.value), kind: 'enum' };
  if (type === 'boolean') return { ...base, type: 'checkbox', value: !!p.value, kind: 'boolean' };
  if (type === 'integer' || type === 'number') return { ...base, type: 'number', value: p.value ?? '', kind: type, min: schema.minimum, max: schema.maximum };
  if (type === 'array' && (!schema.items || ['string', 'integer', 'number'].includes(schema.items?.type))) {
    return { ...base, type: 'textarea', rows: 3, value: (p.value || []).join('\n'), kind: 'list', itemType: schema.items?.type, help: `${help ? `${help} — ` : ''}one value per line` };
  }
  if (type === 'array' || type === 'object') return { ...base, type: 'json', value: p.value ?? (type === 'array' ? [] : {}), kind: 'json', rows: 5 };
  return { ...base, type: (schema.maxLength || 0) > 200 ? 'textarea' : 'text', value: p.value ?? '', kind: 'string' };
}

function coerceParam(field, raw) {
  switch (field.kind) {
    case 'enum': return JSON.parse(raw);
    case 'integer': return raw == null ? null : Math.trunc(raw);
    case 'list': {
      const items = String(raw || '').split('\n').map((x) => x.trim()).filter(Boolean);
      return field.itemType === 'integer' || field.itemType === 'number' ? items.map(Number) : items;
    }
    case 'string': return raw === '' ? null : raw;
    default: return raw;
  }
}

/**
 * Trigger dialog: a typed form generated from the DAG's params, a raw JSON
 * mode, and the advanced TriggerDAGRunPostBody fields.
 */
export async function triggerDag(dagId, { details, prefillConf, onDone, openRun = true } = {}) {
  let dag = details;
  if (!dag) {
    try { dag = await api.get(`/dags/${enc(dagId)}/details`); } catch (err) { toastError(err, 'Could not load DAG'); return null; }
  }
  const params = dag.params || {};
  const names = Object.keys(params);
  const fields = names.map((n) => paramField(n, params[n]));
  if (prefillConf) {
    fields.forEach((f) => {
      if (!(f.name in prefillConf)) return;
      const v = prefillConf[f.name];
      if (f.kind === 'list') f.value = (v || []).join('\n');
      else if (f.kind === 'enum') f.value = JSON.stringify(v);
      else f.value = v;
    });
  }
  const paramsForm = fields.length ? buildForm(fields) : null;
  const defaults = Object.fromEntries(names.map((n) => [n, params[n] && typeof params[n] === 'object' && 'value' in params[n] ? params[n].value : params[n]]));
  const confEditor = jsonEditor(prefillConf || defaults, { rows: 12 });
  const advanced = buildForm([
    { name: 'dag_run_id', label: 'Run id', placeholder: 'auto-generated (manual__…)', help: 'Set your own id to make retries idempotent. Duplicates return 409.' },
    { name: 'logical_date', label: 'Logical date', type: 'datetime', help: 'Empty = none ("now"). Must be unique per DAG when set.' },
    { name: 'note', label: 'Note', type: 'textarea', rows: 2, value: 'Triggered from Afya DataHub' },
  ]);

  let mode = fields.length ? 'form' : 'json';
  const panes = { form: h('div', paramsForm?.el), json: h('div', confEditor.el), advanced: h('div', advanced.el) };
  const host = h('div.af-trigger-pane');
  const collectForm = () => {
    const raw = paramsForm.values();
    return Object.fromEntries(fields.map((f) => [f.name, coerceParam(f, raw[f.name])]));
  };
  const show = (id) => {
    if (id === 'json' && mode === 'form' && paramsForm) {
      try { confEditor.set(collectForm()); } catch { /* keep editor contents */ }
    }
    if (id !== 'advanced') mode = id;
    mount(host, panes[id]);
  };
  const tabItems = [
    fields.length && { id: 'form', label: 'Parameters', icon: 'ui-checks', count: fields.length },
    { id: 'json', label: 'Raw conf JSON', icon: 'braces' },
    { id: 'advanced', label: 'Run options', icon: 'gear' },
  ].filter(Boolean);
  show(mode);

  return new Promise((resolve) => {
    let done = false;
    async function submit(close) {
      const conf = mode === 'form' && paramsForm ? collectForm() : (confEditor.value() || {});
      if (typeof conf !== 'object' || Array.isArray(conf)) throw new Error('conf must be a JSON object');
      const adv = advanced.values();
      const body = { logical_date: adv.logical_date || null, conf, note: adv.note || null };
      if (adv.dag_run_id) body.dag_run_id = adv.dag_run_id;
      const run = await api.post(`/dags/${enc(dagId)}/dagRuns`, body);
      done = true;
      close();
      toast(`Run ${run.dag_run_id} is ${run.state}`, 'success', {
        title: `${dagId} triggered`,
        action: openRun ? null : { label: 'View run →', onClick: () => navigate(urls.run(dagId, run.dag_run_id)) },
      });
      onDone?.(run);
      resolve(run);
      if (openRun) navigate(urls.run(dagId, run.dag_run_id));
    }
    modal({
      title: `Trigger ${dagId}`,
      subtitle: dag.description || null,
      icon: 'play-circle',
      size: 'lg',
      body: h('div',
        dag.is_paused && h('div.alert.alert-warning.py-2.small.d-flex.align-items-center.gap-2', icon('pause-circle'),
          'This DAG is paused — the run stays queued until it is unpaused.'),
        tabs(tabItems, mode, show),
        host),
      actions: [
        { label: 'Cancel' },
        dag.is_paused && {
          label: 'Unpause & trigger', variant: 'outline-primary', icon: 'play',
          onClick: async (close) => {
            await api.patch(`/dags/${enc(dagId)}`, { is_paused: false });
            await submit(close);
          },
        },
        { label: 'Trigger', variant: 'primary', icon: 'play-fill', onClick: submit },
      ].filter(Boolean),
      onClose: () => { if (!done) resolve(null); },
    });
  });
}

// ------------------------------------------------------------------ runs

export async function setRunState(dagId, runId, state) {
  const ok = await confirmDialog({
    title: `Mark run as ${state}?`,
    message: h('span', 'Run ', h('code', runId), ' will be set to ', stateBadge(state),
      state === 'failed' ? '. Running tasks are stopped.' : '.'),
    confirmLabel: `Mark ${state}`, danger: state === 'failed',
  });
  if (!ok) return null;
  try {
    const res = await api.patch(runPath(dagId, runId), { state }, { params: { update_mask: 'state' } });
    toast(`Run marked ${state}`, 'success');
    return res;
  } catch (err) { toastError(err, 'Update failed'); return null; }
}

export function editRunNote(dagId, run) {
  return new Promise((resolve) => {
    formModal({
      title: 'Run note', icon: 'sticky',
      fields: [{ name: 'note', label: 'Note (Markdown)', type: 'textarea', rows: 6, value: run.note || '' }],
      onSubmit: async ({ note }, close) => {
        const res = await api.patch(runPath(dagId, run.dag_run_id), { note }, { params: { update_mask: 'note' } });
        toast('Note saved', 'success');
        close();
        resolve(res);
      },
    });
  });
}

function renderAffected(el, tis) {
  mount(el,
    h('div.small.fw-semibold.mb-1', `${tis.length} task instance(s) affected`),
    tis.length ? dataTable({
      dense: true,
      rows: tis,
      rowKey: (t) => `${t.dag_run_id}|${t.task_id}|${t.map_index}`,
      columns: [
        { key: 'task_id', label: 'Task', render: (t) => h('span.af-mono', t.task_id, t.map_index >= 0 ? `[${t.map_index}]` : '') },
        { key: 'dag_run_id', label: 'Run', render: (t) => h('span.af-mono.small', t.dag_run_id) },
        { key: 'state', label: 'Current state', render: (t) => stateBadge(t.state) },
      ],
    }) : null);
}

/** Generic "options form + dry-run preview + confirm" dialog. */
function previewDialog({ title, subtitle, icon: ic, form, run, confirmLabel, success }) {
  const preview = h('div.af-preview');
  const doPreview = async () => renderAffected(preview, (await run(true)).task_instances || []);
  return new Promise((resolve) => {
    let done = false;
    modal({
      title, subtitle, icon: ic, size: 'lg',
      body: h('div', form.el, preview),
      actions: [
        { label: 'Cancel' },
        { label: 'Preview', icon: 'eye', onClick: doPreview },
        {
          label: confirmLabel, variant: 'primary', icon: ic,
          onClick: async (close) => {
            const res = await run(false);
            done = true;
            toast(success(res), 'success');
            close();
            resolve(res);
          },
        },
      ],
      onClose: () => { if (!done) resolve(null); },
    });
    doPreview().catch((err) => mount(preview, h('div.text-danger.small', err.message)));
    form.el.addEventListener('change', () => doPreview().catch((err) => mount(preview, h('div.text-danger.small', err.message))));
  });
}

/** Clear (re-run) a whole DAG run, previewing affected task instances first. */
export function clearRun(dagId, runId) {
  const form = buildForm([
    { name: 'only_failed', type: 'checkbox', checkLabel: 'Only failed tasks', value: true },
    { name: 'run_on_latest_version', type: 'checkbox', checkLabel: 'Run on latest DAG version', value: false },
  ]);
  return previewDialog({
    title: 'Clear DAG run', subtitle: runId, icon: 'arrow-counterclockwise', form, confirmLabel: 'Clear',
    run: (dry) => api.post(`${runPath(dagId, runId)}/clear`, { ...form.values(), dry_run: dry }),
    success: () => 'Run cleared — tasks will be re-scheduled',
  });
}

export async function deleteRun(dagId, runId) {
  const ok = await confirmDialog({
    title: 'Delete DAG run?', danger: true, confirmLabel: 'Delete run',
    message: h('span', 'This permanently deletes run ', h('code', runId), ' and its task instances.'),
  });
  if (!ok) return null;
  try {
    await api.del(runPath(dagId, runId));
    toast('Run deleted', 'success');
    return true;
  } catch (err) { toastError(err, 'Delete failed'); return null; }
}

// ------------------------------------------------------------------ task instances

const relatives = (values = {}) => [
  { type: 'section', label: 'Also include' },
  { name: 'include_upstream', type: 'checkbox', checkLabel: 'Upstream', value: values.include_upstream ?? false },
  { name: 'include_downstream', type: 'checkbox', checkLabel: 'Downstream', value: values.include_downstream ?? false },
  { name: 'include_past', type: 'checkbox', checkLabel: 'Past runs', value: false },
  { name: 'include_future', type: 'checkbox', checkLabel: 'Future runs', value: false },
];

/** Mark a task instance (and optionally relatives) with a new state. */
export function markTask(ti, newState = 'success') {
  const form = buildForm([
    { name: 'new_state', label: 'New state', type: 'select', options: ['success', 'failed', 'skipped'], value: newState },
    ...relatives(),
    { name: 'note', label: 'Note (optional)', type: 'textarea', rows: 2 },
  ]);
  form.el.classList.add('af-form-relatives');
  const path = tiPath(ti.dag_id, ti.dag_run_id, ti.task_id, ti.map_index);
  const body = () => {
    const b = form.values();
    if (!b.note) delete b.note;
    return b;
  };
  return previewDialog({
    title: 'Mark task instance',
    subtitle: `${ti.task_id}${ti.map_index >= 0 ? ` [${ti.map_index}]` : ''} · ${ti.dag_run_id}`,
    icon: 'check2-square', form, confirmLabel: 'Apply',
    run: (dry) => {
      const b = body();
      return dry
        ? api.patch(`${path}/dry_run`, b)
        : api.patch(path, b, { params: { update_mask: b.note ? ['new_state', 'note'] : ['new_state'] } });
    },
    success: () => `Marked ${form.values().new_state}`,
  });
}

/** Clear task instance(s) via POST /dags/{dag_id}/clearTaskInstances. */
export function clearTasks(dagId, { runId, taskIds, mapIndexes } = {}) {
  const form = buildForm([
    ...relatives({ include_downstream: true }),
    { type: 'section', label: 'Options' },
    { name: 'only_failed', type: 'checkbox', checkLabel: 'Only failed', value: false },
    { name: 'only_running', type: 'checkbox', checkLabel: 'Only running', value: false },
    { name: 'reset_dag_runs', type: 'checkbox', checkLabel: 'Reset DAG runs to queued', value: true },
    { name: 'run_on_latest_version', type: 'checkbox', checkLabel: 'Run on latest DAG version', value: false },
  ]);
  form.el.classList.add('af-form-relatives');
  const body = (dry) => {
    const b = { ...form.values(), dry_run: dry };
    if (runId) b.dag_run_id = runId;
    if (taskIds?.length) b.task_ids = mapIndexes ? taskIds.map((t, i) => (mapIndexes[i] >= 0 ? [t, mapIndexes[i]] : t)) : taskIds;
    return b;
  };
  return previewDialog({
    title: 'Clear task instances', subtitle: taskIds?.length ? taskIds.join(', ') : dagId,
    icon: 'arrow-counterclockwise', form, confirmLabel: 'Clear',
    run: (dry) => api.post(`/dags/${enc(dagId)}/clearTaskInstances`, body(dry)),
    success: (res) => `${(res.task_instances || []).length} task instance(s) cleared`,
  });
}

export function editTaskNote(ti) {
  return new Promise((resolve) => {
    formModal({
      title: 'Task instance note', icon: 'sticky',
      fields: [{ name: 'note', label: 'Note (Markdown)', type: 'textarea', rows: 6, value: ti.note || '' }],
      onSubmit: async ({ note }, close) => {
        const res = await api.patch(tiPath(ti.dag_id, ti.dag_run_id, ti.task_id, ti.map_index), { note }, { params: { update_mask: 'note' } });
        toast('Note saved', 'success');
        close();
        resolve(res);
      },
    });
  });
}

export async function deleteTaskInstance(ti) {
  const ok = await confirmDialog({
    title: 'Delete task instance?', danger: true, confirmLabel: 'Delete',
    message: h('span', 'Delete ', h('code', ti.task_id), ' from run ', h('code', ti.dag_run_id), '?'),
  });
  if (!ok) return null;
  try {
    await api.del(tiPath(ti.dag_id, ti.dag_run_id, ti.task_id), { params: ti.map_index >= 0 ? { map_index: ti.map_index } : undefined });
    toast('Task instance deleted', 'success');
    return true;
  } catch (err) { toastError(err, 'Delete failed'); return null; }
}

/** Bulk-update task instances of one run with PATCH .../taskInstances (BulkBody). */
export async function bulkTaskState(dagId, runId, tis, newState) {
  const ok = await confirmDialog({
    title: `Mark ${tis.length} task instance(s) ${newState}?`, confirmLabel: `Mark ${newState}`, danger: newState === 'failed',
    message: tis.map((t) => t.task_id).slice(0, 12).join(', ') + (tis.length > 12 ? '…' : ''),
  });
  if (!ok) return null;
  try {
    const res = await api.patch(`${runPath(dagId, runId)}/taskInstances`, {
      actions: [{
        action: 'update',
        entities: tis.map((t) => ({ task_id: t.task_id, map_index: t.map_index, new_state: newState })),
        update_mask: ['new_state'],
        action_on_non_existence: 'skip',
      }],
    });
    const errs = res?.update?.errors || [];
    if (errs.length) toast(`${errs.length} failed: ${errs[0].error || JSON.stringify(errs[0])}`, 'warning');
    else toast(`${tis.length} task instance(s) marked ${newState}`, 'success');
    return res;
  } catch (err) { toastError(err, 'Bulk update failed'); return null; }
}
