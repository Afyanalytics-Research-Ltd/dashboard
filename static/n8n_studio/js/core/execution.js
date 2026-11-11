// Pure helpers over n8n execution objects (GET /executions/{id}?includeData=true):
// per-node summaries, timeline, error extraction and grouping, run comparison, and
// turning an execution's outputs into pinned data ("debug in editor").
//
// data.resultData.runData[node] = [{startTime, executionTime, executionStatus,
//   data: {main: [[{json, binary, pairedItem}]]}, error, source, metadata}]

import { deepDiff } from './jsondiff.js';

export const STATUS = {
  success: { label: 'Succeeded', tone: 'success', color: '#16a34a', icon: 'check-circle-fill' },
  error: { label: 'Failed', tone: 'failed', color: '#dc2626', icon: 'x-circle-fill' },
  crashed: { label: 'Crashed', tone: 'failed', color: '#b91c1c', icon: 'exclamation-octagon-fill' },
  running: { label: 'Running', tone: 'running', color: '#0891b2', icon: 'arrow-repeat' },
  waiting: { label: 'Waiting', tone: 'scheduled', color: '#a16207', icon: 'pause-circle-fill' },
  new: { label: 'Queued', tone: 'queued', color: '#64748b', icon: 'hourglass-split' },
  canceled: { label: 'Canceled', tone: 'skipped', color: '#94a3b8', icon: 'slash-circle' },
  unknown: { label: 'Unknown', tone: 'none', color: '#94a3b8', icon: 'question-circle' },
};
export const statusInfo = (s) => STATUS[s] || STATUS.unknown;
export const FAILED = new Set(['error', 'crashed']);
export const ACTIVE = new Set(['running', 'waiting', 'new']);

export function durationMs(e) {
  if (!e?.startedAt) return null;
  const a = new Date(e.startedAt).getTime();
  const b = e.stoppedAt ? new Date(e.stoppedAt).getTime() : (ACTIVE.has(e.status) ? Date.now() : a);
  return Math.max(0, b - a);
}

const runData = (e) => e?.data?.resultData?.runData || {};

export function itemsOf(run, output = 0, type = 'main') {
  return run?.data?.[type]?.[output] || [];
}

/** Per-node summary sorted by start time. */
export function nodeSummaries(e) {
  const out = [];
  for (const [name, runs] of Object.entries(runData(e))) {
    (runs || []).forEach((run, runIndex) => {
      const outputs = Object.entries(run.data || {}).flatMap(([type, outs]) => (outs || []).map((items, i) => ({ type, index: i, count: items?.length || 0 })));
      out.push({
        name,
        runIndex,
        runs: runs.length,
        status: run.error ? 'error' : run.executionStatus || 'success',
        start: run.startTime || 0,
        ms: run.executionTime || 0,
        items: outputs.filter((o) => o.type === 'main').reduce((a, o) => a + o.count, 0),
        outputs,
        error: run.error || null,
        hints: run.hints || [],
        source: run.source || [],
        subExecution: run.metadata?.subExecution || null,
        subExecutions: run.metadata?.subExecutions || null,
      });
    });
  }
  return out.sort((a, b) => a.start - b.start || a.runIndex - b.runIndex);
}

/** Latest status per node name (for colouring the canvas). */
export function nodeStatusMap(e) {
  const map = new Map();
  for (const s of nodeSummaries(e)) {
    const prev = map.get(s.name);
    map.set(s.name, {
      status: prev?.status === 'error' ? 'error' : s.status,
      items: (prev?.items || 0) + s.items,
      runs: s.runs,
      ms: (prev?.ms || 0) + s.ms,
      error: s.error || prev?.error || null,
    });
  }
  const last = e?.data?.resultData?.lastNodeExecuted;
  if (last && ACTIVE.has(e.status) && map.has(last)) map.get(last).status = 'running';
  return map;
}

/** Items flowing along each edge "from\u0000outputIndex" -> count (for edge labels). */
export function edgeCounts(e) {
  const map = new Map();
  for (const [name, runs] of Object.entries(runData(e))) {
    for (const run of runs || []) {
      (run.data?.main || []).forEach((items, i) => {
        const k = `${name}\u0000${i}`;
        map.set(k, (map.get(k) || 0) + (items?.length || 0));
      });
    }
  }
  return map;
}

/** The execution's main error: {message, description, node, stack, type} or null. */
export function mainError(e) {
  const err = e?.data?.resultData?.error;
  if (err) {
    return {
      message: err.message || err.description || 'Unknown error',
      description: err.description || null,
      node: err.node?.name || e.data.resultData.lastNodeExecuted || null,
      stack: err.stack || null,
      type: err.name || err.constructor || null,
      httpCode: err.httpCode || err.context?.httpCode || null,
    };
  }
  const failed = nodeSummaries(e).find((s) => s.error);
  if (failed) return { message: failed.error.message || 'Node failed', description: failed.error.description || null, node: failed.name, stack: failed.error.stack || null, type: failed.error.name || null };
  if (FAILED.has(e?.status)) return { message: e.status === 'crashed' ? 'Execution crashed (out of memory or the worker died).' : 'Execution failed.', node: e?.data?.resultData?.lastNodeExecuted || null };
  return null;
}

/** Normalise an error message for grouping (ids, numbers, quoted values, urls collapse). */
export function errorSignature(message) {
  return String(message || '')
    .replace(/https?:\/\/\S+/g, '<url>')
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, '<uuid>')
    .replace(/(["'`]).*?\1/g, '<str>')
    .replace(/\b\d+(\.\d+)?\b/g, '<n>')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 200);
}

/**
 * Group failed executions: [{key, workflowId, node, signature, message, count, first, last, ids}].
 * `errors` maps execution id -> mainError (when detail was loaded); otherwise groups by workflow.
 */
export function groupFailures(executions, errors = new Map()) {
  const groups = new Map();
  for (const e of executions) {
    if (!FAILED.has(e.status)) continue;
    const err = errors.get(String(e.id)) || mainError(e) || {};
    const signature = errorSignature(err.message || '(error details not loaded)');
    const key = `${e.workflowId}\u0000${err.node || ''}\u0000${signature}`;
    let g = groups.get(key);
    if (!g) {
      g = { key, workflowId: e.workflowId, node: err.node || null, signature, message: err.message || null, count: 0, first: e.startedAt, last: e.startedAt, ids: [] };
      groups.set(key, g);
    }
    g.count++;
    g.ids.push(e.id);
    if (e.startedAt < g.first) g.first = e.startedAt;
    if (e.startedAt > g.last) g.last = e.startedAt;
  }
  return mergeSimilar([...groups.values()]).sort((a, b) => b.count - a.count || (b.last > a.last ? 1 : -1));
}

/**
 * Merge groups of the same workflow + node whose signatures differ in exactly one
 * token ("…limit for carl" / "…limit for eve" -> "…limit for <x>").
 */
function mergeSimilar(groups) {
  const out = [];
  for (const g of groups) {
    const tokens = g.signature.split(' ');
    const twin = out.find((o) => {
      if (o.workflowId !== g.workflowId || o.node !== g.node) return false;
      const t = o.signature.split(' ');
      return t.length === tokens.length && t.filter((x, i) => x !== tokens[i]).length === 1;
    });
    if (!twin) { out.push({ ...g, ids: [...g.ids] }); continue; }
    const t = twin.signature.split(' ');
    twin.signature = t.map((x, i) => (x === tokens[i] ? x : '<x>')).join(' ');
    twin.count += g.count;
    twin.ids.push(...g.ids);
    if (g.first < twin.first) twin.first = g.first;
    if (g.last > twin.last) twin.last = g.last;
  }
  return out;
}

/** Compare two executions node by node: status, item counts and first-item JSON diff. */
export function compareExecutions(a, b) {
  const A = runData(a);
  const B = runData(b);
  const names = [...new Set([...Object.keys(A), ...Object.keys(B)])];
  return names.map((name) => {
    const ra = A[name]?.[A[name].length - 1];
    const rb = B[name]?.[B[name].length - 1];
    const ia = itemsOf(ra);
    const ib = itemsOf(rb);
    return {
      name,
      inA: !!ra,
      inB: !!rb,
      statusA: ra ? (ra.error ? 'error' : ra.executionStatus || 'success') : null,
      statusB: rb ? (rb.error ? 'error' : rb.executionStatus || 'success') : null,
      itemsA: ia.length,
      itemsB: ib.length,
      msA: ra?.executionTime ?? null,
      msB: rb?.executionTime ?? null,
      diff: deepDiff(ia[0]?.json ?? null, ib[0]?.json ?? null),
    };
  });
}

/** Pin the outputs of `names` from an execution: {node: [{json}]} (binary is dropped). */
export function toPinData(e, names) {
  const out = {};
  const rd = runData(e);
  for (const name of names || Object.keys(rd)) {
    const runs = rd[name];
    if (!runs?.length) continue;
    const items = itemsOf(runs[runs.length - 1]);
    if (items.length) out[name] = items.map((it) => ({ json: it.json ?? {} }));
  }
  return out;
}

/** Key paths of the JSON items (merged schema) with a sample value and type. */
export function schemaOf(items, { maxDepth = 4, maxKeys = 400 } = {}) {
  const fields = new Map();
  const walk = (v, path, depth) => {
    if (fields.size > maxKeys) return;
    const type = v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v;
    const key = path.join('.');
    if (key) {
      const f = fields.get(key) || { path: key, depth: path.length, types: new Set(), sample: undefined, count: 0 };
      f.types.add(type);
      f.count++;
      if (f.sample === undefined && type !== 'object' && type !== 'array') f.sample = v;
      fields.set(key, f);
    }
    if (depth >= maxDepth) return;
    if (type === 'object') for (const [k, x] of Object.entries(v)) walk(x, [...path, k], depth + 1);
    else if (type === 'array' && v.length && typeof v[0] === 'object' && v[0] !== null) walk(v[0], [...path, '0'], depth + 1);
  };
  for (const it of items || []) walk(it?.json ?? it, [], 0);
  return [...fields.values()].map((f) => ({ ...f, types: [...f.types] }));
}

/** Flatten items into table rows/columns (top-level keys of json). */
export function itemsTable(items, maxCols = 40) {
  const cols = [];
  const seen = new Set();
  for (const it of items || []) {
    for (const k of Object.keys(it?.json ?? {})) {
      if (!seen.has(k)) { seen.add(k); cols.push(k); }
      if (cols.length >= maxCols) break;
    }
  }
  return { columns: cols, rows: (items || []).map((it) => it?.json ?? {}) };
}

export const parentExecution = (e) => e?.data?.parentExecution || e?.data?.executionData?.metadata?.parentExecution || null;

export function subExecutions(e) {
  const out = [];
  for (const s of nodeSummaries(e)) {
    if (s.subExecution) out.push({ node: s.name, ...s.subExecution });
    for (const x of s.subExecutions || []) out.push({ node: s.name, ...x });
  }
  return out;
}

export function triggerNode(e) {
  return e?.data?.executionData?.runtimeData?.triggerNode?.name
    || nodeSummaries(e).find((s) => !s.source?.length)?.name || null;
}

/** Bucket executions per time slot for a stacked status chart. */
export function timeBuckets(executions, { buckets = 24, end = Date.now(), span = 24 * 3600 * 1000 } = {}) {
  const start = end - span;
  const size = span / buckets;
  const out = Array.from({ length: buckets }, (_, i) => ({ t: start + i * size, success: 0, error: 0, other: 0 }));
  for (const e of executions) {
    const t = new Date(e.startedAt || e.createdAt).getTime();
    if (!(t >= start && t < end)) continue;
    const b = out[Math.min(buckets - 1, Math.floor((t - start) / size))];
    if (e.status === 'success') b.success++;
    else if (FAILED.has(e.status)) b.error++;
    else b.other++;
  }
  return out;
}

export function percentile(values, p) {
  const v = values.filter((x) => x != null).sort((a, b) => a - b);
  if (!v.length) return null;
  const idx = Math.min(v.length - 1, Math.max(0, Math.ceil((p / 100) * v.length) - 1));
  return v[idx];
}

/** Per-workflow reliability stats from a list of executions. */
export function workflowStats(executions) {
  const by = new Map();
  for (const e of executions) {
    let s = by.get(e.workflowId);
    if (!s) { s = { workflowId: e.workflowId, total: 0, failed: 0, durations: [], last: null, lastStatus: null }; by.set(e.workflowId, s); }
    s.total++;
    if (FAILED.has(e.status)) s.failed++;
    const d = durationMs(e);
    if (d != null && e.stoppedAt) s.durations.push(d);
    if (!s.last || e.startedAt > s.last) { s.last = e.startedAt; s.lastStatus = e.status; }
  }
  return [...by.values()].map((s) => ({
    ...s,
    failureRate: s.total ? s.failed / s.total : 0,
    p50: percentile(s.durations, 50),
    p95: percentile(s.durations, 95),
  }));
}
