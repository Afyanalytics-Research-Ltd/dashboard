// Sweep data helpers shared by the sweeps list and the sweep page.

import { h, icon } from '../../../../ui/js/dom.js';
import { gql } from '../../core/api.js';
import { RUN_FIELDS, normRun } from '../../core/data.js';
import { parseYaml } from './shared.js';

/** {metricName, goal, method, parameters:{name: spec}, runCap, name, raw} */
export function sweepConfig(sweep) {
  const cfg = parseYaml(sweep?.config || '');
  const metric = cfg.metric || {};
  return {
    raw: cfg,
    metricName: metric.name || null,
    goal: String(metric.goal || 'minimize').toLowerCase().startsWith('max') ? 'maximize' : 'minimize',
    method: String(cfg.method || sweep?.method || '').toLowerCase(),
    parameters: cfg.parameters || {},
    runCap: cfg.run_cap ?? sweep?.runCountExpected ?? null,
    name: cfg.name || sweep?.displayName || sweep?.name,
  };
}

/** Parameters that actually vary (skip fixed "value:" ones). */
export function tunedParams(parameters) {
  return Object.entries(parameters || {}).filter(([, spec]) => spec && typeof spec === 'object' && !('value' in spec)).map(([k]) => k);
}

export function describeParam(spec) {
  if (!spec || typeof spec !== 'object') return String(spec);
  if ('value' in spec) return `fixed = ${JSON.stringify(spec.value)}`;
  if (Array.isArray(spec.values)) return `one of ${spec.values.map((v) => JSON.stringify(v)).join(', ')}`;
  const dist = spec.distribution || (Number.isInteger(spec.min) && Number.isInteger(spec.max) ? 'int_uniform' : 'uniform');
  const range = spec.min != null || spec.max != null ? ` in [${spec.min ?? '−∞'}, ${spec.max ?? '∞'}]` : '';
  const extra = spec.mu != null ? ` μ=${spec.mu} σ=${spec.sigma}` : '';
  return `${dist.replace(/_/g, ' ')}${range}${extra}`;
}

export const isLogParam = (spec) => /log/.test(spec?.distribution || '');

export async function loadSweepRuns(entity, project, sweepName, { signal, max = 1000 } = {}) {
  const runs = [];
  let after = null;
  let sweep = null;
  for (;;) {
    const d = await gql(`query SweepRuns($e: String!, $p: String!, $s: String!, $after: String) { project(entityName: $e, name: $p) {
      sweep(sweepName: $s) { id name displayName state method config bestLoss runCount runCountExpected createdAt updatedAt heartbeatAt description
        user { username }
        runs(first: 100, after: $after, order: "+createdAt") { pageInfo { hasNextPage endCursor } edges { node { ${RUN_FIELDS} } } } } } }`,
    { e: entity, p: project, s: sweepName, after }, { signal });
    sweep = d.project?.sweep;
    if (!sweep) throw Object.assign(new Error(`Sweep ${sweepName} not found`), { status: 404 });
    runs.push(...sweep.runs.edges.map((e) => {
      const r = normRun(e.node);
      // Some SDK versions report _runtime 0; fall back to wall-clock between start and last heartbeat.
      if (!r.runtime && r.heartbeatAt && r.createdAt) r.runtime = Math.max(0, (new Date(r.heartbeatAt) - new Date(r.createdAt)) / 1000);
      return r;
    }));
    if (!sweep.runs.pageInfo.hasNextPage || runs.length >= max) break;
    after = sweep.runs.pageInfo.endCursor;
  }
  const { runs: _runs, ...meta } = sweep;
  return { sweep: meta, runs };
}

/** Metric value of a run from its summary. */
export function metricOf(run, metricName) {
  if (!metricName) return null;
  const v = run.summary?.[metricName];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** Best-so-far series over runs ordered by creation. */
export function bestSoFar(runs, metricName, goal) {
  let best = null;
  return [...runs].sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt)).map((r, i) => {
    const v = metricOf(r, metricName);
    if (v != null) best = best == null ? v : goal === 'maximize' ? Math.max(best, v) : Math.min(best, v);
    return { i, run: r, value: v, best };
  });
}

const STATE_ICON = { running: 'play-fill', paused: 'pause-fill', finished: 'check2', canceled: 'x', cancelled: 'x', pending: 'hourglass', failed: 'x-octagon', crashed: 'x-octagon' };
export function sweepStateBadge(state) {
  const s = String(state || 'unknown').toLowerCase();
  return h('span.ma-state', { class: s }, s === 'running' ? h('span.af-dot', { style: { background: '#0891b2' } }) : icon(STATE_ICON[s] || 'circle'), s);
}

/** Change sweep state via upsertSweep (RUNNING / PAUSED / FINISHED / CANCELED). */
export async function setSweepState(sweepId, state) {
  const d = await gql(`mutation SweepState($input: UpsertSweepInput!) { upsertSweep(input: $input) { sweep { id state } } }`, { input: { id: sweepId, state } });
  return d.upsertSweep?.sweep;
}
