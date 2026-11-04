// Shared GraphQL loaders for Model Studio pages.

import { gql, collect, parseJSON } from './api.js';
import { parseConfig, parseSummary } from './wb.js';
import { bucketsToPoints } from './charts.js';

export const RUN_FIELDS = `
  id name displayName state createdAt heartbeatAt updatedAt group jobType tags notes sweepName
  historyLineCount commit host
  user { username photoUrl }
  config summaryMetrics`;

/** Normalise a raw run node: parsed config/summary + runtime seconds. */
export function normRun(r) {
  const summary = parseSummary(r.summaryMetrics);
  const raw = parseJSON(r.summaryMetrics, {});
  // The SDK's _runtime can be ~0 for short runs; take the larger of it and heartbeat - start.
  const reported = raw._runtime ?? raw['_wandb.runtime'] ?? null;
  const wall = r.heartbeatAt && r.createdAt ? (new Date(r.heartbeatAt) - new Date(r.createdAt)) / 1000 : null;
  const runtime = reported == null && wall == null ? null : Math.max(reported ?? 0, wall ?? 0);
  return { ...r, config: parseConfig(r.config), summary, rawSummary: raw, runtime };
}

export const ENTITY_QUERY = `query Viewer { viewer { id username name email admin photoUrl entity
  teams { edges { node { id name photoUrl isTeam } } } } serverInfo { latestLocalVersionInfo { versionOnThisInstanceString } } }`;

export async function loadViewer(opts) {
  const d = await gql(ENTITY_QUERY, {}, opts);
  const teams = (d.viewer?.teams?.edges || []).map((e) => e.node);
  return { viewer: d.viewer, teams, version: d.serverInfo?.latestLocalVersionInfo?.versionOnThisInstanceString };
}

/** All projects of an entity (newest activity first). */
export function loadProjects(entity, { max = 500, signal } = {}) {
  return collect((after) => gql(`query Projects($e: String!, $after: String) {
    projects(entityName: $e, first: 100, after: $after, order: "-lastActive") {
      pageInfo { hasNextPage endCursor }
      edges { node { id name entityName description createdAt updatedAt lastActive totalRuns access user { username } } } } }`,
  { e: entity, after }, { signal }), (d) => d.projects, { max });
}

/**
 * One page of runs with server-side filter/order. Returns {runs, total, endCursor, hasNextPage}.
 * filters: W&B Mongo JSON string (see wb.buildFilter); order like "-createdAt".
 */
export async function loadRunsPage(entity, project, { filters = '{}', order = '-createdAt', first = 100, after = null, signal } = {}) {
  const d = await gql(`query Runs($e: String!, $p: String!, $f: JSONString, $o: String, $first: Int, $after: String) {
    project(entityName: $e, name: $p) { id runCount(filters: $f)
      runs(filters: $f, order: $o, first: $first, after: $after) {
        pageInfo { hasNextPage endCursor }
        edges { node { ${RUN_FIELDS} } } } } }`,
  { e: entity, p: project, f: filters, o: order, first, after }, { signal });
  if (!d.project) throw Object.assign(new Error(`Project ${entity}/${project} not found`), { status: 404 });
  const conn = d.project.runs;
  return {
    runs: conn.edges.map((e) => normRun(e.node)),
    total: d.project.runCount,
    endCursor: conn.pageInfo.endCursor,
    hasNextPage: conn.pageInfo.hasNextPage,
  };
}

/** Every run matching filters (paged), up to max. onProgress(loaded, total). */
export async function loadAllRuns(entity, project, { filters, order, max = 2000, signal, onProgress } = {}) {
  const runs = [];
  let after = null;
  let total = 0;
  for (;;) {
    const page = await loadRunsPage(entity, project, { filters, order, after, first: 100, signal });
    runs.push(...page.runs);
    total = page.total;
    onProgress?.(runs.length, total);
    if (!page.hasNextPage || runs.length >= max) break;
    after = page.endCursor;
  }
  return { runs, total };
}

/** Metric keys present in a project (union over a sample of runs' historyKeys). */
export async function loadMetricKeys(entity, project, runNames, { signal } = {}) {
  if (!runNames.length) return [];
  // NB: W&B ignores runs(names: ...) — select runs via filters instead.
  const d = await gql(`query Keys($e: String!, $p: String!, $f: JSONString) {
    project(entityName: $e, name: $p) { runs(filters: $f, first: 50) { edges { node { name historyKeys } } } } }`,
  { e: entity, p: project, f: JSON.stringify({ name: { $in: runNames.slice(0, 50) } }) }, { signal });
  const keys = new Map();
  for (const e of d.project?.runs?.edges || []) {
    const hk = e.node.historyKeys?.keys || {};
    for (const [k, info] of Object.entries(hk)) {
      const types = (info.typeCounts || []).map((t) => t.type);
      const numeric = !types.length || types.some((t) => t === 'number');
      if (!keys.has(k)) keys.set(k, { key: k, numeric, types: new Set(types) });
      else types.forEach((t) => keys.get(k).types.add(t));
      if (numeric) keys.get(k).numeric = true;
    }
  }
  return [...keys.values()].map((k) => ({ ...k, types: [...k.types] }));
}

/**
 * Downsampled histories for many runs × metrics in ONE request (aliases per run).
 * Returns Map(runName -> Map(metricKey -> points[{x,y,min,max}])).
 */
export async function loadHistories(entity, project, runNames, keys, { xKey = '_step', samples = 500, signal } = {}) {
  const out = new Map();
  if (!runNames.length || !keys.length) return out;
  const specs = keys.map((k) => JSON.stringify({ keys: [xKey, k], samples }));
  const chunk = 25;
  for (let i = 0; i < runNames.length; i += chunk) {
    const names = runNames.slice(i, i + chunk);
    const d = await gql(`query Hist($e: String!, $p: String!, $f: JSONString, $specs: [JSONString!]!) {
      project(entityName: $e, name: $p) { runs(filters: $f, first: ${chunk}) { edges { node { name bucketedHistory(specs: $specs) } } } } }`,
    { e: entity, p: project, f: JSON.stringify({ name: { $in: names } }), specs }, { signal });
    for (const e of d.project?.runs?.edges || []) {
      const perKey = new Map();
      (e.node.bucketedHistory || []).forEach((rows, idx) => perKey.set(keys[idx], bucketsToPoints(rows, keys[idx], xKey)));
      out.set(e.node.name, perKey);
    }
  }
  return out;
}

/** Full single run (detail page). */
export async function loadRun(entity, project, runName, { signal } = {}) {
  const d = await gql(`query Run($e: String!, $p: String!, $r: String!) {
    project(entityName: $e, name: $p) { id name entityName
      run(name: $r) { ${RUN_FIELDS} description exitcode stopped running failed fileCount logLineCount eventsLineCount
        storageBytes computeSeconds github patch framework wandbConfig systemMetrics historyKeys
        sweep { name displayName method state } } } }`,
  { e: entity, p: project, r: runName }, { signal });
  if (!d.project?.run) throw Object.assign(new Error(`Run ${runName} not found`), { status: 404 });
  const run = normRun(d.project.run);
  run.projectId = d.project.id;
  run.systemMetrics = parseJSON(run.systemMetrics, {});
  run.wandbConfig = parseJSON(run.wandbConfig, {});
  return run;
}

export async function loadSweeps(entity, project, { signal } = {}) {
  return collect((after) => gql(`query Sweeps($e: String!, $p: String!, $after: String) {
    project(entityName: $e, name: $p) { sweeps(first: 50, after: $after, order: "-createdAt") {
      pageInfo { hasNextPage endCursor }
      edges { node { id name displayName state method runCount runCountExpected bestLoss config createdAt updatedAt heartbeatAt user { username } } } } } }`,
  { e: entity, p: project, after }, { signal }), (d) => d.project?.sweeps, { max: 500 });
}

const byNames = (runNames) => JSON.stringify({ name: { $in: runNames } });

/** Mutations used across pages (input shapes checked against the W&B 0.86 schema). */
export const mutations = {
  /** fields: { displayName, notes, tags, groupName, jobType, description } */
  updateRun: (id, fields) => gql(`mutation UpdateRun($input: UpsertBucketInput!) {
    upsertBucket(input: $input) { bucket { id name displayName notes tags group jobType } } }`, { input: { id, ...fields } }),
  deleteRun: (id, deleteArtifacts = false) => gql(`mutation DeleteRun($input: DeleteRunInput!) {
    deleteRun(input: $input) { clientMutationId } }`, { input: { id, deleteArtifacts } }),
  deleteRuns: (entity, project, runNames, deleteArtifacts = false) => gql(`mutation DeleteRuns($input: DeleteRunsInput!) {
    deleteRuns(input: $input) { clientMutationId } }`,
  { input: { entityName: entity, projectName: project, filters: byNames(runNames), deleteArtifacts } }),
  /**
   * Bulk tag edit via modifyRuns. W&B rejects a call that carries an empty addTags or
   * removeTags list, so each non-empty side is sent as its own call.
   */
  async tagRuns(entity, project, runNames, { addTags = [], removeTags = [] }) {
    const base = { entityName: entity, projectName: project, filters: byNames(runNames) };
    const doc = 'mutation TagRuns($input: ModifyRunsInput!) { modifyRuns(input: $input) { clientMutationId } }';
    if (addTags.length) await gql(doc, { input: { ...base, addTags } });
    if (removeTags.length) await gql(doc, { input: { ...base, removeTags } });
  },
  /** Move runs to another project (e.g. an archive project to keep workspaces fast). */
  moveRuns: (entity, project, runNames, destEntity, destProject) => gql(`mutation MoveRuns($input: MoveRunsInput!) {
    moveRuns(input: $input) { clientMutationId } }`,
  { input: { sourceEntityName: entity, sourceProjectName: project, filters: byNames(runNames), destinationEntityName: destEntity, destinationProjectName: destProject } }),
  stopRun: (id) => gql(`mutation StopRun($input: StopRunInput!) { stopRun(input: $input) { success } }`, { input: { id } }),
};
