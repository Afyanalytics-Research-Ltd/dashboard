// URL builders for app pages. Always use these so ids are encoded consistently.

import { href } from './router.js';

const e = encodeURIComponent;

export const urls = {
  home: () => href('/'),
  dags: (query) => href('/dags', query),
  dag: (dagId, tab, query) => href(`/dags/${e(dagId)}`, { tab, ...query }),
  run: (dagId, runId, tab) => href(`/dags/${e(dagId)}/runs/${e(runId)}`, { tab }),
  task: (dagId, runId, taskId, mapIndex = -1, query) =>
    href(`/dags/${e(dagId)}/runs/${e(runId)}/tasks/${e(taskId)}${mapIndex != null && Number(mapIndex) >= 0 ? `/${mapIndex}` : ''}`, query),
  runs: (query) => href('/runs', query),
  taskInstances: (query) => href('/task-instances', query),
  assets: (query) => href('/assets', query),
  asset: (assetId, query) => href(`/assets/${e(assetId)}`, query),
  backfills: (query) => href('/backfills', query),
  connections: (query) => href('/connections', query),
  variables: (query) => href('/variables', query),
  pools: () => href('/pools'),
  xcoms: (query) => href('/xcoms', query),
  hitl: (query) => href('/hitl', query),
  audit: (query) => href('/audit', query),
  jobs: (query) => href('/jobs', query),
  importErrors: () => href('/import-errors'),
  warnings: () => href('/warnings'),
  plugins: () => href('/plugins'),
  providers: () => href('/providers'),
  config: () => href('/config'),
  health: () => href('/health'),
  explorer: (query) => href('/explorer', query),
};
