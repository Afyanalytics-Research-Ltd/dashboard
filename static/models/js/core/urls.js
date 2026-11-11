// URL builders for Model Studio pages. Always use these so names are encoded consistently.

import { href } from '../../../ui/js/router.js';

const e = encodeURIComponent;
const p = (entity, project) => `/p/${e(entity)}/${e(project)}`;

export const urls = {
  home: () => href('/'),
  projects: (query) => href('/projects', query),
  workspace: (entity, project, query) => href(p(entity, project), query),
  table: (entity, project, query) => href(`${p(entity, project)}/table`, query),
  run: (entity, project, runName, tab, query) => href(`${p(entity, project)}/runs/${e(runName)}`, { tab, ...query }),
  compare: (entity, project, runNames) => href(`${p(entity, project)}/compare`, { runs: runNames.join(',') }),
  sweeps: (entity, project) => href(`${p(entity, project)}/sweeps`),
  sweep: (entity, project, sweepName, tab) => href(`${p(entity, project)}/sweeps/${e(sweepName)}`, { tab }),
  artifacts: (entity, project, query) => href(`${p(entity, project)}/artifacts`, query),
  collection: (entity, project, type, name) => href(`${p(entity, project)}/artifacts/${e(type)}/${e(name)}`),
  artifact: (entity, project, type, name, version, tab) => href(`${p(entity, project)}/artifacts/${e(type)}/${e(name)}/${e(version)}`, { tab }),
  reports: (entity, project) => href(`${p(entity, project)}/reports`),
  report: (entity, project, viewId) => href(`${p(entity, project)}/reports/${e(viewId)}`),
  projectSettings: (entity, project) => href(`${p(entity, project)}/settings`),
  registry: (query) => href('/registry', query),
  teams: () => href('/teams'),
  team: (entity) => href(`/teams/${e(entity)}`),
  users: (query) => href('/users', query),
  system: () => href('/system'),
  cleanup: (query) => href('/cleanup', query),
  explorer: (query) => href('/explorer', query),
};
