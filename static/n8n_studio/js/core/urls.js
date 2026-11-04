// App-internal URLs (all under /automations/).
import { href } from '../../../ui/js/router.js';

const e = (v) => encodeURIComponent(String(v));

export const urls = {
  home: () => href('/'),
  workflows: (q) => href('/workflows', q),
  workflow: (id, q) => href(`/workflows/${e(id)}`, q),
  newWorkflow: (q) => href('/workflows/new', q),
  executions: (q) => href('/executions', q),
  execution: (id, q) => href(`/executions/${e(id)}`, q),
  failures: (q) => href('/failures', q),
  credentials: (q) => href('/credentials', q),
  tags: () => href('/tags'),
  variables: () => href('/variables'),
  projects: () => href('/projects'),
  project: (id) => href(`/projects/${e(id)}`),
  dataTables: () => href('/data-tables'),
  dataTable: (id) => href(`/data-tables/${e(id)}`),
  users: () => href('/users'),
  roles: () => href('/roles'),
  insights: (q) => href('/insights', q),
  audit: () => href('/audit'),
  packages: () => href('/packages'),
  logStreaming: () => href('/log-streaming'),
  settings: (tab) => href('/settings', tab ? { tab } : undefined),
  backups: (q) => href('/backups', q),
  history: (q) => href('/history', q),
  explorer: (q) => href('/explorer', q),
  system: () => href('/system'),
};
