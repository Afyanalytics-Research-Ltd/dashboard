// URL builders for Studio pages. Always use these so ids are encoded consistently.

import { href } from '../../../ui/js/router.js';

const e = encodeURIComponent;

export const urls = {
  home: () => href('/'),
  queries: (query) => href('/queries', query),
  newQuery: (query) => href('/queries/new', query),
  query: (id, query) => href(`/queries/${e(id)}`, query),
  queryViz: (id, vizId) => href(`/queries/${e(id)}`, { viz: vizId }),
  dashboards: (query) => href('/dashboards', query),
  dashboard: (idOrSlug, query) => href(`/dashboards/${e(idOrSlug)}`, query),
  alerts: (query) => href('/alerts', query),
  newAlert: (query) => href('/alerts/new', query),
  alert: (id) => href(`/alerts/${e(id)}`),
  dataSources: () => href('/data-sources'),
  dataSource: (id) => href(`/data-sources/${e(id)}`),
  destinations: () => href('/destinations'),
  users: (query) => href('/users', query),
  user: (id) => href(`/users/${e(id)}`),
  groups: () => href('/groups'),
  group: (id) => href(`/groups/${e(id)}`),
  snippets: () => href('/snippets'),
  settings: () => href('/settings'),
  events: (query) => href('/events', query),
  system: () => href('/system'),
  cleanup: () => href('/cleanup'),
  explorer: (query) => href('/explorer', query),
};
