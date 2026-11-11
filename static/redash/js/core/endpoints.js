// Catalog of every Redash 10.1 API operation (76 routes / 123 operations), built from the
// server's route table (redash/handlers/api.py + blueprint routes) and handler source.
// Paths use {param} placeholders and are relative to the Redash base URL (called via the proxy).

const PAGED = [
  { name: 'page', type: 'integer', desc: 'Page number (1-based).', example: 1 },
  { name: 'page_size', type: 'integer', desc: 'Items per page (max 250).', example: 25 },
  { name: 'q', type: 'string', desc: 'Full-text search term.' },
  { name: 'order', type: 'string', desc: 'Sort field, prefix "-" for descending (name, created_at, executed_at, runtime, schedule, created_by).' },
];
const TAGS = [{ name: 'tags', type: 'string[]', desc: 'Only objects with all of these tags (repeatable).' }];
const MAX_AGE = [{ name: 'maxAge', type: 'integer', desc: 'Return the cached result only if younger than this many seconds.' }];

const QUERY_BODY = {
  name: 'zz_studio_test example', query: 'SELECT 1 AS one', data_source_id: 1, description: '',
  options: { parameters: [] }, schedule: null, tags: [], is_draft: true,
};
const EXEC_BODY = { parameters: {}, max_age: 0, apply_auto_limit: false };
const SCHEDULE_DOC = 'schedule: {interval: seconds, time: "HH:MM" (UTC), day_of_week, until: "YYYY-MM-DD"} or null';

function op(method, path, group, summary, extra = {}) {
  const params = [...path.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
  return {
    id: `${method.toLowerCase()}_${path.replace(/^\/(api\/)?/, '').replace(/[{}]/g, '').replace(/[^\w]+/g, '_').replace(/_+$/, '')}`,
    method, path, group, summary, params,
    query: extra.query || [],
    body: extra.body,
    desc: extra.desc || '',
    download: !!extra.download,
    danger: !!extra.danger,
  };
}

export const ENDPOINTS = [
  // ------------------------------------------------------------ Queries
  op('GET', '/api/queries', 'Queries', 'List queries (paginated)', { query: [...PAGED, ...TAGS], desc: 'Non-draft queries the user can see. Each item includes latest_query_data_id when cached results exist.' }),
  op('POST', '/api/queries', 'Queries', 'Create a query', { body: QUERY_BODY, desc: `Required: query, data_source_id. Optional: name, description, options.parameters, tags, is_draft, ${SCHEDULE_DOC}.` }),
  op('GET', '/api/queries/{query_id}', 'Queries', 'Get a query (with visualizations)'),
  op('POST', '/api/queries/{query_id}', 'Queries', 'Update a query', { body: { name: 'Renamed', version: 1 }, desc: 'Any query field: name, query, description, data_source_id, options, schedule, tags, is_draft, is_archived (false = unarchive). Send "version" to get 409 on concurrent edits.' }),
  op('DELETE', '/api/queries/{query_id}', 'Queries', 'Archive a query', { danger: true }),
  op('GET', '/api/queries/my', 'Queries', 'Queries created by me', { query: PAGED }),
  op('GET', '/api/queries/recent', 'Queries', 'Recently modified queries (up to 10)'),
  op('GET', '/api/queries/archive', 'Queries', 'Archived queries', { query: PAGED }),
  op('GET', '/api/queries/search', 'Queries', 'Search query text, names and descriptions', { query: [{ name: 'q', type: 'string', desc: 'Search term (required).', example: 'select' }, { name: 'include_drafts', type: 'boolean', desc: 'Include drafts when present.' }] }),
  op('GET', '/api/queries/tags', 'Queries', 'All query tags with counts'),
  op('POST', '/api/queries/format', 'Queries', 'Format SQL text', { body: { query: 'select a,b from t where x=1' }, desc: 'Returns {query: formatted}. Uses sqlparse on the server.' }),
  op('POST', '/api/queries/{query_id}/fork', 'Queries', 'Fork (copy) a query', { body: {} }),
  op('POST', '/api/queries/{query_id}/refresh', 'Queries', 'Re-execute a saved query (returns a job)', { body: {}, desc: 'Accepts parameter values as p_<name> query-string arguments.' }),
  op('POST', '/api/queries/{query_id}/regenerate_api_key', 'Queries', 'Regenerate the query API key', { body: {}, danger: true, desc: 'Breaks existing shared result URLs for this query.' }),
  op('GET', '/api/queries/{query_id}/dropdown', 'Queries', 'Options for a query-based dropdown parameter', { desc: 'Returns [{name, value}] from the query\'s latest results.' }),
  op('GET', '/api/queries/{query_id}/dropdowns/{dropdown_query_id}', 'Queries', 'Dropdown options of a parameter of another query'),
  op('GET', '/api/queries/favorites', 'Favorites', 'Favourite queries', { query: [...PAGED, ...TAGS] }),
  op('POST', '/api/queries/{query_id}/favorite', 'Favorites', 'Add a query to favourites', { body: {} }),
  op('DELETE', '/api/queries/{query_id}/favorite', 'Favorites', 'Remove a query from favourites'),

  // ------------------------------------------------------------ Query results & jobs
  op('POST', '/api/query_results', 'Query results', 'Execute ad-hoc SQL (or return a cached result)', { body: { query: 'SELECT 1 AS one', data_source_id: 1, max_age: 0, parameters: {}, apply_auto_limit: false }, desc: 'Returns {query_result} when cached, else {job}. Poll /api/jobs/{id}.' }),
  op('GET', '/api/query_results/{query_result_id}', 'Query results', 'Get a query result', { query: MAX_AGE }),
  op('POST', '/api/query_results/{query_result_id}', 'Query results', 'Execute the query behind a result', { body: EXEC_BODY }),
  op('GET', '/api/query_results/{query_result_id}.{filetype}', 'Query results', 'Download a result (csv, tsv, json, xlsx)', { query: MAX_AGE, download: true }),
  op('POST', '/api/query_results/{query_result_id}.{filetype}', 'Query results', 'Execute and download a result', { body: EXEC_BODY, download: true }),
  op('GET', '/api/queries/{query_id}/results', 'Query results', 'Latest cached result of a query', { query: MAX_AGE, desc: 'Not available for parameterized queries; POST instead.' }),
  op('POST', '/api/queries/{query_id}/results', 'Query results', 'Run a saved query (with parameters / max_age)', { body: EXEC_BODY, desc: 'max_age 0 always executes. Returns {query_result} or {job}.' }),
  op('GET', '/api/queries/{query_id}/results.{filetype}', 'Query results', 'Download the latest result of a query', { query: MAX_AGE, download: true }),
  op('POST', '/api/queries/{query_id}/results.{filetype}', 'Query results', 'Run a saved query and download', { body: EXEC_BODY, download: true }),
  op('GET', '/api/queries/{query_id}/results/{query_result_id}.{filetype}', 'Query results', 'Download a specific result of a query', { query: MAX_AGE, download: true }),
  op('POST', '/api/queries/{query_id}/results/{query_result_id}.{filetype}', 'Query results', 'Run and download a specific result of a query', { body: EXEC_BODY, download: true }),
  op('GET', '/api/jobs/{job_id}', 'Jobs', 'Job status', { desc: 'status: 1 pending, 2 started, 3 success, 4 failure, 5 cancelled. On success includes query_result_id.' }),
  op('DELETE', '/api/jobs/{job_id}', 'Jobs', 'Cancel a running job', { danger: true }),
  op('GET', '/api/queries/{query_id}/jobs/{job_id}', 'Jobs', 'Job status (scoped to a query)'),
  op('DELETE', '/api/queries/{query_id}/jobs/{job_id}', 'Jobs', 'Cancel a job (scoped to a query)', { danger: true }),

  // ------------------------------------------------------------ Visualizations & widgets
  op('POST', '/api/visualizations', 'Visualizations', 'Create a visualization', { body: { query_id: 1, type: 'CHART', name: 'Chart', description: '', options: { globalSeriesType: 'column', columnMapping: { x_col: 'x', y_col: 'y' } } } }),
  op('POST', '/api/visualizations/{visualization_id}', 'Visualizations', 'Update a visualization', { body: { name: 'Renamed', options: {} } }),
  op('DELETE', '/api/visualizations/{visualization_id}', 'Visualizations', 'Delete a visualization', { danger: true }),
  op('POST', '/api/widgets', 'Widgets', 'Add a widget to a dashboard', { body: { dashboard_id: 1, visualization_id: 1, text: '', width: 1, options: { position: { col: 0, row: 0, sizeX: 3, sizeY: 8 } } }, desc: 'For a text widget send visualization_id null and markdown in text.' }),
  op('POST', '/api/widgets/{widget_id}', 'Widgets', 'Update a widget (text / options / position)', { body: { options: { position: { col: 0, row: 0, sizeX: 3, sizeY: 8 } }, text: '' } }),
  op('DELETE', '/api/widgets/{widget_id}', 'Widgets', 'Remove a widget', { danger: true }),

  // ------------------------------------------------------------ Dashboards
  op('GET', '/api/dashboards', 'Dashboards', 'List dashboards (paginated)', { query: [...PAGED, ...TAGS] }),
  op('POST', '/api/dashboards', 'Dashboards', 'Create a dashboard (draft)', { body: { name: 'zz_studio_test dashboard' } }),
  op('GET', '/api/dashboards/{dashboard_id}', 'Dashboards', 'Get a dashboard with widgets', { query: [{ name: 'legacy', type: 'boolean', desc: 'Look up by slug instead of id.' }] }),
  op('POST', '/api/dashboards/{dashboard_id}', 'Dashboards', 'Update a dashboard', { body: { name: 'Renamed', tags: [], is_draft: false, dashboard_filters_enabled: false, version: 1 }, desc: 'Fields: name, layout, options, tags, is_draft, is_archived, dashboard_filters_enabled, version (409 on conflict).' }),
  op('DELETE', '/api/dashboards/{dashboard_id}', 'Dashboards', 'Archive a dashboard', { danger: true }),
  op('GET', '/api/dashboards/my', 'Dashboards', 'Dashboards created by me', { query: PAGED }),
  op('GET', '/api/dashboards/tags', 'Dashboards', 'All dashboard tags with counts'),
  op('GET', '/api/dashboards/public/{token}', 'Dashboards', 'Get a public (shared) dashboard by token'),
  op('POST', '/api/dashboards/{dashboard_id}/share', 'Dashboards', 'Enable public sharing (returns public_url)', { body: {} }),
  op('DELETE', '/api/dashboards/{dashboard_id}/share', 'Dashboards', 'Disable public sharing', { danger: true }),
  op('GET', '/api/dashboards/favorites', 'Favorites', 'Favourite dashboards', { query: [...PAGED, ...TAGS] }),
  op('POST', '/api/dashboards/{object_id}/favorite', 'Favorites', 'Add a dashboard to favourites', { body: {} }),
  op('DELETE', '/api/dashboards/{object_id}/favorite', 'Favorites', 'Remove a dashboard from favourites'),

  // ------------------------------------------------------------ Permissions
  op('GET', '/api/{object_type}/{object_id}/acl', 'Permissions', 'List access permissions (queries | dashboards)'),
  op('POST', '/api/{object_type}/{object_id}/acl', 'Permissions', 'Grant access to a user', { body: { access_type: 'modify', user_id: 1 } }),
  op('DELETE', '/api/{object_type}/{object_id}/acl', 'Permissions', 'Revoke access from a user', { body: { access_type: 'modify', user_id: 1 }, danger: true }),
  op('GET', '/api/{object_type}/{object_id}/acl/{access_type}', 'Permissions', 'Check whether I have an access type'),

  // ------------------------------------------------------------ Alerts
  op('GET', '/api/alerts', 'Alerts', 'List alerts'),
  op('POST', '/api/alerts', 'Alerts', 'Create an alert', { body: { name: 'zz_studio_test alert', query_id: 1, rearm: null, options: { column: 'one', op: '>', value: 1, custom_subject: '', custom_body: '' } }, desc: 'options.op: > >= < <= == != ; rearm = seconds before re-notifying (null = once).' }),
  op('GET', '/api/alerts/{alert_id}', 'Alerts', 'Get an alert'),
  op('POST', '/api/alerts/{alert_id}', 'Alerts', 'Update an alert', { body: { name: 'Renamed', options: {}, rearm: null } }),
  op('DELETE', '/api/alerts/{alert_id}', 'Alerts', 'Delete an alert', { danger: true }),
  op('POST', '/api/alerts/{alert_id}/mute', 'Alerts', 'Mute an alert', { body: {} }),
  op('DELETE', '/api/alerts/{alert_id}/mute', 'Alerts', 'Unmute an alert'),
  op('GET', '/api/alerts/{alert_id}/subscriptions', 'Alerts', 'List alert subscriptions'),
  op('POST', '/api/alerts/{alert_id}/subscriptions', 'Alerts', 'Subscribe me or a destination', { body: { destination_id: null }, desc: 'Omit destination_id to subscribe the current user by email.' }),
  op('DELETE', '/api/alerts/{alert_id}/subscriptions/{subscriber_id}', 'Alerts', 'Remove a subscription', { danger: true }),

  // ------------------------------------------------------------ Destinations
  op('GET', '/api/destinations', 'Destinations', 'List alert destinations'),
  op('POST', '/api/destinations', 'Destinations', 'Create a destination', { body: { name: 'zz_studio_test webhook', type: 'webhook', options: { url: 'https://example.com/hook' } } }),
  op('GET', '/api/destinations/types', 'Destinations', 'Destination types with configuration schemas'),
  op('GET', '/api/destinations/{destination_id}', 'Destinations', 'Get a destination'),
  op('POST', '/api/destinations/{destination_id}', 'Destinations', 'Update a destination', { body: { name: 'Renamed', type: 'webhook', options: {} } }),
  op('DELETE', '/api/destinations/{destination_id}', 'Destinations', 'Delete a destination', { danger: true }),

  // ------------------------------------------------------------ Data sources
  op('GET', '/api/data_sources', 'Data sources', 'List data sources'),
  op('POST', '/api/data_sources', 'Data sources', 'Create a data source', { body: { name: 'zz_studio_test results', type: 'results', options: {} }, desc: 'options must match the type\'s configuration_schema from /api/data_sources/types.' }),
  op('GET', '/api/data_sources/types', 'Data sources', 'Data source types with configuration schemas'),
  op('GET', '/api/data_sources/{data_source_id}', 'Data sources', 'Get a data source (secrets masked)'),
  op('POST', '/api/data_sources/{data_source_id}', 'Data sources', 'Update a data source', { body: { name: 'Renamed', type: 'results', options: {} } }),
  op('DELETE', '/api/data_sources/{data_source_id}', 'Data sources', 'Delete a data source', { danger: true }),
  op('GET', '/api/data_sources/{data_source_id}/schema', 'Data sources', 'Schema (tables & columns) — returns a job in 10.x', { query: [{ name: 'refresh', type: 'boolean', desc: 'Bypass the schema cache.' }] }),
  op('POST', '/api/data_sources/{data_source_id}/test', 'Data sources', 'Test the connection', { body: {} }),
  op('POST', '/api/data_sources/{data_source_id}/pause', 'Data sources', 'Pause a data source', { body: { reason: 'Maintenance' } }),
  op('DELETE', '/api/data_sources/{data_source_id}/pause', 'Data sources', 'Resume a paused data source'),
  op('GET', '/api/databricks/databases/{data_source_id}', 'Data sources', 'Databricks: list databases', { query: [{ name: 'refresh', type: 'boolean' }] }),
  op('GET', '/api/databricks/databases/{data_source_id}/{database_name}/tables', 'Data sources', 'Databricks: tables of a database', { query: [{ name: 'refresh', type: 'boolean' }] }),
  op('GET', '/api/databricks/databases/{data_source_id}/{database_name}/columns/{table_name}', 'Data sources', 'Databricks: columns of a table'),

  // ------------------------------------------------------------ Snippets
  op('GET', '/api/query_snippets', 'Snippets', 'List query snippets'),
  op('POST', '/api/query_snippets', 'Snippets', 'Create a snippet', { body: { trigger: 'zz_test', description: 'Example', snippet: 'SELECT 1' } }),
  op('GET', '/api/query_snippets/{snippet_id}', 'Snippets', 'Get a snippet'),
  op('POST', '/api/query_snippets/{snippet_id}', 'Snippets', 'Update a snippet', { body: { trigger: 'zz_test', description: '', snippet: 'SELECT 2' } }),
  op('DELETE', '/api/query_snippets/{snippet_id}', 'Snippets', 'Delete a snippet', { danger: true }),

  // ------------------------------------------------------------ Users
  op('GET', '/api/users', 'Users', 'List users (paginated)', { query: [...PAGED, { name: 'disabled', type: 'boolean', desc: 'Only disabled users.' }, { name: 'pending', type: 'boolean', desc: 'Only pending invitations.' }] }),
  op('POST', '/api/users', 'Users', 'Create / invite a user', { body: { name: 'Test User', email: 'test@example.com' }, query: [{ name: 'no_invite', type: 'boolean', desc: 'Do not send the invitation email.' }] }),
  op('GET', '/api/users/{user_id}', 'Users', 'Get a user'),
  op('POST', '/api/users/{user_id}', 'Users', 'Update a user (name, email, group_ids, password)', { body: { name: 'New name' } }),
  op('DELETE', '/api/users/{user_id}', 'Users', 'Delete a user (pending invitations only)', { danger: true }),
  op('POST', '/api/users/{user_id}/invite', 'Users', 'Re-send the invitation', { body: {} }),
  op('POST', '/api/users/{user_id}/reset_password', 'Users', 'Send / generate a password reset link', { body: {} }),
  op('POST', '/api/users/{user_id}/regenerate_api_key', 'Users', 'Regenerate a user API key', { body: {}, danger: true, desc: 'Regenerating the dashboard admin key breaks this Studio until .env is updated.' }),
  op('POST', '/api/users/{user_id}/disable', 'Users', 'Disable a user', { body: {}, danger: true }),
  op('DELETE', '/api/users/{user_id}/disable', 'Users', 'Re-enable a user'),

  // ------------------------------------------------------------ Groups
  op('GET', '/api/groups', 'Groups', 'List groups'),
  op('POST', '/api/groups', 'Groups', 'Create a group', { body: { name: 'zz_studio_test group' } }),
  op('GET', '/api/groups/{group_id}', 'Groups', 'Get a group'),
  op('POST', '/api/groups/{group_id}', 'Groups', 'Rename a group', { body: { name: 'Renamed' } }),
  op('DELETE', '/api/groups/{group_id}', 'Groups', 'Delete a group', { danger: true }),
  op('GET', '/api/groups/{group_id}/members', 'Groups', 'Group members'),
  op('POST', '/api/groups/{group_id}/members', 'Groups', 'Add a member', { body: { user_id: 1 } }),
  op('DELETE', '/api/groups/{group_id}/members/{user_id}', 'Groups', 'Remove a member', { danger: true }),
  op('GET', '/api/groups/{group_id}/data_sources', 'Groups', 'Data sources a group can access'),
  op('POST', '/api/groups/{group_id}/data_sources', 'Groups', 'Give a group access to a data source', { body: { data_source_id: 1 } }),
  op('POST', '/api/groups/{group_id}/data_sources/{data_source_id}', 'Groups', 'Set view-only access', { body: { view_only: true } }),
  op('DELETE', '/api/groups/{group_id}/data_sources/{data_source_id}', 'Groups', 'Revoke data source access', { danger: true }),

  // ------------------------------------------------------------ Events & admin
  op('GET', '/api/events', 'Events', 'Audit events (paginated)', { query: PAGED.slice(0, 2) }),
  op('POST', '/api/events', 'Events', 'Record client events', { body: [{ action: 'view', object_type: 'page', object_id: 'studio' }] }),
  op('GET', '/api/admin/queries/outdated', 'Admin', 'Scheduled queries waiting to refresh'),
  op('GET', '/api/admin/queries/rq_status', 'Admin', 'RQ queues and workers'),
  op('GET', '/api/settings/organization', 'Admin', 'Organization settings'),
  op('POST', '/api/settings/organization', 'Admin', 'Update organization settings', { body: { date_format: 'DD/MM/YY' }, danger: true }),
  op('GET', '/api/organization/status', 'Admin', 'Object counters (users, queries, dashboards…)'),
  op('GET', '/status.json', 'Admin', 'Server status: version, sizes, queues, workers'),
  op('GET', '/api/session', 'Session', 'Current user, permissions and client config'),
  op('GET', '/api/config', 'Session', 'Client configuration and org slug'),
];

export const GROUPS = [...new Set(ENDPOINTS.map((e) => e.group))];

/** Fill {param} placeholders; values are URL-encoded except the file type. */
export function buildPath(endpoint, values) {
  return endpoint.path.replace(/\{(\w+)\}/g, (_, k) => (k === 'filetype' ? (values[k] || 'csv') : encodeURIComponent(values[k] ?? '')));
}
