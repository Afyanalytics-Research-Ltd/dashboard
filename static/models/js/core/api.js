// Client for Model Studio's Django endpoints (GraphQL proxy, file proxy, saved views).
//
//   const data = await gql(`query P($e: String!, $p: String!) { project(entityName: $e, name: $p) { id } }`, { e, p })
//   const runs = await collect(cursor => gql(Q, { ...vars, after: cursor }), d => d.project.runs, { max: 500 })
//
// W&B returns many fields as JSON strings (config, summaryMetrics, metadata...); use parseJSON().

const CONFIG = window.APP_UI || {};
export const config = CONFIG;

export class WandbError extends Error {
  constructor(status, errors, extra = {}) {
    const msg = (errors || []).map((e) => e.message).filter(Boolean).join('; ') || (status === 0 ? 'Network error — is the dashboard server reachable?' : `Request failed (${status})`);
    super(msg);
    this.status = status;
    this.errors = errors || [];
    Object.assign(this, extra);
  }
}

function csrf() {
  const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

const listeners = new Set();
export const onRequest = (fn) => (listeners.add(fn), () => listeners.delete(fn));

async function post(url, body, signal) {
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json', 'X-CSRFToken': csrf() },
      body: JSON.stringify(body),
      credentials: 'same-origin',
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new WandbError(0, null);
  }
  let payload = null;
  try { payload = await res.json(); } catch { /* non-JSON */ }
  const message = payload?.errors?.[0]?.message || '';
  // W&B itself also answers 401 ("relogin required"); only Django's own message means our session ended.
  const sessionExpired = res.status === 401 && /session has expired/i.test(message);
  listeners.forEach((fn) => fn({ status: res.status, ms: performance.now() - t0, message, sessionExpired }));
  return { res, payload };
}

/** Run one GraphQL document; resolves to `data`, throws WandbError on transport or GraphQL errors. */
export async function gql(query, variables = {}, { signal, allowPartial = false } = {}) {
  const { res, payload } = await post(CONFIG.graphql || '/models/graphql', { query, variables }, signal);
  const errors = payload?.errors;
  if (!res.ok || (errors?.length && !(allowPartial && payload?.data))) {
    throw new WandbError(res.status, errors || [{ message: `HTTP ${res.status}` }], { data: payload?.data });
  }
  return payload?.data;
}

/**
 * Follow a Relay-style connection: fetchPage(cursor) -> data; pick(data) -> {edges, pageInfo}.
 * Returns the nodes (up to `max`). onPage(nodes, connection) is called after each page.
 */
export async function collect(fetchPage, pick, { max = 1000, onPage } = {}) {
  const nodes = [];
  let cursor = null;
  for (;;) {
    const conn = pick(await fetchPage(cursor));
    if (!conn) break;
    const batch = (conn.edges || []).map((e) => e.node).filter(Boolean);
    nodes.push(...batch);
    onPage?.(nodes, conn);
    if (!conn.pageInfo?.hasNextPage || !batch.length || nodes.length >= max) break;
    cursor = conn.pageInfo.endCursor;
  }
  return nodes.slice(0, max);
}

/** Same-origin URL for a W&B file (image, log, table, artifact file) via the server-side key. */
export function fileUrl(url, { download = false } = {}) {
  if (!url) return '';
  const base = CONFIG.file || '/models/file';
  return `${base}?url=${encodeURIComponent(url)}${download ? '&download=1' : ''}`;
}

/** Fetch a W&B file's text/JSON through the proxy. */
export async function fetchFile(url, { as = 'text', signal } = {}) {
  const res = await fetch(fileUrl(url), { credentials: 'same-origin', signal });
  if (!res.ok) {
    let msg = `File request failed (${res.status})`;
    try { msg = (await res.json()).errors?.[0]?.message || msg; } catch { /* binary */ }
    throw new WandbError(res.status, [{ message: msg }]);
  }
  return as === 'json' ? res.json() : as === 'blob' ? res.blob() : res.text();
}

export function parseJSON(value, fallback = {}) {
  if (value == null || value === '') return fallback;
  if (typeof value !== 'string') return value;
  try { return JSON.parse(value); } catch { return fallback; }
}

// ------------------------------------------------------------------ saved views (Django)

const viewsUrl = (entity, project) => (CONFIG.views || '/models/views/_E_/_P_/')
  .replace('_E_', encodeURIComponent(entity)).replace('_P_', encodeURIComponent(project));
const viewUrl = (id) => `${(CONFIG.views || '/models/views/_E_/_P_/').split('/views/')[0]}/views/${id}/`;

async function viewsRequest(method, url, body) {
  const res = await fetch(url, {
    method, credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf() },
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new WandbError(res.status, payload.errors);
  return payload;
}

export const savedViews = {
  list: (entity, project) => viewsRequest('GET', viewsUrl(entity, project)).then((r) => r.views),
  create: (entity, project, view) => viewsRequest('POST', viewsUrl(entity, project), view),
  update: (id, view) => viewsRequest('POST', viewUrl(id), view),
  remove: (id) => viewsRequest('DELETE', viewUrl(id)),
};

export function wandbLink(path = '') {
  return CONFIG.wandbUrl ? `${CONFIG.wandbUrl}${path}` : null;
}
