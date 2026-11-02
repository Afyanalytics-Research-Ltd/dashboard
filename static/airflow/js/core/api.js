// Client for the Django proxy at /pipelines/api/v2 (which forwards to Airflow's /api/v2).
//
//   await api.get('/dags', { limit: 50, tags: ['etl', 'v3'] })
//   await api.post(`/dags/${enc(dagId)}/dagRuns`, { logical_date: null, conf: {} })
//
// Every path segment that holds an id MUST go through enc(): run ids contain ':' and '+'.

const CONFIG = window.AIRFLOW_UI || {};
const API_BASE = (CONFIG.api || '/pipelines/api/v2/').replace(/\/$/, '');

export const enc = (value) => encodeURIComponent(String(value));

export class ApiError extends Error {
  constructor(status, body, path) {
    super(formatDetail(body, status));
    this.status = status;
    this.body = body;
    this.path = path;
  }
}

/** Airflow errors are {detail: string | [{loc, msg}] | {...}}. */
export function formatDetail(body, status) {
  const detail = body && typeof body === 'object' ? body.detail : body;
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const loc = Array.isArray(d.loc) ? d.loc.filter((p) => p !== 'body').join('.') : '';
      return loc ? `${loc}: ${d.msg}` : d.msg || JSON.stringify(d);
    }).join('; ');
  }
  if (detail && typeof detail === 'object') return detail.reason || detail.message || JSON.stringify(detail);
  if (typeof detail === 'string' && detail) return detail;
  if (status === 0) return 'Network error — is the dashboard server reachable?';
  return `Request failed (${status})`;
}

export function qs(params) {
  if (!params) return '';
  const sp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value == null || value === '') continue;
    if (Array.isArray(value)) value.forEach((v) => v != null && v !== '' && sp.append(key, v));
    else sp.append(key, typeof value === 'boolean' ? String(value) : value);
  }
  const str = sp.toString();
  return str ? `?${str}` : '';
}

function csrfToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  if (match) return decodeURIComponent(match[1]);
  return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

const listeners = new Set();
/** Subscribe to every completed request: fn({method, path, status, ms}). */
export const onRequest = (fn) => (listeners.add(fn), () => listeners.delete(fn));

export async function request(method, path, { params, body, signal, accept = 'application/json', raw = false } = {}) {
  const url = `${API_BASE}${path}${qs(params)}`;
  const headers = { Accept: accept };
  if (method !== 'GET') headers['X-CSRFToken'] = csrfToken();
  if (body !== undefined) headers['Content-Type'] = 'application/json';

  const started = performance.now();
  let res;
  try {
    res = await fetch(url, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
      credentials: 'same-origin',
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError(0, null, path);
  }
  listeners.forEach((fn) => fn({ method, path, status: res.status, ms: performance.now() - started }));

  if (raw) return res;
  const type = res.headers.get('Content-Type') || '';
  const text = await res.text();
  let data = text;
  if (type.includes('json') && text) {
    try { data = JSON.parse(text); } catch { /* keep text */ }
  } else if (!text) {
    data = null;
  }
  if (!res.ok) {
    if (res.status === 401 && data && /session/i.test(data.detail || '')) {
      window.dispatchEvent(new CustomEvent('af:session-expired'));
    }
    throw new ApiError(res.status, data, path);
  }
  return data;
}

export const api = {
  get: (path, params, opts = {}) => request('GET', path, { ...opts, params }),
  post: (path, body = {}, opts = {}) => request('POST', path, { ...opts, body }),
  put: (path, body, opts = {}) => request('PUT', path, { ...opts, body }),
  patch: (path, body, opts = {}) => request('PATCH', path, { ...opts, body }),
  del: (path, opts = {}) => request('DELETE', path, opts),
  text: (path, params, opts = {}) => request('GET', path, { ...opts, params, accept: 'text/plain' }),

  /** Fetch every page of a list endpoint (limit/offset), up to `max` items. */
  async all(path, key, params = {}, { max = 2000, signal } = {}) {
    const items = [];
    let offset = 0;
    let total = Infinity;
    while (offset < total && items.length < max) {
      const page = await request('GET', path, { params: { ...params, limit: 100, offset }, signal });
      const batch = page[key] || [];
      items.push(...batch);
      total = page.total_entries ?? batch.length;
      offset += batch.length;
      if (!batch.length) break;
    }
    return items;
  },
};

export const config = CONFIG;

/** Deep link to the same object in the native Airflow UI. */
export function airflowLink(path = '') {
  return CONFIG.airflowUrl ? `${CONFIG.airflowUrl}${path}` : null;
}
