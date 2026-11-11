// Client for the Django proxy at /studio/proxy/ (which forwards to Redash with the admin key).
//
//   await rd.get('/api/queries', { page: 1, page_size: 25, q: 'admissions' })
//   const result = await runQuery({ query: 'select 1', data_source_id: 1 })  // polls the job
//
// Redash errors look like {"message": "..."}; job statuses are 1 pending, 2 started,
// 3 success, 4 failure, 5 cancelled.

const CONFIG = window.APP_UI || {};
const BASE = (CONFIG.api || '/studio/proxy/').replace(/\/$/, '');

export const config = CONFIG;
export const enc = (v) => encodeURIComponent(String(v));

export class RedashError extends Error {
  constructor(status, body, path) {
    super(errorMessage(body, status));
    this.status = status;
    this.body = body;
    this.path = path;
  }
}

function errorMessage(body, status) {
  if (body && typeof body === 'object') return body.message || body.error || body.detail || JSON.stringify(body);
  if (typeof body === 'string' && body && !body.trim().startsWith('<')) return body.slice(0, 300);
  if (status === 0) return 'Network error — is the dashboard server reachable?';
  if (status === 404) return 'Not found.';
  if (status === 403) return 'Redash denied this action.';
  return `Request failed (${status})`;
}

export function qs(params) {
  if (!params) return '';
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === '') continue;
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, x));
    else sp.append(k, typeof v === 'boolean' ? String(v) : v);
  }
  const s = sp.toString();
  return s ? `?${s}` : '';
}

function csrf() {
  const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

const listeners = new Set();
export const onRequest = (fn) => (listeners.add(fn), () => listeners.delete(fn));

export async function request(method, path, { params, body, signal, raw = false } = {}) {
  const url = `${BASE}${path}${qs(params)}`;
  const headers = { Accept: 'application/json' };
  if (method !== 'GET') headers['X-CSRFToken'] = csrf();
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(url, { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined, signal, credentials: 'same-origin' });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new RedashError(0, null, path);
  }
  listeners.forEach((fn) => fn({ method, path, status: res.status, ms: performance.now() - t0 }));
  if (raw) return res;
  const text = await res.text();
  let data = text || null;
  if (text && (res.headers.get('Content-Type') || '').includes('json')) {
    try { data = JSON.parse(text); } catch { /* keep text */ }
  }
  if (!res.ok) throw new RedashError(res.status, data, path);
  return data;
}

export const rd = {
  get: (path, params, opts = {}) => request('GET', path, { ...opts, params }),
  post: (path, body = {}, opts = {}) => request('POST', path, { ...opts, body }),
  del: (path, opts = {}) => request('DELETE', path, opts),

  /** Fetch a Redash paginated list ({count, page, page_size, results}). */
  page: (path, { page = 1, pageSize = 25, ...params } = {}, opts = {}) =>
    request('GET', path, { ...opts, params: { page, page_size: pageSize, ...params } }),

  /** Every item of a paginated list (page_size max 250 in Redash). */
  async all(path, params = {}, { max = 5000, signal } = {}) {
    const out = [];
    for (let page = 1; out.length < max; page++) {
      const res = await request('GET', path, { params: { ...params, page, page_size: 250 }, signal });
      out.push(...(res.results || []));
      if (!res.results?.length || out.length >= res.count) break;
    }
    return out;
  },
};

/** Proxy URL for a file download, e.g. resultUrl(15, 'csv'). */
export const fileUrl = (path) => `${BASE}${path}`;

// ------------------------------------------------------------------ jobs

export const JOB_STATUS = { 1: 'pending', 2: 'running', 3: 'success', 4: 'failed', 5: 'cancelled' };

const sleep = (ms, signal) => new Promise((resolve, reject) => {
  const t = setTimeout(resolve, ms);
  signal?.addEventListener('abort', () => { clearTimeout(t); reject(new DOMException('Aborted', 'AbortError')); }, { once: true });
});

/**
 * Poll /api/jobs/{id} until it finishes. onProgress({status, label, elapsed, job}).
 * Resolves with the finished job; throws RedashError on failure/cancel.
 */
export async function waitForJob(job, { signal, onProgress, interval = 600 } = {}) {
  const started = Date.now();
  let current = job;
  let delay = interval;
  while (current.status < 3) {
    onProgress?.({ status: current.status, label: JOB_STATUS[current.status], elapsed: (Date.now() - started) / 1000, job: current });
    await sleep(delay, signal);
    delay = Math.min(delay * 1.25, 3000);
    current = (await rd.get(`/api/jobs/${enc(current.id)}`, null, { signal })).job;
  }
  onProgress?.({ status: current.status, label: JOB_STATUS[current.status], elapsed: (Date.now() - started) / 1000, job: current });
  if (current.status === 4) {
    // Redash reports job errors either as a string or as {code, message}.
    const err = current.error;
    const message = (err && typeof err === 'object' ? err.message || JSON.stringify(err) : err) || 'Query failed.';
    throw new RedashError(400, { message }, '/api/jobs');
  }
  if (current.status === 5) throw new RedashError(499, { message: 'Query execution was cancelled.' }, '/api/jobs');
  return current;
}

export const cancelJob = (jobId) => rd.del(`/api/jobs/${enc(jobId)}`);

/**
 * Execute SQL or a saved query and return the query_result object.
 *  - saved query:  runQuery({ queryId, parameters, maxAge })
 *  - ad-hoc SQL:   runQuery({ query, dataSourceId, parameters, maxAge })
 * maxAge 0 always re-executes; -1 / undefined lets Redash return a cached result.
 * onJob(job) fires as soon as a job id is known (so callers can offer Cancel).
 */
export async function runQuery({ queryId, query, dataSourceId, parameters = {}, maxAge = 0, signal, onProgress, onJob, applyAutoLimit = false }) {
  const res = queryId != null
    ? await rd.post(`/api/queries/${enc(queryId)}/results`, { parameters, max_age: maxAge, apply_auto_limit: applyAutoLimit }, { signal })
    : await rd.post('/api/query_results', { query, data_source_id: dataSourceId, parameters, max_age: maxAge, apply_auto_limit: applyAutoLimit }, { signal });
  if (res.query_result) return res.query_result;
  onJob?.(res.job);
  const job = await waitForJob(res.job, { signal, onProgress });
  const result = await rd.get(`/api/query_results/${enc(job.query_result_id)}`, null, { signal });
  return result.query_result;
}

/** Data source schema (Redash 10 returns a job for it). refresh=true bypasses the cache. */
export async function getSchema(dataSourceId, { refresh = false, signal } = {}) {
  const res = await rd.get(`/api/data_sources/${enc(dataSourceId)}/schema`, refresh ? { refresh: true } : null, { signal });
  if (res.schema) return res.schema;
  if (res.job) {
    const job = await waitForJob(res.job, { signal, interval: 800 });
    return job.result || [];
  }
  return [];
}

export function redashLink(path = '') {
  return CONFIG.redashUrl ? `${CONFIG.redashUrl}${path}` : null;
}
