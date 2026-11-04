// Client for the Django proxy at /automations/api/ (forwards to n8n's public API with the
// server-side key) and the Studio endpoints at /automations/studio/.
//
//   await n8n.get('/workflows', { limit: 50, active: true })
//   for await (const wf of n8n.iterate('/workflows')) …      // follows nextCursor
//   await studio.post(`/run/${id}/`, { triggerName })
//
// n8n errors look like {"message": "..."}. Licence-gated features answer 403 with
// "Your license does not allow for feat:xyz" — surfaced as err.license = 'feat:xyz'.

const CONFIG = window.APP_UI || {};
const API = (CONFIG.api || '/automations/api/').replace(/\/$/, '');
const STUDIO = (CONFIG.studio || '/automations/studio/').replace(/\/$/, '');

export const config = CONFIG;
export const enc = (v) => encodeURIComponent(String(v));

export class N8nError extends Error {
  constructor(status, body, path) {
    super(errorMessage(body, status));
    this.status = status;
    this.body = body;
    this.path = path;
    const m = /does not allow for (feat:[\w.:-]+)/.exec(this.message);
    this.license = m ? m[1] : null;
    this.code = body && typeof body === 'object' ? body.code : null;
  }
}

function errorMessage(body, status) {
  if (body && typeof body === 'object') {
    const msg = body.message || body.error || body.detail;
    if (msg) return String(msg);
    return JSON.stringify(body).slice(0, 300);
  }
  if (typeof body === 'string' && body && !body.trim().startsWith('<')) return body.slice(0, 300);
  if (status === 0) return 'Network error — is the dashboard server reachable?';
  if (status === 404) return 'Not found.';
  if (status === 403) return 'n8n denied this action.';
  return `Request failed (${status})`;
}

/** Friendly name for a licence feature flag, e.g. feat:logStreaming -> "Log streaming". */
export function featureName(flag) {
  const raw = String(flag || '').replace(/^feat:/, '').split(':').pop();
  const map = {
    variables: 'Variables', folders: 'Folders', logStreaming: 'Log streaming', saml: 'SAML SSO', oidc: 'OIDC SSO',
    ldap: 'LDAP', sourceControl: 'Source control', personalSpacePolicy: 'Security policy', admin: 'Project roles',
    advancedPermissions: 'Advanced permissions', customRoles: 'Custom roles', workflowHistory: 'Workflow history',
  };
  return map[raw] || raw.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase());
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

// ------------------------------------------------------------------ request log (net indicator, API explorer)

const listeners = new Set();
export const onRequest = (fn) => (listeners.add(fn), () => listeners.delete(fn));
export const requestLog = [];

async function send(base, method, path, { params, body, signal, raw = false, headers: extra, form } = {}) {
  const url = `${base}${path}${qs(params)}`;
  const headers = { Accept: 'application/json', ...(extra || {}) };
  if (method !== 'GET') headers['X-CSRFToken'] = csrf();
  if (body !== undefined && !form) headers['Content-Type'] = 'application/json';
  const t0 = performance.now();
  listeners.forEach((fn) => fn({ phase: 'start', method, path }));
  let res;
  try {
    res = await fetch(url, {
      method, headers, signal, credentials: 'same-origin',
      body: form || (body !== undefined ? JSON.stringify(body) : undefined),
    });
  } catch (err) {
    listeners.forEach((fn) => fn({ phase: 'end', method, path, status: 0, ms: performance.now() - t0 }));
    if (err.name === 'AbortError') throw err;
    throw new N8nError(0, null, path);
  }
  const entry = { phase: 'end', method, path: base === API ? path : `studio:${path}`, status: res.status, ms: performance.now() - t0, at: Date.now() };
  requestLog.unshift(entry);
  if (requestLog.length > 200) requestLog.length = 200;
  listeners.forEach((fn) => fn(entry));
  if (res.status === 401) window.dispatchEvent(new CustomEvent('n8n:session-expired'));
  if (raw) {
    if (!res.ok) {
      const text = await res.text();
      let data = text;
      try { data = JSON.parse(text); } catch { /* keep text */ }
      throw new N8nError(res.status, data, path);
    }
    return res;
  }
  const text = await res.text();
  let data = text || null;
  if (text && (res.headers.get('Content-Type') || '').includes('json')) {
    try { data = JSON.parse(text); } catch { /* keep text */ }
  }
  if (!res.ok) throw new N8nError(res.status, data, path);
  return data;
}

function makeClient(base) {
  const c = {
    request: (method, path, opts) => send(base, method, path, opts),
    get: (path, params, opts = {}) => send(base, 'GET', path, { ...opts, params }),
    post: (path, body = {}, opts = {}) => send(base, 'POST', path, { ...opts, body }),
    put: (path, body = {}, opts = {}) => send(base, 'PUT', path, { ...opts, body }),
    patch: (path, body = {}, opts = {}) => send(base, 'PATCH', path, { ...opts, body }),
    del: (path, opts = {}) => send(base, 'DELETE', path, opts),
  };
  return c;
}

export const n8n = makeClient(API);
export const studio = makeClient(STUDIO);

/** One page of a cursor list: {data, nextCursor}. */
n8n.page = (path, params = {}, opts = {}) => n8n.get(path, { limit: 100, ...params }, opts);

/** Async iterator over every item of a cursor-paginated list. */
n8n.iterate = async function* iterate(path, params = {}, { signal, limit = 250, max = 20000 } = {}) {
  let cursor = null;
  let seen = 0;
  do {
    // eslint-disable-next-line no-await-in-loop
    const page = await n8n.get(path, { ...params, limit, cursor }, { signal });
    const items = Array.isArray(page) ? page : page?.data || [];
    for (const item of items) {
      yield item;
      if (++seen >= max) return;
    }
    cursor = Array.isArray(page) ? null : page?.nextCursor;
  } while (cursor);
};

/** Collect every item of a cursor list (with an onProgress(count) hook). */
n8n.all = async (path, params = {}, { signal, max = 20000, onProgress, limit } = {}) => {
  const out = [];
  for await (const item of n8n.iterate(path, params, { signal, max, limit })) {
    out.push(item);
    if (onProgress && out.length % 250 === 0) onProgress(out.length);
  }
  return out;
};

/** Run fn over items with bounded concurrency; returns [{item, ok, value|error}]. */
export async function pool(items, fn, { concurrency = 4, onProgress } = {}) {
  const results = new Array(items.length);
  let next = 0;
  let done = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      try {
        results[i] = { item: items[i], ok: true, value: await fn(items[i], i) };
      } catch (error) {
        results[i] = { item: items[i], ok: false, error };
      }
      done++;
      onProgress?.(done, items.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return results;
}

/** Download a binary response (e.g. package export) as a file. */
export async function download(method, path, body, fallbackName = 'download') {
  const res = await send(API, method, path, { body, raw: true, headers: { Accept: '*/*' } });
  const blob = await res.blob();
  const cd = res.headers.get('Content-Disposition') || '';
  const name = (/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd) || [])[1] || fallbackName;
  saveBlob(blob, decodeURIComponent(name));
  return { size: blob.size, name };
}

export function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function saveText(text, name, type = 'application/json') {
  saveBlob(new Blob([text], { type }), name);
}

/** Upload multipart (package import). */
export const upload = (path, formData) => send(API, 'POST', path, { form: formData, body: null });

export const iconUrl = (path) => (path ? `${STUDIO}/icons/${String(path).replace(/^\/+/, '')}` : null);
export const n8nLink = (path = '') => `${(CONFIG.n8nUrl || '').replace(/\/$/, '')}${path}`;
