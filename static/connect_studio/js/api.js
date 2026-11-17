// Connect Studio API client. Every call goes through request(): CSRF, JSON, typed errors.

const config = window.APP_UI || {};
export const API = (config.api || '/connect/api/').replace(/\/?$/, '/');

export class ApiError extends Error {
  constructor(message, { status = 0, code = 'error' } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

export function csrf() {
  const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  if (m) return decodeURIComponent(m[1]);
  return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

const listeners = new Set();
export const onRequest = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };

export async function request(path, { method = 'GET', query, body, signal } = {}) {
  const qs = query ? new URLSearchParams(Object.entries(query).filter(([, v]) => v != null && v !== '')).toString() : '';
  const url = `${API}${path.replace(/^\//, '')}${qs ? `?${qs}` : ''}`;
  listeners.forEach((fn) => fn('start'));
  try {
    const resp = await fetch(url, {
      method, signal, credentials: 'same-origin',
      headers: { Accept: 'application/json', ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}), ...(method !== 'GET' ? { 'X-CSRFToken': csrf() } : {}) },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await resp.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = null; }
    if (!resp.ok) {
      const err = data?.error || {};
      throw new ApiError(err.message || `HTTP ${resp.status}`, { status: resp.status, code: err.code || 'http' });
    }
    return data;
  } finally {
    listeners.forEach((fn) => fn('end'));
  }
}

/** One file + form fields as multipart (no 2.5 MB JSON-body limit), JSON response. */
export async function postFile(path, file, fields = {}) {
  const form = new FormData();
  form.append('file', file, file.name);
  Object.entries(fields).forEach(([k, v]) => { if (v != null) form.append(k, String(v)); });
  listeners.forEach((fn) => fn('start'));
  try {
    const resp = await fetch(`${API}${path.replace(/^\//, '')}`, {
      method: 'POST', credentials: 'same-origin', body: form,
      headers: { Accept: 'application/json', 'X-CSRFToken': csrf() },
    });
    const text = await resp.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = null; }
    if (!resp.ok) {
      const err = data?.error || {};
      const fallback = resp.status === 413 ? 'That file is too large for the server.' : `HTTP ${resp.status}`;
      throw new ApiError(err.message || fallback, { status: resp.status, code: err.code || 'http' });
    }
    return data;
  } finally {
    listeners.forEach((fn) => fn('end'));
  }
}

/** Multipart upload with progress (XHR — fetch has no upload progress). */
export function upload(path, files, { onProgress } = {}) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    [...files].forEach((f) => form.append('files', f, f.name));
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}${path.replace(/^\//, '')}`);
    xhr.setRequestHeader('X-CSRFToken', csrf());
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress?.(e.loaded / e.total); };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { data = null; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError(data?.error?.message || (xhr.status === 413 ? 'Upload too large — the server accepts up to 100 MB per upload.' : `Upload failed (HTTP ${xhr.status})`), { status: xhr.status }));
    };
    xhr.onerror = () => reject(new ApiError('Upload failed — network error'));
    xhr.send(form);
  });
}

async function sha256hex(buf) {
  if (!globalThis.crypto?.subtle) return null;          // not a secure context: the server skips the check
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', buf));
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
}

function putChunk(uploadId, index, buf, sha, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', `${API}uploads/${uploadId}/chunks/${index}/`);
    xhr.setRequestHeader('X-CSRFToken', csrf());
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    if (sha) xhr.setRequestHeader('X-Chunk-SHA256', sha);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress?.(e.loaded / e.total); };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { data = null; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError(data?.error?.message || `Chunk ${index} failed (HTTP ${xhr.status})`, { status: xhr.status, code: data?.error?.code || 'http' }));
    };
    xhr.onerror = () => reject(new ApiError('network error', { status: 0, code: 'network' }));
    xhr.send(buf);
  });
}

const RETRYABLE = (e) => !e.status || e.status >= 500 || e.status === 429 || e.status === 408 || e.code === 'chunk';

/** Any-size upload: 16 MB chunks (3 in flight, each retried with backoff and checked with SHA-256), straight to
 * S3 when the server has a bucket. Choosing the same file again resumes at the first missing chunk. */
export async function uploadChunked(sourceId, files, { onProgress, parallel = 3 } = {}) {
  const list = [...files];
  const total = list.reduce((s, f) => s + f.size, 0) || 1;
  let done = 0;
  let last = null;
  listeners.forEach((fn) => fn('start'));
  try {
    for (const file of list) {
      const { upload: u } = await request(`sources/${sourceId}/uploads/`, { method: 'POST', body: { name: file.name, size: file.size, last_modified: file.lastModified } });
      const have = new Set(u.received);
      const partSize = (i) => Math.min(u.chunk_size, file.size - i * u.chunk_size);
      done += [...have].reduce((s, i) => s + partSize(i), 0);
      const inflight = new Map();
      const report = () => onProgress?.(Math.min(1, (done + [...inflight.values()].reduce((s, v) => s + v, 0)) / total));
      report();
      const todo = Array.from({ length: u.total }, (_, i) => i).filter((i) => !have.has(i));
      const worker = async () => {
        while (todo.length) {
          const i = todo.shift();
          const buf = await file.slice(i * u.chunk_size, i * u.chunk_size + partSize(i)).arrayBuffer();
          const sha = await sha256hex(buf);
          for (let attempt = 1; ; attempt++) {
            try {
              await putChunk(u.id, i, buf, sha, (p) => { inflight.set(i, p * buf.byteLength); report(); });
              break;
            } catch (e) {
              inflight.delete(i);
              if (attempt >= 6 || !RETRYABLE(e)) {
                const pct = Math.round((done / total) * 100);
                throw new ApiError(`${file.name}: upload stopped at ${pct}% (${e.message}). Choose the same file again to resume where it stopped.`, { status: e.status, code: 'upload_paused' });
              }
              await new Promise((r) => setTimeout(r, Math.min(30000, 1000 * 2 ** (attempt - 1))));
            }
          }
          inflight.delete(i);
          done += buf.byteLength;
          report();
        }
      };
      await Promise.all(Array.from({ length: Math.min(parallel, Math.max(1, todo.length)) }, worker));
      last = await request(`uploads/${u.id}/`, { method: 'POST', body: {} });
    }
    return last;
  } finally {
    listeners.forEach((fn) => fn('end'));
  }
}

const get = (p, q, o) => request(p, { ...o, query: q });
const post = (p, b, o) => request(p, { ...o, method: 'POST', body: b ?? {} });
const patch = (p, b, o) => request(p, { ...o, method: 'PATCH', body: b ?? {} });
const del = (p, o) => request(p, { ...o, method: 'DELETE' });

let metaPromise = null;
export const api = {
  meta: (fresh) => { if (fresh || !metaPromise) metaPromise = get('meta/').catch((e) => { metaPromise = null; throw e; }); return metaPromise; },
  overview: (o) => get('overview/', null, o),
  // sources
  sources: (q, o) => get('sources/', q, o),
  source: (id, o) => get(`sources/${id}/`, null, o),
  createSource: (b) => post('sources/', b),
  updateSource: (id, b) => patch(`sources/${id}/`, b),
  deleteSource: (id) => del(`sources/${id}/`),
  uploadFiles: (id, files, opts) => uploadChunked(id, files, opts),
  uploadFilesSingle: (id, files, opts) => upload(`sources/${id}/files/`, files, opts),
  importApi: (b, o) => post('import-api/', b, o),
  importApiFile: (file, fields) => postFile('import-api/', file, fields),
  deleteFile: (id, fid) => del(`sources/${id}/files/${fid}/`),
  freeSpace: (id) => post(`sources/${id}/free-space/`, {}),
  testSource: (id) => post(`sources/${id}/test/`),
  extract: (id, b) => post(`sources/${id}/extract/`, b),
  snapshots: (id) => get(`sources/${id}/snapshots/`),
  snapshot: (id, o) => get(`snapshots/${id}/`, null, o),
  compare: (id, against) => get(`snapshots/${id}/compare/`, { against }),
  // mappings
  mappings: (o) => get('mappings/', null, o),
  mapping: (id, q, o) => get(`mappings/${id}/`, q, o),
  createMapping: (b) => post('mappings/', b),
  updateMapping: (id, b) => patch(`mappings/${id}/`, b),
  deleteMapping: (id) => del(`mappings/${id}/`),
  generate: (id, b) => post(`mappings/${id}/generate/`, b),
  fields: (id, b) => patch(`mappings/${id}/fields/`, b),
  bulk: (id, b) => post(`mappings/${id}/bulk/`, b),
  setTable: (id, b) => post(`mappings/${id}/table/`, b),
  preview: (id, b, o) => post(`mappings/${id}/preview/`, b, o),
  exportMapping: (id, q) => get(`mappings/${id}/export/`, q),
  exportUrl: (id, q) => `${API}mappings/${id}/export/?${new URLSearchParams({ ...q, download: 1 })}`,
  // syncs
  connections: (o) => get('connections/', null, o),
  connection: (id, o) => get(`connections/${id}/`, null, o),
  createConnection: (b) => post('connections/', b),
  updateConnection: (id, b) => patch(`connections/${id}/`, b),
  deleteConnection: (id) => del(`connections/${id}/`),
  runConnection: (id) => post(`connections/${id}/run/`),
  // jobs
  jobs: (q, o) => get('jobs/', q, o),
  job: (id, since, o) => get(`jobs/${id}/`, { since }, o),
  cancelJob: (id) => post(`jobs/${id}/cancel/`),
};

export async function connectorSpec(kind) {
  const meta = await api.meta();
  return meta.connectors.find((c) => c.key === kind);
}
