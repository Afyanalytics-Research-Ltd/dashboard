// Shared, short-lived caches over list endpoints so pages and the command palette
// don't refetch everything on each navigation. Mutations call invalidate(kind).

import { n8n } from './api.js';

const TTL = { workflows: 30_000, credentials: 60_000, tags: 60_000, users: 120_000, projects: 120_000, dataTables: 60_000 };
const cache = new Map();
const listeners = new Set();

const LOADERS = {
  workflows: (signal) => n8n.all('/workflows', {}, { signal }),
  credentials: (signal) => n8n.all('/credentials', {}, { signal }),
  tags: (signal) => n8n.all('/tags', {}, { signal }),
  users: (signal) => n8n.all('/users', { includeRole: true }, { signal }),
  projects: (signal) => n8n.all('/projects', {}, { signal }),
  dataTables: (signal) => n8n.all('/data-tables', {}, { signal }),
};

/** Cached list. Rejections are not cached (licence-gated lists reject with err.license). */
export function load(kind, { force = false, signal } = {}) {
  const hit = cache.get(kind);
  if (!force && hit && Date.now() - hit.at < (TTL[kind] || 30_000)) return hit.promise;
  const promise = LOADERS[kind](signal).catch((err) => { if (cache.get(kind)?.promise === promise) cache.delete(kind); throw err; });
  cache.set(kind, { at: Date.now(), promise });
  return promise;
}

/** Like load() but resolves to [] when the feature is licence-gated or forbidden. */
export async function loadOptional(kind, opts) {
  try {
    return await load(kind, opts);
  } catch (err) {
    if (err?.license || err?.status === 403 || err?.status === 404) return [];
    throw err;
  }
}

export function invalidate(...kinds) {
  for (const k of kinds.length ? kinds : [...cache.keys()]) cache.delete(k);
  listeners.forEach((fn) => fn(kinds));
}

export const onInvalidate = (fn) => (listeners.add(fn), () => listeners.delete(fn));

/** Map id -> item for a cached list. */
export async function byId(kind, opts) {
  return new Map((await loadOptional(kind, opts)).map((x) => [String(x.id), x]));
}

/** Update a cached workflow summary in place after a save/activate. */
export function patchWorkflow(wf) {
  const hit = cache.get('workflows');
  if (!hit) return;
  hit.promise = hit.promise.then((list) => {
    const i = list.findIndex((w) => String(w.id) === String(wf.id));
    if (i >= 0) list[i] = { ...list[i], ...wf };
    else list.unshift(wf);
    return list;
  });
}

export function removeWorkflow(id) {
  const hit = cache.get('workflows');
  if (hit) hit.promise = hit.promise.then((list) => list.filter((w) => String(w.id) !== String(id)));
}
