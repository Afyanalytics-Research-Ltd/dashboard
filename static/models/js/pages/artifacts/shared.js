// Shared helpers for the artifacts / registry / sweeps / reports pages.

import { h, icon, mount } from '../../../../ui/js/dom.js';
import { toast, toastError, confirmDialog } from '../../../../ui/js/ui.js';
import { fmtBytes } from '../../../../ui/js/format.js';
import { gql, collect, parseJSON } from '../../core/api.js';

export { fmtBytes };

// ------------------------------------------------------------------ type look

const TYPE_LOOK = {
  model: { icon: 'cpu', color: '#7C3AED', bg: '#ede9fe' },
  dataset: { icon: 'database', color: '#0072CE', bg: '#e8f4ff' },
  run_table: { icon: 'table', color: '#0a9a85', bg: '#e6faf8' },
  code: { icon: 'code-slash', color: '#475569', bg: '#f1f5f9' },
  job: { icon: 'briefcase', color: '#b45309', bg: '#fef3c7' },
  evaluation: { icon: 'clipboard-check', color: '#be185d', bg: '#fce7f3' },
};
export const typeLook = (type) => TYPE_LOOK[type] || { icon: 'box', color: '#64748b', bg: '#f1f5f9' };

export function typeBadge(type) {
  const t = typeLook(type);
  return h('span.ma-type', { style: { color: t.color, background: t.bg } }, icon(t.icon), type);
}

export function typeIcon(type, size = 34) {
  const t = typeLook(type);
  return h('span.ma-type-icon', { style: { color: t.color, background: t.bg, width: `${size}px`, height: `${size}px` } }, icon(t.icon));
}

const VERSION_ALIAS = /^v\d+$/;
/** Aliases a person cares about (W&B also returns the implicit "vN"). */
// A version linked into a registry portfolio also carries that portfolio's aliases; pass `collection`
// to keep only the aliases of the collection being viewed (and never show one twice).
export const userAliases = (aliases, collection) => [...new Set((aliases || [])
  .filter((a) => !collection || typeof a === 'string' || !a.artifactCollectionName || a.artifactCollectionName === collection)
  .map((a) => (typeof a === 'string' ? a : a.alias)).filter((a) => !VERSION_ALIAS.test(a)))];

const ALIAS_LOOK = {
  latest: 'ma-alias-latest', best: 'ma-alias-best', production: 'ma-alias-prod', prod: 'ma-alias-prod',
  staging: 'ma-alias-staging', champion: 'ma-alias-best', candidate: 'ma-alias-staging',
};
export function aliasChip(alias, { onRemove } = {}) {
  return h('span.ma-alias', { class: ALIAS_LOOK[alias] || '' },
    alias === 'production' || alias === 'prod' ? icon('rocket-takeoff') : alias === 'best' ? icon('trophy') : alias === 'latest' ? icon('clock') : icon('tag'),
    alias,
    onRemove && h('button.ma-alias-x', { type: 'button', title: `Remove alias "${alias}"`, onclick: (e) => { e.stopPropagation(); onRemove(alias); } }, icon('x')));
}

// ------------------------------------------------------------------ alias mutations

const aliasInput = (entity, project, collection, alias) => ({ alias, artifactCollectionName: collection, entityName: entity, projectName: project });

export async function addAlias({ entity, project, collection, artifactId, alias }) {
  await gql(`mutation AddAlias($input: AddAliasesInput!) { addAliases(input: $input) { success } }`,
    { input: { artifactID: artifactId, aliases: [aliasInput(entity, project, collection, alias)] } });
}

export async function removeAlias({ entity, project, collection, artifactId, alias }) {
  await gql(`mutation DelAlias($input: DeleteAliasesInput!) { deleteAliases(input: $input) { success } }`,
    { input: { artifactID: artifactId, aliases: [aliasInput(entity, project, collection, alias)] } });
}

/**
 * Inline alias editor for one version. Moving an alias that another version holds is
 * explicit: W&B silently re-points it, so we say so before doing it.
 * holders: Map(alias -> versionLabel) for the collection, used for that warning.
 */
export function aliasEditor({ entity, project, collection, artifact, holders = new Map(), onChange, compact = false }) {
  const el = h('div.ma-aliases', { class: compact && 'compact' });
  let aliases = userAliases(artifact.aliases, collection);
  const render = () => {
    mount(el,
      aliases.map((a) => aliasChip(a, { onRemove: a === 'latest' ? null : remove })),
      h('button.ma-alias-add', { type: 'button', title: 'Add an alias (e.g. production, staging, best)', onclick: (e) => openAdd(e.currentTarget) }, icon('plus'), compact ? '' : ' alias'));
  };
  async function remove(alias) {
    const ok = await confirmDialog({ title: `Remove alias “${alias}”?`, message: `Code that loads ${collection}:${alias} will stop resolving to this version.`, confirmLabel: 'Remove alias', danger: alias === 'production' || alias === 'prod' });
    if (!ok) return;
    try {
      await removeAlias({ entity, project, collection, artifactId: artifact.id, alias });
      aliases = aliases.filter((x) => x !== alias);
      render();
      toast(`Removed “${alias}”`, 'success', { timeout: 2000 });
      onChange?.(aliases);
    } catch (err) { toastError(err, 'Could not remove alias'); }
  }
  function openAdd(anchor) {
    const input = h('input.form-control.form-control-sm.ma-alias-input', { placeholder: 'alias', list: 'ma-alias-suggest', 'aria-label': 'New alias' });
    const dl = h('datalist#ma-alias-suggest', ['production', 'staging', 'best', 'candidate', 'champion', 'baseline'].filter((s) => !aliases.includes(s)).map((s) => h('option', { value: s })));
    anchor.replaceWith(input, dl);
    input.focus();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const v = input.value.trim();
      dl.remove();
      if (!save || !v || aliases.includes(v)) { render(); return; }
      if (VERSION_ALIAS.test(v)) { toast('“vN” aliases are managed by W&B.', 'warning'); render(); return; }
      if (!/^[\w.-]+$/.test(v)) { toast('Aliases may contain letters, digits, “-”, “_” and “.” only.', 'warning'); render(); return; }
      const holder = holders.get(v);
      if (holder && holder !== artifact.versionLabel) {
        const ok = await confirmDialog({ title: `Move “${v}” here?`, message: `“${v}” currently points to ${holder}. It will move to ${artifact.versionLabel}.`, confirmLabel: 'Move alias', danger: v === 'production' || v === 'prod' });
        if (!ok) { render(); return; }
      }
      try {
        await addAlias({ entity, project, collection, artifactId: artifact.id, alias: v });
        aliases = [...aliases, v];
        render();
        toast(`“${v}” → ${artifact.versionLabel || 'this version'}`, 'success', { timeout: 2200 });
        onChange?.(aliases, v);
      } catch (err) { toastError(err, 'Could not add alias'); render(); }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
  }
  render();
  return el;
}

// ------------------------------------------------------------------ queries

export const VERSION_FIELDS = `id digest versionIndex state size fileCount createdAt updatedAt description metadata usedCount ttlDurationSeconds ttlIsInherited
  aliases { alias artifactCollectionName }
  createdBy { __typename ... on Run { id name displayName state project { name entityName } } ... on User { username name } }`;

/** All collections of one type in a project (Relay pages). */
export function loadCollections(entity, project, type, { signal, max = 1000 } = {}) {
  return collect((after) => gql(`query Cols($e: String!, $p: String!, $t: String!, $after: String) {
    project(entityName: $e, name: $p) { artifactType(name: $t) { artifactCollections(first: 100, after: $after) {
      totalCount pageInfo { hasNextPage endCursor }
      edges { node { __typename id name description createdAt updatedAt
        aliases(first: 25) { edges { node { alias } } }
        tags(first: 20) { edges { node { name } } }
        artifacts(first: 1) { totalCount edges { version node { id size createdAt versionIndex createdBy { __typename ... on Run { name displayName } } } } } } } } } } }`,
  { e: entity, p: project, t: type, after }, { signal }), (d) => d.project?.artifactType?.artifactCollections, { max });
}

export async function loadTypes(entity, project, { signal } = {}) {
  const d = await gql(`query Types($e: String!, $p: String!) { project(entityName: $e, name: $p) { id
    artifactTypes(first: 100) { edges { node { id name description createdAt artifactCollections(first: 1) { totalCount } } } } } }`,
  { e: entity, p: project }, { signal });
  if (!d.project) throw Object.assign(new Error(`Project ${entity}/${project} not found`), { status: 404 });
  return d.project.artifactTypes.edges.map((e) => ({ ...e.node, count: e.node.artifactCollections?.totalCount ?? 0 }));
}

/** Collection + all its versions. */
export async function loadCollection(entity, project, name, { signal, max = 500 } = {}) {
  const head = await gql(`query Col($e: String!, $p: String!, $n: String!) { project(entityName: $e, name: $p) { id
    artifactCollection(name: $n) { __typename id name description createdAt updatedAt
      tags(first: 50) { edges { node { id name } } }
      aliases(first: 100) { edges { node { alias } } } } } }`, { e: entity, p: project, n: name }, { signal });
  const col = head.project?.artifactCollection;
  if (!col) throw Object.assign(new Error(`Artifact collection “${name}” not found`), { status: 404 });
  const versions = [];
  let after = null;
  for (;;) {
    const d = await gql(`query Vers($e: String!, $p: String!, $n: String!, $after: String) { project(entityName: $e, name: $p) {
      artifactCollection(name: $n) { artifacts(first: 100, after: $after) { totalCount pageInfo { hasNextPage endCursor }
        edges { version node { ${VERSION_FIELDS} } } } } } }`, { e: entity, p: project, n: name, after }, { signal });
    const conn = d.project.artifactCollection.artifacts;
    versions.push(...conn.edges.map((e) => ({ ...e.node, versionLabel: e.version, meta: parseJSON(e.node.metadata, {}) })));
    if (!conn.pageInfo.hasNextPage || versions.length >= max) break;
    after = conn.pageInfo.endCursor;
  }
  return { ...col, projectId: head.project.id, tags: col.tags.edges.map((e) => e.node), versions };
}

/** One version by "name:alias" (alias may be vN, latest, production...). */
export async function loadArtifact(entity, project, collection, version, { signal } = {}) {
  const d = await gql(`query Art($e: String!, $p: String!, $n: String!) { project(entityName: $e, name: $p) {
    artifact(name: $n) { ${VERSION_FIELDS} digestAlgorithm storageBytes historyStep isLinkedToGlobalRegistry
      artifactType { name }
      artifactSequence { id name }
      currentManifest { file { directUrl sizeBytes } }
      artifactCollections { edges { node { __typename id name project { name entityName } } } }
      usedBy(first: 100) { totalCount edges { node { id name displayName state createdAt jobType project { name entityName } } } } } } }`,
  { e: entity, p: project, n: `${collection}:${version}` }, { signal });
  const a = d.project?.artifact;
  if (!a) throw Object.assign(new Error(`Artifact ${collection}:${version} not found`), { status: 404 });
  const label = `v${a.versionIndex}`;
  return { ...a, versionLabel: label, meta: parseJSON(a.metadata, {}), usedRuns: a.usedBy.edges.map((e) => e.node), usedTotal: a.usedBy.totalCount };
}

export async function loadArtifactFiles(artifactId, { signal, max = 2000 } = {}) {
  return collect((after) => gql(`query Files($id: ID!, $after: String) { artifact(id: $id) {
    files(first: 200, after: $after) { totalCount pageInfo { hasNextPage endCursor } edges { node { id name displayName sizeBytes mimetype digest directUrl md5 } } } } }`,
  { id: artifactId, after }, { signal }), (d) => d.artifact?.files, { max });
}

// ------------------------------------------------------------------ diffing

export function metadataDiff(a = {}, b = {}) {
  const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
  return keys.map((k) => {
    const va = JSON.stringify(a[k]);
    const vb = JSON.stringify(b[k]);
    return { key: k, a: a[k], b: b[k], status: va === vb ? 'same' : va === undefined ? 'added' : vb === undefined ? 'removed' : 'changed' };
  });
}

export function fileDiff(filesA = [], filesB = []) {
  const ma = new Map(filesA.map((f) => [f.name, f]));
  const mb = new Map(filesB.map((f) => [f.name, f]));
  const names = [...new Set([...ma.keys(), ...mb.keys()])].sort();
  return names.map((n) => {
    const fa = ma.get(n);
    const fb = mb.get(n);
    const status = !fa ? 'added' : !fb ? 'removed' : (fa.digest || fa.md5) === (fb.digest || fb.md5) ? 'same' : 'changed';
    return { name: n, a: fa, b: fb, status };
  });
}

// ------------------------------------------------------------------ minimal YAML (W&B sweep configs)

/** Parse the subset of YAML W&B writes for sweep configs: nested maps, "- " lists, scalars. */
export function parseYaml(text) {
  const lines = String(text || '').split('\n').filter((l) => l.trim() && !l.trim().startsWith('#'));
  let i = 0;
  const scalar = (s) => {
    const v = s.trim();
    if (v === '' || v === '~' || v === 'null') return null;
    if (v === 'true') return true;
    if (v === 'false') return false;
    if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(v)) return Number(v);
    if (/^['"].*['"]$/.test(v)) return v.slice(1, -1);
    if (v.startsWith('[') && v.endsWith(']')) return v.slice(1, -1).split(',').map((x) => scalar(x)).filter((x) => x !== null || true);
    return v;
  };
  const indentOf = (l) => l.length - l.trimStart().length;
  function block(indent) {
    if (i >= lines.length) return null;
    const isList = lines[i].trimStart().startsWith('- ') || lines[i].trim() === '-';
    if (isList) {
      const arr = [];
      while (i < lines.length && indentOf(lines[i]) === indent && lines[i].trimStart().startsWith('-')) {
        const rest = lines[i].trimStart().slice(1).trim();
        i++;
        if (!rest) arr.push(block(i < lines.length ? indentOf(lines[i]) : indent + 2));
        else if (/^[\w./-]+:\s/.test(rest) || /^[\w./-]+:$/.test(rest)) {
          lines.splice(i, 0, `${' '.repeat(indent + 2)}${rest}`);
          arr.push(block(indent + 2));
        } else arr.push(scalar(rest));
      }
      return arr;
    }
    const obj = {};
    while (i < lines.length && indentOf(lines[i]) === indent) {
      const line = lines[i].trim();
      const m = line.match(/^("?[^":]+"?)\s*:\s*(.*)$/);
      i++;
      if (!m) continue;
      const key = m[1].replace(/^"|"$/g, '');
      const next = lines[i];
      // YAML lets a list sit at the same indent as its key ("values:\n- 1"), which W&B writes.
      const childList = next != null && indentOf(next) === indent && next.trimStart().startsWith('-');
      if (m[2] !== '') obj[key] = scalar(m[2]);
      else if (childList) obj[key] = block(indent);
      else obj[key] = next != null && indentOf(next) > indent ? block(indentOf(next)) : null;
    }
    return obj;
  }
  try { return block(lines.length ? indentOf(lines[0]) : 0) || {}; } catch { return {}; }
}

// ------------------------------------------------------------------ statistics (parameter importance)

function ranks(values) {
  const idx = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(values.length);
  for (let k = 0; k < idx.length;) {
    let j = k;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[k][0]) j++;
    const avg = (k + j) / 2 + 1;
    for (let t = k; t <= j; t++) r[idx[t][1]] = avg;
    k = j + 1;
  }
  return r;
}

function pearson(x, y) {
  const n = x.length;
  if (n < 3) return null;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
}

export const spearman = (x, y) => pearson(ranks(x), ranks(y));

/** Variance explained by the best single threshold split (numeric) or by grouping (categorical). 0..1 */
function splitImportance(xs, ys, categorical) {
  const n = ys.length;
  const mean = ys.reduce((a, b) => a + b, 0) / n;
  const total = ys.reduce((a, y) => a + (y - mean) ** 2, 0);
  if (!total) return 0;
  if (categorical) {
    const groups = new Map();
    xs.forEach((x, i) => { const k = String(x); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(ys[i]); });
    let within = 0;
    for (const g of groups.values()) { const m = g.reduce((a, b) => a + b, 0) / g.length; within += g.reduce((a, y) => a + (y - m) ** 2, 0); }
    return Math.max(0, 1 - within / total);
  }
  const order = xs.map((x, i) => [x, ys[i]]).sort((a, b) => a[0] - b[0]);
  let best = 0;
  for (let k = 1; k < n; k++) {
    if (order[k][0] === order[k - 1][0]) continue;
    const left = order.slice(0, k).map((p) => p[1]);
    const right = order.slice(k).map((p) => p[1]);
    const sse = (arr) => { const m = arr.reduce((a, b) => a + b, 0) / arr.length; return arr.reduce((a, y) => a + (y - m) ** 2, 0); };
    best = Math.max(best, 1 - (sse(left) + sse(right)) / total);
  }
  return best;
}

/**
 * Importance of each parameter for a metric over runs.
 * rows: [{params:{...}, metric:number}] -> [{param, correlation, importance, categorical, n}]
 */
export function parameterImportance(rows, params) {
  const out = [];
  for (const p of params) {
    const pts = rows.filter((r) => r.params[p] != null && Number.isFinite(r.metric));
    if (pts.length < 3) continue;
    const xs = pts.map((r) => r.params[p]);
    const ys = pts.map((r) => r.metric);
    const categorical = xs.some((x) => typeof x !== 'number');
    if (new Set(xs.map(String)).size < 2) continue;
    let correlation = null;
    if (!categorical) correlation = spearman(xs, ys);
    out.push({ param: p, correlation, importance: splitImportance(xs, ys, categorical), categorical, n: pts.length });
  }
  return out.sort((a, b) => b.importance - a.importance);
}

// ------------------------------------------------------------------ misc

export function shortDigest(d) {
  return d ? String(d).slice(0, 10) : '—';
}

export function ttlText(seconds, inherited) {
  if (seconds == null || seconds < 0) return inherited ? 'Inherited (never expires)' : 'Never expires';
  const d = Math.round(seconds / 86400);
  return d >= 1 ? `${d} day${d === 1 ? '' : 's'}` : `${Math.round(seconds / 3600)} h`;
}

export function creatorLink(createdBy, urlsMod) {
  if (!createdBy) return h('span.text-muted', '—');
  if (createdBy.__typename === 'Run') {
    const pr = createdBy.project || {};
    return h('a.ms-run-name', { href: urlsMod.run(pr.entityName, pr.name, createdBy.name) }, createdBy.displayName || createdBy.name);
  }
  return h('span', icon('person', 'me-1'), createdBy.username || createdBy.name || 'user');
}

export function guard(fn, title) {
  return async (...args) => {
    try { return await fn(...args); } catch (err) { toastError(err, title); return null; }
  };
}
