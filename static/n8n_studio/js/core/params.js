// n8n node-parameter helpers: which description applies to a typeVersion, which
// properties are visible (displayOptions show/hide incl. _cnd conditions and
// @version), defaults, and canvas subtitles.

export function versionsOf(desc) {
  const v = desc?.version ?? 1;
  return Array.isArray(v) ? v : [v];
}

/** Pick the description entry for typeVersion (a type can have one entry per version range). */
export function descriptionFor(descs, typeVersion) {
  if (!Array.isArray(descs) || !descs.length) return null;
  const tv = Number(typeVersion);
  return descs.find((d) => versionsOf(d).includes(tv))
    || descs.slice().sort((a, b) => Math.max(...versionsOf(b)) - Math.max(...versionsOf(a)))[0];
}

function cond(expected, actual) {
  if (expected && typeof expected === 'object' && '_cnd' in expected) {
    const [[op, v]] = Object.entries(expected._cnd);
    switch (op) {
      case 'eq': return actual === v;
      case 'not': return actual !== v;
      case 'gte': return Number(actual) >= v;
      case 'lte': return Number(actual) <= v;
      case 'gt': return Number(actual) > v;
      case 'lt': return Number(actual) < v;
      case 'between': return Number(actual) >= v.from && Number(actual) <= v.to;
      case 'startsWith': return String(actual ?? '').startsWith(v);
      case 'endsWith': return String(actual ?? '').endsWith(v);
      case 'includes': return String(actual ?? '').includes(v);
      case 'regex': try { return new RegExp(v).test(String(actual ?? '')); } catch { return false; }
      case 'exists': return actual !== undefined && actual !== null && actual !== '';
      default: return false;
    }
  }
  return actual === expected || (expected != null && typeof expected !== 'object' && actual != null && String(actual) === String(expected));
}

const getPath = (obj, path) => path.reduce((o, k) => (o == null ? undefined : o[k]), obj);

function lookup(name, local, root, ctx) {
  if (name === '@version') return Number(ctx.typeVersion);
  if (name === '@tool') return !!ctx.isTool;
  if (name.startsWith('/')) return getPath(root, name.slice(1).split('/'));
  return local?.[name];
}

/**
 * Is `prop` visible? `local` = sibling values (same collection level), `root` = all node
 * parameters, ctx = {typeVersion, defaults: values used when a parameter is unset}.
 */
export function isVisible(prop, local, root = local, ctx = {}) {
  const d = prop.displayOptions;
  if (!d) return true;
  const value = (name) => {
    let v = lookup(name, local, root, ctx);
    if (v === undefined && ctx.defaults) v = name.startsWith('/') ? getPath(ctx.defaults, name.slice(1).split('/')) : ctx.defaults[name];
    return v;
  };
  const test = (name, values) => {
    const v = value(name);
    const list = Array.isArray(values) ? values : [values];
    const actual = Array.isArray(v) ? v : [v];
    return list.some((exp) => actual.some((a) => cond(exp, a)));
  };
  for (const [name, values] of Object.entries(d.show || {})) if (!test(name, values)) return false;
  for (const [name, values] of Object.entries(d.hide || {})) if (test(name, values)) return false;
  return true;
}

/** Defaults of the top-level properties visible for `current` values (resolved in passes). */
export function defaults(props, typeVersion, current = {}) {
  let out = {};
  for (let pass = 0; pass < 3; pass++) {
    const next = {};
    const merged = { ...out, ...current };
    for (const p of props || []) {
      if (['notice', 'button', 'callout', 'curlImport'].includes(p.type)) continue;
      if (p.name in next) continue;
      if (!isVisible(p, merged, merged, { typeVersion })) continue;
      if (p.default !== undefined) next[p.name] = p.default;
    }
    out = next;
  }
  return out;
}

/** Visible properties; several props may share a name (per-version variants) — first visible wins. */
export function visibleProps(props, local, root, ctx = {}) {
  const seen = new Set();
  const out = [];
  for (const p of props || []) {
    if (seen.has(p.name) && p.type !== 'notice') continue;
    if (isVisible(p, local, root, ctx)) {
      out.push(p);
      seen.add(p.name);
    }
  }
  return out;
}

/** Canvas subtitle ("resource: operation"), resolving simple $parameter references. */
export function subtitle(desc, params) {
  const tpl = desc?.subtitle;
  if (!tpl || typeof tpl !== 'string') return params?.operation ? String(params.operation) : '';
  const body = tpl.replace(/^=/, '');
  const m = body.match(/^\{\{([\s\S]*)\}\}$/);
  const expr = m ? m[1] : body;
  if (!/\$parameter/.test(expr)) return m ? '' : body;
  try {
    return expr
      .replace(/\$parameter\[\s*["'](\w+)["']\s*\]/g, (_, k) => JSON.stringify(String(params?.[k] ?? '')))
      .split('+')
      .map((s) => s.trim())
      .map((s) => (/^"[^"]*"$|^'[^']*'$/.test(s) ? s.slice(1, -1) : ''))
      .join('');
  } catch {
    return '';
  }
}

/** Option label for a value of an `options` property. */
export function optionLabel(prop, value) {
  const opt = (prop?.options || []).find((o) => o.value === value);
  return opt ? opt.name : value;
}
