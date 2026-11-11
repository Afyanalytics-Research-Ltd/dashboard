// Expression preview. n8n expressions are JavaScript, and workflow content may be
// written by any n8n user, so previews run inside a sandboxed iframe with an opaque
// origin (sandbox="allow-scripts", no allow-same-origin): it cannot read cookies,
// call the dashboard API or touch this page — only postMessage a result back.
//
//   const r = await previewExpression('={{ $json.total * 2 }}', { json: {...}, nodes: {Name: [items]} })
//   r = { ok: true, value, text } | { ok: false, error }

const RUNTIME = `
const isoDate = (d) => d.toISOString();
function luxonish(d) {
  const x = new Date(d.getTime());
  x.toISO = () => isoDate(x);
  x.toISODate = () => isoDate(x).slice(0, 10);
  x.toMillis = () => x.getTime();
  x.toSeconds = () => Math.floor(x.getTime() / 1000);
  x.plus = (o) => luxonish(new Date(x.getTime() + dur(o)));
  x.minus = (o) => luxonish(new Date(x.getTime() - dur(o)));
  x.toFormat = (f) => f.replace(/yyyy/g, x.getFullYear()).replace(/MM/g, String(x.getMonth() + 1).padStart(2, '0'))
    .replace(/dd/g, String(x.getDate()).padStart(2, '0')).replace(/HH/g, String(x.getHours()).padStart(2, '0'))
    .replace(/mm/g, String(x.getMinutes()).padStart(2, '0')).replace(/ss/g, String(x.getSeconds()).padStart(2, '0'));
  return x;
}
function dur(o) {
  if (typeof o === 'number') return o;
  const u = { years: 31536e6, months: 2592e6, weeks: 6048e5, days: 864e5, hours: 36e5, minutes: 6e4, seconds: 1e3, milliseconds: 1 };
  return Object.entries(o || {}).reduce((a, [k, v]) => a + (u[k] || u[k + 's'] || 0) * v, 0);
}
function nodeAccessor(items) {
  const list = items || [];
  const first = list[0] || { json: {} };
  return { item: first, first: () => first, last: () => list[list.length - 1] || { json: {} }, all: () => list,
    json: first.json, params: {}, itemMatching: (i) => list[i] || first, isExecuted: list.length > 0 };
}
// A few of n8n's data-transformation extensions, so common previews work.
const ext = (proto, name, fn) => { if (!proto[name]) Object.defineProperty(proto, name, { value: fn, configurable: true }); };
ext(String.prototype, 'isEmpty', function () { return this.length === 0; });
ext(String.prototype, 'isNotEmpty', function () { return this.length > 0; });
ext(String.prototype, 'toTitleCase', function () { return this.replace(/\\w\\S*/g, (t) => t[0].toUpperCase() + t.slice(1).toLowerCase()); });
ext(String.prototype, 'toNumber', function () { return Number(this); });
ext(String.prototype, 'extractEmail', function () { return (this.match(/[\\w.+-]+@[\\w-]+\\.[\\w.]+/) || [])[0]; });
ext(String.prototype, 'toDateTime', function () { return luxonish(new Date(String(this))); });
ext(Array.prototype, 'first', function () { return this[0]; });
ext(Array.prototype, 'last', function () { return this[this.length - 1]; });
ext(Array.prototype, 'sum', function () { return this.reduce((a, b) => a + Number(b || 0), 0); });
ext(Array.prototype, 'average', function () { return this.length ? this.sum() / this.length : 0; });
ext(Array.prototype, 'unique', function () { return [...new Set(this)]; });
ext(Array.prototype, 'isEmpty', function () { return this.length === 0; });
ext(Array.prototype, 'pluck', function (...keys) { return this.map((o) => (keys.length === 1 ? o?.[keys[0]] : Object.fromEntries(keys.map((k) => [k, o?.[k]])))); });
ext(Object.prototype, 'isEmpty', function () { return Object.keys(this).length === 0; });
ext(Object.prototype, 'keys', function () { return Object.keys(this); });

function evaluate(template, ctx) {
  const nodes = ctx.nodes || {};
  const input = ctx.input || [{ json: ctx.json || {} }];
  const scope = {
    $json: ctx.json || input[0]?.json || {},
    $binary: input[0]?.binary || {},
    $input: nodeAccessor(input),
    $: (name) => { if (!(name in nodes)) throw new Error('Referenced node "' + name + '" has no data in this preview'); return nodeAccessor(nodes[name]); },
    $node: new Proxy({}, { get: (_, name) => nodeAccessor(nodes[name]) }),
    $items: (name) => (name ? nodes[name] : input) || [],
    $now: luxonish(new Date()),
    $today: luxonish(new Date(new Date().setHours(0, 0, 0, 0))),
    $vars: ctx.vars || {},
    $workflow: ctx.workflow || {},
    $execution: { id: 'preview', mode: 'test', resumeUrl: '' },
    $runIndex: 0, $itemIndex: 0, $position: 0,
    $prevNode: { name: ctx.prevNode || '' },
    $parameter: ctx.parameters || {},
    DateTime: { now: () => luxonish(new Date()), fromISO: (s) => luxonish(new Date(s)) },
  };
  const names = Object.keys(scope);
  const values = Object.values(scope);
  let text = String(template);
  if (text.startsWith('=')) text = text.slice(1);
  const parts = [];
  let last = 0;
  const re = /\\{\\{([\\s\\S]*?)\\}\\}/g;
  let m;
  while ((m = re.exec(text))) {
    parts.push({ lit: text.slice(last, m.index) });
    // eslint-disable-next-line no-new-func
    const fn = new Function(...names, '"use strict"; return (' + m[1] + '\\n);');
    parts.push({ val: fn(...values) });
    last = m.index + m[0].length;
  }
  parts.push({ lit: text.slice(last) });
  const vals = parts.filter((p) => 'val' in p);
  const onlyExpr = vals.length === 1 && parts.every((p) => 'val' in p || p.lit === '');
  if (onlyExpr) return vals[0].val;
  return parts.map((p) => ('lit' in p ? p.lit : typeof p.val === 'object' ? JSON.stringify(p.val) : String(p.val))).join('');
}
addEventListener('message', (e) => {
  const { id, template, ctx } = e.data || {};
  let reply;
  try {
    const value = evaluate(template, ctx || {});
    let safe;
    try { safe = JSON.parse(JSON.stringify(value === undefined ? null : value)); } catch { safe = String(value); }
    reply = { id, ok: true, value: safe, type: Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value };
  } catch (err) {
    reply = { id, ok: false, error: String(err && err.message || err) };
  }
  parent.postMessage(reply, '*');
});
parent.postMessage({ ready: true }, '*');
`;

let frame = null;
let ready = null;
let seq = 0;
const pending = new Map();

function ensureFrame() {
  if (ready) return ready;
  ready = new Promise((resolve, reject) => {
    frame = document.createElement('iframe');
    frame.setAttribute('sandbox', 'allow-scripts');
    frame.setAttribute('aria-hidden', 'true');
    frame.style.display = 'none';
    frame.srcdoc = `<!doctype html><meta charset="utf-8"><script>${RUNTIME}<\/script>`;
    const timer = setTimeout(() => reject(new Error('Expression sandbox did not start (blocked by a Content-Security-Policy?)')), 4000);
    window.addEventListener('message', (e) => {
      if (e.source !== frame?.contentWindow) return;
      if (e.data?.ready) { clearTimeout(timer); resolve(); return; }
      const p = pending.get(e.data?.id);
      if (p) { pending.delete(e.data.id); clearTimeout(p.timer); p.resolve(e.data); }
    });
    document.body.appendChild(frame);
  });
  return ready;
}

export const isExpression = (v) => typeof v === 'string' && v.startsWith('=');

/** Evaluate an n8n expression against sample data in the sandbox (2 s timeout). */
export async function previewExpression(template, ctx = {}) {
  await ensureFrame();
  const id = ++seq;
  return new Promise((resolve) => {
    const timer = setTimeout(() => { pending.delete(id); resolve({ ok: false, error: 'Timed out (infinite loop?)' }); }, 2000);
    pending.set(id, { resolve, timer });
    frame.contentWindow.postMessage({ id, template, ctx: JSON.parse(JSON.stringify(ctx)) }, '*');
  });
}

/** Text form of a preview result for display. */
export function previewText(r) {
  if (!r.ok) return `⚠ ${r.error}`;
  if (typeof r.value === 'string') return r.value;
  return JSON.stringify(r.value, null, 2);
}
