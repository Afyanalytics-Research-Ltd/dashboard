// Pure workflow-model helpers (no DOM): graph edges, rename with expression rewrite,
// copy/paste in n8n's clipboard format, auto-layout, lint, diff, search, docs.
//
// n8n connections: { [sourceName]: { [type]: [ /* output index */ [ {node, type, index} ] ] } }

import { deepDiff } from './jsondiff.js';

export const STICKY = 'n8n-nodes-base.stickyNote';
export const GRID = 20;

export const SETTINGS_KEYS = [
  'saveExecutionProgress', 'saveManualExecutions', 'saveDataErrorExecution', 'saveDataSuccessExecution',
  'executionTimeout', 'errorWorkflow', 'timezone', 'executionOrder', 'binaryMode', 'callerPolicy', 'callerIds',
  'timeSavedMode', 'timeSavedPerExecution', 'redactionPolicy', 'availableInMCP', 'customTelemetryTags',
  'credentialResolverId',
];

export const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));

export function uuid() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export const escapeRegExp = (s) => String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
export const isSticky = (node) => node?.type === STICKY;

// ------------------------------------------------------------------ node classification

const TRIGGER_RE = /(trigger|webhook|cron|schedule|interval|start|formtrigger|chattrigger|mcptrigger)$/i;

/** True for trigger nodes. `entry` (catalog index entry) wins over the name heuristic. */
export function isTriggerType(type, entry) {
  if (entry && typeof entry.trigger === 'boolean') return entry.trigger;
  const short = String(type || '').split('.').pop();
  return TRIGGER_RE.test(short) && !/^(respondToWebhook|executeWorkflow)$/i.test(short);
}

export function triggers(wf, catalog) {
  return (wf.nodes || []).filter((n) => !isSticky(n) && isTriggerType(n.type, catalog?.get?.(n.type)));
}

export function triggerKinds(wf, catalog) {
  const kinds = new Set();
  for (const n of triggers(wf, catalog)) {
    const t = n.type.toLowerCase();
    if (t.includes('webhook')) kinds.add('webhook');
    else if (t.includes('schedule') || t.includes('cron') || t.includes('interval')) kinds.add('schedule');
    else if (t.includes('manual')) kinds.add('manual');
    else if (t.includes('form')) kinds.add('form');
    else if (t.includes('chat')) kinds.add('chat');
    else if (t.includes('executeworkflowtrigger')) kinds.add('sub-workflow');
    else if (t.includes('errortrigger')) kinds.add('error');
    else kinds.add('app');
  }
  return [...kinds];
}

export const shortType = (type) => String(type || '').split('.').pop();

// ------------------------------------------------------------------ edges

export function edges(wf) {
  const out = [];
  for (const [from, byType] of Object.entries(wf.connections || {})) {
    for (const [type, outputs] of Object.entries(byType || {})) {
      (outputs || []).forEach((targets, fromIndex) => {
        for (const t of targets || []) {
          if (t && t.node != null) out.push({ from, fromType: type, fromIndex, to: t.node, toType: t.type || type, toIndex: t.index || 0 });
        }
      });
    }
  }
  return out;
}

export function edgesToConnections(list) {
  const conns = {};
  for (const e of list) {
    const byType = (conns[e.from] ||= {});
    const outputs = (byType[e.fromType] ||= []);
    while (outputs.length <= e.fromIndex) outputs.push([]);
    outputs[e.fromIndex].push({ node: e.to, type: e.toType, index: e.toIndex });
  }
  return conns;
}

export const edgeKey = (e) => `${e.from}\u0000${e.fromType}\u0000${e.fromIndex}\u0000${e.to}\u0000${e.toType}\u0000${e.toIndex}`;

export function addConnection(wf, edge) {
  const list = edges(wf);
  if (edge.from === edge.to) return wf;
  if (list.some((e) => edgeKey(e) === edgeKey(edge))) return wf;
  return { ...wf, connections: edgesToConnections([...list, edge]) };
}

export function removeConnection(wf, edge) {
  const key = edgeKey(edge);
  return { ...wf, connections: edgesToConnections(edges(wf).filter((e) => edgeKey(e) !== key)) };
}

export function parents(wf, name) {
  return [...new Set(edges(wf).filter((e) => e.to === name).map((e) => e.from))];
}

export function children(wf, name) {
  return [...new Set(edges(wf).filter((e) => e.from === name).map((e) => e.to))];
}

/** Every node upstream of `name` (main connections), nearest first. */
export function ancestors(wf, name) {
  const list = edges(wf);
  const seen = new Set();
  const order = [];
  let frontier = [name];
  while (frontier.length) {
    const next = [];
    for (const n of frontier) {
      for (const e of list) {
        if (e.to === n && !seen.has(e.from)) { seen.add(e.from); order.push(e.from); next.push(e.from); }
      }
    }
    frontier = next;
  }
  return order.filter((n) => n !== name);
}

// ------------------------------------------------------------------ names

/** n8n-style unique name: "Set", "Set1", "Set2"… */
export function uniqueName(base, existing) {
  const taken = existing instanceof Set ? existing : new Set(existing);
  const clean = String(base || 'Node').trim() || 'Node';
  if (!taken.has(clean)) return clean;
  const stem = clean.replace(/\d+$/, '');
  for (let i = 1; ; i++) {
    const candidate = `${stem}${i}`;
    if (!taken.has(candidate)) return candidate;
  }
}

function rewriteString(str, oldName, newName) {
  let out = str;
  const e = escapeRegExp(oldName);
  out = out.replace(new RegExp(`(\\$\\(\\s*)(['"\`])${e}\\2(\\s*\\))`, 'g'), (_, a, q, b) => `${a}${q}${newName}${q}${b}`);
  out = out.replace(new RegExp(`(\\$node\\[\\s*)(['"\`])${e}\\2(\\s*\\])`, 'g'), (_, a, q, b) => `${a}${q}${newName}${q}${b}`);
  out = out.replace(new RegExp(`(\\$items\\(\\s*)(['"\`])${e}\\2`, 'g'), (_, a, q) => `${a}${q}${newName}${q}`);
  if (/^[A-Za-z_$][\w$]*$/.test(oldName)) {
    const replacement = /^[A-Za-z_$][\w$]*$/.test(newName) ? `$node.${newName}` : `$node["${newName}"]`;
    out = out.replace(new RegExp(`\\$node\\.${e}(?![\\w$])`, 'g'), replacement);
  }
  return out;
}

export function mapStrings(value, fn) {
  if (typeof value === 'string') return fn(value);
  if (Array.isArray(value)) return value.map((v) => mapStrings(v, fn));
  if (value && typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = mapStrings(v, fn);
    return out;
  }
  return value;
}

/** Rename a node, updating connections, pinData and expression references everywhere. */
export function renameNode(wf, oldName, newName) {
  if (!newName || oldName === newName) return wf;
  if ((wf.nodes || []).some((n) => n.name === newName)) throw new Error(`A node named "${newName}" already exists.`);
  const nodes = (wf.nodes || []).map((n) => {
    const renamed = n.name === oldName ? { ...n, name: newName } : n;
    return { ...renamed, parameters: mapStrings(renamed.parameters || {}, (s) => rewriteString(s, oldName, newName)) };
  });
  const list = edges(wf).map((e) => ({ ...e, from: e.from === oldName ? newName : e.from, to: e.to === oldName ? newName : e.to }));
  const pinData = wf.pinData ? Object.fromEntries(Object.entries(wf.pinData).map(([k, v]) => [k === oldName ? newName : k, v])) : wf.pinData;
  return { ...wf, nodes, connections: edgesToConnections(list), pinData };
}

/** Node names referenced in a value's expressions. */
export function expressionRefs(value) {
  const found = new Set();
  const res = [
    /\$\(\s*(['"`])((?:(?!\1).)+)\1\s*\)/g,
    /\$node\[\s*(['"`])((?:(?!\1).)+)\1\s*\]/g,
    /\$items\(\s*(['"`])((?:(?!\1).)+)\1/g,
  ];
  mapStrings(value, (s) => {
    for (const re of res) for (const m of s.matchAll(re)) found.add(m[2]);
    for (const m of s.matchAll(/\$node\.([A-Za-z_$][\w$]*)/g)) found.add(m[1]);
    return s;
  });
  return [...found];
}

export function removeNodes(wf, names) {
  const drop = new Set(names);
  const pinData = wf.pinData ? Object.fromEntries(Object.entries(wf.pinData).filter(([k]) => !drop.has(k))) : wf.pinData;
  return {
    ...wf,
    nodes: (wf.nodes || []).filter((n) => !drop.has(n.name)),
    connections: edgesToConnections(edges(wf).filter((e) => !drop.has(e.from) && !drop.has(e.to))),
    pinData,
  };
}

// ------------------------------------------------------------------ clipboard (n8n-compatible)

/** Serialize selected nodes exactly like the n8n editor's copy (so paste works both ways). */
export function toClipboard(wf, names) {
  const pick = new Set(names);
  const nodes = (wf.nodes || []).filter((n) => pick.has(n.name)).map(clone);
  const connections = edgesToConnections(edges(wf).filter((e) => pick.has(e.from) && pick.has(e.to)));
  const out = { nodes, connections };
  const pins = Object.fromEntries(Object.entries(wf.pinData || {}).filter(([k]) => pick.has(k)));
  if (Object.keys(pins).length) out.pinData = pins;
  return JSON.stringify(out, null, 2);
}

export function parseClipboard(text) {
  let data;
  try { data = JSON.parse(text); } catch { return null; }
  if (!data || typeof data !== 'object') return null;
  if (Array.isArray(data.nodes) && data.nodes.every((n) => n && n.type && n.name)) {
    return { nodes: data.nodes, connections: data.connections || {}, pinData: data.pinData || {} };
  }
  if (data.type && data.name && data.parameters) return { nodes: [data], connections: {}, pinData: {} };
  return null;
}

/** Merge clipboard contents into wf: unique names, fresh ids, offset positions. Returns {wf, names}. */
export function pasteInto(wf, clip, { offset = [40, 40], at } = {}) {
  let incoming = { nodes: clip.nodes.map(clone), connections: clone(clip.connections || {}), pinData: clone(clip.pinData || {}) };
  const taken = new Set((wf.nodes || []).map((n) => n.name));
  // Reserve every incoming name too, so a renamed "Set" never lands on an incoming "Set1".
  const reserved = new Set([...taken, ...incoming.nodes.map((n) => n.name)]);
  for (const original of incoming.nodes.map((n) => n.name)) {
    if (!taken.has(original)) continue;
    const name = uniqueName(original, reserved);
    reserved.add(name);
    incoming = renameNode(incoming, original, name);
  }
  const minX = Math.min(...incoming.nodes.map((n) => n.position?.[0] ?? 0));
  const minY = Math.min(...incoming.nodes.map((n) => n.position?.[1] ?? 0));
  const dx = at ? at[0] - minX : offset[0];
  const dy = at ? at[1] - minY : offset[1];
  const webhookTypes = /webhook|formtrigger|chattrigger|wait/i;
  const nodes = incoming.nodes.map((n) => {
    const copy = { ...n, id: uuid(), position: [snap((n.position?.[0] ?? 0) + dx), snap((n.position?.[1] ?? 0) + dy)] };
    if (n.webhookId || webhookTypes.test(n.type)) copy.webhookId = uuid();
    return copy;
  });
  const merged = {
    ...wf,
    nodes: [...(wf.nodes || []), ...nodes],
    connections: edgesToConnections([...edges(wf), ...edges(incoming)]),
    pinData: { ...(wf.pinData || {}), ...incoming.pinData },
  };
  return { wf: merged, names: nodes.map((n) => n.name) };
}

export const snap = (v, grid = GRID) => Math.round(v / grid) * grid;

// ------------------------------------------------------------------ geometry & layout

export const NODE_W = 100;
export const NODE_H = 100;

export function nodeSize(node) {
  if (isSticky(node)) return [node.parameters?.width || 240, node.parameters?.height || 160];
  return [NODE_W, NODE_H];
}

export function bounds(nodes) {
  if (!nodes.length) return { x: 0, y: 0, w: 0, h: 0 };
  let x1 = Infinity; let y1 = Infinity; let x2 = -Infinity; let y2 = -Infinity;
  for (const n of nodes) {
    const [w, hgt] = nodeSize(n);
    const [x, y] = n.position || [0, 0];
    x1 = Math.min(x1, x); y1 = Math.min(y1, y); x2 = Math.max(x2, x + w); y2 = Math.max(y2, y + hgt);
  }
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

/**
 * Layered left-to-right layout. Rank = longest path from a root over main connections;
 * rows ordered by parent barycentre; AI sub-nodes (ai_* connections) sit under their parent.
 */
export function autoLayout(wf, { gapX = 220, gapY = 140, origin = [0, 0] } = {}) {
  const nodes = (wf.nodes || []).filter((n) => !isSticky(n));
  const names = new Set(nodes.map((n) => n.name));
  const list = edges(wf).filter((e) => names.has(e.from) && names.has(e.to));
  const main = list.filter((e) => e.fromType === 'main');
  const sub = list.filter((e) => e.fromType !== 'main');
  const subNodes = new Set(sub.map((e) => e.from).filter((n) => !main.some((e) => e.from === n || e.to === n)));
  const rank = new Map();
  const incoming = new Map(nodes.map((n) => [n.name, []]));
  for (const e of main) incoming.get(e.to)?.push(e.from);
  const visiting = new Set();
  const rankOf = (name) => {
    if (rank.has(name)) return rank.get(name);
    if (visiting.has(name)) return 0; // cycle: break it
    visiting.add(name);
    const ps = incoming.get(name) || [];
    const r = ps.length ? Math.max(...ps.map(rankOf)) + 1 : 0;
    visiting.delete(name);
    rank.set(name, r);
    return r;
  };
  const layered = nodes.filter((n) => !subNodes.has(n.name));
  layered.forEach((n) => rankOf(n.name));
  const columns = new Map();
  for (const n of layered) {
    const r = rank.get(n.name);
    if (!columns.has(r)) columns.set(r, []);
    columns.get(r).push(n.name);
  }
  const pos = new Map();
  const ranks = [...columns.keys()].sort((a, b) => a - b);
  for (const r of ranks) {
    const col = columns.get(r);
    const key = (name) => {
      const ps = (incoming.get(name) || []).filter((p) => pos.has(p));
      if (!ps.length) return (wf.nodes.find((n) => n.name === name)?.position?.[1] ?? 0) / 1000;
      return ps.reduce((a, p) => a + pos.get(p)[1], 0) / ps.length;
    };
    col.sort((a, b) => key(a) - key(b));
    const height = (col.length - 1) * gapY;
    col.forEach((name, i) => pos.set(name, [origin[0] + r * gapX, origin[1] + i * gapY - height / 2]));
  }
  // sub-nodes in a row under the node they plug into
  const byParent = new Map();
  for (const e of sub) {
    if (!subNodes.has(e.from)) continue;
    if (!byParent.has(e.to)) byParent.set(e.to, []);
    if (!byParent.get(e.to).includes(e.from)) byParent.get(e.to).push(e.from);
  }
  for (const [parent, kids] of byParent) {
    const p = pos.get(parent) || [origin[0], origin[1]];
    kids.forEach((k, i) => { if (!pos.has(k)) pos.set(k, [p[0] + (i - (kids.length - 1) / 2) * 120, p[1] + 180]); });
  }
  let maxY = Math.max(0, ...[...pos.values()].map((p) => p[1]));
  for (const n of nodes) if (!pos.has(n.name)) { maxY += gapY; pos.set(n.name, [origin[0], maxY]); }
  return {
    ...wf,
    nodes: (wf.nodes || []).map((n) => (pos.has(n.name) ? { ...n, position: pos.get(n.name).map((v) => snap(v)) } : n)),
  };
}

// ------------------------------------------------------------------ credentials, webhooks, schedules

export function credentialRefs(wf) {
  const out = [];
  for (const n of wf.nodes || []) {
    for (const [type, ref] of Object.entries(n.credentials || {})) {
      out.push({ node: n.name, nodeType: n.type, type, id: ref?.id != null ? String(ref.id) : null, name: ref?.name || null });
    }
  }
  return out;
}

const WEBHOOK_NODE = /\.(webhook|formTrigger|chatTrigger|mcpTrigger)$/;

export function webhooks(wf) {
  return (wf.nodes || []).filter((n) => WEBHOOK_NODE.test(n.type) && !n.disabled).map((n) => {
    const p = n.parameters || {};
    const kind = n.type.split('.').pop();
    const method = kind === 'webhook' ? (p.multipleMethods ? (p.httpMethod || ['GET', 'POST']) : (p.httpMethod || 'GET')) : kind === 'formTrigger' ? ['GET', 'POST'] : 'POST';
    let path = p.path || n.webhookId || '';
    if (kind === 'formTrigger' || kind === 'chatTrigger' || kind === 'mcpTrigger') path = p.path || n.webhookId;
    return {
      node: n.name,
      kind,
      path: String(path || '').replace(/^\/+/, ''),
      methods: Array.isArray(method) ? method : [method],
      auth: p.authentication || 'none',
      responseMode: p.responseMode || 'onReceived',
      webhookId: n.webhookId,
      prefix: kind === 'formTrigger' ? 'form' : kind === 'mcpTrigger' ? 'mcp' : 'webhook',
    };
  });
}

/** Human summary of a Schedule Trigger / Cron node, or null. */
export function scheduleSummary(node) {
  const p = node?.parameters || {};
  const t = String(node?.type || '');
  if (/scheduleTrigger$/.test(t)) {
    const rules = p.rule?.interval || [{ field: 'days' }];
    return rules.map((r) => {
      const f = r.field || 'days';
      const at = () => `at ${String(r.triggerAtHour ?? 0).padStart(2, '0')}:${String(r.triggerAtMinute ?? 0).padStart(2, '0')}`;
      switch (f) {
        case 'seconds': return `every ${r.secondsInterval || 30}s`;
        case 'minutes': return `every ${r.minutesInterval || 5} min`;
        case 'hours': return `every ${r.hoursInterval || 1}h at :${String(r.triggerAtMinute ?? 0).padStart(2, '0')}`;
        case 'days': return `every ${(r.daysInterval || 1) === 1 ? 'day' : `${r.daysInterval} days`} ${at()}`;
        case 'weeks': return `weekly on ${weekdays(r.triggerAtDay)} ${at()}`;
        case 'months': return `monthly on day ${r.triggerAtDayOfMonth || 1} ${at()}`;
        case 'cronExpression': return `cron ${r.expression || '?'}`;
        default: return f;
      }
    }).join('; ');
  }
  if (/\.cron$/.test(t)) return 'cron (legacy node)';
  if (/\.interval$/.test(t)) return `every ${p.interval || 1} ${p.unit || 'seconds'}`;
  return null;
}

function weekdays(days) {
  const names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const list = Array.isArray(days) ? days : days != null ? [days] : [0];
  return list.map((d) => names[Number(d) % 7] || d).join(', ');
}

// ------------------------------------------------------------------ lint

const SECRET_RE = /\b(sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b/;

/**
 * Static checks. ctx: { catalog (Map type->entry), credentials (Map id->cred), workflowsById }.
 * Returns [{level: 'error'|'warning'|'info', code, node, message}].
 */
export function lint(wf, ctx = {}) {
  const issues = [];
  const add = (level, code, node, message) => issues.push({ level, code, node, message });
  const nodes = (wf.nodes || []).filter((n) => !isSticky(n));
  const names = new Set(nodes.map((n) => n.name));
  const list = edges(wf);
  const trig = triggers(wf, ctx.catalog);
  if (!nodes.length) return issues;
  if (!trig.length) add('warning', 'no-trigger', null, 'No trigger node: this workflow can only run as a sub-workflow or manually.');

  // reachability from triggers
  const reach = new Set();
  const stack = trig.map((t) => t.name);
  while (stack.length) {
    const n = stack.pop();
    if (reach.has(n)) continue;
    reach.add(n);
    for (const e of list) if (e.from === n) stack.push(e.to);
  }
  for (const e of list) if (e.fromType !== 'main' && reach.has(e.to)) reach.add(e.from); // AI sub-nodes
  const subSources = new Set(list.filter((e) => e.fromType !== 'main').map((e) => e.from));

  const seenPaths = new Map();
  for (const n of nodes) {
    const entry = ctx.catalog?.get?.(n.type);
    if (ctx.catalog && ctx.catalog.size && !entry) add('error', 'unknown-type', n.name, `Unknown node type ${n.type} (missing community package?).`);
    const connected = list.some((e) => e.from === n.name || e.to === n.name);
    if (!connected && nodes.length > 1) add('warning', 'disconnected', n.name, 'Not connected to anything.');
    else if (trig.length && !reach.has(n.name) && !subSources.has(n.name) && !trig.includes(n)) add('warning', 'unreachable', n.name, 'Not reachable from any trigger.');
    if (n.disabled) add('info', 'disabled', n.name, 'Disabled — it will pass data through untouched.');
    for (const ref of expressionRefs(n.parameters || {})) {
      if (!names.has(ref)) add('error', 'broken-ref', n.name, `Expression references a node that does not exist: "${ref}".`);
    }
    for (const c of credentialRefs({ nodes: [n] })) {
      if (!c.id) add('error', 'credential-missing', n.name, `Credential "${c.type}" is not selected.`);
      else if (ctx.credentials && ctx.credentials.size && !ctx.credentials.has(c.id)) add('error', 'credential-missing', n.name, `Credential "${c.name || c.id}" (${c.type}) does not exist on this instance.`);
    }
    if (entry?.credentials?.length && !Object.keys(n.credentials || {}).length && !/httpRequest$/.test(n.type)) {
      add('info', 'no-credential', n.name, 'Node has credential types available but none selected.');
    }
    const flat = JSON.stringify(n.parameters || {});
    const sec = SECRET_RE.exec(flat);
    if (sec) add('error', 'hardcoded-secret', n.name, `Looks like a hard-coded secret (${sec[1].slice(0, 6)}…). Move it into a credential.`);
    if (/"=?\s*https?:\/\/[^"]*(?:token|apikey|api_key|password)=[^"&]{6,}/i.test(flat)) add('warning', 'secret-in-url', n.name, 'URL contains a token/password query parameter.');
    if (n.continueOnFail || n.onError === 'continueRegularOutput') add('info', 'continue-on-fail', n.name, 'Errors are swallowed (continue on fail).');
  }
  for (const w of webhooks(wf)) {
    const key = `${w.methods.join(',')} ${w.path}`;
    if (seenPaths.has(key)) add('error', 'duplicate-webhook', w.node, `Webhook path "${w.path}" is also used by "${seenPaths.get(key)}".`);
    seenPaths.set(key, w.node);
    if (w.kind === 'webhook' && w.auth === 'none') add('warning', 'open-webhook', w.node, 'Webhook has no authentication — anyone with the URL can trigger it.');
    if (w.kind === 'webhook' && w.responseMode === 'responseNode' && !nodes.some((n) => /respondToWebhook$/.test(n.type))) {
      add('error', 'no-respond-node', w.node, 'Response mode is "Respond to Webhook node" but there is no such node.');
    }
  }
  const s = wf.settings || {};
  if (trig.length && !s.errorWorkflow && wf.active) add('info', 'no-error-workflow', null, 'Active workflow without an error workflow — failures will go unnoticed.');
  if (s.errorWorkflow && ctx.workflowsById && !ctx.workflowsById.has(s.errorWorkflow)) add('error', 'error-workflow-missing', null, 'The configured error workflow no longer exists.');
  const order = { error: 0, warning: 1, info: 2 };
  return issues.sort((a, b) => order[a.level] - order[b.level]);
}

// ------------------------------------------------------------------ diff

/** Node-level diff between two workflow versions (nodes matched by id, then name). */
export function diffWorkflows(a, b) {
  const A = a?.nodes || [];
  const B = b?.nodes || [];
  const key = (n) => n.id || n.name;
  const bByKey = new Map(B.map((n) => [key(n), n]));
  const bByName = new Map(B.map((n) => [n.name, n]));
  const matched = new Set();
  const added = [];
  const removed = [];
  const changed = [];
  for (const n of A) {
    const m = bByKey.get(key(n)) || bByName.get(n.name);
    if (!m) { removed.push(n.name); continue; }
    matched.add(m);
    const changes = deepDiff(stripNode(n), stripNode(m));
    const moved = JSON.stringify(n.position) !== JSON.stringify(m.position);
    if (changes.length || n.name !== m.name || moved) {
      changed.push({ name: m.name, from: n.name, renamed: n.name !== m.name, moved, changes });
    }
  }
  for (const n of B) if (!matched.has(n)) added.push(n.name);
  const ea = new Set(edges(a || {}).map(edgeKey));
  const eb = new Set(edges(b || {}).map(edgeKey));
  const decode = (k) => { const [from, fromType, fromIndex, to] = k.split('\u0000'); return { from, fromType, fromIndex: Number(fromIndex), to }; };
  return {
    name: a?.name !== b?.name ? { from: a?.name, to: b?.name } : null,
    added, removed, changed,
    connections: { added: [...eb].filter((k) => !ea.has(k)).map(decode), removed: [...ea].filter((k) => !eb.has(k)).map(decode) },
    settings: deepDiff(a?.settings || {}, b?.settings || {}),
    get empty() {
      return !this.name && !added.length && !removed.length && !changed.some((c) => c.changes.length || c.renamed)
        && !this.connections.added.length && !this.connections.removed.length && !this.settings.length;
    },
  };
}

function stripNode(n) {
  const { position, id, ...rest } = n;
  return rest;
}

// ------------------------------------------------------------------ search

/** Lower-cased text blob of everything searchable in a workflow. */
export function searchText(wf) {
  const parts = [wf.name, wf.description, wf.id];
  for (const t of wf.tags || []) parts.push(t.name);
  for (const n of wf.nodes || []) {
    parts.push(n.name, n.type, n.notes);
    if (n.webhookId) parts.push(n.webhookId);
    mapStrings(n.parameters || {}, (s) => { parts.push(s); return s; });
    for (const [type, ref] of Object.entries(n.credentials || {})) parts.push(type, ref?.name, ref?.id);
  }
  return parts.filter(Boolean).join('\n').toLowerCase();
}

/** Parse "foo type:slack cred:abc tag:x node:Set -bar \"exact phrase\"" into a matcher. */
export function parseQuery(q) {
  const terms = [];
  const re = /(-)?(?:(\w+):)?(?:"([^"]*)"|(\S+))/g;
  for (const m of String(q || '').matchAll(re)) {
    const value = (m[3] ?? m[4] ?? '').toLowerCase();
    if (!value) continue;
    terms.push({ neg: !!m[1], field: m[2]?.toLowerCase() || null, value });
  }
  return terms;
}

export function matchWorkflow(wf, terms, blob = null) {
  if (!terms.length) return true;
  let text = blob;
  for (const t of terms) {
    let hit;
    switch (t.field) {
      case 'type': hit = (wf.nodes || []).some((n) => n.type.toLowerCase().includes(t.value)); break;
      case 'node': hit = (wf.nodes || []).some((n) => n.name.toLowerCase().includes(t.value)); break;
      case 'tag': hit = (wf.tags || []).some((x) => x.name.toLowerCase().includes(t.value)); break;
      case 'cred': hit = credentialRefs(wf).some((c) => [c.id, c.name, c.type].some((v) => v && String(v).toLowerCase().includes(t.value))); break;
      case 'id': hit = String(wf.id).toLowerCase() === t.value; break;
      case 'name': hit = (wf.name || '').toLowerCase().includes(t.value); break;
      case 'is':
        hit = t.value === 'active' ? !!wf.active : t.value === 'inactive' ? !wf.active : t.value === 'archived' ? !!wf.isArchived : false;
        break;
      default:
        text ??= searchText(wf);
        hit = text.includes(t.value);
    }
    if (t.neg ? hit : !hit) return false;
  }
  return true;
}

// ------------------------------------------------------------------ docs, export

export function toMermaid(wf) {
  const id = new Map();
  const nodes = (wf.nodes || []).filter((n) => !isSticky(n));
  nodes.forEach((n, i) => id.set(n.name, `n${i}`));
  const esc = (s) => String(s).replace(/"/g, '#quot;');
  const lines = ['flowchart LR'];
  for (const n of nodes) lines.push(`  ${id.get(n.name)}["${esc(n.name)}"]`);
  for (const e of edges(wf)) {
    if (!id.has(e.from) || !id.has(e.to)) continue;
    lines.push(e.fromType === 'main' ? `  ${id.get(e.from)} --> ${id.get(e.to)}` : `  ${id.get(e.from)} -.${e.fromType.replace('ai_', '')}.-> ${id.get(e.to)}`);
  }
  return lines.join('\n');
}

/** Markdown documentation generated from the workflow JSON. */
export function workflowDocs(wf, catalog) {
  const nodes = (wf.nodes || []).filter((n) => !isSticky(n));
  const notes = (wf.nodes || []).filter(isSticky).map((n) => n.parameters?.content).filter(Boolean);
  const display = (t) => catalog?.get?.(t)?.displayName || shortType(t);
  const lines = [`# ${wf.name}`, ''];
  if (wf.description) lines.push(wf.description, '');
  lines.push(`- **ID:** ${wf.id || '(unsaved)'}`, `- **Status:** ${wf.active ? 'active' : 'inactive'}`, `- **Nodes:** ${nodes.length}`);
  if (wf.tags?.length) lines.push(`- **Tags:** ${wf.tags.map((t) => t.name).join(', ')}`);
  if (wf.updatedAt) lines.push(`- **Updated:** ${wf.updatedAt}`);
  lines.push('', '## Triggers', '');
  const trig = triggers(wf, catalog);
  if (!trig.length) lines.push('_No trigger — runs as a sub-workflow or manually._');
  for (const t of trig) {
    const sched = scheduleSummary(t);
    lines.push(`- **${t.name}** (${display(t.type)})${sched ? ` — ${sched}` : ''}`);
  }
  const hooks = webhooks(wf);
  if (hooks.length) {
    lines.push('', '## Webhooks', '', '| Node | Method | Path | Auth |', '|---|---|---|---|');
    for (const w of hooks) lines.push(`| ${w.node} | ${w.methods.join(', ')} | \`/${w.prefix}/${w.path}\` | ${w.auth} |`);
  }
  const creds = credentialRefs(wf);
  if (creds.length) {
    lines.push('', '## Credentials used', '');
    const uniq = new Map();
    for (const c of creds) uniq.set(`${c.type}:${c.id}`, c);
    for (const c of uniq.values()) lines.push(`- ${c.name || c.id || '(not set)'} — \`${c.type}\``);
  }
  lines.push('', '## Steps', '');
  const order = topoOrder(wf);
  order.forEach((name, i) => {
    const n = nodes.find((x) => x.name === name);
    if (!n) return;
    const next = children(wf, name);
    lines.push(`${i + 1}. **${n.name}** — ${display(n.type)}${n.disabled ? ' _(disabled)_' : ''}${next.length ? ` → ${next.join(', ')}` : ''}`);
    if (n.notes) lines.push(`   > ${String(n.notes).replace(/\n/g, ' ')}`);
  });
  if (notes.length) {
    lines.push('', '## Notes', '');
    for (const note of notes) lines.push(note, '');
  }
  lines.push('', '## Diagram', '', '```mermaid', toMermaid(wf), '```');
  return lines.join('\n');
}

/** Node names in execution-ish order (BFS from triggers, then the rest). */
export function topoOrder(wf) {
  const nodes = (wf.nodes || []).filter((n) => !isSticky(n));
  const list = edges(wf);
  const roots = nodes.filter((n) => !list.some((e) => e.to === n.name && e.fromType === 'main'));
  const seen = new Set();
  const out = [];
  const queue = roots.sort((a, b) => (a.position?.[1] ?? 0) - (b.position?.[1] ?? 0)).map((n) => n.name);
  while (queue.length) {
    const n = queue.shift();
    if (seen.has(n)) continue;
    seen.add(n);
    out.push(n);
    for (const e of list) if (e.from === n && !seen.has(e.to)) queue.push(e.to);
  }
  for (const n of nodes) if (!seen.has(n.name)) out.push(n.name);
  return out;
}

// ------------------------------------------------------------------ API bodies

/** Body accepted by PUT/POST /workflows (the public API rejects read-only fields). */
export function saveBody(wf) {
  const settings = {};
  for (const k of SETTINGS_KEYS) if (wf.settings && wf.settings[k] !== undefined && wf.settings[k] !== null) settings[k] = wf.settings[k];
  const body = { name: wf.name || 'My workflow', nodes: wf.nodes || [], connections: wf.connections || {}, settings };
  if (wf.pinData && Object.keys(wf.pinData).length) body.pinData = wf.pinData;
  else if (wf.pinData) body.pinData = {};
  if (wf.staticData !== undefined) body.staticData = wf.staticData;
  if (wf.description !== undefined && wf.description !== null) body.description = wf.description;
  return body;
}

/** Body for duplicating/importing: fresh node ids and webhook ids, no ids/versions. */
export function importBody(wf, { name, freshWebhooks = true } = {}) {
  const nodes = (wf.nodes || []).map((n) => ({
    ...clone(n),
    id: uuid(),
    ...(freshWebhooks && (n.webhookId || WEBHOOK_NODE.test(n.type)) ? { webhookId: uuid() } : {}),
  }));
  return saveBody({ ...wf, name: name || wf.name, nodes, staticData: undefined });
}

export function newWorkflow(name = 'My workflow') {
  return { name, nodes: [], connections: {}, settings: { executionOrder: 'v1' }, pinData: {} };
}

export function stats(wf) {
  const nodes = (wf.nodes || []).filter((n) => !isSticky(n));
  const types = new Map();
  for (const n of nodes) types.set(n.type, (types.get(n.type) || 0) + 1);
  return { nodes: nodes.length, stickies: (wf.nodes || []).length - nodes.length, edges: edges(wf).length, types, disabled: nodes.filter((n) => n.disabled).length };
}
