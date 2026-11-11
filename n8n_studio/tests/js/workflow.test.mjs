// node --test n8n_studio/tests/js
import test from 'node:test';
import assert from 'node:assert/strict';
import * as W from '../../../static/n8n_studio/js/core/workflow.js';

const wf = () => ({
  id: 'w1',
  name: 'Orders',
  nodes: [
    { id: '1', name: 'Webhook', type: 'n8n-nodes-base.webhook', typeVersion: 2, position: [0, 0], webhookId: 'abc', parameters: { path: 'orders', httpMethod: 'POST' } },
    { id: '2', name: 'Set', type: 'n8n-nodes-base.set', typeVersion: 3, position: [200, 0], parameters: { v: '={{ $json.a }}' } },
    { id: '3', name: 'If', type: 'n8n-nodes-base.if', typeVersion: 2, position: [400, 0], parameters: { x: "={{ $('Set').item.json.v }} and {{ $node[\"Set\"].json.v }} {{ $node.Set.json }}" } },
    { id: '4', name: 'Slack', type: 'n8n-nodes-base.slack', typeVersion: 2, position: [600, -100], parameters: {}, credentials: { slackApi: { id: 'c1', name: 'Slack prod' } } },
    { id: '5', name: 'Note', type: W.STICKY, typeVersion: 1, position: [0, 300], parameters: { content: '## hi', width: 300, height: 200 } },
  ],
  connections: {
    Webhook: { main: [[{ node: 'Set', type: 'main', index: 0 }]] },
    Set: { main: [[{ node: 'If', type: 'main', index: 0 }]] },
    If: { main: [[{ node: 'Slack', type: 'main', index: 0 }], []] },
  },
  settings: { executionOrder: 'v1', foo: 'nope', errorWorkflow: 'x' },
  pinData: { Set: [{ json: { a: 1 } }] },
});

test('edges round-trip through edgesToConnections', () => {
  const w = wf();
  const e = W.edges(w);
  assert.equal(e.length, 3);
  assert.deepEqual(W.edges({ connections: W.edgesToConnections(e) }), e);
});

test('add/remove connection is idempotent and ignores self loops', () => {
  let w = wf();
  const edge = { from: 'Webhook', fromType: 'main', fromIndex: 0, to: 'Slack', toType: 'main', toIndex: 0 };
  w = W.addConnection(w, edge);
  assert.equal(W.edges(w).length, 4);
  w = W.addConnection(w, edge);
  assert.equal(W.edges(w).length, 4);
  assert.equal(W.addConnection(w, { ...edge, to: 'Webhook' }), w);
  w = W.removeConnection(w, edge);
  assert.equal(W.edges(w).length, 3);
});

test('renameNode rewrites connections, pinData and every expression style', () => {
  const w = W.renameNode(wf(), 'Set', 'Prepare data');
  assert.ok(w.nodes.some((n) => n.name === 'Prepare data'));
  assert.ok(w.connections['Prepare data']);
  assert.equal(w.connections.Webhook.main[0][0].node, 'Prepare data');
  assert.ok(w.pinData['Prepare data']);
  const x = w.nodes.find((n) => n.name === 'If').parameters.x;
  assert.match(x, /\$\('Prepare data'\)/);
  assert.match(x, /\$node\["Prepare data"\]/);
  assert.doesNotMatch(x, /\$node\.Set/);
  assert.throws(() => W.renameNode(wf(), 'Set', 'If'), /already exists/);
});

test('renameNode does not touch similarly named nodes', () => {
  const w = wf();
  w.nodes[2].parameters.y = "={{ $('Set1').item }}";
  const r = W.renameNode(w, 'Set', 'S');
  assert.equal(r.nodes.find((n) => n.name === 'If').parameters.y, "={{ $('Set1').item }}");
});

test('expressionRefs finds all reference styles', () => {
  assert.deepEqual(W.expressionRefs(wf().nodes[2].parameters).sort(), ['Set']);
  assert.deepEqual(W.expressionRefs({ a: "{{ $items('A B', 0) }} {{ $(\"C\").first() }}" }).sort(), ['A B', 'C']);
});

test('uniqueName follows n8n numbering', () => {
  assert.equal(W.uniqueName('Set', ['If']), 'Set');
  assert.equal(W.uniqueName('Set', ['Set']), 'Set1');
  assert.equal(W.uniqueName('Set', ['Set', 'Set1']), 'Set2');
  assert.equal(W.uniqueName('Set3', ['Set3']), 'Set1');
});

test('removeNodes drops connections and pins', () => {
  const w = W.removeNodes(wf(), ['Set']);
  assert.equal(w.nodes.length, 4);
  assert.equal(W.edges(w).length, 1);
  assert.equal(w.pinData.Set, undefined);
});

test('clipboard is n8n compatible and paste avoids collisions', () => {
  const w = wf();
  const text = W.toClipboard(w, ['Set', 'If']);
  const clip = W.parseClipboard(text);
  assert.equal(clip.nodes.length, 2);
  assert.equal(W.edges(clip).length, 1);
  assert.deepEqual(Object.keys(clip.pinData), ['Set']);
  const { wf: merged, names } = W.pasteInto(w, clip);
  assert.deepEqual(names, ['Set1', 'If1']);
  assert.equal(merged.nodes.length, 7);
  const ifCopy = merged.nodes.find((n) => n.name === 'If1');
  assert.match(ifCopy.parameters.x, /\$\('Set1'\)/, 'expressions in pasted nodes follow the rename');
  assert.ok(W.edges(merged).some((e) => e.from === 'Set1' && e.to === 'If1'));
  assert.notEqual(ifCopy.id, '3');
  assert.equal(new Set(merged.nodes.map((n) => n.id)).size, 7);
});

test('paste rename never collides with another pasted node', () => {
  const base = { nodes: [{ name: 'Set', type: 't', parameters: {}, position: [0, 0] }], connections: {} };
  const clip = { nodes: [{ name: 'Set', type: 't', parameters: {}, position: [0, 0] }, { name: 'Set1', type: 't', parameters: {}, position: [0, 0] }], connections: {} };
  const { names } = W.pasteInto(base, clip);
  assert.equal(new Set(names).size, 2);
  assert.ok(!names.includes('Set'));
});

test('parseClipboard rejects junk and accepts a single node', () => {
  assert.equal(W.parseClipboard('nope'), null);
  assert.equal(W.parseClipboard('{"a":1}'), null);
  assert.equal(W.parseClipboard(JSON.stringify(wf().nodes[1])).nodes.length, 1);
});

test('pasted webhooks get fresh webhookIds', () => {
  const w = wf();
  const { wf: merged } = W.pasteInto(w, W.parseClipboard(W.toClipboard(w, ['Webhook'])));
  const hooks = merged.nodes.filter((n) => n.type.endsWith('webhook'));
  assert.notEqual(hooks[0].webhookId, hooks[1].webhookId);
});

test('autoLayout ranks left to right and handles cycles', () => {
  const w = W.autoLayout(wf());
  const x = (n) => w.nodes.find((m) => m.name === n).position[0];
  assert.ok(x('Webhook') < x('Set') && x('Set') < x('If') && x('If') < x('Slack'));
  assert.deepEqual(w.nodes.find((n) => n.name === 'Note').position, [0, 300], 'stickies stay put');
  const cyc = W.addConnection(wf(), { from: 'Slack', fromType: 'main', fromIndex: 0, to: 'Set', toType: 'main', toIndex: 0 });
  assert.doesNotThrow(() => W.autoLayout(cyc));
});

test('autoLayout puts AI sub-nodes below their parent', () => {
  const w = {
    nodes: [
      { name: 'Chat', type: '@n8n/n8n-nodes-langchain.chatTrigger', position: [0, 0], parameters: {} },
      { name: 'Agent', type: '@n8n/n8n-nodes-langchain.agent', position: [0, 0], parameters: {} },
      { name: 'Model', type: '@n8n/n8n-nodes-langchain.lmChatOpenAi', position: [0, 0], parameters: {} },
    ],
    connections: { Chat: { main: [[{ node: 'Agent', type: 'main', index: 0 }]] }, Model: { ai_languageModel: [[{ node: 'Agent', type: 'ai_languageModel', index: 0 }]] } },
  };
  const out = W.autoLayout(w);
  const p = (n) => out.nodes.find((m) => m.name === n).position;
  assert.equal(p('Model')[0], p('Agent')[0]);
  assert.ok(p('Model')[1] > p('Agent')[1]);
});

test('triggers and trigger kinds', () => {
  assert.deepEqual(W.triggers(wf()).map((n) => n.name), ['Webhook']);
  assert.deepEqual(W.triggerKinds(wf()), ['webhook']);
  assert.equal(W.isTriggerType('n8n-nodes-base.respondToWebhook'), false);
  assert.equal(W.isTriggerType('n8n-nodes-base.scheduleTrigger'), true);
  assert.equal(W.isTriggerType('x.custom', { trigger: true }), true);
});

test('webhooks describe path, methods and auth', () => {
  const [h] = W.webhooks(wf());
  assert.equal(h.path, 'orders');
  assert.deepEqual(h.methods, ['POST']);
  assert.equal(h.auth, 'none');
});

test('scheduleSummary covers interval rules', () => {
  const n = { type: 'n8n-nodes-base.scheduleTrigger', parameters: { rule: { interval: [{ field: 'minutes', minutesInterval: 15 }, { field: 'weeks', triggerAtDay: [1, 5], triggerAtHour: 9 }] } } };
  assert.equal(W.scheduleSummary(n), 'every 15 min; weekly on Mon, Fri at 09:00');
  assert.equal(W.scheduleSummary({ type: 'x.set' }), null);
});

test('lint finds the classic problems', () => {
  const w = wf();
  w.nodes.push({ id: '9', name: 'Orphan', type: 'n8n-nodes-base.code', position: [0, 600], parameters: { jsCode: 'const k = "sk-abcdefghijklmnopqrstuvwxyz123456"' } });
  w.nodes[2].parameters.z = "={{ $('Gone').item }}";
  w.active = true;
  w.settings = {};
  const codes = W.lint(w, { credentials: new Map([['zzz', {}]]) }).map((i) => i.code);
  for (const c of ['disconnected', 'broken-ref', 'credential-missing', 'hardcoded-secret', 'open-webhook', 'no-error-workflow']) assert.ok(codes.includes(c), c);
  const levels = W.lint(w).map((i) => i.level);
  assert.deepEqual(levels, [...levels].sort((a, b) => ['error', 'warning', 'info'].indexOf(a) - ['error', 'warning', 'info'].indexOf(b)));
});

test('lint flags duplicate webhook paths and missing respond node', () => {
  const w = wf();
  w.nodes.push({ ...w.nodes[0], id: 'x', name: 'Webhook2', parameters: { path: 'orders', httpMethod: 'POST', responseMode: 'responseNode', authentication: 'headerAuth' } });
  const codes = W.lint(w).map((i) => i.code);
  assert.ok(codes.includes('duplicate-webhook'));
  assert.ok(codes.includes('no-respond-node'));
});

test('diffWorkflows reports node, connection and settings changes', () => {
  const a = wf();
  const b = W.renameNode(wf(), 'If', 'Check');
  b.nodes = b.nodes.filter((n) => n.name !== 'Slack');
  b.connections = W.edgesToConnections(W.edges(b).filter((e) => e.to !== 'Slack'));
  b.nodes.push({ id: '10', name: 'Email', type: 'n8n-nodes-base.emailSend', position: [0, 0], parameters: {} });
  b.nodes.find((n) => n.name === 'Set').parameters.v = 'changed';
  b.settings = { ...b.settings, timezone: 'UTC' };
  const d = W.diffWorkflows(a, b);
  assert.deepEqual(d.added, ['Email']);
  assert.deepEqual(d.removed, ['Slack']);
  assert.ok(d.changed.some((c) => c.name === 'Check' && c.renamed));
  assert.ok(d.changed.some((c) => c.name === 'Set' && c.changes.some((x) => x.path.join('.') === 'parameters.v')));
  assert.ok(d.connections.removed.length >= 1);
  assert.ok(d.settings.some((s) => s.path[0] === 'timezone'));
  assert.equal(d.empty, false);
  assert.equal(W.diffWorkflows(a, wf()).empty, true);
});

test('search query language', () => {
  const w = { ...wf(), tags: [{ name: 'billing' }], active: true };
  const m = (q) => W.matchWorkflow(w, W.parseQuery(q));
  assert.ok(m('orders'));
  assert.ok(m('type:slack'));
  assert.ok(m('cred:"slack prod"'));
  assert.ok(m('tag:bill is:active'));
  assert.ok(!m('-type:slack'));
  assert.ok(!m('node:Nope'));
  assert.ok(m('$json.a'), 'searches inside parameters');
  assert.ok(m('id:w1'));
});

test('saveBody strips read-only fields and unknown settings', () => {
  const b = W.saveBody({ ...wf(), active: true, tags: [{ id: 1 }], versionId: 'v' });
  assert.deepEqual(Object.keys(b).sort(), ['connections', 'name', 'nodes', 'pinData', 'settings']);
  assert.deepEqual(b.settings, { executionOrder: 'v1', errorWorkflow: 'x' });
});

test('importBody regenerates node and webhook ids', () => {
  const b = W.importBody(wf(), { name: 'Copy' });
  assert.equal(b.name, 'Copy');
  assert.ok(b.nodes.every((n, i) => n.id !== wf().nodes[i].id));
  assert.notEqual(b.nodes[0].webhookId, 'abc');
});

test('docs and mermaid', () => {
  const md = W.workflowDocs(wf());
  assert.match(md, /^# Orders/);
  assert.match(md, /## Webhooks/);
  assert.match(md, /Slack prod/);
  assert.match(md, /```mermaid/);
  const mer = W.toMermaid(wf());
  assert.match(mer, /flowchart LR/);
  assert.equal((mer.match(/-->/g) || []).length, 3);
});

test('topoOrder starts from triggers', () => {
  assert.deepEqual(W.topoOrder(wf()), ['Webhook', 'Set', 'If', 'Slack']);
});

test('ancestors walks upstream', () => {
  assert.deepEqual(W.ancestors(wf(), 'If'), ['Set', 'Webhook']);
});

test('bounds and stats', () => {
  const b = W.bounds(wf().nodes);
  assert.equal(b.x, 0);
  assert.equal(b.y, -100);
  const s = W.stats(wf());
  assert.equal(s.nodes, 4);
  assert.equal(s.stickies, 1);
});
