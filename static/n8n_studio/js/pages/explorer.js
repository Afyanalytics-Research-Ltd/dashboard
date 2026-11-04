// API explorer: every public-API operation (from n8n's own /discover), grouped by
// resource, with a request builder, response viewer, curl export and a live
// coverage table of which operations this browser session has exercised.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, jsonView, jsonEditor, spinner } from '../../../ui/js/ui.js';
import { n8n, requestLog, config } from '../core/api.js';
import { emptyHint } from '../core/widgets.js';

export default async function render(ctx) {
  const left = h('div.n8s-api-list');
  const right = h('div.n8s-api-detail');
  let q = '';
  mount(ctx.el, pageHeader({ title: 'API explorer', icon: 'terminal', subtitle: `All ${config.operations} operations of the n8n public API (spec ${config.specVersion}), proxied with the server-side key.` }),
    h('div.n8s-api', h('div', h('input.form-control.form-control-sm.mb-2', { type: 'search', placeholder: 'Filter operations…', oninput: (e) => { q = e.target.value.toLowerCase(); drawList(); } }), left), right));

  mount(left, spinner());
  const disc = await n8n.get('/discover', { include: 'schemas' }, { signal: ctx.signal }).catch(() => n8n.get('/discover'));
  const resources = disc.data?.resources || disc.resources || {};
  const ops = Object.entries(resources).flatMap(([res, r]) => (r.endpoints || []).map((ep) => ({ res, ...ep, schema: ep.requestSchema || ep.schema || null })));

  const exercised = () => new Set(requestLog.filter((r) => !r.path.startsWith('studio:')).map((r) => `${r.method} ${r.path.split('?')[0].replace(/\/[^/]*\d[^/]*(?=\/|$)/g, '/{id}')}`));

  function drawList() {
    const used = exercised();
    const groups = new Map();
    for (const op of ops) {
      if (q && !`${op.method} ${op.path} ${op.operationId} ${op.res}`.toLowerCase().includes(q)) continue;
      if (!groups.has(op.res)) groups.set(op.res, []);
      groups.get(op.res).push(op);
    }
    mount(left, h('div.small.text-muted.mb-2', `${ops.length} operations · ${[...used].length} distinct calls this session`),
      [...groups.entries()].map(([res, list]) => h('div.n8s-api-group', h('div.n8s-api-res', res),
        list.map((op) => h('button.n8s-api-op', { type: 'button', onclick: () => select(op) }, h('span.n8s-verb', { class: op.method.toLowerCase() }, op.method), h('span.n8s-api-path', op.path.replace('/api/v1', '')))))));
  }

  function select(op) {
    const rel = op.path.replace(/^\/api\/v1/, '');
    const params = [...rel.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
    const inputs = Object.fromEntries(params.map((p) => [p, h('input.form-control.form-control-sm', { placeholder: p })]));
    const query = h('input.form-control.form-control-sm', { placeholder: 'query string, e.g. limit=5&active=true' });
    const body = ['POST', 'PUT', 'PATCH'].includes(op.method) ? jsonEditor({}, { rows: 10 }) : null;
    const out = h('div');
    const build = () => rel.replace(/\{(\w+)\}/g, (_, p) => encodeURIComponent(inputs[p].value || `{${p}}`));
    const send = async () => {
      const missing = params.filter((p) => !inputs[p].value.trim());
      if (missing.length) { mount(out, h('div.text-danger.small', `Fill in ${missing.join(', ')} first.`)); inputs[missing[0]].focus(); return; }
      mount(out, spinner());
      const t0 = performance.now();
      try {
        const res = await n8n.request(op.method, build(), { params: Object.fromEntries(new URLSearchParams(query.value)), body: body ? body.value() ?? {} : undefined });
        mount(out, h('div.small.text-success.mb-1', `${Math.round(performance.now() - t0)} ms`), res == null ? h('div.text-muted', '(empty response)') : typeof res === 'object' ? jsonView(res, { open: 2 }) : h('pre', String(res).slice(0, 5000)));
      } catch (err) {
        mount(out, h('div.text-danger', `HTTP ${err.status}: ${err.message}`), err.body && typeof err.body === 'object' ? jsonView(err.body) : null);
      }
      drawList();
    };
    const curl = () => `curl -X ${op.method} '${(config.n8nUrl || '').replace(/\/$/, '')}/api/v1${build()}${query.value ? `?${query.value}` : ''}' -H 'X-N8N-API-KEY: $N8N_API_KEY'${body ? ` -H 'Content-Type: application/json' -d '${JSON.stringify(safe(body))}'` : ''}`;
    const destructive = op.method === 'DELETE' || /stop|transfer|pull|uninstall|clear/.test(op.path);
    mount(right, card({
      title: h('span', h('span.n8s-verb', { class: op.method.toLowerCase() }, op.method), ' ', h('code', rel)),
      actions: h('div.d-flex.gap-2', h('button.btn.btn-sm.btn-outline-secondary', { onclick: async () => { await navigator.clipboard.writeText(curl()); } }, icon('terminal', 'me-1'), 'curl'),
        h('button.btn.btn-sm', { class: destructive ? 'btn-danger' : 'btn-primary', onclick: send }, icon('send', 'me-1'), destructive ? 'Send (destructive!)' : 'Send')),
      body: h('div',
        h('div.small.text-muted.mb-2', op.operationId),
        params.map((p) => h('div.af-form-row', h('label.form-label.small.fw-semibold', p), inputs[p])),
        h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Query'), query),
        body && h('div.af-form-row', h('label.form-label.small.fw-semibold', 'JSON body'), body.el),
        op.schema && h('details.mb-2', h('summary.small', 'Request schema'), jsonView(op.schema, { open: 3 })),
        out),
    }));
  }

  const safe = (ed) => { try { return ed.value(); } catch { return {}; } };
  drawList();
  mount(right, emptyHint('terminal', 'Pick an operation', 'Requests go through the dashboard proxy (superuser only, audited) — your key never reaches the browser.'));
}
