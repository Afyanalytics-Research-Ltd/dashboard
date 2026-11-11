// System status: instance health, API key, optional login, node catalog (with
// refresh), configuration and the latency of recent API calls.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, kv, dataTable, toast, spinner } from '../../../ui/js/ui.js';
import { studio, requestLog, config, n8nLink } from '../core/api.js';
import { loadCatalog, catalog } from '../core/catalog.js';
import { percentile } from '../core/execution.js';

export default async function render(ctx) {
  const healthEl = h('div', spinner());
  const latEl = h('div');
  mount(ctx.el, pageHeader({ title: 'System status', icon: 'heart-pulse', actions: [h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => go() }, icon('arrow-clockwise', 'me-1'), 'Re-check')] }),
    h('div.n8s-grid-2', card({ title: 'Connection', icon: 'plug', body: healthEl }), card({ title: 'Recent API calls (this tab)', icon: 'speedometer', body: latEl })));

  async function go() {
    mount(healthEl, spinner());
    try {
      const st = await studio.get('/health/');
      const ok = (v) => h('span', { class: v ? 'text-success' : 'text-danger' }, icon(v ? 'check-circle-fill' : 'x-circle-fill', 'me-1'));
      mount(healthEl, kv([
        ['n8n instance', h('span', ok(st.instance?.ok), st.instance?.ok ? 'healthy' : st.instance?.error || `HTTP ${st.instance?.status}`)],
        ['Server-side URL', h('code', st.apiUrl)],
        ['Browser URL', h('a', { href: n8nLink('/'), target: '_blank', rel: 'noopener' }, st.publicUrl)],
        ['API key', h('span', ok(st.key?.ok), st.key?.ok ? `valid (${st.key.ms} ms)` : st.key?.configured ? st.key.error || `HTTP ${st.key.status}` : 'N8N_API_KEY not set')],
        ['Manual runs (login)', h('span', ok(st.login?.ok), st.login?.configured ? (st.login.ok ? 'logged in' : st.login.error) : 'not configured — set N8N_STUDIO_EMAIL / N8N_STUDIO_PASSWORD')],
        ['Node catalog', h('span', `${st.catalog?.nodes} node types (${st.catalog?.source})`, ' ', h('button.af-link-btn', { onclick: async () => { await loadCatalog({ refresh: true }); toast(`Catalog reloaded: ${catalog.nodes.size} types (${catalog.source})`, 'success'); go(); } }, 'reload'))],
        ['API coverage', `${st.operations} operations from spec ${st.specVersion}`],
        ['Signed in as', config.user],
      ], { cols: 1 }));
    } catch (err) { mount(healthEl, h('div.text-danger', err.message)); }
    drawLatency();
  }

  function drawLatency() {
    const rows = requestLog.slice(0, 60);
    const ms = rows.map((r) => r.ms);
    mount(latEl, rows.length ? h('div',
      h('div.small.mb-2', `p50 ${Math.round(percentile(ms, 50))} ms · p95 ${Math.round(percentile(ms, 95))} ms · ${rows.filter((r) => r.status >= 400).length} errors`),
      dataTable({
        rows, dense: true, rowKey: (r) => `${r.at}${r.path}${r.method}`,
        columns: [
          { key: 'method', label: '', render: (r) => h('span.n8s-verb', { class: r.method.toLowerCase() }, r.method) },
          { key: 'path', label: 'Path', render: (r) => h('code.small', r.path.slice(0, 70)) },
          { key: 'status', label: 'Status', render: (r) => h('span', { class: r.status >= 400 ? 'text-danger' : 'text-success' }, r.status) },
          { key: 'ms', label: 'ms', align: 'right', render: (r) => Math.round(r.ms) },
        ],
      })) : h('div.small.text-muted', 'No calls yet.'));
  }

  ctx.every(5000, drawLatency);
  go();
}
