// Security audit: run n8n's audit with chosen categories, then present each risk
// report with locations linked to workflows / credentials / nodes.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, spinner, toastError } from '../../../ui/js/ui.js';
import { n8n } from '../core/api.js';
import { urls } from '../core/urls.js';
import { emptyHint, licenseGate } from '../core/widgets.js';

const CATEGORIES = ['credentials', 'database', 'nodes', 'filesystem', 'instance'];

export default async function render(ctx) {
  const chosen = new Set(CATEGORIES);
  const days = h('input.form-control.form-control-sm', { type: 'number', min: 1, value: 90, style: 'width:90px' });
  const out = h('div');
  mount(ctx.el, pageHeader({
    title: 'Security audit', icon: 'shield-check', subtitle: 'Unused credentials, unprotected webhooks, risky nodes, SQL injection surface, outdated instance…',
    actions: h('div.d-flex.gap-2.align-items-center.flex-wrap',
      CATEGORIES.map((c) => h('label.n8s-chk', h('input.form-check-input', { type: 'checkbox', checked: true, onchange: (e) => (e.target.checked ? chosen.add(c) : chosen.delete(c)) }), c)),
      h('span.small', 'abandoned after'), days, h('span.small', 'days'),
      h('button.btn.btn-sm.btn-primary', { onclick: () => run() }, icon('play-fill', 'me-1'), 'Run audit')),
  }), out);

  async function run() {
    mount(out, h('div.text-center.p-4', spinner(), h('div.small.text-muted.mt-2', 'Auditing… this scans every workflow and credential.')));
    try {
      const report = await n8n.post('/audit', { additionalOptions: { daysAbandonedWorkflow: Number(days.value) || 90, categories: [...chosen] } });
      const reports = Object.entries(report || {}).filter(([, v]) => v && typeof v === 'object');
      if (!reports.length) { mount(out, emptyHint('shield-check', 'No risks found', 'Nothing to report in the selected categories.')); return; }
      const total = reports.reduce((a, [, r]) => a + (r.sections || []).reduce((b, s) => b + (s.location?.length || 0), 0), 0);
      mount(out,
        h('div.n8s-audit-sum', h('b', `${total} finding${total === 1 ? '' : 's'}`), ' across ', reports.map(([name, r], i) => [i ? ', ' : '', h('span', name.replace(' Risk Report', ''), ` (${(r.sections || []).length})`)])),
        reports.map(([name, r]) => card({
          title: name, icon: riskIcon(r.risk),
          body: h('div', (r.sections || []).map((s) => h('div.n8s-audit-sec',
            h('div.fw-semibold', s.title),
            h('div.small.text-muted', s.description),
            s.recommendation && h('div.n8s-errhint', icon('lightbulb', 'me-1'), s.recommendation),
            s.location?.length ? h('div.n8s-audit-locs', s.location.slice(0, 200).map(loc)) : null,
            s.nextVersions && h('div.small', 'Newer versions: ', s.nextVersions.map((v) => v.name).join(', ')),
            s.settings && h('pre.small', JSON.stringify(s.settings, null, 2))))),
        })));
    } catch (err) {
      mount(out, err.license ? licenseGate(err) : h('div.text-danger', err.message));
      toastError(err, 'Audit failed');
    }
  }

  run();
}

function riskIcon(risk) {
  return { credentials: 'key', database: 'database', nodes: 'boxes', filesystem: 'folder2-open', instance: 'hdd-network' }[risk] || 'shield';
}

function loc(l) {
  if (l.kind === 'credential') return h('a.n8s-loc', { href: urls.credentials({ focus: l.id }) }, icon('key'), l.name);
  if (l.kind === 'node' || l.workflowId) {
    return h('a.n8s-loc', { href: urls.workflow(l.workflowId, l.nodeName ? { find: `node:"${l.nodeName}"` } : undefined) }, icon('diagram-3'), l.workflowName || l.workflowId, l.nodeName && h('span.text-muted', ` › ${l.nodeName}`));
  }
  if (l.kind === 'community' || l.packageUrl) return h('a.n8s-loc', { href: l.packageUrl || '#', target: '_blank', rel: 'noopener' }, icon('box-seam'), l.nodeType || l.name);
  return h('span.n8s-loc', icon('dot'), l.name || l.nodeType || JSON.stringify(l));
}
