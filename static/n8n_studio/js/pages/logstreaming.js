// Log streaming: destinations (webhook / syslog / Sentry), event subscriptions,
// test messages.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, toast, toastError, modal, jsonEditor, card } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { gated, emptyHint, confirmDelete } from '../core/widgets.js';

const TEMPLATES = {
  '$$MessageEventBusDestinationWebhook': { __type: '$$MessageEventBusDestinationWebhook', label: 'Webhook', enabled: true, subscribedEvents: ['n8n.workflow.failed'], url: 'https://example.com/n8n-events', method: 'POST', sendPayload: true },
  '$$MessageEventBusDestinationSyslog': { __type: '$$MessageEventBusDestinationSyslog', label: 'Syslog', enabled: true, subscribedEvents: ['n8n.audit'], host: '127.0.0.1', port: 514, protocol: 'udp', facility: 16, app_name: 'n8n' },
  '$$MessageEventBusDestinationSentry': { __type: '$$MessageEventBusDestinationSentry', label: 'Sentry', enabled: true, subscribedEvents: ['n8n.workflow.failed'], dsn: 'https://key@o0.ingest.sentry.io/0' },
};

export default async function render(ctx) {
  const body = h('div');
  mount(ctx.el, pageHeader({
    title: 'Log streaming', icon: 'broadcast-pin', subtitle: 'Send n8n events (workflow, node, audit, AI…) to external systems.',
    actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => edit() }, icon('plus-lg', 'me-1'), 'Destination')],
  }), body);

  let eventTypes = [];
  const refresh = () => gated(body, async () => {
    const [dest, types] = await Promise.all([n8n.get('/settings/log-streaming/destinations', undefined, { signal: ctx.signal }), n8n.get('/settings/log-streaming/event-types').catch(() => ({ data: [] }))]);
    eventTypes = types.data || types || [];
    const list = dest.data || dest || [];
    return h('div',
      list.length ? h('div.n8s-grid-2', list.map((d) => card({
        title: d.label || d.__type?.replace('$$MessageEventBusDestination', ''), icon: d.enabled ? 'broadcast' : 'pause-circle',
        actions: h('div.btn-group.btn-group-sm',
          h('button.btn.btn-outline-secondary', { title: 'Send test message', onclick: async () => { try { const r = await n8n.post(`/settings/log-streaming/destinations/${enc(d.id)}/test`, {}); toast(r.success ? 'Test message delivered' : 'Test failed', r.success ? 'success' : 'error'); } catch (err) { toastError(err); } } }, icon('send')),
          h('button.btn.btn-outline-secondary', { onclick: () => edit(d) }, icon('pencil')),
          h('button.btn.btn-outline-danger', { onclick: async () => { if (await confirmDelete('destination', d.label || d.id)) { await n8n.del(`/settings/log-streaming/destinations/${enc(d.id)}`).catch(toastError); refresh(); } } }, icon('trash'))),
        body: h('div.small', h('div', h('b', 'Type: '), d.__type?.replace('$$MessageEventBusDestination', '')), d.url && h('div.text-break', h('b', 'URL: '), d.url), d.host && h('div', h('b', 'Host: '), `${d.host}:${d.port}`),
          h('div.mt-1', (d.subscribedEvents || []).map((e) => h('span.n8s-chip-sm', e)))),
      }))) : emptyHint('broadcast-pin', 'No destinations'),
      eventTypes.length ? card({ title: `${eventTypes.length} streamable events`, icon: 'list-check', body: h('div.n8s-pkg-nodes', eventTypes.map((e) => h('span.n8s-chip-sm', e))) }) : null);
  }, { alternative: 'Without the licence, an Error Trigger workflow can forward failures to Slack/email/webhooks.' });

  function edit(d) {
    let tpl = d || TEMPLATES.$$MessageEventBusDestinationWebhook;
    const editor = jsonEditor(tpl, { rows: 16, allowEmpty: false });
    const type = h('select.form-select.form-select-sm', { disabled: !!d, onchange: (e) => { tpl = TEMPLATES[e.target.value]; editor.set(tpl); } }, Object.keys(TEMPLATES).map((k) => h('option', { value: k, selected: tpl.__type === k }, k.replace('$$MessageEventBusDestination', ''))));
    modal({
      title: d ? 'Edit destination' : 'New destination', icon: 'broadcast-pin', size: 'lg',
      body: h('div', h('div.mb-2', type), editor.el, eventTypes.length ? h('details.mt-2', h('summary.small', 'Available event names'), h('div.n8s-pkg-nodes', eventTypes.map((e) => h('span.n8s-chip-sm', e)))) : null),
      actions: [{ label: 'Cancel' }, {
        label: 'Save', variant: 'primary',
        onClick: async (close) => {
          const v = editor.value();
          if (d) await n8n.put(`/settings/log-streaming/destinations/${enc(d.id)}`, v); else await n8n.post('/settings/log-streaming/destinations', v);
          close();
          refresh();
        },
      }],
    });
  }

  refresh();
}
