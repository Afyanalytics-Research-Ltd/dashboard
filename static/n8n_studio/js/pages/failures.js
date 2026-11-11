// Failure inbox: failed executions grouped by workflow + node + normalised error,
// with first/last seen, counts, a sparkline, sample links and one-click retry.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, chips, spinner, timeAgo, toast } from '../../../ui/js/ui.js';
import { sparkline } from '../../../ui/js/charts.js';
import { n8n, enc, pool } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load } from '../core/store.js';
import { groupFailures, mainError, FAILED } from '../core/execution.js';
import { bulkRun, emptyHint, workflowLink } from '../core/widgets.js';

const WINDOWS = { '24h': 86400e3, '7d': 7 * 86400e3, '30d': 30 * 86400e3 };

export default async function render(ctx) {
  let since = ctx.query.get('since') || '7d';
  const workflowId = ctx.query.get('workflowId') || '';
  const body = h('div');
  const progress = h('span.small.text-muted');
  mount(ctx.el,
    pageHeader({
      title: 'Failure inbox', icon: 'inbox',
      subtitle: 'Failed runs grouped by workflow, node and error — so one broken API shows up as one problem, not 400 rows.',
      actions: [h('div.n8s-since'), progress],
    }),
    body);

  const draw = () => mount(ctx.el.querySelector('.n8s-since'), chips(Object.keys(WINDOWS).map((k) => ({ value: k, label: `Last ${k}` })), since, (v) => { since = v || '7d'; ctx.setQuery({ since }); go(); }));

  async function go() {
    draw();
    mount(body, spinner());
    const limit = Date.now() - WINDOWS[since];
    const [workflows] = await Promise.all([load('workflows').catch(() => [])]);
    const failed = [];
    for (const status of ['error', 'crashed']) {
      let cursor;
      for (let i = 0; i < 8; i++) {
        // eslint-disable-next-line no-await-in-loop
        const page = await n8n.get('/executions', { status, limit: 250, cursor, workflowId: workflowId || undefined }, { signal: ctx.signal });
        const items = page.data || [];
        failed.push(...items.filter((e) => new Date(e.startedAt || e.createdAt).getTime() >= limit));
        cursor = page.nextCursor;
        if (!cursor || !items.length || new Date(items[items.length - 1].startedAt || items[items.length - 1].createdAt).getTime() < limit) break;
      }
    }
    if (!failed.length) { mount(body, emptyHint('emoji-sunglasses', 'Inbox zero', `No failed executions in the last ${since}.`)); mount(progress); return; }

    // Load error details for a sample of runs per workflow so we can group by message.
    const errors = new Map();
    const perWf = new Map();
    for (const e of failed) { const list = perWf.get(e.workflowId) || []; if (list.length < 15) list.push(e); perWf.set(e.workflowId, list); }
    const sample = [...perWf.values()].flat().slice(0, 150);
    let done = 0;
    await pool(sample, async (e) => {
      const full = await n8n.get(`/executions/${enc(e.id)}`, { includeData: true }, { signal: ctx.signal });
      const err = mainError(full);
      if (err) errors.set(String(e.id), err);
      mount(progress, `analysed ${++done}/${sample.length} errors`);
    }, { concurrency: 6 });
    mount(progress, `${failed.length} failed runs · ${sample.length} analysed`);

    // Runs without loaded detail inherit the most common error of their workflow.
    const commonByWf = new Map();
    for (const e of sample) { const err = errors.get(String(e.id)); if (err && !commonByWf.has(e.workflowId)) commonByWf.set(e.workflowId, err); }
    for (const e of failed) if (!errors.has(String(e.id)) && commonByWf.has(e.workflowId)) errors.set(String(e.id), { ...commonByWf.get(e.workflowId), inferred: true });

    const groups = groupFailures(failed, errors);
    const byWf = new Map(workflows.map((w) => [String(w.id), w]));
    const span = WINDOWS[since];
    const buckets = 24;
    mount(body, h('div.n8s-inbox', groups.map((g) => {
      const series = Array(buckets).fill(0);
      for (const id of g.ids) {
        const e = failed.find((x) => x.id === id);
        const t = new Date(e.startedAt || e.createdAt).getTime();
        series[Math.min(buckets - 1, Math.floor(((t - (Date.now() - span)) / span) * buckets))]++;
      }
      const wf = byWf.get(String(g.workflowId));
      const retryable = g.ids.map((id) => failed.find((x) => x.id === id)).filter((e) => FAILED.has(e.status) && !e.retrySuccessId);
      return h('div.n8s-inbox-item',
        h('div.n8s-inbox-count', h('b', String(g.count)), h('span', 'runs')),
        h('div.n8s-inbox-main',
          h('div.n8s-inbox-title', g.message || 'Unknown error'),
          h('div.n8s-inbox-meta', workflowLink(wf, { fallbackId: g.workflowId }), g.node && [' · node ', h('b', g.node)], ' · first ', timeAgo(g.first), ' · last ', timeAgo(g.last)),
          h('div.n8s-inbox-samples', g.ids.slice(0, 6).map((id) => h('a.n8s-chip-sm', { href: urls.execution(id) }, `#${id}`)), g.ids.length > 6 && h('span.small.text-muted', ` +${g.ids.length - 6}`))),
        h('div.n8s-inbox-spark', sparkline(series, { width: 120, height: 30, color: '#dc2626' })),
        h('div.n8s-inbox-actions',
          wf && h('a.btn.btn-sm.btn-outline-secondary', { href: urls.workflow(wf.id, { execution: g.ids[0] }) }, icon('diagram-3', 'me-1'), 'Open'),
          retryable.length > 0 && h('button.btn.btn-sm.btn-primary', {
            onclick: async () => {
              const r = await bulkRun({ title: `Retry ${retryable.length} run(s)`, items: retryable, label: (e) => `#${e.id}`, run: (e) => n8n.post(`/executions/${enc(e.id)}/retry`, { loadWorkflow: true }), concurrency: 2 });
              if (r.ok.length) toast(`${r.ok.length} retried`, 'success');
            },
          }, icon('arrow-repeat', 'me-1'), `Retry ${retryable.length}`)));
    })), card({ body: h('div.small.text-muted', icon('info-circle', 'me-1'), 'Grouping normalises numbers, ids, quoted values and URLs in error messages. Retry uses the current saved workflow.') }));
  }

  await go();
}
