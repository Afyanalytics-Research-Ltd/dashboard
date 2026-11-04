// System: W&B Server version & updates, license, ingestion health, live latency, storage, features, deployment notes.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, kv, errorState, skeleton, statCard, timeAgo, refreshControl, dataTable, emptyState, chips } from '../../../ui/js/ui.js';
import { sparkline, tip, tipRows } from '../../../ui/js/charts.js';
import { fmtBytes, fmtNumber } from '../../../ui/js/format.js';
import { gql, config, wandbLink } from '../core/api.js';
import { urls } from '../core/urls.js';
import { adminContext, pill } from './admin/shared.js';

// Latency samples survive navigation within the session.
const latency = [];

export default async function render(ctx) {
  const { el, signal } = ctx;
  const banner = h('div');
  const statsEl = h('div.af-stats', skeleton(1, { height: 70 }));
  const versionEl = h('div.msa-kv-wrap', skeleton(3));
  const licenseEl = h('div.msa-kv-wrap', skeleton(3));
  const healthEl = h('div.msa-kv-wrap', skeleton(2));
  const storageEl = h('div', skeleton(5));
  const featuresEl = h('div', skeleton(4));
  const latencyEl = h('div.msa-latency');
  let featureFilter = 'enabled';
  let features = [];

  const refresh = refreshControl(() => load(), { key: 'ms-system', initial: 15 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'System', icon: 'heart-pulse', subtitle: 'Health, version, license and storage of the self-hosted W&B Server behind Model Studio.',
      actions: [wandbLink('/') && h('a.btn.btn-sm.btn-outline-secondary', { href: wandbLink('/'), target: '_blank', rel: 'noopener' }, icon('box-arrow-up-right', 'me-1'), 'Open W&B'), refresh.el],
    }),
    banner, statsEl,
    h('div.af-grid-2',
      card({ title: 'Version', icon: 'tag', body: versionEl }),
      card({ title: 'License', icon: 'patch-check', body: licenseEl })),
    h('div.af-grid-main-side',
      card({ title: 'Response time (this session)', icon: 'speedometer', body: latencyEl }),
      card({ title: 'Ingestion health', icon: 'activity', body: healthEl })),
    card({ title: 'Storage by team and project', icon: 'hdd-stack', body: storageEl, flush: true }),
    card({
      title: 'Server features', icon: 'toggles', body: featuresEl, flush: true,
      actions: chips([{ value: 'enabled', label: 'Enabled' }, { value: 'disabled', label: 'Disabled' }, { value: 'all', label: 'All' }],
        featureFilter, (v) => { featureFilter = v || 'enabled'; drawFeatures(); }),
    }),
    card({ title: 'Deployment', icon: 'box', body: deploymentNotes() }));

  async function timed(query, vars) {
    const t0 = performance.now();
    const d = await gql(query, vars, { signal, allowPartial: true });
    latency.push({ at: Date.now(), ms: performance.now() - t0 });
    if (latency.length > 120) latency.shift();
    return d;
  }

  async function load() {
    const [info, lic, store] = await Promise.allSettled([
      timed(`query Sys { serverInfo { frontendHost deploymentId hasLocalLicense licenseFlags pollingOK userLimitReached viewOnlyUserLimitReached
        latestLocalVersionInfo { versionOnThisInstanceString latestVersionString outOfDate }
        licenseExpirationUI { bannerDaysRemaining bannerTotalDays renewUrl }
        ingestionHealth { status oldestPendingAgeSeconds observedAt }
        features { name description isEnabled } cliVersionInfo }
        users(first: 500) { edges { node { id deletedAt } } } }`),
      timed('query Lic { localLicenses { id customerName contactEmail createdAt expiresAt notes claims { seats teams storageGigs expiresAt flags } } }'),
      loadStorage(),
    ]);
    if (!ctx.active) return;
    const s = info.status === 'fulfilled' ? info.value?.serverInfo : null;
    if (!s) { mount(versionEl, errorState(info.reason || new Error('No server info'))); mount(statsEl); drawLatency(); return; }
    const userCount = (info.value.users?.edges || []).filter((e) => !e.node.deletedAt).length;
    const v = s.latestLocalVersionInfo || {};
    const licenses = lic.status === 'fulfilled' ? lic.value?.localLicenses || [] : [];
    const active = licenses.find((l) => new Date(l.expiresAt) > new Date()) || licenses[0];

    mount(banner, v.outOfDate
      ? h('div.af-callout.af-callout-warn.msa-banner', icon('arrow-up-circle'), h('span', h('b', `W&B Server ${v.latestVersionString} is available`), ` — this instance runs ${v.versionOnThisInstanceString}. Bump the image tag in docker-compose.wandb.yaml and run `, h('code', 'docker compose -f docker-compose.wandb.yaml up -d'), '.'))
      : null);

    const ing = s.ingestionHealth || {};
    mount(statsEl,
      statCard({ label: 'Server version', value: v.versionOnThisInstanceString || '?', icon: 'tag', tone: v.outOfDate ? 'amber' : 'green', sub: v.outOfDate ? `update: ${v.latestVersionString}` : 'up to date' }),
      statCard({ label: 'Ingestion', value: String(ing.status || 'unknown').toLowerCase(), icon: 'activity', tone: /HEALTHY|OK/i.test(ing.status || '') ? 'green' : 'amber' }),
      statCard({ label: 'Users', value: fmtNumber(userCount), icon: 'people', tone: s.userLimitReached ? 'red' : 'blue', href: urls.users(), sub: s.userLimitReached ? 'seat limit reached' : null }),
      statCard({ label: 'License', value: s.hasLocalLicense ? 'Licensed' : 'Unlicensed', icon: 'patch-check', tone: s.hasLocalLicense ? 'green' : 'slate' }));

    mount(versionEl, kv([
      ['Running', h('code', v.versionOnThisInstanceString || '—')],
      ['Latest release', v.latestVersionString
        ? h('span', h('code', v.latestVersionString), ' ', v.outOfDate ? pill('update available', 'warn', 'arrow-up-circle') : pill('current', 'ok', 'check2'))
        : h('span.text-muted', 'not reported (offline update check)')],
      ['Frontend host', s.frontendHost ? h('code', s.frontendHost) : '—'],
      ['Model Studio connects to', h('code', config.wandbUrl || '—')],
      ['Deployment id', s.deploymentId ? h('code.small', s.deploymentId) : '—'],
      ['Supported wandb SDK', sdkRange(s.cliVersionInfo)],
    ]));

    const exp = s.licenseExpirationUI;
    mount(licenseEl, active
      ? h('div',
        exp && exp.bannerDaysRemaining <= 30 && h('div.af-callout.af-callout-warn.mb-2', icon('calendar-x'), h('span', `License expires in ${exp.bannerDaysRemaining} days.`)),
        kv([
          ['Customer', active.customerName || '—'], ['Contact', active.contactEmail || '—'],
          ['Seats', fmtNumber(active.claims?.seats)], ['Teams', fmtNumber(active.claims?.teams)],
          ['Storage allowance', active.claims?.storageGigs ? `${fmtNumber(active.claims.storageGigs)} GB` : '—'],
          ['Expires', h('span', new Date(active.expiresAt).toLocaleDateString(), ' ', timeAgo(active.expiresAt))],
          ['Flags', h('div.d-flex.flex-wrap.gap-1', (active.claims?.flags || s.licenseFlags || []).map((f) => pill(f, 'info'))), { wide: true }],
        ]))
      : h('div',
        h('div.af-callout.af-callout-info', icon('info-circle'), h('span',
          'No license is installed, so the server runs on the free single-user tier. Get a free personal license at ',
          h('a', { href: 'https://deploy.wandb.ai', target: '_blank', rel: 'noopener' }, 'deploy.wandb.ai'),
          ', put it in .env as ', h('code', 'WANDB_LICENSE'), ' and restart the wandb-local container.')),
        s.licenseFlags?.length ? h('div.mt-2.d-flex.gap-1.flex-wrap', s.licenseFlags.map((f) => pill(f, 'muted'))) : null));

    mount(healthEl, kv([
      ['Status', pill(String(ing.status || 'unknown').toLowerCase(), /HEALTHY|OK/i.test(ing.status || '') ? 'ok' : 'warn', 'activity')],
      ['Oldest pending write', ing.oldestPendingAgeSeconds != null ? `${Number(ing.oldestPendingAgeSeconds).toFixed(1)} s` : 'none'],
      ['Observed', ing.observedAt ? timeAgo(ing.observedAt) : '—'],
      ['Live polling', s.pollingOK ? pill('ok', 'ok') : pill('degraded', 'warn')],
    ], { cols: 1 }));

    features = s.features || [];
    drawFeatures();
    drawLatency();
    if (store.status === 'fulfilled') drawStorage(store.value); else mount(storageEl, errorState(store.reason));
  }

  function drawFeatures() {
    const list = features.filter((f) => (featureFilter === 'all' ? true : featureFilter === 'enabled' ? f.isEnabled : !f.isEnabled))
      .sort((a, b) => a.name.localeCompare(b.name));
    mount(featuresEl, dataTable({
      rows: list, rowKey: (f) => f.name, dense: true,
      empty: emptyState({ icon: 'toggles', title: 'No features in this filter' }),
      columns: [
        { key: 'name', label: 'Feature', render: (f) => h('code', f.name) },
        { key: 'description', label: 'Description', render: (f) => h('span.small', f.description || '—') },
        { key: 'isEnabled', label: '', align: 'right', render: (f) => (f.isEnabled ? pill('on', 'ok', 'check2') : pill('off', 'muted')) },
      ],
    }));
  }

  function drawLatency() {
    if (!latency.length) { mount(latencyEl, h('div.text-muted.small', 'Measuring…')); return; }
    const ms = latency.map((l) => l.ms);
    const avg = ms.reduce((a, b) => a + b, 0) / ms.length;
    const p95 = [...ms].sort((a, b) => a - b)[Math.floor(ms.length * 0.95)] ?? ms[ms.length - 1];
    const last = ms[ms.length - 1];
    const spark = sparkline(ms, { width: 520, height: 64, color: last > 1500 ? '#E5484D' : '#0072CE' });
    spark.setAttribute('preserveAspectRatio', 'none');
    tip(spark, () => tipRows('GraphQL round-trip', [['Samples', ms.length], ['Last', `${last.toFixed(0)} ms`], ['Average', `${avg.toFixed(0)} ms`], ['p95', `${p95.toFixed(0)} ms`]]));
    mount(latencyEl,
      h('div.msa-latency-figs',
        h('div', h('b', `${last.toFixed(0)} ms`), h('span', 'last')),
        h('div', h('b', `${avg.toFixed(0)} ms`), h('span', 'average')),
        h('div', h('b', `${p95.toFixed(0)} ms`), h('span', 'p95'))),
      h('div.msa-latency-chart', spark),
      h('div.small.text-muted', `Measured from your browser through the dashboard proxy, ${ms.length} sample${ms.length === 1 ? '' : 's'}; refreshes with the page.`));
  }

  async function loadStorage() {
    const c = await adminContext({ signal });
    const rows = [];
    for (const e of c.entities) {
      const d = await gql(`query S($e: String!) { entity(name: $e) { name storageBytes
        projects(first: 100) { edges { node { name storageBytes totalRuns } } } } }`, { e: e.name }, { signal, allowPartial: true });
      const ent = d?.entity;
      if (!ent) continue;
      rows.push({ kind: 'entity', name: ent.name, bytes: ent.storageBytes || 0 });
      (ent.projects?.edges || []).forEach((p) => rows.push({ kind: 'project', entity: ent.name, name: p.node.name, bytes: p.node.storageBytes || 0, runs: p.node.totalRuns }));
    }
    return rows;
  }

  function drawStorage(rows) {
    const projects = rows.filter((r) => r.kind === 'project').sort((a, b) => b.bytes - a.bytes);
    const max = Math.max(1, ...projects.map((p) => p.bytes));
    if (!projects.length) { mount(storageEl, emptyState({ icon: 'hdd', title: 'Nothing stored yet' })); return; }
    mount(storageEl, h('div.msa-storage', projects.map((p, i) => h('a.msa-storage-row', { href: urls.workspace(p.entity, p.name), style: { animationDelay: `${Math.min(i, 12) * 0.03}s` } },
      h('div.msa-storage-name', h('span.text-muted', `${p.entity}/`), h('b', p.name)),
      h('div.msa-storage-bar', h('span', { style: { width: `${Math.max(1, (p.bytes / max) * 100)}%` } })),
      h('div.msa-storage-val', fmtBytes(p.bytes)),
      h('div.msa-storage-runs.text-muted', `${fmtNumber(p.runs || 0)} runs`)))));
  }

  function sdkRange(info) {
    const min = info?.min_cli_version || info?.minCliVersion;
    const max = info?.max_cli_version || info?.maxCliVersion;
    if (!min && !max) return 'any recent wandb';
    return h('span', h('code', `wandb ${min ? `≥ ${min}` : ''}${min && max ? ', ' : ''}${max ? `≤ ${max}` : ''}`),
      h('div.small.text-muted', 'pip install "wandb', min ? `>=${min}` : '', max ? `,<=${max}` : '', '"'));
  }

  function deploymentNotes() {
    return h('div.msa-deploy',
      h('p.mb-2', 'W&B Server runs as one container defined in ', h('code', 'docker-compose.wandb.yaml'), ' (MySQL, object storage and the app inside, data in the ', h('code', 'dashboard_wandb-data'), ' volume).'),
      h('ul.msa-deploy-list',
        h('li', icon('shield-lock'), ' Port is bound to ', h('code', '127.0.0.1:8090'), ' only — Model Studio reaches it server-side; the API key never reaches browsers.'),
        h('li', icon('memory'), ' Memory is hard-capped at ', h('code', '3500m'), ' so W&B is the one stopped if the host runs low, never Airflow/Redash/Django.'),
        h('li', icon('cloud-upload'), ' Online: run ', h('code', 'docker compose -f docker-compose.wandb.yaml up -d'), ', set ', h('code', 'WANDB_BASE_URL=http://wandb-local:8080'), ' and ', h('code', 'WANDB_API_KEY'), ' in .env, restart the web container.'),
        h('li', icon('globe'), ' Training jobs elsewhere log via ', h('code', 'WANDB_HOST_URL'), ' — give W&B its own subdomain (it cannot live under a path prefix).')));
  }

  await load();
}
