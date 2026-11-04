// One artifact version: overview (aliases, metadata, how to use it), files with preview,
// incremental lineage graph, usage, and version navigation.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, tabs, kv, emptyState, errorState, skeleton, timeAgo, toast, toastError, jsonView, codeBlock, copyButton,
  formModal, confirmDialog, menuButton, dataTable,
} from '../../../ui/js/ui.js';
import { fmtNumber, fmtDate } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { gql, config } from '../core/api.js';
import { urls } from '../core/urls.js';
import { runStateBadge } from '../core/wb.js';
import {
  loadArtifact, loadArtifactFiles, loadCollection, aliasEditor, userAliases, typeIcon, typeBadge, fmtBytes, ttlText, creatorLink, aliasChip,
} from './artifacts/shared.js';
import { fileBrowser } from './artifacts/files.js';
import { lineageGraph } from './artifacts/lineage.js';
import { linkToRegistry } from './artifacts/registry_link.js';

const TABS = [
  { id: 'overview', label: 'Overview', icon: 'info-circle' },
  { id: 'files', label: 'Files', icon: 'folder2-open' },
  { id: 'lineage', label: 'Lineage', icon: 'diagram-3' },
  { id: 'usage', label: 'Usage', icon: 'people' },
];

export default async function render(ctx) {
  const { entity, project, type, collection, version } = ctx.params;
  const { signal } = ctx;
  let tab = ctx.query.get('tab') || 'overview';
  ctx.setTitle(`${collection}:${version}`);

  const headEl = h('div', skeleton(3));
  const bodyEl = h('div');
  mount(ctx.el, headEl, bodyEl);

  let art;
  let siblings = [];
  let filesCache = null;
  try {
    [art, siblings] = await Promise.all([
      loadArtifact(entity, project, collection, version, { signal }),
      loadCollection(entity, project, collection, { signal }).then((c) => c.versions).catch(() => []),
    ]);
  } catch (err) {
    if (err.name === 'AbortError') return;
    mount(headEl, pageHeader({ title: `${collection}:${version}`, icon: 'box', crumbs: [{ label: project, href: urls.workspace(entity, project) }, { label: 'Artifacts', href: urls.artifacts(entity, project) }, { label: collection, href: urls.collection(entity, project, type, collection) }, { label: version }] }));
    mount(bodyEl, errorState(err));
    return;
  }
  if (!ctx.active) return;
  // Resolve aliases in the URL (e.g. ":latest") to the concrete version for clarity.
  if (version !== art.versionLabel) ctx.setQuery({ resolved: art.versionLabel });

  const holders = new Map();
  for (const v of siblings) for (const a of userAliases(v.aliases, collection)) holders.set(a, v.versionLabel);
  const ordered = [...siblings].sort((a, b) => a.versionIndex - b.versionIndex);
  const pos = ordered.findIndex((v) => v.versionIndex === art.versionIndex);
  const prev = ordered[pos - 1];
  const next = ordered[pos + 1];

  function drawHead() {
    const tabBar = tabs(TABS.map((t) => ({ ...t, count: t.id === 'usage' ? art.usedTotal : t.id === 'files' ? art.fileCount : undefined })), tab, (id) => {
      tab = id;
      ctx.setQuery({ tab: id === 'overview' ? null : id });
      drawBody();
    });
    mount(headEl,
      pageHeader({
        title: h('span', collection, h('span.ma-ver-sep', ':'), h('span.ma-ver', art.versionLabel)),
        crumbs: [{ label: 'Projects', href: urls.projects() }, { label: project, href: urls.workspace(entity, project) }, { label: 'Artifacts', href: urls.artifacts(entity, project, { type }) }, { label: collection, href: urls.collection(entity, project, type, collection) }, { label: art.versionLabel }],
        badges: [typeBadge(type), art.state !== 'COMMITTED' && h('span.ms-pill.ms-pill-warn', art.state.toLowerCase())],
        actions: [
          h('div.btn-group.btn-group-sm', { role: 'group', 'aria-label': 'Versions' },
            h('a.btn.btn-outline-secondary', { href: prev ? urls.artifact(entity, project, type, collection, prev.versionLabel, tab === 'overview' ? null : tab) : undefined, class: !prev && 'disabled', title: 'Previous version' }, icon('chevron-left'), prev ? ` ${prev.versionLabel}` : ''),
            h('span.btn.btn-outline-secondary.disabled.ma-ver-pos', `${pos + 1} / ${ordered.length || 1}`),
            h('a.btn.btn-outline-secondary', { href: next ? urls.artifact(entity, project, type, collection, next.versionLabel, tab === 'overview' ? null : tab) : undefined, class: !next && 'disabled', title: 'Next version' }, next ? `${next.versionLabel} ` : '', icon('chevron-right'))),
          type === 'model' && h('button.btn.btn-sm.btn-outline-primary', { type: 'button', onclick: () => linkToRegistry({ entity, project, collection, artifact: art, onDone: () => ctx.navigate(location.pathname + location.search, { replace: true }) }) }, icon('box-arrow-in-up-right', 'me-1'), 'Link to registry'),
          menuButton(null, () => [
            { label: 'Edit description…', icon: 'card-text', onClick: editDescription },
            { label: 'Edit metadata…', icon: 'braces', onClick: editMetadata },
            { label: 'Set retention (TTL)…', icon: 'hourglass', onClick: editTtl },
            { divider: true },
            { label: `Delete ${art.versionLabel}…`, icon: 'trash', danger: true, onClick: remove },
          ]),
        ],
      }),
      h('div.ma-art-strip', typeIcon(type, 40),
        h('div.ma-art-strip-main',
          aliasEditor({ entity, project, collection, artifact: art, holders, onChange: () => ctx.navigate(location.pathname + location.search, { replace: true }) }),
          h('div.ma-col-facts',
            h('span', icon('hdd', 'me-1'), fmtBytes(art.size)),
            h('span', icon('files', 'me-1'), `${fmtNumber(art.fileCount)} file${art.fileCount === 1 ? '' : 's'}`),
            h('span', icon('person-workspace', 'me-1'), 'by ', creatorLink(art.createdBy, urls)),
            h('span', icon('clock', 'me-1'), timeAgo(art.createdAt)),
            h('span', icon('people', 'me-1'), `used by ${fmtNumber(art.usedTotal)} run${art.usedTotal === 1 ? '' : 's'}`)))),
      tabBar);
  }

  function drawBody() {
    if (tab === 'files') return drawFiles();
    if (tab === 'lineage') return drawLineage();
    if (tab === 'usage') return drawUsage();
    return drawOverview();
  }

  function drawOverview() {
    const ref = `${entity}/${project}/${collection}:${art.versionLabel}`;
    const snippet = `import wandb

run = wandb.init(project="${project}", entity="${entity}")
artifact = run.use_artifact("${ref}", type="${type}")
path = artifact.download()   # local folder with the ${fmtNumber(art.fileCount)} file(s)`;
    const env = `export WANDB_BASE_URL=${config.wandbUrl || 'http://localhost:8090'}`;
    const linked = (art.artifactCollections?.edges || []).map((e) => e.node).filter((c) => c.__typename === 'ArtifactPortfolio');
    mount(bodyEl, h('div.af-grid-main-side',
      h('div.d-flex.flex-column.gap-3',
        card({
          title: 'Details', icon: 'info-circle',
          body: kv([
            ['Reference', h('span.d-flex.align-items-center.gap-2', h('code', ref), copyButton(ref))],
            ['Type', typeBadge(type)],
            ['Version', `${art.versionLabel}${version !== art.versionLabel ? ` (requested as “${version}”)` : ''}`],
            ['Digest', h('span.d-flex.align-items-center.gap-2', h('code.small', art.digest), copyButton(art.digest))],
            ['Size', `${fmtBytes(art.size)} · storage ${fmtBytes(art.storageBytes)}`],
            ['Files', fmtNumber(art.fileCount)],
            ['Created', h('span', fmtDate(art.createdAt), ' · ', timeAgo(art.createdAt))],
            ['Created by', creatorLink(art.createdBy, urls)],
            ['Retention', ttlText(art.ttlDurationSeconds, art.ttlIsInherited)],
            ['History step', art.historyStep ?? '—'],
            ['Description', art.description || h('span.text-muted', 'None'), { wide: true }],
          ]),
        }),
        card({
          title: 'Metadata', icon: 'braces',
          actions: h('button.af-link-btn', { onclick: editMetadata }, icon('pencil', 'me-1'), 'Edit'),
          body: Object.keys(art.meta || {}).length ? jsonView(art.meta, { open: 2 }) : emptyState({ icon: 'braces', title: 'No metadata', message: 'Pass metadata={…} when creating the artifact, or add it here.' }),
        })),
      h('div.d-flex.flex-column.gap-3',
        card({ title: 'Use this artifact', icon: 'code-slash', body: h('div', codeBlock(snippet, { lang: 'python' }), h('div.small.text-muted.mt-2', 'Point the SDK at this server first:'), codeBlock(env, { lang: 'text' })) }),
        card({
          title: 'In the registry', icon: 'box-seam',
          body: linked.length
            ? h('div.af-list', linked.map((p) => h('a.af-list-item', { href: urls.registry({ model: p.name }) }, icon('box-seam'), h('div.af-list-main', h('div.af-list-title', p.name), h('div.af-list-sub', p.project?.name)))))
            : h('div.small.text-muted', type === 'model' ? 'Not linked to a registered model yet.' : 'Only model artifacts are usually registered.'),
        }),
        ordered.length > 1 && card({
          title: 'Other versions', icon: 'layers', flush: true,
          body: h('div.af-list', [...ordered].reverse().slice(0, 12).map((v) => h('a.af-list-item', { href: urls.artifact(entity, project, type, collection, v.versionLabel), class: v.versionIndex === art.versionIndex && 'ma-current' },
            h('span.ma-version', v.versionLabel), h('div.af-list-main', h('div.ma-aliases.compact', userAliases(v.aliases, collection).map((a) => aliasChip(a)))),
            h('span.small.text-muted.af-nowrap', timeAgo(v.createdAt))))),
        }))));
  }

  async function drawFiles() {
    mount(bodyEl, card({ title: 'Files', icon: 'folder2-open', body: skeleton(6) }));
    try {
      filesCache = filesCache || await loadArtifactFiles(art.id, { signal });
      if (!ctx.active || tab !== 'files') return;
      mount(bodyEl, filesCache.length ? fileBrowser(filesCache, { initial: ctx.query.get('file') }) : card({ title: 'Files', body: emptyState({ icon: 'folder', title: 'No files', message: 'This version only holds references or metadata.' }) }));
    } catch (err) { mount(bodyEl, errorState(err, drawFiles)); }
  }

  function drawLineage() {
    const graph = lineageGraph({ artifact: art, entity, project, collection, urls, height: Math.max(420, Math.min(680, window.innerHeight - 300)) });
    mount(bodyEl, card({
      title: 'Lineage', icon: 'diagram-3',
      actions: h('span.small.text-muted', 'Starts from this version — expand nodes one hop at a time instead of drawing the whole project'),
      body: graph.el,
    }));
  }

  function drawUsage() {
    const runs = art.usedRuns;
    mount(bodyEl, card({
      title: `Used by ${fmtNumber(art.usedTotal)} run${art.usedTotal === 1 ? '' : 's'}`, icon: 'people', flush: true,
      body: runs.length ? dataTable({
        rows: runs,
        rowKey: (r) => r.id,
        onRowClick: (r) => navigate(urls.run(r.project?.entityName || entity, r.project?.name || project, r.name)),
        columns: [
          { key: 'displayName', label: 'Run', render: (r) => h('a.ms-run-name', { href: urls.run(r.project?.entityName || entity, r.project?.name || project, r.name) }, r.displayName || r.name) },
          { key: 'state', label: 'State', render: (r) => runStateBadge(r.state) },
          { key: 'jobType', label: 'Job type', render: (r) => r.jobType || h('span.text-muted', '—') },
          { key: 'project', label: 'Project', render: (r) => r.project?.name },
          { key: 'createdAt', label: 'Started', render: (r) => timeAgo(r.createdAt) },
        ],
      }) : emptyState({ icon: 'people', title: 'Not used by any run yet', message: 'Runs that call use_artifact() on this version appear here.' }),
    }));
    if (art.usedTotal > runs.length) bodyEl.appendChild(h('div.small.text-muted.mt-2', `Showing the first ${runs.length}.`));
  }

  // ---------------------------------------------------------------- edits

  const reload = () => ctx.navigate(location.pathname + location.search, { replace: true });

  function editDescription() {
    formModal({
      title: `Describe ${art.versionLabel}`, icon: 'card-text',
      fields: [{ name: 'description', label: 'Description', type: 'textarea', rows: 5, value: art.description || '' }],
      onSubmit: async ({ description }, close) => {
        await gql(`mutation U($input: UpdateArtifactInput!) { updateArtifact(input: $input) { artifact { id description } } }`, { input: { artifactID: art.id, description } });
        close(); toast('Saved', 'success', { timeout: 1500 }); reload();
      },
    });
  }

  function editMetadata() {
    formModal({
      title: `Metadata of ${art.versionLabel}`, icon: 'braces', size: 'lg',
      fields: [{ name: 'metadata', label: 'Metadata (JSON object)', type: 'json', rows: 14, value: art.meta || {} }],
      onSubmit: async ({ metadata }, close) => {
        if (metadata && (typeof metadata !== 'object' || Array.isArray(metadata))) throw new Error('Metadata must be a JSON object.');
        await gql(`mutation U($input: UpdateArtifactInput!) { updateArtifact(input: $input) { artifact { id metadata } } }`, { input: { artifactID: art.id, metadata: JSON.stringify(metadata || {}) } });
        close(); toast('Metadata saved', 'success', { timeout: 1500 }); reload();
      },
    });
  }

  function editTtl() {
    formModal({
      title: 'Retention', icon: 'hourglass',
      fields: [{ name: 'days', label: 'Delete this version after (days)', type: 'number', min: 0, value: art.ttlDurationSeconds > 0 ? Math.round(art.ttlDurationSeconds / 86400) : '', help: 'Leave empty to keep it forever (inherit the default).' }],
      onSubmit: async ({ days }, close) => {
        const ttl = days == null || days === '' ? -2 : Math.round(days * 86400);
        await gql(`mutation U($input: UpdateArtifactInput!) { updateArtifact(input: $input) { artifact { id ttlDurationSeconds } } }`, { input: { artifactID: art.id, ttlDurationSeconds: ttl } });
        close(); toast('Retention updated', 'success', { timeout: 1500 }); reload();
      },
    });
  }

  async function remove() {
    const ok = await confirmDialog({ title: `Delete ${collection}:${art.versionLabel}?`, danger: true, confirmLabel: 'Delete version', message: `${fmtNumber(art.fileCount)} file(s) and aliases (${userAliases(art.aliases, collection).join(', ') || 'none'}) are removed. Runs that used it keep their records.` });
    if (!ok) return;
    try {
      await gql(`mutation D($input: DeleteArtifactInput!) { deleteArtifact(input: $input) { artifact { id } } }`, { input: { artifactID: art.id, deleteAliases: true } });
      toast(`Deleted ${art.versionLabel}`, 'success');
      navigate(urls.collection(entity, project, type, collection));
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  drawHead();
  drawBody();
}
