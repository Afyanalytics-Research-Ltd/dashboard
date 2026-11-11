// Model registry: every registered model (portfolio) and model collection across your entities,
// with stage slots (production / staging / best / latest), one-click promotion, history and metric trends.

import { h, icon, mount, prefs } from '../../../ui/js/dom.js';
import {
  pageHeader, card, statCard, searchInput, emptyState, errorState, skeleton, timeAgo, toast, toastError, confirmDialog, tabs, chips, drawer, refreshControl,
} from '../../../ui/js/ui.js';
import { fmtNumber } from '../../../ui/js/format.js';
import { gql, parseJSON } from '../core/api.js';
import { urls } from '../core/urls.js';
import { loadViewer, loadProjects } from '../core/data.js';
import { lineChart } from '../core/charts.js';
import { fmtMetric } from '../core/wb.js';
import { addAlias, typeIcon, fmtBytes, creatorLink, aliasEditor } from './artifacts/shared.js';
import { loadPortfolios } from './artifacts/registry_link.js';

const STAGES = [
  { alias: 'production', label: 'Production', cls: 'prod', icon: 'rocket-takeoff' },
  { alias: 'staging', label: 'Staging', cls: 'staging', icon: 'hourglass-split' },
  { alias: 'best', label: 'Best', cls: 'best', icon: 'trophy' },
];
const METRIC_HINT = /val.*acc|accuracy|f1|auc|score|map|bleu|rouge/i;

const VERSIONS = `artifacts(first: 100) { totalCount edges { version node { id versionIndex size createdAt metadata
  aliases { alias artifactCollectionName }
  createdBy { __typename ... on Run { name displayName project { name entityName } } ... on User { username } } } } }`;

async function loadModelCollections(entity, project, signal) {
  const d = await gql(`query M($e: String!, $p: String!) { project(entityName: $e, name: $p) { artifactType(name: "model") {
    artifactCollections(first: 200) { edges { node { __typename id name description createdAt updatedAt ${VERSIONS} } } } } } }`,
  { e: entity, p: project }, { signal });
  return (d.project?.artifactType?.artifactCollections?.edges || []).map((e) => ({ ...e.node, project: { name: project, entityName: entity }, kind: 'collection' }));
}

async function loadPortfolioVersions(p, signal) {
  const d = await gql(`query PV($id: ID!) { artifactCollection(id: $id) { id ${VERSIONS} } }`, { id: p.id }, { signal });
  return { ...p, artifacts: d.artifactCollection?.artifacts, kind: 'portfolio' };
}

function normalise(m) {
  const versions = (m.artifacts?.edges || []).map((e) => ({ ...e.node, versionLabel: e.version, meta: parseJSON(e.node.metadata, {}) }))
    .sort((a, b) => a.versionIndex - b.versionIndex);
  const holders = new Map();
  for (const v of versions) for (const a of v.aliases.filter((x) => !x.artifactCollectionName || x.artifactCollectionName === m.name).map((x) => x.alias)) {
    if (!/^v\d+$/.test(a)) holders.set(a, v);
  }
  const latest = holders.get('latest') || versions[versions.length - 1];
  const metricKey = [...new Set(versions.flatMap((v) => Object.entries(v.meta).filter(([, x]) => typeof x === 'number').map(([k]) => k)))]
    .find((k) => METRIC_HINT.test(k)) || null;
  return { ...m, versions, holders, latest, metricKey, total: m.artifacts?.totalCount ?? versions.length };
}

export default async function render(ctx) {
  const { signal } = ctx;
  let q = ctx.query.get('q') || '';
  let view = ctx.query.get('view') || prefs.get('ms:registry:view', 'all');
  let stageFilter = ctx.query.get('stage') || '';
  const statsEl = h('div.af-stats', skeleton(1, { height: 70 }));
  const gridEl = h('div', skeleton(6));
  const progressEl = h('div.small.text-muted');
  let models = [];

  const refresh = refreshControl(() => load(), { key: 'registry', initial: 0 });
  ctx.cleanup(refresh.stop);

  const viewTabs = tabs([
    { id: 'all', label: 'All models', icon: 'grid' },
    { id: 'registered', label: 'Registered models', icon: 'box-seam' },
    { id: 'collections', label: 'Model collections', icon: 'collection' },
  ], view, (id) => { view = id; prefs.set('ms:registry:view', id); ctx.setQuery({ view: id === 'all' ? null : id }); draw(); });

  mount(ctx.el,
    pageHeader({
      title: 'Model registry', icon: 'box-seam',
      subtitle: 'Every model you have trained or registered, with where production and staging point right now.',
      actions: [refresh.el],
    }),
    statsEl,
    h('div.ma-reg-filters',
      searchInput({ placeholder: 'Search models, projects, aliases…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } }),
      chips([{ value: '', label: 'Any stage' }, ...STAGES.map((s) => ({ value: s.alias, label: s.label }))], stageFilter, (v) => { stageFilter = v; ctx.setQuery({ stage: v }); draw(); }),
      progressEl),
    viewTabs,
    gridEl);

  async function load() {
    try {
      const { viewer, teams } = await loadViewer({ signal });
      const entities = [...new Set([viewer?.entity, ...teams.map((t) => t.name)].filter(Boolean))];
      const projects = (await Promise.all(entities.map((e) => loadProjects(e, { max: 200, signal }).catch(() => [])))).flat();
      mount(progressEl, `Scanning ${projects.length} project${projects.length === 1 ? '' : 's'}…`);
      const portfolios = (await Promise.all(entities.map((e) => loadPortfolios(e, { signal }).catch(() => [])))).flat();
      const regs = (await Promise.all(portfolios.map((p) => loadPortfolioVersions(p, signal).catch(() => ({ ...p, kind: 'portfolio' }))))).map(normalise);
      if (!ctx.active) return;
      models = [...regs];
      if (models.length) { drawStats(); draw(); }
      // Six projects at a time keeps the server responsive for big orgs; render as results arrive.
      for (let i = 0; i < projects.length; i += 6) {
        const batch = await Promise.all(projects.slice(i, i + 6).map((p) => loadModelCollections(p.entityName, p.name, signal).catch(() => [])));
        if (!ctx.active) return;
        models.push(...batch.flat().map(normalise));
        mount(progressEl, i + 6 < projects.length ? `Scanned ${i + 6} of ${projects.length} projects…` : '');
        if (models.length) { drawStats(); draw(); }
      }
      mount(progressEl);
      drawStats();
      draw();
      if (!models.length) mount(gridEl, emptyState({ icon: 'box-seam', title: 'No models yet', message: 'Log an artifact with type="model" (or link one to the registry) and it appears here.' }));
      const open = ctx.query.get('model');
      if (open) {
        const m = models.find((x) => x.name === open);
        if (m) openModel(m);
      }
    } catch (err) {
      if (err.name !== 'AbortError') mount(gridEl, errorState(err, load));
    }
  }

  function drawStats() {
    const inProd = models.filter((m) => m.holders.has('production')).length;
    const versions = models.reduce((a, m) => a + m.total, 0);
    mount(statsEl,
      statCard({ label: 'Models', value: fmtNumber(models.length), icon: 'cpu', tone: 'purple' }),
      statCard({ label: 'Registered', value: fmtNumber(models.filter((m) => m.kind === 'portfolio').length), icon: 'box-seam', tone: 'blue' }),
      statCard({ label: 'In production', value: fmtNumber(inProd), icon: 'rocket-takeoff', tone: inProd ? 'green' : 'slate' }),
      statCard({ label: 'Versions', value: fmtNumber(versions), icon: 'layers', tone: 'teal' }));
  }

  function filtered() {
    const needle = q.toLowerCase();
    return models.filter((m) => (view === 'all' || (view === 'registered' ? m.kind === 'portfolio' : m.kind === 'collection'))
      && (!stageFilter || m.holders.has(stageFilter))
      && (!needle || m.name.toLowerCase().includes(needle) || m.project?.name?.toLowerCase().includes(needle) || [...m.holders.keys()].some((a) => a.includes(needle))))
      .sort((a, b) => (b.holders.has('production') - a.holders.has('production')) || new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt));
  }

  function draw() {
    const rows = filtered();
    if (!models.length) {
      mount(gridEl, emptyState({ icon: 'box-seam', title: 'No models yet', message: 'Log an artifact with type="model" (or link one to the registry) and it appears here.' }));
      return;
    }
    if (!rows.length) { mount(gridEl, emptyState({ icon: 'search', title: 'No model matches' })); return; }
    mount(gridEl, h('div.ma-reg-grid', rows.slice(0, 300).map((m) => {
      const metric = m.metricKey && (m.holders.get('production') || m.latest)?.meta?.[m.metricKey];
      const el = h('button.ma-reg-card', { type: 'button', onclick: () => openModel(m) },
        h('div.ma-reg-head', typeIcon('model', 38), h('div.ma-reg-title',
          h('div.ma-reg-name', m.name),
          h('div.ma-reg-sub', m.kind === 'portfolio' ? h('span', icon('box-seam', 'me-1'), 'registered · ', m.project?.name) : h('span', icon('folder2', 'me-1'), m.project?.entityName, '/', m.project?.name)))),
        h('div.ma-reg-stage', STAGES.map((s) => {
          const v = m.holders.get(s.alias);
          return h('div.ma-reg-slot', { class: [v && 'filled', s.cls] }, h('span.ma-reg-slot-label', icon(s.icon, 'me-1'), s.label), h('span.ma-reg-slot-ver', v ? v.versionLabel : '—'));
        })),
        h('div.ma-reg-metric', h('span', `${fmtNumber(m.total)} version${m.total === 1 ? '' : 's'} · latest ${m.latest ? m.latest.versionLabel : '—'}`),
          m.metricKey ? h('span', `${m.metricKey} `, h('b', fmtMetric(metric))) : h('span', timeAgo(m.updatedAt || m.createdAt))));
      return el;
    })));
  }

  // ---------------------------------------------------------------- detail drawer

  function openModel(m) {
    ctx.setQuery({ model: m.name });
    const body = h('div.d-flex.flex-column.gap-3');
    const d = drawer({
      title: h('span', typeIcon('model', 26), ' ', m.name),
      subtitle: `${m.kind === 'portfolio' ? 'Registered model' : 'Model collection'} · ${m.project?.entityName}/${m.project?.name}`,
      width: 980, body,
      actions: [h('a.btn.btn-sm.btn-outline-secondary', { href: urls.collection(m.project.entityName, m.project.name, 'model', m.name) }, icon('box-arrow-up-right', 'me-1'), 'Open collection')],
      onClose: () => ctx.setQuery({ model: null }),
    });
    const drawDetail = () => {
      const norm = normalise(m);
      Object.assign(m, norm);
      const track = h('div.ma-stage-track', [{ alias: 'latest', label: 'Latest', cls: '', icon: 'clock' }, ...STAGES.slice().reverse().filter((s) => s.alias !== 'best'), STAGES[2]].map((s, i, arr) => {
        const v = m.holders.get(s.alias);
        return [
          h('div.ma-stage', { class: [v && 'on', s.cls] }, h('span.ma-reg-slot-label', icon(s.icon, 'me-1'), s.label),
            h('span.ma-reg-slot-ver', v ? v.versionLabel : '—'), v && h('span.small.text-muted', timeAgo(v.createdAt))),
          i < arr.length - 1 && i < 2 ? h('span.ma-stage-arrow', icon('arrow-right')) : null,
        ];
      }));
      const chartEl = h('div');
      const metricKeys = [...new Set(m.versions.flatMap((v) => Object.entries(v.meta).filter(([, x]) => typeof x === 'number').map(([k]) => k)))];
      let key = m.metricKey || metricKeys[0];
      const keySel = metricKeys.length ? h('select.form-select.form-select-sm.ma-sort', { onchange: (e) => { key = e.target.value; plot(); } }, metricKeys.map((k) => h('option', { value: k, selected: k === key }, k))) : null;
      const plot = () => key && lineChart(chartEl, [{ name: key, runName: key, color: '#7C3AED', points: m.versions.filter((v) => typeof v.meta[key] === 'number').map((v) => ({ x: v.versionIndex, y: v.meta[key] })) }],
        { height: 220, xLabel: 'version', smoothing: { type: 'none', weight: 0 }, bands: false })
        .then((c) => window.Plotly?.relayout(c.el, { 'xaxis.dtick': 1, 'xaxis.tickprefix': 'v' }));
      const best = key ? Math.max(...m.versions.map((v) => (typeof v.meta[key] === 'number' ? v.meta[key] : -Infinity))) : null;
      const aliasProject = m.project?.name;
      const rows = [...m.versions].reverse();
      mount(body,
        card({ title: 'Stages', icon: 'signpost-split', body: track }),
        metricKeys.length > 0 && m.versions.length > 1 && card({ title: 'Metric across versions', icon: 'graph-up', actions: keySel, body: chartEl }),
        card({
          title: 'Versions', icon: 'layers', flush: true,
          body: h('div.af-table-wrap', h('table.af-table.ma-table',
            h('thead', h('tr', h('th', 'Version'), h('th', 'Aliases'), key && h('th.text-end', key), h('th.text-end', 'Size'), h('th', 'From'), h('th', 'Created'), h('th', 'Promote'))),
            h('tbody', rows.map((v) => h('tr',
              h('td', h('a.ma-version', { href: urls.artifact(m.project.entityName, m.project.name, 'model', m.name, v.versionLabel) }, v.versionLabel)),
              h('td', aliasEditor({ entity: m.project.entityName, project: aliasProject, collection: m.name, artifact: v, holders: new Map([...m.holders].map(([a, x]) => [a, x.versionLabel])), compact: true, onChange: () => refreshModel() })),
              key && h('td.text-end.ms-metric', h('span', { class: v.meta[key] === best && m.versions.length > 1 ? 'ms-best' : '' }, fmtMetric(v.meta[key]))),
              h('td.text-end.ms-metric', fmtBytes(v.size)),
              h('td', creatorLink(v.createdBy, urls)),
              h('td.af-nowrap', timeAgo(v.createdAt)),
              h('td.af-nowrap', STAGES.slice(0, 2).map((s) => h('button.btn.btn-sm.btn-outline-secondary.me-1', {
                type: 'button', disabled: m.holders.get(s.alias)?.id === v.id, title: `Point “${s.alias}” at ${v.versionLabel}`,
                onclick: () => promote(v, s.alias),
              }, icon(s.icon, 'me-1'), s.label)))))))),
        }));
      plot();
    };
    async function refreshModel() {
      try {
        const fresh = m.kind === 'portfolio' ? await loadPortfolioVersions(m) : (await loadModelCollections(m.project.entityName, m.project.name)).find((x) => x.name === m.name);
        if (fresh) Object.assign(m, normalise({ ...m, artifacts: fresh.artifacts }));
        if (d.isOpen()) drawDetail();
        draw();
        drawStats();
      } catch (err) { toastError(err, 'Could not refresh'); }
    }
    async function promote(v, alias) {
      const current = m.holders.get(alias);
      const ok = await confirmDialog({
        title: `Promote ${v.versionLabel} to ${alias}?`, confirmLabel: `Promote to ${alias}`, danger: alias === 'production',
        message: current ? `“${alias}” currently points to ${current.versionLabel}. Anything loading ${m.name}:${alias} will get ${v.versionLabel} instead.` : `Anything loading ${m.name}:${alias} will get ${v.versionLabel}.`,
      });
      if (!ok) return;
      try {
        await addAlias({ entity: m.project.entityName, project: m.project.name, collection: m.name, artifactId: v.id, alias });
        toast(`${m.name}:${alias} → ${v.versionLabel}`, 'success', { title: 'Promoted' });
        refreshModel();
      } catch (err) { toastError(err, 'Promotion failed'); }
    }
    drawDetail();
  }

  await load();
}
