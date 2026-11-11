// Report (W&B view of type "runs/report") data helpers and spec handling.
//
// Spec format (W&B v5): { version: 5, blocks: [...slate blocks], width, panelSettings, authors, ref, discussionThreads }
// Blocks we create are standard W&B blocks; Model Studio charts are stored as a markdown-block
// carrying a `studioChart` key (W&B ignores unknown keys, Studio renders it as a live chart).

import { gql, collect, parseJSON } from '../../core/api.js';

export const REPORT_TYPE = 'runs/report';

export async function listReports(entity, project, { signal } = {}) {
  const nodes = await collect((after) => gql(`query Reports($e: String!, $p: String!, $after: String) { project(entityName: $e, name: $p) {
    allViews(first: 100, after: $after, viewType: "${REPORT_TYPE}") { totalCount pageInfo { hasNextPage endCursor }
      edges { node { id name displayName description spec createdAt updatedAt starCount viewCount user { username photoUrl } } } } } }`,
  { e: entity, p: project, after }, { signal }), (d) => d.project?.allViews, { max: 500 });
  return nodes.map((n) => ({ ...n, specObj: parseJSON(n.spec, {}) }))
    .sort((a, b) => new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt));
}

export async function loadReport(id, { signal } = {}) {
  const d = await gql(`query Report($ids: [ID!]!) { views(ids: $ids) { edges { node {
    id name displayName description spec type createdAt updatedAt entityName projectName user { username photoUrl } updatedBy { username } } } } }`,
  { ids: [id] }, { signal });
  const n = d.views?.edges?.[0]?.node;
  if (!n) throw Object.assign(new Error('Report not found'), { status: 404 });
  return { ...n, specObj: parseJSON(n.spec, {}) };
}

export function emptySpec() {
  return { version: 5, panelSettings: {}, blocks: [], width: 'readable', authors: [], discussionThreads: [], ref: {} };
}

/** Create (no id) or update a report. Returns the view {id, displayName}. */
export async function saveReport({ id, entity, project, displayName, description, spec }) {
  const input = { displayName, description: description || '', spec: JSON.stringify(spec || emptySpec()), type: REPORT_TYPE };
  if (id) input.id = id;
  else Object.assign(input, { entityName: entity, projectName: project, name: crypto.randomUUID().replace(/-/g, '').slice(0, 12) });
  const d = await gql(`mutation SaveReport($input: UpsertViewInput!) { upsertView(input: $input) { inserted view { id displayName updatedAt } } }`, { input });
  return d.upsertView.view;
}

export async function deleteReport(id) {
  await gql(`mutation DelReport($input: DeleteViewInput!) { deleteView(input: $input) { success } }`, { input: { id, deleteDrafts: true } });
}

// ------------------------------------------------------------------ blocks <-> editor model

const text = (children) => (children || []).map((c) => (c.text != null ? c.text : text(c.children))).join('');

/** Editor model block: {kind: heading|paragraph|markdown|quote|code|divider|chart|list, text, level, chart:{metrics, filter, limit, title}} */
export function specToBlocks(spec) {
  return (spec.blocks || []).map((b) => {
    if (b.studioChart) return { kind: 'chart', chart: b.studioChart };
    switch (b.type) {
      case 'heading': return { kind: 'heading', level: b.level || 1, text: text(b.children) };
      case 'paragraph': return { kind: 'paragraph', text: text(b.children) };
      case 'markdown-block': return { kind: 'markdown', text: b.content || '' };
      case 'block-quote': return { kind: 'quote', text: text(b.children) };
      case 'callout-block': return { kind: 'quote', text: text(b.children) };
      case 'code-block': return { kind: 'code', text: (b.children || []).map((l) => text(l.children)).join('\n'), language: b.language };
      case 'horizontal-rule': return { kind: 'divider' };
      case 'list': return { kind: 'list', ordered: !!b.ordered, text: (b.children || []).map((li) => text(li.children)).join('\n') };
      case 'panel-grid': return { kind: 'wandb-panels', raw: b };
      default: return { kind: 'unknown', raw: b, text: text(b.children) };
    }
  });
}

export function blocksToSpec(blocks, base = emptySpec()) {
  const empty = [{ text: '' }];
  const out = blocks.map((b) => {
    switch (b.kind) {
      case 'heading': return { type: 'heading', level: b.level || 1, children: [{ text: b.text || '' }] };
      case 'paragraph': return { type: 'paragraph', children: [{ text: b.text || '' }] };
      case 'markdown': return { type: 'markdown-block', content: b.text || '', children: empty };
      case 'quote': return { type: 'block-quote', children: [{ text: b.text || '' }] };
      case 'code': return { type: 'code-block', language: b.language || 'python', children: String(b.text || '').split('\n').map((l) => ({ type: 'code-line', children: [{ text: l }] })) };
      case 'divider': return { type: 'horizontal-rule', children: empty };
      case 'list': return { type: 'list', ordered: !!b.ordered, children: String(b.text || '').split('\n').filter(Boolean).map((l) => ({ type: 'list-item', children: [{ text: l }] })) };
      case 'chart': return { type: 'markdown-block', content: `**Chart:** ${(b.chart?.metrics || []).join(', ')} _(live chart in Model Studio)_`, children: empty, studioChart: b.chart };
      case 'wandb-panels':
      case 'unknown': return b.raw;
      default: return null;
    }
  }).filter(Boolean);
  return { ...emptySpec(), ...base, blocks: out };
}

export function reportSummary(r) {
  const blocks = specToBlocks(r.specObj || {});
  const firstText = blocks.find((b) => ['paragraph', 'markdown', 'quote'].includes(b.kind) && b.text?.trim());
  return {
    blocks: blocks.length,
    charts: blocks.filter((b) => b.kind === 'chart' || b.kind === 'wandb-panels').length,
    preview: firstText ? firstText.text.replace(/[#*_`>]/g, '').slice(0, 160) : '',
  };
}

/** Metrics referenced by a W&B panel-grid block (line plots), plus its run-set project. */
export function panelGridInfo(raw) {
  const md = raw?.metadata || {};
  const panels = [
    ...(md.panelBankSectionConfig?.panels || []),
    ...((md.panelBankConfig?.sections || []).flatMap((s) => s.panels || [])),
  ];
  const charts = panels.map((p) => ({ type: p.viewType, title: p.config?.chartTitle, metrics: p.config?.metrics || (p.config?.yAxis ? [p.config.yAxis] : []) }));
  const runSet = (md.runSets || [])[0] || {};
  return { charts, project: runSet.project || null };
}
