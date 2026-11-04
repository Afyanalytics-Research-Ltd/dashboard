// Run page — Artifacts: what this run consumed and produced, with alias chips and links.

import { h, icon, mount } from '../../../../ui/js/dom.js';
import { card, emptyState, errorState, skeleton, timeAgo, copyButton } from '../../../../ui/js/ui.js';
import { gql } from '../../core/api.js';
import { urls } from '../../core/urls.js';
import { fmtBytesShort } from './common.js';

const ART_FIELDS = `id versionIndex size digest createdAt description fileCount
  artifactType { name } artifactSequence { name project { name entityName } } aliases { alias }`;

const TYPE_ICON = { dataset: 'database', model: 'box-seam', run_table: 'table', code: 'code-slash', job: 'gear' };

export async function artifactsTab(el, ctx, run) {
  const { entity, project } = ctx.params;
  mount(el, skeleton(5));
  let d;
  try {
    d = await gql(`query RunArtifacts($e: String!, $p: String!, $r: String!) {
      project(entityName: $e, name: $p) { run(name: $r) {
        inputArtifacts(first: 100) { totalCount edges { usedAs node { ${ART_FIELDS} } } }
        outputArtifacts(first: 100) { totalCount edges { node { ${ART_FIELDS} } } } } } }`,
    { e: entity, p: project, r: run.name }, { signal: ctx.signal });
  } catch (err) {
    if (err.name !== 'AbortError') mount(el, errorState(err, () => artifactsTab(el, ctx, run)));
    return;
  }
  const r = d.project?.run || {};
  const inputs = (r.inputArtifacts?.edges || []).filter((e) => e.node).map((e) => ({ ...e.node, usedAs: e.usedAs }));
  const outputs = (r.outputArtifacts?.edges || []).filter((e) => e.node).map((e) => e.node);
  const section = (title, ic, list, emptyMsg, role) => card({
    title, icon: ic, flush: true,
    actions: h('span.ms-pill.ms-pill-muted', String(list.length)),
    body: list.length ? h('div.msr-art-list', groupByType(list).map(([type, arts]) => [
      h('div.msr-art-type', icon(TYPE_ICON[type] || 'archive'), type, h('span.ms-count', String(arts.length))),
      arts.map((a) => artRow(a, entity, project, role)),
    ])) : emptyState({ icon: ic, title: emptyMsg }),
  });
  mount(el, h('div.msr-art',
    h('div.msr-art-flow',
      h('div.msr-art-node.in', icon('box-arrow-in-right'), h('b', inputs.length), ' consumed'),
      h('div.msr-art-arrow', icon('arrow-right')),
      h('div.msr-art-node.run', icon('play-circle'), h('b', run.displayName || run.name)),
      h('div.msr-art-arrow', icon('arrow-right')),
      h('div.msr-art-node.out', icon('box-arrow-right'), h('b', outputs.length), ' produced')),
    h('div.af-grid-2',
      section('Inputs — used by this run', 'box-arrow-in-right', inputs, 'This run used no artifacts.', 'in'),
      section('Outputs — logged by this run', 'box-arrow-right', outputs, 'This run logged no artifacts.', 'out'))));
}

function groupByType(list) {
  const m = new Map();
  for (const a of list) {
    const t = a.artifactType?.name || 'artifact';
    if (!m.has(t)) m.set(t, []);
    m.get(t).push(a);
  }
  return [...m.entries()].sort(([a], [b]) => a.localeCompare(b));
}

function artRow(a, entity, project, role) {
  const coll = a.artifactSequence?.name || '?';
  const e2 = a.artifactSequence?.project?.entityName || entity;
  const p2 = a.artifactSequence?.project?.name || project;
  const version = `v${a.versionIndex ?? 0}`;
  const href = urls.artifact(e2, p2, a.artifactType?.name || 'artifact', coll, version);
  const aliases = (a.aliases || []).map((x) => x.alias).filter((x) => !/^v\d+$/.test(x));
  return h('a.msr-art-row', { href },
    h('div.msr-art-main',
      h('div.msr-art-name', h('span.ms-mono', coll), h('span.msr-art-ver', `:${version}`)),
      h('div.msr-art-meta',
        aliases.map((al) => h('span.msr-alias', { class: al === 'best' ? 'best' : al === 'latest' && 'latest' }, al)),
        role === 'in' && a.usedAs?.length ? h('span.small.text-muted', `as ${a.usedAs.join(', ')}`) : null,
        h('span.small.text-muted', `${fmtBytesShort(a.size)}${a.fileCount != null ? ` · ${a.fileCount} file${a.fileCount === 1 ? '' : 's'}` : ''}`),
        a.createdAt && h('span.small.text-muted', timeAgo(a.createdAt)),
        p2 !== project && h('span.ms-pill.ms-pill-info', `${e2}/${p2}`))),
    h('span.af-no-row-click', { onclick: (ev) => ev.preventDefault() }, copyButton(`${e2}/${p2}/${coll}:${version}`)),
    icon('chevron-right', 'msr-art-chev'));
}
