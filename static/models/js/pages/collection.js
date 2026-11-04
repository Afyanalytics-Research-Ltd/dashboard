// One artifact collection: alias board, versions table with inline alias editing,
// side-by-side version compare, metadata trends, description/tags, link to registry, delete.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, emptyState, errorState, skeleton, timeAgo, toast, toastError, confirmDialog, drawer, formModal,
  menuButton, copyButton, selectionBar, kv,
} from '../../../ui/js/ui.js';
import { fmtNumber, fmtDate } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { gql } from '../core/api.js';
import { urls } from '../core/urls.js';
import { lineChart } from '../core/charts.js';
import { fmtMetric } from '../core/wb.js';
import {
  loadCollection, loadArtifactFiles, aliasEditor, aliasChip, userAliases, typeIcon, typeBadge, fmtBytes, shortDigest,
  metadataDiff, fileDiff, creatorLink,
} from './artifacts/shared.js';
import { linkToRegistry } from './artifacts/registry_link.js';

export default async function render(ctx) {
  const { entity, project, type, collection } = ctx.params;
  const { signal } = ctx;
  ctx.setTitle(`${collection} · Artifacts`);
  const selected = new Set();
  let col;

  const headerEl = h('div');
  const boardEl = h('div', skeleton(2));
  const trendEl = h('div');
  // Bottom room so the last row's menu can scroll clear of the app's floating help buttons.
  const tableEl = h('div.ma-bottom-room', skeleton(8));
  const selEl = h('div');
  mount(ctx.el, headerEl, boardEl, trendEl, tableEl, selEl);

  async function load() {
    try {
      col = await loadCollection(entity, project, collection, { signal });
      if (!ctx.active) return;
      drawHeader();
      drawBoard();
      drawTrend();
      drawTable();
    } catch (err) {
      if (err.name === 'AbortError') return;
      mount(headerEl, pageHeader({ title: collection, icon: 'box', crumbs: [{ label: project, href: urls.workspace(entity, project) }, { label: 'Artifacts', href: urls.artifacts(entity, project) }, { label: collection }] }));
      mount(boardEl, errorState(err, load));
      mount(tableEl);
    }
  }

  // ---------------------------------------------------------------- header

  function drawHeader() {
    const isSeq = col.__typename !== 'ArtifactPortfolio';
    const desc = h('div.ma-desc-edit', { tabindex: 0, title: 'Click to edit the description' },
      col.description || h('span.text-muted.fst-italic', 'Add a description — what is in this collection and how should it be used?'),
      icon('pencil', 'ma-pencil'));
    desc.addEventListener('click', editDescription);
    desc.addEventListener('keydown', (e) => { if (e.key === 'Enter') editDescription(); });
    mount(headerEl,
      pageHeader({
        title: col.name, icon: null,
        crumbs: [{ label: 'Projects', href: urls.projects() }, { label: project, href: urls.workspace(entity, project) }, { label: 'Artifacts', href: urls.artifacts(entity, project, { type }) }, { label: collection }],
        badges: [typeBadge(type), !isSeq && h('span.ms-pill.ms-pill-info', icon('link-45deg'), 'Registry portfolio')],
        actions: [
          type === 'model' && h('button.btn.btn-sm.btn-outline-primary', { type: 'button', onclick: () => col.versions[0] && linkToRegistry({ entity, project, collection, artifact: col.versions[0], onDone: load }) }, icon('box-arrow-in-up-right', 'me-1'), 'Link latest to registry'),
          menuButton(null, () => [
            { label: 'Copy reference', icon: 'clipboard', onClick: () => navigator.clipboard.writeText(`${entity}/${project}/${collection}:latest`).then(() => toast('Copied', 'success', { timeout: 1500 })) },
            { label: 'Rename collection…', icon: 'pencil', onClick: rename, disabled: !isSeq },
            { divider: true },
            { label: 'Delete collection…', icon: 'trash', danger: true, onClick: deleteCollection, disabled: !isSeq },
          ]),
        ],
      }),
      h('div.ma-col-head', typeIcon(type, 44),
        h('div.ma-col-meta',
          desc,
          h('div.ma-col-facts',
            h('span', icon('layers', 'me-1'), `${col.versions.length} version${col.versions.length === 1 ? '' : 's'}`),
            h('span', icon('hdd', 'me-1'), fmtBytes(col.versions.reduce((a, v) => a + (v.size || 0), 0)), ' total'),
            h('span', icon('clock', 'me-1'), 'updated ', timeAgo(col.updatedAt || col.createdAt)),
            h('span.ma-ref', h('code', `${entity}/${project}/${collection}:latest`), copyButton(`${entity}/${project}/${collection}:latest`))),
          tagRow())));
  }

  function tagRow() {
    const el = h('div.ma-tags-edit');
    const render = () => mount(el,
      col.tags.map((t) => h('span.ms-tag', icon('tag'), t.name, h('button.ma-alias-x', { type: 'button', title: `Remove tag ${t.name}`, onclick: () => removeTag(t.name) }, icon('x')))),
      h('button.ma-alias-add', { type: 'button', onclick: (e) => addTag(e.currentTarget) }, icon('plus'), ' tag'));
    async function removeTag(name) {
      try {
        await gql(`mutation DelTag($input: DeleteArtifactCollectionTagAssignmentsInput!) { deleteArtifactCollectionTagAssignments(input: $input) { success } }`,
          { input: { entityName: entity, projectName: project, artifactCollectionName: collection, tags: [{ tagName: name }] } });
        col.tags = col.tags.filter((t) => t.name !== name);
        render();
      } catch (err) { toastError(err, 'Could not remove tag'); }
    }
    function addTag(anchor) {
      const input = h('input.form-control.form-control-sm.ma-alias-input', { placeholder: 'tag', 'aria-label': 'New tag' });
      anchor.replaceWith(input);
      input.focus();
      let done = false;
      const finish = async (save) => {
        if (done) return;
        done = true;
        const v = input.value.trim();
        if (!save || !v || col.tags.some((t) => t.name === v)) { render(); return; }
        try {
          await gql(`mutation AddTag($input: CreateArtifactCollectionTagAssignmentsInput!) { createArtifactCollectionTagAssignments(input: $input) { tags { id name } } }`,
            { input: { entityName: entity, projectName: project, artifactCollectionName: collection, tags: [{ tagName: v }] } });
          col.tags = [...col.tags, { name: v }];
        } catch (err) { toastError(err, 'Could not add tag'); }
        render();
      };
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); finish(true); } if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); } });
      input.addEventListener('blur', () => finish(true));
    }
    render();
    return el;
  }

  function editDescription() {
    formModal({
      title: 'Collection description', icon: 'card-text',
      fields: [{ name: 'description', label: 'Description (Markdown)', type: 'textarea', rows: 6, value: col.description || '' }],
      onSubmit: async ({ description }, close) => {
        await gql(`mutation Desc($input: UpdateArtifactSequenceInput!) { updateArtifactSequence(input: $input) { artifactCollection { id description } } }`,
          { input: { artifactSequenceID: col.id, description } });
        col.description = description;
        close();
        drawHeader();
        toast('Description saved', 'success', { timeout: 2000 });
      },
    });
  }

  function rename() {
    formModal({
      title: 'Rename collection', icon: 'pencil',
      fields: [{ name: 'name', label: 'New name', value: col.name, required: true, help: 'Code that references the old name will stop resolving it.' }],
      onSubmit: async ({ name }, close) => {
        if (!/^[\w.-]+$/.test(name)) throw new Error('Use letters, digits, “-”, “_” and “.” only.');
        await gql(`mutation Ren($input: UpdateArtifactSequenceInput!) { updateArtifactSequence(input: $input) { artifactCollection { id name } } }`,
          { input: { artifactSequenceID: col.id, name } });
        close();
        toast(`Renamed to ${name}`, 'success');
        navigate(urls.collection(entity, project, type, name), { replace: true });
      },
    });
  }

  async function deleteCollection() {
    const ok = await confirmDialog({
      title: `Delete collection “${collection}”?`, danger: true, confirmLabel: 'Delete collection',
      message: `This deletes all ${col.versions.length} version(s) and their files. Runs that used them keep their history but lose the files.`,
    });
    if (!ok) return;
    try {
      await gql(`mutation DelCol($input: DeleteArtifactSequenceInput!) { deleteArtifactSequence(input: $input) { artifactCollection { id } } }`, { input: { artifactSequenceID: col.id } });
      toast(`Deleted ${collection}`, 'success');
      navigate(urls.artifacts(entity, project, { type }));
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  // ---------------------------------------------------------------- alias board

  function holders() {
    const m = new Map();
    for (const v of col.versions) for (const a of userAliases(v.aliases, collection)) m.set(a, v.versionLabel);
    return m;
  }

  function drawBoard() {
    const map = holders();
    const order = ['production', 'prod', 'staging', 'best', 'candidate', 'latest'];
    const aliases = [...map.keys()].sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99) || a.localeCompare(b));
    mount(boardEl, card({
      title: 'Where each alias points', icon: 'signpost-split',
      body: aliases.length
        ? h('div.ma-board', aliases.map((a) => {
          const v = col.versions.find((x) => x.versionLabel === map.get(a));
          return h('a.ma-board-item', { href: urls.artifact(entity, project, type, collection, map.get(a)) },
            aliasChip(a), icon('arrow-right', 'ma-board-arrow'), h('span.ma-board-ver', map.get(a)),
            h('span.ma-board-when', v ? timeAgo(v.createdAt) : ''));
        }))
        : emptyState({ icon: 'signpost', title: 'No aliases yet', message: 'Add one below (e.g. “production”) so code can load a stable name instead of a version number.' }),
    }));
  }

  // ---------------------------------------------------------------- metadata trend

  function drawTrend() {
    const versions = [...col.versions].sort((a, b) => a.versionIndex - b.versionIndex);
    const numericKeys = [...new Set(versions.flatMap((v) => Object.entries(v.meta || {}).filter(([, x]) => typeof x === 'number').map(([k]) => k)))];
    if (versions.length < 2 || !numericKeys.length) { mount(trendEl); return; }
    let key = numericKeys.find((k) => /acc|f1|auc|score|loss/i.test(k)) || numericKeys[0];
    const chartEl = h('div');
    const sel = h('select.form-select.form-select-sm.ma-sort', { 'aria-label': 'Metadata key', onchange: (e) => { key = e.target.value; draw(); } },
      numericKeys.map((k) => h('option', { value: k, selected: k === key }, k)));
    const draw = () => lineChart(chartEl, [{
      name: key, runName: key, color: '#7C3AED',
      points: versions.filter((v) => typeof v.meta?.[key] === 'number').map((v) => ({ x: v.versionIndex, y: v.meta[key] })),
    }], { height: 200, xLabel: 'version', smoothing: { type: 'none', weight: 0 }, bands: false,
      onClick: ({ x }) => navigate(urls.artifact(entity, project, type, collection, `v${x}`)) })
      // Versions are integers: label them v0, v1, … instead of fractional ticks.
      .then((c) => window.Plotly?.relayout(c.el, { 'xaxis.dtick': 1, 'xaxis.tickprefix': 'v' }));
    mount(trendEl, card({ title: 'Metadata across versions', icon: 'graph-up', actions: sel, body: chartEl }));
    draw();
  }

  // ---------------------------------------------------------------- versions table

  function drawTable() {
    const map = holders();
    const rows = [...col.versions].sort((a, b) => b.versionIndex - a.versionIndex);
    if (!rows.length) { mount(tableEl, card({ title: 'Versions', body: emptyState({ icon: 'layers', title: 'No versions' }) })); return; }
    const metaKeys = [...new Set(rows.flatMap((v) => Object.entries(v.meta || {}).filter(([, x]) => typeof x === 'number').map(([k]) => k)))].slice(0, 3);
    const best = Object.fromEntries(metaKeys.map((k) => [k, Math.max(...rows.map((v) => (typeof v.meta?.[k] === 'number' ? v.meta[k] : -Infinity)))]));
    mount(tableEl, card({
      title: 'Versions', icon: 'layers', flush: true,
      actions: h('span.small.text-muted', 'Tick two versions to compare them'),
      body: h('div.af-table-wrap', h('table.af-table.ma-table',
        h('thead', h('tr', h('th.af-col-check'), h('th', 'Version'), h('th', 'Aliases'), metaKeys.map((k) => h('th.text-end', k)),
          h('th.text-end', 'Size'), h('th.text-end', 'Files'), h('th', 'Digest'), h('th', 'Created by'), h('th.text-end', 'Used by'), h('th', 'Created'), h('th'))),
        h('tbody', rows.map((v) => {
          const vurl = urls.artifact(entity, project, type, collection, v.versionLabel);
          const check = h('input.form-check-input', {
            type: 'checkbox', checked: selected.has(v.versionLabel), 'aria-label': `Select ${v.versionLabel}`,
            onchange: (e) => { e.target.checked ? selected.add(v.versionLabel) : selected.delete(v.versionLabel); drawSelection(); },
          });
          return h('tr', { class: selected.has(v.versionLabel) && 'af-row-selected' },
            h('td.af-col-check', check),
            h('td', h('a.ma-version', { href: vurl }, v.versionLabel), v.description && h('div.ma-desc.small', v.description)),
            h('td', aliasEditor({ entity, project, collection, artifact: v, holders: map, compact: true, onChange: () => load() })),
            metaKeys.map((k) => h('td.text-end.ms-metric', h('span', { class: v.meta?.[k] === best[k] && rows.length > 1 ? 'ms-best' : '' }, fmtMetric(v.meta?.[k])))),
            h('td.text-end.ms-metric', fmtBytes(v.size)),
            h('td.text-end.ms-metric', fmtNumber(v.fileCount)),
            h('td', h('code.small', shortDigest(v.digest))),
            h('td', creatorLink(v.createdBy, urls)),
            h('td.text-end', v.usedCount ? h('a', { href: urls.artifact(entity, project, type, collection, v.versionLabel, 'usage') }, fmtNumber(v.usedCount)) : h('span.text-muted', '0')),
            h('td.af-nowrap', timeAgo(v.createdAt)),
            h('td', menuButton(null, () => [
              { label: 'Open', icon: 'box-arrow-up-right', href: vurl },
              { label: 'Files', icon: 'folder2-open', href: urls.artifact(entity, project, type, collection, v.versionLabel, 'files') },
              { label: 'Lineage', icon: 'diagram-3', href: urls.artifact(entity, project, type, collection, v.versionLabel, 'lineage') },
              type === 'model' && { label: 'Link to registry…', icon: 'box-arrow-in-up-right', onClick: () => linkToRegistry({ entity, project, collection, artifact: v, onDone: load }) },
              { label: 'Copy reference', icon: 'clipboard', onClick: () => navigator.clipboard.writeText(`${entity}/${project}/${collection}:${v.versionLabel}`).then(() => toast('Copied', 'success', { timeout: 1500 })) },
              { divider: true },
              { label: `Delete ${v.versionLabel}…`, icon: 'trash', danger: true, onClick: () => deleteVersion(v) },
            ])));
        })))),
    }));
  }

  function drawSelection() {
    const n = selected.size;
    mount(selEl, selectionBar(n, [
      h('button.btn.btn-sm.btn-light', { disabled: n !== 2, onclick: compare }, icon('layout-split', 'me-1'), n === 2 ? 'Compare' : 'Select 2 to compare'),
      h('button.btn.btn-sm.btn-outline-light', { onclick: deleteSelected }, icon('trash', 'me-1'), `Delete ${n}`),
    ], () => { selected.clear(); drawTable(); drawSelection(); }));
    tableEl.querySelectorAll('tbody tr').forEach((tr) => tr.classList.toggle('af-row-selected', !!tr.querySelector('input:checked')));
  }

  async function deleteVersion(v) {
    const ok = await confirmDialog({
      title: `Delete ${collection}:${v.versionLabel}?`, danger: true, confirmLabel: 'Delete version',
      message: userAliases(v.aliases, collection).length ? `Its aliases (${userAliases(v.aliases, collection).join(', ')}) are removed too.` : 'Its files are deleted. This cannot be undone.',
    });
    if (!ok) return;
    try {
      await gql(`mutation DelArt($input: DeleteArtifactInput!) { deleteArtifact(input: $input) { artifact { id } } }`, { input: { artifactID: v.id, deleteAliases: true } });
      toast(`Deleted ${v.versionLabel}`, 'success');
      selected.delete(v.versionLabel);
      load();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  async function deleteSelected() {
    const vs = col.versions.filter((v) => selected.has(v.versionLabel));
    const ok = await confirmDialog({ title: `Delete ${vs.length} version(s)?`, danger: true, confirmLabel: 'Delete', message: vs.map((v) => v.versionLabel).join(', ') });
    if (!ok) return;
    let failed = 0;
    for (const v of vs) {
      try { await gql(`mutation DelArt($input: DeleteArtifactInput!) { deleteArtifact(input: $input) { artifact { id } } }`, { input: { artifactID: v.id, deleteAliases: true } }); } catch { failed++; }
    }
    toast(failed ? `${failed} of ${vs.length} failed` : `Deleted ${vs.length} version(s)`, failed ? 'warning' : 'success');
    selected.clear();
    drawSelection();
    load();
  }

  async function compare() {
    const [a, b] = col.versions.filter((v) => selected.has(v.versionLabel)).sort((x, y) => x.versionIndex - y.versionIndex);
    const body = h('div', skeleton(6));
    drawer({ title: `${collection}: ${a.versionLabel} ↔ ${b.versionLabel}`, subtitle: 'Metadata and file differences', body, width: 900 });
    try {
      const [fa, fb] = await Promise.all([loadArtifactFiles(a.id), loadArtifactFiles(b.id)]);
      const md = metadataDiff(a.meta, b.meta);
      const fd = fileDiff(fa, fb);
      const changed = (rows) => rows.filter((r) => r.status !== 'same');
      let onlyChanges = true;
      const mdEl = h('div');
      const fdEl = h('div');
      const pill = (s) => h(`span.ma-diff-pill.${s}`, s);
      const drawDiff = () => {
        const mrows = onlyChanges ? changed(md) : md;
        const frows = onlyChanges ? changed(fd) : fd;
        mount(mdEl, mrows.length ? h('table.af-table.af-table-dense', h('thead', h('tr', h('th', 'Key'), h('th', a.versionLabel), h('th', b.versionLabel), h('th'))),
          h('tbody', mrows.map((r) => h('tr', { class: `ma-diff-${r.status}` }, h('td.ms-mono', r.key), h('td.ms-mono', fmtMetric(r.a)), h('td.ms-mono', fmtMetric(r.b)), h('td', pill(r.status))))))
          : emptyState({ icon: 'check2-all', title: 'Metadata is identical' }));
        mount(fdEl, frows.length ? h('table.af-table.af-table-dense', h('thead', h('tr', h('th', 'File'), h('th.text-end', a.versionLabel), h('th.text-end', b.versionLabel), h('th'))),
          h('tbody', frows.map((r) => h('tr', { class: `ma-diff-${r.status}` }, h('td.ms-mono', r.name), h('td.text-end', r.a ? fmtBytes(r.a.sizeBytes) : '—'), h('td.text-end', r.b ? fmtBytes(r.b.sizeBytes) : '—'), h('td', pill(r.status))))))
          : emptyState({ icon: 'check2-all', title: 'Files are identical' }));
      };
      const toggle = h('label.form-check.form-switch.small.mb-2', h('input.form-check-input', { type: 'checkbox', checked: true, onchange: (e) => { onlyChanges = e.target.checked; drawDiff(); } }), h('span.ms-1', 'Only show differences'));
      const sum = (rows) => ['added', 'removed', 'changed'].map((s) => `${rows.filter((r) => r.status === s).length} ${s}`).join(' · ');
      mount(body,
        kv([[a.versionLabel, h('span', fmtDate(a.createdAt), ' · ', fmtBytes(a.size))], [b.versionLabel, h('span', fmtDate(b.createdAt), ' · ', fmtBytes(b.size))],
          ['Digest', a.digest === b.digest ? h('span.text-success', icon('check2'), ' identical') : h('span.text-warning', icon('exclamation-triangle'), ' different')]]),
        toggle,
        h('h6.mt-3', 'Metadata ', h('span.small.text-muted', sum(md))), mdEl,
        h('h6.mt-3', 'Files ', h('span.small.text-muted', sum(fd))), fdEl);
      drawDiff();
    } catch (err) { mount(body, errorState(err)); }
  }

  await load();
}
