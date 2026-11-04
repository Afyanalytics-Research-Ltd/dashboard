// Run page — Files: folder tree with breadcrumbs, search, previews (text, JSON, YAML, CSV,
// Markdown, code, images) and downloads, all through the same-origin file proxy.

import { h, icon, mount, debounce } from '../../../../ui/js/dom.js';
import { drawer, emptyState, errorState, jsonView, codeBlock, copyButton, skeleton, timeAgo } from '../../../../ui/js/ui.js';
import { loadMarkdown } from '../../../../ui/js/lib.js';
import { fileUrl, fetchFile } from '../../core/api.js';
import { fileIndex, fmtBytesShort, highlight } from './common.js';

const PREVIEW_LIMIT = 2 * 1024 * 1024;
const IMG = /\.(png|jpe?g|gif|webp|svg|bmp)$/i;
const CODE_LANG = { py: 'python', sql: 'sql' };

function fileIcon(name) {
  if (IMG.test(name)) return 'file-earmark-image';
  if (/\.json$/i.test(name)) return 'filetype-json';
  if (/\.ya?ml$/i.test(name)) return 'filetype-yml';
  if (/\.py$/i.test(name)) return 'filetype-py';
  if (/\.(log|txt)$/i.test(name)) return 'file-earmark-text';
  if (/\.csv$/i.test(name)) return 'filetype-csv';
  if (/\.md$/i.test(name)) return 'filetype-md';
  if (/\.(bin|pt|pth|ckpt|h5|onnx|safetensors)$/i.test(name)) return 'file-earmark-binary';
  return 'file-earmark';
}

/** Build {folders: Map(name -> node), files: []} tree from flat names. */
function buildTree(files) {
  const root = { name: '', folders: new Map(), files: [], size: 0, count: 0 };
  for (const f of files) {
    const parts = f.name.split('/');
    let node = root;
    root.size += f.sizeBytes || 0;
    root.count++;
    for (const part of parts.slice(0, -1)) {
      if (!node.folders.has(part)) node.folders.set(part, { name: part, folders: new Map(), files: [], size: 0, count: 0 });
      node = node.folders.get(part);
      node.size += f.sizeBytes || 0;
      node.count++;
    }
    node.files.push({ ...f, base: parts[parts.length - 1] });
  }
  return root;
}

export async function filesTab(el, ctx, run) {
  const { entity, project } = ctx.params;
  mount(el, skeleton(6));
  let files;
  try {
    files = [...(await fileIndex(entity, project, run.name)).values()].sort((a, b) => a.name.localeCompare(b.name));
  } catch (err) {
    if (err.name !== 'AbortError') mount(el, errorState(err, () => filesTab(el, ctx, run)));
    return;
  }
  if (!files.length) {
    mount(el, emptyState({ icon: 'folder2-open', title: 'No files', message: 'Files saved with wandb.save() or logged media appear here.' }));
    return;
  }
  const tree = buildTree(files);
  let path = (ctx.query.get('path') || '').split('/').filter(Boolean);
  let q = '';
  const listEl = h('div.msr-files-list');
  const crumbsEl = h('nav.msr-crumbs', { 'aria-label': 'Folder' });

  const nodeAt = (p) => p.reduce((n, part) => n?.folders.get(part), tree) || tree;

  const draw = () => {
    const node = nodeAt(path);
    mount(crumbsEl,
      h('button.msr-crumb', { type: 'button', onclick: () => { path = []; sync(); } }, icon('hdd'), ' files'),
      path.map((part, i) => [icon('chevron-right', 'msr-crumb-sep'), h('button.msr-crumb', { type: 'button', onclick: () => { path = path.slice(0, i + 1); sync(); } }, part)]));
    let folderRows = [...node.folders.values()];
    let fileRows = node.files;
    if (q) {
      // Search spans the whole run, not just the open folder.
      folderRows = [];
      fileRows = files.filter((f) => f.name.toLowerCase().includes(q)).map((f) => ({ ...f, base: f.name }));
    }
    if (!folderRows.length && !fileRows.length) { mount(listEl, h('div.msr-kv-empty', q ? `No files match “${q}”.` : 'This folder is empty.')); return; }
    mount(listEl, h('table.msr-files-table',
      h('thead', h('tr', h('th', 'Name'), h('th.text-end', 'Size'), h('th.msr-hide-sm', 'Updated'), h('th'))),
      h('tbody',
        !q && path.length > 0 && h('tr.msr-file-row', { onclick: () => { path = path.slice(0, -1); sync(); } }, h('td', icon('arrow-90deg-up', 'me-2'), '..'), h('td'), h('td.msr-hide-sm'), h('td')),
        folderRows.sort((a, b) => a.name.localeCompare(b.name)).map((f) => h('tr.msr-file-row', { onclick: () => { path = [...path, f.name]; sync(); } },
          h('td', icon('folder-fill', 'me-2 msr-folder-ic'), h('span.fw-semibold', f.name), h('span.small.text-muted.ms-2', `${f.count} file${f.count === 1 ? '' : 's'}`)),
          h('td.text-end.ms-metric', fmtBytesShort(f.size)), h('td.msr-hide-sm'), h('td'))),
        fileRows.map((f) => h('tr.msr-file-row', { onclick: (e) => { if (!e.target.closest('a,button')) preview(f); } },
          h('td', icon(fileIcon(f.name), 'me-2 text-muted'), h('span.msr-file-name', highlight(f.base, q))),
          h('td.text-end.ms-metric', fmtBytesShort(f.sizeBytes)),
          h('td.msr-hide-sm.small', f.updatedAt ? timeAgo(f.updatedAt) : '—'),
          h('td.text-end.msr-file-actions',
            h('button.af-icon-btn', { type: 'button', title: 'Preview', onclick: () => preview(f) }, icon('eye')),
            h('a.af-icon-btn', { href: fileUrl(f.directUrl || f.url, { download: true }), title: 'Download', download: f.base }, icon('download'))))))));
  };
  const sync = () => { ctx.setQuery({ path: path.join('/') || null }); draw(); };

  const searchEl = h('input.form-control.form-control-sm', {
    type: 'search', placeholder: `Search ${files.length} files…`, dataset: { searchShortcut: '' }, 'aria-label': 'Search files',
    oninput: debounce((e) => { q = e.target.value.trim().toLowerCase(); draw(); }, 120),
  });
  mount(el, h('div.msr-files',
    h('div.msr-files-bar', crumbsEl, h('div.af-search.msr-files-search', icon('search', 'af-search-icon'), searchEl),
      h('span.small.text-muted', `${files.length} files · ${fmtBytesShort(tree.size)}`)),
    listEl));
  draw();

  async function preview(f) {
    const url = f.directUrl || f.url;
    const body = h('div', skeleton(8));
    drawer({
      title: h('span.ms-mono', f.name), subtitle: `${fmtBytesShort(f.sizeBytes)}${f.mimetype ? ` · ${f.mimetype}` : ''}`, width: 860, body,
      actions: [copyButton(f.name, 'Path'), h('a.btn.btn-sm.btn-outline-secondary', { href: fileUrl(url, { download: true }), download: f.base || f.name }, icon('download', 'me-1'), 'Download')],
    });
    try {
      if (IMG.test(f.name)) { mount(body, h('div.msr-img-preview', h('img', { src: fileUrl(url), alt: f.name }))); return; }
      if ((f.sizeBytes || 0) > PREVIEW_LIMIT) {
        mount(body, emptyState({ icon: 'file-earmark-binary', title: 'Too large to preview', message: `${fmtBytesShort(f.sizeBytes)} — download it instead.` }));
        return;
      }
      if (/\.(bin|pt|pth|ckpt|h5|onnx|safetensors|pkl|npy|npz|zip|gz|tar)$/i.test(f.name)) {
        mount(body, emptyState({ icon: 'file-earmark-binary', title: 'Binary file', message: 'Download it to open locally.' }));
        return;
      }
      const text = await fetchFile(url);
      const ext = (f.name.split('.').pop() || '').toLowerCase();
      if (ext === 'json') {
        let parsed = null;
        try { parsed = JSON.parse(text); } catch { /* show raw */ }
        if (parsed && Array.isArray(parsed.columns) && Array.isArray(parsed.data)) { mount(body, tableFromJson(parsed)); return; }
        mount(body, parsed != null ? jsonView(parsed, { open: 2 }) : codeBlock(text, { lang: 'text' }));
      } else if (ext === 'md') {
        const render = await loadMarkdown();
        const md = h('div.af-markdown');
        md.innerHTML = render(text);
        mount(body, md);
      } else if (ext === 'csv' || ext === 'tsv') {
        mount(body, csvTable(text, ext === 'tsv' ? '\t' : ','));
      } else {
        mount(body, codeBlock(text, { lang: CODE_LANG[ext] || (ext === 'yaml' || ext === 'yml' ? 'yaml' : 'text') }));
      }
    } catch (err) {
      mount(body, errorState(err));
    }
  }
}

function csvTable(text, sep) {
  const rows = text.split(/\r?\n/).filter(Boolean).slice(0, 1001).map((line) => {
    const out = [];
    let cur = '';
    let inQ = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') { if (inQ && line[i + 1] === '"') { cur += '"'; i++; } else inQ = !inQ; } else if (c === sep && !inQ) { out.push(cur); cur = ''; } else cur += c;
    }
    out.push(cur);
    return out;
  });
  const [head, ...body] = rows;
  return h('div.msr-table-wrap', h('table.msr-data-table', h('thead', h('tr', (head || []).map((c) => h('th', c)))),
    h('tbody', body.map((r) => h('tr', r.map((c) => h('td', c)))))),
  rows.length > 1000 && h('div.small.text-muted.p-2', 'Showing the first 1,000 rows.'));
}

/** Render a W&B table JSON ({columns, data}). Exported for the Media tab. */
export function tableFromJson(t, { searchable = true } = {}) {
  let sortCol = null;
  let dir = 1;
  let q = '';
  const wrap = h('div.msr-table-wrap');
  const draw = () => {
    let rows = t.data.map((r, i) => ({ r, i }));
    if (q) rows = rows.filter(({ r }) => r.some((v) => String(v ?? '').toLowerCase().includes(q)));
    if (sortCol != null) rows.sort((a, b) => {
      const x = a.r[sortCol];
      const y = b.r[sortCol];
      return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true })) * dir;
    });
    mount(wrap, h('table.msr-data-table',
      h('thead', h('tr', t.columns.map((c, ci) => h('th', {
        class: sortCol === ci && 'sorted', role: 'button', title: 'Sort',
        onclick: () => { if (sortCol === ci) dir = -dir; else { sortCol = ci; dir = 1; } draw(); },
      }, c, sortCol === ci && icon(dir > 0 ? 'sort-up' : 'sort-down', 'ms-1'))))),
      h('tbody', rows.map(({ r }) => h('tr', r.map((v) => h('td', { class: typeof v === 'number' && 'ms-metric' },
        typeof v === 'boolean' ? icon(v ? 'check-lg' : 'x-lg', v ? 'text-success' : 'text-danger') : v && typeof v === 'object' ? JSON.stringify(v) : highlight(v ?? '', q))))))),
    !rows.length && h('div.msr-kv-empty', 'No rows match.'));
  };
  draw();
  const search = searchable && h('input.form-control.form-control-sm.msr-table-search', {
    type: 'search', placeholder: `Search ${t.data.length} rows…`, 'aria-label': 'Search table',
    oninput: debounce((e) => { q = e.target.value.trim().toLowerCase(); draw(); }, 120),
  });
  const csv = () => [t.columns, ...t.data].map((r) => r.map((v) => { const s = v && typeof v === 'object' ? JSON.stringify(v) : String(v ?? ''); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }).join(',')).join('\n');
  return h('div.msr-table-block',
    h('div.msr-table-tools', search, h('span.small.text-muted', `${t.data.length} rows × ${t.columns.length} columns`), copyButton(csv, 'CSV')),
    wrap);
}
