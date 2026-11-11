// Artifact file browser: folder tree + searchable list + in-place preview
// (text, JSON, CSV/TSV as a table, images, W&B tables) and downloads.

import { h, icon, mount, debounce } from '../../../../ui/js/dom.js';
import { jsonView, codeBlock, emptyState, errorState, skeleton, copyButton } from '../../../../ui/js/ui.js';
import { fileUrl, fetchFile } from '../../core/api.js';
import { fmtBytes } from './shared.js';

const TEXT_EXT = /\.(txt|log|md|py|yaml|yml|toml|ini|cfg|sh|sql|js|ts|html|xml|env|gitignore|requirements)$/i;
const IMG_EXT = /\.(png|jpe?g|gif|webp|svg|bmp)$/i;
const MAX_PREVIEW_BYTES = 5 * 1024 * 1024;

const extOf = (name) => (name.match(/\.([^.\/]+)$/) || ['', ''])[1].toLowerCase();
const fileIcon = (name) => {
  const e = extOf(name);
  if (IMG_EXT.test(name)) return 'file-earmark-image';
  if (e === 'json') return name.endsWith('.table.json') ? 'table' : 'filetype-json';
  if (e === 'csv' || e === 'tsv') return 'filetype-csv';
  if (e === 'py') return 'filetype-py';
  if (e === 'yaml' || e === 'yml') return 'filetype-yml';
  if (e === 'md') return 'filetype-md';
  if (['bin', 'pt', 'pth', 'ckpt', 'onnx', 'h5', 'safetensors', 'pkl'].includes(e)) return 'file-earmark-binary';
  return 'file-earmark';
};

function parseDelimited(text, sep) {
  const rows = [];
  let row = [];
  let cur = '';
  let q = false;
  for (let i = 0; i < text.length && rows.length < 501; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c;
    } else if (c === '"') q = true;
    else if (c === sep) { row.push(cur); cur = ''; } else if (c === '\n') { row.push(cur); rows.push(row); row = []; cur = ''; } else if (c !== '\r') cur += c;
  }
  if (cur || row.length) { row.push(cur); rows.push(row); }
  return rows;
}

function tablePreview(header, rows, total) {
  return h('div.ma-table-preview',
    h('div.small.text-muted.mb-1', `${total.toLocaleString()} row${total === 1 ? '' : 's'}${total > rows.length ? ` · first ${rows.length} shown` : ''}`),
    h('div.ma-table-scroll', h('table.af-table.af-table-dense',
      h('thead', h('tr', header.map((c) => h('th', String(c))))),
      h('tbody', rows.map((r) => h('tr', r.map((v) => h('td', v == null ? h('span.text-muted', 'null') : typeof v === 'object' ? JSON.stringify(v) : String(v)))))))));
}

async function renderPreview(el, file) {
  const url = fileUrl(file.directUrl);
  const head = h('div.ma-preview-head',
    h('div.ma-preview-name', icon(fileIcon(file.name)), h('span', file.name)),
    h('div.d-flex.gap-2.align-items-center',
      h('span.text-muted.small', fmtBytes(file.sizeBytes)),
      file.digest && copyButton(file.digest, 'digest'),
      h('a.btn.btn-sm.btn-outline-secondary', { href: fileUrl(file.directUrl, { download: true }), download: file.name.split('/').pop() }, icon('download', 'me-1'), 'Download')));
  const body = h('div.ma-preview-body', skeleton(6));
  mount(el, head, body);
  try {
    if (IMG_EXT.test(file.name)) {
      mount(body, h('div.ma-img-preview', h('img', { src: url, alt: file.name, loading: 'lazy' })));
      return;
    }
    const e = extOf(file.name);
    const textual = TEXT_EXT.test(file.name) || ['json', 'csv', 'tsv'].includes(e) || (file.mimetype || '').startsWith('text');
    if (!textual) {
      mount(body, emptyState({ icon: 'file-earmark-binary', title: 'Binary file', message: 'No preview for this file type — download it to inspect.' }));
      return;
    }
    if (file.sizeBytes > MAX_PREVIEW_BYTES) {
      mount(body, emptyState({ icon: 'file-earmark-zip', title: `Large file (${fmtBytes(file.sizeBytes)})`, message: 'Download it to view the full contents.' }));
      return;
    }
    const text = await fetchFile(file.directUrl);
    if (e === 'json') {
      let data;
      try { data = JSON.parse(text); } catch { mount(body, codeBlock(text, { lang: 'json' })); return; }
      // W&B table: {columns, data}
      if (data && Array.isArray(data.columns) && Array.isArray(data.data)) {
        mount(body, tablePreview(data.columns, data.data.slice(0, 500), data.data.length));
        return;
      }
      mount(body, jsonView(data, { open: 2 }));
      return;
    }
    if (e === 'csv' || e === 'tsv') {
      const rows = parseDelimited(text, e === 'tsv' ? '\t' : ',');
      const total = Math.max(0, text.split('\n').filter(Boolean).length - 1);
      mount(body, rows.length ? tablePreview(rows[0], rows.slice(1, 501), total) : emptyState({ title: 'Empty file' }));
      return;
    }
    mount(body, codeBlock(text.length > 400000 ? `${text.slice(0, 400000)}\n… (truncated)` : text, { lang: e === 'py' ? 'python' : e === 'sql' ? 'sql' : e || 'text', maxHeight: '60vh' }));
  } catch (err) {
    mount(body, errorState(err));
  }
}

/** files: [{name, sizeBytes, digest, directUrl, mimetype}] */
export function fileBrowser(files, { initial } = {}) {
  let folder = '';
  let query = '';
  let selected = files.find((f) => f.name === initial) || null;
  const list = h('div.ma-files-list');
  const preview = h('div.ma-files-preview');
  const crumbs = h('div.ma-files-crumbs');
  const total = files.reduce((a, f) => a + (f.sizeBytes || 0), 0);

  const search = h('div.af-search.ma-files-search', icon('search', 'af-search-icon'),
    h('input.form-control.form-control-sm', { type: 'search', placeholder: `Search ${files.length} files…`, oninput: debounce((e) => { query = e.target.value.trim().toLowerCase(); draw(); }, 120) }));

  function children() {
    const dirs = new Map();
    const here = [];
    for (const f of files) {
      if (query) { if (f.name.toLowerCase().includes(query)) here.push(f); continue; }
      if (!f.name.startsWith(folder)) continue;
      const rest = f.name.slice(folder.length);
      const slash = rest.indexOf('/');
      if (slash >= 0) {
        const d = rest.slice(0, slash);
        const cur = dirs.get(d) || { name: d, count: 0, size: 0 };
        cur.count++;
        cur.size += f.sizeBytes || 0;
        dirs.set(d, cur);
      } else here.push(f);
    }
    return { dirs: [...dirs.values()].sort((a, b) => a.name.localeCompare(b.name)), files: here.sort((a, b) => a.name.localeCompare(b.name)) };
  }

  function draw() {
    const parts = folder.split('/').filter(Boolean);
    mount(crumbs,
      h('button.af-link-btn', { onclick: () => { folder = ''; draw(); } }, icon('house-door', 'me-1'), 'root'),
      parts.map((p, i) => [h('span.text-muted', ' / '), h('button.af-link-btn', { onclick: () => { folder = `${parts.slice(0, i + 1).join('/')}/`; draw(); } }, p)]));
    const { dirs, files: fs } = children();
    mount(list,
      !dirs.length && !fs.length ? emptyState({ icon: 'search', title: query ? 'No files match' : 'Empty folder' }) : null,
      dirs.map((d) => h('button.ma-file-row.dir', { type: 'button', onclick: () => { folder = `${folder}${d.name}/`; draw(); } },
        icon('folder-fill'), h('span.ma-file-name', d.name), h('span.ma-file-meta', `${d.count} file${d.count === 1 ? '' : 's'} · ${fmtBytes(d.size)}`))),
      fs.map((f) => h('button.ma-file-row', { type: 'button', class: selected === f && 'active', onclick: () => { selected = f; draw(); renderPreview(preview, f); } },
        icon(fileIcon(f.name)), h('span.ma-file-name', query ? f.name : f.name.slice(folder.length)), h('span.ma-file-meta', fmtBytes(f.sizeBytes)))));
  }

  draw();
  if (selected) renderPreview(preview, selected);
  else if (files.length === 1) { selected = files[0]; renderPreview(preview, files[0]); draw(); }
  else mount(preview, emptyState({ icon: 'hand-index', title: 'Pick a file to preview', message: 'Text, JSON, CSV/TSV, W&B tables and images open right here.' }));

  return h('div.ma-files',
    h('div.ma-files-side',
      h('div.ma-files-top', search, h('div.small.text-muted', `${files.length.toLocaleString()} files · ${fmtBytes(total)}`)),
      crumbs, list),
    preview);
}
