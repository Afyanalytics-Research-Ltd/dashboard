// REST API source: import a Postman collection or OpenAPI/Swagger file → auth, pagination, streams.
// With AI on, Groq picks the endpoints that are real entity tables and names them for mapping.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { toast, toastError } from '../../../ui/js/ui.js';
import { api } from '../api.js';
import { dropzone, seg } from '../kit.js';

const METHOD_CLS = { GET: 'cs-m-get', POST: 'cs-m-post', PUT: 'cs-m-put', PATCH: 'cs-m-put', DELETE: 'cs-m-del' };
const CONFIG_KEYS = ['base_url', 'auth_type', 'token', 'username', 'password', 'api_key_name', 'login_path', 'login_body',
  'token_path', 'headers', 'pagination', 'page_param', 'size_param', 'page_size', 'cursor_path', 'records_path'];

export function streamLine(ep) {
  const m = (ep.method || 'GET').toUpperCase();
  let line = `${ep.name} = ${m === 'GET' ? '' : `${m} `}${ep.path}`;
  if (ep.body && Object.keys(ep.body).length) line += ` ${JSON.stringify(ep.body)}`;
  if (ep.records) line += ` -> ${ep.records}`;
  return line;
}

const slug = (s) => String(s || '').toLowerCase().replace(/[^0-9a-z]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'stream';

// Afya Platform (V3): a Model Gateway Postman collection → Models list (+ URL template when it names real hosts).
export function v3ImportPanel({ form, onApplied }) {
  const el = h('div.cs-import');
  let result = null;

  function intro() {
    mount(el,
      h('div.cs-import-head',
        h('span.cs-import-icon', icon('box-arrow-in-down')),
        h('div.cs-grow', h('div.cs-import-title', 'Import a Model Gateway Postman collection'),
          h('div.cs-muted', 'Picks up the models the collection reads, per service — structure still comes live from describe.'))),
      dropzone({ accept: ['.json'], multiple: false, title: 'Drop the ModelGateway …postman_collection.json',
        sub: 'Postman v2.0 / v2.1', onFiles: ([f]) => run(f) }));
  }

  async function run(file) {
    mount(el, h('div.cs-import-busy', h('span.spinner-border.spinner-border-sm'), h('div', h('b', `Reading ${file.name}`))));
    try {
      if (file.size > 20 * 1024 * 1024) throw new Error('That file is over 20 MB.');
      result = await api.importApiFile(file, { kind: 'afya_platform' });
      review();
    } catch (e) { toastError(e, 'Import failed'); intro(); }
  }

  function review() {
    const countEl = h('span');
    const applyBtn = h('button.cs-btn.cs-btn-brand', { type: 'button', onclick: apply }, icon('magic'), 'Apply to form');
    const drawCount = () => {
      const n = result.models.filter((m) => m.selected).length;
      mount(countEl, h('b', String(n)), ` of ${result.models.length} models selected`);
      applyBtn.disabled = !n;
    };
    const groups = result.services.map((svc) => {
      const ms = result.models.filter((m) => m.service === svc);
      return h('div.cs-v3-group',
        h('div.cs-v3-svc', svc, h('span.cs-muted', ` · ${ms.length}`)),
        h('div.cs-chips', ms.map((m) => {
          const chip = h('button.cs-chip.cs-v3-model', { type: 'button', class: m.selected && 'on', title: m.why,
            onclick: () => { m.selected = !m.selected; chip.classList.toggle('on', m.selected); drawCount(); } },
          icon(m.selected ? 'check2-square' : 'square'), m.alias);
          chip.addEventListener('click', () => { chip.querySelector('i').className = `bi bi-${m.selected ? 'check2-square' : 'square'}`; });
          return chip;
        })));
    });
    mount(el,
      h('div.cs-import-head',
        h('span.cs-import-icon.ok', icon('check2-circle')),
        h('div.cs-grow', h('div.cs-import-title', result.title), h('div.cs-muted', h('span.cs-badge.cs-badge-outline', result.format))),
        h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: intro }, icon('arrow-counterclockwise'), 'Another file')),
      h('div.cs-import-facts',
        h('div', h('span.cs-muted', 'Services'), h('b.cs-ellipsis', result.services.join(', '))),
        h('div', h('span.cs-muted', 'URL template'), h('b.cs-ellipsis', result.url_template || 'production (kept)'))),
      result.notes?.length ? h('details.cs-import-notes', { open: true }, h('summary', icon('info-circle'), `${result.notes.length} notes`),
        h('ul', result.notes.map((nt) => h('li', nt)))) : null,
      h('div.cs-muted', { style: { fontSize: '12px' } }, 'Ticked = read in the collection. Unticked models only appear in copy / insert / describe requests (reference data) — tick them to include.'),
      h('div.cs-v3-groups', groups),
      h('div.cs-row.cs-wrap.cs-import-foot', h('span.cs-muted', countEl), h('span.cs-grow'), applyBtn));
    drawCount();
  }

  function apply() {
    const chosen = result.models.filter((m) => m.selected);
    const vals = { models: chosen.map((m) => m.id).join(', '), services: [...new Set(chosen.map((m) => m.service))].join(', ') };
    if (result.url_template) vals.url_template = result.url_template;
    form.set(vals);
    toast(`${chosen.length} models filled in — add the account’s username and password, then test.`, 'success', { title: 'Imported' });
    onApplied?.(result);
  }

  intro();
  return el;
}

export function importPanel({ form, meta, onApplied }) {
  const aiAvailable = !!meta.llm?.available;
  let useAi = aiAvailable;
  let result = null;
  let filter = 'selected';
  const el = h('div.cs-import');

  function intro() {
    const aiSwitch = h('label.cs-switch', { title: aiAvailable ? '' : 'No Groq key configured — the parser’s choices are used' },
      h('input', { type: 'checkbox', checked: useAi, disabled: !aiAvailable, onchange: (e) => { useAi = e.target.checked; } }),
      h('span.track'), h('span', h('b', 'AI setup'), ' — choose & name the tables for mapping'));
    mount(el,
      h('div.cs-import-head',
        h('span.cs-import-icon', icon('box-arrow-in-down')),
        h('div.cs-grow', h('div.cs-import-title', 'Import from Postman or OpenAPI'),
          h('div.cs-muted', 'Drop a collection or spec and the auth, pagination and streams are filled in for you.')),
        aiSwitch),
      dropzone({ accept: ['.json', '.yaml', '.yml'], multiple: false, title: 'Drop a Postman collection, OpenAPI or Swagger file',
        sub: 'Postman v2.0 / v2.1 · OpenAPI 3.x · Swagger 2.0 — JSON or YAML', onFiles: ([f]) => run(f) }));
  }

  async function run(file) {
    mount(el, h('div.cs-import-busy',
      h('span.spinner-border.spinner-border-sm'),
      h('div', h('b', `Reading ${file.name}`),
        h('div.cs-muted', useAi ? 'Parsing endpoints, then Groq picks the ones that are tables and names them…' : 'Parsing endpoints…'))));
    try {
      if (file.size > 20 * 1024 * 1024) throw new Error('That file is over 20 MB.');
      result = await api.importApiFile(file, { ai: useAi ? '1' : '0' });
      result.file = file.name;
      filter = result.endpoints.some((e) => e.selected) ? 'selected' : 'all';
      review();
    } catch (e) {
      toastError(e, 'Import failed');
      intro();
    }
  }

  function review() {
    const eps = result.endpoints;
    const listBox = h('div.cs-ep-list');
    const countEl = h('span');
    const applyBtn = h('button.cs-btn.cs-btn-brand', { type: 'button', onclick: apply }, icon('magic'), 'Apply to form');

    const drawCount = () => {
      const n = eps.filter((e) => e.selected).length;
      mount(countEl, h('b', String(n)), ` of ${eps.length} endpoints become streams`);
      applyBtn.disabled = !n;
    };
    const drawList = () => {
      const shown = eps.filter((e) => filter === 'all' || (filter === 'selected' ? e.selected : !e.selected));
      mount(listBox, shown.length ? shown.map(row) : h('div.cs-muted.cs-ep-empty', 'Nothing here.'));
    };
    const row = (e) => {
      const name = h('input.cs-input.cs-ep-name', { value: e.name, disabled: !e.selected, spellcheck: 'false', title: 'Table name',
        oninput: (ev) => { e.name = slug(ev.target.value) || e.name; } , onblur: (ev) => { ev.target.value = e.name; } });
      const cb = h('input', { type: 'checkbox', checked: e.selected, onchange: (ev) => {
        e.selected = ev.target.checked; name.disabled = !e.selected; r.classList.toggle('off', !e.selected); drawCount();
      } });
      const r = h('label.cs-ep', { class: !e.selected && 'off', title: streamLine(e) },
        cb,
        h(`span.cs-method.${METHOD_CLS[e.method] || 'cs-m-get'}`, e.method),
        h('div.cs-ep-main',
          h('div.cs-row.cs-gap-sm', name, e.ai?.include && h('span.cs-badge.cs-badge-ai', { title: e.ai.why || 'Chosen by AI' }, icon('stars'), 'AI')),
          h('div.cs-ep-path.cs-mono.cs-ellipsis', e.path, e.body && h('span.cs-muted', ` ${JSON.stringify(e.body)}`)),
          (e.ai?.why || e.reason || e.folder) && h('div.cs-ep-why.cs-muted',
            e.selected ? (e.ai?.why || e.folder) : (e.reason || e.ai?.why))),
        h('div.cs-ep-side',
          e.records != null && e.records !== '' ? h('span.cs-chip.cs-mono', { title: 'Records path' }, icon('list-ul'), e.records) : null,
          e.fields?.length ? h('span.cs-muted.cs-ep-fields', { title: e.fields.join(', ') }, `${e.fields.length} fields`) : null));
      return r;
    };

    const setAll = (on) => { for (const e of eps) e.selected = on && !/login/.test(e.reason || ''); drawCount(); drawList(); };
    const cfg = result.config;
    const facts = [
      ['Base URL', cfg.base_url || h('span.cs-badge.cs-badge-warn', 'fill in')],
      ['Auth', ({ none: 'None', bearer: 'Bearer token', basic: 'Basic', api_key_header: `API key · ${cfg.api_key_name || ''}`,
        api_key_query: `API key (query) · ${cfg.api_key_name || ''}`, login: `Login · ${cfg.login_path || ''}` })[cfg.auth_type] || cfg.auth_type || 'None'],
      ['Pagination', cfg.pagination && cfg.pagination !== 'none' ? `${cfg.pagination} · ${[cfg.page_param, cfg.size_param].filter(Boolean).join(' / ')}` : 'none detected'],
      ['Records', cfg.records_path ?? 'auto-detect'],
    ];

    mount(el,
      h('div.cs-import-head',
        h('span.cs-import-icon.ok', icon('check2-circle')),
        h('div.cs-grow', h('div.cs-import-title', result.title),
          h('div.cs-muted', h('span.cs-badge.cs-badge-outline', result.format), ' ', result.file)),
        h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => { result = null; intro(); } }, icon('arrow-counterclockwise'), 'Another file')),
      result.ai?.used
        ? h('div.cs-callout.cs-callout-ai', icon('stars'), h('div', h('b', 'AI setup: '), result.ai.summary || `${result.ai.included} tables chosen.`))
        : result.ai?.error ? h('div.cs-callout.cs-callout-warn', icon('exclamation-triangle'), h('div', 'AI setup unavailable — showing the parser’s choices. ', h('span.cs-muted', result.ai.error))) : null,
      h('div.cs-import-facts', facts.map(([k, v]) => h('div', h('span.cs-muted', k), h('b.cs-ellipsis', v)))),
      result.notes?.length ? h('details.cs-import-notes', { open: result.notes.length <= 3 },
        h('summary', icon('info-circle'), `${result.notes.length} note${result.notes.length > 1 ? 's' : ''}`),
        h('ul', result.notes.map((nt) => h('li', nt)))) : null,
      h('div.cs-row.cs-wrap.cs-import-bar',
        seg([{ value: 'selected', label: 'Streams', count: eps.filter((e) => e.selected).length },
          { value: 'skipped', label: 'Skipped', count: eps.filter((e) => !e.selected).length },
          { value: 'all', label: 'All', count: eps.length }], filter, (v) => { filter = v; drawList(); }),
        h('span.cs-grow'),
        h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => setAll(false) }, 'None'),
        h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => setAll(true) }, 'All')),
      listBox,
      h('div.cs-row.cs-wrap.cs-import-foot', h('span.cs-muted', countEl), h('span.cs-grow'), applyBtn));
    drawCount(); drawList();
  }

  function apply() {
    const chosen = result.endpoints.filter((e) => e.selected);
    const seen = new Map();
    for (const e of chosen) {             // names must stay unique after manual edits
      const k = e.name; seen.set(k, (seen.get(k) || 0) + 1);
      if (seen.get(k) > 1) e.name = `${k}_${seen.get(k)}`;
    }
    const vals = {};
    for (const k of CONFIG_KEYS) if (result.config[k] != null && result.config[k] !== '') vals[k] = result.config[k];
    vals.streams = chosen.map(streamLine).join('\n');
    const changed = form.set(vals, { name: result.title });
    toast(`${chosen.length} stream${chosen.length > 1 ? 's' : ''} and ${changed.length - 1} settings filled in — review, then test the connection.`, 'success', { title: 'Imported' });
    onApplied?.(result);
  }

  intro();
  return el;
}
