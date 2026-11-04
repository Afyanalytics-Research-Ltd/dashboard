// Projects: every project of the chosen entity as cards or a list, with instant search,
// sorting, run counts, last activity and project creation.

import { h, icon, mount, debounce, prefs } from '../../../ui/js/dom.js';
import { pageHeader, emptyState, errorState, skeleton, timeAgo, formModal, toast, chips } from '../../../ui/js/ui.js';
import { fmtNumber, fmtDate } from '../../../ui/js/format.js';
import { navigate } from '../../../ui/js/router.js';
import { gql } from '../core/api.js';
import { urls } from '../core/urls.js';
import { loadViewer, loadProjects } from '../core/data.js';

const SORTS = [
  { value: 'active', label: 'Last active', fn: (a, b) => new Date(b.lastActive || b.updatedAt || 0) - new Date(a.lastActive || a.updatedAt || 0) },
  { value: 'name', label: 'Name', fn: (a, b) => a.name.localeCompare(b.name) },
  { value: 'runs', label: 'Most runs', fn: (a, b) => (b.totalRuns || 0) - (a.totalRuns || 0) },
  { value: 'created', label: 'Newest', fn: (a, b) => new Date(b.createdAt) - new Date(a.createdAt) },
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  let view = prefs.get('ms:projects:view', 'cards');
  let sort = ctx.query.get('sort') || 'active';
  let q = ctx.query.get('q') || '';
  let entity = ctx.query.get('entity') || '';
  let entities = [];
  let projects = [];

  const listEl = h('div', skeleton(6));
  const toolbarEl = h('div.mpj-toolbar');
  const headerEl = h('div');
  mount(el, headerEl, toolbarEl, listEl);

  const sync = () => ctx.setQuery({ q: q || null, sort: sort !== 'active' ? sort : null, entity: entity || null });

  async function load() {
    try {
      const { viewer, teams } = await loadViewer({ signal });
      // Personal entity first, then the default entity, then teams.
      entities = [...new Set([viewer?.username, viewer?.entity, ...teams.map((t) => t.name)].filter(Boolean))];
      if (!entity || !entities.includes(entity)) {
        const last = prefs.get('ms:projects:entity', '');
        entity = entities.includes(last) ? last : entities[0] || '';
      }
      prefs.set('ms:projects:entity', entity);
      projects = await loadProjects(entity, { max: 1000, signal });
      if (!ctx.active) return;
      renderHeader();
      renderToolbar();
      renderList();
    } catch (err) {
      if (err.name !== 'AbortError') mount(listEl, errorState(err, load));
    }
  }

  function renderHeader() {
    mount(headerEl, pageHeader({
      title: 'Projects', icon: 'folder2',
      subtitle: `${fmtNumber(projects.length)} project${projects.length === 1 ? '' : 's'} in ${entity}`,
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: createDialog }, icon('plus-lg', 'me-1'), 'New project')],
    }));
  }

  function renderToolbar() {
    mount(toolbarEl,
      entities.length > 1 && chips(entities.map((e) => ({ value: e, label: e })), entity, (v) => { entity = v || entities[0]; sync(); load(); }),
      h('div.af-search.mpj-search', icon('search', 'af-search-icon'), h('input.form-control.form-control-sm', {
        type: 'search', placeholder: 'Filter projects…', value: q, dataset: { searchShortcut: '' },
        oninput: debounce((e) => { q = e.target.value.trim(); sync(); renderList(); }, 100),
      })),
      h('select.form-select.form-select-sm.mpj-sort', { 'aria-label': 'Sort', onchange: (e) => { sort = e.target.value; sync(); renderList(); } },
        SORTS.map((s) => h('option', { value: s.value, selected: s.value === sort }, s.label))),
      h('div.mpj-view', ['cards', 'list'].map((v) => h('button.mpj-view-btn', {
        type: 'button', class: view === v && 'active', title: v === 'cards' ? 'Card view' : 'List view',
        onclick: () => { view = v; prefs.set('ms:projects:view', v); renderToolbar(); renderList(); },
      }, icon(v === 'cards' ? 'grid-3x2-gap' : 'list-ul')))));
  }

  function renderList() {
    const query = q.toLowerCase();
    const list = projects.filter((p) => !query || p.name.toLowerCase().includes(query) || (p.description || '').toLowerCase().includes(query))
      .sort(SORTS.find((s) => s.value === sort).fn);
    if (!projects.length) {
      mount(listEl, emptyState({ icon: 'folder-plus', title: 'No projects yet', message: 'Create one, or call wandb.init(project="…") from your training code.', action: h('button.btn.btn-sm.btn-primary', { onclick: createDialog }, 'New project') }));
      return;
    }
    if (!list.length) { mount(listEl, emptyState({ icon: 'search', title: 'No project matches', message: 'Try another search.' })); return; }
    if (view === 'list') {
      mount(listEl, h('section.af-card', h('div.af-list', list.map((p) => h('a.af-list-item', { href: urls.workspace(p.entityName, p.name) },
        h('span.mh-proj-icon', icon('folder2-open')),
        h('div.af-list-main', h('div.af-list-title', p.name), h('div.af-list-sub', p.description || 'No description')),
        h('span.ms-pill.ms-pill-muted', p.access || 'PRIVATE'),
        h('div.text-end.small.af-nowrap', { style: { minWidth: '110px' } }, h('div.fw-semibold', `${fmtNumber(p.totalRuns || 0)} runs`), h('div.text-muted', timeAgo(p.lastActive || p.updatedAt))))))));
      return;
    }
    mount(listEl, h('div.mpj-grid', list.map((p, i) => h('a.mpj-card', { href: urls.workspace(p.entityName, p.name), style: { animationDelay: `${Math.min(i, 12) * 25}ms` } },
      h('div.mpj-card-top', h('span.mpj-card-icon', { style: { background: hue(p.name) } }, p.name.slice(0, 2).toUpperCase()),
        h('span.ms-pill.ms-pill-muted.ms-auto', p.access || 'PRIVATE')),
      h('div.mpj-card-name', p.name),
      h('div.mpj-card-desc', p.description || 'No description'),
      h('div.mpj-card-foot',
        h('span', icon('play-circle', 'me-1'), `${fmtNumber(p.totalRuns || 0)} runs`),
        h('span', { title: fmtDate(p.lastActive || p.updatedAt) }, icon('clock', 'me-1'), timeAgo(p.lastActive || p.updatedAt)))))));
  }

  function createDialog() {
    formModal({
      title: 'New project', icon: 'folder-plus',
      fields: [
        { name: 'name', label: 'Name', required: true, placeholder: 'e.g. chest-xray-classifier', help: 'Letters, numbers, ".", "_" and "-".' },
        { name: 'description', label: 'Description', type: 'textarea', rows: 2 },
        { name: 'access', label: 'Visibility', type: 'select', value: 'PRIVATE', options: [{ value: 'PRIVATE', label: 'Private' }, { value: 'USER_READ', label: 'Team readable' }] },
      ],
      submitLabel: 'Create project',
      onSubmit: async ({ name, description, access }, close) => {
        if (!/^[\w.-]+$/.test(name)) throw new Error('Use only letters, numbers, ".", "_" and "-".');
        if (projects.some((p) => p.name === name)) throw new Error('A project with this name already exists.');
        await gql(`mutation C($input: UpsertModelInput!) { upsertModel(input: $input) { project { id name entityName } } }`,
          { input: { name, entityName: entity, description: description || null, access } });
        toast(`Created ${name}`, 'success');
        close();
        navigate(urls.workspace(entity, name));
      },
    });
  }

  await load();
}

function hue(name) {
  let x = 0;
  for (const c of name) x = (x * 31 + c.charCodeAt(0)) % 360;
  return `linear-gradient(135deg, hsl(${x} 70% 55%), hsl(${(x + 40) % 360} 70% 45%))`;
}
