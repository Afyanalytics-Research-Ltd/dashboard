// Groups: cards with member + data-source counts and permissions, instant search,
// create, delete (built-in groups are protected).

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, emptyState, errorState, skeleton, searchInput, formModal, toast, toastError, confirmDialog } from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { matches, hl, pill, groupDataSources } from './admin_shared.js';

export function createGroup() {
  return new Promise((resolve) => {
    formModal({
      title: 'New group', icon: 'diagram-2', submitLabel: 'Create group',
      subtitle: 'Then add members and the data sources they may use.',
      fields: [{ name: 'name', label: 'Group name', required: true, placeholder: 'e.g. ksh:finance' }],
      onSubmit: async ({ name }, close) => {
        const g = await rd.post('/api/groups', { name: name.trim() });
        toast(`Group ${g.name} created`, 'success');
        close();
        resolve(g);
      },
    });
  });
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  let q = ctx.query.get('q') || '';
  let groups = [];
  const counts = new Map(); // id -> {members, sources}
  const grid = h('div.rd-groups-grid', skeleton(4, { height: 60 }));

  mount(el,
    pageHeader({
      title: 'Groups', icon: 'diagram-2',
      subtitle: 'Groups bundle permissions and data source access. A user can belong to several groups.',
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => createGroup().then((g) => g && navigate(urls.group(g.id))) }, icon('plus-lg', 'me-1'), 'New group')],
    }),
    h('div.af-toolbar', searchInput({ placeholder: 'Search groups…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } })),
    grid);

  async function load() {
    try {
      groups = await rd.get('/api/groups', null, { signal });
      if (!ctx.active) return;
      draw();
      // Counts load lazily, a few groups at a time.
      const queue = [...groups];
      const worker = async () => {
        while (queue.length && ctx.active) {
          const g = queue.shift();
          try {
            const [m, d] = await Promise.all([
              rd.get(`/api/groups/${enc(g.id)}/members`, null, { signal }),
              groupDataSources(g.id, { signal }),
            ]);
            counts.set(g.id, { members: m.length, sources: d.sources.length });
          } catch { counts.set(g.id, { members: '?', sources: '?' }); }
          if (ctx.active) draw();
        }
      };
      await Promise.all([worker(), worker(), worker()]);
    } catch (err) {
      if (err.name !== 'AbortError') mount(grid, errorState(err, load));
    }
  }

  function draw() {
    const rows = groups.filter((g) => matches(g, q, ['name', (x) => (x.permissions || []).join(' ')]))
      .sort((a, b) => (a.type === 'builtin' ? -1 : 0) - (b.type === 'builtin' ? -1 : 0) || a.name.localeCompare(b.name));
    if (!rows.length) { mount(grid, emptyState({ icon: 'search', title: q ? 'No groups match' : 'No groups' })); return; }
    mount(grid, rows.map((g) => {
      const c = counts.get(g.id);
      return h('a.rd-group-card', { href: urls.group(g.id) },
        h('div.rd-group-card-head',
          h('span.rd-group-icon', { class: g.name === 'admin' && 'admin' }, icon(g.name === 'admin' ? 'shield-fill' : g.type === 'builtin' ? 'people-fill' : 'diagram-2-fill')),
          h('div.rd-group-card-title', h('div.fw-semibold', hl(g.name, q)), g.type === 'builtin' ? pill('built-in', 'muted') : h('span.small.text-muted', `created `, new Date(g.created_at).toLocaleDateString())),
          g.type !== 'builtin' && h('button.btn.btn-sm.btn-light.rd-group-del', {
            title: 'Delete group',
            onclick: (e) => { e.preventDefault(); e.stopPropagation(); remove(g); },
          }, icon('trash'))),
        h('div.rd-group-stats',
          h('span', icon('person', 'me-1'), c ? `${c.members} member${c.members === 1 ? '' : 's'}` : h('span.rd-shimmer-text', '… members')),
          h('span', icon('database', 'me-1'), c ? `${c.sources} data source${c.sources === 1 ? '' : 's'}` : h('span.rd-shimmer-text', '… sources'))),
        h('div.rd-chip-row', (g.permissions || []).map((p) => h('span.rd-perm', p.replace(/_/g, ' ')))));
    }));
  }

  async function remove(g) {
    const ok = await confirmDialog({ title: `Delete group ${g.name}?`, danger: true, confirmLabel: 'Delete group', message: 'Members lose any data source access that came only through this group.' });
    if (!ok) return;
    try {
      await rd.del(`/api/groups/${enc(g.id)}`);
      groups = groups.filter((x) => x.id !== g.id);
      toast('Group deleted', 'success');
      draw();
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  await load();
}
