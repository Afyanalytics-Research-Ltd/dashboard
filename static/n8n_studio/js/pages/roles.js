// Roles: global & project roles with usage counts and scopes, custom role editor
// (scopes grouped by resource), and role-mapping rules for SSO.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, toast, modal, tabs, spinner } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { gated, emptyHint, licenseGate } from '../core/widgets.js';

export default async function render(ctx) {
  let tab = 'global';
  const body = h('div');
  const tabBar = tabs([{ id: 'global', label: 'Instance roles', icon: 'globe' }, { id: 'project', label: 'Project roles', icon: 'kanban' }], tab, (t) => { tab = t; refresh(); });
  mount(ctx.el, pageHeader({
    title: 'Roles', icon: 'person-badge', subtitle: 'Who can do what. Custom roles combine individual scopes.',
    actions: [
      h('button.btn.btn-sm.btn-outline-primary', { onclick: () => mappingRule() }, icon('diagram-2', 'me-1'), 'Role-mapping rule'),
      h('button.btn.btn-sm.btn-primary', { onclick: () => editRole() }, icon('plus-lg', 'me-1'), 'Custom role'),
    ],
  }), tabBar, body);

  let data = null;
  const refresh = () => gated(body, async () => {
    data = await n8n.get('/roles', { withUsageCount: 'true' }, { signal: ctx.signal });
    const list = data[tab] || [];
    return h('div.n8s-roles', list.length ? list.map((r) => h('div.n8s-role',
      h('div.n8s-role-head',
        h('div', h('div.fw-semibold', r.displayName, r.systemRole ? h('span.badge.text-bg-light.ms-2', 'system') : h('span.badge.text-bg-info.ms-2', 'custom'), r.licensed === false && h('span.badge.text-bg-warning.ms-1', 'unlicensed')), h('code.small', r.slug)),
        h('div.small.text-muted.ms-auto', `${r.usedByUsers ?? 0} users${r.usedByProjects != null ? ` · ${r.usedByProjects} projects` : ''}`),
        !r.systemRole && h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => editRole(r) }, icon('pencil'))),
      h('details', h('summary.small', `${r.scopes?.length || 0} scopes`), scopeGroups(r.scopes || [])))) : emptyHint('person-badge', 'No roles'));
  });

  function scopeGroups(scopes) {
    const g = new Map();
    for (const s of scopes) { const [res, act] = s.split(':'); if (!g.has(res)) g.set(res, []); g.get(res).push(act); }
    return h('div.n8s-scopes', [...g.entries()].sort().map(([res, acts]) => h('div.n8s-scope-row', h('span.n8s-scope-res', res), acts.map((a) => h('span.n8s-scope', a)))));
  }

  async function editRole(role) {
    const m = modal({ title: role ? `Edit ${role.displayName}` : 'New custom role', icon: 'person-badge', size: 'xl', body: spinner() });
    let available = [];
    try { available = (await n8n.get('/roles', {})); } catch (err) { mount(m.body, err.license ? licenseGate(err) : err.message); return; }
    const allScopes = [...new Set([...(available.global || []), ...(available.project || [])].flatMap((r) => r.scopes || []))].sort();
    const chosen = new Set(role?.scopes || []);
    const name = h('input.form-control.form-control-sm', { value: role?.displayName || '', placeholder: 'e.g. Workflow reviewer' });
    const type = h('select.form-select.form-select-sm', { disabled: !!role }, ['project', 'global'].map((t) => h('option', { selected: (role?.roleType || tab) === t }, t)));
    const filter = h('input.form-control.form-control-sm', { type: 'search', placeholder: 'Filter scopes…' });
    const grid = h('div.n8s-scope-pick');
    const draw = () => {
      const g = new Map();
      for (const s of allScopes.filter((x) => !filter.value || x.includes(filter.value))) { const [res] = s.split(':'); if (!g.has(res)) g.set(res, []); g.get(res).push(s); }
      mount(grid, [...g.entries()].map(([res, list]) => h('div.n8s-scope-group',
        h('label.n8s-scope-res', h('input.form-check-input.me-1', { type: 'checkbox', checked: list.every((s) => chosen.has(s)), onchange: (e) => { list.forEach((s) => (e.target.checked ? chosen.add(s) : chosen.delete(s))); draw(); } }), res),
        list.map((s) => h('label.n8s-scope-opt', h('input.form-check-input', { type: 'checkbox', checked: chosen.has(s), onchange: (e) => { e.target.checked ? chosen.add(s) : chosen.delete(s); } }), s.split(':')[1])))));
    };
    filter.addEventListener('input', draw);
    draw();
    mount(m.body, h('div', h('div.row.g-2.mb-2', h('div.col-md-8', h('label.small.fw-semibold', 'Name'), name), h('div.col-md-4', h('label.small.fw-semibold', 'Applies to'), type)), filter, grid));
    m.el.appendChild(h('div.af-modal-footer',
      h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => m.close() }, 'Cancel'),
      h('button.btn.btn-sm.btn-primary', {
        onclick: async () => {
          try {
            if (role) await n8n.put(`/roles/${enc(role.slug)}`, { displayName: name.value, scopes: [...chosen] });
            else await n8n.post('/roles', { displayName: name.value, roleType: type.value, scopes: [...chosen] });
            toast('Role saved', 'success');
            m.close();
            refresh();
          } catch (err) { toast(err.message, 'error', { title: err.license ? 'Needs licence' : 'Save failed' }); }
        },
      }, 'Save role')));
  }

  function mappingRule() {
    const expr = h('input.form-control.form-control-sm', { placeholder: 'e.g. $claims.groups.includes("n8n-admins")' });
    const role = h('input.form-control.form-control-sm', { placeholder: 'global:admin or project:editor' });
    const type = h('select.form-select.form-select-sm', h('option', 'instance'), h('option', 'project'));
    const order = h('input.form-control.form-control-sm', { type: 'number', value: 0 });
    const projectIds = h('input.form-control.form-control-sm', { placeholder: 'project ids, comma-separated (project rules)' });
    modal({
      title: 'New role-mapping rule', icon: 'diagram-2',
      body: h('div', h('p.small.text-muted', 'Maps SSO claims to n8n roles at login (SAML/OIDC).'), h('label.small.fw-semibold', 'Expression'), expr, h('label.small.fw-semibold.mt-2', 'Role'), role, h('div.row.g-2.mt-1', h('div.col', h('label.small.fw-semibold', 'Type'), type), h('div.col', h('label.small.fw-semibold', 'Order'), order)), h('label.small.fw-semibold.mt-2', 'Projects'), projectIds),
      actions: [{ label: 'Cancel' }, {
        label: 'Create rule', variant: 'primary',
        onClick: async (close) => {
          await n8n.post('/role-mapping-rules', { expression: expr.value, role: role.value, type: type.value, order: Number(order.value), projectIds: projectIds.value.split(',').map((x) => x.trim()).filter(Boolean) });
          toast('Rule created', 'success');
          close();
        },
      }],
    });
  }

  refresh();
}

