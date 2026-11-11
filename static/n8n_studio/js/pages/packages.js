// Community packages: list with nodes & update availability, install (npm name,
// version, verified), update, uninstall — plus which workflows use each package.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, toast, formModal, timeAgo } from '../../../ui/js/ui.js';
import { n8n, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { load } from '../core/store.js';
import { loadCatalog } from '../core/catalog.js';
import { gated, emptyHint, confirmDelete } from '../core/widgets.js';

export default async function render(ctx) {
  const body = h('div');
  mount(ctx.el, pageHeader({
    title: 'Community packages', icon: 'box-seam', subtitle: 'npm packages that add nodes to this instance.',
    actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => install() }, icon('cloud-download', 'me-1'), 'Install')],
  }), body);

  const refresh = () => gated(body, async () => {
    const [pkgs, workflows] = await Promise.all([n8n.get('/community-packages', undefined, { signal: ctx.signal }), load('workflows').catch(() => [])]);
    const list = Array.isArray(pkgs) ? pkgs : pkgs?.data || [];
    if (!list.length) return emptyHint('box-seam', 'No community packages installed', h('span', 'Browse ', h('a', { href: 'https://www.npmjs.com/search?q=keywords:n8n-community-node-package', target: '_blank', rel: 'noopener' }, 'n8n community nodes on npm'), '.'));
    return h('div.n8s-pkgs', list.map((p) => {
      const types = new Set((p.installedNodes || []).map((n) => n.type));
      const users = workflows.filter((w) => (w.nodes || []).some((n) => types.has(n.type)));
      return h('div.n8s-pkg', { class: p.failedLoading && 'bad' },
        h('div.n8s-pkg-head',
          h('div', h('a.fw-semibold', { href: `https://www.npmjs.com/package/${p.packageName}`, target: '_blank', rel: 'noopener' }, p.packageName), h('span.badge.text-bg-light.ms-2', `v${p.installedVersion}`),
            p.updateAvailable && h('span.badge.text-bg-warning.ms-1', `update → ${p.updateAvailable}`), p.failedLoading && h('span.badge.text-bg-danger.ms-1', 'failed to load')),
          h('div.ms-auto.d-flex.gap-1',
            p.updateAvailable && h('button.btn.btn-sm.btn-warning', { onclick: () => update(p, p.updateAvailable) }, icon('arrow-up-circle', 'me-1'), 'Update'),
            h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => update(p) }, icon('arrow-repeat'), ' Version…'),
            h('button.btn.btn-sm.btn-outline-danger', { onclick: () => uninstall(p, users) }, icon('trash')))),
        h('div.small.text-muted', p.authorName && `by ${p.authorName}${p.authorEmail ? ` <${p.authorEmail}>` : ''} · `, 'installed ', timeAgo(p.createdAt)),
        h('div.n8s-pkg-nodes', (p.installedNodes || []).map((n) => h('span.n8s-chip-sm', n.name, n.latestVersion && ` v${n.latestVersion}`))),
        h('div.small', users.length ? ['Used by ', users.slice(0, 6).map((w, i) => [i ? ', ' : '', h('a', { href: urls.workflow(w.id) }, w.name)]), users.length > 6 && ` +${users.length - 6}`] : h('span.text-muted', 'Not used by any workflow')));
    }));
  });

  function install() {
    formModal({
      title: 'Install community package', icon: 'cloud-download',
      fields: [
        { name: 'name', label: 'npm package', required: true, placeholder: 'n8n-nodes-…', help: 'Only install packages you trust — they run code inside n8n.' },
        { name: 'version', label: 'Version', placeholder: 'latest' },
        { name: 'verify', label: '', type: 'checkbox', value: true, checkLabel: 'Require a verified package' },
      ],
      submitLabel: 'Install',
      onSubmit: async (v, close) => {
        const r = await n8n.post('/community-packages', { name: v.name.trim(), ...(v.version ? { version: v.version.trim() } : {}), verify: v.verify });
        toast(`Installed ${r.packageName} ${r.installedVersion} (${(r.installedNodes || []).length} nodes)`, 'success');
        await loadCatalog({ refresh: true }).catch(() => null);
        close();
        refresh();
      },
    });
  }

  function update(p, version) {
    formModal({
      title: `Update ${p.packageName}`, icon: 'arrow-up-circle', size: 'sm',
      fields: [{ name: 'version', label: 'Version', value: version || '', placeholder: 'latest' }, { name: 'verify', label: '', type: 'checkbox', value: true, checkLabel: 'Verified only' }],
      submitLabel: 'Update',
      onSubmit: async (v, close) => {
        const r = await n8n.patch(`/community-packages/${enc(p.packageName)}`, { ...(v.version ? { version: v.version } : {}), verify: v.verify });
        toast(`${r.packageName} is now ${r.installedVersion}`, 'success');
        close();
        refresh();
      },
    });
  }

  async function uninstall(p, users) {
    if (!(await confirmDelete('package', p.packageName, { details: users.length ? h('div.small.text-danger', `${users.length} workflow(s) use its nodes and will break.`) : null }))) return;
    await n8n.del(`/community-packages/${enc(p.packageName)}`).then(() => toast('Uninstalled', 'success')).catch((err) => toast(err.message, 'error'));
    refresh();
  }

  refresh();
}
