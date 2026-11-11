// Instance settings: security policy, SAML, OIDC, LDAP (+ sync history / run),
// OpenTelemetry (+ test trace) and source-control pull. Each tab is licence-aware.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, tabs, toast, card, buildForm, dataTable, timeAgo, confirmDialog, chips } from '../../../ui/js/ui.js';
import { n8n } from '../core/api.js';
import { gated } from '../core/widgets.js';

const TABS = [
  { id: 'security', label: 'Security policy', icon: 'shield-lock' },
  { id: 'saml', label: 'SAML SSO', icon: 'person-badge' },
  { id: 'oidc', label: 'OIDC SSO', icon: 'box-arrow-in-right' },
  { id: 'ldap', label: 'LDAP', icon: 'diagram-2' },
  { id: 'otel', label: 'OpenTelemetry', icon: 'activity' },
  { id: 'git', label: 'Source control', icon: 'git' },
];

export default async function render(ctx) {
  let tab = ctx.query.get('tab') || 'security';
  const body = h('div');
  mount(ctx.el, pageHeader({ title: 'Instance settings', icon: 'sliders', subtitle: 'Everything n8n exposes for administrators through its API.' }),
    tabs(TABS, tab, (t) => { tab = t; ctx.setQuery({ tab: t }); show(); }), body);

  /** Generic GET-form-PUT card from a field spec. */
  function settingsCard({ path, fields, title, iconName, extra, normalize = (v) => v, actions = [] }) {
    return gated(body, async () => {
      const current = await n8n.get(path, undefined, { signal: ctx.signal });
      const form = buildForm(fields(current));
      const saveBtn = h('button.btn.btn-sm.btn-primary', {
        onclick: async () => {
          try {
            const values = normalize(form.values(), current);
            await n8n.put(path, values);
            toast('Settings saved', 'success');
          } catch (err) { toast(err.message, 'error', { title: 'Save failed' }); }
        },
      }, icon('cloud-check', 'me-1'), 'Save');
      return h('div', card({ title, icon: iconName, actions: h('div.d-flex.gap-2', actions.map((a) => a(form, current)), saveBtn), body: form.el }), extra && await extra(current));
    });
  }

  const bool = (name, label, value, help) => ({ name, label: '', type: 'checkbox', value, checkLabel: label, help });

  function show() {
    if (tab === 'security') {
      return settingsCard({
        path: '/settings/security-policy', title: 'Personal space policy', iconName: 'shield-lock',
        fields: (c) => [
          bool('personalSpacePublishing', 'Allow publishing (activating) workflows in personal spaces', c.personalSpacePublishing, `${c.publishedPersonalWorkflowsCount ?? 0} personal workflows are published today.`),
          bool('personalSpaceSharing', 'Allow sharing personal workflows & credentials', c.personalSpaceSharing, `${c.sharedPersonalWorkflowsCount ?? 0} workflows and ${c.sharedPersonalCredentialsCount ?? 0} credentials are shared today.`),
          { name: 'floor', label: 'Execution data redaction floor', type: 'select', value: c.redactionEnforcement?.floor || 'none', options: ['none', 'manual', 'production', 'all'], help: 'Minimum redaction every workflow must apply to stored execution data.' },
        ],
        normalize: (v) => ({ personalSpacePublishing: v.personalSpacePublishing, personalSpaceSharing: v.personalSpaceSharing, redactionEnforcement: { floor: v.floor } }),
      });
    }
    if (tab === 'saml') {
      return settingsCard({
        path: '/settings/sso/saml', title: 'SAML', iconName: 'person-badge',
        fields: (c) => [
          bool('loginEnabled', 'Enable SAML login', c.loginEnabled),
          { name: 'loginLabel', label: 'Login button label', value: c.loginLabel },
          { type: 'static', name: 'entityID', label: 'Entity ID (give to your IdP)', value: c.entityID },
          { type: 'static', name: 'returnUrl', label: 'ACS / return URL', value: c.returnUrl },
          { name: 'metadataUrl', label: 'IdP metadata URL', value: c.metadataUrl },
          { name: 'metadata', label: 'IdP metadata XML (if no URL)', type: 'textarea', rows: 5, value: c.metadata },
          { name: 'loginBinding', label: 'Login binding', type: 'select', value: c.loginBinding, options: ['redirect', 'post'] },
          { name: 'acsBinding', label: 'ACS binding', type: 'select', value: c.acsBinding, options: ['post', 'redirect'] },
          bool('ignoreSSL', 'Ignore SSL errors when fetching metadata', c.ignoreSSL),
          bool('authnRequestsSigned', 'Sign authn requests', c.authnRequestsSigned),
          bool('wantAssertionsSigned', 'Require signed assertions', c.wantAssertionsSigned),
          bool('wantMessageSigned', 'Require signed messages', c.wantMessageSigned),
          { name: 'relayState', label: 'Relay state', value: c.relayState },
          { name: 'mapping', label: 'Attribute mapping', type: 'json', value: c.mapping, rows: 8 },
          { name: 'signatureConfig', label: 'Signature config', type: 'json', value: c.signatureConfig, rows: 4 },
          { name: 'signingCertificate', label: 'Signing certificate', type: 'textarea', rows: 3, value: c.signingCertificate },
          { name: 'signingPrivateKey', label: 'Signing private key', type: 'textarea', rows: 3, value: c.signingPrivateKey },
        ],
        normalize: (v, c) => ({ ...c, ...v }),
      });
    }
    if (tab === 'oidc') {
      return settingsCard({
        path: '/settings/sso/oidc', title: 'OpenID Connect', iconName: 'box-arrow-in-right',
        fields: (c) => [
          bool('loginEnabled', 'Enable OIDC login', c.loginEnabled),
          { name: 'discoveryEndpoint', label: 'Discovery endpoint', value: c.discoveryEndpoint, placeholder: 'https://idp/.well-known/openid-configuration' },
          { name: 'clientId', label: 'Client ID', value: c.clientId },
          { name: 'clientSecret', label: 'Client secret', type: 'password', value: c.clientSecret },
          { name: 'prompt', label: 'Prompt', type: 'select', value: c.prompt, options: ['none', 'login', 'consent', 'select_account', 'create'] },
          { name: 'additionalScopes', label: 'Additional scopes', value: c.additionalScopes },
          { name: 'authenticationContextClassReference', label: 'ACR values', type: 'tags', value: c.authenticationContextClassReference || [] },
          bool('emailVerifiedRequired', 'Require verified email', c.emailVerifiedRequired),
          bool('rpInitiatedLogoutEnabled', 'RP-initiated logout', c.rpInitiatedLogoutEnabled),
        ],
      });
    }
    if (tab === 'ldap') {
      return settingsCard({
        path: '/settings/ldap', title: 'LDAP', iconName: 'diagram-2',
        fields: (c) => [
          bool('loginEnabled', 'Enable LDAP login', c.loginEnabled),
          { name: 'loginLabel', label: 'Login label', value: c.loginLabel },
          { name: 'connectionUrl', label: 'Server', value: c.connectionUrl },
          { name: 'connectionPort', label: 'Port', type: 'number', value: c.connectionPort },
          { name: 'connectionSecurity', label: 'Security', type: 'select', value: c.connectionSecurity, options: ['none', 'tls', 'startTls'] },
          bool('allowUnauthorizedCerts', 'Allow self-signed certificates', c.allowUnauthorizedCerts),
          { name: 'baseDn', label: 'Base DN', value: c.baseDn },
          { name: 'bindingAdminDn', label: 'Bind DN', value: c.bindingAdminDn },
          { name: 'bindingAdminPassword', label: 'Bind password', type: 'password', value: c.bindingAdminPassword },
          { name: 'userFilter', label: 'User filter', value: c.userFilter },
          { type: 'section', label: 'Attribute mapping' },
          { name: 'emailAttribute', label: 'Email', value: c.emailAttribute },
          { name: 'loginIdAttribute', label: 'Login id', value: c.loginIdAttribute },
          { name: 'ldapIdAttribute', label: 'LDAP id', value: c.ldapIdAttribute },
          { name: 'firstNameAttribute', label: 'First name', value: c.firstNameAttribute },
          { name: 'lastNameAttribute', label: 'Last name', value: c.lastNameAttribute },
          { type: 'section', label: 'Synchronisation' },
          bool('synchronizationEnabled', 'Enable periodic sync', c.synchronizationEnabled),
          { name: 'synchronizationInterval', label: 'Interval (minutes)', type: 'number', value: c.synchronizationInterval },
          { name: 'searchPageSize', label: 'Page size', type: 'number', value: c.searchPageSize },
          { name: 'searchTimeout', label: 'Timeout (s)', type: 'number', value: c.searchTimeout },
          bool('enforceEmailUniqueness', 'Enforce email uniqueness', c.enforceEmailUniqueness),
        ],
        actions: [() => h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => syncLdap('dry') }, icon('eye', 'me-1'), 'Dry-run sync'), () => h('button.btn.btn-sm.btn-outline-primary', { onclick: () => syncLdap('live') }, icon('arrow-repeat', 'me-1'), 'Sync now')],
        extra: async () => {
          const hist = await n8n.get('/settings/ldap/sync', { limit: 25 }).catch(() => ({ data: [] }));
          return card({
            title: 'Sync history', icon: 'clock-history',
            body: dataTable({ rows: hist.data || [], dense: true, columns: [{ key: 'startedAt', label: 'Started', render: (r) => timeAgo(r.startedAt) }, { key: 'runMode', label: 'Mode' }, { key: 'status', label: 'Status' }, { key: 'scanned', label: 'Scanned' }, { key: 'created', label: 'Created' }, { key: 'updated', label: 'Updated' }, { key: 'disabled', label: 'Disabled' }, { key: 'error', label: 'Error' }] }),
          });
        },
      });
    }
    if (tab === 'otel') {
      return settingsCard({
        path: '/settings/otel', title: 'OpenTelemetry tracing', iconName: 'activity',
        fields: (c) => [
          bool('enabled', 'Export traces', c.enabled),
          { name: 'exporterEndpoint', label: 'OTLP endpoint', value: c.exporterEndpoint },
          { name: 'exporterTracingPath', label: 'Tracing path', value: c.exporterTracingPath },
          { name: 'exporterServiceName', label: 'Service name', value: c.exporterServiceName },
          { name: 'exporterHeaders', label: 'Headers', value: c.exporterHeaders, placeholder: 'key=value,key2=value2' },
          { name: 'tracesSampleRate', label: 'Sample rate (0–1)', type: 'number', step: 0.01, min: 0, max: 1, value: c.tracesSampleRate },
          { name: 'startupConnectivityTimeoutMs', label: 'Startup connectivity timeout (ms)', type: 'number', value: c.startupConnectivityTimeoutMs },
          bool('includeNodeSpans', 'Include a span per node', c.includeNodeSpans),
          bool('injectOutbound', 'Propagate trace context on outbound HTTP', c.injectOutbound),
          bool('productionExecutionsOnly', 'Production executions only', c.productionExecutionsOnly),
        ],
        actions: [(form) => h('button.btn.btn-sm.btn-outline-secondary', {
          onclick: async () => {
            const v = form.values();
            try {
              const r = await n8n.post('/settings/otel/test-trace', { exporterEndpoint: v.exporterEndpoint, exporterTracingPath: v.exporterTracingPath, exporterServiceName: v.exporterServiceName, exporterHeaders: v.exporterHeaders, startupConnectivityTimeoutMs: v.startupConnectivityTimeoutMs });
              toast(r.success ? 'Collector reachable — test trace sent' : `Test failed: ${r.error}`, r.success ? 'success' : 'error');
            } catch (err) { toast(err.message, 'error'); }
          },
        }, icon('send', 'me-1'), 'Send test trace')],
      });
    }
    if (tab === 'git') return sourceControl();
    return null;
  }

  async function syncLdap(type) {
    if (type === 'live' && !(await confirmDialog({ title: 'Run LDAP sync?', message: 'Creates, updates and disables n8n users to match the directory.' }))) return;
    try {
      const r = await n8n.post('/settings/ldap/sync', { type });
      toast(`Sync ${r.status}: ${r.scanned} scanned, ${r.created} created, ${r.updated} updated, ${r.disabled} disabled${r.error ? ` — ${r.error}` : ''}`, r.error ? 'error' : 'success', { timeout: 10000 });
      show();
    } catch (err) { toast(err.message, 'error'); }
  }

  function sourceControl() {
    let autoPublish = 'none';
    const force = h('input.form-check-input', { type: 'checkbox' });
    const out = h('div');
    mount(body, card({
      title: 'Pull from the connected Git repository', icon: 'git',
      body: h('div',
        h('p.small.text-muted', 'Brings workflows, credentials (stubs), variables, tags and folders from the remote branch into this instance. Configure the repository in n8n → Settings → Environments.'),
        h('div.d-flex.gap-3.align-items-center.flex-wrap',
          h('span.small.fw-semibold', 'After pull, publish:'), chips([{ value: 'none', label: 'nothing' }, { value: 'published', label: 'what was published remotely' }, { value: 'all', label: 'everything' }], autoPublish, (v) => { autoPublish = v || 'none'; }),
          h('label.n8s-chk', force, 'Force (overwrite local changes)'),
          h('button.btn.btn-sm.btn-primary', {
            onclick: async () => {
              if (force.checked && !(await confirmDialog({ title: 'Force pull?', danger: true, confirmLabel: 'Overwrite', message: 'Local changes to pulled items are discarded.' }))) return;
              mount(out, h('div.small.text-muted.mt-2', 'Pulling…'));
              try {
                const r = await n8n.post('/source-control/pull', { force: force.checked, autoPublish });
                mount(out, dataTable({ rows: r || [], dense: true, rowKey: (x) => `${x.type}:${x.id}`, empty: h('div.small.text-success.mt-2', 'Already up to date.'), columns: [{ key: 'type', label: 'Type' }, { key: 'name', label: 'Name' }, { key: 'status', label: 'Status' }, { key: 'conflict', label: 'Conflict', render: (x) => (x.conflict ? h('span.text-danger', 'yes') : '') }, { key: 'publishingError', label: 'Publishing', render: (x) => x.publishingError || (x.isLocalPublished ? 'published' : '') }] }));
                toast('Pull complete', 'success');
              } catch (err) { mount(out, h('div.text-danger.mt-2', err.message)); }
            },
          }, icon('cloud-arrow-down', 'me-1'), 'Pull')),
        out),
    }));
  }

  show();
}
