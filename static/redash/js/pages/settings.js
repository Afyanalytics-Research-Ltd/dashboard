// Organization settings: grouped form with live format previews, conditional
// SAML/JWT sections, change tracking, a review-before-save diff, and a lock-out guard.
// Only changed keys are POSTed to /api/settings/organization.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, errorState, skeleton, modal, toast } from '../../../ui/js/ui.js';
import { rd } from '../core/api.js';
import { invalidate } from '../core/common.js';
import { fmtNumeral, fmtMoment } from '../core/format.js';

const DATE_FORMATS = ['DD/MM/YY', 'DD/MM/YYYY', 'MM/DD/YY', 'MM/DD/YYYY', 'YYYY-MM-DD', 'DD-MM-YYYY', 'DD MMM YYYY'];
const TIME_FORMATS = ['HH:mm', 'HH:mm:ss'];
const INT_FORMATS = ['0,0', '0', '0.0a', '0,0 a'];
const FLOAT_FORMATS = ['0,0.00', '0,0.[00]', '0.00', '0,0.0000', '0.0a'];

const SECTIONS = [
  {
    id: 'formats', title: 'Formats', icon: 'calendar3', sub: 'How dates and numbers display across Redash and Studio.',
    fields: [
      { key: 'date_format', label: 'Date format', type: 'choice', options: DATE_FORMATS, preview: (v) => fmtMoment(new Date(), v) },
      { key: 'time_format', label: 'Time format', type: 'choice', options: TIME_FORMATS, preview: (v) => fmtMoment(new Date(), v) },
      { key: 'integer_format', label: 'Integer format', type: 'choice', options: INT_FORMATS, free: true, preview: (v) => fmtNumeral(1234567, v) },
      { key: 'float_format', label: 'Decimal format', type: 'choice', options: FLOAT_FORMATS, free: true, preview: (v) => fmtNumeral(1234567.891, v) },
    ],
  },
  {
    id: 'features', title: 'Features', icon: 'toggles', sub: 'Organization-wide feature switches.',
    fields: [
      { key: 'feature_show_permissions_control', label: 'Show per-object permissions (ACL) controls', type: 'bool', help: 'Lets owners grant edit access on individual queries and dashboards.' },
      { key: 'send_email_on_failed_scheduled_queries', label: 'Email owners when a scheduled query fails', type: 'bool', help: 'Needs a mail server configured on Redash.' },
      { key: 'disable_public_urls', label: 'Disable public (shareable) dashboard links', type: 'bool', help: 'Existing public links stop working.' },
      { key: 'hide_plotly_mode_bar', label: 'Hide the chart toolbar (Plotly mode bar)', type: 'bool' },
      { key: 'multi_byte_search_enabled', label: 'Multi-byte search (for Chinese, Japanese, Korean text)', type: 'bool', help: 'Slower search, but matches non-Latin text correctly.' },
      { key: 'beacon_consent', label: 'Share anonymous usage statistics with the Redash project', type: 'bool' },
    ],
  },
  {
    id: 'auth', title: 'Authentication', icon: 'shield-lock', sub: 'How people sign in to Redash.',
    fields: [
      { key: 'auth_password_login_enabled', label: 'Password login', type: 'bool' },
      { key: 'auth_google_apps_domains', label: 'Google sign-in domains', type: 'list', help: 'Anyone with a Google account in these domains can sign in. Comma separated.', placeholder: 'afya.ai, example.org' },
      { key: 'auth_saml_enabled', label: 'SAML single sign-on', type: 'bool' },
      { key: 'auth_saml_type', label: 'SAML configuration', type: 'choice', options: ['static', 'dynamic'], when: (s) => s.auth_saml_enabled },
      { key: 'auth_saml_metadata_url', label: 'SAML metadata URL', type: 'text', when: (s) => s.auth_saml_enabled && s.auth_saml_type === 'dynamic' },
      { key: 'auth_saml_entity_id', label: 'SAML entity ID', type: 'text', when: (s) => s.auth_saml_enabled },
      { key: 'auth_saml_nameid_format', label: 'SAML NameID format', type: 'text', when: (s) => s.auth_saml_enabled },
      { key: 'auth_saml_sso_url', label: 'SAML SSO URL', type: 'text', when: (s) => s.auth_saml_enabled && s.auth_saml_type === 'static' },
      { key: 'auth_saml_x509_cert', label: 'SAML x509 certificate', type: 'textarea', when: (s) => s.auth_saml_enabled && s.auth_saml_type === 'static' },
      { key: 'auth_jwt_login_enabled', label: 'JWT login (behind an auth proxy)', type: 'bool' },
      { key: 'auth_jwt_auth_issuer', label: 'JWT issuer', type: 'text', when: (s) => s.auth_jwt_login_enabled },
      { key: 'auth_jwt_auth_public_certs_url', label: 'JWT public certs URL', type: 'text', when: (s) => s.auth_jwt_login_enabled },
      { key: 'auth_jwt_auth_audience', label: 'JWT audience', type: 'text', when: (s) => s.auth_jwt_login_enabled },
      { key: 'auth_jwt_auth_algorithms', label: 'JWT algorithms', type: 'list', when: (s) => s.auth_jwt_login_enabled },
      { key: 'auth_jwt_auth_cookie_name', label: 'JWT cookie name', type: 'text', when: (s) => s.auth_jwt_login_enabled },
      { key: 'auth_jwt_auth_header_name', label: 'JWT header name', type: 'text', when: (s) => s.auth_jwt_login_enabled },
    ],
  },
];

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
const show = (v) => (v == null || v === '' ? '—' : Array.isArray(v) ? (v.length ? v.join(', ') : '—') : typeof v === 'boolean' ? (v ? 'On' : 'Off') : String(v));

export default async function render(ctx) {
  const { el, signal } = ctx;
  mount(el, pageHeader({ title: 'Settings', icon: 'sliders' }), skeleton(8));
  let original;
  try {
    original = (await rd.get('/api/settings/organization', null, { signal })).settings;
  } catch (err) {
    if (err.name !== 'AbortError') mount(el, pageHeader({ title: 'Settings', icon: 'sliders' }), errorState(err));
    return;
  }
  if (!ctx.active) return;
  let draft = structuredClone(original);
  const sectionsEl = h('div.d-flex.flex-column.gap-3');
  const bar = h('div.rd-save-bar');

  const known = new Set(SECTIONS.flatMap((s) => s.fields.map((f) => f.key)));
  const other = Object.keys(original).filter((k) => !known.has(k));

  mount(el,
    pageHeader({
      title: 'Settings', icon: 'sliders',
      subtitle: 'Organization-wide Redash settings. Changes apply to everyone immediately after saving.',
    }),
    h('div.rd-settings-layout',
      h('nav.rd-settings-nav', SECTIONS.map((s) => h('a.rd-settings-nav-link', { href: `#rd-set-${s.id}`, onclick: (e) => { e.preventDefault(); document.getElementById(`rd-set-${s.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); } }, icon(s.icon), s.title))),
      sectionsEl),
    bar);

  const changes = () => Object.keys(draft).filter((k) => !same(draft[k], original[k]));

  function drawBar() {
    const c = changes();
    bar.classList.toggle('show', c.length > 0);
    mount(bar,
      h('span.rd-save-count', icon('pencil-square', 'me-1'), `${c.length} unsaved change${c.length === 1 ? '' : 's'}`),
      h('button.btn.btn-sm.btn-outline-light', { onclick: () => { draft = structuredClone(original); drawSections(); } }, 'Discard'),
      h('button.btn.btn-sm.btn-light', { onclick: review }, icon('eye', 'me-1'), 'Review & save'));
  }

  function fieldEl(f) {
    const id = `rd-set-f-${f.key}`;
    const v = draft[f.key];
    const changed = !same(v, original[f.key]);
    const set = (nv, redraw = false) => { draft[f.key] = nv; if (redraw) drawSections(); else { row.classList.toggle('changed', !same(nv, original[f.key])); drawBar(); if (f.preview) mount(previewEl, f.preview(nv)); } };
    let input;
    if (f.type === 'bool') {
      input = h('div.form-check.form-switch.m-0', h('input.form-check-input', { id, type: 'checkbox', checked: !!v, onchange: (e) => set(e.target.checked, true) }));
    } else if (f.type === 'choice') {
      const opts = [...new Set([...(f.options || []), ...(v ? [v] : [])])];
      if (f.free) {
        input = h('div.d-flex.gap-2',
          h('select.form-select.form-select-sm', { id, onchange: (e) => { if (e.target.value) { txt.value = e.target.value; set(e.target.value); } } },
            h('option', { value: '' }, 'Presets…'), opts.map((o) => h('option', { value: o, selected: o === v }, o))));
        const txt = h('input.form-control.form-control-sm.af-mono', { value: v ?? '', oninput: (e) => set(e.target.value) });
        input.appendChild(txt);
      } else {
        input = h('select.form-select.form-select-sm', { id, onchange: (e) => set(e.target.value, true) }, opts.map((o) => h('option', { value: o, selected: o === v }, o)));
      }
    } else if (f.type === 'list') {
      input = h('input.form-control.form-control-sm', { id, value: (v || []).join(', '), placeholder: f.placeholder, oninput: (e) => set(e.target.value.split(',').map((x) => x.trim()).filter(Boolean)) });
    } else if (f.type === 'textarea') {
      input = h('textarea.form-control.form-control-sm.af-mono', { id, rows: 4, oninput: (e) => set(e.target.value) }, v ?? '');
    } else {
      input = h('input.form-control.form-control-sm', { id, value: v ?? '', oninput: (e) => set(e.target.value) });
    }
    const previewEl = h('span.rda-set-preview', f.preview ? f.preview(v) : '');
    const row = h('div.rda-set-row', { class: [changed && 'changed', f.type === 'bool' && 'bool'] },
      h('label.rda-set-label', { for: id }, f.label, f.help && h('div.rda-set-help', f.help)),
      h('div.rda-set-control', input, f.preview && h('div.small.text-muted.mt-1', 'Preview: ', previewEl)),
      h('button.rda-set-revert', { type: 'button', title: `Revert to ${show(original[f.key])}`, onclick: () => { draft[f.key] = structuredClone(original[f.key]); drawSections(); } }, icon('arrow-counterclockwise')));
    return row;
  }

  function drawSections() {
    const scroll = window.scrollY;
    mount(sectionsEl,
      SECTIONS.map((s) => h('section', { id: `rd-set-${s.id}` }, card({
        title: s.title, icon: s.icon,
        body: h('div',
          h('p.small.text-muted.mb-2', s.sub),
          s.id === 'auth' && lockoutWarning(),
          h('div.rda-set-rows', s.fields.filter((f) => !f.when || f.when(draft)).filter((f) => f.key in draft || f.type === 'bool' || f.when).map(fieldEl))),
      }))),
      other.length && card({
        title: 'Other settings', icon: 'three-dots',
        body: h('div.rda-set-rows', other.map((k) => h('div.rda-set-row', h('label.rda-set-label', h('code', k)), h('div.rda-set-control', h('span.af-mono.small', show(original[k])))))),
      }));
    window.scrollTo({ top: scroll });
    drawBar();
  }

  function lockoutWarning() {
    const anyLogin = draft.auth_password_login_enabled || draft.auth_saml_enabled || draft.auth_jwt_login_enabled || (draft.auth_google_apps_domains || []).length;
    return !anyLogin && h('div.af-callout.af-callout-danger.mb-2', icon('exclamation-octagon-fill'),
      h('span', h('b', 'No sign-in method is enabled.'), ' Saving this would lock everyone out of Redash. Enable at least one.'));
  }

  function review() {
    const c = changes();
    const anyLogin = draft.auth_password_login_enabled || draft.auth_saml_enabled || draft.auth_jwt_login_enabled || (draft.auth_google_apps_domains || []).length;
    const label = (k) => SECTIONS.flatMap((s) => s.fields).find((f) => f.key === k)?.label || k;
    modal({
      title: `Save ${c.length} change${c.length === 1 ? '' : 's'}?`, icon: 'sliders', size: 'lg',
      body: h('div',
        !anyLogin && h('div.af-callout.af-callout-danger.mb-3', icon('exclamation-octagon-fill'), h('span', 'No sign-in method would remain enabled. Saving is blocked.')),
        h('table.af-table.af-table-dense', h('thead', h('tr', h('th', 'Setting'), h('th', 'Before'), h('th', 'After'))),
          h('tbody', c.map((k) => h('tr', h('td', label(k)), h('td.text-muted', show(original[k])), h('td.fw-semibold', show(draft[k]))))))),
      actions: [
        { label: 'Keep editing' },
        {
          label: 'Save settings', variant: 'primary', icon: 'check2', disabled: !anyLogin,
          onClick: async (close) => {
            const payload = Object.fromEntries(c.map((k) => [k, draft[k]]));
            const res = await rd.post('/api/settings/organization', payload);
            original = res.settings;
            draft = structuredClone(original);
            invalidate('orgsettings');
            toast('Settings saved', 'success');
            close();
            drawSections();
          },
        },
      ],
    });
  }

  const onUnload = (e) => { if (changes().length) { e.preventDefault(); e.returnValue = ''; } };
  window.addEventListener('beforeunload', onUnload);
  ctx.cleanup(() => window.removeEventListener('beforeunload', onUnload));
  drawSections();
}
