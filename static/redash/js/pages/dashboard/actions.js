// Dashboard actions with their dialogs: duplicate, share, text widgets, parameter mappings.

import { h, icon, mount, debounce } from '../../../../ui/js/dom.js';
import { modal, toast, toastError, copyButton, confirmDialog, spinner } from '../../../../ui/js/ui.js';
import { rd, enc } from '../../core/api.js';
import { invalidate, getSession } from '../../core/common.js';
import { loadMarkdown } from '../../core/lib.js';
import { bottomRow } from '../../core/add_to_dashboard.js';

/** Copy a dashboard and every widget (visualizations, text boxes, positions, mappings). */
export async function duplicateDashboard(dashboard, { name } = {}) {
  const status = h('div.rdd-dup-status', spinner('sm'), ' Preparing…');
  const progress = h('div.af-meter', h('span', { style: { width: '0%', background: 'var(--af-primary)', transition: 'width .2s' } }));
  const m = modal({ title: 'Duplicating dashboard', icon: 'copy', size: 'sm', persistent: true, body: h('div.d-flex.flex-column.gap-2', status, progress) });
  try {
    const src = dashboard.widgets ? dashboard : await rd.get(`/api/dashboards/${enc(dashboard.id)}`);
    const copy = await rd.post('/api/dashboards', { name: name || `Copy of ${src.name}` });
    const widgets = src.widgets || [];
    let done = 0;
    for (const w of widgets) {
      mount(status, spinner('sm'), ` Copying widget ${done + 1} of ${widgets.length}…`);
      await rd.post('/api/widgets', {
        dashboard_id: copy.id,
        visualization_id: w.visualization?.id || null,
        text: w.text || '',
        width: w.width || 1,
        options: w.options || {},
      });
      done++;
      progress.firstChild.style.width = `${(done / widgets.length) * 100}%`;
    }
    const final = await rd.post(`/api/dashboards/${enc(copy.id)}`, {
      tags: src.tags || [], dashboard_filters_enabled: !!src.dashboard_filters_enabled, options: src.options || {},
    });
    invalidate('dash');
    m.close();
    toast(`Created “${final.name}” with ${widgets.length} widget${widgets.length === 1 ? '' : 's'}`, 'success');
    return final;
  } catch (err) {
    m.close();
    toastError(err, 'Duplicate failed');
    return null;
  }
}

/** Public sharing dialog (enable / copy / disable). onChange(dashboard) after a change. */
export async function shareDialog(dashboard, { onChange } = {}) {
  const session = await getSession().catch(() => null);
  const disabled = session?.client_config?.disablePublicUrls;
  const body = h('div.rdd-share');
  const draw = () => {
    if (disabled) {
      mount(body, h('div.af-callout.af-callout-warn', icon('lock'), h('span', 'Public URLs are disabled for this Redash organization.')));
      return;
    }
    const on = !!dashboard.public_url;
    const toggle = h('input.form-check-input', { type: 'checkbox', checked: on, id: 'rdd-share-toggle', role: 'switch' });
    toggle.addEventListener('change', async () => {
      toggle.disabled = true;
      try {
        if (toggle.checked) {
          const res = await rd.post(`/api/dashboards/${enc(dashboard.id)}/share`, {});
          dashboard.public_url = res.public_url;
          dashboard.api_key = res.api_key;
          toast('Public link enabled', 'success');
        } else {
          const ok = await confirmDialog({ title: 'Disable the public link?', message: 'Anyone using the current link loses access immediately.', confirmLabel: 'Disable link', danger: true });
          if (!ok) { toggle.checked = true; toggle.disabled = false; return; }
          await rd.del(`/api/dashboards/${enc(dashboard.id)}/share`);
          dashboard.public_url = null;
          dashboard.api_key = null;
          toast('Public link disabled', 'success');
        }
        onChange?.(dashboard);
      } catch (err) {
        toggle.checked = !toggle.checked;
        toastError(err, 'Sharing failed');
      }
      draw();
    });
    mount(body,
      h('div.form-check.form-switch.mb-3', toggle, h('label.form-check-label.fw-semibold', { for: 'rdd-share-toggle' }, 'Allow access with a secret link (no login needed)')),
      on
        ? h('div.rdd-share-link',
          h('input.form-control.form-control-sm.af-mono', { readonly: true, value: dashboard.public_url, onclick: (e) => e.target.select() }),
          copyButton(dashboard.public_url, 'Copy'),
          h('a.btn.btn-sm.btn-outline-secondary', { href: dashboard.public_url, target: '_blank', rel: 'noopener' }, icon('box-arrow-up-right')))
        : h('div.rdv-help', 'When enabled, anyone with the link can view this dashboard and its data. Parameters stay editable by viewers.'),
      on && h('div.af-callout.af-callout-warn.mt-3', icon('shield-exclamation'), h('span', 'Treat this link like a password — it exposes the dashboard data to anyone who has it.')));
  };
  draw();
  modal({ title: 'Share dashboard', subtitle: dashboard.name, icon: 'share', body, actions: [{ label: 'Done', variant: 'primary' }] });
}

/** Markdown editor with live preview for text widgets. Resolves with the text or null. */
export async function editTextDialog(initial = '') {
  const md = await loadMarkdown();
  const ta = h('textarea.form-control.af-mono.rdd-text-input', { rows: 12, placeholder: '# Section title\n\nWrite **markdown** — headings starting with # become jump links in the dashboard navigation.' }, initial);
  const preview = h('div.rdd-text-preview.af-markdown');
  const draw = () => { preview.innerHTML = ta.value.trim() ? md(ta.value) : '<p class="text-muted">Preview appears here.</p>'; };
  ta.addEventListener('input', debounce(draw, 120));
  draw();
  return new Promise((resolve) => {
    let done = false;
    modal({
      title: initial ? 'Edit text box' : 'Add text box', icon: 'fonts', size: 'lg',
      body: h('div.rdd-text-editor', h('div', h('div.rdv-section-title', 'Markdown'), ta), h('div', h('div.rdv-section-title', 'Preview'), preview)),
      actions: [
        { label: 'Cancel' },
        { label: initial ? 'Save' : 'Add to dashboard', variant: 'primary', icon: 'check2', onClick: (close) => { if (!ta.value.trim()) throw new Error('Write something first.'); done = true; resolve(ta.value); close(); } },
      ],
      onClose: () => { if (!done) resolve(null); },
    });
    setTimeout(() => ta.focus(), 60);
  });
}

export async function addTextWidget(dashboard, text) {
  return rd.post('/api/widgets', {
    dashboard_id: dashboard.id, visualization_id: null, text, width: 1,
    options: { isHidden: false, position: { col: 0, row: bottomRow(dashboard.widgets || []), sizeX: 6, sizeY: 3, autoHeight: false, minSizeX: 1, maxSizeX: 6, minSizeY: 1, maxSizeY: 1000 } },
  });
}

/** Save a widget's options/text (Redash requires both keys). */
export const saveWidget = (w, { options = w.options, text = w.text } = {}) =>
  rd.post(`/api/widgets/${enc(w.id)}`, { options, text: text || '' });

const MAPPING_TYPES = [
  { value: 'dashboard-level', label: 'Dashboard parameter', help: 'One input at the top of the dashboard controls every widget mapped to it.' },
  { value: 'widget-level', label: 'Widget parameter', help: 'The widget shows its own input.' },
  { value: 'static-value', label: 'Fixed value', help: 'Always runs with the value below; no input is shown.' },
];

/** Edit widget.options.parameterMappings for a visualization widget. Resolves with the new mappings or null. */
export function parameterMappingDialog(widget, query) {
  const params = query?.options?.parameters || [];
  const mappings = JSON.parse(JSON.stringify(widget.options?.parameterMappings || {}));
  const rows = params.map((p) => {
    const m = mappings[p.name] ||= { name: p.name, type: 'widget-level', mapTo: p.name, value: null, title: '' };
    const type = h('select.form-select.form-select-sm', MAPPING_TYPES.map((t) => h('option', { value: t.value, selected: t.value === m.type }, t.label)));
    const mapTo = h('input.form-control.form-control-sm', { value: m.mapTo || p.name, placeholder: 'Dashboard parameter name' });
    const value = h('input.form-control.form-control-sm', { value: m.value == null ? (typeof p.value === 'object' ? JSON.stringify(p.value) : p.value ?? '') : (typeof m.value === 'object' ? JSON.stringify(m.value) : m.value), placeholder: 'Value' });
    const title = h('input.form-control.form-control-sm', { value: m.title || '', placeholder: p.title || p.name });
    const extra = h('div.rdd-map-extra');
    const help = h('div.rdv-help');
    const draw = () => {
      help.textContent = MAPPING_TYPES.find((t) => t.value === type.value)?.help || '';
      mount(extra,
        type.value === 'dashboard-level' && h('label.rdd-map-field', h('span', 'Shared name'), mapTo),
        type.value === 'static-value' && h('label.rdd-map-field', h('span', 'Value'), value),
        type.value !== 'static-value' && h('label.rdd-map-field', h('span', 'Label'), title));
    };
    type.addEventListener('change', draw);
    draw();
    return {
      el: h('div.rdd-map-row', h('div.rdd-map-name', h('code', `{{ ${p.name} }}`), h('span.text-muted.small', p.type)), type, extra, help),
      read() {
        const out = { name: p.name, type: type.value, mapTo: type.value === 'dashboard-level' ? (mapTo.value.trim() || p.name) : p.name, title: title.value.trim(), value: null };
        if (type.value === 'static-value') {
          let v = value.value;
          try { if (/^[[{]/.test(v)) v = JSON.parse(v); } catch { /* keep text */ }
          out.value = v;
        }
        return out;
      },
    };
  });
  return new Promise((resolve) => {
    let done = false;
    modal({
      title: 'Widget parameters', subtitle: widget.visualization?.name, icon: 'sliders', size: 'lg',
      body: params.length ? h('div.rdd-map', rows.map((r) => r.el)) : h('div.rdv-help', 'This query has no parameters.'),
      actions: [
        { label: 'Cancel' },
        params.length && {
          label: 'Save mappings', variant: 'primary', icon: 'check2',
          onClick: async (close) => {
            const next = Object.fromEntries(rows.map((r, i) => [params[i].name, r.read()]));
            done = true;
            resolve(next);
            close();
          },
        },
      ].filter(Boolean),
      onClose: () => { if (!done) resolve(null); },
    });
  });
}
