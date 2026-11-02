// Configuration: full config when the server exposes it, otherwise a single-option lookup.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import {
  pageHeader, card, searchInput, emptyState, errorState, skeleton, copyButton, buildForm, spinner,
} from '../core/ui.js';

const COMMON = [
  ['core', 'executor'], ['core', 'parallelism'], ['core', 'max_active_tasks_per_dag'], ['core', 'max_active_runs_per_dag'],
  ['core', 'dags_folder'], ['core', 'default_timezone'], ['scheduler', 'dag_dir_list_interval'], ['logging', 'base_log_folder'],
  ['api', 'expose_config'], ['webserver', 'default_ui_timezone'],
];

export default async function render(ctx) {
  const { el, signal } = ctx;
  const bodyEl = h('div', skeleton(8));
  mount(el,
    pageHeader({ title: 'Configuration', icon: 'sliders', subtitle: 'Effective airflow.cfg values on the Airflow server (sensitive values are masked by Airflow).' }),
    lookupCard(),
    bodyEl);

  let config;
  try {
    config = await api.get('/config', null, { signal });
  } catch (err) {
    if (err.name === 'AbortError' || !ctx.active) return;
    if (err.status === 403) {
      mount(bodyEl, card({
        body: h('div.sys-locked',
          h('div.sys-locked-icon', icon('shield-lock')),
          h('div',
            h('div.fw-bold.mb-1', 'The Airflow administrator has not exposed the configuration'),
            h('div.small.text-muted', err.message),
            h('div.small.mt-2', 'To allow it, set ', h('code', '[api] expose_config = True'),
              ' (or ', h('code', 'non-sensitive-only'), ') on the Airflow API server. Individual options may still be readable with the lookup above.'))),
      }));
    } else {
      mount(bodyEl, errorState(err, () => render(ctx)));
    }
    return;
  }
  if (!ctx.active) return;
  renderSections(config.sections || []);

  function renderSections(sections) {
    let q = ctx.query.get('q') || '';
    const listEl = h('div.d-flex.flex-column.gap-2');
    const draw = () => {
      const needle = q.toLowerCase();
      const shown = sections.map((s) => ({
        ...s,
        options: s.options.filter((o) => !needle || `${s.name} ${o.key} ${o.value}`.toLowerCase().includes(needle)),
      })).filter((s) => s.options.length);
      if (!shown.length) { mount(listEl, emptyState({ icon: 'search', title: 'No options match' })); return; }
      mount(listEl, shown.map((s) => h('details.sys-cfg', { open: !!needle || shown.length === 1 },
        h('summary.sys-cfg-head', icon('chevron-right', 'sys-ie-caret'), h('span.fw-semibold', `[${s.name}]`), h('span.af-tab-count', String(s.options.length))),
        h('table.af-table.af-table-dense', h('tbody', s.options.map((o) => h('tr',
          h('td.af-mono.small', { style: { width: '34%' } }, o.key),
          h('td.af-mono.small.sys-cfg-val', valueText(o.value)),
          h('td', { style: { width: '40px' } }, copyButton(`${o.key} = ${valueText(o.value)}`)))))))));
    };
    mount(bodyEl, card({
      title: `${sections.length} sections`, icon: 'list-nested',
      body: h('div.d-flex.flex-column.gap-3',
        searchInput({ placeholder: 'Search sections, keys or values…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } }),
        listEl),
    }));
    draw();
  }

  function lookupCard() {
    const form = buildForm([
      { name: 'section', label: 'Section', value: ctx.query.get('section') || 'core', required: true, placeholder: 'core' },
      { name: 'option', label: 'Option', value: ctx.query.get('option') || '', required: true, placeholder: 'executor' },
    ]);
    form.el.classList.add('sys-lookup-form');
    const result = h('div.sys-lookup-result');
    const run = async (section, option) => {
      mount(result, spinner('sm'));
      ctx.setQuery({ section, option });
      try {
        const res = await api.get(`/config/section/${enc(section)}/option/${enc(option)}`, null, { signal });
        const opt = res.sections?.[0]?.options?.[0];
        mount(result, h('div.sys-lookup-ok',
          h('span.af-mono.small.text-muted', `[${section}] ${option} = `),
          h('code.sys-lookup-value', opt ? valueText(opt.value) : JSON.stringify(res)),
          opt && copyButton(valueText(opt.value))));
      } catch (err) {
        if (err.name === 'AbortError') return;
        mount(result, h('div.small', { class: err.status === 403 ? 'text-warning-emphasis' : 'text-danger' },
          icon(err.status === 403 ? 'shield-lock' : 'x-octagon', 'me-1'),
          err.status === 404 ? `No option [${section}] ${option}.` : err.message));
      }
    };
    form.el.addEventListener('submit', (e) => {
      e.preventDefault();
      try { const v = form.values(); run(v.section.trim(), v.option.trim()); } catch (err) { mount(result, h('div.text-danger.small', err.message)); }
    });
    const quick = h('div.af-tags', COMMON.map(([s, o]) => h('button.af-tag', {
      type: 'button',
      onclick: () => {
        form.el.querySelectorAll('input')[0].value = s;
        form.el.querySelectorAll('input')[1].value = o;
        run(s, o);
      },
    }, `${s}.${o}`)));
    return card({
      title: 'Look up an option', icon: 'search',
      body: h('div.d-flex.flex-column.gap-2',
        h('div.d-flex.gap-2.align-items-end.flex-wrap', form.el,
          h('button.btn.btn-sm.btn-primary', { type: 'button', onclick: () => form.el.requestSubmit() }, icon('search', 'me-1'), 'Look up')),
        h('div.small.text-muted', 'Common options:'), quick, result),
    });
  }
}

function valueText(v) {
  if (Array.isArray(v)) return v.join(', ');
  if (v && typeof v === 'object') return JSON.stringify(v);
  return v == null ? '' : String(v);
}
