// Required actions: human-in-the-loop requests raised by HITL operators,
// with a response dialog (options + params input).

import { h, icon, mount, debounce } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  pageHeader, card, pager, refreshControl, emptyState, errorState, skeleton, timeAgo, jsonView, modal,
  jsonEditor, toast, stateBadge, searchInput,
} from '../core/ui.js';
import { fmtDate, fmtNumber } from '../core/format.js';
import { segmented } from './admin_shared.js';

const VIEWS = [
  { value: 'pending', label: 'Awaiting response', icon: 'hourglass-split' },
  { value: 'responded', label: 'Responded', icon: 'check2-all' },
  { value: 'all', label: 'All', icon: 'list' },
];

/** HITL params may be serialized Params ({value, schema, description}) or plain values. */
const paramValue = (p) => (p && typeof p === 'object' && !Array.isArray(p) && 'value' in p ? p.value : p);

export default async function render(ctx) {
  const { el, signal } = ctx;
  const st = {
    view: ctx.query.get('view') || 'pending',
    q: ctx.query.get('q') || '',
    dag: ctx.query.get('dag') || '',
    limit: 25,
    offset: 0,
  };

  const listEl = h('div', skeleton(4, { height: 60 }));
  const pagerEl = h('div');
  const countEl = h('span.text-muted.small');
  const refresh = refreshControl(() => load(), { key: 'hitl', initial: 10 });
  ctx.cleanup(refresh.stop);

  const onDag = debounce((v) => { st.dag = v; st.offset = 0; ctx.setQuery({ dag: v || null }); load(); }, 300);
  mount(el,
    pageHeader({
      title: 'Required actions', icon: 'person-check',
      subtitle: 'Tasks paused until a person approves, chooses an option or supplies input (human-in-the-loop).',
      actions: [refresh.el],
    }),
    card({
      body: h('div',
        h('div.af-toolbar.mb-3',
          segmented(VIEWS, st.view, (v) => { st.view = v; st.offset = 0; ctx.setQuery({ view: v === 'pending' ? null : v }); load(); }),
          searchInput({ placeholder: 'Search subject…', value: st.q, onInput: (q) => { st.q = q; st.offset = 0; ctx.setQuery({ q: q || null }); load(); } }),
          h('input.form-control.form-control-sm.af-hitl-dag', {
            type: 'search', placeholder: 'DAG id contains…', value: st.dag, 'aria-label': 'DAG filter',
            oninput: (e) => onDag(e.target.value.trim()),
          }),
          h('div.af-grow'),
          countEl),
        listEl, pagerEl),
    }));

  async function load() {
    try {
      const res = await api.get('/dags/~/dagRuns/~/hitlDetails', {
        limit: st.limit, offset: st.offset,
        order_by: st.view === 'responded' ? '-responded_at' : '-created_at',
        response_received: st.view === 'all' ? null : st.view === 'responded',
        subject_search: st.q ? `%${st.q}%` : null,
        dag_id_pattern: st.dag ? `%${st.dag}%` : null,
      }, { signal });
      if (!ctx.active) return;
      const items = res.hitl_details || [];
      mount(countEl, `${fmtNumber(res.total_entries || 0)} request${res.total_entries === 1 ? '' : 's'}`);
      mount(listEl, items.length ? h('div.af-hitl-list', items.map(itemCard)) : emptyState(st.view === 'pending'
        ? { icon: 'check2-circle', title: 'Nothing needs your attention', message: 'When a DAG reaches a human-in-the-loop task (ApprovalOperator, HITLOperator…), it shows up here until someone responds.' }
        : { icon: 'person-check', title: 'No matching requests', message: st.q || st.dag ? 'Try clearing the filters.' : null }));
      mount(pagerEl, res.total_entries > st.limit || st.offset ? pager({
        total: res.total_entries, limit: st.limit, offset: st.offset, sizes: [25, 50, 100],
        onChange: (o, l) => { st.offset = o; st.limit = l; load(); },
      }) : null);
    } catch (err) {
      if (err.name !== 'AbortError') mount(listEl, errorState(err, load));
    }
  }

  function itemCard(d) {
    const ti = d.task_instance || {};
    const responded = !!d.response_received;
    const chosen = new Set(d.chosen_options || []);
    const defaults = new Set(d.defaults || []);
    const params = d.params || {};
    const hasParams = Object.keys(params).length > 0;
    return h('article.af-hitl-card', { class: responded ? 'responded' : 'pending' },
      h('div.af-hitl-card-head',
        h('div.af-hitl-icon', icon(responded ? 'check2-circle' : 'person-raised-hand')),
        h('div.af-hitl-main',
          h('div.af-hitl-subject', d.subject || '(no subject)'),
          h('div.af-hitl-where.small',
            ti.dag_id && h('a', { href: urls.dag(ti.dag_id) }, ti.dag_display_name || ti.dag_id),
            ti.dag_run_id && [' · ', h('a.af-mono', { href: urls.run(ti.dag_id, ti.dag_run_id) }, ti.dag_run_id)],
            ti.task_id && [' · ', h('a.af-mono', { href: urls.task(ti.dag_id, ti.dag_run_id, ti.task_id, ti.map_index) }, ti.task_display_name || ti.task_id, ti.map_index >= 0 ? `[${ti.map_index}]` : '')])),
        h('div.af-hitl-side',
          ti.state && stateBadge(ti.state),
          h('div.small.text-muted', 'raised ', timeAgo(d.created_at)))),
      d.body && h('div.af-hitl-body', d.body),
      h('div.af-hitl-options-row',
        h('span.af-section-label.mb-0', d.multiple ? 'Options (multiple)' : 'Options'),
        (d.options || []).map((o) => h('span.af-hitl-opt', { class: [chosen.has(o) && 'chosen', !responded && defaults.has(o) && 'default'] },
          chosen.has(o) && icon('check2'), o, !responded && defaults.has(o) && h('span.af-hitl-def', 'default')))),
      hasParams && h('details.af-hitl-params', h('summary.small', `Parameters (${Object.keys(params).length})`), jsonView(responded && d.params_input ? d.params_input : params, { open: 1 })),
      h('div.af-hitl-foot',
        h('div.small.text-muted',
          (d.assigned_users || []).length
            ? ['Assigned to ', d.assigned_users.map((u, i) => [i ? ', ' : '', h('b', u.name || u.id)])]
            : 'Anyone with access can respond',
          responded && [' · responded by ', h('b', d.responded_by_user?.name || d.responded_by_user?.id || 'unknown'), ' ', h('span', { title: fmtDate(d.responded_at) }, timeAgo(d.responded_at))]),
        !responded && h('button.btn.btn-sm.btn-primary', { onclick: () => respond(d) }, icon('reply-fill', 'me-1'), 'Respond')));
  }

  function respond(d) {
    const ti = d.task_instance;
    const name = `hitl-${Math.random().toString(36).slice(2, 7)}`;
    const defaults = new Set(d.defaults || []);
    const optionsEl = h('div.af-hitl-options', (d.options || []).map((o, i) => {
      const id = `${name}-${i}`;
      return h('label.af-hitl-option', { for: id },
        h('input.form-check-input', { id, type: d.multiple ? 'checkbox' : 'radio', name, value: o, checked: defaults.has(o) || (!d.multiple && !defaults.size && i === 0) }),
        h('span', o));
    }));
    const params = d.params || {};
    const paramEditor = Object.keys(params).length
      ? jsonEditor(Object.fromEntries(Object.entries(params).map(([k, v]) => [k, paramValue(v)])), { rows: 8 })
      : null;
    modal({
      title: d.subject || 'Respond', icon: 'reply', size: 'lg',
      subtitle: `${ti.dag_id} · ${ti.task_id}${ti.map_index >= 0 ? ` [${ti.map_index}]` : ''}`,
      body: h('div.d-flex.flex-column.gap-3',
        d.body && h('div.af-hitl-body', d.body),
        h('div', h('div.af-section-label', d.multiple ? 'Choose one or more' : 'Choose one'), optionsEl),
        paramEditor && h('div', h('div.af-section-label', 'Parameters'), paramEditor.el,
          h('div.form-text.small', 'Sent as params_input. Descriptions of each parameter:'),
          h('ul.small.text-muted.mb-0', Object.entries(params).map(([k, v]) => h('li', h('code', k), v?.description ? ` — ${v.description}` : ''))))),
      actions: [
        { label: 'Cancel' },
        {
          label: 'Send response', variant: 'primary', icon: 'send',
          onClick: async (close) => {
            const chosen = [...optionsEl.querySelectorAll('input:checked')].map((i) => i.value);
            if (!chosen.length) throw new Error('Choose at least one option.');
            const payload = { chosen_options: chosen };
            if (paramEditor) {
              const v = paramEditor.value();
              if (v != null && (typeof v !== 'object' || Array.isArray(v))) throw new Error('Parameters must be a JSON object.');
              payload.params_input = v || {};
            }
            await api.patch(`/dags/${enc(ti.dag_id)}/dagRuns/${enc(ti.dag_run_id)}/taskInstances/${enc(ti.task_id)}/${ti.map_index ?? -1}/hitlDetails`, payload);
            toast(`Responded: ${chosen.join(', ')}`, 'success', { title: d.subject });
            close();
            load();
          },
        },
      ],
    });
  }

  await load();
}
