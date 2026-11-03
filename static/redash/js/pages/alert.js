// Alert editor (/alerts/new and /alerts/:alertId). A guided, live form:
// 1 query → 2 condition (with "would trigger now?") → 3 notification frequency →
// 4 message template (with preview) → destinations. Ctrl/⌘+S saves.

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import {
  pageHeader, card, emptyState, errorState, skeleton, timeAgo, kv, toast, toastError,
  confirmDialog, spinner,
} from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc, runQuery } from '../core/api.js';
import { urls } from '../core/urls.js';
import { userChip, scheduleText, getSession } from '../core/common.js';
import { fmtCell, inferType } from '../core/format.js';
import {
  alertStateBadge, OPERATORS, evaluateAlert, normalizeOp, opInfo, rearmParts, rearmText, REARM_UNITS,
  TEMPLATE_VARS, renderMustache, typeLogo, pill,
} from './admin_shared.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  const alertId = ctx.params.alertId;
  const isNew = !alertId;

  let alert = null;
  if (!isNew) {
    try {
      alert = await rd.get(`/api/alerts/${enc(alertId)}`, null, { signal });
    } catch (err) {
      mount(el, pageHeader({ title: 'Alert', icon: 'bell', crumbs: [{ label: 'Alerts', href: urls.alerts() }, { label: `#${alertId}` }] }), errorState(err, () => ctx.navigate(location.pathname)));
      return;
    }
    ctx.setTitle(alert.name);
  }

  // ---- editable state
  const opts = alert?.options || {};
  const st = {
    name: alert?.name || '',
    nameTouched: !!alert,
    query: alert?.query || null,
    column: opts.column || '',
    op: normalizeOp(opts.op || '>'),
    value: opts.value ?? '',
    rearm: alert?.rearm ?? null,
    custom: !!(opts.custom_subject || opts.custom_body || opts.template),
    subject: opts.custom_subject || '',
    body: opts.custom_body || opts.template || '',
    muted: !!opts.muted,
  };
  let result = null; // latest query_result of the chosen query
  let dirty = false;
  const preQueryId = ctx.query.get('query_id');

  const saveBtn = h('button.btn.btn-sm.btn-primary', { type: 'button', onclick: () => save() }, icon('check2', 'me-1'), isNew ? 'Create alert' : 'Save changes');
  const dirtyEl = h('span.rd-dirty.d-none', icon('circle-fill'), ' Unsaved changes');
  const markDirty = () => { dirty = true; dirtyEl.classList.remove('d-none'); };

  // ---- step containers
  const queryStep = h('div');
  const conditionStep = h('div');
  const verdictEl = h('div.rd-verdict');
  const rearmStep = h('div');
  const templateStep = h('div');
  const subsEl = h('div');
  const sideEl = h('div.d-flex.flex-column.gap-3');

  const nameInput = h('input.form-control.rd-alert-name', {
    value: st.name, placeholder: 'Alert name — suggested automatically from the condition',
    oninput: (e) => { st.name = e.target.value; st.nameTouched = !!e.target.value; markDirty(); },
  });

  mount(el,
    pageHeader({
      title: isNew ? 'New alert' : alert.name, icon: 'bell',
      crumbs: [{ label: 'Alerts', href: urls.alerts() }, { label: isNew ? 'New' : `#${alert.id}` }],
      badges: !isNew && alertStateBadge(alert.state, { muted: st.muted }),
      subtitle: isNew ? 'Watch one value of a query and get notified when it crosses a threshold.' : h('span', 'Created by ', userChip(alert.user), ' · updated ', timeAgo(alert.updated_at)),
      actions: [dirtyEl, !isNew && muteButton(), !isNew && h('button.btn.btn-sm.btn-outline-danger', { onclick: remove }, icon('trash', 'me-1'), 'Delete'), saveBtn],
    }),
    h('div.rd-alert-layout',
      h('div.d-flex.flex-column.gap-3',
        card({ body: h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Name'), nameInput) }),
        step(1, 'Which query should be watched?', 'The alert checks the first row of this query\'s latest result.', queryStep),
        step(2, 'When should it trigger?', 'Pick a column and a condition. Redash evaluates it each time the query runs.', h('div', conditionStep, verdictEl)),
        step(3, 'How often should it notify?', null, rearmStep),
        step(4, 'What should the message say?', 'Optional — otherwise Redash sends its default message.', templateStep),
        !isNew && card({ title: 'Destinations', icon: 'send', body: subsEl })),
      h('aside.rd-alert-side', sideEl)));

  function step(n, title, sub, body) {
    return h('section.af-card.rd-step',
      h('header.rd-step-head', h('span.rd-step-num', String(n)), h('div', h('h2.rd-step-title', title), sub && h('div.rd-step-sub', sub))),
      h('div.rd-step-body', body));
  }

  // ---- 1. query picker
  function renderQueryStep() {
    if (st.query) {
      const q = st.query;
      mount(queryStep, h('div.rd-picked',
        h('div.rd-picked-icon', icon('code-square')),
        h('div.rd-picked-main',
          h('a.fw-semibold', { href: urls.query(q.id), target: '_blank' }, `#${q.id} ${q.name}`),
          h('div.small.text-muted', 'Schedule: ', scheduleText(q.schedule),
            !q.schedule?.interval && h('span.rd-warn-inline', icon('exclamation-triangle-fill'), ' No schedule — the alert only checks when someone runs the query.'))),
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: () => { st.query = null; result = null; renderQueryStep(); renderCondition(); } }, icon('arrow-repeat', 'me-1'), 'Change')));
      return;
    }
    const input = h('input.form-control', { type: 'search', placeholder: 'Search queries by name or SQL…', autofocus: true });
    const list = h('div.rd-picker-list', h('div.small.text-muted.p-2', 'Start typing to search your queries.'));
    let seq = 0;
    const search = async () => {
      const term = input.value.trim();
      const mine = ++seq;
      mount(list, h('div.p-2', spinner('sm')));
      try {
        const res = await rd.get('/api/queries', { q: term || undefined, page_size: 12, order: term ? undefined : '-created_at' }, { signal });
        if (mine !== seq) return;
        const items = res.results || [];
        mount(list, items.length ? items.map((q) => h('button.rd-picker-item', {
          type: 'button', onclick: () => pickQuery(q),
        },
        icon('code-square'),
        h('span.rd-picker-name', `#${q.id} ${q.name}`),
        q.latest_query_data_id ? pill('has results', 'ok') : pill('never run', 'muted'))) : h('div.small.text-muted.p-2', 'No queries match.'));
      } catch (err) { mount(list, h('div.text-danger.small.p-2', err.message)); }
    };
    input.addEventListener('input', debounce(search, 200));
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); list.querySelector('.rd-picker-item')?.click(); } });
    mount(queryStep, h('div.rd-picker', input, list));
    search();
  }

  async function pickQuery(q) {
    st.query = q;
    markDirty();
    renderQueryStep();
    await loadResult();
  }

  async function loadResult() {
    result = null;
    renderCondition();
    if (!st.query) return;
    try {
      const full = st.query.latest_query_data_id === undefined
        ? await rd.get(`/api/queries/${enc(st.query.id)}`, null, { signal }) : st.query;
      st.query = { ...st.query, ...full };
      if (full.latest_query_data_id) {
        result = (await rd.get(`/api/query_results/${enc(full.latest_query_data_id)}`, null, { signal })).query_result;
      }
    } catch (err) {
      if (err.name !== 'AbortError') toastError(err, 'Could not load the query result');
    }
    if (!ctx.active) return;
    if (result && !st.column) {
      const cols = result.data.columns || [];
      const numeric = cols.find((c) => ['integer', 'float'].includes(c.type));
      st.column = (numeric || cols[0])?.name || '';
    }
    renderQueryStep();
    renderCondition();
  }

  // ---- 2. condition + verdict
  function renderCondition() {
    if (!st.query) {
      mount(conditionStep, h('div.small.text-muted', 'Choose a query first.'));
      mount(verdictEl);
      renderSide();
      return;
    }
    if (!result) {
      mount(conditionStep, h('div.af-callout.af-callout-warn', icon('hourglass'),
        h('div', h('div.fw-semibold', 'This query has no results yet'),
          h('div.small', 'Run it once so the alert knows which columns exist.'),
          h('button.btn.btn-sm.btn-outline-primary.mt-2', { onclick: runNow }, icon('play-fill', 'me-1'), 'Run the query now'))),
      h('div.af-form-row.mt-3', h('label.form-label.small.fw-semibold', 'Column name (type it if you know it)'),
        h('input.form-control.form-control-sm', { value: st.column, oninput: (e) => { st.column = e.target.value; markDirty(); renderVerdict(); } })));
      renderVerdict();
      return;
    }
    const cols = result.data.columns || [];
    const colSel = h('select.form-select.form-select-sm', {
      onchange: (e) => { st.column = e.target.value; markDirty(); suggestName(); renderVerdict(); },
    }, cols.map((c) => h('option', { value: c.name, selected: c.name === st.column }, `${c.friendly_name || c.name} (${c.type || inferType(result.data.rows, c.name)})`)),
    st.column && !cols.some((c) => c.name === st.column) && h('option', { value: st.column, selected: true }, `${st.column} (missing from result)`));
    const opSel = h('select.form-select.form-select-sm', {
      onchange: (e) => { st.op = e.target.value; markDirty(); suggestName(); renderVerdict(); },
    }, OPERATORS.map((o) => h('option', { value: o.value, selected: o.value === st.op }, `${o.symbol}  ${o.label}`)));
    const valInput = h('input.form-control.form-control-sm', {
      value: st.value, placeholder: 'Threshold',
      oninput: debounce((e) => { st.value = e.target.value; markDirty(); suggestName(); renderVerdict(); }, 120),
    });
    mount(conditionStep,
      h('div.rd-condition',
        h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Value column'), colSel),
        h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Condition'), opSel),
        h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Threshold'), valInput)),
      h('div.small.text-muted.mt-2', 'Result from ', timeAgo(result.retrieved_at), ` · ${result.data.rows.length.toLocaleString()} rows · `,
        h('button.af-link-btn', { onclick: runNow }, icon('arrow-clockwise'), ' refresh result')));
    renderVerdict();
  }

  function currentValue() {
    if (!result) return undefined;
    const row = result.data.rows?.[0];
    if (!row || !(st.column in row)) return undefined;
    return row[st.column];
  }

  function renderVerdict() {
    const v = currentValue();
    const verdict = evaluateAlert(st.op, v, st.value);
    const colType = result?.data.columns.find((c) => c.name === st.column)?.type;
    const reason = !st.query ? '' : !result ? 'No result to check yet.'
      : v === undefined ? `Column "${st.column}" is not in the first row of the result.`
        : st.value === '' ? 'Enter a threshold to see the verdict.'
          : `First row value is ${fmtCell(v, colType) || 'empty'}; ${fmtCell(v, colType)} ${opInfo(st.op).symbol} ${st.value} is ${verdict === 'triggered' ? 'true' : verdict === 'ok' ? 'false' : 'not comparable'}.`;
    const tone = verdict === 'triggered' ? 'bad' : verdict === 'ok' ? 'ok' : 'muted';
    mount(verdictEl, st.query ? h(`div.rd-verdict-box.rd-verdict-${tone}`,
      h('div.rd-verdict-icon', icon(verdict === 'triggered' ? 'bell-fill' : verdict === 'ok' ? 'check-circle-fill' : 'question-circle')),
      h('div',
        h('div.rd-verdict-title', verdict === 'triggered' ? 'Would trigger right now' : verdict === 'ok' ? 'Would stay OK right now' : 'Cannot evaluate yet'),
        h('div.small', reason))) : null);
    renderSide();
    renderTemplatePreview();
  }

  async function runNow(e) {
    const btn = e?.currentTarget;
    if (btn) { btn.disabled = true; mount(btn, spinner('sm'), ' Running…'); }
    try {
      result = await runQuery({ queryId: st.query.id, parameters: {}, maxAge: 0, signal });
      st.query.latest_query_data_id = result.id;
      toast('Query ran — result refreshed', 'success');
    } catch (err) {
      if (err.name !== 'AbortError') toastError(err, 'Query failed');
    }
    if (ctx.active) { renderQueryStep(); renderCondition(); }
  }

  function suggestName() {
    if (st.nameTouched || !st.query) return;
    st.name = `${st.query.name}: ${st.column || 'value'} ${opInfo(st.op).symbol} ${st.value || '…'}`;
    nameInput.value = st.name;
  }

  // ---- 3. rearm
  function renderRearm() {
    const p = rearmParts(st.rearm);
    const amount = h('input.form-control.form-control-sm.rd-rearm-amount', { type: 'number', min: 1, value: p.amount, disabled: p.mode !== 'every' });
    const unit = h('select.form-select.form-select-sm.rd-rearm-unit', { disabled: p.mode !== 'every' }, REARM_UNITS.map((u) => h('option', { value: u.value, selected: u.value === p.unit }, u.label)));
    const update = () => {
      const mode = rearmStep.querySelector('input[name=rd-rearm]:checked').value;
      amount.disabled = unit.disabled = mode !== 'every';
      st.rearm = mode === 'once' ? null : mode === 'each' ? 1 : Math.max(1, Number(amount.value) || 1) * Number(unit.value);
      markDirty();
      renderSide();
    };
    amount.addEventListener('input', update);
    unit.addEventListener('change', update);
    const radio = (value, title, sub, extra) => h('label.rd-radio-card', { class: p.mode === value && 'checked' },
      h('input.form-check-input', { type: 'radio', name: 'rd-rearm', value, checked: p.mode === value, onchange: (e) => {
        rearmStep.querySelectorAll('.rd-radio-card').forEach((c) => c.classList.remove('checked'));
        e.target.closest('.rd-radio-card').classList.add('checked');
        update();
      } }),
      h('div', h('div.fw-semibold', title), h('div.small.text-muted', sub), extra));
    mount(rearmStep, h('div.rd-radio-grid',
      radio('once', 'Just once', 'Notify when it triggers, then stay quiet until it goes back to OK.'),
      radio('each', 'Each time it is evaluated', 'Notify on every run while the condition is true. Noisy for frequent schedules.'),
      radio('every', 'At most every…', 'Re-notify while still triggered, no more often than:', h('div.d-flex.gap-2.mt-2', amount, unit))));
  }

  // ---- 4. template
  const subjectInput = h('input.form-control.form-control-sm.af-mono', { value: st.subject, placeholder: '({{ALERT_STATUS}}) {{ALERT_NAME}}' });
  const bodyInput = h('textarea.form-control.form-control-sm.af-mono', { rows: 6, placeholder: 'Value is {{QUERY_RESULT_VALUE}} — see {{QUERY_URL}}' }, st.body);
  const previewEl = h('div.rd-template-preview');
  let lastFocused = bodyInput;
  [subjectInput, bodyInput].forEach((inp) => {
    inp.addEventListener('focus', () => { lastFocused = inp; });
    inp.addEventListener('input', debounce(() => { st.subject = subjectInput.value; st.body = bodyInput.value; markDirty(); renderTemplatePreview(); }, 120));
  });

  function renderTemplate() {
    const toggle = h('input.form-check-input', { type: 'checkbox', checked: st.custom, id: 'rd-custom-tpl' });
    const fields = h('div.rd-template', { class: !st.custom && 'd-none' },
      h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Subject'), subjectInput),
      h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Body'), bodyInput),
      h('div.rd-template-vars',
        h('div.small.fw-semibold.mb-1', 'Click a variable to insert it'),
        h('div.d-flex.flex-wrap.gap-1', TEMPLATE_VARS.map(([k, d]) => h('button.rd-var-chip', {
          type: 'button', title: d,
          onclick: () => {
            const t = lastFocused;
            const ins = `{{${k}}}`;
            const s = t.selectionStart ?? t.value.length;
            t.setRangeText(ins, s, t.selectionEnd ?? s, 'end');
            t.focus();
            t.dispatchEvent(new Event('input'));
          },
        }, k)))),
      h('div.af-form-row', h('label.form-label.small.fw-semibold', icon('eye', 'me-1'), 'Preview with the current result'), previewEl));
    toggle.addEventListener('change', () => { st.custom = toggle.checked; fields.classList.toggle('d-none', !st.custom); markDirty(); renderTemplatePreview(); });
    mount(templateStep,
      h('div.form-check.form-switch.mb-2', toggle, h('label.form-check-label', { for: 'rd-custom-tpl' }, 'Use a custom message')),
      fields);
    renderTemplatePreview();
  }

  function renderTemplatePreview() {
    if (!st.custom) return;
    const verdict = evaluateAlert(st.op, currentValue(), st.value);
    const host = location.origin;
    const vars = {
      ALERT_NAME: st.name || '(unnamed alert)',
      ALERT_URL: alert ? `${host}${urls.alert(alert.id)}` : `${host}/alerts/…`,
      ALERT_STATUS: verdict.toUpperCase(),
      ALERT_CONDITION: st.op,
      ALERT_THRESHOLD: st.value,
      QUERY_NAME: st.query?.name || '(no query)',
      QUERY_URL: st.query ? `${host}${urls.query(st.query.id)}` : '',
      QUERY_RESULT_VALUE: currentValue() ?? '',
      QUERY_RESULT_ROWS: result?.data.rows?.slice(0, 20) || [],
      QUERY_RESULT_COLS: result?.data.columns || [],
    };
    mount(previewEl,
      h('div.rd-mail',
        h('div.rd-mail-subject', renderMustache(st.subject || '({{ALERT_STATUS}}) {{ALERT_NAME}}', vars)),
        h('pre.rd-mail-body', renderMustache(st.body || '(default Redash message)', vars))));
  }

  // ---- side summary
  function renderSide() {
    const verdict = evaluateAlert(st.op, currentValue(), st.value);
    mount(sideEl,
      card({
        title: 'Summary', icon: 'card-checklist',
        body: h('div.rd-summary',
          h('div.rd-summary-sentence',
            'Notify when ', h('b', st.column || 'a column'), ' of ', h('b', st.query?.name || 'a query'), ' is ',
            h('b', opInfo(st.op).label), ' ', h('b', st.value === '' ? '…' : st.value), '.'),
          kv([
            ['Right now', st.query ? alertStateBadge(verdict) : '—'],
            ['Notifies', rearmText(st.rearm)],
            !isNew && ['Current state', alertStateBadge(alert.state, { muted: st.muted })],
            !isNew && ['Last triggered', alert.last_triggered_at ? timeAgo(alert.last_triggered_at) : 'never'],
          ], { cols: 1 })),
      }),
      card({
        title: 'How alerts work', icon: 'lightbulb',
        body: h('ul.rd-howto',
          h('li', 'Redash checks the alert every time its query runs — give the query a refresh schedule.'),
          h('li', 'Only the ', h('b', 'first row'), ' of the result is checked. Sort or aggregate in SQL to control which value that is.'),
          h('li', 'Muting keeps evaluating the alert but stops notifications.')),
      }));
  }

  // ---- destinations / subscriptions
  async function renderSubs() {
    if (isNew) return;
    mount(subsEl, skeleton(2));
    try {
      const [subs, dests, session] = await Promise.all([
        rd.get(`/api/alerts/${enc(alert.id)}/subscriptions`, null, { signal }),
        rd.get('/api/destinations', null, { signal }),
        getSession(),
      ]);
      if (!ctx.active) return;
      const me = session?.user;
      const mineEmail = subs.find((s) => !s.destination && s.user?.id === me?.id);
      const usedDest = new Set(subs.filter((s) => s.destination).map((s) => s.destination.id));
      const available = dests.filter((d) => !usedDest.has(d.id));
      const destSel = h('select.form-select.form-select-sm', h('option', { value: '' }, available.length ? 'Add a destination…' : 'No more destinations available'),
        available.map((d) => h('option', { value: d.id }, `${d.name} (${d.type})`)));
      destSel.addEventListener('change', async () => {
        if (!destSel.value) return;
        try {
          await rd.post(`/api/alerts/${enc(alert.id)}/subscriptions`, { destination_id: Number(destSel.value) });
          toast('Destination added', 'success');
          renderSubs();
        } catch (err) { toastError(err, 'Could not add destination'); }
      });
      mount(subsEl,
        subs.length ? h('div.rd-sub-list', subs.map((s) => h('div.rd-sub',
          s.destination ? typeLogo(s.destination.type, 'dest', 24) : h('span.rd-type-logo', icon('envelope')),
          h('div.rd-sub-main',
            h('div.fw-semibold', s.destination ? s.destination.name : `Email ${s.user?.email || ''}`),
            h('div.small.text-muted', s.destination ? s.destination.type : 'Personal email subscription', ' · added by ', s.user?.name || '?')),
          h('button.btn.btn-sm.btn-outline-danger', {
            title: 'Remove',
            onclick: async () => {
              try {
                await rd.del(`/api/alerts/${enc(alert.id)}/subscriptions/${enc(s.id)}`);
                toast('Removed', 'success');
                renderSubs();
              } catch (err) { toastError(err, 'Could not remove'); }
            },
          }, icon('x-lg')))))
          : emptyState({ icon: 'send', title: 'Nobody is notified yet', message: 'Add yourself by email or pick a destination such as Slack or a webhook.' }),
        h('div.d-flex.gap-2.flex-wrap.mt-3',
          !mineEmail && h('button.btn.btn-sm.btn-outline-primary', {
            onclick: async () => {
              try {
                await rd.post(`/api/alerts/${enc(alert.id)}/subscriptions`, {});
                toast('You will be emailed when this alert changes state', 'success');
                renderSubs();
              } catch (err) { toastError(err, 'Could not subscribe'); }
            },
          }, icon('envelope', 'me-1'), 'Email me'),
          h('div.rd-dest-add', destSel),
          h('a.btn.btn-sm.btn-link', { href: urls.destinations() }, icon('gear', 'me-1'), 'Manage destinations')));
    } catch (err) {
      if (err.name !== 'AbortError') mount(subsEl, errorState(err, renderSubs));
    }
  }

  function muteButton() {
    const btn = h('button.btn.btn-sm.btn-outline-secondary', { type: 'button' });
    const draw = () => mount(btn, icon(st.muted ? 'bell' : 'bell-slash', 'me-1'), st.muted ? 'Unmute' : 'Mute');
    btn.addEventListener('click', async () => {
      const next = !st.muted;
      try {
        if (next) await rd.post(`/api/alerts/${enc(alert.id)}/mute`, {});
        else await rd.del(`/api/alerts/${enc(alert.id)}/mute`);
        st.muted = next;
        alert.options = { ...alert.options, muted: next };
        draw();
        renderSide();
        toast(next ? 'Muted — still evaluated, notifications paused' : 'Unmuted', 'success');
      } catch (err) { toastError(err, 'Could not change mute state'); }
    });
    draw();
    return btn;
  }

  // ---- save / delete
  async function save() {
    const problems = [];
    if (!st.query) problems.push('choose a query');
    if (!st.column) problems.push('choose a value column');
    if (st.value === '' || st.value == null) problems.push('enter a threshold');
    if (problems.length) { toast(`Please ${problems.join(', ')}.`, 'warning', { title: 'Almost there' }); return; }
    suggestName();
    const options = {
      ...(alert?.options || {}),
      column: st.column, op: st.op, value: st.value, muted: st.muted,
    };
    delete options.template;
    if (st.custom) { options.custom_subject = st.subject; options.custom_body = st.body; } else { delete options.custom_subject; delete options.custom_body; }
    const body = { name: st.name.trim() || `${st.query.name} alert`, query_id: st.query.id, options, rearm: st.rearm };
    saveBtn.disabled = true;
    try {
      const saved = isNew
        ? await rd.post('/api/alerts', body)
        : await rd.post(`/api/alerts/${enc(alert.id)}`, body);
      dirty = false;
      dirtyEl.classList.add('d-none');
      toast(isNew ? 'Alert created' : 'Alert saved', 'success');
      if (isNew) navigate(urls.alert(saved.id));
      else { alert = saved; renderSide(); }
    } catch (err) {
      toastError(err, 'Could not save the alert');
    } finally {
      saveBtn.disabled = false;
    }
  }

  async function remove() {
    const ok = await confirmDialog({ title: `Delete alert "${alert.name}"?`, danger: true, confirmLabel: 'Delete alert', message: 'Subscriptions are removed too. The query is not affected.' });
    if (!ok) return;
    try {
      await rd.del(`/api/alerts/${enc(alert.id)}`);
      dirty = false;
      toast('Alert deleted', 'success');
      navigate(urls.alerts());
    } catch (err) { toastError(err, 'Delete failed'); }
  }

  const onKey = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); save(); }
  };
  const onUnload = (e) => { if (dirty) { e.preventDefault(); e.returnValue = ''; } };
  document.addEventListener('keydown', onKey);
  window.addEventListener('beforeunload', onUnload);
  ctx.cleanup(() => { document.removeEventListener('keydown', onKey); window.removeEventListener('beforeunload', onUnload); });

  renderQueryStep();
  renderCondition();
  renderRearm();
  renderTemplate();
  renderSide();
  renderSubs();

  if (st.query) await loadResult();
  else if (preQueryId) {
    try { await pickQuery(await rd.get(`/api/queries/${enc(preQueryId)}`, null, { signal })); } catch { /* picker stays open */ }
    dirty = false;
    dirtyEl.classList.add('d-none');
  }
}
