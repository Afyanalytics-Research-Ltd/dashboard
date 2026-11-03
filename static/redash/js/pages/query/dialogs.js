// Query page dialogs: refresh schedule, API key & result URLs, permissions,
// editor settings and the save-conflict resolver.

import { h, icon, mount } from '../../../../ui/js/dom.js';
import { modal, copyButton, confirmDialog, toast, toastError, kv, spinner, emptyState } from '../../../../ui/js/ui.js';
import { rd, enc, request, config } from '../../core/api.js';
import { scheduleText, DAY_NAMES, userChip, getSession } from '../../core/common.js';
import { editorPrefs, saveEditorPrefs } from '../../core/editor.js';

// ------------------------------------------------------------------ schedule

const DEFAULT_INTERVALS = [60, 300, 600, 900, 1800, 3600, 7200, 10800, 14400, 21600, 43200, 86400, 172800, 604800, 1209600];

function intervalLabel(s) {
  return scheduleText({ interval: s }).replace(/^Every /, '');
}

/** Resolves to the new schedule (object or null for "Never"), or undefined on cancel. */
export async function scheduleDialog(schedule) {
  let intervals = DEFAULT_INTERVALS;
  try {
    const s = await getSession();
    if (s?.client_config?.queryRefreshIntervals?.length) intervals = s.client_config.queryRefreshIntervals;
  } catch { /* defaults */ }
  const cur = schedule || {};
  const intervalSel = h('select.form-select.form-select-sm', { 'aria-label': 'Refresh every' },
    h('option', { value: '' }, 'Never (manual refresh only)'),
    intervals.map((s) => h('option', { value: s, selected: Number(cur.interval) === s }, `Every ${intervalLabel(s)}`)));
  const time = h('input.form-control.form-control-sm', { type: 'time', value: cur.time || '00:00' });
  const dow = h('select.form-select.form-select-sm', DAY_NAMES.map((d) => h('option', { value: d, selected: (cur.day_of_week || 'Monday') === d }, d)));
  const untilOn = h('input.form-check-input', { type: 'checkbox', checked: !!cur.until });
  const until = h('input.form-control.form-control-sm', { type: 'date', value: cur.until || '' });
  const summary = h('div.rd-sched-summary');
  const timeRow = h('div.af-form-row', h('label.form-label.small.fw-semibold', 'At (UTC)'), time, h('div.form-text.small', 'Your local time: ', h('span.rd-sched-local')));
  const dowRow = h('div.af-form-row', h('label.form-label.small.fw-semibold', 'On'), dow);
  const untilRow = h('div.af-form-row', h('label.form-check.small', untilOn, h('span.ms-2', 'Stop refreshing after a date')), until);

  const value = () => {
    const interval = Number(intervalSel.value);
    if (!interval) return null;
    const out = { interval, time: null, day_of_week: null, until: null };
    if (interval >= 86400) out.time = time.value || '00:00';
    if (interval >= 604800) out.day_of_week = dow.value;
    if (untilOn.checked && until.value) out.until = until.value;
    return out;
  };
  const draw = () => {
    const interval = Number(intervalSel.value);
    timeRow.hidden = !(interval >= 86400);
    dowRow.hidden = !(interval >= 604800);
    untilRow.hidden = !interval;
    until.disabled = !untilOn.checked;
    const [hh, mm] = (time.value || '00:00').split(':').map(Number);
    const d = new Date(Date.UTC(2000, 0, 1, hh, mm));
    timeRow.querySelector('.rd-sched-local').textContent = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    mount(summary, icon(interval ? 'arrow-repeat' : 'hand-index', 'me-1'), scheduleText(value()));
  };
  [intervalSel, time, dow, untilOn, until].forEach((x) => x.addEventListener('input', draw));
  draw();

  return new Promise((resolve) => {
    let done = false;
    modal({
      title: 'Refresh schedule', icon: 'calendar2-week', size: 'sm',
      body: h('div.af-form', h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Refresh'), intervalSel), timeRow, dowRow, untilRow, summary),
      actions: [
        { label: 'Cancel' },
        { label: 'Save schedule', variant: 'primary', onClick: (close) => { done = true; resolve(value()); close(); } },
      ],
      onClose: () => { if (!done) resolve(undefined); },
    });
  });
}

// ------------------------------------------------------------------ API key

export function apiKeyDialog(query, { onRegenerated } = {}) {
  const base = config.redashUrl || '';
  const body = h('div');
  const draw = () => {
    const key = query.api_key;
    const urlsList = [
      ['CSV', `${base}/api/queries/${query.id}/results.csv?api_key=${key}`],
      ['JSON', `${base}/api/queries/${query.id}/results.json?api_key=${key}`],
    ];
    mount(body,
      h('p.small.text-muted', 'Anyone with this key can read this query’s latest results — nothing else. Share these links with tools such as Google Sheets or Excel.'),
      h('div.rd-key-row', h('code.rd-key', key), copyButton(key, 'Copy key')),
      h('div.rd-key-urls', urlsList.map(([label, url]) => h('div.rd-key-url',
        h('span.rd-type-pill', label),
        h('code.rd-key-url-text', url),
        copyButton(url)))),
      query.options?.parameters?.length && h('div.af-callout.af-callout-warn.mt-2', icon('exclamation-triangle'),
        h('span', 'This query has parameters, so these links return the last cached result. Use POST /api/queries/', String(query.id), '/results with parameters for fresh data.')));
  };
  draw();
  modal({
    title: 'API key & result links', icon: 'key', size: 'lg', body,
    actions: [
      {
        label: 'Regenerate key', variant: 'outline-danger', icon: 'arrow-repeat',
        onClick: async () => {
          const ok = await confirmDialog({ title: 'Regenerate this query’s API key?', message: 'Existing links using the old key stop working immediately.', confirmLabel: 'Regenerate', danger: true });
          if (!ok) return;
          const res = await rd.post(`/api/queries/${enc(query.id)}/regenerate_api_key`, {});
          query.api_key = res.api_key;
          draw();
          onRegenerated?.(res.api_key);
          toast('New API key generated', 'success');
        },
      },
      { label: 'Done', variant: 'primary' },
    ],
  });
}

// ------------------------------------------------------------------ permissions

export function permissionsDialog(query) {
  const body = h('div', spinner('sm'));
  const load = async () => {
    try {
      const [acl, users] = await Promise.all([
        rd.get(`/api/queries/${enc(query.id)}/acl`),
        rd.all('/api/users', {}, { max: 500 }).catch(() => []),
      ]);
      const editors = acl.modify || [];
      const ids = new Set([query.user?.id, ...editors.map((u) => u.id)]);
      const pick = h('select.form-select.form-select-sm', h('option', { value: '' }, 'Choose a user…'),
        users.filter((u) => !ids.has(u.id) && !u.is_disabled).map((u) => h('option', { value: u.id }, `${u.name} <${u.email}>`)));
      mount(body,
        h('p.small.text-muted', 'Admins and the owner can always edit. Grant edit access to more people here.'),
        h('div.rd-acl-list',
          h('div.rd-acl-row', userChip(query.user), h('span.rd-type-pill', 'Owner')),
          editors.map((u) => h('div.rd-acl-row', userChip(u), h('span.rd-type-pill', 'Can edit'),
            h('button.btn.btn-sm.btn-link.text-danger', {
              onclick: async () => {
                try {
                  await request('DELETE', `/api/queries/${enc(query.id)}/acl`, { body: { access_type: 'modify', user_id: u.id } });
                  toast(`${u.name} can no longer edit`, 'success');
                  load();
                } catch (err) { toastError(err, 'Could not remove access'); }
              },
            }, 'Remove')))),
        h('div.rd-acl-add', pick, h('button.btn.btn-sm.btn-primary', {
          onclick: async () => {
            if (!pick.value) return;
            try {
              await rd.post(`/api/queries/${enc(query.id)}/acl`, { access_type: 'modify', user_id: Number(pick.value) });
              toast('Edit access granted', 'success');
              load();
            } catch (err) { toastError(err, 'Could not grant access'); }
          },
        }, icon('person-plus', 'me-1'), 'Grant edit access')),
        users.length <= 1 && h('div.small.text-muted.mt-2', 'Invite more users from Admin → Users to share editing.'));
    } catch (err) {
      mount(body, emptyState({ icon: 'shield-lock', title: 'Permissions unavailable', message: err.message }));
    }
  };
  load();
  modal({ title: 'Who can edit this query', icon: 'shield-lock', body, actions: [{ label: 'Done', variant: 'primary' }] });
}

// ------------------------------------------------------------------ editor settings

export function editorSettingsDialog(onSave) {
  const p = editorPrefs();
  const sw = (name, label, help) => h('label.rd-set-row',
    h('input.form-check-input', { type: 'checkbox', name, checked: !!p[name] }),
    h('span', h('span.fw-semibold', label), help && h('span.d-block.small.text-muted', help)));
  const font = h('input.form-range', { type: 'range', min: 11, max: 20, step: 1, value: p.fontSize });
  const fontVal = h('span.rd-set-font', `${p.fontSize}px`);
  font.addEventListener('input', () => { fontVal.textContent = `${font.value}px`; });
  const body = h('div.rd-settings',
    h('div.rd-set-row.rd-set-font-row', h('span.fw-semibold', 'Font size'), font, fontVal),
    sw('lineWrapping', 'Wrap long lines'),
    sw('liveAutocomplete', 'Suggest as you type', 'Tables, columns and keywords. Ctrl+Space always works.'),
    sw('lowercaseHints', 'Lowercase keyword suggestions'),
    h('label.rd-set-row', h('input.form-check-input', { type: 'checkbox', name: 'dark', checked: p.theme === 'dark' }), h('span.fw-semibold', 'Dark editor')),
    sw('vim', 'Vim key bindings'));
  modal({
    title: 'Editor settings', icon: 'gear', size: 'sm', body,
    actions: [
      { label: 'Cancel' },
      {
        label: 'Save', variant: 'primary',
        onClick: (close) => {
          const get = (n) => body.querySelector(`[name=${n}]`).checked;
          const next = {
            fontSize: Number(font.value), lineWrapping: get('lineWrapping'), liveAutocomplete: get('liveAutocomplete'),
            lowercaseHints: get('lowercaseHints'), vim: get('vim'), theme: get('dark') ? 'dark' : 'light',
          };
          saveEditorPrefs(next);
          close();
          onSave?.(next);
        },
      },
    ],
  });
}

// ------------------------------------------------------------------ conflict

/** Resolves 'reload' | 'overwrite' | null. */
export function conflictDialog(theirs) {
  return new Promise((resolve) => {
    let done = false;
    modal({
      title: 'Someone else changed this query', icon: 'people', size: 'sm',
      body: h('div',
        h('p', 'This query was saved by someone else (or in another tab) after you opened it.'),
        theirs && kv([
          ['Saved by', theirs.last_modified_by?.name || theirs.user?.name || '—'],
          ['Version', String(theirs.version)],
        ], { cols: 2 }),
        h('p.small.text-muted.mt-2', 'Your edits are kept as a local draft either way.')),
      actions: [
        { label: 'Cancel', onClick: (close) => { done = true; resolve(null); close(); } },
        { label: 'Load their version', icon: 'cloud-download', onClick: (close) => { done = true; resolve('reload'); close(); } },
        { label: 'Overwrite with mine', variant: 'danger', icon: 'cloud-upload', onClick: (close) => { done = true; resolve('overwrite'); close(); } },
      ],
      onClose: () => { if (!done) resolve(null); },
    });
  });
}
