// Query snippets: reusable SQL fragments expanded in the editor by typing the
// trigger and pressing Tab. Instant search, drawer editor with SQL highlighting,
// trigger validation, copy and delete.

import { h, icon, mount } from '../../../ui/js/dom.js';
import {
  pageHeader, card, dataTable, searchInput, emptyState, errorState, skeleton, timeAgo, drawer,
  toast, toastError, confirmDialog, copyButton,
} from '../../../ui/js/ui.js';
import { rd, enc } from '../core/api.js';
import { userChip, invalidate } from '../core/common.js';
import { sqlEditor } from '../core/editor.js';
import { matches, hl } from './admin_shared.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  let q = ctx.query.get('q') || '';
  let snippets = [];
  const body = h('div', skeleton(5));

  mount(el,
    pageHeader({
      title: 'Query snippets', icon: 'scissors',
      subtitle: h('span', 'Reusable SQL. In the query editor, type a snippet\'s trigger and press ', h('kbd', 'Tab'), ' to insert it.'),
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => openEditor(null) }, icon('plus-lg', 'me-1'), 'New snippet')],
    }),
    card({
      body: h('div.d-flex.flex-column.gap-3',
        h('div.af-toolbar', searchInput({ placeholder: 'Search triggers, descriptions and SQL…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } })),
        body),
    }));

  async function load() {
    try {
      snippets = await rd.get('/api/query_snippets', null, { signal });
      if (ctx.active) draw();
    } catch (err) {
      if (err.name !== 'AbortError') mount(body, errorState(err, load));
    }
  }

  function draw() {
    if (!snippets.length) {
      mount(body, emptyState({
        icon: 'scissors', title: 'No snippets yet',
        message: 'Save SQL you write over and over — a date filter, a join, a CTE — and insert it with a short trigger word.',
        action: h('button.btn.btn-sm.btn-primary', { onclick: () => openEditor(null) }, icon('plus-lg', 'me-1'), 'Create a snippet'),
      }));
      return;
    }
    const rows = snippets.filter((s) => matches(s, q, ['trigger', 'description', 'snippet']))
      .sort((a, b) => a.trigger.localeCompare(b.trigger));
    mount(body, dataTable({
      rows,
      rowKey: (s) => s.id,
      onRowClick: (s) => openEditor(s),
      empty: emptyState({ icon: 'search', title: 'No snippets match' }),
      columns: [
        { key: 'trigger', label: 'Trigger', width: '180px', render: (s) => h('code.rd-trigger', hl(s.trigger, q)) },
        { key: 'description', label: 'Description', render: (s) => (s.description ? hl(s.description, q) : h('span.text-muted', '—')) },
        { key: 'snippet', label: 'SQL', render: (s) => h('code.rd-snippet-preview', hl(s.snippet.replace(/\s+/g, ' ').slice(0, 120), q)) },
        { key: 'user', label: 'Created by', render: (s) => userChip(s.user) },
        { key: 'updated_at', label: 'Updated', render: (s) => timeAgo(s.updated_at) },
        {
          key: 'actions', label: '', align: 'right', width: '96px',
          render: (s) => h('div.d-flex.gap-1.justify-content-end', copyButton(s.snippet),
            h('button.btn.btn-sm.btn-outline-danger', { title: 'Delete', onclick: () => remove(s) }, icon('trash'))),
        },
      ],
    }));
  }

  async function openEditor(snip) {
    const trigger = h('input.form-control.form-control-sm.af-mono', { value: snip?.trigger || '', placeholder: 'e.g. last30', autofocus: true });
    const desc = h('input.form-control.form-control-sm', { value: snip?.description || '', placeholder: 'What does it do?' });
    const triggerHint = h('div.form-text.small');
    const errorBox = h('div.alert.alert-danger.small.py-2.d-none');
    const edHost = h('div.rd-snippet-editor');
    const validate = () => {
      const t = trigger.value.trim();
      const clash = snippets.find((s) => s.trigger === t && s.id !== snip?.id);
      const msg = !t ? 'A trigger is required.' : /\s/.test(t) ? 'Triggers cannot contain spaces.'
        : clash ? `"${t}" is already used by another snippet.` : null;
      mount(triggerHint, msg ? h('span.text-danger', icon('exclamation-circle'), ` ${msg}`) : h('span.text-success', icon('check2'), ' Type ', h('code', t), ' then Tab in the editor.'));
      return msg;
    };
    trigger.addEventListener('input', validate);
    const saveBtn = h('button.btn.btn-sm.btn-primary', { type: 'button' }, icon('check2', 'me-1'), snip ? 'Save' : 'Create snippet');
    const d = drawer({
      title: snip ? `Snippet ${snip.trigger}` : 'New snippet', width: 720,
      subtitle: snip ? h('span', 'Updated ', timeAgo(snip.updated_at)) : 'Reusable SQL fragment',
      actions: [saveBtn],
      body: h('div.af-form',
        errorBox,
        h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Trigger', h('span.text-danger', ' *')), trigger, triggerHint),
        h('div.af-form-row', h('label.form-label.small.fw-semibold', 'Description'), desc),
        h('div.af-form-row', h('label.form-label.small.fw-semibold', 'SQL', h('span.text-danger', ' *')), edHost),
        snip && h('div.d-flex.justify-content-between.align-items-center',
          h('span.small.text-muted', 'Created by ', userChip(snip.user)),
          h('button.btn.btn-sm.btn-outline-danger', { onclick: async () => { if (await remove(snip)) d.close(); } }, icon('trash', 'me-1'), 'Delete'))),
    });
    validate();
    let editor;
    try {
      editor = await sqlEditor(edHost, { value: snip?.snippet || '', placeholder: 'SELECT …', onSave: () => saveBtn.click() });
    } catch (err) {
      mount(edHost, errorState(err));
      return;
    }
    saveBtn.addEventListener('click', async () => {
      errorBox.classList.add('d-none');
      const problem = validate() || (!editor.getValue().trim() && 'The snippet SQL is empty.');
      if (problem) { mount(errorBox, problem); errorBox.classList.remove('d-none'); return; }
      saveBtn.disabled = true;
      try {
        const payload = { trigger: trigger.value.trim(), description: desc.value.trim(), snippet: editor.getValue() };
        const saved = snip ? await rd.post(`/api/query_snippets/${enc(snip.id)}`, payload) : await rd.post('/api/query_snippets', payload);
        snippets = snip ? snippets.map((s) => (s.id === saved.id ? saved : s)) : [...snippets, saved];
        invalidate('snippets');
        toast(snip ? 'Snippet saved' : `Snippet ${saved.trigger} created`, 'success');
        d.close();
        draw();
      } catch (err) {
        mount(errorBox, err.message);
        errorBox.classList.remove('d-none');
      } finally { saveBtn.disabled = false; }
    });
  }

  async function remove(s) {
    const ok = await confirmDialog({ title: `Delete snippet "${s.trigger}"?`, danger: true, confirmLabel: 'Delete', message: 'Queries that already used it keep their SQL.' });
    if (!ok) return false;
    try {
      await rd.del(`/api/query_snippets/${enc(s.id)}`);
      snippets = snippets.filter((x) => x.id !== s.id);
      invalidate('snippets');
      toast('Snippet deleted', 'success');
      draw();
      return true;
    } catch (err) { toastError(err, 'Delete failed'); return false; }
  }

  await load();
}
