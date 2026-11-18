// W&B connection: use your own W&B API key — for this platform's W&B server or W&B Cloud (wandb.ai) — instead of
// the shared server key. The key is tested on the chosen W&B before it is saved, stored encrypted on the server and
// never sent back to the browser.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, card, kv, timeAgo, toast, confirmDialog } from '../../../ui/js/ui.js';
import { config } from '../core/api.js';

function csrf() {
  const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

async function call(method, body) {
  const res = await fetch(config.connectionApi || '/models/api/connection', {
    method, credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json', 'X-CSRFToken': csrf() },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.errors?.[0]?.message || `Request failed (${res.status})`);
  return data;
}

export default async function render(ctx) {
  const { el } = ctx;
  let state = config.connection || await call('GET');
  const statusEl = h('div');
  const formEl = h('div');
  mount(el,
    pageHeader({
      title: 'W&B connection', icon: 'key',
      subtitle: 'Use your own Weights & Biases account in Model Studio — on this platform’s W&B server or on W&B Cloud (wandb.ai).',
    }),
    h('div.d-flex.flex-column.gap-3', statusEl, formEl));
  draw();

  function draw() {
    const a = state.active;
    const own = state.own;
    const where = a.source === 'own' ? `Your key · ${a.server_label}` : a.source === 'shared' ? `Shared server key · ${a.server_label}` : 'Not connected';
    mount(statusEl, card({
      title: 'In use now', icon: a.source === 'none' ? 'plug' : 'plug-fill',
      body: h('div',
        kv([
          ['Connection', h('span', { class: a.source === 'none' ? 'text-danger' : '' }, where)],
          ['W&B', a.source === 'none' ? '—' : a.wandb_url
            ? h('a', { href: a.wandb_url, target: '_blank', rel: 'noopener' }, a.wandb_url, ' ', icon('box-arrow-up-right'))
            : h('span.text-muted', 'Reached by Model Studio on the server (not published on its own web address)')],
          ['W&B user', a.wandb_user || (a.source === 'shared' ? 'the shared service account' : '—')],
          ['API key', a.key_hint ? h('span.ms-mono', a.key_hint) : '—'],
          ...(own?.verified_at ? [['Checked', timeAgo(own.verified_at)]] : []),
        ]),
        own ? h('div.mt-3.d-flex.gap-2.flex-wrap',
          h('button.btn.btn-sm.btn-outline-danger', { type: 'button', onclick: remove }, icon('x-circle'),
            state.shared_available ? ' Remove my key — use the shared key again' : ' Remove my key')) : null),
    }));
    drawForm();
  }

  function drawForm() {
    const own = state.own;
    let server = own?.server || (state.servers.find((s) => s.key === 'local') ? 'local' : state.servers[0]?.key);
    const keyInput = h('input.form-control.ms-mono', {
      type: 'password', autocomplete: 'off', spellcheck: false, placeholder: own ? 'Paste a new key to replace yours' : 'Paste your W&B API key',
      'aria-label': 'W&B API key',
    });
    const msg = h('div.small.mt-2');
    const hint = h('div.small.text-muted.mt-1');
    const setHint = () => {
      const s = state.servers.find((x) => x.key === server);
      const cloudNote = server === 'cloud' ? 'Model Studio then shows your wandb.ai projects; instance-admin pages (Users, System, Cleanup) are hidden.' : '';
      if (s.key_page) {
        mount(hint, 'Get it at ', h('a', { href: s.key_page, target: '_blank', rel: 'noopener' }, s.key_page), ' (signed in as you). ', cloudNote);
      } else {
        // This platform's W&B is only reachable on the server itself: no page to send people to.
        mount(hint, 'This platform’s W&B has no public web address yet, so its key page can’t be opened from here. ',
          'Ask a W&B admin for a key for your W&B user, or use W&B Cloud.');
      }
    };
    const options = h('div.d-flex.flex-column.gap-2', state.servers.map((s) => h('label.ms-choice.d-flex.gap-2.align-items-start',
      h('input.form-check-input.mt-1', { type: 'radio', name: 'wb-server', value: s.key, checked: s.key === server,
        onchange: () => { server = s.key; setHint(); } }),
      h('span', h('b', s.label), h('div.small.text-muted', s.url)))));
    const save = h('button.btn.btn-primary', { type: 'submit' }, icon('shield-check'), ' Test & save');
    const reveal = h('button.btn.btn-outline-secondary', { type: 'button', title: 'Show / hide',
      onclick: () => { keyInput.type = keyInput.type === 'password' ? 'text' : 'password'; } }, icon('eye'));
    const form = h('form', {
      onsubmit: async (e) => {
        e.preventDefault();
        const key = keyInput.value.trim();
        if (!key) { mount(msg, h('span.text-danger', 'Paste your API key first.')); keyInput.focus(); return; }
        save.disabled = true;
        mount(msg, h('span.text-muted', 'Checking the key with W&B…'));
        try {
          state = await call('POST', { server, api_key: key });
          keyInput.value = '';
          toast(`Connected as ${state.active.wandb_user || 'you'} on ${state.active.server_label}. Reloading…`, 'success', { title: 'W&B connection saved' });
          setTimeout(() => window.location.assign(config.root || '/models/'), 900);   // every page re-reads with the new key
        } catch (err) {
          mount(msg, h('span.text-danger', icon('exclamation-triangle'), ' ', err.message));
        } finally { save.disabled = false; }
      },
    },
      h('div.mb-3', h('div.form-label.fw-semibold', 'W&B server'), options),
      h('div.mb-2', h('label.form-label.fw-semibold', 'API key'), h('div.input-group', keyInput, reveal), hint),
      h('div.d-flex.gap-2.align-items-center', save), msg,
      h('div.small.text-muted.mt-3', icon('lock'), ' The key is checked with W&B first, then stored encrypted on the server. It is never shown again or sent to your browser; Model Studio acts as that W&B user for you only — other people keep their own connection.'));
    setHint();
    mount(formEl, card({ title: own ? 'Change your key' : 'Use your own W&B key', icon: 'key', body: form }));
  }

  async function remove() {
    const ok = await confirmDialog({
      title: 'Remove your W&B key?',
      message: state.shared_available ? 'Model Studio goes back to the shared server key for you.' : 'Model Studio will not be connected to W&B for you until you add a key again.',
      confirmLabel: 'Remove', danger: true,
    });
    if (!ok) return;
    try {
      state = await call('DELETE');
      toast('Your key was removed. Reloading…', 'success');
      setTimeout(() => window.location.assign(config.root || '/models/'), 700);
    } catch (err) { toast(err.message, 'error', { title: 'Could not remove the key' }); }
  }
}
