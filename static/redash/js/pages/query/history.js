// Version history: revisions recorded by the Studio backend on every save,
// side-by-side line diff (LCS) against the current text, one-click restore.

import { h, icon, mount } from '../../../../ui/js/dom.js';
import { drawer, emptyState, errorState, skeleton, timeAgo, confirmDialog } from '../../../../ui/js/ui.js';
import { fmtDate } from '../../../../ui/js/format.js';

/** Line diff via LCS. Returns [{type: 'same'|'add'|'del', text, a, b}] (a/b = 1-based line numbers). */
export function diffLines(oldText, newText) {
  const a = String(oldText ?? '').split('\n');
  const b = String(newText ?? '').split('\n');
  const n = a.length;
  const m = b.length;
  // Guard against pathological sizes (O(n*m) memory).
  if (n * m > 4e6) {
    return [...a.map((t, i) => ({ type: 'del', text: t, a: i + 1 })), ...b.map((t, i) => ({ type: 'add', text: t, b: i + 1 }))];
  }
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push({ type: 'same', text: a[i], a: i + 1, b: j + 1 }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ type: 'del', text: a[i], a: i + 1 }); i++; }
    else { out.push({ type: 'add', text: b[j], b: j + 1 }); j++; }
  }
  while (i < n) { out.push({ type: 'del', text: a[i], a: i + 1 }); i++; }
  while (j < m) { out.push({ type: 'add', text: b[j], b: j + 1 }); j++; }
  return out;
}

export function diffView(oldText, newText, { context = 3 } = {}) {
  const lines = diffLines(oldText, newText);
  const changed = lines.map((l) => l.type !== 'same');
  if (!changed.some(Boolean)) return h('div.rd-diff-identical', icon('check2-circle', 'me-1'), 'Identical to the current text.');
  const keep = lines.map((_, i) => changed.slice(Math.max(0, i - context), i + context + 1).some(Boolean));
  const adds = lines.filter((l) => l.type === 'add').length;
  const dels = lines.filter((l) => l.type === 'del').length;
  const rows = [];
  let skipped = 0;
  lines.forEach((l, i) => {
    if (!keep[i]) { skipped++; return; }
    if (skipped) { rows.push(h('div.rd-diff-gap', `⋯ ${skipped} unchanged line${skipped > 1 ? 's' : ''}`)); skipped = 0; }
    rows.push(h(`div.rd-diff-line.rd-diff-${l.type}`,
      h('span.rd-diff-num', l.a ?? ''), h('span.rd-diff-num', l.b ?? ''),
      h('span.rd-diff-sign', l.type === 'add' ? '+' : l.type === 'del' ? '−' : ' '),
      h('span.rd-diff-text', l.text || ' ')));
  });
  if (skipped) rows.push(h('div.rd-diff-gap', `⋯ ${skipped} unchanged line${skipped > 1 ? 's' : ''}`));
  return h('div',
    h('div.rd-diff-stats', h('span.rd-diff-stat-add', `+${adds}`), h('span.rd-diff-stat-del', `−${dels}`), h('span.text-muted', ' compared with the text in the editor')),
    h('div.rd-diff', rows));
}

/**
 * openHistory({ queryId, current: () => ({query, name}), onRestore(revision) })
 * Revisions come from GET /redash/history/<id>/ (Studio backend).
 */
export function openHistory({ queryId, current, onRestore }) {
  const listEl = h('div.rd-hist-list', skeleton(6));
  const detailEl = h('div.rd-hist-detail', emptyState({ icon: 'clock-history', title: 'Pick a version', message: 'Select a revision on the left to compare it with the editor.' }));
  const d = drawer({
    title: h('span', icon('clock-history', 'me-2'), 'Version history'),
    subtitle: `Query #${queryId} · every save made in Analytics Studio is kept`,
    width: 1100,
    body: h('div.rd-hist', listEl, detailEl),
  });

  let revisions = [];
  let selected = null;

  const select = (rev) => {
    selected = rev;
    listEl.querySelectorAll('.rd-hist-item').forEach((b) => b.classList.toggle('active', Number(b.dataset.id) === rev.id));
    const cur = current();
    mount(detailEl,
      h('div.rd-hist-detail-head',
        h('div',
          h('div.fw-semibold', rev.name || 'Untitled'),
          h('div.small.text-muted', `${fmtDate(rev.created_at)} · ${rev.user || 'unknown user'}${rev.version ? ` · Redash v${rev.version}` : ''}`)),
        h('button.btn.btn-sm.btn-primary', {
          onclick: async () => {
            const ok = await confirmDialog({
              title: 'Restore this version?',
              message: 'The editor text (and name, if different) is replaced with this version. Nothing is saved until you press Save, and you can undo with Ctrl+Z.',
              confirmLabel: 'Restore into editor',
            });
            if (!ok) return;
            onRestore(rev);
            d.close();
          },
        }, icon('arrow-counterclockwise', 'me-1'), 'Restore')),
      rev.name !== cur.name && h('div.af-callout.af-callout-info.my-2', icon('info-circle'), h('span', 'Name at that time: ', h('b', rev.name || 'Untitled'))),
      diffView(rev.query, cur.query));
  };

  fetch(`/redash/history/${encodeURIComponent(queryId)}/`, { credentials: 'same-origin', headers: { Accept: 'application/json' } })
    .then(async (r) => {
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw Object.assign(new Error(body.message || `History unavailable (${r.status})`), { status: r.status });
      return body;
    })
    .then((body) => {
      revisions = body.revisions || [];
      if (!revisions.length) {
        mount(listEl, emptyState({ icon: 'clock-history', title: 'No saved versions yet', message: 'Versions are recorded from now on, every time this query is saved in Analytics Studio.' }));
        return;
      }
      mount(listEl, revisions.map((rev, i) => h('button.rd-hist-item', {
        type: 'button', dataset: { id: rev.id }, onclick: () => select(rev),
      },
      h('div.rd-hist-item-top', h('span.fw-semibold', i === 0 ? 'Latest save' : `Version ${revisions.length - i}`), timeAgo(rev.created_at)),
      h('div.rd-hist-item-sub', (() => { const n = String(rev.query || '').split('\n').length; return `${rev.user || 'unknown'} · ${n} line${n === 1 ? '' : 's'}`; })()))));
      select(revisions[0]);
    })
    .catch((err) => mount(listEl, errorState(err)));

  return d;
}
