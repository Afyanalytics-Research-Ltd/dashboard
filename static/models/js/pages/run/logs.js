// Run page — Logs: virtualized console output with search (highlight + match navigation),
// basic ANSI colour, wrap toggle, stderr filter, live tail for running runs, download.

import { h, icon, mount, debounce, prefs } from '../../../../ui/js/dom.js';
import { emptyState, errorState, toast } from '../../../../ui/js/ui.js';
import { gql } from '../../core/api.js';
import { downloadText } from '../../core/charts.js';

const LINE_H = 20;
const OVERSCAN = 30;
const ANSI = { 30: '#94a3b8', 31: '#f87171', 32: '#4ade80', 33: '#facc15', 34: '#60a5fa', 35: '#c084fc', 36: '#22d3ee', 37: '#e2e8f0',
  90: '#64748b', 91: '#fca5a5', 92: '#86efac', 93: '#fde047', 94: '#93c5fd', 95: '#d8b4fe', 96: '#67e8f9', 97: '#f8fafc' };

/** Split a line with ANSI SGR codes into [{text, color, bold}] segments. */
export function ansiSegments(line) {
  const out = [];
  let color = null;
  let bold = false;
  let last = 0;
  const re = /\x1b\[([\d;]*)m/g;
  let m;
  while ((m = re.exec(line))) {
    if (m.index > last) out.push({ text: line.slice(last, m.index), color, bold });
    for (const code of (m[1] || '0').split(';').map(Number)) {
      if (code === 0) { color = null; bold = false; } else if (code === 1) bold = true;
      else if (code === 22) bold = false;
      else if (code === 39) color = null;
      else if (ANSI[code]) color = ANSI[code];
    }
    last = re.lastIndex;
  }
  if (last < line.length) out.push({ text: line.slice(last), color, bold });
  return out.length ? out : [{ text: '', color: null, bold: false }];
}
const stripAnsi = (s) => s.replace(/\x1b\[[\d;]*m/g, '');

async function fetchLines(entity, project, runName, after, signal) {
  const d = await gql(`query RunLogs($e: String!, $p: String!, $r: String!, $after: String) {
    project(entityName: $e, name: $p) { run(name: $r) { logLineCount state
      logLines(first: 1000, after: $after) { pageInfo { hasNextPage endCursor } edges { node { number line level label timestamp } } } } } }`,
  { e: entity, p: project, r: runName, after }, { signal });
  const run = d.project?.run;
  return { conn: run?.logLines, count: run?.logLineCount, state: run?.state };
}

export async function logsTab(el, ctx, run) {
  const { entity, project } = ctx.params;
  const lines = [];
  let cursor = null;
  let total = run.logLineCount ?? 0;
  let state = run.state;
  const opts = prefs.get('ms:run:logs', { wrap: false, follow: true, stderrOnly: false, timestamps: false });

  mount(el, h('div.ms-panel-loading'));
  try {
    for (;;) {
      const { conn, count, state: st } = await fetchLines(entity, project, run.name, cursor, ctx.signal);
      total = count ?? total;
      state = st || state;
      if (!conn) break;
      const lastNo = lines.length ? (lines[lines.length - 1].number ?? -1) : -1;
      const fresh = conn.edges.map((e) => e.node).filter((n) => lines.length === 0 || (n.number ?? Infinity) > lastNo);
      lines.push(...fresh);
      cursor = conn.pageInfo.endCursor || cursor;
      if (!conn.pageInfo.hasNextPage || !fresh.length || lines.length >= 200000) break;
    }
  } catch (err) {
    if (err.name !== 'AbortError') mount(el, errorState(err, () => logsTab(el, ctx, run)));
    return;
  }
  if (!lines.length && state !== 'running') {
    mount(el, emptyState({ icon: 'terminal', title: 'No console output', message: 'Anything this run printed to stdout/stderr would appear here.' }));
    return;
  }

  // ---- view state
  let q = '';
  let view = [];
  let matches = [];
  let current = -1;
  const scroller = h('div.msr-log', { class: opts.wrap && 'wrap', tabindex: 0, 'aria-label': 'Console output' });
  const spacer = h('div.msr-log-spacer');
  const win = h('div.msr-log-window');
  scroller.append(spacer, win);
  const countEl = h('span.msr-log-count');
  const matchEl = h('span.msr-log-matches');
  const liveEl = h('span.msr-log-live');

  const isErr = (l) => l.level === 'error' || l.label === 'stderr' || /\b(error|exception|traceback)\b/i.test(l.line || '');
  const recompute = () => {
    view = opts.stderrOnly ? lines.filter(isErr) : lines;
    matches = [];
    if (q) view.forEach((l, i) => { if (stripAnsi(l.line || '').toLowerCase().includes(q)) matches.push(i); });
    if (current >= matches.length) current = matches.length - 1;
    countEl.textContent = `${view.length.toLocaleString()} of ${Math.max(total || 0, lines.length).toLocaleString()} lines`;
    matchEl.textContent = q ? (matches.length ? `${current + 1 > 0 ? current + 1 : 0} / ${matches.length}` : 'no matches') : '';
  };

  const renderLine = (l, idx) => {
    const text = l.line || '';
    const segs = ansiSegments(text);
    const content = [];
    for (const s of segs) {
      const node = q ? markText(stripAnsi(s.text), q) : s.text;
      content.push(s.color || s.bold ? h('span', { style: { color: s.color, fontWeight: s.bold ? 700 : null } }, node) : node);
    }
    return h('div.msr-log-line', { class: [isErr(l) && 'err', matches[current] === idx && 'current', q && matches.includes(idx) && 'hit'], dataset: { idx } },
      h('span.msr-log-no', String((l.number ?? idx) + 1)),
      opts.timestamps && h('span.msr-log-ts', (l.timestamp || '').slice(11, 19)),
      h('span.msr-log-text', content.length ? content : ' '));
  };

  const paint = () => {
    if (opts.wrap) {
      // Wrapped lines have variable height: render everything (capped) instead of virtualizing.
      spacer.style.height = '0px';
      win.style.transform = 'none';
      const cap = 20000;
      mount(win, view.slice(0, cap).map(renderLine), view.length > cap && h('div.msr-log-more', `Showing first ${cap.toLocaleString()} lines while wrapping — turn wrap off to scroll everything.`));
      return;
    }
    spacer.style.height = `${view.length * LINE_H}px`;
    const top = scroller.scrollTop;
    const start = Math.max(0, Math.floor(top / LINE_H) - OVERSCAN);
    const end = Math.min(view.length, Math.ceil((top + scroller.clientHeight) / LINE_H) + OVERSCAN);
    win.style.transform = `translateY(${start * LINE_H}px)`;
    const frag = [];
    for (let i = start; i < end; i++) frag.push(renderLine(view[i], i));
    mount(win, frag);
  };
  let raf = 0;
  let programmatic = false;
  scroller.addEventListener('scroll', () => {
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; paint(); });
    if (programmatic) { programmatic = false; return; }
    // Scrolling up by hand pauses Follow; scrolling back to the bottom resumes it.
    const atBottom = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - LINE_H * 3;
    if (opts.follow && !atBottom) setFollow(false);
    else if (!opts.follow && atBottom && state === 'running') setFollow(true);
  });

  const goto = (dir) => {
    if (!matches.length) return;
    current = (current + dir + matches.length) % matches.length;
    recompute();
    const idx = matches[current];
    if (opts.wrap) {
      paint();
      win.querySelector(`[data-idx="${idx}"]`)?.scrollIntoView({ block: 'center' });
    } else {
      scroller.scrollTop = Math.max(0, idx * LINE_H - scroller.clientHeight / 2);
      paint();
    }
  };

  const search = h('input.form-control.form-control-sm', {
    type: 'search', placeholder: 'Search logs…', dataset: { searchShortcut: '' }, 'aria-label': 'Search logs',
    oninput: debounce((e) => { q = e.target.value.trim().toLowerCase(); current = -1; recompute(); paint(); if (matches.length) goto(1); }, 150),
    onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); goto(e.shiftKey ? -1 : 1); } },
  });
  const toggle = (key, label, ic, after) => {
    const b = h('button.btn.btn-sm.btn-outline-secondary.msr-toggle', { type: 'button', class: opts[key] && 'active', 'aria-pressed': String(!!opts[key]), title: label },
      icon(ic, 'me-1'), label);
    b.addEventListener('click', () => {
      opts[key] = !opts[key];
      prefs.set('ms:run:logs', opts);
      b.classList.toggle('active', opts[key]);
      b.setAttribute('aria-pressed', String(opts[key]));
      after?.();
    });
    return b;
  };
  let followBtn;
  const setFollow = (v) => { opts.follow = v; followBtn?.classList.toggle('active', v); followBtn?.setAttribute('aria-pressed', String(v)); };
  followBtn = toggle('follow', 'Follow', 'arrow-down-circle', () => { if (opts.follow) scrollToEnd(); });

  // Grow the spacer first, then scroll — otherwise the new bottom isn't reachable yet.
  const scrollToEnd = () => {
    paint();
    programmatic = true;
    scroller.scrollTop = scroller.scrollHeight;
    paint();
  };
  const fullText = () => lines.map((l) => stripAnsi(l.line || '')).join('\n');

  mount(el, h('div.msr-logs',
    h('div.msr-log-toolbar',
      h('div.af-search.msr-log-search', icon('search', 'af-search-icon'), search),
      h('div.btn-group',
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Previous match (Shift+Enter)', onclick: () => goto(-1) }, icon('chevron-up')),
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Next match (Enter)', onclick: () => goto(1) }, icon('chevron-down'))),
      matchEl,
      h('div.msr-log-spacer-flex'),
      toggle('stderrOnly', 'Errors only', 'exclamation-octagon', () => { recompute(); paint(); }),
      toggle('wrap', 'Wrap', 'text-wrap', () => { scroller.classList.toggle('wrap', opts.wrap); paint(); }),
      toggle('timestamps', 'Time', 'clock', () => paint()),
      state === 'running' && followBtn,
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Jump to first error',
        onclick: () => {
          const i = view.findIndex(isErr);
          if (i < 0) { toast('No error lines found', 'info', { timeout: 1800 }); return; }
          scroller.scrollTop = Math.max(0, i * LINE_H - 60); paint();
        } }, icon('bug')),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Download output.log', onclick: () => downloadText(fullText(), `${run.name}-output.log`, 'text/plain') }, icon('download')),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Copy all', onclick: async () => { await navigator.clipboard.writeText(fullText()); toast('Logs copied', 'success', { timeout: 1500 }); } }, icon('clipboard'))),
    scroller,
    h('div.msr-log-foot', countEl, liveEl)));
  recompute();
  requestAnimationFrame(() => { if (state === 'running' && opts.follow) scrollToEnd(); else paint(); });

  // ---- live tail
  if (state === 'running') {
    mount(liveEl, h('span.af-dot.af-pulse', { style: { background: '#16a34a' } }), ' Live — new lines appear automatically');
    let ended = false;
    ctx.every(3000, async () => {
      if (ended) return;
      try {
        const { conn, count, state: st } = await fetchLines(entity, project, run.name, cursor, ctx.signal);
        total = count ?? total;
        // W&B's logLines(after:) cursor can replay lines already sent; keep only new line numbers.
        const lastNo = lines.length ? (lines[lines.length - 1].number ?? lines.length - 1) : -1;
        const fresh = (conn?.edges || []).map((e) => e.node).filter((n) => (n.number ?? Infinity) > lastNo);
        if (fresh.length) {
          lines.push(...fresh);
          cursor = conn.pageInfo.endCursor || cursor;
          recompute();
          if (opts.follow) scrollToEnd(); else paint();
        }
        if (st && st !== 'running') { ended = true; mount(liveEl, h('span.text-muted', `Run ${st} — live tail stopped`)); }
      } catch { /* transient; next tick retries */ }
    });
  }
}

function markText(text, q) {
  const lower = text.toLowerCase();
  const out = [];
  let i = 0;
  for (;;) {
    const j = lower.indexOf(q, i);
    if (j < 0) { out.push(text.slice(i)); break; }
    out.push(text.slice(i, j), h('mark', text.slice(j, j + q.length)));
    i = j + q.length;
  }
  return out;
}
