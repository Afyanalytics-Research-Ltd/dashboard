// Connect Studio component kit: logos, rings, type chips, rule visuals, live job tracker,
// dynamic connector forms, dropzones, JSON highlighting.

import { h, s, icon, mount } from '../../ui/js/dom.js';
import { relTime, fmtNumber, fmtBytes, fmtDuration } from '../../ui/js/format.js';
import { drawer, toast } from '../../ui/js/ui.js';
import { api } from './api.js';

export { relTime, fmtNumber, fmtBytes, fmtDuration };

export const n = (v) => (v == null ? '—' : fmtNumber(v));
export const compact = (v) => {
  if (v == null) return '—';
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e4) return `${(v / 1e3).toFixed(1)}k`;
  return fmtNumber(v);
};
export const pct = (v, d = 0) => (v == null ? '—' : `${(v * 100).toFixed(d)}%`);
export const ago = (v) => (v ? relTime(v) : 'never');

// ------------------------------------------------------------------ logos

export function logo(spec, { size = '', solid = false } = {}) {
  const sp = spec || {};
  return h(`span.cs-logo${size ? `.cs-logo-${size}` : ''}${solid ? '.cs-logo-solid' : ''}`, { style: { '--c': sp.color || '#64748b' }, title: sp.label || '' },
    icon(sp.icon || 'plug'));
}

// ------------------------------------------------------------------ rings & bars

export function ring(value, { size = 44, stroke = 5, color, label, sub, track } = {}) {
  const v = Math.max(0, Math.min(1, value || 0));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const col = color || (v >= 0.9 ? '#16a34a' : v >= 0.6 ? '#0bb99f' : v >= 0.3 ? '#f59e0b' : '#94a3b8');
  const el = h('span.cs-ring', { style: { width: `${size}px`, height: `${size}px` } },
    s('svg', { width: size, height: size, viewBox: `0 0 ${size} ${size}` },
      s('circle.track', { cx: size / 2, cy: size / 2, r, fill: 'none', 'stroke-width': stroke, style: track ? `stroke:${track}` : undefined }),
      s('circle.val', { cx: size / 2, cy: size / 2, r, fill: 'none', stroke: col, 'stroke-width': stroke, 'stroke-dasharray': c, 'stroke-dashoffset': c * (1 - v) })),
    label !== false && h('span.cs-ring-label', { style: { fontSize: `${Math.max(10, size * 0.27)}px` } }, label ?? `${Math.round(v * 100)}`));
  if (sub) el.title = sub;
  return el;
}

export function bar(value, { cls = '' } = {}) {
  return h(`div.cs-bar${cls ? `.${cls}` : ''}`, h('span', { style: { width: `${Math.max(0, Math.min(100, value))}%` } }));
}

export function confidence(conf) {
  const v = conf ?? 0;
  const col = v >= 0.9 ? '#16a34a' : v >= 0.7 ? '#0bb99f' : v >= 0.5 ? '#f59e0b' : '#dc2626';
  return h('span.cs-conf', { title: `Confidence ${Math.round(v * 100)}%` },
    h('span.cs-conf-bar', h('span', { style: { width: `${v * 100}%`, background: col } })), h('span', { style: { color: col } }, `${Math.round(v * 100)}`));
}

// ------------------------------------------------------------------ types

export const FAMILY = {
  integer: { glyph: '#', color: '#2563eb' }, decimal: { glyph: '1.0', color: '#0891b2' }, text: { glyph: 'Aa', color: '#475569' },
  boolean: { glyph: '◐', color: '#9333ea' }, date: { glyph: '📅', color: '#ea580c', icon: 'calendar3' }, datetime: { icon: 'clock', color: '#ea580c' },
  time: { icon: 'stopwatch', color: '#ea580c' }, json: { glyph: '{}', color: '#db2777' }, uuid: { icon: 'key', color: '#0d9488' },
  binary: { glyph: '01', color: '#64748b' }, other: { glyph: '?', color: '#94a3b8' }, unknown: { glyph: '·', color: '#94a3b8' },
};

export function typeChip(col) {
  const fam = FAMILY[col.family] || FAMILY.unknown;
  const glyph = fam.icon ? icon(fam.icon) : fam.glyph;
  return h('span.cs-type', { title: `${col.type || 'unknown type'} · ${col.family || ''}`, style: { '--c': fam.color } },
    h('b', glyph), (col.type || col.family || '').toLowerCase().slice(0, 22));
}

export function keyBadges(col) {
  const out = [];
  if (col.pk) out.push(h('span.cs-key.cs-key-pk', { title: 'Primary key' }, 'PK'));
  if (col.references) out.push(h('span.cs-key.cs-key-fk', { title: `→ ${col.references.table}.${col.references.column} (${col.references.kind})` }, col.references.kind === 'inferred' ? 'FK?' : 'FK'));
  if (col.unique && !col.pk) out.push(h('span.cs-key.cs-key-uq', { title: 'Unique' }, 'UQ'));
  for (const p of col.profile?.pii || []) out.push(h('span.cs-key.cs-key-pii', { title: `Looks like ${p}` }, 'PII'));
  return out;
}

// ------------------------------------------------------------------ rule kinds

export const KIND = {
  copied: { icon: 'arrow-right', cls: 'copied', label: 'Copy', wire: '#0bb99f', help: 'Use the source value as is.' },
  'worked out': { icon: 'magic', cls: 'worked', label: 'Steps', wire: '#0ea5e9', help: 'Clean the text: lowercase, trim, slug…' },
  'codes converted': { icon: 'arrow-left-right', cls: 'codes', label: 'Code map', wire: '#f59e0b', help: 'Translate codes/labels to the target’s values.' },
  'looked up in another table': { icon: 'link-45deg', cls: 'lookup', label: 'Lookup', wire: '#6366f1', help: 'Fetch a value from a related table.' },
  'one column divided': { icon: 'scissors', cls: 'divide', label: 'Split', wire: '#ec4899', help: 'Take one part of a delimited value.' },
  constant: { icon: 'pin-angle', cls: 'constant', label: 'Constant', wire: '#94a3b8', help: 'Always the same value.' },
};

export function kindIcon(kind) {
  const k = KIND[kind] || KIND.copied;
  return h(`span.kind.cs-kind-${k.cls}`, { title: k.label }, icon(k.icon));
}

export function madeBy(who) {
  const w = String(who || '');
  if (w === 'human') return h('span.cs-madeby.human', { title: 'Set by a person' }, icon('person-fill'));
  if (w.includes('ai')) return h('span.cs-madeby.ai', { title: w === 'rule+ai' ? 'Rule match confirmed by AI' : 'Suggested by AI (Groq)' }, icon('stars'));
  return h('span.cs-madeby.rule', { title: 'Rule-based match' }, icon('lightning-charge-fill'));
}

export function describe(f) {
  const r = f.rule || {};
  const src = f.source || '';
  switch (f.kind) {
    case 'copied': return src + (f.transform ? ` → ${f.transform}` : '');
    case 'worked out': return `${src} → ${(r.steps || []).map((x) => x.op).join(' → ')}`;
    case 'codes converted': {
      const pairs = Object.entries(r.map || {});
      return `${src}: ${pairs.slice(0, 3).map(([a, b]) => `${a}→${b}`).join(', ')}${pairs.length > 3 ? ` +${pairs.length - 3}` : ''}`;
    }
    case 'looked up in another table': return `${r.table}.${r.value} by ${r.key} = ${src}`;
    case 'one column divided': return `${src} ⟂ ${JSON.stringify(r.delimiter || ' ')} part ${r.part}`;
    case 'constant': return `= ${JSON.stringify(r.value)}`;
    default: return src;
  }
}

const STEPS = {
  lowercase: (x) => x.toLowerCase(), uppercase: (x) => x.toUpperCase(), trim: (x) => x.trim(),
  slug: (x) => x.normalize('NFKD').replace(/[^\w\s-]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''),
  first_word: (x) => (x.split(/\s+/).filter(Boolean)[0] || ''), rest_words: (x) => x.split(/\s+/).filter(Boolean).slice(1).join(' '),
  title_case: (x) => x.replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase()),
};

/** Client-side port of rules.apply (lookups need the server preview). */
export function applyRule(f, value) {
  const r = f.rule || {};
  if (f.kind === 'constant') return r.value;
  if (value == null) return null;
  let v = value;
  if (f.kind === 'looked up in another table') return undefined;
  if (r.part != null && typeof v === 'string') {
    const pieces = r.delimiter ? v.split(r.delimiter) : v.split(/\s+/).filter(Boolean);
    if (r.part === 'rest') v = pieces.slice(1).join(r.delimiter || ' ') || null;
    else v = pieces[Number(r.part) - 1] ?? null;
  }
  if (r.map && v != null) {
    const table = Object.fromEntries(Object.entries(r.map).map(([k, x]) => [String(k).toLowerCase(), x]));
    const key = String(v).trim().toLowerCase();
    v = key in table ? table[key] : ('default' in r ? r.default : v);
  }
  for (const st of [...(r.steps || []), ...(f.transform ? [{ op: f.transform }] : [])]) {
    const fn = STEPS[String(st.op || '').toLowerCase()];
    if (fn && typeof v === 'string') v = fn(v);
  }
  return v;
}

// ------------------------------------------------------------------ status badges

export function statusBadge(status) {
  const map = {
    ok: ['ok', 'check-circle-fill', 'Connected'], error: ['bad', 'x-octagon-fill', 'Error'], draft: ['outline', 'pencil', 'Draft'],
    untested: ['warn', 'question-circle', 'Untested'], review: ['warn', 'eye', 'In review'], approved: ['ok', 'patch-check-fill', 'Approved'],
    generating: ['ai', 'stars', 'Generating'], failed: ['bad', 'x-octagon-fill', 'Failed'], succeeded: ['ok', 'check-circle-fill', 'Succeeded'],
    running: ['info', 'arrow-repeat', 'Running'], queued: ['outline', 'hourglass-split', 'Queued'], cancelled: ['outline', 'slash-circle', 'Cancelled'],
    stalled: ['warn', 'exclamation-triangle', 'Stalled'], partial: ['warn', 'exclamation-circle', 'Partial'], suggested: ['ai', 'stars', 'Suggested'],
    rejected: ['bad', 'x-circle', 'Rejected'],
  };
  const [cls, ic, label] = map[status] || ['outline', 'circle', status || '—'];
  return h(`span.cs-badge.cs-badge-${cls}`, icon(status === 'running' ? 'arrow-repeat' : ic, status === 'running' ? 'cs-spin' : ''), label);
}

export function statusDot(status) {
  const cls = { ok: 'ok', succeeded: 'ok', approved: 'ok', error: 'bad', failed: 'bad', running: 'run', generating: 'run', queued: 'run', review: 'warn', untested: 'warn', partial: 'warn', stalled: 'warn' }[status] || '';
  return h(`span.cs-dot${cls ? `.${cls}` : ''}`);
}

// ------------------------------------------------------------------ job tracker

const PHASES = {
  test: [['connect', 'Connect'], ['done', 'Verified']],
  extract: [['connect', 'Connect'], ['introspect', 'Introspect'], ['sample', 'Sample & profile'], ['analyze', 'Relationships'], ['done', 'Snapshot']],
  map: [['extract', 'Snapshots'], ['tables', 'Match tables'], ['fields', 'Map columns'], ['done', 'Ready']],
  remap: [['fields', 'Map columns'], ['done', 'Ready']],
  sync: [['connect', 'Connect'], ['sync', 'Move rows'], ['done', 'Done']],
};

/**
 * Live view of a background job. Returns {el, done: Promise<job>, stop()}.
 * opts: { title, ai, compact, showLog, onUpdate(job) }
 */
export function jobTracker(jobId, opts = {}) {
  let since = 0;
  let stopped = false;
  let timer = null;
  const lines = [];
  const orb = h('div.cs-job-orb.running', { class: opts.ai && 'ai' }, icon(opts.ai ? 'stars' : 'cpu'));
  const titleEl = h('div.cs-job-title', opts.title || 'Working…');
  const msgEl = h('div.cs-job-msg', 'Starting…');
  const pctEl = h('span.cs-mono.cs-muted', '0%');
  const barEl = bar(0, { cls: opts.ai ? 'ai' : '' });
  const phasesEl = h('div.cs-phases');
  const consoleEl = h('div.cs-console', h('div.cs-console-empty', 'Waiting for the first log line…'));
  const cancelBtn = h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => api.cancelJob(jobId).then(() => toast('Cancelling…', 'info')) }, icon('stop-circle'), 'Cancel');
  const logToggle = h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => consoleEl.classList.toggle('cs-hidden') }, icon('terminal'), 'Log');
  if (opts.showLog === false) consoleEl.classList.add('cs-hidden');
  const el = h('div.cs-job',
    h('div.cs-job-top', orb, h('div.cs-grow', titleEl, msgEl), pctEl, cancelBtn, logToggle),
    barEl, phasesEl, consoleEl);
  let resolveDone;
  const done = new Promise((res) => { resolveDone = res; });

  function renderPhases(job) {
    const list = PHASES[job.kind] || [];
    const idx = list.findIndex(([k]) => k === job.phase);
    mount(phasesEl, list.map(([k, label], i) => h('span.cs-phase', { class: [job.status === 'succeeded' || i < idx ? 'done' : i === idx ? 'on' : ''] },
      icon(job.status === 'succeeded' || i < idx ? 'check2' : i === idx ? 'circle-fill' : 'circle'), label)));
  }

  function addLines(newLines) {
    if (!newLines?.length) return;
    if (!lines.length) mount(consoleEl);
    const atBottom = consoleEl.scrollHeight - consoleEl.scrollTop - consoleEl.clientHeight < 30;
    for (const l of newLines) {
      lines.push(l);
      consoleEl.appendChild(h(`div.cs-console-line.${l.level || 'info'}`, h('span.ts', (l.t || '').slice(11, 19)), h('span', l.msg)));
    }
    if (atBottom) consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  async function tick() {
    if (stopped) return;
    try {
      const { job } = await api.job(jobId, since);
      since += job.log?.length || 0;
      addLines(job.log);
      if (job.log?.length) opts.onLines?.(job.log);
      if (!opts.title) mount(titleEl, job.title || job.kind_label);
      mount(msgEl, job.status === 'failed' ? (job.error || job.message) : job.message || '…');
      const p = job.status === 'succeeded' ? 100 : job.progress || 0;
      barEl.firstChild.style.width = `${Math.max(p, job.status === 'running' ? 3 : 0)}%`;
      mount(pctEl, `${Math.round(p)}%`);
      renderPhases(job);
      opts.onUpdate?.(job);
      const final = ['succeeded', 'failed', 'cancelled', 'stalled'].includes(job.status);
      if (final) {
        orb.classList.remove('running');
        orb.classList.add(job.status === 'succeeded' ? 'succeeded' : 'failed');
        mount(orb, icon(job.status === 'succeeded' ? 'check-lg' : 'x-lg'));
        barEl.classList.add('done');
        if (job.status !== 'succeeded') barEl.classList.add('bad');
        cancelBtn.remove();
        stopped = true;
        resolveDone(job);
        return;
      }
    } catch (e) {
      if (e.status === 404) { stopped = true; resolveDone({ status: 'failed', error: 'Job not found' }); return; }
    }
    timer = setTimeout(tick, document.hidden ? 3000 : 900);
  }
  tick();
  return { el, done, stop: () => { stopped = true; clearTimeout(timer); } };
}

/** Wrap overlay content so it picks up the studio's theme variables. */
export function themed(...children) {
  return h('div.cs-root', { 'data-theme': document.getElementById('cs-root')?.dataset.theme || 'light' }, ...children);
}

export function openJobDrawer(job) {
  const tracker = jobTracker(job.id, { title: job.title, ai: ['map', 'remap'].includes(job.kind) });
  const d = drawer({ title: job.title || job.kind_label, subtitle: `Job #${job.id} · ${job.kind_label}`, body: themed(tracker.el), width: 720 });
  d.onCleanup(() => tracker.stop());
  return d;
}

// ------------------------------------------------------------------ dynamic connector form

/**
 * Renders a connector spec's fields. Returns {el, values(), setValues(v)}.
 * opts: { values, secretsSet: [names], include: ['name'] }
 */
export function connectorForm(spec, { values = {}, secretsSet = [], name, onChange } = {}) {
  const inputs = {};
  const rows = {};
  const groups = new Map();
  const advanced = new Map();
  const nameInput = name !== undefined ? h('input.cs-input', { value: name || '', placeholder: `My ${spec.label}`, maxlength: 200 }) : null;

  for (const f of spec.fields.filter((x) => x.type !== 'file')) {
    const id = `cs-f-${f.name}-${Math.random().toString(36).slice(2, 6)}`;
    const v = values[f.name] ?? f.default ?? '';
    let input;
    if (f.type === 'select') {
      input = h('select.cs-input', { id }, (f.options || []).map((o) => h('option', { value: o.value, selected: String(o.value) === String(v) }, o.label)));
    } else if (f.type === 'checkbox') {
      input = h('input', { id, type: 'checkbox', checked: !!(values[f.name] ?? f.default) });
    } else if (f.type === 'textarea' || f.type === 'code') {
      input = h(`textarea.cs-input${f.type === 'code' ? '.code' : ''}`, { id, rows: f.rows || 3, placeholder: f.placeholder || '', spellcheck: 'false' },
        f.secret ? '' : String(v ?? ''));
    } else {
      input = h('input.cs-input', { id, type: f.type === 'password' ? 'password' : f.type === 'number' ? 'number' : 'text', value: f.secret ? '' : (v ?? ''),
        placeholder: f.secret && secretsSet.includes(f.name) ? '•••••••• (saved — leave blank to keep)' : f.placeholder || '', autocomplete: f.secret ? 'new-password' : 'off' });
    }
    input.addEventListener('input', () => { refresh(); onChange?.(); });
    input.addEventListener('change', () => { refresh(); onChange?.(); });
    inputs[f.name] = input;
    const label = h('label', { for: id }, f.label, f.required && h('span.req', '*'), f.secret && h('i.bi.bi-lock-fill.cs-lock', { title: 'Encrypted at rest' }));
    let control = input;
    if (f.type === 'checkbox') {
      control = h('label.cs-switch', input, h('span.track'), h('span', f.label));
    }
    const wide = ['textarea', 'code'].includes(f.type) || f.name === 'base_url' || f.name === 'url';
    const row = h('div.cs-field', { class: wide && 'full' },
      f.type !== 'checkbox' && label,
      f.secret && secretsSet.includes(f.name) && f.type !== 'textarea' ? h('div.cs-input-wrap', control, h('span.cs-secret-set', icon('shield-check'), ' saved')) : control,
      f.help && h('div.help', f.help));
    rows[f.name] = row;
    const bucket = f.advanced ? advanced : groups;
    if (!bucket.has(f.group)) bucket.set(f.group, []);
    bucket.get(f.group).push(row);
  }

  function refresh() {
    for (const f of spec.fields) {
      if (!f.show_if || !rows[f.name]) continue;
      const dep = inputs[f.show_if.field];
      const cur = dep ? (dep.type === 'checkbox' ? String(dep.checked) : dep.value) : '';
      rows[f.name].classList.toggle('cs-hidden', !f.show_if.in.map(String).includes(cur));
    }
  }

  const el = h('div.cs-form',
    nameInput && h('div.cs-field', h('label', 'Name', h('span.req', '*')), nameInput, h('div.help', 'How this connection is shown everywhere in the studio.')),
    [...groups.entries()].map(([g, items]) => [h('div.cs-form-section', g), h('div.cs-form-grid', items)]),
    advanced.size ? h('details.cs-advanced', h('summary', icon('chevron-right'), 'Advanced settings'),
      [...advanced.entries()].map(([g, items]) => [h('div.cs-form-section', g), h('div.cs-form-grid', items)])) : null);
  refresh();
  return {
    el,
    name: () => nameInput?.value.trim(),
    /** Fill fields programmatically (e.g. from an import); opens Advanced when a field there changes. */
    set(vals, { name: newName } = {}) {
      const changed = [];
      for (const f of spec.fields) {
        const inp = inputs[f.name];
        if (!inp || !(f.name in vals) || vals[f.name] == null) continue;
        if (f.type === 'checkbox') inp.checked = !!vals[f.name];
        else inp.value = String(vals[f.name]);
        changed.push(f.name);
        rows[f.name].classList.remove('cs-flash');
        void rows[f.name].offsetWidth;            // restart the highlight animation
        rows[f.name].classList.add('cs-flash');
      }
      if (newName && nameInput && !nameInput.value.trim()) nameInput.value = newName;
      if (changed.some((k) => spec.fields.find((f) => f.name === k)?.advanced)) el.querySelector('details.cs-advanced')?.setAttribute('open', '');
      refresh();
      onChange?.();
      return changed;
    },
    values() {
      const out = {};
      for (const f of spec.fields) {
        const inp = inputs[f.name];
        if (!inp) continue;
        if (f.type === 'checkbox') out[f.name] = inp.checked;
        else if (f.type === 'number') out[f.name] = inp.value === '' ? '' : Number(inp.value);
        else out[f.name] = inp.value;
      }
      return out;
    },
    missing() {
      const vals = this.values();
      return spec.fields.filter((f) => f.required && f.type !== 'file' && !rows[f.name]?.classList.contains('cs-hidden')
        && (vals[f.name] === '' || vals[f.name] == null) && !(f.secret && secretsSet.includes(f.name))).map((f) => f.label);
    },
  };
}

// ------------------------------------------------------------------ dropzone

export function dropzone({ accept = [], multiple = true, title = 'Drop files here', sub, onFiles } = {}) {
  const input = h('input', { type: 'file', multiple, accept: accept.join(','), hidden: true, onchange: () => { if (input.files.length) onFiles([...input.files]); input.value = ''; } });
  const el = h('div.cs-drop', { tabindex: 0, role: 'button', onclick: () => input.click(), onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } } },
    input, h('div.cs-drop-icon', icon('cloud-arrow-up-fill')), h('div.cs-drop-title', title),
    h('div.cs-drop-sub', sub || (accept.length ? `${accept.join(' · ')} — or click to browse` : 'or click to browse')));
  let depth = 0;
  el.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; el.classList.add('over'); });
  el.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) el.classList.remove('over'); });
  el.addEventListener('dragover', (e) => e.preventDefault());
  el.addEventListener('drop', (e) => {
    e.preventDefault(); depth = 0; el.classList.remove('over');
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length) onFiles(multiple ? files : files.slice(0, 1));
  });
  return el;
}

// ------------------------------------------------------------------ JSON

export function jsonCode(value) {
  const text = JSON.stringify(value, null, 2);
  const el = h('pre.cs-code');
  const re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) el.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (m[1]) {
      el.appendChild(h(`span.${m[2] ? 'k' : 's'}`, m[1]));
      if (m[2]) el.appendChild(document.createTextNode(m[2]));
    } else if (m[3]) el.appendChild(h('span.b', m[3]));
    else el.appendChild(h('span.n', m[4]));
    last = re.lastIndex;
  }
  el.appendChild(document.createTextNode(text.slice(last)));
  return el;
}

// ------------------------------------------------------------------ celebration

export function confetti(n = 70) {
  const colors = ['#0072ce', '#0bb99f', '#7c3aed', '#f59e0b', '#ec4899', '#06b6d4'];
  for (let i = 0; i < n; i++) {
    const p = h('span.cs-confetti', { style: {
      left: `${50 + (Math.random() - 0.5) * 30}vw`, top: '28vh', background: colors[i % colors.length],
      '--dx': `${(Math.random() - 0.5) * 80}vw`, '--r': `${Math.random() * 720 - 360}deg`, animationDelay: `${Math.random() * 0.25}s`,
    } });
    document.body.appendChild(p);
    setTimeout(() => p.remove(), 1900);
  }
}

// ------------------------------------------------------------------ misc

export function empty({ icon: ic = 'inbox', title, message, action }) {
  return h('div.cs-empty', h('div.cs-empty-art', icon(ic)), h('h3', title), message && h('p', message), action);
}

export function seg(options, value, onChange) {
  let cur = value;
  const el = h('div.cs-seg');
  const render = () => mount(el, options.map((o) => h('button', { type: 'button', class: o.value === cur && 'on', title: o.title,
    onclick: () => { cur = o.value; render(); onChange(o.value); } }, o.icon && icon(o.icon), o.label, o.count != null && h('span.cs-seg-count', String(o.count)))));
  render();
  el.set = (v) => { cur = v; render(); };
  return el;
}

export function kv(items) {
  return h('dl.cs-kvs', items.filter(Boolean).map(([k, v, title]) => h('div.cs-kv', { title: title || (typeof v === 'string' ? v : '') }, h('dt', k), h('dd', v ?? '—'))));
}

export function scheduleLabel(min) {
  if (!min) return 'Manual';
  if (min % 1440 === 0) return min === 1440 ? 'Daily' : `Every ${min / 1440} days`;
  if (min % 60 === 0) return min === 60 ? 'Hourly' : `Every ${min / 60} h`;
  return `Every ${min} min`;
}
