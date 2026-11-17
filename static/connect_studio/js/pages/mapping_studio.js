// Mapping studio: generating experience + three-pane review (tables · wire board · rule inspector).

import { h, s, icon, mount, debounce } from '../../../ui/js/dom.js';
import { href, navigate } from '../../../ui/js/router.js';
import { confirmDialog, formModal, modal, openMenu, toast, toastError } from '../../../ui/js/ui.js';
import { api } from '../api.js';
import {
  KIND, applyRule, compact, confetti, confidence, describe, empty, jobTracker, jsonCode, keyBadges, kindIcon,
  logo, madeBy, n, ring, statusBadge, themed, typeChip,
} from '../kit.js';

const STEP_OPS = ['lowercase', 'uppercase', 'trim', 'slug', 'first_word', 'rest_words', 'title_case'];

export default async function render(ctx) {
  const id = Number(ctx.params.id);
  const meta = await api.meta();
  const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
  let { mapping: m } = await api.mapping(id, { snapshots: 1 }, { signal: ctx.signal });
  ctx.setTitle(m.name);

  if (m.running_job && (m.status === 'generating' || !m.document?.tables)) return generating(ctx, m, specs);
  if (!m.document?.tables) return failed(ctx, m);

  // ---------------------------------------------------------------- state
  let doc = m.document;
  let tableSummary = m.table_summary || [];
  let summary = m.summary || {};
  const srcIdx = Object.fromEntries((m.source_schema?.tables || []).map((t) => [t.key, t]));
  const tgtIdx = Object.fromEntries((m.target_schema?.tables || []).map((t) => [t.key, t]));
  let filter = ctx.query.get('filter') || 'all';
  let railQ = '';
  let selTable = ctx.query.get('table') || doc.tables[0]?.target;
  let selCol = ctx.query.get('col') || null;
  let busyTable = null;

  const pairOf = (t) => doc.tables.find((p) => p.target === t);
  const fieldsOf = (t) => doc.fields.filter((f) => f.target_table === t);
  const fieldOf = (t, c) => doc.fields.find((f) => f.target_table === t && f.target === c);
  const tsum = (t) => tableSummary.find((x) => x.target === t) || {};

  const headEl = h('div.cs-card');
  const driftEl = h('div');
  const railEl = h('aside.cs-card.cs-rail');
  const boardEl = h('section.cs-card', { style: { minWidth: 0 } });
  const inspEl = h('aside.cs-card.cs-inspector');
  const studio = h('div.cs-studio', railEl, boardEl, inspEl);
  mount(ctx.el, headEl, driftEl, studio);

  function applyResponse(res) {
    if (res.document) doc = res.document;
    if (res.fields) for (const f of res.fields) {
      const i = doc.fields.findIndex((x) => x.id === f.id);
      if (i >= 0) doc.fields[i] = f; else doc.fields.push(f);
    }
    if (res.changed && !res.document) {
      const present = new Set((res.fields || []).map((f) => f.id));
      doc.fields = doc.fields.filter((f) => !res.changed.includes(f.id) || present.has(f.id));
    }
    if (res.unmapped_fields) doc.unmapped_fields = res.unmapped_fields;
    if (res.summary) summary = res.summary;
    if (res.table_summary) tableSummary = res.table_summary;
    m.version = res.version ?? m.version;
    window.dispatchEvent(new Event('cs:changed'));
  }

  // ---------------------------------------------------------------- header
  function renderHead() {
    const s0 = summary;
    mount(headEl, h('div.cs-mhead',
      h('div.cs-grow', { style: { minWidth: '280px' } },
        h('div.cs-crumbs', h('a', { href: href('/mappings') }, 'Mappings'), icon('chevron-right'), 'Studio'),
        h('div.cs-row.cs-wrap', { style: { gap: '10px' } },
          h('h1', { style: { fontFamily: 'var(--cs-display)', fontSize: '21px', fontWeight: 700, margin: 0, cursor: 'text' }, title: 'Rename', onclick: rename }, m.name),
          statusBadge(m.status)),
        h('div.cs-row.cs-wrap', { style: { marginTop: '8px', gap: '8px' } },
          h('a.cs-flowchip', { href: href(`/sources/${m.source.id}`) }, logo(specs[m.source.kind], { size: 'xs' }), h('span.cs-ellipsis', m.source.name)),
          h('span.cs-flow-arrow', h('span.dash'), icon('chevron-right')),
          h('a.cs-flowchip', { href: href(`/sources/${m.target.id}`) }, logo(specs[m.target.kind], { size: 'xs' }), h('span.cs-ellipsis', m.target.name)),
          doc.usage?.calls ? h('span.cs-badge.cs-badge-ai', { title: `${doc.usage.calls} Groq calls · ${doc.usage.seconds}s · ${doc.usage.model || ''}` }, icon('stars'), `${compact(doc.usage.total_tokens)} tokens`) : h('span.cs-badge.cs-badge-teal', icon('lightning-charge'), 'rule-based'))),
      h('div.cs-mstats',
        h('div.cs-mstat', ring(s0.coverage || 0, { size: 52, stroke: 6 }), h('div', h('div.cs-mstat-v', `${n(s0.mapped)}/${n(s0.columns)}`), h('div.cs-mstat-k', 'columns mapped'))),
        h('div.cs-mstat', ring(s0.approved_pct || 0, { size: 52, stroke: 6, color: '#16a34a' }), h('div', h('div.cs-mstat-v', n(s0.approved)), h('div.cs-mstat-k', 'approved'))),
        h('div.cs-mstat', h('div', h('div.cs-mstat-v', { style: { color: s0.review ? 'var(--cs-warn)' : undefined } }, n(s0.review)), h('div.cs-mstat-k', 'low confidence'))),
        h('div.cs-mstat', h('div', h('div.cs-mstat-v', n(s0.unmapped)), h('div.cs-mstat-k', 'unmapped')))),
      h('div.cs-head-actions',
        h('button.cs-btn.cs-btn-ok', { type: 'button', title: 'Approve every suggestion with ≥ 90% confidence', onclick: () => bulk({ action: 'approve', min_confidence: 0.9 }, 'Approved all ≥ 90% suggestions') },
          icon('check2-all'), 'Approve ≥ 90%'),
        h('button.cs-btn', { type: 'button', onclick: (e) => exportMenu(e.currentTarget) }, icon('filetype-json'), 'Export', icon('chevron-down')),
        h('a.cs-btn.cs-btn-brand', { href: href(`/syncs?new=1&mapping=${id}`) }, icon('arrow-repeat'), 'Sync'),
        h('button.cs-icon-btn', { type: 'button', title: 'More', onclick: (e) => openMenu(e.currentTarget, [
          { label: 'Regenerate everything with AI', icon: 'stars', onClick: regenerate },
          { label: 'Rename', icon: 'pencil', onClick: rename },
          { label: 'Keyboard shortcuts', icon: 'keyboard', onClick: shortcuts },
          { divider: true },
          { label: 'Delete mapping', icon: 'trash3', danger: true, onClick: remove },
        ]) }, icon('three-dots-vertical')))));
    const drift = m.drift || {};
    mount(driftEl, drift.source || drift.target ? h('div.cs-callout.cs-callout-warn', icon('shuffle'),
      h('div.cs-grow', h('b', 'Schema drift: '), [drift.source && `source changed (${drift.source})`, drift.target && `target changed (${drift.target})`].filter(Boolean).join(' · '),
        ' since this mapping was generated.'),
      h('button.cs-btn.cs-btn-sm', { type: 'button', onclick: regenerate }, icon('stars'), 'Re-map with the new snapshots')) : null);
  }

  // ---------------------------------------------------------------- rail
  function renderRail() {
    const counts = { all: doc.tables.length, review: 0, unmatched: 0, done: 0 };
    for (const p of doc.tables) {
      const t = tsum(p.target);
      if (!p.source) counts.unmatched++;
      else if (t.mapped && t.approved === t.mapped && !t.unmapped) counts.done++;
      else counts.review++;
    }
    const match = (p) => {
      const t = tsum(p.target);
      if (railQ && !`${p.target} ${p.source || ''}`.toLowerCase().includes(railQ)) return false;
      if (filter === 'unmatched') return !p.source;
      if (filter === 'done') return p.source && t.mapped && t.approved === t.mapped && !t.unmapped;
      if (filter === 'review') return p.source && !(t.mapped && t.approved === t.mapped && !t.unmapped);
      return true;
    };
    const list = doc.tables.filter(match);
    const unused = doc.unused_sources || [];
    mount(railEl,
      h('div.cs-rail-head',
        h('div.cs-seg', { style: { width: '100%' } }, [['all', 'All'], ['review', 'To do'], ['unmatched', 'No source'], ['done', 'Done']].map(([k, label]) =>
          h('button', { type: 'button', class: filter === k && 'on', style: { flex: 1, justifyContent: 'center', padding: '5px 4px' }, onclick: () => { filter = k; ctx.setQuery({ filter: k === 'all' ? null : k }); renderRail(); } },
            label, h('span.cs-seg-count', String(counts[k]))))),
        h('div.cs-input-wrap', icon('search'), h('input.cs-input.cs-input-sm', { placeholder: 'Filter target tables…', value: railQ,
          oninput: (e) => { railQ = e.target.value.trim().toLowerCase(); renderRail(); setTimeout(() => railEl.querySelector('input')?.focus(), 0); } }))),
      h('div.cs-rail-list', list.length ? list.map((p) => {
        const t = tsum(p.target);
        return h('button.cs-ritem', { type: 'button', class: p.target === selTable && 'on', onclick: () => selectTable(p.target) },
          ring(t.columns ? t.mapped / t.columns : 0, { size: 32, stroke: 4, label: false, color: t.mapped && t.approved === t.mapped ? '#16a34a' : undefined }),
          h('div.cs-grow', { style: { minWidth: 0 } },
            h('div.cs-ritem-name.cs-ellipsis', p.target),
            p.source ? h('div.cs-ritem-sub.cs-ellipsis', icon('arrow-return-right'), p.source, p.made_by === 'ai' && icon('stars', 'cs-faint'))
              : h('div.cs-ritem-sub.none', icon('exclamation-triangle'), 'no source table')),
          busyTable === p.target ? h('span.spinner-border.spinner-border-sm.text-primary')
            : t.review ? h('span.cs-badge.cs-badge-warn', { title: 'Low-confidence suggestions' }, String(t.review))
              : t.mapped && t.approved === t.mapped ? icon('patch-check-fill', 'text-success') : null);
      }) : h('div.cs-muted', { style: { padding: '14px', fontSize: '12.5px' } }, 'No tables match.')),
      unused.length ? h('div.cs-rail-foot', h('a', { href: '#', onclick: (e) => { e.preventDefault(); showUnused(); } }, icon('box-seam'), ` ${unused.length} source table(s) not used`)) : null);
    railEl.querySelector('.cs-ritem.on')?.scrollIntoView({ block: 'nearest' });
  }

  function selectTable(t, col = null) {
    selTable = t;
    selCol = col;
    ctx.setQuery({ table: t, col });
    renderRail();
    renderBoard();
    renderInspector();
  }

  // ---------------------------------------------------------------- board
  let wireRedraw = () => {};
  function renderBoard() {
    const p = pairOf(selTable);
    if (!p) { mount(boardEl, empty({ icon: 'table', title: 'No target tables', message: 'This mapping has no target tables.' })); return; }
    const tgt = tgtIdx[p.target];
    const src = p.source ? srcIdx[p.source] : null;
    const fields = fieldsOf(p.target);
    const byTarget = Object.fromEntries(fields.map((f) => [f.target, f]));
    const tcols = tgt?.columns || [];

    // source columns: used (in target order) first, then the rest
    const used = [];
    for (const c of tcols) { const f = byTarget[c.name]; if (f?.source && f.status !== 'rejected' && !used.includes(f.source)) used.push(f.source); }
    const scols = src ? [...used.map((nm) => src.columns.find((c) => c.name === nm)).filter(Boolean), ...src.columns.filter((c) => !used.includes(c.name))] : [];

    const srcEls = {};
    const tgtEls = {};
    const wiresSvg = s('svg.cs-wires');
    const scolEl = (c) => {
      const isUsed = used.includes(c.name);
      const el = h('div.cs-scol', { class: isUsed ? 'used' : 'unused', draggable: 'true', title: `${c.name} · ${c.type}${c.profile ? ` · ${c.profile.null_pct}% null · ${c.profile.distinct ?? '?'} distinct` : ''}\nDrag onto a target column to map it`,
        ondragstart: (e) => { e.dataTransfer.setData('text/cs-column', c.name); e.dataTransfer.effectAllowed = 'link'; el.classList.add('dragging'); },
        ondragend: () => el.classList.remove('dragging'),
        onmouseenter: () => hl({ source: c.name }), onmouseleave: () => hl(null) },
      icon('grip-vertical', 'grip'), h('span.cs-ellipsis.cs-grow', c.name), ...keyBadges(c).slice(0, 1), typeChip(c));
      srcEls[c.name] = el;
      return el;
    };
    const tcolEl = (c) => {
      const f = byTarget[c.name];
      const unmapped = !f;
      const st = f?.status;
      const el = h('div.cs-tcol', {
        class: [unmapped && 'unmapped', st === 'approved' && 'approved', st === 'rejected' && 'rejected', selCol === c.name && 'on'],
        onclick: () => { selCol = c.name; ctx.setQuery({ col: c.name }); boardEl.querySelectorAll('.cs-tcol.on').forEach((x) => x.classList.remove('on')); el.classList.add('on'); renderInspector(); },
        onmouseenter: () => hl({ target: c.name }), onmouseleave: () => hl(null),
        ondragover: (e) => { if (src && e.dataTransfer.types.includes('text/cs-column')) { e.preventDefault(); el.classList.add('drop'); } },
        ondragleave: () => el.classList.remove('drop'),
        ondrop: (e) => { e.preventDefault(); el.classList.remove('drop'); const col = e.dataTransfer.getData('text/cs-column'); if (col) dropMap(p.target, c.name, col); },
      },
      h('div.cs-tcol-name', h('span.cs-ellipsis', c.name), ...keyBadges(c).slice(0, 1), typeChip(c)),
      f ? h('div.cs-expr', kindIcon(f.kind), h('span.txt', { title: describe(f) }, describe(f)), madeBy(f.made_by))
        : h('div.cs-expr.empty', icon('arrow-left-short'), src ? 'drop a source column · or click' : 'pick a source table first'),
      h('div.cs-tcol-acts',
        f && st !== 'rejected' && confidence(f.confidence),
        f && h('button.cs-icon-btn.cs-icon-btn-sm.ok', { type: 'button', class: st === 'approved' && 'on', title: st === 'approved' ? 'Approved (click to undo)' : 'Approve (A)',
          onclick: (e) => { e.stopPropagation(); setStatus(f, st === 'approved' ? 'suggested' : 'approved'); } }, icon('check-lg')),
        f && h('button.cs-icon-btn.cs-icon-btn-sm.bad', { type: 'button', class: st === 'rejected' && 'off', title: st === 'rejected' ? 'Rejected (click to undo)' : 'Reject (X)',
          onclick: (e) => { e.stopPropagation(); setStatus(f, st === 'rejected' ? 'suggested' : 'rejected'); } }, icon('x-lg'))));
      tgtEls[c.name] = el;
      return el;
    };

    const t = tsum(p.target);
    const board = h('div.cs-board', wiresSvg,
      h('div', h('div.cs-board-col-head', icon('box-arrow-right'), src ? `${p.source} · ${src.columns.length}` : 'source'),
        h('div.cs-scol-list', src ? [...scols.slice(0, used.length).map(scolEl), scols.length > used.length && h('div.cs-scol-sep', 'not used'), ...scols.slice(used.length).map(scolEl)]
          : h('div.cs-muted', { style: { fontSize: '12.5px', padding: '8px 4px' } }, 'No source table — pick one above.'))),
      h('div'),
      h('div', h('div.cs-board-col-head', icon('box-arrow-in-right'), `${p.target} · ${tcols.length}`), h('div.cs-tcol-list', tcols.map(tcolEl))));

    const busy = busyTable === p.target;
    mount(boardEl,
      h('div.cs-board-head',
        h('div.cs-board-title',
          h('button.cs-srcpick', { type: 'button', class: !p.source && 'none', title: 'Change the source table', disabled: busy, onclick: (e) => pickSource(e.currentTarget, p) },
            icon('box-arrow-right'), p.source || 'choose a source table', icon('chevron-down')),
          icon('arrow-right', 'cs-faint'),
          h('span', p.target),
          tgt?.model && h('span.cs-badge.cs-badge-outline', { title: 'Target model' }, tgt.model.split('\\').pop())),
        p.source && h('span.cs-badge', { class: p.made_by === 'human' ? 'cs-badge-info' : p.made_by === 'ai' ? 'cs-badge-ai' : 'cs-badge-teal' },
          p.made_by === 'human' ? icon('person-fill') : p.made_by === 'ai' ? icon('stars') : icon('lightning-charge-fill'),
          p.made_by === 'human' ? 'your choice' : `${Math.round((p.confidence || 0) * 100)}% match`),
        p.where && h('span.cs-badge.cs-badge-warn', { title: 'Only these source rows feed this table' }, icon('funnel'), `${p.where.column} in ${p.where.in.slice(0, 3).join(', ')}${p.where.in.length > 3 ? '…' : ''}`),
        h('span.cs-grow'),
        busy ? h('span.cs-badge.cs-badge-ai', h('span.spinner-border.spinner-border-sm'), ' Groq is re-mapping…') : [
          p.source && h('button.cs-btn.cs-btn-sm', { type: 'button', onclick: () => remapTable(p.target, p.source) }, icon('stars'), 'Re-map'),
          fields.length > 0 && h('button.cs-btn.cs-btn-sm.cs-btn-ok', { type: 'button', onclick: () => bulk({ action: 'approve', table: p.target }, `${p.target}: all approved`) }, icon('check2-all'), `Approve table`),
        ]),
      p.reason && h('div.cs-reason', icon('stars'), h('div', p.reason)),
      ...(doc.notes || []).filter((x) => x.startsWith(`${p.target}: `)).map((note) => h('div.cs-callout.cs-callout-warn', { style: { margin: '12px 16px 0' } },
        icon('exclamation-triangle'), h('div.cs-grow', note.slice(p.target.length + 2).replace(/\(Groq call failed: Error code: 429.*$/, '(Groq rate limit reached)')),
        p.source && !busy && h('button.cs-btn.cs-btn-sm', { type: 'button', onclick: () => remapTable(p.target, p.source) }, icon('stars'), 'Re-map'))),
      h('div', { style: { padding: '10px 16px 0', display: 'flex', gap: '14px', fontSize: '12px', color: 'var(--cs-muted)', flexWrap: 'wrap' } },
        h('span', `${t.mapped || 0} of ${t.columns || tcols.length} target columns mapped`), h('span', `${t.approved || 0} approved`),
        t.review ? h('span', { style: { color: 'var(--cs-warn)' } }, `${t.review} low confidence`) : null,
        h('span.cs-grow'), h('span', icon('info-circle'), ' drag source columns onto target columns')),
      board);

    // wires
    const draw = () => {
      if (!board.isConnected) return;
      const br = board.getBoundingClientRect();
      wiresSvg.setAttribute('width', br.width);
      wiresSvg.setAttribute('height', br.height);
      mount(wiresSvg);
      for (const f of fields) {
        if (!f.source || f.status === 'rejected' || f.kind === 'constant') continue;
        const a = srcEls[f.source]; const b = tgtEls[f.target];
        if (!a || !b) continue;
        const ra = a.getBoundingClientRect(); const rb = b.getBoundingClientRect();
        const x1 = ra.right - br.left; const y1 = ra.top + ra.height / 2 - br.top;
        const x2 = rb.left - br.left; const y2 = rb.top + rb.height / 2 - br.top;
        const dx = Math.max(30, (x2 - x1) / 2);
        const col = f.confidence < 0.6 && f.status !== 'approved' ? 'var(--cs-wire-low)' : (KIND[f.kind] || KIND.copied).wire;
        const path = s('path', { d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`, stroke: col,
          class: ['cs-wire', f.status !== 'approved' && 'suggested'], dataset: { s: f.source, t: f.target } });
        wiresSvg.append(path, s('circle.cs-wire-dot', { cx: x1, cy: y1, r: 3.5, fill: col, dataset: { s: f.source, t: f.target } }),
          s('circle.cs-wire-dot', { cx: x2, cy: y2, r: 3.5, fill: col, dataset: { s: f.source, t: f.target } }));
      }
    };
    wireRedraw = draw;
    requestAnimationFrame(draw);
    const ro = new ResizeObserver(() => draw());
    ro.observe(board);
    ctx.cleanup(() => ro.disconnect());

    function hl(sel) {
      const wires = wiresSvg.querySelectorAll('.cs-wire, .cs-wire-dot');
      Object.values(srcEls).forEach((e) => e.classList.remove('hl'));
      Object.values(tgtEls).forEach((e) => e.classList.remove('hl'));
      if (!sel) { wires.forEach((w) => w.classList.remove('hl', 'dim')); return; }
      wires.forEach((w) => {
        const on = (sel.source && w.dataset.s === sel.source) || (sel.target && w.dataset.t === sel.target);
        w.classList.toggle('hl', !!on);
        w.classList.toggle('dim', !on);
        if (on && w.tagName === 'path') { srcEls[w.dataset.s]?.classList.add('hl'); tgtEls[w.dataset.t]?.classList.add('hl'); }
      });
    }
  }

  function pickSource(anchor, p) {
    const cands = (p.candidates || []).map((c) => c.source);
    const all = Object.keys(srcIdx);
    const rest = all.filter((k) => !cands.includes(k));
    openMenu(anchor, [
      { header: 'Best candidates' },
      ...(p.candidates || []).slice(0, 6).map((c) => ({ label: `${c.source} · ${Math.round(c.score * 100)}%`, icon: c.source === p.source ? 'check2' : 'table', onClick: () => remapTable(p.target, c.source) })),
      { divider: true },
      { header: `All source tables (${all.length})` },
      ...rest.slice(0, 60).map((k) => ({ label: k, icon: k === p.source ? 'check2' : 'table', onClick: () => remapTable(p.target, k) })),
      rest.length > 60 && { label: `Search ${rest.length} tables…`, icon: 'search', onClick: () => searchSource(p) },
      { divider: true },
      { label: 'No source (leave unmapped)', icon: 'slash-circle', danger: true, onClick: () => remapTable(p.target, null) },
    ], 'start');
  }

  function searchSource(p) {
    const all = Object.keys(srcIdx);
    const list = h('div.cs-picklist');
    const draw = (q) => mount(list, all.filter((k) => !q || k.toLowerCase().includes(q)).slice(0, 200).map((k) => h('div.cs-pick', { onclick: () => { dlg.close(); remapTable(p.target, k); } },
      icon('table', 'cs-muted'), h('span.cs-mono.cs-grow', k), h('span.cs-muted', { style: { fontSize: '11.5px' } }, `${srcIdx[k].columns.length} cols`))));
    const dlg = modal({ title: `Source for ${p.target}`, icon: 'search', body: themed(h('div.cs-col', h('input.cs-input', { placeholder: 'Search…', autofocus: true, oninput: (e) => draw(e.target.value.trim().toLowerCase()) }), list)) });
    draw('');
  }

  async function remapTable(target, source) {
    try {
      busyTable = target;
      renderRail(); renderBoard();
      const { job } = await api.setTable(id, { target, source, ai: meta.llm.available });
      const t = jobTracker(job.id, { showLog: false });
      const final = await t.done;
      busyTable = null;
      if (final.status === 'succeeded') {
        const res = await api.mapping(id, { snapshots: 0 });
        doc = res.mapping.document; tableSummary = res.mapping.table_summary; summary = res.mapping.summary; m.version = res.mapping.version;
        toast(final.message || 'Re-mapped', 'success', { title: target });
      } else toast(final.error || 'Re-map failed', 'error');
      renderHead(); renderRail(); renderBoard(); renderInspector();
    } catch (e) { busyTable = null; renderRail(); renderBoard(); toastError(e, 'Re-map failed'); }
  }

  // ---------------------------------------------------------------- actions
  async function setStatus(f, status) {
    try {
      applyResponse(await api.fields(id, { updates: [{ id: f.id, status }] }));
      renderHead(); renderRail(); renderBoard(); renderInspector();
    } catch (e) { toastError(e, 'Could not update'); }
  }

  async function dropMap(target, col, sourceCol) {
    const existing = fieldOf(target, col);
    try {
      const res = existing
        ? await api.fields(id, { updates: [{ id: existing.id, source: sourceCol, kind: 'copied', rule: {}, transform: '', status: 'approved', why: 'mapped by drag & drop' }] })
        : await api.fields(id, { create: [{ target_table: target, target: col, source: sourceCol, kind: 'copied', why: 'mapped by drag & drop' }] });
      applyResponse(res);
      selCol = col;
      renderHead(); renderRail(); renderBoard(); renderInspector();
      toast(`${sourceCol} → ${col}`, 'success', { title: 'Mapped', timeout: 1800 });
    } catch (e) { toastError(e, 'Could not map'); }
  }

  async function bulk(body, msg) {
    try {
      const res = await api.bulk(id, body);
      applyResponse(res);
      renderHead(); renderRail(); renderBoard(); renderInspector();
      toast(`${msg} (${res.changed} changed)`, 'success');
      if (summary.approved && summary.approved === summary.mapped && !summary.unmapped) confetti();
    } catch (e) { toastError(e, 'Bulk update failed'); }
  }

  async function regenerate() {
    if (!(await confirmDialog({ title: 'Regenerate with AI?', message: 'Groq re-maps every table from the latest snapshots. Your review decisions on this version are replaced.', confirmLabel: 'Regenerate' }))) return;
    try { await api.generate(id, {}); navigate(`/mappings/${id}`, { replace: true }); } catch (e) { toastError(e, 'Could not start'); }
  }

  function rename() {
    formModal({ title: 'Rename mapping', icon: 'pencil', fields: [{ name: 'name', label: 'Name', value: m.name, required: true }],
      onSubmit: async (v, close) => { await api.updateMapping(id, { name: v.name }); m.name = v.name; renderHead(); close(); } });
  }

  async function remove() {
    if (!(await confirmDialog({ title: `Delete “${m.name}”?`, message: 'Syncs that use this mapping are deleted too.', confirmLabel: 'Delete', danger: true }))) return;
    await api.deleteMapping(id);
    window.dispatchEvent(new Event('cs:changed'));
    navigate('/mappings');
  }

  function showUnused() {
    modal({ title: 'Source tables not used', subtitle: 'No target table chose these', icon: 'box-seam', size: 'lg',
      body: themed(h('div.cs-list', (doc.unused_sources || []).map((u) => h('div.cs-li', icon('table', 'cs-muted'), h('div.cs-grow', h('div.cs-li-title.cs-mono', u.source_table), h('div.cs-li-sub', u.reason)),
        u.rows != null && h('span.cs-badge', `${n(u.rows)} rows`))))), actions: [{ label: 'Close' }] });
  }

  function exportMenu(anchor) {
    const afyaDefault = m.source.kind === 'afya_extraction';
    openMenu(anchor, [
      { header: 'mappings.json' },
      { label: 'Preview & validate…', icon: 'eye', onClick: () => previewExport(afyaDefault ? 'afya' : 'generic') },
      { label: 'Download · generic format', icon: 'download', href: api.exportUrl(id, { flavor: 'generic' }), external: false },
      { label: 'Download · Afya V3 loader format', icon: 'hospital', href: api.exportUrl(id, { flavor: 'afya' }), external: false },
      { label: 'Download · approved fields only', icon: 'patch-check', href: api.exportUrl(id, { flavor: afyaDefault ? 'afya' : 'generic', approved_only: 1 }), external: false },
    ]);
    setTimeout(() => document.querySelectorAll('.af-menu a.af-menu-item').forEach((a) => { a.setAttribute('download', ''); a.removeAttribute('data-link'); }), 0);
  }

  async function previewExport(flavor) {
    const box = h('div', h('div', { style: { textAlign: 'center', padding: '30px' } }, h('span.spinner-border.text-primary')));
    let current = flavor;
    let data = null;
    const load = async () => {
      data = await api.exportMapping(id, { flavor: current });
      mount(box, h('div.cs-col',
        h('div.cs-row.cs-wrap',
          h('div.cs-seg', ['generic', 'afya'].map((f) => h('button', { type: 'button', class: f === current && 'on', onclick: () => { current = f; load(); } }, f === 'afya' ? 'Afya V3 loader' : 'Generic'))),
          h('span.cs-grow'),
          data.problems.length ? h('span.cs-badge.cs-badge-warn', icon('exclamation-triangle'), `${data.problems.length} problem(s)`) : h('span.cs-badge.cs-badge-ok', icon('check2-circle'), 'valid'),
          h('span.cs-badge.cs-badge-outline', `${data.document.tables.length} tables · ${data.document.fields.length} fields · ${data.document.special_actions.length} special`)),
        data.problems.length ? h('div.cs-callout.cs-callout-warn', icon('exclamation-triangle'), h('ul', { style: { margin: 0, paddingLeft: '18px' } }, data.problems.map((p) => h('li', p)))) : null,
        jsonCode(data.document)));
    };
    modal({ title: 'mappings.json', subtitle: m.name, icon: 'filetype-json', size: 'xl', body: themed(box),
      actions: [
        { label: 'Copy', icon: 'clipboard', onClick: async () => { await navigator.clipboard.writeText(JSON.stringify(data.document, null, 4)); toast('Copied', 'success'); } },
        { label: 'Download', icon: 'download', variant: 'primary', onClick: (close) => { window.location.href = api.exportUrl(id, { flavor: current }); close(); } },
      ] });
    load().catch((e) => mount(box, h('div.cs-callout.cs-callout-warn', icon('x-octagon'), e.message)));
  }

  function shortcuts() {
    modal({ title: 'Keyboard shortcuts', icon: 'keyboard', size: 'sm', body: themed(h('div.cs-help-grid',
      [['j / ↓', 'next column'], ['k / ↑', 'previous column'], ['] / [', 'next / previous table'], ['a', 'approve column'], ['x', 'reject column'],
        ['u', 'undo (back to suggested)'], ['⇧A', 'approve the whole table'], ['esc', 'close the inspector'], ['⌘K', 'command palette']]
        .flatMap(([k, v]) => [h('span.cs-kbd', k), h('span', v)]))), actions: [{ label: 'Close' }] });
  }

  // ---------------------------------------------------------------- inspector
  function renderInspector() {
    const p = pairOf(selTable);
    const tgt = tgtIdx[selTable];
    const col = selCol && tgt?.columns.find((c) => c.name === selCol);
    if (!p || !col) {
      studio.classList.remove('no-inspector');
      const t = tsum(selTable);
      mount(inspEl, h('div.cs-insp-body',
        h('div.cs-empty', { style: { padding: '30px 10px' } }, h('div.cs-empty-art', icon('cursor')), h('h3', 'Select a column'),
          h('p', 'Click a target column to see why it is mapped that way, change its rule and preview the result on real values.')),
        p && h('div.cs-kvs', h('div.cs-kv', h('dt', 'Mapped'), h('dd', `${t.mapped || 0}/${t.columns || 0}`)), h('div.cs-kv', h('dt', 'Approved'), h('dd', String(t.approved || 0))),
          h('div.cs-kv', h('dt', 'Unmapped'), h('dd', String(t.unmapped || 0)))),
        h('div.cs-callout.cs-callout-info', icon('keyboard'), h('div', 'Review fast: ', h('span.cs-kbd', 'j'), ' ', h('span.cs-kbd', 'k'), ' move · ', h('span.cs-kbd', 'a'), ' approve · ', h('span.cs-kbd', 'x'), ' reject'))));
      return;
    }
    const src = p.source ? srcIdx[p.source] : null;
    const saved = fieldOf(selTable, col.name);
    const unmappedReason = (doc.unmapped_fields || []).find((u) => u.target_table === selTable && u.target === col.name)?.reason;
    const draft = saved ? JSON.parse(JSON.stringify(saved)) : { source: src?.columns.find((c) => c.name === col.name)?.name || src?.columns[0]?.name || null, kind: 'copied', rule: {}, transform: '' };
    draft.rule = draft.rule || {};

    const editor = h('div.cs-col');
    const previewEl = h('div.cs-preview');
    const sourceProfile = h('div');
    const srcSelect = h('select.cs-input', { disabled: !src, onchange: () => { draft.source = srcSelect.value || null; drawProfile(); drawPreview(); } },
      h('option', { value: '' }, '— none —'),
      (src?.columns || []).map((c) => h('option', { value: c.name, selected: c.name === draft.source }, `${c.name}  ·  ${(c.type || '').toLowerCase()}`)));

    const kinds = h('div.cs-kinds');
    const drawKinds = () => mount(kinds, Object.entries(KIND).map(([k, v]) => h('button.cs-kindbtn', { type: 'button', class: draft.kind === k && 'on', title: v.help,
      onclick: () => {
        draft.kind = k;
        if (k === 'one column divided' && draft.rule.part == null) draft.rule = { part: 1, delimiter: ' ' };
        else if (k === 'codes converted' && !draft.rule.map) draft.rule = { map: {} };
        else if (k === 'worked out' && !draft.rule.steps) draft.rule = { steps: [{ op: 'lowercase' }] };
        else if (k === 'looked up in another table' && !draft.rule.table) {
          const ref = src?.columns.find((c) => c.name === draft.source)?.references;
          const guessTable = () => {
            // department / department_id / dept_no → departments
            const base = String(draft.source || '').toLowerCase().replace(/(_id|_no|_code|id)$/, '');
            const single = (x) => x.toLowerCase().split('.').pop().replace(/ies$/, 'y').replace(/s$/, '');
            return Object.keys(srcIdx).find((t) => t !== p.source && single(t) === single(base))
              || (m.source_schema?.relationships || []).find((x) => x.from === p.source)?.to
              || Object.keys(srcIdx).find((t) => t !== p.source);
          };
          const table = ref?.table || guessTable();
          const lt = srcIdx[table];
          const nameCol = lt?.columns.find((c) => /(^|_)(name|title|label)$/.test(c.name)) || lt?.columns.find((c) => !c.pk) || lt?.columns[0];
          draft.rule = lt ? { table, key: ref?.column || lt.primary_key?.[0] || lt.columns[0].name, value: nameCol?.name } : {};
        } else if (k === 'constant' && !('value' in draft.rule)) draft.rule = { value: '' };
        else if (k === 'copied') draft.rule = {};
        drawKinds(); drawEditor(); drawPreview();
      } }, icon(v.icon), v.label)));

    function drawEditor() {
      const r = draft.rule;
      const k = draft.kind;
      let body = null;
      if (k === 'copied') {
        body = h('div.cs-field', h('label', 'Then (optional)'),
          h('select.cs-input', { onchange: (e) => { draft.transform = e.target.value; drawPreview(); } },
            h('option', { value: '' }, 'nothing — copy as is'), STEP_OPS.map((op) => h('option', { value: op, selected: draft.transform === op }, op))));
      } else if (k === 'worked out') {
        const steps = r.steps || [];
        body = h('div.cs-field', h('label', 'Steps, in order'),
          h('div.cs-chips', steps.map((st, i) => h('span.cs-chip', { style: { borderColor: 'var(--cs-primary)', color: 'var(--cs-primary)' } }, `${i + 1}. ${st.op}`,
            h('button.cs-icon-btn.cs-icon-btn-sm', { type: 'button', style: { width: '18px', height: '18px' }, onclick: () => { steps.splice(i, 1); r.steps = steps; drawEditor(); drawPreview(); } }, icon('x'))))),
          h('div.cs-chips', { style: { marginTop: '8px' } }, STEP_OPS.map((op) => h('button.cs-chip', { type: 'button', onclick: () => { r.steps = [...steps, { op }]; drawEditor(); drawPreview(); } }, icon('plus'), op))));
      } else if (k === 'codes converted') {
        const entries = Object.entries(r.map || {});
        const allowed = col.allowed_values || [];
        const listId = `cs-allowed-${Math.random().toString(36).slice(2, 7)}`;
        const scol = src?.columns.find((c) => c.name === draft.source);
        const srcValues = (scol?.profile?.top || []).map((t) => t.v);
        const rows = h('div.cs-maprows', entries.map(([from, to], i) => h('div.cs-maprow',
          h('input.cs-input.cs-input-sm.cs-mono', { value: from, placeholder: 'source value', onchange: (e) => { const m2 = {}; entries.forEach(([a, b], j) => { m2[j === i ? e.target.value : a] = b; }); r.map = m2; drawEditor(); drawPreview(); } }),
          icon('arrow-right', 'arrow'),
          h('input.cs-input.cs-input-sm.cs-mono', { value: to ?? '', list: listId, placeholder: 'target value', oninput: (e) => { r.map[from] = e.target.value; drawPreview(); } }),
          h('button.cs-icon-btn.cs-icon-btn-sm.bad', { type: 'button', onclick: () => { delete r.map[from]; drawEditor(); drawPreview(); } }, icon('x-lg')))));
        body = h('div.cs-field', h('label', 'Value map', h('span.cs-grow'),
          srcValues.length ? h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', title: 'Add every value seen in the source sample', onclick: () => {
            r.map = { ...(r.map || {}) }; srcValues.forEach((v) => { if (!(v in r.map)) r.map[v] = ''; }); drawEditor(); drawPreview();
          } }, icon('magic'), 'Add source values') : null),
        h('datalist', { id: listId }, allowed.map((v) => h('option', { value: v }))),
        rows,
        h('button.cs-btn.cs-btn-sm', { type: 'button', style: { marginTop: '8px' }, onclick: () => { r.map = { ...(r.map || {}), [`value${entries.length + 1}`]: '' }; drawEditor(); } }, icon('plus-lg'), 'Add pair'),
        h('div.cs-form-grid', { style: { marginTop: '10px' } },
          h('div.cs-field.full', h('label', 'When no pair matches'),
            h('div.cs-row', h('input.cs-input.cs-input-sm.cs-mono', { value: 'default' in r ? (r.default ?? '') : '', list: listId, placeholder: 'keep the source value',
              oninput: (e) => { if (e.target.value === '') delete r.default; else r.default = e.target.value; drawPreview(); } })))),
        allowed.length ? h('div.help', 'Target accepts: ', allowed.slice(0, 12).join(' · ')) : null);
      } else if (k === 'looked up in another table') {
        const tables = Object.keys(srcIdx);
        const related = (m.source_schema?.relationships || []).filter((x) => x.from === p.source).map((x) => x.to);
        const lt = srcIdx[r.table];
        body = h('div.cs-col',
          h('div.cs-field', h('label', 'Look up in table'),
            h('select.cs-input', { onchange: (e) => { r.table = e.target.value; r.key = srcIdx[r.table]?.primary_key?.[0] || srcIdx[r.table]?.columns[0]?.name; r.value = srcIdx[r.table]?.columns[1]?.name || r.key; drawEditor(); drawPreview(); } },
              h('option', { value: '' }, '— choose —'),
              related.length ? h('optgroup', { label: 'Related (FK)' }, related.map((t) => h('option', { value: t, selected: t === r.table }, t))) : null,
              h('optgroup', { label: 'All source tables' }, tables.filter((t) => !related.includes(t)).map((t) => h('option', { value: t, selected: t === r.table }, t))))),
          lt && h('div.cs-form-grid',
            h('div.cs-field', h('label', 'Match on'), h('select.cs-input', { onchange: (e) => { r.key = e.target.value; drawPreview(); } },
              lt.columns.map((c) => h('option', { value: c.name, selected: c.name === r.key }, c.name)))),
            h('div.cs-field', h('label', 'Take'), h('select.cs-input', { onchange: (e) => { r.value = e.target.value; drawPreview(); } },
              lt.columns.map((c) => h('option', { value: c.name, selected: c.name === r.value }, c.name))))),
          lt && h('div.help', `${r.table}.${r.value || '?'} of the row whose ${r.table}.${r.key || '?'} equals this row’s ${draft.source || '?'}.`));
      } else if (k === 'one column divided') {
        body = h('div.cs-form-grid',
          h('div.cs-field', h('label', 'Split on'), h('input.cs-input.cs-mono', { value: !r.delimiter || r.delimiter === ' ' ? '' : r.delimiter, placeholder: 'space (default)', oninput: (e) => { r.delimiter = e.target.value || ' '; drawPreview(); } })),
          h('div.cs-field', h('label', 'Take part'), h('select.cs-input', { onchange: (e) => { r.part = e.target.value === 'rest' ? 'rest' : Number(e.target.value); drawPreview(); } },
            [1, 2, 3, 4, 'rest'].map((v) => h('option', { value: v, selected: String(r.part) === String(v) }, v === 'rest' ? 'everything after the first' : `part ${v}`)))));
      } else if (k === 'constant') {
        body = h('div.cs-field', h('label', 'Value'), h('input.cs-input.cs-mono', { value: r.value ?? '', oninput: (e) => {
          const v = e.target.value;
          r.value = v === 'true' ? true : v === 'false' ? false : v === 'null' ? null : v !== '' && !Number.isNaN(Number(v)) ? Number(v) : v;
          drawPreview();
        } }), h('div.help', 'Numbers, true / false and null are typed; anything else is text.'));
      }
      mount(editor, body);
    }

    function drawProfile() {
      const c = src?.columns.find((x) => x.name === draft.source);
      const pr = c?.profile;
      mount(sourceProfile, c ? h('div.cs-col.cs-gap-sm',
        h('div.cs-row.cs-gap-sm', typeChip(c), ...keyBadges(c), c.references && h('span.cs-muted.cs-mono', { style: { fontSize: '11px' } }, `→ ${c.references.table}.${c.references.column}`)),
        pr && h('div.cs-profile-mini', h('div', h('b', `${pr.null_pct.toFixed(0)}%`), h('span', 'null')), h('div', h('b', pr.distinct != null ? n(pr.distinct) : '—'), h('span', 'distinct')),
          h('div', h('b', pr.rows != null ? n(pr.rows) : '—'), h('span', 'rows sampled')))) : null);
    }

    let previewSeq = 0;
    const serverPreview = debounce(async (seq) => {
      try {
        const res = await api.preview(id, { source_table: p.source, source: draft.source, kind: draft.kind, rule: draft.rule, transform: draft.transform });
        if (seq !== previewSeq) return;
        renderPreviewRows(res.preview.map((x) => [x.in, x.out]), res.note);
      } catch (e) { if (seq === previewSeq) renderPreviewRows([], e.message); }
    }, 350);
    function renderPreviewRows(rows, note) {
      mount(previewEl, h('div.cs-preview-head', icon('play-circle'), 'Preview on sampled values', h('span.cs-grow'), note && h('span', { title: note }, icon('info-circle'))),
        rows.length ? rows.map(([a, b]) => h('div.cs-preview-row', h('span.in', { title: String(a) }, a == null ? 'null' : String(a)), icon('arrow-right'),
          h('span.out', { class: b == null && 'null', title: String(b) }, b == null ? 'null' : typeof b === 'string' ? b : JSON.stringify(b))))
          : h('div.cs-preview-row', h('span.in', note || 'No sample values — extract the source with profiling on.')));
    }
    function drawPreview() {
      const scol = src?.columns.find((c) => c.name === draft.source);
      const values = [...new Set([...(scol?.profile?.top || []).map((t) => t.v), ...(scol?.profile?.samples || [])])].slice(0, 10);
      if (draft.kind === 'looked up in another table') { previewSeq++; mount(previewEl, h('div.cs-preview-head', h('span.spinner-border.spinner-border-sm'), 'Looking up…')); serverPreview(previewSeq); return; }
      if (draft.kind === 'constant') { renderPreviewRows([['every row', applyRule(draft, null)]]); return; }
      renderPreviewRows(values.map((v) => [v, applyRule(draft, v)]));
    }

    async function save(status = 'approved') {
      if (draft.kind === 'looked up in another table' && !(draft.rule.table && draft.rule.key && draft.rule.value)) {
        toast('Pick the table, its key column and the value to take.', 'warning', { title: 'Lookup incomplete' }); return;
      }
      if (draft.kind === 'codes converted' && !Object.keys(draft.rule.map || {}).length) {
        toast('Add at least one value pair.', 'warning', { title: 'Code map is empty' }); return;
      }
      if (draft.kind !== 'constant' && !draft.source) { toast('Pick the source column.', 'warning'); return; }
      try {
        const payload = { source: draft.source, kind: draft.kind, rule: draft.rule, transform: draft.kind === 'copied' ? draft.transform : '', status };
        const res = saved ? await api.fields(id, { updates: [{ id: saved.id, ...payload }] })
          : await api.fields(id, { create: [{ target_table: selTable, target: col.name, ...payload }] });
        applyResponse(res);
        renderHead(); renderRail(); renderBoard(); renderInspector();
        toast(`${col.name} saved`, 'success', { timeout: 1600 });
      } catch (e) { toastError(e, 'Could not save'); }
    }
    async function unmap() {
      try { applyResponse(await api.fields(id, { delete: [saved.id] })); renderHead(); renderRail(); renderBoard(); renderInspector(); } catch (e) { toastError(e, 'Could not unmap'); }
    }

    studio.classList.remove('no-inspector');
    mount(inspEl,
      h('div.cs-insp-head',
        h('div.cs-row', h('div.cs-insp-name.cs-grow', col.name, ...keyBadges(col)), h('button.cs-icon-btn.cs-icon-btn-sm', { type: 'button', title: 'Close (esc)', onclick: () => { selCol = null; ctx.setQuery({ col: null }); renderBoard(); renderInspector(); } }, icon('x-lg'))),
        h('div.cs-row.cs-wrap', { style: { marginTop: '6px', gap: '6px' } }, typeChip(col), !col.nullable && h('span.cs-badge.cs-badge-outline', 'required'),
          saved ? [statusBadge(saved.status), madeBy(saved.made_by), confidence(saved.confidence)] : h('span.cs-badge.cs-badge-outline', 'unmapped')),
        col.allowed_values?.length ? h('div.cs-chips', { style: { marginTop: '8px' } }, col.allowed_values.slice(0, 10).map((v) => h('span.cs-chip.mono', v))) : null),
      h('div.cs-insp-body',
        saved?.why ? h('div.cs-callout', { class: String(saved.made_by).includes('ai') ? 'cs-callout-ai' : 'cs-callout-info' }, icon(String(saved.made_by).includes('ai') ? 'stars' : 'info-circle'), h('div', saved.why))
          : unmappedReason ? h('div.cs-callout.cs-callout-warn', icon('question-circle'), h('div', unmappedReason)) : null,
        !src ? h('div.cs-callout.cs-callout-warn', icon('exclamation-triangle'), 'Pick a source table for this target first.') : [
          h('div.cs-field', h('label', 'Filled from source column'), srcSelect, h('div', { style: { marginTop: '8px' } }, sourceProfile)),
          h('div.cs-field', h('label', 'Rule'), kinds),
          editor,
          previewEl,
        ]),
      src && h('div.cs-insp-foot',
        h('button.cs-btn.cs-btn-primary.cs-grow', { type: 'button', onclick: () => save('approved') }, icon('check2'), saved ? 'Save & approve' : 'Map & approve'),
        saved && h('button.cs-icon-btn', { type: 'button', title: 'Remove this mapping', onclick: unmap }, icon('trash3'))));
    drawKinds(); drawEditor(); drawProfile(); drawPreview();
  }

  // ---------------------------------------------------------------- keyboard
  const onKey = (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (['input', 'textarea', 'select'].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
    const tgt = tgtIdx[selTable];
    const cols = tgt?.columns.map((c) => c.name) || [];
    const i = cols.indexOf(selCol);
    const tables = doc.tables.map((p) => p.target);
    const ti = tables.indexOf(selTable);
    const f = selCol && fieldOf(selTable, selCol);
    if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); selCol = cols[Math.min(cols.length - 1, i + 1)] || cols[0]; ctx.setQuery({ col: selCol }); renderBoard(); renderInspector(); scrollSel(); }
    else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); selCol = cols[Math.max(0, i - 1)] || cols[0]; ctx.setQuery({ col: selCol }); renderBoard(); renderInspector(); scrollSel(); }
    else if (e.key === ']') { e.preventDefault(); selectTable(tables[Math.min(tables.length - 1, ti + 1)]); }
    else if (e.key === '[') { e.preventDefault(); selectTable(tables[Math.max(0, ti - 1)]); }
    else if (e.key === 'A') { e.preventDefault(); bulk({ action: 'approve', table: selTable }, `${selTable}: all approved`); }
    else if (e.key === 'a' && f) { e.preventDefault(); setStatus(f, 'approved').then(() => { const nx = cols[i + 1]; if (nx) { selCol = nx; renderBoard(); renderInspector(); scrollSel(); } }); }
    else if (e.key === 'x' && f) { e.preventDefault(); setStatus(f, 'rejected'); }
    else if (e.key === 'u' && f) { e.preventDefault(); setStatus(f, 'suggested'); }
    else if (e.key === 'Escape' && selCol) { selCol = null; ctx.setQuery({ col: null }); renderBoard(); renderInspector(); }
    else if (e.key === '?') { e.preventDefault(); shortcuts(); }
  };
  const scrollSel = () => boardEl.querySelector('.cs-tcol.on')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  document.addEventListener('keydown', onKey);
  ctx.cleanup(() => document.removeEventListener('keydown', onKey));
  const onResize = debounce(() => wireRedraw(), 80);
  window.addEventListener('resize', onResize);
  ctx.cleanup(() => window.removeEventListener('resize', onResize));

  renderHead();
  renderRail();
  renderBoard();
  renderInspector();
}

// ------------------------------------------------------------------ generating

async function generating(ctx, m, specs) {
  const cards = h('div.cs-gen-tables');
  const tablesSeen = new Set();
  const tracker = jobTracker(m.running_job, {
    title: `Mapping ${m.source.name} → ${m.target.name}`, ai: true,
    onLines: (lines) => {
      for (const l of lines) {
        const d = l.detail;
        if (!d?.target || tablesSeen.has(d.target)) continue;
        tablesSeen.add(d.target);
        cards.appendChild(h('div.cs-gen-table', ring(d.total ? d.mapped / d.total : 0, { size: 34, stroke: 4, label: false }),
          h('div.cs-grow', h('div.cs-mono', { style: { fontWeight: 650, fontSize: '12.5px' } }, d.target), h('div.cs-muted', { style: { fontSize: '11.5px' } }, `${d.mapped}/${d.total} columns`)),
          icon('check2-circle', 'text-success')));
      }
    },
  });
  mount(ctx.el,
    h('div.cs-head', h('div', h('div.cs-crumbs', h('a', { href: href('/mappings') }, 'Mappings'), icon('chevron-right'), 'Generating'),
      h('h1', m.name), h('div.cs-head-sub', 'Groq is matching tables and writing column rules. This usually takes under a minute.'))),
    h('div.cs-gen',
      h('div.cs-card.cs-card-pad',
        h('div.cs-gen-orb', h('div.halo'), h('div.halo.two'), h('div.core', icon('stars'))),
        h('div', { style: { textAlign: 'center', marginBottom: '14px' } },
          h('div.cs-row', { style: { justifyContent: 'center', gap: '10px' } },
            h('span.cs-flowchip', logo(specs[m.source.kind], { size: 'xs' }), m.source.name),
            h('span.cs-flow-arrow', h('span.dash'), icon('chevron-right')),
            h('span.cs-flowchip', logo(specs[m.target.kind], { size: 'xs' }), m.target.name))),
        tracker.el),
      h('div.cs-card',
        h('div.cs-card-head', h('div.cs-card-title', icon('table'), 'Tables mapped so far')),
        h('div.cs-card-body', cards))));
  const final = await tracker.done;
  if (!ctx.active) return;
  if (final.status === 'succeeded') {
    confetti();
    toast(final.message || 'Mapping ready', 'success', { title: 'Ready for review' });
    window.dispatchEvent(new Event('cs:changed'));
    setTimeout(() => ctx.active && navigate(`/mappings/${m.id}`, { replace: true }), 900);
  }
}

function failed(ctx, m) {
  mount(ctx.el, h('div.cs-card', empty({ icon: 'x-octagon', title: 'Mapping failed', message: 'The last generation did not finish. Check Activity for the log, then try again.',
    action: h('div.cs-row', h('a.cs-btn', { href: href('/activity?kind=map') }, icon('activity'), 'Open activity'),
      h('button.cs-btn.cs-btn-ai', { type: 'button', onclick: async () => { await api.generate(m.id, {}); navigate(`/mappings/${m.id}`, { replace: true }); } }, icon('stars'), 'Try again')) })));
}
