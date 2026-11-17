// Source detail: schema explorer, relationship ERD, snapshots & drift, settings, activity.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { href, navigate } from '../../../ui/js/router.js';
import { confirmDialog, modal, toast, toastError } from '../../../ui/js/ui.js';
import { api } from '../api.js';
import { erd } from '../erd.js';
import {
  ago, compact, connectorForm, dropzone, empty, fmtBytes, fmtDuration, jobTracker, keyBadges, kv, logo, n, openJobDrawer,
  seg, statusBadge, statusDot, typeChip,
} from '../kit.js';

export default async function render(ctx) {
  const id = Number(ctx.params.id);
  const [meta, detail] = await Promise.all([api.meta(), api.source(id, { signal: ctx.signal })]);
  let { source } = detail;
  const spec = meta.connectors.find((c) => c.key === source.kind) || { label: source.kind, fields: [], uploads: [] };
  ctx.setTitle(source.name);
  let snapshot = source.snapshot ? (await api.snapshot(source.snapshot.id, { signal: ctx.signal })).snapshot : null;
  let tab = ctx.query.get('tab') || 'schema';
  let selected = ctx.query.get('table') || null;

  const header = h('div');
  const tabsEl = h('div.cs-tabs');
  const content = h('div');

  function renderHeader() {
    const st = snapshot?.stats;
    mount(header, h('div.cs-card',
      h('div.cs-card-body.cs-col.cs-gap-lg',
        h('div.cs-row.cs-wrap', { style: { gap: '16px' } },
          logo(spec, { size: 'lg', solid: true }),
          h('div.cs-grow',
            h('div.cs-crumbs', h('a', { href: href('/sources') }, 'Sources'), icon('chevron-right'), spec.label),
            h('h1', { style: { fontFamily: 'var(--cs-display)', fontSize: '24px', fontWeight: 700, margin: 0, display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' } },
              source.name, statusBadge(source.status)),
            h('div.cs-muted', { style: { fontSize: '13px', marginTop: '2px' } },
              source.description || spec.description, source.last_error && source.status === 'error' ? h('div', { style: { color: 'var(--cs-bad)' } }, icon('exclamation-octagon'), ' ', source.last_error) : null)),
          h('div.cs-head-actions',
            h('button.cs-btn', { type: 'button', onclick: test }, icon('plug'), 'Test'),
            h('button.cs-btn.cs-btn-primary', { type: 'button', onclick: extract }, icon('arrow-clockwise'), snapshot ? 'Re-extract' : 'Extract schema'),
            h('a.cs-btn.cs-btn-ai', { href: href(`/mappings/new?${source.kind === 'schema_file' ? 'target' : 'source'}=${id}`) }, icon('stars'), 'Map with AI'),
            source.storage?.total ? h('button.cs-btn', { type: 'button', onclick: freeSpace,
              title: 'Delete the uploads and the data restored from them; keep snapshots and mappings' }, icon('hdd'), `Free ${fmtBytes(source.storage.total)}`) : null,
            h('button.cs-icon-btn', { type: 'button', title: 'Delete source', onclick: remove }, icon('trash3')))),
        source.storage?.freed_at && !source.files.length ? h('div.cs-callout.cs-callout-info', icon('hdd'),
          h('div', `Files deleted ${ago(source.storage.freed_at)} to free space (${(source.config?.freed_files || []).join(', ')}). `,
            'Snapshots and mappings are kept. Upload the files again (Settings) to test, re-extract or sync.')) : null,
        st ? kv([
          ['Tables', n(st.tables)], ['Columns', n(st.columns)],
          ['Relationships', `${n(st.relationships)} (${n(st.declared_relationships)} declared)`],
          ['Rows', compact(st.rows)], ['Extracted', ago(snapshot.created_at)],
          ['Took', fmtDuration((snapshot.duration_ms || 0) / 1000)], ['Extractor', snapshot.data.extractor || '—', snapshot.data.extractor],
        ]) : h('div.cs-callout.cs-callout-info', icon('info-circle'), h('div', 'No snapshot yet. ', h('b', 'Extract schema'), ' to see tables, columns, profiles and relationships.')),
        snapshot?.data?.warnings?.length ? h('details', h('summary.cs-muted', { style: { cursor: 'pointer', fontSize: '12.5px' } }, icon('exclamation-triangle'), ` ${snapshot.data.warnings.length} extraction warning(s)`),
          h('ul', { style: { fontSize: '12.5px', marginTop: '6px' } }, snapshot.data.warnings.map((w) => h('li', w)))) : null)));
  }

  function renderTabs() {
    const items = [
      ['schema', 'Schema', 'table', snapshot?.stats?.tables],
      ['relationships', 'Relationships', 'diagram-3', snapshot?.stats?.relationships],
      ['snapshots', 'Snapshots', 'clock-history', source.snapshots],
      ['settings', 'Settings', 'sliders'],
      ['activity', 'Activity', 'activity'],
    ];
    mount(tabsEl, items.map(([key, label, ic, count]) => h('button.cs-tab', { type: 'button', class: tab === key && 'on',
      onclick: () => { tab = key; ctx.setQuery({ tab: key === 'schema' ? null : key }); renderTabs(); renderContent(); } },
    icon(ic), label, count != null && h('span.n', String(count)))));
  }

  // ---------------------------------------------------------------- schema explorer
  function schemaTab() {
    if (!snapshot) return emptySnapshot();
    const tables = snapshot.data.tables;
    const maxRows = Math.max(1, ...tables.map((t) => t.rows || 0));
    let q = '';
    const list = h('div.cs-tlist');
    const detailEl = h('div.cs-tdetail');
    const drawList = () => mount(list, tables.filter((t) => !q || t.key.toLowerCase().includes(q) || t.columns.some((c) => c.name.toLowerCase().includes(q)))
      .map((t) => h('button.cs-titem', { type: 'button', class: t.key === selected && 'on', onclick: () => { selected = t.key; ctx.setQuery({ table: t.key }); drawList(); drawDetail(); } },
        icon(t.kind === 'view' ? 'eye' : 'table', 'cs-muted'),
        h('div.cs-grow', h('div.cs-titem-name.cs-ellipsis', t.key),
          h('div.cs-titem-sub', `${t.columns.length} cols${t.rows != null ? ` · ${compact(t.rows)} rows${t.rows_sampled ? ' (sample)' : ''}` : ''}`)),
        t.columns.some((c) => c.profile?.pii?.length) && h('span.cs-key.cs-key-pii', { title: 'Contains likely PII' }, 'PII'),
        h('span.cs-titem-bar', h('span', { style: { width: `${((t.rows || 0) / maxRows) * 100}%` } })))));

    function drawDetail() {
      const t = tables.find((x) => x.key === selected) || tables[0];
      if (!t) { mount(detailEl, empty({ icon: 'table', title: 'No tables' })); return; }
      selected = t.key;
      const rels = snapshot.data.relationships.filter((r) => r.from === t.key || r.to === t.key);
      mount(detailEl,
        h('div.cs-tdetail-head',
          h('span.cs-logo.cs-logo-sm', { style: { '--c': '#0072ce' } }, icon(t.kind === 'view' ? 'eye' : 'table')),
          h('div.cs-grow', h('div.cs-tdetail-name', t.key),
            h('div.cs-muted', { style: { fontSize: '12.5px' } }, [t.kind, `${t.columns.length} columns`, t.rows != null && `${n(t.rows)} rows${t.rows_sampled ? ' sampled' : ''}`,
              t.primary_key?.length && `PK (${t.primary_key.join(', ')})`, t.model && `model ${t.model}`].filter(Boolean).join(' · '))),
          rels.length ? h('span.cs-badge.cs-badge-info', icon('diagram-3'), `${rels.length} relationship${rels.length > 1 ? 's' : ''}`) : null,
          h('button.cs-btn.cs-btn-sm', { type: 'button', onclick: () => { tab = 'relationships'; renderTabs(); renderContent(); } }, icon('diagram-3'), 'ERD')),
        t.comment && h('div.cs-callout.cs-callout-info', { style: { margin: '12px 18px 0' } }, icon('chat-quote'), t.comment),
        h('div', { style: { overflow: 'auto', maxHeight: '620px' } },
          h('table.cs-coltable',
            h('thead', h('tr', h('th', '#'), h('th', 'Column'), h('th', 'Type'), h('th', 'Nulls'), h('th', 'Distinct'), h('th', 'Values'))),
            h('tbody', t.columns.map((c) => {
              const p = c.profile;
              const top = p?.top?.length && p.distinct != null && p.distinct <= 30 ? p.top : null;
              return h('tr',
                h('td.cs-muted.cs-num', String(c.position)),
                h('td', h('div.cs-row.cs-gap-sm', h('span.cs-colname', c.name), ...keyBadges(c), !c.nullable && !c.pk && h('span.cs-faint', { title: 'NOT NULL', style: { fontSize: '11px' } }, '●')),
                  c.references && h('a.cs-fk-link', { onclick: () => { selected = c.references.table; drawList(); drawDetail(); } },
                    `→ ${c.references.table}.${c.references.column}${c.references.kind === 'inferred' ? ` · inferred ${Math.round(c.references.confidence * 100)}%` : ''}`),
                  c.comment && h('div.cs-muted', { style: { fontSize: '11.5px' } }, c.comment)),
                h('td', typeChip(c)),
                h('td', p ? h('div.cs-nullbar', h('div.cs-meter', h('span', { style: { width: `${p.null_pct}%`, background: p.null_pct > 50 ? '#f59e0b' : '#94a3b8' } })), `${p.null_pct.toFixed(p.null_pct < 10 && p.null_pct > 0 ? 1 : 0)}%`) : h('span.cs-faint', '—')),
                h('td.cs-num', p?.distinct != null ? n(p.distinct) : h('span.cs-faint', '—')),
                h('td', { style: { maxWidth: '380px' } }, top ? h('div.cs-chips', top.slice(0, 8).map((v) => h('span.cs-chip.mono', { title: `${v.v}: ${v.n}` }, v.v, h('small', String(v.n)))))
                  : p?.samples?.length ? h('div.cs-chips', p.samples.slice(0, 4).map((v) => h('span.cs-chip.mono', { title: v }, v)))
                    : c.allowed_values?.length ? h('div.cs-chips', c.allowed_values.slice(0, 8).map((v) => h('span.cs-chip.mono', v))) : h('span.cs-faint', '—')));
            })))),
        rels.length ? h('div', { style: { padding: '12px 18px', borderTop: '1px solid var(--cs-border)' } },
          h('div.cs-form-section', 'Relationships'),
          h('div.cs-col.cs-gap-sm', { style: { marginTop: '10px' } }, rels.map((r) => h('div.cs-row.cs-gap-sm', { style: { fontSize: '12.5px' } },
            h('span.cs-badge', { class: r.kind === 'inferred' ? 'cs-badge-ai' : 'cs-badge-info' }, r.kind === 'inferred' ? icon('stars') : icon('key'), r.kind === 'inferred' ? `inferred ${Math.round(r.confidence * 100)}%` : 'declared'),
            h('a.cs-fk-link', { onclick: () => { selected = r.from; drawList(); drawDetail(); } }, `${r.from}.${r.from_columns.join(',')}`),
            icon('arrow-right', 'cs-faint'),
            h('a.cs-fk-link', { onclick: () => { selected = r.to; drawList(); drawDetail(); } }, `${r.to}.${r.to_columns.join(',')}`))))) : null);
    }
    drawList();
    drawDetail();
    return h('div.cs-explorer',
      h('div.cs-explorer-list',
        h('div.cs-explorer-search', h('div.cs-input-wrap', icon('search'), h('input.cs-input.cs-input-sm', { placeholder: `Search ${tables.length} tables & columns…`,
          oninput: (e) => { q = e.target.value.trim().toLowerCase(); drawList(); } }))),
        list),
      detailEl);
  }

  function relationshipsTab() {
    if (!snapshot) return emptySnapshot();
    const rels = snapshot.data.relationships;
    if (!snapshot.data.tables.length) return emptySnapshot();
    return h('div',
      h('div.cs-card-head', h('div.cs-card-title', icon('diagram-3'), `${n(rels.length)} relationships`),
        h('span.cs-badge.cs-badge-info', `${rels.filter((r) => r.kind !== 'inferred').length} declared`),
        h('span.cs-badge.cs-badge-ai', icon('stars'), `${rels.filter((r) => r.kind === 'inferred').length} inferred by dbatlas`)),
      erd(snapshot.data, { onOpen: (key) => { selected = key; tab = 'schema'; ctx.setQuery({ tab: null, table: key }); renderTabs(); renderContent(); } }));
  }

  async function snapshotsTab() {
    const box = h('div.cs-card-body', h('div.af-skeleton-wrap', h('div.af-skeleton', { style: { height: '18px', width: '60%' } })));
    api.snapshots(id).then(({ snapshots }) => {
      if (!snapshots.length) { mount(box, empty({ icon: 'camera', title: 'No snapshots yet', message: 'Every extraction saves a versioned snapshot so you can see schema drift.' })); return; }
      mount(box, h('div.cs-timeline', snapshots.map((sn, i) => h('div.cs-tl-item', { class: sn.diff?.first && 'first' },
        h('div.cs-row.cs-wrap',
          h('b', `Snapshot #${sn.id}`), h('span.cs-muted', ago(sn.created_at)),
          h('span.cs-badge.cs-badge-outline.cs-mono', sn.fingerprint),
          sn.diff?.first ? h('span.cs-badge.cs-badge-teal', 'first') : h('span.cs-badge', { class: sn.diff?.summary === 'no changes' ? 'cs-badge-outline' : 'cs-badge-warn' }, icon('shuffle'), sn.diff?.summary),
          h('span.cs-grow'),
          i < snapshots.length - 1 && h('button.cs-btn.cs-btn-sm', { type: 'button', onclick: () => showDiff(sn, snapshots[i + 1]) }, icon('file-diff'), 'Changes'),
          sn.id !== snapshot?.id && h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: async () => {
            snapshot = (await api.snapshot(sn.id)).snapshot; renderHeader(); toast(`Viewing snapshot #${sn.id}`, 'info'); tab = 'schema'; renderTabs(); renderContent();
          } }, icon('eye'), 'View')),
        h('div.cs-muted', { style: { fontSize: '12.5px', marginTop: '4px' } },
          `${n(sn.stats.tables)} tables · ${n(sn.stats.columns)} columns · ${n(sn.stats.relationships)} relationships · ${fmtDuration((sn.duration_ms || 0) / 1000)} · ${sn.extractor || ''}`)))));
    }).catch((e) => mount(box, h('div.cs-callout.cs-callout-warn', icon('exclamation-triangle'), e.message)));
    return box;
  }

  async function showDiff(a, b) {
    const { diff } = await api.compare(a.id, b.id);
    modal({
      title: `Snapshot #${b.id} → #${a.id}`, subtitle: diff.summary, icon: 'file-diff', size: 'lg',
      body: h('div.cs-col', { style: { gap: '14px' } },
        diff.added_tables.length ? h('div', h('div.cs-form-section', `Added tables (${diff.added_tables.length})`), h('div.cs-chips', { style: { marginTop: '8px' } }, diff.added_tables.map((t) => h('span.cs-chip.mono.cs-diff-add', `+ ${t}`)))) : null,
        diff.removed_tables.length ? h('div', h('div.cs-form-section', `Removed tables (${diff.removed_tables.length})`), h('div.cs-chips', { style: { marginTop: '8px' } }, diff.removed_tables.map((t) => h('span.cs-chip.mono.cs-diff-rem', `− ${t}`)))) : null,
        diff.changed_tables.length ? h('div', h('div.cs-form-section', `Changed tables (${diff.changed_tables.length})`),
          h('div.cs-col.cs-gap-sm', { style: { marginTop: '8px' } }, diff.changed_tables.map((c) => h('div', h('b.cs-mono', c.table), h('div.cs-chips', { style: { marginTop: '4px' } },
            c.added_columns.map((x) => h('span.cs-chip.mono.cs-diff-add', `+ ${x}`)), c.removed_columns.map((x) => h('span.cs-chip.mono.cs-diff-rem', `− ${x}`)),
            c.type_changes.map((x) => h('span.cs-chip.mono.cs-diff-chg', `${x.column}: ${x.from} → ${x.to}`))))))) : null,
        !diff.added_tables.length && !diff.removed_tables.length && !diff.changed_tables.length ? h('div.cs-callout.cs-callout-info', icon('check2-circle'), 'The structure is identical.') : null),
      actions: [{ label: 'Close' }],
    });
  }

  function settingsTab() {
    const form = connectorForm(spec, { values: source.config || {}, secretsSet: source.secrets_set || [], name: source.name });
    const filesEl = h('div.cs-files');
    const drawFiles = () => mount(filesEl, source.files.map((f) => h('div.cs-file', icon('file-earmark-check'), h('span.cs-grow.cs-ellipsis', f.name),
      h('span.cs-muted.cs-mono', fmtBytes(f.size)), h('span.cs-muted', { style: { fontSize: '12px' } }, ago(f.uploaded_at)),
      h('button.cs-icon-btn.cs-icon-btn-sm.bad', { type: 'button', title: 'Remove', onclick: async () => {
        try { await api.deleteFile(id, f.id); ({ source } = await api.source(id)); drawFiles(); renderHeader(); } catch (e) { toastError(e, 'Remove failed'); }
      } }, icon('x-lg')))));
    drawFiles();
    const save = h('button.cs-btn.cs-btn-primary', { type: 'button', onclick: async () => {
      save.disabled = true;
      try {
        ({ source } = await api.updateSource(id, { name: form.name(), config: form.values() }));
        toast('Saved — test the connection to confirm it still works.', 'success'); renderHeader();
      } catch (e) { toastError(e, 'Save failed'); } finally { save.disabled = false; }
    } }, icon('check2'), 'Save changes');
    return h('div.cs-card-body', h('div.cs-col.cs-gap-lg', { style: { maxWidth: '820px' } },
      form.el,
      spec.uploads.length ? h('div.cs-field', h('label', 'Files'),
        dropzone({ accept: spec.uploads, multiple: spec.multi_upload, onFiles: async (files) => {
          try { ({ source } = await api.uploadFiles(id, files)); drawFiles(); renderHeader(); toast('Uploaded — re-extract to refresh the snapshot.', 'success'); } catch (e) { toastError(e, 'Upload failed'); }
        } }), filesEl) : null,
      h('div.cs-row', save, h('button.cs-btn', { type: 'button', onclick: test }, icon('plug'), 'Test connection'))));
  }

  function activityTab() {
    const jobs = detail.jobs || [];
    const maps = detail.mappings || [];
    return h('div.cs-grid.cs-grid-2', { style: { padding: '16px' } },
      h('div.cs-card', h('div.cs-card-head', h('div.cs-card-title', icon('activity'), 'Jobs')),
        h('div.cs-list', jobs.length ? jobs.map((j) => h('div.cs-li.click', { onclick: () => openJobDrawer(j) }, statusDot(j.status),
          h('div.cs-grow', h('div.cs-li-title', j.kind_label), h('div.cs-li-sub.cs-ellipsis', j.message)), h('span.cs-muted', { style: { fontSize: '12px' } }, ago(j.created_at))))
          : h('div.cs-li.cs-muted', 'No jobs yet.'))),
      h('div.cs-card', h('div.cs-card-head', h('div.cs-card-title', icon('stars'), 'Used in mappings')),
        h('div.cs-list', maps.length ? maps.map((m) => h('a.cs-li', { href: href(`/mappings/${m.id}`) }, icon('stars', 'cs-muted'),
          h('div.cs-grow', h('div.cs-li-title', m.name), h('div.cs-li-sub', `${m.source.id === id ? 'as source' : 'as target'} · ${ago(m.updated_at)}`)), statusBadge(m.status)))
          : h('div.cs-li.cs-muted', 'Not used in a mapping yet.'))));
  }

  function emptySnapshot() {
    return empty({ icon: 'camera', title: 'No schema snapshot yet', message: 'Extract the schema to explore tables, profiles and relationships.',
      action: h('button.cs-btn.cs-btn-brand', { type: 'button', onclick: extract }, icon('lightning-charge-fill'), 'Extract schema') });
  }

  async function renderContent() {
    const panel = { schema: schemaTab, relationships: relationshipsTab, snapshots: snapshotsTab, settings: settingsTab, activity: activityTab }[tab] || schemaTab;
    mount(content, h('div', { style: { textAlign: 'center', padding: '30px' } }, h('span.spinner-border.text-primary')));
    mount(content, await panel());
  }

  async function test() {
    try {
      const { job } = await api.testSource(id);
      const t = jobTracker(job.id, { title: `Testing ${source.name}`, showLog: false });
      modal({ title: 'Connection test', icon: 'plug', body: h('div.cs-root', { 'data-theme': document.getElementById('cs-root').dataset.theme }, t.el), actions: [{ label: 'Close' }] });
      const final = await t.done;
      ({ source } = await api.source(id));
      renderHeader();
      if (final.status === 'succeeded') toast(final.result?.message || 'Connected', 'success', { title: 'Connection works' });
    } catch (e) { toastError(e, 'Test failed'); }
  }

  async function extract() {
    try {
      const { job } = await api.extract(id, {});
      const t = jobTracker(job.id, { title: `Extracting ${source.name}` });
      const m = modal({ title: 'Schema extraction', subtitle: 'dbatlas · introspect → profile → relationships', icon: 'lightning-charge', size: 'lg',
        body: h('div.cs-root', { 'data-theme': document.getElementById('cs-root').dataset.theme }, t.el), actions: [{ label: 'Run in background' }] });
      const final = await t.done;
      if (final.status === 'succeeded') {
        m.close();
        toast(final.message, 'success', { title: 'Snapshot saved' });
        const d = await api.source(id);
        source = d.source;
        snapshot = source.snapshot ? (await api.snapshot(source.snapshot.id)).snapshot : null;
        renderHeader(); renderTabs(); renderContent();
        window.dispatchEvent(new Event('cs:changed'));
      }
    } catch (e) { toastError(e, 'Extraction failed'); }
  }

  async function freeSpace() {
    const st = source.storage || {};
    if (!(await confirmDialog({
      title: `Free ${fmtBytes(st.total)} from ${source.name}?`,
      message: `Deletes the uploaded file(s) (${fmtBytes(st.uploads)}) and the data restored or staged from them (${fmtBytes(st.work)}). `
        + 'Snapshots, mappings and anything already synced to a destination are kept. To test, re-extract or sync again, upload the files again.',
      confirmLabel: 'Delete files', danger: true }))) return;
    try {
      const res = await api.freeSpace(id);
      source = res.source;
      toast(`${fmtBytes(res.freed_bytes)} freed`, 'success', { title: 'Files deleted' });
      renderHeader(); renderContent();
    } catch (e) { toastError(e, 'Could not free space'); }
  }

  async function remove() {
    if (!(await confirmDialog({ title: `Delete ${source.name}?`, message: 'Its snapshots, uploads, data restored from them and the mappings that use it are deleted too. Data already synced to a destination stays there.', confirmLabel: 'Delete', danger: true }))) return;
    try { await api.deleteSource(id); toast('Source deleted', 'success'); navigate('/sources'); } catch (e) { toastError(e, 'Delete failed'); }
  }

  mount(ctx.el, header, h('div.cs-card', tabsEl, content));
  renderHeader();
  renderTabs();
  renderContent();
}
