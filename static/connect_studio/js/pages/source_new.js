// Add a source: choose a connector → configure & test → extract the schema with dbatlas.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { href, navigate } from '../../../ui/js/router.js';
import { toast, toastError } from '../../../ui/js/ui.js';
import { api } from '../api.js';
import { confetti, connectorForm, dropzone, fmtBytes, jobTracker, kv, logo, n, seg } from '../kit.js';
import { importPanel, v3ImportPanel } from './rest_import.js';

const GUIDES = {
  Databases: [
    ['What gets extracted', 'Tables, views, columns, types, primary keys, unique constraints and declared foreign keys — then dbatlas infers the relationships nobody declared (e.g. visit.patient_id → patients.id).'],
    ['Profiling', 'A sample of rows per table (you choose how many) is profiled: null rates, distinct counts, top values and PII signals. The AI uses these to write code maps and splits.'],
    ['Tip', 'A read-only database user is enough for mapping. Syncing into this database needs write access.'],
  ],
  Warehouses: [
    ['What gets extracted', 'Schemas, tables, columns and declared constraints plus inferred relationships and sampled profiles.'],
    ['Tip', 'Use a small warehouse — extraction runs a COUNT and a LIMIT sample per table.'],
  ],
  Files: [
    ['One table per file', 'Each CSV/JSON/Parquet file becomes a table; every Excel sheet becomes its own table.'],
    ['Relationships', 'dbatlas infers relationships across your files from column names (patient_id → patients).'],
  ],
  APIs: [
    ['Import', 'Have a Postman collection or an OpenAPI/Swagger file? Drop it in — base URL, auth (including login requests), pagination and streams are filled in, and AI picks the endpoints that are real tables.'],
    ['Streams', 'Each stream is one endpoint = one table. Write `name = /path -> records.path`, or `name = POST /path {"json": "body"} -> data` for APIs that read with POST. The records path is optional.'],
    ['Pagination', 'Page number, offset, cursor or next-link pagination is handled for you; set the parameter names your API uses.'],
    ['Nested JSON', 'Nested objects are flattened one level (address__city); deeper values are kept as JSON text.'],
  ],
  sql_dump: [
    ['Any engine', 'MySQL / MariaDB (mysqldump, phpMyAdmin), PostgreSQL (pg_dump), SQL Server (SSMS ▸ Tasks ▸ Generate Scripts ▸ Schema and data), SQLite (.dump) and Oracle (SQL Developer export). The engine is detected; pick it under “Dump from” if it guesses wrong.'],
    ['No database server needed', 'The dump is restored into a private DuckDB copy on this server. Tables, row counts, declared keys, foreign keys, comments and enum values come from the dump itself; columns get their declared types when every value fits (otherwise they stay text and the test lists why).'],
    ['Big dumps', 'Any size up to 20 GB: files go up in 16 MB chunks, each checked, and a dropped connection resumes — choose the same file again. Compressing (.gz / .zip, ~10× smaller) still makes uploads much faster. “Only these tables” restores just what you need.'],
    ['Where the files live', 'Uploads are kept in encrypted cloud storage (S3), not on the server. Each test, extraction or sync works in a scratch folder that is deleted when it ends; the restored copy of the dump is cached in storage so the next job doesn’t parse the dump again.'],
    ['Not accepted', 'Binary backups: SQL Server .bak and Oracle .dmp — script them out first. PostgreSQL custom-format (-Fc) dumps work only where pg_restore is installed; otherwise export with pg_dump --format=plain.'],
    ['After the migration', 'Source ▸ “Free … MB” deletes the upload and the cached restored copy from storage but keeps snapshots and mappings. Deleting the source removes everything it stored.'],
  ],
  afya_platform: [
    ['Account = tenant', 'Log in with a V3 account of the organization you want to read (e.g. its superadmin). Every read is scoped to that account’s organization — there is no tenant id to type.'],
    ['Models', 'List them as service:alias — reception:patient, evaluation:visits, evaluation:vitals, finance:invoice. Leave blank to discover every readable model on the chosen services (a few hundred on production: slow to extract).'],
    ['Environment', 'Production is https://{service}.afyaanalytics.ai/api/ (the default). The migrate platform is https://{service}migrate.afyaanalytics.ai/api/.'],
    ['Server credentials', 'Instead of typing a password, set a credential key such as WEMA_PROD: AFYA_WEMA_PROD_USERNAME / _PASSWORD / _URL_TEMPLATE are read from the server environment.'],
  ],
  Schema: [
    ['Accepted formats', 'schema.json in many shapes ({"tables": [...]}, {table: {column: type}}), JSON Schema, SQL DDL (CREATE TABLE … in any dialect), a dbatlas.json, or an existing mappings.json (its target side).'],
    ['Relationships', 'Declared foreign keys are read; missing ones are inferred by dbatlas.'],
    ['Value lists', 'enum / allowed_values on columns become the target values the AI maps codes onto.'],
  ],
};

export default async function render(ctx) {
  ctx.setTitle('Add a source');
  const meta = await api.meta();
  let spec = meta.connectors.find((c) => c.key === ctx.query.get('kind')) || null;
  let source = null;     // created lazily (draft) on first test / upload / continue
  let step = spec ? 2 : 1;
  let form = null;
  let testInfo = null;
  const pendingUploads = [];

  const stepper = h('div.cs-steps');
  const body = h('div');
  const renderStepper = () => mount(stepper, [['Connector', 1], ['Configure & test', 2], ['Extract schema', 3]].map(([label, i], idx) => [
    idx ? h('div.cs-step-line', { class: step > i - 1 && 'done' }) : null,
    h('div.cs-step', { class: [step === i && 'on', step > i && 'done'] }, h('span.cs-step-n', step > i ? icon('check-lg') : String(i)), label)]));

  mount(ctx.el,
    h('div.cs-head',
      h('div', h('div.cs-crumbs', h('a', { href: href('/sources') }, 'Sources'), icon('chevron-right'), 'New'),
        h('h1', 'Add a source'), h('div.cs-head-sub', 'Connect anything — the same source can be mapped from, used as a target schema, or synced into.')),
      stepper),
    body);

  // ---------------------------------------------------------------- step 1
  function stepChoose() {
    step = 1; renderStepper();
    let cat = '';
    let q = '';
    const cats = [...new Set(meta.connectors.map((c) => c.category))];
    const grid = h('div.cs-grid.cs-grid-tiles');
    const draw = () => mount(grid, meta.connectors.filter((c) => (!cat || c.category === cat) && (!q || `${c.label} ${c.description} ${c.tagline}`.toLowerCase().includes(q)))
      .map((c) => h('button.cs-tile', { type: 'button', style: { '--c': c.color }, class: !c.driver_ok && 'disabled',
        title: c.driver_ok ? `Add ${c.label}` : `Driver not installed: pip install ${c.pip}`,
        onclick: () => { if (!c.driver_ok) { toast(`Install the driver first: pip install ${c.pip}`, 'warning', { title: `${c.label} unavailable` }); return; } spec = c; ctx.setQuery({ kind: c.key }); stepConfigure(); } },
        h('div.cs-row', logo(c), h('div.cs-grow', h('div.cs-tile-name', c.label), h('div.cs-tile-cat', c.category))),
        h('div.cs-tile-desc', c.description),
        h('div.cs-tile-foot',
          c.tagline && h('span.cs-badge.cs-badge-outline', c.tagline),
          c.roles.includes('destination') && h('span.cs-badge.cs-badge-teal', { title: 'Can be a sync destination' }, icon('box-arrow-in-down'), 'destination'),
          !c.driver_ok && h('span.cs-badge.cs-badge-warn', icon('exclamation-triangle'), 'driver missing')))));
    mount(body, h('div.cs-col.cs-gap-lg',
      h('div.cs-row.cs-wrap',
        seg([{ value: '', label: 'All', count: meta.connectors.length }, ...cats.map((c) => ({ value: c, label: c, count: meta.connectors.filter((x) => x.category === c).length }))], cat, (v) => { cat = v; draw(); }),
        h('span.cs-grow'),
        h('div.cs-input-wrap', { style: { width: '300px' } }, icon('search'),
          h('input.cs-input', { placeholder: `Search ${meta.connectors.length} connectors…`, autofocus: true, oninput: (e) => { q = e.target.value.trim().toLowerCase(); draw(); } }))),
      grid));
    draw();
  }

  // ---------------------------------------------------------------- step 2
  async function ensureSource() {
    const config = form.values();
    const name = form.name() || spec.label;
    if (!source) {
      ({ source } = await api.createSource({ kind: spec.key, name, config }));
    } else {
      ({ source } = await api.updateSource(source.id, { name, config }));
    }
    return source;
  }

  function stepConfigure() {
    step = 2; renderStepper();
    form = connectorForm(spec, { values: source?.config || {}, secretsSet: source?.secrets_set || [], name: source?.name || '' });
    const resultBox = h('div');
    const filesBox = h('div.cs-files');
    const renderFiles = () => mount(filesBox, [
      ...(source?.files || []).map((f) => h('div.cs-file', icon('file-earmark-check'), h('span.cs-grow.cs-ellipsis', f.name), h('span.cs-muted.cs-mono', fmtBytes(f.size)),
        h('button.cs-icon-btn.cs-icon-btn-sm.bad', { type: 'button', title: 'Remove', onclick: async () => {
          await api.deleteFile(source.id, f.id); ({ source } = await api.source(source.id)); renderFiles();
        } }, icon('x-lg')))),
      ...pendingUploads.map((u) => h('div.cs-file', icon('file-earmark'), h('span.cs-grow.cs-ellipsis', u.file.name), h('span.cs-muted.cs-mono', fmtBytes(u.file.size)),
        h('div.cs-upbar', h('span', { style: { width: `${Math.round(u.progress * 100)}%` } })))),
    ]);
    const uploads = spec.uploads.length ? h('div.cs-field',
      h('label', spec.key === 'schema_file' ? 'Schema file' : spec.multi_upload ? 'Files' : 'Database file', spec.key !== 'duckdb' && h('span.req', '*')),
      dropzone({ accept: spec.uploads, multiple: spec.multi_upload, title: spec.key === 'schema_file' ? 'Drop schema.json, a .sql DDL file or mappings.json' : `Drop ${spec.multi_upload ? 'files' : 'a file'} here`,
        onFiles: async (files) => {
          try {
            await ensureSource();
            for (const file of files) {
              const item = { file, progress: 0 };
              pendingUploads.push(item);
              renderFiles();
              const res = await api.uploadFiles(source.id, [file], { onProgress: (p) => { item.progress = p; renderFiles(); } });
              pendingUploads.splice(pendingUploads.indexOf(item), 1);
              source = res.source;
              renderFiles();
            }
            toast(`${files.length} file(s) uploaded`, 'success');
          } catch (e) { pendingUploads.length = 0; renderFiles(); toastError(e, 'Upload failed'); }
        } }),
      filesBox) : null;
    renderFiles();

    const testBtn = h('button.cs-btn', { type: 'button', onclick: () => runTest() }, icon('plug'), 'Test connection');
    const nextBtn = h('button.cs-btn.cs-btn-brand', { type: 'button', onclick: () => saveAndContinue() }, 'Save & extract schema', icon('arrow-right'));

    async function runTest() {
      const missing = form.missing();
      if (missing.length) { toast(`Fill in: ${missing.join(', ')}`, 'warning'); return; }
      testBtn.disabled = true;
      mount(resultBox, h('div.cs-result.run', h('span.cs-result-icon', h('span.spinner-border.spinner-border-sm')), h('div', h('div.cs-result-title', 'Testing the connection…'), h('div.cs-result-msg', `Reaching ${spec.label}`))));
      try {
        await ensureSource();
        const { job } = await api.testSource(source.id);
        const t = jobTracker(job.id, { showLog: false });
        const final = await t.done;
        if (final.status === 'succeeded') {
          testInfo = final.result || {};
          mount(resultBox, h('div.cs-result.ok', h('span.cs-result-icon', icon('check-lg')),
            h('div.cs-grow', h('div.cs-result-title', 'Connection works'), h('div.cs-result-msg', testInfo.message),
              testInfo.version && h('div.cs-result-msg.cs-muted', testInfo.version),
              testInfo.preview?.length ? h('details', { style: { marginTop: '6px' } }, h('summary.cs-muted', { style: { fontSize: '12px', cursor: 'pointer' } }, 'First records'),
                h('pre.cs-code', { style: { maxHeight: '200px', marginTop: '6px' } }, JSON.stringify(testInfo.preview, null, 2))) : null),
            testInfo.latency_ms != null && h('span.cs-badge.cs-badge-ok', `${testInfo.latency_ms} ms`)));
        } else {
          mount(resultBox, h('div.cs-result.bad', h('span.cs-result-icon', icon('x-lg')),
            h('div', h('div.cs-result-title', 'Could not connect'), h('div.cs-result-msg', final.error || final.message || 'Unknown error'))));
        }
      } catch (e) {
        mount(resultBox, h('div.cs-result.bad', h('span.cs-result-icon', icon('x-lg')), h('div', h('div.cs-result-title', 'Could not connect'), h('div.cs-result-msg', e.message))));
      } finally { testBtn.disabled = false; }
    }

    async function saveAndContinue() {
      const missing = form.missing();
      if (missing.length) { toast(`Fill in: ${missing.join(', ')}`, 'warning'); return; }
      if (spec.uploads.length && spec.key !== 'duckdb' && !(source?.files?.length) && !form.values().path && !form.values().schema_text) {
        toast('Upload a file first.', 'warning'); return;
      }
      nextBtn.disabled = true;
      try {
        await ensureSource();
        if (source.status === 'draft') ({ source } = await api.updateSource(source.id, { status: 'untested' }));
        stepExtract();
      } catch (e) { toastError(e, 'Could not save'); } finally { nextBtn.disabled = false; }
    }

    const guide = GUIDES[spec.key] || GUIDES[spec.category] || [];
    mount(body,
      h('div.cs-wizard',
        h('div.cs-card',
          h('div.cs-card-head', logo(spec, { solid: true }), h('div.cs-grow', h('div.cs-card-title', spec.label), h('div.cs-muted', { style: { fontSize: '12.5px' } }, spec.description)),
            !ctx.query.get('locked') && h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => { ctx.setQuery({ kind: null }); stepChoose(); } }, icon('arrow-left-right'), 'Change')),
          h('div.cs-card-body.cs-col.cs-gap-lg',
            spec.key === 'rest_api' && importPanel({ form, meta, onApplied: () => form.el.querySelector('textarea.code')?.scrollIntoView({ behavior: 'smooth', block: 'center' }) }),
            spec.key === 'afya_platform' && v3ImportPanel({ form, onApplied: () => form.el.querySelector('input')?.scrollIntoView({ behavior: 'smooth', block: 'center' }) }),
            form.el, uploads, resultBox)),
        h('aside.cs-card.cs-card-pad.cs-guide',
          h('div.cs-row', logo(spec, { size: 'sm' }), h('b', 'How it works')),
          guide.map(([t, p]) => [h('h4', t), h('p', p)]),
          h('h4', 'Security'), h('p', 'Passwords, keys and tokens are encrypted at rest and never sent back to the browser.'),
          spec.roles.includes('destination') && [h('h4', 'Also a destination'), h('p', `${spec.label} can receive synced data — pick it as a destination when you create a sync.`)])),
      h('div.cs-wizard-foot', { style: { marginTop: '16px' } },
        h('button.cs-btn.cs-btn-ghost', { type: 'button', onclick: () => (ctx.query.get('kind') && !source ? navigate('/sources') : stepChoose()) }, icon('arrow-left'), 'Back'),
        h('span.cs-grow'), testBtn, nextBtn));
  }

  // ---------------------------------------------------------------- step 3
  function stepExtract() {
    step = 3; renderStepper();
    const sql = ['Databases', 'Warehouses'].includes(spec.category);
    const opts = { profile: true, sample_rows: 1000, exact_counts: true, schemas: [] };
    const schemaPicker = sql && testInfo?.schemas?.length > 1 ? h('div.cs-field', h('label', 'Schemas to extract'),
      h('div.cs-chips', testInfo.schemas.slice(0, 40).map((sc) => {
        const chip = h('button.cs-chip', { type: 'button', onclick: () => {
          const i = opts.schemas.indexOf(sc);
          if (i >= 0) opts.schemas.splice(i, 1); else opts.schemas.push(sc);
          chip.style.borderColor = opts.schemas.includes(sc) ? 'var(--cs-primary)' : 'transparent';
          chip.style.color = opts.schemas.includes(sc) ? 'var(--cs-primary)' : '';
        } }, icon('folder2'), sc);
        return chip;
      })), h('div.help', 'None selected = the schemas configured on the source (or every non-system schema).')) : null;
    const runBox = h('div');
    const startBtn = h('button.cs-btn.cs-btn-brand.cs-btn-lg', { type: 'button', onclick: () => start() }, icon('lightning-charge-fill'), 'Extract schema');

    async function start() {
      startBtn.disabled = true;
      try {
        const { job } = await api.extract(source.id, opts.schemas.length ? opts : { ...opts, schemas: undefined });
        const t = jobTracker(job.id, { title: `Extracting ${source.name}` });
        mount(runBox, h('div.cs-card.cs-card-pad', t.el));
        const final = await t.done;
        if (final.status === 'succeeded') {
          confetti();
          const st = final.result?.stats || {};
          const isTarget = spec.key === 'schema_file';
          mount(runBox, h('div.cs-card.cs-card-pad.cs-col.cs-gap-lg',
            h('div.cs-result.ok', h('span.cs-result-icon', icon('check-lg')), h('div', h('div.cs-result-title', 'Schema extracted'),
              h('div.cs-result-msg', `${n(st.tables)} tables · ${n(st.columns)} columns · ${n(st.relationships)} relationships (${n(st.declared_relationships)} declared, ${n((st.relationships || 0) - (st.declared_relationships || 0))} inferred)`))),
            kv([['Tables', n(st.tables)], ['Columns', n(st.columns)], ['Relationships', n(st.relationships)], ['Rows', n(st.rows)], ['Profiled', st.profiled ? 'yes' : 'no']]),
            h('div.cs-row.cs-wrap',
              h('a.cs-btn.cs-btn-lg', { href: href(`/sources/${source.id}`) }, icon('diagram-3'), 'Explore schema & relationships'),
              h('a.cs-btn.cs-btn-ai.cs-btn-lg', { href: href(`/mappings/new?${isTarget ? 'target' : 'source'}=${source.id}`) }, icon('stars'), isTarget ? 'Map a source onto this schema' : 'Map it with AI'))));
          window.dispatchEvent(new Event('cs:changed'));
        } else {
          startBtn.disabled = false;
        }
      } catch (e) { toastError(e, 'Extraction failed to start'); startBtn.disabled = false; }
    }

    mount(body, h('div.cs-wizard',
      h('div.cs-col.cs-gap-lg',
        h('div.cs-card',
          h('div.cs-card-head', logo(spec, { solid: true }), h('div.cs-grow', h('div.cs-card-title', source.name), h('div.cs-muted', { style: { fontSize: '12.5px' } }, `${spec.label} · saved`)),
            h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => stepConfigure() }, icon('pencil'), 'Edit settings')),
          h('div.cs-card-body.cs-col.cs-gap-lg',
            spec.key !== 'schema_file' && h('div.cs-form-grid',
              h('div.cs-field', h('label', 'Profile columns'),
                h('label.cs-switch', h('input', { type: 'checkbox', checked: true, onchange: (e) => { opts.profile = e.target.checked; } }), h('span.track'), h('span', 'Null rates, distinct values, samples, PII'))),
              sql && h('div.cs-field', h('label', 'Exact row counts'),
                h('label.cs-switch', h('input', { type: 'checkbox', checked: true, onchange: (e) => { opts.exact_counts = e.target.checked; } }), h('span.track'), h('span', 'COUNT(*) per table'))),
              h('div.cs-field.full', h('label', 'Rows sampled per table'),
                seg([200, 1000, 5000, 20000].map((v) => ({ value: v, label: n(v) })), 1000, (v) => { opts.sample_rows = v; }),
                h('div.help', 'More rows = better profiles (and better AI code maps), slower extraction.'))),
            schemaPicker,
            h('div.cs-row', startBtn, h('span.cs-muted', { style: { fontSize: '12.5px' } }, 'Runs in the background — you can leave this page.')))),
        runBox),
      h('aside.cs-card.cs-card-pad.cs-guide',
        h('div.cs-row', icon('diagram-3'), h('b', 'What dbatlas does')),
        h('ul', { style: { marginTop: '10px' } },
          h('li', 'Introspects schemas, tables, views, columns, types, keys and constraints'),
          h('li', 'Samples and profiles every table (in DuckDB for server databases)'),
          h('li', 'Infers relationships from naming conventions and key structure'),
          h('li', 'Flags likely PII columns'),
          h('li', 'Saves a versioned snapshot — re-extract later to see schema drift')))));
  }

  renderStepper();
  if (spec) stepConfigure(); else stepChoose();
}
