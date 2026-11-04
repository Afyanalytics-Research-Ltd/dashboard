// Procedures: browse stored procedures, run them with a typed form, see results.
// Only the cost & performance toolkit is callable in read mode; anything else
// needs write mode (and only if the deployment allows writes).

import { h, icon, mount, empty, errorView, skeleton, copyButton, popover, chip, liveTimer, banner, toast, kv } from '../lib/ui.js';
import { api, waitForQuery } from '../lib/api.js';
import { state } from '../lib/store.js';
import * as fmt from '../lib/fmt.js';
import * as x from '../lib/explorercore.js';
import { resultGrid, highlightSql } from './data.js';
import { openQuery } from '../lib/query.js';
import { prefs, debounce } from '../../../ui/js/dom.js';
import { href } from '../../../ui/js/router.js';

const RECENT_KEY = 'ws:procedures:recent';
// Mirrors installer.EXPECTED_PROCEDURES: the allow-list when the optimizer catalog is unreachable.
const TOOLKIT_PROCS = ['SP_DETECT_LONG_COMPILATION', 'SP_DETECT_LONG_EXECUTION', 'SP_DETECT_POOR_PRUNING', 'SP_DETECT_SPILL_AND_OOM',
  'SP_DETECT_QUEUEING', 'SP_DETECT_BLOCKED_TRANSACTIONS', 'SP_DETECT_STORAGE_OVERHEAD', 'SP_DETECT_WAREHOUSE_WASTE',
  'SP_DETECT_WAREHOUSE_SETTINGS', 'SP_DETECT_SELECT_STAR', 'SP_DETECT_REPEAT_OFFENDERS', 'SP_DETECT_JOIN_EXPLOSION',
  'SP_RUN_ALL_DETECTORS', 'SP_RUN_FULL_OPTIMIZATION_ANALYSIS', 'SP_FIND_HIGH_TIME_TRAVEL_TABLES', 'SP_MONITOR_CLUSTERING_COSTS',
  'SP_CLUSTERING_REPORT', 'SP_ESTIMATE_SOS_COST', 'SP_WS_WAREHOUSES', 'SP_WS_RESOURCE_MONITORS', 'SP_WS_QUERY_PROFILE', 'SP_WS_DETECTOR_SCHEDULE'];

export default async function render(ctx) {
  ctx.setTitle('Procedures');
  const toolkit = state.session?.toolkit?.toolkit || window.APP_UI?.toolkit || 'HOSPITALS.ANALYTICS';
  const [tkDb, tkSchema] = toolkit.split('.');
  const sel = {
    database: ctx.query.get('database') || tkDb,
    schema: ctx.query.get('schema') || (ctx.query.get('database') ? '' : tkSchema),
    name: ctx.query.get('name') || '',
  };
  const allowWrites = !!(state.session?.allow_writes ?? window.APP_UI?.allowWrites);
  let allow = new Set(TOOLKIT_PROCS.map((p) => `${toolkit}.${p}`.toUpperCase()));
  const toolkitMeta = new Map();
  let procs = [];            // [{database, schema, name, params, argument_signature, data_type, language, comment}]
  let schemas = [];
  let q = '';
  let onlyAllowed = false;
  let loadToken = 0;
  let loading = true;        // until the first schema scan finishes, never show the empty state

  const headEl = h('div.ws-card.ws-card-pad.ws-mb');
  const listEl = h('div.wsx-proc-list');
  const runEl = h('div.wsx-proc-run');
  const recentEl = h('div');
  mount(ctx.el, headEl, h('div.wsx-proc-layout', h('div.ws-col', { style: { gap: '12px', minWidth: 0 } }, listEl, recentEl), runEl));

  api.optimizer(false, { signal: ctx.signal }).then((cat) => {
    const all = [...(cat.detectors || []), ...(cat.reports || [])];
    for (const p of all) { allow.add(String(p.fq).toUpperCase()); toolkitMeta.set(String(p.fq).toUpperCase(), p); }
    renderList();
    if (current) renderRun(current);
  }).catch(() => { /* static allow-list is enough */ });

  const syncUrl = () => ctx.setQuery({ database: sel.database || null, schema: sel.schema || null, name: sel.name || null });

  function renderHead() {
    mount(headEl, h('div.ws-row.ws-wrap', { style: { gap: '8px' } },
      h('button.ws-pill', { type: 'button', onclick: pickDatabase }, icon('database'), h('span.ws-pill-k', 'Database'), h('b', sel.database || 'pick'), icon('chevron-down')),
      h('button.ws-pill', { type: 'button', onclick: pickSchema }, icon('folder2'), h('span.ws-pill-k', 'Schema'), h('b', sel.schema || 'all schemas'), icon('chevron-down')),
      h('div.ws-search', { style: { flex: '1 1 240px' } }, icon('search'), h('input.ws-input', { type: 'search', placeholder: 'Search procedures, comments, types…', value: q,
        oninput: debounce((e) => { q = e.target.value.toLowerCase(); renderList(); }, 150) })),
      chip('Toolkit only', { on: onlyAllowed, onClick: () => { onlyAllowed = !onlyAllowed; renderHead(); renderList(); } }),
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', onclick: () => load() }, icon('arrow-clockwise'), 'Refresh')),
    h('div.ws-muted', { style: { fontSize: '12px', marginTop: '8px' } },
      icon('shield-lock', 'me-1'), `Read-only mode runs only the cost & performance toolkit in ${toolkit}. `,
      allowWrites ? 'Other procedures can be run after unlocking write mode for that call.' : 'Other procedures are locked on this deployment (WAREHOUSE_STUDIO_ALLOW_WRITES is off).'));
  }

  async function pickDatabase(e) {
    const anchor = e.currentTarget;
    try {
      const { databases } = await api.databases({ role: state.context.role });
      popover(anchor, { search: true, width: 320, items: databases.map((d) => ({ label: d.name, value: d.name, icon: 'database', on: d.name === sel.database })),
        onPick: (i) => { sel.database = i.value; sel.schema = ''; sel.name = ''; syncUrl(); load(); } });
    } catch (err) { toast(err.message, 'error'); }
  }

  function pickSchema(e) {
    popover(e.currentTarget, { search: schemas.length > 8, width: 320,
      items: [{ label: 'All schemas', value: '', icon: 'asterisk', on: !sel.schema }, ...schemas.map((s) => ({ label: s.name, value: s.name, icon: 'folder2', on: s.name === sel.schema }))],
      onPick: (i) => { sel.schema = i.value; syncUrl(); load(); } });
  }

  async function load() {
    const token = ++loadToken;
    loading = true;
    renderHead();
    mount(listEl, skeleton(6, 64));
    procs = [];
    try {
      const sres = await api.schemas(sel.database, { role: state.context.role }, { signal: ctx.signal });
      if (token !== loadToken) return;
      schemas = sres.schemas || [];
      const targets = sel.schema ? schemas.filter((s) => s.name === sel.schema) : schemas;
      if (!targets.length) { loading = false; renderHead(); renderList(); return; }
      let done = 0;
      const failures = [];
      const progress = h('div.ws-muted', { style: { fontSize: '12px' } });
      // Fetch schemas with limited concurrency so big databases stream in.
      const queue = [...targets];
      const worker = async () => {
        while (queue.length && token === loadToken) {
          const s = queue.shift();
          try {
            const res = await api.objects(sel.database, s.name, { role: state.context.role }, { signal: ctx.signal });
            const ps = res.objects?.PROCEDURE || [];
            procs.push(...ps.map((p) => ({ ...p, database: sel.database, schema: s.name, params: p.params?.length || p.argument_signature === '()' ? p.params || [] : x.parseSignature(p.argument_signature) })));
          } catch (err) {
            if (err.name === 'AbortError') return;
            failures.push(`${s.name}: ${err.message}`);
          }
          done += 1;
          if (token === loadToken) { mount(progress, targets.length > 1 && done < targets.length ? `Scanning schemas… ${done}/${targets.length}` : ''); renderList(progress); }
        }
      };
      await Promise.all([...Array(Math.min(4, targets.length))].map(worker));
      if (token !== loadToken) return;
      loading = false;
      renderHead();
      renderList(failures.length ? banner('error', `Could not list procedures in ${failures.length} schema${failures.length > 1 ? 's' : ''}`,
        failures.slice(0, 3).join(' · ')) : undefined);
      if (sel.name) {
        const p = procs.find((pp) => pp.name === sel.name && (!sel.schema || pp.schema === sel.schema));
        if (p) renderRun(p);
      }
    } catch (err) {
      loading = false;
      if (err.name !== 'AbortError') mount(listEl, errorView(err, load));
    }
  }

  const fqOf = (p) => `${p.database}.${p.schema}.${p.name}`.toUpperCase();
  const isAllowed = (p) => allow.has(fqOf(p));

  function renderList(progress) {
    if (loading && !procs.length) { mount(listEl, progress || null, skeleton(6, 64)); return; }
    const shown = procs.filter((p) => (!onlyAllowed || isAllowed(p))
      && (!q || `${p.schema} ${p.name} ${p.comment || ''} ${p.argument_signature || ''} ${p.data_type || ''}`.toLowerCase().includes(q)))
      .sort((a, b) => Number(isAllowed(b)) - Number(isAllowed(a)) || a.schema.localeCompare(b.schema) || a.name.localeCompare(b.name));
    mount(listEl, progress || null,
      shown.length ? shown.map((p) => procCard(p)) : empty({ icon: 'braces-asterisk', title: procs.length ? 'No procedures match' : 'No procedures visible',
        text: procs.length ? 'Clear the search or the toolkit filter.' : 'This role cannot see any stored procedures here. Pick another database or schema.' }));
  }

  let current = null;
  function procCard(p) {
    const allowed = isAllowed(p);
    const meta = toolkitMeta.get(fqOf(p));
    return h(`button.wsx-proc${current && fqOf(current) === fqOf(p) && current.argument_signature === p.argument_signature ? '.on' : ''}`, { type: 'button', onclick: () => renderRun(p) },
      h('div.ws-row', { style: { gap: '8px', minWidth: 0 } },
        icon(meta?.icon || (allowed ? 'lightning-charge' : 'lock'), allowed ? 'wsx-proc-ok' : 'wsx-proc-lock'),
        h('b.ws-ellipsis', meta?.title || p.name),
        allowed ? h('span.ws-badge.ws-sev-INFO.wsx-mini', 'toolkit') : h('span.ws-badge.ws-sev-LOW.wsx-mini', allowWrites ? 'write mode' : 'locked')),
      h('div.ws-mono.wsx-proc-sig', `${p.schema}.${p.name}${p.argument_signature || '()'}`),
      (meta?.summary || p.comment) && h('div.ws-muted.wsx-proc-desc', meta?.summary || p.comment),
      h('div.ws-row.ws-wrap', { style: { gap: '6px' } }, p.data_type && h('span.ws-tag', `→ ${fmt.truncate(p.data_type, 40)}`), p.language && h('span.ws-tag', p.language.toLowerCase())));
  }

  // ---------------------------------------------------------------- run panel

  function renderRun(p) {
    current = p;
    sel.name = p.name;
    syncUrl();
    renderList();
    const allowed = isAllowed(p);
    const meta = toolkitMeta.get(fqOf(p));
    const params = p.params || [];
    let writeMode = false;
    const inputs = params.map((param) => inputFor(param, meta?.params?.find((mp) => mp.name.toUpperCase() === String(param.name).toUpperCase())));
    const errorsEl = h('div');
    const outEl = h('div');
    const runBtn = h('button.ws-btn.ws-btn-primary', { type: 'submit' }, icon('play-fill'), 'Run');
    const lockEl = allowed ? null : allowWrites
      ? h('label.wsx-unlock', h('input', { type: 'checkbox', onchange: (e) => { writeMode = e.target.checked; runBtn.disabled = !writeMode; } }),
        icon('unlock', 'ms-1'), ' Unlock write mode for this call — this procedure is not on the read-only allow-list and may change data.')
      : banner('warn', 'Locked', 'Only the cost & performance toolkit can be called from the Studio on this deployment. Ask an administrator to run it, or enable WAREHOUSE_STUDIO_ALLOW_WRITES.');
    if (!allowed) runBtn.disabled = true;
    const form = h('form.ws-col', { style: { gap: '12px' }, onsubmit: (e) => { e.preventDefault(); run(); } },
      params.length ? inputs.map((i) => i.el) : h('div.ws-muted', 'This procedure takes no arguments.'),
      errorsEl, lockEl,
      h('div.ws-row', runBtn, h('button.ws-btn.ws-btn-ghost', { type: 'button', onclick: () => { inputs.forEach((i) => i.reset()); mount(errorsEl); } }, 'Reset'),
        copyButton(() => callSql(), 'SQL')));
    mount(runEl, h('div.ws-card.ws-card-pad',
      h('div.ws-row.ws-wrap', { style: { gap: '8px', marginBottom: '4px' } }, h('span.ws-badge.ws-sev-INFO', icon('braces-asterisk'), 'procedure'),
        h('span.ws-muted.ws-mono', `${p.database}.${p.schema}`)),
      h('h2.wsx-title', meta?.title || p.name),
      meta?.detects && h('p.ws-text-2', { style: { margin: '0 0 6px' } }, meta.detects),
      meta?.fix && h('div.ws-insight', { style: { marginBottom: '10px' } }, icon('lightbulb'), h('p', { style: { gridColumn: '2' } }, meta.fix)),
      kv([['Signature', h('span.ws-mono', `${p.name}${p.argument_signature || '()'}`)], ['Returns', h('span.ws-mono', p.data_type || '—')],
        ['Language', p.language || '—'], p.comment && ['Comment', p.comment]]),
      h('div.ws-divider'), form, h('div.ws-mt'), outEl));

    function callSql() {
      const v = x.validateParams(params, inputs.map((i) => i.value()));
      const lit = (val, param) => (val === null ? 'NULL' : x.inputKind(param.type) === 'json' ? `PARSE_JSON('${String(val).replace(/'/g, "''")}')`
        : typeof val === 'string' ? `'${val.replace(/'/g, "''")}'` : String(val));
      return `CALL ${x.quoteIdent(p.database, p.schema, p.name)}(${v.values.map((val, i) => lit(val, params[i])).join(', ')});`;
    }

    async function run() {
      const v = x.validateParams(params, inputs.map((i) => i.value()));
      if (!v.ok) { mount(errorsEl, banner('error', 'Check the arguments', null, h('ul', { style: { margin: 0 } }, v.errors.map((e) => h('li', e))))); return; }
      mount(errorsEl);
      const ac = new AbortController();
      ctx.cleanup(() => ac.abort());
      const timer = liveTimer();
      let qid = null;
      const statusEl = h('span.ws-live', h('span.ws-spinner'), 'Submitting…');
      runBtn.disabled = true;
      mount(outEl, h('div.ws-card.ws-card-pad.wsx-running', h('div.ws-row.ws-between', h('div.ws-row', statusEl, timer.el),
        h('button.ws-btn.ws-btn-sm.ws-btn-danger', { type: 'button', onclick: async () => { ac.abort(); if (qid) { try { await api.cancel(qid); } catch { /* already done */ } } } }, icon('stop-fill'), 'Cancel'))));
      const t0 = Date.now();
      let ok = false;
      try {
        const res = await api.callProcedure({ database: p.database, schema: p.schema, name: p.name, params, values: v.values, context: state.context, mode: writeMode ? 'write' : 'read' });
        qid = res.query_id;
        const st = await waitForQuery(qid, { signal: ac.signal, onTick: (s) => mount(statusEl, h('span.ws-spinner'), String(s.status).toLowerCase().replace(/_/g, ' ')) });
        timer.stop();
        if (!st.ok) {
          mount(outEl, failure(st.error?.message || `Query ${String(st.status).toLowerCase()}`, qid, Date.now() - t0));
        } else {
          const result = await api.results(qid, 0, 1000, { signal: ac.signal });
          ok = true;
          mount(outEl, success(result, qid, Date.now() - t0));
        }
      } catch (err) {
        timer.stop();
        if (err.name === 'AbortError') { mount(outEl, banner('info', 'Cancelled', 'The call was cancelled.')); } else if (err.code === 'call') {
          mount(outEl, banner('warn', 'Not on the allow-list', err.message));
        } else if (err.code === 'query') {
          mount(outEl, failure(err.message, err.data?.query_id, Date.now() - t0));
        } else mount(outEl, errorView(err));
      } finally {
        runBtn.disabled = !allowed && !writeMode;
        remember({ fq: fqOf(p), database: p.database, schema: p.schema, name: p.name, values: inputs.map((i) => i.value()), ok, at: new Date().toISOString(), ms: Date.now() - t0, query_id: qid });
      }
    }
  }

  function success(result, qid, elapsed) {
    const cols = result.columns || [];
    const rows = result.rows || [];
    const footer = h('div.ws-row.ws-wrap', { style: { gap: '8px', marginTop: '10px', fontSize: '12px' } }, h('span.ws-muted', `${fmt.ms(elapsed)} · `),
      h('a', { href: '#', onclick: (e) => { e.preventDefault(); openQuery(qid); } }, 'Inspect the call'),
      rows.length > 1 && h('a', { href: api.exportUrl(qid, 'csv') }, 'Download CSV'));
    if (cols.length === 1 && rows.length === 1) {
      const value = rows[0][0];
      const outcome = x.detectorOutcome(value);
      if (outcome) {
        const label = outcome.kind === 'batch' ? 'Detector batch finished' : `${fmt.detectorLabel(outcome.detector)} finished`;
        return h('div.ws-card.ws-card-pad.wsx-success',
          h('div.ws-row', { style: { gap: '12px' } }, h('div.wsx-success-icon', icon('check2-circle')),
            h('div.ws-col', { style: { gap: '2px' } }, h('b', label),
              h('span', h('b.ws-mono', { style: { fontSize: '22px' } }, fmt.fmtInt(outcome.findings)), ` new finding${outcome.findings === 1 ? '' : 's'}`,
                outcome.failed ? h('span.ws-muted', ` · ${outcome.failed} detector(s) failed`) : null)),
            h('a.ws-btn.ws-btn-primary.ws-btn-sm', { style: { marginLeft: 'auto' }, href: href('/optimizer', outcome.kind === 'detector' ? { detector: outcome.detector } : {}) }, icon('lightning-charge'), 'See findings')),
          h('pre.ws-code', { style: { marginTop: '12px' } }, JSON.stringify(outcome.raw, null, 2)), footer);
      }
      const plain = value && typeof value === 'object' && '$json' in value ? value.$json : value;
      let parsed = plain;
      if (typeof plain === 'string') { try { parsed = JSON.parse(plain); } catch { parsed = plain; } }
      return h('div.ws-card.ws-card-pad.wsx-success', h('div.ws-row', { style: { gap: '10px', marginBottom: '10px' } }, h('div.wsx-success-icon', icon('check2-circle')), h('b', 'Returned')),
        typeof parsed === 'object' && parsed !== null
          ? h('div', h('div.ws-row', { style: { marginBottom: '6px' } }, copyButton(JSON.stringify(parsed, null, 2), 'Copy JSON')), h('pre.ws-code', JSON.stringify(parsed, null, 2)))
          : h('pre.ws-code', String(parsed ?? 'NULL')), footer);
    }
    return h('div.ws-card.ws-card-pad.wsx-success', h('div.ws-row', { style: { gap: '10px', marginBottom: '10px' } }, h('div.wsx-success-icon', icon('table')),
      h('b', `${fmt.plural(result.total ?? rows.length, 'row')}`), result.truncated && h('span.ws-tag', 'truncated')), resultGrid(cols, rows, { maxHeight: '50vh' }), footer);
  }

  function failure(message, qid, elapsed) {
    return banner('error', 'The procedure failed', null, h('pre.ws-code', message),
      h('div.ws-row', { style: { gap: '8px', fontSize: '12px' } }, h('span.ws-muted', fmt.ms(elapsed)), qid && h('a', { href: '#', onclick: (e) => { e.preventDefault(); openQuery(qid); } }, 'Inspect the call')));
  }

  // ---------------------------------------------------------------- recent calls

  function remember(entry) {
    const list = [entry, ...prefs.get(RECENT_KEY, []).filter((r) => !(r.fq === entry.fq && JSON.stringify(r.values) === JSON.stringify(entry.values)))].slice(0, 12);
    prefs.set(RECENT_KEY, list);
    renderRecent();
  }

  function renderRecent() {
    const list = prefs.get(RECENT_KEY, []);
    if (!list.length) { mount(recentEl); return; }
    mount(recentEl, h('div.ws-card.ws-card-pad', h('div.ws-row.ws-between', { style: { marginBottom: '8px' } }, h('h3.ws-card-title', icon('clock-history'), 'Recent calls'),
      h('button.ws-btn.ws-btn-sm.ws-btn-ghost', { type: 'button', onclick: () => { prefs.set(RECENT_KEY, []); renderRecent(); } }, 'Clear')),
    h('div.ws-col', { style: { gap: '4px' } }, list.map((r) => h('button.wsx-recent', { type: 'button', title: 'Open with these arguments',
      onclick: () => {
        const p = procs.find((pp) => fqOf(pp) === r.fq);
        if (p) { renderRun(p); setTimeout(() => fillValues(r.values), 0); } else { sel.database = r.database; sel.schema = r.schema; sel.name = r.name; syncUrl(); load(); }
      } },
    icon(r.ok ? 'check-circle-fill' : 'x-octagon-fill', r.ok ? 'wsx-st-ok' : 'wsx-st-fail'),
    h('span.ws-ellipsis.ws-mono', `${r.name}(${(r.values || []).map((v) => (v === '' || v == null ? '·' : fmt.truncate(String(v), 12))).join(', ')})`),
    h('span.ws-muted', { style: { marginLeft: 'auto', fontSize: '11px' } }, fmt.relTime(r.at)))))));
  }

  function fillValues(values) {
    const fields = runEl.querySelectorAll('[data-param]');
    fields.forEach((f, i) => {
      const v = values?.[i];
      if (f.type === 'checkbox') f.checked = v === true || v === 'true';
      else f.value = v ?? '';
    });
  }

  renderHead();
  renderRecent();
  mount(runEl, h('div.ws-card.ws-card-pad', empty({ icon: 'braces-asterisk', title: 'Pick a procedure', text: 'Toolkit procedures (detectors and reports) run straight away; their results land in the Optimizer.' })));
  await load();
}

/** Typed input for one parameter.  Returns {el, value(), reset()}. */
function inputFor(param, meta) {
  const kind = x.inputKind(param.type);
  const label = meta?.label || param.name;
  const def = meta?.default;
  const help = [meta?.help, param.has_default ? 'optional' : null, meta?.min != null && meta?.max != null ? `${meta.min}–${meta.max}` : null].filter(Boolean).join(' · ');
  let input;
  if (kind === 'boolean') {
    input = h('input', { type: 'checkbox', checked: def === true, dataset: { param: param.name } });
  } else if (kind === 'json') {
    input = h('textarea.ws-textarea', { rows: 4, placeholder: param.type.startsWith('ARRAY') ? '["a", "b"]' : '{"key": "value"}', dataset: { param: param.name },
      oninput: () => { try { if (input.value.trim()) JSON.parse(input.value); input.classList.remove('wsx-invalid'); } catch { input.classList.add('wsx-invalid'); } } });
  } else {
    input = h('input.ws-input', { type: kind === 'int' || kind === 'float' ? 'number' : 'text', step: kind === 'float' ? 'any' : kind === 'int' ? '1' : undefined,
      min: meta?.min ?? undefined, max: meta?.max ?? undefined, value: def ?? '', placeholder: def != null ? String(def) : param.has_default ? 'default' : '', dataset: { param: param.name } });
  }
  const el = h('div.ws-field', h('label', label, h('span.ws-tag', { style: { marginLeft: '8px' } }, param.type), !param.has_default && h('span.wsx-req', ' *')),
    kind === 'boolean' ? h('label.ws-row', { style: { gap: '8px' } }, input, h('span.ws-text-2', 'true')) : input, help && h('div.ws-help', help));
  return {
    el,
    value: () => (kind === 'boolean' ? input.checked : input.value),
    reset: () => { if (kind === 'boolean') input.checked = def === true; else input.value = def ?? ''; input.classList?.remove('wsx-invalid'); },
  };
}

export { highlightSql };
