// New mapping: source → target (upload schema.json/DDL or pick a source) → scope & AI → generate.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { href, navigate } from '../../../ui/js/router.js';
import { toast, toastError } from '../../../ui/js/ui.js';
import { api } from '../api.js';
import { ago, dropzone, jobTracker, logo, n, seg } from '../kit.js';

export default async function render(ctx) {
  ctx.setTitle('New mapping');
  const [meta, { sources }] = await Promise.all([api.meta(), api.sources({}, { signal: ctx.signal })]);
  const specs = Object.fromEntries(meta.connectors.map((c) => [c.key, c]));
  const roleOk = (s, role) => (specs[s.kind]?.roles || []).includes(role);
  const state = {
    source: sources.find((s) => String(s.id) === ctx.query.get('source')) || null,
    target: sources.find((s) => String(s.id) === ctx.query.get('target')) || null,
    ai: meta.llm.available, samples: 'full', name: '', sourceTables: null, targetTables: null,
  };
  let step = state.source ? (state.target ? 3 : 2) : 1;
  const stepper = h('div.cs-steps');
  const body = h('div');
  const renderStepper = () => mount(stepper, [['Source', 1], ['Target schema', 2], ['Scope & AI', 3]].map(([label, i], idx) => [
    idx ? h('div.cs-step-line', { class: step > i - 1 && 'done' }) : null,
    h('div.cs-step', { class: [step === i && 'on', step > i && 'done'] }, h('span.cs-step-n', step > i ? icon('check-lg') : String(i)), label)]));

  const sourceCard = (s, selectedId, onPick) => h('button.cs-sel-card', { type: 'button', class: s.id === selectedId && 'on', onclick: () => onPick(s) },
    logo(specs[s.kind]), h('div.cs-grow',
      h('div', { style: { fontWeight: 650 } }, s.name),
      h('div.cs-muted', { style: { fontSize: '12px' } }, specs[s.kind]?.label || s.kind, ' · ',
        s.snapshot ? `${n(s.snapshot.stats.tables)} tables · ${n(s.snapshot.stats.columns)} columns · ${ago(s.snapshot.created_at)}` : 'not extracted yet — will be extracted first')),
    h('span.check', icon('check')));

  function foot(back, next, nextLabel = 'Continue', nextCls = 'cs-btn-brand') {
    return h('div.cs-wizard-foot', { style: { marginTop: '16px' } },
      back ? h('button.cs-btn.cs-btn-ghost', { type: 'button', onclick: back }, icon('arrow-left'), 'Back') : h('a.cs-btn.cs-btn-ghost', { href: href('/mappings') }, 'Cancel'),
      h('span.cs-grow'),
      next && h(`button.cs-btn.${nextCls}`, { type: 'button', onclick: next }, nextLabel, icon(nextCls === 'cs-btn-ai' ? 'stars' : 'arrow-right')));
  }

  // ---------------------------------------------------------------- step 1
  function stepSource() {
    step = 1; renderStepper();
    const list = sources.filter((s) => roleOk(s, 'source') && s.kind !== 'schema_file');
    mount(body,
      h('div.cs-wizard',
        h('div.cs-card', h('div.cs-card-head', h('div.cs-card-title', icon('box-arrow-right'), 'Where does the data come from?'), h('span.cs-grow'),
          h('a.cs-btn.cs-btn-sm', { href: href('/sources/new') }, icon('plus-lg'), 'Add a source')),
        h('div.cs-card-body.cs-col.cs-gap-sm', list.length ? list.map((s) => sourceCard(s, state.source?.id, (x) => { state.source = x; stepSource(); }))
          : h('div.cs-callout.cs-callout-info', icon('info-circle'), h('div', 'No sources yet. ', h('a', { href: href('/sources/new') }, 'Add one'), ' — a database, files or an API.')))),
        guide()),
      foot(null, () => { if (!state.source) { toast('Pick a source', 'warning'); return; } stepTarget(); }));
  }

  // ---------------------------------------------------------------- step 2
  function stepTarget() {
    step = 2; renderStepper();
    const others = sources.filter((s) => s.id !== state.source.id && roleOk(s, 'target'));
    const uploadBox = h('div');
    const drop = dropzone({ accept: ['.json', '.sql', '.ddl'], multiple: false, title: 'Drop your target schema.json, a .sql DDL file or a mappings.json',
      sub: 'JSON Schema, {"tables": …}, dbatlas.json, CREATE TABLE … (any dialect)',
      onFiles: async ([file]) => {
        try {
          const { source } = await api.createSource({ kind: 'schema_file', name: file.name.replace(/\.(json|sql|ddl)$/i, '') });
          await api.uploadFiles(source.id, [file]);
          await api.updateSource(source.id, { status: 'untested' });
          const { job } = await api.extract(source.id, {});
          const t = jobTracker(job.id, { title: `Parsing ${file.name}`, showLog: false });
          mount(uploadBox, h('div.cs-card.cs-card-pad', { style: { marginTop: '12px' } }, t.el));
          const final = await t.done;
          if (final.status !== 'succeeded') return;
          const fresh = (await api.source(source.id)).source;
          sources.push(fresh);
          state.target = fresh;
          toast(final.message, 'success', { title: 'Target schema ready' });
          stepTarget();
        } catch (e) { toastError(e, 'Could not read the schema'); }
      } });
    mount(body,
      h('div.cs-wizard',
        h('div.cs-col.cs-gap-lg',
          h('div.cs-card', h('div.cs-card-head', h('div.cs-card-title', icon('filetype-json'), 'Upload a target schema')),
            h('div.cs-card-body', drop, uploadBox)),
          h('div.cs-card', h('div.cs-card-head', h('div.cs-card-title', icon('database'), '…or use an existing source as the target'), h('span.cs-grow'),
            h('span.cs-muted', { style: { fontSize: '12.5px' } }, 'any database, files or schema file')),
          h('div.cs-card-body.cs-col.cs-gap-sm', others.length ? others.map((s) => sourceCard(s, state.target?.id, (x) => { state.target = x; stepTarget(); }))
            : h('div.cs-muted', 'No other sources yet.')))),
        guide()),
      foot(stepSource, () => { if (!state.target) { toast('Upload or pick a target schema', 'warning'); return; } stepScope(); }));
  }

  // ---------------------------------------------------------------- step 3
  async function stepScope() {
    step = 3; renderStepper();
    state.name = state.name || `${state.source.name} → ${state.target.name}`;
    const tablePicker = async (src, key, label) => {
      if (!src.snapshot) return h('div.cs-callout.cs-callout-info', icon('info-circle'), `${src.name} will be extracted first — all of its tables are included.`);
      const { snapshot } = await api.snapshot(src.snapshot.id);
      const tables = snapshot.data.tables.map((t) => t.key);
      const chosen = new Set(state[key] || tables);
      let q = '';
      const count = h('span.cs-badge');
      const listEl = h('div.cs-picklist');
      const draw = () => {
        mount(count, `${chosen.size}/${tables.length}`);
        mount(listEl, tables.filter((t) => !q || t.toLowerCase().includes(q)).map((t) => {
          const tt = snapshot.data.tables.find((x) => x.key === t);
          return h('label.cs-pick', h('input', { type: 'checkbox', checked: chosen.has(t), onchange: (e) => { e.target.checked ? chosen.add(t) : chosen.delete(t); state[key] = [...chosen]; mount(count, `${chosen.size}/${tables.length}`); } }),
            h('span.cs-mono.cs-grow', t), h('span.cs-muted', { style: { fontSize: '11.5px' } }, `${tt.columns.length} cols${tt.rows != null ? ` · ${n(tt.rows)} rows` : ''}`));
        }));
      };
      draw();
      return h('div.cs-field',
        h('label', label, count, h('span.cs-grow'),
          h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => { tables.forEach((t) => chosen.add(t)); state[key] = [...chosen]; draw(); } }, 'All'),
          h('button.cs-btn.cs-btn-sm.cs-btn-ghost', { type: 'button', onclick: () => { chosen.clear(); state[key] = []; draw(); } }, 'None')),
        h('div.cs-input-wrap', { style: { marginBottom: '6px' } }, icon('search'), h('input.cs-input.cs-input-sm', { placeholder: 'Filter tables…', oninput: (e) => { q = e.target.value.trim().toLowerCase(); draw(); } })),
        listEl);
    };
    const [srcPick, tgtPick] = await Promise.all([tablePicker(state.source, 'sourceTables', 'Source tables'), tablePicker(state.target, 'targetTables', 'Target tables')]);
    const samplesHelp = h('div.help');
    const SAMPLE_HELP = {
      full: 'Column names, types, stats and real sample values go to Groq — the best code maps and splits.',
      mask_pii: 'Values of columns dbatlas flags as PII are masked (digits → 9, letters → x); the rest is sent as is.',
      masked: 'Every sample value is masked — the AI sees shapes and value lists, not content.',
      none: 'Only names, types and keys — no values at all. Code maps get harder.',
    };
    mount(samplesHelp, SAMPLE_HELP[state.samples]);
    const genBtn = h('button.cs-btn.cs-btn-ai.cs-btn-lg', { type: 'button', onclick: generate }, icon('stars'), state.ai ? 'Generate mapping with AI' : 'Generate rule-based mapping');

    async function generate() {
      if (state.sourceTables && !state.sourceTables.length) { toast('Pick at least one source table', 'warning'); return; }
      if (state.targetTables && !state.targetTables.length) { toast('Pick at least one target table', 'warning'); return; }
      genBtn.disabled = true;
      try {
        const { mapping } = await api.createMapping({ source: state.source.id, target: state.target.id, name: state.name, ai: state.ai, samples: state.samples,
          source_tables: state.sourceTables || [], target_tables: state.targetTables || [] });
        window.dispatchEvent(new Event('cs:changed'));
        navigate(`/mappings/${mapping.id}`);
      } catch (e) { toastError(e, 'Could not start'); genBtn.disabled = false; }
    }

    mount(body,
      h('div.cs-wizard',
        h('div.cs-col.cs-gap-lg',
          h('div.cs-card', h('div.cs-card-body.cs-col.cs-gap-lg',
            h('div.cs-row.cs-wrap', { style: { gap: '12px' } },
              h('span.cs-flowchip', logo(specs[state.source.kind], { size: 'xs' }), state.source.name),
              h('span.cs-flow-arrow', h('span.dash'), icon('chevron-right')),
              h('span.cs-flowchip', logo(specs[state.target.kind], { size: 'xs' }), state.target.name)),
            h('div.cs-field', h('label', 'Mapping name'), h('input.cs-input', { value: state.name, oninput: (e) => { state.name = e.target.value; } })),
            h('div.cs-form-grid', srcPick, tgtPick))),
          h('div.cs-card', h('div.cs-card-head', h('div.cs-card-title', icon('stars'), 'AI settings'), h('span.cs-grow'),
            meta.llm.available ? h('span.cs-badge.cs-badge-ai', icon('cpu'), `Groq · ${meta.llm.model}`) : h('span.cs-badge.cs-badge-warn', 'GROQ_API_KEY not set')),
          h('div.cs-card-body.cs-col.cs-gap-lg',
            h('label.cs-switch', h('input', { type: 'checkbox', checked: state.ai, disabled: !meta.llm.available, onchange: (e) => { state.ai = e.target.checked; mount(genBtn, icon('stars'), state.ai ? 'Generate mapping with AI' : 'Generate rule-based mapping'); } }),
              h('span.track'), h('span', 'Use AI to match tables and write column rules')),
            h('div.cs-field', h('label', 'What the AI may see'),
              seg([{ value: 'full', label: 'Samples', icon: 'eye' }, { value: 'mask_pii', label: 'Mask PII', icon: 'incognito' },
                { value: 'masked', label: 'Mask all', icon: 'eye-slash' }, { value: 'none', label: 'Structure only', icon: 'diagram-2' }],
              state.samples, (v) => { state.samples = v; mount(samplesHelp, SAMPLE_HELP[v]); }),
              samplesHelp)))),
        guide(true)),
      h('div.cs-wizard-foot', { style: { marginTop: '16px' } },
        h('button.cs-btn.cs-btn-ghost', { type: 'button', onclick: stepTarget }, icon('arrow-left'), 'Back'), h('span.cs-grow'), genBtn));
  }

  function guide(final = false) {
    return h('aside.cs-card.cs-card-pad.cs-guide',
      h('div.cs-row', h('span.cs-pill-ai', icon('stars'), 'HOW IT WORKS')),
      h('h4', '1 · Tables'), h('p', 'Every target table is scored against every source table (names + column overlap); Groq makes the final call and explains it.'),
      h('h4', '2 · Columns'), h('p', 'For each pair, exact and synonym matches are pre-filled, then Groq maps every remaining column with a rule: copy, clean-up steps, a code map, a lookup through a foreign key, a split, or a constant.'),
      h('h4', '3 · Validation'), h('p', 'Suggestions that reference missing columns or invalid rules are rejected before you see them.'),
      final && [h('h4', '4 · Review'), h('p', 'You approve, fix by drag & drop, preview each rule on real sample values, and export mappings.json — or sync directly.')]);
  }

  mount(ctx.el,
    h('div.cs-head',
      h('div', h('div.cs-crumbs', h('a', { href: href('/mappings') }, 'Mappings'), icon('chevron-right'), 'New'),
        h('h1', 'New mapping', h('span.cs-pill-ai', icon('stars'), 'AI')),
        h('div.cs-head-sub', 'Map a source schema onto a target schema. Groq proposes; you decide.')),
      stepper),
    body);
  renderStepper();
  if (step === 1) stepSource(); else if (step === 2) stepTarget(); else stepScope();
}
