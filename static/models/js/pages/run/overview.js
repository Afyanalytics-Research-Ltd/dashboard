// Run page — Overview: facts, editable group/job/tags/description, notes as a decision log,
// searchable config and summary with JSON/YAML export.

import { h, icon, mount, debounce } from '../../../../ui/js/dom.js';
import { card, kv, timeAgo, copyButton, toast, toastError, emptyState, skeleton, tabs } from '../../../../ui/js/ui.js';
import { fmtDate, fmtDuration } from '../../../../ui/js/format.js';
import { loadMarkdown } from '../../../../ui/js/lib.js';
import { fetchFile, config } from '../../core/api.js';
import { mutations } from '../../core/data.js';
import { urls } from '../../core/urls.js';
import { fmtMetric, runStateBadge } from '../../core/wb.js';
import { downloadText } from '../../core/charts.js';
import { fileIndex, inlineText, tagEditor, toYaml, unflatten, fmtBytesShort, highlight } from './common.js';

export default async function overviewTab(el, ctx, run) {
  const metaEl = h('div', skeleton(3));
  mount(el,
    h('div.msr-overview',
      h('div.msr-ov-main',
        card({ title: 'About this run', icon: 'info-circle', body: aboutBody(run, ctx) }),
        notesCard(run, ctx),
        kvTableCard({
          title: 'Config', icon: 'sliders', data: run.config, file: `${run.name}-config`,
          exportYaml: () => toYaml(unflatten(run.config)), emptyText: 'This run logged no config.',
        }),
        kvTableCard({
          title: 'Summary', icon: 'list-ol', data: run.summary, file: `${run.name}-summary`,
          emptyText: 'No summary metrics yet.', metric: true,
        })),
      h('aside.msr-ov-side',
        card({ title: 'Environment', icon: 'cpu', body: metaEl }))));
  loadEnvironment(metaEl, ctx, run).catch((err) => mount(metaEl, h('div.text-muted.small', `Metadata unavailable: ${err.message}`)));
}

function aboutBody(run, ctx) {
  const { entity, project } = ctx.params;
  const save = (field) => async (v) => {
    await mutations.updateRun(run.id, { [field]: v });
    run[field === 'groupName' ? 'group' : field] = v;
    toast('Saved', 'success', { timeout: 1500 });
  };
  const started = run.createdAt;
  const ended = run.state === 'running' ? null : run.heartbeatAt;
  return kv([
    ['State', h('span.d-inline-flex.align-items-center.gap-2', runStateBadge(run.state), run.exitcode != null && run.exitcode !== 0 && h('span.ms-pill.ms-pill-bad', `exit ${run.exitcode}`))],
    ['Run path', h('span.d-inline-flex.align-items-center.gap-2', h('code.ms-mono', `${entity}/${project}/${run.name}`), copyButton(`${entity}/${project}/${run.name}`))],
    ['Started', h('span', fmtDate(started), ' · ', timeAgo(started))],
    ['Last heartbeat', run.heartbeatAt ? h('span', fmtDate(run.heartbeatAt), ' · ', timeAgo(run.heartbeatAt)) : '—'],
    ['Runtime', fmtDuration(run.runtime)],
    ['Steps logged', run.historyLineCount != null ? Number(run.historyLineCount).toLocaleString() : '—'],
    ['User', run.user ? h('span.d-inline-flex.align-items-center.gap-2', run.user.photoUrl && h('img.msr-avatar', { src: run.user.photoUrl, alt: '' }), run.user.username) : '—'],
    ['Host', run.host ? h('code.ms-mono', run.host) : '—'],
    ['Group', inlineText(run.group, save('groupName'), { placeholder: 'No group — click to set' })],
    ['Job type', inlineText(run.jobType, save('jobType'), { placeholder: 'No job type — click to set' })],
    ['Sweep', run.sweep ? h('a', { href: urls.sweep(entity, project, run.sweep.name) }, icon('shuffle', 'me-1'), run.sweep.displayName || run.sweep.name) : '—'],
    ['Storage', `${fmtBytesShort(run.storageBytes)} · ${run.fileCount ?? 0} files`],
    ['Commit', run.commit ? h('span.d-inline-flex.align-items-center.gap-2', h('code.ms-mono', run.commit.slice(0, 12)), copyButton(run.commit)) : '—'],
    ['Repository', run.github ? h('a', { href: run.github, target: '_blank', rel: 'noopener' }, run.github.replace(/^https?:\/\//, '')) : '—'],
    ['Tags', tagEditor(run.tags, async (tags) => { await mutations.updateRun(run.id, { tags }); run.tags = tags; }, { suggestions: ctx.tagSuggestions || [] }), { wide: true }],
    // NB: W&B's run "description" field is the legacy display name, so it is not shown separately.
    ended && started && ['Ended', fmtDate(ended)],
  ].filter(Boolean));
}

// ------------------------------------------------------------------ notes / decision log

function notesCard(run, ctx) {
  const body = h('div');
  let notes = run.notes || '';
  let mode = 'preview';
  let render;
  const toolbar = h('div.msr-notes-tools');
  const draw = async () => {
    mount(toolbar,
      tabs([{ id: 'preview', label: 'Preview', icon: 'eye' }, { id: 'edit', label: 'Edit', icon: 'pencil' }], mode, (id) => { mode = id; draw(); }),
      h('button.btn.btn-sm.btn-outline-secondary', {
        type: 'button', title: 'Add a dated entry to the decision log',
        onclick: () => {
          const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
          notes = `${notes.trim()}\n\n### ${stamp} — ${config.user || 'me'}\n- Decision: \n- Why: \n- Next: \n`.trimStart();
          mode = 'edit';
          draw();
        },
      }, icon('journal-plus', 'me-1'), 'Log a decision'));
    if (mode === 'edit') {
      const ta = h('textarea.form-control.msr-notes-ta', { rows: 10, placeholder: 'Why did this run happen? What did you learn? What will you try next? (Markdown)' }, notes);
      const saveBtn = h('button.btn.btn-sm.btn-primary', {
        type: 'button',
        onclick: async () => {
          saveBtn.disabled = true;
          try {
            await mutations.updateRun(run.id, { notes: ta.value });
            notes = ta.value;
            run.notes = notes;
            toast('Notes saved to W&B', 'success', { timeout: 1800 });
            mode = 'preview';
            draw();
          } catch (err) { toastError(err, 'Could not save notes'); saveBtn.disabled = false; }
        },
      }, icon('cloud-check', 'me-1'), 'Save notes');
      ta.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); saveBtn.click(); } });
      ta.addEventListener('input', () => { notes = ta.value; });
      mount(body, ta, h('div.msr-notes-foot', h('span.small.text-muted', 'Markdown · Ctrl+S to save · stored as the run\'s W&B notes'),
        h('div.d-flex.gap-2', h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => { notes = run.notes || ''; mode = 'preview'; draw(); } }, 'Cancel'), saveBtn)));
      setTimeout(() => { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }, 0);
    } else if (!notes.trim()) {
      mount(body, emptyState({ icon: 'journal-text', title: 'No notes yet', message: 'Record why you ran this and what you decided — no more spreadsheets on the side.' }));
    } else {
      render ||= await loadMarkdown();
      const md = h('div.af-markdown.msr-notes-md');
      md.innerHTML = render(notes); // sanitized by DOMPurify inside loadMarkdown
      mount(body, md);
    }
  };
  draw();
  return card({ title: 'Notes & decision log', icon: 'journal-text', actions: toolbar, body });
}

// ------------------------------------------------------------------ config / summary tables

function kvTableCard({ title, icon: ic, data, file, exportYaml, emptyText, metric }) {
  const entries = Object.entries(data || {}).sort(([a], [b]) => a.localeCompare(b));
  const listEl = h('div.msr-kv-list');
  let q = '';
  const draw = () => {
    const rows = entries.filter(([k, v]) => !q || k.toLowerCase().includes(q) || String(fmtMetric(v)).toLowerCase().includes(q));
    if (!entries.length) { mount(listEl, emptyState({ icon: ic, title: emptyText })); return; }
    if (!rows.length) { mount(listEl, h('div.msr-kv-empty', `No keys match “${q}”.`)); return; }
    mount(listEl, h('table.msr-kv-table', h('tbody', rows.map(([k, v]) => h('tr',
      h('th', highlight(k, q)),
      h('td', { class: metric && typeof v === 'number' && 'ms-metric' },
        v && typeof v === 'object' ? h('code.msr-json', highlight(JSON.stringify(v), q)) : highlight(fmtMetric(v), q)),
      h('td.msr-kv-copy', copyButton(typeof v === 'object' ? JSON.stringify(v) : String(v ?? ''))))))));
  };
  const search = h('input.form-control.form-control-sm.msr-kv-search', {
    type: 'search', placeholder: `Filter ${entries.length} keys…`, 'aria-label': `Filter ${title}`,
    oninput: debounce((e) => { q = e.target.value.trim().toLowerCase(); draw(); }, 120),
  });
  draw();
  const json = JSON.stringify(exportYaml ? unflatten(data) : data, null, 2);
  return card({
    title: `${title}`, icon: ic, flush: true,
    actions: [
      h('span.ms-pill.ms-pill-muted', String(entries.length)),
      entries.length > 6 && search,
      copyButton(json, 'JSON'),
      exportYaml && copyButton(() => exportYaml(), 'YAML'),
      h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', title: 'Download JSON', onclick: () => downloadText(json, `${file}.json`, 'application/json') }, icon('download')),
    ],
    body: listEl,
  });
}

// ------------------------------------------------------------------ environment (wandb-metadata.json)

async function loadEnvironment(el, ctx, run) {
  const { entity, project } = ctx.params;
  const files = await fileIndex(entity, project, run.name);
  const metaFile = files.get('wandb-metadata.json');
  let meta = {};
  if (metaFile) {
    try { meta = JSON.parse(await fetchFile(metaFile.directUrl || metaFile.url)); } catch { meta = {}; }
  }
  const cmd = [meta.program, ...(meta.args || [])].filter(Boolean).join(' ');
  const gpus = meta.gpu_nvidia || meta.gpu_devices || [];
  const mem = meta.memory?.total ? fmtBytesShort(Number(meta.memory.total)) : null;
  const disk = meta.disk?.['/']?.total ? `${fmtBytesShort(Number(meta.disk['/'].used))} / ${fmtBytesShort(Number(meta.disk['/'].total))}` : null;
  const snap = run.systemMetrics || {};
  const live = Object.entries(snap).filter(([k]) => /cpu|memory_percent|gpu\.\d+\.gpu$|gpu\.\d+\.memory$/.test(k)).slice(0, 6);
  mount(el, kv([
    meta.os && ['OS', meta.os],
    meta.python && ['Python', meta.python],
    meta.executable && ['Executable', h('code.ms-mono.msr-wrap', meta.executable)],
    cmd && ['Command', h('div.msr-cmd', h('code.ms-mono', `python ${cmd}`), copyButton(`python ${cmd}`))],
    meta.codePath && ['Code path', h('code.ms-mono.msr-wrap', meta.codePath)],
    meta.git?.remote && ['Git remote', h('code.ms-mono.msr-wrap', meta.git.remote)],
    meta.git?.commit && ['Git commit', h('code.ms-mono', meta.git.commit.slice(0, 12))],
    meta.cpu?.model && ['CPU', `${meta.cpu.model}${meta.cpu_count ? ` · ${meta.cpu_count} cores` : ''}`],
    gpus.length && ['GPU', gpus.map((g) => g.name || g).join(', ')],
    mem && ['Memory', mem],
    disk && ['Disk', disk],
    meta.startedAt && ['Process started', fmtDate(meta.startedAt)],
    live.length && ['Last system sample', h('div.msr-sys-snap', live.map(([k, v]) => h('div', h('span.text-muted', k.replace(/^system\./, '')), ' ', h('b.ms-metric', fmtMetric(v)))))],
  ].filter(Boolean), { cols: 1 }), !metaFile && h('div.small.text-muted.mt-2', 'This run uploaded no wandb-metadata.json.'));
}
