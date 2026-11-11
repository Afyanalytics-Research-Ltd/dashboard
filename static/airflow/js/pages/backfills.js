// Backfills: list per DAG (or all DAGs), pause/unpause/cancel, and a
// creation wizard that requires a dry-run preview before it can submit.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import {
  pageHeader, card, dataTable, refreshControl, emptyState, errorState, skeleton, modal, buildForm, confirmDialog,
  menuButton, toast, toastError, timeAgo, jsonView, drawer, kv, stateInfo, chips,
} from '../core/ui.js';
import { fmtDate, fmtNumber, toLocalInput } from '../core/format.js';

const REPROCESS = [
  { value: 'none', label: 'None — only dates with no existing run' },
  { value: 'failed', label: 'Failed — also re-run failed runs' },
  { value: 'completed', label: 'Completed — re-run every existing run' },
];

const STATUS = {
  running: { label: 'running', state: 'running', icon: 'arrow-repeat' },
  paused: { label: 'paused', state: 'paused', icon: 'pause-fill' },
  completed: { label: 'completed', state: 'success', icon: 'check2' },
};

const statusOf = (b) => (b.completed_at ? 'completed' : b.is_paused ? 'paused' : 'running');

function statusBadge(b) {
  const s = STATUS[statusOf(b)];
  const info = stateInfo(s.state);
  return h('span.af-badge', { class: s.label === 'running' && 'af-pulse', style: { color: info.color, background: info.bg } }, icon(s.icon), s.label);
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  let dagId = ctx.query.get('dag') || '';
  let status = ctx.query.get('status') || '';
  let dags = [];
  let rows = [];

  const tableEl = h('div', skeleton(6));
  const summaryEl = h('div');
  const dagSelect = h('select.form-select.form-select-sm.af-select-auto', {
    'aria-label': 'DAG',
    onchange: () => { dagId = dagSelect.value; firstLoad = true; ctx.setQuery({ dag: dagId || null }); load(); },
  }, h('option', { value: '' }, 'All DAGs'));
  const refresh = refreshControl(() => load(), { key: 'backfills', initial: 30 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Backfills', icon: 'rewind-circle',
      subtitle: 'Create DAG runs for past date ranges, and steer backfills that are in progress.',
      actions: [
        h('button.btn.btn-sm.btn-primary', { onclick: () => openWizard() }, icon('plus-lg', 'me-1'), 'New backfill'),
        refresh.el,
      ],
    }),
    card({
      flush: true,
      body: h('div',
        h('div.af-toolbar.af-card-toolbar',
          dagSelect,
          summaryEl),
        tableEl),
    }));

  try {
    dags = await api.all('/dags', 'dags', { order_by: 'dag_id' }, { signal });
  } catch (err) {
    if (err.name === 'AbortError') return;
    mount(tableEl, h('div.p-3', errorState(err)));
    return;
  }
  if (!ctx.active) return;
  dags.forEach((d) => dagSelect.appendChild(h('option', { value: d.dag_id, selected: d.dag_id === dagId }, d.dag_display_name || d.dag_id)));

  /** GET /backfills requires dag_id, so "All DAGs" fans out with a small worker pool. */
  async function fetchAll(onProgress) {
    const ids = dagId ? [dagId] : dags.map((d) => d.dag_id);
    const out = [];
    let next = 0;
    let done = 0;
    const worker = async () => {
      while (next < ids.length) {
        const id = ids[next++];
        try {
          out.push(...await api.all('/backfills', 'backfills', { dag_id: id, order_by: '-id' }, { signal }));
        } catch (err) {
          if (err.name === 'AbortError') throw err;
        }
        onProgress?.(++done, ids.length);
      }
    };
    await Promise.all(Array.from({ length: Math.min(12, ids.length) }, worker));
    return out.sort((a, b) => b.id - a.id);
  }

  let firstLoad = true;
  async function load() {
    try {
      const progress = firstLoad && !dagId ? h('div.af-scan', h('div.small.text-muted.mb-2', 'Scanning DAGs for backfills…'), h('div.af-meter', h('span', { style: { width: '0%', background: 'var(--af-primary)' } }))) : null;
      if (progress) mount(tableEl, h('div.p-3', progress));
      rows = await fetchAll(progress && ((n, total) => {
        progress.querySelector('.small').textContent = `Scanning DAGs for backfills… ${n} / ${total}`;
        progress.querySelector('.af-meter span').style.width = `${(n / total) * 100}%`;
      }));
      firstLoad = false;
      if (!ctx.active) return;
      draw();
    } catch (err) {
      if (err.name !== 'AbortError') mount(tableEl, h('div.p-3', errorState(err, load)));
    }
  }

  function draw() {
    const counts = { running: 0, paused: 0, completed: 0 };
    rows.forEach((b) => { counts[statusOf(b)]++; });
    mount(summaryEl, chips([
      { value: '', label: 'All', count: rows.length },
      { value: 'running', label: 'Running', count: counts.running, color: stateInfo('running').color },
      { value: 'paused', label: 'Paused', count: counts.paused, color: stateInfo('paused').color },
      { value: 'completed', label: 'Completed', count: counts.completed, color: stateInfo('success').color },
    ], status, (v) => { status = v; ctx.setQuery({ status: v || null }); draw(); }));
    const visible = status ? rows.filter((b) => statusOf(b) === status) : rows;
    mount(tableEl, dataTable({
      rows: visible,
      rowKey: (b) => b.id,
      onRowClick: (b) => showDetail(b),
      empty: emptyState({
        icon: 'rewind-circle', title: 'No backfills',
        message: dagId ? `${dagId} has never been backfilled.` : 'No DAG has been backfilled yet.',
        action: h('button.btn.btn-sm.btn-primary', { onclick: () => openWizard() }, icon('plus-lg', 'me-1'), 'New backfill'),
      }),
      columns: [
        { key: 'id', label: 'Id', width: '70px', render: (b) => h('span.af-mono', `#${b.id}`) },
        { key: 'dag_id', label: 'DAG', render: (b) => h('a.fw-semibold', { href: urls.dag(b.dag_id) }, b.dag_display_name || b.dag_id) },
        { key: 'status', label: 'Status', render: statusBadge },
        { key: 'range', label: 'Date range', render: (b) => h('span.small.af-nowrap', fmtDate(b.from_date, { seconds: false }), icon('arrow-right', 'mx-1 text-muted'), fmtDate(b.to_date, { seconds: false })) },
        { key: 'reprocess_behavior', label: 'Reprocess', render: (b) => h('span.af-tag', b.reprocess_behavior) },
        { key: 'max_active_runs', label: 'Max active', align: 'right' },
        { key: 'created_at', label: 'Created', render: (b) => timeAgo(b.created_at) },
        { key: 'completed_at', label: 'Completed', render: (b) => (b.completed_at ? timeAgo(b.completed_at) : h('span.text-muted', '—')) },
        { key: 'actions', label: '', align: 'right', width: '48px', render: (b) => menuButton(null, () => actionsFor(b)) },
      ],
    }));
  }

  function actionsFor(b) {
    const done = !!b.completed_at;
    return [
      { label: 'Details', icon: 'info-circle', onClick: () => showDetail(b) },
      { label: 'View runs', icon: 'play-circle', href: urls.runs({ dag: b.dag_id, type: 'backfill' }) },
      { divider: true },
      b.is_paused
        ? { label: 'Unpause', icon: 'play-fill', disabled: done, onClick: () => act(b, 'unpause') }
        : { label: 'Pause', icon: 'pause-fill', disabled: done, onClick: () => act(b, 'pause') },
      { label: 'Cancel backfill', icon: 'x-octagon', danger: true, disabled: done, onClick: () => act(b, 'cancel') },
    ];
  }

  async function act(b, verb) {
    if (verb === 'cancel') {
      const ok = await confirmDialog({
        title: `Cancel backfill #${b.id}?`, danger: true, confirmLabel: 'Cancel backfill',
        message: 'Queued runs that have not started are removed; running ones finish. This cannot be undone.',
      });
      if (!ok) return;
    }
    try {
      await api.put(`/backfills/${enc(b.id)}/${verb}`, undefined);
      toast(`Backfill #${b.id} ${verb === 'cancel' ? 'cancelled' : `${verb}d`}`, 'success');
      load();
    } catch (err) { toastError(err, `Could not ${verb} backfill`); }
  }

  function showDetail(b) {
    const d = drawer({
      title: h('span', `Backfill #${b.id} `, statusBadge(b)),
      subtitle: b.dag_id,
      width: 620,
      actions: menuButton('Actions', () => actionsFor(b).map((a) => (a.onClick ? { ...a, onClick: () => { d.close(); a.onClick(); } } : a)), { icon: 'lightning' }),
      body: h('div.d-flex.flex-column.gap-3',
        kv([
          ['DAG', h('a', { href: urls.dag(b.dag_id) }, b.dag_id)],
          ['Status', statusBadge(b)],
          ['From', fmtDate(b.from_date)],
          ['To', fmtDate(b.to_date)],
          ['Reprocess', b.reprocess_behavior],
          ['Max active runs', b.max_active_runs],
          ['Created', fmtDate(b.created_at)],
          ['Updated', fmtDate(b.updated_at)],
          ['Completed', b.completed_at ? fmtDate(b.completed_at) : null],
        ]),
        h('div', h('div.af-section-label', 'Run conf'), jsonView(b.dag_run_conf || {}))),
    });
    d.el.addEventListener('click', (e) => { if (e.target.closest('a[href]')) d.close(); });
  }

  // ---------------------------------------------------------------- wizard

  function openWizard() {
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 86400e3);
    const form = buildForm([
      { name: 'dag_id', label: 'DAG', type: 'select', required: true, value: dagId || dags[0]?.dag_id, options: dags.map((d) => ({ value: d.dag_id, label: `${d.dag_id}${d.is_paused ? ' (paused)' : ''}` })) },
      { name: 'from_date', label: 'From (logical date)', type: 'datetime', required: true, value: toLocalInput(weekAgo) },
      { name: 'to_date', label: 'To (logical date)', type: 'datetime', required: true, value: toLocalInput(now) },
      { name: 'reprocess_behavior', label: 'Reprocess existing runs', type: 'select', options: REPROCESS, value: 'none' },
      { name: 'max_active_runs', label: 'Max active runs', type: 'number', value: 10, min: 1, step: 1 },
      { name: 'run_backwards', type: 'checkbox', checkLabel: 'Run backwards (newest dates first)', value: false },
      { name: 'run_on_latest_version', type: 'checkbox', checkLabel: 'Run on the latest DAG version', value: true },
      { name: 'dag_run_conf', label: 'Run conf (JSON)', type: 'json', value: {}, rows: 5 },
    ]);
    form.el.classList.add('af-backfill-form');
    const preview = h('div.af-preview', h('div.small.text-muted', icon('info-circle', 'me-1'), 'Preview the runs this backfill will create. Create unlocks after a successful preview.'));
    let previewedFor = null;

    const body = () => {
      const v = form.values();
      if (v.from_date && v.to_date && new Date(v.from_date) > new Date(v.to_date)) throw new Error('“From” must be before “To”.');
      if (v.dag_run_conf != null && (typeof v.dag_run_conf !== 'object' || Array.isArray(v.dag_run_conf))) throw new Error('Run conf must be a JSON object.');
      return {
        dag_id: v.dag_id,
        from_date: v.from_date,
        to_date: v.to_date,
        run_backwards: v.run_backwards,
        dag_run_conf: v.dag_run_conf || {},
        reprocess_behavior: v.reprocess_behavior,
        max_active_runs: v.max_active_runs || 10,
        run_on_latest_version: v.run_on_latest_version,
      };
    };

    const m = modal({
      title: 'New backfill', icon: 'rewind-circle', size: 'lg',
      body: h('div', form.el, preview),
      actions: [
        { label: 'Cancel' },
        { label: 'Preview runs', icon: 'eye', onClick: doPreview },
        {
          label: 'Create backfill', variant: 'primary', icon: 'rewind', disabled: true,
          onClick: async (close) => {
            const b = body();
            if (JSON.stringify(b) !== previewedFor) throw new Error('Inputs changed since the preview — preview again.');
            const created = await api.post('/backfills', b);
            toast(`Backfill #${created.id} started for ${created.dag_id}`, 'success');
            close();
            dagId = created.dag_id;
            dagSelect.value = dagId;
            ctx.setQuery({ dag: dagId });
            load();
          },
        },
      ],
    });
    const createBtn = [...m.el.querySelectorAll('.af-modal-footer button')].pop();
    const invalidate = () => {
      if (!previewedFor) return;
      previewedFor = null;
      createBtn.disabled = true;
      preview.classList.add('af-stale');
    };
    form.el.addEventListener('input', invalidate);
    form.el.addEventListener('change', invalidate);

    async function doPreview() {
      const b = body();
      mount(preview, skeleton(3));
      preview.classList.remove('af-stale');
      try {
        const res = await api.post('/backfills/dry_run', b);
        const runs = res.backfills || [];
        const total = res.total_entries ?? runs.length;
        previewedFor = JSON.stringify(b);
        createBtn.disabled = !total;
        mount(preview,
          h('div.d-flex.justify-content-between.align-items-center.mb-2',
            h('div.fw-semibold', icon(total ? 'check-circle-fill' : 'exclamation-circle', total ? 'text-success me-1' : 'text-warning me-1'),
              total ? `${fmtNumber(total)} run${total === 1 ? '' : 's'} will be created` : 'No runs would be created for this range'),
            h('span.small.text-muted', `${b.dag_id} · reprocess: ${b.reprocess_behavior}`)),
          total ? h('div.af-dryrun-list', runs.slice(0, 500).map((r, i) => h('div.af-dryrun-item',
            h('span.af-dryrun-n', String(i + 1)), h('span.af-mono', fmtDate(r.logical_date || r.partition_key || r.run_after))))) : null,
          runs.length > 500 && h('div.small.text-muted.mt-1', `…and ${fmtNumber(runs.length - 500)} more`));
      } catch (err) {
        previewedFor = null;
        createBtn.disabled = true;
        mount(preview, h('div.text-danger.small', icon('x-octagon', 'me-1'), err.message));
      }
    }
  }

  await load();
}
