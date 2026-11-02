// Import errors: DAG files that failed to parse, plus plugin import errors.

import { h, icon, mount } from '../core/dom.js';
import { api, enc } from '../core/api.js';
import {
  pageHeader, card, pager, refreshControl, searchInput, timeAgo, codeBlock, errorState, skeleton,
  copyButton, emptyState,
} from '../core/ui.js';
import { fmtDate } from '../core/format.js';

export default async function render(ctx) {
  const { el, signal } = ctx;
  const state = { q: ctx.query.get('q') || '', offset: Number(ctx.query.get('offset') || 0), limit: 25 };
  const listEl = h('div', skeleton(6));
  const pluginEl = h('div', skeleton(2));
  const countEl = h('span.af-muted.small');
  const refresh = refreshControl(() => load(), { key: 'import-errors', initial: 30 });
  ctx.cleanup(refresh.stop);

  mount(el,
    pageHeader({
      title: 'Import errors', icon: 'bug',
      subtitle: 'DAG files the DAG processor could not parse. Fix the file and it is retried on the next parse.',
      actions: [refresh.el],
    }),
    card({
      title: 'DAG files', icon: 'file-earmark-x', actions: countEl,
      body: h('div.d-flex.flex-column.gap-3',
        h('div.af-toolbar', searchInput({
          placeholder: 'Filter by file name…', value: state.q,
          onInput: (q) => { state.q = q; state.offset = 0; ctx.setQuery({ q, offset: null }); load(); },
        })),
        listEl),
    }),
    card({ title: 'Plugin import errors', icon: 'puzzle', body: pluginEl }));

  async function load() {
    loadPlugins();
    try {
      const params = { limit: state.limit, offset: state.offset, order_by: '-timestamp' };
      if (state.q) params.filename_pattern = state.q;
      const data = await api.get('/importErrors', params, { signal });
      if (!ctx.active) return;
      mount(countEl, `${data.total_entries} file${data.total_entries === 1 ? '' : 's'}`);
      if (!data.import_errors.length) {
        mount(listEl, state.q
          ? emptyState({ icon: 'search', title: 'No matching import errors' })
          : h('div.sys-celebrate', h('div.sys-celebrate-icon', icon('patch-check-fill')),
            h('div.af-empty-title', 'Every DAG file parses cleanly'),
            h('div.af-empty-msg', 'No import errors right now. Nice work.')));
        return;
      }
      mount(listEl,
        h('div.d-flex.flex-column.gap-2', data.import_errors.map(errorCard)),
        data.total_entries > state.limit ? pager({
          total: data.total_entries, limit: state.limit, offset: state.offset, sizes: [25, 50, 100],
          onChange: (offset, limit) => { state.offset = offset; state.limit = limit; ctx.setQuery({ offset: offset || null }); load(); },
        }) : null);
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(listEl, errorState(err, load));
    }
  }

  function errorCard(e) {
    const firstLine = (e.stack_trace || '').trim().split('\n').filter(Boolean).pop() || 'Unknown error';
    const body = h('div.sys-ie-body');
    let loaded = false;
    const details = h('details.sys-ie', {
      ontoggle: async () => {
        if (!details.open || loaded) return;
        loaded = true;
        mount(body, skeleton(4));
        try {
          const full = await api.get(`/importErrors/${enc(e.import_error_id)}`, null, { signal });
          mount(body,
            h('div.d-flex.gap-3.flex-wrap.small.text-muted.mb-2',
              h('span', icon('folder2', 'me-1'), full.bundle_name || 'bundle —'),
              h('span', icon('clock', 'me-1'), fmtDate(full.timestamp)),
              h('span', icon('hash', 'me-1'), `error #${full.import_error_id}`)),
            codeBlock(full.stack_trace || '', { lang: 'text', maxHeight: '420px' }));
        } catch (err) {
          loaded = false;
          mount(body, errorState(err));
        }
      },
    },
    h('summary.sys-ie-summary',
      icon('chevron-right', 'sys-ie-caret'),
      icon('file-earmark-code-fill', 'text-danger'),
      h('div.sys-ie-main',
        h('div.sys-ie-file', e.filename),
        h('div.sys-ie-msg', firstLine)),
      h('div.sys-ie-side', timeAgo(e.timestamp), copyButton(e.filename))),
    body);
    return details;
  }

  async function loadPlugins() {
    try {
      const data = await api.get('/plugins/importErrors', null, { signal });
      if (!ctx.active) return;
      if (!data.import_errors.length) {
        mount(pluginEl, h('div.d-flex.align-items-center.gap-2.text-success.small', icon('check-circle-fill'), 'All plugins loaded without errors.'));
        return;
      }
      mount(pluginEl, h('div.d-flex.flex-column.gap-2', data.import_errors.map((p) => h('div.sys-plugin-err',
        h('div.fw-semibold.af-mono.small.mb-1', icon('puzzle-fill', 'text-danger me-1'), p.source),
        codeBlock(p.error || '', { lang: 'text', maxHeight: '260px' })))));
    } catch (err) {
      if (err.name === 'AbortError' || !ctx.active) return;
      mount(pluginEl, errorState(err));
    }
  }

  await load();
}
