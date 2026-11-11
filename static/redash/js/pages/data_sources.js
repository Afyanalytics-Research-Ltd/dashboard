// Data sources: cards with logos, live status, inline connection test,
// pause/resume, and a searchable type picker for new sources.

import { h, icon, mount } from '../../../ui/js/dom.js';
import { pageHeader, emptyState, errorState, skeleton, searchInput, toast, toastError, spinner, openMenu } from '../../../ui/js/ui.js';
import { navigate } from '../../../ui/js/router.js';
import { rd, enc } from '../core/api.js';
import { urls } from '../core/urls.js';
import { invalidate } from '../core/common.js';
import { typeLogo, pickType, matches, hl, pill } from './admin_shared.js';
import { pauseDialog } from './data_source.js';

export const POPULAR_DB = ['pg', 'mysql', 'snowflake', 'bigquery', 'redshift', 'mssql', 'results', 'url', 'json', 'clickhouse', 'athena', 'google_spreadsheets'];

export async function newDataSource() {
  const types = await rd.get('/api/data_sources/types');
  const t = await pickType(types, { title: 'Connect a new data source', kind: 'db', popular: POPULAR_DB });
  if (t) navigate(urls.dataSource('new') + `?type=${encodeURIComponent(t.type)}`);
}

export default async function render(ctx) {
  const { el, signal } = ctx;
  let q = ctx.query.get('q') || '';
  let sources = [];
  let typeNames = {};
  const tests = new Map(); // id -> {state: 'running'|'ok'|'bad', message}
  const grid = h('div.rd-ds-grid', skeleton(4, { height: 60 }));

  mount(el,
    pageHeader({
      title: 'Data sources', icon: 'database',
      subtitle: 'Databases and APIs that queries run against. Credentials stay on the Redash server.',
      actions: [h('button.btn.btn-sm.btn-primary', { onclick: () => newDataSource().catch((e) => toastError(e)) }, icon('plus-lg', 'me-1'), 'New data source')],
    }),
    h('div.af-toolbar', searchInput({ placeholder: 'Search data sources by name or type…', value: q, onInput: (v) => { q = v; ctx.setQuery({ q: v }); draw(); } })),
    grid);

  async function load() {
    try {
      const [ds, types] = await Promise.all([
        rd.get('/api/data_sources', null, { signal }),
        rd.get('/api/data_sources/types', null, { signal }).catch(() => []),
      ]);
      sources = ds;
      typeNames = Object.fromEntries(types.map((t) => [t.type, t.name]));
      if (ctx.active) draw();
    } catch (err) {
      if (err.name !== 'AbortError') mount(grid, errorState(err, load));
    }
  }

  function draw() {
    if (!sources.length) {
      mount(grid, emptyState({
        icon: 'database-add', title: 'No data sources yet',
        message: 'Connect a database, warehouse or API to start querying.',
        action: h('button.btn.btn-sm.btn-primary', { onclick: () => newDataSource() }, icon('plus-lg', 'me-1'), 'Connect a data source'),
      }));
      return;
    }
    const rows = sources.filter((d) => matches(d, q, ['name', 'type', (x) => typeNames[x.type]]));
    if (!rows.length) { mount(grid, emptyState({ icon: 'search', title: 'No data sources match', message: 'Try a different search.' })); return; }
    mount(grid, rows.map((d) => cardFor(d)));
  }

  function cardFor(d) {
    const t = tests.get(d.id);
    const testEl = t && h(`div.rd-ds-test.rd-ds-test-${t.state}`,
      t.state === 'running' ? [spinner('sm'), ' Testing connection…'] : [icon(t.state === 'ok' ? 'check-circle-fill' : 'x-octagon-fill'), ' ', t.message]);
    return h('a.rd-ds-card', { href: urls.dataSource(d.id), class: d.paused && 'paused' },
      h('div.rd-ds-head',
        typeLogo(d.type, 'db', 36),
        h('div.rd-ds-title',
          h('div.rd-ds-name', hl(d.name, q)),
          h('div.rd-ds-type', hl(typeNames[d.type] || d.type, q))),
        h('button.btn.btn-sm.btn-light.rd-ds-more', {
          title: 'Actions',
          onclick: (e) => {
            e.preventDefault(); e.stopPropagation();
            openMenu(e.currentTarget, [
              { label: 'Test connection', icon: 'plug', onClick: () => test(d) },
              d.paused ? { label: 'Resume', icon: 'play-fill', onClick: () => resume(d) } : { label: 'Pause…', icon: 'pause-fill', onClick: () => pause(d) },
              { label: 'Browse schema', icon: 'diagram-3', href: `${urls.dataSource(d.id)}?tab=schema` },
              { label: 'Write a query', icon: 'code-square', href: urls.newQuery({ data_source_id: d.id }) },
            ]);
          },
        }, icon('three-dots-vertical'))),
      h('div.rd-ds-meta',
        d.paused ? pill(`Paused${d.pause_reason ? `: ${d.pause_reason}` : ''}`, 'warn', 'pause-circle-fill') : pill('Active', 'ok', 'circle-fill'),
        d.view_only && pill('View only', 'muted', 'eye'),
        h('span.rd-ds-syntax', d.syntax || 'sql')),
      testEl,
      h('div.rd-ds-foot',
        h('button.btn.btn-sm.btn-outline-secondary', { onclick: (e) => { e.preventDefault(); e.stopPropagation(); test(d); }, disabled: t?.state === 'running' }, icon('plug', 'me-1'), 'Test'),
        h('span.btn.btn-sm.btn-link', 'Configure', icon('arrow-right', 'ms-1'))));
  }

  async function test(d) {
    tests.set(d.id, { state: 'running' });
    draw();
    try {
      const r = await rd.post(`/api/data_sources/${enc(d.id)}/test`, {});
      tests.set(d.id, { state: r.ok ? 'ok' : 'bad', message: r.ok ? 'Connection works' : r.message });
    } catch (err) {
      tests.set(d.id, { state: 'bad', message: err.message });
    }
    if (ctx.active) draw();
  }

  async function pause(d) {
    const updated = await pauseDialog(d);
    if (updated) { Object.assign(d, updated); invalidate('ds'); draw(); }
  }

  async function resume(d) {
    try {
      Object.assign(d, await rd.del(`/api/data_sources/${enc(d.id)}/pause`));
      invalidate('ds');
      toast(`${d.name} resumed`, 'success');
      draw();
    } catch (err) { toastError(err, 'Could not resume'); }
  }

  await load();
}
