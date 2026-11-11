// Visualization editor: options form on the left, live preview on the right.
//
//   const saved = await openVizEditor({ query, result, viz, onSaved, onDeleted })
//   await deleteViz(viz)
//
// `result` is a query_result ({data: {columns, rows}}); when omitted the query's latest
// cached result is loaded. Options are kept Redash-compatible; Studio-only extras are
// stored under `studio*` keys (ignored by Redash's own renderer).

import { h, icon, mount, debounce } from '../../../ui/js/dom.js';
import { modal, confirmDialog, toast, toastError, tabs, emptyState, spinner } from '../../../ui/js/ui.js';
import { rd, enc, runQuery } from './api.js';
import {
  renderViz, suggestViz, defaultChartOptions, VIZ_TYPES, CHART_TYPES, PIVOT_AGGREGATORS, PALETTE, vizTypeInfo,
} from './viz.js';
import { tableColumns } from './results.js';
import { inferType, isNumericType, isDateType } from './format.js';

const NUMBER_FORMATS = ['0,0', '0,0.00', '0,0[.]00', '0,0[.]00000', '0.0%', '0[.]00%', '$0,0.00', '0.0a'];
const DATE_FORMATS = ['DD/MM/YY', 'DD/MM/YY HH:mm', 'YYYY-MM-DD', 'YYYY-MM-DD HH:mm', 'YYYY-MM-DD HH:mm:ss', 'MMM YYYY', 'DD MMM YYYY'];
const TABLE_DISPLAY = ['string', 'number', 'datetime', 'boolean', 'json', 'image', 'link'];
const RULE_OPS = ['>', '>=', '<', '<=', '=', '!=', 'contains', 'empty'];

const clone = (o) => (o == null ? o : JSON.parse(JSON.stringify(o)));

function columnsOf(result) {
  const data = result?.data || { columns: [], rows: [] };
  return (data.columns || []).map((c) => ({
    name: c.name,
    label: c.friendly_name || c.name,
    type: c.type && c.type !== 'string' ? c.type : inferType(data.rows || [], c.name),
  }));
}

/** Sensible starting options for a viz type given the result columns. */
export function defaultOptionsFor(type, result) {
  const cols = columnsOf(result);
  const nums = cols.filter((c) => isNumericType(c.type));
  const cats = cols.filter((c) => !isNumericType(c.type));
  const first = (list, fallback) => (list[0] || fallback || cols[0] || {}).name;
  switch (type) {
    case 'CHART': {
      const s = suggestViz(result?.data || {});
      if (s.type === 'CHART') return s.options;
      const x = first(cols.filter((c) => isDateType(c.type)), cats[0]);
      return defaultChartOptions('column', x, nums.filter((n) => n.name !== x).slice(0, 2).map((n) => n.name));
    }
    case 'COUNTER':
      return { counterColName: first(nums), rowNumber: 1, targetRowNumber: 1, stringDecimal: 0, stringDecChar: '.', stringThouSep: ',', countRow: false, formatTargetValue: false, counterLabel: '' };
    case 'TABLE':
      return { itemsPerPage: 25, columns: [] };
    case 'PIVOT':
      return { controls: { enabled: false }, rows: cats[0] ? [cats[0].name] : [], cols: [], vals: nums[0] ? [nums[0].name] : [], aggregatorName: nums[0] ? 'Sum' : 'Count', rendererName: 'Table' };
    case 'FUNNEL':
      return { stepCol: { colName: first(cats), displayAs: 'Steps' }, valueCol: { colName: first(nums), displayAs: 'Value' }, autoSort: true, sortKeyCol: { colName: '' }, itemsPerPage: 100 };
    default:
      return {};
  }
}

// ------------------------------------------------------------------ tiny form kit

const row = (label, control, help) => h('div.rdv-row', h('label.rdv-label', label), h('div.rdv-control', control), help && h('div.rdv-help', help));
const section = (title, ...children) => h('div.rdv-section', h('div.rdv-section-title', title), ...children);

function select(value, options, onChange, { allowEmpty, emptyLabel = '— none —' } = {}) {
  const el = h('select.form-select.form-select-sm',
    allowEmpty && h('option', { value: '' }, emptyLabel),
    options.map((o) => {
      const opt = typeof o === 'object' ? o : { value: o, label: o };
      return h('option', { value: opt.value, selected: String(opt.value) === String(value ?? '') }, opt.label);
    }));
  el.addEventListener('change', () => onChange(el.value));
  return el;
}
function text(value, onChange, { placeholder, type = 'text', list } = {}) {
  const el = h('input.form-control.form-control-sm', { type, value: value ?? '', placeholder, list });
  el.addEventListener('input', () => onChange(type === 'number' ? (el.value === '' ? null : Number(el.value)) : el.value));
  return el;
}
function check(value, onChange, label) {
  const id = `rdv-${Math.random().toString(36).slice(2, 8)}`;
  const input = h('input.form-check-input', { type: 'checkbox', id, checked: !!value });
  input.addEventListener('change', () => onChange(input.checked));
  return h('div.form-check.form-switch.rdv-check', input, h('label.form-check-label', { for: id }, label));
}
function color(value, onChange) {
  const el = h('input.form-control.form-control-sm.form-control-color.rdv-color', { type: 'color', value: value || '#0072CE', title: 'Colour' });
  el.addEventListener('input', () => onChange(el.value));
  return el;
}
function formatField(value, onChange, presets, placeholder) {
  const id = `rdv-fmt-${Math.random().toString(36).slice(2, 8)}`;
  return h('div', text(value, onChange, { placeholder, list: id }), h(`datalist#${id}`, presets.map((p) => h('option', { value: p }))));
}

// ------------------------------------------------------------------ CHART

function chartForm(o, cols, data, changed) {
  o.columnMapping ||= {};
  o.series ||= { stacking: null };
  o.seriesOptions ||= {};
  o.xAxis ||= { type: '-', labels: { enabled: true } };
  o.xAxis.labels ||= { enabled: true };
  o.yAxis ||= [{ type: 'linear' }, { type: 'linear', opposite: true }];
  while (o.yAxis.length < 2) o.yAxis.push({ type: 'linear', opposite: true });
  o.legend ||= { enabled: true, placement: 'auto', traceorder: 'normal' };
  o.studioRefLines ||= [];

  const roleOf = (role) => Object.entries(o.columnMapping).filter(([, r]) => r === role).map(([c]) => c);
  const setRole = (col, role) => {
    for (const [c, r] of Object.entries(o.columnMapping)) if (r === role && role !== 'y') delete o.columnMapping[c];
    if (col) o.columnMapping[col] = role;
  };
  const colOpts = cols.map((c) => ({ value: c.name, label: `${c.label} (${c.type})` }));
  const pane = h('div');
  let current = 'general';

  const typeButtons = () => h('div.rdv-charttypes', CHART_TYPES.map((t) => h('button.rdv-charttype', {
    type: 'button', class: o.globalSeriesType === t.value && 'active', title: t.label,
    onclick: () => { o.globalSeriesType = t.value; if (t.value === 'pie') o.showDataLabels = true; changed(); draw(); },
  }, icon(t.icon), h('span', t.label))));

  const yChecks = () => h('div.rdv-checklist', cols.filter((c) => o.columnMapping[c.name] !== 'x').map((c) => {
    const on = o.columnMapping[c.name] === 'y';
    return check(on, (v) => { if (v) o.columnMapping[c.name] = 'y'; else if (o.columnMapping[c.name] === 'y') delete o.columnMapping[c.name]; changed(); draw(); }, `${c.label}`);
  }));

  // Series keys exactly as chartFigure() computes them.
  const seriesKeys = () => {
    const seriesCol = roleOf('series')[0];
    if (seriesCol) return [...new Set((data.rows || []).map((r) => String(r[seriesCol])))].slice(0, 40);
    return roleOf('y');
  };

  const panes = {
    general: () => [
      section('Chart type', typeButtons()),
      section('Columns',
        row('X column', select(roleOf('x')[0], colOpts, (v) => { setRole(v, 'x'); changed(); draw(); }, { allowEmpty: true })),
        o.globalSeriesType !== 'heatmap' && row('Y columns', yChecks()),
        o.globalSeriesType === 'heatmap' && row('Y column', select(roleOf('y')[0], colOpts, (v) => { for (const c of roleOf('y')) delete o.columnMapping[c]; if (v) o.columnMapping[v] = 'y'; changed(); }, { allowEmpty: true })),
        o.globalSeriesType === 'heatmap' && row('Colour (Z) column', select(roleOf('zVal')[0], colOpts, (v) => { setRole(v, 'zVal'); changed(); }, { allowEmpty: true })),
        !['pie', 'heatmap'].includes(o.globalSeriesType) && row('Group by', select(roleOf('series')[0], colOpts, (v) => { setRole(v, 'series'); changed(); draw(); }, { allowEmpty: true }), 'Split one Y column into a series per distinct value.'),
        o.globalSeriesType === 'bubble' && row('Bubble size', select(roleOf('size')[0], colOpts, (v) => { setRole(v, 'size'); changed(); }, { allowEmpty: true })),
        !['pie', 'heatmap', 'box'].includes(o.globalSeriesType) && row('Error bars', select(roleOf('yError')[0], colOpts, (v) => { setRole(v, 'yError'); changed(); }, { allowEmpty: true }))),
      section('Display',
        !['pie', 'heatmap', 'box', 'scatter', 'bubble'].includes(o.globalSeriesType) && row('Stacking', select(o.series.stacking || '', [{ value: '', label: 'Disabled' }, { value: 'stack', label: 'Stack' }], (v) => { o.series.stacking = v || null; changed(); })),
        o.globalSeriesType === 'column' && check(o.swappedAxes, (v) => { o.swappedAxes = v; changed(); }, 'Horizontal bars'),
        o.globalSeriesType !== 'pie' && check(o.sortX !== false, (v) => { o.sortX = v; changed(); }, 'Sort X values'),
        check(o.missingValuesAsZero, (v) => { o.missingValuesAsZero = v; changed(); }, 'Treat missing values as 0'),
        check(o.showDataLabels, (v) => { o.showDataLabels = v; changed(); }, 'Show data labels'),
        check(o.legend.enabled !== false, (v) => { o.legend.enabled = v; changed(); }, 'Show legend'),
        o.globalSeriesType === 'pie' && row('Direction', select(o.direction?.type || 'counterclockwise', ['counterclockwise', 'clockwise'], (v) => { o.direction = { type: v }; changed(); })),
        o.globalSeriesType === 'bubble' && row('Bubble size coefficient', text(o.coefficient ?? 1, (v) => { o.coefficient = v || 1; changed(); }, { type: 'number' }))),
    ],
    series: () => {
      const keys = seriesKeys();
      if (!keys.length) return [h('div.rdv-help', 'Choose Y columns first.')];
      return [section('Series', h('div.rdv-series', keys.map((k, i) => {
        o.seriesOptions[k] ||= { index: i, zIndex: i, type: o.globalSeriesType, yAxis: 0 };
        const so = o.seriesOptions[k];
        return h('div.rdv-series-row',
          color(so.color || PALETTE[i % PALETTE.length], (v) => { so.color = v; changed(); }),
          h('div.rdv-series-key', { title: k }, k),
          text(so.name, (v) => { so.name = v || undefined; changed(); }, { placeholder: 'Label' }),
          !['pie', 'heatmap', 'box'].includes(o.globalSeriesType) && select(so.type || o.globalSeriesType, CHART_TYPES.filter((t) => ['column', 'line', 'area', 'scatter'].includes(t.value)).map((t) => ({ value: t.value, label: t.label })), (v) => { so.type = v; changed(); }),
          !['pie', 'heatmap'].includes(o.globalSeriesType) && select(String(so.yAxis || 0), [{ value: '0', label: 'Left' }, { value: '1', label: 'Right' }], (v) => { so.yAxis = Number(v); changed(); }),
          so.color && h('button.btn.btn-sm.btn-link.p-0', { type: 'button', title: 'Reset colour', onclick: () => { delete so.color; changed(); draw(); } }, icon('arrow-counterclockwise')));
      })))];
    },
    axes: () => [
      section('X axis',
        row('Scale', select(o.xAxis.type || '-', [{ value: '-', label: 'Auto detect' }, { value: 'category', label: 'Category' }, { value: 'linear', label: 'Linear' }, { value: 'logarithmic', label: 'Logarithmic' }, { value: 'datetime', label: 'Date/time' }], (v) => { o.xAxis.type = v; changed(); })),
        row('Title', text(o.xAxis.title?.text, (v) => { o.xAxis.title = { text: v }; changed(); })),
        check(o.xAxis.labels.enabled !== false, (v) => { o.xAxis.labels.enabled = v; changed(); }, 'Show labels'),
        check(o.studioShowAllXLabels, (v) => { o.studioShowAllXLabels = v; changed(); }, 'Show every X label (no skipping)')),
      ...[0, 1].map((i) => section(i ? 'Right Y axis' : 'Left Y axis',
        row('Scale', select(o.yAxis[i].type || 'linear', [{ value: 'linear', label: 'Linear' }, { value: 'logarithmic', label: 'Logarithmic' }, { value: 'datetime', label: 'Date/time' }, { value: 'category', label: 'Category' }], (v) => { o.yAxis[i].type = v; changed(); })),
        row('Title', text(o.yAxis[i].title?.text, (v) => { o.yAxis[i].title = { text: v }; changed(); })),
        h('div.rdv-pair',
          row('Min', text(o.yAxis[i].rangeMin, (v) => { o.yAxis[i].rangeMin = v ?? undefined; changed(); }, { type: 'number', placeholder: 'auto' })),
          row('Max', text(o.yAxis[i].rangeMax, (v) => { o.yAxis[i].rangeMax = v ?? undefined; changed(); }, { type: 'number', placeholder: 'auto' }))))),
    ],
    format: () => [
      section('Number formats',
        row('Numbers', formatField(o.numberFormat, (v) => { o.numberFormat = v; changed(); }, NUMBER_FORMATS, '0,0[.]00'), 'numeral.js syntax, e.g. 0,0.00 · 0.0% · $0,0 · 0.0a'),
        row('Percentages', formatField(o.percentFormat, (v) => { o.percentFormat = v; changed(); }, ['0%', '0.0%', '0[.]00%'], '0[.]00%')),
        row('Dates', formatField(o.dateTimeFormat, (v) => { o.dateTimeFormat = v; changed(); }, DATE_FORMATS, 'DD/MM/YY HH:mm'))),
      section('Reference lines',
        h('div.rdv-help', 'Horizontal lines such as a target or average (shown in Analytics Studio).'),
        h('div.rdv-reflines', o.studioRefLines.map((l, i) => h('div.rdv-refline',
          text(l.label, (v) => { l.label = v; changed(); }, { placeholder: 'Label (e.g. Target)' }),
          text(l.value, (v) => { l.value = v; changed(); }, { type: 'number', placeholder: 'Value' }),
          color(l.color || '#E5484D', (v) => { l.color = v; changed(); }),
          h('button.btn.btn-sm.btn-outline-danger', { type: 'button', title: 'Remove line', onclick: () => { o.studioRefLines.splice(i, 1); changed(); draw(); } }, icon('trash'))))),
        h('button.btn.btn-sm.btn-outline-secondary', {
          type: 'button',
          onclick: () => {
            const yc = roleOf('y')[0];
            const vals = (data.rows || []).map((r) => Number(r[yc])).filter(Number.isFinite);
            const avg = vals.length ? Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 100) / 100 : 0;
            o.studioRefLines.push({ label: 'Average', value: avg, color: '#E5484D' });
            changed(); draw();
          },
        }, icon('plus', 'me-1'), 'Add line (starts at the average)')),
    ],
  };

  const bar = tabs([
    { id: 'general', label: 'General', icon: 'sliders' },
    { id: 'series', label: 'Series', icon: 'palette' },
    { id: 'axes', label: 'Axes', icon: 'rulers' },
    { id: 'format', label: 'Format', icon: '123' },
  ], current, (id) => { current = id; draw(); });
  function draw() { mount(pane, panes[current]()); }
  draw();
  return h('div', bar, pane);
}

// ------------------------------------------------------------------ COUNTER

function counterForm(o, cols, changed, redraw) {
  const colOpts = cols.map((c) => ({ value: c.name, label: c.label }));
  return h('div',
    section('Value',
      check(o.countRow, (v) => { o.countRow = v; changed(); redraw(); }, 'Count rows instead of a column value'),
      !o.countRow && row('Value column', select(o.counterColName, colOpts, (v) => { o.counterColName = v; changed(); }, { allowEmpty: true })),
      !o.countRow && row('Row number', text(o.rowNumber ?? 1, (v) => { o.rowNumber = v || 1; changed(); }, { type: 'number' })),
      row('Label', text(o.counterLabel, (v) => { o.counterLabel = v; changed(); }, { placeholder: 'Shown under the value' }))),
    section('Target (shows % change)',
      row('Target column', select(o.targetColName, colOpts, (v) => { o.targetColName = v || undefined; changed(); }, { allowEmpty: true, emptyLabel: '— no target —' })),
      row('Target row number', text(o.targetRowNumber ?? 1, (v) => { o.targetRowNumber = v || 1; changed(); }, { type: 'number' })),
      check(o.formatTargetValue, (v) => { o.formatTargetValue = v; changed(); }, 'Format the target like the value')),
    section('Formatting',
      h('div.rdv-pair',
        row('Prefix', text(o.stringPrefix, (v) => { o.stringPrefix = v; changed(); }, { placeholder: 'e.g. KES ' })),
        row('Suffix', text(o.stringSuffix, (v) => { o.stringSuffix = v; changed(); }, { placeholder: 'e.g. %' }))),
      h('div.rdv-pair',
        row('Decimals', text(o.stringDecimal ?? 0, (v) => { o.stringDecimal = Math.max(0, Math.min(10, v || 0)); changed(); }, { type: 'number' })),
        row('Decimal char', text(o.stringDecChar ?? '.', (v) => { o.stringDecChar = v; changed(); })),
        row('Thousands', text(o.stringThouSep ?? ',', (v) => { o.stringThouSep = v; changed(); })))));
}

// ------------------------------------------------------------------ TABLE

function tableForm(o, result, changed, redraw) {
  const data = result?.data || { columns: [], rows: [] };
  if (!o.columns?.length || o.columns.length !== (data.columns || []).length) {
    // Normalise to the full column list (keeps existing per-column settings).
    o.columns = tableColumns(data, o).map(({ type, rules, ...c }) => ({ ...c, studioRules: rules }));
  }
  o.columns.sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  const reorder = () => { o.columns.forEach((c, i) => { c.order = i; }); changed(); redraw(); };
  const open = new Set();
  const list = h('div.rdv-cols');
  let dragFrom = null;

  const colRow = (c, i) => {
    const isOpen = open.has(c.name);
    c.studioRules ||= [];
    const head = h('div.rdv-col-head', {
      draggable: 'true',
      ondragstart: (e) => { dragFrom = i; e.dataTransfer.effectAllowed = 'move'; },
      ondragover: (e) => { e.preventDefault(); head.classList.add('drop'); },
      ondragleave: () => head.classList.remove('drop'),
      ondrop: (e) => {
        e.preventDefault();
        head.classList.remove('drop');
        if (dragFrom == null || dragFrom === i) return;
        const [moved] = o.columns.splice(dragFrom, 1);
        o.columns.splice(i, 0, moved);
        dragFrom = null;
        reorder();
      },
    },
    h('span.rdv-grip', { title: 'Drag to reorder' }, icon('grip-vertical')),
    h('button.rdv-eye', { type: 'button', title: c.visible === false ? 'Show column' : 'Hide column', onclick: () => { c.visible = c.visible === false; changed(); redraw(); } }, icon(c.visible === false ? 'eye-slash' : 'eye')),
    h('span.rdv-col-name', { class: c.visible === false && 'hidden' }, c.name),
    c.studioRules.length ? h('span.rd-type-pill', icon('palette2'), c.studioRules.length) : null,
    h('div.rdv-col-move',
      h('button.af-icon-btn', { type: 'button', title: 'Move up', disabled: i === 0, onclick: () => { [o.columns[i - 1], o.columns[i]] = [o.columns[i], o.columns[i - 1]]; reorder(); } }, icon('arrow-up')),
      h('button.af-icon-btn', { type: 'button', title: 'Move down', disabled: i === o.columns.length - 1, onclick: () => { [o.columns[i + 1], o.columns[i]] = [o.columns[i], o.columns[i + 1]]; reorder(); } }, icon('arrow-down')),
      h('button.af-icon-btn', { type: 'button', title: 'Column settings', onclick: () => { isOpen ? open.delete(c.name) : open.add(c.name); draw(); } }, icon(isOpen ? 'chevron-up' : 'gear'))));
    if (!isOpen) return h('div.rdv-col', head);
    const body = h('div.rdv-col-body',
      row('Title', text(c.title, (v) => { c.title = v; changed(); })),
      h('div.rdv-pair',
        row('Display as', select(c.displayAs, TABLE_DISPLAY, (v) => { c.displayAs = v; changed(); draw(); })),
        row('Align', select(c.alignContent, ['left', 'center', 'right'], (v) => { c.alignContent = v; changed(); }))),
      c.displayAs === 'number' && row('Number format', formatField(c.numberFormat, (v) => { c.numberFormat = v; changed(); }, NUMBER_FORMATS, '0,0[.]00')),
      c.displayAs === 'datetime' && row('Date format', formatField(c.dateTimeFormat, (v) => { c.dateTimeFormat = v; changed(); }, DATE_FORMATS, 'DD/MM/YY HH:mm')),
      c.displayAs === 'link' && [
        row('URL template', text(c.linkUrlTemplate, (v) => { c.linkUrlTemplate = v; changed(); }, { placeholder: 'https://…/{{ @ }}' }), 'Use {{ @ }} for this value or {{ column }} for others.'),
        row('Text template', text(c.linkTextTemplate, (v) => { c.linkTextTemplate = v; changed(); }, { placeholder: '{{ @ }}' })),
        check(c.linkOpenInNewTab !== false, (v) => { c.linkOpenInNewTab = v; changed(); }, 'Open in new tab')],
      c.displayAs === 'image' && row('Image URL template', text(c.imageUrlTemplate, (v) => { c.imageUrlTemplate = v; changed(); }, { placeholder: '{{ @ }}' })),
      check(c.allowSearch, (v) => { c.allowSearch = v; changed(); }, 'Use in Redash table search'),
      h('div.rdv-rules',
        h('div.rdv-section-title', 'Conditional colours'),
        c.studioRules.map((r, ri) => h('div.rdv-rule',
          select(r.op, RULE_OPS, (v) => { r.op = v; changed(); draw(); }),
          r.op !== 'empty' && text(r.value, (v) => { r.value = v; changed(); }, { placeholder: 'value' }),
          h('label.rdv-mini', 'bg', color(r.bg || '#fef3c7', (v) => { r.bg = v; changed(); })),
          h('label.rdv-mini', 'text', color(r.color || '#1e293b', (v) => { r.color = v; changed(); })),
          check(r.bold, (v) => { r.bold = v; changed(); }, 'Bold'),
          h('button.btn.btn-sm.btn-outline-danger', { type: 'button', title: 'Remove rule', onclick: () => { c.studioRules.splice(ri, 1); changed(); draw(); } }, icon('trash')))),
        h('button.btn.btn-sm.btn-outline-secondary', { type: 'button', onclick: () => { c.studioRules.push({ op: '>', value: '', bg: '#fef3c7', color: '#92400e', bold: false }); changed(); draw(); } }, icon('plus', 'me-1'), 'Add rule')));
    return h('div.rdv-col.open', head, body);
  };
  function draw() { mount(list, o.columns.map(colRow)); }
  draw();
  return h('div',
    section('Table',
      row('Rows per page', select(o.itemsPerPage || 25, [10, 15, 20, 25, 50, 100, 250], (v) => { o.itemsPerPage = Number(v); changed(); }))),
    section('Columns', h('div.rdv-help', 'Drag to reorder, click the eye to hide, the gear for formatting and colour rules.'), list));
}

// ------------------------------------------------------------------ PIVOT / FUNNEL

function pivotForm(o, cols, changed, redraw) {
  const multi = (key) => h('div.rdv-checklist', cols.map((c) => check((o[key] || []).includes(c.name), (v) => {
    o[key] = (o[key] || []).filter((x) => x !== c.name);
    if (v) { o[key].push(c.name); for (const other of ['rows', 'cols']) if (other !== key) o[other] = (o[other] || []).filter((x) => x !== c.name); }
    changed(); redraw();
  }, c.label)));
  return h('div',
    section('Layout', row('Rows', multi('rows')), row('Columns', multi('cols'))),
    section('Values',
      row('Aggregation', select(o.aggregatorName || 'Count', PIVOT_AGGREGATORS, (v) => { o.aggregatorName = v; changed(); redraw(); })),
      o.aggregatorName !== 'Count' && row('Value column', select((o.vals || [])[0], cols.map((c) => ({ value: c.name, label: c.label })), (v) => { o.vals = v ? [v] : []; changed(); }, { allowEmpty: true })),
      row('Style', select(o.rendererName || 'Table', ['Table', 'Table Heatmap'], (v) => { o.rendererName = v; changed(); }))));
}

function funnelForm(o, cols, changed) {
  o.stepCol ||= { colName: '', displayAs: 'Steps' };
  o.valueCol ||= { colName: '', displayAs: 'Value' };
  o.sortKeyCol ||= { colName: '' };
  const colOpts = cols.map((c) => ({ value: c.name, label: c.label }));
  return h('div',
    section('Steps',
      row('Step column', select(o.stepCol.colName, colOpts, (v) => { o.stepCol.colName = v; changed(); }, { allowEmpty: true })),
      row('Step title', text(o.stepCol.displayAs, (v) => { o.stepCol.displayAs = v; changed(); }))),
    section('Values',
      row('Value column', select(o.valueCol.colName, colOpts, (v) => { o.valueCol.colName = v; changed(); }, { allowEmpty: true })),
      row('Value title', text(o.valueCol.displayAs, (v) => { o.valueCol.displayAs = v; changed(); }))),
    section('Sorting',
      check(o.autoSort !== false, (v) => { o.autoSort = v; changed(); }, 'Sort steps by value automatically'),
      row('Sort by column', select(o.sortKeyCol.colName, colOpts, (v) => { o.sortKeyCol.colName = v; changed(); }, { allowEmpty: true }))));
}

/** Options as they should be stored (Studio-normalised fields only where Redash expects them). */
function serializeOptions(type, o) {
  const out = clone(o);
  if (type === 'TABLE' && out.columns) {
    out.columns = out.columns.map((c, i) => {
      const { rules, type: _t, ...rest } = c;
      return { ...rest, order: i, studioRules: (c.studioRules || []).filter((r) => r.op === 'empty' || r.value !== '') };
    });
  }
  if (type === 'CHART') {
    out.studioRefLines = (out.studioRefLines || []).filter((l) => l.value != null && l.value !== '');
    // Funnel-style autoSort maps onto Redash's sortKeyCol for the renderer.
  }
  if (type === 'FUNNEL' && out.autoSort !== false && !out.sortKeyCol?.colName) out.sortKeyCol = { colName: out.valueCol?.colName || '' };
  return out;
}

// ------------------------------------------------------------------ public API

export async function deleteViz(viz) {
  const ok = await confirmDialog({
    title: `Delete “${viz.name}”?`, danger: true, confirmLabel: 'Delete visualization',
    message: 'It is removed from the query and from every dashboard that shows it.',
  });
  if (!ok) return false;
  try {
    await rd.del(`/api/visualizations/${enc(viz.id)}`);
    toast('Visualization deleted', 'success');
    return true;
  } catch (err) {
    toastError(err, 'Delete failed');
    return false;
  }
}

/**
 * Open the editor. Resolves with the saved visualization, or null when cancelled.
 * query: needs at least {id, name, latest_query_data_id}; viz: existing visualization or null.
 */
/** Best available result for a query: stored latest, else a cached/fresh run (non-parameterized only). */
async function resultFor(query) {
  if (query?.latest_query_data_id) {
    try { return (await rd.get(`/api/query_results/${enc(query.latest_query_data_id)}`)).query_result; } catch { /* fall through */ }
  }
  if (!query?.id || (query.options?.parameters || []).length) return null;
  const wait = modal({ title: 'Getting results…', size: 'sm', icon: 'hourglass-split', persistent: true, body: h('div.d-flex.align-items-center.gap-2', spinner('sm'), h('span', 'Fetching the latest results so the preview has data…')) });
  try {
    return await runQuery({ queryId: query.id, maxAge: -1 });
  } catch (err) {
    toast(`No results to preview: ${err.message}`, 'warning', { timeout: 6000 });
    return null;
  } finally {
    wait.close();
  }
}

export async function openVizEditor({ query, result, viz, onSaved, onDeleted } = {}) {
  let res = result?.data ? result : await resultFor(query);
  const data = res?.data || { columns: [], rows: [] };
  const cols = columnsOf(res);
  const isNew = !viz?.id;
  const suggestion = isNew ? suggestViz(data) : null;
  const draft = {
    type: viz?.type || suggestion?.type || 'TABLE',
    name: viz?.name || suggestion?.name || 'Chart',
    description: viz?.description || '',
    options: clone(viz?.options) || (suggestion ? clone(suggestion.options) : {}),
  };
  if (!viz?.options && draft.type === 'TABLE') draft.options = defaultOptionsFor('TABLE', res);
  const original = JSON.stringify(draft);
  let dirty = isNew;
  const optionsCache = { [draft.type]: draft.options };

  const nameInput = h('input.form-control.rdv-name', { value: draft.name, placeholder: 'Visualization name', 'aria-label': 'Visualization name' });
  nameInput.addEventListener('input', () => { draft.name = nameInput.value; markDirty(); });
  const typeSel = h('div.rdv-types', VIZ_TYPES.map((t) => h('button.rdv-type', {
    type: 'button', dataset: { type: t.type },
    onclick: () => switchType(t.type),
  }, icon(t.icon), h('span', t.label))));
  const formEl = h('div.rdv-form');
  const previewEl = h('div.rdv-preview-body');
  const previewMeta = h('div.rdv-preview-meta');
  let handle = null;
  let renderSeq = 0;

  function markDirty() { dirty = JSON.stringify(draft) !== original || isNew; }
  const changed = () => { markDirty(); schedulePreview(); };
  const schedulePreview = debounce(async () => {
    const seq = ++renderSeq;
    try { handle?.destroy?.(); } catch { /* ignore */ }
    const v = { type: draft.type, name: draft.name, options: serializeOptions(draft.type, draft.options) };
    const target = h('div.rdv-preview-canvas');
    mount(previewEl, target);
    const hdl = await renderViz(target, v, res, { height: 380, gridMaxHeight: '420px' });
    if (seq === renderSeq) handle = hdl; else hdl?.destroy?.();
  }, 220);

  function drawForm() {
    typeSel.querySelectorAll('.rdv-type').forEach((b) => b.classList.toggle('active', b.dataset.type === draft.type));
    const o = draft.options;
    const redraw = () => drawForm();
    let form;
    switch (draft.type) {
      case 'CHART': form = chartForm(o, cols, data, changed); break;
      case 'COUNTER': form = counterForm(o, cols, changed, redraw); break;
      case 'TABLE': form = tableForm(o, res, changed, redraw); break;
      case 'PIVOT': form = pivotForm(o, cols, changed, redraw); break;
      case 'FUNNEL': form = funnelForm(o, cols, changed); break;
      default: form = h('div.rdv-help.p-2', `${vizTypeInfo(draft.type).label} has no options — it shows the result one row at a time.`);
    }
    mount(formEl, form);
  }

  function switchType(type) {
    if (type === draft.type) return;
    optionsCache[draft.type] = draft.options;
    draft.type = type;
    draft.options = optionsCache[type] || defaultOptionsFor(type, res);
    // Rename default-looking names to match the new type.
    if (!nameInput.value || VIZ_TYPES.some((t) => t.label === nameInput.value) || nameInput.value === suggestion?.name) {
      draft.name = vizTypeInfo(type).label;
      nameInput.value = draft.name;
    }
    drawForm();
    changed();
  }

  const body = h('div.rdv',
    h('div.rdv-left',
      h('div.rdv-head', nameInput, typeSel),
      formEl),
    h('div.rdv-right',
      h('div.rdv-preview-head', h('span.fw-semibold', icon('eye', 'me-1'), 'Live preview'), previewMeta),
      previewEl));

  if (!data.columns.length) {
    mount(previewEl, emptyState({ icon: 'hourglass', title: 'No results to preview', message: 'Run the query first, then come back to design the visualization.' }));
  }
  mount(previewMeta, data.columns.length ? `${data.rows.length.toLocaleString()} rows · ${data.columns.length} columns` : '');

  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => { if (!settled) { settled = true; resolve(value); } };
    const actions = [
      !isNew && {
        label: 'Delete', variant: 'outline-danger', icon: 'trash',
        onClick: async (close) => {
          if (await deleteViz(viz)) { dirty = false; onDeleted?.(viz.id); finish(null); close(); }
        },
      },
      { label: 'Cancel', onClick: (close) => attemptClose(close) },
      {
        label: isNew ? 'Create visualization' : 'Save', variant: 'primary', icon: 'check2',
        onClick: async (close) => {
          if (!draft.name.trim()) throw new Error('Give the visualization a name.');
          const payload = { type: draft.type, name: draft.name.trim(), description: draft.description, options: serializeOptions(draft.type, draft.options) };
          const saved = isNew
            ? await rd.post('/api/visualizations', { ...payload, query_id: query.id })
            : await rd.post(`/api/visualizations/${enc(viz.id)}`, payload);
          toast(isNew ? `“${saved.name}” created` : `“${saved.name}” saved`, 'success');
          dirty = false;
          onSaved?.(saved);
          finish(saved);
          close();
        },
      },
    ].filter(Boolean);

    const m = modal({
      title: isNew ? 'New visualization' : `Edit visualization`,
      subtitle: query?.name ? `${query.name}` : null,
      icon: 'bar-chart-line', size: 'xl', persistent: true,
      body, actions,
      onClose: () => { try { handle?.destroy?.(); } catch { /* ignore */ } finish(null); },
    });
    m.el.classList.add('rdv-modal');

    async function attemptClose(close) {
      if (dirty && JSON.stringify(draft) !== original) {
        const ok = await confirmDialog({ title: 'Discard changes?', message: 'Your visualization changes have not been saved.', confirmLabel: 'Discard', danger: true });
        if (!ok) return;
      }
      close();
    }
    // Header X and Escape go through the unsaved-changes check.
    const x = m.el.querySelector('.af-modal-header .af-icon-btn');
    const x2 = x.cloneNode(true);
    x.replaceWith(x2);
    x2.addEventListener('click', () => attemptClose(m.close));
    m.el.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !document.querySelector('.af-menu') && e.target.closest('.rdv-modal')) {
        e.preventDefault();
        e.stopPropagation();
        attemptClose(m.close);
      }
    });

    drawForm();
    if (data.columns.length) {
      mount(previewEl, h('div.rdv-loading', spinner(), ' Rendering…'));
      schedulePreview();
    }
    setTimeout(() => nameInput.focus(), 50);
  });
}
