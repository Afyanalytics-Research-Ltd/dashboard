// Value formatting compatible with Redash's numeral.js / moment-style formats.

const pad = (n) => String(n).padStart(2, '0');

/**
 * Minimal numeral.js-compatible formatter: '0,0', '0,0.00', '0[.]00', '0.0%', '$0,0.[00]', '0.0a'.
 * Unknown formats fall back to locale formatting.
 */
export function fmtNumeral(value, format = '0,0[.]00') {
  if (value == null || value === '') return '';
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  let f = String(format || '0,0[.]00');
  const prefix = (f.match(/^[^0#.,[\]%a]*/) || [''])[0];
  const suffix = (f.match(/[^0#.,[\]%a]*$/) || [''])[0];
  f = f.slice(prefix.length, f.length - suffix.length || undefined);
  let n = num;
  let unit = '';
  if (f.includes('%')) { n *= 100; unit = '%'; f = f.replace('%', ''); }
  if (f.includes('a')) {
    const abs = Math.abs(n);
    for (const [lim, u] of [[1e12, 't'], [1e9, 'b'], [1e6, 'm'], [1e3, 'k']]) {
      if (abs >= lim) { n /= lim; unit = u + unit; break; }
    }
    f = f.replace(/\s?a/, '');
  }
  const optional = /\[\.\]|\.\[/.test(f);
  const decPart = (f.split(/\[?\.\]?/)[1] || '').replace(/[[\]]/g, '');
  const decimals = decPart.length;
  const grouping = f.includes(',');
  let out = n.toLocaleString('en-US', {
    minimumFractionDigits: optional ? 0 : decimals,
    maximumFractionDigits: decimals,
    useGrouping: grouping,
  });
  return `${prefix}${out}${unit}${suffix}`;
}

/** moment-like date format subset: DD/MM/YY, YYYY-MM-DD HH:mm, etc. */
export function fmtMoment(value, format = 'DD/MM/YY HH:mm') {
  if (value == null || value === '') return '';
  const d = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  const map = {
    YYYY: d.getFullYear(), YY: String(d.getFullYear()).slice(-2), MMM: d.toLocaleString('en-US', { month: 'short' }),
    MM: pad(d.getMonth() + 1), DD: pad(d.getDate()), HH: pad(d.getHours()), mm: pad(d.getMinutes()), ss: pad(d.getSeconds()),
  };
  return format.replace(/YYYY|YY|MMM|MM|DD|HH|mm|ss/g, (t) => map[t]);
}

/** Convert a numeral format to a d3-format string for Plotly hover/tick labels. */
export function numeralToD3(format) {
  if (!format) return '';
  const pct = format.includes('%');
  const dec = ((format.split(/\[?\.\]?/)[1] || '').replace(/[[\]%a\s]/g, '')).length;
  const grouping = format.includes(',') ? ',' : '';
  const trim = /\[\.\]|\.\[/.test(format) ? '~' : '';
  return `${grouping}.${dec}${trim}${pct ? '%' : 'f'}`;
}

const NUMERIC = new Set(['integer', 'float', 'number']);
export const isNumericType = (t) => NUMERIC.has(t);
export const isDateType = (t) => t === 'date' || t === 'datetime';

/** Guess a column type from its values when Redash reports none / string. */
export function inferType(rows, name) {
  let seen = 0;
  let nums = 0;
  let dates = 0;
  for (const r of rows) {
    const v = r[name];
    if (v == null || v === '') continue;
    seen++;
    if (typeof v === 'number') nums++;
    else if (typeof v === 'string' && /^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2})?/.test(v)) dates++;
    if (seen >= 50) break;
  }
  if (!seen) return 'string';
  if (nums === seen) return 'float';
  if (dates === seen) return 'datetime';
  return 'string';
}

/** Default display of a cell given the column type and org settings. */
export function fmtCell(value, type, settings = {}) {
  if (value == null) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  if (type === 'integer') return fmtNumeral(value, settings.integer_format || '0,0');
  if (type === 'float') return fmtNumeral(value, settings.float_format || '0,0.00');
  if (type === 'boolean') return value ? 'true' : 'false';
  if (type === 'datetime') return fmtMoment(value, `${settings.date_format || 'DD/MM/YY'} ${settings.time_format || 'HH:mm'}`);
  if (type === 'date') return fmtMoment(value, settings.date_format || 'DD/MM/YY');
  return String(value);
}
