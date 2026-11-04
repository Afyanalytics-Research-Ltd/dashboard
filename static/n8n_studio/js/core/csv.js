// RFC 4180 CSV parse/serialize. Strips a UTF-8 BOM (n8n's own data-table export adds
// one, which breaks its re-import) and sniffs , ; tab | delimiters.

export function detectDelimiter(text) {
  const sample = text.slice(0, 4096).split(/\r?\n/).slice(0, 5);
  let best = ',';
  let bestScore = -1;
  for (const d of [',', ';', '\t', '|']) {
    const counts = sample.map((line) => line.split(d).length - 1);
    const consistent = counts.length && counts.every((c) => c === counts[0]) ? 2 : 1;
    const score = (counts[0] || 0) * consistent;
    if (score > bestScore) { best = d; bestScore = score; }
  }
  return best;
}

export function parseCSV(input, { delimiter } = {}) {
  let text = String(input ?? '');
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const d = delimiter || detectDelimiter(text);
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"' && field === '') quoted = true;
    else if (c === d) { row.push(field); field = ''; } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => !(r.length === 1 && r[0] === ''));
}

/** CSV text -> {headers, records: [{header: value}]} */
export function csvToRecords(text, opts) {
  const rows = parseCSV(text, opts);
  if (!rows.length) return { headers: [], records: [] };
  const headers = rows[0].map((h, i) => h.trim() || `column_${i + 1}`);
  const records = rows.slice(1).map((r) => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ''])));
  return { headers, records };
}

const needsQuote = (s, d) => s.includes(d) || s.includes('"') || s.includes('\n') || s.includes('\r') || /^\s|\s$/.test(s);

export function toCSV(records, headers, { delimiter = ',', bom = false } = {}) {
  const cols = headers || [...new Set(records.flatMap((r) => Object.keys(r)))];
  const cell = (v) => {
    if (v == null) return '';
    const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
    return needsQuote(s, delimiter) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [cols.map(cell).join(delimiter), ...records.map((r) => cols.map((c) => cell(r[c])).join(delimiter))];
  return (bom ? '﻿' : '') + lines.join('\r\n');
}

/** Guess a data-table column type from sample values. */
export function inferType(values) {
  const vals = values.filter((v) => v !== '' && v != null);
  if (!vals.length) return 'string';
  if (vals.every((v) => /^(true|false)$/i.test(String(v)))) return 'boolean';
  if (vals.every((v) => /^-?\d+(\.\d+)?$/.test(String(v).trim()))) return 'number';
  if (vals.every((v) => /^\d{4}-\d{2}-\d{2}([T ][\d:.]+(Z|[+-]\d{2}:?\d{2})?)?$/.test(String(v).trim()))) return 'date';
  return 'string';
}

export function coerce(value, type) {
  if (value === '' || value == null) return null;
  switch (type) {
    case 'number': { const n = Number(value); return Number.isFinite(n) ? n : null; }
    case 'boolean': return /^(true|1|yes)$/i.test(String(value));
    case 'date': { const d = new Date(value); return Number.isNaN(d.getTime()) ? null : d.toISOString(); }
    default: return String(value);
  }
}

/** Columns n8n manages itself; importing them as user columns fails. */
export const RESERVED_COLUMNS = ['id', 'createdAt', 'updatedAt'];
