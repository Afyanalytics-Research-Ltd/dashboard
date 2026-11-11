// Date, duration and number formatting. All API dates are ISO-8601 UTC.

const pad = (n) => String(n).padStart(2, '0');

export function toDate(value) {
  if (!value) return null;
  const d = value instanceof Date ? value : new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** 2026-11-02 08:10:23 in the viewer's local time. */
export function fmtDate(value, { seconds = true } = {}) {
  const d = toDate(value);
  if (!d) return '—';
  const base = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return seconds ? `${base}:${pad(d.getSeconds())}` : base;
}

export function fmtShortDate(value) {
  const d = toDate(value);
  if (!d) return '—';
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function fmtDay(value) {
  const d = toDate(value);
  return d ? `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` : '';
}

/** "3m ago", "in 2h", "just now". */
export function relTime(value, now = Date.now()) {
  const d = toDate(value);
  if (!d) return '—';
  const diff = (d.getTime() - now) / 1000;
  const abs = Math.abs(diff);
  const units = [[60, 's', 1], [3600, 'm', 60], [86400, 'h', 3600], [86400 * 30, 'd', 86400], [86400 * 365, 'mo', 86400 * 30], [Infinity, 'y', 86400 * 365]];
  if (abs < 10) return 'just now';
  for (const [limit, unit, div] of units) {
    if (abs < limit) {
      const n = Math.floor(abs / div);
      return diff < 0 ? `${n}${unit} ago` : `in ${n}${unit}`;
    }
  }
  return fmtDate(d);
}

/** Seconds -> "1h 02m 03s" / "4.2s" / "—". */
export function fmtDuration(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return '—';
  let s = Number(seconds);
  if (s < 0) s = 0;
  if (s < 10) return `${s.toFixed(s < 1 ? 2 : 1)}s`;
  s = Math.round(s);
  const d = Math.floor(s / 86400);
  const hh = Math.floor((s % 86400) / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (d) return `${d}d ${hh}h ${pad(mm)}m`;
  if (hh) return `${hh}h ${pad(mm)}m ${pad(ss)}s`;
  if (mm) return `${mm}m ${pad(ss)}s`;
  return `${ss}s`;
}

/** Duration of something that may still be running. */
export function liveDuration(start, end, fallback) {
  if (fallback != null) return fallback;
  const a = toDate(start);
  if (!a) return null;
  const b = toDate(end) || new Date();
  return (b - a) / 1000;
}

export function fmtNumber(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString();
}

export function fmtPercent(part, total, digits = 0) {
  if (!total) return '—';
  return `${((part / total) * 100).toFixed(digits)}%`;
}

export function fmtBytes(n) {
  if (n == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = Number(n);
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
}

/** ISO string for a <input type="datetime-local"> value (local time). */
export function toLocalInput(value) {
  const d = toDate(value);
  if (!d) return '';
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** <input type="datetime-local"> value -> UTC ISO string (or null). */
export function fromLocalInput(value) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

export function truncate(text, max = 80) {
  if (text == null) return '';
  const str = String(text);
  return str.length > max ? `${str.slice(0, max - 1)}…` : str;
}

export function pluralize(n, word, plural = `${word}s`) {
  return `${fmtNumber(n)} ${n === 1 ? word : plural}`;
}
