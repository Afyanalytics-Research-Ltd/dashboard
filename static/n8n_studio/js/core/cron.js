// Cron parsing, description and next-run calculation (5 or 6 fields, n8n style:
// [second] minute hour day-of-month month day-of-week).

const RANGES = [[0, 59], [0, 59], [0, 23], [1, 31], [1, 12], [0, 6]];
const NAMES = {
  jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12,
  sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5, sat: 6,
};
const MACROS = {
  '@yearly': '0 0 1 1 *', '@annually': '0 0 1 1 *', '@monthly': '0 0 1 * *', '@weekly': '0 0 * * 0',
  '@daily': '0 0 * * *', '@midnight': '0 0 * * *', '@hourly': '0 * * * *',
};
const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function parseField(src, [lo, hi], isDow) {
  const out = new Set();
  for (const part of src.toLowerCase().split(',')) {
    const [rangePart, stepPart] = part.split('/');
    const step = stepPart ? Number(stepPart) : 1;
    if (!Number.isInteger(step) || step < 1) throw new Error(`Bad step "${part}"`);
    let a;
    let b;
    if (rangePart === '*' || rangePart === '?') { a = lo; b = hi; } else {
      const [x, y] = rangePart.split('-');
      const num = (s) => (s in NAMES ? NAMES[s] : Number(s));
      a = num(x);
      b = y !== undefined ? num(y) : stepPart ? hi : a;
    }
    if (isDow) { if (a === 7) a = 0; if (b === 7) b = 6; }
    if (![a, b].every(Number.isInteger) || a < lo || b > hi || a > b) throw new Error(`Value out of range in "${part}"`);
    for (let v = a; v <= b; v += step) out.add(v);
  }
  return out;
}

function normalize(expr) {
  const text = String(expr || '').trim();
  return MACROS[text.toLowerCase()] || text;
}

export function parseCron(expr) {
  const parts = normalize(expr).split(/\s+/);
  if (parts.length !== 5 && parts.length !== 6) throw new Error('Cron needs 5 or 6 fields');
  const fields = parts.length === 5 ? ['0', ...parts] : parts;
  const sets = fields.map((f, i) => parseField(f, RANGES[i], i === 5));
  return {
    second: sets[0], minute: sets[1], hour: sets[2], dom: sets[3], month: sets[4], dow: sets[5],
    domStar: fields[3] === '*' || fields[3] === '?', dowStar: fields[5] === '*' || fields[5] === '?',
  };
}

/** Next `count` run times after `from` (viewer's local time). */
export function nextRuns(expr, count = 5, from = new Date()) {
  const c = parseCron(expr);
  const out = [];
  const d = new Date(from.getTime());
  d.setMilliseconds(0);
  d.setSeconds(d.getSeconds() + 1);
  const limit = from.getTime() + 2 * 366 * 86400 * 1000;
  while (out.length < count && d.getTime() < limit) {
    if (!c.month.has(d.getMonth() + 1)) { d.setMonth(d.getMonth() + 1, 1); d.setHours(0, 0, 0); continue; }
    const domOk = c.dom.has(d.getDate());
    const dowOk = c.dow.has(d.getDay());
    let dayOk;
    if (c.domStar && c.dowStar) dayOk = true;
    else if (c.domStar) dayOk = dowOk;
    else if (c.dowStar) dayOk = domOk;
    else dayOk = domOk || dowOk;
    if (!dayOk) { d.setDate(d.getDate() + 1); d.setHours(0, 0, 0); continue; }
    if (!c.hour.has(d.getHours())) { d.setHours(d.getHours() + 1, 0, 0); continue; }
    if (!c.minute.has(d.getMinutes())) { d.setMinutes(d.getMinutes() + 1, 0); continue; }
    if (!c.second.has(d.getSeconds())) { d.setSeconds(d.getSeconds() + 1); continue; }
    out.push(new Date(d.getTime()));
    d.setSeconds(d.getSeconds() + 1);
  }
  return out;
}

export function describeCron(expr) {
  try {
    const text = normalize(expr);
    const p = text.split(/\s+/);
    const [s, m, h, dom, mon, dow] = p.length === 5 ? ['0', ...p] : p;
    const every = (x) => /^\*\/(\d+)$/.exec(x)?.[1];
    if (every(s) && m === '*') return `Every ${every(s)} seconds`;
    if (s === '0' && every(m) && h === '*' && dom === '*' && mon === '*' && dow === '*') return `Every ${every(m)} minutes`;
    if (s === '0' && m === '*' && h === '*') return 'Every minute';
    if (s === '0' && /^\d+$/.test(m) && every(h)) return `Every ${every(h)} hours at :${m.padStart(2, '0')}`;
    if (s === '0' && /^\d+$/.test(m) && h === '*') return `Hourly at :${m.padStart(2, '0')}`;
    if (/^\d+$/.test(m) && /^\d+(,\d+)*$/.test(h)) {
      const times = h.split(',').map((x) => `${x.padStart(2, '0')}:${m.padStart(2, '0')}`).join(', ');
      if (dom === '*' && mon === '*' && dow === '*') return `Daily at ${times}`;
      if (dom === '*' && mon === '*' && dow === '1-5') return `Weekdays at ${times}`;
      if (dom === '*' && mon === '*') return `${[...parseField(dow, RANGES[5], true)].map((d) => DAY_NAMES[d]).join(', ')} at ${times}`;
      if (/^\d+$/.test(dom) && mon === '*' && dow === '*') return `Monthly on day ${dom} at ${times}`;
    }
    return `Cron: ${text}`;
  } catch {
    return `Cron: ${expr}`;
  }
}
