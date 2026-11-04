// Small fuzzy matcher for palettes: exact > prefix > substring > subsequence, with
// bonuses for word starts and contiguous runs. score() is -Infinity on no match.

const SEP = /[\s\-_./:]/;

export function score(query, text) {
  const q = String(query || '').toLowerCase().trim();
  const t = String(text || '').toLowerCase();
  if (!q) return 0;
  if (t === q) return 1000;
  if (t.startsWith(q)) return 800 - t.length;
  const idx = t.indexOf(q);
  if (idx >= 0) return 600 - idx - t.length / 10 + (SEP.test(t[idx - 1] || ' ') ? 100 : 0);
  let s = 0;
  let ti = 0;
  let run = 0;
  for (const ch of q) {
    if (ch === ' ') continue;
    const found = t.indexOf(ch, ti);
    if (found < 0) return -Infinity;
    run = found === ti ? run + 1 : 0;
    s += 10 + run * 5 + (found === 0 || SEP.test(t[found - 1]) ? 15 : 0) - Math.min(found - ti, 10);
    ti = found + 1;
  }
  return s - t.length / 20;
}

/** Rank items by the best score over keys(item); empty query keeps the order. */
export function rank(items, query, keys, limit = 50) {
  if (!String(query || '').trim()) return items.slice(0, limit);
  const scored = [];
  for (const item of items) {
    let best = -Infinity;
    for (const k of keys(item)) if (k) best = Math.max(best, score(query, k));
    if (best > -Infinity) scored.push([best, item]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, limit).map((x) => x[1]);
}
