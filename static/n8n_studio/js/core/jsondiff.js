// Structural diff of two JSON values: [{path: ['a', 0, 'b'], kind: 'added'|'removed'|'changed', a, b}].

const isObj = (v) => v !== null && typeof v === 'object';

export function deepDiff(a, b, path = [], out = []) {
  if (a === b) return out;
  if (Array.isArray(a) && Array.isArray(b)) {
    const n = Math.max(a.length, b.length);
    for (let i = 0; i < n; i++) {
      if (i >= a.length) out.push({ path: [...path, i], kind: 'added', b: b[i] });
      else if (i >= b.length) out.push({ path: [...path, i], kind: 'removed', a: a[i] });
      else deepDiff(a[i], b[i], [...path, i], out);
    }
    return out;
  }
  if (isObj(a) && isObj(b) && !Array.isArray(a) && !Array.isArray(b)) {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (!(k in b)) out.push({ path: [...path, k], kind: 'removed', a: a[k] });
      else if (!(k in a)) out.push({ path: [...path, k], kind: 'added', b: b[k] });
      else deepDiff(a[k], b[k], [...path, k], out);
    }
    return out;
  }
  if (Number.isNaN(a) && Number.isNaN(b)) return out;
  out.push({ path, kind: 'changed', a, b });
  return out;
}

export const pathString = (path) => path.map((p, i) => (typeof p === 'number' ? `[${p}]` : i ? `.${p}` : p)).join('') || '(root)';

/** Line diff (LCS) for text such as Code nodes: [{op: ' '|'+'|'-', text}]. */
export function lineDiff(aText, bText) {
  const a = String(aText ?? '').split('\n');
  const b = String(bText ?? '').split('\n');
  const n = a.length;
  const m = b.length;
  if (n * m > 4_000_000) return [...a.map((t) => ({ op: '-', text: t })), ...b.map((t) => ({ op: '+', text: t }))];
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  }
  const out = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push({ op: ' ', text: a[i] }); i++; j++; } else if (dp[i + 1][j] >= dp[i][j + 1]) out.push({ op: '-', text: a[i++] });
    else out.push({ op: '+', text: b[j++] });
  }
  while (i < n) out.push({ op: '-', text: a[i++] });
  while (j < m) out.push({ op: '+', text: b[j++] });
  return out;
}
