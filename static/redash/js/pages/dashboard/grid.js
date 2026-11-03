// Dashboard grid: Redash-compatible 6-column layout (options.position {col,row,sizeX,sizeY}),
// absolute positioning with animated reflow, vertical compaction, and drag / resize editing
// that snaps to the grid and pushes colliding widgets down.

import { h, icon } from '../../../../ui/js/dom.js';

export const COLS = 6;
export const ROW_H = 50;
export const GAP = 15;
export const MOBILE_BP = 800;

const overlaps = (a, b) => a.col < b.col + b.sizeX && b.col < a.col + a.sizeX && a.row < b.row + b.sizeY && b.row < a.row + a.sizeY;

/** Clamp a position into the grid using the widget's min/max sizes. */
export function clampPos(p) {
  const minX = Math.max(1, p.minSizeX || 1);
  const maxX = Math.min(COLS, p.maxSizeX || COLS);
  const minY = Math.max(1, p.minSizeY || 1);
  const maxY = p.maxSizeY || 1000;
  const sizeX = Math.min(Math.max(Math.round(p.sizeX ?? 3), minX), maxX);
  const sizeY = Math.min(Math.max(Math.round(p.sizeY ?? 8), minY), maxY);
  const col = Math.min(Math.max(Math.round(p.col || 0), 0), COLS - sizeX);
  const row = Math.max(Math.round(p.row || 0), 0);
  return { ...p, col, row, sizeX, sizeY };
}

/**
 * Positions for every widget. Widgets without a position (older Redash dashboards) are
 * auto-placed in reading order: charts 3×8, tables 6×10, counters 2×5, text 6×3.
 */
export function initialLayout(widgets) {
  const layout = new Map();
  const unplaced = [];
  for (const w of widgets) {
    const p = w.options?.position;
    if (p && p.sizeX != null && p.sizeY != null && p.col != null && p.row != null) layout.set(w.id, clampPos(p));
    else unplaced.push(w);
  }
  for (const w of unplaced) {
    const t = w.visualization?.type;
    const size = !w.visualization ? { sizeX: 6, sizeY: 3 } : t === 'TABLE' ? { sizeX: 6, sizeY: 10 } : t === 'COUNTER' ? { sizeX: 2, sizeY: 5 } : { sizeX: 3, sizeY: 8 };
    layout.set(w.id, findSlot(layout, { ...size, minSizeX: 1, maxSizeX: 6, minSizeY: 1, maxSizeY: 1000, autoHeight: false }));
  }
  return compact(layout);
}

/** First free slot scanning rows top-to-bottom, columns left-to-right. */
function findSlot(layout, size) {
  const placed = [...layout.values()];
  for (let row = 0; row < 2000; row++) {
    for (let col = 0; col <= COLS - size.sizeX; col++) {
      const cand = { ...size, col, row };
      if (!placed.some((p) => overlaps(p, cand))) return cand;
    }
  }
  return { ...size, col: 0, row: 0 };
}

/**
 * Gravity compaction: every widget floats up as far as it can. `priority` (an id) is placed
 * first at its row so the widget being dragged wins its spot and others flow around it.
 */
export function compact(layout, priority, depth = 0) {
  const entries = [...layout.entries()].map(([id, p]) => [id, clampPos(p)]);
  const placed = [];
  const out = new Map();
  const fixed = priority != null && layout.has(priority) ? entries.find(([id]) => id === priority) : null;
  if (fixed) {
    const cand = { ...fixed[1] };
    placed.push(cand);
    out.set(priority, cand);
  }
  entries
    .filter(([id]) => !fixed || id !== priority)
    .sort(([, a], [, b]) => a.row - b.row || a.col - b.col)
    .forEach(([id, p]) => {
      const cand = { ...p, row: 0 };
      while (placed.some((q) => overlaps(q, cand))) cand.row++;
      placed.push(cand);
      out.set(id, cand);
    });
  const result = new Map([...layout.keys()].map((id) => [id, out.get(id)]));
  if (fixed) {
    // Let the dragged widget float up into any gap left above it, then re-flow the others
    // around its final spot (repeat until stable so no gaps remain).
    const cand = out.get(priority);
    const others = placed.filter((q) => q !== cand);
    const startRow = cand.row;
    while (cand.row > 0 && !others.some((q) => overlaps(q, { ...cand, row: cand.row - 1 }))) cand.row--;
    if (cand.row !== startRow && depth < 4) return compact(result, priority, depth + 1);
  }
  return result;
}

export const layoutHeight = (layout) => Math.max(0, ...[...layout.values()].map((p) => p.row + p.sizeY));

/**
 * Mount the grid. items: [{id, el}] (el = widget card element).
 * opts: { layout: Map, editing, onChange(layout) }
 * Returns { setEditing, setLayout, getLayout, relayout, add(item), remove(id), destroy }
 */
export function gridView(container, items, opts = {}) {
  let layout = opts.layout || new Map();
  let editing = !!opts.editing;
  const cells = new Map();
  const canvas = h('div.rdd-canvas');
  const ghost = h('div.rdd-ghost', { hidden: true });
  canvas.appendChild(ghost);
  container.appendChild(canvas);

  const metrics = () => {
    const w = canvas.clientWidth || container.clientWidth || 1000;
    return { width: w, colW: (w - GAP * (COLS - 1)) / COLS, mobile: w < MOBILE_BP };
  };
  const rect = (p, m) => ({
    left: p.col * (m.colW + GAP),
    top: p.row * (ROW_H + GAP),
    width: p.sizeX * m.colW + (p.sizeX - 1) * GAP,
    height: p.sizeY * ROW_H + (p.sizeY - 1) * GAP,
  });

  function place(cell, p, m) {
    const r = rect(p, m);
    Object.assign(cell.style, { transform: `translate(${r.left}px, ${r.top}px)`, width: `${r.width}px`, height: `${r.height}px` });
  }

  function relayout() {
    const m = metrics();
    canvas.classList.toggle('mobile', m.mobile);
    canvas.classList.toggle('editing', editing && !m.mobile);
    if (m.mobile) {
      // Single column, reading order, natural heights.
      const order = [...cells.keys()].sort((a, b) => {
        const pa = layout.get(a); const pb = layout.get(b);
        return pa.row - pb.row || pa.col - pb.col;
      });
      order.forEach((id, i) => {
        const cell = cells.get(id);
        const p = layout.get(id);
        cell.style.order = i;
        Object.assign(cell.style, { transform: '', width: '', height: `${Math.max(p.sizeY * ROW_H * 0.8, 160)}px` });
      });
      canvas.style.height = '';
      return;
    }
    for (const [id, cell] of cells) place(cell, layout.get(id), m);
    canvas.style.height = `${Math.max(layoutHeight(layout) * (ROW_H + GAP) - GAP, ROW_H)}px`;
  }

  function addCell({ id, el }) {
    const cell = h('div.rdd-cell', { dataset: { id } }, el,
      h('div.rdd-resize', { title: 'Drag to resize', 'aria-hidden': 'true' }, icon('arrows-angle-expand')));
    cells.set(id, cell);
    canvas.appendChild(cell);
    if (!layout.has(id)) layout.set(id, findSlot(layout, { sizeX: 3, sizeY: 8, minSizeX: 1, maxSizeX: 6, minSizeY: 1, maxSizeY: 1000 }));
    bindDrag(id, cell);
  }

  // ---------------------------------------------------------------- drag & resize

  function bindDrag(id, cell) {
    cell.addEventListener('pointerdown', (e) => {
      if (!editing || metrics().mobile || e.button !== 0) return;
      const resize = e.target.closest('.rdd-resize');
      const handle = e.target.closest('.rdd-drag');
      if (!resize && !handle) return;
      if (e.target.closest('button:not(.rdd-drag), a, input, select, textarea')) return;
      e.preventDefault();
      start(id, cell, e, resize ? 'resize' : 'move');
    });
  }

  function start(id, cell, e, mode) {
    const m = metrics();
    const origin = { ...layout.get(id) };
    const before = new Map(layout);
    const startRect = rect(origin, m);
    const sx = e.clientX;
    const sy = e.clientY;
    cell.setPointerCapture(e.pointerId);
    cell.classList.add(mode === 'move' ? 'dragging' : 'resizing');
    canvas.classList.add('interacting');
    ghost.hidden = false;
    place(ghost, origin, m);
    let target = origin;
    let raf = 0;

    const onMove = (ev) => {
      const dx = ev.clientX - sx;
      const dy = ev.clientY - sy;
      if (mode === 'move') {
        cell.style.transform = `translate(${startRect.left + dx}px, ${startRect.top + dy}px)`;
        target = clampPos({ ...origin, col: origin.col + Math.round(dx / (m.colW + GAP)), row: origin.row + Math.round(dy / (ROW_H + GAP)) });
      } else {
        const w = Math.max(startRect.width + dx, m.colW * 0.5);
        const hgt = Math.max(startRect.height + dy, ROW_H * 0.5);
        cell.style.width = `${w}px`;
        cell.style.height = `${hgt}px`;
        target = clampPos({ ...origin, sizeX: Math.round((w + GAP) / (m.colW + GAP)), sizeY: Math.round((hgt + GAP) / (ROW_H + GAP)) });
        if (target.col + target.sizeX > COLS) target.sizeX = COLS - target.col;
      }
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const next = new Map(before);
        next.set(id, target);
        layout = compact(next, id);
        place(ghost, layout.get(id), m);
        for (const [oid, c] of cells) if (oid !== id) place(c, layout.get(oid), m);
        canvas.style.height = `${Math.max(layoutHeight(layout) * (ROW_H + GAP) - GAP, ROW_H)}px`;
      });
      // Auto-scroll near the viewport edges while dragging.
      if (ev.clientY > window.innerHeight - 60) window.scrollBy(0, 18);
      else if (ev.clientY < 80) window.scrollBy(0, -18);
    };
    const onUp = () => {
      cancelAnimationFrame(raf);
      cell.removeEventListener('pointermove', onMove);
      cell.removeEventListener('pointerup', onUp);
      cell.removeEventListener('pointercancel', onUp);
      cell.classList.remove('dragging', 'resizing');
      canvas.classList.remove('interacting');
      ghost.hidden = true;
      const next = new Map(before);
      next.set(id, target);
      layout = compact(next, id);
      relayout();
      const changed = [...layout].some(([k, p]) => {
        const b = before.get(k);
        return !b || b.col !== p.col || b.row !== p.row || b.sizeX !== p.sizeX || b.sizeY !== p.sizeY;
      });
      if (changed) {
        cell.classList.add('dropped');
        setTimeout(() => cell.classList.remove('dropped'), 400);
        opts.onChange?.(layout, { id, mode });
      }
      // Charts need to know their box changed.
      cell.dispatchEvent(new CustomEvent('rdd:resized', { bubbles: true }));
    };
    cell.addEventListener('pointermove', onMove);
    cell.addEventListener('pointerup', onUp);
    cell.addEventListener('pointercancel', onUp);
  }

  // ---------------------------------------------------------------- keyboard nudging (accessibility)

  canvas.addEventListener('keydown', (e) => {
    if (!editing) return;
    const cell = e.target.closest('.rdd-cell');
    if (!cell || !e.target.classList.contains('rdd-drag')) return;
    const id = Number(cell.dataset.id) || cell.dataset.id;
    const p = layout.get(id);
    const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (!step || !p) return;
    e.preventDefault();
    const next = new Map(layout);
    next.set(id, clampPos(e.shiftKey ? { ...p, sizeX: p.sizeX + step[0], sizeY: p.sizeY + step[1] } : { ...p, col: p.col + step[0], row: p.row + step[1] }));
    layout = compact(next, id);
    relayout();
    opts.onChange?.(layout, { id, mode: e.shiftKey ? 'resize' : 'move' });
  });

  // Only react to real width changes (ignore height churn and scrollbar flicker < 20px),
  // otherwise re-rendering charts can feed back into the observer.
  let lastWidth = 0;
  const ro = new ResizeObserver(() => {
    const w = canvas.clientWidth;
    if (Math.abs(w - lastWidth) < 20) return;
    lastWidth = w;
    relayout();
    for (const cell of cells.values()) cell.dispatchEvent(new CustomEvent('rdd:resized', { bubbles: true }));
  });
  ro.observe(canvas);

  items.forEach(addCell);
  layout = compact(layout);
  relayout();

  return {
    canvas,
    setEditing(on) { editing = on; relayout(); },
    getLayout: () => new Map(layout),
    setLayout(next) { layout = compact(new Map(next)); relayout(); },
    relayout,
    add(item, pos) {
      if (pos) layout.set(item.id, clampPos(pos));
      addCell(item);
      layout = compact(layout, item.id);
      relayout();
      cells.get(item.id).classList.add('entering');
      setTimeout(() => cells.get(item.id)?.classList.remove('entering'), 500);
    },
    remove(id) {
      const cell = cells.get(id);
      if (!cell) return;
      cell.classList.add('leaving');
      setTimeout(() => {
        cell.remove();
        cells.delete(id);
        layout.delete(id);
        layout = compact(layout);
        relayout();
      }, 220);
    },
    cell: (id) => cells.get(id),
    destroy() { ro.disconnect(); canvas.remove(); },
  };
}
