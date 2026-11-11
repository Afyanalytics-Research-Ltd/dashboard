// Undo/redo stack of JSON snapshots with coalescing: rapid edits sharing a `group`
// (typing in one field, dragging one node) merge into a single undo step.

export class UndoStack {
  constructor({ limit = 200, coalesceMs = 800 } = {}) {
    this.limit = limit;
    this.coalesceMs = coalesceMs;
    this.past = [];
    this.future = [];
    this.lastGroup = null;
    this.lastAt = 0;
  }

  /** Record `state` — the state BEFORE a change. Returns false when coalesced. */
  push(state, { group = null, now = Date.now() } = {}) {
    const snapshot = typeof state === 'string' ? state : JSON.stringify(state);
    if (group && group === this.lastGroup && now - this.lastAt < this.coalesceMs) {
      this.lastAt = now;
      return false;
    }
    if (this.past[this.past.length - 1] !== snapshot) this.past.push(snapshot);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
    this.lastGroup = group;
    this.lastAt = now;
    return true;
  }

  undo(current) {
    if (!this.past.length) return null;
    this.future.push(typeof current === 'string' ? current : JSON.stringify(current));
    this.lastGroup = null;
    return JSON.parse(this.past.pop());
  }

  redo(current) {
    if (!this.future.length) return null;
    this.past.push(typeof current === 'string' ? current : JSON.stringify(current));
    this.lastGroup = null;
    return JSON.parse(this.future.pop());
  }

  get canUndo() { return this.past.length > 0; }

  get canRedo() { return this.future.length > 0; }

  clear() {
    this.past = [];
    this.future = [];
    this.lastGroup = null;
  }
}
