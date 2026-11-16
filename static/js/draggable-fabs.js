/* ==========================================================================
   DRAGGABLE FABS — lets the user drag the chatbot and feedback buttons
   anywhere on screen so they never cover what's behind them.

   - Drag with mouse, pen or touch; a plain tap/click still works as before.
   - The position is remembered per browser (localStorage) and kept on screen
     when the window is resized.
   - The feedback speed-dial opens away from the nearest screen edges.
   ========================================================================== */
(function () {
  'use strict';

  var DRAG_THRESHOLD = 6;   // px of movement before a press becomes a drag
  var MARGIN = 8;           // keep at least this much of a gap from the edges

  function load(key) {
    try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch (e) { return null; }
  }
  function save(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) { /* storage off */ }
  }

  /**
   * handle: the button the user presses.
   * box:    the fixed-position element that moves (the button itself, or its wrapper).
   * key:    localStorage key.
   * flip:   add edge classes so a menu inside `box` opens towards the screen centre.
   */
  function makeDraggable(handle, box, key, flip) {
    if (!handle || !box) return;
    handle.style.touchAction = 'none';

    // Saved position = centre of the handle as a fraction of the viewport,
    // so it survives window resizes sensibly.
    var pos = load(key);

    function place(cx, cy) {
      var r = handle.getBoundingClientRect();
      var w = window.innerWidth, h = window.innerHeight;
      // clamp so the whole button stays visible
      cx = Math.min(Math.max(cx, r.width / 2 + MARGIN), w - r.width / 2 - MARGIN);
      cy = Math.min(Math.max(cy, r.height / 2 + MARGIN), h - r.height / 2 - MARGIN);

      var left = cx < w / 2, top = cy < h / 2;
      if (flip) {
        box.classList.toggle('fab-at-left', left);
        box.classList.toggle('fab-at-top', top);
      }
      // Anchor on the handle's edge nearest the screen edge; the wrapper's
      // menu then grows towards the centre.
      if (left) { box.style.left = (cx - r.width / 2) + 'px'; box.style.right = 'auto'; }
      else      { box.style.right = (w - cx - r.width / 2) + 'px'; box.style.left = 'auto'; }
      if (top)  { box.style.top = (cy - r.height / 2) + 'px'; box.style.bottom = 'auto'; }
      else      { box.style.bottom = (h - cy - r.height / 2) + 'px'; box.style.top = 'auto'; }
      return { x: cx / w, y: cy / h };
    }

    function restore() {
      if (pos) place(pos.x * window.innerWidth, pos.y * window.innerHeight);
    }
    restore();
    window.addEventListener('resize', restore);

    var start = null, dragging = false, suppressClick = false;

    handle.addEventListener('pointerdown', function (e) {
      if (e.button !== 0) return;
      var r = handle.getBoundingClientRect();
      start = { x: e.clientX, y: e.clientY,
                dx: e.clientX - (r.left + r.width / 2),
                dy: e.clientY - (r.top + r.height / 2), id: e.pointerId };
      dragging = false;
    });

    // Listen on the window: a quick flick leaves the button before the drag starts.
    window.addEventListener('pointermove', function (e) {
      if (!start || e.pointerId !== start.id) return;
      if (!dragging) {
        if (Math.abs(e.clientX - start.x) + Math.abs(e.clientY - start.y) < DRAG_THRESHOLD) return;
        dragging = true;
        try { handle.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
        box.classList.add('fab-dragging');
      }
      e.preventDefault();
      pos = place(e.clientX - start.dx, e.clientY - start.dy);
    });

    function end(e) {
      if (!start || (e && e.pointerId !== start.id)) return;
      if (dragging) {
        box.classList.remove('fab-dragging');
        save(key, pos);
        suppressClick = true;           // the click that follows a drag is not a tap
        setTimeout(function () { suppressClick = false; }, 0);
      }
      start = null;
      dragging = false;
    }
    window.addEventListener('pointerup', end);
    window.addEventListener('pointercancel', end);

    // Capture phase on the button runs before chat.js / tickets.js handlers.
    handle.addEventListener('click', function (e) {
      if (suppressClick) {
        suppressClick = false;
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    }, true);

    handle.title = (handle.title ? handle.title + ' — ' : '') + 'drag to move';
  }

  makeDraggable(document.getElementById('chatbotFab'),
                document.getElementById('chatbotFab'), 'afya.fabPos.chatbot', false);
  makeDraggable(document.getElementById('ticketFab'),
                document.getElementById('ticketFabWrap'), 'afya.fabPos.feedback', true);
})();
