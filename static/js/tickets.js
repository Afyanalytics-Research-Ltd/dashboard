/* ==========================================================================
   TICKETING SYSTEM — feedback FAB speed-dial, the 3 ticket modals, and
   (on the Support & Ticketing page) the staff Kanban board.
   Depends on: Bootstrap 5 JS, AfyaToast (main.js).
   ========================================================================== */
(function () {
  'use strict';

  var TICKET_CREATE_URL = document.body.dataset.ticketCreateUrl;

  // ---------------------------------------------------------------------
  // FAB speed-dial
  // ---------------------------------------------------------------------
  var fabWrap = document.getElementById('ticketFabWrap');
  var fabBtn = document.getElementById('ticketFab');

  function closeDial() {
    if (fabWrap) fabWrap.classList.remove('dial-open');
    if (fabBtn) fabBtn.classList.remove('dial-open');
  }

  if (fabBtn && fabWrap) {
    fabBtn.addEventListener('click', function () {
      var open = fabWrap.classList.toggle('dial-open');
      fabBtn.classList.toggle('dial-open', open);
    });
    document.addEventListener('click', function (e) {
      if (!fabWrap.contains(e.target)) closeDial();
    });
    document.querySelectorAll('.ticket-dial-item').forEach(function (item) {
      item.addEventListener('click', function () {
        closeDial();
        var targetModal = document.getElementById(item.dataset.modalTarget);
        if (targetModal) new bootstrap.Modal(targetModal).show();
      });
    });
  }

  // ---------------------------------------------------------------------
  // Character counters
  // ---------------------------------------------------------------------
  document.querySelectorAll('[data-char-counter]').forEach(function (textarea) {
    var max = parseInt(textarea.getAttribute('maxlength') || '2000', 10);
    var counter = document.querySelector(textarea.dataset.charCounter);
    if (!counter) return;
    function update() {
      var remaining = max - textarea.value.length;
      counter.textContent = textarea.value.length + ' / ' + max;
      counter.classList.toggle('near-limit', remaining < max * 0.1);
    }
    textarea.addEventListener('input', update);
    update();
  });

  // ---------------------------------------------------------------------
  // Priority chips
  // ---------------------------------------------------------------------
  document.querySelectorAll('.priority-chips').forEach(function (group) {
    var hidden = document.querySelector(group.dataset.hiddenInput);
    group.querySelectorAll('.priority-chip').forEach(function (chip) {
      chip.addEventListener('click', function () {
        group.querySelectorAll('.priority-chip').forEach(function (c) { c.classList.remove('active'); });
        chip.classList.add('active');
        if (hidden) hidden.value = chip.dataset.priority;
      });
    });
  });

  // ---------------------------------------------------------------------
  // File dropzone (Issue modal attachment)
  // ---------------------------------------------------------------------
  document.querySelectorAll('.ticket-dropzone').forEach(function (zone) {
    var input = zone.querySelector('input[type="file"]');
    var preview = document.querySelector(zone.dataset.preview);
    if (!input) return;

    zone.addEventListener('click', function () { input.click(); });
    ['dragenter', 'dragover'].forEach(function (evt) {
      zone.addEventListener(evt, function (e) { e.preventDefault(); zone.classList.add('drag-over'); });
    });
    ['dragleave', 'drop'].forEach(function (evt) {
      zone.addEventListener(evt, function (e) { e.preventDefault(); zone.classList.remove('drag-over'); });
    });
    zone.addEventListener('drop', function (e) {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event('change'));
      }
    });
    input.addEventListener('change', function () {
      if (!input.files.length || !preview) return;
      var file = input.files[0];
      var img = preview.querySelector('img');
      var name = preview.querySelector('.dz-filename');
      if (name) name.textContent = file.name;
      if (img && file.type.startsWith('image/')) {
        var reader = new FileReader();
        reader.onload = function (e) { img.src = e.target.result; img.style.display = ''; };
        reader.readAsDataURL(file);
      } else if (img) {
        img.style.display = 'none';
      }
      preview.classList.add('show');
    });
  });

  // ---------------------------------------------------------------------
  // Confetti burst — small, dependency-free success flourish
  // ---------------------------------------------------------------------
  function fireConfetti() {
    var colors = ['#0072CE', '#0BB99F', '#f5a623', '#ef4444', '#60c6ff'];
    for (var i = 0; i < 26; i++) {
      (function () {
        var piece = document.createElement('div');
        piece.className = 'ticket-confetti-piece';
        piece.style.background = colors[Math.floor(Math.random() * colors.length)];
        piece.style.borderRadius = Math.random() > 0.5 ? '50%' : '2px';
        document.body.appendChild(piece);

        var angle = Math.random() * Math.PI * 2;
        var distance = 90 + Math.random() * 160;
        var dx = Math.cos(angle) * distance;
        var dy = Math.sin(angle) * distance - 60;
        var rotate = (Math.random() - 0.5) * 720;
        var duration = 700 + Math.random() * 500;

        var anim = piece.animate([
          { transform: 'translate(-50%, -50%) rotate(0deg) scale(1)', opacity: 1 },
          { transform: 'translate(calc(-50% + ' + dx + 'px), calc(-50% + ' + dy + 'px)) rotate(' + rotate + 'deg) scale(0.4)', opacity: 0 },
        ], { duration: duration, easing: 'cubic-bezier(.2,.8,.2,1)' });
        anim.onfinish = function () { piece.remove(); };
      })();
    }
  }

  // ---------------------------------------------------------------------
  // Ticket form submission (shared across all 3 modals + the Support page)
  // ---------------------------------------------------------------------
  function submitTicketForm(form) {
    var modalEl = form.closest('.modal');
    var submitBtn = form.querySelector('[data-submit-btn]');
    var btnText = submitBtn ? submitBtn.querySelector('.btn-text') : null;
    var btnSpinner = submitBtn ? submitBtn.querySelector('.spinner-border') : null;
    var errorBox = form.querySelector('[data-form-error]');

    if (errorBox) { errorBox.classList.add('d-none'); errorBox.textContent = ''; }

    var subject = form.querySelector('[name="subject"]');
    var description = form.querySelector('[name="description"]');
    if (subject && !subject.value.trim()) {
      if (errorBox) { errorBox.textContent = 'Please give it a short subject.'; errorBox.classList.remove('d-none'); }
      subject.focus();
      return;
    }
    if (description && !description.value.trim()) {
      if (errorBox) { errorBox.textContent = 'Please describe it a bit more.'; errorBox.classList.remove('d-none'); }
      description.focus();
      return;
    }

    var pageUrlField = form.querySelector('[name="page_url"]');
    if (pageUrlField) pageUrlField.value = window.location.href;

    if (submitBtn) submitBtn.disabled = true;
    if (btnText) btnText.classList.add('d-none');
    if (btnSpinner) btnSpinner.classList.remove('d-none');

    var formData = new FormData(form);

    fetch(TICKET_CREATE_URL, { method: 'POST', body: formData, credentials: 'same-origin' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok) {
          if (errorBox) { errorBox.textContent = data.error || 'Something went wrong. Please try again.'; errorBox.classList.remove('d-none'); }
          return;
        }
        if (modalEl) {
          var instance = bootstrap.Modal.getInstance(modalEl);
          if (instance) instance.hide();
        }
        form.reset();
        document.querySelectorAll('.priority-chip').forEach(function (c) { c.classList.remove('active'); });
        var dzPreview = form.querySelector('.ticket-dropzone-preview');
        if (dzPreview) dzPreview.classList.remove('show');

        fireConfetti();
        var typeLabel = data.ticket.ticket_type_display || 'Ticket';
        window.AfyaToast.show(
          typeLabel + ' submitted — thank you! We’ll take it from here.',
          'success'
        );

        if (typeof window.onTicketCreated === 'function') {
          window.onTicketCreated(data.ticket);
        }
      })
      .catch(function () {
        if (errorBox) { errorBox.textContent = 'Network error. Please try again.'; errorBox.classList.remove('d-none'); }
      })
      .finally(function () {
        if (submitBtn) submitBtn.disabled = false;
        if (btnText) btnText.classList.remove('d-none');
        if (btnSpinner) btnSpinner.classList.add('d-none');
      });
  }

  document.querySelectorAll('.js-ticket-form').forEach(function (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      submitTicketForm(form);
    });
  });

  // Reset error state when a ticket modal is reopened
  document.querySelectorAll('.ticket-modal').forEach(function (modalEl) {
    modalEl.addEventListener('hidden.bs.modal', function () {
      var errorBox = modalEl.querySelector('[data-form-error]');
      if (errorBox) errorBox.classList.add('d-none');
    });
  });
})();
