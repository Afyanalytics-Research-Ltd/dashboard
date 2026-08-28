/* ==========================================================================
   SPREADSHEET ANALYST — chat page interactivity.
   Depends on: AfyaToast (main.js). Config read from #analystChatConfig.
   ========================================================================== */
(function () {
  'use strict';

  var configEl = document.getElementById('analystChatConfig');
  if (!configEl) return;
  var cfg = JSON.parse(configEl.textContent);

  var scrollEl = document.getElementById('anChatScroll');
  var textarea = document.getElementById('anQuestionInput');
  var sendBtn = document.getElementById('anSendBtn');
  var form = document.getElementById('anAskForm');

  function scrollToBottom() {
    scrollEl.scrollTop = scrollEl.scrollHeight;
  }
  scrollToBottom();

  // ---- Auto-resize textarea ----
  function autoResize() {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
    sendBtn.disabled = !textarea.value.trim();
  }
  textarea.addEventListener('input', autoResize);
  autoResize();

  // Enter to send, Shift+Enter for newline
  textarea.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) form.requestSubmit();
    }
  });

  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function appendUserBubble(text) {
    var wrap = document.createElement('div');
    wrap.className = 'chat-msg chat-msg-user';
    wrap.innerHTML = '<div class="chat-bubble chat-bubble-user"></div>';
    wrap.querySelector('.chat-bubble-user').textContent = text;
    scrollEl.appendChild(wrap);
    scrollToBottom();
  }

  function showTyping() {
    var wrap = document.createElement('div');
    wrap.className = 'chat-msg chat-msg-assistant';
    wrap.id = 'anTypingIndicator';
    wrap.innerHTML =
      '<div class="chat-bubble chat-bubble-assistant">' +
      '<div class="chat-bubble-avatar"><i class="bi bi-robot"></i></div>' +
      '<div class="chat-bubble-body chat-typing-bubble">' +
      '<span class="chat-typing-dot"></span><span class="chat-typing-dot"></span><span class="chat-typing-dot"></span>' +
      '</div></div>';
    scrollEl.appendChild(wrap);
    scrollToBottom();
  }

  function hideTyping() {
    var el = document.getElementById('anTypingIndicator');
    if (el) el.remove();
  }

  function sendQuestion(question) {
    if (!question.trim()) return;

    appendUserBubble(question);
    textarea.value = '';
    autoResize();
    hideSuggestions();
    showTyping();
    sendBtn.disabled = true;

    var body = new FormData();
    body.append('csrfmiddlewaretoken', cfg.csrfToken);
    body.append('question', question);

    fetch(cfg.askUrl, { method: 'POST', body: body, credentials: 'same-origin' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        hideTyping();
        var temp = document.createElement('div');
        temp.innerHTML = data.html;
        scrollEl.appendChild(temp.firstElementChild);
        scrollToBottom();
        if (!data.ok) window.AfyaToast.show('The analysis hit a snag — see the message above.', 'danger');
      })
      .catch(function () {
        hideTyping();
        window.AfyaToast.show('Network error. Please try again.', 'danger');
      })
      .finally(function () {
        sendBtn.disabled = !textarea.value.trim();
      });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    sendQuestion(textarea.value);
  });

  function hideSuggestions() {
    var el = document.getElementById('anSuggestions');
    if (el) el.style.display = 'none';
  }

  document.querySelectorAll('.an-suggestion-chip').forEach(function (chip) {
    chip.addEventListener('click', function () { sendQuestion(chip.dataset.question); });
  });

  // ---- Reset kernel ----
  var resetBtn = document.getElementById('anResetKernelBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', function () {
      if (!confirm('Reset the analysis kernel? Computed variables will be cleared — the workbook and chat history stay intact.')) return;
      var body = new FormData();
      body.append('csrfmiddlewaretoken', cfg.csrfToken);
      fetch(cfg.resetUrl, { method: 'POST', body: body, credentials: 'same-origin' })
        .then(function (r) { return r.json(); })
        .then(function () { window.AfyaToast.show('Kernel reset — the next question reloads the workbook.', 'success'); });
    });
  }

  // ---- Lightbox for chart artifacts ----
  window.openLightbox = function (src, title) {
    var lb = document.getElementById('anLightbox');
    lb.querySelector('img').src = src;
    lb.querySelector('img').alt = title;
    lb.classList.add('show');
  };
  var lightbox = document.getElementById('anLightbox');
  if (lightbox) {
    lightbox.addEventListener('click', function (e) {
      if (e.target === lightbox || e.target.closest('.an-lightbox-close')) {
        lightbox.classList.remove('show');
      }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') lightbox.classList.remove('show');
    });
  }
})();
