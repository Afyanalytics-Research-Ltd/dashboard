/* ==========================================================================
   Afya DataHub — Self-Service Analytics Chatbot
   Connects to the Django Channels WebSocket consumer at /ws/analytics/chat/
   ========================================================================== */

(function () {
  'use strict';

  // ---- Configuration ------------------------------------------------------
  var WS_PATH = '/ws/analytics/chat/';
  var RECONNECT_DELAY_MS = 3000;
  var MAX_RECONNECTS = 5;

  // ---- DOM references (populated in init) ---------------------------------
  var elFab, elPanel, elCloseBtn, elMessages, elTyping;
  var elInput, elSendBtn, elStatus, elStatusDot, elBadge, elSuggestions;

  // ---- State --------------------------------------------------------------
  var ws = null;
  var reconnectCount = 0;
  var panelOpen = false;
  var unreadCount = 0;
  var connected = false;

  // =========================================================================
  // Bootstrap
  // =========================================================================

  document.addEventListener('DOMContentLoaded', function () {
    elFab        = document.getElementById('chatbotFab');
    elPanel      = document.getElementById('chatbotPanel');
    elCloseBtn   = document.getElementById('chatbotCloseBtn');
    elMessages   = document.getElementById('chatbotMessages');
    elTyping     = document.getElementById('chatbotTyping');
    elInput      = document.getElementById('chatbotInput');
    elSendBtn    = document.getElementById('chatbotSendBtn');
    elStatus     = document.getElementById('chatbotStatusText');
    elStatusDot  = document.getElementById('chatbotStatusDot');
    elBadge      = document.getElementById('chatbotBadge');
    elSuggestions = document.getElementById('chatbotSuggestions');

    // If the FAB doesn't exist the user isn't authenticated — bail out.
    if (!elFab) return;

    bindEvents();
  });

  // =========================================================================
  // Event wiring
  // =========================================================================

  function bindEvents() {
    elFab.addEventListener('click', togglePanel);
    elCloseBtn.addEventListener('click', closePanel);
    elSendBtn.addEventListener('click', sendMessage);

    elInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Auto-resize textarea
    elInput.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    // Suggestion chips
    if (elSuggestions) {
      elSuggestions.addEventListener('click', function (e) {
        var chip = e.target.closest('.chatbot-chip');
        if (chip) {
          elInput.value = chip.dataset.query;
          elInput.dispatchEvent(new Event('input'));
          sendMessage();
        }
      });
    }

    // Keyboard: close on Escape
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && panelOpen) closePanel();
    });

    // Mobile: tap outside to close
    document.addEventListener('click', function (e) {
      if (
        panelOpen &&
        window.innerWidth <= 768 &&
        !elPanel.contains(e.target) &&
        !elFab.contains(e.target)
      ) {
        closePanel();
      }
    });
  }

  // =========================================================================
  // Panel open / close
  // =========================================================================

  function togglePanel() {
    panelOpen ? closePanel() : openPanel();
  }

  function openPanel() {
    panelOpen = true;
    elPanel.classList.add('open');
    elFab.classList.add('panel-open');
    elFab.setAttribute('aria-expanded', 'true');

    resetUnread();
    elInput.focus();

    if (!ws || ws.readyState === WebSocket.CLOSED) {
      connect();
    }
  }

  function closePanel() {
    panelOpen = false;
    elPanel.classList.remove('open');
    elFab.classList.remove('panel-open');
    elFab.setAttribute('aria-expanded', 'false');
  }

  // =========================================================================
  // WebSocket lifecycle
  // =========================================================================

  function wsUrl() {
    var proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return proto + '//' + window.location.host + WS_PATH;
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
      return;
    }

    setStatus(false);
    ws = new WebSocket(wsUrl());

    ws.onopen = function () {
      reconnectCount = 0;
      setStatus(true);
      if (elSuggestions) elSuggestions.classList.remove('d-none');
    };

    ws.onmessage = function (event) {
      try {
        handleServerMessage(JSON.parse(event.data));
      } catch (err) {
        console.error('[Afya Chat] JSON parse error:', err);
      }
    };

    ws.onerror = function () {
      // onclose fires after onerror; do nothing here
    };

    ws.onclose = function () {
      setStatus(false);
      if (elSuggestions) elSuggestions.classList.add('d-none');

      if (panelOpen && reconnectCount < MAX_RECONNECTS) {
        reconnectCount++;
        setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };
  }

  // =========================================================================
  // Incoming message handling
  // =========================================================================

  function handleServerMessage(data) {
    if (data.type === 'typing') {
      elTyping.classList.toggle('d-none', !data.status);
      return;
    }

    elTyping.classList.add('d-none');

    if (data.type === 'message') {
      appendBubble(data.role || 'assistant', data.content || '');
      if (!panelOpen) bumpUnread();
    }
  }

  // =========================================================================
  // Sending
  // =========================================================================

  function sendMessage() {
    var text = elInput.value.trim();
    if (!text) return;

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      appendSystemNote('Not connected. Reconnecting…');
      connect();
      return;
    }

    appendBubble('user', text);
    ws.send(JSON.stringify({ message: text }));

    elInput.value = '';
    elInput.style.height = 'auto';
    if (elSuggestions) elSuggestions.classList.add('d-none');
  }

  // =========================================================================
  // DOM helpers
  // =========================================================================

  function appendBubble(role, content) {
    var div = document.createElement('div');
    div.className = 'chat-bubble ' + role;
    div.innerHTML = renderMarkdown(content);
    elMessages.appendChild(div);
    scrollToBottom();
  }

  function appendSystemNote(text) {
    var div = document.createElement('div');
    div.className = 'text-center text-muted my-1';
    div.style.fontSize = '11px';
    div.textContent = text;
    elMessages.appendChild(div);
    scrollToBottom();
  }

  function scrollToBottom() {
    elMessages.scrollTop = elMessages.scrollHeight;
  }

  function setStatus(isOnline) {
    connected = isOnline;
    if (elStatusDot) {
      elStatusDot.classList.toggle('online', isOnline);
    }
    if (elStatus) {
      elStatus.textContent = isOnline ? 'Online' : 'Connecting…';
    }
    if (elSendBtn) {
      elSendBtn.disabled = !isOnline;
    }
  }

  function bumpUnread() {
    unreadCount++;
    if (elBadge) {
      elBadge.textContent = unreadCount > 9 ? '9+' : unreadCount;
      elBadge.classList.remove('d-none');
    }
  }

  function resetUnread() {
    unreadCount = 0;
    if (elBadge) elBadge.classList.add('d-none');
  }

  // =========================================================================
  // Minimal Markdown renderer (bold, italic, bullet lists, line breaks)
  // Escapes HTML first to prevent XSS from server responses.
  // =========================================================================

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderMarkdown(text) {
    var escaped = escapeHtml(text);

    // Bold: **text**
    escaped = escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic: *text*
    escaped = escaped.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Bullet list items starting with "• " or "- "
    var lines = escaped.split('\n');
    var inList = false;
    var out = [];

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      var isBullet = /^(&bull;|•|-)\s/.test(line) || /^&lt;li&gt;/.test(line);

      // Normalise "• " and "- " to list items
      var cleaned = line.replace(/^(•|-)\s/, '');

      if (isBullet) {
        if (!inList) { out.push('<ul>'); inList = true; }
        out.push('<li>' + cleaned.replace(/^(&bull;)\s/, '') + '</li>');
      } else {
        if (inList) { out.push('</ul>'); inList = false; }
        out.push(line);
      }
    }
    if (inList) out.push('</ul>');

    // Remaining newlines → <br>
    return out.join('\n').replace(/\n/g, '<br>');
  }

})();
