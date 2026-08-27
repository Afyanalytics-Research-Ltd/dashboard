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
  var SESSION_STORAGE_KEY = 'afyaChatSessionKey';

  // ---- DOM references (populated in init) ---------------------------------
  var elFab, elPanel, elCloseBtn, elMessages, elTyping;
  var elInput, elSendBtn, elStatus, elStatusDot, elBadge, elSuggestions;
  var elHistoryBtn, elNewChatBtn, elHistoryOverlay, elHistoryCloseBtn, elHistoryList;
  var elLightbox;

  // ---- State --------------------------------------------------------------
  var ws = null;
  var reconnectCount = 0;
  var panelOpen = false;
  var unreadCount = 0;
  var connected = false;
  var currentSessionKey = null;
  try {
    currentSessionKey = window.localStorage.getItem(SESSION_STORAGE_KEY) || null;
  } catch (err) { /* localStorage unavailable (private mode, etc.) — fine, just no persistence */ }

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
    elHistoryBtn      = document.getElementById('chatbotHistoryBtn');
    elNewChatBtn      = document.getElementById('chatbotNewChatBtn');
    elHistoryOverlay  = document.getElementById('chatbotHistoryOverlay');
    elHistoryCloseBtn = document.getElementById('chatbotHistoryCloseBtn');
    elHistoryList     = document.getElementById('chatbotHistoryList');
    elLightbox        = document.getElementById('chatbotLightbox');

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

    if (elHistoryBtn) elHistoryBtn.addEventListener('click', openHistory);
    if (elHistoryCloseBtn) elHistoryCloseBtn.addEventListener('click', closeHistory);
    if (elNewChatBtn) elNewChatBtn.addEventListener('click', startNewChat);

    if (elLightbox) {
      elLightbox.addEventListener('click', function (e) {
        if (e.target === elLightbox || e.target.closest('.chatbot-lightbox-close')) {
          closeLightbox();
        }
      });
    }

    // Keyboard: close on Escape — lightbox first, then history, then panel.
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape') return;
      if (elLightbox && elLightbox.classList.contains('show')) {
        closeLightbox();
      } else if (panelOpen && elHistoryOverlay && !elHistoryOverlay.classList.contains('d-none')) {
        closeHistory();
      } else if (panelOpen) {
        closePanel();
      }
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
    var url = proto + '//' + window.location.host + WS_PATH;
    if (currentSessionKey) {
      url += '?session=' + encodeURIComponent(currentSessionKey);
    }
    return url;
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
    if (data.type === 'session') {
      currentSessionKey = data.session_key;
      try { window.localStorage.setItem(SESSION_STORAGE_KEY, currentSessionKey); } catch (err) { /* ignore */ }
      // A resumed session already has messages in the DB — pull them in.
      // A brand-new session has nothing to load; the server's canned
      // welcome arrives next as an ordinary 'message' event.
      if (!data.is_new) loadSessionHistory(currentSessionKey);
      return;
    }

    if (data.type === 'typing') {
      elTyping.classList.toggle('d-none', !data.status);
      return;
    }

    elTyping.classList.add('d-none');

    if (data.type === 'message') {
      if (data.chart && data.chart.image_base64) {
        var src = 'data:' + (data.chart.mime || 'image/png') + ';base64,' + data.chart.image_base64;
        appendAnswerWithChart(data.role || 'assistant', data.content || '', src, data.chart.caption);
      } else {
        appendBubble(data.role || 'assistant', data.content || '');
      }
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
  // Conversation history — session switching, listing, resuming
  // =========================================================================

  function loadSessionHistory(sessionKey) {
    fetch('/analytics/chat/history/?session=' + encodeURIComponent(sessionKey), { credentials: 'same-origin' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        elMessages.innerHTML = '';
        (data.messages || []).forEach(function (m) {
          if (m.chart_url) {
            appendAnswerWithChart(m.role, m.content, m.chart_url, m.chart_caption);
          } else {
            appendBubble(m.role, m.content);
          }
        });
        if (elSuggestions) elSuggestions.classList.add('d-none');
      })
      .catch(function (err) { console.error('[Afya Chat] history load error:', err); });
  }

  function openHistory() {
    if (!elHistoryOverlay) return;
    loadSessionList();
    elHistoryOverlay.classList.remove('d-none');
  }

  function closeHistory() {
    if (elHistoryOverlay) elHistoryOverlay.classList.add('d-none');
  }

  function loadSessionList() {
    if (!elHistoryList) return;
    elHistoryList.innerHTML = '<div class="chatbot-history-empty">Loading…</div>';
    fetch('/analytics/chat/sessions/', { credentials: 'same-origin' })
      .then(function (r) { return r.json(); })
      .then(function (data) { renderSessionList(data.sessions || []); })
      .catch(function () {
        elHistoryList.innerHTML = '<div class="chatbot-history-empty">Could not load history.</div>';
      });
  }

  function renderSessionList(sessions) {
    elHistoryList.innerHTML = '';

    if (!sessions.length) {
      elHistoryList.innerHTML = '<div class="chatbot-history-empty">' +
        '<i class="bi bi-chat-square-dots d-block mb-2" style="font-size:24px;opacity:.4;"></i>' +
        'No past conversations yet.</div>';
      return;
    }

    sessions.forEach(function (s) {
      var item = document.createElement('button');
      item.type = 'button';
      item.className = 'chatbot-history-item' + (s.session_key === currentSessionKey ? ' active' : '');
      item.innerHTML =
        '<div class="chatbot-history-item-title">' + escapeHtml(s.title) + '</div>' +
        (s.preview ? '<div class="chatbot-history-item-preview">' + escapeHtml(s.preview) + '</div>' : '') +
        '<div class="chatbot-history-item-time">' + formatRelativeTime(s.last_activity) +
        ' · ' + s.message_count + ' message' + (s.message_count === 1 ? '' : 's') + '</div>';
      item.addEventListener('click', function () { switchToSession(s.session_key); });
      elHistoryList.appendChild(item);
    });
  }

  function switchToSession(sessionKey) {
    closeHistory();
    if (sessionKey === currentSessionKey && connected) return;
    currentSessionKey = sessionKey;
    try { window.localStorage.setItem(SESSION_STORAGE_KEY, currentSessionKey); } catch (err) { /* ignore */ }
    elMessages.innerHTML = '';
    reconnect();
  }

  function startNewChat() {
    closeHistory();
    currentSessionKey = null;
    try { window.localStorage.removeItem(SESSION_STORAGE_KEY); } catch (err) { /* ignore */ }
    elMessages.innerHTML = '';
    reconnect();
  }

  function reconnect() {
    reconnectCount = 0;
    if (ws) {
      ws.onclose = null; // this is a deliberate switch, not a drop — skip auto-reconnect
      ws.close();
    }
    connect();
  }

  function formatRelativeTime(iso) {
    var d = new Date(iso);
    var mins = Math.round((Date.now() - d.getTime()) / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return mins + 'm ago';
    var hrs = Math.round(mins / 60);
    if (hrs < 24) return hrs + 'h ago';
    var days = Math.round(hrs / 24);
    if (days < 7) return days + 'd ago';
    return d.toLocaleDateString();
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

  function appendAnswerWithChart(role, content, chartSrc, chartCaption) {
    var caption = chartCaption || 'Chart';

    var wrap = document.createElement('div');
    wrap.className = 'chat-bubble ' + role + ' chat-bubble-tabbed';

    var tabs = document.createElement('div');
    tabs.className = 'chat-tabs';
    tabs.innerHTML =
      '<button type="button" class="chat-tab active" data-tab="answer"><i class="bi bi-chat-left-text"></i> Answer</button>' +
      '<button type="button" class="chat-tab" data-tab="chart"><i class="bi bi-bar-chart-line"></i> Chart</button>';
    tabs.addEventListener('click', function (e) {
      var btn = e.target.closest('.chat-tab');
      if (!btn) return;
      var target = btn.dataset.tab;
      tabs.querySelectorAll('.chat-tab').forEach(function (t) { t.classList.toggle('active', t === btn); });
      wrap.querySelectorAll('.chat-tab-panel').forEach(function (p) {
        p.classList.toggle('active', p.classList.contains('chat-tab-panel-' + target));
      });
    });
    wrap.appendChild(tabs);

    var answerPanel = document.createElement('div');
    answerPanel.className = 'chat-tab-panel chat-tab-panel-answer active';
    answerPanel.innerHTML = renderMarkdown(content);
    wrap.appendChild(answerPanel);

    var chartPanel = document.createElement('div');
    chartPanel.className = 'chat-tab-panel chat-tab-panel-chart';

    var img = document.createElement('img');
    img.src = chartSrc;
    img.alt = caption;
    img.style.maxWidth = '100%';
    img.style.borderRadius = '8px';
    img.style.display = 'block';
    img.style.margin = '0 auto';
    img.title = 'Click to view full size';
    img.addEventListener('click', function () { openLightbox(chartSrc, caption); });
    chartPanel.appendChild(img);

    var expandBtn = document.createElement('button');
    expandBtn.type = 'button';
    expandBtn.className = 'chat-chart-expand';
    expandBtn.innerHTML = '<i class="bi bi-arrows-fullscreen"></i> View full size';
    expandBtn.addEventListener('click', function () { openLightbox(chartSrc, caption); });
    chartPanel.appendChild(expandBtn);

    wrap.appendChild(chartPanel);
    elMessages.appendChild(wrap);
    scrollToBottom();
  }

  function openLightbox(src, caption) {
    if (!elLightbox) return;
    var img = elLightbox.querySelector('img');
    img.src = src;
    img.alt = caption || 'Chart';
    elLightbox.classList.add('show');
  }

  function closeLightbox() {
    if (elLightbox) elLightbox.classList.remove('show');
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
