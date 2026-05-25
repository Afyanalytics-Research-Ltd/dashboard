/**
 * Afya DataHub — Main JavaScript
 * Bootstrap 5.3 companion utilities.
 */

/* ============================================================
   Sidebar toggle
   ============================================================ */
(function () {
  'use strict';

  const SIDEBAR_KEY = 'afya_sidebar_collapsed';

  const sidebar = document.getElementById('afyaSidebar');
  const mainEl  = document.getElementById('afyaMain');
  const toggleBtn = document.getElementById('sidebarToggle');
  const collapseBtn = document.getElementById('sidebarCollapseBtn');

  function isCollapsed() {
    return localStorage.getItem(SIDEBAR_KEY) === '1';
  }

  function setCollapsed(state) {
    if (!sidebar || !mainEl) return;
    if (state) {
      sidebar.classList.add('collapsed');
      mainEl.classList.add('sidebar-collapsed');
      localStorage.setItem(SIDEBAR_KEY, '1');
    } else {
      sidebar.classList.remove('collapsed');
      mainEl.classList.remove('sidebar-collapsed');
      localStorage.setItem(SIDEBAR_KEY, '0');
    }
  }

  // Restore state on load
  if (isCollapsed()) setCollapsed(true);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', function () {
      if (window.innerWidth <= 768) {
        sidebar && sidebar.classList.toggle('mobile-open');
      } else {
        setCollapsed(!isCollapsed());
      }
    });
  }

  if (collapseBtn) {
    collapseBtn.addEventListener('click', function () {
      setCollapsed(!isCollapsed());
    });
  }

  // Close mobile sidebar on outside click
  document.addEventListener('click', function (e) {
    if (window.innerWidth <= 768 && sidebar && sidebar.classList.contains('mobile-open')) {
      if (!sidebar.contains(e.target) && e.target !== toggleBtn) {
        sidebar.classList.remove('mobile-open');
      }
    }
  });

  // Submenu toggles
  document.querySelectorAll('[data-submenu-toggle]').forEach(function (trigger) {
    trigger.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('data-submenu-toggle');
      const submenu  = document.getElementById(targetId);
      const chevron  = this.querySelector('.sidebar-chevron');
      if (!submenu) return;

      const isOpen = submenu.classList.toggle('open');
      if (chevron) chevron.classList.toggle('open', isOpen);

      // Save open state
      const openMenus = JSON.parse(sessionStorage.getItem('afya_open_menus') || '[]');
      if (isOpen) {
        if (!openMenus.includes(targetId)) openMenus.push(targetId);
      } else {
        const idx = openMenus.indexOf(targetId);
        if (idx > -1) openMenus.splice(idx, 1);
      }
      sessionStorage.setItem('afya_open_menus', JSON.stringify(openMenus));
    });
  });

  // Restore open submenus
  const openMenus = JSON.parse(sessionStorage.getItem('afya_open_menus') || '[]');
  openMenus.forEach(function (menuId) {
    const submenu = document.getElementById(menuId);
    const trigger = document.querySelector('[data-submenu-toggle="' + menuId + '"]');
    const chevron = trigger && trigger.querySelector('.sidebar-chevron');
    if (submenu) {
      submenu.classList.add('open');
      if (chevron) chevron.classList.add('open');
    }
  });
}());


/* ============================================================
   Toast notifications
   ============================================================ */
(function () {
  'use strict';

  const ICON_MAP = {
    success   : 'bi-check-circle-fill',
    danger    : 'bi-x-circle-fill',
    warning   : 'bi-exclamation-triangle-fill',
    info      : 'bi-info-circle-fill',
    secondary : 'bi-bell-fill',
  };

  /**
   * Show a programmatic toast.
   * @param {string} message
   * @param {string} [type='info'] - success | danger | warning | info | secondary
   * @param {number} [duration=5000]
   */
  window.AfyaToast = {
    show: function (message, type, duration) {
      type     = type || 'info';
      duration = duration !== undefined ? duration : 5000;

      const container = document.getElementById('afyaToastContainer');
      if (!container) return;

      const icon = ICON_MAP[type] || ICON_MAP.info;

      const toast = document.createElement('div');
      toast.className = 'afya-toast';
      toast.innerHTML = [
        '<div class="toast-accent toast-accent-' + type + '"></div>',
        '<div class="toast-body-inner d-flex align-items-center gap-2">',
        '  <i class="bi ' + icon + ' toast-icon text-' + (type === 'secondary' ? 'secondary' : type === 'danger' ? 'danger' : type) + '"></i>',
        '  <span class="toast-msg">' + _escapeHtml(message) + '</span>',
        '</div>',
        '<button class="toast-close" aria-label="Close"><i class="bi bi-x"></i></button>',
      ].join('');

      container.appendChild(toast);

      // Close button
      toast.querySelector('.toast-close').addEventListener('click', function () {
        _dismiss(toast);
      });

      // Auto-dismiss
      if (duration > 0) {
        setTimeout(function () { _dismiss(toast); }, duration);
      }
    },
  };

  function _dismiss(el) {
    el.style.transition = 'opacity 0.3s, transform 0.3s';
    el.style.opacity    = '0';
    el.style.transform  = 'translateX(40px)';
    setTimeout(function () { el.remove(); }, 300);
  }

  function _escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  // Auto-render Django messages that were server-rendered as data attributes
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-django-message]').forEach(function (el) {
      AfyaToast.show(el.dataset.djangoMessage, el.dataset.djangoMessageType || 'info');
    });
  });
}());


/* ============================================================
   Auto-dismiss Bootstrap alerts
   ============================================================ */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.alert-dismissible[data-auto-dismiss]').forEach(function (el) {
    const delay = parseInt(el.dataset.autoDismiss, 10) || 4000;
    setTimeout(function () {
      var bsAlert = window.bootstrap && bootstrap.Alert.getOrCreateInstance(el);
      if (bsAlert) bsAlert.close();
    }, delay);
  });
});


/* ============================================================
   Confirm dialogs (data-confirm attribute)
   ============================================================ */
document.addEventListener('click', function (e) {
  const el = e.target.closest('[data-confirm]');
  if (!el) return;
  const msg = el.getAttribute('data-confirm') || 'Are you sure?';
  if (!window.confirm(msg)) {
    e.preventDefault();
    e.stopImmediatePropagation();
  }
});

/* Form-level confirm */
document.addEventListener('submit', function (e) {
  const form = e.target;
  const msg  = form.getAttribute('data-confirm');
  if (msg && !window.confirm(msg)) {
    e.preventDefault();
  }
});


/* ============================================================
   DataTable helpers — client-side search + pagination
   ============================================================ */
(function () {
  'use strict';

  /**
   * Initialise a simple searchable table.
   * @param {string} tableId     - ID of the <table> element
   * @param {string} searchId    - ID of the <input> to use as search box
   * @param {number} [pageSize]  - rows per page (0 = no pagination)
   */
  window.AfyaTable = {
    init: function (tableId, searchId, pageSize) {
      const table  = document.getElementById(tableId);
      const search = document.getElementById(searchId);
      if (!table) return;

      pageSize = pageSize || 0;
      let currentPage = 1;
      const tbody = table.querySelector('tbody');

      function allRows() {
        return Array.from(tbody.querySelectorAll('tr'));
      }

      function filterRows(query) {
        const q = query.toLowerCase().trim();
        allRows().forEach(function (row) {
          const text = row.textContent.toLowerCase();
          row.style.display = (!q || text.includes(q)) ? '' : 'none';
        });
        currentPage = 1;
        if (pageSize) paginate();
      }

      function paginate() {
        const visible = allRows().filter(function (r) {
          return r.style.display !== 'none';
        });
        const start = (currentPage - 1) * pageSize;
        const end   = start + pageSize;
        visible.forEach(function (r, i) {
          r.style.display = (i >= start && i < end) ? '' : 'none';
        });
      }

      if (search) {
        search.addEventListener('input', function () {
          filterRows(this.value);
        });
      }

      if (pageSize) paginate();
    },
  };
}());


/* ============================================================
   AJAX CSRF helper
   ============================================================ */
(function () {
  'use strict';

  function getCookie(name) {
    if (!document.cookie) return null;
    const parts = document.cookie.split(';')
      .map(function (c) { return c.trim(); })
      .filter(function (c) { return c.startsWith(name + '='); });
    if (!parts.length) return null;
    return decodeURIComponent(parts[0].split('=')[1]);
  }

  /**
   * Make a CSRF-safe fetch request.
   * Usage: AfyaFetch('/api/v1/...', { method: 'POST', body: JSON.stringify({}) })
   */
  window.AfyaFetch = function (url, options) {
    options = options || {};
    options.headers = Object.assign({
      'X-CSRFToken': getCookie('csrftoken') || '',
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }, options.headers || {});
    options.credentials = 'same-origin';
    return fetch(url, options);
  };
}());


/* ============================================================
   Notification mark-read (via AJAX)
   ============================================================ */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('[data-mark-read]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const pk  = btn.getAttribute('data-mark-read');
      const url = '/core/notifications/' + pk + '/read/';
      AfyaFetch(url, { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.ok) {
            const item = btn.closest('.notif-item');
            if (item) item.classList.remove('unread');
            // Update badge
            const badge = document.getElementById('notifBadge');
            if (badge) {
              let count = parseInt(badge.textContent, 10);
              if (count > 1) {
                badge.textContent = count - 1;
              } else {
                badge.remove();
              }
            }
          }
        })
        .catch(function () {});
    });
  });
});
