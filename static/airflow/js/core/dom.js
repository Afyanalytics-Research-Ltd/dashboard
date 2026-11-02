// Tiny hyperscript DOM toolkit. No innerHTML with untrusted data anywhere.
//
//   h('div.card#main', { onclick, style: { color: 'red' }, dataset: { id: 1 } }, 'text', child, [more])

const SVG_NS = 'http://www.w3.org/2000/svg';

function applyAttrs(el, attrs, isSvg) {
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === 'class' || key === 'className') {
      const cls = Array.isArray(value) ? value.filter(Boolean).join(' ') : value;
      if (cls) isSvg ? el.setAttribute('class', cls) : (el.className = [el.className, cls].filter(Boolean).join(' '));
    } else if (key === 'style') {
      if (typeof value === 'string') el.style.cssText = value;
      else Object.assign(el.style, value);
    } else if (key === 'dataset') {
      Object.assign(el.dataset, value);
    } else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'ref' && typeof value === 'function') {
      value(el);
    } else if (!isSvg && (key === 'value' || key === 'checked' || key === 'disabled' || key === 'selected')) {
      el[key] = value;
    } else {
      el.setAttribute(key, value === true ? '' : value);
    }
  }
}

function appendChildren(el, children) {
  for (const child of children) {
    if (child == null || child === false || child === true) continue;
    if (Array.isArray(child)) appendChildren(el, child);
    else if (child instanceof Node) el.appendChild(child);
    else el.appendChild(document.createTextNode(String(child)));
  }
}

function parseTag(tag) {
  const [, name = 'div', rest = ''] = tag.match(/^([a-zA-Z0-9-]*)(.*)$/) || [];
  const classes = [];
  let id = null;
  for (const part of rest.match(/[.#][^.#]+/g) || []) {
    if (part[0] === '.') classes.push(part.slice(1));
    else id = part.slice(1);
  }
  return { name: name || 'div', classes, id };
}

function build(isSvg, tag, attrs, children) {
  if (attrs == null || typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs)) {
    children = [attrs, ...children];
    attrs = {};
  }
  const { name, classes, id } = parseTag(tag);
  const el = isSvg ? document.createElementNS(SVG_NS, name) : document.createElement(name);
  if (classes.length) el.setAttribute('class', classes.join(' '));
  if (id) el.id = id;
  applyAttrs(el, attrs, isSvg);
  appendChildren(el, children);
  return el;
}

export const h = (tag, attrs, ...children) => build(false, tag, attrs, children);
export const s = (tag, attrs, ...children) => build(true, tag, attrs, children);

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function mount(el, ...children) {
  clear(el);
  appendChildren(el, children);
  return el;
}

export function debounce(fn, ms = 250) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export function icon(name, cls = '') {
  return h(`i.bi.bi-${name}`, { class: cls, 'aria-hidden': 'true' });
}

/** Read and write per-viewer UI preferences; storage may be unavailable. */
export const prefs = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(`af:${key}`);
      return raw == null ? fallback : JSON.parse(raw);
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`af:${key}`, JSON.stringify(value)); } catch { /* ignore */ }
  },
};
