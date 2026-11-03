// Lazy loaders for third-party libraries (pinned versions, cdnjs).

const CDN = 'https://cdnjs.cloudflare.com/ajax/libs';
const loaded = new Map();

export function loadScript(src) {
  if (!loaded.has(src)) {
    loaded.set(src, new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.async = false; // preserve order for addons that depend on the core
      s.onload = resolve;
      s.onerror = () => { loaded.delete(src); reject(new Error(`Failed to load ${src}`)); };
      document.head.appendChild(s);
    }));
  }
  return loaded.get(src);
}

const CM = `${CDN}/codemirror/5.65.16`;

/** CodeMirror 5 with SQL mode, hints, bracket matching, search and comment toggling. */
export async function loadCodeMirror() {
  await loadScript(`${CM}/codemirror.min.js`);
  await Promise.all([
    `${CM}/mode/sql/sql.min.js`,
    `${CM}/addon/hint/show-hint.min.js`,
    `${CM}/addon/hint/sql-hint.min.js`,
    `${CM}/addon/edit/matchbrackets.min.js`,
    `${CM}/addon/edit/closebrackets.min.js`,
    `${CM}/addon/comment/comment.min.js`,
    `${CM}/addon/selection/active-line.min.js`,
    `${CM}/addon/display/placeholder.min.js`,
  ].map(loadScript));
  return window.CodeMirror;
}

export async function loadVimKeymap() {
  await loadCodeMirror();
  await loadScript(`${CM}/keymap/vim.min.js`);
}

/** Plotly (basic + statistical + financial is enough for every Redash chart type). */
export async function loadPlotly() {
  await loadScript(`${CDN}/plotly.js/2.35.0/plotly.min.js`);
  return window.Plotly;
}

/** Markdown for query descriptions and dashboard text widgets (sanitized with DOMPurify). */
export async function loadMarkdown() {
  await Promise.all([
    loadScript(`${CDN}/marked/12.0.2/marked.min.js`),
    loadScript(`${CDN}/dompurify/3.1.6/purify.min.js`),
  ]);
  return (text) => window.DOMPurify.sanitize(window.marked.parse(String(text ?? ''), { breaks: true }));
}
