// SQL editor built on CodeMirror 5.
//
//   const ed = await sqlEditor(el, { value, onChange, onRun, onSave, onFormat, schema, snippets })
//   ed.getValue(); ed.getSelection(); ed.setSchema(schema); ed.insert(text); ed.focus()
//
// Shortcuts: Ctrl/⌘+Enter run (selection if any), Ctrl/⌘+S save, Ctrl/⌘+Shift+F format,
// Ctrl/⌘+/ toggle comment, Ctrl+Space autocomplete, Tab expands a snippet trigger.

import { h, prefs } from '../../../ui/js/dom.js';
import { loadCodeMirror, loadVimKeymap } from './lib.js';

export const EDITOR_DEFAULTS = { fontSize: 13, lineWrapping: false, lowercaseHints: false, vim: false, theme: 'light', liveAutocomplete: true };
export const editorPrefs = () => ({ ...EDITOR_DEFAULTS, ...prefs.get('rd:editor', {}) });
export const saveEditorPrefs = (p) => prefs.set('rd:editor', { ...editorPrefs(), ...p });

/** Redash schema [{name, columns:[name|{name,type}]}] -> CodeMirror sql-hint tables {table: [cols]}. */
export function hintTables(schema = []) {
  const tables = {};
  for (const t of schema) {
    const cols = (t.columns || []).map((c) => (typeof c === 'string' ? c : c.name));
    tables[t.name] = cols;
    const short = t.name.split('.').pop();
    if (short !== t.name && !(short in tables)) tables[short] = cols;
  }
  return tables;
}

export async function sqlEditor(host, opts = {}) {
  const CodeMirror = await loadCodeMirror();
  const p = { ...editorPrefs(), ...(opts.prefs || {}) };
  if (p.vim) await loadVimKeymap();
  let schema = opts.schema || [];
  let snippets = opts.snippets || [];

  const wrap = h('div.rd-editor', { class: p.theme === 'dark' && 'rd-editor-dark', style: { '--rd-editor-font': `${p.fontSize}px` } });
  host.appendChild(wrap);
  const mod = navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl';
  const cm = CodeMirror(wrap, {
    value: opts.value || '',
    mode: opts.mode || 'text/x-sql',
    lineNumbers: true,
    lineWrapping: p.lineWrapping,
    matchBrackets: true,
    autoCloseBrackets: true,
    styleActiveLine: true,
    indentUnit: 2,
    tabSize: 2,
    readOnly: opts.readOnly ? 'nocursor' : false,
    placeholder: opts.placeholder || 'SELECT * FROM … — press Ctrl+Space for suggestions, Ctrl+Enter to run',
    keyMap: p.vim ? 'vim' : 'default',
    viewportMargin: 50,
    extraKeys: {
      [`${mod}-Enter`]: () => opts.onRun?.(cm.somethingSelected() ? cm.getSelection() : null),
      [`Shift-${mod}-Enter`]: () => opts.onRun?.(null),
      [`${mod}-S`]: () => opts.onSave?.(),
      [`Shift-${mod}-F`]: () => opts.onFormat?.(),
      [`${mod}-/`]: 'toggleComment',
      'Ctrl-Space': 'autocomplete',
      Tab: (ed) => {
        if (expandSnippet(ed)) return;
        if (ed.somethingSelected()) ed.indentSelection('add');
        else ed.replaceSelection('  ', 'end');
      },
      'Shift-Tab': (ed) => ed.indentSelection('subtract'),
    },
    hintOptions: { completeSingle: false },
  });

  function expandSnippet(ed) {
    const cur = ed.getCursor();
    const line = ed.getLine(cur.line).slice(0, cur.ch);
    const word = (line.match(/[\w-]+$/) || [''])[0];
    const snip = word && snippets.find((s) => s.trigger === word);
    if (!snip) return false;
    ed.replaceRange(snip.snippet, { line: cur.line, ch: cur.ch - word.length }, cur);
    return true;
  }

  const setHints = () => {
    const hintOptions = { completeSingle: false, tables: hintTables(schema) };
    if (p.lowercaseHints) {
      // CodeMirror's sql-hint uppercases keywords; lowercase them on request.
      hintOptions.hint = (ed, o) => {
        const res = CodeMirror.hint.sql(ed, o);
        if (res?.list) res.list = res.list.map((it) => (typeof it === 'string' && /^[A-Z ]+$/.test(it) ? it.toLowerCase() : it));
        return res;
      };
    }
    cm.setOption('hintOptions', hintOptions);
  };
  setHints();

  if (p.liveAutocomplete) {
    cm.on('inputRead', (ed, change) => {
      if (change.origin !== '+input' || ed.state.completionActive) return;
      const ch = change.text[0];
      if (/[\w.]/.test(ch)) {
        const line = ed.getLine(ed.getCursor().line).slice(0, ed.getCursor().ch);
        if (/(\w{2,}|\.)$/.test(line)) ed.showHint({ completeSingle: false });
      }
    });
  }

  if (opts.onChange) cm.on('change', () => opts.onChange(cm.getValue()));
  setTimeout(() => cm.refresh(), 0);

  return {
    cm,
    el: wrap,
    getValue: () => cm.getValue(),
    setValue: (v) => { cm.setValue(v || ''); cm.clearHistory(); },
    replaceAll: (v) => {
      // Keeps undo history (used by Format / Restore so the user can undo them).
      cm.operation(() => cm.replaceRange(v, { line: 0, ch: 0 }, { line: cm.lastLine(), ch: cm.getLine(cm.lastLine()).length }));
    },
    getSelection: () => (cm.somethingSelected() ? cm.getSelection() : ''),
    insert: (text) => { cm.replaceSelection(text); cm.focus(); },
    focus: () => cm.focus(),
    refresh: () => cm.refresh(),
    setSchema: (s) => { schema = s || []; setHints(); },
    setSnippets: (s) => { snippets = s || []; },
    setReadOnly: (ro) => cm.setOption('readOnly', ro ? 'nocursor' : false),
    markError: (lineNumber) => {
      if (lineNumber == null) return null;
      const handle = cm.addLineClass(lineNumber - 1, 'background', 'rd-line-error');
      return () => cm.removeLineClass(handle, 'background', 'rd-line-error');
    },
  };
}
