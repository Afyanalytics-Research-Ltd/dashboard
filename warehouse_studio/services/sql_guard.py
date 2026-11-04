"""SQL statement splitting and safety classification for Warehouse Studio.

The old warehouse console blocked any query containing a word like UPDATE or
CREATE anywhere — including inside strings, comments and column names.  This
module lexes Snowflake SQL properly (single-quoted strings with '' escapes,
double-quoted identifiers, $$ dollar-quoted bodies, -- // and /* */ comments)
so decisions are based on what a statement actually does.

Policy
------
* **read** mode (default): SELECT / WITH…SELECT / SHOW / DESCRIBE / EXPLAIN /
  LIST, plus CALL of a procedure on the studio allow-list.  SYSTEM$ functions
  that change state (cancel, abort, enable, set …) are refused.
* **write** mode (only when WAREHOUSE_STUDIO_ALLOW_WRITES is on *and* the
  user explicitly unlocks it for a run): anything the Snowflake role permits
  except account / security administration, which is never allowed from a
  web UI.
* ``USE …`` is refused in both modes — context is chosen with the pickers, and
  a ``USE`` would silently change a pooled session other people share.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

READ = 'read'
WRITE = 'write'


class SqlGuardError(Exception):
    """The statement is not allowed (or not parseable) under the current policy."""

    def __init__(self, message: str, *, code: str = 'blocked', statement: str = ''):
        super().__init__(message)
        self.code = code
        self.statement = statement


# ----------------------------------------------------------------------------- lexer

@dataclass
class Token:
    kind: str          # word | string | ident | dollar | number | punct | comment | ws
    text: str
    start: int
    end: int

    @property
    def upper(self) -> str:
        return self.text.upper()


# Unicode-aware: identifiers like "café" must lex as one word (str.isalpha() admits them).
_WORD_RE = re.compile(r'[^\W\d][\w$]*')
_NUM_RE = re.compile(r'\d+(\.\d*)?([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?')


def tokenize(sql: str) -> list[Token]:
    """Lex ``sql`` into tokens.  Unterminated strings/comments run to end of input."""
    tokens: list[Token] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            j = i + 1
            while j < n and sql[j].isspace():
                j += 1
            tokens.append(Token('ws', sql[i:j], i, j))
            i = j
        elif sql.startswith('--', i) or sql.startswith('//', i):
            j = sql.find('\n', i)
            j = n if j < 0 else j
            tokens.append(Token('comment', sql[i:j], i, j))
            i = j
        elif sql.startswith('/*', i):
            j = sql.find('*/', i + 2)
            j = n if j < 0 else j + 2
            tokens.append(Token('comment', sql[i:j], i, j))
            i = j
        elif c == "'":
            j = i + 1
            while j < n:
                if sql[j] == '\\':
                    j += 2
                    continue
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            else:
                j = n
            tokens.append(Token('string', sql[i:j], i, min(j, n)))
            i = min(j, n)
        elif c == '"':
            j = i + 1
            while j < n:
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            else:
                j = n
            tokens.append(Token('ident', sql[i:j], i, min(j, n)))
            i = min(j, n)
        elif sql.startswith('$$', i):
            j = sql.find('$$', i + 2)
            j = n if j < 0 else j + 2
            tokens.append(Token('dollar', sql[i:j], i, j))
            i = j
        elif c.isalpha() or c == '_':
            m = _WORD_RE.match(sql, i)
            tokens.append(Token('word', m.group(0), i, m.end()))
            i = m.end()
        elif c.isdigit() or (c == '.' and i + 1 < n and sql[i + 1].isdigit()):
            m = _NUM_RE.match(sql, i)
            tokens.append(Token('number', m.group(0), i, m.end()))
            i = m.end()
        else:
            tokens.append(Token('punct', c, i, i + 1))
            i += 1
    return tokens


# ----------------------------------------------------------------------------- splitting

@dataclass
class Statement:
    text: str           # statement text, trimmed, without the trailing semicolon
    start: int          # offset of the first character in the original input
    end: int            # offset just past the last character (before ';')
    tokens: list[Token] = field(default_factory=list, repr=False)

    @property
    def code_tokens(self) -> list[Token]:
        return [t for t in self.tokens if t.kind not in ('ws', 'comment')]

    @property
    def keyword(self) -> str:
        for t in self.code_tokens:
            return t.upper if t.kind == 'word' else ''
        return ''


def split_statements(sql: str) -> list[Statement]:
    """Split a script on top-level semicolons.  Empty / comment-only pieces are dropped.

    Snowflake Scripting blocks (``BEGIN … END``, ``DECLARE … END``,
    ``EXECUTE IMMEDIATE $$…$$``) contain semicolons; a block is kept whole by
    tracking BEGIN/END nesting for statements that *start* a block.
    """
    tokens = tokenize(sql)
    statements: list[Statement] = []
    current: list[Token] = []
    depth = 0           # BEGIN … END nesting inside an anonymous block
    in_block = False
    declare_pending = False

    def flush(end: int):
        code = [t for t in current if t.kind not in ('ws', 'comment')]
        if code:
            start, stop = code[0].start, code[-1].end
            statements.append(Statement(sql[start:stop], start, stop, list(current)))
        current.clear()

    code_idx = [i for i, t in enumerate(tokens) if t.kind not in ('ws', 'comment')]
    next_code = {}
    for a, b in zip(code_idx, code_idx[1:]):
        next_code[a] = tokens[b]

    for idx, tok in enumerate(tokens):
        if tok.kind == 'word':
            word = tok.upper
            code_before = [t for t in current if t.kind not in ('ws', 'comment')]
            nxt = next_code.get(idx)
            nxt_word = nxt.upper if nxt is not None and nxt.kind == 'word' else ''
            nxt_semi = nxt is not None and nxt.kind == 'punct' and nxt.text == ';'
            if not code_before and word in ('BEGIN', 'DECLARE'):
                # BEGIN; / BEGIN TRANSACTION / BEGIN WORK open a transaction, not a block.
                if not (word == 'BEGIN' and (nxt is None or nxt_semi or nxt_word in ('TRANSACTION', 'WORK', 'NAME'))):
                    in_block = True
                    if word == 'DECLARE':
                        # DECLARE … ; BEGIN … END — the declarations' semicolons
                        # are inside the block; its BEGIN does not nest again.
                        depth, declare_pending = 1, True
                        current.append(tok)
                        continue
            if in_block:
                if word == 'BEGIN' and declare_pending:
                    declare_pending = False
                elif word in ('BEGIN', 'CASE', 'IF', 'FOR', 'WHILE', 'LOOP', 'REPEAT'):
                    prev = code_before[-1].upper if code_before else ''
                    # END IF / END FOR close a block; IF NOT EXISTS is DDL; CURSOR FOR declares a cursor.
                    if prev != 'END' and not (word == 'IF' and nxt_word in ('NOT', 'EXISTS')) \
                            and not (word == 'FOR' and prev == 'CURSOR'):
                        depth += 1
                elif word == 'END':
                    depth -= 1
        if tok.kind == 'punct' and tok.text == ';' and (not in_block or depth <= 0):
            flush(tok.start)
            in_block, depth, declare_pending = False, 0, False
            continue
        current.append(tok)
    flush(len(sql))
    return statements


def statement_at(sql: str, offset: int) -> Statement | None:
    """The statement containing (or immediately preceding) character ``offset``."""
    statements = split_statements(sql)
    if not statements:
        return None
    best = statements[0]
    for st in statements:
        if st.start <= offset:
            best = st
    return best


# ----------------------------------------------------------------------------- classification

READ_KEYWORDS = {'SELECT', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN', 'LIST', 'LS', 'VALUES'}
WRITE_DML = {'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'TRUNCATE', 'COPY', 'PUT', 'GET', 'REMOVE', 'RM', 'UNDROP'}
DDL = {'CREATE', 'ALTER', 'DROP', 'COMMENT', 'GRANT', 'REVOKE', 'EXECUTE', 'BEGIN', 'DECLARE',
       'COMMIT', 'ROLLBACK', 'START', 'SET', 'UNSET', 'CALL'}

# Never from a web UI, whatever the mode.
_ADMIN_OBJECTS = {'USER', 'ROLE', 'ACCOUNT', 'INTEGRATION', 'NETWORK', 'SECURITY', 'SHARE',
                  'REPLICATION', 'FAILOVER', 'CONNECTION', 'ORGANIZATION', 'PASSWORD', 'SESSION'}

_STATEFUL_SYSTEM_FN = re.compile(
    r'^SYSTEM\$(?:.*(?:CANCEL|ABORT|ENABLE|DISABLE|SET_|UNSET|RESUME|SUSPEND|MIGRATE|REFRESH|'
    r'AUTHORIZE|REVOKE|LINK|UNLINK|CREATE|DROP|ADD_|REMOVE|ROTATE|RESET|START|STOP|TRIGGER|'
    r'FORCE|PIPE_REBINDING|SEND|WAIT|RECLUSTER|CONVERT|COMMIT|ROLLBACK|RESTORE|FAILOVER|'
    r'PROMOTE|SNOWPIPE_STREAMING_UPDATE|REGISTER|DEREGISTER|ALLOWLIST_PRIVATELINK|SET_RETURN_VALUE).*)$')


@dataclass
class Verdict:
    kind: str                     # read | write | call | admin | session
    keyword: str
    procedure: str = ''           # fully-qualified upper-case name for CALL
    reason: str = ''


def _names_after(code: list[Token], idx: int) -> str:
    """Dotted object name starting at ``code[idx]`` (identifiers unquoted, upper-cased)."""
    parts, expect_part = [], True
    for tok in code[idx:]:
        if expect_part and tok.kind in ('word', 'ident'):
            parts.append(tok.text[1:-1].replace('""', '"') if tok.kind == 'ident' else tok.upper)
            expect_part = False
        elif not expect_part and tok.kind == 'punct' and tok.text == '.':
            expect_part = True
        else:
            break
    return '.'.join(parts)


def classify(statement: Statement) -> Verdict:
    code = statement.code_tokens
    if not code:
        raise SqlGuardError('Empty statement.', code='empty')
    first = code[0]
    kw = first.upper if first.kind == 'word' else ''
    if first.kind == 'punct' and first.text == '(':
        # (SELECT …) UNION … — find the first word inside the parentheses.
        kw = next((t.upper for t in code if t.kind == 'word'), '')

    if kw == 'USE':
        return Verdict('session', kw, reason='Pick the role, warehouse, database and schema with the context bar instead of USE.')

    if kw in ('CREATE', 'ALTER', 'DROP', 'GRANT', 'REVOKE', 'UNDROP', 'SHOW') or kw in ('DESCRIBE', 'DESC'):
        words = [t.upper for t in code[1:8] if t.kind == 'word']
        if kw in ('GRANT', 'REVOKE'):
            return Verdict('admin', kw, reason='Privileges are managed by an administrator, not from the Studio.')
        if kw in ('CREATE', 'ALTER', 'DROP', 'UNDROP'):
            target = [w for w in words if w not in ('OR', 'REPLACE', 'IF', 'NOT', 'EXISTS', 'SECURE', 'TEMPORARY',
                                                    'TEMP', 'TRANSIENT', 'LOCAL', 'GLOBAL', 'VOLATILE', 'RECURSIVE')]
            obj = target[0] if target else ''
            if kw == 'ALTER' and obj == 'SESSION':
                return Verdict('session', kw, reason='Session parameters are managed by the Studio.')
            if obj in _ADMIN_OBJECTS or (obj == 'API' and 'INTEGRATION' in words) or (obj in ('STORAGE', 'NOTIFICATION', 'EXTERNAL', 'SECURITY', 'CATALOG') and 'INTEGRATION' in words):
                return Verdict('admin', kw, reason=f'{kw} {obj} is account administration and is never run from the Studio.')
            return Verdict('write', kw)
        return Verdict('read', kw)

    if kw == 'WITH':
        # A CTE can front a DML statement in Snowflake (WITH … INSERT is not
        # valid, but WITH … SELECT inside CTAS etc. can be).  Any top-level
        # write keyword turns it into a write.
        depth = 0
        for tok in code[1:]:
            if tok.kind == 'punct' and tok.text == '(':
                depth += 1
            elif tok.kind == 'punct' and tok.text == ')':
                depth -= 1
            elif depth == 0 and tok.kind == 'word' and tok.upper in WRITE_DML | {'CREATE'}:
                return Verdict('write', tok.upper)
        kw = 'SELECT'

    if kw in READ_KEYWORDS:
        for i, tok in enumerate(code):
            if tok.kind == 'word' and tok.upper.startswith('SYSTEM$') and _STATEFUL_SYSTEM_FN.match(tok.upper):
                return Verdict('write', kw, reason=f'{tok.upper} changes account state.')
        return Verdict('read', kw)

    if kw == 'CALL':
        name = _names_after(code, 1)
        if not name:
            raise SqlGuardError('CALL needs a procedure name.', code='parse')
        return Verdict('call', kw, procedure=name)

    if kw in ('SET', 'UNSET'):
        return Verdict('session', kw, reason='Session variables are not kept between Studio runs; inline the value instead.')

    if kw in WRITE_DML or kw in DDL or kw in ('BEGIN', 'DECLARE', 'EXECUTE', 'COMMIT', 'ROLLBACK', 'START'):
        return Verdict('write', kw)

    if not kw:
        raise SqlGuardError('Could not understand this statement.', code='parse')
    return Verdict('write', kw, reason=f'{kw} is not a read-only statement.')


def qualify_procedure(name: str, database: str, schema: str) -> str:
    """Resolve a 1-, 2- or 3-part procedure name against the session context."""
    parts = [p for p in name.split('.') if p]
    if len(parts) == 1:
        parts = [database, schema] + parts
    elif len(parts) == 2:
        parts = [database] + parts
    return '.'.join(p.upper() for p in parts)


def check(sql: str, *, mode: str = READ, allow_writes: bool = False, callable_procedures: set[str] | frozenset = frozenset(),
          database: str = '', schema: str = '', single: bool = True) -> list[tuple[Statement, Verdict]]:
    """Split and validate ``sql``.  Returns ``[(statement, verdict)]`` or raises :class:`SqlGuardError`.

    ``callable_procedures`` holds fully-qualified upper-case names
    (``DB.SCHEMA.PROC``) that may be CALLed in read mode.
    """
    if not sql or not sql.strip():
        raise SqlGuardError('Write a SQL statement first.', code='empty')
    statements = split_statements(sql)
    if not statements:
        raise SqlGuardError('Nothing to run — the editor only contains comments.', code='empty')
    if single and len(statements) > 1:
        raise SqlGuardError(
            f'{len(statements)} statements were sent together; run them one at a time (Run all does this for you).',
            code='multiple')
    write_ok = mode == WRITE and allow_writes
    out = []
    for st in statements:
        verdict = classify(st)
        if verdict.kind == 'admin':
            raise SqlGuardError(verdict.reason, code='admin', statement=st.text)
        if verdict.kind == 'session':
            raise SqlGuardError(verdict.reason, code='session', statement=st.text)
        if verdict.kind == 'call':
            fq = qualify_procedure(verdict.procedure, database, schema)
            verdict.procedure = fq
            if fq not in callable_procedures and not write_ok:
                raise SqlGuardError(
                    f'{fq} is not on the Studio allow-list. Procedures can change data, so only the cost & '
                    f'performance toolkit runs in read mode' + ('' if allow_writes else '.') +
                    (' — unlock write mode to run it.' if allow_writes else ''),
                    code='call', statement=st.text)
        if verdict.kind == 'write' and not write_ok:
            why = verdict.reason or f'{verdict.keyword} changes data or objects.'
            hint = ' Unlock write mode to run it.' if allow_writes else ' The Studio is read-only on this deployment.'
            raise SqlGuardError(why + hint, code='write', statement=st.text)
        out.append((st, verdict))
    return out


_ERR_POS_RE = re.compile(r'line (\d+) at position (\d+)', re.IGNORECASE)


def error_position(message: str) -> dict | None:
    """Pull ``{line, column}`` (1-based line, 0-based column) out of a Snowflake compilation error."""
    m = _ERR_POS_RE.search(message or '')
    if not m:
        return None
    return {'line': int(m.group(1)), 'column': int(m.group(2))}
