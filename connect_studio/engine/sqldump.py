"""SQL dumps → DuckDB: restore a database dump without the database server.

A dump is a script of CREATE TABLE / INSERT / COPY statements. This module reads one (or several) as a stream and
rebuilds every table in a DuckDB file, so the rest of Connect Studio treats it like any other database: dbatlas
extraction, profiling, mapping and full-table reads for syncs.

Dialects (auto-detected; can be forced):
  mysql     mysqldump, phpMyAdmin, MariaDB, Adminer, HeidiSQL — backtick names, backslash escapes, extended
            INSERTs, `/*!…*/` comments, keys added later with ALTER TABLE, DELIMITER blocks (routines are skipped)
  postgres  pg_dump plain format — COPY … FROM stdin blocks or INSERT style, schema.table names, $$ bodies,
            E'' strings, psql meta commands (\\connect, \\restrict); custom / tar format via pg_restore when installed
  tsql      SQL Server (SSMS "Generate Scripts", sqlcmd) — [dbo].[x] names, N'' strings, GO batches,
            statements without semicolons, CAST(N'…' AS DateTime) and CAST(0x… AS DateTime) values
  sqlite    sqlite3 .dump
  oracle    SQL Developer / exp-to-SQL scripts — TO_DATE(…), q'' strings, '/' terminators, PL/SQL blocks skipped

Files: .sql (UTF-8, Latin-1 or UTF-16 with BOM), compressed .gz / .bz2 / .xz, and .zip archives of .sql files.

How values land: every table is loaded as text first (dumps are full of values a strict type would reject —
'0000-00-00', '' in numeric columns), then each column is converted to the type its CREATE TABLE declared when
every value converts. Columns that don't are kept as text and reported. Declared primary keys, foreign keys,
uniques, comments, defaults and enum values are kept in a side file (dump.meta.json) and merged into the snapshot,
because the DuckDB copy carries no constraints (dumps routinely violate their own keys).
"""

from __future__ import annotations

import bz2
import codecs
import gzip
import io
import lzma
import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

VERSION = 1                    # bump when restore output changes → cached restores are rebuilt
CHUNK = 4 * 1024 * 1024        # characters read per refill
FLUSH_ROWS = 50_000            # rows buffered per table before they're written to DuckDB
DIALECTS = ("mysql", "postgres", "tsql", "sqlite", "oracle")
DIALECT_LABELS = {"mysql": "MySQL / MariaDB", "postgres": "PostgreSQL", "tsql": "SQL Server", "sqlite": "SQLite",
                  "oracle": "Oracle"}

Progress = Callable[..., None]


def _noop(*_a, **_k):
    return None


class DumpError(Exception):
    """The dump can't be read; the message is user-facing."""


# ─── opening files ────────────────────────────────────────────────────────

class _Counting(io.RawIOBase):
    """Counts the compressed bytes consumed, for progress on .gz / .bz2 / .xz files."""

    def __init__(self, raw):
        self.raw, self.count = raw, 0

    def readable(self):
        return True

    def readinto(self, b):
        n = self.raw.readinto(b)
        self.count += n or 0
        return n

    def close(self):
        self.raw.close()
        super().close()


@dataclass
class DumpFile:
    name: str               # what the log calls it ("dump.sql.gz", "export.zip:tables/patients.sql")
    size: int               # bytes on disk (compressed size for compressed files)
    open_binary: Callable[[], tuple[Any, Callable[[], int]]]   # → (binary stream, bytes-consumed-so-far)


def _magic(path: Path) -> bytes:
    """The first 512 bytes: enough for compression magics, PGDMP, and a tar header (pg_dump -Ft)."""
    with open(path, "rb") as f:
        return f.read(512)


def dump_files(paths: list[Path]) -> list[DumpFile]:
    """Every SQL script in the uploads: plain, compressed, or members of zip archives (in name order)."""
    out: list[DumpFile] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise DumpError(f"{path.name}: file not found on the server.")
        magic = _magic(path)
        if magic.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(path) as z:
                members = sorted(i.filename for i in z.infolist()
                                 if not i.is_dir() and not i.filename.startswith("__MACOSX/")
                                 and re.search(r"\.(sql|txt|dump|ddl)(\.(gz|bz2|xz))?$", i.filename, re.I))
                if not members:
                    raise DumpError(f"{path.name}: the zip holds no .sql files.")
                for m in members:
                    info = z.getinfo(m)
                    out.append(DumpFile(f"{path.name}:{m}", info.compress_size,
                                        (lambda p=path, m=m: _open_zip_member(p, m))))
            continue
        out.append(DumpFile(path.name, path.stat().st_size, (lambda p=path, mg=magic: _open_compressed(p, mg))))
    return out


def _open_zip_member(path: Path, member: str):
    z = zipfile.ZipFile(path)
    raw = z.open(member)
    counter = _Counting(raw)
    stream = io.BufferedReader(counter, 1 << 20)
    if member.lower().endswith((".gz", ".bz2", ".xz")):
        stream = _decompressor(member.lower().rsplit(".", 1)[1], stream)
    return stream, lambda: counter.count


def _decompressor(kind: str, stream):
    return {"gz": lambda s: gzip.GzipFile(fileobj=s), "bz2": lambda s: bz2.BZ2File(s),
            "xz": lambda s: lzma.LZMAFile(s)}[kind](stream)


def _open_compressed(path: Path, magic: bytes):
    raw = open(path, "rb")
    counter = _Counting(raw)
    buffered = io.BufferedReader(counter, 1 << 20)
    if magic.startswith(b"\x1f\x8b"):
        return gzip.GzipFile(fileobj=buffered), lambda: counter.count
    if magic.startswith(b"BZh"):
        return bz2.BZ2File(buffered), lambda: counter.count
    if magic.startswith(b"\xfd7zXZ"):
        return lzma.LZMAFile(buffered), lambda: counter.count
    return buffered, lambda: counter.count


def text_stream(binary) -> io.TextIOBase:
    """Decode a dump: UTF-16/UTF-8 BOMs are honoured (SSMS writes UTF-16); otherwise UTF-8, falling back to
    Latin-1 per undecodable byte so a stray legacy character never stops a restore."""
    head = binary.peek(4)[:4] if hasattr(binary, "peek") else b""
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return io.TextIOWrapper(binary, encoding="utf-16", newline="")
    if len(head) >= 4 and head[1:2] == b"\x00" and head[3:4] == b"\x00":     # UTF-16 LE without BOM
        return io.TextIOWrapper(binary, encoding="utf-16-le", newline="")
    enc = "utf-8-sig" if head.startswith(codecs.BOM_UTF8) else "utf-8"
    return io.TextIOWrapper(binary, encoding=enc, errors="latin1_fallback", newline="")


def _latin1_fallback(err: UnicodeDecodeError):
    return err.object[err.start:err.end].decode("latin-1"), err.end


codecs.register_error("latin1_fallback", _latin1_fallback)


def binary_kind(magic: bytes) -> str | None:
    """Dumps that aren't SQL text: what they are (for a useful error), or None."""
    if magic.startswith(b"PGDMP"):
        return "pg_custom"
    if magic[257:262] == b"ustar" and magic[:7] == b"toc.dat":     # pg_dump --format=tar
        return "pg_custom"
    if magic[:5] == b"TAPE\x00" or magic[:4] == b"TAPE":
        return "mssql_bak"
    if magic.startswith(b"SQLite format 3"):
        return "sqlite_db"
    if b"EXPORT:V" in magic[:64]:                                   # Oracle exp (classic export) header
        return "oracle_dmp"
    return None


# ─── dialect detection ────────────────────────────────────────────────────

_SNIFF = [
    ("postgres", r"PostgreSQL database dump|^\s*COPY\s+\S+.*FROM\s+stdin;|SET\s+search_path\s*=|pg_catalog\.|^\\connect\s"),
    ("tsql", r"^\s*GO\s*$|\[dbo\]\.|SET\s+ANSI_NULLS|SET\s+IDENTITY_INSERT|CAST\(N'|NVARCHAR\s*\(|\bINSERT\s+\[[^\]]+\]"),
    ("mysql", r"MySQL dump|phpMyAdmin|MariaDB dump|ENGINE\s*=|/\*!\d{5}|`\w+`|AUTO_INCREMENT|DELIMITER\s"),
    ("oracle", r"VARCHAR2\s*\(|\bTO_DATE\s*\(|\bNUMBER\s*\(|^\s*/\s*$|SET\s+DEFINE\s+OFF|\bNVARCHAR2\b"),
    ("sqlite", r"PRAGMA\s+foreign_keys|BEGIN TRANSACTION;|sqlite_sequence|CREATE TABLE IF NOT EXISTS \"?\w+\"? \("),
]


def detect_dialect(sample: str) -> str:
    """Best guess from the first few MB: the dialect whose markers appear most (MySQL when nothing stands out)."""
    scores = {}
    for name, pattern in _SNIFF:
        scores[name] = len(re.findall(pattern, sample, re.I | re.M))
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] else "mysql"


# ─── statement splitting ──────────────────────────────────────────────────

_TSQL_LINE_START = re.compile(
    r"[ \t]*(?:INSERT|SET|GO|CREATE|ALTER|USE|PRINT|DROP|IF|EXEC|EXECUTE|DECLARE|BEGIN|COMMIT|ROLLBACK|UPDATE|DELETE|"
    r"TRUNCATE|GRANT|DENY|REVOKE|DBCC|RAISERROR|WHILE|CHECKPOINT)\b", re.I)
_ROUTINE = re.compile(r"\s*CREATE\s+(?:OR\s+(?:REPLACE|ALTER)\s+)?(?:DEFINER\s*=\s*\S+\s+)?(?:EDITIONABLE\s+|NONEDITIONABLE\s+)?"
                      r"(PROCEDURE|PROC|FUNCTION|TRIGGER|PACKAGE|TYPE\s+BODY|EVENT)\b", re.I)


_WS = re.compile(r"\s*")


@dataclass
class Statement:
    kind: str               # "sql" | "copy_data" | "meta"
    text: str = ""
    lines: list[str] = field(default_factory=list)


class Splitter:
    """Streams statements out of a dump, honouring each dialect's quoting, comments and terminators."""

    def __init__(self, stream: io.TextIOBase, dialect: str):
        self.stream, self.d = stream, dialect
        self.buf, self.pos, self.eof = "", 0, False
        self.delimiter = ";"
        self.copy_mode = False
        self._compile()

    def _compile(self) -> None:
        d = self.d
        if d == "mysql":
            s = r"'(?:[^'\\]|\\.|'')*'"
            q = r'"(?:[^"\\]|\\.|"")*"'
        else:
            s = r"'(?:[^']|'')*'"
            q = r'"(?:[^"]|"")*"'
        parts = [f"(?P<s>{s})", f"(?P<q>{q})"]
        if d in ("mysql", "sqlite"):
            parts.append(r"(?P<b>`[^`]*`)")
        if d in ("tsql", "sqlite"):
            parts.append(r"(?P<br>\[(?:[^\]]|\]\])*\])")
        if d == "postgres":
            parts.append(r"(?P<dq>\$(?P<tag>[A-Za-z_]\w*)?\$.*?\$(?P=tag)\$)")
        line_comment = r"--(?=[ \t\r\n]|$)[^\n]*" if d == "mysql" else r"--[^\n]*"
        if d == "mysql":
            line_comment += r"|\#[^\n]*"
        parts += [rf"(?P<lc>{line_comment})", r"(?P<bc>/\*.*?\*/)", r"(?P<nl>\n)"]
        delim_first = re.escape(self.delimiter[0])
        stop = "'\"`\\[\\-#/\n$" + delim_first
        parts.append(f"(?P<plain>[^{stop}]+)")
        parts.append(r"(?P<other>.)")
        self.token = re.compile("|".join(parts), re.S)

    def _refill(self) -> bool:
        if self.eof:
            return False
        chunk = self.stream.read(CHUNK)
        if not chunk:
            self.eof = True
            return False
        self.buf = self.buf[self.pos:] + chunk
        self.pos = 0
        return True

    def __iter__(self) -> Iterator[Statement]:
        while True:
            st = self._next()
            if st is None:
                return
            yield st

    def _next(self) -> Statement | None:
        if self.copy_mode:
            return self._copy_data()
        while True:   # skip blank space between statements
            m = _WS.match(self.buf, self.pos)
            self.pos = m.end()
            if self.pos < len(self.buf):
                break
            if not self._refill():
                return None
        # line-level commands
        line_end = self.buf.find("\n", self.pos)
        while line_end < 0 and not self.eof and self._refill():
            line_end = self.buf.find("\n", self.pos)
        line = self.buf[self.pos: line_end if line_end >= 0 else len(self.buf)]
        if self.d == "mysql" and re.match(r"DELIMITER\s+\S+", line, re.I):
            self.delimiter = line.split()[1]
            self._compile()
            self.pos = line_end + 1 if line_end >= 0 else len(self.buf)
            return Statement("meta", line)
        if self.d == "postgres" and line.startswith("\\"):
            self.pos = line_end + 1 if line_end >= 0 else len(self.buf)
            return Statement("meta", line)
        if self.d in ("tsql",) and re.fullmatch(r"\s*GO\s*(\d+)?\s*", line, re.I):
            self.pos = line_end + 1 if line_end >= 0 else len(self.buf)
            return Statement("meta", "GO")
        if self.d == "oracle" and line.strip() == "/":
            self.pos = line_end + 1 if line_end >= 0 else len(self.buf)
            return Statement("meta", "/")
        # T-SQL / Oracle routine bodies hold inner ';' — they run to GO / '/'. MySQL routines sit in DELIMITER blocks.
        routine = self.d in ("tsql", "oracle") and bool(_ROUTINE.match(self.buf, self.pos))
        return self._statement(routine)

    def _statement(self, routine: bool) -> Statement:
        start = self.pos
        i = start
        while True:
            m = self.token.match(self.buf, i)
            if m is None or (m.end() >= len(self.buf) and not self.eof):
                # a token may continue past the buffer (unterminated string, comment, plain run): read more
                if self._refill_keep(start):
                    i, start = i - start, 0
                    continue
                if m is None:     # end of input
                    break
            kind = m.lastgroup
            if kind == "other" and not self.eof and self._opener(i):
                # a quote / comment opener with no match yet: its end is further on
                if self._refill_keep(start):
                    i, start = i - start, 0
                    continue
            if kind == "nl":
                nxt = i + 1
                if self.d == "tsql":
                    rest = self.buf[nxt: nxt + 64]
                    if not self.eof and len(rest) < 64 and self._refill_keep(start):
                        i, start = i - start, 0
                        continue
                    if re.match(r"[ \t]*GO[ \t]*(\d+[ \t]*)?\r?(\n|$)", rest, re.I):
                        return self._emit(start, i, nxt)
                    if not routine and _TSQL_LINE_START.match(rest) and self.buf[start:i].strip():
                        return self._emit(start, i, nxt)
                if self.d == "oracle":
                    rest = self.buf[nxt: nxt + 8]
                    if re.match(r"[ \t]*/[ \t]*\r?(\n|$)", rest):
                        return self._emit(start, i, nxt)
            if kind == "other" and not routine and self.buf.startswith(self.delimiter[0], i):
                if len(self.buf) - i < len(self.delimiter) and not self.eof and self._refill_keep(start):
                    i, start = i - start, 0
                    continue
                if self.buf.startswith(self.delimiter, i):
                    return self._emit(start, i, i + len(self.delimiter))
            i = m.end()
            if i >= len(self.buf) and self.eof:
                break
        text = self.buf[start:]
        self.pos = len(self.buf)
        return Statement("sql", text)

    def _opener(self, i: int) -> bool:
        """Does an unmatched character at i open a quote / comment whose end hasn't been read yet?"""
        ch = self.buf[i]
        if ch in "'\"":
            return True
        if ch == "`":
            return self.d in ("mysql", "sqlite")
        if ch == "[":
            return self.d in ("tsql", "sqlite")
        if ch == "/":
            return self.buf.startswith("/*", i) or i + 1 >= len(self.buf)
        if ch == "$":
            return self.d == "postgres" and bool(re.match(r"\$(?:[A-Za-z_]\w*)?(\$|$)", self.buf[i:i + 64]))
        return False

    def _refill_keep(self, start: int) -> bool:
        """Refill, keeping the current statement (from `start`) at the front of the buffer."""
        if self.eof:
            return False
        self.pos = start
        return self._refill()

    def _emit(self, start: int, end: int, resume: int) -> Statement:
        text = self.buf[start:end]
        self.pos = resume
        if self.d == "postgres" and re.match(r"\s*COPY\s.+\bFROM\s+stdin\b", strip_comments(text[:4000]), re.I | re.S):
            self.copy_mode = True
            # the data starts on the next line (the rest of the COPY line is just its newline)
            while (nl := self.buf.find("\n", self.pos)) < 0 and self._refill():
                pass
            self.pos = nl + 1 if nl >= 0 else len(self.buf)
        return Statement("sql", text)

    def _copy_data(self) -> Statement:
        """pg_dump COPY data: tab-separated lines up to a line holding only \\. — yielded in batches."""
        lines: list[str] = []
        while True:
            nl = self.buf.find("\n", self.pos)
            if nl < 0:
                if self._refill():
                    continue
                tail = self.buf[self.pos:]
                self.pos = len(self.buf)
                if tail and tail.rstrip("\r") != "\\.":
                    lines.append(tail.rstrip("\r"))
                self.copy_mode = False
                return Statement("copy_data", lines=lines + ["\\."])
            line = self.buf[self.pos:nl].rstrip("\r")
            self.pos = nl + 1
            if line == "\\.":
                self.copy_mode = False
                return Statement("copy_data", lines=lines + ["\\."])
            lines.append(line)
            if len(lines) >= FLUSH_ROWS:
                return Statement("copy_data", lines=lines)


_COMMENTS = re.compile(r"--[^\n]*|/\*(?!!).*?\*/|\#[^\n]*", re.S)


def strip_comments(text: str) -> str:
    """Leading comments and blank space off a statement (MySQL /*!…*/ version comments are kept: they're code)."""
    t = text
    while True:
        t2 = re.sub(r"^\s+", "", t)
        m = re.match(r"--[^\n]*\n?|/\*(?!!).*?\*/|\#[^\n]*\n?", t2, re.S)
        if not m:
            return t2
        t = t2[m.end():]


# ─── identifiers ──────────────────────────────────────────────────────────

_IDENT_PART = r'(?:`[^`]+`|"(?:[^"]|"")+"|\[(?:[^\]]|\]\])+\]|[A-Za-z_#@$][\w$#@]*)'
NAME = rf"{_IDENT_PART}(?:\s*\.\s*{_IDENT_PART}){{0,2}}"


def unquote(part: str) -> str:
    part = part.strip()
    if len(part) >= 2 and part[0] == part[-1] and part[0] in "`\"":
        return part[1:-1].replace(part[0] * 2, part[0])
    if part.startswith("[") and part.endswith("]"):
        return part[1:-1].replace("]]", "]")
    return part


def split_name(text: str) -> tuple[str, str]:
    """'`db`.`patients`' → ('db', 'patients'); 'public.visits' → ('public', 'visits'); 'x' → ('', 'x')."""
    parts = [unquote(p) for p in re.findall(_IDENT_PART, text)]
    if not parts:
        return "", text.strip()
    return (parts[-2] if len(parts) >= 2 else ""), parts[-1]


def ident_list(text: str) -> list[str]:
    """'(`a`, "b" ASC, [c](10))' → ['a', 'b', 'c'] (index prefixes and sort order dropped)."""
    out = []
    for item in _split_top(text.strip().strip("()")):
        item = item.strip()
        m = re.match(_IDENT_PART, item)
        if m:
            out.append(unquote(m.group(0)))
    return out


def _split_top(text: str, sep: str = ",") -> list[str]:
    """Split on `sep` outside parentheses and quotes."""
    out, depth, cur, i, quote = [], 0, [], 0, None
    while i < len(text):
        ch = text[i]
        if quote:
            cur.append(ch)
            if ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    cur.append(text[i + 1])
                    i += 1
                else:
                    quote = None
            elif ch == "\\" and quote == "'" and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
        elif ch in "'\"`":
            quote = ch
            cur.append(ch)
        elif ch == "[":
            j = text.find("]", i)
            j = len(text) - 1 if j < 0 else j
            cur.append(text[i:j + 1])
            i = j
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    if "".join(cur).strip():
        out.append("".join(cur))
    return out


# ─── DDL ──────────────────────────────────────────────────────────────────

_CONSTRAINT_WORDS = re.compile(
    r"\b(NOT\s+NULL|NULL|DEFAULT|PRIMARY\s+KEY|UNIQUE|REFERENCES|CHECK|COLLATE|CHARACTER\s+SET|CHARSET|AUTO_INCREMENT|"
    r"AUTOINCREMENT|IDENTITY|COMMENT|CONSTRAINT|GENERATED|ON\s+UPDATE|ENCODE|SPARSE|ROWGUIDCOL|FILESTREAM|"
    r"MASKED|INVISIBLE|VISIBLE|STORAGE|COLUMN_FORMAT|ENABLE|DISABLE|AS\s*\()", re.I)
_TABLE_CONSTRAINT = re.compile(r"\s*(CONSTRAINT\b|PRIMARY\s+KEY\b|UNIQUE\b|FOREIGN\s+KEY\b|KEY\b|INDEX\b|FULLTEXT\b|"
                               r"SPATIAL\b|CHECK\b|EXCLUDE\b|PERIOD\b)", re.I)


@dataclass
class Column:
    name: str
    declared: str = ""
    nullable: bool = True
    default: str | None = None
    comment: str | None = None
    pk: bool = False
    unique: bool = False
    enum: list[str] = field(default_factory=list)
    auto: bool = False
    ref: tuple | None = None       # inline REFERENCES (table text, columns)
    added: bool = False            # came from ALTER TABLE … ADD COLUMN


@dataclass
class Table:
    schema: str
    name: str
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    uniques: list[list[str]] = field(default_factory=list)
    foreign_keys: list[dict] = field(default_factory=list)
    comment: str | None = None
    declared: bool = False      # a CREATE TABLE was seen (vs. a table known only from its INSERTs)

    def col(self, name: str) -> Column | None:
        low = name.lower()
        return next((c for c in self.columns if c.name.lower() == low), None)


def _string_value(lit: str) -> str:
    lit = lit.strip()
    m = re.match(r"^[NnEe]?'(.*)'$", lit, re.S)
    return m.group(1).replace("''", "'").replace("\\'", "'") if m else lit


def parse_create_table(stmt: str) -> Table | None:
    m = re.match(rf"\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:(?:GLOBAL|LOCAL)\s+)?(?:TEMPORARY\s+|TEMP\s+|UNLOGGED\s+)?TABLE\s+"
                 rf"(?:IF\s+NOT\s+EXISTS\s+)?(?:ONLY\s+)?(?P<name>{NAME})\s*\(", stmt, re.I | re.S)
    if not m:
        return None
    schema, name = split_name(m.group("name"))
    body_start = m.end() - 1
    depth, i, quote = 0, body_start, None
    while i < len(stmt):            # find the closing parenthesis of the column list
        ch = stmt[i]
        if quote:
            if ch == quote:
                if i + 1 < len(stmt) and stmt[i + 1] == quote:
                    i += 1
                else:
                    quote = None
            elif ch == "\\" and quote == "'":
                i += 1
        elif ch in "'\"`":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body, tail = stmt[body_start + 1:i], stmt[i + 1:]
    t = Table(schema, name, declared=True)
    for item in _split_top(body):
        item = item.strip()
        if not item:
            continue
        if _TABLE_CONSTRAINT.match(item):
            _table_constraint(t, item)
            continue
        col = _column_def(item)
        if col:
            t.columns.append(col)
    cm = re.search(r"\bCOMMENT\s*=?\s*('(?:[^'\\]|\\.|'')*')", tail, re.I)
    if cm:
        t.comment = _string_value(cm.group(1))
    for c in t.columns:
        if c.pk and c.name not in t.primary_key:
            t.primary_key.append(c.name)
    return t


def _column_def(item: str) -> Column | None:
    m = re.match(rf"\s*({_IDENT_PART})\s*(.*)$", item, re.S)
    if not m:
        return None
    name, rest = unquote(m.group(1)), m.group(2)
    # the type runs until the first constraint keyword at parenthesis depth 0
    depth, cut = 0, len(rest)
    for tok in re.finditer(r"\(|\)|'(?:[^'\\]|\\.|'')*'|" + _CONSTRAINT_WORDS.pattern, rest, re.I | re.S):
        g = tok.group(0)
        if g == "(":
            depth += 1
        elif g == ")":
            depth -= 1
        elif depth == 0 and not g.startswith("'"):
            cut = tok.start()
            break
    declared = " ".join(rest[:cut].split())
    flags = rest[cut:]
    col = Column(name, declared=declared.replace("[", "").replace("]", ""))
    if re.search(r"\bNOT\s+NULL\b", flags, re.I):
        col.nullable = False
    if re.search(r"\bPRIMARY\s+KEY\b", flags, re.I):
        col.pk, col.nullable = True, False
    if re.search(r"\bUNIQUE\b", flags, re.I):
        col.unique = True
    if re.search(r"\b(AUTO_INCREMENT|AUTOINCREMENT|IDENTITY)\b", flags, re.I) or re.match(r"(big|small)?serial\b", declared, re.I):
        col.auto = True
    dm = re.search(r"\bDEFAULT\s+(\((?:[^()]|\([^()]*\))*\)|'(?:[^'\\]|\\.|'')*'|[^\s,]+)", flags, re.I | re.S)
    if dm:
        col.default = dm.group(1).strip()
    cm = re.search(r"\bCOMMENT\s+('(?:[^'\\]|\\.|'')*')", flags, re.I | re.S)
    if cm:
        col.comment = _string_value(cm.group(1))
    em = re.match(r"(?:enum|set)\s*\((.*)\)\s*$", declared, re.I | re.S)
    if em:
        col.enum = [_string_value(v) for v in _split_top(em.group(1))]
    ref = re.search(rf"\bREFERENCES\s+({NAME})\s*(\([^)]*\))?", flags, re.I | re.S)
    if ref:
        col.ref = (ref.group(1), ident_list(ref.group(2) or ""))
    return col


def _fk_clauses(text: str) -> list[dict]:
    out = []
    for m in re.finditer(rf"FOREIGN\s+KEY\s*(?:{_IDENT_PART}\s*)?(\([^)]*\))\s*REFERENCES\s+({NAME})\s*(\([^)]*\))?",
                         text, re.I | re.S):
        out.append({"columns": ident_list(m.group(1)), "ref": split_name(m.group(2)),
                    "ref_columns": ident_list(m.group(3) or "")})
    return out


def _table_constraint(t: Table, item: str) -> None:
    pk = re.search(r"PRIMARY\s+KEY\s*(?:CLUSTERED|NONCLUSTERED)?\s*(?:USING\s+\w+\s*)?(\([^)]*\))", item, re.I)
    if pk:
        t.primary_key = ident_list(pk.group(1))
        return
    fks = _fk_clauses(item)
    if fks:
        t.foreign_keys.extend(fks)
        return
    uq = re.search(rf"UNIQUE\s*(?:KEY|INDEX|CONSTRAINT)?\s*(?:{_IDENT_PART}\s*)?(?:CLUSTERED|NONCLUSTERED)?\s*(?:USING\s+\w+\s*)?(\([^)]*\))",
                   item, re.I)
    if uq:
        t.uniques.append(ident_list(uq.group(1)))


def parse_alter_table(stmt: str) -> tuple[tuple[str, str], Table] | None:
    """ALTER TABLE … ADD PRIMARY KEY / FOREIGN KEY / UNIQUE (phpMyAdmin, pg_dump and SSMS add keys this way)."""
    m = re.match(rf"\s*ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?(?P<name>{NAME})(?P<rest>.*)$", stmt, re.I | re.S)
    if not m:
        return None
    key = split_name(m.group("name"))
    patch = Table(*key)
    rest = m.group("rest")
    for clause in _split_top(rest):
        if re.search(r"\bDROP\b", clause, re.I) and not re.search(r"\bADD\b", clause, re.I):
            continue
        dm = re.search(rf"\bADD\s+(?:CONSTRAINT\s+{_IDENT_PART}\s+)?DEFAULT\s+(.+?)\s+FOR\s+({_IDENT_PART})\s*$", clause, re.I | re.S)
        if dm:          # T-SQL: ADD DEFAULT ((1)) FOR [IsActive]
            expr = dm.group(1).strip()
            while expr.startswith("(") and expr.endswith(")") and len(_split_top(expr[1:-1])) == 1 \
                    and expr[1:-1].count("(") == expr[1:-1].count(")"):
                expr = expr[1:-1].strip()
            patch.columns.append(Column(unquote(dm.group(2)), default=expr))
            continue
        om = re.search(rf"\bMODIFY\s*\(\s*({_IDENT_PART})\s+(?:CONSTRAINT\s+{_IDENT_PART}\s+)?NOT\s+NULL", clause, re.I | re.S)
        if om:          # Oracle: MODIFY ("NAME" NOT NULL ENABLE)
            patch.columns.append(Column(unquote(om.group(1)), nullable=False))
            continue
        _table_constraint(patch, clause)
        cm = (re.search(rf"\bMODIFY\s+(?:COLUMN\s+)?({_IDENT_PART}\s+.*)$", clause, re.I | re.S)
              or re.search(rf"\bCHANGE\s+(?:COLUMN\s+)?{_IDENT_PART}\s+({_IDENT_PART}\s+.*)$", clause, re.I | re.S))
        if cm:          # MODIFY `id` int(11) NOT NULL AUTO_INCREMENT — keep its nullability / auto flag
            col = _column_def(cm.group(1))
            if col:
                patch.columns.append(col)
        am = re.search(rf"\bADD\s+(?:COLUMN\s+)?(?!(?:CONSTRAINT|PRIMARY|UNIQUE|FOREIGN|KEY|INDEX|CHECK|FULLTEXT|SPATIAL|"
                       rf"DEFAULT|PERIOD|PARTITION)\b)({_IDENT_PART}\s+\w.*)$", clause, re.I | re.S)
        if am:
            col = _column_def(am.group(1))
            if col:
                col.added = True
                patch.columns.append(col)
    return key, patch


# ─── values ───────────────────────────────────────────────────────────────

_WRAPPERS = {"cast", "convert", "to_date", "to_timestamp", "to_timestamp_tz", "date", "timestamp", "datetime", "time",
             "n", "unhex", "x", "char", "nchar", "str_to_date", "parsedatetime", "datetime2fromparts"}
_MYSQL_ESC = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a", "\\": "\\", "'": "'", '"': '"',
              "%": "\\%", "_": "\\_"}


def _unescape_mysql(s: str) -> str:
    if "\\" not in s and "''" not in s:
        return s
    s = s.replace("''", "'")
    return re.sub(r"\\(.)", lambda m: _MYSQL_ESC.get(m.group(1), m.group(1)), s, flags=re.S)


def _unescape_c(s: str) -> str:
    """PostgreSQL E'' strings."""
    def rep(m):
        g = m.group(1)
        if g[0] in "01234567":
            return chr(int(g, 8))
        if g[0] == "x":
            return chr(int(g[1:], 16))
        return {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}.get(g, g)
    return re.sub(r"\\(x[0-9A-Fa-f]{1,2}|[0-7]{1,3}|.)", rep, s.replace("''", "'"), flags=re.S)


def _tsql_binary_datetime(hexstr: str, sqltype: str) -> str | None:
    """SSMS (pre-2012) scripts datetimes as raw bytes: CAST(0x00009E2C00A4CB80 AS DateTime)."""
    h = hexstr[2:] if hexstr.lower().startswith("0x") else hexstr
    t = sqltype.lower()
    try:
        if t == "datetime" and len(h) == 16:
            days = int.from_bytes(bytes.fromhex(h[:8]), "big", signed=True)
            ticks = int(h[8:], 16)
            return (datetime(1900, 1, 1) + timedelta(days=days, milliseconds=round(ticks * 10 / 3))).isoformat(" ")
        if t == "smalldatetime" and len(h) == 8:
            return (datetime(1900, 1, 1) + timedelta(days=int(h[:4], 16), minutes=int(h[4:], 16))).isoformat(" ")
        if t == "date" and len(h) in (6, 8):     # 3 bytes little-endian days since 0001-01-01 (SSMS pads to 4)
            days = int.from_bytes(bytes.fromhex(h[:6]), "little")
            return (datetime(1, 1, 1) + timedelta(days=days)).date().isoformat()
    except (ValueError, OverflowError):
        return None
    return None


_ORA_FMT = [("YYYY", "%Y"), ("RRRR", "%Y"), ("SYYYY", "%Y"), ("HH24", "%H"), ("HH12", "%I"), ("MONTH", "%B"),
            ("MON", "%b"), ("DAY", "%A"), ("DY", "%a"), ("SSXFF", "%S.%f"), ("XFF", ".%f"), ("FF9", "%f"), ("FF6", "%f"),
            ("FF3", "%f"), ("FF", "%f"), ("YY", "%y"), ("RR", "%y"), ("MM", "%m"), ("DD", "%d"), ("HH", "%I"),
            ("MI", "%M"), ("SS", "%S"), ("AM", "%p"), ("PM", "%p"), ("A.M.", "%p"), ("P.M.", "%p")]
_ORA_TOKEN = re.compile("|".join(re.escape(k) for k, _ in _ORA_FMT) + r'|"[^"]*"|.', re.I)


def _oracle_datetime(value: str, fmt: str) -> str | None:
    """TO_DATE('01-MAY-90', 'DD-MON-RR') → '1990-05-01' (Oracle format mask → strptime)."""
    out, has_time = [], False
    lookup = {k: v for k, v in _ORA_FMT}
    for m in _ORA_TOKEN.finditer(fmt.strip()):
        g = m.group(0)
        up = g.upper()
        if up in lookup:
            out.append(lookup[up])
            has_time = has_time or up in ("HH24", "HH12", "HH", "MI", "SS", "SSXFF")
        elif g.startswith('"'):
            out.append(g[1:-1].replace("%", "%%"))
        else:
            out.append(g.replace("%", "%%"))
    v = re.sub(r"(\.\d{6})\d+", r"\1", value.strip().replace("A.M.", "AM").replace("P.M.", "PM"))
    try:
        dt = datetime.strptime(v, "".join(out))
    except ValueError:
        return None
    return dt.isoformat(" ") if has_time else dt.date().isoformat()


def _mysql_datetime(value: str, fmt: str) -> str | None:
    """STR_TO_DATE('31/12/2020', '%d/%m/%Y') — MySQL's specifiers are strptime's, bar a few."""
    f = fmt.replace("%i", "%M").replace("%s", "%S").replace("%e", "%d").replace("%c", "%m").replace("%k", "%H")
    try:
        dt = datetime.strptime(value.strip(), f)
    except ValueError:
        return None
    return dt.isoformat(" ") if re.search(r"%[HIMSkl]", f) else dt.date().isoformat()


_PREFIXED = re.compile(r"(_\w+\s*|[NnEeXxBb])?'(.*)'$", re.S)


class ValueParser:
    """Rows out of an INSERT … VALUES (…), (…) list — literals, NULL, and the wrapper calls dumps use for typed values."""

    def __init__(self, dialect: str):
        self.d = dialect
        s = r"'(?:[^'\\]|\\.|'')*'" if dialect == "mysql" else r"'(?:[^']|'')*'"
        self.tok = re.compile(
            r"(?P<ws>\s+)|(?P<str>(?:_\w+\s*|[NnEeXxBb])?" + s + r")|(?P<hex>0[xX][0-9A-Fa-f]*)|"
            r"(?P<num>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)|(?P<word>[A-Za-z_][\w$]*)|(?P<lp>\()|(?P<rp>\))|"
            r"(?P<comma>,)|(?P<cast>::\s*[A-Za-z_][\w ]*(?:\([^)]*\))?)|(?P<other>.)", re.S)

    def literal(self, tok: str) -> str:
        if tok[0] == "'":                                   # the common case: no prefix
            body = tok[1:-1]
            if self.d == "mysql":
                return _unescape_mysql(body) if ("\\" in body or "''" in body) else body
            return body.replace("''", "'") if "''" in body else body
        m = _PREFIXED.match(tok)
        prefix, body = (m.group(1) or "").strip().lower(), m.group(2)
        if prefix == "x":
            return "0x" + body
        if prefix == "b":
            return str(int(body, 2)) if body and set(body) <= {"0", "1"} else body
        if self.d == "mysql":
            return _unescape_mysql(body)
        if prefix == "e":
            return _unescape_c(body)
        return body.replace("''", "'")

    def rows(self, text: str, start: int = 0) -> Iterator[list[str | None]]:
        tok, n = self.tok, len(text)
        i = start
        row: list | None = None
        while i < n:
            m = tok.match(text, i)
            kind = m.lastgroup
            i = m.end()
            if kind in ("ws", "comma") and row is None:
                continue
            if kind == "lp" and row is None:
                row = []
                i = self._row(text, i, row)
                yield row
                row = None
                continue
            if row is None:       # ON DUPLICATE KEY UPDATE …, RETURNING …, a trailing ')' — the values are over
                return

    def _row(self, text: str, i: int, row: list) -> int:
        match, n = self.tok.match, len(text)
        append = row.append
        expect_value = True
        while i < n:
            m = match(text, i)
            kind = m.lastgroup
            if kind == "ws":
                i = m.end()
                continue
            if kind == "str":                    # fast paths for the bulk of a dump: plain literals
                append(self.literal(m.group(0)))
                i, expect_value = m.end(), False
                continue
            if kind == "num":
                g = m.group(0)
                append(g[1:] if g[0] == "+" else g)
                i, expect_value = m.end(), False
                continue
            if kind == "comma":
                if expect_value:
                    append(None)
                expect_value = True
                i = m.end()
                continue
            if kind == "rp":
                if expect_value and row:
                    append(None)
                return m.end()
            if kind == "cast":       # postgres 'x'::date — the literal is already in the row
                i = m.end()
                continue
            value, i = self._value(text, i, m)
            append(value)
            expect_value = False
        return i

    def _value(self, text: str, i: int, m=None) -> tuple[str | None, int]:
        m = m or self.tok.match(text, i)
        kind, g = m.lastgroup, m.group(0)
        if kind == "str":
            return self.literal(g), m.end()
        if kind in ("num", "hex"):
            return g.lstrip("+"), m.end()
        if kind == "word":
            j = m.end()
            ws = _WS.match(text, j).end()
            if ws < len(text) and text[ws] == "(":
                return self._call(g, text, ws)
            low = g.lower()
            if low.startswith("_") and ws < len(text):          # MySQL charset introducer: _binary 0x…, _utf8mb4'…'
                return self._value(text, ws)
            if low in ("null", "default"):
                return None, j
            if low in ("true", "false"):
                return low, j
            if low in ("date", "timestamp", "time", "interval"):   # DATE '2020-01-01'
                m2 = self.tok.match(text, ws)
                if m2 and m2.lastgroup == "str":
                    return self.literal(m2.group(0)), m2.end()
            return g, j
        if kind == "other" and g in "-+":     # signed expression like - 5
            v, j = self._value(text, _WS.match(text, m.end()).end())
            return (g + v if g == "-" and v is not None else v), j
        if kind == "lp":                      # parenthesised value: (1), ('x')
            inner: list = []
            j = self._row(text, m.end(), inner)
            return (inner[0] if inner else None), j
        return g, m.end()

    def _call(self, fn: str, text: str, lp: int) -> tuple[str | None, int]:
        """FUNC( … ): the matching ')' and, for type wrappers, the literal inside."""
        depth, i = 0, lp
        lits: list[tuple[str, str]] = []
        while i < len(text):
            m = self.tok.match(text, i)
            kind, g = m.lastgroup, m.group(0)
            if kind == "lp":
                depth += 1
            elif kind == "rp":
                depth -= 1
                if depth == 0:
                    i = m.end()
                    break
            elif kind in ("str", "num", "hex"):
                lits.append((kind, g))
            i = m.end()
        raw = fn + text[lp:i]
        low = fn.lower()
        strs = [self.literal(g) for k, g in lits if k == "str"]
        if low in ("to_date", "to_timestamp", "to_timestamp_tz", "str_to_date") and len(strs) >= 2:
            conv = (_mysql_datetime if low == "str_to_date" else _oracle_datetime)(strs[0], strs[1])
            if conv:
                return conv, i
        if low in _WRAPPERS and lits:
            # the first string literal (CONVERT(varchar(10), '…') has a number first), else the first literal
            kind, g = next((x for x in lits if x[0] == "str"), lits[0])
            if kind == "hex" and low == "cast":
                tm = re.search(r"\bAS\s+\[?(\w+)\]?", raw, re.I)
                dt = _tsql_binary_datetime(g, tm.group(1)) if tm else None
                if dt:
                    return dt, i
            return (self.literal(g) if kind == "str" else g), i
        if low in ("empty_clob", "empty_blob"):
            return None, i
        return raw, i


_INSERT = re.compile(rf"\s*(?:INSERT|REPLACE)\s+(?:(?:LOW_PRIORITY|DELAYED|HIGH_PRIORITY|IGNORE|OR\s+\w+)\s+)*(?:INTO\s+)?"
                     rf"(?P<name>{NAME})\s*(?P<cols>\((?:[^()'\"`\[]|'[^']*'|\"[^\"]*\"|`[^`]*`|\[[^\]]*\])*\))?\s*"
                     rf"(?:OVERRIDING\s+\w+\s+VALUE\s*)?(?P<kw>VALUES?|SELECT|WITH|DEFAULT\s+VALUES)\b", re.I | re.S)
_COPY = re.compile(rf"\s*COPY\s+(?P<name>{NAME})\s*(?P<cols>\([^)]*\))?\s*FROM\s+stdin\b(?P<opts>.*)$", re.I | re.S)


def copy_unescape(field_: str) -> str | None:
    if field_ == "\\N":
        return None
    if "\\" not in field_:
        return field_
    return _unescape_c(field_)


# ─── types ────────────────────────────────────────────────────────────────

def duck_type(declared: str, dialect: str) -> str:
    """The DuckDB type a declared column type converts to (VARCHAR when nothing safer fits)."""
    t = " ".join(declared.lower().replace("[", "").replace("]", "").split())
    base = re.sub(r"\(.*", "", t).strip()
    args = [a.strip() for a in re.findall(r"\(([^)]*)\)", t)[0].split(",")] if "(" in t else []
    unsigned = "unsigned" in t
    if re.match(r"(tinyint|smallint|mediumint|int|integer|bigint|int2|int4|int8|serial|bigserial|smallserial|"
                r"byteint|year)\b", base):
        return "HUGEINT" if base.startswith("bigint") and unsigned else "BIGINT"
    if base in ("bit",) and (not args or args == ["1"]):
        return "BOOLEAN" if dialect == "tsql" else "BIGINT"
    if base in ("bool", "boolean"):
        return "BOOLEAN"
    if re.match(r"(decimal|numeric|number|dec|fixed)\b", base):
        if len(args) >= 1 and args[0].isdigit():
            p = int(args[0])
            s = int(args[1]) if len(args) > 1 and args[1].lstrip("-").isdigit() else 0
            if base == "number" and s == 0 and p <= 18:
                return "BIGINT"
            if 0 < p <= 38 and 0 <= s <= p:
                return f"DECIMAL({p},{s})"
        return "DOUBLE"
    if re.match(r"(float|double|real|float4|float8|binary_float|binary_double|money|smallmoney)\b", base):
        return "DOUBLE"
    if base == "date":
        return "TIMESTAMP" if dialect == "oracle" else "DATE"
    if re.match(r"(datetime|datetime2|smalldatetime|timestamp)\b", base):
        if "with time zone" in t or "with local time zone" in t:
            return "TIMESTAMPTZ"
        return "TIMESTAMP"
    if base in ("timestamptz", "datetimeoffset"):
        return "TIMESTAMPTZ"
    if base in ("time", "timetz") and "zone" not in t:
        return "TIME"
    return "VARCHAR"


# ─── restore ──────────────────────────────────────────────────────────────

def _key(schema: str, name: str) -> tuple[str, str]:
    return (schema or "").lower(), name.lower()


class Restore:
    """One pass over the dump files into a DuckDB database."""

    DEFAULT_SCHEMAS = {"", "public", "dbo", "main"}

    def __init__(self, con, *, dialect: str | None, tables: list[str] | None, max_rows: int, typed: bool,
                 progress: Progress):
        self.con, self.forced, self.typed, self.max_rows = con, dialect, typed, max_rows
        self.only = {t.lower() for t in (tables or [])}
        self.progress = progress
        self.tables: dict[tuple[str, str], Table] = {}
        self.final: dict[tuple[str, str], str] = {}         # source table → DuckDB table name
        self.rows: dict[tuple[str, str], int] = {}
        self.buffers: dict[tuple[str, str], list] = {}
        self.cols_loaded: dict[tuple[str, str], list[str]] = {}   # physical column order of the DuckDB table
        self.skipped: dict[str, int] = {}
        self.warnings: list[str] = []
        self.dialects: set[str] = set()
        self.schemas: set[str] = set()
        self.statements = 0
        self.copy_target: tuple | None = None
        self.capped: dict[tuple[str, str], int] = {}
        self.ragged: dict[tuple[str, str], int] = {}
        self.dialect = dialect or "mysql"
        self._targets: dict[tuple, tuple | None] = {}     # INSERT header → (key, positions)
        self._parsers: dict[str, "ValueParser"] = {}
        self._n = 0

    # tables ────────────────────────────────────────────────────────────
    def wanted(self, key: tuple[str, str]) -> bool:
        if not self.only:
            return True
        return key[1] in self.only or f"{key[0]}.{key[1]}" in self.only

    def _physical(self, key: tuple[str, str]) -> str:
        """The DuckDB table holding a source table: an internal name while loading, the final one after rename()."""
        if key not in self.final:
            self._n += 1
            self.final[key] = f"__dump_t{self._n}"
        return self.final[key]

    def rename(self) -> None:
        """Final names, once every schema is known: the bare table name, schema__name only when the dump spans
        several non-default schemas (or two schemas share a table name)."""
        keys = list(self.cols_loaded)
        named = {k: self.tables[k] for k in keys}
        schemas = {(t.schema or "").lower() for t in named.values()} - self.DEFAULT_SCHEMAS
        name_count: dict[str, int] = {}
        for t in named.values():
            name_count[t.name.lower()] = name_count.get(t.name.lower(), 0) + 1
        taken: set[str] = set()
        for k in keys:
            t = named[k]
            own = (t.schema or "").lower()
            # default-schema tables (public / dbo / …) keep their bare names
            prefix = own not in self.DEFAULT_SCHEMAS and (len(schemas) > 1 or name_count[t.name.lower()] > 1)
            base = f"{t.schema}__{t.name}" if prefix else t.name
            base = re.sub(r"[^\w$]+", "_", base).strip("_") or "table"
            cand, n = base, 2
            while cand.lower() in taken:
                cand, n = f"{base}_{n}", n + 1
            taken.add(cand.lower())
            self.con.execute(f"ALTER TABLE {_qd(self.final[k])} RENAME TO {_qd(cand)}")
            self.final[k] = cand

    def parser(self, dialect: str) -> "ValueParser":
        if dialect not in self._parsers:
            self._parsers[dialect] = ValueParser(dialect)
        return self._parsers[dialect]

    def _create(self, key: tuple[str, str], columns: list[str]) -> None:
        name = self._physical(key)
        cols, seen = [], set()
        for c in columns:
            c2, n = c, 2
            while c2.lower() in seen:
                c2, n = f"{c}_{n}", n + 1
            seen.add(c2.lower())
            cols.append(c2)
        self._targets.clear()          # column positions of cached INSERT headers may have changed
        self.con.execute(f'DROP TABLE IF EXISTS {_qd(name)}')
        self.cols_loaded[key] = cols
        self.rows.setdefault(key, 0)
        if cols:                       # no columns known yet (data-only dump): made at the first row, col_1 … col_n
            self.con.execute(f'CREATE TABLE {_qd(name)} (' + ", ".join(f"{_qd(c)} VARCHAR" for c in cols) + ")")

    def on_create_table(self, t: Table) -> None:
        key = _key(t.schema, t.name)
        if not self.wanted(key):
            self.skipped["tables not selected"] = self.skipped.get("tables not selected", 0) + 1
            self.tables.setdefault(key, t)
            return
        self.schemas.add(t.schema or "")
        prev = self.tables.get(key)
        if prev and prev.declared and self.rows.get(key):
            self.warnings.append(f"{t.name}: created twice — the rows loaded before the second CREATE TABLE were dropped.")
            self.rows[key] = 0
            self.buffers.pop(key, None)
        elif prev and not prev.declared and self.cols_loaded.get(key):
            # data came before the DDL (two files, or a data-only file first): keep the rows, adopt the declaration
            self._flush(key)
            loaded = self.cols_loaded[key]
            positional = all(re.fullmatch(r"col_\d+", c) for c in loaded)
            if positional and len(loaded) <= len(t.columns):
                for old, c in zip(list(loaded), t.columns):
                    self.con.execute(f"ALTER TABLE {_qd(self._physical(key))} RENAME COLUMN {_qd(old)} TO {_qd(c.name)}")
                    loaded[loaded.index(old)] = c.name
                self._targets.clear()
            for c in t.columns:
                if c.name.lower() not in {x.lower() for x in loaded}:
                    self._add_column(key, c.name)
            self.tables[key] = t
            return
        self.tables[key] = t
        self._create(key, [c.name for c in t.columns])

    def _add_column(self, key, name: str) -> None:
        self._flush(key)
        self.con.execute(f"ALTER TABLE {_qd(self._physical(key))} ADD COLUMN {_qd(name)} VARCHAR")
        self.cols_loaded[key].append(name)

    def on_alter(self, key: tuple[str, str], patch: Table) -> None:
        t = self.tables.get(key)
        if not t:
            return
        if patch.primary_key:
            t.primary_key = patch.primary_key
        t.uniques += patch.uniques
        t.foreign_keys += patch.foreign_keys
        for pc in patch.columns:
            c = t.col(pc.name)
            if c is None:
                if pc.added and key in self.cols_loaded:
                    t.columns.append(pc)
                    self._add_column(key, pc.name)
                continue
            c.nullable = c.nullable and pc.nullable
            c.auto = c.auto or pc.auto
            if pc.declared:
                c.declared = pc.declared
            if pc.default is not None:
                c.default = pc.default

    def _target(self, name_text: str, cols_text: str | None) -> tuple[tuple[str, str], list[int] | None] | None:
        """(table key, positions of the INSERT's columns in the DuckDB table) — the table is made if unknown.
        Cached per header text: SSMS and Oracle scripts repeat the same INSERT header once per row."""
        ck = (name_text, cols_text)
        if ck in self._targets:
            return self._targets[ck]
        res = self._resolve_target(name_text, cols_text)
        self._targets[ck] = res
        return res

    def _resolve_target(self, name_text: str, cols_text: str | None):
        schema, name = split_name(name_text)
        key = _key(schema, name)
        if not self.wanted(key):
            return None
        if name.lower().startswith("sqlite_") or (schema or "").lower() in ("pg_catalog", "information_schema", "sys"):
            self._skip("rows of the database's own system tables")
            return None
        cols = ident_list(cols_text) if cols_text else None
        if key not in self.cols_loaded:
            t = self.tables.get(key)
            if not t:
                t = Table(schema, name)
                self.tables[key] = t
                self.schemas.add(schema or "")
            if cols:
                for c in cols:
                    if not t.col(c):
                        t.columns.append(Column(c))
            self._create(key, [c.name for c in t.columns] or (cols or []))
        if cols is None:
            return key, None
        loaded = self.cols_loaded[key]
        lower = [c.lower() for c in loaded]
        pos = []
        for c in cols:
            if c.lower() not in lower:
                self._add_column(key, c)
                if not self.tables[key].col(c):
                    self.tables[key].columns.append(Column(c))
                lower.append(c.lower())
            pos.append(lower.index(c.lower()))
        return key, pos

    # rows ──────────────────────────────────────────────────────────────
    def add_rows(self, key: tuple[str, str], positions: list[int] | None, rows) -> None:
        width = len(self.cols_loaded[key])
        buf = self.buffers.setdefault(key, [])
        for r in rows:
            if width == 0:             # a table seen only through INSERTs without a column list
                cols = [f"col_{k + 1}" for k in range(len(r))] or ["value"]
                self.con.execute(f'CREATE TABLE {_qd(self._physical(key))} (' + ", ".join(f"{_qd(c)} VARCHAR" for c in cols) + ")")
                self.cols_loaded[key] = cols
                t = self.tables[key]
                t.columns = [Column(c) for c in cols]
                width = len(cols)
            if self.max_rows and self.rows[key] >= self.max_rows:
                self.capped[key] = self.capped.get(key, 0) + 1
                continue
            if positions is not None:
                full = [None] * width
                for p, v in zip(positions, r):
                    full[p] = v
                if len(r) != len(positions):
                    self.ragged[key] = self.ragged.get(key, 0) + 1
                r = full
            elif len(r) != width:
                self.ragged[key] = self.ragged.get(key, 0) + 1
                if len(r) > width:          # more values than declared columns: keep them in extra columns
                    for k in range(width, len(r)):
                        self._add_column(key, f"col_{k + 1}")
                    width = len(self.cols_loaded[key])
                    buf = self.buffers.setdefault(key, [])
                r = (list(r) + [None] * width)[:width]
            buf.append(r)
            self.rows[key] += 1
            if len(buf) >= FLUSH_ROWS:
                self._flush(key)
                buf = self.buffers.setdefault(key, [])

    def _flush(self, key) -> None:
        buf = self.buffers.get(key)
        if not buf:
            return
        import pandas as pd

        cols = self.cols_loaded[key]
        df = pd.DataFrame(buf, columns=[f"c{i}" for i in range(len(cols))], dtype=object)
        self.con.register("_rows", df)
        try:
            self.con.execute(f"INSERT INTO {_qd(self._physical(key))} SELECT * FROM _rows")
        finally:
            self.con.unregister("_rows")
        self.buffers[key] = []

    def flush_all(self) -> None:
        for key in list(self.buffers):
            self._flush(key)

    # statements ────────────────────────────────────────────────────────
    def handle(self, st: Statement, dialect: str) -> None:
        if st.kind == "copy_data":
            if self.copy_target:
                key, pos = self.copy_target
                rows = ([copy_unescape(f) for f in line.split("\t")] for line in st.lines if line != "\\.")
                self.add_rows(key, pos, rows)
            if st.lines and st.lines[-1] == "\\.":
                self.copy_target = None
            return
        if st.kind == "meta":
            return
        self.statements += 1
        text = strip_comments(st.text)
        if not text:
            return
        head = text[:40].upper()
        if head.startswith(("INSERT", "REPLACE")):
            m = _INSERT.match(text)
            if not m:
                self._skip("INSERT statements not understood")
                return
            if m.group("kw").upper() != "VALUES" and m.group("kw").upper() != "VALUE":
                self._skip(f"INSERT … {m.group('kw').split()[0].upper()} statements (no literal values)")
                return
            tgt = self._target(m.group("name"), m.group("cols"))
            if tgt:
                self.add_rows(tgt[0], tgt[1], self.parser(dialect).rows(text, m.end()))
            return
        if head.startswith("COPY"):
            m = _COPY.match(text)
            if not m or re.search(r"\bCSV\b|FORMAT\s+csv|BINARY", m.group("opts") or "", re.I):
                self._skip("COPY statements in CSV / binary format")
                self.copy_target = None
                return
            tgt = self._target(m.group("name"), m.group("cols"))
            self.copy_target = tgt
            return
        if re.match(r"CREATE\s+(OR\s+REPLACE\s+)?((GLOBAL|LOCAL)\s+)?(TEMPORARY\s+|TEMP\s+|UNLOGGED\s+)?TABLE\b", head):
            t = parse_create_table(text)
            if t:
                self.on_create_table(t)
            else:
                self._skip("CREATE TABLE statements not understood")
            return
        if head.startswith("ALTER TABLE"):
            res = parse_alter_table(text)
            if res:
                self.on_alter(_key(*res[0]), res[1])
            return
        if re.match(r"CREATE\s+(OR\s+REPLACE\s+)?(ALGORITHM\s*=\s*\w+\s+)?(DEFINER\s*=\s*\S+\s+)?(SQL\s+SECURITY\s+\w+\s+)?"
                     r"(MATERIALIZED\s+)?VIEW\b", head):
            self._skip("views (no data of their own)")
            return
        if _ROUTINE.match(text):
            self._skip("procedures / functions / triggers")
            return
        if re.match(r"COMMENT\s+ON\s+(TABLE|COLUMN)", head):
            self._comment_on(text)
            return
        if re.match(r"EXEC(UTE)?\s+(sys\.)?sp_addextendedproperty", text, re.I):
            self._ms_description(text)
            return

    def _skip(self, what: str) -> None:
        self.skipped[what] = self.skipped.get(what, 0) + 1

    def _comment_on(self, text: str) -> None:
        m = re.match(rf"COMMENT\s+ON\s+(TABLE|COLUMN)\s+({NAME})\s+IS\s+('(?:[^']|'')*'|NULL)", text, re.I | re.S)
        if not m or m.group(3).upper() == "NULL":
            return
        parts = [unquote(p) for p in re.findall(_IDENT_PART, m.group(2))]
        value = _string_value(m.group(3))
        if m.group(1).upper() == "TABLE":
            t = self.tables.get(_key(parts[-2] if len(parts) > 1 else "", parts[-1]))
            if t:
                t.comment = value
        elif len(parts) >= 2:
            t = self.tables.get(_key(parts[-3] if len(parts) > 2 else "", parts[-2]))
            c = t.col(parts[-1]) if t else None
            if c:
                c.comment = value

    def _ms_description(self, text: str) -> None:
        """SSMS: EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'…', … 'TABLE',N'x', 'COLUMN',N'y'."""
        lits = [_string_value(x) for x in re.findall(r"N?'(?:[^']|'')*'", text)]
        if len(lits) < 2 or lits[0] != "MS_Description":
            return
        value = lits[1]
        low = [x.lower() for x in lits]
        try:
            tname = lits[low.index("table") + 1]
        except (ValueError, IndexError):
            return
        schema = lits[low.index("schema") + 1] if "schema" in low and low.index("schema") + 1 < len(lits) else ""
        t = self.tables.get(_key(schema, tname)) or self.tables.get(_key("", tname))
        if not t:
            return
        if "column" in low and low.index("column") + 1 < len(lits):
            c = t.col(lits[low.index("column") + 1])
            if c:
                c.comment = value
        else:
            t.comment = value

    # finishing ─────────────────────────────────────────────────────────
    def convert_types(self) -> None:
        """Text columns → their declared types, where every value converts."""
        keys = [k for k in self.cols_loaded]
        for n, key in enumerate(keys, start=1):
            t, name = self.tables[key], self._physical(key)
            self.progress(pct=88 + 8 * n / max(len(keys), 1), msg=f"Typing {name}", level="debug", phase="types")
            loaded = self.cols_loaded[key]
            plans = []
            for col in loaded:
                c = t.col(col)
                target = duck_type(c.declared, self.dialect) if c and c.declared else "VARCHAR"
                if target == "DOUBLE" and c and re.fullmatch(r"\s*(number|numeric|decimal)\s*", c.declared, re.I) \
                        and self.rows.get(key):
                    # NUMBER with no precision usually holds ids / counts: integers when every value is one
                    fractional = self.con.execute(
                        f"SELECT count(*) FROM {_qd(name)} WHERE {_qd(col)} IS NOT NULL AND trim({_qd(col)}) <> '' "
                        f"AND TRY_CAST(trim({_qd(col)}) AS BIGINT) IS NULL").fetchone()[0]
                    if not fractional:
                        target = "BIGINT"
                plans.append((col, target))
            check = [(col, tgt) for col, tgt in plans if tgt != "VARCHAR"]
            bad: dict[str, int] = {}
            if check and self.rows.get(key):
                exprs = ", ".join(f"count_if({_clean_expr(col, tgt)} IS NULL AND {_qd(col)} IS NOT NULL "
                                  f"AND trim({_qd(col)}) <> '' AND NOT {_zero_date(col, tgt)}) AS {_qd('b' + str(i))}"
                                  for i, (col, tgt) in enumerate(check))
                counts = self.con.execute(f"SELECT {exprs} FROM {_qd(name)}").fetchone()
                bad = {col: int(v) for (col, _), v in zip(check, counts) if v}
            selects = []
            for col, tgt in plans:
                if tgt == "VARCHAR" or col in bad:
                    if col in bad:
                        c = t.col(col)
                        sample = self.con.execute(
                            f"SELECT {_qd(col)} FROM {_qd(name)} WHERE {_clean_expr(col, tgt)} IS NULL AND {_qd(col)} IS NOT NULL "
                            f"AND trim({_qd(col)}) <> '' AND NOT {_zero_date(col, tgt)} LIMIT 3").fetchall()
                        self.warnings.append(f"{name}.{col}: declared {c.declared if c else tgt}, but {bad[col]:,} value(s) "
                                             f"aren't valid {tgt} (e.g. {', '.join(repr(s[0])[:40] for s in sample)}) — kept as text.")
                    selects.append(_qd(col))
                else:
                    selects.append(f"{_clean_expr(col, tgt)} AS {_qd(col)}")
            if any(tgt != "VARCHAR" and col not in bad for col, tgt in plans):
                tmp = f"{name}__typed"
                self.con.execute(f"CREATE TABLE {_qd(tmp)} AS SELECT {', '.join(selects)} FROM {_qd(name)}")
                self.con.execute(f"DROP TABLE {_qd(name)}")
                self.con.execute(f"ALTER TABLE {_qd(tmp)} RENAME TO {_qd(name)}")

    def meta(self) -> dict:
        out_tables = {}
        by_source = {k: self._physical(k) for k in self.cols_loaded}
        for key, phys in by_source.items():
            t = self.tables[key]
            fks = []
            for c in t.columns:
                ref = c.ref
                if ref:
                    t.foreign_keys.append({"columns": [c.name], "ref": split_name(ref[0]), "ref_columns": ref[1]})
            for fk in t.foreign_keys:
                rkey = _key(*fk["ref"])
                target = by_source.get(rkey) or next((v for k, v in by_source.items() if k[1] == rkey[1]), None)
                if not target:
                    continue
                rt = self.tables.get(rkey) or next((self.tables[k] for k in by_source if k[1] == rkey[1]), None)
                ref_cols = fk["ref_columns"] or (rt.primary_key if rt else [])
                if len(ref_cols) == len(fk["columns"]):
                    fks.append({"columns": fk["columns"], "table": target, "ref_columns": ref_cols})
            out_tables[phys] = {
                "source_schema": t.schema, "source_name": t.name, "declared": t.declared, "comment": t.comment,
                "rows": self.rows.get(key, 0), "primary_key": [c for c in t.primary_key if t.col(c)],
                "uniques": [u for u in t.uniques if u], "foreign_keys": fks,
                "capped_rows": self.capped.get(key, 0), "ragged_rows": self.ragged.get(key, 0),
                "columns": {c.name: {"declared": c.declared, "nullable": c.nullable, "default": c.default,
                                     "comment": c.comment, "pk": c.pk or c.name in t.primary_key, "unique": c.unique,
                                     "enum": c.enum, "auto": c.auto} for c in t.columns},
            }
            if self.ragged.get(key):
                self.warnings.append(f"{phys}: {self.ragged[key]:,} row(s) had a different number of values than columns.")
            if self.capped.get(key):
                self.warnings.append(f"{phys}: kept the first {self.max_rows:,} rows; {self.capped[key]:,} more were not loaded.")
        return {"tables": out_tables, "skipped": self.skipped, "warnings": self.warnings[:200],
                "statements": self.statements, "rows": sum(self.rows.get(k, 0) for k in by_source)}


def _qd(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def _zero_date(col: str, tgt: str) -> str:
    if tgt in ("DATE", "TIMESTAMP", "TIMESTAMPTZ"):
        return f"({_qd(col)} LIKE '0000-00-00%')"
    return "false"


def _clean_expr(col: str, tgt: str) -> str:
    """The typed value: blanks → NULL, MySQL zero dates → NULL, T/Z ISO forms accepted."""
    c = _qd(col)
    if tgt in ("DATE", "TIMESTAMP", "TIMESTAMPTZ"):
        return (f"TRY_CAST(CASE WHEN {c} LIKE '0000-00-00%' OR trim({c}) = '' THEN NULL ELSE {c} END AS {tgt})")
    if tgt == "BOOLEAN":
        return f"TRY_CAST(NULLIF(trim({c}), '') AS BOOLEAN)"
    return f"TRY_CAST(NULLIF(trim({c}), '') AS {tgt})"


def _pg_restore_to_sql(path: Path, workdir: Path, progress: Progress) -> Path:
    exe = shutil.which("pg_restore")
    if not exe:
        raise DumpError(f"{path.name} is a PostgreSQL custom-format dump. This server has no pg_restore to read it — "
                        f"re-export as plain SQL (pg_dump --format=plain, or pg_restore -f dump.sql {path.name}) and upload that.")
    out = workdir / (path.stem + ".plain.sql")
    progress(msg=f"{path.name}: PostgreSQL custom format — converting it to SQL with pg_restore…", phase="read")
    r = subprocess.run([exe, "--no-owner", "--no-privileges", "-f", str(out), str(path)], capture_output=True, text=True,
                       timeout=6 * 3600)
    if r.returncode != 0 and not out.exists():
        raise DumpError(f"pg_restore could not read {path.name}: {r.stderr.strip()[:300]}")
    return out


def restore(paths: list[Path], db_path: Path, *, dialect: str | None = None, tables: list[str] | None = None,
            max_rows: int = 0, typed: bool = True, progress: Progress = _noop, workdir: Path | None = None) -> dict:
    """Read every dump in `paths` into a fresh DuckDB file at `db_path`. Returns the restore report / metadata."""
    import duckdb

    t0 = time.time()
    workdir = Path(workdir or db_path.parent)
    plain: list[Path] = []
    for p in paths:
        p = Path(p)
        kind = binary_kind(_magic(p)) if p.exists() else None
        if kind == "pg_custom":
            plain.append(_pg_restore_to_sql(p, workdir, progress))
        elif kind == "mssql_bak":
            raise DumpError(f"{p.name} is a SQL Server backup (.bak), not a script. Restore it on a SQL Server and use the "
                            f"SQL Server connector, or script it out (SSMS: Tasks ▸ Generate Scripts ▸ Types of data: "
                            f"Schema and data) and upload the .sql.")
        elif kind == "sqlite_db":
            raise DumpError(f"{p.name} is a SQLite database file, not a dump — use the SQLite connector for it.")
        elif kind == "oracle_dmp":
            raise DumpError(f"{p.name} is an Oracle binary export (.dmp), not a script. Export it as SQL instead "
                            f"(SQL Developer: Tools ▸ Database Export ▸ format: insert) and upload that.")
        else:
            plain.append(p)
    files = dump_files(plain)
    if not files:
        raise DumpError("Upload at least one SQL dump.")
    total = sum(f.size for f in files) or 1
    tmp = db_path.with_suffix(".restoring.duckdb")
    for f in (tmp, Path(str(tmp) + ".wal")):
        if f.exists():
            f.unlink()
    con = duckdb.connect(str(tmp))
    done_bytes = 0
    used: list[str] = []
    try:
        r = None
        for f in files:
            binary, consumed = f.open_binary()
            try:
                stream = text_stream(binary)
                head = stream.read(4 * 1024 * 1024)
                d = dialect if dialect in DIALECTS else detect_dialect(head)
                used.append(d)
                if r is None:
                    r = Restore(con, dialect=dialect, tables=tables, max_rows=max_rows, typed=typed, progress=progress)
                r.dialects.add(d)
                r.dialect = d
                progress(msg=f"{f.name}: {DIALECT_LABELS[d]} dump{'' if dialect else ' (detected)'} — "
                             f"{f.size / 1e6:,.1f} MB", phase="read")
                stream = _Prepend(head, stream)
                last = time.time()
                for st in Splitter(stream, d):
                    r.handle(st, d)
                    if time.time() - last > 1.0:
                        last = time.time()
                        frac = min((done_bytes + consumed()) / total, 1.0)
                        progress(pct=2 + 84 * frac, msg=f"{f.name}: {consumed() / 1e6:,.0f} of {f.size / 1e6:,.0f} MB · "
                                 f"{sum(r.rows.values()):,} rows in {len(r.cols_loaded)} tables", level="debug", phase="read")
            except UnicodeError as e:
                raise DumpError(f"{f.name}: can't decode the text ({e}).") from None
            finally:
                binary.close()
            done_bytes += f.size
            r.flush_all()
        if not r or not r.cols_loaded:
            what = ", ".join(f"{k} ({v})" for k, v in (r.skipped.items() if r else []))
            raise DumpError("No tables found in the dump — it holds no CREATE TABLE / INSERT / COPY statements"
                            + (f" that were read (skipped: {what})." if what else ".")
                            + (" None of the listed tables is in it." if tables else ""))
        r.dialect = used[0]
        r.flush_all()
        r.rename()
        if typed:
            progress(pct=88, msg="Converting columns to their declared types…", phase="types")
            r.convert_types()
        meta = r.meta()
        con.execute("CHECKPOINT")
    finally:
        con.close()
    os.replace(tmp, db_path)
    meta.update({"dialect": used[0], "dialects": sorted(set(used)), "files": [f.name for f in files],
                 "bytes": sum(f.size for f in files), "seconds": round(time.time() - t0, 1), "version": VERSION})
    return meta


class _Prepend(io.TextIOBase):
    """A text stream with an already-read head put back in front."""

    def __init__(self, head: str, rest: io.TextIOBase):
        self.head, self.rest = head, rest

    def readable(self):
        return True

    def read(self, n: int = -1) -> str:
        if self.head:
            out, self.head = (self.head, "") if n < 0 or n >= len(self.head) else (self.head[:n], self.head[n:])
            return out
        return self.rest.read(n)
