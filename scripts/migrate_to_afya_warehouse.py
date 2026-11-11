#!/usr/bin/env python3
"""Codemod: route every Snowflake connection through afya_backends.warehouse.

Rewrites, in place:
  snowflake.connector.connect(...)            -> afya_warehouse.connect(...)
  <alias>.connect(...)  (import snowflake.connector as <alias>
                         / from snowflake import connector [as <alias>])
  from snowflake.connector.pandas_tools import write_pandas
                                              -> from afya_backends.warehouse import write_pandas
  snowflake.connector.pandas_tools.write_pandas(...) -> afya_warehouse.write_pandas(...)
  from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
                                              -> WarehouseHook as SnowflakeHook

and adds ``from afya_backends import warehouse as afya_warehouse`` right
after the snowflake import that the connect call relies on (same
indentation, so function-local lazy imports stay lazy).

Idempotent.  Usage:  python scripts/migrate_to_afya_warehouse.py [--apply] [paths...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {"env", "node_modules", "afya_backends", ".git", "static", "logs", "redash", "__pycache__"}
IMPORT_LINE = "from afya_backends import warehouse as afya_warehouse"

RE_IMPORT_MOD = re.compile(r"^(?P<ind>[ \t]*)import snowflake\.connector(?P<alias>\s+as\s+\w+)?\s*(#.*)?$", re.M)
RE_FROM_CONNECTOR = re.compile(r"^(?P<ind>[ \t]*)from snowflake import connector(?:\s+as\s+(?P<alias>\w+))?\s*(#.*)?$", re.M)
RE_WRITE_PANDAS_IMPORT = re.compile(r"^(?P<ind>[ \t]*)from snowflake\.connector\.pandas_tools import (?P<names>[^\n#]+)", re.M)
RE_HOOK_IMPORT = re.compile(r"^(?P<ind>[ \t]*)from airflow\.providers\.snowflake\.hooks\.snowflake import SnowflakeHook\b(?P<rest>[^\n]*)$", re.M)


def migrate(text: str) -> tuple[str, list[str]]:
    changes = []
    out = text

    # pandas_tools import
    def wp(m):
        names = [n.strip() for n in m.group("names").split(",")]
        if names == ["write_pandas"]:
            changes.append("write_pandas import")
            return f"{m.group('ind')}from afya_backends.warehouse import write_pandas"
        return m.group(0)
    out = RE_WRITE_PANDAS_IMPORT.sub(wp, out)
    if "snowflake.connector.pandas_tools.write_pandas(" in out:
        out = out.replace("snowflake.connector.pandas_tools.write_pandas(", "afya_warehouse.write_pandas(")
        changes.append("write_pandas call")

    # Airflow hook
    def hook(m):
        changes.append("SnowflakeHook import")
        return (f"{m.group('ind')}from afya_backends.warehouse.airflow_hook import WarehouseHook as SnowflakeHook"
                f"{m.group('rest')}")
    out = RE_HOOK_IMPORT.sub(hook, out)

    # connect calls
    aliases = {"snowflake.connector"}
    for m in RE_IMPORT_MOD.finditer(out):
        if m.group("alias"):
            aliases.add(m.group("alias").split()[-1])
    for m in RE_FROM_CONNECTOR.finditer(out):
        aliases.add(m.group("alias") or "connector")
    n_calls = 0
    for alias in sorted(aliases, key=len, reverse=True):
        pat = re.compile(r"(?<![\w.])" + re.escape(alias) + r"\.connect\(")
        out, n = pat.subn("afya_warehouse.connect(", out)
        n_calls += n
    if n_calls:
        changes.append(f"{n_calls} connect call(s)")

    needs_import = ("afya_warehouse." in out) and IMPORT_LINE not in out
    if needs_import:
        # Put the import after each snowflake import (keeps lazy imports lazy).
        anchors = list(RE_IMPORT_MOD.finditer(out)) + list(RE_FROM_CONNECTOR.finditer(out))
        if not anchors:
            anchors = [m for m in re.finditer(r"^(?P<ind>[ \t]*)(?:import|from) snowflake[^\n]*$", out, re.M)]
        if anchors:
            for m in sorted(anchors, key=lambda m: m.end(), reverse=True):
                ind = m.group("ind")
                out = out[: m.end()] + f"\n{ind}{IMPORT_LINE}" + out[m.end():]
        else:
            # No snowflake import (connect via write_pandas-only modules): add after the last top import.
            lines = out.split("\n")
            idx = max((i for i, ln in enumerate(lines[:200]) if re.match(r"^(import|from) \w", ln)), default=0)
            lines.insert(idx + 1, IMPORT_LINE)
            out = "\n".join(lines)
        changes.append("import added")
    return out, changes


def iter_files(paths):
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            for f in p.rglob("*.py"):
                if f.is_symlink() or any(part in SKIP_PARTS for part in f.relative_to(ROOT).parts):
                    continue
                if "/tests/" in str(f) or f.name.startswith("test_") or f.name == "tests.py":
                    continue
                # Deliberately Snowflake-only: the switch's own health check,
                # and this script.
                if f.relative_to(ROOT).as_posix() in ("core/backends.py", "scripts/migrate_to_afya_warehouse.py"):
                    continue
                yield f


def main(argv):
    apply = "--apply" in argv
    paths = [a for a in argv if not a.startswith("--")] or [ROOT]
    total = 0
    for f in sorted(set(iter_files(paths))):
        raw = f.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        if "snowflake" not in text and "write_pandas" not in text:
            continue
        new, changes = migrate(text)
        if new != text:
            total += 1
            print(f"{f.relative_to(ROOT)}: {', '.join(changes)}")
            if apply:
                # Keep the file's own line endings (several are CRLF).
                f.write_bytes((new.replace("\n", "\r\n") if crlf else new).encode("utf-8"))
    print(f"{'changed' if apply else 'would change'} {total} file(s)")


if __name__ == "__main__":
    main(sys.argv[1:])
