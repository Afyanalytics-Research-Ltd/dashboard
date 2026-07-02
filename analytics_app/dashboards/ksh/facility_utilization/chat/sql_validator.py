"""
sql_validator.py — SQL safety and semantic validation for the KSH AI Chat Layer.

Called by sql_executor.py before executing any generated SQL.
Returns ValidationResult dict: {valid, errors, warnings, tables_used, columns_used}.

Hard errors → block execution, trigger one retry.
Warnings    → logged in sql_requests.jsonl, execution proceeds.

Checks:
  1. Parseable SQL
  2. SELECT-only — no DML/DDL keywords
  3. Table allowlist — only tables defined in schema_catalog.json
  4. No CROSS JOIN
  5. No recursive CTEs (WITH RECURSIVE)
  6. Max 3 tables joined
  7. Max subquery nesting depth: 2
  8. Date filter required on large time-series tables (warning if missing)
  9. LIMIT recommended on large tables (warning if absent)
 10. Column catalog check on qualified column references (warning if unknown)
"""

import json
import re
from pathlib import Path

import sqlglot
import sqlglot.expressions as exp

_CHAT = Path(__file__).resolve().parent
_CATALOG_PATH = _CHAT / "schema_catalog.json"

# Tables where an unbounded scan is expensive — warn if no date filter present
_LARGE_TABLES = frozenset({"bed_occupancy", "readmissions", "doctor_performance"})

# DML/DDL keywords that must not appear anywhere in the SQL text
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|MERGE|CALL|TRUNCATE|ALTER|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# CROSS JOIN patterns
_CROSS_JOIN = re.compile(r"\bCROSS\s+JOIN\b", re.IGNORECASE)

# Recursive CTE
_RECURSIVE_CTE = re.compile(r"\bWITH\s+RECURSIVE\b", re.IGNORECASE)


def _load_catalog() -> dict:
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))


def _build_alias_to_columns(catalog: dict) -> dict:
    """alias → frozenset of column names (lowercase)."""
    result = {}
    for table_entry in catalog.get("tables", []):
        alias = table_entry["alias"]
        cols = frozenset(c.lower() for c in table_entry.get("columns", {}).keys())
        result[alias] = cols
    return result


def _extract_tables(statement) -> tuple[list[str], dict[str, str]]:
    """
    Return (table_names, alias_to_name_map).
    table_names: list of table name stems (lowercased, last segment of dotted path).
    alias_to_name_map: SQL alias → table name stem (for column resolution).
    """
    tables = []
    alias_map: dict[str, str] = {}
    for node in statement.walk():
        if isinstance(node, exp.Table):
            # node.name is the last identifier (e.g. "rpt_bed_occupancy" or "bed_occupancy")
            name = node.name.lower() if node.name else None
            if not name:
                continue
            # Strip common rpt_ prefix if present so catalog lookup works
            stem = name.removeprefix("rpt_")
            tables.append(stem)
            sql_alias = node.alias.lower() if node.alias else stem
            alias_map[sql_alias] = stem
    return list(dict.fromkeys(tables)), alias_map


def _max_nesting_depth(statement, depth: int = 0) -> int:
    """Count the deepest subquery nesting level."""
    max_d = depth
    for node in statement.args.values():
        if node is None:
            continue
        nodes = node if isinstance(node, list) else [node]
        for child in nodes:
            if isinstance(child, (exp.Subquery, exp.CTE)):
                d = _max_nesting_depth(child, depth + 1)
                if d > max_d:
                    max_d = d
    return max_d


def _has_date_filter(statement, alias: str) -> bool:
    """
    Rough check: returns True if a WHERE clause references a date-like column
    on the given table alias. Heuristic — avoids false positives on computed cols.
    """
    sql_text = statement.sql(dialect="snowflake").upper()
    date_keywords = ["_MONTH", "_DATE", "DATEADD", "DATE_TRUNC", "CURRENT_DATE", "DATEDIFF"]
    return any(kw in sql_text for kw in date_keywords)


def _has_limit(statement) -> bool:
    for node in statement.walk():
        if isinstance(node, exp.Limit):
            return True
    return False


def validate(sql: str) -> dict:
    """
    Validate a generated SQL string.

    Returns:
        {
            "valid":       bool,
            "errors":      list[str],   # hard failures — block execution
            "warnings":    list[str],   # soft issues — logged, execution proceeds
            "tables_used": list[str],   # alias names from catalog
            "columns_used": list[str],  # column names found in the query
        }
    """
    errors: list[str] = []
    warnings: list[str] = []
    tables_used: list[str] = []
    columns_used: list[str] = []

    # ── 1. Forbidden keyword check (text level, before parsing) ──────────────
    match = _FORBIDDEN_KEYWORDS.search(sql)
    if match:
        errors.append(f"forbidden_keyword:{match.group(0).upper()}")
        return {"valid": False, "errors": errors, "warnings": warnings,
                "tables_used": tables_used, "columns_used": columns_used}

    # ── 2. Cross join ─────────────────────────────────────────────────────────
    if _CROSS_JOIN.search(sql):
        errors.append("cross_join_not_allowed")

    # ── 3. Recursive CTE ──────────────────────────────────────────────────────
    if _RECURSIVE_CTE.search(sql):
        errors.append("recursive_cte_not_allowed")

    # ── 4. Parse ─────────────────────────────────────────────────────────────
    try:
        statements = sqlglot.parse(sql, dialect="snowflake")
    except Exception as exc:
        errors.append(f"parse_error:{exc}")
        return {"valid": False, "errors": errors, "warnings": warnings,
                "tables_used": tables_used, "columns_used": columns_used}

    if not statements:
        errors.append("empty_statement")
        return {"valid": False, "errors": errors, "warnings": warnings,
                "tables_used": tables_used, "columns_used": columns_used}

    if len(statements) > 1:
        errors.append("multiple_statements_not_allowed")

    statement = statements[0]

    # ── 5. SELECT-only ───────────────────────────────────────────────────────
    if not isinstance(statement, exp.Select):
        errors.append(f"non_select_statement:{type(statement).__name__}")
        return {"valid": False, "errors": errors, "warnings": warnings,
                "tables_used": tables_used, "columns_used": columns_used}

    # ── 6. Load catalog ───────────────────────────────────────────────────────
    try:
        catalog = _load_catalog()
    except Exception as exc:
        errors.append(f"catalog_load_error:{exc}")
        return {"valid": False, "errors": errors, "warnings": warnings,
                "tables_used": tables_used, "columns_used": columns_used}

    alias_to_cols = _build_alias_to_columns(catalog)
    allowed_aliases = frozenset(alias_to_cols.keys())

    # ── 7. Table allowlist ────────────────────────────────────────────────────
    tables_found, sql_alias_map = _extract_tables(statement)
    tables_used = [t for t in tables_found if t in allowed_aliases]
    unknown_tables = [t for t in tables_found if t not in allowed_aliases]
    if unknown_tables:
        errors.append(f"unknown_tables:{unknown_tables}")

    # ── 8. Max joined tables ──────────────────────────────────────────────────
    if len(tables_found) > 3:
        errors.append(f"too_many_tables:{len(tables_found)}_max_3")

    # ── 9. Nesting depth ──────────────────────────────────────────────────────
    depth = _max_nesting_depth(statement)
    if depth > 2:
        errors.append(f"nesting_too_deep:{depth}_max_2")

    # ── 10. Date filter on large tables (warning) ─────────────────────────────
    large_used = [t for t in tables_used if t in _LARGE_TABLES]
    if large_used and not _has_date_filter(statement, large_used[0]):
        warnings.append(f"missing_date_filter_on:{large_used}")

    # ── 11. LIMIT on large tables (warning) ───────────────────────────────────
    if large_used and not _has_limit(statement):
        warnings.append(f"no_limit_on_large_table:{large_used}")

    # ── 12. Column catalog check (warning only) ───────────────────────────────
    for node in statement.walk():
        if isinstance(node, exp.Column):
            col_name = node.name.lower() if node.name else None
            if not col_name:
                continue
            columns_used.append(col_name)
            # Resolve SQL alias (e.g. "a") to catalog alias (e.g. "bed_occupancy")
            sql_table_ref = node.table.lower() if node.table else None
            catalog_alias = sql_alias_map.get(sql_table_ref, sql_table_ref) if sql_table_ref else None
            if catalog_alias and catalog_alias in alias_to_cols:
                known_cols = alias_to_cols[catalog_alias]
                if col_name not in known_cols:
                    warnings.append(f"unknown_column:{catalog_alias}.{col_name}")

    columns_used = list(dict.fromkeys(columns_used))  # deduplicate

    valid = len(errors) == 0
    return {
        "valid":        valid,
        "errors":       errors,
        "warnings":     warnings,
        "tables_used":  tables_used,
        "columns_used": columns_used,
    }
