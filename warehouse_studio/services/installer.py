"""Render and apply ``snowflake/warehouse_studio/*.sql`` (the cost & performance toolkit).

The SQL files are templates with ``{{DB}}``, ``{{SCHEMA}}``, ``{{APP_ROLE}}``
and ``{{TASK_WAREHOUSE}}`` placeholders.  Only plain identifiers may be
substituted, so a template can never be turned into an injection vector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from ..conf import IDENT_RE
from .sql_guard import split_statements

SQL_DIR = Path(getattr(settings, 'BASE_DIR', Path(__file__).resolve().parents[2])) / 'snowflake' / 'warehouse_studio'
PLACEHOLDER_RE = re.compile(r'\{\{([A-Z_]+)\}\}')
REQUIRED = ('DB', 'SCHEMA', 'APP_ROLE', 'TASK_WAREHOUSE')

# Objects install.sql creates, used by verify() and the Studio's status panel.
EXPECTED_PROCEDURES = (
    'SP_DETECT_LONG_COMPILATION', 'SP_DETECT_LONG_EXECUTION', 'SP_DETECT_POOR_PRUNING',
    'SP_DETECT_SPILL_AND_OOM', 'SP_DETECT_QUEUEING', 'SP_DETECT_BLOCKED_TRANSACTIONS',
    'SP_DETECT_STORAGE_OVERHEAD', 'SP_DETECT_WAREHOUSE_WASTE', 'SP_DETECT_WAREHOUSE_SETTINGS',
    'SP_DETECT_SELECT_STAR', 'SP_DETECT_REPEAT_OFFENDERS', 'SP_DETECT_JOIN_EXPLOSION',
    'SP_RUN_ALL_DETECTORS', 'SP_RUN_FULL_OPTIMIZATION_ANALYSIS', 'SP_FIND_HIGH_TIME_TRAVEL_TABLES',
    'SP_MONITOR_CLUSTERING_COSTS', 'SP_CLUSTERING_REPORT', 'SP_ESTIMATE_SOS_COST',
    'SP_WS_WAREHOUSES', 'SP_WS_RESOURCE_MONITORS', 'SP_WS_QUERY_PROFILE', 'SP_WS_DETECTOR_SCHEDULE',
)
EXPECTED_RELATIONS = (
    'TUNING_FINDINGS', 'TUNING_SCAN_LOG', 'WS_DETECTOR_RUNS', 'V_TUNING_FINDINGS', 'V_TUNING_PRIORITIES',
    'V_CLUSTERING_CANDIDATES', 'WS_QUERY_HISTORY', 'WS_QUERY_ATTRIBUTION', 'WS_WAREHOUSE_METERING',
    'WS_METERING_DAILY', 'WS_WAREHOUSE_LOAD', 'WS_WAREHOUSE_EVENTS', 'WS_STORAGE_DAILY',
    'WS_DATABASE_STORAGE', 'WS_TABLE_STORAGE', 'WS_TASK_HISTORY', 'WS_CLUSTERING_HISTORY',
)


class InstallError(Exception):
    pass


def render(template: str, values: dict[str, str]) -> str:
    """Substitute ``{{NAME}}`` placeholders with validated identifiers."""
    clean = {}
    for key in REQUIRED:
        value = str(values.get(key, '')).strip().upper()
        if not IDENT_RE.match(value):
            raise InstallError(f'{key} must be a plain Snowflake identifier, got {values.get(key)!r}')
        clean[key] = value

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in clean:
            raise InstallError(f'Unknown placeholder {{{{{name}}}}} in template')
        return clean[name]

    return PLACEHOLDER_RE.sub(sub, template)


@dataclass
class Step:
    index: int
    sql: str

    @property
    def summary(self) -> str:
        first = ' '.join(self.sql.split())
        m = re.match(r'(CREATE(?: OR REPLACE)?(?: SECURE)? (?:TABLE|VIEW|PROCEDURE|TASK)(?: IF NOT EXISTS)? [\w.$]+'
                     r'|GRANT [\w ]+? ON [\w ]+? [\w.$]+(?:\([^)]*\))?|USE ROLE \w+|DROP [\w ]+? (?:IF EXISTS )?[\w.$]+(?:\([^)]*\))?)', first)
        return m.group(1) if m else first[:90]


def load_steps(name: str, values: dict[str, str]) -> list[Step]:
    path = SQL_DIR / name
    if not path.exists():
        raise InstallError(f'{path} not found')
    sql = render(path.read_text(encoding='utf-8'), values)
    return [Step(i, st.text) for i, st in enumerate(split_statements(sql), 1)]


def apply(conn, steps: list[Step], *, on_step=None) -> list[dict]:
    """Execute every step on ``conn``; stop at the first failure (and report it)."""
    results = []
    cur = conn.cursor()
    try:
        for step in steps:
            try:
                cur.execute(step.sql)
                row = cur.fetchone() if cur.description else None
                outcome = {'step': step.index, 'summary': step.summary, 'ok': True,
                           'message': str(row[0]) if row else 'ok'}
            except Exception as exc:  # noqa: BLE001 — surface the Snowflake message
                outcome = {'step': step.index, 'summary': step.summary, 'ok': False,
                           'message': getattr(exc, 'msg', None) or str(exc)}
                results.append(outcome)
                if on_step:
                    on_step(outcome)
                raise InstallError(f'Step {step.index} ({step.summary}) failed: {outcome["message"]}') from exc
            results.append(outcome)
            if on_step:
                on_step(outcome)
    finally:
        cur.close()
    return results


def verify(conn, database: str, schema: str) -> dict:
    """Report which toolkit objects exist in ``database.schema`` (visible to the session's role)."""
    cur = conn.cursor()
    try:
        cur.execute(
            f'SELECT procedure_name FROM "{database}".INFORMATION_SCHEMA.PROCEDURES WHERE procedure_schema = %s',
            (schema,))
        procs = {r[0] for r in cur.fetchall()}
        cur.execute(
            f'SELECT table_name FROM "{database}".INFORMATION_SCHEMA.TABLES WHERE table_schema = %s', (schema,))
        rels = {r[0] for r in cur.fetchall()}
    finally:
        cur.close()
    return {
        'procedures': {p: p in procs for p in EXPECTED_PROCEDURES},
        'relations': {r: r in rels for r in EXPECTED_RELATIONS},
        'installed': all(p in procs for p in EXPECTED_PROCEDURES) and all(r in rels for r in EXPECTED_RELATIONS),
        'missing': sorted([p for p in EXPECTED_PROCEDURES if p not in procs] + [r for r in EXPECTED_RELATIONS if r not in rels]),
    }
