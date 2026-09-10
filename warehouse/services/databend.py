"""Databend HTTP query client for the Afya DataHub warehouse module.

Talks to Databend's HTTP handler REST API (POST /v1/query/, see
docker-compose.databend.yaml) using basic auth. All destructive SQL
keywords are blocked before execution, mirroring the read-only guarantee
of the Snowflake console (see warehouse/services/snowflake.py).
"""

import logging
import re

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Keywords that should never appear in a read-only query interface.
BLOCKED_KEYWORDS = frozenset({
    'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE',
    'INSERT', 'UPDATE', 'GRANT', 'REVOKE',
})

_BLOCKED_RE = re.compile(
    r'\b(' + '|'.join(BLOCKED_KEYWORDS) + r')\b',
    re.IGNORECASE,
)


class DatabendQueryError(Exception):
    """Raised when a Databend query fails or is rejected."""


def _validate_sql(sql: str) -> None:
    """Raise DatabendQueryError if ``sql`` contains any blocked keyword."""
    match = _BLOCKED_RE.search(sql)
    if match:
        raise DatabendQueryError(
            f"The keyword '{match.group(0).upper()}' is not permitted. "
            "Only read-only SELECT queries are allowed."
        )


class DatabendClient:
    """Client for executing read-only queries against Databend's HTTP handler.

    Connection parameters are read from Django settings (which in turn read
    them from environment variables): DATABEND_HTTP_URL, DATABEND_USER,
    DATABEND_PASSWORD.
    """

    def query(self, sql: str, max_rows: int = 10_000) -> tuple[list[str], list[list]]:
        """Run ``sql`` and return (column_names, rows), following pagination
        (Databend's ``next_uri``) until ``max_rows`` is reached or exhausted.
        """
        _validate_sql(sql)

        auth = (settings.DATABEND_USER, settings.DATABEND_PASSWORD)
        try:
            resp = requests.post(
                f"{settings.DATABEND_HTTP_URL}/v1/query/",
                json={"sql": sql},
                auth=auth,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("Databend query failed: %s", exc)
            raise DatabendQueryError(str(exc)) from exc

        if data.get("error"):
            raise DatabendQueryError(data["error"].get("message", str(data["error"])))

        columns = [c["name"] for c in data.get("schema", [])]
        rows = list(data.get("data", []))

        next_uri = data.get("next_uri")
        while next_uri and len(rows) < max_rows:
            try:
                resp = requests.get(
                    f"{settings.DATABEND_HTTP_URL}{next_uri}",
                    auth=auth,
                    timeout=60,
                )
                resp.raise_for_status()
                page = resp.json()
            except requests.RequestException as exc:
                logger.error("Databend pagination failed: %s", exc)
                break
            if page.get("error"):
                break
            rows.extend(page.get("data", []))
            next_uri = page.get("next_uri")

        return columns, rows[:max_rows]

    def list_tables(self) -> list[dict]:
        """Return [{"schema_name": ..., "table_name": ...}, ...] for every
        user table (system/information_schema tables excluded).
        """
        columns, rows = self.query(
            "SELECT database, name FROM system.tables "
            "WHERE database NOT IN ('system', 'information_schema') "
            "ORDER BY database, name",
            max_rows=1000,
        )
        return [
            {"schema_name": row[0], "table_name": row[1]}
            for row in rows
        ]
