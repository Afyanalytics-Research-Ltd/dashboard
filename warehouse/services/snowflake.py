"""Snowflake query client for the Afya DataHub warehouse module.

Connections are established per-call (Snowflake connector handles its own
connection pooling internally). All destructive SQL keywords are blocked
before execution to prevent accidental data loss from the UI.
"""

import logging
import re
import os
import time
from typing import Optional

import pandas as pd
import snowflake.connector
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


class SnowflakeQueryError(Exception):
    """Raised when a Snowflake query fails or is rejected."""


def _validate_sql(sql: str) -> None:
    """Raise SnowflakeQueryError if ``sql`` contains any blocked keyword."""
    match = _BLOCKED_RE.search(sql)
    if match:
        raise SnowflakeQueryError(
            f"The keyword '{match.group(0).upper()}' is not permitted. "
            "Only read-only SELECT queries are allowed."
        )


class SnowflakeClient:
    """Client for executing read-only queries against the configured Snowflake account.

    Connection parameters are read from Django settings (which in turn read
    them from environment variables):
        SNOWFLAKE_USER, SNOWFLAKE_ACCOUNT, SNOWFLAKE_WAREHOUSE,
        SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, SNOWFLAKE_PASSWORD (optional),
        SNOWFLAKE_PRIVATE_KEY_PATH (optional, preferred over password).
    """

    def _connect(self) -> snowflake.connector.SnowflakeConnection:
        """Open a new Snowflake connection.

        Tries private-key auth first (``SNOWFLAKE_PRIVATE_KEY_PATH``); falls
        back to password auth if no key file is configured.
        """
        common = dict(
            user=os.getenv('SNOWFLAKE_USER', '').strip(),
            account=os.getenv('SNOWFLAKE_ACCOUNT', '').strip(),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', '').strip(),
            database=os.getenv('SNOWFLAKE_DATABASE', '').strip(),
            schema=os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC').strip(),
        )
        private_key_path = os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH', '').strip()
        if private_key_path:
            common['private_key_file'] = private_key_path
        else:
            common['password'] = os.getenv('SNOWFLAKE_PASSWORD', '').strip()

        return snowflake.connector.connect(**common)

    def query(self, sql: str, max_rows: int = 10_000) -> pd.DataFrame:
        """Execute a read-only SQL query and return results as a DataFrame.

        Args:
            sql: A SELECT statement (no destructive keywords allowed).
            max_rows: Soft limit — adds ``LIMIT max_rows`` only if the query
                does not already contain a LIMIT clause.

        Returns:
            A :class:`pandas.DataFrame` with query results.

        Raises:
            SnowflakeQueryError: If the SQL contains blocked keywords or the
                query execution fails.
        """
        _validate_sql(sql)

        # Append a safety LIMIT if the query doesn't already have one.
        normalized = sql.rstrip().rstrip(';')
        if not re.search(r'\bLIMIT\b', normalized, re.IGNORECASE):
            normalized = f"{normalized} LIMIT {max_rows}"

        logger.debug("Snowflake query: %.200s", normalized)
        t0 = time.monotonic()
        conn = self._connect()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(normalized)
                df: pd.DataFrame = cursor.fetch_pandas_all()
            finally:
                cursor.close()
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.error("Snowflake query failed after %dms: %s", elapsed_ms, exc)
            raise SnowflakeQueryError(str(exc)) from exc
        finally:
            conn.close()

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Snowflake query returned %d rows in %dms",
            len(df), elapsed_ms,
        )
        return df

    def get_tables(self) -> pd.DataFrame:
        """Return a DataFrame listing all tables in the configured schema.

        Columns: SCHEMA_NAME, TABLE_NAME, TABLE_TYPE, ROW_COUNT, BYTES,
        LAST_ALTERED.

        Raises:
            SnowflakeQueryError: On connection or query failure.
        """
        sql = """
            SELECT
                TABLE_SCHEMA  AS SCHEMA_NAME,
                TABLE_NAME,
                TABLE_TYPE,
                ROW_COUNT,
                BYTES,
                LAST_ALTERED
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA != 'INFORMATION_SCHEMA'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        try:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                try:
                    cursor.execute(sql)
                    return cursor.fetch_pandas_all()
                finally:
                    cursor.close()
            finally:
                conn.close()
        except SnowflakeQueryError:
            raise
        except Exception as exc:
            raise SnowflakeQueryError(str(exc)) from exc

    def get_table_sample(
        self, schema: str, table: str, rows: int = 10
    ) -> pd.DataFrame:
        """Return a sample of ``rows`` rows from ``schema.table``.

        Args:
            schema: Schema name (will be uppercased and quoted).
            table: Table name (will be uppercased and quoted).
            rows: Number of sample rows to return (max 1000).

        Raises:
            SnowflakeQueryError: On connection or query failure.
        """
        rows = min(rows, 1000)
        # Identifiers are uppercased and double-quoted to handle mixed case.
        safe_schema = schema.upper().replace('"', '')
        safe_table = table.upper().replace('"', '')
        sql = f'SELECT * FROM "{safe_schema}"."{safe_table}" LIMIT {rows}'
        return self.query(sql, max_rows=rows)
