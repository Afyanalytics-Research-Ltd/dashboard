import logging
import os
import sys
from pathlib import Path

# Project root (…/analytics_app/dashboards/snowflake_service/ -> 3 up). Streamlit
# dashboards import this module with only their own folder on sys.path, so
# `afya_backends` (a top-level package in the project root) is not importable
# unless the root is added here -- otherwise every Streamlit dashboard using
# SnowflakeClient fails with "ModuleNotFoundError: No module named 'afya_backends'".
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import snowflake.connector  # noqa: E402
from afya_backends import warehouse as afya_warehouse  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# 390114: "Authentication token has expired. The user must authenticate
# again." Raised as ProgrammingError once the connector's own internal
# session-renewal (_renew_session, using the master token) has itself
# failed -- i.e. the connection has been idle long enough that BOTH the
# session token and the master token that would normally refresh it are
# gone. A single reconnect (a fresh login, not a renewal) clears it.
_TOKEN_EXPIRED_ERRNO = 390114


class SnowflakeClient:
    """A dashboard-facing client instantiated ONCE per process (many
    dashboards bind `SnowflakeClient().query` at module import time and
    keep it for the life of the container -- see db.py-style callers
    across analytics_app/dashboards/*). Long-lived Streamlit containers
    routinely sit idle for days between visits, which outlives Snowflake's
    session/master token lifetimes; query() reconnects once and retries
    rather than making every dashboard caller handle that itself."""

    def __init__(self, schema_=None):
        self._schema = schema_
        self.conn = self._connect()

    def _connect(self):
        return afya_warehouse.connect(
            user=os.getenv("SNOWFLAKE_USER").strip(),
            account=os.getenv("SNOWFLAKE_ACCOUNT").strip(),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE").strip(),
            database=os.getenv("SNOWFLAKE_DATABASE").strip(),
            schema=self._schema if self._schema else os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC").strip(),
            private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH").strip(),
        )

    def query(self, sql):
        try:
            return self._execute(sql)
        except snowflake.connector.errors.ProgrammingError as exc:
            if getattr(exc, "errno", None) != _TOKEN_EXPIRED_ERRNO:
                raise
            logger.warning("Snowflake session token expired — reconnecting and retrying once.")
            try:
                self.conn.close()
            except Exception:
                pass  # the old connection is already dead either way
            self.conn = self._connect()
            return self._execute(sql)

    def _execute(self, sql):
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetch_pandas_all()
        finally:
            cursor.close()