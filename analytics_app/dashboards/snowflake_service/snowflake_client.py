import logging
import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

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
        return snowflake.connector.connect(
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