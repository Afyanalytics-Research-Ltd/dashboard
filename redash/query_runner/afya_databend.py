"""Redash 10.1 query runner: Snowflake SQL, executed on Databend.

Bind-mounted into the Redash containers by docker-compose.redash.yaml and
enabled with REDASH_ADDITIONAL_QUERY_RUNNERS=redash.query_runner.afya_databend.

Every saved Redash query is written in Snowflake SQL.  When the dashboard's
warehouse switch moves queries onto a "<name> [databend]" data source (see
core/backends.py: sync_redash), this runner:

  1. sends the SQL to the dashboard's translate endpoint
     (POST /core/api/warehouse/translate/, header X-Afya-Token) - the same
     Snowflake->Databend translator the pipelines use, so a report means the
     same thing on both warehouses;
  2. runs the result on Databend's HTTP API (session cookies on, UTC, Oracle
     date formats), following pagination;
  3. returns typed columns like the Snowflake runner does.

Redash 10.1 runs Python 3.7 and ships requests, so this file uses nothing
else (no f-strings with '=' etc.).
"""
import json
import logging

import requests

from redash.query_runner import (
    TYPE_BOOLEAN,
    TYPE_DATE,
    TYPE_DATETIME,
    TYPE_FLOAT,
    TYPE_INTEGER,
    TYPE_STRING,
    BaseQueryRunner,
    register,
)
from redash.utils import json_dumps

logger = logging.getLogger(__name__)

SESSION_SETTINGS = {
    "timezone": "UTC",
    "date_format_style": "Oracle",
    "enable_auto_detect_datetime_format": "1",
}


def _redash_type(databend_type):
    t = databend_type or ""
    if t.startswith("Nullable(") and t.endswith(")"):
        t = t[len("Nullable("):-1]
    if t.startswith(("Int", "UInt")):
        return TYPE_INTEGER
    if t.startswith("Decimal"):
        return TYPE_INTEGER if t.replace(" ", "").endswith(",0)") else TYPE_FLOAT
    if t.startswith("Float"):
        return TYPE_FLOAT
    if t == "Boolean":
        return TYPE_BOOLEAN
    if t == "Date":
        return TYPE_DATE
    if t.startswith("Timestamp"):
        return TYPE_DATETIME
    return TYPE_STRING


def _convert(value, rtype):
    if value is None:
        return None
    try:
        if rtype == TYPE_INTEGER:
            return int(value)
        if rtype == TYPE_FLOAT:
            return float(value)
        if rtype == TYPE_BOOLEAN:
            return value in ("1", "true", "TRUE", "True")
    except ValueError:
        return value
    return value


class AfyaDatabend(BaseQueryRunner):
    noop_query = "SELECT 1"
    should_annotate_query = False

    @classmethod
    def name(cls):
        return "Databend (Afya, Snowflake SQL)"

    @classmethod
    def type(cls):
        return "afya_databend"

    @classmethod
    def configuration_schema(cls):
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "title": "Databend HTTP URL", "default": "http://databend:8000"},
                "user": {"type": "string"},
                "password": {"type": "string"},
                "database": {"type": "string", "title": "Snowflake database the queries use",
                             "default": "HOSPITALS"},
                "translate_url": {"type": "string", "title": "Translate endpoint",
                                  "default": "http://web:8000/core/api/warehouse/translate/"},
                "translate_token": {"type": "string", "title": "AFYA_INTERNAL_API_TOKEN"},
                "timeout": {"type": "number", "default": 300},
            },
            "order": ["url", "user", "password", "database", "translate_url", "translate_token", "timeout"],
            "required": ["url", "user", "translate_url"],
            "secret": ["password", "translate_token"],
        }

    @classmethod
    def enabled(cls):
        return True

    # -- helpers --------------------------------------------------------------
    def _translate(self, sql):
        resp = requests.post(
            self.configuration["translate_url"],
            data=json.dumps({"sql": sql, "database": self.configuration.get("database") or "HOSPITALS"}),
            headers={"Content-Type": "application/json",
                     "X-Afya-Token": self.configuration.get("translate_token") or ""},
            timeout=30,
        )
        if resp.status_code == 401:
            raise Exception("Translate endpoint rejected the token (check AFYA_INTERNAL_API_TOKEN).")
        body = resp.json()
        if resp.status_code >= 400:
            raise Exception("Could not translate query for Databend: {}".format(body.get("error")))
        return body.get("statements") or []

    def _run(self, http, sql, session):
        url = self.configuration["url"].rstrip("/")
        timeout = float(self.configuration.get("timeout") or 300)
        r = http.post(url + "/v1/query",
                      data=json.dumps({"sql": sql, "session": session,
                                       "pagination": {"wait_time_secs": 30, "max_rows_per_page": 50000}}),
                      headers={"Content-Type": "application/json"}, timeout=timeout)
        body = r.json()
        schema, rows = None, []
        while True:
            if body.get("session"):
                session = body["session"]
                session.setdefault("settings", {}).update(SESSION_SETTINGS)
            if body.get("error"):
                raise Exception("Databend: {}".format(body["error"].get("message")))
            if body.get("schema") and schema is None:
                schema = body["schema"]
            rows.extend(body.get("data") or [])
            nxt = body.get("next_uri")
            if not nxt:
                break
            body = http.get(url + nxt, timeout=timeout).json()
            if nxt.endswith("/final"):
                if body.get("error"):
                    raise Exception("Databend: {}".format(body["error"].get("message")))
                break
        return schema or [], rows, session

    # -- Redash API -----------------------------------------------------------
    def run_query(self, query, user):
        http = requests.Session()
        http.auth = (self.configuration.get("user") or "", self.configuration.get("password") or "")
        http.headers["X-DATABEND-CLIENT-CAPS"] = "session_cookie"
        session = {"settings": dict(SESSION_SETTINGS)}
        try:
            statements = self._translate(query)
            if not statements:
                return json_dumps({"columns": [], "rows": []}), None
            schema, rows = [], []
            for sql in statements:
                schema, rows, session = self._run(http, sql, session)
            columns = self.fetch_columns([(c["name"], _redash_type(c["type"])) for c in schema])
            names = [c["name"] for c in columns]
            types = [c["type"] for c in columns]
            out = [dict(zip(names, [_convert(v, t) for v, t in zip(r, types)])) for r in rows]
            return json_dumps({"columns": columns, "rows": out}), None
        except Exception as exc:  # Redash shows the message to the user
            logger.warning("afya_databend query failed: %s", exc)
            return None, str(exc)
        finally:
            http.close()

    def get_schema(self, get_stats=False):
        http = requests.Session()
        http.auth = (self.configuration.get("user") or "", self.configuration.get("password") or "")
        try:
            _, rows, _ = self._run(
                http,
                "SELECT database, \"table\", name FROM system.columns WHERE database NOT IN "
                "('system', 'information_schema', 'AFYA_SF_COMPAT', 'default') ORDER BY 1, 2",
                {"settings": dict(SESSION_SETTINGS)})
        finally:
            http.close()
        schema = {}
        for db, table, col in rows:
            # Shown as Snowflake names (HOSPITALS.<schema>.<table>) since
            # queries are written in Snowflake SQL.
            key = "{}.{}".format(db, table)
            schema.setdefault(key, {"name": key, "columns": []})["columns"].append(col)
        return list(schema.values())


register(AfyaDatabend)
