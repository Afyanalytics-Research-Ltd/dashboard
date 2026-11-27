# dags/v3_platform_account.py
"""The V3 platform account a counting pipeline logs in with (patient_counts_to_snowflake, commercialization_counts).

`apply_platform_account(key)` puts the account for `key` into AFYA_<KEY>_* (what v3_to_v3_client.V3Client reads),
first source wins and Airflow values REPLACE pipelines/.env so the two never mix:

  1. Connection  afya_v3_<key>  (or an explicit conn id)
       Login / Password   the account (counting across organisations needs PLATFORM STATISTICS access)
       Host               URL template, e.g. https://{service}.afyaanalytics.ai/api/  (or Extra "url_template")
       Extra (optional)   {"organization_id", "facility_id", "migration_key", "urls": {...}, "environment"}
  2. Variables   AFYA_<K>_USERNAME / _PASSWORD / _URL_TEMPLATE (+ _ORGANIZATION_ID / _FACILITY_ID / _MIGRATION_KEY)
  3. pipelines/.env  AFYA_<K>_*

The domain changes from time to time: only the Connection / Variables need updating, never the code.
"""
from __future__ import annotations

import logging
import os

from pipelines_common import _connection, _variable, load_airflow_config

log = logging.getLogger(__name__)

ACCOUNT_FIELDS = ("USERNAME", "PASSWORD", "URL_TEMPLATE", "ORGANIZATION_ID", "FACILITY_ID", "MIGRATION_KEY")


def environment_label(url_template: str) -> str:
    u = (url_template or "").lower()
    if "migrate" in u:
        return "migrate"
    if "staging" in u:
        return "staging"
    return "prod"


def apply_platform_account(key: str, conn_id: str | None = None) -> str:
    """Put the platform account for `key` into AFYA_<KEY>_* from Airflow (Connection, then Variables),
    replacing any .env values so the two never mix. Returns where it came from."""
    up = key.upper()
    conn_id = conn_id or f"afya_v3_{key}"
    conn = _connection(conn_id)
    has_vars = bool(_variable(f"AFYA_{up}_USERNAME"))
    if conn is None and not has_vars:
        return "pipelines/.env"

    # Airflow has this account: drop the .env copy first (also per-service *_URL overrides).
    for name in list(os.environ):
        if name.startswith(f"AFYA_{up}_") and (name[len(f"AFYA_{up}_"):] in ACCOUNT_FIELDS or name.endswith("_URL")):
            os.environ.pop(name)
    os.environ.pop("UPD_RAW_ENVIRONMENT", None)

    if conn is not None:
        extra = conn.extra_dejson or {}
        template = (extra.get("url_template") or conn.host or "").strip()
        values = {"USERNAME": conn.login, "PASSWORD": conn.password, "URL_TEMPLATE": template,
                  "ORGANIZATION_ID": extra.get("organization_id"), "FACILITY_ID": extra.get("facility_id"),
                  "MIGRATION_KEY": extra.get("migration_key")}
        for field, value in values.items():
            if value not in (None, ""):
                os.environ[f"AFYA_{up}_{field}"] = str(value).strip()
        for svc, url in (extra.get("urls") or {}).items():
            os.environ[f"AFYA_{up}_{str(svc).upper()}_URL"] = str(url).strip()
        if extra.get("environment"):
            os.environ["UPD_RAW_ENVIRONMENT"] = str(extra["environment"]).strip()
        source = f"connection {conn_id}"
    else:
        load_airflow_config((key,))  # Variables AFYA_<K>_*
        source = f"variables AFYA_{up}_*"

    if not os.environ.get(f"AFYA_{up}_USERNAME") or not os.environ.get(f"AFYA_{up}_PASSWORD"):
        raise RuntimeError(f"{source}: platform account needs a login and password")
    if not os.environ.get(f"AFYA_{up}_URL_TEMPLATE") and not any(
            n.startswith(f"AFYA_{up}_") and n.endswith("_URL") for n in os.environ):
        raise RuntimeError(f"{source}: set the URL template (Connection Host / Extra url_template, "
                           f"or Variable AFYA_{up}_URL_TEMPLATE) so it never falls back to .env's URL")
    os.environ.setdefault("UPD_RAW_ENVIRONMENT", environment_label(os.environ.get(f"AFYA_{up}_URL_TEMPLATE", "")))
    log.info("platform account %r: %s as %s, environment=%s", key, source,
             os.environ[f"AFYA_{up}_USERNAME"], os.environ["UPD_RAW_ENVIRONMENT"])
    return source
