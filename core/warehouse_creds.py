"""Snowflake connection kwargs from the environment (Django side).

Shared by the backend health check and anything else in Django that needs
"the platform's Snowflake login" without its own config.  Mirrors what the
apps already read: key-pair auth when SNOWFLAKE_PRIVATE_KEY_PATH is set,
password otherwise.
"""

import os


def snowflake_kwargs(prefix: str = "SNOWFLAKE_") -> dict:
    g = lambda k, d="": (os.getenv(prefix + k) or d).strip()
    kw = dict(user=g("USER"), account=g("ACCOUNT"), warehouse=g("WAREHOUSE") or None,
              database=g("DATABASE") or None, schema=g("SCHEMA", "PUBLIC"), role=g("ROLE") or None)
    key = g("PRIVATE_KEY_PATH")
    if key:
        kw["private_key_file"] = key
    else:
        kw["password"] = g("PASSWORD")
    return {k: v for k, v in kw.items() if v is not None}
