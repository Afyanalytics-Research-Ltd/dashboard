"""Snowflake-shaped metadata on Databend.

Code all over the platform asks Snowflake's INFORMATION_SCHEMA questions
("which columns does HOSPITALS.RAW.X have?", "does table Y exist?") and
reads the answers by Snowflake's column names (TABLE_SCHEMA, COLUMN_NAME,
DATA_TYPE = 'NUMBER' / 'TEXT' / 'TIMESTAMP_NTZ' ...).  Databend's own
information_schema has lower-case columns, MySQL type names and no
ordinal positions, so the translator points every INFORMATION_SCHEMA
reference at the views created here instead.

Also creates the PUBLIC database (Snowflake's default schema always
exists).  Runs once per process per Databend server; cheap and idempotent.
"""

from __future__ import annotations

import logging
import threading

from .sql_translate import COMPAT_DB, default_primary_database

logger = logging.getLogger(__name__)

_done: set = set()
_lock = threading.Lock()

_EXCLUDED = f"('system', 'information_schema', '{COMPAT_DB}')"

_SF_TYPE = """CASE
    WHEN UPPER(c.data_type) LIKE 'DECIMAL%' OR UPPER(c.data_type) IN
         ('TINYINT','SMALLINT','INT','INTEGER','BIGINT','INT8','INT16','INT32','INT64',
          'UINT8','UINT16','UINT32','UINT64','TINYINT UNSIGNED','SMALLINT UNSIGNED',
          'INT UNSIGNED','BIGINT UNSIGNED') THEN 'NUMBER'
    WHEN UPPER(c.data_type) LIKE 'FLOAT%' OR UPPER(c.data_type) LIKE 'DOUBLE%' THEN 'FLOAT'
    WHEN UPPER(c.data_type) IN ('VARCHAR','STRING','TEXT') THEN 'TEXT'
    WHEN UPPER(c.data_type) LIKE 'TIMESTAMP%' THEN 'TIMESTAMP_NTZ'
    WHEN UPPER(c.data_type) = 'DATE' THEN 'DATE'
    WHEN UPPER(c.data_type) = 'BOOLEAN' THEN 'BOOLEAN'
    WHEN UPPER(c.data_type) = 'VARIANT' THEN 'VARIANT'
    WHEN UPPER(c.data_type) LIKE 'ARRAY%' THEN 'ARRAY'
    WHEN UPPER(c.data_type) LIKE 'TUPLE%' OR UPPER(c.data_type) LIKE 'MAP%' THEN 'OBJECT'
    WHEN UPPER(c.data_type) = 'BINARY' THEN 'BINARY'
    ELSE UPPER(c.data_type) END"""


def _views(primary: str) -> list[str]:
    cat = "'" + primary.replace("'", "''") + "'"
    return [
        f'CREATE DATABASE IF NOT EXISTS "{COMPAT_DB}"',
        'CREATE DATABASE IF NOT EXISTS "PUBLIC"',
        f'''CREATE OR REPLACE VIEW "{COMPAT_DB}"."COLUMNS" AS
SELECT {cat} AS "TABLE_CATALOG", c.database AS "TABLE_SCHEMA", c."table" AS "TABLE_NAME",
       c.name AS "COLUMN_NAME",
       ROW_NUMBER() OVER (PARTITION BY c.database, c."table") AS "ORDINAL_POSITION",
       NULLIF(c.default_expression, '') AS "COLUMN_DEFAULT", c.is_nullable AS "IS_NULLABLE",
       {_SF_TYPE} AS "DATA_TYPE",
       CASE WHEN {_SF_TYPE} = 'TEXT' THEN 16777216 END AS "CHARACTER_MAXIMUM_LENGTH",
       CASE WHEN {_SF_TYPE} = 'NUMBER' THEN 38 END AS "NUMERIC_PRECISION",
       CASE WHEN {_SF_TYPE} = 'NUMBER' AND UPPER(c.data_type) NOT LIKE 'DECIMAL%' THEN 0 END AS "NUMERIC_SCALE",
       NULLIF(c.comment, '') AS "COMMENT", c.type AS "DATABEND_TYPE"
FROM system.columns c
WHERE c.database NOT IN {_EXCLUDED}''',
        f'''CREATE OR REPLACE VIEW "{COMPAT_DB}"."TABLES" AS
SELECT {cat} AS "TABLE_CATALOG", t.database AS "TABLE_SCHEMA", t.name AS "TABLE_NAME",
       t.owner AS "TABLE_OWNER",
       CASE WHEN t.engine = 'VIEW' THEN 'VIEW' ELSE 'BASE TABLE' END AS "TABLE_TYPE",
       CASE WHEN t.is_transient = 'TRANSIENT' THEN 'YES' ELSE 'NO' END AS "IS_TRANSIENT",
       NULLIF(t.cluster_by, '') AS "CLUSTERING_KEY",
       t.num_rows AS "ROW_COUNT", t.data_size AS "BYTES",
       t.created_on AS "CREATED", t.updated_on AS "LAST_ALTERED", t.updated_on AS "LAST_DDL",
       NULLIF(t.comment, '') AS "COMMENT"
FROM system.tables t
WHERE t.database NOT IN {_EXCLUDED}''',
        f'''CREATE OR REPLACE VIEW "{COMPAT_DB}"."SCHEMATA" AS
SELECT {cat} AS "CATALOG_NAME", d.name AS "SCHEMA_NAME", d.owner AS "SCHEMA_OWNER",
       'NO' AS "IS_TRANSIENT", NULL AS "COMMENT"
FROM system.databases d
WHERE d.name NOT IN {_EXCLUDED} AND d.name <> 'default' ''',
        f'''CREATE OR REPLACE VIEW "{COMPAT_DB}"."VIEWS" AS
SELECT {cat} AS "TABLE_CATALOG", v.database AS "TABLE_SCHEMA", v.name AS "TABLE_NAME",
       v.view_query AS "VIEW_DEFINITION", v.owner AS "TABLE_OWNER", 'NO' AS "IS_SECURE",
       v.created_on AS "CREATED", v.updated_on AS "LAST_ALTERED"
FROM system.views v
WHERE v.database NOT IN {_EXCLUDED}''',
        f'''CREATE OR REPLACE VIEW "{COMPAT_DB}"."DATABASES" AS
SELECT {cat} AS "DATABASE_NAME", 'DATABEND' AS "DATABASE_OWNER", 'NO' AS "IS_TRANSIENT"''',
    ]


def ensure(transport) -> None:
    """Create/refresh the compatibility objects once per process."""
    key = transport.url
    if key in _done:
        return
    with _lock:
        if key in _done:
            return
        primary = default_primary_database()
        for sql in _views(primary):
            try:
                transport.run(sql)
            except Exception as exc:  # concurrent CREATE OR REPLACE from another process
                logger.warning("Databend compat setup: %s", str(exc)[:200])
        _done.add(key)
