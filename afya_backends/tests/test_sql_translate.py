"""Snowflake -> Databend translation (no database needed).

Each case pins down a difference found by running the Snowflake construct
on databend-query v1.2.881; test_databend_live.py runs the translated SQL
for real.
"""

import pytest

from afya_backends.warehouse import sql_translate as tr

CTX = tr.Context(database="HOSPITALS", schema="RAW", primary_database="HOSPITALS")


def one(sql, ctx=CTX):
    out = tr.translate(sql, ctx)
    assert len(out) == 1, out
    return out[0]


def sql(s, ctx=CTX):
    st = one(s, ctx)
    assert st.action == "sql", st
    return st.sql


# -- names ------------------------------------------------------------------

def test_three_part_names_map_schema_to_database():
    assert sql("select * from HOSPITALS.STAGING.STG_VISITS") == 'SELECT * FROM "STAGING"."STG_VISITS"'


def test_two_part_and_one_part_names():
    assert sql("select * from staging.t") == 'SELECT * FROM "STAGING"."T"'
    assert sql("select * from t") == 'SELECT * FROM "T"'


def test_other_snowflake_database_gets_prefixed():
    assert sql("select * from AFYA_PROD.RAW.X") == 'SELECT * FROM "AFYA_PROD__RAW"."X"'


def test_unquoted_identifiers_uppercased_quoted_kept():
    out = sql('select id, "source_schema", Name as pName from t')
    assert out == 'SELECT "ID", "source_schema", "NAME" AS "PNAME" FROM "T"'


def test_information_schema_goes_to_compat_views():
    out = sql("select column_name from HOSPITALS.INFORMATION_SCHEMA.COLUMNS where table_schema = 'RAW'")
    assert '"AFYA_SF_COMPAT"."COLUMNS"' in out and '"COLUMN_NAME"' in out
    assert '"AFYA_SF_COMPAT"."TABLES"' in sql("select * from information_schema.tables")


# -- functions ----------------------------------------------------------------

@pytest.mark.parametrize("snow, expect", [
    ("a ILIKE 'x%'", "LOWER(\"A\") LIKE LOWER('x%')"),
    ("DAYNAME(ts)", "TO_CHAR(\"TS\", 'DY')"),
    ("MONTHNAME(ts)", "TO_CHAR(\"TS\", 'Mon')"),
    ("HOUR(ts)", "TO_HOUR(\"TS\")"),
    ("TO_DOUBLE(x)", "TO_FLOAT64(\"X\")"),
    ("TRY_TO_DOUBLE(x)", "TRY_TO_FLOAT64(\"X\")"),
    ("TO_TIMESTAMP_NTZ(x)", "TO_TIMESTAMP(\"X\")"),
    ("TRY_TO_TIMESTAMP_NTZ(x)", "TRY_TO_TIMESTAMP(\"X\")"),
    ("TRY_TO_NUMBER(x)", "TRY_CAST(\"X\" AS DECIMAL(38, 0))"),
    ("TRY_TO_NUMBER(x, 10, 2)", "TRY_CAST(\"X\" AS DECIMAL(10, 2))"),
    ("SYSDATE()", "NOW()"),
    ("CURRENT_TIMESTAMP()", "NOW()"),
    ("UUID_STRING()", "UUID()"),
    ("BOOLAND_AGG(x)", "BOOL_AND(\"X\")"),
    ("MIN_BY(a, b)", "ARG_MIN(\"A\", \"B\")"),
    ("VARIANCE(x)", "POW(STDDEV_SAMP(\"X\"), 2)"),
    ("MD5_HEX(x)", "MD5(\"X\")"),
    ("SHA2(x)", "SHA2(\"X\", 256)"),
    ("DAYOFWEEK(ts)", "(TO_DAY_OF_WEEK(\"TS\") % 7)"),
    ("CURRENT_SCHEMA()", "CURRENT_DATABASE()"),
    ("TO_BOOLEAN(x)", "CASE WHEN LOWER(TRIM(CAST(\"X\" AS VARCHAR))) IN"),
])
def test_function_rewrites(snow, expect):
    assert expect in sql(f"select {snow} from t")


def test_date_format_strings_are_strftime_or_oracle():
    out = sql("select to_date(x, 'DD/MM/YYYY'), to_char(ts, 'YYYY-MM') from t")
    assert "TO_DATE(\"X\", '%d/%m/%Y')" in out
    assert "TO_CHAR(\"TS\", 'YYYY-MM')" in out


def test_to_char_quoted_literal_text_split():
    out = sql("select to_char(ts, 'YYYY-MM-DD\"T\"HH24:MI:SS') from t")
    assert out.startswith("SELECT CONCAT(TO_CHAR(") and "'T'" in out


def test_regexp_like_is_anchored():
    assert "'^(?:a.*)$'" in sql("select * from t where x rlike 'a.*'")


def test_percentile_cont_desc():
    out = sql("select percentile_cont(0.9) within group (order by x desc) from t")
    assert "QUANTILE_CONT(0.09" in out or "QUANTILE_CONT(0.1" in out


def test_split_is_zero_indexed_variant():
    assert "CAST(SPLIT(" in sql("select split(a, ',')[0] from t")


def test_flatten_columns_stay_lowercase_and_alias_list_dropped():
    out = sql("select f.value:id::string, f.index from t, lateral flatten(input => t.payload:items) f")
    assert '"F"."value"' in out and '"F"."index"' in out
    assert '("SEQ"' not in out
    assert "FLATTEN(input => " in out


def test_values_columns_renamed():
    out = sql("select column1, column2 from values (1, 'a')")
    assert '"col0"' in out and '"col1"' in out


def test_union_order_by_ordinal_wrapped():
    out = sql("select a from t union all select b from u order by 1")
    assert out.startswith("SELECT * FROM (SELECT")


def test_convert_timezone_three_arg():
    out = sql("select convert_timezone('Africa/Nairobi', 'UTC', ts) from t")
    assert "CONVERT_TIMEZONE('UTC', DATEADD(SECOND, -DATEDIFF(SECOND" in out


def test_generator_becomes_numbers():
    out = sql("SELECT DATEADD(month, -SEQ4(), d) AS m FROM x, TABLE(GENERATOR(ROWCOUNT => 12))")
    assert "NUMBERS(12)" in out and "-number" in out and "GENERATOR" not in out


def test_hash_star_marker():
    out = sql("select a, hash(*) as h from staging.t")
    assert "AFYA_HASH_STAR('\"STAGING\".\"T\"')" in out


# -- DDL / DML ------------------------------------------------------------------

def test_create_table_types_and_constraints():
    out = sql("CREATE OR REPLACE TRANSIENT TABLE HOSPITALS.RAW.X (id NUMBER(38,0) AUTOINCREMENT, ts TIMESTAMP_NTZ(9),"
              " v VARIANT, o OBJECT, a ARRAY, s VARCHAR(16777216), t TIME, f FLOAT, i INT, PRIMARY KEY (id))")
    assert '"RAW"."X"' in out
    for frag in ('"TS" TIMESTAMP', '"O" VARIANT', '"A" VARIANT', '"S" VARCHAR', '"T" VARCHAR', '"F" DOUBLE',
                 '"I" BIGINT'):
        assert frag in out, (frag, out)
    assert "PRIMARY KEY" not in out and "AUTOINCREMENT" not in out


def test_merge_target_alias_renamed_and_set_unqualified():
    out = sql("MERGE INTO t AS tgt USING s ON tgt.id = s.id WHEN MATCHED THEN UPDATE SET tgt.name = s.name "
              "WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)")
    assert 'AS "afya_tgt"' in out and '"afya_tgt"."ID" = "S"."ID"' in out
    assert 'UPDATE SET "NAME" = "S"."NAME"' in out


def test_delete_using_becomes_exists():
    out = sql("DELETE FROM t USING s WHERE t.id = s.id")
    assert "WHERE EXISTS(SELECT 1 FROM" in out.replace("EXISTS (", "EXISTS(")


def test_clone_becomes_ctas():
    assert "AS SELECT * FROM" in sql("CREATE TABLE IF NOT EXISTS b.x CLONE c.y")


@pytest.mark.parametrize("stmt", [
    "USE WAREHOUSE COMPUTE_WH", "USE ROLE ACCOUNTADMIN", "ALTER SESSION SET TIMEZONE = 'UTC'",
    "GRANT SELECT ON ALL TABLES IN SCHEMA HOSPITALS.REPORTING TO ROLE R",
    "CREATE OR REPLACE ROW ACCESS POLICY p AS (x VARCHAR) RETURNS BOOLEAN -> TRUE",
    "ALTER TABLE t ADD ROW ACCESS POLICY p ON (x)",
])
def test_noops(stmt):
    assert one(stmt).action == "noop"


def test_schema_statements():
    st = one("CREATE SCHEMA IF NOT EXISTS HOSPITALS.STAGING")
    assert st.action == "create_schema" and st.params["schema"] == "STAGING"
    assert 'CREATE DATABASE IF NOT EXISTS "STAGING"' == st.sql
    st = one("USE SCHEMA HOSPITALS.RAW")
    assert st.action == "use_schema" and st.params == {"schema": "RAW", "database": "HOSPITALS"}
    assert one("DROP SCHEMA IF EXISTS HOSPITALS.X").sql == 'DROP DATABASE IF EXISTS "X"'


def test_client_side_actions():
    assert one("TRUNCATE TABLE IF EXISTS raw.x").action == "truncate_if_exists"
    assert one("ALTER TABLE a SWAP WITH b").action == "swap"
    st = one("PUT file:///tmp/x.csv @%MY_TABLE AUTO_COMPRESS=TRUE OVERWRITE=TRUE")
    assert st.action == "put" and st.params["stage"] == "tblstage_my_table" and st.params["overwrite"]


def test_copy_from_external_stage_with_select():
    st = one("""COPY INTO HOSPITALS.KISUMU_RAW.EVENTS_RAW (facility_id, ingested_at, payload)
        FROM (SELECT 'kisumu'::STRING AS facility_id, '2026-01-01T00:00:00+03:00'::TIMESTAMP_TZ AS ingested_at,
              PARSE_JSON($1) AS payload FROM @HOSPITALS.SHARED.FACILITY_RAW_STAGE)
        FILES = ('raw/facilities/x.jsonl.gz') FILE_FORMAT = (FORMAT_NAME = HOSPITALS.SHARED.JSON_FF)
        ON_ERROR = 'CONTINUE';""")
    assert st.action == "copy"
    assert st.sql.startswith('COPY INTO "KISUMU_RAW"."EVENTS_RAW" ("FACILITY_ID", "INGESTED_AT", "PAYLOAD") FROM (SELECT')
    assert "FROM @facility_raw_stage" in st.sql
    assert "FORMAT_NAME = 'json_ff'" in st.sql and "ON_ERROR = continue" in st.sql
    assert "AS TIMESTAMP" in st.sql and "TIMESTAMP_TZ" not in st.sql
    assert st.params == {"stages": ["facility_raw_stage"], "file_formats": ["json_ff"]}


def test_copy_from_table_stage_csv():
    st = one("COPY INTO staging FROM @%\"x_staging\" FILE_FORMAT = (TYPE=CSV SKIP_HEADER=1 "
             "FIELD_OPTIONALLY_ENCLOSED_BY='\"') ON_ERROR = 'CONTINUE' PURGE = TRUE")
    assert "FROM @tblstage_x_staging" in st.sql and "QUOTE=" in st.sql.replace(" ", "")


def test_show_and_describe():
    st = one("SHOW TABLES LIKE 'x%' IN SCHEMA HOSPITALS.MEDHANIT_V1")
    assert "database = 'MEDHANIT_V1'" in st.sql and "LOWER(name) LIKE LOWER('x%')" in st.sql
    st = one("SHOW COLUMNS IN TABLE HOSPITALS.MEDHANIT_V1.patients")
    assert "\"TABLE_NAME\" = 'PATIENTS'" in st.sql and "\"TABLE_SCHEMA\" = 'MEDHANIT_V1'" in st.sql
    assert '"COLUMN_NAME"' in one("DESC TABLE raw.x").sql


def test_split_statements_respects_quotes_and_comments():
    parts = tr.split_statements("select 'a;b'; -- c;\nselect \"x;\" from t; /* ; */ select 3;")
    assert len(parts) == 3 and parts[0] == "select 'a;b'"


def test_identifier_function_inlined():
    assert sql("select * from identifier('HOSPITALS.RAW.T')") == 'SELECT * FROM "RAW"."T"'


def test_context_changes_mapping():
    ctx = tr.Context(database="AFYA_PROD", schema="X", primary_database="HOSPITALS")
    assert sql("select * from raw.t", ctx) == 'SELECT * FROM "AFYA_PROD__RAW"."T"'
