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
    ("TRY_TO_TIMESTAMP_NTZ(x)", "TRY_TO_TIMESTAMP(CAST(\"X\" AS VARCHAR))"),
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


# -- constructs found failing in HOSPITALS views (2026-11-16) ----------------

def test_try_to_timestamp_one_arg_accepts_timestamps():
    # Databend's TRY_TO_TIMESTAMP rejects a TIMESTAMP argument; Snowflake returns it.
    assert sql("select try_to_timestamp(created_at) from t") == \
        'SELECT TRY_TO_TIMESTAMP(CAST("CREATED_AT" AS VARCHAR)) FROM "T"'
    # the formatted form is unchanged
    assert "TRY_TO_TIMESTAMP(\"X\", '%Y-%m-%d')" in sql("select try_to_timestamp(x, 'YYYY-MM-DD') from t")


def test_regexp_substr_group_becomes_regexp_extract():
    out = sql("select regexp_substr(v, '([0-9]+)x', 1, 1, 'e', 1) from t")
    assert out == "SELECT NULLIF(REGEXP_EXTRACT(\"V\", '([0-9]+)x', 1), '') FROM \"T\""
    assert "REGEXP_EXTRACT(\"V\", '(?i)(a)(b)', 2)" in sql("select regexp_substr(v, '(a)(b)', 1, 1, 'ie', 2) from t")
    # plain forms and other positions/occurrences keep REGEXP_SUBSTR
    assert sql("select regexp_substr(v, '^[0-9]+') from t") == "SELECT REGEXP_SUBSTR(\"V\", '^[0-9]+') FROM \"T\""
    assert "REGEXP_SUBSTR" in sql("select regexp_substr(v, '(a)', 1, 2, 'e', 1) from t")


def test_filter_runs_on_native_array():
    out = sql("select get(filter(split(s, ' '), t -> length(t) > 1), 0)::string from t")
    assert "CAST(ARRAY_FILTER(SPLIT(\"S\", ' '), " in out and "AS VARIANT)" in out
    assert "FILTER(CAST(SPLIT" not in out


def test_trim_casts_non_text():
    assert sql("select lower(trim(type)) from t") == 'SELECT LOWER(TRIM(CAST("TYPE" AS VARCHAR))) FROM "T"'
    assert sql("select trim('  a ') from t") == "SELECT TRIM('  a ') FROM \"T\""
    assert "CAST(\"X\" AS VARCHAR)" in sql("select ltrim(x, 'a') from t")


def test_timestamp_from_parts_date_and_time():
    out = sql("select timestamp_from_parts(d, try_to_time(t)) from x")
    assert out.startswith("SELECT TRY_CAST(CONCAT(CAST(CAST(\"D\" AS DATE) AS VARCHAR), ' ', ")
    assert "()" not in out


def test_view_ctes_get_view_specific_names():
    # read twice -> stays a CTE, renamed per view (single-use CTEs are inlined, see below)
    out = sql("create or replace view staging.v as with deduped as (select id from raw.t) "
              "select deduped.id from deduped join raw.u u on u.id = deduped.id "
              "union all select d2.id from deduped d2")
    name = out.split('WITH "')[1].split('"')[0]
    assert name.startswith("DEDUPED_V") and len(name) == len("DEDUPED_V") + 8
    assert f'FROM "{name}" AS "DEDUPED"' in out and '"DEDUPED"."ID"' in out and f'FROM "{name}" AS "D2"' in out
    # a different view gets a different name; plain queries are untouched
    other = sql("create or replace view staging.w as with deduped as (select 1 as id) "
                "select * from deduped union all select * from deduped")
    assert name not in other
    assert 'WITH "DEDUPED" AS' in sql("with deduped as (select 1 as id) select * from deduped")


def test_view_single_use_ctes_are_inlined():
    """Databend registers a view's CTE names at every expansion: reading such a view in a NOT EXISTS and a JOIN
    failed with "Duplicate common table expression" (STAGING.STG_ADMISSIONS from REPORTING.VW_RPT_ORTHO_OPD)."""
    out = sql("create or replace view staging.adm as with eav as (select id, w from raw.t), "
              "ow as (select e.id, max(e.w) w from eav e group by 1) select a.*, ow.w from raw.a a left join ow on ow.id = a.id")
    assert "WITH" not in out                                # no CTE left to collide
    assert 'LEFT JOIN (SELECT "E"."ID", MAX("E"."W") AS "W" FROM (SELECT "ID", "W" FROM "RAW"."T") AS "E"' in out
    assert 'AS "OW" ON' in out


def test_view_unqualified_names_resolve_in_the_views_schema():
    out = sql("CREATE OR REPLACE VIEW patients_current_v AS WITH aged AS (SELECT p.* FROM patients p) "
              "SELECT a.*, g.age_group FROM aged a LEFT JOIN age_group_reference g ON a.x = g.x",
              tr.Context(database="HOSPITALS", schema="STAGING_V2", primary_database="HOSPITALS"))
    assert out.startswith('CREATE OR REPLACE VIEW "STAGING_V2"."PATIENTS_CURRENT_V" AS')
    assert '"STAGING_V2"."PATIENTS" AS "P"' in out and '"STAGING_V2"."AGE_GROUP_REFERENCE" AS "G"' in out


def test_scaled_to_timestamp_and_asof_match_condition():
    assert "TO_TIMESTAMP(CAST(\"APTDATE\" AS BIGINT), 6)" in sql("select to_timestamp(aptdate, 6) from t")
    out = sql("select u.k, f.v from u asof join f match_condition (u.d >= f.d) on u.k = f.k")
    assert "MATCH_CONDITION" not in out
    assert 'ASOF LEFT JOIN "F" ON "U"."D" >= "F"."D" AND "U"."K" = "F"."K"' in out


def test_flatten_value_only_where_flatten_is_read():
    out = sql("with neg as (select month, sum(v) as value from t group by 1), "
              "fl as (select x.value:m::string m from s, lateral flatten(input => s.arr) x) "
              "select month, value from neg union all select m, null from fl")
    assert 'SELECT "MONTH", "VALUE" FROM' in out          # ordinary column: Snowflake case
    assert 'GET_PATH("X"."value", \'m\')' in out          # FLATTEN's column: Databend's lower case


def test_coerce_text_aggregates_is_opt_in():
    q = "select sum(amount), sum(case when c is not null then amount else 0 end), avg(x) from t"
    assert "TRY_CAST" not in sql(q)
    out = sql(q, tr.Context(database="HOSPITALS", schema="RAW", primary_database="HOSPITALS",
                            coerce_text_aggregates=True))
    assert 'SUM(TRY_CAST("AMOUNT" AS DECIMAL(38, 6)))' in out
    assert 'WHEN NOT "C" IS NULL THEN TRY_CAST("AMOUNT" AS DECIMAL(38, 6)) ELSE 0 END' in out
    assert 'AVG(TRY_CAST("X" AS DECIMAL(38, 6)))' in out
