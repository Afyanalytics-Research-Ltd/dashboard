"""End-to-end against a real Databend (skipped when none is reachable).

Run:  DATABEND_HTTP_URL=http://localhost:8008 pytest afya_backends/tests -q

Every test works in its own throw-away Snowflake schema (= Databend
database) and drops it afterwards.  The statements are the shapes the
pipelines, dashboards and agents actually send (see the pipeline study in
afya_backends/README.md).
"""

import datetime as dt
import json
import os
import uuid

import pytest
import requests

from afya_backends import toggles, warehouse

URL = os.getenv("DATABEND_HTTP_URL", "http://localhost:8008")


def _reachable():
    try:
        return requests.get(URL + "/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="no Databend at DATABEND_HTTP_URL")


@pytest.fixture()
def conn(monkeypatch):
    monkeypatch.setenv("DATABEND_HTTP_URL", URL)
    schema = "T_" + uuid.uuid4().hex[:10].upper()
    with toggles.override(toggles.WAREHOUSE, "databend"):
        c = warehouse.connect(account="ignored", user="ignored", warehouse="COMPUTE_WH", role="ACCOUNTADMIN",
                              database="HOSPITALS", schema=schema, private_key_file="/nonexistent")
        c.test_schema = schema
        yield c
        try:
            c.cursor().execute(f"DROP SCHEMA IF EXISTS HOSPITALS.{schema}")
        finally:
            c.close()


def q(conn, sql, params=None):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur


def test_connect_returns_databend(conn):
    assert warehouse.is_databend_connection(conn)


def test_session_schema_auto_created_and_one_part_names(conn):
    q(conn, "CREATE TABLE PATIENTS (id NUMBER(38,0), name VARCHAR)")
    q(conn, "INSERT INTO PATIENTS VALUES (1, 'a')")
    rows = q(conn, f"SELECT * FROM HOSPITALS.{conn.test_schema}.PATIENTS").fetchall()
    assert rows == [(1, "a")]


def test_execute_returns_cursor_for_chaining(conn):
    assert conn.cursor().execute("SELECT 1 AS x").fetchall() == [(1,)]


def test_uppercase_result_columns_like_snowflake(conn):
    cur = q(conn, 'SELECT 1 AS patient_id, 2 AS "mixedCase"')
    assert [d[0] for d in cur.description] == ["PATIENT_ID", "mixedCase"]


def test_param_styles(conn):
    q(conn, "CREATE TABLE P (id INT, s VARCHAR, ts TIMESTAMP_NTZ, d DATE, j VARIANT)")
    q(conn, "INSERT INTO P (id, s, ts, d) VALUES (%s, %s, %s, %s)",
      (1, "it's a \\ test %", dt.datetime(2026, 1, 2, 3, 4, 5), dt.date(2026, 1, 2)))
    q(conn, "INSERT INTO P (id, s) SELECT %(i)s, %(s)s", {"i": 2, "s": "named"})
    q(conn, "INSERT INTO P (id, s) VALUES (?, ?)", [3, "qmark"])
    assert q(conn, "SELECT s FROM P WHERE id = %s", (1,)).fetchone()[0] == "it's a \\ test %"
    # literal % with no params stays literal; all three strings contain "a"
    assert q(conn, "SELECT COUNT(*) FROM P WHERE s LIKE '%a%'").fetchone()[0] == 3
    assert q(conn, "SELECT COUNT(*) FROM P WHERE s LIKE '%%q%%' AND id > %s", (0,)).fetchone()[0] == 1
    assert q(conn, "SELECT ts, d FROM P WHERE id = 1").fetchone() == (dt.datetime(2026, 1, 2, 3, 4, 5),
                                                                        dt.date(2026, 1, 2))


def test_executemany_bulk_and_function_wrapped_placeholders(conn):
    q(conn, "CREATE TABLE B (id INT, ts TIMESTAMP_NTZ, j VARIANT)")
    cur = conn.cursor()
    cur.executemany("INSERT INTO B (id, ts) VALUES (%s, TO_TIMESTAMP_NTZ(%s))",
                    [(i, "2026-03-04 05:06:07") for i in range(3000)])
    assert cur.rowcount == 3000
    cur.executemany("INSERT INTO B (id, j) SELECT %s, PARSE_JSON(%s)", [(9001, json.dumps({"a": [1]}))])
    assert q(conn, "SELECT COUNT(*), MIN(ts) FROM B WHERE id < 3000").fetchone() == (3000, dt.datetime(2026, 3, 4, 5, 6, 7))
    assert json.loads(q(conn, "SELECT j FROM B WHERE id = 9001").fetchone()[0]) == {"a": [1]}


def test_variant_json_flatten_qualify(conn):
    q(conn, "CREATE TABLE EVENTS_RAW (facility_id STRING, ingested_at TIMESTAMP_TZ, payload VARIANT)")
    for i, fac in enumerate(["k", "k", "t"]):
        q(conn, "INSERT INTO EVENTS_RAW SELECT %s, %s::TIMESTAMP_TZ, PARSE_JSON(%s)",
          (fac, f"2026-01-0{i + 1}T10:00:00+03:00", json.dumps({"id": i, "items": [{"code": "A"}, {"code": "B"}]})))
    rows = q(conn, """SELECT facility_id, payload:id::int AS id FROM EVENTS_RAW
                      QUALIFY ROW_NUMBER() OVER (PARTITION BY facility_id ORDER BY ingested_at DESC) = 1
                      ORDER BY facility_id""").fetchall()
    assert rows == [("k", 1), ("t", 2)]
    codes = q(conn, """SELECT f.value:code::string AS code, COUNT(*) AS n FROM EVENTS_RAW r,
                       LATERAL FLATTEN(input => r.payload:items) f GROUP BY 1 ORDER BY 1""").fetchall()
    assert codes == [("A", 3), ("B", 3)]
    # TIMESTAMP_TZ literal with offset lands as the UTC instant (NTZ semantics)
    assert q(conn, "SELECT MIN(ingested_at) FROM EVENTS_RAW").fetchone()[0] == dt.datetime(2026, 1, 1, 7, 0)


def test_merge_with_alias_and_result_counts(conn):
    q(conn, "CREATE TABLE T (id INT, name VARCHAR)")
    q(conn, "INSERT INTO T VALUES (1, 'a')")
    rows = q(conn, """MERGE INTO T tgt USING (SELECT 1 AS id, 'A' AS name UNION ALL SELECT 2, 'B') s
                      ON tgt.id = s.id WHEN MATCHED THEN UPDATE SET tgt.name = s.name
                      WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)""").fetchall()
    assert rows == [(1, 1)]  # (inserted, updated) like Snowflake
    assert q(conn, "SELECT id, name FROM T ORDER BY id").fetchall() == [(1, "A"), (2, "B")]


def test_delete_using_and_update_from(conn):
    q(conn, "CREATE TABLE T (id INT, name VARCHAR)")
    q(conn, "INSERT INTO T VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    assert q(conn, "DELETE FROM T USING (SELECT 2 AS id) s WHERE T.id = s.id").rowcount == 1
    q(conn, "UPDATE T SET name = s.n FROM (SELECT 3 AS id, 'z' AS n) s WHERE T.id = s.id")
    assert q(conn, "SELECT id, name FROM T ORDER BY id").fetchall() == [(1, "a"), (3, "z")]


def test_information_schema_and_show(conn):
    s = conn.test_schema
    q(conn, "CREATE TABLE X (id NUMBER(38,0), amount NUMBER(10,2), s VARCHAR, ts TIMESTAMP_NTZ, v VARIANT)")
    cols = q(conn, f"""SELECT COLUMN_NAME, DATA_TYPE FROM HOSPITALS.INFORMATION_SCHEMA.COLUMNS
                       WHERE TABLE_SCHEMA = '{s}' AND TABLE_NAME = 'X' ORDER BY ORDINAL_POSITION""").fetchall()
    assert cols == [("ID", "NUMBER"), ("AMOUNT", "NUMBER"), ("S", "TEXT"), ("TS", "TIMESTAMP_NTZ"), ("V", "VARIANT")]
    tables = q(conn, f"SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = '{s}'").fetchall()
    assert tables == [("X", "BASE TABLE")]
    dcur = conn.cursor(type("DictCursor", (), {}))
    dcur.execute(f"SHOW TABLES LIKE 'x' IN SCHEMA HOSPITALS.{s}")
    assert [r["name"] for r in dcur.fetchall()] == ["X"]
    rows = q(conn, f"SHOW COLUMNS IN TABLE HOSPITALS.{s}.X").fetchall()
    assert [r[2] for r in rows] == ["ID", "AMOUNT", "S", "TS", "V"]


def test_missing_table_error_is_snowflake_like(conn):
    import snowflake.connector.errors as se
    with pytest.raises(se.ProgrammingError) as ei:
        q(conn, "SELECT * FROM HOSPITALS.NOPE_SCHEMA.NOPE")
    assert ei.value.errno == 2003 and "does not exist" in str(ei.value)


def test_noop_statements_and_truncate_if_exists(conn):
    for s in ("USE WAREHOUSE COMPUTE_WH", "USE ROLE ACCOUNTADMIN", "ALTER SESSION SET QUERY_TAG='x'",
              "TRUNCATE TABLE IF EXISTS NOT_THERE"):
        q(conn, s).fetchall()


def test_swap_and_clone(conn):
    q(conn, "CREATE TABLE A AS SELECT 1 AS x")
    q(conn, "CREATE TABLE B AS SELECT 2 AS x UNION ALL SELECT 3")
    q(conn, "ALTER TABLE A SWAP WITH B")
    assert q(conn, "SELECT COUNT(*) FROM A").fetchone()[0] == 2
    q(conn, "CREATE TABLE IF NOT EXISTS C CLONE A")
    assert q(conn, "SELECT COUNT(*) FROM C").fetchone()[0] == 2


def test_temp_table_and_write_pandas_into_it(conn):
    import pandas as pd
    q(conn, "CREATE TEMPORARY TABLE IF NOT EXISTS _STG_X (_SKEY VARCHAR, _SDATA VARCHAR)")
    df = pd.DataFrame({"_SKEY": ["a", "b"], "_SDATA": [json.dumps({"x": 1}), None]})
    ok, nchunks, nrows, _ = warehouse.write_pandas(conn, df, "_STG_X", database="HOSPITALS",
                                                   schema=conn.test_schema, quote_identifiers=False)
    assert ok and nrows == 2
    assert q(conn, "SELECT PARSE_JSON(_SDATA):x::int FROM _STG_X WHERE _SKEY = 'a'").fetchone()[0] == 1


def test_write_pandas_auto_create_types_and_edge_values(conn):
    import pandas as pd
    df = pd.DataFrame({
        "ID": [1, 2, 3], "NAME": ["a", None, 'q"uote, comma\nnewline'],
        "TS": pd.to_datetime(["2026-01-01 00:00:00", None, "2026-01-03 04:05:06"]),
        "AMT": [1.5, float("nan"), 3.0], "FLAG": [True, False, True],
        "obj": [{"k": 1}, None, [1, 2]], "EMPTY": ["", "x", "\\N"],
    })
    res = warehouse.write_pandas(conn, df, "WP", schema=conn.test_schema, auto_create_table=True, overwrite=True)
    assert res[0] and res[2] == 3
    rows = q(conn, 'SELECT ID, NAME, TS, AMT, FLAG, "obj", EMPTY FROM WP ORDER BY ID').fetchall()
    assert rows[0] == (1, "a", dt.datetime(2026, 1, 1), 1.5, True, '{"k":1}', "")
    assert rows[1][1] is None and rows[1][2] is None and rows[1][3] is None
    assert rows[2][1] == 'q"uote, comma\nnewline' and rows[2][6] == "\\N"
    df2 = q(conn, "SELECT * FROM WP ORDER BY ID").fetch_pandas_all()
    assert list(df2.columns) == ["ID", "NAME", "TS", "AMT", "FLAG", "obj", "EMPTY"]
    assert str(df2["ID"].dtype) == "int64"


def test_transactions(conn):
    q(conn, "CREATE TABLE T (id INT)")
    cur = conn.cursor()
    cur.execute("BEGIN")
    cur.execute("INSERT INTO T VALUES (1)")
    cur.execute("ROLLBACK")
    conn.autocommit(False)
    cur.execute("INSERT INTO T VALUES (2)")
    conn.commit()
    cur.execute("INSERT INTO T VALUES (3)")
    conn.rollback()
    conn.autocommit(True)
    assert q(conn, "SELECT id FROM T").fetchall() == [(2,)]


def test_context_manager_commits(monkeypatch):
    monkeypatch.setenv("DATABEND_HTTP_URL", URL)
    schema = "T_" + uuid.uuid4().hex[:10].upper()
    with toggles.override(toggles.WAREHOUSE, "databend"):
        with warehouse.connect(database="HOSPITALS", schema=schema) as c:
            c.cursor().execute("CREATE TABLE T AS SELECT 1 AS x")
        with warehouse.connect(database="HOSPITALS", schema=schema) as c:
            assert c.cursor().execute("SELECT x FROM T").fetchall() == [(1,)]
            c.cursor().execute(f"DROP SCHEMA HOSPITALS.{schema}")


def test_hash_star_dedup_query(conn):
    q(conn, "CREATE TABLE D (id INT, v VARCHAR)")
    q(conn, "INSERT INTO D VALUES (1, 'a'), (1, 'a'), (2, NULL)")
    rows = q(conn, "SELECT COUNT(DISTINCT HASH(*)) FROM D").fetchone()
    assert rows[0] == 2
    rows = q(conn, 'SELECT id, HASH(*) AS "__H" FROM D QUALIFY ROW_NUMBER() OVER (PARTITION BY TO_VARCHAR(id) '
                   'ORDER BY "__H") = 1 ORDER BY id').fetchall()
    assert [r[0] for r in rows] == [1, 2]


def test_array_union_agg_and_object_keys(conn):
    q(conn, "CREATE TABLE E (source_table VARCHAR, payload VARIANT)")
    q(conn, "INSERT INTO E SELECT 'a', PARSE_JSON('{\"x\":1,\"y\":2}') UNION ALL SELECT 'a', PARSE_JSON('{\"y\":3,\"z\":4}')")
    keys = q(conn, "SELECT ARRAY_UNION_AGG(OBJECT_KEYS(payload)) FROM E WHERE IS_OBJECT(payload)").fetchone()[0]
    assert sorted(json.loads(keys)) == ["x", "y", "z"]


def test_put_and_copy_from_table_stage(conn, tmp_path):
    p = tmp_path / "x_chunk.csv"
    p.write_text('ID,NAME\n1,"a, b"\n2,c\n')
    q(conn, "CREATE TABLE X_STAGING (ID INT, NAME VARCHAR)")
    q(conn, f"PUT file://{p} @%X_STAGING AUTO_COMPRESS=TRUE OVERWRITE=TRUE")
    cur = q(conn, "COPY INTO X_STAGING FROM @%X_STAGING FILE_FORMAT = (TYPE=CSV SKIP_HEADER=1 "
                  "FIELD_OPTIONALLY_ENCLOSED_BY='\"') ON_ERROR = 'CONTINUE' PURGE = TRUE")
    res = cur.fetchall()
    assert [d[0] for d in cur.description][:4] == ["file", "status", "rows_parsed", "rows_loaded"]
    assert sum(r[3] for r in res) == 2 and cur.rowcount == 2
    assert q(conn, "SELECT NAME FROM X_STAGING ORDER BY ID").fetchall() == [("a, b",), ("c",)]


def test_async_query(conn):
    cur = conn.cursor()
    qid = cur.execute_async("SELECT COUNT(*) AS N FROM numbers(1000)")["queryId"]
    cur.get_results_from_sfqid(qid)
    assert cur.fetchall() == [(1000,)]
    assert conn.get_query_status(qid).name == "SUCCESS"


def test_functions_semantics(conn):
    row = q(conn, """SELECT DATEDIFF('day', '2026-01-01 23:00'::TIMESTAMP_NTZ, '2026-01-02 01:00'::TIMESTAMP_NTZ),
                            DATEADD(month, 1, '2026-01-31'::DATE), TRY_TO_NUMBER('12.5'), TRY_TO_NUMBER('x'),
                            TRY_TO_DATE('05/01/2026', 'DD/MM/YYYY'), IFF(1 > 0, 'y', 'n'), ZEROIFNULL(NULL),
                            DAYNAME('2026-01-04'::DATE), DAYOFWEEK('2026-01-04'::DATE), TO_BOOLEAN('yes'),
                            CONVERT_TIMEZONE('Africa/Nairobi', 'UTC', '2026-01-01 10:00:00'::TIMESTAMP_NTZ),
                            TO_VARCHAR('2026-01-05 14:07:09'::TIMESTAMP_NTZ, 'YYYY-MM-DD"T"HH24:MI:SS'),
                            'abc' RLIKE 'b', 'abc' RLIKE 'a.*', SPLIT('a,b', ',')[0]::string,
                            VARIANCE(x) FROM (SELECT 1 AS x UNION ALL SELECT 3)""").fetchone()
    assert row[:15] == (1, dt.date(2026, 2, 28), 13, None, dt.date(2026, 1, 5), "y", 0, "Sun", 0, True,
                        dt.datetime(2026, 1, 1, 7, 0), "2026-01-05T14:07:09", False, True, "a")
    assert row[15] == pytest.approx(2.0)


def test_generator_trend_months(conn):
    rows = q(conn, """SELECT DATEADD(MONTH, -SEQ4(), DATE_TRUNC('MONTH', '2026-05-17'::DATE)) AS m
                      FROM TABLE(GENERATOR(ROWCOUNT => 3)) ORDER BY m""").fetchall()
    assert [r[0] for r in rows] == [dt.date(2026, 3, 1), dt.date(2026, 4, 1), dt.date(2026, 5, 1)]


def test_airflow_hook_on_databend(monkeypatch):
    monkeypatch.setenv("DATABEND_HTTP_URL", URL)
    from afya_backends.warehouse.airflow_hook import WarehouseHook, DatabendHook
    with toggles.override(toggles.WAREHOUSE, "databend"):
        hook = WarehouseHook(snowflake_conn_id="snowflake_default", database="HOSPITALS", schema="PUBLIC")
        assert isinstance(hook, DatabendHook)
        assert hook.get_first("SELECT 1 + 1") == (2,)
        assert hook.get_records("SELECT number FROM numbers(3) ORDER BY 1") == [(0,), (1,), (2,)]
        assert list(hook.get_pandas_df("SELECT 1 AS a").columns) == ["A"]
