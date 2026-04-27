from collections import defaultdict
import os 
import snowflake.connector
from dotenv import load_dotenv 

load_dotenv()

# Environment variables for Snowflake connection
SF_ACCOUNT = os.getenv("SF_ACCOUNT") 
SF_USER = os.getenv("SF_USER")
SF_AUTHENTICATOR = os.getenv("SF_AUTHENTICATOR")
SF_PASSWORD = os.getenv("SF_PASSWORD")
SF_ROLE = os.getenv("SF_ROLE")
SF_WAREHOUSE = os.getenv("SF_WAREHOUSE")
SF_DATABASE = os.getenv("SF_DATABASE")
SF_SCHEMA = os.getenv("SF_SCHEMA")

# Different schema for raw and transformed data 
RAW_SCHEMAS = ["KISUMU_RAW", "KAKAMEGA_RAW", "LODWAR_RAW"] 

# Final schema for flattened tables 
CLEAN_SCHEMAS = ["KISUMU_CLEAN","KAKAMEGA_CLEAN","LODWAR_CLEAN"]


SCHEMA_PAIRS = [
    ("KISUMU_RAW", "KISUMU_CLEAN"),
    ("KAKAMEGA_RAW", "KAKAMEGA_CLEAN"),
    ("LODWAR_RAW", "LODWAR_CLEAN"),
]

# SNOWFLAKE CONNECTION

def _snowflake_connect(**kwargs):
    """
    Helper function to establish a connection to Snowflake using provided parameters.
    """
    return snowflake.connector.connect(
        account=SF_ACCOUNT,
        user=SF_USER,
        password=SF_PASSWORD,
        role=SF_ROLE,
        warehouse=SF_WAREHOUSE,
        database=SF_DATABASE,
        authenticator=SF_AUTHENTICATOR,
        **kwargs
    )
# HELPER FUNCTIONS

# resolves when we have multiple observed types for a field, e.g. ["INTEGER", "NULL_VALUE"] or ["INTEGER", "VARCHAR"]
def resolve_type(types):
    # Remove NULL_VALUE — it's "absence of data," not a real type
    real_types = [t for t in types if t != 'NULL_VALUE']
    
    # All observations were null (shouldn't happen if SQL filter works, but defensive)
    if not real_types:
        return 'VARCHAR'
    
    # Only one real type (possibly after removing NULL_VALUE)
    types_set = set(real_types)
    if len(types_set) == 1:
        return real_types[0]
    
    # Mixed scalar + nested → preserve as VARIANT
    if types_set & {'OBJECT', 'ARRAY'}:
        print(f"    ⚠ mixed scalar/nested {types} → VARIANT")
        return 'VARIANT'
    
    # All numeric → widen to biggest
    if types_set <= {'INTEGER', 'DECIMAL', 'DOUBLE'}:
        if 'DOUBLE' in types_set:
            return 'DOUBLE'
        if 'DECIMAL' in types_set:
            return 'DECIMAL'
        return 'INTEGER'  # unreachable given types_set has >1 element, but explicit
    
    # Genuinely mixed scalars → VARCHAR
    print(f"    ⚠ mixed scalars {types} → VARCHAR")
    return 'VARCHAR'

# Map Snowflake types to our target types for casting in the final SELECT statement
def infer_type(snowflake_type):
    mapping = {
        'INTEGER': 'NUMBER',
        'DECIMAL': 'NUMBER(18,2)',
        'DOUBLE': 'FLOAT',
        'VARCHAR': 'STRING',
        'TEXT': 'STRING',
        'BOOLEAN': 'BOOLEAN',
        'DATE': 'DATE',
        'TIMESTAMP_NTZ': 'TIMESTAMP',
        'TIMESTAMP_TZ': 'TIMESTAMP',
        'TIMESTAMP_LTZ': 'TIMESTAMP',
        'ARRAY': 'ARRAY',
        'OBJECT': 'OBJECT',
    }
    return mapping.get(snowflake_type.upper(), 'STRING')  # safe fallback

# Check how often a JSON path is populated to decide whether to expand it or keep as VARIANT
def check_fill_rate(cursor, raw_schema, table, json_path):
    """Return fraction of records where json_path is populated. Accepts dotted paths like 'bed_type.type'."""
    sql_path = ':'.join(f'"{p}"' for p in json_path.split('.'))
    query = f"""
        SELECT
            COUNT(*) AS total,
            COUNT(f.value:{sql_path}) AS populated
        FROM {raw_schema}.EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = %(table)s
          AND IS_ARRAY(payload) AND ARRAY_SIZE(payload) > 0
          AND IS_OBJECT(f.value)
    """
    total, populated = cursor.execute(query, {"table": table}).fetchone()
    if total == 0:
        return 0.0
    return populated / total

# Discover inner fields for a given JSON path (used for expanding nested objects) nested columns. 
# For example, if we have a top-level field "type" that is an OBJECT, we might want to discover "type.id", "type.name", etc.
def _discover_inner_fields(cursor, raw_schema: str, table: str, json_path: str) -> list[tuple[str, str]]:
    """Discover fields at json_path. Accepts dotted paths like 'bed_type.type'."""
    sql_path = ':'.join(f'"{p}"' for p in json_path.split('.'))

    discover_inner_query = f"""
    SELECT
        obj.key AS field_name,
        TYPEOF(obj.value) AS field_type
        FROM {raw_schema}.EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f,
        LATERAL FLATTEN(input => f.value:{sql_path}) obj
        WHERE source_table = %(table)s
        AND ARRAY_SIZE(payload) > 0
        AND IS_OBJECT(f.value:{sql_path})
        AND IS_ARRAY(payload)
        GROUP BY 1, 2;
    """
    rows = cursor.execute(discover_inner_query, {"table": table}).fetchall()

    type_by_field = defaultdict(list)
    for name, dtype in rows:
        type_by_field[name.replace('-', '_')].append(dtype)

    return [(name, resolve_type(types)) for name, types in type_by_field.items()]

# MAIN LOGIC
# I have MFA enabled on my Snowflake account, so the connection flow is a bit more complex. 
# This function tries the push notification method first, and if it fails (e.g. due to timeout), it falls back to asking for a passcode. 
# You can adapt this logic based on your specific MFA setup (e.g. if you only use passcodes, you can simplify this).
# got the code here: https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#:~:text=Using%20multi%2Dfactor%20authentication%20(MFA)%C2%B6
def connect_with_mfa_push():
    """
    Example 1: MFA with Push Notification (Duo Push)

    When no passcode is provided, Snowflake sends a push notification
    to your registered device. You need to approve it to complete login.
    """
    #print("Connecting with MFA push notification...")
    #print("Please approve the push notification on your device.")

    try: 
        return _snowflake_connect() 
    except Exception as e: 
        if 'MFA' in str(e) or 'passcode' in str(e).lower():
            print(f"MFA push timed out. Falling back to passcode.")
            passcode = input("Enter MFA passcode: ").strip()
            return _snowflake_connect(passcode=passcode)
        raise  # re-raise anything not MFA-related

# get the tables we want to process by looking at the source_table field in the raw events, and filtering to the relevant prefixes.
def get_source_tables(cursor, raw_schema: str) -> list[str]:
    tables = cursor.execute(
            f"""SELECT 
                SOURCE_TABLE, COUNT(*) 
            FROM {raw_schema}.EVENTS_RAW, 
            LATERAL FLATTEN(input => PAYLOAD) d 
            WHERE SOURCE_TABLE ILIKE ANY ('evaluation_%', 'finance_%', 'inventory_%','reception_%','inpatient_%','theatre_%') 
                AND IS_ARRAY(PAYLOAD) 
                AND ARRAY_SIZE(PAYLOAD) > 0 
            GROUP BY 1 
            ORDER BY 2 DESC""" ).fetchall()
    #listed_tables = [row[0] for row in tables]
    #print(f"Table list: {listed_tables}")
    return [row[0] for row in tables]

# Discover top-level fields and their types for a given table by inspecting the JSON payloads.
# This is the first step in our flattening process, where we identify what fields exist and what types they have,
# so we can decide how to handle them (e.g. whether to expand nested objects or keep as VARIANT).
def discover_fields(cursor, raw_schema: str, table: str) -> list[tuple[str, str]]:
    discover_query = f"""
    SELECT 
        obj.key AS field_name,
        TYPEOF(obj.value) AS field_type
        FROM {raw_schema}.EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f,
        LATERAL FLATTEN(input => f.value) obj
        WHERE source_table = %(table)s
        AND ARRAY_SIZE(payload) > 0
        AND IS_OBJECT(f.value)
        AND IS_ARRAY(payload)
        GROUP BY 1, 2;
    """

    fields = cursor.execute(discover_query, {"table": table}).fetchall()
    #2. Resolve conflicts and map to Snowflake data types
    type_by_field = defaultdict(list)
    
    for name, dtype in fields:
        type_by_field[name.replace('-', '_')].append(dtype)


    return [(name, resolve_type(types)) for name, types in type_by_field.items()]


def _add_field(result, seen_names, col, json_path, dtype):
    # Appends '_' until col is unique — prevents collisions like BED_ID (top-level) vs bed.id (expanded)
    while col.upper() in seen_names:
        col = col + '_'
    seen_names.add(col.upper())
    result.append((col, json_path, dtype))

# expand OBJECT fields into their inner fields, but only if they are sufficiently populated (e.g. >20% fill rate).
def expand_objects(cursor, raw_schema, table,
                   top_level_fields) -> list[tuple[str, str, str]]:

    result = []
    seen_names = set()

    for name, dtype in top_level_fields:
        if dtype != 'OBJECT':
            _add_field(result, seen_names, name, name, dtype)
            continue

        if check_fill_rate(cursor, raw_schema, table, name) < 0.2:
            _add_field(result, seen_names, name, name, 'VARIANT')
            continue

        inner_fields = _discover_inner_fields(cursor, raw_schema, table, name)
        for inner_name, inner_type in inner_fields:
            inner_path = f"{name}.{inner_name}"
            col_l1 = f"{name}_{inner_name}"

            if inner_type != 'OBJECT':
                _add_field(result, seen_names, col_l1, inner_path, inner_type)
                continue

            # Level 2 expansion
            inner2_fields = _discover_inner_fields(cursor, raw_schema, table, inner_path)
            if not inner2_fields:
                _add_field(result, seen_names, col_l1, inner_path, 'VARIANT')
                continue

            for inner2_name, inner2_type in inner2_fields:
                col_l2 = f"{name}_{inner_name}_{inner2_name}"
                _add_field(result, seen_names, col_l2, f"{inner_path}.{inner2_name}", inner2_type)

    return result


# Build the final SQL statement to create or replace the flattened view for a given table, based on the discovered and expanded fields. 
# This function constructs a dynamic SELECT list that extracts the appropriate JSON paths and casts them to the inferred types
def build_flatten_sql(raw_schema: str, clean_schema: str, table: str, expanded_fields: list[tuple[str, str, str]]) -> str:
    # 3. Build the dynamic SELECT list
    select_parts = []
    for name, json_path, dtype in expanded_fields:
        path_parts = json_path.split('.')
        sql_path = ':'.join(f'"{p}"' for p in path_parts)  # "id" or "type":"name"

        if dtype in ('ARRAY', 'OBJECT', 'VARIANT'):
            select_parts.append(f'record:{sql_path} AS {name.upper()}')
        else:
            select_parts.append(
                f'TRY_CAST(record:{sql_path}::STRING AS {infer_type(dtype)}) AS {name.upper()}'
            )
    select_list = ",\n".join(select_parts)

    # 4. Construct the final CREATE OR REPLACE TABLE statement
    create_table_query = f"""
    CREATE OR REPLACE VIEW {clean_schema}.{table} AS
    WITH deduped AS (
        SELECT 
            DISTINCT facility_id,
            f.value as record
        FROM {raw_schema}.EVENTS_RAW,
        LATERAL FLATTEN(input => payload) f
        WHERE source_table = '{table}'
        AND ARRAY_SIZE(payload) > 0
        AND IS_ARRAY(payload)   
        AND IS_OBJECT(f.value)
    )
    SELECT
        facility_id as source_schema, 
        {select_list}
    FROM deduped;
    """
    return create_table_query

# Main function to orchestrate the flattening process for all tables in all schema pairs. 
# It iterates through each raw/clean schema pair, discovers tables, fields, expands objects, builds SQL, and executes it to create the flattened views. 
# It also keeps track of successes and failures for reporting.
def flatten_all(conn):
    cursor = conn.cursor()
    total_ok, total_fail = 0, 0
    try:
        for raw_schema, clean_schema in SCHEMA_PAIRS:
            print(f"\n=== {raw_schema} → {clean_schema} ===")
            tables = get_source_tables(cursor, raw_schema)
            print(f"  Found {len(tables)} tables to process")
            ok, fail = 0, 0
            for table in tables:
                try:
                    fields = discover_fields(cursor, raw_schema, table)
                    expanded = expand_objects(cursor, raw_schema, table, fields)
                    sql = build_flatten_sql(raw_schema, clean_schema, table, expanded)
                    cursor.execute(sql)
                    print(f"  ✓ {table} ({len(expanded)} cols)")
                    ok += 1
                except Exception as e:
                    print(f"  ✗ {table}: {e}")
                    fail += 1
            print(f"  {raw_schema} done: {ok} ok, {fail} failed")
            total_ok += ok
            total_fail += fail
        print(f"\n=== TOTAL: {total_ok} ok, {total_fail} failed ===")
    finally:
        cursor.close()

# When this script is run, it will execute the main function which connects to Snowflake, 
# processes all tables, and creates the flattened views and closes the connection.
def main():
    conn = None
    try:
        conn = connect_with_mfa_push()
        flatten_all(conn)
    finally:
        if conn:
            conn.close()    
        
# TEST FUNCTION 
def test_expand_objects():
    conn = connect_with_mfa_push()
    try:
        cursor = conn.cursor()
        
        test_top_level = [
            ("id", "INTEGER"),
            ("name", "VARCHAR"),
            ("type", "OBJECT"),              # should expand into type_id, type_name, etc.
            ("consumable_type_id", "INTEGER"),
        ]
        
        result = expand_objects(
            cursor=cursor,
            raw_schema="KISUMU_RAW",
            table="evaluation_consumables",
            top_level_fields=test_top_level
        )
        
        print(f"\nGot {len(result)} expanded fields:\n")
        print(f"{'column_name':<30} {'json_path':<30} {'dtype'}")
        print("-" * 75)
        for col_name, json_path, dtype in result:
            print(f"{col_name:<30} {json_path:<30} {dtype}")
        
        cursor.close()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
    #test_expand_objects()