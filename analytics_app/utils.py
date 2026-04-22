import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_postgres_table(conn_str, table):
    engine = create_engine(conn_str)
    return pd.read_sql(f"SELECT * FROM {table}", engine)

def load_mysql_table(conn_str, table):
    engine = create_engine(conn_str)
    return pd.read_sql(f"SELECT * FROM {table}", engine)

def load_snowflake_table(user, password, account, warehouse, database, schema, table):
    import snowflake.connector
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema,
    )
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def aggregate_data(df, group_cols, agg_map):
    return df.groupby(group_cols).agg(agg_map).reset_index()
