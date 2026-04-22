import snowflake.connector
import pandas as pd
import os

class SnowflakeClient:

    def __init__(self):
        self.conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER").strip(),
            password=os.getenv("SNOWFLAKE_PASSWORD").strip(),
            account=os.getenv("SNOWFLAKE_ACCOUNT").strip(),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE").strip(),
            database=os.getenv("SNOWFLAKE_DATABASE").strip(),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC").strip(),
        )

    def query(self, sql):
        return pd.read_sql(sql, self.conn)