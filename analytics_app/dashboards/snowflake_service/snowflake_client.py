import os
from pathlib import Path
import snowflake.connector
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


class SnowflakeClient:

    def __init__(self, schema_=None):

        with open(os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH").strip(), "rb") as key:
            private_key = key.read()

        self.conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER").strip(),
            account=os.getenv("SNOWFLAKE_ACCOUNT").strip(),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE").strip(),
            database=os.getenv("SNOWFLAKE_DATABASE").strip(),
            schema=schema_ if schema_ else os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC").strip(),
            private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH").strip(),
        )

    def query(self, sql):
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetch_pandas_all()
        finally:
            cursor.close()
