import os
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


class SnowflakeClient:
    def __init__(self):
        passphrase = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
        password_bytes = passphrase.encode() if passphrase else b""

        with open(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"], "rb") as key_file:
            p_key = serialization.load_pem_private_key(
                key_file.read(),
                password=password_bytes,
                backend=default_backend(),
            )

        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        self.conn = snowflake.connector.connect(
            user=os.environ["SNOWFLAKE_USER"],
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            private_key=pkb,
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
            role=os.environ["SNOWFLAKE_ROLE"],
        )