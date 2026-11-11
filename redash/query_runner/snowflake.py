"""Patched Redash 10.1 Snowflake query runner with key-pair auth.

Bind-mounted over /app/redash/query_runner/snowflake.py by
docker-compose.redash.yaml. Stock 10.1 only supports user+password, which
Snowflake rejects for this account ("Multi-factor authentication is
required"). This version adds an optional ``private_key`` option (PEM text,
unencrypted PKCS#8, or encrypted with ``private_key_passphrase``) and makes
``password`` optional. Original file: snowflake.py.orig next to this one.
"""
try:
    import snowflake.connector
    enabled = True
except ImportError:
    enabled = False


from redash.query_runner import BaseQueryRunner, register
from redash.query_runner import (
    TYPE_STRING,
    TYPE_DATE,
    TYPE_DATETIME,
    TYPE_INTEGER,
    TYPE_FLOAT,
    TYPE_BOOLEAN,
)
from redash.utils import json_dumps, json_loads

TYPES_MAP = {
    0: TYPE_INTEGER,
    1: TYPE_FLOAT,
    2: TYPE_STRING,
    3: TYPE_DATE,
    4: TYPE_DATETIME,
    5: TYPE_STRING,
    6: TYPE_DATETIME,
    7: TYPE_DATETIME,
    8: TYPE_DATETIME,
    13: TYPE_BOOLEAN,
}


def _private_key_der(pem, passphrase=None):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    pem = pem.strip().replace("\\n", "\n")
    if "-----BEGIN" not in pem:
        pem = "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----".format(pem)
    key = serialization.load_pem_private_key(
        pem.encode(),
        password=passphrase.encode() if passphrase else None,
        backend=default_backend(),
    )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


class Snowflake(BaseQueryRunner):
    noop_query = "SELECT 1"

    @classmethod
    def configuration_schema(cls):
        return {
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "region": {"type": "string", "default": "us-west"},
                "user": {"type": "string"},
                "password": {"type": "string"},
                "private_key": {
                    "type": "string",
                    "title": "Private Key (PEM, key-pair auth - use instead of password)",
                },
                "private_key_passphrase": {"type": "string"},
                "role": {"type": "string"},
                "warehouse": {"type": "string"},
                "database": {"type": "string"},
                "host": {"type": "string"},
            },
            "order": [
                "account",
                "region",
                "user",
                "password",
                "private_key",
                "private_key_passphrase",
                "role",
                "warehouse",
                "database",
                "host",
            ],
            "required": ["user", "account", "database", "warehouse"],
            "secret": ["password", "private_key", "private_key_passphrase"],
            "extra_options": [
                "host",
                "private_key_passphrase",
                "role",
            ],
        }

    @classmethod
    def enabled(cls):
        return enabled

    @classmethod
    def determine_type(cls, data_type, scale):
        t = TYPES_MAP.get(data_type, None)
        if t == TYPE_INTEGER and scale > 0:
            return TYPE_FLOAT
        return t

    def _get_connection(self):
        region = self.configuration.get("region")
        account = self.configuration["account"]

        # for us-west we don't need to pass a region (and if we do, it fails to connect)
        if region == "us-west":
            region = None

        if self.configuration.__contains__("host"):
            host = self.configuration.get("host")
        else:
            if region:
                host = "{}.{}.snowflakecomputing.com".format(account, region)
            else:
                host = "{}.snowflakecomputing.com".format(account)

        kwargs = dict(
            user=self.configuration["user"],
            account=account,
            region=region,
            host=host,
        )
        if self.configuration.get("role"):
            kwargs["role"] = self.configuration["role"]

        private_key = self.configuration.get("private_key")
        if private_key:
            kwargs["private_key"] = _private_key_der(
                private_key, self.configuration.get("private_key_passphrase")
            )
        else:
            kwargs["password"] = self.configuration.get("password")

        return snowflake.connector.connect(**kwargs)

    def _parse_results(self, cursor):
        columns = self.fetch_columns(
            [(i[0], self.determine_type(i[1], i[5])) for i in cursor.description]
        )
        rows = [
            dict(zip((column["name"] for column in columns), row)) for row in cursor
        ]

        data = {"columns": columns, "rows": rows}
        return data

    def run_query(self, query, user):
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("USE WAREHOUSE {}".format(self.configuration["warehouse"]))
            cursor.execute("USE {}".format(self.configuration["database"]))

            cursor.execute(query)

            data = self._parse_results(cursor)
            error = None
            json_data = json_dumps(data)
        finally:
            cursor.close()
            connection.close()

        return json_data, error

    def _run_query_without_warehouse(self, query):
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("USE {}".format(self.configuration["database"]))

            cursor.execute(query)

            data = self._parse_results(cursor)
            error = None
        finally:
            cursor.close()
            connection.close()

        return data, error

    def _database_name_includes_schema(self):
        return '.' in self.configuration.get('database')

    def get_schema(self, get_stats=False):
        if self._database_name_includes_schema():
            query = "SHOW COLUMNS"
        else:
            query = "SHOW COLUMNS IN DATABASE"

        results, error = self._run_query_without_warehouse(query)

        if error is not None:
            raise Exception("Failed getting schema.")

        schema = {}
        for row in results["rows"]:
            if row["kind"] == "COLUMN":
                table_name = "{}.{}".format(row["schema_name"], row["table_name"])

                if table_name not in schema:
                    schema[table_name] = {"name": table_name, "columns": []}

                schema[table_name]["columns"].append(row["column_name"])

        return list(schema.values())


register(Snowflake)
