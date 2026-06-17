"""Snowflake read-only tools for LangGraph agents.

Wraps the existing SnowflakeClient so agents can query the data warehouse
without direct database access. All queries are validated as read-only
before execution.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def query_snowflake(sql: str) -> str:
    """Execute a read-only SELECT query against the Snowflake data warehouse.

    Use this to retrieve data for analysis, reporting, or decision-making.
    Only SELECT queries are permitted — INSERT, UPDATE, DELETE, and DDL are blocked.

    The warehouse is HOSPITALS, default schema is XANALIFE_CLEAN.
    Call list_available_tables first if you are unsure which tables exist.

    Args:
        sql: A valid SELECT statement.

    Returns:
        JSON array of records, or a JSON error object on failure.
    """
    from warehouse.services.snowflake import SnowflakeClient, SnowflakeQueryError
    try:
        client = SnowflakeClient()
        df = client.query(sql, max_rows=500)
        return df.to_json(orient="records", date_format="iso")
    except SnowflakeQueryError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": f"Unexpected error: {exc}"})


@tool
def list_available_tables() -> str:
    """List all tables available in the Snowflake data warehouse.

    Returns table names, schemas, and row counts so you can discover
    what data is available before writing a query.

    Returns:
        JSON array with fields: SCHEMA_NAME, TABLE_NAME, ROW_COUNT.
    """
    from warehouse.services.snowflake import SnowflakeClient
    try:
        client = SnowflakeClient()
        df = client.get_tables()
        cols = [c for c in ("SCHEMA_NAME", "TABLE_NAME", "ROW_COUNT") if c in df.columns]
        return df[cols].to_json(orient="records")
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def get_table_sample(schema: str, table: str) -> str:
    """Get 10 sample rows from a Snowflake table to understand its columns and data.

    Args:
        schema: Schema name (e.g. 'XANALIFE_CLEAN').
        table: Table name (e.g. 'PHARMACY_TRANSACTIONS').

    Returns:
        JSON array of 10 sample rows.
    """
    from warehouse.services.snowflake import SnowflakeClient
    try:
        client = SnowflakeClient()
        df = client.get_table_sample(schema, table, rows=10)
        return df.to_json(orient="records", date_format="iso")
    except Exception as exc:
        return json.dumps({"error": str(exc)})
