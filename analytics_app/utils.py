"""
Utility helpers for loading data from relational databases and Snowflake,
and for aggregating DataFrames.

These are low-level building blocks used by dashboard modules that need to
pull data directly from a database and reshape it before visualisation.
Non-technical summary: think of these as "data-fetching recipes" — each
function connects to a database, runs a query, and hands back a table of
results that the dashboard can then chart or analyse.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_postgres_table(conn_str: str, table: str) -> pd.DataFrame:
    """Load every row of a PostgreSQL table into a DataFrame.

    Connects to the database identified by ``conn_str``, runs
    ``SELECT * FROM <table>``, and returns the results as a pandas DataFrame
    so callers can filter, aggregate, or chart the data immediately.

    Non-technical explanation:
        Imagine opening a filing cabinet labelled ``table`` inside the
        PostgreSQL office (``conn_str``), photocopying every document inside,
        and bringing the copies back as a spreadsheet you can work with.

    Args:
        conn_str: A SQLAlchemy-style connection URL, e.g.
            ``"postgresql+psycopg2://user:pass@localhost:5432/mydb"``.
        table: The unqualified table name to query, e.g. ``"patients"``.
            No schema prefix is needed if the table is in the default schema.

    Returns:
        A :class:`pandas.DataFrame` containing all rows and columns from
        the specified table.  The column names match the database column
        names exactly.

    Example:
        >>> df = load_postgres_table("postgresql://...", "revenue_summary")
        >>> df.head()
           facility_id  total_revenue  period
        0            1      450000.00  2024-Q1
        1            2      320000.00  2024-Q1
    """
    engine = create_engine(conn_str)
    return pd.read_sql(f"SELECT * FROM {table}", engine)


def load_mysql_table(conn_str: str, table: str) -> pd.DataFrame:
    """Load every row of a MySQL table into a DataFrame.

    Identical behaviour to :func:`load_postgres_table` but targets a MySQL
    (or MariaDB) database.  The ``conn_str`` must use a MySQL dialect.

    Non-technical explanation:
        Same as the PostgreSQL version above — opens a different kind of
        database cabinet (MySQL instead of PostgreSQL) and brings back all
        the records as a usable spreadsheet.

    Args:
        conn_str: A SQLAlchemy-style MySQL connection URL, e.g.
            ``"mysql+pymysql://user:pass@localhost:3306/mydb"``.
        table: The table name to read, e.g. ``"dispensing_records"``.

    Returns:
        A :class:`pandas.DataFrame` with all rows from the table.

    Example:
        >>> df = load_mysql_table("mysql+pymysql://...", "stock_levels")
        >>> len(df)
        1200
    """
    engine = create_engine(conn_str)
    return pd.read_sql(f"SELECT * FROM {table}", engine)


def load_snowflake_table(
    user: str,
    password: str,
    account: str,
    warehouse: str,
    database: str,
    schema: str,
    table: str,
) -> pd.DataFrame:
    """Load every row of a Snowflake table into a DataFrame using password auth.

    Opens a direct Snowflake connection using username/password credentials,
    reads the requested table in full, then closes the connection before
    returning results.

    Non-technical explanation:
        Snowflake is a cloud-based data warehouse — think of it as a very
        large, powerful spreadsheet system living in the cloud.  This function
        logs in with your username and password, opens the right folder
        (database → schema → table), copies everything out into a local
        spreadsheet, then logs out cleanly.

    Note:
        For production use, prefer the :class:`warehouse.services.snowflake.SnowflakeClient`
        which supports private-key authentication, validates SQL for safety,
        and adds automatic LIMIT guardrails.  This helper is a quick
        convenience loader for dashboard modules.

    Args:
        user: Snowflake username, e.g. ``"analytics_user"``.
        password: Corresponding password (keep this in environment variables,
            never hard-code it).
        account: Snowflake account identifier, e.g. ``"xy12345.us-east-1"``.
        warehouse: The compute warehouse to use, e.g. ``"ANALYTICS_WH"``.
        database: The Snowflake database name, e.g. ``"AFYA_PROD"``.
        schema: The schema inside the database, e.g. ``"PUBLIC"``.
        table: The table name to read, e.g. ``"PATIENT_VISITS"``.

    Returns:
        A :class:`pandas.DataFrame` containing all rows and columns from
        the specified Snowflake table.

    Example:
        >>> df = load_snowflake_table(
        ...     user="analytics_user",
        ...     password="...",
        ...     account="xy12345.us-east-1",
        ...     warehouse="ANALYTICS_WH",
        ...     database="AFYA_PROD",
        ...     schema="PUBLIC",
        ...     table="PATIENT_VISITS",
        ... )
        >>> df.columns.tolist()
        ['VISIT_ID', 'PATIENT_ID', 'VISIT_DATE', 'DIAGNOSIS']
    """
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


def aggregate_data(df: pd.DataFrame, group_cols: list, agg_map: dict) -> pd.DataFrame:
    """Group a DataFrame and apply aggregation functions to produce summary figures.

    Non-technical explanation:
        Imagine you have a long list of individual sales receipts.  This
        function lets you say "group all receipts by store and month, then
        add up the totals and count the transactions" — and it hands back a
        tidy summary table.  The ``group_cols`` are the things you want to
        group by (e.g. store name, month), and ``agg_map`` is the set of
        calculations to run on each group (e.g. sum, mean, count).

    Args:
        df: The source DataFrame to group.  Every column referenced in
            ``group_cols`` and ``agg_map`` must exist.
        group_cols: Column name(s) to group by, e.g.
            ``["facility_name", "month"]``.
        agg_map: A mapping of ``{column_name: aggregation_function}``, e.g.
            ``{"revenue": "sum", "transactions": "count", "avg_cost": "mean"}``.
            Any aggregation accepted by :meth:`pandas.DataFrame.agg` is valid.

    Returns:
        A new :class:`pandas.DataFrame` with one row per unique combination
        of ``group_cols`` values and one column per entry in ``agg_map``.
        The index is reset so the result is a clean, flat table.

    Example:
        >>> import pandas as pd
        >>> sales = pd.DataFrame({
        ...     "store": ["A", "A", "B", "B"],
        ...     "revenue": [100, 200, 150, 250],
        ... })
        >>> aggregate_data(sales, group_cols=["store"], agg_map={"revenue": "sum"})
          store  revenue
        0     A      300
        1     B      400
    """
    return df.groupby(group_cols).agg(agg_map).reset_index()
