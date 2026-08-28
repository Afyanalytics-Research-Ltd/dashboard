"""Utility layer: Snowflake connectivity and facility horizons.

Kept import-light: ``snowflake_conn`` lazily imports the Snowflake connector
and ``cryptography`` so offline runs never need them installed.
"""
