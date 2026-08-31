"""SPH Inventory Intelligence — the serving dashboard for St. Peter's Orthopaedic.

Reads pre-computed analytics tables (``output/analytics/``) and read-only
Snowflake for item names, procurement and stock. Layers:

- ``config``    — explicit, provenance-tracked business inputs.
- ``utils``     — Snowflake connectivity and facility horizons.
- ``data``      — parameterized SQL and v1+v2 stitched ingestion.
- ``dashboard`` — the Streamlit presentation layer.

Every number shown comes from a fitted model, an estimated distribution, or an
explicit documented business input — never a buried constant. Nothing is
imported at top level, so importing the package never pulls the Snowflake
connector.
"""

__version__ = "0.1.0"
