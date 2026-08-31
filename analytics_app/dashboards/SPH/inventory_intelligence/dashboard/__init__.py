"""SPH inventory intelligence dashboard (Streamlit).

Presentation layer only: every number shown here is read from the engine's
analytics tables (``output/analytics/``) or from read-only Snowflake queries
anchored by the facility registry. The dashboard computes **no** thresholds,
tiers, or model outputs of its own — any color banding is derived at runtime
from quantiles of the displayed distribution and labeled as such.
"""
