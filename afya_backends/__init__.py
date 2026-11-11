"""Switchable platform backends.

* ``afya_backends.warehouse`` - Snowflake or Databend behind one
  snowflake.connector-compatible API (connect, write_pandas, Airflow hook,
  Snowflake->Databend SQL translation).
* ``afya_backends.llm``       - hosted LLM APIs (Groq/xAI/Anthropic/OpenAI
  keys) or a local Ollama server behind one factory.
* ``afya_backends.toggles``   - which of each is active (System Settings
  switches, Airflow Variables / DAG params, env overrides).

Used by Django, Airflow (mounted at /opt/airflow/lib, see
docker-compose.airflow.yaml) and CLI scripts alike, so nothing here imports
Django or Airflow at module level.
"""

__all__ = ["toggles", "warehouse", "llm"]
