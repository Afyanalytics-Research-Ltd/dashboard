# dags/canary_end_to_end_dag.py
"""
Canary through the whole workflow: N records of a facility go JSON mappings →
Snowflake RAW → CLEAN → V3, each stage checked before the next starts, and
(optionally) N records of a document folder go Drive → LandingAI → Snowflake.
See CANARY_README.md.

  plan                read the mappings, resolve the facility (schema) name,
                      list the V3 tables the canary will touch
  json_to_snowflake   triggers from_json_mappings_to_snowflake (canary=N) and
                      waits: the first N records per V3 table are upserted
                      into <FAC>_RAW.EVENTS_RAW and read back
  flatten             triggers snowflake_flatten_clean for those V3 tables
                      (rebuilds their <FAC>_CLEAN views), waits
  verify_clean        the canary's records, by id, are in each CLEAN view
  migrate_to_v3       triggers snowflake_to_v3_migration (canary=N, parents
                      first) and waits: N records per table posted to V3 and
                      read back field by field
  documents           (run_documents) triggers gdrive_landingai_extract
                      (canary=N) with document_conf, in parallel with the above
  summary             every stage's canary report in one log (runs regardless)
  done                runs only when no stage failed: the run's state is the verdict

Each stage only starts when the previous one passed, so the first stage to
fail is where the run stops — its task is red (so is the triggered DAG's
failing task), the run fails, and the central failure callback sends the
failures to Sentry. Each triggered run is a normal
run of that DAG (run id canary_e2e__<this run id>), with its own logs and
report under output/canary/<pipeline>/.

Writes go to the real schemas and the real V3 org, exactly as the full runs
would: those runs afterwards skip (V3, documents) or replace (EVENTS_RAW)
what the canary wrote.

Trigger form options:
  mappings        as for from_json_mappings_to_snowflake: a file name (repo
                  root), a path, or the mappings JSON itself.
  name            facility / schema name; blank = the connection's name in the
                  extraction tool. It must also be a facility the V3
                  migration knows (V3 connection afya_v3_<name>).
  canary          records per table (default 3).
  tables          only these mapping source tables. Empty = all.
  dedup           run dedup in the flatten step (as a full run does).
  migrate         run the V3 stage (off = stop after CLEAN).
  run_documents   also run the document pipeline canary.
  document_conf   trigger conf for gdrive_landingai_extract (schema_path,
                  file_ids, name_contains, snowflake_table …); canary is added.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, get_current_context, task
from airflow.task.trigger_rule import TriggerRule

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import PIPELINES_DIR, clean_list, clean_schema, use_pipelines_dir

log = logging.getLogger(__name__)

DAG_ID = "canary_end_to_end"
JSON_DAG_ID = "from_json_mappings_to_snowflake"
FLATTEN_DAG_ID = "snowflake_flatten_clean"
V3_DAG_ID = "snowflake_to_v3_migration"
DOCUMENT_DAG_ID = "gdrive_landingai_extract"
POKE_SECONDS = 30


def _trigger(task_id: str, dag_id: str, conf_key: str) -> TriggerDagRunOperator:
    """Trigger `dag_id` with plan's conf[conf_key] and wait for it: this task
    fails when that run fails. Clearing this task re-runs the same run id."""
    return TriggerDagRunOperator(
        task_id=task_id,
        trigger_dag_id=dag_id,
        trigger_run_id=f"canary_e2e__{{{{ run_id }}}}",
        conf=f"{{{{ ti.xcom_pull(task_ids='plan')['{conf_key}'] }}}}",
        wait_for_completion=True,
        poke_interval=POKE_SECONDS,
        allowed_states=["success"],
        failed_states=["failed"],
        reset_dag_run=True,
    )


def _mappings_value(value):
    """The mappings param as the JSON DAG takes it: an object or JSON text as is,
    a relative file name resolved against the repo root (checked here, so a
    typo fails before anything runs)."""
    import from_json_mappings_to_snowflake as fj

    if isinstance(value, (dict, list)) or (isinstance(value, str) and fj.looks_like_json(value)):
        return value
    path = Path(str(value or "").strip()).expanduser()
    path = path if path.is_absolute() else PIPELINES_DIR / path
    if not path.is_file():
        raise AirflowFailException(f"Mappings file not found: {path}")
    return str(path)


@dag(
    dag_id=DAG_ID,
    description="Canary of N records through JSON → Snowflake RAW → CLEAN → V3 (+ documents), stage by stage",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    render_template_as_native_obj=True,
    tags=["canary", "snowflake", "v3", "migration"],
    default_args={"owner": "data-eng", "retries": 0},
    params={
        "mappings": Param("dags/jsons/silverwood_data_mappings.json", type=["string", "object"],
                          title="Mappings JSON",
                          description="File (relative to the repo root), an absolute path, or the mappings JSON."),
        "name": Param("", type=["null", "string"], title="Facility / schema name",
                      description="Blank = the connection's name in the extraction tool."),
        "canary": Param(3, type="integer", minimum=1, title="Canary records per table"),
        "tables": Param([], type="array", items={"type": "string"}, title="Source tables",
                        description="Only these mapping source tables (one per line). Empty = all."),
        "dedup": Param(True, type="boolean", title="Dedup in the flatten step"),
        "migrate": Param(True, type="boolean", title="Run the V3 stage",
                         description="Writes the canary records to the facility's V3 org."),
        "run_documents": Param(False, type="boolean", title="Also run the document pipeline canary"),
        "document_conf": Param({}, type="object", title="Document pipeline conf",
                               description='Trigger conf for gdrive_landingai_extract, e.g. {"schema_path": '
                                           '"dags/jsons/schema.json", "name_contains": "siaya"}.'),
    },
)
def canary_end_to_end():

    @task
    def plan() -> dict:
        """Resolve the facility and every downstream DAG's trigger conf."""
        use_pipelines_dir()
        import from_json_mappings_to_snowflake as fj

        p = get_current_context()["params"]
        mappings = _mappings_value(p["mappings"])
        try:
            spec = fj.Mappings.load(Path(mappings) if isinstance(mappings, str) and not fj.looks_like_json(mappings)
                                    else mappings)
        except fj.MappingsError as e:
            raise AirflowFailException(str(e)) from None
        wanted = clean_list(p["tables"])
        unknown = sorted(set(wanted) - set(spec.source_tables))
        if unknown:
            raise AirflowFailException(f"Not in {spec.label}: {unknown}")
        name = (p.get("name") or "").strip() or fj.resolve_name(spec, fj.ExtractionClient(spec.connection_id))
        facility = fj.slug(name)
        if not facility:
            raise AirflowFailException(f"Schema name {name!r} produces an empty slug — set 'name'")
        v3_tables = sorted({m.table for m in spec.models if not wanted or m.source_table in wanted})
        n = int(p["canary"])
        log.info("canary %d record(s) per table · facility %s · %d V3 table(s): %s",
                 n, facility, len(v3_tables), ", ".join(v3_tables))
        return {
            "facility": facility,
            "v3_tables": v3_tables,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "json_conf": {"mappings": mappings, "name": facility, "tables": wanted, "canary": n, "dry_run": False},
            "flatten_conf": {"facilities": [facility], "tables": v3_tables, "dedup": bool(p["dedup"]),
                             "trigger_v3": False},
            "v3_conf": {"facilities": [facility], "tables": v3_tables, "canary": n, "mode": "migrate",
                        "order": "by_tier"},
            "document_conf": {**(p.get("document_conf") or {}), "canary": n},
        }

    @task
    def verify_clean(plan_out: dict) -> dict:
        """The records the JSON canary landed, by id, in each CLEAN view the
        flatten step rebuilt. A record missing there was dropped by the
        flatten (or merged away by dedup) and would never reach V3."""
        use_pipelines_dir([plan_out["facility"]])
        import canary
        import flatten_jsons_schemas as fl

        ctx = get_current_context()
        latest = canary.load_latest(JSON_DAG_ID, plan_out["facility"])
        if not latest or (latest.get("finished_at") or "") < plan_out["started_at"]:
            raise AirflowFailException(f"no {JSON_DAG_ID} canary report for {plan_out['facility']} from this run "
                                       f"(output/canary/{JSON_DAG_ID}/{plan_out['facility']}_latest.json)")
        report = canary.CanaryReport(f"{DAG_ID}.clean", int(latest["size"]), facility=plan_out["facility"],
                                     run_id=latest["run_id"])
        clean = clean_schema(plan_out["facility"])
        conn = fl._snowflake_connect()
        try:
            cur = conn.cursor()
            for t in latest["targets"]:
                ids = [i for i in t.get("ids") or [] if i not in ("", "None")]
                res = report.add(canary.TargetResult(f"{clean}.{t['target'].upper()}", expected=len(ids), ids=ids))
                if t["status"] != canary.PASSED or not ids:
                    res.untested = f"not landed in RAW by the canary ({t['status']})"
                    continue
                in_list = ", ".join("'" + i.replace("'", "''") + "'" for i in ids)
                try:
                    cur.execute(f'SELECT DISTINCT "ID"::STRING FROM HOSPITALS.{clean}.{t["target"].upper()} '
                                f'WHERE "ID"::STRING IN ({in_list})')
                    found = {r[0] for r in cur.fetchall()}
                except Exception as e:
                    res.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
                    continue
                res.landed = len(found)
                res.missing = [i for i in ids if i not in found]
                if res.missing:
                    res.notes.append("missing from the CLEAN view: dropped by the flatten, or merged away "
                                     "by dedup (HOSPITALS.DEDUP.CROSSWALK)")
        finally:
            conn.close()
        return canary.finish_in_airflow(report, ctx)

    @task.short_circuit(ignore_downstream_trigger_rules=False)
    def migrate_enabled() -> bool:
        return bool(get_current_context()["params"]["migrate"])

    @task.short_circuit(ignore_downstream_trigger_rules=False)
    def documents_enabled() -> bool:
        return bool(get_current_context()["params"]["run_documents"])

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def summary(plan_out: dict) -> dict:
        """Every stage's canary report from this run, in one log — whatever
        happened, so a failed run still shows how far each stage got."""
        use_pipelines_dir()
        import canary

        stages = [(JSON_DAG_ID, plan_out["facility"]), (f"{DAG_ID}.clean", plan_out["facility"]),
                  (V3_DAG_ID, plan_out["facility"]), ("document_pipeline", None)]
        lines, out = [], {}
        for pipeline, facility in stages:
            rep = (canary.load_latest(pipeline, facility) if facility else
                   _newest(canary.CANARY_DIR / pipeline))
            if not rep or (rep.get("finished_at") or "") < plan_out["started_at"]:
                lines.append(f"{pipeline}: not run in this canary")
                continue
            out[pipeline] = rep
            lines.append(f"{pipeline}: {'PASSED' if rep['ok'] else 'FAILED'} — " + ", ".join(
                f"{t['target']} {t['status']}" for t in rep["targets"]))
        log.info("End-to-end canary (%s):\n  %s", plan_out["facility"], "\n  ".join(lines))
        return out

    @task(trigger_rule=TriggerRule.NONE_FAILED)
    def done() -> None:
        """Runs only when no stage failed (a stage switched off counts as passed),
        so this run's own state says whether the canary passed."""
        log.info("Every canary stage passed — the full runs can start.")

    planned = plan()
    to_raw = _trigger("json_to_snowflake", JSON_DAG_ID, "json_conf")
    to_clean = _trigger("flatten", FLATTEN_DAG_ID, "flatten_conf")
    clean_ok = verify_clean(planned)
    gate_v3 = migrate_enabled()
    to_v3 = _trigger("migrate_to_v3", V3_DAG_ID, "v3_conf")
    gate_docs = documents_enabled()
    docs = _trigger("documents", DOCUMENT_DAG_ID, "document_conf")
    stages = [to_raw, to_clean, clean_ok, gate_v3, to_v3, gate_docs, docs]

    planned >> to_raw >> to_clean >> clean_ok >> gate_v3 >> to_v3
    planned >> gate_docs >> docs
    stages >> summary(planned)
    [planned, *stages] >> done()


def _newest(directory: Path) -> dict | None:
    """The most recently written *_latest.json report in a pipeline's canary folder."""
    import json

    files = sorted(directory.glob("*_latest.json"), key=lambda p: p.stat().st_mtime) if directory.is_dir() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


canary_end_to_end()
