# dags/v3_to_snowflake_sync_dag.py
"""
V3 → Snowflake, kept up to date (v3_to_snowflake_sync.py — same code as the command line).

Every run, for each source (a V3 tenant login, AFYA_<KEY>_* in the server's .env) and each gateway model:
read all its rows, MERGE into HOSPITALS.V3_SYNC.<SERVICE>__<ALIAS> only the rows new or edited since the
model's watermark, flag rows V3 no longer returns as _IS_DELETED, then move the watermark. Watermarks and
the last run's counts / errors per model: HOSPITALS.V3_SYNC._SYNC_STATE.

    resolve ──► plan ──► sync_service (one mapped task per source × service) ──► check_duplicates ──► summary

A model that fails doesn't stop the others; its service task fails at the end, its watermark stays put,
and the next run re-sends the same window (the MERGE is idempotent). No Airflow retries for that reason.
A model V3 itself refuses (a 4xx on its first page, e.g. 422 "Unknown column organization_id") is
recorded as `unreadable` and is not a failure.

Sources (every facility): the `sources` param, else V3_SYNC_SOURCES, else — when that is empty or "auto" —
every production AFYA_<KEY>_* login in pipelines/.env (URL template https://{service}.afyaanalytics.ai/),
minus V3_SYNC_EXCLUDE. Add a facility = add its login to the .env; no DAG change, no restart (the .env is
re-read each run). Each login is tried on its own (a bad one is reported, the rest still sync), and logins
that land in the same organisation are collapsed so each tenant is synced exactly once.

No duplicates: one task per (tenant, service), max_active_runs=1, the MERGE keeps one row per
(_TENANT_ID, _KEY), and check_duplicates verifies that (Snowflake doesn't enforce PRIMARY KEY) after
every run, repairing any table that broke it.
`full_refresh` re-sends every row (hash-guarded) — run it after edits made straight in a V3 database,
which don't move updated_at.
"""
from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, Variable, dag, get_current_context, task

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pipelines_common
from pipelines_common import clean_list, use_pipelines_dir

log = logging.getLogger(__name__)

DAG_ID = "v3_to_snowflake_sync"
SERVICES = ["core", "reception", "evaluation", "inventory", "finance", "theatre", "inpatient", "dialysis"]


def sources_from(param) -> list[str]:
    """The `sources` param, else V3_SYNC_SOURCES (.env, then Airflow Variable); empty or "auto" = every
    production AFYA_<KEY>_* login in the .env (minus V3_SYNC_EXCLUDE) — see S.configured_sources."""
    import v3_to_snowflake_sync as S
    env = dict(os.environ)
    try:
        # the containers read pipelines/.env once at start; read it again so a login added there (or a changed
        # V3_SYNC_* setting) counts on the next run without restarting Airflow
        from dotenv import dotenv_values
        from pipelines_common import PIPELINES_DIR
        env.update({k: v for k, v in dotenv_values(Path(PIPELINES_DIR) / ".env").items()
                    if v is not None and (k.startswith("V3_SYNC_") or k.startswith("AFYA_"))})
    except Exception as e:
        log.warning("could not re-read pipelines/.env (%s) — using the environment as started", e)
    if not env.get("V3_SYNC_SOURCES"):
        env["V3_SYNC_SOURCES"] = Variable.get("V3_SYNC_SOURCES", default="") or ""
    if not env.get("V3_SYNC_EXCLUDE"):
        env["V3_SYNC_EXCLUDE"] = Variable.get("V3_SYNC_EXCLUDE", default="") or ""
    return S.configured_sources(clean_list(param), env)


def group_by_service(source: str, models, tenant_id: int | None = None) -> list[dict]:
    """One job per (source, service) — the unit a mapped task syncs."""
    by_service: dict[str, list[str]] = defaultdict(list)
    for m in models:
        by_service[m.service].append(m.name)
    return [{"source": source, "tenant_id": tenant_id, "service": s, "models": names}
            for s, names in sorted(by_service.items())]


@dag(
    dag_id=DAG_ID,
    description="V3 gateway → Snowflake HOSPITALS.V3_SYNC — watermarked upserts, deletes flagged",
    schedule="0 */3 * * *",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,          # never two runs MERGEing the same (tenant, model)
    max_active_tasks=6,         # each task reads 4 models × 4 pages at once — keep prod V3 and this box breathing
    tags=["v3", "snowflake", "incremental"],
    default_args={"owner": "data-eng", "retries": 0},
    params={
        "sources": Param([], type="array", items={"type": "string"}, title="V3 sources (AFYA_<KEY>_* logins)",
                         description="e.g. org284, gyno159. Empty = V3_SYNC_SOURCES, or when that is empty / "
                                     "'auto': every production AFYA_<KEY>_* login in the .env."),
        "models": Param([], type="array", items={"type": "string"}, title="Models (service:alias)",
                        description="e.g. reception:patient, finance:invoice. Empty = every readable model "
                                    "on the services below."),
        "services": Param(SERVICES, type="array", items={"type": "string", "enum": SERVICES}, title="Services"),
        "full_refresh": Param(False, type="boolean", title="Full refresh",
                              description="Ignore watermarks and re-send every row (unchanged rows are "
                                          "still skipped by hash)."),
        "lookback_minutes": Param(60, type="integer", minimum=0, title="Watermark overlap (minutes)"),
        "detect_deletes": Param(True, type="boolean", title="Flag rows V3 no longer returns as deleted"),
        "max_delete_fraction": Param(0.2, type="number", minimum=0, maximum=1, title="Delete guard",
                                     description="Skip flagging when more than this share of live rows "
                                                 "would be flagged in one run."),
        "workers": Param(4, type="integer", minimum=1, maximum=16, title="Models at once per service"),
        "page_workers": Param(4, type="integer", minimum=1, maximum=16, title="Pages read ahead per model"),
        "batch_size": Param(5000, type="integer", minimum=100, maximum=100_000, title="Rows per MERGE"),
        "limit": Param(0, type="integer", minimum=0, title="At most N rows per model (0 = all; 3 = canary)"),
        "dry_run": Param(False, type="boolean", title="Dry run", description="Read and count only."),
        "database": Param("HOSPITALS", type="string", title="Snowflake database"),
        "schema": Param("V3_SYNC", type="string", title="Snowflake schema"),
    },
)
def v3_to_snowflake_sync():

    @task
    def resolve() -> dict:
        """Log every source in, alone (a bad login is reported, the rest still sync), list each one's
        models, and collapse logins that land in the same organisation (one sync per tenant — no duplicates)."""
        p = get_current_context()["params"]
        use_pipelines_dir()
        sources = sources_from(p["sources"])
        if not sources:
            raise AirflowFailException("no V3 sources: no production AFYA_<KEY>_* logins in the .env and no "
                                       "`sources` / V3_SYNC_SOURCES given")
        use_pipelines_dir(sources)
        import v3_to_snowflake_sync as S
        from v3_to_v3_client import V3Client
        resolved, problems, notes = S.resolve_sources(sources)
        for n in notes:
            log.warning(n)
        for pr in problems:
            log.error("source skipped: %s", pr)
        chosen = S.parse_models(p["models"])
        jobs, synced = [], []
        for src in resolved:
            try:
                models = chosen or S.discover_models(V3Client(src.name), p["services"])
            except Exception as e:                       # e.g. one service down for this tenant
                problems.append(f"{src.name}: listing models failed: {type(e).__name__}: {str(e)[:300]}")
                continue
            jobs += group_by_service(src.name, models, src.tenant_id)
            synced.append({"source": src.name, "tenant_id": src.tenant_id, "models": len(models)})
            log.info("%s (org %s): %d model(s) on %s", src.name, src.tenant_id, len(models),
                     sorted({m.service for m in models}))
        if not jobs:
            raise AirflowFailException("no source could be planned: " + "; ".join(problems or ["no models"]))
        return {"jobs": jobs, "problems": problems, "notes": notes, "sources": synced}

    @task
    def plan(resolved: dict) -> list[dict]:
        """The mapped tasks' input (Airflow 3 maps only over a task's plain return value)."""
        return resolved["jobs"]

    @task(map_index_template="{{ job_label }}", execution_timeout=timedelta(hours=12))
    def sync_service(job: dict) -> dict:
        ctx = get_current_context()
        ctx["job_label"] = f"{job['source']} · {job['service']}"
        p = ctx["params"]
        use_pipelines_dir([job["source"]])
        import v3_to_snowflake_sync as S
        opts = S.SyncOptions(full_refresh=p["full_refresh"], lookback=timedelta(minutes=p["lookback_minutes"]),
                             batch_size=p["batch_size"], detect_deletes=p["detect_deletes"],
                             max_delete_fraction=p["max_delete_fraction"], page_workers=p["page_workers"],
                             limit=p["limit"], dry_run=p["dry_run"])
        results = S.sync(job["source"], [S.parse_model(m) for m in job["models"]], opts=opts,
                         workers=p["workers"], database=p["database"], schema=p["schema"])
        log.info("%s\n%s", ctx["job_label"], S.summarize(results))
        failed = [r.model for r in results if r.status == "failed"]
        if failed:
            raise AirflowFailException(f"{len(failed)} of {len(results)} model(s) failed: {failed} — the rest are "
                                       f"in; these keep their watermark and are retried next run")
        return {**job, "results": [r.as_dict() for r in results]}

    @task(trigger_rule="all_done")
    def check_duplicates(done) -> dict:
        """Snowflake doesn't enforce PRIMARY KEY: prove every table still has one row per (_TENANT_ID, _KEY),
        and repair any table that doesn't (keeping the most recently loaded row)."""
        p = get_current_context()["params"]
        if p["dry_run"]:
            return {}
        use_pipelines_dir()
        import v3_to_snowflake_sync as S
        with S._sink(p["database"], p["schema"]) as sink:
            found = S.check_duplicates(sink)
        log.info("duplicate check: %s", found or "every table has one row per (_TENANT_ID, _KEY)")
        return found

    @task(trigger_rule="all_done")
    def summary(planned: dict, done, duplicates) -> None:
        # Airflow 3 tasks can't query task states; a failed mapped task has no return XCom, so it's simply
        # absent from `done` — compare against the plan to name it.
        if planned is None:
            raise AirflowFailException("plan failed — nothing was synced")
        jobs, problems, notes = planned["jobs"], planned["problems"], planned["notes"]
        done = [d for d in (done or []) if d]
        reported = {(d["source"], d["service"]) for d in done}
        failed = [f"{j['source']} · {j['service']}" for j in jobs if (j["source"], j["service"]) not in reported]
        use_pipelines_dir()
        import v3_to_snowflake_sync as S
        rows = [r for d in done for r in d["results"]]
        if rows:
            log.info("\n%s", S.summarize(rows))
        log.info("sources synced: %s", ", ".join(f"{s['source']} (org {s['tenant_id']})" for s in planned["sources"]))
        for n in notes:
            log.warning(n)
        refused = sorted({r["model"] for r in rows if r["status"] == "unreadable"})
        if refused:
            log.warning("V3 refuses to read %d model(s) — skipped, not failures: %s", len(refused), refused)
        if duplicates:
            log.warning("duplicate rows were found and removed: %s", duplicates)
        errors = []
        if problems:
            errors.append(f"sources not synced: {problems}")
        if failed:
            errors.append(f"service tasks that failed (see their logs; the next run retries): {failed}")
        if errors:
            raise AirflowFailException(" | ".join(errors))

    resolved = resolve()
    synced = sync_service.expand(job=plan(resolved))
    summary(resolved, synced, check_duplicates(synced))


v3_to_snowflake_sync()
