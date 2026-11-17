"""Glue between the Django models and the framework-free engine."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

from django.db import close_old_connections, transaction
from django.utils import timezone

from . import crypto, jobs
from .engine import connectors, document, mapper, snapshot as snap_mod, sync
from .models import Connection, Job, Mapping, Snapshot, Source

logger = logging.getLogger(__name__)


# ─── connectors ───────────────────────────────────────────────────────────

def connector_for(source: Source):
    config = {**(source.config or {}), **crypto.decrypt(source.secrets)}
    return connectors.create(source.kind, config, files=source.file_paths(), workdir=source.workdir())


def split_config(kind: str, values: dict, existing_secrets: dict | None = None) -> tuple[dict, dict]:
    """Form values → (plain config, secrets). Blank secret fields keep the stored value."""
    spec = connectors.get_spec(kind)
    secret_names = spec.secret_fields
    known = {f.name for f in spec.fields}
    plain, secrets = {}, dict(existing_secrets or {})
    for k, v in (values or {}).items():
        if k not in known:
            continue
        if k in secret_names:
            if v not in (None, ""):
                secrets[k] = v
            elif v == "" and values.get(f"__clear_{k}"):
                secrets.pop(k, None)
        else:
            plain[k] = v
    return plain, secrets


# ─── serializers ──────────────────────────────────────────────────────────

def source_dict(s: Source, *, detail: bool = False) -> dict:
    spec = connectors.REGISTRY.get(s.kind, (None,))[0]
    secrets = crypto.decrypt(s.secrets)
    latest = s.latest_snapshot
    d = {
        "id": s.id, "name": s.name, "kind": s.kind, "description": s.description, "status": s.status,
        "label": spec.label if spec else s.kind, "icon": spec.icon if spec else "plug", "color": spec.color if spec else "#64748b",
        "category": spec.category if spec else "", "roles": list(spec.roles) if spec else [],
        "last_tested_at": s.last_tested_at, "last_error": s.last_error, "last_test": s.last_test,
        "created_at": s.created_at, "updated_at": s.updated_at,
        "snapshot": snapshot_meta(latest) if latest else None,
        "snapshots": s.snapshots.count(),
        "files": [{"id": f.id, "name": f.original_name, "size": f.size, "uploaded_at": f.uploaded_at} for f in s.files.all()],
        "mappings": s.mappings_as_source.count() + s.mappings_as_target.count(),
    }
    if detail:
        d["config"] = s.config
        d["secrets_set"] = sorted(k for k, v in secrets.items() if v)
    return d


def snapshot_meta(snap: Snapshot) -> dict:
    return {"id": snap.id, "created_at": snap.created_at, "stats": snap.stats, "fingerprint": snap.fingerprint,
            "diff": {k: v for k, v in (snap.diff or {}).items() if k in ("summary", "first", "added_tables", "removed_tables")}
            | {"changed": len((snap.diff or {}).get("changed_tables", []))},
            "duration_ms": snap.duration_ms, "dialect": snap.data.get("dialect"), "extractor": snap.data.get("extractor"),
            "warnings": (snap.data.get("warnings") or [])[:20]}


def mapping_dict(m: Mapping, *, detail: bool = False) -> dict:
    d = {
        "id": m.id, "name": m.name, "status": m.status, "version": m.version,
        "source": {"id": m.source_id, "name": m.source.name, "kind": m.source.kind},
        "target": {"id": m.target_id, "name": m.target.name, "kind": m.target.kind},
        "source_snapshot": m.source_snapshot_id, "target_snapshot": m.target_snapshot_id,
        "summary": {k: v for k, v in (m.summary or {}).items() if k != "tables"},
        "settings": m.settings, "created_at": m.created_at, "updated_at": m.updated_at,
        "usage": (m.document or {}).get("usage"),
        "connections": m.connections.count(),
        "running_job": _running_job_id(mapping=m),
    }
    if detail:
        d["document"] = m.document
        d["table_summary"] = (m.summary or {}).get("tables", [])
    return d


def connection_dict(c: Connection) -> dict:
    last = c.jobs.order_by("-created_at").first()
    return {
        "id": c.id, "name": c.name, "mode": c.mode, "mode_label": dict(Connection.MODES).get(c.mode, c.mode),
        "schedule_minutes": c.schedule_minutes, "enabled": c.enabled, "next_run_at": c.next_run_at,
        "settings": c.settings, "state": c.state, "last_status": c.last_status, "last_run_at": c.last_run_at,
        "last_rows": c.last_rows, "created_at": c.created_at,
        "mapping": {"id": c.mapping_id, "name": c.mapping.name, "source": c.mapping.source.name, "source_kind": c.mapping.source.kind},
        "destination": {"id": c.destination_id, "name": c.destination.name, "kind": c.destination.kind},
        "last_job": job_dict(last) if last else None,
        "running_job": _running_job_id(connection=c),
    }


def job_dict(j: Job, *, since: int | None = None) -> dict:
    d = {
        "id": j.id, "kind": j.kind, "kind_label": dict(Job.KINDS).get(j.kind, j.kind), "status": "stalled" if j.stalled else j.status,
        "title": j.title, "progress": j.progress, "phase": j.phase, "message": j.message, "error": (j.error or "").split("\n\n")[0],
        "created_at": j.created_at, "started_at": j.started_at, "finished_at": j.finished_at, "duration": j.duration,
        "source": j.source_id, "mapping": j.mapping_id, "connection": j.connection_id, "trigger": j.trigger,
        "log_size": len(j.log or []), "user": j.created_by.get_username() if j.created_by_id else None,
    }
    if since is not None:
        d["log"] = (j.log or [])[since:]
        d["result"] = j.result
    return d


def _running_job_id(**kw) -> int | None:
    j = Job.objects.filter(status__in=("queued", "running"), **kw).order_by("-created_at").first()
    return j.id if j and not j.stalled else None


# ─── jobs ─────────────────────────────────────────────────────────────────

def _new_job(kind: str, title: str, user=None, **links) -> Job:
    return Job.objects.create(kind=kind, title=title, created_by=user if getattr(user, "is_authenticated", False) else None, **links)


def run_test(source: Source, user=None) -> Job:
    job = _new_job("test", f"Test {source.name}", user, source=source)

    def work(ctx):
        ctx(pct=10, msg=f"Connecting to {source.name}…", phase="connect")
        conn = connector_for(source)
        try:
            res = conn.test()
        except connectors.ConnectorError as e:
            Source.objects.filter(pk=source.pk).update(status="error", last_error=str(e), last_tested_at=timezone.now(),
                                                       last_test={"ok": False, "message": str(e)})
            raise
        finally:
            conn.close()
        res = {k: v for k, v in res.items() if k != "preview"} | ({"preview": res["preview"][:3]} if res.get("preview") else {})
        Source.objects.filter(pk=source.pk).update(status="ok", last_error="", last_tested_at=timezone.now(), last_test=res)
        ctx(pct=100, msg=res.get("message") or "Connected.")
        return res

    return jobs.start(job, work)


def run_extract(source: Source, options: dict | None = None, user=None) -> Job:
    job = _new_job("extract", f"Extract schema · {source.name}", user, source=source)

    def work(ctx):
        t0 = time.time()
        ctx(pct=2, msg=f"Extracting the schema of {source.name} with dbatlas…", phase="connect")
        conn = connector_for(source)
        try:
            data = conn.discover(options or {}, progress=lambda **k: (ctx.check(), ctx(**k)))
        except connectors.ConnectorError as e:
            Source.objects.filter(pk=source.pk).update(status="error", last_error=str(e))
            raise
        finally:
            conn.close()
        prev = source.latest_snapshot
        diff = snap_mod.diff(prev.data if prev else None, data)
        snap = Snapshot.objects.create(source=source, data=data, fingerprint=data.get("fingerprint", ""), stats=data.get("stats", {}),
                                       diff=diff, options=options or {}, duration_ms=int((time.time() - t0) * 1000))
        Source.objects.filter(pk=source.pk).update(status="ok", last_error="", updated_at=timezone.now())
        s = data["stats"]
        ctx(pct=100, msg=f"Snapshot #{snap.id}: {s['tables']} tables · {s['columns']} columns · {s['relationships']} relationships"
            + (f" · drift: {diff['summary']}" if prev else ""))
        return {"snapshot": snap.id, "stats": s, "diff": diff.get("summary")}

    return jobs.start(job, work)


def _ensure_snapshot(source: Source, ctx, label: str, span: tuple[float, float]) -> Snapshot:
    snap = source.latest_snapshot
    if snap:
        return snap
    ctx(msg=f"No snapshot of the {label} yet — extracting {source.name} first…", phase="extract")
    conn = connector_for(source)
    lo, hi = span
    try:
        data = conn.discover({}, progress=lambda pct=None, **k: ctx(pct=lo + (hi - lo) * (pct or 0) / 100 if pct is not None else None, **k))
    finally:
        conn.close()
    return Snapshot.objects.create(source=source, data=data, fingerprint=data.get("fingerprint", ""), stats=data.get("stats", {}),
                                   diff=snap_mod.diff(None, data))


def refresh_summary(m: Mapping) -> None:
    target = m.target_snapshot.data if m.target_snapshot else None
    m.summary = document.summary(m.document or {}, target)


def merge_human_decisions(old: dict, new: dict, src_columns: dict[str, set[str]]) -> int:
    """Carry review work into a regenerated document: table choices and fields made by a
    human, and row filters. Only kept while they still fit the new source snapshot."""
    kept = 0
    old_pairs = {p["target"]: p for p in old.get("tables", [])}
    new_pairs = {p["target"]: p for p in new.get("tables", [])}
    restored = set()  # targets whose human-chosen source table overrides the new suggestion
    for target, op in old_pairs.items():
        np_ = new_pairs.get(target)
        if not np_:
            continue
        human_choice = op.get("made_by") == "human" and (op.get("source") is None or op.get("source") in src_columns)
        # a weaker run (AI off, AI failed) must never un-match a table the previous document had matched
        lost_match = not np_.get("source") and op.get("source") in src_columns
        if (human_choice or lost_match) and np_.get("source") != op.get("source"):
            np_.update(source=op.get("source"), made_by=op.get("made_by") or "human", confidence=op.get("confidence", 1.0))
            new["fields"] = [f for f in new["fields"] if f["target_table"] != target]
            new["unmapped_fields"] = [u for u in new.get("unmapped_fields", []) if u["target_table"] != target]
            restored.add(target)
            kept += 1
        if op.get("where") and np_.get("source") == op.get("source"):
            np_["where"] = op["where"]
    by_id = {f["id"]: i for i, f in enumerate(new["fields"])}
    for f in old.get("fields", []):
        pair = new_pairs.get(f["target_table"])
        if (f.get("made_by") != "human" and f["target_table"] not in restored) or not pair \
                or pair.get("source") != f.get("source_table"):
            continue
        if f["kind"] != "constant" and f.get("source") not in src_columns.get(f.get("source_table") or "", set()):
            continue
        if f["id"] in by_id:
            new["fields"][by_id[f["id"]]] = f
        else:
            new["fields"].append(f)
            by_id[f["id"]] = len(new["fields"]) - 1
        new["unmapped_fields"] = [u for u in new.get("unmapped_fields", [])
                                  if not (u["target_table"] == f["target_table"] and u["target"] == f["target"])]
        kept += 1
    return kept


def run_generate(m: Mapping, user=None) -> Job:
    job = _new_job("map", f"AI mapping · {m.name}", user, mapping=m)
    Mapping.objects.filter(pk=m.pk).update(status="generating")

    def work(ctx):
        try:
            # always the newest snapshot of each side (extracted now if there is none)
            src_snap = _ensure_snapshot(m.source, ctx, "source", (0, 10))
            tgt_snap = _ensure_snapshot(m.target, ctx, "target", (10, 15))
            s_data = snap_mod.subset(src_snap.data, (m.settings or {}).get("source_tables"))
            t_data = snap_mod.subset(tgt_snap.data, (m.settings or {}).get("target_tables"))
            ctx(msg=f"Source: {s_data['stats']['tables']} tables · Target: {len(t_data['tables'])} tables", pct=15)
            old = Mapping.objects.get(pk=m.pk).document or {}
            doc = mapper.generate(s_data, t_data, use_ai=(m.settings or {}).get("ai", True),
                                  samples=(m.settings or {}).get("samples", "full"),
                                  progress=lambda pct=None, **k: (ctx.check(), ctx(pct=15 + 0.85 * pct if pct is not None else None, **k)))
            src_keys = {t["key"]: {c["name"] for c in t["columns"]} for t in s_data["tables"]}
            kept = merge_human_decisions(old, doc, src_keys)
            if kept:
                ctx(msg=f"Kept {kept} decision(s) made in review.")
            had = sum(1 for p in old.get("tables", []) if p.get("source"))
            if had and not any(p.get("source") for p in doc["tables"]):
                raise RuntimeError(f"Regeneration matched no tables (AI {'on' if doc['settings'].get('ai') else 'off'}) — "
                                   f"the previous mapping ({had} tables) was kept unchanged.")
            fresh = Mapping.objects.get(pk=m.pk)
            fresh.document = doc
            fresh.source_snapshot = src_snap
            fresh.target_snapshot = tgt_snap
            fresh.status = "review"
            fresh.version += 1
            refresh_summary(fresh)
            fresh.save()
            s = fresh.summary
            failed_calls = int(doc["usage"].get("failures") or 0)
            ctx(msg=f"{s['matched_tables']}/{s['target_tables']} tables matched · {s['mapped']}/{s['columns']} columns mapped "
                    f"· {doc['usage'].get('calls', 0)} Groq calls · {doc['usage'].get('total_tokens', 0):,} tokens"
                    + (f" · {failed_calls} AI call(s) failed — those parts are rule-based only; generate again to retry"
                       if failed_calls else ""),
                level="warning" if failed_calls else "info")
            return {"summary": {k: v for k, v in s.items() if k != "tables"}, "usage": doc["usage"]}
        except Exception:
            Mapping.objects.filter(pk=m.pk).update(status="failed" if not m.document else "review")
            raise

    return jobs.start(job, work)


def run_remap(m: Mapping, target_table: str, source_table: str | None, user=None, use_ai: bool = True) -> Job:
    job = _new_job("remap", f"Re-map {target_table} · {m.name}", user, mapping=m)

    def work(ctx):
        ctx(pct=10, msg=f"{'AI ' if use_ai else ''}mapping {source_table or '∅'} → {target_table}…", phase="fields")
        fresh = Mapping.objects.get(pk=m.pk)
        doc = mapper.remap_table(fresh.document, fresh.source_snapshot.data, fresh.target_snapshot.data, target_table,
                                 source_table, use_ai=use_ai, samples=(fresh.settings or {}).get("samples", "full"))
        with transaction.atomic():
            locked = Mapping.objects.select_for_update().get(pk=m.pk)
            # splice in only this table — edits made to other tables while the AI ran must survive
            cur = dict(locked.document or {})
            new_pair = next((p for p in doc["tables"] if p["target"] == target_table), None)
            cur["tables"] = [new_pair if (p["target"] == target_table and new_pair) else p for p in cur.get("tables", [])]
            cur["fields"] = [f for f in cur.get("fields", []) if f["target_table"] != target_table] + \
                [f for f in doc["fields"] if f["target_table"] == target_table]
            cur["unmapped_fields"] = [u for u in cur.get("unmapped_fields", []) if u["target_table"] != target_table] + \
                [u for u in doc.get("unmapped_fields", []) if u["target_table"] == target_table]
            for k in ("usage", "unused_sources", "notes"):
                if k in doc:
                    cur[k] = doc[k]
            doc = cur
            locked.document = doc
            locked.version += 1
            refresh_summary(locked)
            locked.save()
        n = sum(1 for f in doc["fields"] if f["target_table"] == target_table)
        ctx(pct=100, msg=f"{target_table}: {n} columns mapped")
        return {"target": target_table, "fields": n}

    return jobs.start(job, work)


def run_sync(c: Connection, user=None, trigger: str = "manual") -> Job:
    job = _new_job("sync", f"Sync · {c.name}", user, connection=c, mapping=c.mapping)
    job.trigger = trigger
    job.save(update_fields=["trigger"])

    def work(ctx):
        m = c.mapping
        if not m.source_snapshot or not m.target_snapshot or not m.document.get("fields"):
            raise ValueError("The mapping has no reviewed fields yet — generate and review it first.")
        ctx(pct=2, msg=f"{m.source.name} → {c.destination.name} · {dict(Connection.MODES)[c.mode]}", phase="connect")
        src = connector_for(m.source)
        dst = connector_for(c.destination)
        try:
            report = sync.run(source=src, destination=dst, doc=m.document, source_snap=m.source_snapshot.data,
                              target_snap=m.target_snapshot.data, mode=c.mode, state=c.state, settings=c.settings,
                              progress=ctx, cancelled=ctx.cancelled,
                              row_limit=(c.settings or {}).get("row_limit") or None)
        except sync.SyncCancelled:
            raise jobs.Cancelled() from None
        finally:
            src.close()
            dst.close()
        state = report.pop("state", {})
        status = "failed" if report["failed"] == len(report["tables"]) and report["tables"] else ("partial" if report["failed"] else "succeeded")
        Connection.objects.filter(pk=c.pk).update(state=state, last_status=status, last_run_at=timezone.now(), last_rows=report["rows"])
        ctx(pct=100, msg=f"{report['rows']:,} rows into {len(report['tables'])} tables"
            + (f" · {report['failed']} failed" if report["failed"] else ""), level="warning" if report["failed"] else "info")
        if report["failed"] and status == "failed":
            raise RuntimeError("; ".join(f"{t['target']}: {t['error']}" for t in report["tables"] if t["error"])[:900])
        return report

    return jobs.start(job, work)


# ─── scheduler ────────────────────────────────────────────────────────────

def run_due(now=None) -> list[int]:
    """Start every enabled scheduled connection that is due. Claiming is atomic,
    so several processes (or the cron command) never double-run a sync."""
    now = now or timezone.now()
    started = []
    for c in Connection.objects.filter(enabled=True, schedule_minutes__gt=0, next_run_at__lte=now):
        nxt = now + timezone.timedelta(minutes=c.schedule_minutes)
        if Connection.objects.filter(pk=c.pk, next_run_at=c.next_run_at).update(next_run_at=nxt) != 1:
            continue
        if Job.objects.filter(connection=c, status__in=("queued", "running")).exists():
            continue
        started.append(run_sync(c, trigger="schedule").id)
    return started


_scheduler_started = False


def start_scheduler() -> None:
    global _scheduler_started
    if _scheduler_started or os.getenv("CONNECT_STUDIO_SCHEDULER", "1") != "1":
        return
    argv = " ".join(sys.argv).lower()
    if not any(k in argv for k in ("runserver", "daphne", "gunicorn", "uvicorn")):
        return
    _scheduler_started = True

    def loop():
        time.sleep(20)
        try:
            close_old_connections()
            jobs.recover_interrupted()
        except Exception:
            logger.exception("connect-studio: recovering interrupted jobs failed")
        while True:
            try:
                close_old_connections()
                run_due()
            except Exception:
                logger.exception("connect-studio scheduler tick failed")
            time.sleep(60)

    threading.Thread(target=loop, name="connect-studio-scheduler", daemon=True).start()
