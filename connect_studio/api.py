"""Connect Studio JSON API (mounted at ``/connect/api/``). Superusers only.

Errors: ``{"error": {"message": "...", "code": "..."}}`` with a 4xx/5xx status.
"""

from __future__ import annotations

import functools
import json
import logging
import uuid
from collections import Counter
from pathlib import Path

from django.core.exceptions import RequestDataTooBig
from django.db import transaction
from django.db.models import Q, Sum
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone

from . import crypto, services
from .engine import connectors, document, llm, rules, sync as sync_engine
from .engine.snapshot import diff as snapshot_diff
from .models import Connection, Job, Mapping, Snapshot, Source, SourceFile

logger = logging.getLogger(__name__)
MAX_UPLOAD = 100 * 1024 * 1024  # nginx client_max_body_size is 100M — larger requests never reach Django


class ApiError(Exception):
    def __init__(self, message, status=400, code="invalid"):
        super().__init__(message)
        self.status, self.code = status, code


def _default(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def ok(data, status=200):
    return JsonResponse(data, status=status, safe=False, json_dumps_params={"default": _default})


def endpoint(*methods):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return JsonResponse({"error": {"message": "Sign in first.", "code": "auth"}}, status=401)
            if not request.user.is_superuser:
                return JsonResponse({"error": {"message": "Connect Studio is for administrators.", "code": "forbidden"}}, status=403)
            if request.method not in methods:
                return JsonResponse({"error": {"message": f"{request.method} not allowed", "code": "method"}}, status=405)
            try:
                return fn(request, *args, **kwargs)
            except ApiError as e:
                return JsonResponse({"error": {"message": str(e), "code": e.code}}, status=e.status)
            except connectors.ConnectorError as e:
                return JsonResponse({"error": {"message": str(e), "code": "connector"}}, status=400)
            except RequestDataTooBig:
                return JsonResponse({"error": {"message": "Request too large — upload the file instead of pasting it.",
                                               "code": "too_large"}}, status=413)
            except (ValueError, TypeError) as e:
                # bad query/body values (int("abc"), float(None)…) are the client's fault, not a server error
                logger.info("connect-studio bad request: %s", e)
                return JsonResponse({"error": {"message": f"Invalid value: {e}", "code": "invalid"}}, status=400)
            except Exception as e:  # pragma: no cover - surfaced to the UI
                if e.__class__.__name__ == "Http404":
                    return JsonResponse({"error": {"message": "Not found.", "code": "not_found"}}, status=404)
                logger.exception("connect-studio api error")
                return JsonResponse({"error": {"message": f"{type(e).__name__}: {e}", "code": "server"}}, status=500)
        return wrapper
    return deco


def body(request) -> dict:
    if not request.body or "json" not in (request.content_type or ""):
        return {}  # an empty / form POST is "no options", not an error
    try:
        data = json.loads(request.body)
    except ValueError:
        raise ApiError("Request body is not valid JSON.") from None
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object.")
    return data


# ─── API definition import (Postman / OpenAPI → REST connector) ───────────

@endpoint("POST")
def import_api(request):
    """Parse a Postman collection or OpenAPI/Swagger document into a REST API
    connector proposal; with ai (default on) Groq picks and names the tables."""
    from .engine import api_import

    if request.content_type and request.content_type.startswith("multipart/"):
        f = request.FILES.get("file")
        if not f:
            raise ApiError("Attach the collection / OpenAPI file as “file”.")
        if f.size > 20 * 1024 * 1024:
            raise ApiError("That file is over 20 MB — export just the API definition.")
        content, filename = f.read(), f.name
        use_ai = request.POST.get("ai", "1") not in ("0", "false", "")
        kind = request.POST.get("kind") or "rest_api"
    else:
        data = body(request)
        content, filename = data.get("text") or "", data.get("filename") or ""
        use_ai = data.get("ai", True) not in (False, 0, "0", "false")
        kind = data.get("kind") or "rest_api"
    if kind == "afya_platform":                      # Model Gateway collection → V3 connector settings
        try:
            return ok(api_import.v3_from_postman(content, filename))
        except api_import.ImportError_ as e:
            raise ApiError(str(e)) from None
    try:
        proposal = api_import.parse(content, filename)
    except api_import.ImportError_ as e:
        raise ApiError(str(e)) from None
    if use_ai:
        usage = llm.Usage()
        proposal = api_import.ai_refine(proposal, usage=usage)
        proposal["ai"]["usage"] = usage.as_dict()
    else:
        proposal["ai"] = {"used": False}
    for e in proposal["endpoints"]:
        e["line"] = api_import.stream_line(e)
    return ok(proposal)


# ─── meta / overview ──────────────────────────────────────────────────────

@endpoint("GET")
def meta(request):
    return ok({
        "connectors": connectors.catalog(),
        "llm": llm.settings(),
        "kinds": rules.KINDS,
        "steps": list(rules.STEPS),
        "sync_modes": [{"value": v, "label": label} for v, label in Connection.MODES],
        "user": request.user.get_username(),
    })


@endpoint("GET")
def overview(request):
    week = timezone.now() - timezone.timedelta(days=7)
    jobs_week = Job.objects.filter(created_at__gte=week)
    by_status = Counter(jobs_week.values_list("status", flat=True))
    by_kind = Counter(Source.objects.values_list("kind", flat=True))
    rows = sum((j.result or {}).get("rows", 0) for j in jobs_week.filter(kind="sync", status="succeeded").only("result"))
    tokens = 0
    for doc in Mapping.objects.values_list("document", flat=True):
        tokens += int(((doc or {}).get("usage") or {}).get("total_tokens") or 0)
    daily = []
    for i in range(13, -1, -1):
        day = (timezone.now() - timezone.timedelta(days=i)).date()
        qs = Job.objects.filter(created_at__date=day)
        daily.append({"day": day.isoformat(), "succeeded": qs.filter(status="succeeded").count(),
                      "failed": qs.filter(status="failed").count(), "total": qs.count()})
    spec = {s["key"]: s for s in connectors.catalog()}
    return ok({
        "counts": {
            "sources": Source.objects.exclude(status="draft").count(),
            "sources_error": Source.objects.filter(status="error").count(),
            "snapshots": Snapshot.objects.count(),
            "mappings": Mapping.objects.count(),
            "mappings_review": Mapping.objects.filter(status="review").count(),
            "connections": Connection.objects.count(),
            "connections_scheduled": Connection.objects.filter(enabled=True, schedule_minutes__gt=0).count(),
            "jobs_week": jobs_week.count(), "jobs_failed_week": by_status.get("failed", 0),
            "rows_week": rows, "ai_tokens": tokens,
        },
        "sources_by_kind": [{"kind": k, "count": n, "label": spec.get(k, {}).get("label", k), "color": spec.get(k, {}).get("color"),
                             "icon": spec.get(k, {}).get("icon")} for k, n in by_kind.most_common()],
        "daily": daily,
        "recent_jobs": [services.job_dict(j) for j in Job.objects.select_related("created_by")[:12]],
        "recent_mappings": [services.mapping_dict(m) for m in Mapping.objects.select_related("source", "target")[:6]],
        "connections": [services.connection_dict(c) for c in Connection.objects.select_related("mapping__source", "destination")[:6]],
        "llm": llm.settings(),
    })


# ─── sources ──────────────────────────────────────────────────────────────

@endpoint("GET", "POST")
def sources(request):
    if request.method == "GET":
        qs = list(Source.objects.prefetch_related("files", "snapshots"))
        role = request.GET.get("role")
        if role:
            qs = [s for s in qs if s.kind in connectors.REGISTRY and role in connectors.REGISTRY[s.kind][0].roles]
        if not request.GET.get("drafts"):
            qs = [s for s in qs if s.status != "draft"]
        return ok({"sources": [services.source_dict(s) for s in qs]})
    data = body(request)
    kind = data.get("kind")
    spec = connectors.get_spec(kind)
    name = (data.get("name") or "").strip() or spec.label
    plain, secrets = services.split_config(kind, data.get("config") or {})
    s = Source.objects.create(name=name[:200], kind=kind, description=data.get("description") or "", config=plain,
                              secrets=crypto.encrypt(secrets), status="draft", created_by=request.user)
    return ok({"source": services.source_dict(s, detail=True)}, status=201)


@endpoint("GET", "PATCH", "DELETE")
def source_detail(request, pk):
    s = get_object_or_404(Source, pk=pk)
    if request.method == "GET":
        return ok({"source": services.source_dict(s, detail=True),
                   "jobs": [services.job_dict(j) for j in s.jobs.all()[:15]],
                   "mappings": [services.mapping_dict(m) for m in Mapping.objects.filter(Q(source=s) | Q(target=s))[:20]]})
    if request.method == "DELETE":
        if Connection.objects.filter(destination=s).exists():
            raise ApiError("This source is the destination of a sync connection — delete that first.", 409, "in_use")
        for f in s.files.all():
            f.file.delete(save=False)
        s.delete()
        return ok({"deleted": pk})
    data = body(request)
    if "name" in data:
        s.name = (data["name"] or s.name)[:200]
    if "description" in data:
        s.description = data["description"] or ""
    if "config" in data:
        plain, secrets = services.split_config(s.kind, data["config"] or {}, crypto.decrypt(s.secrets))
        s.config = plain
        s.secrets = crypto.encrypt(secrets)
        if s.status == "ok":
            s.status = "untested"
    if data.get("status") in ("ok", "untested") and s.status == "draft":
        s.status = data["status"]
    s.save()
    return ok({"source": services.source_dict(s, detail=True)})


@endpoint("POST")
def source_files(request, pk):
    s = get_object_or_404(Source, pk=pk)
    spec = connectors.get_spec(s.kind)
    uploaded = request.FILES.getlist("files") or request.FILES.getlist("file")
    if not uploaded:
        raise ApiError("No files in the upload.")
    if not spec.multi_upload:
        for f in s.files.all():
            f.file.delete(save=False)
            f.delete()
        uploaded = uploaded[:1]
    added = []
    for up in uploaded:
        ext = Path(up.name).suffix.lower()
        if spec.uploads and ext not in spec.uploads:
            raise ApiError(f"{up.name}: {spec.label} accepts {', '.join(spec.uploads)}")
        if up.size > MAX_UPLOAD:
            raise ApiError(f"{up.name} is larger than {MAX_UPLOAD // (1024 * 1024)} MB.")
        sf = SourceFile(source=s, original_name=up.name[:300], size=up.size)
        sf.file.save(up.name, up, save=True)
        added.append(sf.id)
    s.status = "untested" if s.status == "ok" else s.status
    s.save(update_fields=["status", "updated_at"])
    return ok({"source": services.source_dict(s, detail=True), "added": added}, status=201)


@endpoint("DELETE")
def source_file_delete(request, pk, fid):
    sf = get_object_or_404(SourceFile, pk=fid, source_id=pk)
    sf.file.delete(save=False)
    sf.delete()
    return ok({"deleted": fid})


@endpoint("POST")
def source_test(request, pk):
    s = get_object_or_404(Source, pk=pk)
    problems = connectors.validate_config(s.kind, s.config, has_files=s.files.exists(),
                                          existing_secrets={k for k, v in crypto.decrypt(s.secrets).items() if v})
    if problems:
        raise ApiError(" ".join(problems), 400, "config")
    job = services.run_test(s, request.user)
    return ok({"job": services.job_dict(job)}, status=202)


@endpoint("POST")
def source_extract(request, pk):
    s = get_object_or_404(Source, pk=pk)
    data = body(request)
    options = {k: data[k] for k in ("profile", "sample_rows", "exact_counts", "max_tables", "schemas", "tables") if k in data}
    if Job.objects.filter(source=s, kind="extract", status__in=("queued", "running")).exists():
        raise ApiError("An extraction is already running for this source.", 409, "busy")
    job = services.run_extract(s, options, request.user)
    return ok({"job": services.job_dict(job)}, status=202)


@endpoint("GET")
def source_snapshots(request, pk):
    s = get_object_or_404(Source, pk=pk)
    return ok({"snapshots": [services.snapshot_meta(sn) for sn in s.snapshots.all()[:50]]})


@endpoint("GET")
def snapshot_detail(request, pk):
    sn = get_object_or_404(Snapshot, pk=pk)
    return ok({"snapshot": {**services.snapshot_meta(sn), "source": sn.source_id, "data": sn.data, "diff_full": sn.diff}})


@endpoint("GET")
def snapshot_compare(request, pk):
    sn = get_object_or_404(Snapshot, pk=pk)
    other = get_object_or_404(Snapshot, pk=request.GET.get("against"))
    return ok({"diff": snapshot_diff(other.data, sn.data)})


# ─── mappings ─────────────────────────────────────────────────────────────

@endpoint("GET", "POST")
def mappings(request):
    if request.method == "GET":
        qs = Mapping.objects.select_related("source", "target")
        return ok({"mappings": [services.mapping_dict(m) for m in qs]})
    data = body(request)
    src = get_object_or_404(Source, pk=data.get("source"))
    tgt = get_object_or_404(Source, pk=data.get("target"))
    if src.pk == tgt.pk and not data.get("allow_same"):
        raise ApiError("Pick different source and target.")
    if "source" not in connectors.get_spec(src.kind).roles:
        raise ApiError(f"{src.name} cannot be a mapping source.")
    settings_ = {"ai": bool(data.get("ai", True)), "samples": data.get("samples") or "full",
                 "source_tables": data.get("source_tables") or [], "target_tables": data.get("target_tables") or []}
    m = Mapping.objects.create(name=(data.get("name") or f"{src.name} → {tgt.name}")[:200], source=src, target=tgt,
                               settings=settings_, status="draft", created_by=request.user)
    job = services.run_generate(m, request.user) if data.get("generate", True) else None
    return ok({"mapping": services.mapping_dict(m), "job": services.job_dict(job) if job else None}, status=201)


@endpoint("GET", "PATCH", "DELETE")
def mapping_detail(request, pk):
    m = get_object_or_404(Mapping.objects.select_related("source", "target", "source_snapshot", "target_snapshot"), pk=pk)
    if request.method == "DELETE":
        m.delete()
        return ok({"deleted": pk})
    if request.method == "PATCH":
        data = body(request)
        if "name" in data:
            m.name = (data["name"] or m.name)[:200]
        if data.get("status") in ("review", "approved"):
            m.status = data["status"]
        if "settings" in data and isinstance(data["settings"], dict):
            m.settings = {**(m.settings or {}), **data["settings"]}
        m.save()
        return ok({"mapping": services.mapping_dict(m)})
    out = services.mapping_dict(m, detail=True)
    if request.GET.get("snapshots"):
        out["source_schema"] = m.source_snapshot.data if m.source_snapshot else None
        out["target_schema"] = m.target_snapshot.data if m.target_snapshot else None
        latest_src = m.source.latest_snapshot
        latest_tgt = m.target.latest_snapshot
        out["drift"] = {
            "source": (latest_src.diff or {}).get("summary") if latest_src and m.source_snapshot and latest_src.pk != m.source_snapshot_id else None,
            "target": (latest_tgt.diff or {}).get("summary") if latest_tgt and m.target_snapshot and latest_tgt.pk != m.target_snapshot_id else None,
        }
    return ok({"mapping": out})


@endpoint("POST")
def mapping_generate(request, pk):
    m = get_object_or_404(Mapping, pk=pk)
    if Job.objects.filter(mapping=m, kind__in=("map", "remap"), status__in=("queued", "running")).exists():
        raise ApiError("This mapping is already being generated.", 409, "busy")
    data = body(request)
    if "ai" in data or "samples" in data:
        m.settings = {**(m.settings or {}), **{k: data[k] for k in ("ai", "samples") if k in data}}
        m.save(update_fields=["settings"])
    job = services.run_generate(m, request.user)
    return ok({"job": services.job_dict(job)}, status=202)


def _field_from(data: dict, *, target_table: str, target: str, source_table: str | None) -> dict:
    f = rules.normalize({"target": target, "source": data.get("source"), "kind": data.get("kind") or "copied",
                         "rule": data.get("rule"), "transform": data.get("transform")})
    return {"id": f"{target_table}::{target}", "target_table": target_table, "target": target, "source_table": source_table,
            "source": f["source"], "kind": f["kind"], "rule": f["rule"], "transform": f["transform"],
            "confidence": 1.0, "why": data.get("why") or "set in review", "made_by": "human",
            "status": data.get("status") or "approved"}


@endpoint("PATCH")
def mapping_fields(request, pk):
    """{updates: [{id, source?, kind?, rule?, transform?, status?, why?}], create: [...], delete: [id]}"""
    data = body(request)
    with transaction.atomic():
        # lock only the mapping row: Postgres refuses FOR UPDATE across the nullable snapshot join
        m = Mapping.objects.select_for_update(of=("self",)).select_related("source_snapshot").get(pk=pk)
        doc = m.document or {}
        src_tables = {t["key"]: t for t in (m.source_snapshot.data["tables"] if m.source_snapshot else [])}
        by_id = {f["id"]: f for f in doc.get("fields", [])}
        pairs = {p["target"]: p for p in doc.get("tables", [])}
        changed = []
        for u in data.get("updates") or []:
            f = by_id.get(u.get("id"))
            if not f:
                raise ApiError(f"Unknown field {u.get('id')}", 404, "not_found")
            semantic = {k: u[k] for k in ("source", "kind", "rule", "transform") if k in u}
            if semantic:
                candidate = rules.normalize({**f, **semantic, "source_table": f["source_table"]})
                problem = rules.validate(candidate, src_tables)
                if problem:
                    raise ApiError(f"{f['target']}: {problem}", 400, "rule")
                f.update({k: candidate[k] for k in ("source", "kind", "rule", "transform")})
                f["made_by"] = "human"
                f["confidence"] = 1.0
                f.setdefault("status", "approved")
                if "status" not in u:
                    f["status"] = "approved"
            if u.get("status") in ("suggested", "approved", "rejected"):
                f["status"] = u["status"]
            if "why" in u:
                f["why"] = str(u["why"] or "")[:500]
            changed.append(f["id"])
        for c in data.get("create") or []:
            tt, tc = c.get("target_table"), c.get("target")
            pair = pairs.get(tt)
            if not pair or not pair.get("source"):
                raise ApiError(f"{tt} has no source table yet — pick one first.", 400, "no_source")
            fid = f"{tt}::{tc}"
            if fid in by_id:
                raise ApiError(f"{tt}.{tc} is already mapped — edit it instead.", 409, "exists")
            f = _field_from(c, target_table=tt, target=tc, source_table=pair["source"])
            problem = rules.validate({**f, "source_table": pair["source"]}, src_tables)
            if problem:
                raise ApiError(f"{tc}: {problem}", 400, "rule")
            doc.setdefault("fields", []).append(f)
            by_id[fid] = f
            doc["unmapped_fields"] = [x for x in doc.get("unmapped_fields", []) if not (x["target_table"] == tt and x["target"] == tc)]
            changed.append(fid)
        for fid in data.get("delete") or []:
            f = by_id.pop(fid, None)
            if f:
                doc["fields"] = [x for x in doc["fields"] if x["id"] != fid]
                doc.setdefault("unmapped_fields", []).append({"target_table": f["target_table"], "target": f["target"],
                                                              "reason": "removed in review"})
                changed.append(fid)
        m.document = doc
        m.version += 1
        if m.status == "approved" and changed:
            m.status = "review"
        services.refresh_summary(m)
        m.save()
    return ok({"version": m.version, "changed": changed, "fields": [by_id[i] for i in changed if i in by_id],
               "summary": {k: v for k, v in m.summary.items() if k != "tables"}, "table_summary": m.summary.get("tables", []),
               "unmapped_fields": m.document.get("unmapped_fields", [])})


@endpoint("POST")
def mapping_bulk(request, pk):
    """{action: approve|reject|reset, min_confidence?, table?, ids?}"""
    data = body(request)
    action = data.get("action")
    status = {"approve": "approved", "reject": "rejected", "reset": "suggested"}.get(action)
    if not status:
        raise ApiError("action must be approve, reject or reset")
    thr = float(data.get("min_confidence") or 0)
    table = data.get("table")
    ids = set(data.get("ids") or [])
    with transaction.atomic():
        m = Mapping.objects.select_for_update().get(pk=pk)
        n = 0
        for f in m.document.get("fields", []):
            if table and f["target_table"] != table:
                continue
            if ids and f["id"] not in ids:
                continue
            if action == "approve" and (f["confidence"] < thr or f["status"] == "rejected"):
                continue
            if f["status"] != status:
                f["status"] = status
                n += 1
        if action == "approve" and table is None and not ids and thr == 0:
            m.status = "approved"
        m.version += 1
        services.refresh_summary(m)
        m.save()
    return ok({"changed": n, "version": m.version, "document": m.document,
               "summary": {k: v for k, v in m.summary.items() if k != "tables"}, "table_summary": m.summary.get("tables", [])})


@endpoint("POST")
def mapping_table(request, pk):
    """Change (or clear) the source table of one target table and re-map it."""
    m = get_object_or_404(Mapping.objects.select_related("source_snapshot"), pk=pk)
    data = body(request)
    target, source_table = data.get("target"), data.get("source") or None
    if not any(p["target"] == target for p in m.document.get("tables", [])):
        raise ApiError(f"Unknown target table {target}", 404, "not_found")
    if source_table and source_table not in {t["key"] for t in m.source_snapshot.data["tables"]}:
        raise ApiError(f"Unknown source table {source_table}", 404, "not_found")
    if data.get("where") is not None:
        with transaction.atomic():
            locked = Mapping.objects.select_for_update().get(pk=pk)
            for p in locked.document["tables"]:
                if p["target"] == target:
                    p["where"] = data["where"] or None
            locked.version += 1
            locked.save(update_fields=["document", "version", "updated_at"])
        if "source" not in data:
            return ok({"version": locked.version})
    job = services.run_remap(m, target, source_table, request.user, use_ai=bool(data.get("ai", True)))
    return ok({"job": services.job_dict(job)}, status=202)


@endpoint("POST")
def mapping_preview(request, pk):
    """Run a (draft) field rule over the source column's sampled values."""
    m = get_object_or_404(Mapping.objects.select_related("source_snapshot", "source"), pk=pk)
    data = body(request)
    f = rules.normalize({"source": data.get("source"), "kind": data.get("kind") or "copied", "rule": data.get("rule"),
                         "transform": data.get("transform")})
    st = next((t for t in m.source_snapshot.data["tables"] if t["key"] == data.get("source_table")), None) if m.source_snapshot else None
    if not st:
        raise ApiError("Unknown source table.")
    col = next((c for c in st["columns"] if c["name"] == f["source"]), None)
    p = (col or {}).get("profile") or {}
    values = list(dict.fromkeys([t["v"] for t in p.get("top", [])] + list(p.get("samples", []))))[:12]
    lookups = {}
    note = None
    if f["kind"] == "looked up in another table" and f["rule"].get("table"):
        r = f["rule"]
        try:
            conn = services.connector_for(m.source)
            found = {}
            for batch in conn.read(r["table"], columns=[r["key"], r["value"]], batch_size=5000, limit=20000):
                for row in batch:
                    found[str(row.get(r["key"]))] = row.get(r["value"])
            conn.close()
            lookups[(r["table"], r["key"], r["value"])] = found
        except Exception as e:
            note = f"Lookup preview unavailable: {str(e)[:160]}"
    out = []
    for v in values:
        try:
            out.append({"in": v, "out": rules.apply(f, v, lookups)})
        except Exception as e:
            out.append({"in": v, "out": None, "error": str(e)[:120]})
    if f["kind"] == "constant":
        out = [{"in": None, "out": f["rule"].get("value")}]
    return ok({"preview": out, "describe": rules.describe(f), "note": note})


def _export(m: Mapping, request):
    flavor = request.GET.get("flavor") or ("afya" if m.source.kind == "afya_extraction" else "generic")
    conn_id = None
    if flavor == "afya":
        cid = (m.source.config or {}).get("connection_id")
        try:
            conn_id = int(cid) if cid not in (None, "") else m.source_id
        except (TypeError, ValueError):
            conn_id = m.source_id
    exported = document.export(
        m.document or {}, source_meta={"id": m.source_id, "name": m.source.name, "kind": m.source.kind},
        target_meta={"id": m.target_id, "name": m.target.name, "kind": m.target.kind},
        target=m.target_snapshot.data if m.target_snapshot else None, source=m.source_snapshot.data if m.source_snapshot else None,
        flavor=flavor, include_suggested=request.GET.get("approved_only") not in ("1", "true"), connection_id=conn_id)
    return exported, flavor


@endpoint("GET")
def mapping_export(request, pk):
    m = get_object_or_404(Mapping.objects.select_related("source", "target", "source_snapshot", "target_snapshot"), pk=pk)
    exported, flavor = _export(m, request)
    if request.GET.get("download"):
        slug = rules.slug(m.name).replace("-", "_") or f"mapping_{m.pk}"
        resp = HttpResponse(json.dumps(exported, indent=4, ensure_ascii=False, default=_default), content_type="application/json")
        resp["Content-Disposition"] = f'attachment; filename="{slug}_data_mappings.json"'
        return resp
    return ok({"flavor": flavor, "document": exported, "problems": document.validate_export(exported, flavor)})


# ─── connections (syncs) ──────────────────────────────────────────────────

def _apply_connection(c: Connection, data: dict) -> None:
    if "name" in data:
        c.name = (data["name"] or c.name or "Sync")[:200]
    if data.get("mode"):
        if data["mode"] not in sync_engine.MODES:
            raise ApiError("Unknown sync mode.")
        c.mode = data["mode"]
    if "schedule_minutes" in data:
        c.schedule_minutes = max(0, int(data["schedule_minutes"] or 0))
    if "enabled" in data:
        c.enabled = bool(data["enabled"])
    if "settings" in data and isinstance(data["settings"], dict):
        c.settings = {**(c.settings or {}), **data["settings"]}
    if data.get("reset_state"):
        c.state = {}
    c.schedule_next()


@endpoint("GET", "POST")
def connections_view(request):
    if request.method == "GET":
        qs = Connection.objects.select_related("mapping__source", "destination")
        return ok({"connections": [services.connection_dict(c) for c in qs]})
    data = body(request)
    m = get_object_or_404(Mapping, pk=data.get("mapping"))
    dest = get_object_or_404(Source, pk=data.get("destination"))
    if not connectors.REGISTRY[dest.kind][1].can_write:
        raise ApiError(f"{dest.name} ({connectors.get_spec(dest.kind).label}) cannot be a sync destination — pick a SQL database.")
    c = Connection(mapping=m, destination=dest, created_by=request.user, name=data.get("name") or f"{m.source.name} → {dest.name}")
    _apply_connection(c, data)
    c.save()
    job = services.run_sync(c, request.user) if data.get("run_now") else None
    return ok({"connection": services.connection_dict(c), "job": services.job_dict(job) if job else None}, status=201)


@endpoint("GET", "PATCH", "DELETE")
def connection_detail(request, pk):
    c = get_object_or_404(Connection.objects.select_related("mapping__source", "destination"), pk=pk)
    if request.method == "DELETE":
        c.delete()
        return ok({"deleted": pk})
    if request.method == "PATCH":
        _apply_connection(c, body(request))
        c.save()
    return ok({"connection": services.connection_dict(c), "jobs": [services.job_dict(j) for j in c.jobs.all()[:30]]})


@endpoint("POST")
def connection_run(request, pk):
    c = get_object_or_404(Connection, pk=pk)
    if Job.objects.filter(connection=c, status__in=("queued", "running")).exists():
        raise ApiError("This sync is already running.", 409, "busy")
    job = services.run_sync(c, request.user)
    return ok({"job": services.job_dict(job)}, status=202)


# ─── jobs ─────────────────────────────────────────────────────────────────

@endpoint("GET")
def jobs_view(request):
    qs = Job.objects.select_related("created_by")
    for key in ("kind", "status", "source", "mapping", "connection"):
        if request.GET.get(key):
            qs = qs.filter(**{key: request.GET[key]})
    limit = min(int(request.GET.get("limit") or 50), 200)
    offset = int(request.GET.get("offset") or 0)
    return ok({"jobs": [services.job_dict(j) for j in qs[offset:offset + limit]], "total": qs.count()})


@endpoint("GET")
def job_detail(request, pk):
    j = get_object_or_404(Job.objects.select_related("created_by"), pk=pk)
    since = int(request.GET.get("since") or 0)
    return ok({"job": services.job_dict(j, since=since)})


@endpoint("POST")
def job_cancel(request, pk):
    j = get_object_or_404(Job, pk=pk)
    if j.status in ("queued", "running"):
        Job.objects.filter(pk=pk).update(cancel_requested=True)
        if j.stalled:
            Job.objects.filter(pk=pk).update(status="cancelled", finished_at=timezone.now())
    return ok({"job": services.job_dict(Job.objects.get(pk=pk))})
