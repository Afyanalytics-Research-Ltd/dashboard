"""
v1_to_v1.py — copy what an old Collabmed V1 server recorded after it was cloned onto a new V1 server, through both
servers' data gateways.

    POST {url}/api/access/data/login  {"username", "password"}            → success.token
    POST {url}/api/access/data/point  {"table", "action": "get", ...}      → rows (names/phones come back decrypted)
    POST {url}/api/access/data/point  {"namespace", "action": "post", "data": [..]}
                                      → {"created": n, "ids": [..]}  (through the Eloquent model, so encrypted columns
                                        are encrypted again; one request is all-or-nothing)

The situation it handles (CTTM, 2026-11): the new server (cttm.collabmed.net) was cloned from the old one
(cttm.afyaanalytics.ai) on the morning of 2026-11-20 and went live on 2026-11-21. The old server kept being used for
the rest of the 20th, so those records exist only there — and the new server has since given the *same ids* to its
own, different records. So a record cannot keep its id: each is posted without one, and every reference to a record
created after the clone (a visit's patient, an investigation's visit, a result's investigation, ...) is remapped to
the new id. References to records from before the clone are the same on both sides and are left alone.

Stateless and idempotent: a source row counts as already copied when the target holds, past the clone point, a row
with the source's created_at — a time before the target went live, which the target's own records can't have — and the
same content and (remapped) references. Re-running posts nothing twice; the src→dst id map is rebuilt from the target
every time. Patients are also matched on name + date of birth / phone, so a patient the hospital re-registered on the
new server is linked, not duplicated.

Before writing, a preflight checks every other table: if the source has rows past the shared prefix in a table this
pipeline doesn't copy, it stops (that data would otherwise be silently left behind).

Result PDFs (result_files.file_path) live on the source server's disk, which the gateway can't upload. They are
downloaded from the source into output/v1_to_v1/<target>/files/ (and S3 when AWS credentials are set) for someone with
server access to drop into the target's public/evaluation/files/.

    python v1_to_v1.py cttm                        # dry run: the plan, nothing posted
    python v1_to_v1.py cttm --apply --limit 1      # canary: at most 1 new row per table
    python v1_to_v1.py cttm --apply                # everything
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("v1_to_v1")

PER_PAGE = 100                 # the gateway's cap
MIN_INTERVAL = 1.05            # seconds between requests to one server: 60 a minute is the limit
MAX_BATCH_ROWS = 25            # rows per post
MAX_BATCH_BYTES = 400_000      # ...and bytes per post (patient documents are 0.2–1.5 MB of base64 each)

R, E = "Ignite\\Reception\\Entities\\", "Ignite\\Evaluation\\Entities\\"

# Copy order: a table comes after every table it references. refs: column → parent table (a parent in this plan).
# keyed tables have no id of their own; their primary key is a reference (one row per patient / visit).
PLAN: list[dict] = [
    {"table": "reception_patients", "model": R + "Patients", "refs": {}},
    {"table": "reception_patients_nok", "model": R + "NextOfKin", "refs": {"patient": "reception_patients"},
     "key": "patient"},
    {"table": "reception_patient_documents", "model": R + "PatientDocuments", "refs": {"patient": "reception_patients"}},
    {"table": "evaluation_visits", "model": E + "Visit", "refs": {"patient": "reception_patients"}},
    {"table": "evaluation_visit_metas", "model": E + "VisitMeta", "refs": {"visit": "evaluation_visits"}, "key": "visit"},
    {"table": "evaluation_visit_destinations", "model": E + "VisitDestinations", "refs": {"visit": "evaluation_visits"}},
    {"table": "evaluation_vitals", "model": E + "Vitals", "refs": {"visit": "evaluation_visits"}},
    {"table": "evaluation_doctor_notes", "model": E + "DoctorNotes", "refs": {"visit": "evaluation_visits"}},
    {"table": "evaluation_nursing_notes", "model": E + "NursingNotes", "refs": {"visit": "evaluation_visits"}},
    {"table": "evaluation_investigations", "model": E + "Investigations", "refs": {"visit": "evaluation_visits"}},
    {"table": "evaluation_investigation_results", "model": E + "InvestigationResult",
     "refs": {"investigation": "evaluation_investigations"}},
    {"table": "result_files", "model": E + "ResultFile",
     "refs": {"visit_id": "evaluation_visits", "investigation_id": "evaluation_investigations"}},
    {"table": "evaluation_sensitivity", "model": E + "Sensitivity",
     "refs": {"visit_id": "evaluation_visits", "result_id": "evaluation_investigation_results"}},
]
PLANNED = {p["table"] for p in PLAN}

# target → servers, the moment the target went live, and each id table's clone point (the last id both servers share)
TARGETS: dict[str, dict] = {
    "cttm": {
        "source_conn": "afya_v1_cttm_source", "source_url": "https://cttm.afyaanalytics.ai",
        "target_conn": "afya_v1_cttm_target", "target_url": "https://cttm.collabmed.net",
        "golive": "2026-11-21 00:00:00",
        "clone_ids": {
            "reception_patients": 63821, "reception_patient_documents": 16856, "evaluation_visits": 106527,
            "evaluation_visit_destinations": 191200, "evaluation_vitals": 30961, "evaluation_doctor_notes": 45411,
            "evaluation_nursing_notes": 15895, "evaluation_investigations": 336893,
            "evaluation_investigation_results": 158735, "result_files": 35343, "evaluation_sensitivity": 12525,
        },
        # never copied: the gateway's own login (made on both servers after the clone) and the practice settings row
        "ignore": {"users", "roles", "role_users", "users_user_profiles", "settings_practice", "page_count",
                   "core_notifications", "core_dashboard_widgets"},
    },
}

IGNORED_COLS = {"id", "updated_at"}
FILE_COLS = {"result_files": "file_path", "evaluation_investigation_results": "file_path"}


class GatewayError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, data=None):
        super().__init__(msg)
        self.status, self.data = status, data


class V1Gateway:
    """One V1 server's data gateway: paced under its 60/min limit, re-login on 401, back-off on 429/5xx/network."""

    def __init__(self, base_url: str, username: str, password: str, name: str = "", session=None, sleep=time.sleep):
        self.base, self.username, self.password = base_url.rstrip("/"), username, password
        self.name = name or self.base
        self.s = session or requests.Session()
        self.sleep = sleep
        self.token: str | None = None
        self._last = 0.0
        self.calls = 0

    def login(self) -> None:
        r = self.s.post(f"{self.base}/api/access/data/login", json={"username": self.username, "password": self.password},
                        timeout=60)
        try:
            token = (r.json().get("success") or {}).get("token") if r.ok else None
        except ValueError:
            token = None
        if not token:
            raise GatewayError(f"{self.name}: login failed: HTTP {r.status_code} {r.text[:200]}")
        self.token = token

    def call(self, body: dict, timeout: int = 180) -> dict:
        if not self.token:
            self.login()
        waits = [10, 30, 60, 120, 300]
        relogged = False
        while True:
            pause = MIN_INTERVAL - (time.time() - self._last)
            if pause > 0:
                self.sleep(pause)
            self._last = time.time()
            self.calls += 1
            try:
                r = self.s.post(f"{self.base}/api/access/data/point", json=body, timeout=timeout,
                                headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"})
            except requests.RequestException as e:
                if body.get("action") == "post" or not waits:
                    # a post that timed out may have been saved: never resend it blindly (the next run re-matches)
                    raise GatewayError(f"{self.name} {_what(body)}: {type(e).__name__}: {e}") from e
                w = waits.pop(0)
                log.warning("  %s %s: %s — retrying in %ds", self.name, _what(body), type(e).__name__, w)
                self.sleep(w)
                continue
            if r.status_code == 429:
                self.sleep(int(r.headers.get("Retry-After") or 30) + 1)
                continue
            if r.status_code == 401 and not relogged:
                relogged = True
                self.login()
                continue
            if r.status_code in (502, 503, 504) and waits and body.get("action") != "post":
                w = waits.pop(0)
                log.warning("  %s %s: HTTP %d — retrying in %ds", self.name, _what(body), r.status_code, w)
                self.sleep(w)
                continue
            try:
                data = r.json()
            except ValueError:
                raise GatewayError(f"{self.name} {_what(body)}: HTTP {r.status_code} {r.text[:300]}")
            if r.status_code not in (200, 201) or (isinstance(data, dict) and data.get("success") is False):
                msg = data.get("message") if isinstance(data, dict) else data
                errs = data.get("errors") if isinstance(data, dict) else None
                raise GatewayError(f"{self.name} {_what(body)}: HTTP {r.status_code} {msg}"
                                   + (f" {json.dumps(errs, default=str)[:600]}" if errs else ""),
                                   status=r.status_code, data=data)
            return data

    # ── reads ───────────────────────────────────────────────────────────────────
    def describe(self, table: str) -> dict:
        return self.call({"action": "describe", "table": table})

    def tables(self) -> list[str]:
        return [t["table"] for t in self.call({"action": "tables"})["tables"] if t.get("exists")]

    def rows_after(self, table: str, after_id: int) -> list[dict]:
        """Every row with id > after_id (soft-deleted ones too), in id order."""
        out, after = [], int(after_id)
        while True:
            res = self.call({"action": "get", "table": table, "after_id": after, "per_page": PER_PAGE, "with_trashed": True})
            rows = res.get("data") or []
            out += rows
            pg = res.get("pagination") or {}
            if not rows or not pg.get("has_more_pages"):
                return out
            nxt = int(pg.get("next_after_id") or rows[-1]["id"])
            if nxt <= after:
                raise GatewayError(f"{self.name} {table}: paging stuck at id {after}")
            after = nxt

    def rows_by_ids(self, model: str, ids: list) -> list[dict]:
        out = []
        for i in range(0, len(ids), PER_PAGE):
            res = self.call({"action": "get", "namespace": model, "ids": ids[i:i + PER_PAGE], "with_trashed": True,
                             "per_page": PER_PAGE})
            out += res.get("data") or []
        return out

    # ── writes ──────────────────────────────────────────────────────────────────
    def writable(self, model: str) -> tuple[bool, list]:
        """Whether the server accepts writes to a model (MODEL_GATEWAY_WRITE_MODELS), asked with a record the server
        refuses before saving anything: 403 = not enabled (lists what is), 422 = enabled."""
        try:
            self.call({"action": "post", "namespace": model, "data": [1]})
        except GatewayError as e:
            if e.status == 403:
                return False, list((e.data or {}).get("writable") or []) if isinstance(e.data, dict) else []
            if e.status == 422:
                return True, []
            raise
        raise GatewayError(f"{self.name}: the writability probe for {model} was accepted — check the target")

    def post(self, model: str, rows: list[dict]) -> list:
        res = self.call({"action": "post", "namespace": model, "data": rows}, timeout=300)
        ids = res.get("ids")
        if not isinstance(ids, list) or len(ids) != len(rows):
            raise GatewayError(f"{self.name} post {model}: expected {len(rows)} ids, got {str(res)[:300]}")
        return ids


def _what(body: dict) -> str:
    return f'{body.get("action", "?")} {body.get("table") or body.get("namespace") or ""}'.strip()


# ── matching ────────────────────────────────────────────────────────────────────
def norm(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    s = " ".join(str(v).split())
    if re.fullmatch(r"\d{4}-\d\d-\d\d 00:00:00", s):          # a date the model hands back with a time
        s = s[:10]
    if re.fullmatch(r"-?\d+\.\d*?0+", s):                      # 36.50 == 36.5, 2.0 == 2
        s = s.rstrip("0").rstrip(".")
    return s.upper()


def name_key(p: dict) -> tuple:
    return norm(p.get("first_name")), norm(p.get("last_name"))


def phone(v) -> str:
    d = re.sub(r"\D", "", str(v or ""))
    return d[-9:] if len(d) >= 9 else ""


def same_person(a: dict, b: dict) -> bool:
    """Same first + last name and the same date of birth or phone."""
    if not a.get("first_name") or name_key(a) != name_key(b):
        return False
    dob = norm(a.get("dob"))[:10]
    return (bool(dob) and dob == norm(b.get("dob"))[:10]) or (bool(phone(a.get("mobile"))) and
                                                              phone(a.get("mobile")) == phone(b.get("mobile")))


def score(src_row: dict, dst_row: dict, mapped: dict) -> float:
    """Share of the row's columns that agree (references compared after remapping); -1 when created_at or a reference
    disagrees, which rules the pair out."""
    cols = [c for c in src_row if c not in IGNORED_COLS]
    if "created_at" in src_row and norm(src_row.get("created_at")) != norm(dst_row.get("created_at")):
        return -1
    agree = 0
    for c in cols:
        want = mapped.get(c, src_row.get(c))
        if c in mapped and norm(want) != norm(dst_row.get(c)):
            return -1
        agree += norm(want) == norm(dst_row.get(c))
    return agree / max(1, len(cols))


MATCH_MIN = 0.8                # at least this share of columns must agree for a target row to count as the copy


# ── the run ─────────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, target: str, apply: bool = False, limit: int = 0, tables: list[str] | None = None,
                 preflight: bool = True, files: bool = True, src: V1Gateway | None = None,
                 dst: V1Gateway | None = None, out_dir: Path | None = None):
        if target not in TARGETS:
            raise SystemExit(f"unknown target {target!r}; known: {sorted(TARGETS)}")
        self.target, self.cfg = target, TARGETS[target]
        self.apply, self.limit, self.preflight_on, self.files_on = apply, int(limit or 0), preflight, files
        self.only = set(tables or [])
        unknown = self.only - PLANNED
        if unknown:
            raise SystemExit(f"not in the plan: {sorted(unknown)}; planned: {[p['table'] for p in PLAN]}")
        self.src = src or self._gateway("source")
        self.dst = dst or self._gateway("target")
        self.out_dir = out_dir or Path(os.getenv("PIPELINES_DIR", Path(__file__).resolve().parent)) / "output" / \
            "v1_to_v1" / target
        self.golive = self.cfg["golive"]
        self.idmap: dict[str, dict[str, object]] = {p["table"]: {} for p in PLAN}   # table → {src id: dst id}
        self.report: dict = {"target": target, "apply": apply, "limit": self.limit, "tables": {}, "failures": [],
                             "files": {}, "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def _gateway(self, side: str) -> V1Gateway:
        conn_id, default_url = self.cfg[f"{side}_conn"], self.cfg[f"{side}_url"]
        prefix = f"{self.target.upper()}_V1_{side.upper()}_"
        url, user, pw = os.getenv(prefix + "URL"), os.getenv(prefix + "USERNAME"), os.getenv(prefix + "PASSWORD")
        if not (user and pw):
            try:                                             # Airflow: the connection's host / login / password
                from airflow.sdk import BaseHook
                c = BaseHook.get_connection(conn_id)
                url, user, pw = (c.host or url or default_url), c.login, c.password
            except Exception as e:
                raise SystemExit(f"no {side} login: set {prefix}USERNAME/PASSWORD or Airflow connection {conn_id} ({e})")
        url = url or default_url
        if not url.startswith("http"):
            url = "https://" + url
        return V1Gateway(url, user, pw, name=side)

    # preflight ─────────────────────────────────────────────────────────────────
    def preflight(self) -> dict:
        """Every table this pipeline doesn't copy must have nothing new on the source: its last source row must be a
        row the target has too (same id, same created_at / content)."""
        ignore = self.cfg.get("ignore", set())
        problems, checked = [], 0
        for t in self.src.tables():
            if t in PLANNED or t in ignore:
                continue
            sd = self.src.describe(t)
            if sd.get("paging") != "keyset" or sd.get("final_id") in (None, 0, "0"):
                continue
            sf = int(sd["final_id"])
            pk = (sd.get("primary_key") or ["id"])[0]
            if pk != "id":
                continue
            checked += 1
            dd = self.dst.describe(t)
            df = int(dd.get("final_id") or 0)
            if sf > df:
                problems.append(f"{t}: source has ids up to {sf}, target only {df}")
                continue
            s_last = (self.src.call({"action": "get", "table": t, "after_id": sf - 1, "per_page": 1,
                                     "with_trashed": True}).get("data") or [None])[0]
            d_same = (self.dst.call({"action": "get", "table": t, "after_id": sf - 1, "per_page": 1,
                                     "with_trashed": True}).get("data") or [None])[0]
            if s_last and (not d_same or d_same.get("id") != s_last.get("id") or score(s_last, d_same, {}) < MATCH_MIN):
                problems.append(f"{t}: source row {s_last.get('id')} (created {s_last.get('created_at')}) is not on "
                                f"the target")
        self.report["preflight"] = {"checked": checked, "problems": problems}
        log.info("preflight: %d other table(s) checked, %d problem(s)", checked, len(problems))
        if problems:
            raise GatewayError("the source has new data in tables this pipeline does not copy — stopping: "
                               + "; ".join(problems[:20]))
        return self.report["preflight"]

    # per table ─────────────────────────────────────────────────────────────────
    def source_rows(self, p: dict) -> list[dict]:
        t = p["table"]
        if p.get("key"):          # keyed by a parent: the rows of the parents created after the clone
            parent = p["refs"][p["key"]]
            ids = sorted(int(i) for i in self.new_source_ids.get(parent, []))
            return self.src.rows_by_ids(p["model"], ids) if ids else []
        return self.src.rows_after(t, self.cfg["clone_ids"][t])

    def target_candidates(self, p: dict, src_rows: list[dict]) -> list[dict]:
        t = p["table"]
        if p.get("key"):
            mapped = [self.idmap[p["refs"][p["key"]]].get(str(r.get(p["key"]))) for r in src_rows]
            mapped = [m for m in mapped if m is not None]
            return self.dst.rows_by_ids(p["model"], mapped) if mapped else []
        return self.dst.rows_after(t, self.cfg["clone_ids"][t])

    def remap(self, p: dict, row: dict) -> tuple[dict, list]:
        """The row's references as target ids; the list names references whose parent isn't on the target yet."""
        out, waiting = {}, []
        for col, parent in p["refs"].items():
            v = row.get(col)
            if v in (None, "", 0, "0"):
                continue
            if parent in self.cfg["clone_ids"] and int(v) <= self.cfg["clone_ids"][parent]:
                continue                                    # a record from before the clone: same id on both sides
            got = self.idmap[parent].get(str(v))
            if got is None:
                waiting.append(f"{col}={v}")
            else:
                out[col] = got
        return out, waiting

    def match(self, p: dict, src_rows: list[dict], dst_rows: list[dict]) -> dict:
        """src id → dst id for rows already on the target. Copies carry the source's pre-go-live created_at; the
        target's own rows (created after go-live) are only considered for the patient re-registration match."""
        t, key = p["table"], p.get("key") or "id"
        copies = [d for d in dst_rows if not d.get("created_at") or norm(d.get("created_at")) < norm(self.golive)]
        taken, found = set(), {}
        for s in src_rows:
            mapped, waiting = self.remap(p, s)
            if waiting:
                continue
            if p.get("key"):
                want = mapped.get(key, s.get(key))
                d = next((d for d in dst_rows if str(d.get(key)) == str(want)), None)
                if d is not None:
                    found[str(s[key])] = d[key]
                continue
            best, best_sc = None, MATCH_MIN
            for d in copies:
                if d["id"] in taken:
                    continue
                sc = score(s, d, mapped)
                if sc >= best_sc and (best is None or sc > best_sc):
                    best, best_sc = d, sc
            if best is not None:
                taken.add(best["id"])
                found[str(s["id"])] = best["id"]
        if t == "reception_patients":                 # re-registered on the target after go-live
            natives = [d for d in dst_rows if d["id"] not in taken and d not in copies]
            for s in src_rows:
                if str(s["id"]) in found:
                    continue
                hits = [d for d in natives if same_person(s, d)]
                if len(hits) == 1:
                    found[str(s["id"])] = hits[0]["id"]
                    self.report.setdefault("reregistered", []).append(
                        {"source_id": s["id"], "target_id": hits[0]["id"], "name": " ".join(name_key(s)),
                         "dob": s.get("dob")})
                    taken.add(hits[0]["id"])
        return found

    def copy_table(self, p: dict, apply: bool) -> dict:
        t, key = p["table"], p.get("key") or "id"
        src_rows = self.source_rows(p)
        self.src_rows[t] = src_rows
        self.new_source_ids[t] = [r[key] for r in src_rows if r.get(key) is not None]
        dst_rows = self.target_candidates(p, src_rows)
        self.idmap[t].update(self.match(p, src_rows, dst_rows))
        todo, waiting = [], []
        for s in src_rows:
            if str(s[key]) in self.idmap[t]:
                continue
            mapped, wait = self.remap(p, s)
            if wait:
                waiting.append({"id": s[key], "missing": wait})
                continue
            data = {c: v for c, v in s.items() if c != "id"}
            data.update(mapped)
            todo.append((s, data))
        after_go_live = [s[key] for s in src_rows if s.get("created_at") and norm(s["created_at"]) >= norm(self.golive)]
        st = {"source_new": len(src_rows), "already_on_target": len(src_rows) - len(waiting) - len(todo),
              "to_insert": len(todo), "waiting_for_parent": len(waiting), "inserted": 0, "failed": 0}
        if self.limit and len(todo) > self.limit:
            st["held_back_by_limit"] = len(todo) - self.limit
            todo = todo[:self.limit]
        if after_go_live:
            st["created_after_go_live"] = after_go_live
        if waiting:
            st["waiting_sample"] = waiting[:5]
        if todo and p["model"] in self.not_writable:
            st["blocked_not_writable"] = len(todo)
            self.report["blocked"][p["model"]] = self.report["blocked"].get(p["model"], 0) + len(todo)
            todo = []
        if apply and todo:
            self._insert(p, todo, st)
            # rebuild the map from what the target now holds (never trust response order alone)
            dst_rows = self.target_candidates(p, src_rows)
            self.idmap[t].update(self.match(p, src_rows, dst_rows))
            missing = [s for s, _ in todo if str(s[key]) not in self.idmap[t]]
            if missing:
                st["not_found_after_insert"] = [s[key] for s in missing][:20]
                self.report["failures"].append({"table": t, "error": f"{len(missing)} posted row(s) not found on the "
                                                f"target afterwards", "ids": [s[key] for s in missing][:20]})
        st["mapped"] = len(self.idmap[t])
        log.info("  %-34s source new %4d · on target %4d · insert %4d · waiting %4d · inserted %4d · failed %d",
                 t, st["source_new"], st["already_on_target"], st["to_insert"], st["waiting_for_parent"],
                 st["inserted"], st["failed"])
        return st

    def _insert(self, p: dict, todo: list[tuple[dict, dict]], st: dict) -> None:
        key = p.get("key") or "id"
        batches, cur, size = [], [], 0
        for s, data in todo:
            n = len(json.dumps(data, default=str))
            if cur and (len(cur) >= MAX_BATCH_ROWS or size + n > MAX_BATCH_BYTES):
                batches.append(cur)
                cur, size = [], 0
            cur.append((s, data))
            size += n
        if cur:
            batches.append(cur)
        for b in batches:
            try:
                ids = self.dst.post(p["model"], [d for _, d in b])
                for (s, _), new in zip(b, ids):
                    self.idmap[p["table"]][str(s[key])] = new
                st["inserted"] += len(b)
            except GatewayError as e:
                if len(b) == 1 or "Timeout" in str(e) or "ConnectionError" in str(e):
                    st["failed"] += len(b)
                    self.report["failures"].append({"table": p["table"], "ids": [s[key] for s, _ in b],
                                                    "error": str(e)[:600]})
                    log.error("  %s %s: %s", p["table"], [s[key] for s, _ in b], str(e)[:300])
                    continue
                log.warning("  %s: batch of %d refused (%s) — one at a time", p["table"], len(b), str(e)[:200])
                for s, d in b:          # all-or-nothing per request: nothing of the batch was saved
                    try:
                        new = self.dst.post(p["model"], [d])[0]
                        self.idmap[p["table"]][str(s[key])] = new
                        st["inserted"] += 1
                    except GatewayError as e1:
                        st["failed"] += 1
                        self.report["failures"].append({"table": p["table"], "ids": [s[key]], "error": str(e1)[:600]})
                        log.error("  %s %s: %s", p["table"], s[key], str(e1)[:300])

    # files ─────────────────────────────────────────────────────────────────────
    def collect_files(self, src_rows_by_table: dict) -> dict:
        """Download every file a copied row points to from the source, unless the target already serves it."""
        out = {"needed": 0, "already_on_target": 0, "downloaded": 0, "failed": [], "dir": None, "s3": None}
        paths = sorted({r.get(col) for t, col in FILE_COLS.items() for r in src_rows_by_table.get(t, [])
                        if r.get(col)})
        out["needed"] = len(paths)
        if not paths:
            return out
        fdir = self.out_dir / "files"
        out["dir"] = str(fdir)
        s3 = _s3_client()
        bucket = os.getenv("V1_TO_V1_S3_BUCKET", "collabmedbucket")
        for path in paths:
            rel = str(path).lstrip("/")
            try:
                h = requests.head(f"{self.dst.base}/{rel}", timeout=60, allow_redirects=True)
                if h.status_code == 200 and "html" not in h.headers.get("content-type", ""):
                    out["already_on_target"] += 1
                    continue
                if not self.apply:                   # dry run: only say what's missing
                    out.setdefault("missing_on_target", []).append(path)
                    continue
                g = requests.get(f"{self.src.base}/{rel}", timeout=180)
                if g.status_code != 200 or "html" in g.headers.get("content-type", ""):
                    out["failed"].append({"path": path, "error": f"source HTTP {g.status_code}"})
                    continue
                dest = fdir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(g.content)
                out["downloaded"] += 1
                if s3:
                    key = f"v1_to_v1/{self.target}/files/{rel}"
                    s3.put_object(Bucket=bucket, Key=key, Body=g.content,
                                  ContentType=g.headers.get("content-type", "application/octet-stream"))
                    out["s3"] = f"s3://{bucket}/v1_to_v1/{self.target}/files/"
            except Exception as e:                   # a file is reported, never fails the data copy
                out["failed"].append({"path": path, "error": f"{type(e).__name__}: {e}"[:300]})
        return out

    # main ──────────────────────────────────────────────────────────────────────
    def run(self) -> dict:
        log.info("V1 → V1 %s: %s → %s (%s%s)", self.target, self.src.base, self.dst.base,
                 "APPLY" if self.apply else "dry run", f", limit {self.limit}" if self.limit else "")
        if self.preflight_on:
            self.preflight()
        self.new_source_ids: dict[str, list] = {}
        self.src_rows: dict[str, list] = {}
        self.not_writable: set[str] = set()
        self.report["blocked"] = {}
        for p in PLAN:
            ok, enabled = self.dst.writable(p["model"])
            if not ok:
                self.not_writable.add(p["model"])
                self.report["writable_on_target"] = enabled
        if self.not_writable:
            log.warning("target refuses writes to %d model(s): %s", len(self.not_writable), sorted(self.not_writable))
        for p in PLAN:
            t = p["table"]
            # a table not selected is still read and matched (its ids map the selected tables' references), never written
            selected = not self.only or t in self.only
            try:
                st = self.copy_table(p, apply=self.apply and selected)
                if not selected:
                    st["not_selected"] = True
            except GatewayError as e:
                st = {"error": str(e)[:600]}
                self.report["failures"].append({"table": t, "error": str(e)[:600]})
                log.error("  %s FAILED: %s", t, str(e)[:400])
            self.report["tables"][t] = st
        if self.report["blocked"]:
            need = sorted(set(self.report.get("writable_on_target", [])) | {p["model"] for p in PLAN})
            self.report["to_enable"] = "MODEL_GATEWAY_WRITE_MODELS=" + ",".join(need)
            log.warning("blocked: %s — on %s set %s, then run again", self.report["blocked"], self.dst.base,
                        self.report["to_enable"])
        if self.files_on:
            self.report["files"] = self.collect_files({t: self.src_rows.get(t, []) for t in FILE_COLS})
        self.report["idmap"] = {t: m for t, m in self.idmap.items() if m}
        self.report["calls"] = {"source": self.src.calls, "target": self.dst.calls}
        self.report["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.save_report()
        return self.report

    def save_report(self) -> Path | None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{'apply' if self.apply else 'dry'}.json"
            path.write_text(json.dumps(self.report, indent=1, default=str))
            self.report["report_path"] = str(path)
            return path
        except OSError as e:
            log.warning("report not saved: %s", e)
            return None


def _s3_client():
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        return None
    try:
        import boto3
        return boto3.client("s3", region_name=os.getenv("AWS_REGION") or "us-east-1")
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", choices=sorted(TARGETS))
    ap.add_argument("--apply", action="store_true", help="post to the target (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="at most N new rows per table (canary)")
    ap.add_argument("--tables", nargs="*", help="only insert into these tables (others are still read to map ids)")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--no-files", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s · %(levelname)-7s · %(message)s", datefmt="%H:%M:%S")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    except ImportError:
        pass
    rep = Run(a.target, apply=a.apply, limit=a.limit, tables=a.tables, preflight=not a.no_preflight,
              files=not a.no_files).run()
    print(json.dumps({k: v for k, v in rep.items() if k != "idmap"}, indent=1, default=str))
    return 1 if rep["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
