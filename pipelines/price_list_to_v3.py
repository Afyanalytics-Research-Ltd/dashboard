"""
price_list_to_v3.py — a hospital's procedure & service price list (spreadsheet) → V3 procedures + facility prices,
through the model gateway.

First use: Trans Marigold Hospital, production org 87 / facility 85 (target transmarigold_prod), from
"TMH procedure & servive list.xlsx" (sheet MASTER: #, Other Service, Department, Bill TERM, INCOME CATEGORY, Cash,
INSURANCE), converted to dags/jsons/price_lists/transmarigold_prod.json.

The org already holds a template catalogue (~14k procedures) plus most of this list from an earlier import
(2026-09-29), mostly with the insurance price missing. So the sheet is reconciled, not bulk-loaded:
  * each sheet row is matched to an existing org procedure by name (case / punctuation / "&amp;" ignored). When
    several rows or procedures share a name (ELBOW at 200 and at 1500), a procedure already carrying the row's cash
    price wins, then any priced one, then an unpriced one; each procedure is used once.
  * matched → its facility price is set to the sheet's cash / insurance price (only those two columns are sent; the
    old values go in the run report). Nothing else on the procedure changes.
  * unmatched → a new procedure (code <PREFIX>-<row #>, the sheet's department → the org's own category; a name
    another row or procedure already has gets " (<department>)" appended) + its price.
  * a sheet price of 0 / 0 means "not fixed" (implants, surgeon's fee): an existing non-zero price is kept, otherwise
    the price is posted as 0 with variable_pricing on.
New rows carry stable uuid5s of target + row #, so a re-run finds and updates them instead of duplicating.

    python price_list_to_v3.py convert "TMH procedure & servive list.xlsx" --target transmarigold_prod
    python price_list_to_v3.py load --target transmarigold_prod --dry-run      # plan only, posts nothing
    python price_list_to_v3.py load --target transmarigold_prod --limit 3      # canary: 3 changes per kind
    python price_list_to_v3.py load --target transmarigold_prod
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("price_list_to_v3")

_HERE = Path(__file__).resolve().parent
# dags/jsons is where the repo keeps tracked data (pipelines/ ignores *.json); in Airflow it is /opt/airflow/dags/jsons
DATA_DIRS = (_HERE.parent / "dags" / "jsons" / "price_lists", Path("/opt/airflow/dags/jsons/price_lists"))
NS = uuid.UUID("5d3c1b0e-6a51-4f3e-9b8e-2f1f0b7c9a11")

# target → what its price list needs; the V3 login is AFYA_<TARGET>_* / Airflow connection afya_v3_<target>
TARGETS = {
    "transmarigold_prod": {"code_prefix": "TMH", "label": "Trans Marigold Hospital price list (2026-11)"},
}
# sheet department → org category name, used only when no matched row of that department shows the org's own choice
DEPARTMENT_CATEGORY = {
    "theatre": "Theatre", "radiology": "Radiology", "general opd": "OPD", "general in patient": "Inpatient",
    "dental": "Dental", "maternity": "MATERNITY", "lab": "Lab", "pharmacy": "Pharmacy", "nbu": "NBU UNIT",
    "orthopaedic": "ORTHOPAEDIC", "nutrition": "NUTRITION", "physiotherapy": "PHYSIOTHERAPY",
    "nursing station": "Nursing", "administration": "Others", "obstetrics and gynecology": "OBS/GYN",
}
FALLBACK_CATEGORY = "Others"


def key(v) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(str(v or "")).lower()).strip()


def clean(v) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(v or ""))).strip()


def money(v) -> float:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return 0.0


def gid(target: str, kind: str, k) -> str:
    return str(uuid.uuid5(NS, f"{target}|{kind}|{k}"))


def data_path(target: str) -> Path:
    for d in DATA_DIRS:
        if (d / f"{target}.json").exists():
            return d / f"{target}.json"
    return DATA_DIRS[0] / f"{target}.json"


# ── convert ──────────────────────────────────────────────────────────────────
def convert(xlsx: str, target: str, sheet: str = "MASTER") -> Path:
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)[sheet]
    it = ws.iter_rows(values_only=True)
    head = [key(h) for h in next(it)]
    col = {k: head.index(k) for k in ("other service", "department", "bill term", "income category", "cash", "insurance")}
    rows = []
    for n, r in enumerate(it, start=2):
        name = clean(r[col["other service"]])
        if not name:
            continue
        rows.append({"row": int(r[0]) if isinstance(r[0], (int, float)) else n - 1, "name": name,
                     "department": clean(r[col["department"]]), "bill_term": clean(r[col["bill term"]]),
                     "income_category": clean(r[col["income category"]]),
                     "cash": money(r[col["cash"]]), "insurance": money(r[col["insurance"]])})
    out = DATA_DIRS[0] / f"{target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": Path(xlsx).name, "sheet": sheet, "rows": rows}, indent=1), encoding="utf-8")
    log.info("%d rows → %s", len(rows), out)
    return out


# ── load ─────────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, target: str, dry_run: bool = False, limit: int = 0, workers: int = 8):
        try:                                        # Airflow: settings from the afya_v3_<target> connection
            from pipelines_common import use_pipelines_dir
            use_pipelines_dir([target])
        except (ImportError, RuntimeError):         # command line: pipelines/.env
            from dotenv import load_dotenv
            load_dotenv(_HERE / ".env", override=False)
            sys.path.insert(0, str(_HERE))
        from v3_to_v3_client import V3Client
        if target not in TARGETS:
            raise SystemExit(f"unknown target {target}; known: {list(TARGETS)}")
        self.target, self.dry_run, self.limit, self.workers = target, dry_run, limit, workers
        self.cfg = TARGETS[target]
        self.c = V3Client(target)
        self.c.send_facility_header = True          # prices land in this facility
        self.c.token()
        self.org, self.fac = self.c.organization_id, self.c.facility_id
        if not self.org or not self.fac:
            raise SystemExit(f"{target}: organization_id / facility_id not set")
        self.rows = json.loads(data_path(target).read_text(encoding="utf-8"))["rows"]
        log.info("%r: %d sheet rows", self.c, len(self.rows))

    def read(self):
        self.cats = [c for c in self.c.read_all("evaluation", "procedure_categories") if not c.get("deleted_at")]
        self.procs = [p for p in self.c.read_all("evaluation", "procedures", workers=8) if not p.get("deleted_at")]
        self.prices = {}
        for p in self.c.read_all("evaluation", "procedure_prices", workers=8):
            if p.get("facility_id") == self.fac and not p.get("deleted_at"):
                self.prices.setdefault(p["evaluation_procedure_id"], p)
        self.facility_name = next((f.get("name") for f in self.c.read_all("core", "facilities") if f.get("id") == self.fac), None)
        log.info("org %s: %d categories, %d procedures, %d prices at facility %s (%s)", self.org, len(self.cats),
                 len(self.procs), len(self.prices), self.fac, self.facility_name)

    # which existing procedure each sheet row is
    def match(self) -> dict[int, dict]:
        by_uuid = {str(p.get("uuid")).lower(): p for p in self.procs if p.get("uuid")}
        by_name = defaultdict(list)
        for p in self.procs:
            by_name[key(p["name"])].append(p)
        got, used = {}, set()
        for r in self.rows:                                   # a procedure this loader made on an earlier run
            p = by_uuid.get(gid(self.target, "procedure", r["row"]))
            if p:
                got[r["row"]] = p
                used.add(p["id"])

        def cash(p):
            pr = self.prices.get(p["id"])
            return money(pr["cash_charge"]) if pr else None

        passes = (lambda r, p: cash(p) == r["cash"] and cash(p) is not None,
                  lambda r, p: cash(p) is not None,
                  lambda r, p: True)
        for ok in passes:
            for r in self.rows:
                if r["row"] in got:
                    continue
                p = next((p for p in by_name.get(key(r["name"]), []) if p["id"] not in used and ok(r, p)), None)
                if p:
                    got[r["row"]] = p
                    used.add(p["id"])
        return got

    def categories(self, matched: dict[int, dict]) -> dict[str, int]:
        """sheet department → org category id: the category the org's own matched procedures of that department
        mostly sit in, else DEPARTMENT_CATEGORY by name, else FALLBACK_CATEGORY."""
        by_name = {key(c["name"]): c["id"] for c in self.cats}
        votes = defaultdict(Counter)
        for r in self.rows:
            p = matched.get(r["row"])
            if p and p.get("category"):
                votes[key(r["department"])][p["category"]] += 1
        out = {}
        for d in sorted({key(r["department"]) for r in self.rows}):
            if votes.get(d):
                out[d] = votes[d].most_common(1)[0][0]
            else:
                mapped = DEPARTMENT_CATEGORY.get(d) or ("MCH Services" if d.startswith("mch") else FALLBACK_CATEGORY)
                out[d] = by_name.get(key(mapped)) or by_name.get(key(FALLBACK_CATEGORY))
        return out

    def plan(self) -> dict:
        self.read()
        matched = self.match()
        cat = self.categories(matched)
        cat_name = {c["id"]: c["name"] for c in self.cats}
        taken = Counter(key(p["name"]) for p in self.procs)
        sheet_names = Counter(key(r["name"]) for r in self.rows)
        new_procs, price_new, price_upd, kept, same, inactive = [], [], [], [], 0, []
        for r in self.rows:
            p = matched.get(r["row"])
            cash, ins = r["cash"], r["insurance"]
            variable = cash == 0 and ins == 0
            if p is None:
                name = r["name"]
                if taken[key(name)] or sheet_names[key(name)] > 1:
                    name = f"{name} ({r['department']})"
                    if taken[key(name)]:
                        name = f"{name} #{r['row']}"
                taken[key(name)] += 1
                p = {"uuid": gid(self.target, "procedure", r["row"]), "code": f"{self.cfg['code_prefix']}-{r['row']:03d}",
                     "name": name[:255], "category": cat[key(r["department"])], "active": 1,
                     "description": f"{self.cfg['label']}: {r['department']} · {r['bill_term']} · {r['income_category']}",
                     "_row": r["row"]}
                new_procs.append(p)
            elif not p.get("active", True):
                inactive.append({"row": r["row"], "id": p["id"], "name": p["name"]})
            pr = self.prices.get(p.get("id")) if p.get("id") else None
            want = {"cash_charge": cash, "insurance_charge": ins}
            if variable:
                want["variable_pricing"] = 1
            if pr is None:
                price_new.append({"_row": r["row"], "_procedure_uuid": p["uuid"], "evaluation_procedure_id": p.get("id"),
                                  "uuid": gid(self.target, "price", f"{r['row']}|{self.fac}"), "facility_id": self.fac,
                                  "facility_name": self.facility_name, "charge_insurance": 0, "precharge": 0, "status": 1,
                                  **want})
                continue
            before = {"cash_charge": money(pr["cash_charge"]), "insurance_charge": money(pr["insurance_charge"])}
            if variable and (before["cash_charge"] or before["insurance_charge"]):
                kept.append({"row": r["row"], "name": r["name"], "procedure_id": p["id"], "v3": before})
                continue
            if before == {k: want[k] for k in before} and (not variable or pr.get("variable_pricing")):
                same += 1
                continue
            price_upd.append({"_row": r["row"], "id": pr["id"], "procedure_id": p["id"], "name": p["name"], "code": p.get("code"),
                              "before": before, "data": want})
        return {"matched": len(matched), "new_procedures": new_procs, "price_inserts": price_new,
                "price_updates": price_upd, "kept_v3_price_sheet_has_none": kept, "unchanged": same,
                "matched_but_inactive": inactive,
                "department_category": {d: f"{cid} {cat_name.get(cid)}" for d, cid in cat.items()}}

    # ── posting ──
    def _ok(self, r) -> tuple[bool, str | None]:
        ok = r.ok and (r.json() if r.content else {}).get("success", True)
        return ok, None if ok else f"{r.status_code} {r.text[:300]}"

    def _many(self, items: list, fn) -> tuple[Counter, list]:
        if self.limit:
            items = items[:self.limit]
        res, fails = Counter(), []

        def one(x):
            try:
                ok, err = fn(x)
            except Exception as e:
                ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
            return x, ok, err
        with ThreadPoolExecutor(self.workers) as ex:
            for x, ok, err in ex.map(one, items):
                res["ok" if ok else "failed"] += 1
                if not ok:
                    fails.append({"row": x.get("_row"), "error": err})
                    log.error("row %s: %s", x.get("_row"), err)
        return res, fails

    def insert(self, alias: str, data: dict):
        return self._ok(self.c.insert("evaluation", alias, {k: v for k, v in data.items()
                                                            if not k.startswith("_") and v is not None}, match_on="uuid"))

    def update(self, alias: str, rid: int, data: dict):
        return self._ok(self.c.post("evaluation", {"action": "update", "model": alias, "destination_tenant_id": self.org,
                                                   "ids": [rid], "data": data}, timeout=120))

    def load(self) -> dict:
        plan = self.plan()
        summary = {k: (len(v) if isinstance(v, list) else v) for k, v in plan.items() if k != "department_category"}
        log.info("plan: %s", summary)
        log.info("department → category: %s", plan["department_category"])
        out = {"target": self.target, "org": self.org, "facility": self.fac, "dry_run": self.dry_run, "limit": self.limit,
               "plan": summary, "detail": plan}
        if self.dry_run:
            for k in ("new_procedures", "price_inserts", "price_updates"):
                for x in plan[k][:3]:
                    log.info("  dry-run %s: %s", k, json.dumps(x, default=str)[:600])
            return out
        res, fails = self._many(plan["new_procedures"], lambda p: self.insert("procedures", p))
        out["procedures"] = dict(res)
        if plan["new_procedures"]:
            ids = {str(p.get("uuid")).lower(): p["id"] for p in self.c.read_all("evaluation", "procedures", workers=8)
                   if p.get("uuid")}
            for pr in plan["price_inserts"]:
                pr["evaluation_procedure_id"] = pr["evaluation_procedure_id"] or ids.get(pr["_procedure_uuid"])
        missing = [pr for pr in plan["price_inserts"] if not pr["evaluation_procedure_id"]]
        if missing and not self.limit:
            fails += [{"row": pr["_row"], "error": "procedure not created"} for pr in missing]
        ins = [pr for pr in plan["price_inserts"] if pr["evaluation_procedure_id"]]
        res, f2 = self._many(ins, lambda pr: self.insert("procedure_prices", pr))
        out["price_inserts"] = dict(res)
        res, f3 = self._many(plan["price_updates"], lambda u: self.update("procedure_prices", u["id"], u["data"]))
        out["price_updates"] = dict(res)
        out["failures"] = fails + f2 + f3
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="spreadsheet → dags/jsons/price_lists/<target>.json")
    c.add_argument("xlsx")
    c.add_argument("--target", required=True, choices=list(TARGETS))
    c.add_argument("--sheet", default="MASTER")
    lo = sub.add_parser("load", help="reconcile the price list into V3")
    lo.add_argument("--target", required=True, choices=list(TARGETS))
    lo.add_argument("--dry-run", action="store_true")
    lo.add_argument("--limit", type=int, default=0, help="at most N changes of each kind (canary)")
    lo.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "convert":
        convert(a.xlsx, a.target, a.sheet)
        return
    report = Run(a.target, a.dry_run, a.limit, a.workers).load()
    out = _HERE / "output" / "price_list_to_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.target}-{time.strftime('%Y%m%d_%H%M%S')}{'-dry' if a.dry_run else ''}.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log.info("result: %s", {k: report.get(k) for k in ("plan", "procedures", "price_inserts", "price_updates")})
    log.info("report: %s", path)
    sys.exit(1 if report.get("failures") else 0)


if __name__ == "__main__":
    main()
