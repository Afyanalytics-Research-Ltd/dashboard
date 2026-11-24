"""
stock_take_to_v3.py — a hospital's filled-in V3 stock-take templates (one spreadsheet per store) → V3 products, batches,
stock, adjustment movements and a completed stock take per store, through the model gateway.

First use: Kerugoya Medical Centre, production org 357 / facility 367 (target kerugoya_prod), counts of 2026-11-23:
  "23.11.26- PHARMACY STOCK.xlsx" → Pharmacy (945, the org's pharmacy store that already holds facility-367 stock)
  "Dental Stock Take.xlsx"        → "Dental Store - Kerugoya Medical Centre"   (new, facility 367 — user's choice)
  "Optical Stock Take.xlsx"       → "Optical Store - Kerugoya Medical Centre"  (new, facility 367 — user's choice)
("Dental Store" / "Optical Store" already name facility 368's stores; store names are unique per org.)
Columns: product_code, product_name, counted_batch_no, counted_expiry_date, counted_qty (system_qty / product_id blank).

  products  every sheet row needs a facility-367 product (the user asked for missing ones to be added):
              our own (uuid) → facility-367 product of the same name ("(pcs)" ignored) → an org-level product that
              already holds stock in the target store (this morning's in-app "FieldMatch" count) → a NEW facility-367
              product: code = the sheet code (or <code>-F367 when another org product owns it; codes are unique per
              org), item_no = sheet code, unit PCS, category = what the sheet's matched products mostly use, prices
              from the hospital's pharmacy / cost price lists (dags/jsons/rate_cards/<target>.json).
              Org-level products without stock are not reused: stock on them did not show at the facility (Beta Care).
  stock     each counted row (qty > 0) → a batch (sheet batch no., else KMC-ST-<date>-<code>; sheet expiry) + a stock
            row + an adjustment-in movement. A batch already holding exactly that count is left as is; one holding a
            different count is set to the count (movement = the difference). A row counted 0 whose product holds
            stock in that store is set to 0 (the sheet is the newer count) — only those batches, nothing else.
  stock_take one completed stock take per store (reference <prefix>-<store code>-<date>) + an item per sheet row
            (expected = V3 quantity before, counted = sheet, variance).
New products without any sheet expiry are created with track_expiry off (cotton rolls, frames: no expiry); existing
products counted without an expiry get none (noted on the batch).

Stable uuid5s (target + kind + key) make a re-run update instead of duplicate. Inventory uuids are unindexed on prod
(match_on=uuid times out on 28k products), so rows are matched through a uuid → id map read up front.

    python stock_take_to_v3.py convert <folder> --target kerugoya_prod
    python stock_take_to_v3.py load --target kerugoya_prod --dry-run
    python stock_take_to_v3.py load --target kerugoya_prod
"""
from __future__ import annotations

import argparse
import datetime as dt
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

log = logging.getLogger("stock_take_to_v3")

_HERE = Path(__file__).resolve().parent
DATA_DIRS = (_HERE.parent / "dags" / "jsons" / "stock_takes", Path("/opt/airflow/dags/jsons/stock_takes"))
PRICE_DIRS = (_HERE.parent / "dags" / "jsons" / "rate_cards", Path("/opt/airflow/dags/jsons/rate_cards"))
NS = uuid.UUID("7a4e2c1d-9b3f-4e8a-b6d5-1f0c3e2a9d47")
STEPS = ["stores", "products", "stock", "store_rows"]

TARGETS = {
    "kerugoya_prod": {
        "label": "Kerugoya Medical Centre stock take 2026-11-23",
        "count_date": "2026-11-23",
        "prefix": "KMC-ST",
        "facility": 367,
        "unit": "PCS",
        # sheet file → store: an existing store id, or a store to create in the facility
        "sheets": {
            "pharmacy": {"file": "23.11.26- PHARMACY STOCK.xlsx", "store_id": 945, "code": "PHARM"},
            "dental": {"file": "Dental Stock Take.xlsx", "code": "DENT",
                       "create": {"name": "Dental Store - Kerugoya Medical Centre", "code": "KMC-DENTAL",
                                  "type": "department", "is_consumable": 1, "parent_store_id": 944,
                                  "description": "Dental store, Kerugoya Medical Centre (facility 367)"}},
            "optical": {"file": "Optical Stock Take.xlsx", "code": "OPT",
                        "create": {"name": "Optical Store - Kerugoya Medical Centre", "code": "KMC-OPTICAL",
                                   "type": "department", "is_consumable": 1, "parent_store_id": 944,
                                   "description": "Optical store, Kerugoya Medical Centre (facility 367)"}},
        },
        "fallback_category": {"pharmacy": "PHARMACY", "dental": "Consumable", "optical": "STORES"},
        # (sheet, spreadsheet row) → counted quantity, confirmed with the user 2026-11-24:
        # ATORVASTATIN 20 MG batch AP15024 had counted_qty 0 and "166" typed in the notes column
        "qty_fixes": {("pharmacy", 205): 166},
        "dental_costs": "KMC Dental Items(+cost price).xlsx",
    },
    # Kerugoya Town Clinic (level 3) MIRRORS Kerugoya Medical Centre (367): same products, prices and — user's choice
    # 2026-11-24 — the same 23.11 stock counts. 368 has no pharmacy store of its own (945 is org-wide and holds 367's
    # count), so one is created; dental / optical go to 368's existing Dental Store 957 / Optical Store 958.
    "kerugoya_town_prod": {
        "label": "Kerugoya Town Clinic stock take 2026-11-23 (mirror of Kerugoya Medical Centre)",
        "count_date": "2026-11-23",
        "prefix": "KTC-ST",
        "facility": 368,
        "unit": "PCS",
        "data_target": "kerugoya_prod",            # same sheets + price lists
        "mirror_of": {"target": "kerugoya_prod", "facility": 367},
        "sheets": {
            "pharmacy": {"file": "23.11.26- PHARMACY STOCK.xlsx", "code": "PHARM",
                         "create": {"name": "Pharmacy - Kerugoya Town Clinic", "code": "KTC-PHARMACY", "type": "pharmacy",
                                    "is_consumable": 0, "parent_store_id": 944,
                                    "description": "Pharmacy, Kerugoya Town Clinic (facility 368)"}},
            "dental": {"file": "Dental Stock Take.xlsx", "store_id": 957, "code": "DENT"},
            "optical": {"file": "Optical Stock Take.xlsx", "store_id": 958, "code": "OPT"},
        },
        "fallback_category": {"pharmacy": "PHARMACY", "dental": "Consumable", "optical": "STORES"},
        "qty_fixes": {("pharmacy", 205): 166},
    },
}


# a drug-like name (strength / dosage form); checked against the 1,329 hand-classified Kerugoya products: 96% of the
# names it flags sit in PHARMACY/Drug. Optical lens names ("22%tint", "DS", "DCX") are not drugs.
DRUG = re.compile(r"\b(\d+(\.\d+)?\s*(mg|mcg|g|iu|i\.u|ml|mls|%)|tabs?|tablets?|caps?|capsules?|inj|injection|syrup|syr|"
                  r"susp|suspension|cream|oint|ointment|drops?|gel|lotion|vial|amp|ampoule|inhaler|pessar\w*|suppos\w*|"
                  r"sachets?|elixir|solution|mixture)\b", re.I)
NOT_DRUG = re.compile(r"\b(ds|dcx|tint|lens|lenses|bifocal|frame|needle|syringes?|collector|catheters?|bags?)\b", re.I)


def is_duplicate(err) -> bool:
    """a V3 refusal because the row already exists (unique key): a warning, not a failure — nothing was written"""
    e = str(err or "")
    return "Duplicate entry" in e or "UniqueConstraintViolation" in e or "1062" in e and "23000" in e


def is_drug(name: str) -> bool:
    return bool(DRUG.search(name or "")) and not NOT_DRUG.search(name or "")


def key(v) -> str:
    v = re.sub(r"\(\s*pcs\s*\)", " ", html.unescape(str(v or "")), flags=re.I)
    return re.sub(r"[^a-z0-9]+", " ", v.lower()).strip()


def clean(v) -> str:
    s = "" if v is None else (str(int(v)) if isinstance(v, float) and v == int(v) else str(v))
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def num(v) -> float:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return 0.0


def gid(target: str, kind: str, k) -> str:
    return str(uuid.uuid5(NS, f"{target}|{kind}|{k}"))


def find(dirs, name: str) -> Path:
    for d in dirs:
        if (d / name).exists():
            return d / name
    return dirs[0] / name


# ── convert ──────────────────────────────────────────────────────────────────
def convert(folder: str, target: str, prices_folder: str | None = None) -> Path:
    import openpyxl
    cfg, folder = TARGETS[target], Path(folder)
    out = {"count_date": cfg["count_date"], "sheets": {}}
    for name, sc in cfg["sheets"].items():
        ws = openpyxl.load_workbook(folder / sc["file"], data_only=True, read_only=True).worksheets[0]
        it = ws.iter_rows(values_only=True)
        head = [clean(h).lower() for h in next(it)]
        rows = []
        for n, r in enumerate(it, start=2):
            d = dict(zip(head, r))
            if not clean(d.get("product_name")) and not clean(d.get("product_code")):
                continue                            # (a code-only row gets its name from the price list at load)
            exp = d.get("counted_expiry_date")
            exp = exp.date().isoformat() if isinstance(exp, dt.datetime) else (exp.isoformat() if isinstance(exp, dt.date)
                                                                              else clean(exp)[:10] or None)
            qty = d.get("counted_qty")
            if qty not in (None, "") and not isinstance(qty, (int, float)):
                raise SystemExit(f"{sc['file']} row {n}: counted_qty {qty!r} is not a number")
            if isinstance(qty, float) and qty != int(qty):
                raise SystemExit(f"{sc['file']} row {n}: counted_qty {qty} is fractional (V3 stock is whole units)")
            rows.append({"row": n, "code": clean(d.get("product_code")).upper(), "name": clean(d.get("product_name")),
                         "batch_no": clean(d.get("counted_batch_no")) or None, "expiry": exp,
                         "qty": int(qty or 0), "notes": clean(d.get("notes")) or None})
        dup = [c for c, k in Counter(r["code"] for r in rows if r["code"]).items() if k > 1]
        if dup:
            raise SystemExit(f"{sc['file']}: product codes repeated: {dup[:10]}")
        out["sheets"][name] = {"file": sc["file"], "rows": rows}
        log.info("%s: %d rows, %d counted > 0, %d units", sc["file"], len(rows), sum(r["qty"] > 0 for r in rows),
                 sum(r["qty"] for r in rows))
    if cfg.get("dental_costs") and prices_folder:
        ws = openpyxl.load_workbook(Path(prices_folder) / cfg["dental_costs"], data_only=True, read_only=True).worksheets[0]
        it = ws.iter_rows(values_only=True)
        next(it)
        out["dental_costs"] = {clean(r[0]).upper(): num(r[2]) for r in it if r and clean(r[0])}
    path = DATA_DIRS[0] / f"{target}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("→ %s", path)
    return path


# ── load ─────────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, target: str, dry_run: bool = False, workers: int = 8, steps=None):
        try:
            from pipelines_common import use_pipelines_dir
            use_pipelines_dir([target])
        except (ImportError, RuntimeError):
            from dotenv import load_dotenv
            load_dotenv(_HERE / ".env", override=False)
            sys.path.insert(0, str(_HERE))
        from v3_to_v3_client import V3Client
        if target not in TARGETS:
            raise SystemExit(f"unknown target {target}; known: {list(TARGETS)}")
        self.target, self.dry_run, self.workers = target, dry_run, workers
        self.steps = [s for s in STEPS if s in (steps or STEPS)]
        self.cfg = TARGETS[target]
        self.c = V3Client(target)
        self.c.send_facility_header = True          # new stores / products land in this facility
        self.c.token()
        self.org, self.fac = self.c.organization_id, self.c.facility_id
        if self.fac != self.cfg["facility"]:
            raise SystemExit(f"{target}: login facility {self.fac} ≠ {self.cfg['facility']}")
        dt_ = self.cfg.get("data_target", target)
        self.data = json.loads(find(DATA_DIRS, f"{dt_}.json").read_text(encoding="utf-8"))
        rc = find(PRICE_DIRS, f"{dt_}.json")
        self.prices = json.loads(rc.read_text(encoding="utf-8")) if rc.exists() else {"pharmacy": [], "costs": []}
        self.when = f"{self.data['count_date']} 12:00:00"
        names = {clean(x["code"]).upper(): clean(x["name"]) for x in self.prices.get("pharmacy", []) + self.prices.get("costs", [])}
        for sheet, sd in self.data["sheets"].items():
            for r in sd["rows"]:
                if not r["name"]:
                    r["name"] = names.get(r["code"]) or ""
                fix = self.cfg.get("qty_fixes", {}).get((sheet, r["row"]))
                if fix is not None:
                    log.info("%s row %s %s: counted %s → %s (confirmed fix)", sheet, r["row"], r["name"], r["qty"], fix)
                    r["notes"] = f"counted_qty was {r['qty']}, notes '{r['notes']}' — corrected to {fix}"
                    r["qty"] = fix
            missing = [r["code"] for r in sd["rows"] if not r["name"]]
            if missing:
                log.warning("%s: rows with neither a name nor a known code (skipped): %s", sd["file"], missing)
            sd["rows"] = [r for r in sd["rows"] if r["name"]]
        self.report: dict = {"target": target, "org": self.org, "facility": self.fac, "dry_run": dry_run,
                             "steps": self.steps, "plan": {}, "result": {}, "failures": [], "detail": {}}
        log.info("%r user %s: %s", self.c, self.c.user_id,
                 {k: len(v["rows"]) for k, v in self.data["sheets"].items()})

    # ── gateway helpers ──
    def live(self, svc, alias, workers=8):
        return [r for r in self.c.read_all(svc, alias, workers=workers) if not r.get("deleted_at")]

    def _ok(self, r):
        body = (r.json() if r.content else {}) if r.ok else {}
        ok = r.ok and (body.get("success", True) if isinstance(body, dict) else True)
        data = body.get("data") if isinstance(body, dict) else None
        rid = data.get("id") if isinstance(data, dict) else None      # update answers with a list of rows
        return ok, rid if ok else None, None if ok else f"{r.status_code} {r.text[:300]}"

    def insert(self, svc, alias, data, match_on=None):
        return self._ok(self.c.insert(svc, alias, {k: v for k, v in data.items() if not k.startswith("_") and v is not None},
                                      match_on=match_on))

    def update(self, svc, alias, rid, data):
        return self._ok(self.c.post(svc, {"action": "update", "model": alias, "destination_tenant_id": self.org,
                                          "ids": [rid], "data": data}, timeout=120))

    def many(self, kind, items, fn) -> dict:
        """fn(item) → (ok, id, err); returns {item _key: new id}"""
        if self.dry_run:
            for x in items[:3]:
                log.info("  dry-run %s: %s", kind, json.dumps(x, default=str)[:700])
            return {}
        res, ids = Counter(), {}

        def one(x):
            try:
                return x, *fn(x)
            except Exception as e:
                return x, False, None, f"{type(e).__name__}: {str(e)[:200]}"
        with ThreadPoolExecutor(self.workers) as ex:
            for n, (x, ok, rid, err) in enumerate(ex.map(one, items), 1):
                if ok:
                    res["ok"] += 1
                    ids[x.get("_key")] = rid
                elif is_duplicate(err):              # V3 refused it as a duplicate: nothing written, run goes on
                    res["duplicate (skipped)"] += 1
                    self.report.setdefault("duplicates", []).append({"kind": kind, "key": x.get("_key"), "error": err[:200]})
                    log.warning("%s %s: duplicate, not written — %s", kind, x.get("_key"), err[:200])
                else:
                    res["failed"] += 1
                    self.report["failures"].append({"kind": kind, "key": x.get("_key"), "error": err})
                    log.error("%s %s: %s", kind, x.get("_key"), err)
                if n % 500 == 0:
                    log.info("  %s: %d / %d (%s)", kind, n, len(items), dict(res))
        self.report["result"][kind] = dict(res)
        log.info("%s: %s", kind, dict(res))
        return ids

    # ── stores ──
    def stores(self) -> dict[str, int | None]:
        have = self.live("inventory", "store", 1)
        by_name = {s["name"].strip().lower(): s for s in have}
        out, new = {}, []
        for name, sc in self.cfg["sheets"].items():
            if sc.get("store_id"):
                s = next((s for s in have if s["id"] == sc["store_id"]), None)
                if not s:
                    raise SystemExit(f"store {sc['store_id']} for {name} not found in org {self.org}")
                out[name] = s["id"]
                continue
            s = by_name.get(sc["create"]["name"].lower())
            if s:
                if s.get("facility_id") != self.fac:
                    raise SystemExit(f"store '{s['name']}' exists but in facility {s.get('facility_id')}")
                out[name] = s["id"]
                continue
            out[name] = None
            new.append({"_key": name, "uuid": gid(self.target, "store", name), "is_active": 1, **sc["create"]})
        self.report["plan"]["stores_new"] = [s["name"] for s in new]
        if new and "stores" in self.steps:
            ids = self.many("stores", new, lambda s: self.insert("inventory", "store", s, match_on="uuid"))
            out.update({k: v for k, v in ids.items() if v})
            if not self.dry_run:
                fresh = {s["name"].strip().lower(): s for s in self.live("inventory", "store", 1)}
                for s in new:
                    got = fresh.get(s["name"].lower())
                    if got:
                        out[s["_key"]] = got["id"]
                        if got.get("facility_id") != self.fac:
                            raise SystemExit(f"new store '{s['name']}' landed in facility {got.get('facility_id')}")
        self.store_ids = out
        return out

    # ── products ──
    def read_inventory(self):
        self.products = self.live("inventory", "product")
        self.pid = {p["id"]: p for p in self.products}
        self.batches = self.live("inventory", "batch")
        self.stock = self.live("inventory", "stock")
        self.all_cats = self.live("inventory", "product_category", 1)
        self.cats = {c["name"]: c["id"] for c in self.all_cats if c.get("facility_id") in (None, self.fac)}
        # a facility's own category wins over an org-wide one of the same name
        self.cats.update({c["name"]: c["id"] for c in self.all_cats if c.get("facility_id") == self.fac})
        units = {u["name"].strip().upper(): u["id"] for u in self.live("inventory", "unit", 1)}
        self.unit = units.get(self.cfg["unit"].upper())
        if not self.unit:
            raise SystemExit(f"unit {self.cfg['unit']} not in org {self.org}")
        log.info("org %s: %d products (%d facility %s), %d batches, %d stock rows", self.org, len(self.products),
                 sum(p.get("facility_id") == self.fac for p in self.products), self.fac, len(self.batches), len(self.stock))

    # ── mirror: the source facility's products + its categories, recreated in this facility ──
    def ensure_mirror_categories(self):
        """category id of the source facility → this facility's category of the same name (created when missing);
        org-wide categories map to themselves."""
        mf = self.cfg["mirror_of"]["facility"]
        self.cat_map = {c["id"]: c["id"] for c in self.all_cats if c.get("facility_id") is None}
        own = {c["name"]: c for c in self.all_cats if c.get("facility_id") == self.fac}
        todo = []
        for c in self.all_cats:
            if c.get("facility_id") != mf:
                continue
            if c["name"] in own:
                self.cat_map[c["id"]] = own[c["name"]]["id"]
            elif any(o["name"].strip().lower() == c["name"].strip().lower() for o in self.all_cats):
                # category names are unique per ORG (inv_product_categories_org_name_unique): the source's own
                # category is the only one that can carry that name, so the mirror shares it (prod 2026-11-24)
                self.cat_map[c["id"]] = c["id"]
            else:
                todo.append({"_key": c["id"], "uuid": gid(self.target, "category", c["id"]), "name": c["name"],
                             "description": c.get("description"), "code": c.get("code"),
                             **{k: int(bool(c.get(k))) for k in ("is_consumable", "is_drug", "is_optical")},
                             "cash_markup": c.get("cash_markup"), "credit_markup": c.get("credit_markup")})
        self.report["plan"]["categories_new"] = [c["name"] for c in todo]
        ids = self.many("categories", todo, lambda c: self.insert("inventory", "product_category", c, match_on="uuid"))
        if todo and not self.dry_run:
            fresh = {c["name"]: c for c in self.live("inventory", "product_category", 1) if c.get("facility_id") == self.fac}
            for c in todo:
                got = fresh.get(c["name"]) or ({"id": ids.get(c["_key"])} if ids.get(c["_key"]) else None)
                if got:
                    self.cat_map[c["_key"]] = got["id"]
                    self.cats[c["name"]] = got["id"]
                else:                                # could not make one: keep the source category (still valid)
                    self.cat_map[c["_key"]] = c["_key"]

    def mirror_twins(self) -> dict[str, dict]:
        """sheet key → the source facility's product for it (the one its own run used)"""
        m = self.cfg["mirror_of"]
        by_uuid = {str(p.get("uuid")).lower(): p for p in self.products if p.get("uuid")}
        src_name, org_any = defaultdict(list), defaultdict(list)
        codes = {}
        stocked = {s["product_id"] for s in self.stock if (s.get("quantity") or 0) > 0}
        for p in sorted(self.products, key=lambda p: p["id"]):
            if p.get("facility_id") == m["facility"]:
                src_name[key(p.get("generic_name"))].append(p)
            elif p.get("facility_id") is None and p["id"] in stocked:
                org_any[key(p.get("generic_name"))].append(p)
            if p.get("code"):
                codes.setdefault(str(p["code"]).strip().upper(), p)
        twins = {}
        for sheet, sd in self.data["sheets"].items():
            for r in sd["rows"]:
                k = r["code"] or f"{sheet}:{key(r['name'])}"
                t = (by_uuid.get(gid(m["target"], "product", k)) or next(iter(src_name.get(key(r["name"]), [])), None)
                     or (codes.get(r["code"]) if codes.get(r["code"]) and codes[r["code"]]["id"] in stocked else None)
                     or next(iter(org_any.get(key(r["name"]), [])), None))
                if t:
                    twins[k] = t
        return twins

    def plan_products(self):
        fac_name, org_name = defaultdict(list), defaultdict(list)
        by_uuid = {str(p.get("uuid")).lower(): p for p in self.products if p.get("uuid")}
        codes = {str(p.get("code")).strip().upper(): p for p in self.products if p.get("code")}
        stocked = defaultdict(set)                   # store → product ids holding quantity there
        for s in self.stock:
            if (s.get("quantity") or 0) > 0:
                stocked[s["store_id"]].add(s["product_id"])
        for p in sorted(self.products, key=lambda p: (not p.get("is_active", True), p["id"])):
            if p.get("facility_id") == self.fac:
                fac_name[key(p.get("generic_name"))].append(p)
            elif p.get("facility_id") is None:
                org_name[key(p.get("generic_name"))].append(p)
        sell = {clean(x["code"]).upper(): x["price"] for x in self.prices.get("pharmacy", []) if x.get("price") is not None}
        cost = {}
        for x in self.prices.get("costs", []):
            if x.get("cost") is not None:
                cost.setdefault(clean(x["code"]).upper(), x["cost"])
        cost.update(self.data.get("dental_costs", {}))
        self.product_of, new, how, votes = {}, {}, Counter(), defaultdict(Counter)
        taken_codes = set(codes)
        twins = self.mirror_twins() if self.cfg.get("mirror_of") else {}
        self.report["plan"]["mirror_twins_found"] = len(twins) if twins else None
        for sheet, sd in self.data["sheets"].items():
            store = self.store_ids.get(sheet)
            for r in sd["rows"]:
                k = r["code"] or f"{sheet}:{key(r['name'])}"
                if k in self.product_of or k in new:  # same item on two sheets → one product
                    how["same item, other sheet"] += 1
                    continue
                p = by_uuid.get(gid(self.target, "product", k))
                kind = "ours (earlier run)"
                if not p:
                    p, kind = next(iter(fac_name.get(key(r["name"]), [])), None), "facility product, same name"
                if not p and r["code"] in codes and codes[r["code"]].get("facility_id") == self.fac:
                    p, kind = codes[r["code"]], "facility product, same code"
                if not p:
                    cand = [q for q in org_name.get(key(r["name"]), []) + ([codes[r["code"]]] if r["code"] in codes else [])
                            if q.get("facility_id") is None and q["id"] in stocked.get(store, set())]
                    p, kind = next(iter(cand), None), "org product already stocked in this store"
                if p:
                    self.product_of[k] = p
                    how[kind] += 1
                    if p.get("facility_id") == self.fac and not is_drug(p.get("generic_name")):
                        votes[sheet][p.get("category_id")] += 1
                    continue
                code = r["code"] or None
                if code and code in taken_codes:
                    code = f"{code}-F{self.fac}"
                if code:
                    taken_codes.add(code)
                s, c = sell.get(r["code"]), cost.get(r["code"])
                new[k] = {"_key": k, "_sheet": sheet, "uuid": gid(self.target, "product", k), "generic_name": r["name"][:255],
                          "code": code, "item_no": r["code"] or None, "unit_id": self.unit, "pack_size": 1,
                          "selling_price": s or 0, "insurance_price": s or 0, "cost_price": c or 0, "trade_price": c,
                          "is_active": 1, "track_batch": 1, "track_expiry": 1,
                          "description": f"Added from {sd['file']} ({self.cfg['label']})"}
                t = twins.get(k)
                if t:                                # mirror: the source facility's product, field for field
                    base = (str(t.get("code") or "").strip().upper() or r["code"] or "").replace(f"-F{self.cfg['mirror_of']['facility']}", "")
                    code = f"{base}-F{self.fac}" if base else None
                    if code and code in taken_codes:
                        code = f"{code}-{r['row']}"
                    if code:
                        taken_codes.add(code)
                    new[k].update({
                        "generic_name": t["generic_name"][:255], "code": code, "item_no": t.get("item_no") or r["code"] or None,
                        "unit_id": t.get("unit_id") or self.unit, "pack_size": t.get("pack_size") or 1,
                        "selling_price": num(t.get("selling_price")), "insurance_price": num(t.get("insurance_price")),
                        "cost_price": num(t.get("cost_price")), "trade_price": t.get("trade_price"),
                        "track_batch": int(bool(t.get("track_batch", 1))), "track_expiry": int(bool(t.get("track_expiry", 1))),
                        "requires_prescription": int(bool(t.get("requires_prescription"))),
                        "reorder_level": t.get("reorder_level"), "strength": t.get("strength"), "formulation": t.get("formulation"),
                        "_category": self.cat_map.get(t.get("category_id"), t.get("category_id")), "_twin": t["id"],
                        "description": f"Mirror of facility {self.cfg['mirror_of']['facility']} product {t['id']} "
                                       f"({self.cfg['label']})"})
                    how["new facility product (mirror)"] += 1
                    continue
                how["new facility product"] += 1
        exp_keys = {r["code"] or f"{sheet}:{key(r['name'])}" for sheet, sd in self.data["sheets"].items()
                    for r in sd["rows"] if r["expiry"]}
        counted_keys = {r["code"] or f"{sheet}:{key(r['name'])}" for sheet, sd in self.data["sheets"].items()
                        for r in sd["rows"] if r["qty"] > 0}
        drug_cat = self.cats.get(self.cfg.get("drug_category", "PHARMACY"))
        for k, p in new.items():
            sheet = p.pop("_sheet")
            if "_category" in p:                     # mirror copy: category + flags come from the twin
                p["category_id"] = p.pop("_category")
                p.pop("_twin", None)
                continue
            fb = self.cats.get(self.cfg["fallback_category"][sheet])
            if sheet == "pharmacy" and is_drug(p["generic_name"]) and drug_cat:
                p["category_id"] = drug_cat
            else:                                    # the non-drug category the sheet's matched products mostly use
                p["category_id"] = (votes[sheet].most_common(1)[0][0] if votes[sheet] else None) or fb
            # optical goods never expire; drugs always track expiry; anything else counted without a date doesn't
            if sheet == "optical":
                p["track_expiry"] = 0
            elif is_drug(p["generic_name"]) or k in exp_keys:
                p["track_expiry"] = 1
            else:
                p["track_expiry"] = 0 if k in counted_keys else 1
        self.new_products = list(new.values())
        cat_name = {v: n for n, v in self.cats.items()}
        self.report["plan"].update(products_by=dict(how), products_new=len(new),
                                   products_new_by_category=dict(Counter(cat_name.get(p["category_id"], p["category_id"])
                                                                         for p in new.values())),
                                   products_new_without_price=sum(1 for p in new.values() if not p["selling_price"]),
                                   products_new_code_suffixed=sum(1 for p in new.values() if p["code"] and p["code"].endswith(f"-F{self.fac}")))
        self.report["detail"]["new_products"] = self.new_products

    def run_products(self):
        self.plan_products()
        if "products" not in self.steps or not self.new_products:
            return
        ids = self.many("products", self.new_products, lambda p: self.insert("inventory", "product", p))
        if not self.dry_run:
            fresh = {str(p.get("uuid")).lower(): p for p in self.live("inventory", "product") if p.get("uuid")}
            for p in self.new_products:
                got = fresh.get(p["uuid"])
                if got:
                    if got.get("facility_id") != self.fac:
                        self.report["failures"].append({"kind": "products", "key": p["_key"],
                                                        "error": f"landed in facility {got.get('facility_id')}"})
                    self.product_of[p["_key"]] = got
        else:
            for p in self.new_products:              # dry run: placeholders so the stock plan can be counted
                self.product_of[p["_key"]] = {"id": None, "uuid": p["uuid"], "cost_price": p["cost_price"],
                                              "track_expiry": p["track_expiry"],
                                              "selling_price": p["selling_price"], "insurance_price": p["insurance_price"],
                                              "generic_name": p["generic_name"]}

    # ── stock ──
    def run_stock(self):
        cfg, date = self.cfg, self.data["count_date"]
        user = self.c.user_id
        if not user:
            raise SystemExit("login user id unknown (stock movements need user_id)")
        batch_by = {(b["product_id"], b["store_id"], clean(b["batch_no"])): b for b in self.batches}
        stock_by_batch = {(s["product_id"], s["store_id"], s.get("batch_id")): s for s in self.stock}
        held = defaultdict(list)                     # (product, store) → batches with quantity
        for b in self.batches:
            if (b.get("quantity_available") or 0) > 0:
                held[(b["product_id"], b["store_id"])].append(b)
        # quantity before this count = V3 now minus what this loader's own batches added (a re-run keeps the
        # stock-take lines' expected / variance as first recorded)
        ours = {gid(self.target, "batch", f"{sheet}|{r['code'] or sheet + ':' + key(r['name'])}|"
                                          f"{r['batch_no'] or cfg['prefix'] + '-' + date.replace('-', '') + '-' + str(r['code'] or r['row'])}")
                for sheet, sd in self.data["sheets"].items() for r in sd["rows"]}
        own_batch = {b["id"]: int(b.get("quantity_available") or 0) for b in self.batches if str(b.get("uuid")).lower() in ours}
        qty_now = defaultdict(int)
        for s in self.stock:
            if s.get("batch_id") not in own_batch:
                qty_now[(s["product_id"], s["store_id"])] += int(s.get("quantity") or 0)
        self.report["plan"]["own_batches_already_in_v3"] = len(own_batch)
        new_b, upd_b, same, zeroed, take_items, no_exp, no_exp_off = [], [], 0, [], defaultdict(list), [], {}
        for sheet, sd in self.data["sheets"].items():
            store = self.store_ids.get(sheet)
            for r in sd["rows"]:
                k = r["code"] or f"{sheet}:{key(r['name'])}"
                p = self.product_of.get(k)
                if not p:
                    self.report["failures"].append({"kind": "stock", "key": k, "error": "no product"})
                    continue
                pid = p.get("id")
                before = qty_now.get((pid, store), 0) if pid else 0
                take_items[sheet].append({"k": k, "p": p, "r": r, "expected": before})
                cost, sell = num(p.get("cost_price")), num(p.get("selling_price")) or None
                if r["qty"] > 0:
                    bno = r["batch_no"] or f"{cfg['prefix']}-{date.replace('-', '')}-{r['code'] or r['row']}"
                    b = batch_by.get((pid, store, bno)) if pid else None
                    if b and int(b.get("quantity_available") or 0) == r["qty"]:
                        same += 1
                        continue
                    if r["expiry"] is None and p.get("track_expiry") in (1, True):
                        no_exp.append(f"{sheet}: {r['name']} x{r['qty']}")
                        if pid and p.get("facility_id") == self.fac and not is_drug(r["name"]):
                            no_exp_off[pid] = r["name"]     # frames, lenses, ejectors: stop tracking expiry
                    note = (f"{cfg['label']} ({sd['file']} row {r['row']})" +
                            ("" if r["batch_no"] else "; no batch no. on the sheet") +
                            ("" if r["expiry"] else "; no expiry on the sheet"))
                    if b:
                        upd_b.append({"_key": f"{sheet}|{k}|{bno}", "batch": b, "p": p, "store": store, "qty": r["qty"],
                                      "before": int(b.get("quantity_available") or 0), "note": note, "cost": cost})
                    else:
                        new_b.append({"_key": f"{sheet}|{k}|{bno}", "p": p, "store": store, "qty": r["qty"], "bno": bno,
                                      "expiry": r["expiry"], "cost": cost, "sell": sell, "note": note})
                elif pid and held.get((pid, store)):
                    for b in held[(pid, store)]:
                        zeroed.append({"_key": f"{sheet}|{k}|{b['batch_no']}|zero", "batch": b, "p": p, "store": store,
                                       "qty": 0, "before": int(b.get("quantity_available") or 0), "cost": cost,
                                       "note": f"{cfg['label']}: counted 0 ({sd['file']} row {r['row']})"})
        self.report["plan"].update(batches_new=len(new_b), units_new=sum(x["qty"] for x in new_b),
                                   batches_recount=len(upd_b), batches_zeroed=len(zeroed),
                                   units_zeroed=sum(x["before"] for x in zeroed), batches_unchanged=same,
                                   counted_without_expiry_on_expiry_tracked=len(no_exp),
                                   products_expiry_tracking_off=len(no_exp_off),
                                   stock_take_items={s: len(v) for s, v in take_items.items()})
        self.report["detail"].update(
            counted_without_expiry=no_exp,
            batches_recount=[{"key": x["_key"], "before": x["before"], "after": x["qty"]} for x in upd_b],
            batches_zeroed=[{"key": x["_key"], "product": x["p"].get("generic_name"), "before": x["before"]} for x in zeroed])
        if self.dry_run:
            for kind, xs in (("new batch", new_b), ("recount", upd_b), ("zero", zeroed)):
                for x in xs[:3]:
                    log.info("  dry-run %s: %s", kind, json.dumps({k: v for k, v in x.items() if k not in ("p", "batch")}, default=str))
            return

        self.many("expiry_tracking_off", [{"_key": pid, "id": pid} for pid in no_exp_off],
                  lambda x: self.update("inventory", "product", x["id"], {"track_expiry": 0}))

        # 1. one completed stock take per store (insert-only model: the uuid makes a re-run match it)
        take_ids = {}
        for sheet, items in take_items.items():
            store, sc = self.store_ids[sheet], cfg["sheets"][sheet]
            ref = f"{cfg['prefix']}-{sc['code']}-{date.replace('-', '')}"
            ok, rid, err = self.insert("inventory", "stock_take", {
                "uuid": gid(self.target, "stock_take", sheet), "reference_number": ref, "store_id": store,
                "initiated_by": user, "completed_by": user, "status": "completed", "started_at": self.when,
                "completed_at": self.when, "notes": f"{cfg['label']}: {self.data['sheets'][sheet]['file']} "
                                                   f"({len(items)} lines) loaded by stock_take_to_v3"}, match_on="uuid")
            if not ok or not rid:
                raise SystemExit(f"stock take {ref}: {err or 'no id returned'}")
            take_ids[sheet] = rid
            log.info("stock take %s → id %s", ref, rid)
        self.report["result"]["stock_takes"] = take_ids

        def mv(x, b_id, delta, opening, closing, sheet):
            p = x["p"]
            return {"uuid": gid(self.target, "movement", x["_key"]), "product_id": p["id"], "batch_id": b_id,
                    "batch_no": x.get("bno") or x["batch"]["batch_no"], "expiry_date": x.get("expiry") or (str((x.get("batch") or {}).get("expiry_date") or "")[:10] or None),
                    "type": "adjustment", "direction": "in" if delta >= 0 else "out", "to_store_id": x["store"] if delta >= 0 else None,
                    "from_store_id": None if delta >= 0 else x["store"], "quantity": abs(delta), "unit_cost": x["cost"],
                    "total_value": round(abs(delta) * x["cost"], 2), "opening_balance": opening, "closing_balance": closing,
                    "reference_type": "stock_take", "reference_id": take_ids.get(sheet),
                    "idempotency_key": f"stock_take_to_v3:{self.org}:{x['_key']}"[:191], "user_id": user, "done_by": user,
                    "reason": "Stock take", "notes": x["note"], "created_at": self.when, "updated_at": self.when}

        # 2. new batches → stock rows → movements
        def put_new(x):
            sheet = x["_key"].split("|")[0]
            ok, b_id, err = self.insert("inventory", "batch", {
                "uuid": gid(self.target, "batch", x["_key"]), "batch_no": x["bno"][:255], "product_id": x["p"]["id"],
                "store_id": x["store"], "expiry_date": x["expiry"], "cost_price": x["cost"], "selling_price": x["sell"],
                "quantity_received": x["qty"], "quantity_available": x["qty"], "status": "active", "notes": x["note"]})
            if not ok:
                return False, None, f"batch: {err}"
            if not b_id:
                return False, None, "batch: no id returned"
            ok, s_id, err = self.insert("inventory", "stock", {
                "uuid": gid(self.target, "stock", x["_key"]), "product_id": x["p"]["id"], "store_id": x["store"],
                "batch_id": b_id, "quantity": x["qty"], "available_quantity": x["qty"], "reserved_quantity": 0,
                "average_cost": x["cost"], "total_value": round(x["qty"] * x["cost"], 2), "selling_price": x["sell"],
                "insurance_price": num(x["p"].get("insurance_price")) or x["sell"]})
            if not ok:
                return False, None, f"stock (batch {b_id} made): {err}"
            ok, _, err = self.insert("inventory", "stock_movement", mv(x, b_id, x["qty"], 0, x["qty"], sheet))
            x["batch_id"] = b_id
            return ok, b_id, None if ok else f"movement (batch {b_id}, stock {s_id} made): {err}"
        self.many("new_batches", new_b, put_new)

        # 3. recounts and zeroes: batch + its stock row set to the count, movement = the difference
        def put_count(x):
            sheet, b = x["_key"].split("|")[0], x["batch"]
            ok, _, err = self.update("inventory", "batch", b["id"], {"quantity_available": x["qty"],
                                                                      "status": "active" if x["qty"] else "depleted"})
            if not ok:
                return False, None, f"batch: {err}"
            s = stock_by_batch.get((b["product_id"], b["store_id"], b["id"]))
            if s:
                ok, _, err = self.update("inventory", "stock", s["id"], {
                    "quantity": x["qty"], "available_quantity": max(x["qty"] - int(s.get("reserved_quantity") or 0), 0),
                    "total_value": round(x["qty"] * num(s.get("average_cost") or x["cost"]), 2)})
                if not ok:
                    return False, None, f"stock: {err}"
            x["batch_id"] = b["id"]
            ok, _, err = self.insert("inventory", "stock_movement", mv(x, b["id"], x["qty"] - x["before"], x["before"], x["qty"], sheet))
            return ok, b["id"], None if ok else f"movement: {err}"
        self.many("recounted_batches", upd_b, put_count)
        self.many("zeroed_batches", zeroed, put_count)

        # 4. stock take lines: one per sheet row
        by_key = {x["_key"].split("|")[1]: x for x in new_b + upd_b}
        lines = []
        for sheet, items in take_items.items():
            for it in items:
                x, r = by_key.get(it["k"]), it["r"]
                lines.append({"_key": f"{sheet}|{it['k']}", "uuid": gid(self.target, "stock_take_item", f"{sheet}|{it['k']}"),
                              "stock_take_id": take_ids[sheet], "product_id": it["p"]["id"],
                              "batch_id": (x or {}).get("batch_id"), "counted_batch_no": r["batch_no"],
                              "counted_expiry_date": r["expiry"], "expected_quantity": it["expected"],
                              "counted_quantity": r["qty"], "variance": r["qty"] - it["expected"],
                              "is_reconciled": 1, "is_unexpected": 0, "notes": r["notes"]})
        self.many("stock_take_items", lines, lambda l: self.insert("inventory", "stock_take_item", l, match_on="uuid"))

    # ── per-store base rows ──
    def run_store_rows(self):
        """A product shows in a store only through a base stock row (batch NULL, quantity 0) — the app makes one in
        every store when a product is created; products made through the gateway had none, so Dennis could not find
        e.g. PARACETAMOL 500MG TABS (1,030 units in Pharmacy) — 2026-11-24. Adds the missing base rows for every active
        facility product, and every product holding stock, in every store of the facility (and the org-wide stores)."""
        stores = [s for s in self.live("inventory", "store", 1)
                  if s.get("is_active", True) and s.get("facility_id") in (None, self.fac)]
        stocked = {s["product_id"] for s in self.stock if (s.get("quantity") or 0) > 0}
        prods = [p for p in self.products if p.get("is_active", True)
                 and (p.get("facility_id") == self.fac or (p.get("facility_id") is None and p["id"] in stocked))]
        have = {(s["product_id"], s["store_id"]) for s in self.stock if not s.get("batch_id")}
        rows = [{"_key": f"{p['id']}|{st['id']}", "uuid": gid(self.target, "base_row", f"{p['id']}|{st['id']}"),
                 "product_id": p["id"], "store_id": st["id"], "quantity": 0, "available_quantity": 0,
                 "reserved_quantity": 0, "average_cost": 0, "total_value": 0}
                for p in prods for st in stores if (p["id"], st["id"]) not in have]
        self.report["plan"].update(base_row_stores=[f"{s['id']} {s['name']}" for s in stores], base_row_products=len(prods),
                                   base_rows_missing=len(rows),
                                   base_rows_missing_by_store=dict(Counter(r["store_id"] for r in rows)))
        self.many("base_rows", rows, lambda r: self.insert("inventory", "stock", r))

    def load(self) -> dict:
        if self.steps == ["store_rows"]:              # only the base rows: no stores / products / stock planning
            self.read_inventory()
            self.run_store_rows()
            log.info("plan: %s", json.dumps(self.report["plan"], default=str))
            log.info("result: %s", json.dumps(self.report["result"], default=str))
            return self.report
        self.stores()
        self.read_inventory()
        if self.cfg.get("mirror_of"):
            self.ensure_mirror_categories()
        self.run_products()
        if "stock" in self.steps:
            if not self.dry_run:
                self.read_inventory()                # new products' ids, fresh quantities
                self.product_of.update({k: self.pid.get(p["id"], p) for k, p in self.product_of.items() if p.get("id")})
            self.run_stock()
        if "store_rows" in self.steps:
            if not self.dry_run:
                self.read_inventory()
            self.run_store_rows()
        log.info("plan: %s", json.dumps(self.report["plan"], default=str))
        log.info("result: %s", json.dumps(self.report["result"], default=str))
        return self.report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("folder")
    c.add_argument("--target", required=True, choices=list(TARGETS))
    c.add_argument("--prices-folder", help="folder with the dental cost sheet")
    lo = sub.add_parser("load")
    lo.add_argument("--target", required=True, choices=list(TARGETS))
    lo.add_argument("--dry-run", action="store_true")
    lo.add_argument("--workers", type=int, default=8)
    lo.add_argument("--steps", nargs="*", choices=STEPS, default=STEPS)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "convert":
        convert(a.folder, a.target, a.prices_folder)
        return
    report = Run(a.target, a.dry_run, a.workers, a.steps).load()
    out = _HERE / "output" / "stock_take_to_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.target}-{time.strftime('%Y%m%d_%H%M%S')}{'-dry' if a.dry_run else ''}.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log.info("report: %s", path)
    sys.exit(1 if report.get("failures") else 0)


if __name__ == "__main__":
    main()
