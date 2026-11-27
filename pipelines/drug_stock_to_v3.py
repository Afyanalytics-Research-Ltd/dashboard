"""
drug_stock_to_v3.py — a hospital's hand-counted drug stock sheet (one row per item, extra batch rows underneath, with
cost / selling / insurance prices) → a STOCK TAKE of one V3 store, through the model gateway.

First use: Chacawest Healthcare, production org 131 / facility 129 (target chacawest_prod),
"CHACAWEST_Drug_Stock (1).xlsx updated with prizes (3).xlsx" (sheet "Stock", 262 items, 2026-11-27).
The sheet's headers are shifted: "Categories" holds the item NAME, "Unit" the counted QUANTITY, "Quantity" the
STRENGTH / pack size. A row without a name is one more batch of the item above it, with no quantity of its own.

Decisions (user, 2026-11-27):
  store     Main store 288, where the facility's October stock already sits — the sheet is a recount of it.
  products  each sheet item → the V3 product that already holds one of its batch numbers (names must share a drug
            word, or the row is listed in accept_batch_match), else a facility / stocked product of the same name
            (word set), else a NEW facility-129 product. Matched products are RENAMED to the sheet's spelling and get
            the sheet's cost / selling / insurance prices.
  batches   one total per item: it goes on the FIRST batch listed; the other batch numbers are recorded with 0 (real
            batch no. + expiry) so they can be recounted. Matched products' other batches in the store → 0. An item
            whose quantity the sheet leaves blank keeps its V3 stock untouched (listed in the report).
  expiry    "MM/YY" = last day of that month (V3 stores 2028-07-31 for 07/28, as the October receipts did). Excel turned
            some "MM/YY" into dates of 2026 (07/28 → 2026-07-28: month 7, year 2028) or, for a day that doesn't exist,
            into the 1st (2/29 → 2029-02-01) — both read back as month / year. An impossible month (16/28) → no expiry.
Same name, different pack size ("Paracetamol 120mg/5ml syrup" 100ml / 60ml) → two products, named with the size.
Products not on the sheet are not touched. Each run records a completed stock take (one line per sheet batch).

Stable uuid5s (target + kind + key) make a re-run update instead of duplicate.

    python drug_stock_to_v3.py convert <xlsx> --target chacawest_prod
    python drug_stock_to_v3.py load --target chacawest_prod --dry-run
    python drug_stock_to_v3.py load --target chacawest_prod
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from stock_take_to_v3 import DATA_DIRS, clean, find, gid, is_drug, is_duplicate, key, local_date, num  # noqa: E402

log = logging.getLogger("drug_stock_to_v3")

STEPS = ["products", "stock", "expiry", "store_rows", "verify"]

TARGETS = {
    "chacawest_prod": {
        "label": "Chacawest drug stock count 2026-11-27",
        "file": "CHACAWEST_Drug_Stock (1).xlsx updated with prizes (3).xlsx",
        "count_date": "2026-11-27",
        "prefix": "CHW-ST",
        "code_prefix": "CHW",
        "facility": 129,
        "store_id": 288,                       # Main
        "base_row_stores": [288, 289],         # a product shows in a store only through a base row (Main, Pharmacy)
        "unit": "PCS",
        "drug_category": "Drugs",
        "other_category": "Medical Supplies",
        # sheet row → the V3 product (stocked in Main) that IS this item under another / brand name, reviewed by hand
        # from the 2026-11-27 dry run (same drug, strength and form only; doubtful pairs left out → new product):
        "force_match": {
            # batch-number matches whose names share no word (Piriton = chlorpheniramine, Franol = theophylline +
            # ephedrine, Dyrade-M / Eflaron = metronidazole + diloxanide, Pronerve, Esose HP-kit, Cotrimox, Ampiclox…)
            41: 2446550, 45: 2447234, 49: 2447233, 60: 2444216, 68: 2446652, 116: 2431401, 130: 2425654,
            140: 2446056, 145: 2446299, 146: 2445450, 148: 2445451, 186: 2446400, 213: 2431839, 230: 2442232,
            # same item, other name, no shared batch number
            2: 2445275, 15: 2426407, 17: 2445291, 19: 2445293, 27: 2446783, 32: 2429188, 33: 4756252, 35: 2445894,
            38: 2448115, 40: 2437691, 43: 2447497, 46: 2447652, 48: 3925588, 55: 2424706, 58: 2446092, 59: 2424951,
            64: 2446990, 65: 2427162, 69: 2426390, 72: 2446133, 74: 2428893, 78: 4756255, 81: 2446792, 82: 2426408,
            85: 2448024, 98: 2424460, 99: 2447816, 104: 2445693, 106: 2441367, 107: 2424656, 108: 2425192,
            109: 2424525, 110: 2442253, 112: 2445547, 117: 3488936, 124: 2446179, 127: 2425388, 128: 2446554,
            132: 2424785, 133: 2424405, 134: 2447305, 142: 2446055, 144: 2424301, 152: 2446533, 153: 3461439,
            154: 3319163, 155: 3319164, 156: 2442744, 166: 2425781, 168: 3319162, 181: 2425877, 182: 2446781,
            183: 2425240, 192: 2955285, 195: 2427779, 202: 2447167, 206: 2424344, 207: 2446907, 215: 2434549,
            221: 2424306, 233: 2425296, 245: 2447657, 247: 2430773, 251: 2446778, 253: 2425264, 263: 2446666,
            269: 2447505, 281: 2955302, 285: 2443443, 287: 4756249, 288: 3925583, 293: 3925573, 297: 4756250,
            300: 2446926, 301: 2446209,
        },
        # batch-number matches that are a different product (same number, other manufacturer)
        "reject_batch_match": [17, 19, 109],
    },
}

MONTHS_OK = range(1, 13)
NOISE = {"tablet", "tablets", "tabs", "tab", "capsule", "capsules", "caps", "syrup", "suspension", "susp", "injection",
         "inj", "cream", "drops", "drop", "solution", "oral", "with", "plus", "and", "for", "hydrochloride", "sodium",
         "potassium", "sulphate", "sulfate", "maleate", "acid", "usp", "dry", "extended", "release", "sustained",
         "chewable", "intravenous", "infusion", "vials", "vial", "ampoules", "gel", "child", "adult", "units", "tube",
         "eye", "ear", "nasal", "spray", "mouthwash", "lotion", "jelly", "kit", "mega", "trihydrate", "bisulfate",
         "dihydrochloride", "besylate", "dipropionate", "acetate", "phosphate", "decanoate", "furoate", "propionate"}
SYN = {"tabs": "tablet", "tab": "tablet", "tablets": "tablet", "caps": "capsule", "capsules": "capsule",
       "mls": "ml", "gm": "g", "susp": "suspension", "inj": "injection", "syp": "syrup",
       "cephalexin": "cefalexin", "gentamycin": "gentamicin", "gentamicicn": "gentamicin"}


def words(name: str) -> set[str]:
    """drug words of a name (brands in brackets included), forms / salts dropped"""
    out = set()
    for w in re.findall(r"[a-z]{4,}", str(name or "").lower()):
        w = SYN.get(w, w)
        if w not in NOISE:
            out.add(w)
    return out


def share_word(a: str, b: str) -> bool:
    return any(x == y or SequenceMatcher(None, x, y).ratio() >= 0.8 for x in words(a) for y in words(b))


def tokens(name: str) -> frozenset:
    """order-free name key: 'PARACETAMOL TABS 500MG' == 'Paracetamol 500mg tablets'"""
    s = re.sub(r"(\d)\s+(mg|mcg|g|ml|iu)\b", r"\1\2", str(name or "").lower())
    return frozenset(SYN.get(t, t) for t in re.findall(r"[a-z0-9.%/]+", s) if t not in {"s", "the", "of"})


def nb(v) -> str:
    """batch numbers compared without spaces / punctuation / case"""
    return re.sub(r"[^A-Z0-9]", "", str(v or "").upper())


def close(a: str, b: str) -> bool:
    """two normalized batch numbers that differ by a misread character (57D01426 / 57001426, AZR503 / A2R503)"""
    if a == b or min(len(a), len(b)) < 4:
        return a == b
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.88


def near_batch(batches: dict, want: str, taken: set):
    """the one store batch (normalized no. → batch) close to `want` and not claimed by another sheet line"""
    hits = [b for n, b in batches.items() if n not in taken and close(n, want)]
    return hits[0] if len(hits) == 1 else None


def month_end(y: int, m: int) -> str:
    return dt.date(y, m, calendar.monthrange(y, m)[1]).isoformat()


def parse_expiry(v) -> tuple[str | None, str | None]:
    """→ (YYYY-MM-DD last day of the month, problem)"""
    if v is None or clean(v) == "":
        return None, "no expiry on the sheet"
    if isinstance(v, dt.datetime):
        # Excel read "MM/YY" as month/day of 2026, or (no such day) as month/year on the 1st
        if v.year == 2026 and v.day >= 20:
            return month_end(2000 + v.day, v.month), None
        if v.day == 1:
            return month_end(v.year, v.month), None
        return None, f"expiry {v.date()} not readable as MM/YY"
    m = re.fullmatch(r"\s*(\d{1,2})\s*/\s*(\d{2}|\d{4})\s*", str(v))
    if not m:
        return None, f"expiry {v!r} not readable"
    mo, yr = int(m.group(1)), int(m.group(2))
    yr = yr if yr > 99 else 2000 + yr
    if mo not in MONTHS_OK:
        return None, f"expiry {v!r}: no month {mo}"
    return month_end(yr, mo), None


# ── convert ──────────────────────────────────────────────────────────────────
def convert(xlsx: str, target: str) -> Path:
    import openpyxl
    cfg = TARGETS[target]
    ws = openpyxl.load_workbook(xlsx, data_only=True).worksheets[0]
    items, cur = [], None
    for n, r in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        r = (list(r) + [None] * 10)[:10]
        no, name, qty, bno, exp, strength, cost, sell, ins, notes = r
        if not any(clean(x) for x in r):
            continue
        e, eprob = parse_expiry(exp)
        if qty not in (None, "") and not isinstance(qty, (int, float)):
            raise SystemExit(f"row {n}: quantity {qty!r} is not a number")
        if isinstance(qty, float) and qty != int(qty):
            raise SystemExit(f"row {n}: quantity {qty} is fractional")
        batch = {"row": n, "batch_no": clean(bno) or None, "expiry": e, "expiry_sheet": clean(exp) or None,
                 "expiry_problem": eprob, "notes": clean(notes) or None}
        if clean(name):
            cur = {"row": n, "no": clean(no) or None, "sheet_name": clean(name), "strength": clean(strength) or None,
                   "cost": num(cost) if cost not in (None, "") else None,
                   "sell": num(sell) if sell not in (None, "") else None,
                   "ins": num(ins) if ins not in (None, "") else None,
                   "qty": int(qty) if qty not in (None, "") else None, "notes": clean(notes) or None,
                   "batches": [batch]}
            items.append(cur)
        else:
            if cur is None:
                raise SystemExit(f"row {n}: batch row before any item")
            if qty not in (None, ""):
                raise SystemExit(f"row {n}: a batch row with its own quantity — handle explicitly")
            cur["batches"].append(batch)
    # names: the same name with different pack sizes → the size goes into the name; same size → one product
    groups = defaultdict(list)
    for it in items:
        groups[key(it["sheet_name"])].append(it)
    for k, its in groups.items():
        sizes = {key(i["strength"]) for i in its if i["strength"]}
        for i in its:
            i["name"] = i["sheet_name"]
            if len(sizes) > 1 and i["strength"] and key(i["strength"]) not in key(i["sheet_name"]):
                i["name"] = f"{i['sheet_name']} {i['strength']}"
    out = {"count_date": cfg["count_date"], "file": Path(xlsx).name, "items": items}
    path = DATA_DIRS[0] / f"{target}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("%s: %d items, %d batch rows, %d units → %s", Path(xlsx).name, len(items),
             sum(len(i["batches"]) for i in items), sum(i["qty"] or 0 for i in items), path)
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
        from v3_to_v3_client import V3Client
        if target not in TARGETS:
            raise SystemExit(f"unknown target {target}; known: {list(TARGETS)}")
        self.target, self.dry_run, self.workers = target, dry_run, workers
        self.steps = [s for s in STEPS if s in (steps or STEPS)]
        self.cfg = TARGETS[target]
        self.c = V3Client(target)
        self.c.send_facility_header = True
        self.c.token()
        self.org, self.fac = self.c.organization_id, self.c.facility_id
        if self.fac != self.cfg["facility"]:
            raise SystemExit(f"{target}: login facility {self.fac} ≠ {self.cfg['facility']}")
        self.data = json.loads(find(DATA_DIRS, f"{target}.json").read_text(encoding="utf-8"))
        self.items = self.data["items"]
        # items with the same final name are one product (Gentamicin drops on two rows)
        self.groups = defaultdict(list)
        for it in self.items:
            self.groups[key(it["name"])].append(it)
        self.store = self.cfg["store_id"]
        self.when = f"{self.data['count_date']} 12:00:00"
        self.report: dict = {"target": target, "org": self.org, "facility": self.fac, "dry_run": dry_run,
                             "steps": self.steps, "plan": {}, "result": {}, "failures": [], "detail": {}}
        log.info("%r user %s: %d items / %d products", self.c, self.c.user_id, len(self.items), len(self.groups))

    # ── gateway helpers (as stock_take_to_v3) ──
    def live(self, alias, workers=8, svc="inventory"):
        return [r for r in self.c.read_all(svc, alias, workers=workers) if not r.get("deleted_at")]

    def _ok(self, r):
        body = (r.json() if r.content else {}) if r.ok else {}
        ok = r.ok and (body.get("success", True) if isinstance(body, dict) else True)
        data = body.get("data") if isinstance(body, dict) else None
        rid = data.get("id") if isinstance(data, dict) else None
        return ok, rid if ok else None, None if ok else f"{r.status_code} {r.text[:300]}"

    def insert(self, alias, data, match_on="uuid"):
        return self._ok(self.c.insert("inventory", alias, {k: v for k, v in data.items() if not k.startswith("_")},
                                      match_on=match_on))

    def update(self, alias, rid, data):
        return self._ok(self.c.post("inventory", {"action": "update", "model": alias, "destination_tenant_id": self.org,
                                                  "ids": [rid], "data": data}, timeout=120))

    def many(self, kind, items, fn, workers=None) -> dict:
        if self.dry_run:
            for x in items[:3]:
                log.info("  dry-run %s: %s", kind, json.dumps({k: v for k, v in x.items() if not k.startswith("__")},
                                                              default=str)[:700])
            self.report["result"][kind] = {"planned": len(items)}
            return {}
        res, ids = Counter(), {}

        def one(x):
            try:
                return x, *fn(x)
            except Exception as e:  # noqa: BLE001 — one bad row must not stop the rest
                return x, False, None, f"{type(e).__name__}: {str(e)[:200]}"
        with ThreadPoolExecutor(workers or self.workers) as ex:
            for n, (x, ok, rid, err) in enumerate(ex.map(one, items), 1):
                if ok:
                    res["ok"] += 1
                    ids[x.get("_key")] = rid
                elif is_duplicate(err):
                    res["duplicate (skipped)"] += 1
                    self.report.setdefault("duplicates", []).append({"kind": kind, "key": x.get("_key"), "error": err[:200]})
                    log.warning("%s %s: duplicate, not written — %s", kind, x.get("_key"), err[:200])
                else:
                    res["failed"] += 1
                    self.report["failures"].append({"kind": kind, "key": x.get("_key"), "error": err})
                    log.error("%s %s: %s", kind, x.get("_key"), err)
                if n % 200 == 0:
                    log.info("  %s: %d / %d (%s)", kind, n, len(items), dict(res))
        self.report["result"][kind] = dict(res)
        log.info("%s: %s", kind, dict(res))
        return ids

    def read_inventory(self):
        self.products = self.live("product")
        self.pid = {p["id"]: p for p in self.products}
        self.batches = self.live("batch")
        self.bid = {b["id"]: b for b in self.batches}
        self.stock = self.live("stock")
        cats = self.live("product_category", 1)
        self.cats = {c["name"]: c["id"] for c in cats if c.get("facility_id") in (None, self.fac)}
        units = {u["name"].strip().upper(): u["id"] for u in self.live("unit", 1)}
        self.unit = units.get(self.cfg["unit"].upper())
        if not self.unit:
            raise SystemExit(f"unit {self.cfg['unit']} not in org {self.org}")
        log.info("org %s: %d products, %d batches, %d stock rows", self.org, len(self.products), len(self.batches),
                 len(self.stock))

    # ── products ──
    def pkey(self, it) -> str:
        return key(it["name"])

    def plan_products(self):
        cfg = self.cfg
        by_uuid = {str(p.get("uuid")).lower(): p for p in self.products if p.get("uuid")}
        pos = defaultdict(int)                       # product → units held in the store
        for s in self.stock:
            if s["store_id"] == self.store:
                pos[s["product_id"]] += int(s.get("quantity") or 0)
        # batch numbers held in the facility's stores (any quantity) → products
        bmap = defaultdict(set)
        for b in self.batches:
            p = self.pid.get(b["product_id"])
            if p and p.get("facility_id") in (None, self.fac) and len(nb(b["batch_no"])) >= 4:
                bmap[nb(b["batch_no"])].add(b["product_id"])
        fac_tok, stocked_tok = defaultdict(list), defaultdict(list)
        for p in sorted(self.products, key=lambda p: p["id"]):
            if not p.get("is_active", True):
                continue
            if p.get("facility_id") == self.fac:
                fac_tok[tokens(p.get("generic_name"))].append(p)
            elif pos.get(p["id"]):
                stocked_tok[tokens(p.get("generic_name"))].append(p)
        self.product_of, how, cand_log, claimed = {}, Counter(), [], {}
        force = {int(k): v for k, v in cfg.get("force_match", {}).items()}
        reject = set(cfg.get("reject_batch_match", []))
        plan = []
        for k, its in self.groups.items():
            it0 = its[0]
            p = by_uuid.get(gid(self.target, "product", k))
            kind = "ours (earlier run)"
            fid = next((force[i["row"]] for i in its if i["row"] in force), None)
            if not p and fid:
                p, kind = self.pid.get(fid), "reviewed match"
                if not p:
                    raise SystemExit(f"force_match row {it0['row']}: product {fid} not in org {self.org}")
            if not p:
                kind = None
                for it in its:
                    for b in it["batches"]:
                        for cpid in sorted(bmap.get(nb(b["batch_no"]), ())):
                            q = self.pid[cpid]
                            ok = share_word(it["sheet_name"], q.get("generic_name")) and it["row"] not in reject
                            cand_log.append({"row": it["row"], "sheet": it["name"], "batch": b["batch_no"], "v3_id": cpid,
                                             "v3": q.get("generic_name"), "v3_units_in_store": pos.get(cpid, 0),
                                             "accepted": ok})
                            if ok and not p:
                                p, kind = q, "batch number"
            if not p:
                hit = fac_tok.get(tokens(it0["name"])) or fac_tok.get(tokens(it0["sheet_name"]))
                if hit:
                    p, kind = hit[0], "facility product, same name"
            if not p:
                hit = stocked_tok.get(tokens(it0["name"])) or stocked_tok.get(tokens(it0["sheet_name"]))
                if hit:
                    p, kind = hit[0], "stocked product, same name"
            if p and p["id"] in claimed and kind != "ours (earlier run)":
                how[f"{kind} — already taken by row {claimed[p['id']]} → new"] += 1
                p = None
            if p:
                claimed[p["id"]] = it0["row"]
                self.product_of[k] = p
                how[kind] += 1
                plan.append({"row": it0["row"], "sheet": it0["name"], "v3_id": p["id"], "v3_name": p.get("generic_name"),
                             "how": kind})
            else:
                how["new facility product"] += 1
                plan.append({"row": it0["row"], "sheet": it0["name"], "v3_id": None, "how": "new"})
        self.report["plan"]["products_by"] = dict(how)
        self.report["detail"]["product_plan"] = plan
        self.report["detail"]["batch_match_candidates"] = cand_log
        # new products
        taken = {str(p.get("code")).strip().upper() for p in self.products if p.get("code")}
        drug_cat, other_cat = self.cats.get(cfg["drug_category"]), self.cats.get(cfg["other_category"])
        self.new_products, self.product_updates = [], []
        for k, its in self.groups.items():
            it = its[0]
            sell, ins, cost = self.price(its, "sell"), self.price(its, "ins"), self.price(its, "cost")
            if k not in self.product_of:
                code = f"{cfg['code_prefix']}-{it['no'] or it['row']}"
                if code.upper() in taken:
                    code = f"{code}-R{it['row']}"
                taken.add(code.upper())
                drug = is_drug(it["name"]) or is_drug(f"{it['name']} {it['strength'] or ''}")
                has_exp = any(b["expiry"] for i in its for b in i["batches"])
                self.new_products.append({
                    "_key": k, "uuid": gid(self.target, "product", k), "generic_name": it["name"][:255], "code": code,
                    "item_no": code, "unit_id": self.unit, "pack_size": 1, "strength": (it["strength"] or None),
                    "selling_price": sell or 0, "insurance_price": ins or sell or 0, "cost_price": cost or 0,
                    "trade_price": cost, "is_active": 1, "track_batch": 1, "track_expiry": 1 if (drug or has_exp) else 0,
                    "category_id": drug_cat if drug else (other_cat or drug_cat),
                    "description": f"Added from {self.data['file']} row {it['row']} ({cfg['label']})"})
                continue
            p, upd = self.product_of[k], {}
            if p.get("generic_name") != it["name"][:255]:
                upd["generic_name"] = it["name"][:255]
            for f, v in (("selling_price", sell), ("insurance_price", ins), ("cost_price", cost)):
                if v is not None and num(p.get(f)) != v:
                    upd[f] = v
            if it["strength"] and not p.get("strength"):
                upd["strength"] = it["strength"]
            if not p.get("is_active", True):
                upd["is_active"] = 1
            if upd:
                self.product_updates.append({"_key": k, "id": p["id"], "before": {f: p.get(f) for f in upd}, "data": upd})
        self.report["plan"].update(
            products_new=len(self.new_products), products_updated=len(self.product_updates),
            products_renamed=sum("generic_name" in u["data"] for u in self.product_updates),
            products_without_selling_price=[i["name"] for its in self.groups.values() for i in its[:1]
                                            if self.price(its, "sell") is None])
        self.report["detail"]["new_products"] = self.new_products
        self.report["detail"]["product_updates"] = self.product_updates

    @staticmethod
    def price(its, f):
        """the item's price (rows of one product agree; else the first given)"""
        vals = [i[f] for i in its if i[f] is not None]
        return vals[0] if vals else None

    def run_products(self):
        self.plan_products()
        if "products" not in self.steps:
            return
        self.many("products_new", self.new_products, lambda p: self.insert("product", p))
        self.many("products_update", self.product_updates, lambda u: self.update("product", u["id"], u["data"]))
        if self.dry_run:
            for p in self.new_products:
                self.product_of[p["_key"]] = {"id": None, "uuid": p["uuid"], "generic_name": p["generic_name"],
                                              "cost_price": p["cost_price"], "selling_price": p["selling_price"],
                                              "insurance_price": p["insurance_price"]}
            return
        fresh = {str(p.get("uuid")).lower(): p for p in self.live("product") if p.get("uuid")}
        for p in self.new_products:
            got = fresh.get(p["uuid"])
            if not got:
                self.report["failures"].append({"kind": "products_new", "key": p["_key"], "error": "not found after insert"})
                continue
            if got.get("facility_id") != self.fac:
                self.report["failures"].append({"kind": "products_new", "key": p["_key"],
                                                "error": f"landed in facility {got.get('facility_id')}"})
            self.product_of[p["_key"]] = got

    # ── stock ──
    def run_stock(self):
        cfg, date, user, st = self.cfg, self.data["count_date"], self.c.user_id, self.store
        if not user:
            raise SystemExit("login user id unknown (stock movements need user_id)")
        stock_of_batch = defaultdict(list)
        for s in self.stock:
            if s.get("batch_id"):
                stock_of_batch[s["batch_id"]].append(s)
        pos_rows = defaultdict(list)                 # product → stock rows with quantity in the store
        for s in self.stock:
            if s["store_id"] == st and (s.get("quantity") or 0) > 0:
                pos_rows[s["product_id"]].append(s)
        store_batches = defaultdict(dict)            # product → normalized batch no. → batch in the store
        for b in sorted(self.batches, key=lambda b: b["id"]):
            if b["store_id"] == st:
                store_batches[b["product_id"]].setdefault(nb(b["batch_no"]), b)
        new_b, set_b, zero, same, lines, untouched, near_kept = [], [], [], 0, [], [], []
        for k, its in self.groups.items():
            p = self.product_of.get(k)
            if not p:
                self.report["failures"].append({"kind": "stock", "key": k, "error": "no product"})
                continue
            pid = p.get("id")
            sell = self.price(its, "sell") or num(p.get("selling_price")) or None
            ins = self.price(its, "ins") or num(p.get("insurance_price")) or sell
            cost = self.price(its, "cost") if self.price(its, "cost") is not None else num(p.get("cost_price"))
            if all(i["qty"] is None for i in its):
                untouched.append(f"row {its[0]['row']} {its[0]['name']}: no quantity on the sheet — V3 stock "
                                 f"{sum(int(s['quantity']) for s in pos_rows.get(pid, []))} left as is")
                continue
            sheet_bnos = {nb(b["batch_no"] or f"{cfg['prefix']}-{date.replace('-', '')}-R{b['row']}")
                          for it in its for b in it["batches"]}
            for it in its:
                for j, b in enumerate(it["batches"]):
                    qty = (it["qty"] or 0) if j == 0 else 0
                    bno = b["batch_no"] or f"{cfg['prefix']}-{date.replace('-', '')}-R{b['row']}"
                    have = store_batches.get(pid, {}).get(nb(bno)) if pid else None
                    if not have and pid and b["batch_no"]:
                        have = near_batch(store_batches.get(pid, {}), nb(bno), sheet_bnos)
                        if have:                     # same batch, number misread on one side: keep V3's
                            near_kept.append(f"{p.get('generic_name')}: sheet {bno} = V3 {have['batch_no']}")
                            b = {**b, "notes": "; ".join(filter(None, [b["notes"], f"sheet batch no. {bno}"]))}
                            bno = have["batch_no"]
                            sheet_bnos.add(nb(bno))
                    before = int(have.get("quantity_available") or 0) if have else 0
                    rows = [s for s in stock_of_batch.get(have["id"], []) if s["store_id"] == st] if have else []
                    held = sum(int(s.get("quantity") or 0) for s in rows) if have else 0
                    note = (f"{cfg['label']} ({self.data['file']} row {b['row']})"
                            + ("" if b["batch_no"] else "; no batch no. on the sheet")
                            + (f"; {b['expiry_problem']}" if b["expiry_problem"] else "")
                            + ("; the item's total is on its first batch — recount this one" if j else "")
                            + (f"; sheet note: {b['notes']}" if b["notes"] else ""))
                    x = {"_key": f"{k}|{nb(bno)}", "p": p, "qty": qty, "bno": bno, "expiry": b["expiry"],
                         "cost": cost or 0, "sell": sell, "ins": ins, "note": note, "before": held if have else 0,
                         "batch": have, "rows": rows, "row": b["row"]}
                    lines.append(x)
                    if have:
                        if before == qty and held == qty and len(rows) == 1 and (
                                not b["expiry"] or local_date(have.get("expiry_date")) == b["expiry"]):
                            same += 1
                        else:
                            set_b.append(x)
                    else:
                        new_b.append(x)
            # the product's other stock in the store (batches not on the sheet) → 0
            for s in pos_rows.get(pid, []) if pid else []:
                b = self.bid.get(s.get("batch_id")) or {}
                if b and nb(b.get("batch_no")) in sheet_bnos:
                    continue
                zero.append({"_key": f"{k}|zero|{s['id']}", "p": p, "stock": s, "batch": b or None,
                             "before": int(s["quantity"]), "cost": num(s.get("average_cost")) or cost or 0,
                             "note": f"{cfg['label']}: batch not on the count sheet — set to 0"})
        self.report["plan"].update(
            batches_new=len(new_b), units_new=sum(x["qty"] for x in new_b), batches_set=len(set_b),
            batches_unchanged=same, other_batches_zeroed=len(zero), units_zeroed=sum(z["before"] for z in zero),
            items_without_quantity=len(untouched), stock_take_lines=len(lines),
            units_counted=sum(x["qty"] for x in lines), batch_numbers_near_match=len(near_kept))
        self.report["detail"].update(
            items_without_quantity=untouched, batch_numbers_near_match=near_kept,
            batches_set=[f"{x['p'].get('generic_name')} {x['bno']}: {x['before']} → {x['qty']}" for x in set_b],
            batches_zeroed=[f"{z['p'].get('generic_name')} {(z['batch'] or {}).get('batch_no')}: {z['before']} → 0"
                            for z in zero],
            expiry_problems=[f"row {b['row']} {i['name']} {b['batch_no']}: {b['expiry_problem']}"
                             for i in self.items for b in i["batches"] if b["expiry_problem"]])
        if self.dry_run:
            for kind, xs in (("new batch", new_b), ("set", set_b), ("zero", zero)):
                for x in xs[:3]:
                    log.info("  dry-run %s: %s", kind,
                             json.dumps({k: v for k, v in x.items() if k not in ("p", "batch", "rows", "stock")}, default=str))
            return

        ref = f"{cfg['prefix']}-MAIN-{date.replace('-', '')}"
        ok, take_id, err = self.insert("stock_take", {
            "uuid": gid(self.target, "stock_take", date), "reference_number": ref, "store_id": st,
            "initiated_by": user, "completed_by": user, "status": "completed", "started_at": self.when,
            "completed_at": self.when, "notes": f"{cfg['label']}: {self.data['file']} ({len(lines)} lines) "
                                                f"loaded by drug_stock_to_v3"})
        if not ok or not take_id:
            raise SystemExit(f"stock take {ref}: {err or 'no id returned'}")
        self.report["result"]["stock_take"] = take_id

        def mv(x, b_id, bno, expiry, delta, opening, closing):
            return {"uuid": gid(self.target, "movement", f"{x['_key']}|{opening}>{closing}"), "product_id": x["p"]["id"],
                    "batch_id": b_id, "batch_no": bno, "expiry_date": expiry, "type": "adjustment",
                    "direction": "in" if delta >= 0 else "out", "to_store_id": st if delta >= 0 else None,
                    "from_store_id": None if delta >= 0 else st, "quantity": abs(delta), "unit_cost": x["cost"],
                    "total_value": round(abs(delta) * x["cost"], 2), "opening_balance": opening, "closing_balance": closing,
                    "reference_type": "stock_take", "reference_id": take_id,
                    "idempotency_key": f"drug_stock_to_v3:{self.org}:{x['_key']}:{opening}>{closing}"[:191],
                    "user_id": user, "done_by": user, "reason": "Stock take", "notes": x["note"],
                    "created_at": self.when, "updated_at": self.when}

        def put_new(x):
            ok, b_id, err = self.insert("batch", {
                "uuid": gid(self.target, "batch", x["_key"]), "batch_no": x["bno"][:255], "product_id": x["p"]["id"],
                "store_id": st, "expiry_date": x["expiry"], "cost_price": x["cost"], "selling_price": x["sell"],
                "quantity_received": x["qty"], "quantity_available": x["qty"], "status": "active", "notes": x["note"]})
            if not ok or not b_id:
                return False, None, f"batch: {err or 'no id returned'}"
            ok, s_id, err = self.insert("stock", {
                "uuid": gid(self.target, "stock", x["_key"]), "product_id": x["p"]["id"], "store_id": st,
                "batch_id": b_id, "quantity": x["qty"], "available_quantity": x["qty"], "reserved_quantity": 0,
                "average_cost": x["cost"], "total_value": round(x["qty"] * x["cost"], 2), "selling_price": x["sell"],
                "insurance_price": x["ins"]})
            if not ok:
                return False, None, f"stock (batch {b_id} made): {err}"
            x["batch_id"] = b_id
            if x["qty"]:
                ok, _, err = self.insert("stock_movement", mv(x, b_id, x["bno"], x["expiry"], x["qty"], 0, x["qty"]))
            return ok, b_id, None if ok else f"movement (batch {b_id}, stock {s_id} made): {err}"

        def put_set(x):
            b = x["batch"]
            data = {"quantity_available": x["qty"], "status": "active" if x["qty"] else "depleted"}
            if x["expiry"] and local_date(b.get("expiry_date")) != x["expiry"]:
                data["expiry_date"] = x["expiry"]
            ok, _, err = self.update("batch", b["id"], data)
            if not ok:
                return False, None, f"batch: {err}"
            rows = sorted(x["rows"], key=lambda s: s["id"])
            if rows:
                for i, s in enumerate(rows):         # one row carries the count, any extra row → 0
                    q = x["qty"] if i == 0 else 0
                    ok, _, err = self.update("stock", s["id"], {
                        "quantity": q, "available_quantity": max(q - int(s.get("reserved_quantity") or 0), 0),
                        "total_value": round(q * x["cost"], 2), "selling_price": x["sell"], "insurance_price": x["ins"]})
                    if not ok:
                        return False, None, f"stock {s['id']}: {err}"
            else:
                ok, _, err = self.insert("stock", {
                    "uuid": gid(self.target, "stock", x["_key"]), "product_id": x["p"]["id"], "store_id": st,
                    "batch_id": b["id"], "quantity": x["qty"], "available_quantity": x["qty"], "reserved_quantity": 0,
                    "average_cost": x["cost"], "total_value": round(x["qty"] * x["cost"], 2),
                    "selling_price": x["sell"], "insurance_price": x["ins"]})
                if not ok:
                    return False, None, f"stock: {err}"
            x["batch_id"] = b["id"]
            if x["qty"] != x["before"]:
                ok, _, err = self.insert("stock_movement", mv(x, b["id"], b["batch_no"], x["expiry"] or local_date(b.get("expiry_date")),
                                                              x["qty"] - x["before"], x["before"], x["qty"]))
                return ok, b["id"], None if ok else f"movement: {err}"
            return True, b["id"], None

        def put_zero(z):
            s, b = z["stock"], z["batch"]
            ok, _, err = self.update("stock", s["id"], {"quantity": 0, "available_quantity": 0, "total_value": 0})
            if not ok:
                return False, None, f"stock: {err}"
            if b:
                left = sum(int(r.get("quantity") or 0) for r in self.stock
                           if r.get("batch_id") == b["id"] and r["id"] != s["id"])
                ok, _, err = self.update("batch", b["id"], {"quantity_available": left, "status": "active" if left else "depleted"})
                if not ok:
                    return False, None, f"batch: {err}"
            return self.insert("stock_movement", mv(z, (b or {}).get("id"), (b or {}).get("batch_no"),
                                                    local_date((b or {}).get("expiry_date")), -z["before"], z["before"], 0))

        self.many("batches_new", new_b, put_new)
        self.many("batches_set", set_b, put_set)
        self.many("batches_zeroed", zero, put_zero)
        take_lines = [{"_key": x["_key"], "uuid": gid(self.target, "stock_take_item", x["_key"]),
                       "stock_take_id": take_id, "product_id": x["p"]["id"], "batch_id": x.get("batch_id"),
                       "counted_batch_no": x["bno"][:255], "counted_expiry_date": x["expiry"],
                       "expected_quantity": x["before"], "counted_quantity": x["qty"],
                       "variance": x["qty"] - x["before"], "is_reconciled": 1, "is_unexpected": 0,
                       "notes": x["note"][:1000]} for x in lines]
        self.many("stock_take_items", take_lines, lambda l: self.insert("stock_take_item", l))

    # ── base rows: a product shows in a store only through a base stock row (batch NULL, qty 0) ──
    def run_store_rows(self):
        ids = {p["id"] for p in self.product_of.values() if p.get("id")}
        have = {(s["product_id"], s["store_id"]) for s in self.stock if not s.get("batch_id")}
        rows = [{"_key": f"{pid}|{st}", "uuid": gid(self.target, "base_row", f"{pid}|{st}"), "product_id": pid,
                 "store_id": st, "quantity": 0, "available_quantity": 0, "reserved_quantity": 0, "average_cost": 0,
                 "total_value": 0} for pid in sorted(ids) for st in self.cfg["base_row_stores"] if (pid, st) not in have]
        self.report["plan"]["base_rows_missing"] = len(rows)
        self.many("base_rows", rows, lambda r: self.insert("stock", r))

    # ── verify: every sheet line against V3, read back ──
    def run_verify(self):
        by_uuid = {str(p.get("uuid")).lower(): p for p in self.products if p.get("uuid")}
        res, issues = Counter(), []
        rows_of = defaultdict(list)
        for s in self.stock:
            if s["store_id"] == self.store and (s.get("quantity") or 0) > 0:
                rows_of[s["product_id"]].append(s)
        sb = defaultdict(dict)
        for b in self.batches:
            if b["store_id"] == self.store:
                sb[b["product_id"]].setdefault(nb(b["batch_no"]), b)
        names = Counter()
        for k, its in self.groups.items():
            p0 = self.product_of.get(k) or by_uuid.get(gid(self.target, "product", k))
            p = self.pid.get((p0 or {}).get("id")) if p0 else None
            if not p:
                res["no product"] += 1
                issues.append(f"row {its[0]['row']} {its[0]['name']}: no product")
                continue
            names[key(p.get("generic_name"))] += 1
            prob = []
            if p.get("generic_name") != its[0]["name"][:255]:
                prob.append(f"name {p.get('generic_name')!r}")
            for f, v in (("selling_price", self.price(its, "sell")), ("insurance_price", self.price(its, "ins")),
                         ("cost_price", self.price(its, "cost"))):
                if v is not None and num(p.get(f)) != v:
                    prob.append(f"{f} {p.get(f)}≠{v}")
            if any(i["qty"] is not None for i in its):
                gen = lambda b: b["batch_no"] or f"{self.cfg['prefix']}-{self.data['count_date'].replace('-', '')}-R{b['row']}"
                want = {nb(gen(b)) for it in its for b in it["batches"]}
                for it in its:
                    for j, b in enumerate(it["batches"]):
                        q = (it["qty"] or 0) if j == 0 else 0
                        bno = gen(b)
                        vb = sb[p["id"]].get(nb(bno)) or (near_batch(sb[p["id"]], nb(bno), want) if b["batch_no"] else None)
                        if vb:
                            want.add(nb(vb["batch_no"]))
                        held = sum(int(s["quantity"]) for s in rows_of[p["id"]] if s.get("batch_id") == (vb or {}).get("id"))
                        if not vb:
                            prob.append(f"batch {bno} missing")
                        elif held != q:
                            prob.append(f"batch {bno} holds {held}≠{q}")
                        elif b["expiry"] and local_date(vb.get("expiry_date")) != b["expiry"]:
                            prob.append(f"batch {bno} expiry {local_date(vb.get('expiry_date'))}≠{b['expiry']}")
                extra = [s for s in rows_of[p["id"]] if nb((self.bid.get(s.get("batch_id")) or {}).get("batch_no")) not in want]
                if extra:
                    prob.append(f"{sum(int(s['quantity']) for s in extra)} units on batches not on the sheet")
            if prob:
                res["mismatch"] += 1
                issues.append(f"row {its[0]['row']} {its[0]['name']}: {'; '.join(prob)}")
            else:
                res["ok"] += 1
        self.report["plan"]["verify"] = dict(res)
        self.report["detail"]["verify_issues"] = issues
        log.info("verify: %s", dict(res))
        for i in issues:
            log.info("  verify issue: %s", i)
        if res["mismatch"] or res["no product"]:
            self.report["failures"].append({"kind": "verify", "key": "sheet", "error": f"{len(issues)} issues (see log)"})

    def run_expiry(self):
        """only the sheet's expiry dates on the store's batches (exact or misread-char batch no.) — no quantities, so it
        is safe after the facility has started dispensing (2026-11-27: 4 batches whose count already matched kept
        their old expiry)"""
        sb = defaultdict(dict)
        for b in self.batches:
            if b["store_id"] == self.store:
                sb[b["product_id"]].setdefault(nb(b["batch_no"]), b)
        todo = []
        for k, its in self.groups.items():
            p = self.product_of.get(k)
            if not p or not p.get("id"):
                continue
            want = {nb(b["batch_no"]) for it in its for b in it["batches"] if b["batch_no"]}
            for it in its:
                for b in it["batches"]:
                    if not b["batch_no"] or not b["expiry"]:
                        continue
                    vb = sb[p["id"]].get(nb(b["batch_no"])) or near_batch(sb[p["id"]], nb(b["batch_no"]), want)
                    if vb and local_date(vb.get("expiry_date")) != b["expiry"]:
                        todo.append({"_key": f"{k}|{vb['batch_no']}", "id": vb["id"],
                                     "before": local_date(vb.get("expiry_date")), "data": {"expiry_date": b["expiry"]}})
        self.report["plan"]["expiry_updates"] = len(todo)
        self.report["detail"]["expiry_updates"] = [f"{x['_key']}: {x['before']} → {x['data']['expiry_date']}" for x in todo]
        self.many("expiry_updates", todo, lambda x: self.update("batch", x["id"], x["data"]))

    def load(self) -> dict:
        self.read_inventory()
        self.run_products()
        if not self.dry_run and set(self.steps) & {"stock", "store_rows", "verify"}:
            self.read_inventory()
            self.product_of = {k: self.pid.get(p.get("id"), p) for k, p in self.product_of.items()}
        if "stock" in self.steps:
            self.run_stock()
        if "expiry" in self.steps:
            if not self.dry_run and "stock" in self.steps:
                self.read_inventory()
            self.run_expiry()
        if "store_rows" in self.steps:
            self.run_store_rows()
        if "verify" in self.steps and not self.dry_run:
            self.read_inventory()
            self.run_verify()
        log.info("plan: %s", json.dumps(self.report["plan"], default=str))
        log.info("result: %s", json.dumps(self.report["result"], default=str))
        return self.report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("xlsx")
    c.add_argument("--target", required=True, choices=list(TARGETS))
    lo = sub.add_parser("load")
    lo.add_argument("--target", required=True, choices=list(TARGETS))
    lo.add_argument("--dry-run", action="store_true")
    lo.add_argument("--workers", type=int, default=8)
    lo.add_argument("--steps", nargs="*", choices=STEPS, default=STEPS)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "convert":
        convert(a.xlsx, a.target)
        return
    report = Run(a.target, a.dry_run, a.workers, a.steps).load()
    out = _HERE / "output" / "drug_stock_to_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.target}-{time.strftime('%Y%m%d_%H%M%S')}{'-dry' if a.dry_run else ''}.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log.info("report: %s", path)
    sys.exit(1 if report.get("failures") else 0)


if __name__ == "__main__":
    main()
