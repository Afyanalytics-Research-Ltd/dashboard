"""
rate_cards_to_v3.py — a hospital's per-payer rate cards (Cash, SHA, each insurer) + its insurer/scheme list + item
cost prices → V3, through the model gateway.

First use: Kerugoya Medical Centre, production org 357 / facility 367 (target kerugoya_prod), from kerugoya.zip
(2026-11-24): "KMC - <payer> Rates.xls[x]" (sheets Lab / Radiology / Procedure Rates: Item Code, Category, Item Name,
Charges), "KMC_Insurance & Schemes.xlsx" (Insurance, Scheme, Code), "KMC - Pharmacy Rates.xls" (Item Name, Fixed
Price) and "KMC Item Cost Price.xlsx" (ItemName, CostPrice). `convert` turns them into
dags/jsons/rate_cards/<target>.json; Airflow only reads that file.

The org was seeded (2026-10/11) from the hospital's own catalogue, so this reconciles rather than bulk-loads:

  procedures      each sheet item → the org procedure with its item code (else same name; else a new procedure with
                  the sheet's code, in the V3 category the org's matched items of that sheet category mostly use).
                  Facility price: cash_charge = the Cash card, insurance_charge = the "Other Insurance" card (the
                  price every insurer without a card of its own pays). A missing card price never blanks a V3 price.
  schemes         each sheet scheme → core insurance_schemes under its insurer's existing V3 company (COMPANIES);
                  a scheme already in V3 (SCHEME_ALIAS, or same code / name under that company) is reused, not copied.
  scheme_pricing  each payer with its own card (PAYER_INSURERS) → core scheme_pricing on every scheme of that
                  insurer, for the items whose card price differs from the default insurance price.
  product_prices  facility products (matched by name) get the cost sheet's cost_price and the pharmacy sheet's
                  selling_price where they differ. Products are not created here.

Every new row carries a stable uuid5 (target + kind + sheet key), so a re-run finds and updates it instead of
duplicating. The V3 gateway has no delete: check the dry run before posting.

    python rate_cards_to_v3.py convert <folder of the xls files> --target kerugoya_prod
    python rate_cards_to_v3.py load --target kerugoya_prod --dry-run
    python rate_cards_to_v3.py load --target kerugoya_prod --limit 3          # canary: 3 changes per kind
    python rate_cards_to_v3.py load --target kerugoya_prod [--steps procedures schemes ...]
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

log = logging.getLogger("rate_cards_to_v3")

_HERE = Path(__file__).resolve().parent
DATA_DIRS = (_HERE.parent / "dags" / "jsons" / "rate_cards", Path("/opt/airflow/dags/jsons/rate_cards"))
NS = uuid.UUID("0c1f6f0e-3b0a-4d55-9a3e-7c2b8e5d4a21")
STEPS = ["procedures", "schemes", "scheme_pricing", "product_prices"]

# target → what its rate cards need; the V3 login is AFYA_<TARGET>_* / Airflow connection afya_v3_<target>
TARGETS = {
    "kerugoya_prod": {
        "label": "Kerugoya Medical Centre rate cards (2026-11)",
        "default_payer": "Other",          # the card insurance_charge holds
        # sheet insurer → existing V3 core insurance_companies id (org 357, checked 2026-11-24)
        "companies": {
            "AAR INSURANCE COMPANY LIMITED": 2818, "APA INSURANCE COMPANY LTD": 2824, "CYKA MANPOWER LTD": 2953,
            "FIRST ASSURANCE COMPANY LTD": 2954, "GA INSURANCE COMPANY LTD": 2957, "HERITAGE INSURANCE": 2959,
            "JUBILEE INSURANCE COMPANY LIMITED": 2819, "KENGEN": 2960, "KENYA POWER AND LIGHTING COMPANY": 2967,
            "KENYAN ALLIANCE": 2971, "LASER INSURANCE BROKERS": 2970, "LIAISON INSURANCE": 2969,
            "MADISON INSURANCE": 2822, "MINET INSURANCE LTD": 2966, "NIS RETIREES": 2965, "NSSF": 2964,
            "PACIS INSURANCE": 2962, "MUA KENYA": 2961, "SANLAM INSURANCE": 2958, "TAKAFUL INSURANCE": 2956,
            "OLD MUTUAL GENERAL INSURANCE": 2955, "CIC INSURANCE": 2820, "KMC CREDIT NOTE": 2976,
            "KENBRIGHT HEALTHCARE ADMINISTRATORS LTD": 2975, "ZHONGHAO OVERSEAS CONSTRUCTION ENG CO LTD": 2974,
            "CAREPAY": 2973, "SHA-SHIF": 2817, "SHA-PHC": 2817, "SHA-POMSF": 2817, "KCB BANCASSURANCE": 2972,
            "MTIBA-JUBILEE": 2968, "EQUITY GENERAL INSURANCE KENYA": 2963, "BRITAM INSURANCE": 2821,
        },
        # sheet scheme code → scheme already in V3 (SHA's SHIF and POMSF schemes are in use there)
        "scheme_alias": {"02-1100-451": 28938, "02-1100-468": 33078},
        # rate card → the sheet insurers whose schemes it prices (MTIBA-JUBILEE: own company, default card)
        "payer_insurers": {
            "AAR": ["AAR INSURANCE COMPANY LIMITED"], "APA": ["APA INSURANCE COMPANY LTD"],
            "Britam": ["BRITAM INSURANCE"], "Jubilee": ["JUBILEE INSURANCE COMPANY LIMITED"],
            "Kenya Alliance": ["KENYAN ALLIANCE"], "Liaison": ["LIAISON INSURANCE"], "Pacis": ["PACIS INSURANCE"],
            "SHA": ["SHA-SHIF", "SHA-PHC", "SHA-POMSF"],
        },
        # payers whose WHOLE card is written as scheme prices (not only where it differs from the default), on the sheet's
        # schemes AND every other V3 scheme of these companies (SHA's template schemes Standard / Enhanced / UHC too):
        # Dennis (2026-11-24) found SHA rates not matching where the system fell back to the default price
        "full_card_payers": {"SHA": [2817]},
        "fallback_category": "Others",
        # sheet category → V3 category name, where the vote of matched items is wrong (COVID PCR voted "MRI")
        "category_override": {"molecular biology": "Lab", "histo pathology": "Lab", "histopathology": "Lab"},
        "skip_names": {"test"},            # placeholder rows LAB001 / LAB002
    },
}
# Kerugoya Town Clinic (facility 368) mirrors Kerugoya Medical Centre (user, 2026-11-24): the same facility prices.
# Only the procedures step — schemes / scheme prices are org-wide and already loaded by kerugoya_prod (a run here
# would make duplicate schemes under this target's uuids).
TARGETS["kerugoya_town_prod"] = {**TARGETS["kerugoya_prod"], "data_target": "kerugoya_prod", "only_steps": ["procedures"],
                                 "label": "Kerugoya Town Clinic prices (mirror of Kerugoya Medical Centre, 2026-11)"}


def key(v) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(str(v or "")).lower()).strip()


def clean(v) -> str:
    s = "" if v is None else str(v)
    return "" if s.lower() == "nan" else re.sub(r"\s+", " ", html.unescape(s)).strip()


def money(v) -> float | None:
    """sheet price → float; '800/HR' → 800; blanks, text and negatives (package credits) → None"""
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", clean(v).replace(",", ""))
    if not m:
        return None
    x = round(float(m.group(1)), 2)
    return x if x >= 0 else None


def num(v) -> float:
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
def _frames(path: Path):
    import pandas as pd
    for sheet, df in pd.read_excel(path, sheet_name=None, header=None, dtype=object).items():
        df = df.dropna(how="all")
        if df.empty:
            continue
        head = next((i for i in df.index if {"item name", "itemname", "item", "bed name", "insurance"}
                     & {clean(x).lower() for x in df.loc[i]}), None)
        if head is None:
            continue
        cols = [clean(c).lower() for c in df.loc[head]]
        body = df.loc[df.index > head]
        yield sheet, [dict(zip(cols, r)) for r in body.itertuples(index=False, name=None)]


def convert(folder: str, target: str) -> Path:
    folder = Path(folder)
    items: dict[str, dict] = {}
    pharmacy, costs, schemes, sources = [], [], [], []
    for f in sorted(folder.iterdir()):
        n = f.name
        if not f.suffix.lower().startswith(".xls"):
            continue
        sources.append(n)
        if n.startswith("KMC - ") and "Rates" in n and "Pharmacy" not in n:
            payer = re.sub(r"^KMC - | (Insurance )?(Procedure )?Rates.*$", "", n)
            for sheet, rows in _frames(f):
                for r in rows:
                    if "bed name" in r:          # bed charges: per bed, not a procedure price — not loaded
                        continue
                    code, name = clean(r.get("item code")).upper(), clean(r.get("item name"))
                    if not code or not name:
                        continue
                    it = items.setdefault(code, {"code": code, "name": name, "category": clean(r.get("category")),
                                                 "sheet": re.sub(r"^\S+ ", "", sheet), "rates": {}})
                    if payer == "Cash" or not it.get("_from_cash"):
                        it.update(name=name, category=clean(r.get("category")) or it["category"])
                        it["_from_cash"] = payer == "Cash"
                    price = money(r.get("charges") if "charges" in r else r.get("unit charges"))
                    if price is not None:
                        it["rates"][payer] = price
        elif "Pharmacy Rates" in n:
            for _, rows in _frames(f):
                pharmacy += [{"code": clean(r.get("item code")), "name": clean(r.get("item name")),
                              "price": money(r.get("fixed price"))} for r in rows if clean(r.get("item name"))]
        elif "Cost Price" in n and "Dental" not in n:
            for sheet, rows in _frames(f):
                costs += [{"code": clean(r.get("itemcode")), "name": clean(r.get("itemname")), "sheet": sheet,
                           "cost": money(r.get("costprice"))} for r in rows if clean(r.get("itemname"))]
        elif "Insurance & Schemes" in n:
            insurer = None
            for _, rows in _frames(f):
                for r in rows:
                    insurer = clean(r.get("insurance")) or insurer
                    if clean(r.get("scheme")):
                        schemes.append({"insurer": insurer, "name": clean(r.get("scheme")), "code": clean(r.get("code"))})
    for it in items.values():
        it.pop("_from_cash", None)
    out = DATA_DIRS[0] / f"{target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sources": sources, "items": sorted(items.values(), key=lambda x: x["code"]),
                               "schemes": schemes, "pharmacy": pharmacy, "costs": costs}, indent=1), encoding="utf-8")
    log.info("%d items, %d schemes, %d pharmacy prices, %d cost prices → %s", len(items), len(schemes), len(pharmacy),
             len(costs), out)
    return out


# ── load ─────────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, target: str, dry_run: bool = False, limit: int = 0, workers: int = 8, steps=None):
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
        self.steps = [s for s in STEPS if s in (steps or STEPS) and s in self.cfg.get("only_steps", STEPS)]
        self.c = V3Client(target)
        self.c.send_facility_header = True          # prices land in this facility
        self.c.token()
        self.org, self.fac = self.c.organization_id, self.c.facility_id
        if not self.org or not self.fac:
            raise SystemExit(f"{target}: organization_id / facility_id not set")
        self.data = json.loads(data_path(self.cfg.get("data_target", target)).read_text(encoding="utf-8"))
        skip = {key(s) for s in self.cfg.get("skip_names", ())}
        self.items = [i for i in self.data["items"] if key(i["name"]) not in skip and i["rates"]]
        log.info("%r: %d items, %d schemes, steps %s", self.c, len(self.items), len(self.data["schemes"]), self.steps)
        self.report: dict = {"target": target, "org": self.org, "facility": self.fac, "dry_run": dry_run,
                             "limit": limit, "steps": self.steps, "plan": {}, "result": {}, "failures": [],
                             "detail": {}}

    # ── posting ──
    def _ok(self, r) -> tuple[bool, str | None]:
        ok = r.ok and (r.json() if r.content else {}).get("success", True)
        return ok, None if ok else f"{r.status_code} {r.text[:300]}"

    def _many(self, kind: str, items: list, fn) -> None:
        if self.limit:
            items = items[:self.limit]
        res, done = Counter(), 0
        if self.dry_run:
            for x in items[:3]:
                log.info("  dry-run %s: %s", kind, json.dumps(x, default=str)[:600])
            return

        def one(x):
            try:
                ok, err = fn(x)
            except Exception as e:
                ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
            return x, ok, err
        with ThreadPoolExecutor(self.workers) as ex:
            for x, ok, err in ex.map(one, items):
                res["ok" if ok else "failed"] += 1
                done += 1
                if not ok:
                    self.report["failures"].append({"kind": kind, "key": x.get("_key"), "error": err})
                    log.error("%s %s: %s", kind, x.get("_key"), err)
                if done % 1000 == 0:
                    log.info("  %s: %d / %d posted (%s)", kind, done, len(items), dict(res))
        self.report["result"][kind] = dict(res)
        log.info("%s: %s", kind, dict(res))

    def insert(self, svc: str, alias: str, data: dict):
        return self._ok(self.c.insert(svc, alias, {k: v for k, v in data.items()
                                                   if not k.startswith("_") and v is not None}, match_on="uuid"))

    def update(self, svc: str, alias: str, rid: int, data: dict):
        return self._ok(self.c.post(svc, {"action": "update", "model": alias, "destination_tenant_id": self.org,
                                          "ids": [rid], "data": data}, timeout=120))

    def live(self, svc: str, alias: str, workers: int = 8) -> list[dict]:
        return [r for r in self.c.read_all(svc, alias, workers=workers) if not r.get("deleted_at")]

    # ── procedures + facility prices ──
    def read_procedures(self):
        self.cats = self.live("evaluation", "procedure_categories", 1)
        self.procs = self.live("evaluation", "procedures")
        self.prices = {}
        for p in self.live("evaluation", "procedure_prices"):
            if p.get("facility_id") == self.fac:
                self.prices.setdefault(p["evaluation_procedure_id"], p)
        log.info("org %s: %d categories, %d procedures, %d prices at facility %s", self.org, len(self.cats),
                 len(self.procs), len(self.prices), self.fac)

    def match(self) -> dict[str, dict]:
        by_uuid = {str(p.get("uuid")).lower(): p for p in self.procs if p.get("uuid")}
        by_code, by_name = defaultdict(list), defaultdict(list)
        for p in sorted(self.procs, key=lambda p: (not p.get("active", True), p["id"] not in self.prices, p["id"])):
            if p.get("code"):
                by_code[clean(p["code"]).upper()].append(p)
            by_name[key(p["name"])].append(p)
        got, used, how = {}, set(), Counter()
        for kind, find in (("uuid", lambda i: [by_uuid.get(gid(self.target, "procedure", i["code"]))]),
                           ("code", lambda i: by_code.get(i["code"], [])),
                           ("name", lambda i: by_name.get(key(i["name"]), []))):
            for i in self.items:
                if i["code"] in got:
                    continue
                p = next((p for p in find(i) if p and p["id"] not in used), None)
                if p:
                    got[i["code"]], how[kind] = p, how[kind] + 1
                    used.add(p["id"])
        self.report["plan"]["procedures_matched_by"] = dict(how)
        return got

    def categories(self, matched: dict[str, dict]) -> dict[str, int]:
        """sheet category → the V3 category the org's matched items of that sheet category mostly sit in, else the
        same-named V3 category, else the fallback."""
        by_name = {key(c["name"]): c["id"] for c in self.cats}
        votes = defaultdict(Counter)
        for i in self.items:
            p = matched.get(i["code"])
            if p and p.get("category"):
                votes[key(i["category"])][p["category"]] += 1
        override = {key(k): by_name.get(key(v)) for k, v in self.cfg.get("category_override", {}).items()}
        out = {}
        for d in sorted({key(i["category"]) for i in self.items}):
            out[d] = (override.get(d) or votes[d].most_common(1)[0][0] if override.get(d) or votes.get(d) else
                      by_name.get(d) or by_name.get(key(self.cfg["fallback_category"])))
        return out

    def plan_procedures(self) -> dict:
        self.read_procedures()
        self.matched = matched = self.match()
        cat = self.categories(matched)
        cat_name = {c["id"]: c["name"] for c in self.cats}
        taken = Counter(key(p["name"]) for p in self.procs)
        dp = self.cfg["default_payer"]
        new_procs, price_new, price_upd, same, inactive = [], [], [], 0, []
        for i in self.items:
            rates = i["rates"]
            p = matched.get(i["code"])
            if p is None:
                name = i["name"] if not taken[key(i["name"])] else f"{i['name']} ({i['code']})"
                taken[key(name)] += 1
                p = {"uuid": gid(self.target, "procedure", i["code"]), "code": i["code"], "name": name[:255],
                     "category": cat[key(i["category"])], "active": 1,
                     "description": f"{self.cfg['label']}: {i['sheet']} · {i['category']}", "_key": i["code"]}
                new_procs.append(p)
            elif not p.get("active", True):
                inactive.append({"code": i["code"], "id": p["id"], "name": p["name"]})
            pr = self.prices.get(p.get("id")) if p.get("id") else None
            want = {}
            if rates.get("Cash") is not None:
                want["cash_charge"] = rates["Cash"]
            if rates.get(dp) is not None:
                want["insurance_charge"] = rates[dp]
            if pr is None:
                fill = rates.get("Cash", rates.get(dp, max(rates.values())))
                price_new.append({"_key": i["code"], "_procedure_uuid": p["uuid"], "evaluation_procedure_id": p.get("id"),
                                  "uuid": gid(self.target, "price", f"{i['code']}|{self.fac}"), "facility_id": self.fac,
                                  "cash_charge": want.get("cash_charge", fill),
                                  "insurance_charge": want.get("insurance_charge", fill),
                                  "charge_insurance": 1, "precharge": 0, "status": 1})
                continue
            before = {k: num(pr.get(k)) for k in want}
            if before == want:
                same += 1
                continue
            price_upd.append({"_key": i["code"], "id": pr["id"], "procedure_id": p["id"], "name": p["name"],
                              "before": before, "data": want})
        self.report["detail"].update(new_procedures=new_procs, price_inserts=price_new, price_updates=price_upd,
                                     matched_but_inactive=inactive,
                                     category_map={d: f"{c} {cat_name.get(c)}" for d, c in cat.items()})
        self.report["plan"].update(items=len(self.items), new_procedures=len(new_procs), price_inserts=len(price_new),
                                   price_updates=len(price_upd), prices_unchanged=same,
                                   matched_but_inactive=len(inactive))
        return {"new": new_procs, "ins": price_new, "upd": price_upd}

    def run_procedures(self):
        plan = self.plan_procedures()
        self._many("new_procedures", plan["new"], lambda p: self.insert("evaluation", "procedures", p))
        if plan["new"] and not self.dry_run:
            ids = {str(p.get("uuid")).lower(): p for p in self.live("evaluation", "procedures") if p.get("uuid")}
            for pr in plan["ins"]:
                pr["evaluation_procedure_id"] = pr["evaluation_procedure_id"] or (ids.get(pr["_procedure_uuid"]) or {}).get("id")
            for p in plan["new"]:                   # scheme pricing needs the new procedures too
                if ids.get(p["uuid"]):
                    self.matched[p["_key"]] = ids[p["uuid"]]
        ins = [pr for pr in plan["ins"] if pr["evaluation_procedure_id"] or self.dry_run]
        if not self.dry_run and not self.limit and len(ins) < len(plan["ins"]):
            self.report["failures"] += [{"kind": "price_inserts", "key": pr["_key"], "error": "procedure not created"}
                                        for pr in plan["ins"] if not pr["evaluation_procedure_id"]]
        self._many("price_inserts", ins, lambda pr: self.insert("evaluation", "procedure_prices", pr))
        self._many("price_updates", plan["upd"],
                   lambda u: self.update("evaluation", "procedure_prices", u["id"], u["data"]))

    # ── insurance schemes ──
    def run_schemes(self):
        companies = self.cfg["companies"]
        missing = sorted({s["insurer"] for s in self.data["schemes"]} - set(companies))
        if missing:
            raise SystemExit(f"insurers with no V3 company in TARGETS['{self.target}']['companies']: {missing}")
        have = self.live("core", "insurance_schemes", 1)
        live_cos = {c["id"] for c in self.live("core", "insurance_companies", 1)}
        gone = sorted({v for v in companies.values()} - live_cos)
        if gone:
            raise SystemExit(f"V3 companies {gone} no longer exist in org {self.org}")
        by_uuid = {str(s.get("uuid")).lower(): s for s in have if s.get("uuid")}
        by_id = {s["id"]: s for s in have}
        by_code = {clean(s.get("code")): s for s in have if s.get("code")}
        by_name = {(s.get("company_id"), key(s["name"])): s for s in have}
        self.scheme_ids, new, reused = {}, [], Counter()
        for s in self.data["schemes"]:
            co = companies[s["insurer"]]
            hit = (by_uuid.get(gid(self.target, "scheme", s["code"])) or by_id.get(self.cfg["scheme_alias"].get(s["code"]))
                   or by_code.get(s["code"]) or by_name.get((co, key(s["name"]))))
            if hit:
                self.scheme_ids[s["code"]] = hit["id"]
                reused["ours" if str(hit.get("uuid")).lower() == gid(self.target, "scheme", s["code"]) else "existing"] += 1
                continue
            new.append({"_key": s["code"], "uuid": gid(self.target, "scheme", s["code"]), "company_id": co,
                        "name": s["name"][:255], "code": s["code"], "status": "active",
                        "description": f"{self.cfg['label']}: {s['insurer']}"})
        self.report["plan"].update(schemes=len(self.data["schemes"]), schemes_new=len(new),
                                   schemes_found=dict(reused))
        self.report["detail"]["new_schemes"] = new
        if "schemes" in self.steps:
            self._many("schemes", new, lambda s: self.insert("core", "insurance_schemes", s))
            if new and not self.dry_run:
                ids = {str(s.get("uuid")).lower(): s["id"] for s in self.live("core", "insurance_schemes", 1)}
                for s in new:
                    if ids.get(s["uuid"]):
                        self.scheme_ids[s["_key"]] = ids[s["uuid"]]

    # ── per-scheme prices ──
    def run_scheme_pricing(self):
        if not hasattr(self, "matched"):
            self.read_procedures()
            self.matched = self.match()
        if not hasattr(self, "scheme_ids"):
            self.run_schemes()
        dp = self.cfg["default_payer"]
        existing = {}
        for r in self.live("core", "scheme_pricing"):
            existing.setdefault((r.get("scheme_id"), r.get("priceable_type"), r.get("priceable_id")), r)
        rows, upd, same, held, per_payer = [], [], 0, Counter(), Counter()
        full = self.cfg.get("full_card_payers", {})
        all_schemes = [s for s in self.live("core", "insurance_schemes", 1)]
        for payer, insurers in self.cfg["payer_insurers"].items():
            schemes = [s for s in self.data["schemes"] if s["insurer"] in insurers]
            if payer in full:                        # + every other V3 scheme of the payer's companies
                have = {self.scheme_ids.get(s["code"]) for s in schemes}
                for v in all_schemes:
                    if v.get("company_id") in full[payer] and v["id"] not in have:
                        code = f"V3-{v['id']}"
                        self.scheme_ids[code] = v["id"]
                        schemes.append({"insurer": f"company {v['company_id']}", "name": v["name"], "code": code})
            for i in self.items:
                price = i["rates"].get(payer)
                if price is None or (price == i["rates"].get(dp) and payer not in full):
                    continue
                p = self.matched.get(i["code"])
                for s in schemes:
                    sid = self.scheme_ids.get(s["code"])
                    if not p or not sid:
                        held["procedure" if not p else "scheme"] += 1
                        if not self.dry_run:
                            continue
                    pid = p.get("id") if p else None
                    std = i["rates"].get("Cash", i["rates"].get(dp, price))
                    data = {"scheme_id": sid, "priceable_type": "procedure", "priceable_id": pid,
                            "priceable_code": i["code"], "standard_price": std, "scheme_price": price,
                            "is_covered": 1, "status": "active"}
                    ex = existing.get((sid, "procedure", pid)) if sid and pid else None
                    if ex:
                        if num(ex.get("scheme_price")) == price and ex.get("is_covered") in (1, True):
                            same += 1
                        else:
                            upd.append({"_key": f"{s['code']}|{i['code']}", "id": ex["id"],
                                        "before": num(ex.get("scheme_price")),
                                        "data": {"scheme_price": price, "standard_price": std, "is_covered": 1}})
                        continue
                    per_payer[payer] += 1
                    rows.append({"_key": f"{s['code']}|{i['code']}", "uuid": gid(self.target, "scheme_price",
                                                                                 f"{s['code']}|{i['code']}"), **data})
        self.report["plan"].update(scheme_prices_new=len(rows), scheme_prices_new_by_payer=dict(per_payer),
                                   scheme_prices_update=len(upd), scheme_prices_unchanged=same,
                                   scheme_prices_held=dict(held))
        self.report["detail"]["scheme_price_updates"] = upd
        self._many("scheme_prices", rows, lambda r: self.insert("core", "scheme_pricing", r))
        self._many("scheme_price_updates", upd, lambda u: self.update("core", "scheme_pricing", u["id"], u["data"]))

    # ── facility product cost / selling prices ──
    def run_product_prices(self):
        prods = [p for p in self.live("inventory", "product") if p.get("facility_id") == self.fac]
        by_name = defaultdict(list)
        for p in prods:
            by_name[key(p.get("generic_name"))].append(p)
        cost = {key(c["name"]): c["cost"] for c in self.data["costs"] if c["cost"] is not None and c["sheet"] == "All Products"}
        for c in self.data["costs"]:              # the optical store's own sheet only fills gaps
            if c["cost"] is not None:
                cost.setdefault(key(c["name"]), c["cost"])
        sell = {key(x["name"]): x["price"] for x in self.data["pharmacy"] if x["price"] is not None}
        upd, same, unmatched = [], 0, 0
        for k, ps in by_name.items():
            want = {}
            if k in cost:
                want["cost_price"] = cost[k]
            if k in sell:
                want["selling_price"] = sell[k]
            if not want:
                unmatched += len(ps)
                continue
            for p in ps:
                before = {f: num(p.get(f)) for f in want}
                if before == want:
                    same += 1
                    continue
                upd.append({"_key": p["id"], "id": p["id"], "name": p.get("generic_name"), "before": before, "data": want})
        self.report["plan"].update(facility_products=len(prods), product_price_updates=len(upd),
                                   product_prices_unchanged=same, products_not_on_sheets=unmatched)
        self.report["detail"]["product_price_updates"] = upd
        self._many("product_price_updates", upd, lambda u: self.update("inventory", "product", u["id"], u["data"]))

    def load(self) -> dict:
        if "procedures" in self.steps:
            self.run_procedures()
        if "schemes" in self.steps:
            self.run_schemes()
        if "scheme_pricing" in self.steps:
            self.run_scheme_pricing()
        if "product_prices" in self.steps:
            self.run_product_prices()
        log.info("plan: %s", self.report["plan"])
        log.info("result: %s", self.report["result"])
        return self.report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="folder of rate-card spreadsheets → dags/jsons/rate_cards/<target>.json")
    c.add_argument("folder")
    c.add_argument("--target", required=True, choices=list(TARGETS))
    lo = sub.add_parser("load", help="reconcile the rate cards into V3")
    lo.add_argument("--target", required=True, choices=list(TARGETS))
    lo.add_argument("--dry-run", action="store_true")
    lo.add_argument("--limit", type=int, default=0, help="at most N changes of each kind (canary)")
    lo.add_argument("--workers", type=int, default=8)
    lo.add_argument("--steps", nargs="*", choices=STEPS, default=STEPS)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "convert":
        convert(a.folder, a.target)
        return
    report = Run(a.target, a.dry_run, a.limit, a.workers, a.steps).load()
    out = _HERE / "output" / "rate_cards_to_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.target}-{time.strftime('%Y%m%d_%H%M%S')}{'-dry' if a.dry_run else ''}.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log.info("report: %s", path)
    sys.exit(1 if report.get("failures") else 0)


if __name__ == "__main__":
    main()
