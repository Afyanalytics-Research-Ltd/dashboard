"""
deactivate_unused_products.py — switch off (is_active = 0) an org's template inventory products that nothing uses, so a
facility's product lists show only the hospital's own items. Reversible: run with --reactivate on the same target.

First use: Kerugoya Medical Centre, production org 357 (target kerugoya_prod), 2026-11-24. The org carried a 27k-item
template catalogue (facility_id NULL, loaded 2026-11-16) on top of the client's Excel items (facility 367); the client
wants only their Excel products listed. User's choice: deactivate the unused template products, KEEP the ones that hold
stock (this morning's in-app "FieldMatch" count) or are referenced anywhere.

A product is deactivated only if ALL hold: facility_id is NULL (template), currently active, and no reference in
  inventory stock (quantity > 0), inventory batch, core scheme_pricing (priceable product), evaluation prescriptions /
  consumables / procedure_consumables, inventory sale / purchase_order_item / requisition_item / goods_received_item.
Facility products (the client's items) are never touched.

    python deactivate_unused_products.py --target kerugoya_prod --dry-run
    python deactivate_unused_products.py --target kerugoya_prod
    python deactivate_unused_products.py --target kerugoya_prod --reactivate   # undo (uses the run's logged ids file)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("deactivate_unused_products")
_HERE = Path(__file__).resolve().parent
TARGETS = {"kerugoya_prod": {"organization_id": 357}}
REFS = (("evaluation", "prescriptions"), ("evaluation", "consumables"), ("evaluation", "procedure_consumables"),
        ("inventory", "sale"), ("inventory", "purchase_order_item"), ("inventory", "requisition_item"),
        ("inventory", "goods_received_item"))
ID_KEYS = ("product_id", "inventory_product_id", "drug_id", "item_id")


class Run:
    def __init__(self, target: str, dry_run: bool = False, workers: int = 12):
        try:
            from pipelines_common import use_pipelines_dir
            use_pipelines_dir([target])
        except (ImportError, RuntimeError):
            from dotenv import load_dotenv
            load_dotenv(_HERE / ".env", override=False)
            sys.path.insert(0, str(_HERE))
        from v3_to_v3_client import V3Client
        if target not in TARGETS:
            raise SystemExit(f"unknown target {target}")
        self.target, self.dry_run, self.workers = target, dry_run, workers
        self.c = V3Client(target)
        self.c.token()
        if self.c.organization_id != TARGETS[target]["organization_id"]:
            raise SystemExit(f"login org {self.c.organization_id} ≠ {TARGETS[target]['organization_id']}")
        self.report = {"target": target, "org": self.c.organization_id, "dry_run": dry_run, "plan": {}, "result": {},
                       "failures": []}

    def live(self, svc, alias, workers=8):
        return [r for r in self.c.read_all(svc, alias, workers=workers) if not r.get("deleted_at")]

    def plan(self) -> list[dict]:
        prods = self.live("inventory", "product")
        used = defaultdict(set)
        for s in self.live("inventory", "stock"):
            if (s.get("quantity") or 0) > 0:
                used[s["product_id"]].add("stock")
        for b in self.live("inventory", "batch"):
            used[b["product_id"]].add("batch")
        for r in self.live("core", "scheme_pricing", 1):
            if r.get("priceable_type") == "product":
                used[r["priceable_id"]].add("scheme_pricing")
        for svc, alias in REFS:
            rows = self.live(svc, alias, 4)        # a failed read raises: never deactivate on a partial picture
            for r in rows:
                for k in ID_KEYS:
                    if r.get(k):
                        used[r[k]].add(alias)
                        break
            log.info("references %s/%s: %d rows", svc, alias, len(rows))
        groups, out = Counter(), []
        for p in prods:
            tpl = p.get("facility_id") is None
            if not p.get("is_active", True):
                groups["already inactive"] += 1
            elif not tpl:
                groups[f"facility {p.get('facility_id')} product (kept)"] += 1
            elif p["id"] in used:
                groups["template, in use (kept): " + ",".join(sorted(used[p["id"]]))] += 1
            else:
                groups["template, unused → deactivate"] += 1
                out.append({"id": p["id"], "name": p.get("generic_name")})
        self.report["plan"] = dict(groups)
        log.info("plan: %s", json.dumps(dict(groups)))
        return out

    def apply(self, items: list[dict], active: int) -> None:
        if self.dry_run:
            log.info("dry run: would set is_active=%s on %d products, e.g. %s", active, len(items), items[:5])
            return
        res = Counter()

        def one(x):
            try:
                r = self.c.post("inventory", {"action": "update", "model": "product",
                                              "destination_tenant_id": self.c.organization_id,
                                              "ids": [x["id"]], "data": {"is_active": active}}, timeout=120)
                body = r.json() if r.content else {}
                ok = r.ok and (body.get("success", True) if isinstance(body, dict) else True)
                return x, ok, None if ok else f"{r.status_code} {r.text[:200]}"
            except Exception as e:
                return x, False, f"{type(e).__name__}: {str(e)[:200]}"
        with ThreadPoolExecutor(self.workers) as ex:
            for n, (x, ok, err) in enumerate(ex.map(one, items), 1):
                res["ok" if ok else "failed"] += 1
                if not ok:
                    self.report["failures"].append({"id": x["id"], "error": err})
                if n % 2000 == 0:
                    log.info("  %d / %d (%s)", n, len(items), dict(res))
        self.report["result"] = dict(res)
        log.info("result: %s", dict(res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reactivate", metavar="IDS_JSON", help="undo: set is_active=1 on the ids in this run file")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = Run(a.target, a.dry_run, a.workers)
    out = _HERE / "output" / "deactivate_unused_products"
    out.mkdir(parents=True, exist_ok=True)
    if a.reactivate:
        r.apply(json.loads(Path(a.reactivate).read_text())["deactivated"], 1)
    else:
        items = r.plan()
        r.apply(items, 0)
        (out / f"{a.target}-{time.strftime('%Y%m%d_%H%M%S')}{'-dry' if a.dry_run else ''}.json").write_text(
            json.dumps({**r.report, "deactivated": items}, default=str))
    sys.exit(1 if r.report["failures"] else 0)


if __name__ == "__main__":
    main()
