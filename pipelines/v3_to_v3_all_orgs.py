"""
v3_to_v3_all_orgs.py — v3_to_v3_migration's tail mode for EVERY organisation of a V3
instance that is a clone of another (test*.afyaanalytics.ai → prod), with one platform
account (e.g. "analyst") on each side instead of a login per org.

  survey   read-only: per org, the rows the source has that the target lacks (would_post,
           by the day they were created), rows waiting on such a parent (held), and empty
           links a migrate would fill (would_patch). Nothing is written.
  migrate  the same, posted: tail mode, model by model in tier order, then relink.

Each org gets a client pair acting for it (AFYA_<KEY>_PLATFORM=1, see v3_to_v3_client):
reads send source_tenant_id, writes destination_tenant_id / destination_facility_id of
that org — never the platform account's own org. Records keep their own facility (ids
are the same on both instances).

The plan (models, kinds, links, tiers) is the same for every org: it comes from a
template plan (--plan, default: the test284 → prod284 plan) instead of being rebuilt
per org. Only rows CREATED in the window are posted (window_cols=created_at): a row
created before the clone and missing on the target was deleted there.

Reads (--read-mode): filter (default) asks each instance only for the rows updated since
--since (gateway updated_since / after_id: two calls a table); an instance without the
filter falls back to tail. --since must lie BEFORE the instances diverged: a parent the
window didn't touch is then the same row on both. The platform account's grant (60 calls
a minute per instance) is respected client-side (V3V3_PLATFORM_RATE, default 55).

    python v3_to_v3_all_orgs.py survey  --since 2026-11-16 --orgs auto
    python v3_to_v3_all_orgs.py migrate --since 2026-11-16 --orgs 53 314 --parallel-orgs 2

Settings: V3V3_PLATFORM_{SOURCE,TARGET}_{USERNAME,PASSWORD,URL_TEMPLATE,MIGRATION_KEY}.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import v3_to_v3_migration as vv
from v3_to_v3_client import GatewayError, V3Client

log = logging.getLogger("v3_all_orgs")

TEMPLATE_PLAN = vv.STATE_ROOT / "v3v3_test284_to_prod284" / "plan.json"
SIDES = ("SOURCE", "TARGET")


def side_env(side: str, name: str) -> str:
    return (os.getenv(f"V3V3_PLATFORM_{side}_{name}") or "").strip()


def org_client(side: str, org: int, facility: int) -> V3Client:
    """A platform client acting for org / facility on one side ("SOURCE" | "TARGET")."""
    key = f"{'s' if side == 'SOURCE' else 't'}o{org}"
    up = key.upper()
    os.environ.update({
        f"AFYA_{up}_USERNAME": side_env(side, "USERNAME"), f"AFYA_{up}_PASSWORD": side_env(side, "PASSWORD"),
        f"AFYA_{up}_URL_TEMPLATE": side_env(side, "URL_TEMPLATE"), f"AFYA_{up}_ORGANIZATION_ID": str(org),
        f"AFYA_{up}_FACILITY_ID": str(facility), f"AFYA_{up}_PLATFORM": "1",
        f"AFYA_{up}_MIGRATION_KEY": side_env(side, "MIGRATION_KEY")})
    return V3Client(key)


def discover_orgs(since: str) -> dict[int, list[int]]:
    """{org: [facilities, busiest first]} with patients / visits / invoices created on the
    source since `since` (the gateway's counts action, tenants: all)."""
    tmpl = side_env("SOURCE", "URL_TEMPLATE")
    core = tmpl.format(service="core").rstrip("/")
    tok = requests.post(f"{core}/v1/login", json={"username": side_env("SOURCE", "USERNAME"),
                                                  "password": side_env("SOURCE", "PASSWORD")}, timeout=60).json()
    h = {"Authorization": f"Bearer {tok['access_token']}", "Accept": "application/json",
         "Content-Type": "application/json", "X-Tenant-Id": str((tok.get("tenant") or {}).get("id") or "")}
    group = [{"field": "organization_id"}, {"field": "facility_id"}]
    where = [{"field": "created_at", "op": ">=", "value": since}]
    queries = {f"q{i}": {"service": s, "model": m, "group_by": group, "limit": 1000, "where": where}
               for i, (s, m) in enumerate((("reception", "patient"), ("evaluation", "visits"),
                                           ("finance", "invoice")))}
    r = requests.post(f"{core}/v1/gateway/counts", headers=h, json={"tenants": "all", "queries": queries}, timeout=300)
    if not r.ok:
        raise GatewayError(f"counts: {r.status_code} {r.text[:300]}")
    load: dict[int, dict[int, int]] = {}
    for res in (r.json().get("data") or {}).get("results", {}).values():
        for g in ((res or {}).get("data") or {}).get("groups") or []:
            k = g.get("key") or {}
            org, fac = k.get("organization_id"), k.get("facility_id")
            if isinstance(org, int):
                f = load.setdefault(org, {})
                if isinstance(fac, int):
                    f[fac] = f.get(fac, 0) + int(g.get("count") or 0)
    return {o: sorted(f, key=lambda x: -f[x]) for o, f in load.items() if f}


def changed_models(plan: dict, since: str, batch: int = 20) -> tuple[dict[str, set[int]], set[str], set[str]]:
    """Which models the source changed since `since`, per org — the counts action, several
    queries per call, so the whole instance costs ~ models/batch calls instead of 2+ per table
    per org. Returns ({model: {orgs}} for models with an org column, {tenant-less models with
    any change}, {models whose count failed — copied anyway})."""
    tmpl = side_env("SOURCE", "URL_TEMPLATE")
    core = tmpl.format(service="core").rstrip("/")
    tok = requests.post(f"{core}/v1/login", json={"username": side_env("SOURCE", "USERNAME"),
                                                  "password": side_env("SOURCE", "PASSWORD")}, timeout=60).json()
    h = {"Authorization": f"Bearer {tok['access_token']}", "Accept": "application/json",
         "Content-Type": "application/json", "X-Tenant-Id": str((tok.get("tenant") or {}).get("id") or "")}
    where = [{"field": "updated_at", "op": ">=", "value": since}]
    models = {m: x for m, x in plan["models"].items() if x.get("selected")}
    per_org: dict[str, set[int]] = {}
    tenantless: set[str] = set()
    unknown: set[str] = set()
    names = sorted(models)
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        q = {}
        for n, m in enumerate(chunk):
            x = models[m]
            org_col = x.get("tenant") or ("organization_id" if x.get("kind") == "org_rows" else None)
            spec = {"service": x["service"], "model": x["alias"], "limit": 1000, "where": where}
            if org_col:
                spec["group_by"] = [{"field": org_col}]
            q[f"m{n}"] = spec
        for attempt in range(6):
            r = requests.post(f"{core}/v1/gateway/counts", headers=h, json={"tenants": "all", "queries": q}, timeout=300)
            if r.status_code != 429:
                break
            time.sleep(35)
        results = (r.json().get("data") or {}).get("results", {}) if r.ok else {}
        for n, m in enumerate(chunk):
            res = results.get(f"m{n}") or {}
            if not res.get("success"):
                unknown.add(m)
                continue
            data = res.get("data") or {}
            x = models[m]
            org_col = x.get("tenant") or ("organization_id" if x.get("kind") == "org_rows" else None)
            if not org_col:
                # no org column, no group_by: the gateway answers {"total": N, "groups": []}
                if int(data.get("total") or data.get("count") or 0) > 0:
                    tenantless.add(m)
                continue
            if data.get("truncated"):
                unknown.add(m)
            per_org[m] = {(g.get("key") or {}).get(org_col) for g in data.get("groups") or []
                          if isinstance((g.get("key") or {}).get(org_col), int) and int(g.get("count") or 0) > 0}
    log.info("changed since %s: %d models with an org column (%d org/model pairs), %d tenant-less, %d unknown",
             since, len(per_org), sum(len(v) for v in per_org.values()), len(tenantless), len(unknown))
    return per_org, tenantless, unknown


def is_shared(x: dict) -> bool:
    """A table without an org column (every org's rows in one table)."""
    return not x.get("tenant") and x.get("kind") != "org_rows"


def shared_pass(plan: dict, models: list[str], window, orgs: dict[int, list[int]], mode: str, args) -> dict:
    """The tables without an org column, ONCE for all orgs: each side's changed rows read a
    single time; a new row goes to the org that owns its parent (that org's id map from the
    per-org pass: a lab value → its result's org), posted through that org's clients. A row
    whose parent the window didn't place in any org is skipped and counted (no_owner) —
    never guessed."""
    t0 = time.time()
    out: dict[str, dict] = {}
    if not models or not orgs:
        return out
    states = {o: vv.State(f"src_o{o}", f"dst_o{o}") for o in orgs}
    copiers: dict[int, vv.Copier] = {}

    def copier_for(o: int) -> vv.Copier:
        if o not in copiers:
            c = vv.Copier(org_client("SOURCE", o, orgs[o][0]), org_client("TARGET", o, orgs[o][0]), states[o],
                          json.loads(json.dumps(plan)), workers=args.workers, read_workers=args.read_workers,
                          window=window, tail=True, identity_links=args.identity_links,
                          window_cols=("created_at",), filter_reads=True)
            c.map_only = mode == "survey"
            copiers[o] = c
        return copiers[o]

    owners: dict[str, dict[str, int]] = {}           # parent model → {source id: org} from every org's map

    def owner_index(parent: str) -> dict[str, int]:
        if parent not in owners:
            idx: dict[str, int] = {}
            for o, st in states.items():
                for sid in st.id_map(parent, reload=True):
                    idx.setdefault(str(sid), o)
            owners[parent] = idx
        return owners[parent]

    any_org = next(iter(orgs))
    reader = copier_for(any_org)
    for m in sorted(models, key=lambda m: (plan["models"][m]["tier"] or 0, m)):
        x = plan["models"][m]
        counts = {"read": 0, "already": 0, "posted": 0, "would_post": 0, "no_owner": 0, "held": 0,
                  "failed": 0, "outside_window": 0}
        try:
            src_rows = list(reader._filtered("src", m))          # every org's changed rows, once
            if not src_rows:
                out[m] = {"read": 0}
                continue
            on_target = {str(r["uuid"]).lower() for r in reader._filtered("dst", m) if r.get("uuid")}
            org_links = {c: t for c, t in x["links"].items() if t in plan["models"] and not is_shared(plan["models"][t])}
            for row in src_rows:
                counts["read"] += 1
                if str(row.get("uuid") or "").lower() in on_target:
                    counts["already"] += 1
                    continue
                if not vv.in_window(row, window, ("created_at",)):
                    counts["outside_window"] += 1
                    continue
                o = next((owner_index(t).get(str(row.get(c))) for c, t in org_links.items()
                          if row.get(c) not in (None, "") and owner_index(t).get(str(row.get(c))) is not None), None)
                if o is None:
                    counts["no_owner"] += 1
                    continue
                cp = copier_for(o)
                payload, missing, _ = cp.translate(m, row)
                if missing:
                    counts["held"] += 1
                    continue
                if cp.map_only:
                    counts["would_post"] += 1
                    continue
                new_id, err = cp._insert(m, payload)
                if err:
                    counts["failed"] += 1
                    cp.record_error(m, row, payload, err)
                    continue
                counts["posted"] += 1
                if new_id is not None:
                    cp.state.id_map(m)[str(row["id"])] = new_id
            for cp in copiers.values():
                cp.state.save_map(m)
        except GatewayError as e:
            counts["error"] = str(e)[:300]
        out[m] = {k: v for k, v in counts.items() if v}
        log.info("shared %s: %s", m, out[m])
    log.info("shared pass: %d tables in %ds", len(models), time.time() - t0)
    return out


def run_org(org: int, facilities: list[int], plan: dict, window, mode: str, args, changed=None) -> dict:
    t0 = time.time()
    out = {"org": org, "facilities": facilities, "mode": mode, "models": {}}
    try:
        src, dst = org_client("SOURCE", org, facilities[0]), org_client("TARGET", org, facilities[0])
        state = vv.State(f"src_o{org}", f"dst_o{org}")
        copier = vv.Copier(src, dst, state, json.loads(json.dumps(plan)), workers=args.workers,
                           read_workers=args.read_workers, window=window, tail=True,
                           identity_links=args.identity_links, window_cols=("created_at",),
                           filter_reads=args.read_mode == "filter")
        copier.map_only = mode == "survey"
        order = sorted((m for m, x in copier.models.items() if x["selected"]),
                       key=lambda m: (copier.models[m]["tier"] or 0, m))
        if changed is not None:
            # only the models the source changed for this org since `since` (unchanged ones are the
            # same on both instances: nothing to copy, and their ids link as-is)
            per_org, tenantless, unknown = changed
            mine = {m for m, orgs in per_org.items() if org in orgs} | unknown
            # tables without an org column hold every org's rows: copied once for all orgs by
            # shared_pass(), not re-read per org
            order = [m for m in order if m in mine and not is_shared(copier.models[m])]
            out["models_checked"] = len(order)
        for m in order:
            try:
                res = copier.copy_model(m)
            except GatewayError as e:
                res = {"status": "error", "error": str(e)[:300]}
            keep = {k: v for k, v in res.items() if k in ("posted", "held", "failed", "would_post", "would_post_by_day",
                                                       "would_patch", "links_filled", "updated_at_restored",
                                                       "foreign", "status", "error") and v}
            if set(keep) - {"status"}:
                out["models"][m] = keep
        if mode == "migrate":
            for m in order:
                try:
                    copier.relink(m)
                except GatewayError as e:
                    out["models"].setdefault(m, {})["relink_error"] = str(e)[:300]
    except Exception as e:                                   # one org's failure never stops the others
        out["error"] = f"{type(e).__name__}: {e}"[:500]
    tot = lambda k: sum(int(x.get(k) or 0) for x in out["models"].values())
    out.update({"would_post": tot("would_post"), "held": tot("held"), "would_patch": tot("would_patch"),
                "posted": tot("posted"), "failed": tot("failed"), "seconds": round(time.time() - t0)})
    if mode == "migrate" and "error" not in out:
        # a restart skips this org (its records are on the target either way: a re-run only re-checks)
        vv.State(f"src_o{org}", f"dst_o{org}").save("all_orgs", "done", {
            "since": args.since, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **{k: out[k] for k in ("posted", "held", "failed")}})
    log.info("org %s (%s): %s", org, mode, {k: out[k] for k in ("would_post", "held", "would_patch", "posted",
                                                                 "failed", "seconds")} | ({"error": out["error"]}
                                                                                          if "error" in out else {}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["survey", "migrate"])
    ap.add_argument("--since", required=True, help="window start (rows CREATED from then on)")
    ap.add_argument("--until", default="")
    ap.add_argument("--orgs", nargs="*", default=["auto"], help="org ids, or auto (busy on the source since --since)")
    ap.add_argument("--exclude-orgs", nargs="*", type=int, default=[])
    ap.add_argument("--plan", type=Path, default=TEMPLATE_PLAN)
    ap.add_argument("--parallel-orgs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--read-workers", type=int, default=4)
    ap.add_argument("--identity-links", nargs="*", default=["inventory/"])
    ap.add_argument("--read-mode", choices=["filter", "tail"], default="filter",
                    help="filter: only rows updated since --since on each side (gateway updated_since / after_id; "
                         "--since must be before the instances diverged); tail: bisect the diverged pages")
    ap.add_argument("--out", type=Path, default=None, help="summary JSON (default: state dir)")
    ap.add_argument("--redo", action="store_true", help="re-check orgs a previous migrate already finished")
    ap.add_argument("--skip-shared", action="store_true", help="skip the pass over the tables without an org column")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s · %(levelname)-7s · %(message)s", datefmt="%H:%M:%S")
    for side in SIDES:
        for name in ("USERNAME", "PASSWORD", "URL_TEMPLATE"):
            if not side_env(side, name):
                ap.error(f"set V3V3_PLATFORM_{side}_{name}")
    # "since a date" here means from then until today, not that one day (resolve_window's
    # meaning for a bare date with no until): no until = the end of today
    until = a.until or vv.datetime.now(vv.SOURCE_TZ).date().isoformat()
    window = vv.resolve_window(a.since, until)
    log.info("window: rows created %s … %s (exclusive)", *window)
    plan = json.loads(a.plan.read_text())
    found = discover_orgs(a.since)
    if a.orgs != ["auto"]:
        found = {int(o): found.get(int(o)) or [] for o in a.orgs}
    missing_fac = [o for o, f in found.items() if not f]
    if missing_fac:
        log.warning("orgs with no facility seen since %s (skipped): %s", a.since, missing_fac)
    todo = {o: f for o, f in sorted(found.items()) if f and o not in set(a.exclude_orgs)}
    log.info("%s %d orgs since %s: %s", a.mode, len(todo), a.since, list(todo))
    changed = None
    if a.read_mode == "filter":
        changed = changed_models(plan, window[0].astimezone(vv.SOURCE_TZ).strftime("%Y-%m-%d %H:%M:%S"))
        # an org with no changed model at all has nothing to copy
        busy = {o for orgs in changed[0].values() for o in orgs}
        if not changed[1] and not changed[2]:
            todo = {o: f for o, f in todo.items() if o in busy}
        log.info("%d orgs have changed models", len(todo))
    all_orgs = dict(todo)                                   # the shared pass needs every org's map
    if a.mode == "migrate" and not a.redo:
        done = {o for o in todo
                if (vv.State(f"src_o{o}", f"dst_o{o}").load("all_orgs", "done", {}) or {}).get("since") == a.since}
        if done:
            log.info("skipping %d orgs a previous run finished (since %s; --redo to re-check): %s",
                     len(done), a.since, sorted(done))
        todo = {o: f for o, f in todo.items() if o not in done}
    with ThreadPoolExecutor(max(1, a.parallel_orgs)) as ex:
        results = list(ex.map(lambda kv: run_org(kv[0], kv[1], plan, window, a.mode, a, changed), todo.items()))
    shared = {}
    if changed is not None and not a.skip_shared:
        shared_models = sorted(m for m, x in plan["models"].items()
                               if x.get("selected") and is_shared(x) and (m in changed[1] or m in changed[2]))
        shared = shared_pass(plan, shared_models, window, all_orgs, a.mode, a)
    shared_tot = lambda k: sum(int(v.get(k) or 0) for v in shared.values())
    summary = {"mode": a.mode, "since": a.since, "until": a.until, "orgs": results, "shared": shared,
               "totals": {k: sum(r.get(k) or 0 for r in results) + shared_tot(k)
                          for k in ("would_post", "held", "would_patch", "posted", "failed")},
               "shared_no_owner": shared_tot("no_owner"),
               "errors": {**{r["org"]: r["error"] for r in results if r.get("error")},
                          **{m: v["error"] for m, v in shared.items() if v.get("error")}}}
    out = a.out or vv.STATE_ROOT / f"all_orgs_{a.mode}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    log.info("totals: %s — errors in %d orgs — %s", summary["totals"], len(summary["errors"]), out)
    return 1 if summary["errors"] or summary["totals"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
