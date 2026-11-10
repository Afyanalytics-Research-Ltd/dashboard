"""Patient dataset discovery: the one authoritative path.

  cd analytics_app/dashboards/internal_reporting
  python -m patient_discovery.run plan    [--schemas S1,S2]   metadata only: catalogue, lineage, registry; writes the
                                                             plan (next measurements + SQL) and the question pack
  python -m patient_discovery.run compile [--schemas ...]     EXPLAIN every measurement kind and the planned SQL
                                                             (compiles; reads no table data)
  python -m patient_discovery.run profile --schemas S1,S2 --budget ROWS --max-statements N --approved
                                                             reads source data, aggregates only; needs approval
  python -m patient_discovery.run answer  [--pack PATH] --by NAME
                                                             records filled-in answers in model/statements.json

Nothing here writes to Snowflake, activates a source or changes a count. Registry rows with status 'proposed' come
from the retired in-database discovery and are reported, never used.
Outputs (git-ignored) go to ./discovery_output: plan.json, questions.json, observations.json (aggregates only).
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from .model import measure as M
from .model import questions as Q
from .model import statements as S
from .model import store
from .model.concepts import Gap, Observation
from .model.planner import understand

OUT = Path("discovery_output")
EXPLANATORY = ("categories", "conditional")      # never run on real data: they read free-text-like columns


def _tup(x):
    return tuple(_tup(i) for i in x) if isinstance(x, list) else x


def load_observations():
    p = OUT / "observations.json"
    if not p.exists():
        return []
    return [Observation(o["kind"], _tup(o["subject"]), _tup(o["args"]), o["fp"], o["values"], o.get("at"))
            for o in json.loads(p.read_text(encoding="utf-8"))]


def save_observations(obs):
    (OUT / "observations.json").write_text(json.dumps(
        [dict(kind=o.kind, subject=o.subject, args=o.args, fp=o.fp, values=o.values, at=o.at) for o in obs],
        default=str), encoding="utf-8")


def load_env(path):
    """Load Snowflake credentials from a .env file; a relative private-key path is resolved against its folder."""
    import os
    from dotenv import load_dotenv
    load_dotenv(path)
    key = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    if key and not Path(key).is_absolute():
        os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"] = str(Path(path).resolve().parent / key)


def setup(args, allow_data=False):
    from .model.snowflake import Snowflake
    if args.env:
        load_env(args.env)
    OUT.mkdir(exist_ok=True)
    (OUT / ".gitignore").write_text("*\n", encoding="utf-8")
    wh = Snowflake(allow_data=allow_data, max_statements=getattr(args, "max_statements", None))
    reg = wh.registry()
    statements, legacy = S.load(reg, stored=store.read_statements(wh.fetch))
    keep = [r["SOURCE_OBJECT"] for r in reg if (r["STATUS"] or "").lower() == "active"]
    keep += [m[0] for s in statements if s.kind == "namespace" for m in s.assertion["members"]]
    schemas = set(args.schemas.upper().split(",")) if args.schemas else None
    cached = OUT / "catalogue.json"
    if getattr(args, "cached_catalogue", False) and cached.exists():
        full = json.loads(cached.read_text(encoding="utf-8"))          # names, types, columns, lineage only
        for o in full:
            o["columns"] = [tuple(c) for c in o["columns"]]
        cat = [o for o in full if o["store"] or not schemas or o["schema"] in schemas or o["fqn"].upper() in
               {k.upper() for k in keep}]
    else:
        cat = wh.catalogue(schemas=schemas, keep=keep, progress=lambda m: print("  " + m, flush=True))
        if not schemas:
            cached.write_text(json.dumps(cat, default=str), encoding="utf-8")
    return wh, statements, legacy, cat


def report(res, legacy, wh, cat, statements, now, publish=True):
    b = res.beliefs
    live = [a for a in b.assessments.values() if a.stage != "not_a_source" or not a.reasons
            or not a.reasons[0].startswith("raw store")]
    print(f"objects {len(cat)} (raw stores {sum(1 for o in cat if o.get('store'))}); stages:",
          dict(Counter(a.stage for a in live)))
    print(f"metadata queries {wh.n_meta}, EXPLAIN {wh.n_explain}, data statements {wh.n_queries}, "
          f"measured {len(res.measured)}, failed {len(res.failed)}, waiting {len(res.unmet)}")
    for g, err in res.failed:
        print(f"  failed {g.kind} {g.subject}: {err}")
    if legacy:
        print(f"legacy proposals in the registry (not used): {len(legacy)}")
        for x in legacy:
            print(f"  {x['status']:9} {x['object']}  ({x['source']})")
    for a in sorted(b.assessments.values(), key=lambda a: a.fqn):
        if a.stage in ("auto", "replacement"):
            print(f"  {a.stage:11} {a.fqn}  key={a.config.get('key')}  via {' -> '.join(a.selected.provenance)}")
    groups = Q.with_outcomes(Q.build(b), cat, res.observations, statements, now)
    Q.write(groups, OUT / "questions.json")
    print(f"questions: {len(groups)} groups ({dict(Counter(g['type'] for g in groups))}) -> {OUT / 'questions.json'}")
    if publish and _tables_exist(wh):
        wh.write(store.publish_questions(groups))
        print(f"questions published to {store.QUESTIONS} (the dashboard's Discovery tab)")


def _tables_exist(wh):
    recs = wh.fetch("SELECT COUNT(*) AS N FROM HOSPITALS.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'ANALYTICS' "
                    "AND TABLE_NAME IN ('AFYA_PATIENT_DISCOVERY_QUESTIONS', 'AFYA_PATIENT_DISCOVERY_STATEMENTS')")
    return recs[0]["N"] == 2


def plan_gaps(res, cat, dialect=M.Snowflake):
    gaps = sorted(res.unmet, key=lambda g: (not g.blocking, g.priority, g.cost))
    return [dict(kind=g.kind, objects=list(g.subject), blocking=g.blocking, est_rows=g.cost, why=g.why,
                 sql=M.statements_for(dialect, {o["fqn"]: o for o in cat}, g)) for g in gaps]


def cmd_plan(args):
    wh, statements, legacy, cat = setup(args)
    now = datetime.now()
    res = understand(wh, statements, now=now, catalogue=cat, dialect=M.Snowflake,
                     observations=load_observations(), skip_kinds=EXPLANATORY, dry_run=True)
    plan = plan_gaps(res, cat)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    report(res, legacy, wh, cat, statements, now)
    print(f"next measurements: {len(plan)} gaps, {sum(len(p['sql']) for p in plan)} statements, "
          f"~{sum(p['est_rows'] for p in plan):,.0f} rows estimated -> {OUT / 'plan.json'}")


def sample_gaps(cat, statements):
    """One gap of every measurement kind on real objects (for compiling): a counted source with a record key and
    a sibling object holding the same identifier column."""
    from .model.interpret import nominate
    src = {s.subject: s.assertion for s in statements if s.kind == "source"}
    byf = {o["fqn"]: o for o in cat}
    for f, s in src.items():
        o = byf.get(f)
        if not o or not s.get("record"):
            continue
        cols = dict(o["columns"])
        key, rk = s["key"], s["record"]
        if key not in cols or rk not in cols:
            continue
        sib = next((x for x in cat if x["fqn"] != f and x["schema"] == o["schema"] and key in dict(x["columns"])
                    and rk in dict(x["columns"])), None)
        if not sib:
            continue
        date = next((c for c, t in o["columns"] if t.startswith(("DATE", "TIMESTAMP"))), None)
        texts = tuple(c for c, t in o["columns"] if t == "TEXT" and c not in (key, rk))[:3]
        cands = tuple((c.column, c.data_type) for c in nominate(o["columns"], {rk.upper()})[0][:3])
        k, kt, r, rt = (key, cols[key]), (key, cols[key]), (rk, cols[rk]), (rk, cols[rk])
        return [Gap("screen", (f,), (cands, r, date)), Gap("profile", (f,), (k, rk, s.get("deleted"), s.get("version"), date)),
                Gap("pair", (f,), (k, (rk, cols[rk]))), Gap("shape", (f,), (k,)),
                Gap("containment", (f, sib["fqn"]), (k, (key, dict(sib["columns"])[key]))),
                Gap("align", (f, sib["fqn"]), (r, k, (rk, dict(sib["columns"])[rk]), (key, dict(sib["columns"])[key]))),
                Gap("formats", (f, sib["fqn"]), (k, (key, dict(sib["columns"])[key]))),
                Gap("window", (f,), (k, date or rk, "2026-01-01")),
                Gap("categories", (f,), texts), Gap("conditional", (f,), (k, texts[0] if texts else rk))]
    return []


def cmd_compile(args):
    wh, statements, legacy, cat = setup(args)
    byf = {o["fqn"]: o for o in cat}
    res = understand(wh, statements, now=datetime.now(), catalogue=cat, dialect=M.Snowflake,
                     observations=load_observations(), skip_kinds=EXPLANATORY, dry_run=True)
    todo = [(g, "sample") for g in sample_gaps(cat, statements)] + [(g, "plan") for g in res.unmet][:args.limit]
    ok, bad = Counter(), []
    for g, origin in todo:
        for sql in M.statements_for(M.Snowflake, byf, g):
            try:
                wh.explain(sql)
                ok[g.kind] += 1
            except Exception as e:
                bad.append((g.kind, origin, g.subject, str(e).replace("\n", " ")[:300]))
    print(f"compiled: {dict(ok)}; failures {len(bad)}")
    for b in bad:
        print("  FAIL", b)
    return 1 if bad else 0


def cmd_profile(args):
    if not args.approved:
        sys.exit("profile reads source data: add --approved only after explicit approval")
    if not (args.schemas and args.budget and args.max_statements):
        sys.exit("profile needs --schemas, --budget and --max-statements")
    wh, statements, legacy, cat = setup(args, allow_data=True)
    now = datetime.now()
    res = understand(wh, statements, now=now, budget=args.budget, catalogue=cat,
                     observations=load_observations(), max_statements=args.max_statements,
                     skip_kinds=EXPLANATORY, dialect=M.Snowflake)
    # measure what a 'person' answer would still need, so each question's outcome is known before it is answered
    groups = Q.build(res.beliefs)
    obs, spent, cost = res.observations, res.queries, res.cost
    for g in groups:
        hyp = Q.hypothesis(g, statements)
        if hyp is None:
            continue
        left = args.max_statements - spent
        if left <= 0 or cost >= args.budget:
            break
        r = understand(wh, hyp, now=now, budget=args.budget - cost, catalogue=cat, observations=obs,
                       max_statements=left, skip_kinds=EXPLANATORY, dialect=M.Snowflake)
        obs, spent, cost = r.observations, spent + r.queries, cost + r.cost
    res.observations = obs
    save_observations(obs)
    print(f"outcome measurements: {spent - res.queries} statements")
    report(res, legacy, wh, cat, statements, now)


def cmd_tables(args):
    from .model.snowflake import Snowflake
    if args.env:
        load_env(args.env)
    wh = Snowflake()
    wh.write([(sql, None) for sql in store.DDL])
    print(f"ready: {', '.join(store.TABLES)}")


def cmd_answer(args):
    if args.env:                                   # record in Snowflake, where the dashboard reads them
        from .model.snowflake import Snowflake
        load_env(args.env)
        wh = Snowflake()
        pack = json.loads(Path(args.pack).read_text(encoding="utf-8"))["questions"]
        new = [s for q in pack for s in Q.to_statements(q, q.get("answer"), args.by, q.get("note"))]
        existing = {s["id"]: s for s in store.read_statements(wh.fetch)}
        new = store.merge(new, existing)
        if new:
            wh.write(store.record_statements(new))
        print(f"recorded {len(new)} statements in {store.STATEMENTS}")
        register_due(wh)
    else:
        new = Q.record(args.pack, args.by)
        print(f"recorded {len(new)} statements in {S.FILE}")
    for s in new:
        print(f"  {s['id']}: {s['assertion'].get('meaning') or s['kind']}")


def register_due(wh):
    """Answers take effect at once: accepted sources whose required answers are all in are registered (insert-only,
    with provenance), and an audit entry is recorded."""
    questions, _ = store.read_questions(wh.fetch)
    registry = wh.fetch(f"SELECT SOURCE, SOURCE_OBJECT FROM {store.REGISTRY}")
    due = Q.due_registrations(questions, store.read_statements(wh.fetch), registry)
    if due:
        wh.write([store.register_sql(r) for r in due] + store.record_statements(
            [store.registered_statement(r) for r in due]))
    for r in due:
        print(f"  registered '{r['name']}': {r['object']} key={r['config']['key']} (counted on the next refresh)")
    return due


def main(argv=None):
    ap = argparse.ArgumentParser(prog="patient_discovery.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "compile", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--schemas", help="comma-separated schemas to assess (counted sources are always included)")
        p.add_argument("--env", help="path to the .env with Snowflake credentials (e.g. the main repo's .env)")
        p.add_argument("--cached-catalogue", action="store_true", dest="cached_catalogue",
                       help="reuse discovery_output/catalogue.json from the last full metadata read")
        if name == "compile":
            p.add_argument("--limit", type=int, default=200, help="planned gaps to compile (all kinds are sampled)")
        if name == "profile":
            p.add_argument("--budget", type=float, help="estimated rows to read, at most")
            p.add_argument("--max-statements", type=int, dest="max_statements")
            p.add_argument("--approved", action="store_true")
    a = sub.add_parser("answer")
    a.add_argument("--pack", default=str(OUT / "questions.json"))
    a.add_argument("--by", required=True)
    a.add_argument("--env", help="record in Snowflake (AFYA_PATIENT_DISCOVERY_STATEMENTS) instead of statements.json")
    t = sub.add_parser("tables")
    t.add_argument("--env")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "compile": cmd_compile, "profile": cmd_profile, "answer": cmd_answer,
            "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
