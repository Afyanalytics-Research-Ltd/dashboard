"""Generate API endpoint reference docs from the erp-platform-3.0 Laravel route files.

    python support_kb/gen_api_docs.py /path/to/erp-platform-3.0 support_kb/sources/API_ENDPOINTS.md

Static parse (no PHP needed): follows Route::prefix / ->middleware / ->group nesting,
resolves `$var = 'permission:...'` gates, expands apiResource/resource, and keeps the
nearest `//` comment above a route as its section heading. Output is Markdown with one
`##` per service and one `###` per resource, which is how the RAG chunker splits it.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

SERVICE_INFO = {
    "core-service": ("Core", "sign-in, users, roles, permissions, organizations, facilities, settings, dashboards"),
    "reception-service": ("Reception", "patients, check-in, visits, queues, appointments, follow-ups, documents"),
    "patient-evaluation-service": ("Patient evaluation", "outpatient clinical work, triage, notes, lab, pharmacy, procedures, SHA claims"),
    "finance-service": ("Finance", "billing, cash workbench, invoices, receipts, credit notes, waivers, insurance"),
    "inventory-service": ("Inventory", "products, stores, suppliers, requisitions, purchase orders, GRNs, POS"),
    "inpatient-service": ("Inpatient", "wards, beds, admissions, charge sheets, discharges"),
    "theatre-service": ("Theatre", "theatre bookings, schedules, ward checks, operations"),
    "dialysis-service": ("Dialysis", "dialysis patients, schedules, sessions, machines"),
    "oncology-service": ("Oncology", "screening, diagnosis, treatment plans, chemo cycles, palliative care"),
    "morgue-service": ("Morgue", "bodies, storage, releases, mortuary charges"),
    "psychiatry-service": ("Psychiatry", "psychiatric episodes, forms, risk, legal status, alerts"),
    "integration-service": ("Integration", "per-organization integration switches"),
    "reports-service": ("Reports", "report definitions and report data"),
}
VERBS = ("get", "post", "put", "patch", "delete", "any", "match")
RESOURCE_ACTIONS = [  # (action, method, suffix)
    ("index", "GET", ""), ("store", "POST", ""), ("show", "GET", "/{id}"),
    ("update", "PUT/PATCH", "/{id}"), ("destroy", "DELETE", "/{id}"),
]

STR = r"""['"]([^'"]*)['"]"""


def _strings(s: str) -> list[str]:
    return re.findall(STR, s)


def _perm(mw: list[str]) -> str:
    out = []
    for m in mw:
        for part in m.split(","):
            part = part.strip()
            if part.startswith(("permission:", "role:")):
                out.append(part)
    return "; ".join(dict.fromkeys(out))


def parse(text: str):
    # Strip block comments but keep // comments as headings.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    variables: dict[str, str] = {}
    for name, val in re.findall(r"\$(\w+)\s*=\s*" + STR + r"\s*;", text):
        variables[name] = val

    def mw_from(s: str) -> list[str]:
        res = []
        for m in re.finditer(r"middleware\((.*?)\)(?=\s*(->|;|$))", s, flags=re.S):
            arg = m.group(1)
            res += _strings(arg)
            res += [variables.get(v, "") for v in re.findall(r"\$(\w+)", arg)]
        return [r for r in res if r]

    # Split into statements terminated by ';' or a group opener '{'.
    stack = [{"prefix": "", "mw": []}]
    routes = []
    comment = ""
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("//"):
            c = line.lstrip("/ ").strip()
            if c and not c.startswith(("use ", "---")):
                comment = c
            continue
        if not line:
            continue
        buf += " " + line
        # group open
        while True:
            m = re.search(r"group\(\s*function\s*\([^)]*\)\s*(use\s*\([^)]*\)\s*)?\{", buf)
            if not m:
                break
            head = buf[: m.start()]
            prefixes = re.findall(r"prefix\(" + STR + r"\)", head)
            frame = {
                "prefix": stack[-1]["prefix"] + "".join("/" + p.strip("/") for p in prefixes if p.strip("/")),
                "mw": stack[-1]["mw"] + mw_from(head),
            }
            stack.append(frame)
            buf = buf[m.end():]
        # group close(s)
        while re.match(r"\s*\}\s*\)\s*;", buf):
            if len(stack) > 1:
                stack.pop()
            buf = re.sub(r"^\s*\}\s*\)\s*;", "", buf, count=1)
        if ";" not in buf:
            continue
        stmts, buf = buf.rsplit(";", 1)[0], buf.rsplit(";", 1)[1]
        for st in stmts.split(";"):
            st = st.strip()
            if "Route::" not in st and not st.startswith("->"):
                # closing braces mixed in
                for _ in re.findall(r"\}\s*\)", st):
                    if len(stack) > 1:
                        stack.pop()
                continue
            fr = stack[-1]
            mw = fr["mw"] + mw_from(st)
            ctrl = re.search(r"\[\s*\\?([\w\\]+)::class\s*,\s*" + STR + r"\s*\]", st)
            ctrl_s = f"{ctrl.group(1).split(chr(92))[-1]}@{ctrl.group(2)}" if ctrl else ""
            vm = re.search(r"Route::(" + "|".join(VERBS) + r")\(\s*" + STR, st)
            if vm:
                path = fr["prefix"] + "/" + vm.group(2).strip("/")
                routes.append({"method": vm.group(1).upper(), "path": path.rstrip("/") or "/",
                               "action": ctrl_s, "perm": _perm(mw), "section": comment})
                continue
            rm = re.search(r"Route::(apiResource|resource)\(\s*" + STR + r"\s*,\s*\\?([\w\\]+)::class", st)
            if rm:
                base = fr["prefix"] + "/" + rm.group(2).strip("/")
                cls = rm.group(3).split("\\")[-1]
                only = re.search(r"only\(\s*\[(.*?)\]", st)
                exc = re.search(r"except\(\s*\[(.*?)\]", st)
                acts = RESOURCE_ACTIONS
                if only:
                    keep = set(_strings(only.group(1)))
                    acts = [a for a in acts if a[0] in keep]
                if exc:
                    drop = set(_strings(exc.group(1)))
                    acts = [a for a in acts if a[0] not in drop]
                for act, meth, suf in acts:
                    routes.append({"method": meth, "path": base + suf, "action": f"{cls}@{act}",
                                   "perm": _perm(mw), "section": comment})
    return routes


def resource_of(path: str) -> str:
    parts = [p for p in path.split("/") if p and not p.startswith("{") and p not in ("v1", "v2", "api")]
    return parts[0] if parts else "(root)"


def main(repo: str, out: str) -> None:
    root = Path(repo) / "services"
    md = [
        "# Afya Analytics HMIS 3.0 — Backend API endpoint reference",
        "",
        "Generated from the erp-platform-3.0 route files. Each service is a separate backend; "
        "the frontend reaches it at the address set in its `*_SERVICE_URL` variable (which ends in `/api/v1`). "
        "Every endpoint except `/health` needs a signed-in user token, and most also need the permission shown. "
        "A permission like `permission:a|b` means the user needs any one of them; superadmin always passes. "
        "If a user gets 403 Forbidden on an action, the permission listed for that endpoint is missing from their role.",
        "",
    ]
    total = 0
    for svc in sorted(SERVICE_INFO):
        f = root / svc / "routes" / "api.php"
        if not f.exists():
            continue
        routes = parse(f.read_text(encoding="utf-8", errors="ignore"))
        if not routes:
            continue
        name, about = SERVICE_INFO[svc]
        total += len(routes)
        md += [f"## {name} service API ({svc})", "",
               f"Covers: {about}. {len(routes)} endpoints.", ""]
        groups = defaultdict(list)
        for r in routes:
            groups[resource_of(r["path"])].append(r)
        for res in sorted(groups):
            md += [f"### {name} service — `{res}` endpoints", ""]
            secs = sorted({r['section'] for r in groups[res] if r['section']})
            if secs:
                md += ["Area: " + "; ".join(secs[:6]), ""]
            for r in groups[res]:
                bits = [f"- `{r['method']} {r['path']}`"]
                if r["action"]:
                    bits.append(f"→ {r['action']}")
                if r["perm"]:
                    bits.append(f"(needs {r['perm']})")
                md.append(" ".join(bits))
            md.append("")
    Path(out).write_text("\n".join(md), encoding="utf-8")
    print(f"{total} endpoints → {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
