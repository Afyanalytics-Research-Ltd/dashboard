#!/usr/bin/env python
"""Run the pipeline on Kaggle (headless) from this repo.

    python tools/kaggle_run.py secrets                 # once: private dataset btp-secrets (W&B key only)
    python tools/kaggle_run.py push --job smoke        # build notebook with JOB=smoke and start a GPU run
    python tools/kaggle_run.py push --job full --langs a --resume
    python tools/kaggle_run.py status                  # running / complete / error
    python tools/kaggle_run.py logs [--out DIR]        # download the run's output (reports, logs, summary)

Credentials come from ~/.kaggle (kaggle.json / access_token), never from the repo.
--resume attaches this notebook's previous output as an input, so the workspace carries over.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import build_notebook as B  # noqa: E402

USER = os.environ.get("KAGGLE_USERNAME", "lutherlunyamwi")
SLUG = B.KAGGLE_SLUG
TITLE = "adventproclaimer_multilingual_translator"
PRIVATE_DATASET = f"{USER}/btp-secrets"   # holds only the W&B key (tools/kaggle_secrets.py)


def kaggle_exe() -> str:
    """Locate the Kaggle CLI: PATH, next to this Python, or the pipeline's own environment."""
    for cand in (shutil.which("kaggle"), str(Path(sys.executable).parent / "kaggle"),
                 str(Path.home() / ".venvs" / "btp" / "bin" / "kaggle")):
        if cand and Path(cand).is_file():
            return cand
    raise SystemExit(
        "The Kaggle CLI was not found. Use the pipeline's environment:\n"
        "    source ~/.venvs/btp/bin/activate\n"
        "or install it into the current one:\n"
        "    pip install kaggle six")


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run([kaggle_exe(), *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


def summary_score(s: dict | None) -> tuple:
    """Progress of the workspace a run saved, from its run_summary.json -- the same ordering the notebook
    uses for workspaces: (#finished trainings, #done data units, sum of checkpoint steps)."""
    p = (s or {}).get("progress") or {}
    steps = sum(v for v in (p.get("checkpoint_steps") or {}).values() if isinstance(v, int))
    return (len(p.get("trained") or []), sum((p.get("status_done") or {}).values()), steps)


def pick_best(summaries: dict[int, dict | None]) -> int | None:
    """Version whose saved workspace is the most advanced (ties: the newer version)."""
    scored = [(summary_score(s), v) for v, s in summaries.items() if s]
    return max(scored)[1] if scored else None


def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def current_version(api=None) -> int | None:
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        api = api or kaggle_api()
        with api.build_kaggle_client() as k:
            r = ApiGetKernelRequest()
            r.user_name, r.kernel_slug = USER, SLUG
            return int(k.kernels.kernels_api_client.get_kernel(r).metadata.current_version_number)
    except Exception:  # noqa: BLE001
        return None


def gpu_quota(api=None) -> dict | None:
    """Kaggle's own weekly GPU accounting (all notebooks + editor sessions), not just this notebook's."""
    try:
        api = api or kaggle_api()
        q = api.quota_view()
        g = q.gpu_quota
        return {"used_h": g.time_used.total_seconds() / 3600, "allowed_h": g.total_time_allowed.total_seconds() / 3600,
                "reserved_h": (g.time_reserved.total_seconds() / 3600) if g.time_reserved else 0.0,
                "refresh": str(q.quota_refresh_time)}
    except Exception:  # noqa: BLE001
        return None


def run_summary(version: int) -> dict | None:
    with tempfile.TemporaryDirectory() as d:
        r = kaggle("kernels", "output", f"{USER}/{SLUG}/{version}", "-p", d, "--file-pattern",
                   r"run_summary\.json", check=False)
        f = Path(d) / "run_summary.json"
        if r.returncode or not f.is_file():
            return None
        return json.loads(f.read_text(encoding="utf-8"))


def best_version(back: int = 8) -> int | None:
    """The recent version whose output holds the most advanced workspace. The newest version is not
    always it: Kaggle's daily schedule started version 13 while version 12 was training; 13 got no GPU,
    skipped, and saved the OLD workspace -- as the newest version, its output is what `latest` returns."""
    cur = current_version()
    if cur is None:
        return None
    return pick_best({v: run_summary(v) for v in range(cur, max(cur - back, 0), -1)})


def set_params(nb: dict, params: dict[str, str]) -> dict:
    """Rewrite `NAME = value` lines of the parameters cell."""
    cell = next(c for c in nb["cells"] if c["cell_type"] == "code" and "JOB = " in "".join(c["source"]))
    src = "".join(cell["source"])
    for k, v in params.items():
        src, n = re.subn(rf"^{k} = .*?(\s+#.*)?$", lambda m, k=k, v=v: f"{k} = {v!r}{m.group(1) or ''}", src,
                         flags=re.M)
        if n != 1:
            raise SystemExit(f"parameter {k} not found in the notebook")
    cell["source"] = src.splitlines(keepends=True)
    return nb


def write_secrets_dataset() -> None:
    """`secrets`: a one-time developer step, in its own file so it is not exported to Airflow."""
    try:
        from kaggle_secrets import write_secrets_dataset as write
    except ImportError:
        raise SystemExit("`secrets` runs from a full checkout of book-translation-pipeline "
                         "(tools/kaggle_secrets.py), not from the Airflow copy") from None
    write()


def kaggle_notebook() -> dict:
    """The Kaggle notebook. A full checkout builds it from the sources; a tools-only copy (the Airflow
    deployment: tools/ + notebooks/, no btp/) uses the prebuilt notebooks/kaggle_pipeline.ipynb, which CI
    keeps identical to the build (tools/build_notebook.py --check)."""
    if (REPO / "btp").is_dir():
        return B.build_kaggle()
    if not B.OUT_KAGGLE.is_file():
        raise SystemExit(f"no pipeline sources and no prebuilt notebook at {B.OUT_KAGGLE}")
    return json.loads(B.OUT_KAGGLE.read_text(encoding="utf-8"))


ACTIVE = ("running", "queued", "new")


def active_status() -> str | None:
    """Kaggle's status line if a version of the notebook is running or waiting for a GPU, else None."""
    r = kaggle("kernels", "status", f"{USER}/{SLUG}", check=False)
    out = (r.stdout + r.stderr).strip()
    line = out.splitlines()[-1] if out else ""
    return line if r.returncode == 0 and any(s in line.lower() for s in ACTIVE) else None


def cmd_push(args) -> None:
    busy = None if args.allow_overlap else active_status()
    if busy and not args.dry_run:
        # a push while a version runs or queues starts a second version that restores the OLD workspace
        # (2026-11-23: versions 13/14 overlapped 12) -- cancel the running one on kaggle.com first
        raise SystemExit(f"not pushing: {busy}. Wait for it to finish (or cancel it on kaggle.com); "
                         "--allow-overlap overrides.")
    if busy:
        print(f"dry run -- a real push would be refused now: {busy}")
    nb = set_params(kaggle_notebook(), {"JOB": args.job, "LANGS": args.langs, "SMOKE_LANG": args.smoke_lang,
                                        "B_LANG": args.b_lang, "DEADLINE_HOURS": float(args.deadline_hours)})
    with tempfile.TemporaryDirectory() as d:
        Path(d, f"{TITLE}.ipynb").write_text(B.render(nb), encoding="utf-8")
        meta = {"id": f"{USER}/{SLUG}", "title": TITLE, "code_file": f"{TITLE}.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [PRIVATE_DATASET], "competition_sources": [],
                "kernel_sources": [f"{USER}/{SLUG}"] if args.resume else [], "model_sources": []}
        if args.machine:
            meta["machine_shape"] = args.machine
        Path(d, "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        push = ["kernels", "push", "-p", d]
        if args.accelerator:
            push += ["--accelerator", args.accelerator]
        if args.dry_run:
            cell = next(c for c in nb["cells"] if c["cell_type"] == "code" and "JOB = " in "".join(c["source"]))
            print("dry run -- would run: kaggle " + " ".join(push))
            print("kernel-metadata.json:", json.dumps(meta, indent=1))
            print(f"notebook: {len(nb['cells'])} cells; parameters:\n" + "".join(cell["source"])[:1200])
            return
        r = kaggle(*push, check=False)
        print(r.stdout.strip(), r.stderr.strip())
        if r.returncode or "error" in (r.stdout + r.stderr).lower():
            raise SystemExit(1)
    print(f"started: https://www.kaggle.com/code/{USER}/{SLUG}  (JOB={args.job}, LANGS={args.langs})")


def cmd_status(args) -> None:
    r = kaggle("kernels", "status", f"{USER}/{SLUG}", check=False)
    print((r.stdout + r.stderr).strip(), "| version", current_version())
    q = gpu_quota()
    if q:
        print(f"GPU quota: {q['used_h']:.1f} h used of {q['allowed_h']:.0f} h (in use now: {q['reserved_h']:.1f} h), "
              f"resets {q['refresh']} UTC")


def cmd_logs(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = f"{USER}/{SLUG}" + (f"/{args.version}" if args.version else "")
    r = kaggle("kernels", "output", ref, "-p", str(out), "--file-pattern",
               r".*(\.log|\.json|\.txt)$", check=False)
    if r.returncode:
        raise SystemExit(f"kaggle kernels output {ref} failed:\n{(r.stdout + r.stderr).strip()[-2000:]}")
    print((r.stdout + r.stderr).strip()[-2000:])
    summ = out / "run_summary.json"
    if summ.is_file():
        print("run_summary:", summ.read_text())
    for f in sorted(out.rglob("stage_*.json")):
        d = json.loads(f.read_text())
        bad = "error" in d or any(isinstance(v, dict) and "error" in v for v in (d.get("result") or {}).values())
        print(("ERROR " if bad else "ok    ") + str(f.relative_to(out)))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("secrets")
    p = sub.add_parser("push")
    p.add_argument("--job", default="smoke", choices=["smoke", "full", "b", "book"])
    p.add_argument("--langs", default="a")
    p.add_argument("--smoke-lang", default="saq")
    p.add_argument("--b-lang", default="guz")
    p.add_argument("--resume", action="store_true", help="attach the previous run's output (workspace)")
    p.add_argument("--machine", default=None, help="e.g. NvidiaTeslaT4 (default: Kaggle's GPU default)")
    p.add_argument("--accelerator", default="NvidiaTeslaT4",
                   help="request this accelerator explicitly (a run once started with no GPU when it was "
                        "left to the enable_gpu default); '' to leave it to Kaggle")
    p.add_argument("--deadline-hours", type=float, default=11.0,
                   help="stages stop cleanly after this many hours (Kaggle's hard limit is 12)")
    p.add_argument("--dry-run", action="store_true", help="build the notebook + metadata and print them; push nothing")
    p.add_argument("--allow-overlap", action="store_true",
                   help="push even while a version is running/queued (it would restore the older workspace)")
    sub.add_parser("status")
    lg = sub.add_parser("logs")
    lg.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "btp_kaggle_output"))
    lg.add_argument("--version", type=int, default=None, help="notebook version (default: the newest)")
    args = ap.parse_args()
    {"secrets": lambda a: write_secrets_dataset(), "push": cmd_push, "status": cmd_status, "logs": cmd_logs}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
