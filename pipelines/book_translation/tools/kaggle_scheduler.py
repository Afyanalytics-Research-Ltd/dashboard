#!/usr/bin/env python
"""Scheduled Kaggle training that protects the weekly GPU quota.

Run `tick` every 30 minutes (cron). Each tick is safe to repeat; it looks at the Kaggle notebook and a
local state file and does at most one thing:

  phase resume_test_1  short full run on Luo (deadline 45 min) from a fresh workspace
  phase resume_test_2  the same with --resume: must restore the saved workspace AND get further
                       -> proves stop/save/restore/continue on Kaggle itself before any long run
  phase full           11 h runs over all target languages, each continuing from the previous one;
                       after every run the finished models are copied to Google Drive
  phase done           every language trained (the run reports all_done) -> nothing more is launched

Before each launch: no run active, the previous run succeeded, and the GPU hours used in the last 7 days
plus this run fit the weekly budget (30 h). Any failure HALTS the schedule until a human runs
`tick --resume-schedule` -- a broken run is never retried blindly.

    python tools/kaggle_scheduler.py tick              # what cron runs
    python tools/kaggle_scheduler.py status            # phase, ledger, hours used this week
    python tools/kaggle_scheduler.py tick --dry-run    # decide, but do not launch or sync
    python tools/kaggle_scheduler.py resume-schedule   # clear a halt after fixing the cause
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = Path(os.environ.get("BTP_SCHED_STATE", Path.home() / ".local/state/btp/scheduler.json"))
LOG = STATE.with_name("scheduler.log")

WEEKLY_BUDGET_H = 30.0
OVERHEAD_H = 0.25            # session start-up not covered by the run's own timer
RESUME_TEST = {"job": "full", "langs": "luo", "deadline_hours": 0.75}
FULL_RUN = {"job": "full", "langs": "a", "deadline_hours": 11.0}


# ---------------------------------------------------------------------------
# pure decision logic (unit-tested)
# ---------------------------------------------------------------------------
def new_state() -> dict:
    return {"phase": "resume_test_1", "pending": None, "halted": None, "runs": []}


def hours_last_7d(state: dict, now: datetime) -> float:
    cutoff = now - timedelta(days=7)
    return sum(r["hours"] for r in state["runs"] if datetime.fromisoformat(r["finished"]) >= cutoff)


def progress_score(summary: dict) -> tuple[int, int, int]:
    p = (summary or {}).get("progress") or {}
    steps = sum(v for v in (p.get("checkpoint_steps") or {}).values() if isinstance(v, int))
    return (len(p.get("trained") or []), sum((p.get("status_done") or {}).values()), steps)


def run_params(phase: str) -> dict:
    if phase == "resume_test_1":
        return {**RESUME_TEST, "resume": False}
    if phase == "resume_test_2":
        return {**RESUME_TEST, "resume": True}
    return {**FULL_RUN, "resume": True}


def decide(state: dict, kaggle_status: str, summary: dict | None, now: datetime) -> dict:
    """-> {"action": wait|record|launch|halt|idle, ...}. Never launches while anything is unclear."""
    st = kaggle_status.lower()
    if state.get("halted"):
        return {"action": "idle", "why": f"halted: {state['halted']}"}
    if state["phase"] == "done":
        return {"action": "idle", "why": "training complete"}
    if any(s in st for s in ("running", "queued", "new")):
        return {"action": "wait", "why": f"Kaggle run in progress ({kaggle_status})"}
    pending = state.get("pending")
    if pending:
        if "complete" not in st:
            return {"action": "halt", "why": f"run ended with status {kaggle_status!r}", "hours": pending["deadline_hours"] + 1}
        if not summary:
            return {"action": "halt", "why": "run finished but wrote no run_summary.json", "hours": pending["deadline_hours"] + 1}
        hours = float(summary.get("hours", 0)) + OVERHEAD_H
        if summary.get("failed"):
            return {"action": "halt", "why": f"stages failed: {summary['failed']}", "hours": hours}
        phase, nxt = state["phase"], state["phase"]
        if phase == "resume_test_1":
            if progress_score(summary) == (0, 0, 0):
                return {"action": "halt", "why": "resume test 1 saved no progress", "hours": hours}
            nxt = "resume_test_2"
        elif phase == "resume_test_2":
            first = next((r for r in reversed(state["runs"]) if r["phase"] == "resume_test_1"), None)
            if not summary.get("restored_from"):
                return {"action": "halt", "why": "resume test 2 did not restore the saved workspace", "hours": hours}
            if first is None or not progress_score(summary) > tuple(first["progress_score"]):
                return {"action": "halt", "why": "resume test 2 did not get further than test 1", "hours": hours}
            nxt = "full"
        elif phase == "full":
            if not summary.get("restored_from") and any(r["phase"] != "resume_test_1" for r in state["runs"]):
                return {"action": "halt", "why": "full run started from an empty workspace", "hours": hours}
            nxt = "done" if summary.get("all_done") else "full"
        # a run that skipped (no GPU / no budget) produced no new models: nothing to copy
        return {"action": "record", "hours": hours, "next_phase": nxt, "sync": phase == "full" and not summary.get("skip"),
                "progress_score": list(progress_score(summary))}
    if state.get("hold_after_pending"):
        # finish recording whatever was running, but launch nothing new (e.g. hand-over to Kaggle's scheduler)
        return {"action": "idle", "why": f"hold: {state['hold_after_pending']}"}
    if state.get("monitor_only"):
        # Kaggle's own scheduler launches runs; this machine only records results and copies models to Drive
        return {"action": "idle", "why": "monitor-only (Kaggle scheduler launches runs)"}
    params = run_params(state["phase"])
    need = params["deadline_hours"] + OVERHEAD_H + 0.5
    used = hours_last_7d(state, now)
    if used + need > WEEKLY_BUDGET_H:
        return {"action": "wait", "why": f"weekly budget: {used:.1f} h used, next run needs {need:.1f} h"}
    return {"action": "launch", "params": params}


# ---------------------------------------------------------------------------
# side effects
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else new_state()


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def kaggle_status() -> str:
    sys.path.insert(0, str(REPO / "tools"))
    from kaggle_run import SLUG, USER, kaggle
    r = kaggle("kernels", "status", f"{USER}/{SLUG}", check=False)
    return (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "unknown"


def fetch_summary() -> dict | None:
    sys.path.insert(0, str(REPO / "tools"))
    from kaggle_run import SLUG, USER, kaggle
    with tempfile.TemporaryDirectory() as d:
        kaggle("kernels", "output", f"{USER}/{SLUG}", "-p", d, "--file-pattern", r"run_summary\.json", check=False)
        f = Path(d) / "run_summary.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def launch(params: dict) -> None:
    cmd = [sys.executable, str(REPO / "tools/kaggle_run.py"), "push", "--job", params["job"], "--langs", params["langs"],
           "--deadline-hours", str(params["deadline_hours"])] + (["--resume"] if params["resume"] else [])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if r.returncode:
        raise RuntimeError(f"push failed: {r.stdout[-500:]} {r.stderr[-500:]}")


def sync_models(token: str) -> str:
    r = subprocess.run([sys.executable, str(REPO / "tools/sync_models_to_drive.py"), "--token", token],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode:
        raise RuntimeError(f"model sync failed: {r.stderr[-500:]}")
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "synced"


def monitor_tick(state: dict, status: str, now: datetime, args) -> int:
    """Kaggle's own scheduler launches runs: record each newly finished one and copy its models to Drive."""
    if "complete" not in status.lower() and "error" not in status.lower():
        log(f"monitor: Kaggle {status!r} -- nothing to record")
        return 0
    summary = fetch_summary()
    if not summary or not summary.get("finished"):
        log("monitor: latest run has no run_summary.json -- needs attention")
        state["attention"] = "latest Kaggle run wrote no summary (check its log on Kaggle)"
        save_state(state)
        return 0
    if summary["finished"] == state.get("last_seen_finished"):
        log(f"monitor: no new run since {summary['finished']}")
        return 0
    state["last_seen_finished"] = summary["finished"]
    ran = not summary.get("skip")
    state["runs"].append({"phase": "kaggle-scheduled", "started": None, "finished": now.isoformat(),
                          "hours": round(float(summary.get("hours", 0)) + (OVERHEAD_H if ran else 0), 2),
                          "summary": summary, "progress_score": list(progress_score(summary))})
    if summary.get("failed") or "error" in status.lower():
        state["attention"] = f"run {summary['finished']}: failed stages {summary.get('failed')} / status {status}"
        log(f"ATTENTION: {state['attention']}")
    log(f"monitor: recorded run finished {summary['finished']} (skip={summary.get('skip')}, "
        f"hours={summary.get('hours')}, all_done={summary.get('all_done')})")
    if ran and summary.get("job") == "full":
        try:
            log("models -> Drive: " + sync_models(args.token))
        except Exception as e:  # noqa: BLE001
            log(f"WARNING model sync failed: {e}")
    if summary.get("all_done"):
        state["phase"] = "done"
        log("all languages trained -- turn off the Kaggle schedule (Settings -> Schedule) to stop evaluation-only runs")
    save_state(state)
    return 0


_TICK_LOCK = None


def acquire_tick_lock():
    """Exclusive, non-blocking lock so only one tick runs at a time (cron may start one while a slow
    model copy from the previous tick is still running; two ticks would record the same run twice).

    A kernel file lock (flock) is released automatically when its process exits, even if killed, so
    it can never go stale. (The earlier /proc command-line scan matched cron's own `/bin/sh -c ...
    kaggle_scheduler.py tick` parent and made every tick exit.) Returns the open lock file, or None
    if another tick holds the lock.
    """
    import fcntl
    path = STATE.with_name("tick.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def tick(args) -> int:
    global _TICK_LOCK
    lock = acquire_tick_lock()
    if lock is None:
        log("another tick is still running (e.g. copying models to Drive) -- this tick exits")
        return 0
    _TICK_LOCK = lock          # held until this process exits
    try:
        return _tick(args)
    finally:
        _TICK_LOCK = None
        lock.close()


def _tick(args) -> int:
    state = load_state()
    now = datetime.now(timezone.utc)
    status = kaggle_status()
    if state.get("monitor_only") and not state.get("pending"):
        return 0 if args.dry_run else monitor_tick(state, status, now, args)
    summary = fetch_summary() if state.get("pending") and "complete" in status.lower() else None
    d = decide(state, status, summary, now)
    log(f"phase={state['phase']} kaggle={status!r} -> {d['action']}: {d.get('why') or d.get('params') or ''}")
    if args.dry_run or d["action"] in ("wait", "idle"):
        return 0
    if d["action"] in ("record", "halt"):
        pend = state["pending"]
        state["runs"].append({"phase": state["phase"], "started": pend["started"], "finished": now.isoformat(),
                              "params": pend, "hours": round(d["hours"], 2), "summary": summary,
                              "progress_score": d.get("progress_score") or list(progress_score(summary))})
        state["pending"] = None
        if summary and summary.get("finished"):
            state["last_seen_finished"] = summary["finished"]   # monitor mode must not record it again
        if d["action"] == "halt":
            state["halted"] = d["why"]
            log(f"HALTED: {d['why']} -- fix, then run: python tools/kaggle_scheduler.py resume-schedule")
        else:
            if d.get("sync"):
                try:
                    log("models -> Drive: " + sync_models(args.token))
                except Exception as e:  # noqa: BLE001 -- keep the schedule, report the sync problem
                    log(f"WARNING model sync failed (training continues): {e}")
            state["phase"] = d["next_phase"]
            log(f"run recorded ({d['hours']:.2f} h); next phase: {state['phase']}")
        save_state(state)
        if d["action"] == "halt":
            return 0
        # Recorded a finished run: decide again now instead of idling until the next cron tick
        # (the same budget/phase checks apply -- this only removes a 30-minute gap between runs).
        d = decide(state, status, None, now)
        log(f"phase={state['phase']} -> {d['action']}: {d.get('why') or d.get('params') or ''}")
        if d["action"] != "launch":
            return 0
    if d["action"] == "launch":
        launch(d["params"])
        state["pending"] = {**d["params"], "started": now.isoformat()}
        save_state(state)
        log(f"launched {d['params']}")
    return 0


def cmd_status(_args) -> int:
    state = load_state()
    now = datetime.now(timezone.utc)
    print(json.dumps({"phase": state["phase"], "halted": state.get("halted"), "pending": state.get("pending"),
                      "monitor_only": bool(state.get("monitor_only")), "attention": state.get("attention"),
                      "gpu_hours_last_7_days": round(hours_last_7d(state, now), 2), "weekly_budget": WEEKLY_BUDGET_H,
                      "runs": [{k: r[k] for k in ("phase", "finished", "hours", "progress_score")}
                               for r in state["runs"]]}, indent=1))
    if LOG.is_file():
        print("\nlast log lines:\n" + "\n".join(LOG.read_text(encoding="utf-8").splitlines()[-8:]))
    return 0


def cmd_monitor_only(_args) -> int:
    state = load_state()
    state["monitor_only"] = True
    state.pop("hold_after_pending", None)
    save_state(state)
    log("monitor-only: Kaggle's scheduler launches runs; this machine records them and copies models to Drive")
    return 0


def cmd_resume_schedule(_args) -> int:
    state = load_state()
    state["halted"] = None
    save_state(state)
    log("halt cleared by user")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick")
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--token", default=str(REPO.parent / "gdrive_token.json"))
    sub.add_parser("status")
    sub.add_parser("resume-schedule")
    sub.add_parser("monitor-only")
    args = ap.parse_args()
    return {"tick": tick, "status": cmd_status, "resume-schedule": cmd_resume_schedule,
            "monitor-only": cmd_monitor_only}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
