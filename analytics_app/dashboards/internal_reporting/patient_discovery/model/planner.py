"""Planner: interpret, measure what the beliefs are missing (blocking gaps first, cheapest first, within a budget),
interpret again; stop when nothing is missing or a limit is reached.

Limits: `budget` (estimated rows read), `max_statements` (SQL statements sent), `skip_kinds` (measurement kinds
never run, e.g. the explanatory 'categories' and 'conditional' on real data). A failing measurement is recorded
(masked) and never retried in the same run; everything measured before it is kept.

Persistence is the caller's: pass earlier observations in, keep the returned ones.
"""
import re
from dataclasses import dataclass, field

from . import measure as M
from .concepts import Observation
from .interpret import fingerprints, fp_of, interpret


@dataclass
class Result:
    beliefs: object
    observations: list
    measured: list = field(default_factory=list)     # gaps executed this run, in order
    failed: list = field(default_factory=list)       # (gap, masked error)
    cost: float = 0.0
    queries: int = 0
    rounds: int = 0
    unmet: list = field(default_factory=list)        # gaps left when a limit was reached


def _mask(err):
    return re.sub(r"'[^']*'", "'<value>'", str(err))[:300]


def understand(wh, statements, now=None, budget=None, dialect=M.SQLite, observations=(), max_rounds=40,
               max_statements=None, skip_kinds=(), catalogue=None, dry_run=False):
    """dry_run: interpret once and return every missing measurement as unmet; nothing is executed."""
    catalogue = catalogue if catalogue is not None else wh.catalogue()
    cat = {o["fqn"]: o for o in catalogue}
    fps = fingerprints(catalogue)
    store = {(o.kind, o.subject, o.args): o for o in observations}
    res = Result(None, [])
    q0 = getattr(wh, "n_queries", 0)
    failed, beliefs = set(), None
    at = now.isoformat(timespec="seconds") if now else None
    for res.rounds in range(1, max_rounds + 1):
        beliefs = interpret(catalogue, list(store.values()), statements, now)
        # the interpreter asks only for missing or stale observations (fingerprint changed, or a view's has expired)
        todo = [g for g in beliefs.gaps if g.kind not in skip_kinds and (g.kind, g.subject, g.args) not in failed]
        if not todo or dry_run:
            res.unmet = todo
            break
        todo.sort(key=lambda g: (not g.blocking, g.priority, g.cost))
        ran = False
        for g in todo:
            if budget is not None and res.cost + g.cost > budget:
                continue
            n = len(M.statements_for(dialect, cat, g))
            if max_statements is not None and getattr(wh, "n_queries", 0) - q0 + n > max_statements:
                continue
            try:
                values = M.run(wh, dialect, cat, g)
            except PermissionError:
                raise
            except Exception as e:                         # keep what was measured; never retry this gap
                failed.add((g.kind, g.subject, g.args))
                res.failed.append((g, _mask(e)))
                continue
            store[(g.kind, g.subject, g.args)] = Observation(g.kind, g.subject, g.args, fp_of(fps, g.subject),
                                                             values, at)
            res.cost += g.cost
            res.measured.append(g)
            ran = True
        if not ran:
            res.unmet = todo
            break
    res.beliefs = beliefs
    res.observations = list(store.values())
    res.queries = getattr(wh, "n_queries", 0) - q0
    return res
