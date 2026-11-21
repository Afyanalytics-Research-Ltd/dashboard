"""Runs enabled domains in dependency order. Failures are isolated: a failed domain
only causes its *dependents* to be skipped; everything else still runs."""
from __future__ import annotations

import argparse
import logging
from graphlib import TopologicalSorter
from typing import Mapping

from .config import DomainConfig, load_configs
from .context import IngestResult
from .hooks import Hooks
from .registry import load_ingestor_cls

log = logging.getLogger("ingestion")


def _selected_domains(
    configs: Mapping[str, DomainConfig],
    only: list[str] | None,
) -> set[str]:
    if not only:
        return set(configs)

    selected = set(only)
    unknown = selected - set(configs)
    if unknown:
        raise ValueError(f"Unknown domain(s): {sorted(unknown)}. Known: {sorted(configs)}")

    pending = list(selected)
    while pending:
        name = pending.pop()
        for dependency in configs[name].depends_on:
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return selected


def run_all(
    config_path: str,
    only: list[str] | None = None,
    hooks: Hooks | None = None,
) -> list[IngestResult]:
    configs = load_configs(config_path)
    selected = _selected_domains(configs, only)
    graph = {n: c.depends_on for n, c in configs.items()}
    results: dict[str, IngestResult] = {}

    for name in TopologicalSorter(graph).static_order():
        if name not in selected:
            continue
        cfg = configs[name]
        if not cfg.enabled:
            results[name] = IngestResult(name, "-", "skipped", error="domain is disabled")
            continue
        blocked = [
            dependency
            for dependency in cfg.depends_on
            if dependency not in results or results[dependency].status != "success"
        ]
        if blocked:
            results[name] = IngestResult(
                name,
                "-",
                "skipped",
                error=f"upstream unavailable: {blocked}",
            )
            continue
        try:
            ingestor = load_ingestor_cls(cfg.ingestor)(cfg, hooks=hooks)
            results[name] = ingestor.run()
        except Exception as exc:
            log.exception("%s failed", name)
            results[name] = IngestResult(name, "-", "failed", error=str(exc))
    return list(results.values())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="domains.yaml")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    for r in run_all(args.config, args.only):
        print(f"{r.domain:<15} {r.status:<8} {r.counts} {r.error or ''}")
