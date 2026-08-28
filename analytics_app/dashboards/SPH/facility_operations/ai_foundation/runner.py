"""AI Foundation Intelligence runner.

Executes the full pipeline for all registered metric cards and persists
an IntelligenceRun JSON to disk. Standalone — no Streamlit imports.
get_connection is injected.

Usage:
    python -m ai_foundation.runner
    INTELLIGENCE_RUN_PATH=ai_foundation/latest_run.json python -m ai_foundation.runner

Exit codes:
    0 — ok or no_trigger (pipeline ran cleanly)
    1 — pipeline_failed (DB/query error; empty problems written)
    2 — synthesis_failed (pipeline ok, LLM unavailable; evidence_payload written as fallback)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ai_foundation.contracts import OperationalBriefing, TriggerResult
from ai_foundation.engine import (
    build_problem,
    evaluate_trigger,
    group_problems,
    populate_metric_state,
    prioritise_problems,
    run_card,
)
from ai_foundation.persistence import (
    IntelligenceRun,
    PersistedBriefing,
    PersistedProblem,
    PersistedSignature,
)
from ai_foundation.registry import (
    CARD_REGISTRY,
    CONSULT_P50,
    CONSULT_P50_TRIGGER,
    IMAGING_TAT_P50,
    IMAGING_TAT_P50_TRIGGER,
    LAB_COLLECT_P50,
    LAB_COLLECT_P50_TRIGGER,
    PHARMACY_P50,
    PHARMACY_P50_TRIGGER,
)
from ai_foundation.synthesise import _build_evidence_payload, synthesise

_DEFAULT_OUTPUT = Path(__file__).parent / "latest_run.json"

# All active metric/trigger/card combinations — extend here as new cards are built
_PIPELINE = [
    (CONSULT_P50,      CONSULT_P50_TRIGGER,      "consult_p50"),
    (PHARMACY_P50,     PHARMACY_P50_TRIGGER,     "pharmacy_p50"),
    (LAB_COLLECT_P50,  LAB_COLLECT_P50_TRIGGER,  "lab_collect_p50"),
    (IMAGING_TAT_P50,  IMAGING_TAT_P50_TRIGGER,  "imaging_tat_p50"),
]


def _make_connection():
    """Snowflake connection — mirrors dashboard.db.get_connection without Streamlit."""
    import snowflake.connector
    from cryptography.hazmat.primitives import serialization
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "")
    with open(key_path, "rb") as f:
        p_key = serialization.load_pem_private_key(f.read(), password=None)
    private_key_bytes = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        private_key=private_key_bytes,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )


def _to_persisted_briefing(b: OperationalBriefing) -> PersistedBriefing:
    return PersistedBriefing(
        what=b.what, where=b.where, when=b.when,
        mechanism=b.mechanism, downstream=b.downstream,
        unknowns=b.unknowns, action=b.action,
    )


def _write_atomic(run: IntelligenceRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, output_path)


def run(
    get_connection,
    output_path: Path = _DEFAULT_OUTPUT,
    provider: str = "groq",
) -> IntelligenceRun:
    """Execute pipeline for all registered cards. Returns the written IntelligenceRun.

    Collects problems from every card whose trigger fires. Cards that fail
    individually are skipped (logged to stderr); the run continues for
    remaining cards. Raises RuntimeError only when ALL cards fail.
    """
    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    all_problems = []
    failed_cards: list[str] = []

    for metric_def, trigger_def, card_key in _PIPELINE:
        try:
            state = populate_metric_state(metric_def, get_connection=get_connection)
            trigger = evaluate_trigger(state, trigger_def)
        except Exception as exc:
            print(f"SKIP {card_key} — metric/trigger error: {exc}", file=sys.stderr)
            failed_cards.append(card_key)
            continue

        if trigger != TriggerResult.FIRE:
            print(f"  {card_key}: no_trigger (change={state.change})")
            continue

        print(f"  {card_key}: FIRE (change={state.change:+.1%})")

        try:
            card = CARD_REGISTRY[card_key]
            step_results = run_card(card, state, get_connection=get_connection)
            problem = build_problem(card, state, step_results)
            if problem is not None:
                all_problems.append(problem)
            else:
                print(f"SKIP {card_key} — build_problem returned None", file=sys.stderr)
                failed_cards.append(card_key)
        except Exception as exc:
            print(f"SKIP {card_key} — investigation error: {exc}", file=sys.stderr)
            failed_cards.append(card_key)
            continue

    # All cards failed — write error run and raise
    if failed_cards and not all_problems:
        result = IntelligenceRun(run_ts=run_ts, status="pipeline_failed", problems=[])
        _write_atomic(result, output_path)
        raise RuntimeError(f"pipeline_failed — all cards failed: {failed_cards}")

    # Nothing fired
    if not all_problems:
        result = IntelligenceRun(run_ts=run_ts, status="no_trigger", problems=[])
        _write_atomic(result, output_path)
        return result

    grouped = group_problems(all_problems)
    ranked = prioritise_problems(grouped)

    status = "ok"
    persisted_problems: list[PersistedProblem] = []

    for pp in ranked:
        evidence_payload = _build_evidence_payload(pp)
        sig = pp.problem.signature
        persisted_sig = PersistedSignature(
            attribution=sig.attribution,
            temporal_pattern=sig.temporal_pattern,
            cohort=sig.cohort,
            mechanism=sig.mechanism,
        )

        try:
            briefing = synthesise(pp, provider=provider)
            persisted_briefing = _to_persisted_briefing(briefing)
        except Exception as exc:
            print(f"synthesis_failed for {pp.problem.metric_id}: {exc}", file=sys.stderr)
            persisted_briefing = None
            status = "synthesis_failed"

        _card = CARD_REGISTRY.get(pp.problem.metric_id)
        persisted_problems.append(PersistedProblem(
            metric_id=pp.problem.metric_id,
            priority_score=pp.priority_score,
            severity=_card.severity if _card else None,
            impact_domain=_card.impact_domain if _card else None,
            signature=persisted_sig,
            briefing=persisted_briefing,
            evidence_payload=evidence_payload,
        ))

    result = IntelligenceRun(run_ts=run_ts, status=status, problems=persisted_problems)
    _write_atomic(result, output_path)
    return result


def main() -> int:
    output_path = Path(os.getenv("INTELLIGENCE_RUN_PATH", str(_DEFAULT_OUTPUT)))
    provider = os.getenv("INTELLIGENCE_PROVIDER", "groq")

    print(f"output={output_path}  provider={provider}")

    try:
        result = run(
            get_connection=_make_connection,
            output_path=output_path,
            provider=provider,
        )
    except RuntimeError as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        return 1

    print(f"status={result.status}  problems={len(result.problems)}")

    if result.status == "pipeline_failed":
        return 1
    if result.status == "synthesis_failed":
        print("WARNING: synthesis failed — evidence_payload written as fallback", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
