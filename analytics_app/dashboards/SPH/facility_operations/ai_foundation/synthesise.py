"""Briefing synthesis — translation layer between PrioritisedProblem and management narrative.

Claude/Groq receives a deterministically-formatted evidence payload and synthesises
management language. It does not analyse, discover, or interpret beyond the supplied evidence.

Usage:
    from ai_foundation.synthesise import synthesise
    briefing = synthesise(prioritised_problem, provider="groq")  # testing
    briefing = synthesise(prioritised_problem, provider="claude") # production
"""
from __future__ import annotations

import json
import os

from ai_foundation.contracts import OperationalBriefing, PrioritisedProblem
from ai_foundation.registry import CARD_REGISTRY

_GROQ_MODEL = "llama-3.3-70b-versatile"
_CLAUDE_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are a medical operations analyst producing structured operational briefings for hospital management.

Your role is to SYNTHESISE — not analyse. A deterministic analytical system has already done the analysis. Your job is to translate verified evidence into clear management language.

CRITICAL OUTPUT RULES — violating any of these invalidates the response:

SPACING: Every word must be separated by exactly one space character. Never run adjacent words together. Check every word boundary before outputting.

STATUS LANGUAGE: When describing whether a mechanism or effect was found, use exactly the word "supported" (not "confirmed", "proven", "established", "demonstrated", or any stronger synonym). When a mechanism was not tested due to missing data, say "not applicable". Mirror the evidence — do not upgrade its certainty.

ACTION SCOPE: The action field covers only investigation and review of the demand pattern — not operational changes. Only recommend staffing, scheduling, or capacity actions if those mechanisms are explicitly marked SUPPORTED in the evidence. If only volume is supported, the action should direct the responsible person to investigate the demand pattern and its operational cause. Do not use the word "adjust" unless a capacity mechanism is supported.

BREVITY: Each field must be 1 sentence, maximum 2. Do not pad. Stop writing as soon as the field is complete.

COMPLETENESS: Every field must contain a full sentence. Do not truncate mid-sentence. If a field would be too long at 2 sentences, cut detail — never cut the sentence ending.

EVIDENCE ONLY: Use only what is in the EVIDENCE block. Do not add domain knowledge, assumptions, or inferences. Do not use internal identifiers, file paths, schema names, or metric IDs — translate to plain operational language.
"""

_USER_TEMPLATE = """\
EVIDENCE:
{evidence}

Respond with a JSON object containing exactly these seven string fields. Each value must be a complete sentence (maximum two sentences). Do not truncate any field.

{{
  "what": "State the metric elevation: percentage above baseline, current value, baseline value, and sample size. If the SIGNAL section contains a scope note about deterioration vs absolute level, incorporate that framing — e.g. 'currently worsening' or 'recent deterioration'.",
  "where": "Name the attributed department, its median wait time, and the data confidence qualifier in plain terms.",
  "when": "State peak day and peak hour as clock time (e.g. 09:00). Describe the concentration window.",
  "mechanism": "State which mechanism is supported and what the evidence shows. State which mechanisms are not applicable and why, in operational terms only — no file or schema references.",
  "downstream": "State what downstream effect is observed for this cohort. State what is not applicable and why, in plain operational terms.",
  "unknowns": "For each missing data source, name what operational question it would answer if available.",
  "action": "Name the responsible role and the single investigative step they should take regarding the demand pattern. No operational changes unless capacity is supported. No thresholds cited."
}}
"""


def _build_evidence_payload(pp: PrioritisedProblem) -> str:
    """Format the PrioritisedProblem evidence graph into a structured text block.

    Deterministic — no LLM involvement. Every field traces to a StepResult.
    """
    state = pp.problem.metric_state
    sig = pp.problem.signature
    by_id = {r.step_id: r for r in pp.problem.step_results}

    lines: list[str] = []

    # --- Signal ---
    lines.append("SIGNAL")
    lines.append(f"  Metric        : {pp.problem.metric_id}")
    lines.append(f"  Current value : {state.value:.0f} min  (28-day rolling average P50)")
    lines.append(f"  Baseline      : {state.baseline:.0f} min  (prior 28-day average P50)")
    change_pct = f"{state.change:+.1%}" if state.change is not None else "N/A"
    lines.append(f"  Change        : {change_pct} above baseline")
    _card_def = CARD_REGISTRY.get(pp.problem.metric_id)
    q = by_id.get("quantify")
    if q and q.evidence:
        _slabel = _card_def.sample_label if _card_def else "records"
        lines.append(f"  Sample        : {q.evidence.get('n', 'N/A')} {_slabel}")
    if _card_def and _card_def.scope_note:
        lines.append(f"  Scope         : {_card_def.scope_note}")
    lines.append(f"  Priority score: {pp.priority_score}  ({pp.problem.card_id}: severity={pp.severity_weight} x impact={pp.impact_weight})")

    # --- Attribution ---
    attr = by_id.get("dept_attribution")
    status = attr.status if attr else "MISSING"
    lines.append(f"\nATTRIBUTION  [status: {status}]")
    if attr and attr.status == "SUPPORTED" and attr.evidence:
        ev = attr.evidence
        lines.append(f"  Department  : {ev.get('top_attribution')}")
        depts = ev.get("departments", [])
        if depts:
            top = depts[0]
            lines.append(f"  P50 wait    : {top.get('p50_mins')} min")
        lines.append(f"  Limitation  : {ev.get('limitation')}")

    # --- Temporal ---
    temp = by_id.get("temporal_pattern")
    status = temp.status if temp else "MISSING"
    lines.append(f"\nTEMPORAL  [status: {status}]")
    if temp and temp.status == "SUPPORTED" and temp.evidence:
        ev = temp.evidence
        pd = ev.get("peak_dow", {})
        ph = ev.get("peak_hour", {})
        lines.append(f"  Peak day  : {pd.get('day_name')}  (avg {pd.get('avg_median_wait_mins')} min across all hours that day)")
        lines.append(f"  Peak hour : {ph.get('hour_of_day'):02d}:00  (avg {ph.get('avg_median_wait_mins')} min)")

    # --- Mechanism ---
    mech = by_id.get("mechanism_test")
    status = mech.status if mech else "MISSING"
    lines.append(f"\nMECHANISM  [step status: {status}]")
    if mech and mech.evidence:
        mechs = mech.evidence.get("mechanisms", {})
        for name, m in mechs.items():
            mstatus = m.get("status", "UNKNOWN")
            if mstatus == "SUPPORTED":
                lines.append(f"  {name}  [SUPPORTED]")
                lines.append(f"    Peak window hours : {m.get('peak_window_hours')}")
                lines.append(f"    Peak arrivals/hr  : {m.get('peak_window_avg_daily_arrivals')}")
                lines.append(f"    Overall avg/hr    : {m.get('overall_avg_hourly_arrivals')}")
                lines.append(f"    Ratio             : {m.get('peak_ratio')}x  (threshold {m.get('threshold')}x)")
            else:
                lines.append(f"  {name}  [{mstatus}]  — {m.get('reason', '')}")

    # --- Downstream: pharmacy (consult_p50 only — pharmacy_p50 is the metric, not a downstream) ---
    pharm = by_id.get("downstream_pharmacy")
    if pharm is not None:
        status = pharm.status
        lines.append(f"\nDOWNSTREAM — pharmacy  [status: {status}]")
        if pharm.evidence:
            ev = pharm.evidence
            if status == "NOT_APPLICABLE":
                lines.append(f"  Reason : {ev.get('reason')}")
            elif status == "SUPPORTED":
                lines.append(f"  Cohort pharmacy P50 : {ev.get('cohort_pharm_p50_mins')} min")
                lines.append(f"  Baseline P50        : {ev.get('baseline_pharm_p50_mins')} min")
                lines.append(f"  Ratio               : {ev.get('downstream_ratio')}x  elevated={ev.get('downstream_elevated')}")

    # --- Downstream: incomplete care ---
    inc = by_id.get("downstream_incomplete_care")
    status = inc.status if inc else "MISSING"
    lines.append(f"\nDOWNSTREAM — incomplete care  [status: {status}]")
    if inc and inc.evidence:
        ev = inc.evidence
        if status == "SUPPORTED":
            lines.append(f"  Cohort incomplete rate  : {ev.get('cohort_incomplete_pct')}%  ({ev.get('cohort_incomplete_n')} of {ev.get('cohort_visits')} visits)")
            lines.append(f"  OPD baseline rate       : {ev.get('baseline_incomplete_pct')}%")
            lines.append(f"  Ratio                   : {ev.get('incomplete_ratio')}x  (threshold {ev.get('elevation_threshold')}x)")
            lines.append(f"  Downstream elevated     : {ev.get('downstream_elevated')}")
        elif status == "NOT_APPLICABLE":
            lines.append(f"  Reason : {ev.get('reason')}")

    return "\n".join(lines)


def _load_env() -> None:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _call_groq(evidence: str) -> OperationalBriefing:
    from groq import Groq

    _load_env()
    api_key = os.environ.get("GROQ_API") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API not set. Add it to C:\\Users\\HomePC\\Documents\\TKLK\\.env"
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(evidence=evidence)},
        ],
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    return OperationalBriefing(**data)


def _call_claude(evidence: str) -> OperationalBriefing:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _USER_TEMPLATE.format(evidence=evidence)},
        ],
    )
    raw = response.content[0].text
    # Claude returns JSON in the response text when instructed
    start = raw.find("{")
    end = raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    return OperationalBriefing(**data)


def synthesise(pp: PrioritisedProblem, provider: str = "groq") -> OperationalBriefing:
    """Translate a PrioritisedProblem into a management briefing.

    provider: "groq" (testing) | "claude" (production)
    """
    evidence = _build_evidence_payload(pp)
    if provider == "groq":
        return _call_groq(evidence)
    if provider == "claude":
        return _call_claude(evidence)
    raise ValueError(f"Unknown provider: {provider!r}. Use 'groq' or 'claude'.")
