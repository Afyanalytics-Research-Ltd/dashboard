"""Configuration for the SPH inventory intelligence engine.

Two configuration surfaces exist, and the distinction is a design law:

- :class:`BusinessInputs` — economic quantities that only the hospital can
  supply (holding rate, ordering cost, shortage-cost weights). These are
  *not* heuristics, but they must be explicit and provenance-documented.
  Any value not yet confirmed by St. Peter's carries
  ``provenance = PLACEHOLDER_PROVENANCE`` and is surfaced by
  :meth:`BusinessInputs.placeholders` so every downstream output can flag it.
- :class:`EngineSettings` — reproducibility and statistical-procedure
  parameters (seeds, sample counts, CV geometry, FDR level). These are
  documented procedural choices, not tunable heuristics: no smoothing
  constants, thresholds, or demand multipliers live here — those are fitted
  upstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Optional

#: Sentinel provenance for business inputs the hospital has not yet confirmed.
#: Every analytics output that depends on such an input must surface it.
PLACEHOLDER_PROVENANCE = "PLACEHOLDER — business input required"


@dataclass
class BusinessInputs:
    """Explicit economic inputs, each paired with a provenance string.

    Replace a default by constructing with the confirmed value *and* a real
    provenance, e.g.::

        BusinessInputs(
            holding_rate_annual=0.22,
            holding_rate_annual_provenance="SPH finance office, 2026-08 email",
        )

    Until then the defaults remain flagged placeholders: they exist only so
    the pipeline can run end-to-end while clearly reporting which numbers are
    unconfirmed. EOQ in particular is emitted as NULL
    unless ``ordering_cost`` has real provenance.
    """

    # Annual inventory holding rate as a fraction of unit value
    # (drives overage cost c_o in the newsvendor critical ratio).
    holding_rate_annual: float = 0.25
    holding_rate_annual_provenance: str = PLACEHOLDER_PROVENANCE

    # Fixed cost per purchase order (KES). None → EOQ outputs are NULL with
    # reason, never a fabricated constant.
    ordering_cost: Optional[float] = None
    ordering_cost_provenance: str = PLACEHOLDER_PROVENANCE

    # Shortage-cost weight: underage cost c_u = shortage_weight × unit_price.
    # The neutral default (1.0 — a stockout costs one unit's price) is a
    # placeholder, not an estimate.
    shortage_weight_default: float = 1.0
    shortage_weight_default_provenance: str = PLACEHOLDER_PROVENANCE

    # Optional per-therapeutic-class overrides of the shortage weight,
    # e.g. {"Opioid analgesics": 10.0}. Empty until the hospital ranks
    # clinical criticality.
    shortage_weight_by_class: dict[str, float] = field(default_factory=dict)
    shortage_weight_by_class_provenance: str = PLACEHOLDER_PROVENANCE

    def shortage_weight(self, therapeutic_class: Optional[str] = None) -> float:
        """Resolve the shortage weight for an item's therapeutic class."""
        if therapeutic_class is not None and therapeutic_class in self.shortage_weight_by_class:
            return self.shortage_weight_by_class[therapeutic_class]
        return self.shortage_weight_default

    def placeholders(self) -> list[str]:
        """Names of business inputs still on placeholder provenance.

        Downstream tables copy this list into ``run_metadata`` /
        ``inputs_provenance`` so no unconfirmed economics hide in an output.
        """
        suffix = "_provenance"
        return [
            f.name[: -len(suffix)]
            for f in fields(self)
            if f.name.endswith(suffix) and getattr(self, f.name) == PLACEHOLDER_PROVENANCE
        ]

    def provenance_map(self) -> dict[str, str]:
        """Full input → provenance mapping for run metadata."""
        suffix = "_provenance"
        return {
            f.name[: -len(suffix)]: getattr(self, f.name)
            for f in fields(self)
            if f.name.endswith(suffix)
        }


def _default_anthropic_model() -> str:
    """Anthropic model id, overridable via env ``ANTHROPIC_MODEL``."""
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5").strip()


@dataclass
class EngineSettings:
    """Reproducibility and statistical-procedure settings.

    None of these is a heuristic acting on the data: they fix the *geometry*
    of statistical procedures (how many Monte Carlo draws, which quantiles to
    report, how CV folds are laid out) and the error rate the anomaly layer
    controls. Model parameters themselves (smoothing constants, seasonality,
    overdispersion, cluster assignments) are always fitted or selected by
    cross-validation upstream.
    """

    # Seed for every Monte Carlo simulation and clustering run.
    random_seed: int = 42

    # Monte Carlo sample count for predictive distributions and the
    # demand-over-replenishment-cycle convolution.
    mc_samples: int = 2000

    # Reported predictive quantiles (distributions, not point estimates).
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

    # Rolling-origin cross-validation geometry for model selection:
    # items whose history cannot support >= cv_min_folds folds of
    # cv_horizon_days route to hierarchical pooling instead.
    cv_min_folds: int = 3
    cv_horizon_days: int = 28

    # Benjamini–Hochberg false-discovery rate for anomaly surveillance.
    # This is a chosen, documented statistical *error rate* — the
    # probability of false flags we accept across the item portfolio under
    # multiple testing — not a data heuristic like a per-item z-cutoff.
    fdr_level: float = 0.05

    # Anthropic model id (env-overridable via ANTHROPIC_MODEL).
    anthropic_model: str = field(default_factory=_default_anthropic_model)
