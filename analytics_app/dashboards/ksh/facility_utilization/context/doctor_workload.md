# Notice Type: Doctor Workload / Staffing
**keywords:** doctor, doctors, burnout, staffing, concentration, workload, ogutu, eawando, lowino, clinician, staff
**last_updated:** 2026-06-08
**decay_days:** 30
**Covers:** Rule 25 (individual burnout), Rule 26 (concentration risk)
**Facility:** KSH only (EVALUATION_VISITS exists only in KISUMU_CLEAN schema)
**Sources:** Inv 24

---

## Why This Matters

A doctor seeing significantly more patients than their personal baseline over sustained periods is both a burnout signal and a continuity-of-care risk. Past staffing departures at KSH demonstrated that workload redistribution can occur silently — elevated loads sustained for two or more consecutive months — with no automated detection until discovered through investigation. This motivated both the individual burnout rule and the concentration risk rule.

Concentration risk (one doctor handling a large share of all evaluations) means a single departure or absence creates immediate capacity failure across multiple wards simultaneously.

Current workloads, personal baselines, and concentration shares are in the snapshot — do not use named individuals or specific volumes from this file as current state.

---

## Workload Tiers (interpretive framework — current values from snapshot)

KSH evaluation activity is distributed across doctors who fall into two interpretive tiers based on typical monthly volume. The snapshot provides current volumes and personal baselines per doctor.

- **High-volume clinicians (Tier 1):** doctors whose monthly evaluation volume consistently exceeds 150 visits. These carry the majority of facility evaluation load. A departure or sustained absence in this tier creates an immediate redistribution gap.
- **Supporting clinicians (Tier 2):** doctors whose typical monthly volume is below 150 visits. These are the primary redistribution recipients — they have spare capacity relative to their personal baseline and are the first candidates for redirected load. Clinical scope must be confirmed before any redirection.

The notice fires for whoever leads the current month — this may vary. If the flagged doctor's personal baseline is well below the concentration threshold, verify that the data month is complete before acting.

---

## Thresholds

**Individual burnout rule (rolling 3-month personal average baseline):**

| Level | Threshold | Rule |
|-------|-----------|------|
| WATCH | Monthly visits > 150% of personal 3-month rolling avg | 2 consecutive months |
| CRITICAL | Monthly visits > 200% of personal 3-month rolling avg | 2 consecutive months |

**Concentration risk rule (share of total monthly visits):**

| Level | Threshold |
|-------|-----------|
| WATCH | Top doctor handles >40% of monthly visits |
| CRITICAL | Top doctor handles >50% of monthly visits |

---

## Recommended Actions

> **Important:** KSH has no formal designation or specialty data in the system. All redistribution recommendations are based on volume capacity only. For any cross-doctor reassignment, the clinical lead must confirm that the receiving doctor's clinical scope covers the case type before redirecting.
>
> **Redistribution order: always Step 1. Hiring: only Step 3, after redistribution has been actioned and proven insufficient over 2+ months. Never recommend hiring as an immediate action.**

**If individual burnout WATCH fires (doctor >150% of their baseline for 2 months):**
- Name the doctor and their current load vs personal baseline in the notice
- **Step 1 — Check who has headroom:** From the snapshot, identify clinicians currently operating below their personal baseline. Supporting (Tier 2) clinicians with spare capacity are the first redistribution targets. Clinical lead confirms case-type compatibility before actioning.
- **Step 2 — Redistribute first:** Propose redirecting 20–30 visits/month from the overloaded doctor to those with available headroom.
- **Step 3 — Monitor:** If load drops back within baseline in the next month, redistribution was sufficient. No hiring needed.
- Look at the 3-month trend — is load still rising or plateauing?

**If individual burnout CRITICAL fires (>200% baseline):**
- Escalate to clinical director — this is a patient safety and retention risk
- **Step 1 — Redistribute immediately:** Identify the highest-volume overloaded doctor. Redirect non-urgent evaluations to the clinicians with the most available headroom relative to their personal baseline. Redistribution is the first action, not hiring.
- **Step 2 — Check if a departure triggered the cascade:** If another doctor reduced activity recently, the gap may be structural rather than a temporary spike. After redistribution is actioned, assess whether the remaining team can sustain current load long-term.
- **Step 3 — Consider a locum only if:** Redistribution has been actioned for 2+ months AND the overloaded doctor is still above 150% of baseline. A locum or new hire fills a confirmed structural gap, not a redistribution problem.

**If concentration risk WATCH fires (top doctor >40% of visits):**
- First: check if this is a partial-month artifact (data cutoff mid-month). If the flagged doctor's personal baseline is well below 40% of total facility volume, treat as data noise and monitor next full month.
- If genuine: **Step 1 — Redistribute 3–5 visits/week** from the overloaded doctor to clinicians currently below their personal baseline. Clinical lead confirms clinical scope first.
- **Step 2 — Monitor concentration share** the following month. Target: reduce top-doctor share below 35%.
- New hire only after redistribution has been actioned and proven insufficient over 2 months.

**If concentration risk CRITICAL fires (top doctor >50% of visits):**
- Emergency staffing review with clinical director
- Immediate redistribution across all available supporting clinicians for non-urgent cases
- Cross-cover with a locum for non-urgent evaluations while redistribution is assessed — not as a replacement for redistribution
