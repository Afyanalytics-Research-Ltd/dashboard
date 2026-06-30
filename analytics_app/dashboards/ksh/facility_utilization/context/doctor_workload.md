# Notice Type: Doctor Workload / Staffing
**keywords:** doctor, doctors, burnout, staffing, concentration, workload, ogutu, eawando, lowino, clinician, staff
**last_updated:** 2026-06-08
**decay_days:** 30
**Covers:** Rule 25 (individual burnout), Rule 26 (concentration risk)
**Facility:** KSH only (EVALUATION_VISITS exists only in KISUMU_CLEAN schema)
**Sources:** Inv 24

---

## Why This Matters
A doctor seeing significantly more patients than their baseline over sustained periods is both a burnout signal and a continuity-of-care risk. When `makinyi` stopped Dec 2025, their 172 visits/month load redistributed silently — `lowino` hit 66% above baseline, `eawando` hit 57% above baseline within weeks. No notice fired. Concentration risk (one doctor handling 33%+ of all evaluations) means a single departure or absence creates immediate capacity failure.

---

## Baselines (KSH, all-time clean accounts)

| Doctor | Avg monthly visits | Active months | Notes |
|--------|--------------------|---------------|-------|
| eawando | 557 | 23 | **33% of all visits** — extreme single-point dependency |
| lowino | 392 | 17 | Started Jan 2025; rapid ramp-up post-makinyi |
| jogutu | 259 | 21 | Consistent Tier 1 |
| makinyi | 172 | 17 | **Stopped Dec 2025** — last visit Dec 15 2025 |
| NODEDE | 67 | 21 | Consistent Tier 2 |
| souma | 42 | 21 | Tier 2 |
| danyango | 40 | 23 | Consistent Tier 2 |
| DACHIENG | 35 | 14 | Thinned out Dec 2025 |

**Tier 1 (>150/mo):** eawando, lowino, jogutu, makinyi — carried ~85% of all evaluation activity before makinyi departure.

**CONCENTRATION RISK:** The notice fires for whoever leads the CURRENT month — this may vary. Historically eawando averages 33% of all visits and is the chronic bottleneck, but any doctor can lead a given month (especially partial months near the data cutoff).
**REDISTRIBUTION TARGETS (doctors with available capacity):** jogutu (259/mo baseline), NODEDE (67/mo), souma (42/mo), danyango (40/mo). These doctors RECEIVE redistributed load. If the notice names jogutu as the top doctor in a month, verify whether it is a partial-month artifact before acting — jogutu's baseline is only 259/mo and they are typically a redistribution recipient, not the concentration risk.

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

## Key Findings (Inv 24)

**The makinyi cascade (already happened — undetected):**
- makinyi stopped Dec 2025 (~172 visits/month absorbed by remaining team)
- lowino: peaked at **652/month Feb 2026** vs 392 baseline — **66% above baseline**
- eawando: peaked at **873/month Mar 2026** vs 557 baseline — **57% above baseline**
- Both doctors sustained elevated load for 2+ consecutive months — exactly the pattern the notice rule catches
- Neither doctor had any automated alert — this was discovered only through investigation

**Current concentration risk:**
- eawando alone: 33% of all KSH evaluation visits, consistently
- Top 3 doctors (eawando, lowino, jogutu): carry **64% of all visits**
- eawando unavailability = immediate 33% outpatient capacity loss with zero warning
- eawando already operating at burnout-risk level (873/month in Mar 2026)

**DACHIENG departure signal:**
- Also thinned out Dec 2025 simultaneously with makinyi — suggests a staffing event in Dec 2025
- Combined effect: two doctors reduced activity in the same month

---

## Recommended Actions

> **Important:** KSH has no formal designation or specialty data in the system. All redistribution recommendations are based on volume capacity only. For any cross-doctor reassignment, the clinical lead must confirm that the receiving doctor's clinical scope covers the case type before redirecting.
>
> **Redistribution order: always Step 1. Hiring: only Step 3, after redistribution has been actioned and proven insufficient over 2+ months. Never recommend hiring as an immediate action.**

**If individual burnout WATCH fires (doctor >150% of their baseline for 2 months):**
- Name the doctor and their current load vs personal baseline in the notice
- **Step 1 — Check who has headroom:** Compare current month volume against baselines. jogutu (259/mo baseline) is typically the most underloaded Tier 1 doctor and the first redistribution target. NODEDE (67/mo), souma (42/mo), and danyango (40/mo) are Tier 2 with spare capacity.
- **Step 2 — Redistribute first:** Propose redirecting 20–30 visits/month from the overloaded doctor to jogutu or available Tier 2 doctors. Clinical lead confirms case-type compatibility before actioning.
- **Step 3 — Monitor:** If load drops back within baseline in the next month, redistribution was sufficient. No hiring needed.
- Look at the 3-month trend — is load still rising or plateauing?

**If individual burnout CRITICAL fires (>200% baseline):**
- Escalate to clinical director — this is a patient safety and retention risk
- **Step 1 — Redistribute immediately:** Identify the highest-volume overloaded doctor. Redirect non-urgent evaluations to the two most available Tier 2 doctors (check current month volumes). Redistribution is the first action, not hiring.
- **Step 2 — Check if a departure triggered the cascade:** If another doctor stopped recently (as with makinyi Dec 2025), the gap is structural, not a temporary spike. In that case, after redistribution is actioned, assess whether the team can sustain current load long-term.
- **Step 3 — Consider a locum only if:** Redistribution has been actioned for 2+ months AND the overloaded doctor is still above 150% of baseline. A locum or new hire fills a confirmed structural gap, not a redistribution problem.

**If concentration risk WATCH fires (top doctor >40% of visits):**
- First: check if this is a partial-month artifact (data cutoff mid-month). If the flagged doctor's baseline is well below 40%, treat as data noise and monitor next full month.
- If genuine: **Step 1 — Redistribute 3–5 visits/week** from the overloaded doctor to available Tier 2 doctors (NODEDE, souma, danyango) or jogutu if they have headroom. Clinical lead confirms clinical scope first.
- **Step 2 — Monitor concentration share** the following month. Target: reduce top-doctor share below 35%.
- New hire only after redistribution has been actioned and proven insufficient over 2 months.

**If concentration risk CRITICAL fires (top doctor >50% of visits):**
- Emergency staffing review with clinical director
- Immediate redistribution across all available Tier 2 doctors for non-urgent cases
- Cross-cover with a locum for non-urgent evaluations while redistribution is assessed — not as a replacement for redistribution
