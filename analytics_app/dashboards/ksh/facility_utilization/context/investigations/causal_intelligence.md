# Causal Intelligence — Cross-Domain Findings (KSH)
**keywords:** causal, cross-domain, all investigations, investigation findings, full story, go deeper, complete picture, all findings, summary of investigations, what were all the findings, all causal findings, overview, ruled out, disproved, hypothesis, tested hypotheses, what was disproved, what was ruled out, private female anomaly, general maternity dama, future investigations, what has been tested, areas investigated, cross domain findings
**Facility:** KSH only
**Last updated:** 2026-06-09 (Phase 14 — CD1–CD12 complete + supplemental investigations)

---

## What This Section Covers

Twelve cross-domain investigations were run (CD1–CD12) plus supplementals. Two confirmed causal chains, one patient safety finding, and several operational observations with direct data evidence. Five hypotheses disproved. When asked why something is happening, what affects a service, or how one area impacts another — draw from this section.

---

## CONFIRMED FINDING 1 — Monday Peak Drives Physician Concentration in General Female (CD5)

**The chain:**
Monday 14:00–18:00 is KSH's peak evaluation window. During this window, physician availability across the facility drops from 12 distinct active doctors to 3–4. The 3–4 doctors who remain active concentrate disproportionately in General Female — the ward absorbing the highest admission volume during peak.

**What the data shows:**

Admission distribution shift (Monday 14–18 vs all other times):
- General Female: 31.0% off-peak → **39.8% peak** (+8.8 pp)
- Pediatric General: 23.1% off-peak → 18.6% peak (−4.5 pp)
- Private Male: 6.2% off-peak → 3.5% peak (−2.7 pp)
- Private Female: 5.7% off-peak → 4.4% peak (−1.3 pp)

Physician presence during peak (distinct active doctors per ward):
- General Female: 12 off-peak → **4 peak**
- General Male: 12 off-peak → **4 peak**
- General Maternity: 12 off-peak → **3 peak**
- Pediatric General: 12 off-peak → **3 peak**
- Private Female: 11 off-peak → **0 peak**
- Private Male: 9 off-peak → **0 peak**
- Private Maternity: 9 off-peak → **0 peak**

Doctors active during peak shift disproportionately toward General Female — the ward absorbing the highest admission volume in that window.

**What this means:**
During Monday 14–18, fewer physicians are covering more patients. Those available are drawn toward the ward with highest admission volume. Private wards receive zero physician evaluations and zero new admissions during this window — not because patients are absent but because physician availability has collapsed to a small pool concentrated elsewhere in the facility.

---

## CONFIRMED FINDING 2 — Single-Clinician Concentration Risk Across All Wards (CD6)

**The chain:**
E.Awando evaluates 34–46% of admissions in every ward at KSH. This concentration is not ward-specific — it spans the entire facility. When this capacity is unavailable, every ward is simultaneously affected, not just one.

**What the data shows:**

E.Awando's share of admissions per ward:
- General Male: 45.5% of ward admissions
- Private Female: 44.8%
- Private Male: 42.0%
- General Female: 41.4%
- General Maternity: 34.8%
- Pediatric General: 33.9%
- Private Maternity: 20.0% (S.Ouma leads here at 24%)

J.Ogutu is second across most wards at 14–17%. The top two doctors together account for 48–60% of admissions in every ward.

**Historical evidence — M.Akinyi departure:**
M.Akinyi contributed 10–12% of admissions across General Female, General Male, General Maternity, and Private Male before departing. Her load redistributed silently onto the remaining doctors — E.Awando absorbed the majority of it — with no structural redistribution plan in place. This is the clearest documented example at KSH of how a contributing evaluator's departure concentrates load further onto already high-volume clinicians.

**Note — Private Female is E.Awando-dependent, not M.Akinyi-linked (confirmed 2026-06-09):**
M.Akinyi covered Private Female only briefly in 2025 (3 total admissions), then stopped entirely — months before any Private Female anomaly. From September 2025, E.Awando is the sole consistent physician for Private Female. When E.Awando's activity in Private Female is low, the ward records near-zero admissions; when multiple doctors are simultaneously active, admissions recover. Private Female performance tracks E.Awando's presence, not any departure event.

**What this means:**
Any period of unavailability for E.Awando creates simultaneous evaluation gaps across all seven wards. Private wards are most exposed — Private Female has only 5 distinct evaluating doctors in total, Private Male has 7, compared to General Female's 10. The M.Akinyi departure demonstrates that when a contributing evaluator leaves, load concentrates further onto the remaining high-volume doctors without a redistribution plan.

---

## CONFIRMED FINDING 3 — Renal Care Pathway Gap (CD12) ⚠️ PATIENT SAFETY

**The finding:**
134 distinct critical Creatinine visit_ids (CL/CH flags in EVENTS_RAW) since 2024-01-01, representing 126 unique patients (8 patients had 2+ critical events). 41% of critical Creatinine visits result in no inpatient admission. DAMA rate for those admitted (57%) equals the facility baseline of 56.82% (CD7) — not elevated. KSH's dialysis programme recorded zero sessions from May 2025 through the investigation period end.

**What the data shows:**

Critical Creatinine placement (134 distinct visit_ids, 126 unique patients, since 2024-01-01):
- Outpatient — no admission: **56 visits (41%)**
- Admitted — Pediatric General: 21 patients
- Admitted — General Female: 20 patients
- Admitted — General Maternity: 15 patients
- Admitted — General Male: 13 patients
- Admitted — Private Male: 5 patients
- Admitted — Private Maternity: 4 patients
- Admitted — Private Female: 1 patient

Discharge outcomes (admitted patients with critical Creatinine):
- Patient Request (DAMA): **44 patients · avg LOS 3.6d · 4.5% readmission rate**
- Patient is Stable: 28 patients · avg LOS 4.8d · 0% readmission
- Referral: 2 patients · avg LOS 18.0d
- Death: 1 patient · LOS 1.0d

DAMA rate (57%) = facility baseline (56.82%, CD7). Critical renal patients do not leave against medical advice at a higher rate than any other patient group.

Dialysis programme — full history from rpt_dialysis (KISUMU_CLEAN):
- March 2025: 2 sessions · 2 patients · avg 3.5h · KES 52,200
- April 2025: 1 session · 1 patient · avg 2.0h · KES 119,100
- **May 2025 onward through investigation window: zero sessions**

Return visit tracking — all 126 critical creatinine patients (EVALUATION_VISITS.PATIENT join):

| Index Disposition | Total | Returned | Never Returned | Admitted on Return | Outpatient Only | Median Days |
|---|---|---|---|---|---|---|
| Initially admitted | 78 | 62 (79%) | 16 | 3 (5%) | 59 (95%) | 1d |
| Not admitted at index | 48 | 29 (60%) | 19 | 7 (24%) | 22 (76%) | 2d |

- 72% of critical creatinine patients returned to KSH after their index event.
- Of initially admitted patients who returned: 95% came back outpatient only — they did not escalate to inpatient again.
- Of never-admitted patients who returned: 24% were admitted on return — delayed escalation after the index visit.
- 28% (35 patients) never returned to KSH. Destination unknown.
- No critical renal patient at KSH accessed dialysis — the programme was idle throughout this patient cohort's activity window.

**Classification: Patient Safety — not an ops dashboard metric.**
This finding must be escalated to clinical/medical leadership, not surfaced on the operations dashboard. The gap is in the clinical care pathway (admission decision, inpatient management, dialysis access) — not in operational throughput or billing.

---

## OPERATIONAL OBSERVATIONS — CD8 through CD11

These are pattern findings with direct data evidence. Not causal chains; no confirmed driver identified.

**CD8 — Ward Occupancy (1–14% across all wards):**
Using INPATIENT_BEDS × INPATIENT_WARD for bed counts and stg_inpatient_admissions for LOS:
- All wards run between 1% and 14% occupancy. No ward is capacity-constrained.
- `admission_cost` in stg_inpatient_admissions is a flat per-admission ward fee — not daily_rate × LOS. Revenue efficiency computation against daily capacity is not valid from this column.
- Occupancy is a demand problem, not a capacity problem. Beds are not the bottleneck.

**CD9 — Private Ward Revenue Potential:**
Private ward RevPAB (KES 2,438/bed-day) is 1.34× General (KES 1,824/bed-day). Private occupancy sits at 1–6% — the same structural under-use as general wards but with higher per-unit revenue. Any marginal increase in private ward utilisation produces disproportionate revenue gain relative to general ward growth.
- Reframed from "revenue efficiency invalid" to: private wards hold the highest unrealised revenue-per-bed potential in the facility.

**CD10 — Payment Mode Routing (85.5% insured → general wards):**
Direct test of whether insured patients are routed to general wards during peak by physician concentration (CD5 link):
- 85.5% of insured admissions go to general wards, 14.5% to private.
- Three-way test (insured × peak vs off-peak × ward): routing pattern is **identical** during Mon 14–18 peak and off-peak hours. Peak does not change where insured patients go.
- **Routing is structural, not peak-driven.** CD5 physician-routing link (insured patients → General Female because that's where doctors are during peak) was directly tested and disproved. The routing follows insurance coverage tier, not physician availability.

**CD11 — Demand vs Conversion (5–7% stable admission rate):**
Of all outpatient evaluations, 5–7% result in an inpatient admission — stable across all measured months. No trend toward improvement or deterioration.
- The bottleneck is upstream demand (evaluation volume), not downstream conversion (whether evaluated patients get admitted).
- Low occupancy is a demand problem, not a conversion problem. Increasing the admission rate from 6% to 7% would add ~100 admissions/year — meaningful but not structural.

---

## DISPROVED HYPOTHESES (tested, no signal found)

These are as important as what was confirmed — they show what does NOT drive the outcomes that might appear obvious.

**CD1 — Lab TAT does not drive LOS:**
Short-stay patients (<2 days) have median lab turnaround of 219 minutes — they leave before results return. Long-stay patients have 5-minute median TAT because they are still present when results arrive. The relationship is reversed. Renal and GXM delays are a clinical decision quality issue, not a capacity driver.

**CD2 — Critical lab flags do not explain LOS differences:**
Patients with critical labs (panic values: CL/CH) stay 0.8 days longer median than standard abnormal patients — but this reflects case severity, not a causal lab-to-stay relationship. Only 87 critical events across 1,167+ admitted patients (2.21%).

**CD3 — Abnormal discharge labs do not predict readmission:**
Patients with no labs in their final 48h have the highest 30-day readmission rate (3.4%), not those with abnormal labs (2.2%). Patients with active lab monitoring near discharge appear to be discharged with clearer clinical plans. Overall 30-day readmission rate: 2.9%.

**CD4 — Admission TAT does not drive LOS or DAMA:**
LOS flattens above 60 minutes of TAT — no gradient in the high-TAT range. DAMA rate shows no relationship with TAT bucket (lowest DAMA in the 120–240 min bucket at 52.9%). Private Female's operational anomaly (highest TAT at 158 min + highest LOS at 3.6d despite lowest acuity) is an observation without a confirmed causal link.

**CD7 — DAMA patients do not disproportionately leave with unresolved critical labs:**
Lab status distribution at discharge is virtually identical between Patient Request (DAMA) and Patient is Stable (clinical discharge) — both ~0.8% critical, ~19% H/L, ~80% no labs in final 24h.

---

## OPERATIONAL OBSERVATIONS (not causal, but pattern-worthy)

**Private Female anomaly:**
Private Female has the longest admission TAT (158 min), the highest median LOS (3.6d), and the lowest clinical acuity (zero critical lab patients) of all wards. Multiple angles of investigation failed to establish a causal chain, but the convergence of three outlier metrics in the same ward is a pattern that may warrant operational review.

**General Maternity DAMA baseline:**
69.3% of General Maternity patients leave on their own request — consistent with Inv 23's confirmed baseline of 70%. This is a structural characteristic of the ward, not a signal of deteriorating care.

---

## AREAS IDENTIFIED FOR FUTURE CROSS-DOMAIN INVESTIGATION

The following have not yet been investigated but represent plausible causal chains given available data:

- **Theatre completion rate × ward LOS**: On days with cancelled theatre cases, do surgical ward patients stay longer awaiting rescheduled procedures?
- **Lab volume drop × readmission rate**: Did the Oct 2025 44% lab volume drop (Inv 25) coincide with worsening readmission outcomes?
- **Imaging TAT × LOS**: Do wards with higher radiology volumes show longer stays when imaging is delayed?
- **Discharge timing clustering**: Are there identifiable discharge time patterns (e.g., morning discharge clusters) that could reduce bed turnaround time and improve occupancy?
