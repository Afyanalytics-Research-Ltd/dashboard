# Renal Patient Pathway — KSH (CD12)
**keywords:** renal, creatinine, critical creatinine, renal patient, never admitted, not admitted, renal pathway, critical kidney, renal outcome, nephrology, critical renal, patient outcomes, never returned, renal patients ksh, what happens renal, creatinine not admitted, kidney patient, kidney outcome, dialysis idle, renal care gap, critical lab admitted, renal follow up, renal escalation, admitted creatinine, 41 percent, dialysis programme
**Facility:** KSH only
**Last updated:** 2026-06-10 (monthly monitoring live)
**Covers:** Rule CD12 — patient safety finding + live monthly monitor

---

## Finding

134 distinct critical Creatinine visit_ids (CL/CH flags) since 2024-01-01, representing 126 unique patients.

**41% of critical Creatinine visits result in no inpatient admission.**

Of those not admitted at index: the majority returned to KSH, and a meaningful subset of those returning were admitted on return — delayed escalation after the index visit. A significant minority never returned to KSH — destination unknown. See Return Visit Tracking table below for the full breakdown.

---

## Placement of Critical Creatinine Patients (admitted cohort — 78 patients)

| Ward | Patients |
|------|---------|
| Outpatient only (no admission) | 56 (41%) |
| Pediatric General | 21 |
| General Female | 20 |
| General Maternity | 15 |
| General Male | 13 |
| Private Male | 5 |
| Private Maternity | 4 |
| Private Female | 1 |

---

## Discharge Outcomes (admitted patients)

| Discharge Mode | Patients | Avg LOS | Readmission Rate |
|----------------|----------|---------|-----------------|
| Patient Request (DAMA) | 44 | 3.6d | 4.5% |
| Patient is Stable | 28 | 4.8d | 0% |
| Referral | 2 | 18.0d | — |
| Death | 1 | 1.0d | — |

DAMA rate (57%) equals the facility baseline of 56.82% (CD7) — critical renal patients do not leave against medical advice at a higher rate than any other patient group.

Referral patients wait an average of 18 days before transfer — the longest LOS of any group.

---

## Dialysis Access

Current dialysis programme status and the referral routing gap are documented in `dialysis_idle.md`. For the patient pathway: the clinical question is whether critical creatinine patients are being systematically referred into the programme. Investigation CD12 found the referral rate was critically low.

---

## Return Visit Tracking (all 126 patients)

| Index Disposition | Returned | Never Returned | Admitted on Return |
|---|---|---|---|
| Initially admitted (78) | 62 (79%) | 16 (21%) | 3 of 62 (5%) |
| Not admitted at index (48) | 29 (60%) | 19 (40%) | 7 of 29 (24%) |

- 72% of all critical creatinine patients returned to KSH after their index event.
- Of initially-admitted patients who returned: 95% returned outpatient only — they did not escalate to inpatient again.

---

## Classification

**Patient Safety — not an ops dashboard metric.**
This finding must be escalated to clinical/medical leadership. The gap is in the clinical care pathway (admission decision, inpatient management, dialysis access) — not in operational throughput or billing.

---

## Live Monthly Monitoring

Monthly non-admission rate is tracked live in the snapshot (`cd12_non_admission_rate`). Current value always comes from the snapshot — do not cite historical figures from this file as current.

Monthly volumes are small — a single patient can shift the monthly rate by a large margin. Use the 3-month weighted average for trend assessment, not single-month figures.
