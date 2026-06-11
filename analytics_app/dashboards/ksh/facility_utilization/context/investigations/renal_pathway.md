# Renal Patient Pathway — KSH (CD12)
**keywords:** renal, creatinine, critical creatinine, renal patient, never admitted, not admitted, renal pathway, critical kidney, renal outcome, nephrology, critical renal, patient outcomes, never returned, renal patients ksh, what happens renal, creatinine not admitted, kidney patient, kidney outcome, dialysis idle, renal care gap, critical lab admitted, renal follow up, renal escalation, admitted creatinine, 41 percent, 28 percent, dialysis programme
**Facility:** KSH only
**Last updated:** 2026-06-10 (monthly monitoring live)
**Covers:** Rule CD12 — patient safety finding + live monthly monitor

---

## Finding

134 distinct critical Creatinine visit_ids (CL/CH flags) since 2024-01-01, representing 126 unique patients.

**41% of critical Creatinine visits result in no inpatient admission.**

Of those never admitted at index:
- 60% returned to KSH
- 24% of returning never-admitted patients were admitted on return (delayed escalation)
- 19 patients (28% of not-admitted cohort) never returned — destination unknown

**28% of initially-admitted critical creatinine patients also never returned** — 16 patients with no subsequent encounter recorded.

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

## Dialysis Programme

No critical renal patient at KSH accessed dialysis. The programme has been idle since May 2025 (13+ months):
- March 2025: 2 sessions · 2 patients · KES 52,200
- April 2025: 1 session · 1 patient · KES 119,100
- May 2025 onward: **zero sessions**

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

## Live Monthly Monitoring (from 2026-06-10)

Monthly non-admission rate is now tracked live on the **Lab & Diagnostics** page ("Critical Creatinine — Admission Outcome" section). Data available from Jul 2025 when the CL/CH critical flag format was introduced in the lab system.

**Technical note on flag format:** Critical creatinine is flagged in EVENTS_RAW as HTML-encoded strings — `L<span style="color:red">(CL)</span>` (critically low) and `H<span style="color:red">(CH)</span>` (critically high). Plain `H`/`L` = general abnormal only, not critical.

**Monthly data (Jul 2025–Feb 2026):**

| Month | Total Critical | Not Admitted | Rate |
|-------|---------------|--------------|------|
| Jul 2025 | 13 | 7 | 53.8% |
| Aug 2025 | 28 | 10 | 35.7% |
| Sep 2025 | 17 | 9 | 52.9% |
| Oct 2025 | 16 | 3 | 18.8% |
| Nov 2025 | 25 | 11 | 44.0% |
| Dec 2025 | 14 | 8 | 57.1% |
| Jan 2026 | 20 | 7 | 35.0% |
| Feb 2026 | 4 | 1 | 25.0% |

Monthly volumes are small (4–28/month). One patient = up to 25% rate swing. Use 3-month weighted average for trend assessment, not single-month figures.

**Email notifier:** A "Clinical Safety Monitor" section is sent with each Executive Digest showing the trailing 3-month non-admission rate + latest month count. No WATCH/ALERT badge — routes directly to Clinical/Medical Lead for review.
