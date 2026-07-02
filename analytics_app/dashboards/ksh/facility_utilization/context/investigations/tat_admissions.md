# Notice Type: Admission TAT / Lab Turnaround
**keywords:** tat, turnaround, admission delay, admission time, lab delay, slow admission, renal, blood bank, gxm, lab turnaround, slow, delay
**last_updated:** 2026-06-08
**decay_days:** 60
**Covers:** Inv 27 (Steps 5–18)
**Facility:** KSH only
**Sources:** HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS, HOSPITALS.KISUMU_RAW.EVENTS_RAW

---

## Finding 1 — Admission TAT is Bimodal

Investigation Inv 27 found that admission TAT follows two distinct pathways: a fast-track pathway (under 60 min) and a slow pathway (2–8 hours). The slow pathway is not randomly distributed — it concentrates in specific day/hour combinations.

Morning slots (9–11am, Monday and Thursday) produce the worst admission TAT in the facility. The mechanism is peak-load: the same staff cannot simultaneously triage high-acuity inpatient admissions and process outpatient evaluations during the facility's morning surge. This is a demand bottleneck, not a capacity or staffing shortfall.

**Shift-change hours (14:00, 22:00) are not the driver.** Shift-change does not appear in the slow-admission slot list. Do not treat shift-change as the source of admission TAT delays — the investigation found no supporting evidence.

Outpatient lab TAT is clean — all lab delay issues are inpatient-specific.

---

## Finding 2 — Renal and GXM Delays Are Workflow-Specific, Not System-Wide

Lab TAT is healthy across most hours and test categories. The exceptions are test-type-specific:

**Renal panel batching:** Renal results show extreme delays at early-morning and evening hours — consistent with samples being held for batch processing rather than processed on receipt. This is an inpatient-only issue; outpatient renal TAT is clean.

**GXM / Blood Bank:** Group and cross-match samples show extreme delays at early-morning hours — a structural process failure, not a workload peak. If GXM is required for urgent transfusion, extended processing delays are a patient safety risk.

These are not system-wide slowdowns. They affect specific test categories at specific hours and require targeted workflow interventions, not facility-wide lab investment.

---

## Recommended Actions

**For elevated admission TAT:**
- Identify whether the slow pathway concentrates in morning slots (9–11am Monday and Thursday) — if yes, the intervention is a dedicated admission coordinator for those slots, not additional evaluating doctors. The goal is to decouple triage from outpatient evaluation throughput.

**For Renal panel delays:**
- Investigate whether Renal panels are being held for batch processing or processed individually on receipt.
- If batching: set a maximum batch-hold window regardless of sample count.
- Evening draws are the higher-risk batch window — escalate if turnaround on evening Renal panels is elevated.

**For GXM / Blood Bank delays:**
- Escalate to lab director when GXM turnaround is elevated — extended delays for blood bank matching are a patient safety risk for urgent transfusions.
- Determine whether the delay is in processing, result logging, or overnight sample transport. A policy fix (no overnight sample holds for GXM) is the likely intervention.
