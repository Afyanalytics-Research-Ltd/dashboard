# Notice Type: Peak Hours / Staffing Demand
**keywords:** peak, busy, staffing hours, demand, peak hour, visit volume, monday, afternoon, saturday, 4pm, doctor ratio, visit load
**last_updated:** 2026-06-08
**decay_days:** 60
**Covers:** Inv 27e, 27h, 27i, 27j
**Facility:** KSH only
**Sources:** HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS + USERS

---

## Confirmed Patterns

Investigation Inv 27 established four structural operational patterns at KSH. Current visit volumes and rankings come from the snapshot, not this file.

---

## Pattern 1 — Monday Afternoon is the Highest-Risk Peak Slot

Monday 16:00 concentrates more visits per active clinician than any other slot in the investigation period — the highest visit-to-doctor ratio in the facility. The pattern is not unique to Monday: 4pm is the dominant peak across Monday through Thursday, representing a consistent afternoon surge, not a Monday-specific anomaly.

During the investigation period, a single clinician handled the large majority of Monday 16:00 visits — a single-point-of-failure concentration. If that clinician is absent, Monday 4pm has no comparable coverage available.

---

## Pattern 2 — Saturday Morning is an Understaffed Gap

Saturday 11:00–12:00 outpatient volume reaches levels comparable to weekday mornings, with likely reduced weekend staffing. If outpatient visits are being delayed or backed up, Saturday late morning is where that backlog forms.

---

## Pattern 3 — Morning Slots Carry Admission Risk, Not Volume Risk

Monday morning and Thursday morning slots are not the highest-volume slots in the facility, but they produce the worst admission TAT. The bottleneck is structural: the same headcount cannot simultaneously triage high-acuity inpatient admissions and process outpatient evaluations. Volume is secondary — the role conflict is the bottleneck.

---

## Pattern 4 — Shift-Change Hours Are Not the Bottleneck (Disproved)

Shift-change hours (14:00, 22:00) do not appear as high-risk admission slots. Do not treat shift-change as the source of admission TAT delays — the investigation found no supporting evidence.

---

## Operational Implications

**Monday 16:00 concentration risk:**
Identify clinicians with afternoon capacity for load distribution. A single-point-of-failure concentration in this slot is a structural risk — redistribution is the intervention, not additional headcount. Current clinician workloads come from the snapshot.

**Saturday 11am–12pm staffing:**
Verify that weekend staffing is not running on reduced rosters during these two hours.

**Monday morning / Thursday morning triage:**
A dedicated admission coordinator (not an additional evaluating doctor) for these slots would decouple triage from evaluation throughput. The goal is structural separation of roles — not simply adding headcount.
