# Notice Type: Admission TAT / Lab Turnaround
**keywords:** tat, turnaround, admission delay, admission time, lab delay, slow admission, renal, blood bank, gxm, lab turnaround, slow, delay
**last_updated:** 2026-06-08
**decay_days:** 60
**Covers:** Inv 27 (Steps 5–18)
**Facility:** KSH only
**Sources:** HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS, HOSPITALS.KISUMU_RAW.EVENTS_RAW

---

## Admission Rate

- **Total visits (Jan 2024–Apr 2026, 65 weeks):** 34,405
- **Inpatient admissions:** 1,994 (4.9%)
- **Outpatient (no admission):** 32,716 (95.1%)
- Outpatient lab TAT is clean — only 5 tests over 2h across the entire outpatient cohort. All lab delay issues below are inpatient-specific.

---

## Admission TAT Distribution (n = 1,994)

| Bucket | Count | Share |
|--------|-------|-------|
| 0–30 min | 734 | 37% |
| 31–60 min | 211 | 11% |
| 1–2 h | 269 | 13% |
| 2–4 h | 381 | 19% |
| 4–8 h | 344 | 17% |
| 8–24 h | 55 | 3% |

**Pattern:** Bimodal. 48% fast-track (under 60 min). 36% slow pathway (2–8h). The slow pathway is not random — it concentrates at specific day/hour combinations.

---

## Slow Admission Slots (worst by case count × avg TAT)

| Slot | Cases | Avg TAT |
|------|-------|---------|
| Mon 10:00 | 22 | 304 min |
| Thu 09:00 | 17 | 295 min |

**Root cause confirmed:** Morning peak-load. Shift-change hours (14:00, 22:00) do NOT dominate the slow list. Triage and admission staff face the highest simultaneous demand in the 9–11am window on Mon and Thu.

---

## Lab TAT by Hour (all tests, inpatient)

- **Healthy range:** Median 5–18 min across most hours.
- **Outlier hours (avg TAT):**
  - H12: avg 50.5 min, max 1,759 min
  - H18: avg 96.5 min, max 2,611 min
  - H20: avg 171.2 min, max 1,277 min

These outliers are driven by specific test categories (see below), not system-wide slowdowns.

---

## Lab TAT by Test Category (inpatient)

### Renal Panel
| Hour | Avg TAT |
|------|---------|
| H7 | 216 min |
| H18 | 1,383 min |

**Pattern:** Renal results are being batched — not processed on receipt. H7 = early morning batch held from overnight. H18 = evening batch. This is an inpatient-only issue (outpatient renal TAT is clean).

### GXM / Blood Bank (Group & Cross-Match)
| Hour | Avg TAT |
|------|---------|
| H7 | 3,716 min (62 hours) |

**This is a critical outlier.** A 62-hour average for blood bank matching at H7 indicates a structural process failure — either samples are held overnight without processing, or a system/logging delay exists. This is not a workload peak; it is a blood bank workflow issue.

---

## Recommended Actions

**For slow admissions (Mon 10am, Thu 9am):**
- Add one dedicated triage support resource to Mon 10:00–11:00 and Thu 09:00–10:00 slots.
- Target: bring slow-pathway cases (currently 304 min avg on Mon 10am) below 120 min avg.

**For Renal batch TAT (H7: 216 min, H18: 1,383 min):**
- Investigate whether Renal panels are held for batch processing or processed individually.
- If batching: set a maximum batch-hold window of 60 min regardless of sample count.
- H18 is the priority — 1,383 min avg is a 23-hour delay on evening draws.

**For GXM/Blood Bank (H7: 3,716 min):**
- Escalate to lab director: 62-hour average is a patient safety risk if GXM is required for urgent transfusions.
- Determine if the delay is in processing, in result logging, or in sample transport overnight.
- A policy fix (no overnight sample holds for GXM) is likely the intervention.

**For broad lab monitoring:**
- H12, H18, and H20 are the three highest-risk lab hours for delayed results. These slots should be staffed for lab review.
