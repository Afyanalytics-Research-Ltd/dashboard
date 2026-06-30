# Notice Type: Peak Hours / Staffing Demand
**keywords:** peak, busy, staffing hours, demand, peak hour, visit volume, monday, afternoon, saturday, 4pm, doctor ratio, visit load
**last_updated:** 2026-06-08
**decay_days:** 60
**Covers:** Inv 27e, 27h, 27i, 27j
**Facility:** KSH only
**Sources:** HOSPITALS.KISUMU_CLEAN.EVALUATION_VISITS + USERS

---

## Top 15 Peak Slots (Jan 2024–Apr 2026, 65 weeks)

| Rank | Day | Hour | Visits | Doctors Active | Visits/Doctor |
|------|-----|------|--------|----------------|---------------|
| 1 | Mon | 16:00 | 786 | 15 | **52.4** |
| 2 | Tue | 16:00 | 764 | 22 | 34.7 |
| 3 | Thu | 16:00 | 743 | 18 | 41.3 |
| 4 | Wed | 11:00 | 694 | 17 | 40.8 |
| 5 | Wed | 16:00 | 682 | 16 | 42.6 |
| 6 | Mon | 11:00 | 681 | 20 | 34.1 |
| 7 | Tue | 11:00 | 652 | 20 | 32.6 |
| 8 | Mon | 10:00 | 640 | 14 | 45.7 |
| 9 | Thu | 12:00 | 640 | 22 | 29.1 |
| 10 | Thu | 10:00 | 638 | 17 | 37.5 |
| 11 | Thu | 11:00 | 624 | 19 | 32.8 |
| 12 | Wed | 10:00 | 617 | 17 | 36.3 |
| 13 | Tue | 10:00 | 615 | 18 | 34.2 |
| 14 | Thu | 09:00 | 599 | 18 | 33.3 |
| 15 | Wed | 09:00 | 594 | 19 | 31.3 |

**Note:** DOCTORS_ACTIVE is cumulative across the full 65-week period — it is the number of distinct doctors who ever worked that slot, not concurrent per-shift headcount. Use ratios for relative comparison, not as absolute per-hour staffing counts.

---

## Key Findings

**Mon 16:00 is the highest-risk slot:**
- 786 visits across 15 doctors = 52.4 visits/doctor ratio — highest of any slot
- **lowino handles 587/786 (75%) of all Mon 16:00 visits** — single point of failure. If lowino is absent, Mon 4pm collapses.
- Next largest: eawando (101 visits), jogutu (45), makinyi (25, departed Dec 2025)

**4pm is the dominant peak across Mon–Thu:**
- Mon, Tue, Thu 16:00 all rank in the top 3 by raw visit count
- Wed 16:00 ranks 5th
- This is a consistent afternoon surge pattern, not a Monday anomaly

**Saturday gap (outpatient visits only):**
- Sat 11:00 (579 visits) and Sat 12:00 (525 visits) appear in the top 15 of outpatient peak slots
- Saturday is NOT in the overall top 15 but has significant volume with likely reduced weekend staffing
- Outpatient volume on Sat 11–12am matches weekday morning levels

**Morning slots — admission risk, not volume risk:**
- Mon 10:00 has only 640 visits (rank 8) but produces the worst admission TAT (304 min avg, 22 cases)
- Thu 09:00 (rank 14): 17 slow admissions, 295 min avg TAT
- Morning volume is secondary to morning triage bottleneck — the same headcount cannot simultaneously triage high-acuity admissions and process outpatient evaluations

---

## Mon 16:00 Doctor Breakdown

| Doctor | Visits (Mon 16:00) |
|--------|-------------------|
| lowino | 587 (75%) |
| eawando | 101 (13%) |
| jogutu | 45 (6%) |
| makinyi | 25 (3%, departed) |
| bawuor | 7 |
| praburu | 5 |
| danyango | 4 |
| NODEDE | 4 |
| + 7 others | 1 each |

---

## Recommended Actions

**Mon 16:00 load redistribution:**
- lowino's 75% share of Mon 4pm is an operational single point of failure. Identify 2–3 doctors with afternoon capacity and redirect 100–150 Mon 16:00 evaluations/month to them.
- eawando (101 Mon 16:00 visits) already contributes — check if they have more capacity in that slot before adding load.

**Saturday 11am–12pm staffing:**
- Saturday volume rivals weekday mornings. Verify that weekend staffing is not operating on reduced rosters during these 2 hours. If outpatient visits are being delayed, Sat 11–12 is where the backlog forms.

**Mon 10:00 / Thu 09:00 admission triage:**
- These are the two worst admission bottleneck slots. A dedicated admission coordinator (not an additional evaluating doctor) for Mon 10–11am and Thu 09–10am would decouple triage from evaluation throughput.
- Target: reduce Mon 10am avg admission TAT from 304 min to under 120 min.

**Shift-change monitoring (NOT a current issue):**
- Shift-change hours (14:00, 22:00) do NOT appear in the slow-admission list. Do not treat shift-change as the source of admission delays — the data does not support it.
