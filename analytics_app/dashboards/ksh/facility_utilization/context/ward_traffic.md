# Notice Type: Ward Traffic / Admission Volume Pressure
**keywords:** traffic, admission volume, admissions, volume, ward pressure, surge
**last_updated:** 2026-06-04
**decay_days:** 30
**Covers:** Rules 10 (Medical Female), 11 (Paediatric), 12 (Medical Male), 13 (Maternity), 14 (Private/Amenity)
**Facility:** KSH only
**Sources:** Inv 21

---

## Why This Matters
Sustained high admission volume signals capacity pressure before it becomes a staffing or bed crisis. True occupancy rate is not computable (no available beds denominator in data) — admission count is the best available proxy. Private/Amenity surged 3–4× its baseline for 2 consecutive months (Jan–Feb 2026) with zero automated detection.

---

## Baselines (KSH all-time)

| Ward | Avg Monthly Admissions | Peak | Avg Bed Days | Avg LOS |
|------|----------------------|------|-------------|---------|
| Medical — Female | 31.3 | 45 (Apr 2025) | 113 | 3.8d |
| Paediatric | 22.9 | 37 (May 2025) | 60 | 2.7d |
| Medical — Male | 19.6 | 27 | 70.5 | 3.8d |
| Maternity | 8.4 | 29 (May 2025) | 29.5 | 3.5d |
| Private / Amenity | 6.0 | 26 (Jan 2026) | 22.8 | 3.7d |

---

## Thresholds (2 consecutive months rule)

| Ward | WATCH | CRITICAL |
|------|-------|----------|
| Medical — Female | >40/month | >45/month |
| Paediatric | >32/month | >37/month |
| Medical — Male | >25/month | N/A |
| Maternity | >20/month | >25/month |
| Private / Amenity | >18/month | >22/month |

---

## Key Findings (Inv 21)

**Private/Amenity Jan–Feb 2026 surge (the missed signal):**
- Jan 2026: 26 admissions (4.3× baseline avg of 6) — revenue KES 306K vs avg KES 138K/month
- Feb 2026: 22 admissions (3.7× baseline) — revenue KES 252K
- Two consecutive months of extreme pressure — would have triggered both WATCH and CRITICAL
- Went completely undetected under previous monitoring
- Likely cause: overflow from other wards or a specific clinical event driving demand

**Paediatric Q1 2026 rising trend:**
- Mar 2026: 32 admissions — second highest on record
- Jan (20) → Feb (22) → Mar (32) → Apr (21) — volatility but upward pressure
- Not yet sustained above threshold but trending toward WATCH boundary

**Medical Female declining:**
- Volume peaked Apr 2025 (45 admissions) and has been falling
- Apr 2026 at 22 — below average; no current pressure signal

**Data quality flags:**
- Maternity Nov 2025: 12 admissions, 164 bed days (avg LOS 9.8d) — 1–2 very long-stay patients distorting monthly average; not a data error
- Medical Female Oct 2025: 17 admissions, 144 bed days (avg LOS 9.6d) — same outlier pattern

---

## Recommended Actions
- When WATCH fires on a ward: check staffing roster for the firing month — ward surges without staffing adjustment create burnout risk
- For Private/Amenity surge: investigate source of the demand spike — referrals, seasonal pattern, or overflow from a busier ward
- For any ward sustaining above WATCH: escalate to ward lead for bed management review; assess whether auxiliary space (overflow beds, step-down beds) can be activated
- Cross-reference with doctor workload data — a ward traffic spike should correlate with higher evaluation visit volumes for the same period
