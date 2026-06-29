# Operational Demand Outlook — KSH
**keywords:** demand, forecast, projection, expected, next month, admissions next month, visits next month, expected admissions, expected visits, demand forecast, demand projection, operational planning, admission outlook, visit outlook, ward forecast, ward admissions next month, how many patients, planned admissions, volume forecast, ema projection
**Facility:** KSH only
**Last updated:** 2026-06-10 (EMA projections live)
**Covers:** 1-month forward demand projections — facility visits + ward admissions

---

## What This Is

Two self-updating 1-month forward projections displayed on the dashboard:

1. **Facility Visit Outlook** (Business Overview, 4th KPI card) — projected total visits next month
2. **Ward Admission Outlook** (Beds & Wards tab) — projected admissions per ward next month

Both use **Exponential Moving Average (EMA, span=3)** — adapts to structural breaks (e.g. staffing changes) within 1–2 months. More responsive than linear regression.

---

## Methodology

- **EMA span=3**: weights the last 3 months, with recent months weighted higher
- **Partial month exclusion**: KSH data ends 2026-04-21 (day 21 < threshold of day 25) → the final partial month is excluded before fitting the EMA, so a low partial month does not depress the projection
- **Minimum data**: requires ≥3 usable months. If fewer exist, projection is suppressed.

---

## Current Projections (based on data through Mar 2026)

Verify against dashboard on next run — these update automatically with new data.

**Facility visits (Expected Next Month):** ~2,323 (displayed on Business Overview)
- Note: this figure may reflect genuine recent upward trend. Verify against `q_visit_summary()` monthly output — expected range ~1,500–2,000/month based on Sep 2024–Apr 2026 total of 34,405 visits / ~20 months.

**Ward admissions (Operational Demand Outlook — Next Month):**
| Ward | Projected Admissions |
|------|---------------------|
| Medical — Male | EMA(3) of recent monthly actuals |
| Medical — Female | EMA(3) of recent monthly actuals |
| Maternity | EMA(3) of recent monthly actuals |
| Private / Amenity | EMA(3) of recent monthly actuals |
| Paediatric | EMA(3) of recent monthly actuals |

---

## Important Caveats

- **Ward projections are physician-dependent**: E.Awando handles 34–46% of admissions in every ward (CD6). Any staffing change will shift actual admissions from the projection within 1–2 months.
- **Monday 09:00–12:00 and 16:00 remain peak demand windows** — independent of monthly volume trend. Operational planning should account for these regardless of projected volume.
- **Projection horizon is 1 month only** — not a seasonal or long-term forecast. Seasonal modelling requires 24+ months of data (available from Sep 2026 at earliest).
- **Not a staffing prescription** — projections inform planning, not hiring or redistribution decisions.

---

## Scope Boundaries (not built — reasons documented)

| Scope | Decision |
|-------|----------|
| 7-day forward projection | Daily grain is too noisy at KSH visit volumes — weekly swings exceed signal |
| Lab volume projection | Follows visit volume with no independent signal — adds no information |
| Occupancy bed-days | LOS variance too wide to project reliably at ward level |
| Revenue projection | Sep 2025 dispatch cliff breaks the trend — projection would extrapolate a structural break |
| Staffing demand | No workforce capacity data available (Inv 26: USERS has no designation/role fields) |
