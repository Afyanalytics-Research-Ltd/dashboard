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
- **Partial month exclusion**: if the most recent month has fewer than 25 days of data, it is excluded before fitting the EMA — so a low partial month does not depress the projection
- **Minimum data**: requires ≥3 usable months. If fewer exist, projection is suppressed.

---

## Projections

Projections update automatically from the dashboard and are not stored here. Current projected values are always live from the dashboard or snapshot — do not cite figures from this file as current.

Historical visit volumes have been sufficiently stable for short-term EMA forecasting, while remaining responsive to recent operational changes. The EMA(3) adapts to structural breaks within 1–2 months.

---

## Important Caveats

- **Ward projections are physician-dependent**: a single physician contributes a large share of admissions across all wards (CD6). Any staffing change will shift actual admissions from the projection within 1–2 months.
- **Historically, demand has concentrated on Monday morning and late afternoon** — independent of monthly volume trend. These patterns should be confirmed against current operational metrics before planning decisions.
- **Projection horizon is 1 month only** — not a seasonal or long-term forecast. Seasonal modelling requires at least 24 months of representative data before deployment.
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
