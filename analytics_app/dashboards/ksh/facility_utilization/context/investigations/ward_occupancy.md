# Ward Occupancy, Insured Routing & Conversion (CD8–CD11)
**keywords:** occupancy, ward occupancy, capacity, demand, demand problem, capacity problem, bed occupancy, low occupancy, beds, insured, insured routing, insured patients, conversion, admission rate, private ward revenue, why are wards empty, capacity vs demand, private ward potential, bed utilization, revpab, revenue per bed, general wards insured, routing pattern, routing structural, why insured general, conversion rate, outpatient admission rate, how many admitted, what percent admitted, occupancy rates, ward utilization, beds available, wards underused, empty beds, why low occupancy, occupancy finding, what drives occupancy
**Facility:** KSH only
**Last updated:** 2026-06-09 (CD8–CD11 confirmed)
**Covers:** CD8, CD9, CD10, CD11

---

## CD8 — Ward Occupancy: 1–14% Across All Wards

All wards run between 1% and 14% bed occupancy. No ward is capacity-constrained.

- Bed counts: from INPATIENT_BEDS × INPATIENT_WARD
- LOS data: from stg_inpatient_admissions
- `admission_cost` is a flat per-admission ward fee — not daily_rate × LOS

**Occupancy is a demand problem, not a capacity problem. Beds are not the bottleneck.**

---

## CD9 — Private Ward Revenue Potential

Private ward RevPAB (KES 2,438/bed-day) is 1.34× General ward RevPAB (KES 1,824/bed-day).

Private occupancy sits at 1–6% — the same structural under-use as general wards but with higher per-unit revenue. Any marginal increase in private ward utilisation produces disproportionate revenue gain relative to general ward growth.

**Private wards hold the highest unrealised revenue-per-bed potential in the facility.**

---

## CD10 — Payment Mode Routing: 85.5% of Insured Admissions Go to General Wards

| Payment type | General ward share | Private ward share |
|---|---|---|
| Insured | 85.5% | 14.5% |

**Direct test of CD5 link (insured patients routed to General Female because doctors concentrate there during peak):**

Three-way test (insured × peak vs off-peak × ward): routing pattern is **identical** during Monday 14–18 peak and off-peak hours. Peak does not change where insured patients go.

**Routing is structural, not peak-driven.** The CD5 physician-routing link (insured patients follow doctors to General Female during peak) was directly tested and disproved. The routing follows insurance coverage tier, not physician availability.

---

## CD11 — Demand vs Conversion: 5–7% Stable Admission Rate

Of all outpatient evaluations, 5–7% result in an inpatient admission — stable across all measured months. No trend toward improvement or deterioration.

- The bottleneck is upstream demand (evaluation volume), not downstream conversion (whether evaluated patients get admitted).
- Low occupancy is a demand problem, not a conversion problem.
- Increasing the admission rate from 6% to 7% would add ~100 admissions/year — meaningful but not structural.
