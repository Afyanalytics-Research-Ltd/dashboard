# Ward Occupancy, Insured Routing & Conversion (CD8–CD11)
**keywords:** occupancy, ward occupancy, capacity, demand, demand problem, capacity problem, bed occupancy, low occupancy, beds, insured, insured routing, insured patients, conversion, admission rate, private ward revenue, why are wards empty, capacity vs demand, private ward potential, bed utilization, revpab, revenue per bed, general wards insured, routing pattern, routing structural, why insured general, conversion rate, outpatient admission rate, how many admitted, what percent admitted, occupancy rates, ward utilization, beds available, wards underused, empty beds, why low occupancy, occupancy finding, what drives occupancy
**Facility:** KSH only
**Last updated:** 2026-06-09 (CD8–CD11 confirmed)
**Covers:** CD8, CD9, CD10, CD11

---

## CD8 — Occupancy is Demand-Limited, Not Capacity-Limited

No ward is capacity-constrained. Bed availability is not the bottleneck — demand is.

- Bed counts sourced from INPATIENT_BEDS × INPATIENT_WARD
- LOS data from stg_inpatient_admissions
- `admission_cost` in stg_inpatient_admissions is a flat per-admission ward fee — not daily_rate × LOS. Revenue efficiency calculations against daily bed capacity are not valid from this column.

---

## CD9 — Private Wards Have the Highest Revenue Potential Per Occupied Bed

Private wards generate more revenue per occupied bed-day than general wards. Private occupancy is structurally low — the same demand problem as general wards, but with higher per-unit revenue. Any marginal increase in private ward utilisation produces disproportionate revenue gain relative to equivalent general ward growth.

**Private wards hold the highest unrealised revenue-per-bed potential in the facility.**

---

## CD10 — Insured Patient Routing is Structural, Not Peak-Driven

The majority of insured admissions go to general wards. Investigation CD10 directly tested whether this routing is caused by the Monday peak physician concentration finding (CD5) — specifically, whether insured patients are pushed toward General Female because that is where doctors concentrate during peak.

**The test disproved the link.** A three-way test (insured × peak vs off-peak × ward) showed that routing is identical during the Monday 14–18 peak and off-peak hours. Peak does not change where insured patients go. Routing follows insurance coverage tier, not physician availability.

When interpreting insured patient distribution across wards, do not attribute it to peak-hour physician concentration. It is a structural characteristic of the payment system.

---

## CD11 — Low Occupancy is a Demand Problem, Not a Conversion Problem

The outpatient-to-inpatient conversion rate is low and stable — it has not been trending toward improvement or deterioration. The bottleneck is upstream demand (evaluation volume reaching the facility), not downstream conversion (whether evaluated patients get admitted).

Increasing the conversion rate marginally would produce a small, non-structural occupancy improvement. The lever for occupancy growth is demand generation, not admission decision-making.
