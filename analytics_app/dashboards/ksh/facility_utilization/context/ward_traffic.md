# Notice Type: Ward Traffic / Admission Volume Pressure
**keywords:** traffic, admission volume, admissions, volume, ward pressure, surge
**last_updated:** 2026-06-04
**decay_days:** 30
**Covers:** Rules 10 (Medical Female), 11 (Paediatric), 12 (Medical Male), 13 (Maternity), 14 (Private/Amenity)
**Facility:** KSH only
**Sources:** Inv 21

---

## Why This Matters

Sustained high admission volume signals demand pressure before it becomes a staffing or clinical strain. At KSH, occupancy across all wards is well below physical capacity (confirmed via CD8 — beds are not the bottleneck). A volume surge signals increased demand, not a physical capacity constraint. Admission count is the primary demand signal available in the data — it moves before staffing or clinical quality metrics do.

A ward can surge 3–4× above its baseline for consecutive months without triggering any alert under manual monitoring. This notice was built to catch that pattern automatically.

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

**Private/Amenity surge — the rule's origin:**
A sustained surge at 3–4× baseline for two consecutive months went completely undetected under previous monitoring. This is the primary evidence that automated volume monitoring is necessary. Current Private/Amenity position comes from the snapshot.

**Paediatric:**
Admission count is volatile month to month. The WATCH threshold is reachable — the investigation period showed the trend approaching the boundary. Current position from the snapshot.

**Medical Female:**
Volume is volatile across seasons. Current position from the snapshot — do not treat any prior period as the current state.

**Demand vs capacity:**
Ward traffic surges are demand signals, not capacity failures. When a ward fires WATCH or CRITICAL, the response is clinical coordination and staffing review — not bed activation or overflow planning (CD8 confirmed beds are not the constraint at KSH).

---

## Recommended Actions
- When WATCH fires on a ward: check staffing roster for the firing month — a demand surge without staffing adjustment creates clinical workload pressure
- For Private/Amenity surge: investigate the source of the demand spike — referral pattern change, seasonal effect, or a specific clinical driver. Do not assume overflow from another ward without confirming it in the data.
- For any ward sustaining above WATCH: escalate to ward lead for clinical coordination review; check whether evaluation visit volume also increased in the same period (snapshot — doctor workload data)
- Cross-reference with readmission data: a sustained volume surge that precedes a readmission spike may indicate the ward was absorbing more acuity than staffing could support
