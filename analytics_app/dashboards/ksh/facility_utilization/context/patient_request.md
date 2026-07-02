# Notice Type: Patient Request Discharge Rate
**keywords:** patient request, self discharge, ama, against medical advice, discharge rate
**last_updated:** 2026-06-04
**decay_days:** 30
**Covers:** Rules 20 (Maternity), 21 (Paediatric), 22 (Medical Female), 23 (Private/Amenity), 24 (Medical Male)
**Facility:** KSH only
**Sources:** Inv 17, 23

---

## Why This Matters

"Patient Request" = patient leaves before medical clearance (against medical advice or before doctor-approved discharge). Patient Request discharges carry significantly elevated readmission risk. During the investigation period, the readmission rate among Medical Male Patient Request discharges escalated dramatically across successive months and was accelerating — not plateauing. This is the single strongest confirmed causal link between discharge type and near-term readmissions at KSH. The notice fires BEFORE the readmission rate rises — it is the early warning system for ward readmission rules.

Patient Request is the most common discharge type at KSH. The notice fires when a ward exceeds its own ward-specific baseline — not a general threshold — because baseline rates differ dramatically across wards.

---

## Thresholds (2 consecutive months above threshold, minimum 10 admissions gate)

| Ward | Baseline | WATCH | CRITICAL | Notes |
|------|----------|-------|----------|-------|
| Maternity | 70% | >82% | N/A | Structurally high; CRITICAL not meaningful |
| Paediatric | 54% | >68% | >78% | Stable baseline |
| Medical — Female | 54% | >68% | >78% | Historical spike to 76.5% would have triggered WATCH |
| Private / Amenity | 53% | >75% | N/A | Too volatile for CRITICAL |
| Medical — Male | 47% | >62% | >72% | Historical spike to 72.2% would have triggered CRITICAL |

---

## Key Findings (Inv 17, 23)

**Connection to readmissions (Inv 17):**
Patient Request rate is a genuine leading indicator of readmissions, not a coincidental correlation. The investigation demonstrated that as Patient Request rate rose month-over-month in Medical Male, readmission rate followed — escalating rather than stabilising. There is no natural floor; once the discharge pathway is broken, the readmission rate can continue to worsen.

**Why patients leave early:**
- Root cause hypothesis for Medical Male: financial pressure (insurer not covering full stay → patient leaves to avoid accumulating bills) OR inadequate discharge counseling
- 60+ insured males are the highest-risk cohort: shorter stays than cash patients of the same age, substantially higher readmission rates — consistent with insurer authorization pressure forcing discharge before clinical readiness
- The mechanism is insurer authorization, not patient preference — these patients are not choosing to leave; they are responding to financial pressure

**Notable patterns (Inv 23):**
- Medical Female: spikes above WATCH threshold have preceded readmission spikes in subsequent months — leading indicator pattern confirmed
- Medical Male: reached CRITICAL-level rates in the investigation period — thresholds are calibrated to observed extremes, not theoretical projections
- Private/Amenity: highly volatile month-to-month — the minimum 10-admissions volume gate is critical to prevent false fires in low-volume months
- Maternity: structurally elevated and rarely drops close to its baseline — any sustained move above the WATCH threshold (>82%) is significant

---

## Recommended Actions
- When Patient Request WATCH fires: pull all Patient Request discharges for that ward in the firing month and flag for social work callback within 7 days
- Discharge counseling protocol: mandatory documentation of patient's reason for leaving early + follow-up appointment scheduled at discharge
- For 60+ insured Medical Male patients: challenge insurer authorization for early discharge; document medical necessity for continued stay
- For Maternity: any Patient Request above 82% should trigger a review of whether post-delivery patients are leaving within 24h of delivery
- Cross-reference with readmission data for the same ward 30 days later — Patient Request rate is a leading indicator, readmissions are lagging
