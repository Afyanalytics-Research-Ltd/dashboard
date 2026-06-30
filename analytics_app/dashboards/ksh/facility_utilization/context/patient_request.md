# Notice Type: Patient Request Discharge Rate
**keywords:** patient request, self discharge, ama, against medical advice, discharge rate
**last_updated:** 2026-06-04
**decay_days:** 30
**Covers:** Rules 20 (Maternity), 21 (Paediatric), 22 (Medical Female), 23 (Private/Amenity), 24 (Medical Male)
**Facility:** KSH only
**Sources:** Inv 17, 23

---

## Why This Matters
"Patient Request" = patient leaves before medical clearance (against medical advice or before doctor-approved discharge). At KSH, patients discharged on Patient Request readmit at **50% within 30 days** (Inv 17, Apr 2026 Medical Male). This is the single strongest leading indicator of readmissions. The notice fires BEFORE the readmission rate rises — it is the early warning system for ward readmission rules.

---

## Baselines (KSH, all-time)

| Ward | Total Admissions | Patient Request Count | Patient Request % |
|------|-----------------|----------------------|------------------|
| Maternity | 287 | 202 | **70.38%** — structural |
| Paediatric | 457 | 249 | 54.49% |
| Medical — Female | 626 | 338 | 53.99% |
| Private / Amenity | 235 | 125 | 53.19% |
| Medical — Male | 391 | 183 | 46.80% |

Patient Request is the **most common discharge type facility-wide** (~55% of all discharges). The notice fires when a ward exceeds ITS OWN elevated baseline — not a general threshold.

---

## Thresholds (2 consecutive months above threshold, minimum 10 admissions gate)

| Ward | Baseline | WATCH | CRITICAL | Notes |
|------|----------|-------|----------|-------|
| Maternity | 70% | >82% | N/A | Structurally high; CRITICAL not meaningful |
| Paediatric | 54% | >68% | >78% | Stable baseline |
| Medical — Female | 54% | >68% | >78% | Jan 2026 spike (76.5%) would have triggered WATCH |
| Private / Amenity | 53% | >75% | N/A | Too volatile for CRITICAL |
| Medical — Male | 47% | >62% | >72% | Feb 2026 spike (72.2%) would have triggered CRITICAL |

---

## Key Findings (Inv 17, 23)

**Connection to readmissions (Inv 17):**
- Medical Male Patient Request patients: readmit at 50% within 30 days (Apr 2026)
- Nov 2025: 11 Patient Request discharges → 1 readmission (9.1%)
- Dec 2025: 8 Patient Request discharges → 1 readmission (12.5%)
- Jan 2026: 10 Patient Request discharges → 2 readmissions (20.0%)
- Apr 2026: 4 Patient Request discharges → 2 readmissions (**50.0%**)
- The rate is accelerating — 50% is not a ceiling

**Why patients leave early:**
- Root cause hypothesis for Medical Male: financial pressure (insurer not covering full stay → patient leaves to avoid accumulating bills) OR inadequate discharge counseling
- 60+ insured males most at risk: shorter LOS (3.9d vs 5.1d for cash) + 3× higher readmission rate (18.8% vs 6.3%)
- Insurer authorization pressure forcing early discharge of elderly patients who are not clinically ready

**Notable monthly escalations (Inv 23):**
- Medical Female Jan 2026: 76.5% vs 54% baseline — preceded Mar 2026 readmission spike
- Medical Male Feb 2026: 72.2% vs 47% baseline (above CRITICAL) — then dropped to 21.1% Apr 2026
- Private/Amenity: highly volatile (14.3%–76.9% range), wide monthly swings — needs volume gate to prevent false fires
- Maternity: consistently high, rarely below 55%; spikes to 85.7% (Apr 2025)

**Facility-wide context:**
- Patient Request is not a KSH-specific anomaly — TENRI Patient Request rate is 8.11% (all-time), similar structural pattern
- The column is `discharge_type`, NOT `discharge_reason` — data is in `stg_inpatient_admissions`

---

## Recommended Actions
- When Patient Request WATCH fires: pull all Patient Request discharges for that ward in the firing month and flag for social work callback within 7 days
- Discharge counseling protocol: mandatory documentation of patient's reason for leaving early + follow-up appointment scheduled at discharge
- For 60+ insured Medical Male patients: challenge insurer authorization for early discharge; document medical necessity for continued stay
- For Maternity: any Patient Request above 82% should trigger a review of whether post-delivery patients are leaving within 24h of delivery
- Cross-reference with readmission data for the same ward 30 days later — Patient Request rate is a leading indicator, readmissions are lagging
