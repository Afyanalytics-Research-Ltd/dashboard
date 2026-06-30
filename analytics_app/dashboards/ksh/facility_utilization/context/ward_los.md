# Notice Type: Ward LOS Deviation
**keywords:** los, length of stay, stay, discharge time, length, median los
**last_updated:** 2026-06-04
**decay_days:** 30
**Covers:** Rules 15 (Medical Male), 16 (Medical Female), 17 (Private/Amenity), 18 (Maternity), 19 (Paediatric)
**Facility:** KSH only
**Sources:** Inv 22

---

## Why This Matters
Rising length of stay signals discharge inefficiency (patients staying longer than clinically necessary) or acuity increase (patients arriving sicker). Both have operational consequences: occupied beds block new admissions, increase nursing load, and reduce effective ward capacity. LOS increase often precedes readmission rate increases — earlier detection window.

---

## Baselines (KSH, median-based — extreme outliers excluded from avg)

| Ward | Admissions | Avg LOS | Median LOS | Max recorded |
|------|-----------|---------|------------|--------------|
| Medical — Male | 374 | 3.78d | 3.0d | 73d |
| Medical — Female | 614 | 3.67d | 3.0d | 82d |
| Private / Amenity | 229 | 3.83d | 3.0d | 93d |
| Maternity | 280 | 3.60d | 2.0d | 139d |
| Paediatric | 447 | 2.70d | 2.0d | 47d |

**Why median not average:** Maternity max 139d, Private/Amenity max 93d, Medical Female max 82d. A single outlier in Maternity Nov 2025 would have fired CRITICAL on an avg-based rule (avg 13.67d) while median stayed at 2.0d — one 139-day patient. Median eliminates this class of false positive entirely.

---

## Thresholds (median monthly LOS, 2 consecutive months rule)

| Ward | Median baseline | WATCH | CRITICAL |
|------|----------------|-------|----------|
| Medical — Male | 3.0d | >5.0d | >7.0d |
| Medical — Female | 3.0d | >5.0d | >7.0d |
| Private / Amenity | 3.0d | >5.5d | >8.0d |
| Maternity | 2.0d | >4.0d | >6.0d |
| Paediatric | 2.0d | >3.5d | >5.0d |

---

## Key Findings (Inv 22)

**% of patients over 7 days (structural long-stay load):**
- Medical Male: 12.0% — most skewed ward
- Medical Female: 10.1%
- Private/Amenity: 8.3%
- Paediatric: 5.6%
- Maternity: 5.0%

**Notable monthly spikes:**
- Medical Female Oct 2025: avg 9.60d, median 4.0d, max 82d — median elevated; genuine acuity spike. Not a false positive.
- Maternity Nov 2025: avg 13.67d, median 2.0d — single 139d outlier. Median correctly suppresses false positive.
- Private/Amenity Nov 2025: avg 7.27d, median 3.0d — outlier-driven; no clinical concern.
- Paediatric Jan–Feb 2026: median crept to 4.0d and 2.5d vs usual 1.0–2.5d — mild upward trend, worth watching.

---

## Recommended Actions
- When LOS WATCH fires: identify whether rising median is driven by a cluster of long-stay patients (discharge barrier) or a broad shift across all patients (acuity)
- Pull the LOS distribution for the firing month: if P90 is rising but P50 is stable → outlier load. If P50 is rising → systemic.
- Discharge barrier causes: social circumstances (patient has nowhere to go), insurance authorization delays, awaiting specialist review, medication not available
- Acuity causes: check if readmission rate is rising in the same month (sicker patients, longer recovery, more readmissions)
- For Maternity: never act on LOS alone without checking whether the extreme outlier drove the average — always use median
- For Medical Male: LOS increase + rising readmission rate together = strong signal of acuity increase or discharge quality failure
