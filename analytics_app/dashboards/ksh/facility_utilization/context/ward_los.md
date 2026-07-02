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

**Why median not average:** Long-stay outliers occur at every ward and can make monthly averages highly misleading. A single extreme-LOS admission can push the monthly average above a CRITICAL threshold while the median remains stable. Median eliminates this class of false positive entirely.

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

## Recommended Actions
- When LOS WATCH fires: identify whether rising median is driven by a cluster of long-stay patients (discharge barrier) or a broad shift across all patients (acuity)
- Pull the LOS distribution for the firing month: if P90 is rising but P50 is stable → outlier load. If P50 is rising → systemic.
- Discharge barrier causes: social circumstances (patient has nowhere to go), insurance authorization delays, awaiting specialist review, medication not available
- Acuity causes: check if readmission rate is rising in the same month (sicker patients, longer recovery, more readmissions)
- For Maternity: never act on LOS alone without checking whether an extreme-LOS outlier drove the result — always use median
- For Medical Male: LOS increase + rising readmission rate together = strong signal of acuity increase or discharge quality failure
