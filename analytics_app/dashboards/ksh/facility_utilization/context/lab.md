# Notice Type: Lab / Diagnostics
**keywords:** lab, abnormal, diagnostic, test, results, laboratory, radiology, imaging
**last_updated:** 2026-06-10
**decay_days:** 90
**Covers:** Rule 27 (lab volume drop), Rule 28 (abnormal rate spike)
**Facility:** KSH only
**Sources:** Inv 25, 25b

---

## Why This Matters

Lab volume drops signal equipment downtime, staffing failure, or patient flow reduction before any other metric catches it. A facility-wide volume collapse — all test categories dropping simultaneously — can go completely undetected for a full month without automated monitoring. Rising abnormal rates signal increasing patient acuity across the facility — a leading indicator before ward readmission rates rise.

---

## Thresholds

**Lab volume (distinct visits per month):**

| Level | Threshold | Rule |
|-------|-----------|------|
| WATCH | < 430 distinct visits/month | 2 consecutive months |
| CRITICAL | < 350 distinct visits/month | Single month (acute signal) |

Volume monitoring uses **distinct visits** (not component count) — more patient-meaningful. FBC dominates test volume and is the first category to check when investigating a drop.

**Abnormal rate (% of results flagged H or L):**

| Level | Threshold | Rule |
|-------|-----------|------|
| WATCH | Abnormal % > 9.0% | 2 consecutive months |
| CRITICAL | Abnormal % > 11.0% | 2 consecutive months |

**Not monitored as threshold alert — clinical safety only:**
- Critical creatinine non-admission rate: monitored live on Lab & Diagnostics page and in the Clinical Safety Monitor section. NOT a threshold alert — patient safety finding requiring clinical lead review. See `renal_pathway.md` for full pathway context and current rates from the snapshot.
- Result entry TAT: rising avg `updated_at - created_at` = lab backlog proxy. Feasible but not yet in notices.

---

## Key Findings (Inv 25, 25b)

**Facility-wide volume collapse pattern:**
- When a genuine volume drop occurs, all test categories fall simultaneously — not a single panel. Proportional drops across FBC, Stool Microscopy, Urinalysis, and Malaria together confirm a facility-level event rather than equipment failure in one area.
- Two hypotheses when a drop is confirmed real: (1) genuine patient flow reduction, (2) partial pipeline failure in data ingestion. First diagnostic step: confirm whether raw component count also dropped proportionally.
- A previous facility-wide collapse reached single-month CRITICAL threshold levels and recovered the following month — one-month events are possible.

**Malaria category shift:**
- A sustained decline in malaria testing volume was observed during the investigation period and had not recovered by investigation end. Possible causes: seasonal pattern, or change in testing protocol (rapid test vs microscopy). This is a category-level shift, not a total volume problem — it requires protocol investigation, not a volume alert response.

**Abnormal rate stability:**
- Historically stable across the investigation period — a tight, predictable band. WATCH (>9%) and CRITICAL (>11%) thresholds are deliberately conservative with no false positives in historical data. If the abnormal rate notice fires, treat it as a genuine clinical signal.

---

## Recommended Actions

**If lab volume WATCH fires (< 430 visits for 2 months):**
- First check: is this a data pipeline issue or a genuine volume drop? Pull the raw component count — if component count also dropped proportionally, it is real.
- If real: check whether the volume drop correlates with reduced ward admissions (lower patient flow) or with a lab equipment issue.
- Contact lab manager: ask whether any equipment was down, whether staffing was reduced, or whether any test types were sent to external labs.

**If lab volume CRITICAL fires (< 350 in a single month):**
- Escalate immediately — this is a facility-level signal.
- When all test categories drop simultaneously, the pattern is consistent with equipment or pipeline failure rather than demand reduction. Confirm whether the lab was operational for the full month within 48 hours.

**If abnormal rate WATCH fires (> 9% for 2 months):**
- Rising abnormal rate = patients arriving sicker than baseline OR lab analyzer drift (false positive).
- Cross-reference with ward admissions volume and readmission rate for the same period.
- If admissions are up AND abnormal rate is up: acuity increase — escalate to clinical lead.
- If admissions are flat AND abnormal rate is up: possible analyzer calibration issue — lab QC review.
