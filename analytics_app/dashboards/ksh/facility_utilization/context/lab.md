# Notice Type: Lab / Diagnostics
**keywords:** lab, abnormal, diagnostic, test, results, laboratory, radiology, imaging
**last_updated:** 2026-06-10
**decay_days:** 90
**Covers:** Rule 27 (lab volume drop), Rule 28 (abnormal rate spike)
**Facility:** KSH only (EVENTS_RAW exists only in KISUMU_RAW schema)
**Sources:** Inv 25, 25b

---

## Why This Matters
Lab volume drops signal equipment downtime, staffing failure, or patient flow reduction before any other metric catches it. The Oct 2025 dip (371 visits vs 550–700 baseline — a 44% drop across all categories) was completely undetected and went unresolved for that month. Rising abnormal rates signal increasing patient acuity across the facility — a leading indicator before ward readmission rates rise.

---

## Baselines (KSH, Sep 2024–Apr 2026, 19 months)

| Period | Components/month | Distinct Visits | Abnormal % |
|--------|-----------------|-----------------|------------|
| Sep–Dec 2024 | 12,005–13,383 | 450–551 | 5.98–6.82% |
| Jan–Jun 2025 | 15,006–18,894 | 563–692 | 6.55–7.75% |
| Jul–Sep 2025 | 15,863–16,940 | 609–700 | 5.76–6.44% |
| **Oct 2025 (dip)** | **8,993** | **371** | 6.43% |
| Nov–Dec 2025 | 10,821–13,735 | 473–554 | 6.18–6.22% |
| Jan–Apr 2026 | 15,406–19,262 | 609–735 | 5.85–7.18% |

- Volume monitoring uses **distinct visits** (not component count) — more patient-meaningful
- FBC dominates volume: 50–55% of all components every month
- Abnormal rate: tight band 5.76%–7.75% across all 19 months — very predictable baseline

---

## Thresholds

**Lab volume (distinct visits per month):**

| Level | Threshold | Rule |
|-------|-----------|------|
| WATCH | < 430 distinct visits/month | 2 consecutive months |
| CRITICAL | < 350 distinct visits/month | Single month (acute signal) |

**Abnormal rate (% of results flagged H or L):**

| Level | Threshold | Rule |
|-------|-----------|------|
| WATCH | Abnormal % > 9.0% | 2 consecutive months |
| CRITICAL | Abnormal % > 11.0% | 2 consecutive months |

**Not monitored as threshold alert — clinical safety only:**
- Critical creatinine non-admission rate: monitored live on Lab & Diagnostics page ("Critical Creatinine — Admission Outcome" section) and in email notifier (Clinical Safety Monitor section). NOT a threshold alert — patient safety finding requiring clinical lead review.
- Result entry TAT: rising avg `updated_at - created_at` = lab backlog proxy. Feasible but not yet in notices.

**Critical creatinine monitoring (live from 2026-06-10):**
- Source: `EVENTS_RAW` flags `CONTAINS(flag, '(CL)') OR CONTAINS(flag, '(CH)')` for test = 'Creatinine'
- Data available Jul 2025+ (HTML-encoded CL/CH format introduced mid-2025)
- Jul 2025–Feb 2026: non-admission rate 18.8%–57.1% per month (overall cohort baseline 41%)
- See `renal_pathway.md` for full patient pathway data and monthly breakdown

---

## Key Findings (Inv 25, 25b)

**Oct 2025 volume dip (confirmed facility-wide):**
- FBC: −42%, Stool Microscopy: −44%, Urinalysis: −48%, Malaria: −57%
- All categories dropped simultaneously — not panel-specific equipment failure
- Two hypotheses: (1) genuine volume reduction (fewer patients seeking care), (2) partial pipeline failure in data ingestion
- 371 visits — would have triggered CRITICAL on single-month rule
- Volume recovered Nov 2025 onwards — Oct was a one-month event

**Malaria permanent step-down post-Oct 2025:**
- Was 42–86/month Jun–Sep 2025
- Dropped to 18–30/month from Oct 2025 — did not recover
- Possible: seasonal pattern, or change in testing protocol (rapid test vs microscopy?)
- Not a volume problem — a category change

**Abnormal rate stability:**
- 5.76%–7.75% across 19 months — no month has ever exceeded 7.75%
- WATCH (>9%) and CRITICAL (>11%) thresholds are conservative by design: no false positives in history
- If abnormal rate fires, it is a genuine clinical signal

**Data source technical notes:**
- Table: `HOSPITALS.KISUMU_RAW.EVENTS_RAW`
- All fields via JSON payload syntax: `payload:created_at::STRING`, `payload:flag::STRING`, `payload:visit_id::STRING`, `payload:test::STRING`
- `payload:title` is 99.6% NULL — group by `payload:test` patterns instead
- Abnormal flag parsing: `H` = High, `L` = Low, `H<span style="color:red">(CH)</span>` = Critically High (HTML-encoded), `L<span style="color:red">(CL)</span>` = Critically Low

---

## Recommended Actions

**If lab volume WATCH fires (< 430 visits for 2 months):**
- First check: is this a data pipeline issue or a genuine volume drop? Pull the raw component count — if component count also dropped proportionally, it is real.
- If real: check whether the volume drop correlates with reduced ward admissions (lower patient flow) or with a lab equipment issue
- Contact lab manager: ask whether any equipment was down, whether staffing was reduced, or whether any test types were sent to external labs

**If lab volume CRITICAL fires (< 350 in a single month):**
- Escalate immediately — this is a facility-level signal
- Oct 2025 precedent: 371 visits = 44% drop across all test categories simultaneously = likely equipment or pipeline failure, not demand
- Action within 48 hours: confirm whether lab was operational for the full month

**If abnormal rate WATCH fires (> 9% for 2 months):**
- Rising abnormal rate = patients arriving sicker than baseline OR lab analyzer drift (false positive)
- Cross-reference with ward admissions volume and readmission rate for same period
- If admissions are up AND abnormal rate is up: acuity increase — escalate to clinical lead
- If admissions are flat AND abnormal rate is up: possible analyzer calibration issue — lab QC review
