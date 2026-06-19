# Notice Type: Dialysis Programme — Routing Gap
**keywords:** dialysis, renal, nephrology, kidney, dialysis sessions, dialysis programme, dialysis routing, creatinine dialysis
**last_updated:** 2026-06-19
**decay_days:** 90
**Covers:** Rule 4 (Equipment Idle alert) + CD12 routing gap
**Facility:** KSH only
**Sources:** Inv 4, Inv 61, Inv 62, Inv 63, Inv 64

---

## Correction Notice (Inv 61–62, 2026-06-19)

Earlier investigations (Inv 4) concluded the programme was idle with only 3 sessions ever recorded. This was wrong. `DIALYSIS_SESSIONS` is an abandoned admin table with 3 records. The programme has been running continuously since March 2025 via `FINANCE_INVOICES`. All prior "idle" framing is superseded by Inv 62–64.

---

## Programme Status

The KSH dialysis programme is **operational**. It is predominantly NHIF-funded at the confirmed tariff of KES 10,650 per session ("Dialysis Service Fee (Renal)").

| Metric | Value |
|--------|-------|
| Programme start | March 2025 |
| Sessions Mar 2025 | 5 |
| Peak sessions | 135 (December 2025) |
| Q1 2026 avg | ~112 sessions/month |
| Data end | April 21 2026 (last FINANCE_INVOICES dialysis invoice) |
| Payer mix | Predominantly NHIF — cash sessions minimal |
| Session tariff | KES 10,650 (NHIF contract rate, confirmed May 2025 onward) |
| Peak monthly revenue | ~KES 1.4M (session fees only, Dec 2025) |

---

## Capacity

| Metric | Value |
|--------|-------|
| Machines | 6 |
| Theoretical max (one shift) | 264 sessions/month (6 × 2 × 22 operating days) |
| Peak utilisation | 51.1% (Dec 2025) |
| Headroom | ~49% — programme can absorb double current volume without capital investment |

---

## The Gap — CD12 Routing

The programme has capacity. The clinical gap is the referral pathway from critical creatinine detection to dialysis enrolment.

| Metric | Value |
|--------|-------|
| Critical creatinine patients (Jul 2025+) | 126 |
| Ever billed for dialysis | 4 (3.2%) |
| Never billed for dialysis | 122 (96.8%) |

This is not an equipment gap. It is a clinical routing failure — patients with critical creatinine flags are not being referred into the programme.

---

## Alert Rule 4 — Equipment Idle

**Threshold:** WATCH when ≥ 6 consecutive months with zero dialysis sessions.

**Current KSH status:** NOT firing. Last complete session month = March 2026. Months idle = ~1. Well below the 6-month threshold.

If the programme were to go idle again, this alert would fire after 6 months. The alert body and action remain valid for TENRI.

---

## Recommended Actions

The programme is running — no launch action needed. The priority is routing:

- **Fix the CD12 pathway:** Ensure patients flagged with critical creatinine (CL/CH flags in lab results) are systematically referred for dialysis assessment. 122 of 126 such patients at KSH have never been billed for a dialysis session.
- **Track routing monthly:** The CD12 non-admission rate chart (Lab & Diagnostics tab) monitors whether the routing gap is improving.
- **No capital spend required** — 49% capacity headroom means the programme can absorb significantly more patients without procurement.
