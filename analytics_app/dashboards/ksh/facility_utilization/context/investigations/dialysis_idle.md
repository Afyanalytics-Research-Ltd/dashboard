# Notice Type: Dialysis Programme — Routing Gap
**keywords:** dialysis, renal, nephrology, kidney, dialysis sessions, dialysis programme, dialysis routing, creatinine dialysis
**last_updated:** 2026-06-19
**decay_days:** 90
**Covers:** Rule 4 (Equipment Idle alert) + CD12 routing gap
**Facility:** KSH only
**Sources:** Inv 4, Inv 61, Inv 62, Inv 63, Inv 64

---

## Programme Status

The KSH dialysis programme was confirmed **operational** via `FINANCE_INVOICES` (Inv 61–64) — the earlier `DIALYSIS_SESSIONS` table was abandoned and does not reflect actual programme activity. The programme is predominantly NHIF-funded at a confirmed tariff of KES 10,650 per session ("Dialysis Service Fee (Renal)"). Current session volume is in the snapshot.

| Metric | Value |
|--------|-------|
| Programme start | March 2025 |
| Payer mix | Predominantly NHIF — cash sessions minimal |
| Session tariff | KES 10,650 (NHIF contract rate) |

---

## Capacity

| Metric | Value |
|--------|-------|
| Machines | 6 |
| Theoretical max (one shift) | 264 sessions/month (6 × 2 × 22 operating days) |

Significant headroom exists — the programme can absorb substantially more volume without capital investment. Current utilisation is in the snapshot.

---

## The Gap — CD12 Routing

The programme has capacity. The clinical gap is the referral pathway from critical creatinine detection to dialysis enrolment.

Investigation CD12 found that the large majority of patients flagged with critical creatinine had no dialysis billing history — indicating a referral routing failure, not an equipment gap.

This is not an equipment problem. It is a clinical routing failure — patients with critical creatinine flags are not being systematically referred into the programme.

---

## Alert Rule 4 — Equipment Idle

**Threshold:** WATCH when ≥ 6 consecutive months with zero dialysis sessions.

Current KSH programme status is tracked in the snapshot (dialysis sessions per month). Alert fires when ≥ 6 consecutive months with zero sessions — current status is always live from the snapshot, not this file.

If the programme were to go idle, this alert would fire after 6 months. The alert body and action remain valid for TENRI.

---

## Recommended Actions

If the programme is active (verify in snapshot), no launch action is needed. The priority is routing:

- **Fix the CD12 pathway:** Ensure patients flagged with critical creatinine (CL/CH flags in lab results) are systematically referred for dialysis assessment. Investigation CD12 found the referral rate was critically low.
- **Track routing monthly:** The CD12 non-admission rate chart (Lab & Diagnostics tab) monitors whether the routing gap is improving.
- **No capital spend required** — the programme has substantial spare capacity and can absorb significantly more patients without procurement.
