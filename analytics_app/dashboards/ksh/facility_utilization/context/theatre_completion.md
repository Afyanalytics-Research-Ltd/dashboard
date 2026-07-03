# Notice Type: Theatre Completion Rate
**keywords:** theatre, completion, session, surgery, operating, theatre rate, surgical, surgeries, operations, operation, procedure, procedures, operating room, theatre utilisation, theatre performance, elective, how many surgeries, how many operations, completed sessions
**last_updated:** 2026-06-10
**decay_days:** 60
**Covers:** Rule 3
**Facility:** KSH only (TENRI theatre data = test data, excluded)
**Sources:** Inv 11

---

## Why This Matters

Each uncompleted theatre session is direct revenue loss. At KSH's billing rates, a session that is booked but not completed costs the hospital the revenue it would have collected. Theatre completion drops have coincided with broader operational disruptions at KSH — the signal often appears alongside other financial and operational anomalies in the same period.

---

## Thresholds

| Level | Threshold | Signal used |
|-------|-----------|-------------|
| WATCH | Last-month completion rate < 90% | `th_last_rate` (most recent month) |
| ALERT | Last-month completion rate < 75% | `th_last_rate` |

The 3-month weighted average is shown as context (`th_comp_rate`) but the threshold check uses the **last month's rate** — more responsive to current performance, not smoothed by older months.

---

## Key Findings (Inv 11)

**Lead with these facts in this order:**

1. Theatre performance has demonstrated it can recover to near-peak levels — this proves poor completion is NOT necessarily a structural problem (staffing, equipment, infrastructure). The rate can fall and recover. Do not treat a sustained drop as permanent until the booking log has been reviewed.
2. Financial stakes: a single month of poor completion represents substantial idle capacity at KSH's billing rates. Every uncompleted booked session is direct lost revenue — the financial cost compounds quickly across multiple underperforming months.
3. Mechanism: completion drops when booked sessions are not completed — sessions are being cancelled or abandoned after booking.
4. **The specific reasons for cancellations are NOT in the dashboard data.** The gold data records sessions booked vs completed but does not capture why. Do not speculate on causes — no data exists to confirm any of them. The only next step is the operations team's theatre booking log.
5. When performance improves, identify what operational changes coincided with the improvement — booking protocol, patient follow-up, or staffing. Replicating those changes is the fastest path to sustained recovery.

- One theatre type only: Major Theatre (no minor/day cases in gold data)

---

## Recommended Actions
- When the notice fires: ask the operations team what has changed in booking protocol, patient follow-up, or staffing. If a prior recovery period exists, ask what was different then.
- Request the theatre booking log from the operations team to identify cancellation reasons: patient no-show, hospital-side (staffing, equipment, bed), or administrative.
