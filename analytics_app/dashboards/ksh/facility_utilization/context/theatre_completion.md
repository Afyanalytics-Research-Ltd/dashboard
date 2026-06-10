# Notice Type: Theatre Completion Rate
**keywords:** theatre, completion, session, surgery, operating, theatre rate
**last_updated:** 2026-06-10
**decay_days:** 60
**Covers:** Rule 3
**Facility:** KSH only (TENRI theatre data = test data, excluded)
**Sources:** Inv 11

---

## Why This Matters
Each uncompleted theatre session is direct revenue loss. At KSH's billing rates, a session that is booked but not completed costs the hospital the revenue it would have collected. The Oct 2025 theatre slump coincided exactly with the insurance dispatch cliff — both operational and financial signals in the same month.

---

## Baselines

| Period | Sessions/month | Revenue | Completion rate |
|--------|---------------|---------|----------------|
| All-time avg | — | — | 90.25% |
| Peak (May 2025) | 256 | KES 27.1M | 98.4% |
| Oct 2025 (first cliff month) | — | — | 79.4% |
| Nov 2025 (worst) | 78 booked | — | 73.6% |
| Jan–Feb 2026 (brief recovery) | — | — | 94–98% |
| Mar–Apr 2026 (current) | — | — | ~76–77% |

---

## Thresholds

| Level | Threshold | Signal used |
|-------|-----------|-------------|
| WATCH | Last-month completion rate < 90% | `th_last_rate` (most recent month) |
| ALERT | Last-month completion rate < 75% | `th_last_rate` |

The 3-month weighted average is shown as context (`th_comp_rate`) but the threshold check and Operational Pulse badge use the **last month's rate** — more responsive to current performance, not smoothed by older months.

**Operational Pulse:** Theatre is the 6th domain on the Operational Pulse strip (Business Overview). Status card shows last-month rate as headline with 3-month avg in parentheses (e.g. "Apr 2026: 77% (3-mo avg 82%) — monitor surgical throughput").

---

## Key Findings (Inv 11)

**Lead with these facts in this order:**

1. Current rate: Mar–Apr 2026 = 76–77%. WATCH threshold = <85%. The notice is firing because the trailing 3-month rate is below that threshold.
2. Jan–Feb 2026 recovery to 94–98% is the critical finding — it proves this is NOT a structural problem (staffing, equipment, infrastructure). The theatre can operate at near-peak. Something reversed in Mar–Apr 2026.
3. Financial stakes: Nov 2025 alone = 78 sessions booked but not completed = KES 7.8M idle capacity in a single month.
4. Mechanism: completion drops when booked sessions are not completed — sessions are being cancelled or abandoned after booking.
5. **The specific reasons for cancellations are NOT in the dashboard data.** The gold data records sessions booked vs completed but does not capture why. Do not speculate on causes — no data exists to confirm any of them. The only next step is the operations team's theatre booking log.
6. Key question for operations team: what changed in Jan–Feb 2026 that drove recovery to 94–98%? Replicating that is the fastest path to fixing the current drop.

- One theatre type only: Major Theatre (no minor/day cases in gold data)

---

## Recommended Actions
- The Jan–Feb 2026 recovery is the key lead: ask the operations team what was different in those two months — was there a change in booking protocol, follow-up with patients, or staffing? Replicating that is the fastest path to recovery.
- Request the theatre booking log from the operations team to identify cancellation reasons: patient no-show, hospital-side (staffing, equipment, bed), or administrative.
- KES target: restore to May 2025 throughput = KES 25–27M/month. Nov 2025 alone = KES 7.8M lost capacity in one month.
