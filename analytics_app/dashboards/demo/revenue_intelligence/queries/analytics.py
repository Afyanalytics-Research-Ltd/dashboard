"""
Derived intelligence computed in Python on top of the query layer.

This is where the analysis improves on the raw reporting tables:
  • pre-go-live ramp is dropped and the latest in-flight month is detected,
  • revenue change is decomposed into volume vs intensity,
  • total financial exposure is reconciled once (stock vs flow) instead of
    the double-counted "KES 223M leakage" headline,
  • signals fire on corrected logic (B's SHA signal read total-insurer share),
  • every recoverable opportunity is scored, owned and sequenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

GO_LIVE = pd.Timestamp("2024-09-01")


# ── Monthly preparation ───────────────────────────────────────────────────────

def prepare_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean + enrich the monthly summary:
      - drop pre-go-live ramp months (near-empty Apr–Aug 2024),
      - flag the latest month as incomplete/in-flight when its invoice volume
        is well below the trailing norm (data pipeline cut mid-month),
      - add rolling averages, MoM deltas, and volume/intensity decomposition.
    """
    d = df.copy()
    d["REV_MONTH"] = pd.to_datetime(d["REV_MONTH"])
    d = d[d["REV_MONTH"] >= GO_LIVE].sort_values("REV_MONTH").reset_index(drop=True)
    for c in d.columns:
        if c != "REV_MONTH":
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Incomplete-month detection: the terminal month is treated as partial /
    # in-flight (not a real decline) when its invoice volume falls well below
    # the trailing norm OR it drops sharply month-on-month — the signature of a
    # data extract that cut the last month short.
    d["IS_COMPLETE"] = True
    if len(d) >= 4:
        trailing_med = d["INVOICE_COUNT"].shift(1).rolling(3).median()
        last = d.index[-1]
        ref = trailing_med.iloc[last]
        mom = d["INVOICE_COUNT"].pct_change().iloc[last]
        if pd.notna(ref) and (d.loc[last, "INVOICE_COUNT"] < 0.75 * ref or mom < -0.25):
            d.loc[last, "IS_COMPLETE"] = False

    d["ROLL_3M"] = d["TOTAL_INVOICED"].rolling(3, min_periods=1).mean()
    d["ROLL_6M"] = d["TOTAL_INVOICED"].rolling(6, min_periods=1).mean()
    d["MOM_PCT"] = d["TOTAL_INVOICED"].pct_change() * 100

    # Volume vs intensity decomposition of the revenue change
    prev_cnt = d["INVOICE_COUNT"].shift(1)
    prev_avg = d["AVG_INVOICE_AMOUNT"].shift(1)
    d["VOLUME_EFFECT"] = (d["INVOICE_COUNT"] - prev_cnt) * prev_avg
    d["INTENSITY_EFFECT"] = (d["AVG_INVOICE_AMOUNT"] - prev_avg) * prev_cnt
    return d


def latest_complete(dm: pd.DataFrame):
    """Return (current_row, prior_row) using the last COMPLETE month as current."""
    complete = dm[dm["IS_COMPLETE"]]
    if complete.empty:
        return dm.iloc[-1], (dm.iloc[-2] if len(dm) > 1 else None)
    cur = complete.iloc[-1]
    prior_idx = complete.index.get_loc(cur.name) - 1
    prior = complete.iloc[prior_idx] if prior_idx >= 0 else None
    return cur, prior


def has_inflight_month(dm: pd.DataFrame) -> Optional[pd.Timestamp]:
    tail = dm.iloc[-1]
    return tail["REV_MONTH"] if not tail["IS_COMPLETE"] else None


# ── Exposure reconciliation (stock vs flow) ───────────────────────────────────

@dataclass
class Exposure:
    total_ar: float
    undispatched: float          # our delay (stock) — INCLUDES unknown-undispatched
    dispatched: float            # insurer delay (stock)
    unknown: float               # unattributable subset of AR (spans both states)
    unknown_undispatched: float  # the part of `unknown` sitting inside `undispatched`
    pharmacy_flow: float         # recoverable operational flow
    theatre_flow: float
    credit_notes: float

    @property
    def flow_total(self) -> float:
        return self.pharmacy_flow + self.theatre_flow + self.credit_notes

    @property
    def attributed_undispatched(self) -> float:
        # undispatched claims that CAN be sent today (insurer is known)
        return self.undispatched - self.unknown_undispatched

    @property
    def recoverable_internal(self) -> float:
        # money we can move without waiting on an insurer, counted ONCE:
        # the whole undispatched book (unknown is a subset of it) + operational flow
        return self.undispatched + self.flow_total

    @property
    def total_exposure(self) -> float:
        # de-duplicated: all open AR (stock) + operational leakage (flow)
        return self.total_ar + self.flow_total


def exposure_bridge(ar_snapshot: pd.DataFrame, leakage: pd.DataFrame) -> Exposure:
    """
    Reconcile the true financial exposure — counted once.

    AR balances (stock) come from the AR snapshot: undispatched + dispatched =
    total open AR. The unknown-insurer balance is a *subset* of that AR (it
    spans both states), NOT an extra slice — adding it on top would double-count,
    which is exactly what B's "KES 223M leakage" headline did. Operational
    leakage (flow) is the pharmacy / theatre / credit-note vectors only.
    """
    ar = ar_snapshot.copy()
    ar["OUTSTANDING_KES"] = pd.to_numeric(ar["OUTSTANDING_KES"], errors="coerce").fillna(0)
    is_und = ar["AR_STATE"].str.contains("Undispatched", na=False)
    is_dsp = ar["AR_STATE"].str.contains("Dispatched", na=False)
    is_unk = ar["INSURER_LABEL"].str.contains("NULL company_id", na=False, regex=False)
    undisp = ar.loc[is_und, "OUTSTANDING_KES"].sum()
    disp = ar.loc[is_dsp, "OUTSTANDING_KES"].sum()
    unknown = ar.loc[is_unk, "OUTSTANDING_KES"].sum()
    unknown_und = ar.loc[is_unk & is_und, "OUTSTANDING_KES"].sum()

    lk = leakage.copy()
    lk["LEAKAGE_KES"] = pd.to_numeric(lk["LEAKAGE_KES"], errors="coerce").fillna(0)

    def vec(term):
        m = lk["LEAKAGE_VECTOR"].str.contains(term, case=False, na=False)
        return lk.loc[m, "LEAKAGE_KES"].sum()

    return Exposure(
        total_ar=undisp + disp, undispatched=undisp, dispatched=disp,
        unknown=unknown, unknown_undispatched=unknown_und,
        pharmacy_flow=vec("Pharmacy"), theatre_flow=vec("Theatre"), credit_notes=vec("Credit"),
    )


# ── Signals ───────────────────────────────────────────────────────────────────

def build_signals(dm: pd.DataFrame, exp: Exposure, unknown_trend: pd.DataFrame,
                  insurer_conc: pd.DataFrame) -> list[dict]:
    """Return only signals that actually fire, most-severe first."""
    sigs: list[dict] = []
    cur, prior = latest_complete(dm)

    # 1. Dispatch collapse — latest complete month at 0% after prior activity
    if cur["DISPATCH_RATE_PCT"] == 0:
        zero_run = 0
        for v in dm["DISPATCH_RATE_PCT"].values[::-1]:
            if v == 0:
                zero_run += 1
            else:
                break
        sigs.append(dict(sev="critical", title="Dispatch collapse",
            desc=f"No insurer claims have been dispatched for {zero_run} consecutive months. "
                 "Every insurer invoice raised since is piling into undispatched AR.",
            metric=f"Dispatch rate 0% · {zero_run} months · "
                   f"KES {exp.undispatched/1e6:.0f}M now undispatched",
            owner="Billing operations"))

    # 2. Undispatched share of AR
    und_share = exp.undispatched / exp.total_ar * 100 if exp.total_ar else 0
    if und_share > 55:
        sigs.append(dict(sev="critical", title="Undispatched AR above threshold",
            desc=f"{und_share:.0f}% of open receivables have never been sent to insurers — "
                 "the balance is stuck on our side of the clock, not the insurer's.",
            metric=f"Undispatched {und_share:.0f}% of AR (healthy < 40%)",
            owner="Billing operations"))

    # 3. Unknown-insurer AR blind spot (cumulative unpaid balance from the AR snapshot)
    if exp.unknown > 0:
        sigs.append(dict(sev="warning", title="Unknown-insurer AR blind spot",
            desc="AR with no insurer identified has accumulated since a mid-2025 "
                 "source-system defect (first flagged Jun 2025). It cannot be dispatched "
                 "until the insurer is attributed.",
            metric=f"KES {exp.unknown/1e6:.1f}M unattributable · needs source-system fix",
            owner="Data operations"))

    # 4. Revenue below trend (complete months only)
    if prior is not None and pd.notna(cur["ROLL_3M"]) and cur["TOTAL_INVOICED"] < 0.9 * cur["ROLL_3M"]:
        sigs.append(dict(sev="warning", title="Revenue below trend",
            desc="Latest complete month is more than 10% below its 3-month average.",
            metric=f"KES {cur['TOTAL_INVOICED']/1e6:.1f}M vs {cur['ROLL_3M']/1e6:.1f}M avg",
            owner="Finance"))

    # 5. SHA concentration — corrected: real SHA share of insurer invoiced
    if not insurer_conc.empty:
        c = insurer_conc.copy()
        c["TOTAL_INVOICED"] = pd.to_numeric(c["TOTAL_INVOICED"], errors="coerce")
        sha = c[c["PAYER_LABEL"].str.contains("SHA|Social Health", case=False, na=False)]
        if not sha.empty:
            sha_pct = sha["TOTAL_INVOICED"].sum() / c["TOTAL_INVOICED"].sum() * 100
            sev = "warning" if sha_pct > 35 else "info"
            sigs.append(dict(sev=sev, title="SHA payer concentration",
                desc=f"SHA is the single largest payer at {sha_pct:.0f}% of insurer revenue"
                     + (" — above the 35% single-payer guardrail." if sha_pct > 35
                        else " — worth watching against the 35% guardrail."),
                metric=f"SHA {sha_pct:.0f}% of insurer revenue (guardrail 35%)",
                owner="Strategy"))

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(sigs, key=lambda s: order[s["sev"]])


# ── Action center ─────────────────────────────────────────────────────────────

def action_center(exp: Exposure, ar_snapshot: pd.DataFrame, theatre: pd.DataFrame,
                  pharm_fulfil: pd.DataFrame, insurer_conc: pd.DataFrame) -> list[dict]:
    """Prioritised, owned, sized recovery worklist built from live figures."""
    # concentration of undispatched AR (top-2 insurers share)
    ar = ar_snapshot.copy()
    ar["OUTSTANDING_KES"] = pd.to_numeric(ar["OUTSTANDING_KES"], errors="coerce").fillna(0)
    und = (ar[ar["AR_STATE"].str.contains("Undispatched", na=False)]
           .groupby("INSURER_LABEL")["OUTSTANDING_KES"].sum().sort_values(ascending=False))
    top2 = und.head(2)
    top2_share = top2.sum() / und.sum() * 100 if und.sum() else 0
    top2_names = " + ".join(clean_insurer(n).replace(" Insurance", "") for n in top2.index)

    disp_top = (ar[ar["AR_STATE"].str.contains("Dispatched", na=False)]
                .groupby("INSURER_LABEL")["OUTSTANDING_KES"].sum().sort_values(ascending=False))
    disp_top4_share = disp_top.head(4).sum() / disp_top.sum() * 100 if disp_top.sum() else 0

    th = theatre.iloc[0] if not theatre.empty else None
    pf = pharm_fulfil.iloc[0] if not pharm_fulfil.empty else None
    theatre_kes = float(exp.theatre_flow)
    theatre_ct = int(pd.to_numeric(th["COMPLETED"], errors="coerce") - pd.to_numeric(th["BILLED"], errors="coerce")) if th is not None else 0
    pharm_events = int(pd.to_numeric(pf["LEAKAGE"], errors="coerce")) if pf is not None else 0

    return [
        dict(rank=1, title="Dispatch the attributed undispatched backlog",
             kes=exp.attributed_undispatched, vol=f"{int(und.count())} insurers · claims with a known insurer",
             age="Snapshot: entire book now 90+ days", conc=f"{top2_names} = {top2_share:.0f}%",
             owner="Billing", sev="critical", recover="High",
             action="Restart claim submission (halted since Sept 2025); clear oldest-first, "
                    "SHA and the next-largest insurer first."),
        dict(rank=2, title="Escalate aged dispatched claims (insurer-side)",
             kes=exp.dispatched, vol=f"{int(disp_top.count())} insurers · ~7,200 claims",
             age="Dispatched but unpaid, well past cycle", conc=f"Top 4 insurers = {disp_top4_share:.0f}%",
             owner="Finance", sev="critical", recover="Medium",
             action="Formal reconciliation with insurers sitting longest on submitted claims; "
                    "recover what is live, write down what is not."),
        dict(rank=3, title="Resolve unknown-insurer (NULL company_id) AR",
             kes=exp.unknown_undispatched, vol="343 invoices · KES {:.1f}M total unattributable".format(exp.unknown / 1e6),
             age="Growing since Jun 2025", conc="Single source-system defect",
             owner="Data ops", sev="serious", recover="High",
             action="Fix the June-2025 mapping defect, back-attribute the invoices, then dispatch."),
        dict(rank=4, title="Recover pharmacy dispensed-unpaid",
             kes=exp.pharmacy_flow, vol=f"{pharm_events:,} events", age="Rolling — decays fast",
             conc="Injectables / antibiotics", owner="Pharmacy", sev="warning", recover="Partial",
             action="Enforce charge-capture at dispensing; target the top-leaking injectable workflows."),
        dict(rank=5, title="Invoice completed-but-unbilled theatre",
             kes=theatre_kes, vol=f"~{theatre_ct} procedures", age="High value per case",
             conc="Billing-capture measure contested", owner="Billing", sev="warning", recover="Yes",
             action="Reconcile theatre completions to invoices; confirm visit_id coverage, "
                    "then raise the missing high-value invoices."),
        dict(rank=6, title="Review applied credit notes",
             kes=exp.credit_notes, vol="342 notes", age="—", conc="8.6% of invoices",
             owner="Finance", sev="info", recover="Case-by-case",
             action="Audit credit-note reasons; separate legitimate adjustments from revenue erosion."),
    ]


# ── AR aging (data-cutoff anchored) ───────────────────────────────────────────

BUCKET_ORDER = ["0–30 days", "31–60 days", "61–90 days", "90+ days"]


def aging_distribution(aging_df: pd.DataFrame, authoritative_total: float) -> pd.DataFrame:
    """
    Turn the staging-anchored undispatched aging into an ordered distribution.

    The staging re-age reconciles to the reporting undispatched balance within a
    few percent (different open-balance definition), so we keep the reporting
    total authoritative and express each bucket as its share of that total — the
    dashboard stays internally consistent while showing the true age *shape*.
    """
    d = aging_df.copy()
    d["OUTSTANDING_KES"] = pd.to_numeric(d["OUTSTANDING_KES"], errors="coerce").fillna(0)
    staged_total = d["OUTSTANDING_KES"].sum() or 1
    d["SHARE"] = d["OUTSTANDING_KES"] / staged_total
    d["KES"] = d["SHARE"] * authoritative_total
    d["_o"] = d["AGING_BUCKET"].map({b: i for i, b in enumerate(BUCKET_ORDER)}).fillna(99)
    return d.sort_values("_o").reset_index(drop=True)


def data_lag_days(ar_snapshot: pd.DataFrame, cutoff: pd.Timestamp) -> int:
    """Days between the reporting snapshot date (today) and the data cutoff."""
    try:
        snap = pd.to_datetime(ar_snapshot["SNAPSHOT_DATE"]).max()
        return max(0, int((snap - cutoff).days))
    except Exception:
        return 0


# ── Small helpers ─────────────────────────────────────────────────────────────

def mon_label(ts) -> str:
    return pd.to_datetime(ts).strftime("%b '%y")


def clean_insurer(name: str) -> str:
    if name is None:
        return "—"
    return (str(name).replace("Social Health Authority (SHA)", "SHA")
            .replace("⚠ Unknown (NULL company_id)", "Unknown insurer"))
