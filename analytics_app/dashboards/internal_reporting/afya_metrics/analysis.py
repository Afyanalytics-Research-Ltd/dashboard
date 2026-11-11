"""Week-over-week change and pace against target.

Figures are running totals, so status compares the recent weekly run rate with the
weekly gain still needed to reach the target by quarter end (needs two weeks of data).
"""

from dataclasses import dataclass

from .data import period_bounds

RUN_RATE_WINDOW = 4

# (key, label, value column, target column, format)
TARGETS = [
    ("upds",       "Unique patient datasets", "UPDS_PROCESSED",        "UPDS_TARGET",        "num"),
    ("revenue",    "Revenue booked",          "REVENUE_BOOKED",        "REVENUE_TARGET",     "usd"),
    ("mou",        "County MoUs",             "MOU_SIGNED",            "MOU_TOTAL_COUNTIES", "num"),
    ("facilities", "Facilities on HMIS",      "FACILITIES_PRIVATE",    "FACILITIES_TARGET",  "num"),
    ("associates", "Associates",              "APPRENTICES_ONBOARDED", "APPRENTICES_TARGET", "num"),
]
STATUS_LABEL = {"met": "Target met", "on": "On track", "risk": "At risk", "off": "Off track"}


def fmt(v, kind="num", compact=False):
    v = float(v or 0)
    pre = "$" if kind == "usd" else ""
    if compact:
        for div, suf, dp in ((1e6, "M", 2), (1e3, "K", 1)):
            if abs(v) >= div:
                return pre + f"{v / div:.{dp}f}".rstrip("0").rstrip(".") + suf
    return pre + f"{v:,.0f}"


def pct(num, den):
    return num / den * 100 if den else 0.0


@dataclass
class Clock:
    start: object
    end: object
    week: object

    @property
    def weeks_left(self):
        return max((self.end - self.week).days, 0) / 7 if self.end else 0

    @property
    def days_left(self):
        return max((self.end - self.week).days, 0) if self.end else 0

    @property
    def elapsed_pct(self):
        if not self.start:
            return None
        total = (self.end - self.start).days + 1
        return min(max(((self.week - self.start).days + 1) / total * 100, 0), 100)

    @property
    def week_count(self):
        return round(((self.end - self.start).days + 1) / 7) if self.start else None

    @property
    def week_no(self):
        return min((self.week - self.start).days // 7 + 1, self.week_count) if self.start else None


@dataclass
class Track:
    key: str
    label: str
    kind: str
    value: int
    target: int
    delta: int | None
    status: str | None

    @property
    def pct(self):
        return pct(self.value, self.target)


def _rate(hist):
    pts = hist[-(RUN_RATE_WINDOW + 1):]
    if len(pts) < 2:
        return None
    days = (pts[-1][0] - pts[0][0]).days
    return (pts[-1][1] - pts[0][1]) / days * 7 if days > 0 else None


def _status(value, target, rate, need):
    if target and value >= target:
        return "met"
    if rate is None or not target:
        return None
    if need is None:
        return "off"
    return "on" if rate >= need * 0.95 else ("risk" if rate >= need * 0.7 else "off")


def clock_for(period, week):
    return Clock(*period_bounds(period), week)


def tracks(series, clock):
    out = []
    for key, label, vcol, tcol, kind in TARGETS:
        hist = [(r["WEEK_ENDING"], r[vcol]) for r in series]
        value, target = series[-1][vcol], series[-1][tcol]
        delta = value - series[-2][vcol] if len(series) > 1 else None
        need = max(target - value, 0) / clock.weeks_left if clock.weeks_left > 0 else None
        out.append(Track(key, label, kind, value, target, delta, _status(value, target, _rate(hist), need)))
    return out
