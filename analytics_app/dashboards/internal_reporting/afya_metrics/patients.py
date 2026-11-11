"""Patient counts tab: totals and a per-source table."""

from . import theme as T
from .analysis import fmt, pct


def _share_cell(r):
    """Unique %; flags 100% (no duplicates usually means de-duplication wasn't applied)."""
    share = pct(r["UNIQUE_PATIENTS"], r["PATIENTS"])
    if r["PATIENTS"] > 100 and share >= 100:
        return (f'<td class="x-num x-flag" title="No duplicate records — check whether '
                f'de-duplication was applied">{share:.0f}%</td>')
    return f'<td class="x-num x-dim">{share:.{1 if share >= 99.5 else 0}f}%</td>'


def render(rows, updated):
    if not rows:
        T.html('<div class="x-empty"><b>No patient counts recorded yet</b>Use Update to add them.</div>')
        return

    total = sum(r["PATIENTS"] for r in rows)
    uniq = sum(r["UNIQUE_PATIENTS"] for r in rows)
    upd = f'<span>Updated {updated:%d %b %Y}</span>' if updated else ""
    T.html(f'<div class="x-ctx"><div class="x-strip"><span><b>{len(rows)}</b> data sources</span>'
           f'<div style="flex:1"></div>{upd}</div></div>')

    T.html('<div class="x-grid">'
           + T.tile("Patient records", fmt(total), meta="across all sources")
           + T.tile("Unique patients", fmt(uniq), meta="unique within each source", primary=True)
           + T.tile("Unique share", f"{pct(uniq, total):.1f}%", meta="of patient records")
           + '</div>')

    top = max(r["PATIENTS"] for r in rows) or 1
    body = "".join(
        f'<tr><td>{r["SOURCE"]}</td>'
        f'<td class="x-num">{fmt(r["PATIENTS"])}</td>'
        f'<td class="x-num">{fmt(r["UNIQUE_PATIENTS"])}</td>'
        f'{_share_cell(r)}'
        f'<td class="x-barcell"><div class="x-hbar">'
        f'<i style="width:{pct(r["PATIENTS"], top):.1f}%;background:{T.SOFT}"></i>'
        f'<i style="width:{pct(r["UNIQUE_PATIENTS"], top):.1f}%;background:{T.NAVY}"></i></div></td></tr>'
        for r in rows)
    T.html(f'<table class="x-table"><thead><tr><th>Source</th><th class="x-num">Patients</th>'
           f'<th class="x-num">Unique</th><th class="x-num">Unique %</th><th class="x-barcell">'
           f'<span class="x-key"><i style="background:{T.NAVY}"></i>Unique</span>'
           f'<span class="x-key"><i style="background:{T.SOFT}"></i>All records</span></th></tr></thead>'
           f'<tbody>{body}</tbody><tfoot><tr><td>Total</td><td class="x-num">{fmt(total)}</td>'
           f'<td class="x-num">{fmt(uniq)}</td><td class="x-num">{pct(uniq, total):.0f}%</td>'
           f'<td></td></tr></tfoot></table>')
