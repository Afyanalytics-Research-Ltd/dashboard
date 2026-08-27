"""
Excel loading and deterministic profiling.

The profile is computed in Python, not by the model. The LLM never guesses
dtypes or null counts - it is handed the facts and reasons about them. This
is what keeps a "data analyst agent" from hallucinating its way through a
schema it never actually inspected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_PREVIEW_ROWS = 8
MAX_CATEGORY_EXAMPLES = 8
#: Columns with this fraction of unique values or fewer are treated as categorical.
CATEGORICAL_UNIQUE_RATIO = 0.5
CATEGORICAL_MAX_UNIQUE = 50


def slugify_sheet(name: str) -> str:
    """Turn a sheet name into a valid Python identifier for the namespace.

    'Q1 Sales (2026)' -> 'q1_sales_2026'
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    if not slug:
        slug = "sheet"
    if slug[0].isdigit():
        slug = f"s_{slug}"
    return slug


@dataclass(slots=True)
class SheetProfile:
    name: str
    var_name: str
    n_rows: int
    n_cols: int
    columns: list[dict[str, Any]]
    preview: str
    duplicate_rows: int

    def to_markdown(self) -> str:
        lines = [
            f"### Sheet `{self.name}`  →  variable `{self.var_name}`",
            f"{self.n_rows:,} rows × {self.n_cols} columns"
            + (f" · {self.duplicate_rows:,} fully duplicated rows" if self.duplicate_rows else ""),
            "",
            "| column | dtype | nulls | unique | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
        for col in self.columns:
            lines.append(
                f"| `{col['name']}` | {col['dtype']} | {col['null_pct']} | "
                f"{col['n_unique']:,} | {col['notes']} |"
            )
        lines += ["", "First rows:", "```", self.preview, "```"]
        return "\n".join(lines)


def _column_notes(series: pd.Series) -> str:
    """A short, factual hint per column: range for numerics, examples for
    categoricals, span for dates."""
    non_null = series.dropna()
    if non_null.empty:
        return "all null"

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        try:
            return (
                f"min {non_null.min():,.4g} · median {non_null.median():,.4g} "
                f"· max {non_null.max():,.4g}"
            )
        except (TypeError, ValueError):
            return ""

    if pd.api.types.is_datetime64_any_dtype(series):
        return f"{non_null.min():%Y-%m-%d} → {non_null.max():%Y-%m-%d}"

    n_unique = non_null.nunique(dropna=True)
    ratio = n_unique / max(len(non_null), 1)
    is_categorical = n_unique <= CATEGORICAL_MAX_UNIQUE and (
        ratio <= CATEGORICAL_UNIQUE_RATIO or n_unique <= 20
    )
    if is_categorical:
        examples = list(non_null.unique()[:MAX_CATEGORY_EXAMPLES])
        rendered = ", ".join(str(e)[:24] for e in examples)
        suffix = ", …" if n_unique > MAX_CATEGORY_EXAMPLES else ""
        return f"categorical: {rendered}{suffix}"

    longest = non_null.astype(str).str.len().max()
    return f"free text, longest {int(longest)} chars"


def profile_frame(name: str, var_name: str, df: pd.DataFrame) -> SheetProfile:
    """Deterministic profile of one sheet."""
    columns: list[dict[str, Any]] = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        n_null = int(series.isna().sum())
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "n_null": n_null,
                "null_pct": f"{(n_null / n * 100):.1f}%" if n else "—",
                "n_unique": int(series.nunique(dropna=True)),
                "notes": _column_notes(series),
            }
        )

    with pd.option_context("display.max_columns", 40, "display.width", 200):
        preview = df.head(MAX_PREVIEW_ROWS).to_string()

    try:
        duplicate_rows = int(df.duplicated().sum())
    except TypeError:  # unhashable cell contents
        duplicate_rows = 0

    return SheetProfile(
        name=name,
        var_name=var_name,
        n_rows=n,
        n_cols=df.shape[1],
        columns=columns,
        preview=preview,
        duplicate_rows=duplicate_rows,
    )


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort datetime parsing for object columns that look like dates.

    Excel usually types real dates correctly; this catches the ones exported
    as text. Conservative: only converts when >=90% of non-null values parse.
    """
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        non_null = series.dropna()
        if non_null.empty or len(non_null) < 5:
            continue
        sample = non_null.head(200).astype(str)
        if not sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.9:
            continue
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if parsed.notna().sum() >= 0.9 * len(non_null):
            df[col] = parsed
    return df


def load_workbook(
    path: str | Path,
    *,
    coerce_dates: bool = True,
    max_rows_per_sheet: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Read every sheet of an .xlsx/.xls/.csv file.

    Returns `(frames, sheet_to_var)` where `frames` is keyed by the *variable*
    name the agent will use, and `sheet_to_var` maps original sheet names to
    those variable names.
    """
    path = Path(path)
    frames: dict[str, pd.DataFrame] = {}
    sheet_to_var: dict[str, str] = {}

    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else None
        raw = {path.stem: pd.read_csv(path, sep=sep, engine="python", nrows=max_rows_per_sheet)}
    else:
        raw = pd.read_excel(path, sheet_name=None, nrows=max_rows_per_sheet)

    used: set[str] = set()
    for sheet_name, df in raw.items():
        var = slugify_sheet(str(sheet_name))
        base, i = var, 2
        while var in used:
            var, i = f"{base}_{i}", i + 1
        used.add(var)

        df.columns = pd.Index([str(c).strip() for c in df.columns])
        # Drop the empty "Unnamed: N" columns Excel round-trips leave behind.
        unnamed = np.asarray(df.columns.str.match(r"^Unnamed: \d+$"), dtype=bool)
        all_empty = np.asarray(df.isna().all(), dtype=bool)
        df = df.loc[:, ~(unnamed & all_empty)]
        if coerce_dates:
            df = _coerce_dates(df)

        frames[var] = df
        sheet_to_var[str(sheet_name)] = var

    return frames, sheet_to_var


def workbook_overview(frames: dict[str, pd.DataFrame], sheet_to_var: dict[str, str]) -> str:
    """Full markdown profile of every sheet - this is what primes the agent."""
    var_to_sheet = {v: k for k, v in sheet_to_var.items()}
    blocks = [
        f"The workbook has {len(frames)} sheet(s). "
        "Each is already loaded as a pandas DataFrame in your execution namespace.",
        "",
    ]
    for var, df in frames.items():
        blocks.append(profile_frame(var_to_sheet.get(var, var), var, df).to_markdown())
        blocks.append("")
    return "\n".join(blocks)


def suggest_joins(frames: dict[str, pd.DataFrame]) -> str:
    """Flag columns shared between sheets - candidate join keys.

    Deliberately conservative: name match plus overlapping values. It only
    suggests; the model decides.
    """
    if len(frames) < 2:
        return ""

    names = list(frames)
    lines: list[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(frames[a].columns) & set(frames[b].columns)
            for col in sorted(shared):
                try:
                    va = set(frames[a][col].dropna().unique()[:5000])
                    vb = set(frames[b][col].dropna().unique()[:5000])
                except TypeError:
                    continue
                if not va or not vb:
                    continue
                overlap = len(va & vb) / min(len(va), len(vb))
                if overlap >= 0.3:
                    lines.append(
                        f"- `{a}` and `{b}` share `{col}` "
                        f"({overlap:.0%} of values overlap) - likely join key"
                    )
    if not lines:
        return ""
    return "Possible relationships between sheets:\n" + "\n".join(lines)


def build_namespace(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The execution namespace handed to the sandbox.

    pandas, numpy and matplotlib are pre-imported (the sandbox forbids
    `import`), plus every sheet as its own DataFrame. `df` aliases the first
    sheet so single-sheet workbooks read naturally.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless - required under Django/gunicorn
    import matplotlib.pyplot as plt

    ns: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "dfs": frames,
    }
    ns.update(frames)
    if frames:
        ns["df"] = next(iter(frames.values()))
    return ns
