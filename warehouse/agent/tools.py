"""
The agent's toolbelt, built per session.

`build_tools(session)` returns LangChain `StructuredTool`s closed over one
`AnalysisSession`. Every docstring here is part of the prompt the model sees,
so they are written for the model, not for a human reader. Parameter
descriptions come from the `Annotated` metadata.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from langchain_core.tools import BaseTool, tool

from .sandbox import execute
from .session import AnalysisSession
from .workbook import profile_frame


def build_tools(session: AnalysisSession) -> list[BaseTool]:
    """Return tools bound to `session`. Cheap - call it per request."""

    @tool
    def list_sheets() -> str:
        """List every sheet in the workbook with its DataFrame variable name, row
        count and column names. Call this first if you are unsure what the
        workbook contains.
        """
        lines = []
        var_to_sheet = {v: k for k, v in session.sheet_to_var.items()}
        for var, df in session.frames.items():
            cols = ", ".join(f"`{c}`" for c in df.columns[:25])
            more = f" (+{df.shape[1] - 25} more)" if df.shape[1] > 25 else ""
            lines.append(
                f"- sheet '{var_to_sheet.get(var, var)}' -> `{var}`: "
                f"{len(df):,} rows x {df.shape[1]} cols\n    {cols}{more}"
            )
        return "\n".join(lines) or "The workbook is empty."

    @tool
    def profile_sheet(
        variable: Annotated[str, "DataFrame variable name, e.g. 'sales' or 'df'"],
    ) -> str:
        """Return a full profile of one sheet: per-column dtype, null rate,
        cardinality, numeric ranges, categorical values, date spans, duplicate row
        count and a preview. Use this before analysing a sheet you have not
        inspected - it is computed exactly, so trust it over your assumptions.
        """
        df = session.frames.get(variable)
        if df is None:
            available = ", ".join(f"`{k}`" for k in session.frames)
            return f"No sheet named `{variable}`. Available: {available}"
        var_to_sheet = {v: k for k, v in session.sheet_to_var.items()}
        return profile_frame(
            var_to_sheet.get(variable, variable), variable, df
        ).to_markdown()

    @tool
    def run_python(
        code: Annotated[
            str, "pandas/numpy code; end with a bare expression to see its value"
        ],
    ) -> str:
        """Execute pandas/numpy code against the workbook and return the output.

        The namespace persists across calls like a notebook kernel, so variables
        you define stay available. Pre-loaded: `pd`, `np`, `plt`, every sheet as
        its own DataFrame, `dfs` (dict of all sheets) and `df` (the first sheet).

        `import` is not permitted - everything you need is already in scope. Print
        or end with a bare expression to see results; long output is truncated, so
        aggregate or use .head() rather than dumping raw rows.
        """
        return execute(code, session.namespace, timeout=session.exec_timeout).as_text()

    @tool
    def make_chart(
        code: Annotated[str, "matplotlib code that draws onto the current figure"],
        title: Annotated[str, "short human-readable title for the chart"],
    ) -> str:
        """Draw a matplotlib chart and save it as a PNG shown to the user.

        Write plotting code using `plt` and the DataFrames in scope. Do NOT call
        plt.show(), plt.savefig() or plt.close() - the tool saves and closes the
        figure itself. Always label the axes and give the plot a title. Prefer one
        clear chart over a crowded subplot grid.
        """
        setup = "plt.close('all')\n_fig = plt.figure(figsize=(9, 5), dpi=140)\n"
        result = execute(setup + code, session.namespace, timeout=session.exec_timeout)
        if not result.ok:
            return result.as_text()

        plt = session.namespace["plt"]
        fig = plt.gcf()
        if not fig.get_axes():
            plt.close("all")
            return "ERROR: the code ran but drew nothing. Plot onto the current figure."

        path = session.new_artifact_path(title.lower().replace(" ", "-"), ".png")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close("all")
        session.record("chart", title, path)

        note = result.as_text(max_chars=800).strip()
        prefix = f"Chart saved and shown to the user: '{title}' ({path.name})."
        return prefix if "no output" in note or not note else f"{prefix}\n{note}"

    @tool
    def export_table(
        code: Annotated[
            str, "code whose final bare expression evaluates to a DataFrame"
        ],
        filename: Annotated[str, "file stem, without extension"],
        title: Annotated[str, "short description of what the table contains"],
    ) -> str:
        """Export a DataFrame to .xlsx and attach it as a downloadable file.

        Use this when the user wants results as a spreadsheet, or when a result
        table is too large to read in chat. The code must END with a bare
        expression that evaluates to a DataFrame.
        """
        capture_src = _capture_tail(code, "_export_target")
        if capture_src is None:
            return (
                "ERROR: the last line must be a bare expression evaluating to a "
                "DataFrame (e.g. end with `summary` or `df.groupby('x').sum()`)."
            )

        session.namespace.pop("_export_target", None)
        result = execute(capture_src, session.namespace, timeout=session.exec_timeout)
        if not result.ok:
            return result.as_text()

        obj = session.namespace.pop("_export_target", None)
        if obj is None:
            return "ERROR: the final expression evaluated to None, nothing to export."
        if not isinstance(obj, pd.DataFrame):
            obj = pd.DataFrame(obj)

        path = session.new_artifact_path(filename, ".xlsx")
        keep_index = not isinstance(obj.index, pd.RangeIndex)
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            obj.to_excel(writer, sheet_name="data", index=keep_index)
        session.record("table", title, path)

        return (
            f"Table exported: '{title}' ({path.name}, {len(obj):,} rows x "
            f"{obj.shape[1]} cols). The user can download it."
        )

    @tool
    def write_report(
        title: Annotated[str, "report title"],
        markdown: Annotated[str, "the full report body in Markdown"],
    ) -> str:
        """Save a written analysis report as a downloadable Markdown file.

        Call this once, at the END of a substantive analysis, when the user asked
        for a report or a summary. Ground every number in a computation you
        actually ran - never estimate. Reference charts you created by their
        title. A good report covers: what the data is, the key findings with
        figures, data-quality caveats, and what you would look at next.
        """
        path = session.new_artifact_path(title.lower().replace(" ", "-"), ".md")
        path.write_text(f"# {title}\n\n{markdown.strip()}\n", encoding="utf-8")
        session.record("report", title, path)
        return f"Report saved as '{title}' ({path.name}) and attached for download."

    return [list_sheets, profile_sheet, run_python, make_chart, export_table, write_report]


def _capture_tail(code: str, target: str) -> str | None:
    """Rewrite ``...\\nfinal_expression`` as ``...\\ntarget = final_expression``.

    Returns None when the snippet does not end in a bare expression, so the
    caller can tell the model what to fix. Parse errors pass through unchanged -
    the sandbox reports those with a line number.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return None
    tail = tree.body[-1]
    tree.body[-1] = ast.Assign(
        targets=[ast.Name(id=target, ctx=ast.Store())], value=tail.value
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
