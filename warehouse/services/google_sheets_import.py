"""
Fetch a Google Sheet's data and turn it into a real .xlsx file, so the
spreadsheet analyst pipeline (warehouse.agent.workbook.load_workbook,
profiling, sandboxed pandas execution) can treat a *linked* sheet exactly
like an *uploaded* file — no separate data-loading path to build or trust.

Reuses the existing GoogleSheetsService (warehouse/sheet_service.py), the
same service-account-authenticated client already backing the Spreadsheets
tool elsewhere in this app. A sheet must be shared with that service
account's e-mail (found in the key JSON under `client_email`) to be
readable here — same requirement as every other Sheets operation in this
project, nothing new introduced.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from warehouse.sheet_service import SheetsServiceError, get_service

ID_FROM_URL_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")


class GoogleSheetImportError(Exception):
    """Raised when a linked spreadsheet can't be fetched or has nothing to import."""


def extract_spreadsheet_id(id_or_url: str) -> str:
    """Pull the spreadsheetId out of a pasted URL, or pass through a raw ID."""
    value = (id_or_url or "").strip()
    match = ID_FROM_URL_RE.search(value)
    return match.group(1) if match else value


def fetch_google_sheet_as_xlsx(spreadsheet_id: str) -> tuple[bytes, str]:
    """Pull every tab's values and pack them into one in-memory .xlsx.

    Returns (xlsx_bytes, spreadsheet_title). Each Sheets tab becomes one
    Excel sheet, first row treated as the header — the same shape
    workbook.py already expects from an uploaded .xlsx.

    Raises:
        GoogleSheetImportError: spreadsheet not found/not shared with the
            service account, or every tab is empty.
    """
    service = get_service()
    try:
        meta = service.get_spreadsheet(spreadsheet_id)
    except SheetsServiceError as exc:
        raise GoogleSheetImportError(
            f"Could not open that spreadsheet: {exc}. Make sure it's shared "
            "with this app's service account (Editor or Viewer)."
        ) from exc

    title = (meta.get("properties") or {}).get("title") or spreadsheet_id
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if not tabs:
        raise GoogleSheetImportError("That spreadsheet has no tabs.")

    # Fetch and shape every tab BEFORE opening the writer — openpyxl raises
    # its own (confusing) IndexError trying to save a workbook with zero
    # sheets, so "is there anything to write at all" must be decided first,
    # not discovered after the fact while the writer is already open.
    sheets: list[tuple[str, pd.DataFrame]] = []
    seen_names: dict[str, int] = {}
    for tab_title in tabs:
        try:
            values = service.read_values(spreadsheet_id, tab_title)
        except SheetsServiceError as exc:
            raise GoogleSheetImportError(f"Could not read tab '{tab_title}': {exc}") from exc
        if not values:
            continue

        header, *rows = values
        width = len(header)
        # Sheets omits trailing empty cells per row — pad so every row
        # aligns with the header before handing off to pandas.
        padded_rows = [row + [""] * (width - len(row)) for row in rows]
        df = pd.DataFrame(padded_rows, columns=header)

        safe_name = re.sub(r"[\\\[\]/*?:]", "_", tab_title)[:31] or "Sheet1"
        # Two distinct tab names can sanitize/truncate to the same Excel
        # name (e.g. "Sales/Q1" and "Sales?Q1" both -> "Sales_Q1"). Without
        # deduping, pandas' ExcelWriter doesn't error on the second
        # to_excel() with a repeated sheet_name -- it silently clobbers the
        # first tab's data instead, so this isn't optional polish.
        count = seen_names.get(safe_name, 0)
        seen_names[safe_name] = count + 1
        if count:
            suffix = f"_{count + 1}"
            safe_name = safe_name[: 31 - len(suffix)] + suffix
        sheets.append((safe_name, df))

    if not sheets:
        raise GoogleSheetImportError("Every tab in that spreadsheet is empty.")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for safe_name, df in sheets:
            df.to_excel(writer, sheet_name=safe_name, index=False)

    buf.seek(0)
    return buf.read(), title
