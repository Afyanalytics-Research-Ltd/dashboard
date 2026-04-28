"""Thin wrapper around the Google Sheets and Drive APIs.

Auth: a service account. Create one in Google Cloud Console, enable the
Google Sheets API and Google Drive API, download the JSON key, and point
``GOOGLE_SERVICE_ACCOUNT_FILE`` at it (or paste it into
``GOOGLE_SERVICE_ACCOUNT_JSON``).

Important: a service account is its own identity. To touch an existing
spreadsheet that wasn't created by the service account, share that
spreadsheet with the service account's email (found in the JSON key).
"""
from __future__ import annotations

import json
import threading
from typing import Any, Iterable, Sequence

from django.conf import settings
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class SheetsServiceError(Exception):
    """Raised when an upstream Google API call fails."""


_lock = threading.Lock()
_singleton: "GoogleSheetsService | None" = None


def get_service() -> "GoogleSheetsService":
    """Return a process-wide singleton (cheap to call from views)."""
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = GoogleSheetsService()
        return _singleton


class GoogleSheetsService:
    """High-level operations on Google Sheets / Drive.

    Every method either returns a plain ``dict`` (the raw Google response)
    or raises ``SheetsServiceError``.
    """

    def __init__(self) -> None:
        creds = self._load_credentials()
        # cache_discovery=False avoids a noisy warning when oauth2client is absent.
        self.sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
        self.drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    # ------------------------------------------------------------------ auth
    @staticmethod
    def _load_credentials() -> service_account.Credentials:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        raw_json = getattr(settings, "GOOGLE_SERVICE_ACCOUNT_JSON", "") or ""
        if raw_json.strip():
            info = json.loads(raw_json)
            return service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            )
        path = settings.GOOGLE_SERVICE_ACCOUNT_FILE
        return service_account.Credentials.from_service_account_file(
            path, scopes=scopes
        )

    # ----------------------------------------------------------- spreadsheets
    def create_spreadsheet(
        self,
        title: str,
        sheet_titles: Sequence[str] | None = None,
    ) -> dict:
        """Create a new spreadsheet and return the API response.

        ``sheet_titles`` lets you create multiple tabs at once. If empty,
        Google creates a single default ``Sheet1`` tab.
        """
        body: dict[str, Any] = {"properties": {"title": title}}
        if sheet_titles:
            body["sheets"] = [
                {"properties": {"title": t}} for t in sheet_titles if t
            ]
        try:
            return (
                self.sheets.spreadsheets()
                .create(body=body, fields="spreadsheetId,spreadsheetUrl,properties,sheets")
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    def get_spreadsheet(self, spreadsheet_id: str) -> dict:
        try:
            return (
                self.sheets.spreadsheets()
                .get(spreadsheetId=spreadsheet_id)
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    def delete_spreadsheet(self, spreadsheet_id: str) -> None:
        """Permanently deletes the spreadsheet via the Drive API."""
        try:
            self.drive.files().delete(fileId=spreadsheet_id).execute()
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    # ----------------------------------------------------------------- values
    def read_values(
        self,
        spreadsheet_id: str,
        range_a1: str,
        value_render_option: str = "FORMATTED_VALUE",
    ) -> list[list[Any]]:
        """Read a range. ``range_a1`` is e.g. ``Sheet1!A1:D20``."""
        try:
            resp = (
                self.sheets.spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=range_a1,
                    valueRenderOption=value_render_option,
                )
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e
        return resp.get("values", [])

    def update_values(
        self,
        spreadsheet_id: str,
        range_a1: str,
        values: Sequence[Sequence[Any]],
        value_input_option: str = "USER_ENTERED",
    ) -> dict:
        """Overwrite a range. ``USER_ENTERED`` lets formulas like ``=A1+B1`` work."""
        try:
            return (
                self.sheets.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=range_a1,
                    valueInputOption=value_input_option,
                    body={"values": list(values)},
                )
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    def append_values(
        self,
        spreadsheet_id: str,
        range_a1: str,
        values: Sequence[Sequence[Any]],
        value_input_option: str = "USER_ENTERED",
        insert_data_option: str = "INSERT_ROWS",
    ) -> dict:
        """Append rows below the existing data in ``range_a1``."""
        try:
            return (
                self.sheets.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=range_a1,
                    valueInputOption=value_input_option,
                    insertDataOption=insert_data_option,
                    body={"values": list(values)},
                )
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    def clear_values(self, spreadsheet_id: str, range_a1: str) -> dict:
        try:
            return (
                self.sheets.spreadsheets()
                .values()
                .clear(spreadsheetId=spreadsheet_id, range=range_a1, body={})
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    def batch_update_values(
        self,
        spreadsheet_id: str,
        data: Sequence[dict],
        value_input_option: str = "USER_ENTERED",
    ) -> dict:
        """Update many disjoint ranges in one HTTP round-trip.

        ``data`` is a list of ``{"range": "Sheet1!A1:B2", "values": [[...]]}``.
        """
        body = {"valueInputOption": value_input_option, "data": list(data)}
        try:
            return (
                self.sheets.spreadsheets()
                .values()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    # --------------------------------------------------------------- low level
    def batch_update(self, spreadsheet_id: str, requests: Sequence[dict]) -> dict:
        """Send raw spreadsheets.batchUpdate requests.

        This is the escape hatch for anything not covered by the helpers
        below: conditional formatting, charts, protected ranges, etc.
        """
        try:
            return (
                self.sheets.spreadsheets()
                .batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={"requests": list(requests)},
                )
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    # ------------------------------------------------------- worksheet (tabs)
    def add_sheet(self, spreadsheet_id: str, title: str) -> dict:
        return self.batch_update(
            spreadsheet_id,
            [{"addSheet": {"properties": {"title": title}}}],
        )

    def delete_sheet(self, spreadsheet_id: str, sheet_id: int) -> dict:
        return self.batch_update(
            spreadsheet_id, [{"deleteSheet": {"sheetId": sheet_id}}]
        )

    def rename_sheet(
        self, spreadsheet_id: str, sheet_id: int, new_title: str
    ) -> dict:
        return self.batch_update(
            spreadsheet_id,
            [
                {
                    "updateSheetProperties": {
                        "properties": {"sheetId": sheet_id, "title": new_title},
                        "fields": "title",
                    }
                }
            ],
        )

    def list_sheets(self, spreadsheet_id: str) -> list[dict]:
        """Return ``[{sheetId, title, index, gridProperties}, ...]`` for tabs."""
        meta = self.get_spreadsheet(spreadsheet_id)
        return [s["properties"] for s in meta.get("sheets", [])]

    def find_sheet_id(self, spreadsheet_id: str, title: str) -> int | None:
        for props in self.list_sheets(spreadsheet_id):
            if props.get("title") == title:
                return props.get("sheetId")
        return None

    # ------------------------------------------------------------- formatting
    def format_cells(
        self,
        spreadsheet_id: str,
        sheet_id: int,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        *,
        bold: bool | None = None,
        italic: bool | None = None,
        font_size: int | None = None,
        background_rgb: tuple[float, float, float] | None = None,
        foreground_rgb: tuple[float, float, float] | None = None,
        horizontal_alignment: str | None = None,  # LEFT, CENTER, RIGHT
        number_format: dict | None = None,        # e.g. {"type": "CURRENCY", "pattern": "$#,##0.00"}
    ) -> dict:
        """Apply common formatting to a rectangular range.

        Indices are 0-based, half-open: rows [start_row, end_row),
        columns [start_col, end_col).
        """
        text_format: dict[str, Any] = {}
        if bold is not None:
            text_format["bold"] = bold
        if italic is not None:
            text_format["italic"] = italic
        if font_size is not None:
            text_format["fontSize"] = font_size
        if foreground_rgb is not None:
            r, g, b = foreground_rgb
            text_format["foregroundColor"] = {"red": r, "green": g, "blue": b}

        cell_format: dict[str, Any] = {}
        if text_format:
            cell_format["textFormat"] = text_format
        if background_rgb is not None:
            r, g, b = background_rgb
            cell_format["backgroundColor"] = {"red": r, "green": g, "blue": b}
        if horizontal_alignment is not None:
            cell_format["horizontalAlignment"] = horizontal_alignment
        if number_format is not None:
            cell_format["numberFormat"] = number_format

        # Build the "fields" mask covering only what the caller actually set.
        field_parts: list[str] = []
        if "textFormat" in cell_format:
            tf_keys = ",".join(text_format.keys())
            field_parts.append(f"userEnteredFormat.textFormat({tf_keys})")
        if "backgroundColor" in cell_format:
            field_parts.append("userEnteredFormat.backgroundColor")
        if "horizontalAlignment" in cell_format:
            field_parts.append("userEnteredFormat.horizontalAlignment")
        if "numberFormat" in cell_format:
            field_parts.append("userEnteredFormat.numberFormat")
        if not field_parts:
            raise SheetsServiceError("format_cells called with no formatting options")

        request = {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": start_row,
                    "endRowIndex": end_row,
                    "startColumnIndex": start_col,
                    "endColumnIndex": end_col,
                },
                "cell": {"userEnteredFormat": cell_format},
                "fields": ",".join(field_parts),
            }
        }
        return self.batch_update(spreadsheet_id, [request])

    def freeze_rows(self, spreadsheet_id: str, sheet_id: int, row_count: int) -> dict:
        return self.batch_update(
            spreadsheet_id,
            [
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": sheet_id,
                            "gridProperties": {"frozenRowCount": row_count},
                        },
                        "fields": "gridProperties.frozenRowCount",
                    }
                }
            ],
        )

    # -------------------------------------------------------- row / column ops
    def delete_rows(
        self, spreadsheet_id: str, sheet_id: int, start_row: int, end_row: int
    ) -> dict:
        """Delete rows [start_row, end_row), 0-based."""
        return self.batch_update(
            spreadsheet_id,
            [
                {
                    "deleteDimension": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": start_row,
                            "endIndex": end_row,
                        }
                    }
                }
            ],
        )

    def insert_rows(
        self, spreadsheet_id: str, sheet_id: int, start_row: int, count: int
    ) -> dict:
        return self.batch_update(
            spreadsheet_id,
            [
                {
                    "insertDimension": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": start_row,
                            "endIndex": start_row + count,
                        },
                        "inheritFromBefore": start_row > 0,
                    }
                }
            ],
        )

    # --------------------------------------------------------------- sharing
    def share(
        self,
        spreadsheet_id: str,
        email: str,
        role: str = "writer",  # reader | commenter | writer | owner
        notify: bool = False,
    ) -> dict:
        """Share the spreadsheet with a user by email."""
        if role not in {"reader", "commenter", "writer", "owner"}:
            raise SheetsServiceError(f"Unknown role: {role}")
        body = {"type": "user", "role": role, "emailAddress": email}
        try:
            return (
                self.drive.permissions()
                .create(
                    fileId=spreadsheet_id,
                    body=body,
                    sendNotificationEmail=notify,
                    fields="id,emailAddress,role",
                )
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e

    def list_permissions(self, spreadsheet_id: str) -> list[dict]:
        try:
            resp = (
                self.drive.permissions()
                .list(
                    fileId=spreadsheet_id,
                    fields="permissions(id,emailAddress,role,type,displayName)",
                )
                .execute()
            )
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e
        return resp.get("permissions", [])

    def remove_permission(self, spreadsheet_id: str, permission_id: str) -> None:
        try:
            self.drive.permissions().delete(
                fileId=spreadsheet_id, permissionId=permission_id
            ).execute()
        except HttpError as e:
            raise SheetsServiceError(str(e)) from e


# --------------------------------------------------------------------- helpers
def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """Convert ``#rrggbb`` (or ``rrggbb``) to a (r, g, b) triple in [0, 1]."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {hex_color!r}")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rows_from_dicts(
    rows: Iterable[dict], header: Sequence[str]
) -> list[list[Any]]:
    """Turn a list of dicts into a 2D list aligned to ``header``."""
    return [[row.get(col, "") for col in header] for row in rows]