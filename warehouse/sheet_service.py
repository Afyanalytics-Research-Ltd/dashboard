"""Thin wrapper around the Google Sheets and Drive v3 APIs.

Authentication: a service account JSON key. Either set the path in
``GOOGLE_SERVICE_ACCOUNT_FILE`` or paste the JSON content into
``GOOGLE_SERVICE_ACCOUNT_JSON`` (useful for Docker / env-var deployments).

A service account is its own Google identity. To access a spreadsheet that
was NOT created by the service account, share it with the service account
e-mail address (found in the JSON key under ``client_email``).
"""

import json
import logging
import threading
from typing import Any, Iterable, Optional, Sequence

from django.conf import settings
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


class SheetsServiceError(Exception):
    """Raised when an upstream Google Sheets or Drive API call fails."""


# ─────────────────────────────────────────────────── singleton plumbing

_lock = threading.Lock()
_singleton: Optional["GoogleSheetsService"] = None


def get_service() -> "GoogleSheetsService":
    """Return a process-wide singleton — cheap to call from views."""
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = GoogleSheetsService()
    return _singleton


# ──────────────────────────────────────────────────── service class

class GoogleSheetsService:
    """High-level operations on Google Sheets / Drive.

    Every public method either returns a plain Python value (dict, list, …)
    or raises :class:`SheetsServiceError`.
    """

    def __init__(self) -> None:
        creds = self._load_credentials()
        # cache_discovery=False avoids a noisy warning when oauth2client is absent.
        self.sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
        self.drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        logger.info("GoogleSheetsService initialised")

    # ─────────────────────────────────────────────────── credentials

    @staticmethod
    def _load_credentials() -> service_account.Credentials:
        """Build service-account credentials from settings.

        Prefers ``GOOGLE_SERVICE_ACCOUNT_JSON`` (inline JSON string) over
        ``GOOGLE_SERVICE_ACCOUNT_FILE`` (path to file on disk).
        """
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
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

    # ──────────────────────────────────────────── spreadsheet CRUD

    def create_spreadsheet(
        self,
        title: str,
        sheet_titles: Sequence[str] | None = None,
    ) -> dict:
        """Create a new spreadsheet and return the API response dict.

        Args:
            title: Title shown in Google Drive.
            sheet_titles: Optional list of tab names. If empty, Google
                creates a single default ``Sheet1`` tab.

        Returns:
            The raw ``spreadsheets.create`` response (includes
            ``spreadsheetId``, ``spreadsheetUrl``, ``properties``, and
            ``sheets``).

        Raises:
            SheetsServiceError: On any Google API error.
        """
        body: dict[str, Any] = {"properties": {"title": title}}
        if sheet_titles:
            body["sheets"] = [
                {"properties": {"title": t}} for t in sheet_titles if t
            ]
        try:
            response = (
                self.sheets.spreadsheets()
                .create(
                    body=body,
                    fields="spreadsheetId,spreadsheetUrl,properties,sheets",
                )
                .execute()
            )
        except HttpError as exc:
            logger.error("create_spreadsheet failed: %s", exc)
            raise SheetsServiceError(str(exc)) from exc
        logger.info("Created spreadsheet id=%s title=%r", response.get("spreadsheetId"), title)
        return response

    def get_spreadsheet(self, spreadsheet_id: str) -> dict:
        """Return full spreadsheet metadata (sheets, properties, etc.).

        Raises:
            SheetsServiceError: If the spreadsheet doesn't exist or the
                service account lacks access.
        """
        try:
            return (
                self.sheets.spreadsheets()
                .get(spreadsheetId=spreadsheet_id)
                .execute()
            )
        except HttpError as exc:
            logger.warning("get_spreadsheet %s failed: %s", spreadsheet_id, exc)
            raise SheetsServiceError(str(exc)) from exc

    def delete_spreadsheet(self, spreadsheet_id: str) -> None:
        """Permanently delete the spreadsheet via the Drive API.

        Raises:
            SheetsServiceError: On any Drive API error.
        """
        try:
            self.drive.files().delete(fileId=spreadsheet_id).execute()
        except HttpError as exc:
            logger.error("delete_spreadsheet %s failed: %s", spreadsheet_id, exc)
            raise SheetsServiceError(str(exc)) from exc
        logger.info("Deleted spreadsheet %s", spreadsheet_id)

    # ─────────────────────────────────────────────────────── values

    def read_values(
        self,
        spreadsheet_id: str,
        range_a1: str,
        value_render_option: str = "FORMATTED_VALUE",
    ) -> list[list[Any]]:
        """Read a rectangular range and return a 2-D list of cell values.

        Args:
            spreadsheet_id: The spreadsheet's ``spreadsheetId``.
            range_a1: A1 notation, e.g. ``Sheet1!A1:D20``.
            value_render_option: One of FORMATTED_VALUE, UNFORMATTED_VALUE,
                FORMULA.

        Returns:
            List of rows; each row is a list of cell values.  May be shorter
            than the requested range if trailing empty cells are omitted.

        Raises:
            SheetsServiceError: On API error.
        """
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
        except HttpError as exc:
            logger.error("read_values %s %r failed: %s", spreadsheet_id, range_a1, exc)
            raise SheetsServiceError(str(exc)) from exc
        return resp.get("values", [])

    def update_values(
        self,
        spreadsheet_id: str,
        range_a1: str,
        values: Sequence[Sequence[Any]],
        value_input_option: str = "USER_ENTERED",
    ) -> dict:
        """Overwrite ``range_a1`` with ``values``.

        ``USER_ENTERED`` lets formulas like ``=A1+B1`` and date strings work;
        ``RAW`` writes exact strings.

        Raises:
            SheetsServiceError: On API error.
        """
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
        except HttpError as exc:
            logger.error("update_values %s %r failed: %s", spreadsheet_id, range_a1, exc)
            raise SheetsServiceError(str(exc)) from exc

    def append_values(
        self,
        spreadsheet_id: str,
        range_a1: str,
        values: Sequence[Sequence[Any]],
        value_input_option: str = "USER_ENTERED",
        insert_data_option: str = "INSERT_ROWS",
    ) -> dict:
        """Append rows below the existing data in ``range_a1``.

        Args:
            insert_data_option: INSERT_ROWS (always add rows) or OVERWRITE
                (overwrite existing rows if present).

        Raises:
            SheetsServiceError: On API error.
        """
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
        except HttpError as exc:
            logger.error("append_values %s %r failed: %s", spreadsheet_id, range_a1, exc)
            raise SheetsServiceError(str(exc)) from exc

    def clear_values(self, spreadsheet_id: str, range_a1: str) -> dict:
        """Clear all values in ``range_a1`` (formatting is preserved).

        Raises:
            SheetsServiceError: On API error.
        """
        try:
            return (
                self.sheets.spreadsheets()
                .values()
                .clear(spreadsheetId=spreadsheet_id, range=range_a1, body={})
                .execute()
            )
        except HttpError as exc:
            logger.error("clear_values %s %r failed: %s", spreadsheet_id, range_a1, exc)
            raise SheetsServiceError(str(exc)) from exc

    def batch_update_values(
        self,
        spreadsheet_id: str,
        data: Sequence[dict],
        value_input_option: str = "USER_ENTERED",
    ) -> dict:
        """Update many disjoint ranges in a single HTTP round-trip.

        Args:
            data: List of dicts: ``[{"range": "Sheet1!A1:B2", "values": [[…]]}]``.

        Raises:
            SheetsServiceError: On API error.
        """
        body = {"valueInputOption": value_input_option, "data": list(data)}
        try:
            return (
                self.sheets.spreadsheets()
                .values()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
                .execute()
            )
        except HttpError as exc:
            logger.error("batch_update_values %s failed: %s", spreadsheet_id, exc)
            raise SheetsServiceError(str(exc)) from exc

    # ──────────────────────────────────────────── low-level batchUpdate

    def batch_update(self, spreadsheet_id: str, requests: Sequence[dict]) -> dict:
        """Send raw ``spreadsheets.batchUpdate`` requests.

        This is the escape hatch for operations not covered by the helpers:
        conditional formatting, charts, protected ranges, etc.

        Raises:
            SheetsServiceError: On API error.
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
        except HttpError as exc:
            logger.error("batch_update %s failed: %s", spreadsheet_id, exc)
            raise SheetsServiceError(str(exc)) from exc

    # ───────────────────────────────────────────────── worksheet tabs

    def add_sheet(self, spreadsheet_id: str, title: str) -> dict:
        """Add a new tab (worksheet) to the spreadsheet.

        Raises:
            SheetsServiceError: On API error.
        """
        return self.batch_update(
            spreadsheet_id,
            [{"addSheet": {"properties": {"title": title}}}],
        )

    def delete_sheet(self, spreadsheet_id: str, sheet_id: int) -> dict:
        """Delete a tab by its numeric ``sheetId``.

        Raises:
            SheetsServiceError: On API error.
        """
        return self.batch_update(
            spreadsheet_id,
            [{"deleteSheet": {"sheetId": sheet_id}}],
        )

    def rename_sheet(
        self, spreadsheet_id: str, sheet_id: int, new_title: str
    ) -> dict:
        """Rename a tab.

        Raises:
            SheetsServiceError: On API error.
        """
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
        """Return ``[{sheetId, title, index, gridProperties}, …]`` for all tabs.

        Raises:
            SheetsServiceError: On API error.
        """
        meta = self.get_spreadsheet(spreadsheet_id)
        return [s["properties"] for s in meta.get("sheets", [])]

    def find_sheet_id(self, spreadsheet_id: str, title: str) -> Optional[int]:
        """Return the numeric ``sheetId`` for the tab named ``title``, or None."""
        for props in self.list_sheets(spreadsheet_id):
            if props.get("title") == title:
                return props.get("sheetId")
        return None

    # ────────────────────────────────────────────────── formatting

    def format_cells(
        self,
        spreadsheet_id: str,
        sheet_id: int,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        *,
        bold: Optional[bool] = None,
        italic: Optional[bool] = None,
        font_size: Optional[int] = None,
        background_rgb: Optional[tuple[float, float, float]] = None,
        foreground_rgb: Optional[tuple[float, float, float]] = None,
        horizontal_alignment: Optional[str] = None,
        number_format: Optional[dict] = None,
    ) -> dict:
        """Apply common formatting to a rectangular range.

        All indices are 0-based, half-open: rows ``[start_row, end_row)``,
        columns ``[start_col, end_col)``.

        Args:
            bold: Apply / remove bold text.
            italic: Apply / remove italic.
            font_size: Point size.
            background_rgb: ``(r, g, b)`` in [0, 1].
            foreground_rgb: ``(r, g, b)`` in [0, 1].
            horizontal_alignment: One of LEFT, CENTER, RIGHT.
            number_format: Dict ``{"type": "CURRENCY", "pattern": "$#,##0.00"}``.

        Raises:
            SheetsServiceError: If no formatting options are provided, or on
                API error.
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

        # Build the field mask covering only what the caller actually set.
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
            raise SheetsServiceError("format_cells called with no formatting options.")

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
        """Freeze the first ``row_count`` rows on a sheet.

        Set ``row_count=0`` to unfreeze all rows.

        Raises:
            SheetsServiceError: On API error.
        """
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

    # ─────────────────────────────────────────── row / column ops

    def delete_rows(
        self, spreadsheet_id: str, sheet_id: int, start_row: int, end_row: int
    ) -> dict:
        """Delete rows ``[start_row, end_row)`` (0-based, half-open).

        Raises:
            SheetsServiceError: On API error.
        """
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
        """Insert ``count`` blank rows before row ``start_row`` (0-based).

        Raises:
            SheetsServiceError: On API error.
        """
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

    # ──────────────────────────────────────────────────── sharing

    def share(
        self,
        spreadsheet_id: str,
        email: str,
        role: str = "writer",
        notify: bool = False,
    ) -> dict:
        """Share the spreadsheet with a user by email.

        Args:
            role: One of ``reader``, ``commenter``, ``writer``.
            notify: Whether Google should send a notification email.

        Raises:
            SheetsServiceError: On unknown role or API error.
        """
        if role not in {"reader", "commenter", "writer"}:
            raise SheetsServiceError(f"Unknown role: {role!r}. Use reader/commenter/writer.")
        body = {"type": "user", "role": role, "emailAddress": email}
        try:
            result = (
                self.drive.permissions()
                .create(
                    fileId=spreadsheet_id,
                    body=body,
                    sendNotificationEmail=notify,
                    fields="id,emailAddress,role",
                )
                .execute()
            )
        except HttpError as exc:
            logger.error("share %s with %s failed: %s", spreadsheet_id, email, exc)
            raise SheetsServiceError(str(exc)) from exc
        logger.info("Shared %s with %s as %s", spreadsheet_id, email, role)
        return result

    def list_permissions(self, spreadsheet_id: str) -> list[dict]:
        """Return current permission list for the spreadsheet.

        Raises:
            SheetsServiceError: On API error.
        """
        try:
            resp = (
                self.drive.permissions()
                .list(
                    fileId=spreadsheet_id,
                    fields="permissions(id,emailAddress,role,type,displayName)",
                )
                .execute()
            )
        except HttpError as exc:
            logger.error("list_permissions %s failed: %s", spreadsheet_id, exc)
            raise SheetsServiceError(str(exc)) from exc
        return resp.get("permissions", [])

    def remove_permission(self, spreadsheet_id: str, permission_id: str) -> None:
        """Remove a single permission entry.

        Raises:
            SheetsServiceError: On API error.
        """
        try:
            self.drive.permissions().delete(
                fileId=spreadsheet_id, permissionId=permission_id
            ).execute()
        except HttpError as exc:
            logger.error(
                "remove_permission %s / %s failed: %s",
                spreadsheet_id, permission_id, exc,
            )
            raise SheetsServiceError(str(exc)) from exc
        logger.info("Removed permission %s from %s", permission_id, spreadsheet_id)


# ──────────────────────────────────────────────────── standalone helpers

def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """Convert ``#rrggbb`` (or ``rrggbb``) to a ``(r, g, b)`` triple in [0, 1].

    Raises:
        ValueError: If ``hex_color`` is not a valid 6-digit hex string.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {hex_color!r}")
    return tuple(int(h[i: i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rows_from_dicts(
    rows: Iterable[dict], header: Sequence[str]
) -> list[list[Any]]:
    """Turn a list of dicts into a 2-D list aligned to ``header``."""
    return [[row.get(col, "") for col in header] for row in rows]
