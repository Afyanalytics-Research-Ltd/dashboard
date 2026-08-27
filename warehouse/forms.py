"""Django forms for every Google Sheets and Snowflake operation in the web UI.

Each form ships Bootstrap-friendly widgets (``form-control`` / ``form-select`` /
``form-check-input``) so templates stay thin. Form parsing helpers turn
textarea input into the 2D lists the Sheets API expects.
"""

import csv
import io
import re
from typing import Any

from django import forms


# ──────────────────────────────────────────── shared widget helpers

def _ctrl(extra: str = "") -> dict:
    return {"class": f"form-control {extra}".strip()}


def _select(extra: str = "") -> dict:
    return {"class": f"form-select {extra}".strip()}


def _check() -> dict:
    return {"class": "form-check-input"}


# ──────────────────────────────────────────── value parsing helpers

def parse_table_text(text: str) -> list[list[str]]:
    """Parse a textarea blob into a 2D list.

    Lines are rows; cells are split by tab if any tab is present, otherwise
    by comma (CSV-style, with quoting handled). Empty and whitespace-only
    lines are dropped.
    """
    text = (text or "").strip()
    if not text:
        return []
    if "\t" in text:
        return [
            line.split("\t")
            for line in text.splitlines()
            if line.strip() != ""
        ]
    reader = csv.reader(io.StringIO(text))
    return [row for row in reader if any(c.strip() != "" for c in row)]


def format_table_text(values: list[list[Any]]) -> str:
    """Render a 2D list back to TSV text for textarea preview."""
    return "\n".join(
        "\t".join("" if c is None else str(c) for c in row)
        for row in values
    )


_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _validate_hex(value: str) -> str:
    if not value:
        return value
    if not _HEX_RE.match(value):
        raise forms.ValidationError("Must be a 6-digit hex color, e.g. #1f77b4.")
    return value if value.startswith("#") else f"#{value}"


# ══════════════════════════════════════════════════════════════════ forms

# ── home page ────────────────────────────────────────────────────

class CreateSpreadsheetForm(forms.Form):
    title = forms.CharField(
        max_length=512,
        label="Spreadsheet Title",
        widget=forms.TextInput(attrs={**_ctrl(), "placeholder": "My Healthcare Data"}),
        help_text="Title shown in Google Drive.",
        error_messages={"required": "Please enter a title for the spreadsheet."},
    )
    sheet_titles = forms.CharField(
        required=False,
        label="Tab Names (optional)",
        widget=forms.TextInput(attrs={**_ctrl(), "placeholder": "Sheet1, Sheet2, Summary"}),
        help_text="Optional: comma-separated tab names, e.g. 'Sales, Costs, Summary'.",
    )

    def clean_sheet_titles(self) -> list[str]:
        raw = self.cleaned_data.get("sheet_titles", "") or ""
        return [t.strip() for t in raw.split(",") if t.strip()]


class OpenSpreadsheetForm(forms.Form):
    """Used on the home page to jump to an existing spreadsheet by ID or URL."""

    id_or_url = forms.CharField(
        label="Spreadsheet ID or URL",
        widget=forms.TextInput(attrs={
            **_ctrl("font-monospace"),
            "placeholder": "https://docs.google.com/spreadsheets/d/… or just the ID",
        }),
        help_text="Paste the full Google Sheets URL or just the spreadsheet ID.",
        error_messages={"required": "Please enter a spreadsheet ID or URL."},
    )

    _ID_FROM_URL = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")

    def clean_id_or_url(self) -> str:
        val = (self.cleaned_data["id_or_url"] or "").strip()
        m = self._ID_FROM_URL.search(val)
        if m:
            return m.group(1)
        return val


# ── values ───────────────────────────────────────────────────────

class ReadValuesForm(forms.Form):
    range_notation = forms.CharField(
        label="Range (A1 notation)",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "Sheet1!A1:Z100"}),
        initial="Sheet1!A1:Z100",
        help_text="A1 notation, e.g. Sheet1!A1:D20 or just A1:Z100.",
    )


class UpdateValuesForm(forms.Form):
    range_notation = forms.CharField(
        label="Range (A1 notation)",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "Sheet1!A1"}),
        initial="Sheet1!A1",
        help_text="Top-left cell or full range where data will be written.",
    )
    values = forms.CharField(
        label="Values (TSV or CSV)",
        widget=forms.Textarea(attrs={**_ctrl("font-monospace"), "rows": 8,
                                     "placeholder": "Name\tAge\nAlice\t30\nBob\t42"}),
        help_text="One row per line. Tabs OR commas separate cells. Formulas like =A1+B1 are supported with USER_ENTERED.",
        error_messages={"required": "Please provide at least one row of data."},
    )
    value_input_option = forms.ChoiceField(
        choices=[
            ("USER_ENTERED", "USER_ENTERED — parses formulas and dates"),
            ("RAW", "RAW — literal string values only"),
        ],
        initial="USER_ENTERED",
        widget=forms.Select(attrs=_select()),
        label="Input Mode",
    )

    def clean_values(self) -> list[list[str]]:
        rows = parse_table_text(self.cleaned_data["values"])
        if not rows:
            raise forms.ValidationError("Provide at least one row of data.")
        return rows


class AppendValuesForm(UpdateValuesForm):
    insert_data_option = forms.ChoiceField(
        choices=[
            ("INSERT_ROWS", "INSERT_ROWS — always add new rows"),
            ("OVERWRITE", "OVERWRITE — write over existing rows if present"),
        ],
        initial="INSERT_ROWS",
        widget=forms.Select(attrs=_select()),
        label="Append Mode",
    )


class ClearRangeForm(forms.Form):
    range_notation = forms.CharField(
        label="Range to Clear (A1 notation)",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "Sheet1!A1:Z1000"}),
        initial="Sheet1!A1:Z1000",
        help_text="All cell values in this range will be cleared (formatting is preserved).",
    )


# ── batch update ─────────────────────────────────────────────────

class BatchUpdateForm(forms.Form):
    """Several disjoint range updates in one round-trip.

    Format (one update per blank-line-separated block)::

        Sheet1!A1:B2
        Alice,30
        Bob,42

        Sheet1!D1:D2
        =A1+B1
        =A2+B2
    """

    multi_block = forms.CharField(
        label="Batch Blocks",
        widget=forms.Textarea(attrs={**_ctrl("font-monospace"), "rows": 14,
                                     "placeholder": "Sheet1!A1:B2\nAlice,30\nBob,42\n\nSheet1!D1\n=SUM(A:A)"}),
        help_text=(
            "Each block: first line is the A1 range, following lines are rows. "
            "Separate blocks with a blank line."
        ),
    )
    value_input_option = forms.ChoiceField(
        choices=[("USER_ENTERED", "USER_ENTERED"), ("RAW", "RAW")],
        initial="USER_ENTERED",
        widget=forms.Select(attrs=_select()),
        label="Input Mode",
    )

    def clean_multi_block(self) -> list[dict]:
        text = self.cleaned_data["multi_block"]
        chunks = [c for c in re.split(r"\n\s*\n", text.strip()) if c.strip()]
        if not chunks:
            raise forms.ValidationError("Provide at least one block.")
        out: list[dict] = []
        for i, chunk in enumerate(chunks, 1):
            lines = chunk.splitlines()
            if len(lines) < 2:
                raise forms.ValidationError(
                    f"Block {i} needs a range line plus at least one data row."
                )
            range_a1 = lines[0].strip()
            values = parse_table_text("\n".join(lines[1:]))
            if not values:
                raise forms.ValidationError(f"Block {i} has no data rows.")
            out.append({"range": range_a1, "values": values})
        return out


# ── tabs ─────────────────────────────────────────────────────────

class AddTabForm(forms.Form):
    tab_title = forms.CharField(
        max_length=255,
        label="Tab Name",
        widget=forms.TextInput(attrs={**_ctrl(), "placeholder": "Sheet2"}),
        error_messages={"required": "Please enter a name for the new tab."},
    )


class RenameTabForm(forms.Form):
    sheet_id = forms.IntegerField(widget=forms.HiddenInput())
    new_title = forms.CharField(
        max_length=255,
        label="New Tab Name",
        widget=forms.TextInput(attrs=_ctrl()),
        error_messages={"required": "Please enter the new tab name."},
    )


class DeleteTabForm(forms.Form):
    sheet_id = forms.IntegerField(widget=forms.HiddenInput())
    confirm = forms.BooleanField(
        required=True,
        label="Yes, I want to delete this tab",
        widget=forms.CheckboxInput(attrs=_check()),
        error_messages={"required": "You must confirm tab deletion."},
    )


# ── formatting ───────────────────────────────────────────────────

class FormatCellsForm(forms.Form):
    sheet_id = forms.IntegerField(
        label="Sheet ID",
        widget=forms.NumberInput(attrs={**_ctrl(), "placeholder": "0"}),
        help_text="Numeric sheet ID (sheetId), visible in the Tabs list.",
    )
    start_row = forms.IntegerField(
        min_value=0, initial=0,
        label="Start Row",
        widget=forms.NumberInput(attrs=_ctrl()),
        help_text="0-based, inclusive.",
    )
    end_row = forms.IntegerField(
        min_value=1, initial=1,
        label="End Row",
        widget=forms.NumberInput(attrs=_ctrl()),
        help_text="0-based, exclusive.",
    )
    start_col = forms.IntegerField(
        min_value=0, initial=0,
        label="Start Column",
        widget=forms.NumberInput(attrs=_ctrl()),
        help_text="0-based, inclusive.",
    )
    end_col = forms.IntegerField(
        min_value=1, initial=1,
        label="End Column",
        widget=forms.NumberInput(attrs=_ctrl()),
        help_text="0-based, exclusive.",
    )

    bold = forms.BooleanField(required=False, label="Bold", widget=forms.CheckboxInput(attrs=_check()))
    italic = forms.BooleanField(required=False, label="Italic", widget=forms.CheckboxInput(attrs=_check()))
    font_size = forms.IntegerField(
        required=False, min_value=6, max_value=400,
        label="Font Size",
        widget=forms.NumberInput(attrs={**_ctrl(), "placeholder": "10"}),
    )
    background_hex = forms.CharField(
        required=False, label="Background Color (#hex)",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "#fff2cc"}),
    )
    foreground_hex = forms.CharField(
        required=False, label="Text Color (#hex)",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "#1f77b4"}),
    )
    horizontal_alignment = forms.ChoiceField(
        required=False,
        label="Horizontal Alignment",
        choices=[("", "— no change —"), ("LEFT", "Left"), ("CENTER", "Center"), ("RIGHT", "Right")],
        widget=forms.Select(attrs=_select()),
    )
    number_format_type = forms.ChoiceField(
        required=False,
        label="Number Format Type",
        choices=[
            ("", "— no change —"),
            ("TEXT", "Text"),
            ("NUMBER", "Number"),
            ("PERCENT", "Percent"),
            ("CURRENCY", "Currency"),
            ("DATE", "Date"),
            ("TIME", "Time"),
            ("DATE_TIME", "Date & Time"),
            ("SCIENTIFIC", "Scientific"),
        ],
        widget=forms.Select(attrs=_select()),
    )
    number_format_pattern = forms.CharField(
        required=False,
        label="Number Format Pattern",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "$#,##0.00"}),
        help_text="Optional pattern, e.g. $#,##0.00 or YYYY-MM-DD.",
    )

    def clean_background_hex(self) -> str:
        return _validate_hex(self.cleaned_data.get("background_hex", "") or "")

    def clean_foreground_hex(self) -> str:
        return _validate_hex(self.cleaned_data.get("foreground_hex", "") or "")

    def clean(self) -> dict:
        cleaned = super().clean()
        if (cleaned.get("end_row") is not None
                and cleaned.get("start_row") is not None
                and cleaned["end_row"] <= cleaned["start_row"]):
            self.add_error("end_row", "End row must be greater than start row.")
        if (cleaned.get("end_col") is not None
                and cleaned.get("start_col") is not None
                and cleaned["end_col"] <= cleaned["start_col"]):
            self.add_error("end_col", "End column must be greater than start column.")
        return cleaned


class FreezeRowsForm(forms.Form):
    sheet_id = forms.IntegerField(
        label="Sheet ID",
        widget=forms.NumberInput(attrs={**_ctrl(), "placeholder": "0"}),
    )
    row_count = forms.IntegerField(
        min_value=0, max_value=1000, initial=1,
        label="Number of Rows to Freeze",
        widget=forms.NumberInput(attrs=_ctrl()),
        help_text="Set to 0 to unfreeze all rows.",
    )


# ── rows ─────────────────────────────────────────────────────────

class DeleteRowsForm(forms.Form):
    sheet_id = forms.IntegerField(
        label="Sheet ID",
        widget=forms.NumberInput(attrs={**_ctrl(), "placeholder": "0"}),
    )
    start_row = forms.IntegerField(
        min_value=0,
        label="Start Row (inclusive)",
        widget=forms.NumberInput(attrs=_ctrl()),
        help_text="0-based index of first row to delete.",
    )
    end_row = forms.IntegerField(
        min_value=1,
        label="End Row (exclusive)",
        widget=forms.NumberInput(attrs=_ctrl()),
        help_text="0-based index — rows up to (but not including) this will be deleted.",
    )

    def clean(self) -> dict:
        cleaned = super().clean()
        if (cleaned.get("end_row") is not None
                and cleaned.get("start_row") is not None
                and cleaned["end_row"] <= cleaned["start_row"]):
            self.add_error("end_row", "End row must be greater than start row.")
        return cleaned


class InsertRowsForm(forms.Form):
    sheet_id = forms.IntegerField(
        label="Sheet ID",
        widget=forms.NumberInput(attrs={**_ctrl(), "placeholder": "0"}),
    )
    start_row = forms.IntegerField(
        min_value=0,
        label="Insert Before Row",
        widget=forms.NumberInput(attrs=_ctrl()),
        help_text="0-based index. Rows will be inserted before this position.",
    )
    count = forms.IntegerField(
        min_value=1, initial=1,
        label="Number of Rows",
        widget=forms.NumberInput(attrs=_ctrl()),
    )


# ── sharing ──────────────────────────────────────────────────────

class ShareForm(forms.Form):
    email = forms.EmailField(
        label="Email Address",
        widget=forms.EmailInput(attrs={**_ctrl(), "placeholder": "user@example.com"}),
        error_messages={"required": "Please enter an email address."},
    )
    role = forms.ChoiceField(
        label="Permission Level",
        choices=[
            ("reader", "Reader — can view"),
            ("commenter", "Commenter — can view and comment"),
            ("writer", "Writer (Editor) — can edit"),
        ],
        initial="writer",
        widget=forms.Select(attrs=_select()),
    )
    notify = forms.BooleanField(
        required=False,
        label="Send email notification to the user",
        widget=forms.CheckboxInput(attrs=_check()),
    )


class RemovePermissionForm(forms.Form):
    permission_id = forms.CharField(widget=forms.HiddenInput())


class DeleteSpreadsheetForm(forms.Form):
    confirm = forms.BooleanField(
        required=True,
        label="Yes, permanently delete this spreadsheet from Google Drive",
        widget=forms.CheckboxInput(attrs=_check()),
        error_messages={"required": "You must check the confirmation box to delete."},
    )


# ── snowflake ────────────────────────────────────────────────────

_BLOCKED = {'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE', 'GRANT', 'REVOKE'}

_BLOCKED_RE = re.compile(
    r'\b(' + '|'.join(_BLOCKED) + r')\b',
    re.IGNORECASE,
)


class SnowflakeQueryForm(forms.Form):
    query = forms.CharField(
        label="SQL Query",
        widget=forms.Textarea(attrs={
            **_ctrl("font-monospace"),
            "rows": 12,
            "placeholder": "SELECT * FROM my_table LIMIT 100",
            "spellcheck": "false",
        }),
        help_text="Only SELECT statements are permitted. Destructive keywords are blocked.",
        error_messages={"required": "Please enter a SQL query."},
    )

    def clean_query(self) -> str:
        sql = self.cleaned_data["query"].strip()
        match = _BLOCKED_RE.search(sql)
        if match:
            raise forms.ValidationError(
                f"The keyword '{match.group(0).upper()}' is not permitted. "
                "Only read-only SELECT queries are allowed."
            )
        return sql


# ──────────────────────────────────────────── spreadsheet analyst

from pathlib import Path as _Path

from django.conf import settings as _settings

from .models import Workbook

ANALYST_ALLOWED_SUFFIXES = {".xlsx", ".xlsm", ".xls", ".csv", ".tsv"}
ANALYST_MAX_UPLOAD_BYTES = getattr(_settings, "ANALYST_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)


class WorkbookUploadForm(forms.ModelForm):
    class Meta:
        model = Workbook
        fields = ["file"]
        widgets = {
            "file": forms.ClearableFileInput(
                attrs={"accept": ".xlsx,.xlsm,.xls,.csv,.tsv", "class": "file-input"}
            )
        }

    def clean_file(self):
        upload = self.cleaned_data["file"]
        suffix = _Path(upload.name).suffix.lower()
        if suffix not in ANALYST_ALLOWED_SUFFIXES:
            raise forms.ValidationError(
                f"Unsupported file type '{suffix}'. "
                f"Upload one of: {', '.join(sorted(ANALYST_ALLOWED_SUFFIXES))}."
            )
        if upload.size > ANALYST_MAX_UPLOAD_BYTES:
            raise forms.ValidationError(
                f"File is {upload.size / 1e6:.1f} MB; the limit is "
                f"{ANALYST_MAX_UPLOAD_BYTES / 1e6:.0f} MB."
            )
        return upload


class AnalystQuestionForm(forms.Form):
    question = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "rows": 2,
                "placeholder": "Ask about the data, or say 'analyse this workbook'…",
                "class": "question-input",
            }
        ),
        max_length=4000,
    )
