"""Django forms for every Google Sheets operation in the web UI.

Each form ships Bootstrap-friendly widgets (``form-control`` / ``form-select`` /
``form-check-input``) so templates stay thin. Form parsing helpers turn
textarea input into the 2D lists the Sheets API expects.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Any

from django import forms


# -------------------------------------------------------- shared widget mixin
def _ctrl(extra: str = "") -> dict:
    return {"class": f"form-control {extra}".strip()}


def _select(extra: str = "") -> dict:
    return {"class": f"form-select {extra}".strip()}


def _check() -> dict:
    return {"class": "form-check-input"}


# ------------------------------------------------------------- value parsing
def parse_table_text(text: str) -> list[list[str]]:
    """Parse a textarea blob into a 2D list.

    Lines are rows; cells are split by tab if any tab is present, otherwise by
    comma (CSV-style, with quoting handled). Empty lines are dropped.
    """
    text = (text or "").strip("\n")
    if not text:
        return []
    if "\t" in text:
        return [
            line.split("\t")
            for line in text.splitlines()
            if line.strip() != ""
        ]
    reader = csv.reader(io.StringIO(text))
    return [row for row in reader if any(c != "" for c in row)]


def format_table_text(values: list[list[Any]]) -> str:
    """Render a 2D list back to TSV text for textarea preview."""
    return "\n".join("\t".join("" if c is None else str(c) for c in row) for row in values)


_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _validate_hex(value: str) -> str:
    if not value:
        return value
    if not _HEX_RE.match(value):
        raise forms.ValidationError("Must be a 6-digit hex color, e.g. #1f77b4.")
    return value if value.startswith("#") else f"#{value}"


# ====================================================================== forms
class CreateSpreadsheetForm(forms.Form):
    title = forms.CharField(
        max_length=512,
        widget=forms.TextInput(attrs=_ctrl()),
        help_text="Title shown in Drive.",
    )
    sheet_titles = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs=_ctrl()),
        help_text="Optional: comma-separated tab names, e.g. 'Sales, Costs'.",
    )

    def clean_sheet_titles(self) -> list[str]:
        raw = self.cleaned_data.get("sheet_titles", "") or ""
        return [t.strip() for t in raw.split(",") if t.strip()]


class OpenSpreadsheetForm(forms.Form):
    """Used on the home page to jump to an existing spreadsheet by ID or URL."""

    id_or_url = forms.CharField(
        label="Spreadsheet ID or URL",
        widget=forms.TextInput(attrs=_ctrl("font-monospace")),
    )

    _ID_FROM_URL = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")

    def clean_id_or_url(self) -> str:
        val = (self.cleaned_data["id_or_url"] or "").strip()
        m = self._ID_FROM_URL.search(val)
        if m:
            return m.group(1)
        return val


# ----- values
class ReadValuesForm(forms.Form):
    range = forms.CharField(
        label="Range (A1)",
        widget=forms.TextInput(attrs=_ctrl("font-monospace")),
        initial="Sheet1!A1:Z100",
    )


class UpdateValuesForm(forms.Form):
    range = forms.CharField(
        label="Range (A1)",
        widget=forms.TextInput(attrs=_ctrl("font-monospace")),
        initial="Sheet1!A1",
    )
    values = forms.CharField(
        label="Values (TSV or CSV)",
        widget=forms.Textarea(attrs={**_ctrl("font-monospace"), "rows": 8}),
        help_text="One row per line. Tabs OR commas separate cells.",
    )
    value_input_option = forms.ChoiceField(
        choices=[("USER_ENTERED", "USER_ENTERED (parses formulas)"), ("RAW", "RAW")],
        initial="USER_ENTERED",
        widget=forms.Select(attrs=_select()),
    )

    def clean_values(self) -> list[list[str]]:
        rows = parse_table_text(self.cleaned_data["values"])
        if not rows:
            raise forms.ValidationError("Provide at least one row.")
        return rows


class AppendValuesForm(UpdateValuesForm):
    insert_data_option = forms.ChoiceField(
        choices=[("INSERT_ROWS", "INSERT_ROWS"), ("OVERWRITE", "OVERWRITE")],
        initial="INSERT_ROWS",
        widget=forms.Select(attrs=_select()),
    )


class ClearRangeForm(forms.Form):
    range = forms.CharField(
        label="Range (A1)",
        widget=forms.TextInput(attrs=_ctrl("font-monospace")),
        initial="Sheet1!A1:Z1000",
    )


# ----- tabs
class AddTabForm(forms.Form):
    title = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs=_ctrl()),
    )


class RenameTabForm(forms.Form):
    sheet_id = forms.IntegerField(widget=forms.HiddenInput())
    new_title = forms.CharField(max_length=255, widget=forms.TextInput(attrs=_ctrl()))


class DeleteTabForm(forms.Form):
    sheet_id = forms.IntegerField(widget=forms.HiddenInput())


# ----- formatting
class FormatCellsForm(forms.Form):
    sheet_id = forms.IntegerField(widget=forms.NumberInput(attrs=_ctrl()))
    start_row = forms.IntegerField(min_value=0, initial=0, widget=forms.NumberInput(attrs=_ctrl()))
    end_row = forms.IntegerField(min_value=1, initial=1, widget=forms.NumberInput(attrs=_ctrl()))
    start_col = forms.IntegerField(min_value=0, initial=0, widget=forms.NumberInput(attrs=_ctrl()))
    end_col = forms.IntegerField(min_value=1, initial=1, widget=forms.NumberInput(attrs=_ctrl()))

    bold = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs=_check()))
    italic = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs=_check()))
    font_size = forms.IntegerField(
        required=False, min_value=6, max_value=400,
        widget=forms.NumberInput(attrs=_ctrl()),
    )
    background_hex = forms.CharField(
        required=False, label="Background (#hex)",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "#fff2cc"}),
    )
    foreground_hex = forms.CharField(
        required=False, label="Text color (#hex)",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "#1f77b4"}),
    )
    horizontal_alignment = forms.ChoiceField(
        required=False,
        choices=[("", "—"), ("LEFT", "LEFT"), ("CENTER", "CENTER"), ("RIGHT", "RIGHT")],
        widget=forms.Select(attrs=_select()),
    )
    number_format_type = forms.ChoiceField(
        required=False,
        choices=[
            ("", "—"),
            ("TEXT", "TEXT"),
            ("NUMBER", "NUMBER"),
            ("PERCENT", "PERCENT"),
            ("CURRENCY", "CURRENCY"),
            ("DATE", "DATE"),
            ("TIME", "TIME"),
            ("DATE_TIME", "DATE_TIME"),
            ("SCIENTIFIC", "SCIENTIFIC"),
        ],
        widget=forms.Select(attrs=_select()),
    )
    number_format_pattern = forms.CharField(
        required=False, label="Number format pattern",
        widget=forms.TextInput(attrs={**_ctrl("font-monospace"), "placeholder": "$#,##0.00"}),
    )

    def clean_background_hex(self) -> str:
        return _validate_hex(self.cleaned_data.get("background_hex", "") or "")

    def clean_foreground_hex(self) -> str:
        return _validate_hex(self.cleaned_data.get("foreground_hex", "") or "")

    def clean(self) -> dict:
        cleaned = super().clean()
        if cleaned.get("end_row") is not None and cleaned.get("start_row") is not None:
            if cleaned["end_row"] <= cleaned["start_row"]:
                self.add_error("end_row", "Must be greater than start_row.")
        if cleaned.get("end_col") is not None and cleaned.get("start_col") is not None:
            if cleaned["end_col"] <= cleaned["start_col"]:
                self.add_error("end_col", "Must be greater than start_col.")
        return cleaned


class FreezeRowsForm(forms.Form):
    sheet_id = forms.IntegerField(widget=forms.NumberInput(attrs=_ctrl()))
    row_count = forms.IntegerField(
        min_value=0, max_value=1000, initial=1,
        widget=forms.NumberInput(attrs=_ctrl()),
    )


# ----- batch
class BatchUpdateForm(forms.Form):
    """Several disjoint range updates in one go.

    Format (one update per blank-line-separated block):

        Sheet1!A1:B2
        Alice,30
        Bob,42

        Sheet1!D1:D2
        =A1+B1
        =A2+B2
    """

    blocks = forms.CharField(
        widget=forms.Textarea(attrs={**_ctrl("font-monospace"), "rows": 14}),
        help_text="Each block: first line is the A1 range, following lines are the rows. Separate blocks with a blank line.",
    )
    value_input_option = forms.ChoiceField(
        choices=[("USER_ENTERED", "USER_ENTERED"), ("RAW", "RAW")],
        initial="USER_ENTERED",
        widget=forms.Select(attrs=_select()),
    )

    def clean_blocks(self) -> list[dict]:
        text = self.cleaned_data["blocks"]
        chunks = [c for c in re.split(r"\n\s*\n", text.strip()) if c.strip()]
        if not chunks:
            raise forms.ValidationError("Provide at least one block.")
        out: list[dict] = []
        for i, chunk in enumerate(chunks, 1):
            lines = chunk.splitlines()
            if len(lines) < 2:
                raise forms.ValidationError(
                    f"Block {i} needs a range line plus at least one row."
                )
            range_a1 = lines[0].strip()
            values = parse_table_text("\n".join(lines[1:]))
            if not values:
                raise forms.ValidationError(f"Block {i} has no data rows.")
            out.append({"range": range_a1, "values": values})
        return out


# ----- rows
class DeleteRowsForm(forms.Form):
    sheet_id = forms.IntegerField(widget=forms.NumberInput(attrs=_ctrl()))
    start_row = forms.IntegerField(
        min_value=0, help_text="0-based, inclusive.",
        widget=forms.NumberInput(attrs=_ctrl()),
    )
    end_row = forms.IntegerField(
        min_value=1, help_text="0-based, exclusive.",
        widget=forms.NumberInput(attrs=_ctrl()),
    )

    def clean(self) -> dict:
        cleaned = super().clean()
        if (cleaned.get("end_row") is not None
                and cleaned.get("start_row") is not None
                and cleaned["end_row"] <= cleaned["start_row"]):
            self.add_error("end_row", "Must be greater than start_row.")
        return cleaned


class InsertRowsForm(forms.Form):
    sheet_id = forms.IntegerField(widget=forms.NumberInput(attrs=_ctrl()))
    start_row = forms.IntegerField(min_value=0, widget=forms.NumberInput(attrs=_ctrl()))
    count = forms.IntegerField(min_value=1, initial=1, widget=forms.NumberInput(attrs=_ctrl()))


# ----- sharing
class ShareForm(forms.Form):
    email = forms.EmailField(widget=forms.EmailInput(attrs=_ctrl()))
    role = forms.ChoiceField(
        choices=[
            ("reader", "Reader"),
            ("commenter", "Commenter"),
            ("writer", "Writer (editor)"),
            ("owner", "Owner"),
        ],
        initial="writer",
        widget=forms.Select(attrs=_select()),
    )
    notify = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs=_check()))


class RemovePermissionForm(forms.Form):
    permission_id = forms.CharField(widget=forms.HiddenInput())


class DeleteSpreadsheetForm(forms.Form):
    confirm = forms.BooleanField(
        required=True, label="Yes, permanently delete this spreadsheet",
        widget=forms.CheckboxInput(attrs=_check()),
    )