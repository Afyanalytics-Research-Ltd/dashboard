"""
Comprehensive tests for the warehouse module.

All Google Sheets API calls and Snowflake connections are mocked so the test
suite can run without real credentials.

Test classes:
    TrackedSpreadsheetModelTests        — model CRUD + helpers
    SnowflakeQueryLogModelTests         — model CRUD + choices
    WarehouseViewPermissionTests        — role-based access control
    WarehouseHomeViewTests              — index GET / POST
    SpreadsheetDetailViewTests          — detail page
    SnowflakeQueryViewTests             — Snowflake query UI
    FormsTests                          — all form validations
    WarehouseAPITests                   — REST API (SpreadsheetViewSet)
    SnowflakeAPITests                   — REST API (Snowflake endpoints)
    FacilityScopeUnitTests              — facility_scope.py pure-function behavior
    FacilityScopedQueryViewTests        — SnowflakeQueryView enforcing facility scope
    FacilityScopedAPITests              — Snowflake API endpoints enforcing facility scope
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from authentication.roles import ROLE_CLIENT_ADMIN
from core.models import Client as OrgClient

from .forms import (
    AddTabForm,
    AppendValuesForm,
    BatchUpdateForm,
    ClearRangeForm,
    CreateSpreadsheetForm,
    DeleteRowsForm,
    DeleteSpreadsheetForm,
    DeleteTabForm,
    FormatCellsForm,
    FreezeRowsForm,
    InsertRowsForm,
    OpenSpreadsheetForm,
    ReadValuesForm,
    RemovePermissionForm,
    RenameTabForm,
    ShareForm,
    SnowflakeQueryForm,
    UpdateValuesForm,
    format_table_text,
    parse_table_text,
)
from .models import SnowflakeQueryLog, TrackedSpreadsheet

User = get_user_model()

# ──────────────────────────────────────────────── test helpers


def _make_user(username: str, is_superuser: bool = False, role: str | None = None):
    """Create a user, optionally assigning a role on the profile."""
    user = User.objects.create_user(
        username=username,
        password="testpass123",
        email=f"{username}@test.com",
    )
    if is_superuser:
        user.is_superuser = True
        user.is_staff = True
        user.save()
    if role and hasattr(user, "profile"):
        user.profile.role = role
        user.profile.save()
    return user


def _make_admin(**kwargs):
    return _make_user("admin_user", role=ROLE_CLIENT_ADMIN, **kwargs)


def _mock_sheet_meta(spreadsheet_id: str = "abc123", title: str = "Test Sheet") -> dict:
    return {
        "spreadsheetId": spreadsheet_id,
        "spreadsheetUrl": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
        "properties": {"title": title},
        "sheets": [
            {"properties": {"sheetId": 0, "title": "Sheet1", "index": 0}},
        ],
    }


# ════════════════════════════════════════ MODEL TESTS

class TrackedSpreadsheetModelTests(TestCase):

    def setUp(self):
        self.user = _make_admin()
        self.client_org = OrgClient.objects.create(name="Test Org", slug="test-org")

    def test_create_minimal(self):
        """Can create a TrackedSpreadsheet with only spreadsheet_id."""
        obj = TrackedSpreadsheet.objects.create(spreadsheet_id="abc123")
        self.assertEqual(obj.spreadsheet_id, "abc123")
        self.assertEqual(str(obj), "abc123")  # __str__ falls back to id

    def test_str_uses_title(self):
        obj = TrackedSpreadsheet.objects.create(
            spreadsheet_id="abc123", title="My Sheet"
        )
        self.assertEqual(str(obj), "My Sheet")

    def test_get_absolute_url(self):
        obj = TrackedSpreadsheet.objects.create(spreadsheet_id="abc123")
        url = obj.get_absolute_url()
        self.assertIn("abc123", url)
        self.assertIn("/warehouse/", url)

    def test_unique_spreadsheet_id(self):
        TrackedSpreadsheet.objects.create(spreadsheet_id="abc123")
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            TrackedSpreadsheet.objects.create(spreadsheet_id="abc123")

    def test_client_fk(self):
        obj = TrackedSpreadsheet.objects.create(
            spreadsheet_id="abc123",
            client=self.client_org,
        )
        self.assertEqual(obj.client, self.client_org)

    def test_created_by_fk(self):
        obj = TrackedSpreadsheet.objects.create(
            spreadsheet_id="abc123",
            created_by=self.user,
        )
        self.assertEqual(obj.created_by, self.user)

    def test_ordering_newest_first(self):
        TrackedSpreadsheet.objects.create(spreadsheet_id="first")
        TrackedSpreadsheet.objects.create(spreadsheet_id="second")
        ids = list(TrackedSpreadsheet.objects.values_list("spreadsheet_id", flat=True))
        self.assertEqual(ids[0], "second")  # -updated_at ordering

    def test_update_or_create(self):
        TrackedSpreadsheet.objects.create(spreadsheet_id="abc123", title="Old")
        TrackedSpreadsheet.objects.update_or_create(
            spreadsheet_id="abc123",
            defaults={"title": "New"},
        )
        obj = TrackedSpreadsheet.objects.get(spreadsheet_id="abc123")
        self.assertEqual(obj.title, "New")


class SnowflakeQueryLogModelTests(TestCase):

    def setUp(self):
        self.user = _make_admin()

    def test_create_pending(self):
        log = SnowflakeQueryLog.objects.create(user=self.user, query="SELECT 1")
        self.assertEqual(log.status, "pending")
        self.assertEqual(log.rows_returned, 0)

    def test_str_format(self):
        log = SnowflakeQueryLog.objects.create(
            user=self.user, query="SELECT 1", status="success"
        )
        self.assertIn("success", str(log))
        self.assertIn(self.user.username, str(log))

    def test_status_choices(self):
        for status_val, _ in SnowflakeQueryLog.STATUS_CHOICES:
            log = SnowflakeQueryLog(
                user=self.user, query="SELECT 1", status=status_val
            )
            self.assertEqual(log.status, status_val)

    def test_error_message(self):
        log = SnowflakeQueryLog.objects.create(
            user=self.user, query="SELECT 1",
            status="error", error_message="Connection refused",
        )
        self.assertEqual(log.error_message, "Connection refused")

    def test_ordering_newest_first(self):
        SnowflakeQueryLog.objects.create(user=self.user, query="SELECT 1")
        SnowflakeQueryLog.objects.create(user=self.user, query="SELECT 2")
        first = SnowflakeQueryLog.objects.first()
        self.assertIn("SELECT 2", first.query)


# ════════════════════════════════════════ VIEW PERMISSION TESTS

class WarehouseViewPermissionTests(TestCase):

    def setUp(self):
        self.anon_client = Client()
        self.regular_user = _make_user("regular")
        self.admin_user = _make_admin()

    def _login(self, user):
        c = Client()
        c.force_login(user)
        return c

    def test_anon_redirect_to_login(self):
        resp = self.anon_client.get(reverse("warehouse:index"))
        self.assertIn(resp.status_code, [302, 403])

    def test_non_admin_forbidden(self):
        c = self._login(self.regular_user)
        resp = c.get(reverse("warehouse:index"))
        # Should redirect (302) to login or return 403
        self.assertIn(resp.status_code, [302, 403])

    def test_admin_can_access_index(self):
        c = self._login(self.admin_user)
        with patch("warehouse.views.SnowflakeClient"):
            resp = c.get(reverse("warehouse:index"))
        self.assertEqual(resp.status_code, 200)

    def test_superuser_can_access_index(self):
        su = _make_user("superuser", is_superuser=True)
        c = self._login(su)
        resp = c.get(reverse("warehouse:index"))
        self.assertEqual(resp.status_code, 200)

    def test_non_admin_cannot_access_snowflake(self):
        c = self._login(self.regular_user)
        resp = c.get(reverse("warehouse:snowflake"))
        self.assertIn(resp.status_code, [302, 403])

    def test_admin_can_access_snowflake(self):
        c = self._login(self.admin_user)
        resp = c.get(reverse("warehouse:snowflake"))
        self.assertEqual(resp.status_code, 200)


# ════════════════════════════════════════ HOME VIEW TESTS

class WarehouseHomeViewTests(TestCase):

    def setUp(self):
        self.user = _make_admin()
        self.c = Client()
        self.c.force_login(self.user)

    def test_get_renders_template(self):
        resp = self.c.get(reverse("warehouse:index"))
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "warehouse/index.html")

    def test_get_has_forms_in_context(self):
        resp = self.c.get(reverse("warehouse:index"))
        self.assertIn("create_form", resp.context)
        self.assertIn("open_form", resp.context)

    def test_get_paginates_recent(self):
        for i in range(15):
            TrackedSpreadsheet.objects.create(spreadsheet_id=f"sid_{i}", title=f"Sheet {i}")
        resp = self.c.get(reverse("warehouse:index"))
        self.assertEqual(resp.status_code, 200)
        self.assertIn("recent_page", resp.context)
        self.assertEqual(len(resp.context["recent_page"].object_list), 10)

    @patch("warehouse.views.get_service")
    def test_post_create_spreadsheet(self, mock_get_service):
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        mock_service.create_spreadsheet.return_value = {
            "spreadsheetId": "new_sid",
            "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/new_sid",
            "properties": {"title": "My New Sheet"},
            "sheets": [{"properties": {"sheetId": 0, "title": "Sheet1", "index": 0}}],
        }

        resp = self.c.post(reverse("warehouse:index"), {
            "action": "create",
            "title": "My New Sheet",
            "sheet_titles": "",
        })

        # Verify redirect (without following — the detail view requires its own mock setup)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("new_sid", resp["Location"])
        self.assertTrue(TrackedSpreadsheet.objects.filter(spreadsheet_id="new_sid").exists())

    @patch("warehouse.views.get_service")
    def test_post_create_api_error(self, mock_get_service):
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        from .sheet_service import SheetsServiceError
        mock_service.create_spreadsheet.side_effect = SheetsServiceError("403 Forbidden")

        resp = self.c.post(reverse("warehouse:index"), {
            "action": "create",
            "title": "Bad Sheet",
            "sheet_titles": "",
        })

        self.assertEqual(resp.status_code, 200)
        # Django messages level 40 = ERROR; tags contain 'danger' because of MESSAGE_TAGS mapping
        msg_list = list(resp.wsgi_request._messages)
        self.assertTrue(len(msg_list) > 0, "Expected at least one message to be set")
        self.assertTrue(
            any(m.level >= 40 or "error" in m.tags or "danger" in m.tags for m in msg_list)
        )

    def test_post_open_redirects_to_detail(self):
        resp = self.c.post(reverse("warehouse:index"), {
            "action": "open",
            "id_or_url": "some_spreadsheet_id",
        })
        self.assertRedirects(
            resp,
            reverse("warehouse:detail", args=["some_spreadsheet_id"]),
        )

    def test_post_open_url_extracts_id(self):
        resp = self.c.post(reverse("warehouse:index"), {
            "action": "open",
            "id_or_url": "https://docs.google.com/spreadsheets/d/extracted_id/edit#gid=0",
        })
        self.assertRedirects(
            resp,
            reverse("warehouse:detail", args=["extracted_id"]),
        )

    def test_search_filters_recent(self):
        TrackedSpreadsheet.objects.create(spreadsheet_id="find_me", title="Special Sheet")
        TrackedSpreadsheet.objects.create(spreadsheet_id="other", title="Other Sheet")
        resp = self.c.get(reverse("warehouse:index"), {"q": "Special"})
        self.assertEqual(resp.context["recent_page"].paginator.count, 1)


# ════════════════════════════════════════ DETAIL VIEW TESTS

class SpreadsheetDetailViewTests(TestCase):

    def setUp(self):
        self.user = _make_admin()
        self.c = Client()
        self.c.force_login(self.user)
        self.spreadsheet_id = "test_sid_123"
        TrackedSpreadsheet.objects.create(
            spreadsheet_id=self.spreadsheet_id,
            title="Test Sheet",
            web_view_link="https://docs.google.com/spreadsheets/d/test_sid_123",
        )

    @patch("warehouse.views.get_service")
    def test_get_renders_detail_template(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        mock_svc.get_spreadsheet.return_value = _mock_sheet_meta(self.spreadsheet_id)
        mock_svc.list_permissions.return_value = []

        resp = self.c.get(reverse("warehouse:detail", args=[self.spreadsheet_id]))
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "warehouse/detail.html")

    @patch("warehouse.views.get_service")
    def test_context_has_all_forms(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        mock_svc.get_spreadsheet.return_value = _mock_sheet_meta(self.spreadsheet_id)
        mock_svc.list_permissions.return_value = []

        resp = self.c.get(reverse("warehouse:detail", args=[self.spreadsheet_id]))
        ctx = resp.context
        for form_key in [
            "read_form", "update_form", "append_form", "clear_form", "batch_form",
            "add_tab_form", "rename_tab_form", "delete_tab_form",
            "format_form", "freeze_form", "delete_rows_form", "insert_rows_form",
            "share_form", "delete_spreadsheet_form",
        ]:
            self.assertIn(form_key, ctx, f"{form_key} missing from context")

    @patch("warehouse.views.get_service")
    def test_api_error_shows_in_context(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        from .sheet_service import SheetsServiceError
        mock_svc.get_spreadsheet.side_effect = SheetsServiceError("Not found")
        mock_svc.list_permissions.return_value = []

        resp = self.c.get(reverse("warehouse:detail", args=["bad_id"]))
        self.assertEqual(resp.status_code, 200)
        self.assertIn("error", resp.context["meta"])


# ════════════════════════════════════════ SNOWFLAKE QUERY VIEW TESTS

class SnowflakeQueryViewTests(TestCase):

    def setUp(self):
        self.user = _make_admin()
        self.c = Client()
        self.c.force_login(self.user)

    def test_get_renders_template(self):
        resp = self.c.get(reverse("warehouse:snowflake"))
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "warehouse/snowflake.html")

    @patch("warehouse.views.SnowflakeClient")
    def test_post_valid_query_returns_results(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.query.return_value = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [30, 42],
        })

        resp = self.c.post(reverse("warehouse:snowflake"), {
            "query": "SELECT name, age FROM patients LIMIT 10",
        })

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context["exec_stats"]["rows_returned"], 2)
        self.assertIsNone(resp.context["error_msg"])

    def test_post_blocked_keyword_rejected(self):
        resp = self.c.post(reverse("warehouse:snowflake"), {
            "query": "DROP TABLE patients",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.context["form"].is_valid())

    @patch("warehouse.views.SnowflakeClient")
    def test_post_creates_query_log(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.query.return_value = pd.DataFrame({"x": [1]})

        self.c.post(reverse("warehouse:snowflake"), {
            "query": "SELECT 1 AS x",
        })
        log = SnowflakeQueryLog.objects.filter(user=self.user).first()
        self.assertIsNotNone(log)
        self.assertEqual(log.status, "success")

    @patch("warehouse.views.SnowflakeClient")
    def test_post_connection_error_logged(self, MockClient):
        from .services.snowflake import SnowflakeQueryError
        mock_client = MockClient.return_value
        mock_client.query.side_effect = SnowflakeQueryError("Connection refused")

        resp = self.c.post(reverse("warehouse:snowflake"), {
            "query": "SELECT 1",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(resp.context["error_msg"])
        log = SnowflakeQueryLog.objects.filter(user=self.user).first()
        self.assertEqual(log.status, "error")


# ════════════════════════════════════════ AJAX VIEWS TESTS

class AjaxViewsTests(TestCase):
    """Test the JSON-returning AJAX views."""

    def setUp(self):
        self.user = _make_admin()
        self.c = Client()
        self.c.force_login(self.user)
        self.sid = "sid_for_ajax"
        TrackedSpreadsheet.objects.create(spreadsheet_id=self.sid, title="AJAX Sheet")

    @patch("warehouse.views.get_service")
    def test_read_values_success(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        mock_svc.read_values.return_value = [["Alice", "30"], ["Bob", "42"]]

        resp = self.c.post(
            reverse("warehouse:read_values", args=[self.sid]),
            {"range_notation": "Sheet1!A1:B10"},
        )
        data = json.loads(resp.content)
        self.assertTrue(data["success"])
        self.assertEqual(data["row_count"], 2)

    @patch("warehouse.views.get_service")
    def test_update_values_success(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        mock_svc.update_values.return_value = {
            "updatedCells": 2, "updatedRange": "Sheet1!A1:B1"
        }

        resp = self.c.post(
            reverse("warehouse:update_values", args=[self.sid]),
            {
                "range_notation": "Sheet1!A1",
                "values": "Alice\t30",
                "value_input_option": "USER_ENTERED",
            },
        )
        data = json.loads(resp.content)
        self.assertTrue(data["success"])

    @patch("warehouse.views.get_service")
    def test_add_tab_success(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        mock_svc.add_sheet.return_value = {}

        resp = self.c.post(
            reverse("warehouse:add_tab", args=[self.sid]),
            {"tab_title": "New Tab"},
        )
        data = json.loads(resp.content)
        self.assertTrue(data["success"])

    @patch("warehouse.views.get_service")
    def test_delete_spreadsheet_success(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        mock_svc.delete_spreadsheet.return_value = None

        resp = self.c.post(
            reverse("warehouse:delete_spreadsheet", args=[self.sid]),
            {"confirm": "on"},
        )
        data = json.loads(resp.content)
        self.assertTrue(data["success"])
        self.assertFalse(
            TrackedSpreadsheet.objects.filter(spreadsheet_id=self.sid).exists()
        )

    def test_ajax_view_rejects_get(self):
        resp = self.c.get(reverse("warehouse:read_values", args=[self.sid]))
        data = json.loads(resp.content)
        self.assertIn("error", data)

    def test_ajax_view_rejects_non_admin(self):
        regular = _make_user("regular2")
        c2 = Client()
        c2.force_login(regular)
        resp = c2.post(
            reverse("warehouse:read_values", args=[self.sid]),
            {"range_notation": "A1:Z100"},
        )
        self.assertEqual(resp.status_code, 403)


# ════════════════════════════════════════ FORMS TESTS

class FormsTests(TestCase):

    # ── parse helpers ────────────────────────────────────────────────

    def test_parse_table_text_tsv(self):
        result = parse_table_text("Alice\t30\nBob\t42")
        self.assertEqual(result, [["Alice", "30"], ["Bob", "42"]])

    def test_parse_table_text_csv(self):
        result = parse_table_text("Alice,30\nBob,42")
        self.assertEqual(result, [["Alice", "30"], ["Bob", "42"]])

    def test_parse_table_text_empty(self):
        self.assertEqual(parse_table_text(""), [])
        self.assertEqual(parse_table_text("   \n\n  "), [])

    def test_format_table_text(self):
        result = format_table_text([["Alice", "30"], ["Bob", None]])
        self.assertEqual(result, "Alice\t30\nBob\t")

    # ── CreateSpreadsheetForm ────────────────────────────────────────

    def test_create_form_valid(self):
        form = CreateSpreadsheetForm({"title": "My Sheet", "sheet_titles": "Tab1, Tab2"})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["sheet_titles"], ["Tab1", "Tab2"])

    def test_create_form_missing_title(self):
        form = CreateSpreadsheetForm({"title": "", "sheet_titles": ""})
        self.assertFalse(form.is_valid())
        self.assertIn("title", form.errors)

    # ── OpenSpreadsheetForm ──────────────────────────────────────────

    def test_open_form_url_extraction(self):
        form = OpenSpreadsheetForm({
            "id_or_url": "https://docs.google.com/spreadsheets/d/EXTRACTED_ID/edit#gid=0"
        })
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["id_or_url"], "EXTRACTED_ID")

    def test_open_form_bare_id(self):
        form = OpenSpreadsheetForm({"id_or_url": "bare_id_here"})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["id_or_url"], "bare_id_here")

    # ── UpdateValuesForm ─────────────────────────────────────────────

    def test_update_form_valid(self):
        form = UpdateValuesForm({
            "range_notation": "Sheet1!A1",
            "values": "Alice\t30",
            "value_input_option": "USER_ENTERED",
        })
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["values"], [["Alice", "30"]])

    def test_update_form_empty_values(self):
        form = UpdateValuesForm({
            "range_notation": "Sheet1!A1",
            "values": "",
            "value_input_option": "USER_ENTERED",
        })
        self.assertFalse(form.is_valid())

    # ── FormatCellsForm ──────────────────────────────────────────────

    def test_format_form_end_row_must_exceed_start(self):
        form = FormatCellsForm({
            "sheet_id": 0,
            "start_row": 5, "end_row": 3,
            "start_col": 0, "end_col": 2,
        })
        self.assertFalse(form.is_valid())
        self.assertIn("end_row", form.errors)

    def test_format_form_invalid_hex(self):
        form = FormatCellsForm({
            "sheet_id": 0,
            "start_row": 0, "end_row": 1,
            "start_col": 0, "end_col": 1,
            "background_hex": "zzz",
        })
        self.assertFalse(form.is_valid())

    def test_format_form_valid_hex(self):
        form = FormatCellsForm({
            "sheet_id": 0,
            "start_row": 0, "end_row": 1,
            "start_col": 0, "end_col": 1,
            "background_hex": "#ff0000",
        })
        self.assertTrue(form.is_valid())

    # ── BatchUpdateForm ──────────────────────────────────────────────

    def test_batch_form_valid(self):
        form = BatchUpdateForm({
            "multi_block": "Sheet1!A1:B2\nAlice,30\nBob,42",
            "value_input_option": "USER_ENTERED",
        })
        self.assertTrue(form.is_valid())
        blocks = form.cleaned_data["multi_block"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["range"], "Sheet1!A1:B2")

    def test_batch_form_multi_block(self):
        form = BatchUpdateForm({
            "multi_block": "Sheet1!A1\nHello\n\nSheet2!B1\nWorld",
            "value_input_option": "RAW",
        })
        self.assertTrue(form.is_valid())
        self.assertEqual(len(form.cleaned_data["multi_block"]), 2)

    def test_batch_form_missing_data_row(self):
        form = BatchUpdateForm({
            "multi_block": "Sheet1!A1",
            "value_input_option": "USER_ENTERED",
        })
        self.assertFalse(form.is_valid())

    # ── DeleteTabForm ────────────────────────────────────────────────

    def test_delete_tab_requires_confirm(self):
        form = DeleteTabForm({"sheet_id": 0})
        self.assertFalse(form.is_valid())
        self.assertIn("confirm", form.errors)

    # ── DeleteRowsForm ───────────────────────────────────────────────

    def test_delete_rows_end_must_exceed_start(self):
        form = DeleteRowsForm({"sheet_id": 0, "start_row": 10, "end_row": 5})
        self.assertFalse(form.is_valid())

    # ── ShareForm ────────────────────────────────────────────────────

    def test_share_form_valid(self):
        form = ShareForm({
            "email": "test@example.com",
            "role": "writer",
            "notify": False,
        })
        self.assertTrue(form.is_valid())

    def test_share_form_invalid_email(self):
        form = ShareForm({"email": "not-an-email", "role": "writer"})
        self.assertFalse(form.is_valid())

    # ── SnowflakeQueryForm ───────────────────────────────────────────

    def test_snowflake_form_blocks_drop(self):
        form = SnowflakeQueryForm({"query": "DROP TABLE patients"})
        self.assertFalse(form.is_valid())
        self.assertIn("query", form.errors)

    def test_snowflake_form_blocks_delete(self):
        form = SnowflakeQueryForm({"query": "DELETE FROM patients WHERE id=1"})
        self.assertFalse(form.is_valid())

    def test_snowflake_form_blocks_truncate(self):
        form = SnowflakeQueryForm({"query": "TRUNCATE TABLE orders"})
        self.assertFalse(form.is_valid())

    def test_snowflake_form_allows_select(self):
        form = SnowflakeQueryForm({"query": "SELECT * FROM patients LIMIT 10"})
        self.assertTrue(form.is_valid())

    def test_snowflake_form_case_insensitive_block(self):
        form = SnowflakeQueryForm({"query": "drop table foo"})
        self.assertFalse(form.is_valid())


# ════════════════════════════════════════ API TESTS

class WarehouseAPITests(TestCase):

    def setUp(self):
        self.user = _make_admin()
        self.regular = _make_user("regular_api")
        self.api_client = APIClient()
        self.api_client.force_authenticate(user=self.user)
        self.org = OrgClient.objects.create(name="API Org", slug="api-org")

    def _url(self, name: str, *args):
        return reverse(name, args=args)

    def test_list_spreadsheets(self):
        TrackedSpreadsheet.objects.create(spreadsheet_id="s1", title="S1", created_by=self.user)
        resp = self.api_client.get("/api/v1/warehouse/spreadsheets/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)

    def test_create_spreadsheet(self):
        resp = self.api_client.post("/api/v1/warehouse/spreadsheets/", {
            "spreadsheet_id": "new_api_sid",
            "title": "API Created Sheet",
        }, format="json")
        self.assertEqual(resp.status_code, 201)
        self.assertTrue(TrackedSpreadsheet.objects.filter(spreadsheet_id="new_api_sid").exists())

    def test_retrieve_spreadsheet(self):
        obj = TrackedSpreadsheet.objects.create(spreadsheet_id="retrieve_sid", title="Retrieve Me")
        resp = self.api_client.get(f"/api/v1/warehouse/spreadsheets/{obj.pk}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["title"], "Retrieve Me")

    def test_destroy_spreadsheet(self):
        obj = TrackedSpreadsheet.objects.create(spreadsheet_id="delete_sid")
        resp = self.api_client.delete(f"/api/v1/warehouse/spreadsheets/{obj.pk}/")
        self.assertEqual(resp.status_code, 204)
        self.assertFalse(TrackedSpreadsheet.objects.filter(pk=obj.pk).exists())

    def test_search_by_title(self):
        TrackedSpreadsheet.objects.create(spreadsheet_id="sa", title="Alpha Report")
        TrackedSpreadsheet.objects.create(spreadsheet_id="sb", title="Beta Report")
        resp = self.api_client.get("/api/v1/warehouse/spreadsheets/?search=Alpha")
        self.assertEqual(resp.data["count"], 1)

    def test_filter_by_client(self):
        TrackedSpreadsheet.objects.create(
            spreadsheet_id="sc", client=self.org, created_by=self.user
        )
        TrackedSpreadsheet.objects.create(spreadsheet_id="sd")
        resp = self.api_client.get(
            f"/api/v1/warehouse/spreadsheets/?client={self.org.pk}"
        )
        self.assertEqual(resp.data["count"], 1)

    def test_non_admin_blocked(self):
        api = APIClient()
        api.force_authenticate(user=self.regular)
        resp = api.get("/api/v1/warehouse/spreadsheets/")
        self.assertEqual(resp.status_code, 403)

    @patch("warehouse.api.get_service")
    def test_values_action(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        mock_svc.read_values.return_value = [["A", "B"], ["1", "2"]]
        obj = TrackedSpreadsheet.objects.create(spreadsheet_id="val_sid")
        resp = self.api_client.get(
            f"/api/v1/warehouse/spreadsheets/{obj.pk}/values/?range=Sheet1!A1:B10"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["row_count"], 2)

    @patch("warehouse.api.get_service")
    def test_share_action(self, mock_get_service):
        mock_svc = MagicMock()
        mock_get_service.return_value = mock_svc
        mock_svc.share.return_value = {"id": "perm123", "emailAddress": "x@y.com", "role": "reader"}
        obj = TrackedSpreadsheet.objects.create(spreadsheet_id="share_sid")
        resp = self.api_client.post(
            f"/api/v1/warehouse/spreadsheets/{obj.pk}/share/",
            {"email": "x@y.com", "role": "reader"},
            format="json",
        )
        self.assertEqual(resp.status_code, 201)


class SnowflakeAPITests(TestCase):

    def setUp(self):
        self.user = _make_admin()
        self.api_client = APIClient()
        self.api_client.force_authenticate(user=self.user)

    @patch("warehouse.api.SnowflakeClient")
    def test_query_api_success(self, MockClient):
        mock_client = MockClient.return_value
        mock_client.query.return_value = pd.DataFrame({"name": ["Alice"], "age": [30]})

        resp = self.api_client.post(
            "/api/v1/warehouse/snowflake/query/",
            {"query": "SELECT name, age FROM patients LIMIT 1"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["row_count"], 1)
        self.assertIn("name", resp.data["columns"])

    def test_query_api_blocks_destructive(self):
        resp = self.api_client.post(
            "/api/v1/warehouse/snowflake/query/",
            {"query": "DELETE FROM patients WHERE 1=1"},
            format="json",
        )
        self.assertIn(resp.status_code, [400, 422])

    @patch("warehouse.api.SnowflakeClient")
    def test_query_api_creates_log(self, MockClient):
        MockClient.return_value.query.return_value = pd.DataFrame({"x": [1]})
        self.api_client.post(
            "/api/v1/warehouse/snowflake/query/",
            {"query": "SELECT 1 AS x"},
            format="json",
        )
        log = SnowflakeQueryLog.objects.filter(user=self.user).first()
        self.assertIsNotNone(log)
        self.assertEqual(log.status, "success")

    @patch("warehouse.api.SnowflakeClient")
    def test_query_api_error_logged(self, MockClient):
        from .services.snowflake import SnowflakeQueryError
        MockClient.return_value.query.side_effect = SnowflakeQueryError("Bad query")
        self.api_client.post(
            "/api/v1/warehouse/snowflake/query/",
            {"query": "SELECT xyz FROM nonexistent"},
            format="json",
        )
        log = SnowflakeQueryLog.objects.filter(user=self.user).first()
        self.assertEqual(log.status, "error")
        self.assertIn("Bad query", log.error_message)

    def test_query_log_list_own_history(self):
        SnowflakeQueryLog.objects.create(user=self.user, query="SELECT 1", status="success")
        other = _make_user("other_user2")
        SnowflakeQueryLog.objects.create(user=other, query="SELECT 2", status="success")
        resp = self.api_client.get("/api/v1/warehouse/snowflake/queries/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)  # only own queries

    @patch("warehouse.api.SnowflakeClient")
    def test_tables_api_success(self, MockClient):
        MockClient.return_value.get_tables.return_value = pd.DataFrame({
            "SCHEMA_NAME": ["PUBLIC"],
            "TABLE_NAME": ["PATIENTS"],
            "TABLE_TYPE": ["BASE TABLE"],
            "ROW_COUNT": [1000],
            "BYTES": [204800],
            "LAST_ALTERED": [None],
        })
        resp = self.api_client.get("/api/v1/warehouse/snowflake/tables/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)
        self.assertEqual(resp.data["tables"][0]["TABLE_NAME"], "PATIENTS")

    @patch("warehouse.api.SnowflakeClient")
    def test_tables_api_error(self, MockClient):
        from .services.snowflake import SnowflakeQueryError
        MockClient.return_value.get_tables.side_effect = SnowflakeQueryError("conn fail")
        resp = self.api_client.get("/api/v1/warehouse/snowflake/tables/")
        self.assertEqual(resp.status_code, 503)


# ════════════════════════════════════════ FACILITY SCOPE TESTS

def _make_facility_user(username: str, facility_name: str, reporting_source_schema: str):
    """Create a user whose profile is linked to a real Facility row.

    Mirrors how agents/facility.py's resolve_facility_from_user() actually
    resolves a canonical key in production: through a real UserProfile →
    Facility relationship, not a mock — so these tests exercise the real
    resolution path end to end, only mocking the Snowflake connection itself.
    """
    from core.models import Client as OrgClient
    from core.models import Facility

    org, _ = OrgClient.objects.get_or_create(name="Facility Scope Org", slug="fs-org")
    facility = Facility.objects.create(
        client=org,
        name=facility_name,
        slug=facility_name.lower().replace(" ", "-"),
        reporting_source_schema=reporting_source_schema,
    )
    user = _make_user(username, role=ROLE_CLIENT_ADMIN)
    user.profile.facility = facility
    user.profile.save(update_fields=["facility"])
    return user


class FacilityScopeUnitTests(TestCase):
    """Pure-function tests for warehouse/services/facility_scope.py."""

    def setUp(self):
        self.kisumu_user = _make_facility_user("kisumu_fac_user", "Kisumu County Hospital", "Kisumu")
        self.client_admin_no_facility = _make_admin()  # no facility linked → unrestricted

    def test_get_scope_none_for_unlinked_facility(self):
        from .services.facility_scope import get_facility_scope
        self.assertIsNone(get_facility_scope(self.client_admin_no_facility))

    def test_get_scope_resolves_kisumu(self):
        from .services.facility_scope import get_facility_scope
        scope = get_facility_scope(self.kisumu_user)
        self.assertIsNotNone(scope)
        self.assertEqual(scope.facility_key, "KISUMU")
        self.assertEqual(scope.clean_schema, "KISUMU_CLEAN")
        self.assertEqual(scope.allowed_schemas, frozenset({"KISUMU_CLEAN"}))

    def test_validate_noop_when_unrestricted(self):
        from .services.facility_scope import validate_query_scope
        # Should not raise, even for a schema/table the user shouldn't otherwise see.
        validate_query_scope('SELECT * FROM "KAKAMEGA_CLEAN"."SOME_TABLE"', None)

    def test_validate_allows_own_clean_schema(self):
        from .services.facility_scope import get_facility_scope, validate_query_scope
        scope = get_facility_scope(self.kisumu_user)
        validate_query_scope('SELECT * FROM "KISUMU_CLEAN"."PATIENTS" LIMIT 10', scope)

    def test_validate_blocks_other_facility_clean_schema(self):
        from .services.facility_scope import FacilityScopeError, get_facility_scope, validate_query_scope
        scope = get_facility_scope(self.kisumu_user)
        with self.assertRaises(FacilityScopeError):
            validate_query_scope('SELECT * FROM "KAKAMEGA_CLEAN"."PATIENTS"', scope)

    def test_validate_blocks_raw_schema(self):
        from .services.facility_scope import FacilityScopeError, get_facility_scope, validate_query_scope
        scope = get_facility_scope(self.kisumu_user)
        with self.assertRaises(FacilityScopeError):
            validate_query_scope('SELECT * FROM "KISUMU_RAW"."PATIENTS"', scope)

    def test_validate_blocks_reporting_schema(self):
        """REPORTING is always blocked for facility-scoped users, filtered or not —
        it pools every facility's rows behind one column, which a raw-SQL
        query can't be reliably forced to filter, so it's off-limits entirely."""
        from .services.facility_scope import FacilityScopeError, get_facility_scope, validate_query_scope
        scope = get_facility_scope(self.kisumu_user)
        with self.assertRaises(FacilityScopeError):
            validate_query_scope("SELECT * FROM REPORTING.RPT_CASE_MIX", scope)
        with self.assertRaises(FacilityScopeError):
            validate_query_scope(
                "SELECT * FROM REPORTING.RPT_CASE_MIX WHERE source_schema ILIKE '%Kisumu%'",
                scope,
            )

    def test_filter_tables_unrestricted_passthrough(self):
        from .services.facility_scope import filter_tables_for_scope
        tables = [{"SCHEMA_NAME": "KISUMU_CLEAN"}, {"SCHEMA_NAME": "STAGING"}]
        self.assertEqual(filter_tables_for_scope(tables, None), tables)

    def test_filter_tables_scoped(self):
        from .services.facility_scope import filter_tables_for_scope, get_facility_scope
        scope = get_facility_scope(self.kisumu_user)
        tables = [
            {"SCHEMA_NAME": "KISUMU_CLEAN", "TABLE_NAME": "A"},
            {"SCHEMA_NAME": "KISUMU_RAW", "TABLE_NAME": "B"},
            {"SCHEMA_NAME": "KAKAMEGA_CLEAN", "TABLE_NAME": "C"},
            {"SCHEMA_NAME": "REPORTING", "TABLE_NAME": "D"},
            {"SCHEMA_NAME": "STAGING", "TABLE_NAME": "E"},
        ]
        result = filter_tables_for_scope(tables, scope)
        schemas = {t["SCHEMA_NAME"] for t in result}
        self.assertEqual(schemas, {"KISUMU_CLEAN"})


class FacilityScopedQueryViewTests(TestCase):
    """SnowflakeQueryView enforcing facility scope for a facility-linked user."""

    def setUp(self):
        self.user = _make_facility_user("kisumu_view_user", "Kisumu County Hospital", "Kisumu")
        self.c = Client()
        self.c.force_login(self.user)

    @patch("warehouse.views.SnowflakeClient")
    def test_own_clean_schema_query_succeeds(self, MockClient):
        MockClient.return_value.query.return_value = pd.DataFrame({"x": [1]})
        resp = self.c.post(reverse("warehouse:snowflake"), {
            "query": 'SELECT * FROM "KISUMU_CLEAN"."SOME_TABLE" LIMIT 10',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.context["error_msg"])

    @patch("warehouse.views.SnowflakeClient")
    def test_other_facility_schema_blocked(self, MockClient):
        resp = self.c.post(reverse("warehouse:snowflake"), {
            "query": 'SELECT * FROM "KAKAMEGA_CLEAN"."SOME_TABLE" LIMIT 10',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(resp.context["error_msg"])
        self.assertIn("KAKAMEGA_CLEAN", resp.context["error_msg"])
        MockClient.return_value.query.assert_not_called()

    @patch("warehouse.views.SnowflakeClient")
    def test_reporting_always_blocked(self, MockClient):
        """REPORTING is off-limits for facility-scoped users, filtered or not."""
        resp = self.c.post(reverse("warehouse:snowflake"), {
            "query": "SELECT * FROM REPORTING.RPT_CASE_MIX LIMIT 10",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(resp.context["error_msg"])
        MockClient.return_value.query.assert_not_called()

        resp2 = self.c.post(reverse("warehouse:snowflake"), {
            "query": "SELECT * FROM REPORTING.RPT_CASE_MIX WHERE source_schema ILIKE '%Kisumu%' LIMIT 10",
        })
        self.assertIsNotNone(resp2.context["error_msg"])
        MockClient.return_value.query.assert_not_called()

    @patch("warehouse.views.SnowflakeClient")
    def test_raw_schema_blocked(self, MockClient):
        resp = self.c.post(reverse("warehouse:snowflake"), {
            "query": 'SELECT * FROM "KISUMU_RAW"."SOME_TABLE" LIMIT 10',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(resp.context["error_msg"])
        MockClient.return_value.query.assert_not_called()

    @patch("warehouse.views.SnowflakeClient")
    def test_unrestricted_admin_unaffected(self, MockClient):
        """A Client Admin with no linked Facility keeps full, unscoped access."""
        MockClient.return_value.query.return_value = pd.DataFrame({"x": [1]})
        unrestricted = _make_user("unrestricted_admin", role=ROLE_CLIENT_ADMIN)
        c = Client()
        c.force_login(unrestricted)
        resp = c.post(reverse("warehouse:snowflake"), {
            "query": 'SELECT * FROM "KAKAMEGA_CLEAN"."SOME_TABLE" LIMIT 10',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.context["error_msg"])


class FacilityScopedAPITests(TestCase):
    """DRF Snowflake endpoints enforcing facility scope."""

    def setUp(self):
        self.user = _make_facility_user("kisumu_api_user", "Kisumu County Hospital", "Kisumu")
        self.api_client = APIClient()
        self.api_client.force_authenticate(user=self.user)

    @patch("warehouse.api.SnowflakeClient")
    def test_query_api_blocks_other_facility(self, MockClient):
        resp = self.api_client.post(
            "/api/v1/warehouse/snowflake/query/",
            {"query": 'SELECT * FROM "KAKAMEGA_CLEAN"."SOME_TABLE"'},
            format="json",
        )
        self.assertEqual(resp.status_code, 403)
        MockClient.return_value.query.assert_not_called()

    @patch("warehouse.api.SnowflakeClient")
    def test_query_api_allows_own_clean_schema(self, MockClient):
        MockClient.return_value.query.return_value = pd.DataFrame({"x": [1]})
        resp = self.api_client.post(
            "/api/v1/warehouse/snowflake/query/",
            {"query": 'SELECT * FROM "KISUMU_CLEAN"."SOME_TABLE"'},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

    @patch("warehouse.api.SnowflakeClient")
    def test_query_api_blocks_unfiltered_reporting(self, MockClient):
        resp = self.api_client.post(
            "/api/v1/warehouse/snowflake/query/",
            {"query": "SELECT * FROM REPORTING.RPT_CASE_MIX"},
            format="json",
        )
        self.assertEqual(resp.status_code, 403)
        MockClient.return_value.query.assert_not_called()

    @patch("warehouse.api.SnowflakeClient")
    def test_tables_api_scopes_to_facility(self, MockClient):
        MockClient.return_value.get_tables.return_value = pd.DataFrame({
            "SCHEMA_NAME": ["KISUMU_CLEAN", "KISUMU_RAW", "KAKAMEGA_CLEAN", "REPORTING"],
            "TABLE_NAME": ["A", "B", "C", "D"],
            "TABLE_TYPE": ["BASE TABLE"] * 4,
            "ROW_COUNT": [1, 2, 3, 4],
            "BYTES": [1, 2, 3, 4],
            "LAST_ALTERED": [None, None, None, None],
        })
        resp = self.api_client.get("/api/v1/warehouse/snowflake/tables/")
        self.assertEqual(resp.status_code, 200)
        schemas = {t["SCHEMA_NAME"] for t in resp.data["tables"]}
        self.assertEqual(schemas, {"KISUMU_CLEAN"})


# ════════════════════════════════════════ SPREADSHEET ANALYST — agent internals
# These exercise warehouse/agent/*.py directly — it has no Django or LLM
# dependency (see its own docstrings), so these are real, unmocked tests that
# confirm the package still works correctly after being relocated from a
# standalone project into warehouse/agent/.

class AnalystSandboxTests(TestCase):
    """warehouse/agent/sandbox.py — AST validation + guarded execution."""

    def test_blocks_import(self):
        from warehouse.agent.sandbox import execute
        result = execute("import os", {})
        self.assertFalse(result.ok)
        self.assertIn("not allowed", result.error)

    def test_blocks_dunder_attribute_access(self):
        from warehouse.agent.sandbox import execute
        # __class__ is deliberately allow-listed (harmless, commonly needed);
        # __subclasses__ is not — this is what should actually be rejected.
        result = execute("int.__subclasses__()", {})
        self.assertFalse(result.ok)

    def test_allows_explicitly_allowed_dunder(self):
        from warehouse.agent.sandbox import execute
        result = execute("(1).__class__.__name__", {})
        self.assertTrue(result.ok)

    def test_blocks_eval_and_exec(self):
        from warehouse.agent.sandbox import execute
        self.assertFalse(execute("eval('1')", {}).ok)
        self.assertFalse(execute("exec('1')", {}).ok)

    def test_allows_basic_arithmetic(self):
        from warehouse.agent.sandbox import execute
        result = execute("2 + 2", {})
        self.assertTrue(result.ok)
        self.assertIn("4", result.value_repr)

    def test_namespace_persists_across_calls(self):
        from warehouse.agent.sandbox import execute
        ns = {}
        execute("x = 41", ns)
        result = execute("x + 1", ns)
        self.assertTrue(result.ok)
        self.assertIn("42", result.value_repr)

    def test_syntax_error_reported_not_raised(self):
        from warehouse.agent.sandbox import execute
        result = execute("def bad(:", {})
        self.assertFalse(result.ok)
        self.assertIn("SyntaxError", result.error)


class AnalystWorkbookTests(TestCase):
    """warehouse/agent/workbook.py — CSV/Excel loading and profiling."""

    def _write_csv(self, name="sample.csv"):
        import tempfile
        from pathlib import Path
        tmp_dir = Path(tempfile.mkdtemp())
        path = tmp_dir / name
        path.write_text("region,revenue\nWest,100\nEast,150\nWest,120\n", encoding="utf-8")
        return path

    def test_load_csv_workbook(self):
        from warehouse.agent.workbook import load_workbook
        path = self._write_csv()
        frames, sheet_to_var = load_workbook(path)
        self.assertEqual(len(frames), 1)
        var = next(iter(frames))
        self.assertEqual(list(frames[var].columns), ["region", "revenue"])
        self.assertEqual(len(frames[var]), 3)

    def test_slugify_sheet(self):
        from warehouse.agent.workbook import slugify_sheet
        self.assertEqual(slugify_sheet("Q1 Sales (2026)"), "q1_sales_2026")
        self.assertEqual(slugify_sheet("2026 Data"), "s_2026_data")

    def test_profile_frame(self):
        import pandas as pd
        from warehouse.agent.workbook import profile_frame
        df = pd.DataFrame({"region": ["West", "East", "West"], "revenue": [100, 150, 120]})
        profile = profile_frame("sales", "sales", df)
        self.assertEqual(profile.n_rows, 3)
        self.assertEqual(profile.n_cols, 2)
        md = profile.to_markdown()
        self.assertIn("sales", md)
        self.assertIn("region", md)

    def test_build_namespace_includes_pandas_and_sheets(self):
        import pandas as pd
        from warehouse.agent.workbook import build_namespace
        frames = {"sales": pd.DataFrame({"x": [1, 2]})}
        ns = build_namespace(frames)
        self.assertIn("pd", ns)
        self.assertIn("np", ns)
        self.assertIn("plt", ns)
        self.assertIn("sales", ns)
        self.assertIn("df", ns)  # alias for the first/only sheet


class AnalysisSessionTests(TestCase):
    """warehouse/agent/session.py — end to end against a real small CSV."""

    def test_open_session_and_overview(self):
        import tempfile
        from pathlib import Path
        from warehouse.agent.session import AnalysisSession

        tmp_dir = Path(tempfile.mkdtemp())
        source = tmp_dir / "sales.csv"
        source.write_text("region,revenue\nWest,100\nEast,150\n", encoding="utf-8")

        session = AnalysisSession.open(source, tmp_dir / "artifacts")
        self.assertIn("region", session.overview())
        self.assertIn("pd", session.namespace)

        path = session.new_artifact_path("my chart!", ".png")
        self.assertTrue(str(path).endswith(".png"))
        artifact = session.record("chart", "My Chart", path)
        self.assertEqual(len(session.artifacts), 1)
        self.assertEqual(artifact.to_dict()["kind"], "chart")


# ════════════════════════════════════════ SPREADSHEET ANALYST — Django layer

class AnalystModelTests(TestCase):

    def setUp(self):
        self.user = _make_user("wb_owner")

    def test_create_workbook(self):
        from warehouse.models import Workbook
        from django.core.files.uploadedfile import SimpleUploadedFile
        wb = Workbook.objects.create(
            owner=self.user,
            file=SimpleUploadedFile("sales.csv", b"a,b\n1,2\n"),
            original_name="sales.csv",
        )
        self.assertEqual(str(wb), "sales.csv")
        self.assertIn(str(wb.id), wb.get_absolute_url())

    def test_conversation_and_chatmessage(self):
        from warehouse.models import ChatMessage, Conversation, Workbook
        from django.core.files.uploadedfile import SimpleUploadedFile
        wb = Workbook.objects.create(
            owner=self.user, file=SimpleUploadedFile("x.csv", b"a\n1\n"), original_name="x.csv",
        )
        conv = Conversation.objects.create(workbook=wb, owner=self.user)
        msg = ChatMessage.objects.create(conversation=conv, role="user", content="Hello")
        self.assertEqual(conv.messages.count(), 1)
        self.assertEqual(str(msg), "user: Hello")


class AnalystFormTests(TestCase):

    def test_upload_form_rejects_bad_extension(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from warehouse.forms import WorkbookUploadForm
        form = WorkbookUploadForm(files={"file": SimpleUploadedFile("virus.exe", b"x")})
        self.assertFalse(form.is_valid())

    def test_upload_form_accepts_csv(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from warehouse.forms import WorkbookUploadForm
        form = WorkbookUploadForm(files={"file": SimpleUploadedFile("sales.csv", b"a,b\n1,2\n")})
        self.assertTrue(form.is_valid())

    def test_question_form_requires_question(self):
        from warehouse.forms import AnalystQuestionForm
        self.assertFalse(AnalystQuestionForm({"question": ""}).is_valid())
        self.assertTrue(AnalystQuestionForm({"question": "What drove growth?"}).is_valid())


class AnalystViewTests(TestCase):

    def setUp(self):
        self.user = _make_user("analyst_view_user")
        self.other_user = _make_user("analyst_other_user")
        self.c = Client()

    def _upload_csv(self, user, filename="sales.csv"):
        from django.core.files.uploadedfile import SimpleUploadedFile
        self.c.force_login(user)
        return self.c.post(reverse("warehouse:analyst_workbook_upload"), {
            "file": SimpleUploadedFile(filename, b"region,revenue\nWest,100\nEast,150\n"),
        })

    def test_workbook_list_requires_login(self):
        resp = self.c.get(reverse("warehouse:analyst_workbook_list"))
        self.assertEqual(resp.status_code, 302)

    def test_workbook_list_shows_only_own_workbooks(self):
        from warehouse.models import Workbook
        self._upload_csv(self.user)
        self._upload_csv(self.other_user)
        self.c.force_login(self.user)
        resp = self.c.get(reverse("warehouse:analyst_workbook_list"))
        self.assertEqual(Workbook.objects.filter(owner=self.user).count(), 1)
        self.assertEqual(len(resp.context["workbooks"]), 1)

    def test_upload_real_csv_creates_workbook_and_redirects_to_chat(self):
        from warehouse.models import Conversation, Workbook
        resp = self._upload_csv(self.user)
        wb = Workbook.objects.get(owner=self.user)
        self.assertEqual(wb.load_error, "")
        self.assertIn("region", wb.overview)
        conv = Conversation.objects.get(workbook=wb)
        self.assertRedirects(resp, conv.get_absolute_url())

    def test_upload_rejects_bad_file_type(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from warehouse.models import Workbook
        self.c.force_login(self.user)
        self.c.post(reverse("warehouse:analyst_workbook_upload"), {
            "file": SimpleUploadedFile("virus.exe", b"x"),
        })
        self.assertFalse(Workbook.objects.filter(owner=self.user).exists())

    def test_workbook_detail_ownership_enforced(self):
        self._upload_csv(self.user)
        from warehouse.models import Workbook
        wb = Workbook.objects.get(owner=self.user)
        self.c.force_login(self.other_user)
        resp = self.c.get(reverse("warehouse:analyst_workbook_detail", args=[wb.id]))
        self.assertEqual(resp.status_code, 404)

    def test_new_conversation_creates_second_conversation(self):
        from warehouse.models import Conversation, Workbook
        self._upload_csv(self.user)
        wb = Workbook.objects.get(owner=self.user)
        self.assertEqual(Conversation.objects.filter(workbook=wb).count(), 1)
        self.c.force_login(self.user)
        self.c.post(reverse("warehouse:analyst_new_conversation", args=[wb.id]))
        self.assertEqual(Conversation.objects.filter(workbook=wb).count(), 2)

    @patch("warehouse.analyst_views.submit_question")
    def test_ask_returns_rendered_message(self, mock_submit):
        from warehouse.models import ChatMessage, Conversation, Workbook
        self._upload_csv(self.user)
        wb = Workbook.objects.get(owner=self.user)
        conv = Conversation.objects.get(workbook=wb)

        mock_submit.return_value = ChatMessage.objects.create(
            conversation=conv, role="assistant", content="Revenue was **$250**.",
        )
        self.c.force_login(self.user)
        resp = self.c.post(reverse("warehouse:analyst_ask", args=[conv.id]), {"question": "Total revenue?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertIn("Revenue", data["html"])

    @patch("warehouse.analyst_views.submit_question")
    def test_ask_requires_a_question(self, mock_submit):
        self._upload_csv(self.user)
        from warehouse.models import Conversation
        conv = Conversation.objects.get(workbook__owner=self.user)
        self.c.force_login(self.user)
        resp = self.c.post(reverse("warehouse:analyst_ask", args=[conv.id]), {"question": ""})
        self.assertEqual(resp.status_code, 400)
        mock_submit.assert_not_called()

    @patch("warehouse.analyst_views.drop_session")
    def test_reset_kernel_drops_session(self, mock_drop):
        self._upload_csv(self.user)
        from warehouse.models import Conversation
        conv = Conversation.objects.get(workbook__owner=self.user)
        self.c.force_login(self.user)
        resp = self.c.post(reverse("warehouse:analyst_reset_kernel", args=[conv.id]))
        self.assertEqual(resp.status_code, 200)
        mock_drop.assert_called_once_with(str(conv.id))

    def test_artifact_download_ownership_enforced(self):
        from warehouse.models import Artifact, Conversation
        from django.core.files.base import ContentFile
        self._upload_csv(self.user)
        conv = Conversation.objects.get(workbook__owner=self.user)
        artifact = Artifact.objects.create(conversation=conv, kind="report", title="Report")
        artifact.file.save("report.md", ContentFile(b"# Report"), save=True)

        self.c.force_login(self.other_user)
        resp = self.c.get(reverse("warehouse:analyst_artifact_download", args=[artifact.pk]))
        self.assertEqual(resp.status_code, 404)

        self.c.force_login(self.user)
        resp = self.c.get(reverse("warehouse:analyst_artifact_download", args=[artifact.pk]))
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# GOOGLE SHEETS LINKING (warehouse/services/google_sheets_import.py)
# =============================================================================

class GoogleSheetsImportServiceTests(TestCase):
    """extract_spreadsheet_id() and fetch_google_sheet_as_xlsx() — no real
    Google API calls, get_service() is mocked."""

    def test_extract_spreadsheet_id_from_full_url(self):
        from warehouse.services.google_sheets_import import extract_spreadsheet_id
        url = "https://docs.google.com/spreadsheets/d/1AbC-XyZ_123/edit#gid=0"
        self.assertEqual(extract_spreadsheet_id(url), "1AbC-XyZ_123")

    def test_extract_spreadsheet_id_passthrough_raw_id(self):
        from warehouse.services.google_sheets_import import extract_spreadsheet_id
        self.assertEqual(extract_spreadsheet_id("1AbC-XyZ_123"), "1AbC-XyZ_123")

    def test_extract_spreadsheet_id_blank_input(self):
        from warehouse.services.google_sheets_import import extract_spreadsheet_id
        self.assertEqual(extract_spreadsheet_id(""), "")
        self.assertEqual(extract_spreadsheet_id(None), "")

    def _mock_service(self, sheets_meta, values_by_tab):
        service = MagicMock()
        service.get_spreadsheet.return_value = {
            "properties": {"title": "My Sheet"},
            "sheets": [{"properties": {"title": t}} for t in sheets_meta],
        }
        service.read_values.side_effect = lambda sid, tab: values_by_tab.get(tab, [])
        return service

    def test_fetch_builds_valid_xlsx_from_single_tab(self):
        from warehouse.services.google_sheets_import import fetch_google_sheet_as_xlsx
        service = self._mock_service(
            ["Sheet1"],
            {"Sheet1": [["region", "revenue"], ["West", "100"], ["East", "150"]]},
        )
        with patch("warehouse.services.google_sheets_import.get_service", return_value=service):
            xlsx_bytes, title = fetch_google_sheet_as_xlsx("abc123")

        self.assertEqual(title, "My Sheet")
        import io
        df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name="Sheet1")
        self.assertEqual(list(df.columns), ["region", "revenue"])
        self.assertEqual(len(df), 2)

    def test_fetch_handles_multiple_tabs(self):
        from warehouse.services.google_sheets_import import fetch_google_sheet_as_xlsx
        service = self._mock_service(
            ["Sales", "Costs"],
            {
                "Sales": [["region", "revenue"], ["West", "100"]],
                "Costs": [["region", "cost"], ["West", "40"]],
            },
        )
        with patch("warehouse.services.google_sheets_import.get_service", return_value=service):
            xlsx_bytes, _ = fetch_google_sheet_as_xlsx("abc123")

        import io
        book = pd.ExcelFile(io.BytesIO(xlsx_bytes))
        self.assertEqual(set(book.sheet_names), {"Sales", "Costs"})

    def test_fetch_pads_short_rows_to_header_width(self):
        from warehouse.services.google_sheets_import import fetch_google_sheet_as_xlsx
        # Sheets omits trailing empty cells -- second row is short.
        service = self._mock_service(
            ["Sheet1"],
            {"Sheet1": [["a", "b", "c"], ["1", "2", "3"], ["4"]]},
        )
        with patch("warehouse.services.google_sheets_import.get_service", return_value=service):
            xlsx_bytes, _ = fetch_google_sheet_as_xlsx("abc123")

        import io
        df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name="Sheet1")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.shape[1], 3)

    def test_fetch_skips_empty_tabs_but_succeeds_if_others_have_data(self):
        from warehouse.services.google_sheets_import import fetch_google_sheet_as_xlsx
        service = self._mock_service(
            ["Empty", "Sheet1"],
            {"Sheet1": [["a"], ["1"]]},
        )
        with patch("warehouse.services.google_sheets_import.get_service", return_value=service):
            xlsx_bytes, _ = fetch_google_sheet_as_xlsx("abc123")

        import io
        book = pd.ExcelFile(io.BytesIO(xlsx_bytes))
        self.assertEqual(book.sheet_names, ["Sheet1"])

    def test_fetch_raises_when_all_tabs_empty(self):
        from warehouse.services.google_sheets_import import (
            GoogleSheetImportError, fetch_google_sheet_as_xlsx,
        )
        service = self._mock_service(["Sheet1"], {"Sheet1": []})
        with patch("warehouse.services.google_sheets_import.get_service", return_value=service):
            with self.assertRaises(GoogleSheetImportError):
                fetch_google_sheet_as_xlsx("abc123")

    def test_fetch_raises_when_no_tabs(self):
        from warehouse.services.google_sheets_import import (
            GoogleSheetImportError, fetch_google_sheet_as_xlsx,
        )
        service = self._mock_service([], {})
        with patch("warehouse.services.google_sheets_import.get_service", return_value=service):
            with self.assertRaises(GoogleSheetImportError):
                fetch_google_sheet_as_xlsx("abc123")

    def test_fetch_raises_on_service_error(self):
        from warehouse.sheet_service import SheetsServiceError
        from warehouse.services.google_sheets_import import (
            GoogleSheetImportError, fetch_google_sheet_as_xlsx,
        )
        service = MagicMock()
        service.get_spreadsheet.side_effect = SheetsServiceError("not shared with service account")
        with patch("warehouse.services.google_sheets_import.get_service", return_value=service):
            with self.assertRaises(GoogleSheetImportError):
                fetch_google_sheet_as_xlsx("abc123")


class GoogleSheetLinkFormTests(TestCase):

    def test_accepts_raw_id(self):
        from warehouse.forms import GoogleSheetLinkForm
        form = GoogleSheetLinkForm({"id_or_url": "1AbC-XyZ_123"})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["id_or_url"], "1AbC-XyZ_123")

    def test_extracts_id_from_full_url(self):
        from warehouse.forms import GoogleSheetLinkForm
        form = GoogleSheetLinkForm({
            "id_or_url": "https://docs.google.com/spreadsheets/d/1AbC-XyZ_123/edit#gid=0"
        })
        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["id_or_url"], "1AbC-XyZ_123")

    def test_rejects_empty_input(self):
        from warehouse.forms import GoogleSheetLinkForm
        form = GoogleSheetLinkForm({"id_or_url": ""})
        self.assertFalse(form.is_valid())


class AnalystGoogleSheetViewTests(TestCase):
    """analyst_link_google_sheet / analyst_refresh_google_sheet — the Google
    API call itself is mocked (fetch_google_sheet_as_xlsx), everything
    downstream (Workbook creation, profiling, chat) runs for real."""

    def setUp(self):
        self.user = _make_user("sheet_view_user")
        self.other_user = _make_user("sheet_other_user")
        self.c = Client()

    def _fake_xlsx(self, columns=("region", "revenue"), rows=(("West", 100), ("East", 150))):
        import io
        buf = io.BytesIO()
        pd.DataFrame(list(rows), columns=list(columns)).to_excel(buf, index=False, sheet_name="Sheet1")
        return buf.getvalue(), "My Linked Sheet"

    def test_link_sheet_requires_login(self):
        resp = self.c.post(reverse("warehouse:analyst_link_google_sheet"), {"id_or_url": "abc123"})
        self.assertEqual(resp.status_code, 302)

    @patch("warehouse.analyst_views.fetch_google_sheet_as_xlsx")
    def test_link_sheet_creates_workbook_and_redirects_to_chat(self, mock_fetch):
        from warehouse.models import Conversation, Workbook
        mock_fetch.return_value = self._fake_xlsx()
        self.c.force_login(self.user)
        resp = self.c.post(reverse("warehouse:analyst_link_google_sheet"), {
            "id_or_url": "https://docs.google.com/spreadsheets/d/1AbC-XyZ_123/edit",
        })

        wb = Workbook.objects.get(owner=self.user)
        self.assertEqual(wb.source_type, Workbook.SOURCE_GOOGLE_SHEET)
        self.assertEqual(wb.google_sheet_id, "1AbC-XyZ_123")
        self.assertEqual(wb.load_error, "")
        self.assertIn("region", wb.overview)
        conv = Conversation.objects.get(workbook=wb)
        self.assertRedirects(resp, conv.get_absolute_url())
        mock_fetch.assert_called_once_with("1AbC-XyZ_123")

    @patch("warehouse.analyst_views.fetch_google_sheet_as_xlsx")
    def test_link_sheet_shows_error_on_import_failure(self, mock_fetch):
        from warehouse.models import Workbook
        from warehouse.services.google_sheets_import import GoogleSheetImportError
        mock_fetch.side_effect = GoogleSheetImportError("not shared with the service account")
        self.c.force_login(self.user)
        resp = self.c.post(reverse("warehouse:analyst_link_google_sheet"), {"id_or_url": "abc123"})

        self.assertRedirects(resp, reverse("warehouse:analyst_workbook_list"))
        self.assertFalse(Workbook.objects.filter(owner=self.user).exists())

    def test_link_sheet_rejects_invalid_form(self):
        from warehouse.models import Workbook
        self.c.force_login(self.user)
        self.c.post(reverse("warehouse:analyst_link_google_sheet"), {"id_or_url": ""})
        self.assertFalse(Workbook.objects.filter(owner=self.user).exists())

    @patch("warehouse.analyst_views.fetch_google_sheet_as_xlsx")
    def test_refresh_requires_ownership(self, mock_fetch):
        from warehouse.models import Workbook
        mock_fetch.return_value = self._fake_xlsx()
        self.c.force_login(self.user)
        self.c.post(reverse("warehouse:analyst_link_google_sheet"), {"id_or_url": "abc123"})
        wb = Workbook.objects.get(owner=self.user)

        self.c.force_login(self.other_user)
        resp = self.c.post(reverse("warehouse:analyst_refresh_google_sheet", args=[wb.id]))
        self.assertEqual(resp.status_code, 404)

    def test_refresh_404_for_non_sheet_workbook(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from warehouse.models import Workbook
        self.c.force_login(self.user)
        self.c.post(reverse("warehouse:analyst_workbook_upload"), {
            "file": SimpleUploadedFile("sales.csv", b"a,b\n1,2\n"),
        })
        wb = Workbook.objects.get(owner=self.user)
        resp = self.c.post(reverse("warehouse:analyst_refresh_google_sheet", args=[wb.id]))
        self.assertEqual(resp.status_code, 404)

    @patch("warehouse.analyst_views.fetch_google_sheet_as_xlsx")
    def test_refresh_updates_file_and_reprofiles(self, mock_fetch):
        from warehouse.models import Workbook
        mock_fetch.return_value = self._fake_xlsx(columns=("a", "b"), rows=(("x", 1),))
        self.c.force_login(self.user)
        self.c.post(reverse("warehouse:analyst_link_google_sheet"), {"id_or_url": "abc123"})
        wb = Workbook.objects.get(owner=self.user)
        old_overview = wb.overview
        self.assertIn("region", old_overview)

        mock_fetch.return_value = self._fake_xlsx(columns=("totally", "different"), rows=(("y", 2),))
        resp = self.c.post(reverse("warehouse:analyst_refresh_google_sheet", args=[wb.id]))
        wb.refresh_from_db()
        self.assertRedirects(resp, wb.get_absolute_url())
        self.assertIn("totally", wb.overview)
        self.assertNotIn("region", wb.overview)
