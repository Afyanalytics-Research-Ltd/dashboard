"""
Tests for the agents module.

All LLM calls, Snowflake connections, email sends, and Twilio calls are mocked
so the suite runs without real credentials or network access.

Test classes:
    SnowflakeToolTests          — query_snowflake, list_available_tables, get_table_sample
    EmailToolTests              — send_email
    WhatsAppToolTests           — send_whatsapp_message (with and without Twilio config)
    AgentStateTests             — state TypedDict and _merge_dicts reducer
    LLMFactoryTests             — get_llm provider priority (Groq → Grok → Claude)
    SupervisorNodeTests         — routing decisions, iteration guard, JSON parsing, final response
    SQLAgentNodeTests           — sql_agent_node invocation and output shape
    MetricsAgentNodeTests       — metrics_agent_node invocation and output shape
    ProcurementAgentNodeTests   — procurement_agent_node invocation and output shape
    OperationsAgentNodeTests    — operations_agent_node invocation and output shape
    CommunicationsAgentNodeTests — communications_agent_node invocation and output shape
    AgentGraphTests             — build_graph, _route_from_supervisor, run_agents
    RunAgentsAPITests           — POST /api/v1/agents/run/ authentication, validation, response

Run with:
    python manage.py test agents
"""

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from authentication.roles import ROLE_CLIENT_ADMIN, ROLE_FACILITY_ADMIN

User = get_user_model()

# ──────────────────────────────────────────────── helpers


def _make_user(username: str, role: str = ROLE_FACILITY_ADMIN, is_superuser: bool = False):
    user = User.objects.create_user(
        username=username,
        password="testpass123",
        email=f"{username}@test.com",
    )
    if is_superuser:
        user.is_superuser = True
        user.is_staff = True
        user.save()
    if hasattr(user, "profile"):
        user.profile.role = role
        user.profile.save()
    return user


def _make_admin():
    return _make_user("admin_agent", role=ROLE_CLIENT_ADMIN)


def _base_state(**overrides):
    """Return a minimal valid AgentState dict for tests."""
    state = {
        "messages": [],
        "task": "Show me current stockout alerts",
        "user_role": "Client Admin",
        "facility": "KSH",
        "next_agent": "",
        "agent_outputs": {},
        "final_response": "",
        "iteration_count": 0,
        "evaluation": None,
    }
    state.update(overrides)
    return state


def _ai_message(content: str):
    from langchain_core.messages import AIMessage
    return AIMessage(content=content)


# ════════════════════════════════════════════════ TOOL TESTS


class SnowflakeToolTests(TestCase):
    """Tests for the three Snowflake LangChain tools."""

    # ── query_snowflake ──────────────────────────────────────────────────────

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_query_snowflake_success(self, MockClient):
        """Returns JSON records on a successful SELECT query."""
        from agents.tools.snowflake_tools import query_snowflake

        mock_instance = MockClient.return_value
        mock_instance.query.return_value = pd.DataFrame({
            "DRUG": ["Amoxicillin", "Paracetamol"],
            "STOCK": [0, 5],
        })

        result = query_snowflake.invoke({"sql": "SELECT DRUG, STOCK FROM PHARMACY LIMIT 10"})
        data = json.loads(result)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["DRUG"], "Amoxicillin")

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_query_snowflake_blocked_keyword_returns_error_json(self, MockClient):
        """Blocked SQL keywords return a JSON error object (not raise)."""
        from agents.tools.snowflake_tools import query_snowflake
        from warehouse.services.snowflake import SnowflakeQueryError

        mock_instance = MockClient.return_value
        mock_instance.query.side_effect = SnowflakeQueryError(
            "The keyword 'DROP' is not permitted."
        )

        result = query_snowflake.invoke({"sql": "DROP TABLE pharmacy"})
        data = json.loads(result)

        self.assertIn("error", data)
        self.assertIn("DROP", data["error"])

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_query_snowflake_unexpected_exception_returns_error_json(self, MockClient):
        """Unexpected exceptions are caught and returned as JSON error."""
        from agents.tools.snowflake_tools import query_snowflake

        mock_instance = MockClient.return_value
        mock_instance.query.side_effect = RuntimeError("Connection timed out")

        result = query_snowflake.invoke({"sql": "SELECT 1"})
        data = json.loads(result)

        self.assertIn("error", data)
        self.assertIn("timed out", data["error"])

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_query_snowflake_empty_result(self, MockClient):
        """Empty DataFrame returns an empty JSON array."""
        from agents.tools.snowflake_tools import query_snowflake

        MockClient.return_value.query.return_value = pd.DataFrame(columns=["COL"])
        result = query_snowflake.invoke({"sql": "SELECT COL FROM T WHERE 1=0"})
        self.assertEqual(json.loads(result), [])

    # ── list_available_tables ────────────────────────────────────────────────

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_list_available_tables_success(self, MockClient):
        """Returns JSON with table metadata columns."""
        from agents.tools.snowflake_tools import list_available_tables

        MockClient.return_value.get_tables.return_value = pd.DataFrame({
            "SCHEMA_NAME": ["XANALIFE_CLEAN"],
            "TABLE_NAME": ["PHARMACY_TRANSACTIONS"],
            "ROW_COUNT": [50000],
        })

        result = list_available_tables.invoke({})
        data = json.loads(result)

        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["TABLE_NAME"], "PHARMACY_TRANSACTIONS")
        self.assertEqual(data[0]["ROW_COUNT"], 50000)

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_list_available_tables_error(self, MockClient):
        """Connection errors return a JSON error object."""
        from agents.tools.snowflake_tools import list_available_tables

        MockClient.return_value.get_tables.side_effect = Exception("Auth failed")

        result = list_available_tables.invoke({})
        data = json.loads(result)

        self.assertIn("error", data)

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_list_available_tables_missing_columns_handled(self, MockClient):
        """Works even if ROW_COUNT column is absent from the result."""
        from agents.tools.snowflake_tools import list_available_tables

        MockClient.return_value.get_tables.return_value = pd.DataFrame({
            "SCHEMA_NAME": ["XANALIFE_CLEAN"],
            "TABLE_NAME": ["PATIENTS"],
            # ROW_COUNT deliberately omitted
        })

        result = list_available_tables.invoke({})
        data = json.loads(result)

        self.assertEqual(data[0]["TABLE_NAME"], "PATIENTS")

    # ── get_table_sample ─────────────────────────────────────────────────────

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_get_table_sample_success(self, MockClient):
        """Returns JSON with sample rows from the requested table."""
        from agents.tools.snowflake_tools import get_table_sample

        MockClient.return_value.get_table_sample.return_value = pd.DataFrame({
            "ID": [1, 2],
            "NAME": ["Alice", "Bob"],
        })

        result = get_table_sample.invoke({"schema": "XANALIFE_CLEAN", "table": "PATIENTS"})
        data = json.loads(result)

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["NAME"], "Alice")

    @patch("warehouse.services.snowflake.SnowflakeClient")
    def test_get_table_sample_error(self, MockClient):
        """Returns JSON error on failure."""
        from agents.tools.snowflake_tools import get_table_sample

        MockClient.return_value.get_table_sample.side_effect = Exception("Table not found")

        result = get_table_sample.invoke({"schema": "BAD", "table": "MISSING"})
        data = json.loads(result)

        self.assertIn("error", data)


# ════════════════════════════════════════════════ EMAIL TOOL TESTS


class EmailToolTests(TestCase):
    """Tests for the send_email LangChain tool."""

    @patch("django.core.mail.send_mail")
    def test_send_email_success(self, mock_send):
        """Returns confirmation string on successful send."""
        from agents.tools.email_tools import send_email

        mock_send.return_value = 1

        result = send_email.invoke({
            "recipient_email": "manager@hospital.co.ke",
            "subject": "Stockout Alert — Amoxicillin",
            "body": "Stock has reached zero. Please reorder immediately.",
        })

        self.assertIn("manager@hospital.co.ke", result)
        self.assertIn("sent", result.lower())
        mock_send.assert_called_once()

    @patch("django.core.mail.send_mail")
    def test_send_email_smtp_failure_returns_error_string(self, mock_send):
        """SMTP failures return an error string, not an exception."""
        from agents.tools.email_tools import send_email

        mock_send.side_effect = Exception("SMTP connection refused")

        result = send_email.invoke({
            "recipient_email": "manager@hospital.co.ke",
            "subject": "Test",
            "body": "Test body",
        })

        self.assertIn("Failed", result)
        self.assertIn("manager@hospital.co.ke", result)

    @patch("django.core.mail.send_mail")
    def test_send_email_passes_correct_args(self, mock_send):
        """Passes subject, body, and recipient to Django's send_mail."""
        from agents.tools.email_tools import send_email

        mock_send.return_value = 1

        send_email.invoke({
            "recipient_email": "ceo@afya.ai",
            "subject": "Monthly KPI Report",
            "body": "Revenue is up 12% this month.",
        })

        call_kwargs = mock_send.call_args
        self.assertIn("ceo@afya.ai", call_kwargs[1]["recipient_list"])
        self.assertEqual(call_kwargs[1]["subject"], "Monthly KPI Report")


# ════════════════════════════════════════════════ WHATSAPP TOOL TESTS


class WhatsAppToolTests(TestCase):
    """Tests for the send_whatsapp_message LangChain tool."""

    def test_send_whatsapp_stub_when_no_twilio_config(self):
        """Returns a stub message when Twilio credentials are absent."""
        from agents.tools.whatsapp_tools import send_whatsapp_message

        # Ensure no Twilio env vars are set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TWILIO_ACCOUNT_SID", None)
            os.environ.pop("TWILIO_AUTH_TOKEN", None)
            os.environ.pop("TWILIO_WHATSAPP_FROM", None)

            result = send_whatsapp_message.invoke({
                "phone_number": "+254700000001",
                "message": "URGENT: Paracetamol stockout at KSH pharmacy.",
            })

        self.assertIn("stub", result.lower())
        self.assertIn("+254700000001", result)

    @patch.dict(os.environ, {
        "TWILIO_ACCOUNT_SID": "ACTEST",
        "TWILIO_AUTH_TOKEN": "test_token",
        "TWILIO_WHATSAPP_FROM": "whatsapp:+14155238886",
    })
    @patch("agents.tools.whatsapp_tools.Client")
    def test_send_whatsapp_via_twilio(self, MockTwilioClient):
        """Calls Twilio API and returns SID confirmation when credentials exist."""
        from agents.tools.whatsapp_tools import send_whatsapp_message

        mock_client_instance = MockTwilioClient.return_value
        mock_message = MagicMock()
        mock_message.sid = "SM_TEST_SID_12345"
        mock_client_instance.messages.create.return_value = mock_message

        result = send_whatsapp_message.invoke({
            "phone_number": "+254700000001",
            "message": "Stock alert: Amoxicillin below safety level.",
        })

        self.assertIn("SM_TEST_SID_12345", result)
        self.assertIn("+254700000001", result)

    @patch.dict(os.environ, {
        "TWILIO_ACCOUNT_SID": "ACTEST",
        "TWILIO_AUTH_TOKEN": "test_token",
        "TWILIO_WHATSAPP_FROM": "whatsapp:+14155238886",
    })
    @patch("agents.tools.whatsapp_tools.Client")
    def test_send_whatsapp_twilio_failure_returns_error(self, MockTwilioClient):
        """Twilio API failures return an error string."""
        from agents.tools.whatsapp_tools import send_whatsapp_message

        mock_client_instance = MockTwilioClient.return_value
        mock_client_instance.messages.create.side_effect = Exception("Auth error")

        result = send_whatsapp_message.invoke({
            "phone_number": "+254700000001",
            "message": "Test",
        })

        self.assertIn("Failed", result)

    @patch.dict(os.environ, {
        "TWILIO_ACCOUNT_SID": "ACTEST",
        "TWILIO_AUTH_TOKEN": "test_token",
        "TWILIO_WHATSAPP_FROM": "whatsapp:+14155238886",
    })
    @patch("agents.tools.whatsapp_tools.Client")
    def test_send_whatsapp_truncates_long_message(self, MockTwilioClient):
        """Messages are truncated to 1600 characters before sending."""
        from agents.tools.whatsapp_tools import send_whatsapp_message

        mock_client_instance = MockTwilioClient.return_value
        mock_message = MagicMock()
        mock_message.sid = "SM_XYZ"
        mock_client_instance.messages.create.return_value = mock_message

        long_message = "A" * 2000
        send_whatsapp_message.invoke({
            "phone_number": "+254700000001",
            "message": long_message,
        })

        sent_body = mock_client_instance.messages.create.call_args[1]["body"]
        self.assertLessEqual(len(sent_body), 1600)


# ════════════════════════════════════════════════ STATE TESTS


class AgentStateTests(TestCase):
    """Tests for the AgentState TypedDict and _merge_dicts reducer."""

    def test_merge_dicts_combines_two_dicts(self):
        """_merge_dicts combines two dicts with b taking precedence."""
        from agents.state import _merge_dicts

        result = _merge_dicts({"a": 1, "b": 2}, {"b": 99, "c": 3})
        self.assertEqual(result, {"a": 1, "b": 99, "c": 3})

    def test_merge_dicts_empty_first(self):
        from agents.state import _merge_dicts

        result = _merge_dicts({}, {"sql_agent": "Found 10 rows."})
        self.assertEqual(result, {"sql_agent": "Found 10 rows."})

    def test_merge_dicts_empty_second(self):
        from agents.state import _merge_dicts

        result = _merge_dicts({"sql_agent": "data"}, {})
        self.assertEqual(result, {"sql_agent": "data"})

    def test_merge_dicts_accumulates_across_multiple_calls(self):
        """Simulates outputs accumulating across multiple agent invocations."""
        from agents.state import _merge_dicts

        accumulated = {}
        accumulated = _merge_dicts(accumulated, {"sql_agent": "sql output"})
        accumulated = _merge_dicts(accumulated, {"metrics_agent": "metrics output"})
        accumulated = _merge_dicts(accumulated, {"communications_agent": "sent"})

        self.assertEqual(len(accumulated), 3)
        self.assertIn("sql_agent", accumulated)
        self.assertIn("metrics_agent", accumulated)

    def test_base_state_has_required_keys(self):
        """_base_state helper returns all AgentState keys."""
        state = _base_state()
        required = {
            "messages", "task", "user_role", "facility",
            "next_agent", "agent_outputs", "final_response",
            "iteration_count", "evaluation",
        }
        self.assertTrue(required.issubset(state.keys()))


# ════════════════════════════════════════════════ LLM FACTORY TESTS


class LLMFactoryTests(TestCase):
    """Tests for the get_llm provider selection logic."""

    @patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}, clear=False)
    def test_groq_key_returns_openai_client_pointing_to_groq(self):
        """When GROQ_API_KEY is set, returns a ChatOpenAI configured for Groq."""
        # Remove other keys to ensure priority
        env = {"GROQ_API_KEY": "gsk_test"}
        with patch.dict(os.environ, env):
            os.environ.pop("XAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with patch("agents.llm.ChatOpenAI") as MockOpenAI:
                from importlib import reload
                import agents.llm
                reload(agents.llm)
                agents.llm.get_llm()
            MockOpenAI.assert_called_once()
            call_kwargs = MockOpenAI.call_args[1]
            self.assertIn("groq.com", call_kwargs.get("base_url", ""))

    @patch.dict(os.environ, {"XAI_API_KEY": "xai_test"}, clear=False)
    def test_xai_key_returns_openai_client_pointing_to_xai(self):
        """When XAI_API_KEY is set (and no GROQ key), returns ChatOpenAI for xAI."""
        with patch.dict(os.environ, {"XAI_API_KEY": "xai_test"}):
            os.environ.pop("GROQ_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with patch("agents.llm.ChatOpenAI") as MockOpenAI:
                from importlib import reload
                import agents.llm
                reload(agents.llm)
                agents.llm.get_llm()
            MockOpenAI.assert_called_once()
            call_kwargs = MockOpenAI.call_args[1]
            self.assertIn("x.ai", call_kwargs.get("base_url", ""))

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
    def test_anthropic_key_returns_chat_anthropic(self):
        """When only ANTHROPIC_API_KEY is set, returns a ChatAnthropic instance."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            os.environ.pop("GROQ_API_KEY", None)
            os.environ.pop("XAI_API_KEY", None)
            with patch("agents.llm.ChatAnthropic") as MockAnthropic:
                from importlib import reload
                import agents.llm
                reload(agents.llm)
                agents.llm.get_llm()
            MockAnthropic.assert_called_once()

    def test_no_key_raises_runtime_error(self):
        """Raises RuntimeError when no LLM API key is configured."""
        with patch.dict(os.environ, {}):
            os.environ.pop("GROQ_API_KEY", None)
            os.environ.pop("XAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            from importlib import reload
            import agents.llm
            reload(agents.llm)
            with self.assertRaises(RuntimeError):
                agents.llm.get_llm()

    def test_groq_takes_priority_over_xai(self):
        """Groq is selected when both GROQ and XAI keys are present."""
        env = {"GROQ_API_KEY": "gsk_test", "XAI_API_KEY": "xai_test"}
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env):
            with patch("agents.llm.ChatOpenAI") as MockOpenAI:
                from importlib import reload
                import agents.llm
                reload(agents.llm)
                agents.llm.get_llm()
            call_kwargs = MockOpenAI.call_args[1]
            self.assertIn("groq.com", call_kwargs.get("base_url", ""))


# ════════════════════════════════════════════════ SUPERVISOR NODE TESTS


class SupervisorNodeTests(TestCase):
    """Tests for supervisor_node routing, evaluation, and fallback logic."""

    def _mock_llm_response(self, next_agent: str, reasoning: str = "test", evaluation: str = "N/A"):
        """Return a MagicMock LLM that yields a routing JSON decision."""
        mock_llm = MagicMock()
        payload = json.dumps({
            "next_agent": next_agent,
            "reasoning": reasoning,
            "evaluation": evaluation,
        })
        mock_llm.invoke.return_value = MagicMock(content=payload)
        return mock_llm

    @patch("agents.nodes.supervisor.get_llm")
    def test_routes_to_sql_agent_on_first_call(self, mock_get_llm):
        """Routes to sql_agent when supervisor decides so on the first call."""
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response("sql_agent")
        result = supervisor_node(_base_state())

        self.assertEqual(result["next_agent"], "sql_agent")

    @patch("agents.nodes.supervisor.get_llm")
    def test_routes_to_metrics_agent(self, mock_get_llm):
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response("metrics_agent")
        result = supervisor_node(_base_state())

        self.assertEqual(result["next_agent"], "metrics_agent")

    @patch("agents.nodes.supervisor.get_llm")
    def test_routes_to_procurement_agent(self, mock_get_llm):
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response("procurement_agent")
        result = supervisor_node(_base_state())

        self.assertEqual(result["next_agent"], "procurement_agent")

    @patch("agents.nodes.supervisor.get_llm")
    def test_routes_to_operations_agent(self, mock_get_llm):
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response("operations_agent")
        result = supervisor_node(_base_state())

        self.assertEqual(result["next_agent"], "operations_agent")

    @patch("agents.nodes.supervisor.get_llm")
    def test_routes_to_communications_agent(self, mock_get_llm):
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response("communications_agent")
        result = supervisor_node(_base_state())

        self.assertEqual(result["next_agent"], "communications_agent")

    @patch("agents.nodes.supervisor.get_llm")
    def test_routes_to_finish_and_sets_final_response(self, mock_get_llm):
        """When next_agent is FINISH, final_response is populated."""
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response("FINISH", evaluation="Good output.")
        state = _base_state(agent_outputs={"sql_agent": "10 rows found."})
        result = supervisor_node(state)

        self.assertEqual(result["next_agent"], "FINISH")
        self.assertIn("final_response", result)
        self.assertIn("10 rows found.", result["final_response"])

    @patch("agents.nodes.supervisor.get_llm")
    def test_unknown_agent_falls_back_to_finish(self, mock_get_llm):
        """Unknown agent names from LLM are normalised to FINISH."""
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response("made_up_agent_xyz")
        result = supervisor_node(_base_state())

        self.assertEqual(result["next_agent"], "FINISH")

    @patch("agents.nodes.supervisor.get_llm")
    def test_llm_error_defaults_to_sql_agent_on_first_call(self, mock_get_llm):
        """When LLM raises an exception and no outputs exist, defaults to sql_agent."""
        from agents.nodes.supervisor import supervisor_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        mock_get_llm.return_value = mock_llm

        result = supervisor_node(_base_state(agent_outputs={}))
        self.assertEqual(result["next_agent"], "sql_agent")

    @patch("agents.nodes.supervisor.get_llm")
    def test_llm_error_defaults_to_finish_when_outputs_exist(self, mock_get_llm):
        """When LLM errors and agent outputs already exist, defaults to FINISH."""
        from agents.nodes.supervisor import supervisor_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        mock_get_llm.return_value = mock_llm

        result = supervisor_node(_base_state(agent_outputs={"sql_agent": "some data"}))
        self.assertEqual(result["next_agent"], "FINISH")

    @patch("agents.nodes.supervisor.get_llm")
    def test_iteration_count_increments(self, mock_get_llm):
        """Each supervisor call increments iteration_count by 1."""
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response("sql_agent")
        result = supervisor_node(_base_state(iteration_count=2))

        self.assertEqual(result["iteration_count"], 3)

    @patch("agents.nodes.supervisor.get_llm")
    def test_max_iterations_forces_finish(self, mock_get_llm):
        """At iteration_count >= 5, supervisor forces FINISH without calling LLM."""
        from agents.nodes.supervisor import supervisor_node

        state = _base_state(iteration_count=5, agent_outputs={"sql_agent": "data"})
        result = supervisor_node(state)

        self.assertEqual(result["next_agent"], "FINISH")
        mock_get_llm.assert_not_called()

    @patch("agents.nodes.supervisor.get_llm")
    def test_parses_json_wrapped_in_markdown_fences(self, mock_get_llm):
        """Handles LLM responses wrapped in ```json ... ``` code fences."""
        from agents.nodes.supervisor import supervisor_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=(
            "```json\n"
            '{"next_agent": "metrics_agent", "reasoning": "needs KPIs", "evaluation": "N/A"}'
            "\n```"
        ))
        mock_get_llm.return_value = mock_llm

        result = supervisor_node(_base_state())
        self.assertEqual(result["next_agent"], "metrics_agent")

    @patch("agents.nodes.supervisor.get_llm")
    def test_evaluation_stored_in_result(self, mock_get_llm):
        """Supervisor stores the evaluation field from LLM response."""
        from agents.nodes.supervisor import supervisor_node

        mock_get_llm.return_value = self._mock_llm_response(
            "FINISH", evaluation="Output quality is high."
        )
        result = supervisor_node(_base_state(agent_outputs={"sql_agent": "ok"}))

        self.assertEqual(result["evaluation"], "Output quality is high.")

    def test_build_final_response_with_outputs(self):
        """_build_final_response returns all agent output blocks."""
        from agents.nodes.supervisor import _build_final_response

        state = _base_state(
            task="Show stockouts",
            agent_outputs={
                "sql_agent": "Zero stock: Amoxicillin",
                "metrics_agent": "KPI summary here",
            },
            evaluation="Looks complete.",
        )
        response = _build_final_response(state)

        self.assertIn("Show stockouts", response)
        self.assertIn("Zero stock: Amoxicillin", response)
        self.assertIn("KPI summary here", response)

    def test_build_final_response_empty_outputs(self):
        """_build_final_response handles the case where no outputs exist."""
        from agents.nodes.supervisor import _build_final_response

        state = _base_state(agent_outputs={})
        response = _build_final_response(state)

        self.assertIn("No results", response)


# ════════════════════════════════════════════════ EXECUTION AGENT NODE TESTS


class SQLAgentNodeTests(TestCase):
    """Tests for sql_agent_node."""

    @patch("agents.nodes.sql_agent.create_react_agent")
    @patch("agents.nodes.sql_agent.get_llm")
    def test_returns_agent_output_in_correct_key(self, mock_get_llm, mock_create_agent):
        """sql_agent_node stores result under agent_outputs['sql_agent']."""
        from agents.nodes.sql_agent import sql_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [_ai_message("Found 5 stockout items.")]
        }
        mock_create_agent.return_value = mock_agent

        result = sql_agent_node(_base_state())

        self.assertIn("sql_agent", result["agent_outputs"])
        self.assertEqual(result["agent_outputs"]["sql_agent"], "Found 5 stockout items.")

    @patch("agents.nodes.sql_agent.create_react_agent")
    @patch("agents.nodes.sql_agent.get_llm")
    def test_uses_temperature_zero(self, mock_get_llm, mock_create_agent):
        """SQL agent uses temperature=0.0 for deterministic queries."""
        from agents.nodes.sql_agent import sql_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        sql_agent_node(_base_state())

        mock_get_llm.assert_called_once_with(temperature=0.0)

    @patch("agents.nodes.sql_agent.create_react_agent")
    @patch("agents.nodes.sql_agent.get_llm")
    def test_messages_appended_to_state(self, mock_get_llm, mock_create_agent):
        """Messages returned by the inner agent are included in state updates."""
        from agents.nodes.sql_agent import sql_agent_node

        inner_msg = _ai_message("Result data here.")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [inner_msg]}
        mock_create_agent.return_value = mock_agent

        result = sql_agent_node(_base_state())

        self.assertIn("messages", result)
        self.assertEqual(result["messages"], [inner_msg])

    @patch("agents.nodes.sql_agent.create_react_agent")
    @patch("agents.nodes.sql_agent.get_llm")
    def test_passes_snowflake_tools_to_agent(self, mock_get_llm, mock_create_agent):
        """create_react_agent is called with the three Snowflake tools."""
        from agents.nodes.sql_agent import sql_agent_node, _TOOLS

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        sql_agent_node(_base_state())

        _, tools_arg, _ = mock_create_agent.call_args[0]
        self.assertEqual(len(tools_arg), 3)


class MetricsAgentNodeTests(TestCase):
    """Tests for metrics_agent_node."""

    @patch("agents.nodes.metrics_agent.create_react_agent")
    @patch("agents.nodes.metrics_agent.get_llm")
    def test_returns_agent_output_in_correct_key(self, mock_get_llm, mock_create_agent):
        from agents.nodes.metrics_agent import metrics_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [_ai_message("Revenue up 12%. Bed occupancy 78%.")]
        }
        mock_create_agent.return_value = mock_agent

        result = metrics_agent_node(_base_state())

        self.assertIn("metrics_agent", result["agent_outputs"])
        self.assertIn("Revenue up 12%", result["agent_outputs"]["metrics_agent"])

    @patch("agents.nodes.metrics_agent.create_react_agent")
    @patch("agents.nodes.metrics_agent.get_llm")
    def test_prompt_includes_user_role_and_facility(self, mock_get_llm, mock_create_agent):
        """Prompt sent to inner agent includes user role and facility context."""
        from agents.nodes.metrics_agent import metrics_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        metrics_agent_node(_base_state(user_role="Facilities Admin", facility="Lodwar"))

        invoke_call = mock_agent.invoke.call_args[0][0]
        prompt_text = invoke_call["messages"][0].content
        self.assertIn("Facilities Admin", prompt_text)
        self.assertIn("Lodwar", prompt_text)

    @patch("agents.nodes.metrics_agent.create_react_agent")
    @patch("agents.nodes.metrics_agent.get_llm")
    def test_passes_two_snowflake_tools(self, mock_get_llm, mock_create_agent):
        """Metrics agent uses query_snowflake and list_available_tables."""
        from agents.nodes.metrics_agent import metrics_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        metrics_agent_node(_base_state())

        _, tools_arg, _ = mock_create_agent.call_args[0]
        self.assertEqual(len(tools_arg), 2)


class ProcurementAgentNodeTests(TestCase):
    """Tests for procurement_agent_node."""

    @patch("agents.nodes.procurement_agent.create_react_agent")
    @patch("agents.nodes.procurement_agent.get_llm")
    def test_returns_agent_output_in_correct_key(self, mock_get_llm, mock_create_agent):
        from agents.nodes.procurement_agent import procurement_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [_ai_message("PO #1234 submitted for Amoxicillin.")]
        }
        mock_create_agent.return_value = mock_agent

        result = procurement_agent_node(_base_state())

        self.assertIn("procurement_agent", result["agent_outputs"])
        self.assertIn("PO #1234", result["agent_outputs"]["procurement_agent"])

    @patch("agents.nodes.procurement_agent.create_react_agent")
    @patch("agents.nodes.procurement_agent.get_llm")
    def test_passes_three_tools(self, mock_get_llm, mock_create_agent):
        """Procurement agent has: query_snowflake, list_available_tables, send_email."""
        from agents.nodes.procurement_agent import procurement_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        procurement_agent_node(_base_state())

        _, tools_arg, _ = mock_create_agent.call_args[0]
        self.assertEqual(len(tools_arg), 3)

    @patch("agents.nodes.procurement_agent.create_react_agent")
    @patch("agents.nodes.procurement_agent.get_llm")
    def test_prompt_includes_facility(self, mock_get_llm, mock_create_agent):
        from agents.nodes.procurement_agent import procurement_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        procurement_agent_node(_base_state(facility="Kakamega"))

        prompt_text = mock_agent.invoke.call_args[0][0]["messages"][0].content
        self.assertIn("Kakamega", prompt_text)


class OperationsAgentNodeTests(TestCase):
    """Tests for operations_agent_node."""

    @patch("agents.nodes.operations_agent.create_react_agent")
    @patch("agents.nodes.operations_agent.get_llm")
    def test_returns_agent_output_in_correct_key(self, mock_get_llm, mock_create_agent):
        from agents.nodes.operations_agent import operations_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [_ai_message("Bed occupancy: 82%. Theatre: 60% utilised.")]
        }
        mock_create_agent.return_value = mock_agent

        result = operations_agent_node(_base_state())

        self.assertIn("operations_agent", result["agent_outputs"])
        self.assertIn("Bed occupancy", result["agent_outputs"]["operations_agent"])

    @patch("agents.nodes.operations_agent.create_react_agent")
    @patch("agents.nodes.operations_agent.get_llm")
    def test_passes_two_snowflake_tools(self, mock_get_llm, mock_create_agent):
        """Operations agent uses query_snowflake and list_available_tables only."""
        from agents.nodes.operations_agent import operations_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        operations_agent_node(_base_state())

        _, tools_arg, _ = mock_create_agent.call_args[0]
        self.assertEqual(len(tools_arg), 2)

    @patch("agents.nodes.operations_agent.create_react_agent")
    @patch("agents.nodes.operations_agent.get_llm")
    def test_prompt_includes_task_and_role(self, mock_get_llm, mock_create_agent):
        from agents.nodes.operations_agent import operations_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        operations_agent_node(_base_state(
            task="Check theatre utilisation",
            user_role="Facility Admin",
        ))

        prompt_text = mock_agent.invoke.call_args[0][0]["messages"][0].content
        self.assertIn("theatre utilisation", prompt_text)
        self.assertIn("Facility Admin", prompt_text)


class CommunicationsAgentNodeTests(TestCase):
    """Tests for communications_agent_node."""

    @patch("agents.nodes.communications_agent.create_react_agent")
    @patch("agents.nodes.communications_agent.get_llm")
    def test_returns_agent_output_in_correct_key(self, mock_get_llm, mock_create_agent):
        from agents.nodes.communications_agent import communications_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [_ai_message("Email sent to manager@ksh.co.ke. WhatsApp sent to +25470000.")]
        }
        mock_create_agent.return_value = mock_agent

        result = communications_agent_node(_base_state())

        self.assertIn("communications_agent", result["agent_outputs"])
        self.assertIn("Email sent", result["agent_outputs"]["communications_agent"])

    @patch("agents.nodes.communications_agent.create_react_agent")
    @patch("agents.nodes.communications_agent.get_llm")
    def test_passes_two_communication_tools(self, mock_get_llm, mock_create_agent):
        """Communications agent uses send_email and send_whatsapp_message."""
        from agents.nodes.communications_agent import communications_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("ok")]}
        mock_create_agent.return_value = mock_agent

        communications_agent_node(_base_state())

        _, tools_arg, _ = mock_create_agent.call_args[0]
        self.assertEqual(len(tools_arg), 2)

    @patch("agents.nodes.communications_agent.create_react_agent")
    @patch("agents.nodes.communications_agent.get_llm")
    def test_prior_agent_outputs_injected_into_prompt(self, mock_get_llm, mock_create_agent):
        """Outputs from preceding agents are included in the communications prompt."""
        from agents.nodes.communications_agent import communications_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("Sent.")]}
        mock_create_agent.return_value = mock_agent

        state = _base_state(agent_outputs={
            "metrics_agent": "Revenue fell 15% this week."
        })
        communications_agent_node(state)

        prompt_text = mock_agent.invoke.call_args[0][0]["messages"][0].content
        self.assertIn("Revenue fell 15%", prompt_text)

    @patch("agents.nodes.communications_agent.create_react_agent")
    @patch("agents.nodes.communications_agent.get_llm")
    def test_no_prior_outputs_does_not_crash(self, mock_get_llm, mock_create_agent):
        """Works cleanly when agent_outputs is empty (no prior context)."""
        from agents.nodes.communications_agent import communications_agent_node

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [_ai_message("Sent.")] }
        mock_create_agent.return_value = mock_agent

        # Should not raise
        result = communications_agent_node(_base_state(agent_outputs={}))
        self.assertIn("communications_agent", result["agent_outputs"])


# ════════════════════════════════════════════════ GRAPH TESTS


class AgentGraphTests(TestCase):
    """Tests for graph construction, routing logic, and run_agents()."""

    # ── _route_from_supervisor ───────────────────────────────────────────────

    def test_route_returns_end_for_finish(self):
        """_route_from_supervisor returns END when next_agent is FINISH."""
        from langgraph.graph import END
        from agents.graph import _route_from_supervisor

        state = _base_state(next_agent="FINISH")
        self.assertEqual(_route_from_supervisor(state), END)

    def test_route_returns_end_for_unknown_agent(self):
        """_route_from_supervisor treats unknown agent names as FINISH → END."""
        from langgraph.graph import END
        from agents.graph import _route_from_supervisor

        state = _base_state(next_agent="made_up_agent")
        self.assertEqual(_route_from_supervisor(state), END)

    def test_route_returns_correct_agent_names(self):
        """_route_from_supervisor returns the agent name for each known agent."""
        from agents.graph import _route_from_supervisor, _EXECUTION_AGENTS

        for agent in _EXECUTION_AGENTS:
            state = _base_state(next_agent=agent)
            self.assertEqual(_route_from_supervisor(state), agent)

    def test_route_returns_end_for_empty_next_agent(self):
        """Empty next_agent string maps to END."""
        from langgraph.graph import END
        from agents.graph import _route_from_supervisor

        state = _base_state(next_agent="")
        self.assertEqual(_route_from_supervisor(state), END)

    # ── build_graph ──────────────────────────────────────────────────────────

    def test_build_graph_returns_compiled_graph(self):
        """build_graph() compiles without error and returns a runnable graph."""
        from agents.graph import build_graph

        graph = build_graph()
        # Compiled LangGraph graphs have a .invoke method
        self.assertTrue(hasattr(graph, "invoke"))

    def test_get_graph_returns_singleton(self):
        """get_graph() returns the same compiled graph on repeated calls."""
        from agents.graph import get_graph

        g1 = get_graph()
        g2 = get_graph()
        self.assertIs(g1, g2)

    # ── run_agents ───────────────────────────────────────────────────────────

    @patch("agents.graph.get_graph")
    def test_run_agents_returns_expected_keys(self, mock_get_graph):
        """run_agents() returns dict with final_response, agent_outputs, evaluation, iterations."""
        from agents.graph import run_agents

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_response": "Stockout summary: 3 items at zero.",
            "agent_outputs": {"sql_agent": "3 rows"},
            "evaluation": "Good.",
            "iteration_count": 2,
        }
        mock_get_graph.return_value = mock_graph

        result = run_agents("Show stockout alerts", user_role="Client Admin", facility="KSH")

        self.assertIn("final_response", result)
        self.assertIn("agent_outputs", result)
        self.assertIn("evaluation", result)
        self.assertIn("iterations", result)

    @patch("agents.graph.get_graph")
    def test_run_agents_passes_initial_state_correctly(self, mock_get_graph):
        """run_agents() builds initial state with the provided task and role."""
        from agents.graph import run_agents

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_response": "done",
            "agent_outputs": {},
            "evaluation": "",
            "iteration_count": 1,
        }
        mock_get_graph.return_value = mock_graph

        run_agents("Check bed occupancy", user_role="Facility Admin", facility="Lodwar")

        invoke_call = mock_graph.invoke.call_args[0][0]
        self.assertEqual(invoke_call["task"], "Check bed occupancy")
        self.assertEqual(invoke_call["user_role"], "Facility Admin")
        self.assertEqual(invoke_call["facility"], "Lodwar")
        self.assertEqual(invoke_call["iteration_count"], 0)
        self.assertEqual(invoke_call["agent_outputs"], {})

    @patch("agents.graph.get_graph")
    def test_run_agents_handles_graph_exception_gracefully(self, mock_get_graph):
        """Graph exceptions are caught and returned as a user-facing error message."""
        from agents.graph import run_agents

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("LLM service unavailable")
        mock_get_graph.return_value = mock_graph

        result = run_agents("Any task")

        self.assertIn("error", result["final_response"].lower())
        self.assertEqual(result["agent_outputs"], {})
        self.assertEqual(result["iterations"], 0)

    @patch("agents.graph.get_graph")
    def test_run_agents_facility_defaults_to_none(self, mock_get_graph):
        """Facility defaults to None when not provided."""
        from agents.graph import run_agents

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_response": "ok", "agent_outputs": {},
            "evaluation": "", "iteration_count": 1,
        }
        mock_get_graph.return_value = mock_graph

        run_agents("Any task", user_role="Client Admin")

        invoke_call = mock_graph.invoke.call_args[0][0]
        self.assertIsNone(invoke_call["facility"])

    @patch("agents.graph.get_graph")
    def test_run_agents_returns_empty_strings_on_missing_keys(self, mock_get_graph):
        """Handles graph result dicts that are missing optional keys."""
        from agents.graph import run_agents

        mock_graph = MagicMock()
        # Graph returns minimal state (missing evaluation and iteration_count)
        mock_graph.invoke.return_value = {}
        mock_get_graph.return_value = mock_graph

        result = run_agents("Any task")

        self.assertEqual(result["final_response"], "")
        self.assertEqual(result["agent_outputs"], {})


# ════════════════════════════════════════════════ API TESTS


class RunAgentsAPITests(TestCase):
    """Tests for POST /api/v1/agents/run/"""

    def setUp(self):
        self.user = _make_admin()
        self.api_client = APIClient()
        self.url = "/api/v1/agents/run/"

    # ── authentication ───────────────────────────────────────────────────────

    def test_unauthenticated_request_returns_401(self):
        """Unauthenticated calls are rejected with 401."""
        resp = self.api_client.post(self.url, {"task": "Show KPIs"}, format="json")
        self.assertEqual(resp.status_code, 401)

    # ── validation ───────────────────────────────────────────────────────────

    @patch("agents.api.run_agents")
    def test_missing_task_returns_400(self, mock_run):
        """Request without 'task' field returns 400."""
        self.api_client.force_authenticate(user=self.user)
        resp = self.api_client.post(self.url, {}, format="json")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.data)
        mock_run.assert_not_called()

    @patch("agents.api.run_agents")
    def test_blank_task_returns_400(self, mock_run):
        """Whitespace-only 'task' returns 400."""
        self.api_client.force_authenticate(user=self.user)
        resp = self.api_client.post(self.url, {"task": "   "}, format="json")
        self.assertEqual(resp.status_code, 400)
        mock_run.assert_not_called()

    # ── success ──────────────────────────────────────────────────────────────

    @patch("agents.api.run_agents")
    def test_valid_request_returns_200(self, mock_run):
        """Valid request returns 200 with agent result fields."""
        mock_run.return_value = {
            "final_response": "3 stockouts identified.",
            "agent_outputs": {"sql_agent": "3 rows"},
            "evaluation": "High quality.",
            "iterations": 2,
        }
        self.api_client.force_authenticate(user=self.user)

        resp = self.api_client.post(self.url, {"task": "Show stockouts"}, format="json")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["final_response"], "3 stockouts identified.")
        self.assertIn("agent_outputs", resp.data)
        self.assertIn("iterations", resp.data)

    @patch("agents.api.run_agents")
    def test_user_role_extracted_from_authenticated_user(self, mock_run):
        """The authenticated user's role is passed to run_agents."""
        mock_run.return_value = {
            "final_response": "ok",
            "agent_outputs": {},
            "evaluation": "",
            "iterations": 1,
        }
        self.api_client.force_authenticate(user=self.user)

        self.api_client.post(self.url, {"task": "Show KPIs"}, format="json")

        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["user_role"], ROLE_CLIENT_ADMIN)

    @patch("agents.api.run_agents")
    def test_facility_passed_when_provided(self, mock_run):
        """Optional facility field is forwarded to run_agents."""
        mock_run.return_value = {
            "final_response": "ok",
            "agent_outputs": {},
            "evaluation": "",
            "iterations": 1,
        }
        self.api_client.force_authenticate(user=self.user)

        self.api_client.post(
            self.url,
            {"task": "Show KPIs", "facility": "Lodwar"},
            format="json",
        )

        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["facility"], "Lodwar")

    @patch("agents.api.run_agents")
    def test_facility_defaults_to_none_when_omitted(self, mock_run):
        """Facility defaults to None when not included in the request."""
        mock_run.return_value = {
            "final_response": "ok",
            "agent_outputs": {},
            "evaluation": "",
            "iterations": 1,
        }
        self.api_client.force_authenticate(user=self.user)

        self.api_client.post(self.url, {"task": "Show KPIs"}, format="json")

        call_kwargs = mock_run.call_args[1]
        self.assertIsNone(call_kwargs["facility"])

    @patch("agents.api.run_agents")
    def test_facility_whitespace_stripped_to_none(self, mock_run):
        """Whitespace-only facility is treated as absent (None)."""
        mock_run.return_value = {
            "final_response": "ok",
            "agent_outputs": {},
            "evaluation": "",
            "iterations": 1,
        }
        self.api_client.force_authenticate(user=self.user)

        self.api_client.post(
            self.url,
            {"task": "Show KPIs", "facility": "   "},
            format="json",
        )

        call_kwargs = mock_run.call_args[1]
        self.assertIsNone(call_kwargs["facility"])

    @patch("agents.api.run_agents")
    def test_non_admin_user_can_call_endpoint(self, mock_run):
        """Any authenticated user (regardless of role) can invoke the agents."""
        mock_run.return_value = {
            "final_response": "ok",
            "agent_outputs": {},
            "evaluation": "",
            "iterations": 1,
        }
        regular_user = _make_user("regular_agent_user", role=ROLE_FACILITY_ADMIN)
        self.api_client.force_authenticate(user=regular_user)

        resp = self.api_client.post(self.url, {"task": "Check my facility"}, format="json")

        self.assertEqual(resp.status_code, 200)

    @patch("agents.api.run_agents")
    def test_task_is_trimmed_before_passing_to_run_agents(self, mock_run):
        """Leading/trailing whitespace in task is stripped before processing."""
        mock_run.return_value = {
            "final_response": "ok",
            "agent_outputs": {},
            "evaluation": "",
            "iterations": 1,
        }
        self.api_client.force_authenticate(user=self.user)

        self.api_client.post(
            self.url,
            {"task": "  Show stockouts  "},
            format="json",
        )

        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["task"], "Show stockouts")
