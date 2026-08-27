"""
Tests for the self_service (Self-Service Analytics) app.

Coverage:
  - ChatSession / ChatMessage models
  - security.py  — access context, topic gating, CLS, masking
  - handlers.py  — intent detection, per-topic handlers, access denial
  - consumers.py — WebSocket connect / receive / disconnect lifecycle
  - views.py     — ChatHistoryView, AccessContextView
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, TransactionTestCase
from django.urls import reverse

from analytics_app.models import Dashboard
from authentication.roles import (
    ROLE_CLIENT_ADMIN,
    ROLE_FACILITIES_ADMIN,
    ROLE_FACILITY_ADMIN,
)
from core.models import Client, Facility
from self_service.consumers import AnalyticsChatConsumer
from self_service.handlers import _detect_intent, process_query
from self_service.models import ChatMessage, ChatSession
from self_service.security import (
    _mask,
    apply_data_filters,
    check_topic_access,
    get_user_access_context,
)

User = get_user_model()


# =============================================================================
# Helpers
# =============================================================================

def _make_user(username='testuser', password='testpass123',
               role=None, is_superuser=False):
    """Create and return a User, optionally assigning a role group + profile."""
    user = User.objects.create_user(
        username=username, password=password, email=f'{username}@test.com'
    )
    if is_superuser:
        user.is_superuser = True
        user.is_staff = True
        user.save()
    if role:
        group, _ = Group.objects.get_or_create(name=role)
        user.groups.add(group)
        try:
            user.profile.role = role
            user.profile.save()
        except Exception:
            pass
    return user


def _make_client(**kwargs):
    defaults = {'name': 'Test Hospital Group', 'slug': 'test-hosp-grp', 'is_active': True}
    defaults.update(kwargs)
    return Client.objects.create(**defaults)


def _make_facility(client, **kwargs):
    defaults = {'name': 'Main Clinic', 'slug': 'main-clinic', 'is_active': True}
    defaults.update(kwargs)
    return Facility.objects.create(client=client, **defaults)


def _make_dashboard(name='Test Dashboard', client=None, **kwargs):
    defaults = {
        'slug': name.lower().replace(' ', '-'),
        'category': 'operational',
        'is_active': True,
        'streamlit_url': 'http://localhost:8501',
        'description': 'A test dashboard',
    }
    defaults.update(kwargs)
    return Dashboard.objects.create(name=name, client=client, **defaults)


def _build_access_context(role=ROLE_FACILITY_ADMIN, client=None, facility=None,
                          is_superuser=False):
    """Return a minimal access context dict without hitting the DB."""
    from self_service.security import _ROLE_ACCESS, _FALLBACK
    config = _ROLE_ACCESS.get(role, _FALLBACK)
    return {
        'role': role,
        'role_display': config['display'],
        'allowed_topics': list(config['allowed_topics']),
        'denied_columns': set(config['denied_columns']),
        'masked_columns': set(config['masked_columns']),
        'row_scope': config['row_scope'],
        'client': client,
        'facility': facility,
        'is_superuser': is_superuser,
    }


# =============================================================================
# MODEL TESTS
# =============================================================================

class ChatSessionModelTests(TestCase):
    """ChatSession model creation, constraints, and string representation."""

    def setUp(self):
        self.user = _make_user('session_user')

    def test_create_session(self):
        session = ChatSession.objects.create(user=self.user)
        self.assertIsNotNone(session.pk)
        self.assertTrue(session.is_active)

    def test_session_key_is_unique_uuid(self):
        s1 = ChatSession.objects.create(user=self.user)
        s2 = ChatSession.objects.create(user=self.user)
        self.assertNotEqual(s1.session_key, s2.session_key)

    def test_is_active_defaults_to_true(self):
        session = ChatSession.objects.create(user=self.user)
        self.assertTrue(session.is_active)

    def test_str_includes_session_key_and_username(self):
        session = ChatSession.objects.create(user=self.user)
        self.assertIn(str(session.session_key), str(session))
        self.assertIn(self.user.username, str(session))

    def test_ordering_most_recent_first(self):
        s1 = ChatSession.objects.create(user=self.user)
        s2 = ChatSession.objects.create(user=self.user)
        sessions = list(ChatSession.objects.filter(user=self.user))
        self.assertEqual(sessions[0].pk, s2.pk)

    def test_cascade_delete_with_user(self):
        ChatSession.objects.create(user=self.user)
        self.user.delete()
        self.assertEqual(ChatSession.objects.count(), 0)


class ChatMessageModelTests(TestCase):
    """ChatMessage model creation, role choices, and ordering."""

    def setUp(self):
        self.user = _make_user('msg_user')
        self.session = ChatSession.objects.create(user=self.user)

    def test_create_user_message(self):
        msg = ChatMessage.objects.create(
            session=self.session, role=ChatMessage.ROLE_USER, content='Hello'
        )
        self.assertEqual(msg.role, 'user')
        self.assertEqual(msg.content, 'Hello')

    def test_create_assistant_message(self):
        msg = ChatMessage.objects.create(
            session=self.session, role=ChatMessage.ROLE_ASSISTANT,
            content='Hi there!', query_intent='help'
        )
        self.assertEqual(msg.role, 'assistant')
        self.assertEqual(msg.query_intent, 'help')

    def test_str_includes_role_and_content_prefix(self):
        msg = ChatMessage.objects.create(
            session=self.session, role=ChatMessage.ROLE_USER, content='Test message'
        )
        self.assertIn('user', str(msg))
        self.assertIn('Test message', str(msg))

    def test_ordering_by_created_at(self):
        m1 = ChatMessage.objects.create(
            session=self.session, role=ChatMessage.ROLE_USER, content='First'
        )
        m2 = ChatMessage.objects.create(
            session=self.session, role=ChatMessage.ROLE_ASSISTANT, content='Second'
        )
        msgs = list(self.session.messages.all())
        self.assertEqual(msgs[0].pk, m1.pk)
        self.assertEqual(msgs[1].pk, m2.pk)

    def test_query_intent_can_be_blank(self):
        msg = ChatMessage.objects.create(
            session=self.session, role=ChatMessage.ROLE_USER, content='Hi'
        )
        self.assertEqual(msg.query_intent, '')

    def test_cascade_delete_with_session(self):
        ChatMessage.objects.create(
            session=self.session, role=ChatMessage.ROLE_USER, content='Hi'
        )
        self.session.delete()
        self.assertEqual(ChatMessage.objects.count(), 0)


# =============================================================================
# SECURITY TESTS
# =============================================================================

class AccessContextTests(TestCase):
    """get_user_access_context returns correct config per role."""

    def test_facility_admin_context(self):
        user = _make_user('fa_user', role=ROLE_FACILITY_ADMIN)
        ctx = get_user_access_context(user)
        self.assertEqual(ctx['role'], ROLE_FACILITY_ADMIN)
        self.assertEqual(ctx['row_scope'], 'facility')
        self.assertIn('salary', ctx['denied_columns'])
        self.assertIn('phone_number', ctx['masked_columns'])
        self.assertNotIn('financials', ctx['allowed_topics'])
        # Row-scoped to own facility, but can still query it
        self.assertIn('facilities', ctx['allowed_topics'])

    def test_facilities_admin_context(self):
        user = _make_user('fsa_user', role=ROLE_FACILITIES_ADMIN)
        ctx = get_user_access_context(user)
        self.assertEqual(ctx['row_scope'], 'multi_facility')
        self.assertIn('salary', ctx['denied_columns'])
        self.assertIn('national_id', ctx['masked_columns'])
        self.assertNotIn('financials', ctx['allowed_topics'])
        self.assertIn('facilities', ctx['allowed_topics'])

    def test_client_admin_context(self):
        user = _make_user('ca_user', role=ROLE_CLIENT_ADMIN)
        ctx = get_user_access_context(user)
        self.assertEqual(ctx['row_scope'], 'client')
        self.assertEqual(len(ctx['denied_columns']), 0)
        self.assertEqual(len(ctx['masked_columns']), 0)
        self.assertIn('financials', ctx['allowed_topics'])

    def test_superuser_treated_as_client_admin(self):
        user = _make_user('su', is_superuser=True)
        ctx = get_user_access_context(user)
        self.assertTrue(ctx['is_superuser'])
        # Superuser gets ROLE_CLIENT_ADMIN via get_user_role()
        self.assertIn('financials', ctx['allowed_topics'])
        self.assertEqual(len(ctx['denied_columns']), 0)

    def test_context_includes_client_and_facility_from_profile(self):
        user = _make_user('profile_user', role=ROLE_FACILITY_ADMIN)
        client_obj = _make_client(slug='ctx-client')
        facility_obj = _make_facility(client_obj, slug='ctx-facility')
        try:
            user.profile.client = client_obj
            user.profile.facility = facility_obj
            user.profile.save()
        except Exception:
            pass
        ctx = get_user_access_context(user)
        # profile fields should be accessible via context
        self.assertIn('client', ctx)
        self.assertIn('facility', ctx)


class TopicAccessTests(TestCase):
    """check_topic_access enforces allowed_topics per role."""

    def test_facility_admin_cannot_access_financials(self):
        ctx = _build_access_context(role=ROLE_FACILITY_ADMIN)
        self.assertFalse(check_topic_access('financials', ctx))

    def test_facility_admin_can_access_patients(self):
        ctx = _build_access_context(role=ROLE_FACILITY_ADMIN)
        self.assertTrue(check_topic_access('patients', ctx))

    def test_facilities_admin_cannot_access_financials(self):
        ctx = _build_access_context(role=ROLE_FACILITIES_ADMIN)
        self.assertFalse(check_topic_access('financials', ctx))

    def test_facilities_admin_can_access_facilities(self):
        ctx = _build_access_context(role=ROLE_FACILITIES_ADMIN)
        self.assertTrue(check_topic_access('facilities', ctx))

    def test_client_admin_can_access_all_topics(self):
        ctx = _build_access_context(role=ROLE_CLIENT_ADMIN)
        for topic in ['summary', 'facilities', 'staff', 'financials',
                      'patients', 'operations', 'dashboards']:
            self.assertTrue(check_topic_access(topic, ctx), f'Failed for topic: {topic}')

    def test_superuser_bypasses_topic_check(self):
        ctx = _build_access_context(role=ROLE_FACILITY_ADMIN, is_superuser=True)
        self.assertTrue(check_topic_access('financials', ctx))
        self.assertTrue(check_topic_access('anything', ctx))


class DataFilterTests(TestCase):
    """apply_data_filters enforces column-level security and masking."""

    def _records(self):
        return [
            {
                'name': 'Nurse Jane',
                'salary': 50000,
                'national_id': 'A1234567',
                'phone_number': '+254712345678',
                'department': 'ICU',
            }
        ]

    def test_denied_columns_are_removed(self):
        ctx = _build_access_context(role=ROLE_FACILITY_ADMIN)
        result = apply_data_filters(self._records(), ctx)
        self.assertNotIn('salary', result[0])

    def test_allowed_columns_pass_through(self):
        ctx = _build_access_context(role=ROLE_FACILITY_ADMIN)
        result = apply_data_filters(self._records(), ctx)
        self.assertIn('name', result[0])
        self.assertIn('department', result[0])

    def test_masked_columns_are_partially_obscured(self):
        ctx = _build_access_context(role=ROLE_FACILITY_ADMIN)
        result = apply_data_filters(self._records(), ctx)
        self.assertIn('national_id', result[0])
        self.assertNotEqual(result[0]['national_id'], 'A1234567')
        self.assertIn('*', result[0]['national_id'])

    def test_client_admin_sees_all_columns_unmasked(self):
        ctx = _build_access_context(role=ROLE_CLIENT_ADMIN)
        result = apply_data_filters(self._records(), ctx)
        self.assertEqual(result[0]['salary'], 50000)
        self.assertEqual(result[0]['national_id'], 'A1234567')

    def test_empty_records_returns_empty(self):
        ctx = _build_access_context(role=ROLE_FACILITY_ADMIN)
        self.assertEqual(apply_data_filters([], ctx), [])

    def test_multiple_records_all_filtered(self):
        ctx = _build_access_context(role=ROLE_FACILITY_ADMIN)
        records = self._records() + self._records()
        result = apply_data_filters(records, ctx)
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertNotIn('salary', row)


class MaskTests(TestCase):
    """_mask correctly obscures sensitive field values."""

    def test_long_value_shows_first_and_last_two_chars(self):
        result = _mask('A1234567')
        self.assertTrue(result.startswith('A1'))
        self.assertTrue(result.endswith('67'))
        self.assertIn('*', result)

    def test_short_value_fully_obscured(self):
        self.assertEqual(_mask('abc'), '***')
        self.assertEqual(_mask('ab'), '***')

    def test_none_returns_none(self):
        self.assertIsNone(_mask(None))

    def test_empty_string_returns_empty(self):
        self.assertEqual(_mask(''), '')

    def test_numeric_value_is_stringified_then_masked(self):
        result = _mask(12345678)
        self.assertIn('*', result)


# =============================================================================
# HANDLER TESTS
# =============================================================================

class IntentDetectionTests(TestCase):
    """_detect_intent maps query strings to correct intent labels."""

    cases = [
        ('help me',                    'help'),
        ('what can you do?',           'help'),
        ('give me a summary',          'summary'),
        ('overall status',             'summary'),
        ('show facilities',            'facilities'),
        ('how many hospitals',         'facilities'),
        ('patient admissions',         'patients'),
        ('outpatient visits',          'patients'),
        ('how many staff members',     'staff'),
        ('employee headcount',         'staff'),
        ('revenue this month',         'financials'),
        ('billing breakdown',          'financials'),
        ('show available dashboards',  'dashboards'),
        ('analytics report',           'dashboards'),
        ('KPIs for this quarter',      'operations'),
        ('operational efficiency',     'operations'),
        ('random unrecognised text',   'general'),
    ]

    def test_intent_patterns(self):
        for query, expected in self.cases:
            with self.subTest(query=query):
                self.assertEqual(_detect_intent(query), expected)


class ProcessQueryAccessTests(TestCase):
    """process_query blocks denied topics and routes allowed ones."""

    def setUp(self):
        self.user = _make_user('query_user', role=ROLE_FACILITY_ADMIN)
        self.ctx = _build_access_context(role=ROLE_FACILITY_ADMIN)

    def test_denied_topic_returns_access_error(self):
        result = process_query('show me financial breakdown', self.user, self.ctx)
        self.assertEqual(result['intent'], 'financials')
        self.assertIn('access level', result['content'].lower())

    def test_allowed_topic_returns_content(self):
        result = process_query('give me a summary', self.user, self.ctx)
        self.assertNotIn("don't have access", result['content'].lower())
        self.assertIn('content', result)

    def test_result_always_includes_intent(self):
        result = process_query('patient admissions', self.user, self.ctx)
        self.assertIn('intent', result)

    def test_help_is_always_accessible(self):
        """Help intent should never be blocked, regardless of role."""
        result = process_query('help', self.user, self.ctx)
        self.assertEqual(result['intent'], 'help')
        self.assertNotIn("don't have access", result['content'].lower())

    def test_general_is_always_accessible(self):
        result = process_query('xyzzy frobulate', self.user, self.ctx)
        self.assertEqual(result['intent'], 'general')
        self.assertNotIn("don't have access", result['content'].lower())


class HandlerContentTests(TestCase):
    """Individual topic handlers return well-formed content."""

    def setUp(self):
        self.user = _make_user('content_user', role=ROLE_CLIENT_ADMIN)
        self.ctx = _build_access_context(role=ROLE_CLIENT_ADMIN)

    def test_help_lists_all_allowed_topics(self):
        result = process_query('help', self.user, self.ctx)
        for topic in self.ctx['allowed_topics']:
            self.assertIn(topic, result['content'].lower())

    def test_summary_includes_facility_count(self):
        client_obj = _make_client(slug='summ-client')
        _make_facility(client_obj, slug='summ-fac-1')
        _make_facility(client_obj, slug='summ-fac-2')
        ctx = _build_access_context(role=ROLE_CLIENT_ADMIN, client=client_obj)
        result = process_query('give me an overview', self.user, ctx)
        self.assertIn('2', result['content'])

    def test_facilities_client_scope_lists_facilities(self):
        client_obj = _make_client(slug='fac-scope-client')
        _make_facility(client_obj, slug='fac-scope-a', name='Facility Alpha')
        ctx = _build_access_context(role=ROLE_CLIENT_ADMIN, client=client_obj)
        result = process_query('show facilities', self.user, ctx)
        self.assertIn('Facility Alpha', result['content'])

    def test_facilities_facility_scope_shows_single_facility(self):
        client_obj = _make_client(slug='single-fac-client')
        fac = _make_facility(client_obj, slug='single-fac', name='Solo Clinic')
        ctx = _build_access_context(
            role=ROLE_FACILITY_ADMIN,
            facility=fac,
        )
        user = _make_user('single_fac_user', role=ROLE_FACILITY_ADMIN)
        result = process_query('show facilities', user, ctx)
        self.assertIn('Solo Clinic', result['content'])

    def test_dashboards_handler_returns_active_dashboards(self):
        client_obj = _make_client(slug='dash-client')
        _make_dashboard('Active Dash', client=client_obj)
        _make_dashboard('Hidden Dash', client=client_obj, is_active=False)
        ctx = _build_access_context(role=ROLE_CLIENT_ADMIN, client=client_obj)
        result = process_query('show me the dashboards', self.user, ctx)
        self.assertIn('Active Dash', result['content'])
        self.assertNotIn('Hidden Dash', result['content'])

    def test_patients_handler_returns_guidance(self):
        result = process_query('how many patients', self.user, self.ctx)
        self.assertIn('patient', result['content'].lower())
        self.assertEqual(result['intent'], 'patients')

    def test_staff_handler_returns_guidance(self):
        result = process_query('staff headcount', self.user, self.ctx)
        self.assertIn('staff', result['content'].lower())

    def test_operations_handler_returns_kpi_list(self):
        result = process_query('operational KPIs', self.user, self.ctx)
        self.assertIn('content', result)
        self.assertIn('KPI', result['content'].upper())

    def test_general_handler_suggests_help(self):
        result = process_query('sdfghjkl', self.user, self.ctx)
        self.assertIn('help', result['content'].lower())


# =============================================================================
# CONSUMER TESTS  (async WebSocket via channels.testing)
# =============================================================================

def _run(coro):
    """Run an async coroutine synchronously inside a TestCase method."""
    return asyncio.get_event_loop().run_until_complete(coro)


class ConsumerAuthTests(TransactionTestCase):
    """WebSocket consumer rejects unauthenticated connections."""

    def test_anonymous_user_is_rejected(self):
        from django.contrib.auth.models import AnonymousUser

        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = AnonymousUser()
            connected, code = await comm.connect()
            # Connection must be rejected with the auth-failure close code
            self.assertFalse(connected)
            self.assertEqual(code, 4001)
            # Do NOT call disconnect() — the consumer already closed the socket

        _run(_go())


class ConsumerConnectTests(TransactionTestCase):
    """Authenticated connections receive a welcome message."""

    def setUp(self):
        self.user = _make_user('ws_user', role=ROLE_FACILITY_ADMIN)

    def test_authenticated_user_connects_successfully(self):
        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            connected, _ = await comm.connect()
            self.assertTrue(connected)
            await comm.disconnect()

        _run(_go())

    def test_welcome_message_is_sent_on_connect(self):
        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # session
            response = await comm.receive_json_from()
            self.assertEqual(response['type'], 'message')
            self.assertEqual(response['role'], 'assistant')
            self.assertIn('Analytics Assistant', response['content'])
            await comm.disconnect()

        _run(_go())

    def test_welcome_message_mentions_role(self):
        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # session
            response = await comm.receive_json_from()
            # Role display name should appear in welcome
            self.assertIn('Facility Administrator', response['content'])
            await comm.disconnect()

        _run(_go())

    def test_chat_session_created_in_db_on_connect(self):
        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # session
            await comm.receive_json_from()  # consume welcome
            await comm.disconnect()

        _run(_go())
        self.assertEqual(ChatSession.objects.filter(user=self.user).count(), 1)


class ConsumerSessionResumeTests(TransactionTestCase):
    """A returning client that supplies ?session=<key> resumes that row
    instead of spawning a new one each time it connects."""

    def setUp(self):
        self.user = _make_user('resume_user', role=ROLE_FACILITY_ADMIN)
        self.other_user = _make_user('resume_other_user', role=ROLE_FACILITY_ADMIN)

    def _connect(self, path):
        result = {}

        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, path)
            comm.scope['user'] = self.user
            await comm.connect()
            result['session_msg'] = await comm.receive_json_from()
            if result['session_msg'].get('is_new'):
                result['welcome'] = await comm.receive_json_from()
            await comm.disconnect()

        _run(_go())
        return result

    def test_new_connection_without_session_param_creates_session(self):
        res = self._connect('/ws/analytics/chat/')
        self.assertEqual(res['session_msg']['type'], 'session')
        self.assertTrue(res['session_msg']['is_new'])
        self.assertIn('welcome', res)
        self.assertEqual(ChatSession.objects.filter(user=self.user).count(), 1)

    def test_two_connections_racing_on_the_same_not_yet_created_key_share_one_session(self):
        """Reproduces the "hot reload sometimes starts a new conversation"
        bug: a stale reconnect timer firing at the same moment as a fresh
        connect() both requesting a session_key that doesn't exist in the
        DB yet (e.g. the client already has a key from localStorage, but
        this is the very first connection of the process to actually
        create it). Both must land on the SAME ChatSession — the client
        picked the key, so the server should never silently mint a second,
        different one for either side of the race."""
        import uuid as uuid_module

        shared_key = str(uuid_module.uuid4())

        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm_a = WebsocketCommunicator(app, f'/ws/analytics/chat/?session={shared_key}')
            comm_a.scope['user'] = self.user
            comm_b = WebsocketCommunicator(app, f'/ws/analytics/chat/?session={shared_key}')
            comm_b.scope['user'] = self.user

            await asyncio.gather(comm_a.connect(), comm_b.connect())
            msg_a = await comm_a.receive_json_from()
            msg_b = await comm_b.receive_json_from()
            await comm_a.disconnect()
            await comm_b.disconnect()
            return msg_a, msg_b

        msg_a, msg_b = _run(_go())
        self.assertEqual(msg_a['session_key'], shared_key)
        self.assertEqual(msg_b['session_key'], shared_key)
        self.assertEqual(ChatSession.objects.filter(user=self.user).count(), 1)

    def test_resuming_own_session_does_not_create_a_second_one(self):
        session = ChatSession.objects.create(user=self.user)
        res = self._connect('/ws/analytics/chat/?session=' + str(session.session_key))
        self.assertFalse(res['session_msg']['is_new'])
        self.assertEqual(res['session_msg']['session_key'], str(session.session_key))
        self.assertNotIn('welcome', res)
        self.assertEqual(ChatSession.objects.filter(user=self.user).count(), 1)

    def test_session_param_owned_by_another_user_is_ignored(self):
        foreign_session = ChatSession.objects.create(user=self.other_user)
        res = self._connect('/ws/analytics/chat/?session=' + str(foreign_session.session_key))
        # Must not resume someone else's session — a fresh one is created instead.
        self.assertTrue(res['session_msg']['is_new'])
        self.assertNotEqual(res['session_msg']['session_key'], str(foreign_session.session_key))
        self.assertEqual(ChatSession.objects.filter(user=self.user).count(), 1)

    def test_malformed_session_param_falls_back_to_new_session(self):
        res = self._connect('/ws/analytics/chat/?session=not-a-real-uuid')
        self.assertTrue(res['session_msg']['is_new'])
        self.assertEqual(ChatSession.objects.filter(user=self.user).count(), 1)

    def test_disconnect_does_not_deactivate_session(self):
        self._connect('/ws/analytics/chat/')
        session = ChatSession.objects.get(user=self.user)
        self.assertTrue(session.is_active)


class ConsumerSessionPersistenceTests(TransactionTestCase):
    """Sending a message sets the session title and bumps last_activity.

    These calls hit the real OpenAI API (this suite has no mock for it —
    see ConsumerMessagingTests), so response receives use a generous
    timeout: WebsocketCommunicator.receive_json_from() defaults to 1s,
    which a live network round trip can easily exceed.
    """

    _REPLY_TIMEOUT = 20

    def setUp(self):
        self.user = _make_user('title_user', role=ROLE_CLIENT_ADMIN)

    def test_title_is_set_from_first_user_message(self):
        # 'help' is a recognised platform intent (routes through the fast,
        # single-call OpenAI path) — matches the pattern the rest of this
        # suite uses to avoid exercising the slow, network-heavy LangGraph
        # agent path in tests (see ConsumerMessagingTests).
        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # session
            await comm.receive_json_from()  # welcome
            await comm.send_json_to({'message': 'help'})
            await comm.receive_json_from()  # typing
            await comm.receive_json_from(timeout=self._REPLY_TIMEOUT)  # answer
            await comm.disconnect()

        _run(_go())
        session = ChatSession.objects.get(user=self.user)
        self.assertEqual(session.title, 'help')

    def test_title_is_not_overwritten_by_later_messages(self):
        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()
            await comm.receive_json_from()
            await comm.send_json_to({'message': 'help'})
            await comm.receive_json_from()
            await comm.receive_json_from(timeout=self._REPLY_TIMEOUT)
            await comm.send_json_to({'message': 'give me a summary'})
            await comm.receive_json_from()
            await comm.receive_json_from(timeout=self._REPLY_TIMEOUT)
            await comm.disconnect()

        _run(_go())
        session = ChatSession.objects.get(user=self.user)
        self.assertEqual(session.title, 'help')
        self.assertEqual(session.messages.count(), 4)


class SaveMessagesChartPersistenceTests(TestCase):
    """_save_messages() writes a chart's PNG bytes to storage and links
    them to the assistant's ChatMessage row — charts used to be ephemeral
    (regenerated per request, never saved); this is what makes them
    survive a session reload / history replay."""

    def setUp(self):
        self.user = _make_user('chart_persist_user')
        self.session = ChatSession.objects.create(user=self.user)
        # Bypass __init__ (which expects a Channels scope) — these are
        # plain sync helpers that only touch self.user/self.session_obj.
        self.consumer = AnalyticsChatConsumer.__new__(AnalyticsChatConsumer)
        self.consumer.user = self.user
        self.consumer.session_obj = self.session

    def _fake_png_base64(self):
        import base64
        # Minimal valid PNG signature + IHDR-ish bytes is unnecessary here —
        # _save_messages just writes whatever bytes it's given; content
        # correctness of the PNG itself is chart_codegen's job, tested
        # separately in agents/tests_charts.py.
        return base64.b64encode(b'fake-png-bytes').decode('ascii')

    def test_chart_image_written_to_storage_and_linked(self):
        response = {
            'content': 'Here is the breakdown.',
            'intent': 'metric_query',
            'chart': {
                'image_base64': self._fake_png_base64(),
                'mime': 'image/png',
                'caption': 'Admissions by Sex',
            },
        }
        self.consumer._save_messages('how many admissions', response)

        assistant_msg = ChatMessage.objects.get(session=self.session, role=ChatMessage.ROLE_ASSISTANT)
        self.assertTrue(assistant_msg.chart_image)
        self.assertEqual(assistant_msg.chart_caption, 'Admissions by Sex')
        with assistant_msg.chart_image.open('rb') as f:
            self.assertEqual(f.read(), b'fake-png-bytes')

    def test_no_chart_leaves_fields_blank(self):
        response = {'content': 'General help text.', 'intent': 'help'}
        self.consumer._save_messages('help', response)

        assistant_msg = ChatMessage.objects.get(session=self.session, role=ChatMessage.ROLE_ASSISTANT)
        self.assertFalse(assistant_msg.chart_image)
        self.assertEqual(assistant_msg.chart_caption, '')

    def test_both_user_and_assistant_rows_created(self):
        response = {'content': 'Answer text.', 'intent': 'metric_query'}
        self.consumer._save_messages('a question', response)
        self.assertEqual(self.session.messages.count(), 2)

    def tearDown(self):
        for msg in ChatMessage.objects.filter(session=self.session):
            if msg.chart_image:
                msg.chart_image.delete(save=False)


class RunAgentAlwaysChartsTests(TestCase):
    """_run_agent() attempts a chart for every metric answer now — no more
    gating on the user's wording ("show me a chart") or a size-based
    offer. Mocks the graph + get_chart_for_thread so this is a fast,
    deterministic test of the wiring, not a live LLM call."""

    def setUp(self):
        self.user = _make_user('always_chart_user', role=ROLE_CLIENT_ADMIN)
        self.session = ChatSession.objects.create(user=self.user)
        self.consumer = AnalyticsChatConsumer.__new__(AnalyticsChatConsumer)
        self.consumer.user = self.user
        self.consumer.session_obj = self.session

    def _fake_graph_output(self):
        return {
            'formatted_result': {
                'summary': 'There are 42 admissions.',
                'thread_id': 'fake-thread-id',
                'metric_id': 'admissions_count',
                'data': [{'x': 1}],
            }
        }

    def test_chart_attempted_even_without_chart_wording_in_query(self):
        fake_chart = {'image_base64': 'Zm9v', 'mime': 'image/png', 'caption': 'Admissions'}
        with patch('agents.graph.graph') as mock_graph, \
             patch('agents.facility.resolve_facility_from_user', return_value=None), \
             patch('agents.charts.get_chart_for_thread', return_value=(fake_chart, None)) as mock_chart:
            mock_graph.invoke.return_value = self._fake_graph_output()
            # Deliberately no "chart"/"graph"/"visualize" wording at all.
            result = self.consumer._run_agent('how many admissions do we have')

        mock_chart.assert_called_once()
        self.assertEqual(mock_chart.call_args.kwargs.get('question'), 'how many admissions do we have')
        self.assertEqual(result['chart'], fake_chart)

    def test_no_chart_offer_text_appended_to_content(self):
        with patch('agents.graph.graph') as mock_graph, \
             patch('agents.facility.resolve_facility_from_user', return_value=None), \
             patch('agents.charts.get_chart_for_thread', return_value=(None, 'not chartable')):
            mock_graph.invoke.return_value = self._fake_graph_output()
            result = self.consumer._run_agent('how many admissions do we have')

        self.assertEqual(result['content'], 'There are 42 admissions.')
        self.assertNotIn('would you like', result['content'].lower())
        self.assertNotIn('chart_offer', result)
        self.assertIsNone(result['chart'])


class RunAgentThreadContinuityTests(TestCase):
    """_run_agent() must reuse ChatSession.thread_id across turns — that's
    the entire mechanism agents/state.py relies on for a follow-up
    question ("now break that down by month") to see the previous turn's
    conversation at all. A fresh thread_id per call (the previous bug)
    meant every question was classified in total isolation."""

    def setUp(self):
        self.user = _make_user('continuity_user', role=ROLE_CLIENT_ADMIN)
        self.session = ChatSession.objects.create(user=self.user)
        self.consumer = AnalyticsChatConsumer.__new__(AnalyticsChatConsumer)
        self.consumer.user = self.user
        self.consumer.session_obj = self.session

    def _fake_graph_output(self, thread_id):
        return {
            'formatted_result': {
                'summary': 'answer',
                'thread_id': thread_id,
                'metric_id': 'some_metric',
                'data': [],
            }
        }

    def test_thread_id_is_persisted_to_the_session_on_first_call(self):
        self.assertEqual(self.session.thread_id, '')
        with patch('agents.graph.graph') as mock_graph, \
             patch('agents.facility.resolve_facility_from_user', return_value=None), \
             patch('agents.charts.get_chart_for_thread', return_value=(None, None)):
            mock_graph.invoke.side_effect = lambda state, config: self._fake_graph_output(
                config['configurable']['thread_id']
            )
            self.consumer._run_agent('first question')

        self.session.refresh_from_db()
        self.assertTrue(self.session.thread_id)

    def test_same_thread_id_reused_across_two_calls(self):
        seen_thread_ids = []
        with patch('agents.graph.graph') as mock_graph, \
             patch('agents.facility.resolve_facility_from_user', return_value=None), \
             patch('agents.charts.get_chart_for_thread', return_value=(None, None)):
            def _invoke(state, config):
                seen_thread_ids.append(config['configurable']['thread_id'])
                return self._fake_graph_output(config['configurable']['thread_id'])
            mock_graph.invoke.side_effect = _invoke

            self.consumer._run_agent('first question')
            self.consumer._run_agent('a follow-up question')

        self.assertEqual(len(seen_thread_ids), 2)
        self.assertEqual(seen_thread_ids[0], seen_thread_ids[1])

    def test_messages_key_omitted_so_graph_state_can_accumulate(self):
        """initial_state must NOT include 'messages' or 'last_matched_metric'
        — LangGraph merges by key, so including them would reset the
        checkpointed history instead of letting it accumulate (see
        agents/state.py's _append_and_trim reducer)."""
        captured_state = {}
        with patch('agents.graph.graph') as mock_graph, \
             patch('agents.facility.resolve_facility_from_user', return_value=None), \
             patch('agents.charts.get_chart_for_thread', return_value=(None, None)):
            def _invoke(state, config):
                captured_state.update(state)
                return self._fake_graph_output(config['configurable']['thread_id'])
            mock_graph.invoke.side_effect = _invoke

            self.consumer._run_agent('a question')

        self.assertNotIn('messages', captured_state)
        self.assertNotIn('last_matched_metric', captured_state)
        self.assertEqual(captured_state['question'], 'a question')

    def test_different_sessions_get_independent_thread_ids(self):
        other_session = ChatSession.objects.create(user=self.user)
        other_consumer = AnalyticsChatConsumer.__new__(AnalyticsChatConsumer)
        other_consumer.user = self.user
        other_consumer.session_obj = other_session

        with patch('agents.graph.graph') as mock_graph, \
             patch('agents.facility.resolve_facility_from_user', return_value=None), \
             patch('agents.charts.get_chart_for_thread', return_value=(None, None)):
            mock_graph.invoke.side_effect = lambda state, config: self._fake_graph_output(
                config['configurable']['thread_id']
            )
            self.consumer._run_agent('question in session A')
            other_consumer._run_agent('question in session B')

        self.session.refresh_from_db()
        other_session.refresh_from_db()
        self.assertTrue(self.session.thread_id)
        self.assertTrue(other_session.thread_id)
        self.assertNotEqual(self.session.thread_id, other_session.thread_id)


class ConsumerMessagingTests(TransactionTestCase):
    """Messages sent over WebSocket produce typed responses."""

    def setUp(self):
        self.user = _make_user('ws_msg_user', role=ROLE_CLIENT_ADMIN)

    def _communicate(self, query):
        """Helper: connect, drain welcome, send query, return response dict."""
        result = {}

        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # discard session
            await comm.receive_json_from()  # discard welcome

            await comm.send_json_to({'message': query})

            # First response is the typing indicator
            typing = await comm.receive_json_from()
            result['typing'] = typing

            # Second response is the actual answer — this is a real OpenAI
            # round trip, which routinely exceeds the library's 1s default.
            answer = await comm.receive_json_from(timeout=20)
            result['answer'] = answer

            await comm.disconnect()

        _run(_go())
        return result

    def test_typing_indicator_precedes_response(self):
        res = self._communicate('help')
        self.assertEqual(res['typing']['type'], 'typing')
        self.assertTrue(res['typing']['status'])

    def test_response_has_assistant_role(self):
        res = self._communicate('help')
        self.assertEqual(res['answer']['role'], 'assistant')

    def test_response_includes_intent(self):
        res = self._communicate('help')
        self.assertEqual(res['answer']['intent'], 'help')

    def test_response_content_is_non_empty(self):
        res = self._communicate('give me a summary')
        self.assertGreater(len(res['answer']['content']), 0)

    def test_messages_are_persisted_to_db(self):
        self._communicate('how many facilities')
        session = ChatSession.objects.filter(user=self.user).first()
        self.assertIsNotNone(session)
        messages = list(session.messages.all())
        # User query + assistant response
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, 'user')
        self.assertEqual(messages[1].role, 'assistant')

    def test_empty_message_does_not_create_db_record(self):
        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # session
            await comm.receive_json_from()  # welcome
            await comm.send_json_to({'message': '   '})
            # Consumer should silently discard blank input; no response queued
            self.assertTrue(await comm.receive_nothing())
            await comm.disconnect()

        _run(_go())

    def test_invalid_json_does_not_crash_consumer(self):
        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # session
            await comm.receive_json_from()  # welcome
            await comm.send_to(text_data='NOT VALID JSON')
            self.assertTrue(await comm.receive_nothing())
            await comm.disconnect()

        _run(_go())


class ConsumerAccessDenialTests(TransactionTestCase):
    """Topics outside a user's role produce an informative denial message."""

    def setUp(self):
        self.user = _make_user('denied_user', role=ROLE_FACILITY_ADMIN)

    def test_financials_denied_for_facility_admin(self):
        result = {}

        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # session
            await comm.receive_json_from()  # welcome

            await comm.send_json_to({'message': 'show revenue breakdown'})
            await comm.receive_json_from()  # typing
            answer = await comm.receive_json_from(timeout=20)
            result['answer'] = answer
            await comm.disconnect()

        _run(_go())
        content = result['answer']['content'].lower()
        self.assertIn('access level', content)

    def test_denial_message_lists_allowed_topics(self):
        result = {}

        async def _go():
            app = AnalyticsChatConsumer.as_asgi()
            comm = WebsocketCommunicator(app, '/ws/analytics/chat/')
            comm.scope['user'] = self.user
            await comm.connect()
            await comm.receive_json_from()  # session
            await comm.receive_json_from()  # welcome

            await comm.send_json_to({'message': 'financial data please'})
            await comm.receive_json_from()  # typing
            answer = await comm.receive_json_from(timeout=20)
            result['answer'] = answer
            await comm.disconnect()

        _run(_go())
        # Should suggest what the user CAN ask about
        self.assertIn('patients', result['answer']['content'].lower())


# =============================================================================
# VIEW TESTS
# =============================================================================

class ChatHistoryViewTests(TestCase):
    """GET /analytics/chat/history/ returns session messages."""

    def setUp(self):
        self.user = _make_user('history_user')
        self.url = reverse('self_service:history')

    def _login(self):
        self.client.force_login(self.user)

    def test_redirects_unauthenticated(self):
        resp = self.client.get(self.url)
        self.assertIn(resp.status_code, [302, 403])

    def test_returns_empty_when_no_sessions(self):
        self._login()
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['messages'], [])

    def test_returns_messages_from_active_session(self):
        self._login()
        session = ChatSession.objects.create(user=self.user)
        ChatMessage.objects.create(
            session=session, role=ChatMessage.ROLE_USER, content='Hello'
        )
        ChatMessage.objects.create(
            session=session, role=ChatMessage.ROLE_ASSISTANT,
            content='Hi there!', query_intent='general'
        )
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(len(data['messages']), 2)
        self.assertEqual(data['messages'][0]['role'], 'user')
        self.assertEqual(data['messages'][1]['role'], 'assistant')

    def test_returns_session_key_in_response(self):
        self._login()
        session = ChatSession.objects.create(user=self.user)
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertEqual(data['session'], str(session.session_key))

    def test_only_returns_own_sessions(self):
        self._login()
        other_user = _make_user('other_hist_user')
        other_session = ChatSession.objects.create(user=other_user)
        ChatMessage.objects.create(
            session=other_session, role=ChatMessage.ROLE_USER, content='Secret'
        )
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertEqual(data['messages'], [])

    def test_returns_most_recent_active_session(self):
        self._login()
        old_session = ChatSession.objects.create(user=self.user, is_active=False)
        ChatMessage.objects.create(
            session=old_session, role=ChatMessage.ROLE_USER, content='Old message'
        )
        new_session = ChatSession.objects.create(user=self.user, is_active=True)
        ChatMessage.objects.create(
            session=new_session, role=ChatMessage.ROLE_USER, content='New message'
        )
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        # Should return only the active session's messages
        contents = [m['content'] for m in data['messages']]
        self.assertIn('New message', contents)
        self.assertNotIn('Old message', contents)

    def test_content_type_is_json(self):
        self._login()
        resp = self.client.get(self.url)
        self.assertEqual(resp['Content-Type'], 'application/json')

    def test_session_param_returns_that_specific_session(self):
        self._login()
        older = ChatSession.objects.create(user=self.user, title='Older chat')
        ChatMessage.objects.create(session=older, role=ChatMessage.ROLE_USER, content='Old one')
        newer = ChatSession.objects.create(user=self.user, title='Newer chat')
        ChatMessage.objects.create(session=newer, role=ChatMessage.ROLE_USER, content='New one')

        resp = self.client.get(self.url, {'session': str(older.session_key)})
        data = json.loads(resp.content)
        self.assertEqual(data['session'], str(older.session_key))
        self.assertEqual([m['content'] for m in data['messages']], ['Old one'])

    def test_session_param_owned_by_another_user_returns_404(self):
        self._login()
        other_user = _make_user('history_other_user')
        other_session = ChatSession.objects.create(user=other_user)
        resp = self.client.get(self.url, {'session': str(other_session.session_key)})
        self.assertEqual(resp.status_code, 404)

    def test_malformed_session_param_returns_404(self):
        self._login()
        resp = self.client.get(self.url, {'session': 'not-a-uuid'})
        self.assertEqual(resp.status_code, 404)

    def test_session_param_response_includes_title(self):
        self._login()
        session = ChatSession.objects.create(user=self.user, title='My conversation')
        resp = self.client.get(self.url, {'session': str(session.session_key)})
        data = json.loads(resp.content)
        self.assertEqual(data['title'], 'My conversation')


class ChatSessionListViewTests(TestCase):
    """GET /analytics/chat/sessions/ lists the caller's own conversations."""

    def setUp(self):
        self.user = _make_user('list_user')
        self.url = reverse('self_service:sessions')

    def _login(self):
        self.client.force_login(self.user)

    def test_redirects_unauthenticated(self):
        resp = self.client.get(self.url)
        self.assertIn(resp.status_code, [302, 403])

    def test_returns_empty_list_when_no_sessions(self):
        self._login()
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data['sessions'], [])

    def test_excludes_other_users_sessions(self):
        self._login()
        other_user = _make_user('list_other_user')
        ChatSession.objects.create(user=other_user, title='Not mine')
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertEqual(data['sessions'], [])

    def test_orders_by_most_recently_active_first(self):
        self._login()
        first = ChatSession.objects.create(user=self.user, title='First')
        second = ChatSession.objects.create(user=self.user, title='Second')
        # Touch `first` so it becomes the most recently active.
        first.save(update_fields=['last_activity'])

        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        keys = [s['session_key'] for s in data['sessions']]
        self.assertEqual(keys, [str(first.session_key), str(second.session_key)])

    def test_includes_title_preview_and_message_count(self):
        self._login()
        session = ChatSession.objects.create(user=self.user, title='Patient counts')
        ChatMessage.objects.create(session=session, role=ChatMessage.ROLE_USER, content='How many patients?')
        ChatMessage.objects.create(session=session, role=ChatMessage.ROLE_ASSISTANT, content='You have 42 patients.')

        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertEqual(len(data['sessions']), 1)
        entry = data['sessions'][0]
        self.assertEqual(entry['title'], 'Patient counts')
        self.assertEqual(entry['preview'], 'You have 42 patients.')
        self.assertEqual(entry['message_count'], 2)

    def test_blank_title_defaults_to_new_conversation(self):
        self._login()
        ChatSession.objects.create(user=self.user)
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertEqual(data['sessions'][0]['title'], 'New conversation')


class AccessContextViewTests(TestCase):
    """GET /analytics/chat/access/ returns the user's access context JSON."""

    def setUp(self):
        self.url = reverse('self_service:access')

    def test_redirects_unauthenticated(self):
        resp = self.client.get(self.url)
        self.assertIn(resp.status_code, [302, 403])

    def test_returns_200_for_authenticated_user(self):
        user = _make_user('ctx_view_user', role=ROLE_FACILITY_ADMIN)
        self.client.force_login(user)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_response_contains_role_and_allowed_topics(self):
        user = _make_user('ctx_topic_user', role=ROLE_FACILITIES_ADMIN)
        self.client.force_login(user)
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertIn('role', data)
        self.assertIn('allowed_topics', data)
        self.assertIsInstance(data['allowed_topics'], list)

    def test_denied_and_masked_columns_are_lists(self):
        user = _make_user('ctx_col_user', role=ROLE_FACILITY_ADMIN)
        self.client.force_login(user)
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertIsInstance(data['denied_columns'], list)
        self.assertIsInstance(data['masked_columns'], list)

    def test_facility_admin_has_expected_denied_columns(self):
        user = _make_user('ctx_denied_user', role=ROLE_FACILITY_ADMIN)
        self.client.force_login(user)
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertIn('salary', data['denied_columns'])

    def test_client_admin_has_empty_denied_columns(self):
        user = _make_user('ctx_ca_user', role=ROLE_CLIENT_ADMIN)
        self.client.force_login(user)
        resp = self.client.get(self.url)
        data = json.loads(resp.content)
        self.assertEqual(data['denied_columns'], [])

    def test_content_type_is_json(self):
        user = _make_user('ctx_ct_user', role=ROLE_FACILITY_ADMIN)
        self.client.force_login(user)
        resp = self.client.get(self.url)
        self.assertEqual(resp['Content-Type'], 'application/json')
