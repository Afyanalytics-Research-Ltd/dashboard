"""
Comprehensive tests for the core app.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase, Client as TestClient
from django.urls import reverse

from .models import AuditLog, Client, Facility, Notification, SystemSettings

User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user(username='testuser', password='testpass123', superuser=False):
    if superuser:
        return User.objects.create_superuser(username=username, password=password, email=f'{username}@test.com')
    return User.objects.create_user(username=username, password=password, email=f'{username}@test.com')


def make_client(**kwargs):
    defaults = {'name': 'Test Hospital', 'slug': 'test-hospital', 'is_active': True}
    defaults.update(kwargs)
    return Client.objects.create(**defaults)


def make_facility(client, **kwargs):
    defaults = {'name': 'Main Clinic', 'slug': 'main-clinic', 'is_active': True}
    defaults.update(kwargs)
    return Facility.objects.create(client=client, **defaults)


# ---------------------------------------------------------------------------
# Client model tests
# ---------------------------------------------------------------------------

class ClientModelTests(TestCase):

    def test_create_client(self):
        client = make_client()
        self.assertEqual(client.name, 'Test Hospital')
        self.assertEqual(client.slug, 'test-hospital')
        self.assertTrue(client.is_active)

    def test_str_representation(self):
        client = make_client(name='Nairobi Medical Centre', slug='nairobi-medical')
        self.assertEqual(str(client), 'Nairobi Medical Centre')

    def test_default_ordering(self):
        make_client(name='Zeta Clinic', slug='zeta-clinic')
        make_client(name='Alpha Hospital', slug='alpha-hospital')
        names = list(Client.objects.values_list('name', flat=True))
        self.assertEqual(names, sorted(names))

    def test_slug_uniqueness(self):
        make_client(slug='unique-slug')
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            make_client(name='Another', slug='unique-slug')

    def test_active_facilities_count(self):
        client = make_client()
        make_facility(client, slug='f1')
        make_facility(client, slug='f2')
        make_facility(client, slug='f3', is_active=False)
        self.assertEqual(client.active_facilities_count, 2)

    def test_inactive_client(self):
        client = make_client(is_active=False)
        self.assertFalse(client.is_active)


# ---------------------------------------------------------------------------
# Facility model tests
# ---------------------------------------------------------------------------

class FacilityModelTests(TestCase):

    def setUp(self):
        self.client_obj = make_client()

    def test_create_facility(self):
        facility = make_facility(self.client_obj)
        self.assertEqual(facility.name, 'Main Clinic')
        self.assertEqual(facility.client, self.client_obj)

    def test_str_representation(self):
        facility = make_facility(self.client_obj)
        self.assertIn('Test Hospital', str(facility))
        self.assertIn('Main Clinic', str(facility))

    def test_unique_together_constraint(self):
        make_facility(self.client_obj, slug='same-slug')
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            make_facility(self.client_obj, slug='same-slug', name='Other')

    def test_same_slug_different_clients(self):
        client2 = make_client(name='Second Hospital', slug='second-hospital')
        f1 = make_facility(self.client_obj, slug='shared-slug')
        f2 = make_facility(client2, slug='shared-slug')
        self.assertNotEqual(f1.pk, f2.pk)

    def test_cascade_delete(self):
        facility = make_facility(self.client_obj)
        facility_pk = facility.pk
        self.client_obj.delete()
        self.assertFalse(Facility.objects.filter(pk=facility_pk).exists())


# ---------------------------------------------------------------------------
# AuditLog tests
# ---------------------------------------------------------------------------

class AuditLogTests(TestCase):

    def setUp(self):
        self.user = make_user()

    def test_create_audit_log(self):
        log = AuditLog.log(
            user=self.user,
            action='create',
            resource='Dashboard',
            resource_id='42',
            detail='Created new dashboard',
            ip_address='127.0.0.1',
        )
        self.assertEqual(log.action, 'create')
        self.assertEqual(log.resource, 'Dashboard')
        self.assertEqual(log.user, self.user)
        self.assertIsNotNone(log.timestamp)

    def test_log_with_anonymous_user(self):
        from django.contrib.auth.models import AnonymousUser
        anon = AnonymousUser()
        log = AuditLog.log(user=anon, action='read', resource='Public')
        self.assertIsNone(log.user)

    def test_str_representation(self):
        log = AuditLog.log(user=self.user, action='login', resource='Auth')
        self.assertIn('testuser', str(log))
        self.assertIn('login', str(log))

    def test_ordering_newest_first(self):
        for i in range(3):
            AuditLog.log(user=self.user, action='read', resource=f'Resource{i}')
        logs = list(AuditLog.objects.all())
        timestamps = [l.timestamp for l in logs]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))

    def test_action_choices(self):
        valid_actions = [a[0] for a in AuditLog.ACTION_CHOICES]
        for action in valid_actions:
            log = AuditLog.log(user=self.user, action=action, resource='Test')
            self.assertEqual(log.action, action)


# ---------------------------------------------------------------------------
# Notification tests
# ---------------------------------------------------------------------------

class NotificationTests(TestCase):

    def setUp(self):
        self.user = make_user()
        self.other_user = make_user(username='other')

    def test_create_notification(self):
        notif = Notification.send(
            user=self.user,
            title='Welcome',
            message='Welcome to Afya DataHub!',
        )
        self.assertEqual(notif.title, 'Welcome')
        self.assertFalse(notif.is_read)
        self.assertEqual(notif.notification_type, 'info')

    def test_mark_read(self):
        notif = Notification.send(user=self.user, title='Test', message='Body')
        self.assertFalse(notif.is_read)
        notif.mark_read()
        notif.refresh_from_db()
        self.assertTrue(notif.is_read)

    def test_filter_by_user(self):
        Notification.send(user=self.user, title='For me', message='...')
        Notification.send(user=self.other_user, title='For other', message='...')
        mine = Notification.objects.filter(user=self.user)
        self.assertEqual(mine.count(), 1)
        self.assertEqual(mine.first().title, 'For me')

    def test_unread_count(self):
        Notification.send(user=self.user, title='A', message='...')
        Notification.send(user=self.user, title='B', message='...')
        n = Notification.send(user=self.user, title='C', message='...')
        n.mark_read()
        unread = Notification.objects.filter(user=self.user, is_read=False).count()
        self.assertEqual(unread, 2)

    def test_str_representation(self):
        notif = Notification.send(user=self.user, title='Alert', message='...')
        self.assertIn('testuser', str(notif))
        self.assertIn('Alert', str(notif))

    def test_notification_types(self):
        for ntype in ('info', 'success', 'warning', 'danger'):
            n = Notification.send(user=self.user, title='T', message='M', notification_type=ntype)
            self.assertEqual(n.notification_type, ntype)


# ---------------------------------------------------------------------------
# SystemSettings tests
# ---------------------------------------------------------------------------

class SystemSettingsTests(TestCase):

    def setUp(self):
        self.user = make_user(superuser=True)

    def test_set_and_get(self):
        SystemSettings.set('feature_x', True, user=self.user)
        value = SystemSettings.get('feature_x')
        self.assertTrue(value)

    def test_get_missing_key(self):
        result = SystemSettings.get('nonexistent_key', default='fallback')
        self.assertEqual(result, 'fallback')

    def test_update_existing(self):
        SystemSettings.set('my_key', 'original')
        SystemSettings.set('my_key', 'updated')
        self.assertEqual(SystemSettings.get('my_key'), 'updated')
        self.assertEqual(SystemSettings.objects.filter(key='my_key').count(), 1)

    def test_key_uniqueness(self):
        SystemSettings.objects.create(key='unique_key', value={'x': 1})
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            SystemSettings.objects.create(key='unique_key', value={'x': 2})

    def test_str_representation(self):
        obj = SystemSettings.objects.create(key='test_setting', value={})
        self.assertEqual(str(obj), 'test_setting')


# ---------------------------------------------------------------------------
# Core API tests
# ---------------------------------------------------------------------------

class CoreAPITests(TestCase):

    def setUp(self):
        self.client = TestClient()
        self.user = make_user()
        self.admin = make_user(username='admin', superuser=True)
        self.client_obj = make_client()

    def _auth(self, user):
        self.client.force_login(user)

    def test_clients_list_requires_auth(self):
        resp = self.client.get('/api/v1/core/clients/')
        self.assertIn(resp.status_code, [401, 403])

    def test_clients_list_authenticated(self):
        self._auth(self.user)
        resp = self.client.get('/api/v1/core/clients/')
        self.assertEqual(resp.status_code, 200)

    def test_client_detail(self):
        self._auth(self.user)
        resp = self.client.get(f'/api/v1/core/clients/{self.client_obj.pk}/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data['name'], 'Test Hospital')

    def test_create_client_requires_admin(self):
        self._auth(self.user)
        resp = self.client.post('/api/v1/core/clients/', {'name': 'New', 'slug': 'new'})
        self.assertIn(resp.status_code, [403, 405])

    def test_notifications_api(self):
        Notification.send(user=self.user, title='Test', message='Body')
        self._auth(self.user)
        resp = self.client.get('/api/v1/core/notifications/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data['count'], 1)

    def test_audit_log_own_entries(self):
        AuditLog.log(user=self.user, action='read', resource='Dashboard')
        AuditLog.log(user=self.admin, action='delete', resource='Setting')
        self._auth(self.user)
        resp = self.client.get('/api/v1/core/audit-logs/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Regular user sees only their own logs
        for entry in data['results']:
            self.assertEqual(entry['username'], 'testuser')

    def test_superuser_sees_all_audit_logs(self):
        AuditLog.log(user=self.user, action='read', resource='Dashboard')
        AuditLog.log(user=self.admin, action='delete', resource='Setting')
        self._auth(self.admin)
        resp = self.client.get('/api/v1/core/audit-logs/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data['count'], 2)

    def test_system_settings_superuser_only_write(self):
        self._auth(self.user)
        resp = self.client.post(
            '/api/v1/core/system-settings/',
            {'key': 'k', 'value': 'v'},
            content_type='application/json',
        )
        self.assertIn(resp.status_code, [403, 405])


# ---------------------------------------------------------------------------
# Permission tests
# ---------------------------------------------------------------------------

class PermissionsTests(TestCase):

    def setUp(self):
        self.client = TestClient()
        self.user = make_user()
        self.superuser = make_user(username='super', superuser=True)

    def test_settings_page_superuser_only(self):
        self.client.force_login(self.user)
        resp = self.client.get('/core/settings/')
        self.assertEqual(resp.status_code, 403)

    def test_settings_page_accessible_to_superuser(self):
        self.client.force_login(self.superuser)
        resp = self.client.get('/core/settings/')
        self.assertEqual(resp.status_code, 200)

    def test_notifications_page_requires_login(self):
        resp = self.client.get('/core/notifications/')
        self.assertEqual(resp.status_code, 302)

    def test_notifications_page_authenticated(self):
        self.client.force_login(self.user)
        resp = self.client.get('/core/notifications/')
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# PermissionsView — facility-scoped user management
# ---------------------------------------------------------------------------

class PermissionsViewTests(TestCase):

    def setUp(self):
        from django.contrib.auth.models import Group

        from authentication.roles import ROLE_CLIENT_ADMIN, ROLE_FACILITY_ADMIN

        def _set_role(user, role):
            # RoleRequiredMixin checks Django Group membership (authentication.roles.in_role),
            # so the test user needs the group, not just profile.role, to pass the view's gate.
            user.profile.role = role
            user.profile.save()
            group, _ = Group.objects.get_or_create(name=role)
            user.groups.add(group)

        self.client_org = make_client(name='Perm Test Hosp', slug='perm-test-hosp')
        self.facility_a = make_facility(self.client_org, name='Facility A', slug='facility-a')
        self.facility_b = make_facility(self.client_org, name='Facility B', slug='facility-b')

        self.facility_admin = make_user('perm_facility_admin')
        _set_role(self.facility_admin, ROLE_FACILITY_ADMIN)
        self.facility_admin.profile.facility = self.facility_a
        self.facility_admin.profile.client = self.client_org
        self.facility_admin.profile.save()

        self.same_facility_user = make_user('perm_same_facility')
        _set_role(self.same_facility_user, ROLE_FACILITY_ADMIN)
        self.same_facility_user.profile.facility = self.facility_a
        self.same_facility_user.profile.client = self.client_org
        self.same_facility_user.profile.save()

        self.other_facility_user = make_user('perm_other_facility')
        _set_role(self.other_facility_user, ROLE_FACILITY_ADMIN)
        self.other_facility_user.profile.facility = self.facility_b
        self.other_facility_user.profile.client = self.client_org
        self.other_facility_user.profile.save()

        self.client_admin = make_user('perm_client_admin')
        _set_role(self.client_admin, ROLE_CLIENT_ADMIN)
        self.client_admin.profile.client = self.client_org
        self.client_admin.profile.save()

        self.c = TestClient()

    def test_requires_login(self):
        resp = self.c.get(reverse('core:permissions'))
        self.assertEqual(resp.status_code, 302)

    def test_facility_admin_sees_only_own_facility_users(self):
        self.c.force_login(self.facility_admin)
        resp = self.c.get(reverse('core:permissions'))
        self.assertEqual(resp.status_code, 200)
        managed = resp.context['managed_users']
        self.assertIn(self.same_facility_user, managed)
        self.assertNotIn(self.other_facility_user, managed)
        self.assertNotIn(self.facility_admin, managed)  # never manages self

    def test_client_admin_sees_whole_client(self):
        self.c.force_login(self.client_admin)
        resp = self.c.get(reverse('core:permissions'))
        managed = resp.context['managed_users']
        self.assertIn(self.same_facility_user, managed)
        self.assertIn(self.other_facility_user, managed)
        self.assertIn(self.facility_admin, managed)

    def test_facility_admin_can_grant_module_to_own_user(self):
        self.c.force_login(self.facility_admin)
        resp = self.c.post(reverse('core:permissions'), {
            'action': 'set_module',
            'user_id': self.same_facility_user.pk,
            'module_key': 'warehouse',
            'is_granted': 'true',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()['ok'])
        from authentication.module_access import has_module_access
        self.assertTrue(has_module_access(self.same_facility_user, 'warehouse'))

    def test_facility_admin_cannot_modify_other_facility_user(self):
        self.c.force_login(self.facility_admin)
        resp = self.c.post(reverse('core:permissions'), {
            'action': 'set_module',
            'user_id': self.other_facility_user.pk,
            'module_key': 'warehouse',
            'is_granted': 'true',
        })
        self.assertEqual(resp.status_code, 404)

    def test_clear_module_reverts_to_default(self):
        from authentication.models import UserModuleGrant
        from authentication.module_access import has_module_access

        UserModuleGrant.objects.create(
            user=self.same_facility_user, module_key='warehouse', is_granted=True,
        )
        self.c.force_login(self.facility_admin)
        resp = self.c.post(reverse('core:permissions'), {
            'action': 'clear_module',
            'user_id': self.same_facility_user.pk,
            'module_key': 'warehouse',
        })
        self.assertTrue(resp.json()['ok'])
        self.assertFalse(has_module_access(self.same_facility_user, 'warehouse'))  # back to role default

    def test_toggle_dashboard_hides_for_target_user(self):
        from analytics_app.models import Dashboard

        dashboard = Dashboard.objects.create(
            name='Perm Test Dashboard', slug='perm-test-dashboard',
            client=self.client_org, streamlit_url='http://localhost:8501/?d=1',
        )
        self.c.force_login(self.facility_admin)
        resp = self.c.post(reverse('core:permissions'), {
            'action': 'toggle_dashboard',
            'user_id': self.same_facility_user.pk,
            'dashboard_id': dashboard.pk,
            'hidden': 'true',
        })
        self.assertTrue(resp.json()['ok'])
        self.assertTrue(
            dashboard.hidden_from_users.filter(pk=self.same_facility_user.pk).exists()
        )


# ---------------------------------------------------------------------------
# Ticket / TicketComment model tests
# ---------------------------------------------------------------------------

class TicketModelTests(TestCase):

    def setUp(self):
        self.user = make_user('ticket_reporter')
        self.staff = make_user('ticket_staff')
        self.staff.is_staff = True
        self.staff.save()

    def test_create_ticket(self):
        from .models import Ticket
        t = Ticket.objects.create(
            ticket_type=Ticket.TYPE_ISSUE, subject='Login broken', description='Cannot log in.',
            created_by=self.user,
        )
        self.assertEqual(t.status, Ticket.STATUS_OPEN)
        self.assertEqual(t.priority, Ticket.PRIORITY_MEDIUM)
        self.assertIn('Login broken', str(t))

    def test_status_color_and_type_icon(self):
        from .models import Ticket
        t = Ticket.objects.create(ticket_type=Ticket.TYPE_SUGGESTION, subject='X', description='Y', created_by=self.user)
        self.assertEqual(t.status_color, 'amber')  # open
        self.assertEqual(t.type_icon, 'bi-lightbulb-fill')

    def test_set_status_stamps_resolved_at_once(self):
        from .models import Ticket
        t = Ticket.objects.create(ticket_type=Ticket.TYPE_ISSUE, subject='X', description='Y', created_by=self.user)
        self.assertIsNone(t.resolved_at)
        t.set_status(Ticket.STATUS_RESOLVED, actor=self.staff)
        self.assertIsNotNone(t.resolved_at)
        first_resolved_at = t.resolved_at
        t.set_status(Ticket.STATUS_CLOSED, actor=self.staff)
        t.set_status(Ticket.STATUS_RESOLVED, actor=self.staff)
        self.assertEqual(t.resolved_at, first_resolved_at)  # not re-stamped

    def test_set_status_notifies_creator_not_self(self):
        from .models import Notification, Ticket
        t = Ticket.objects.create(ticket_type=Ticket.TYPE_ISSUE, subject='X', description='Y', created_by=self.user)
        t.set_status(Ticket.STATUS_IN_PROGRESS, actor=self.staff)
        self.assertTrue(Notification.objects.filter(user=self.user, title__icontains='Ticket updated').exists())

        # Creator changing their own ticket's status should not self-notify
        Notification.objects.all().delete()
        t.set_status(Ticket.STATUS_CLOSED, actor=self.user)
        self.assertFalse(Notification.objects.filter(user=self.user).exists())

    def test_set_status_rejects_unknown_status(self):
        from .models import Ticket
        t = Ticket.objects.create(ticket_type=Ticket.TYPE_ISSUE, subject='X', description='Y', created_by=self.user)
        with self.assertRaises(ValueError):
            t.set_status('not_a_real_status')


# ---------------------------------------------------------------------------
# Ticketing views/API tests
# ---------------------------------------------------------------------------

class TicketingAPITests(TestCase):

    def setUp(self):
        self.user = make_user('tk_user')
        self.other_user = make_user('tk_other_user')
        self.staff = make_user('tk_staff')
        self.staff.is_staff = True
        self.staff.save()
        self.c = TestClient()

    def _create_ticket(self, user=None, **overrides):
        from .models import Ticket
        defaults = dict(
            ticket_type=Ticket.TYPE_ISSUE, subject='Something broke', description='Details here.',
            created_by=user or self.user,
        )
        defaults.update(overrides)
        return Ticket.objects.create(**defaults)

    # ---- TicketCreateAPIView ----

    def test_create_ticket_requires_login(self):
        resp = self.c.post(reverse('core:ticket-create'), {
            'ticket_type': 'issue', 'subject': 'X', 'description': 'Y',
        })
        self.assertEqual(resp.status_code, 302)

    def test_create_ticket_success(self):
        from .models import Ticket
        self.c.force_login(self.user)
        resp = self.c.post(reverse('core:ticket-create'), {
            'ticket_type': 'suggestion', 'subject': 'Add dark mode', 'description': 'Would love a dark theme.',
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['ok'])
        self.assertTrue(Ticket.objects.filter(subject='Add dark mode', created_by=self.user).exists())

    def test_create_ticket_missing_subject_rejected(self):
        self.c.force_login(self.user)
        resp = self.c.post(reverse('core:ticket-create'), {
            'ticket_type': 'issue', 'subject': '', 'description': 'Y',
        })
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.json()['ok'])

    def test_create_ticket_invalid_type_rejected(self):
        self.c.force_login(self.user)
        resp = self.c.post(reverse('core:ticket-create'), {
            'ticket_type': 'not_a_type', 'subject': 'X', 'description': 'Y',
        })
        self.assertEqual(resp.status_code, 400)

    def test_create_ticket_notifies_staff(self):
        from .models import Notification
        self.c.force_login(self.user)
        self.c.post(reverse('core:ticket-create'), {
            'ticket_type': 'issue', 'subject': 'Broken export', 'description': 'It just spins forever.',
        })
        self.assertTrue(Notification.objects.filter(user=self.staff).exists())
        self.assertFalse(Notification.objects.filter(user=self.other_user).exists())

    # ---- SupportView ----

    def test_support_page_requires_login(self):
        resp = self.c.get(reverse('core:support'))
        self.assertEqual(resp.status_code, 302)

    def test_support_page_shows_only_own_tickets_for_regular_user(self):
        mine = self._create_ticket(user=self.user, subject='Mine')
        self._create_ticket(user=self.other_user, subject='Not mine')
        self.c.force_login(self.user)
        resp = self.c.get(reverse('core:support'))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(mine, resp.context['my_tickets'])
        self.assertEqual(len(resp.context['my_tickets']), 1)
        self.assertNotIn('board_columns', resp.context)

    def test_support_page_staff_sees_board(self):
        self._create_ticket(user=self.user)
        self._create_ticket(user=self.other_user)
        self.c.force_login(self.staff)
        resp = self.c.get(reverse('core:support'))
        self.assertIn('board_columns', resp.context)
        total_on_board = sum(col['count'] for col in resp.context['board_columns'])
        self.assertEqual(total_on_board, 2)

    # ---- TicketDetailAPIView ----

    def test_detail_visible_to_creator(self):
        t = self._create_ticket()
        self.c.force_login(self.user)
        resp = self.c.get(reverse('core:ticket-detail', args=[t.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['ticket']['subject'], t.subject)

    def test_detail_visible_to_staff(self):
        t = self._create_ticket()
        self.c.force_login(self.staff)
        resp = self.c.get(reverse('core:ticket-detail', args=[t.pk]))
        self.assertEqual(resp.status_code, 200)

    def test_detail_hidden_from_other_users(self):
        t = self._create_ticket()
        self.c.force_login(self.other_user)
        resp = self.c.get(reverse('core:ticket-detail', args=[t.pk]))
        self.assertEqual(resp.status_code, 404)

    def test_detail_hides_internal_comments_from_creator(self):
        from .models import TicketComment
        t = self._create_ticket()
        TicketComment.objects.create(ticket=t, author=self.staff, body='Internal note', is_internal=True)
        TicketComment.objects.create(ticket=t, author=self.staff, body='Public reply', is_internal=False)

        self.c.force_login(self.user)
        resp = self.c.get(reverse('core:ticket-detail', args=[t.pk]))
        bodies = [c['body'] for c in resp.json()['comments']]
        self.assertNotIn('Internal note', bodies)
        self.assertIn('Public reply', bodies)

        self.c.force_login(self.staff)
        resp = self.c.get(reverse('core:ticket-detail', args=[t.pk]))
        bodies = [c['body'] for c in resp.json()['comments']]
        self.assertIn('Internal note', bodies)

    # ---- TicketCommentAPIView ----

    def test_creator_can_comment_on_own_ticket(self):
        t = self._create_ticket()
        self.c.force_login(self.user)
        resp = self.c.post(reverse('core:ticket-comment', args=[t.pk]), {'body': 'Any update?'})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()['ok'])

    def test_other_user_cannot_comment(self):
        t = self._create_ticket()
        self.c.force_login(self.other_user)
        resp = self.c.post(reverse('core:ticket-comment', args=[t.pk]), {'body': 'Butting in'})
        self.assertEqual(resp.status_code, 404)

    def test_non_staff_cannot_force_internal_comment(self):
        from .models import TicketComment
        t = self._create_ticket()
        self.c.force_login(self.user)
        self.c.post(reverse('core:ticket-comment', args=[t.pk]), {'body': 'Trying to be sneaky', 'is_internal': 'true'})
        comment = TicketComment.objects.get(body='Trying to be sneaky')
        self.assertFalse(comment.is_internal)  # ignored for non-staff

    def test_comment_notifies_creator(self):
        from .models import Notification
        t = self._create_ticket()
        self.c.force_login(self.staff)
        self.c.post(reverse('core:ticket-comment', args=[t.pk]), {'body': 'Looking into it now.'})
        self.assertTrue(Notification.objects.filter(user=self.user, title__icontains='New reply').exists())

    # ---- TicketStatusAPIView ----

    def test_status_update_requires_staff(self):
        t = self._create_ticket()
        self.c.force_login(self.user)
        resp = self.c.post(reverse('core:ticket-status', args=[t.pk]), {'status': 'resolved'})
        self.assertEqual(resp.status_code, 403)

    def test_staff_can_update_status(self):
        from .models import Ticket
        t = self._create_ticket()
        self.c.force_login(self.staff)
        resp = self.c.post(reverse('core:ticket-status', args=[t.pk]), {'status': 'resolved'})
        self.assertEqual(resp.status_code, 200)
        t.refresh_from_db()
        self.assertEqual(t.status, Ticket.STATUS_RESOLVED)
        self.assertIsNotNone(t.resolved_at)

    def test_status_update_rejects_unknown_value(self):
        t = self._create_ticket()
        self.c.force_login(self.staff)
        resp = self.c.post(reverse('core:ticket-status', args=[t.pk]), {'status': 'bogus'})
        self.assertEqual(resp.status_code, 400)
