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
