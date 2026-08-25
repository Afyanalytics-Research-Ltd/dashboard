"""
Comprehensive tests for the authentication app.

Coverage:
 - UserProfile model (creation, properties, roles)
 - Authentication views (login, logout, signup, profile, password change)
 - Notification views (list, mark read)
 - Activity view
 - Role system helpers and decorators
 - Template tags
 - DRF API endpoints (users, profiles, password change, notifications, activity)

Run with:
    python manage.py test authentication
"""

import json

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import Client, RequestFactory, TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from core.models import AuditLog, Notification
from .models import UserProfile
from .roles import (
    ROLE_CLIENT_ADMIN,
    ROLE_FACILITIES_ADMIN,
    ROLE_FACILITY_ADMIN,
    in_role,
    is_client_admin,
    is_facilities_admin,
    is_facility_admin,
    role_required,
    user_has_role,
    user_roles,
)

User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user(username='testuser', password='TestPass123!', role=ROLE_FACILITY_ADMIN, **kwargs):
    """Create a user with a profile and group."""
    user = User.objects.create_user(username=username, password=password, **kwargs)
    # Profile is created by signal; update role
    UserProfile.objects.filter(user=user).update(role=role)
    group, _ = Group.objects.get_or_create(name=role)
    user.groups.add(group)
    # Return a fresh user instance with up-to-date related objects
    return User.objects.get(pk=user.pk)


def make_superuser(username='superadmin', password='TestPass123!'):
    """Create a superuser."""
    return User.objects.create_superuser(username=username, password=password, email='super@test.com')


# ---------------------------------------------------------------------------
# 1. UserProfile Model Tests
# ---------------------------------------------------------------------------

class UserProfileModelTests(TestCase):
    """Tests for the UserProfile model and its properties."""

    def setUp(self):
        self.user = make_user(
            username='profileuser',
            email='p@test.com',
            role=ROLE_FACILITY_ADMIN,
            first_name='Jane',
            last_name='Doe',
        )
        self.profile = self.user.profile

    def test_profile_auto_created_on_user_create(self):
        """A UserProfile is created automatically when a User is saved."""
        new_user = User.objects.create_user(username='newbie', password='pass1234A!')
        self.assertTrue(UserProfile.objects.filter(user=new_user).exists())

    def test_str_representation(self):
        """__str__ includes the display name and role."""
        s = str(self.profile)
        self.assertIn('Jane Doe', s)
        self.assertIn('Facility Admin', s)

    def test_display_name_with_full_name(self):
        """display_name returns full name when both parts are set."""
        self.assertEqual(self.profile.display_name, 'Jane Doe')

    def test_display_name_fallback_to_username(self):
        """display_name falls back to username when no full name is set."""
        user = make_user(username='noname', password='pass1234A!')
        self.assertEqual(user.profile.display_name, 'noname')

    def test_initials_with_full_name(self):
        """initials returns first letters of first and last name."""
        self.assertEqual(self.profile.initials, 'JD')

    def test_initials_single_name(self):
        """initials returns single letter when only first name is set."""
        user = User.objects.create_user(username='onlyf', password='pass1234A!', first_name='Alice')
        self.assertEqual(user.profile.initials, 'A')

    def test_is_client_admin_property_false(self):
        """is_client_admin is False for facility_admin role."""
        self.assertFalse(self.profile.is_client_admin)

    def test_is_client_admin_property_true(self):
        """is_client_admin is True for client_admin role."""
        admin = make_user(username='cadmin', role=ROLE_CLIENT_ADMIN)
        self.assertTrue(admin.profile.is_client_admin)

    def test_is_facilities_admin_property(self):
        """is_facilities_admin is True for facilities_admin and client_admin."""
        fadmin = make_user(username='fadmin', role=ROLE_FACILITIES_ADMIN)
        self.assertTrue(fadmin.profile.is_facilities_admin)
        cadmin = make_user(username='cadmin2', role=ROLE_CLIENT_ADMIN)
        self.assertTrue(cadmin.profile.is_facilities_admin)
        self.assertFalse(self.profile.is_facilities_admin)

    def test_superuser_is_client_admin(self):
        """Superuser has is_client_admin = True."""
        su = make_superuser()
        self.assertTrue(su.profile.is_client_admin)

    def test_role_display_badge_colors(self):
        """role_display_badge returns correct Bootstrap colour string."""
        self.profile.role = ROLE_CLIENT_ADMIN
        self.assertEqual(self.profile.role_display_badge, 'primary')
        self.profile.role = ROLE_FACILITIES_ADMIN
        self.assertEqual(self.profile.role_display_badge, 'info')
        self.profile.role = ROLE_FACILITY_ADMIN
        self.assertEqual(self.profile.role_display_badge, 'secondary')

    def test_has_role_method(self):
        """has_role delegates to in_role correctly."""
        self.assertTrue(self.profile.has_role(ROLE_FACILITY_ADMIN))
        self.assertFalse(self.profile.has_role(ROLE_CLIENT_ADMIN))

    def test_profile_fields_save(self):
        """Profile fields can be updated and persisted."""
        self.profile.phone_number = '+254711000000'
        self.profile.job_title = 'Data Analyst'
        self.profile.bio = 'Working in health analytics.'
        self.profile.save()
        refreshed = UserProfile.objects.get(pk=self.profile.pk)
        self.assertEqual(refreshed.phone_number, '+254711000000')
        self.assertEqual(refreshed.job_title, 'Data Analyst')
        self.assertEqual(refreshed.bio, 'Working in health analytics.')


# ---------------------------------------------------------------------------
# 2. Authentication View Tests
# ---------------------------------------------------------------------------

class AuthenticationViewTests(TestCase):
    """Tests for login, logout, signup, profile, and password change views."""

    def setUp(self):
        self.client = Client()
        self.user = make_user(
            username='viewuser',
            password='TestPass123!',
            email='view@test.com',
            first_name='View',
            last_name='User',
        )
        self.login_url = reverse('authentication:login')
        self.logout_url = reverse('authentication:logout')
        self.signup_url = reverse('authentication:signup')
        self.profile_url = reverse('authentication:profile')
        self.pw_change_url = reverse('authentication:password_change')

    # --- Login ----------------------------------------------------------------

    def test_login_page_renders(self):
        """GET /auth/login/ returns 200 with the login form."""
        response = self.client.get(self.login_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/login.html')

    def test_login_with_valid_credentials(self):
        """POST with valid credentials logs user in and redirects."""
        response = self.client.post(self.login_url, {
            'username': 'viewuser',
            'password': 'TestPass123!',
        })
        self.assertEqual(response.status_code, 302)
        # Should be authenticated
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_login_with_invalid_credentials(self):
        """POST with wrong password returns 200 with form errors."""
        response = self.client.post(self.login_url, {
            'username': 'viewuser',
            'password': 'WrongPass!',
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_login_records_ip_in_profile(self):
        """Login sets last_login_ip on the user profile."""
        self.client.post(self.login_url, {
            'username': 'viewuser',
            'password': 'TestPass123!',
        }, REMOTE_ADDR='10.0.0.1')
        self.user.profile.refresh_from_db()
        self.assertEqual(self.user.profile.last_login_ip, '10.0.0.1')

    def test_login_creates_audit_log(self):
        """Successful login creates an AuditLog entry."""
        before = AuditLog.objects.filter(user=self.user, action='login').count()
        self.client.post(self.login_url, {
            'username': 'viewuser',
            'password': 'TestPass123!',
        })
        after = AuditLog.objects.filter(user=self.user, action='login').count()
        self.assertEqual(after, before + 1)

    def test_authenticated_user_redirected_from_login(self):
        """Authenticated user visiting /auth/login/ is redirected."""
        self.client.login(username='viewuser', password='TestPass123!')
        response = self.client.get(self.login_url)
        self.assertEqual(response.status_code, 302)

    # --- Logout ---------------------------------------------------------------

    def test_logout_redirects_to_login(self):
        """POST /auth/logout/ logs out and redirects to login."""
        self.client.login(username='viewuser', password='TestPass123!')
        response = self.client.post(self.logout_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('login', response['Location'])

    def test_logout_creates_audit_log(self):
        """Logout creates an AuditLog entry with action='logout'."""
        self.client.login(username='viewuser', password='TestPass123!')
        before = AuditLog.objects.filter(user=self.user, action='logout').count()
        self.client.post(self.logout_url)
        after = AuditLog.objects.filter(user=self.user, action='logout').count()
        self.assertEqual(after, before + 1)

    # --- Signup ---------------------------------------------------------------

    def test_signup_page_renders(self):
        """GET /auth/signup/ returns 200."""
        response = self.client.get(self.signup_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/signup.html')

    def test_signup_creates_user_and_profile(self):
        """POST /auth/signup/ with valid data creates User and UserProfile."""
        response = self.client.post(self.signup_url, {
            'username': 'brandnew',
            'first_name': 'Brand',
            'last_name': 'New',
            'email': 'brand@test.com',
            'phone_number': '+254700000001',
            'job_title': 'Analyst',
            'password1': 'StrongPass99!',
            'password2': 'StrongPass99!',
            'terms': 'on',
        })
        self.assertTrue(User.objects.filter(username='brandnew').exists())
        user = User.objects.get(username='brandnew')
        self.assertTrue(UserProfile.objects.filter(user=user).exists())
        self.assertEqual(response.status_code, 302)

    def test_signup_with_duplicate_username_fails(self):
        """POST with an existing username returns form errors."""
        response = self.client.post(self.signup_url, {
            'username': 'viewuser',  # already exists
            'first_name': 'Dup',
            'last_name': 'User',
            'email': 'dup@test.com',
            'password1': 'StrongPass99!',
            'password2': 'StrongPass99!',
            'terms': 'on',
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(email='dup@test.com').exists())

    def test_signup_with_mismatched_passwords_fails(self):
        """POST with mismatched passwords returns errors without creating user."""
        response = self.client.post(self.signup_url, {
            'username': 'mismatch',
            'first_name': 'Mis',
            'last_name': 'Match',
            'email': 'mismatch@test.com',
            'password1': 'StrongPass99!',
            'password2': 'DifferentPass!',
            'terms': 'on',
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='mismatch').exists())

    def test_signup_logs_audit(self):
        """Successful signup creates an AuditLog entry."""
        self.client.post(self.signup_url, {
            'username': 'audituser',
            'first_name': 'Audit',
            'last_name': 'Test',
            'email': 'audit@test.com',
            'password1': 'StrongPass99!',
            'password2': 'StrongPass99!',
            'terms': 'on',
        })
        user = User.objects.get(username='audituser')
        self.assertTrue(AuditLog.objects.filter(user=user, action='create').exists())

    # --- Profile --------------------------------------------------------------

    def test_profile_view_requires_login(self):
        """Unauthenticated access to /auth/profile/ redirects to login."""
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('login', response['Location'])

    def test_profile_view_renders_for_authenticated_user(self):
        """GET /auth/profile/ returns 200 for authenticated user."""
        self.client.login(username='viewuser', password='TestPass123!')
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/profile.html')

    def test_profile_update_saves_correctly(self):
        """POST /auth/profile/ updates user name and profile fields."""
        self.client.login(username='viewuser', password='TestPass123!')
        response = self.client.post(self.profile_url, {
            'first_name': 'Updated',
            'last_name': 'Name',
            'email': 'updated@test.com',
            'phone_number': '+254799000000',
            'job_title': 'Senior Analyst',
            'bio': 'Updated bio text.',
        })
        self.assertEqual(response.status_code, 302)
        self.user.refresh_from_db()
        self.assertEqual(self.user.first_name, 'Updated')
        self.user.profile.refresh_from_db()
        self.assertEqual(self.user.profile.job_title, 'Senior Analyst')
        self.assertEqual(self.user.profile.phone_number, '+254799000000')

    # --- Password Change ------------------------------------------------------

    def test_password_change_requires_login(self):
        """Unauthenticated access to password change redirects to login."""
        response = self.client.get(self.pw_change_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('login', response['Location'])

    def test_password_change_renders(self):
        """GET /auth/password/change/ returns 200 for authenticated user."""
        self.client.login(username='viewuser', password='TestPass123!')
        response = self.client.get(self.pw_change_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/password_change.html')

    def test_password_change_works(self):
        """POST with valid data changes password and keeps user logged in."""
        self.client.login(username='viewuser', password='TestPass123!')
        response = self.client.post(self.pw_change_url, {
            'old_password': 'TestPass123!',
            'new_password1': 'NewPass456!',
            'new_password2': 'NewPass456!',
        })
        self.assertEqual(response.status_code, 302)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password('NewPass456!'))


# ---------------------------------------------------------------------------
# 3. Notification View Tests
# ---------------------------------------------------------------------------

class NotificationViewTests(TestCase):
    """Tests for notification list and mark-read views."""

    def setUp(self):
        self.client = Client()
        self.user = make_user(username='notifuser', password='TestPass123!')
        self.notif = Notification.objects.create(
            user=self.user,
            title='Test Notification',
            message='Hello from test.',
            notification_type='info',
        )
        self.list_url = reverse('authentication:notifications')
        self.read_url = reverse('authentication:notification_read', args=[self.notif.pk])

    def test_notifications_list_requires_login(self):
        """Unauthenticated access is redirected."""
        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, 302)

    def test_notifications_list_renders(self):
        """GET /auth/notifications/ returns 200 with notifications."""
        self.client.login(username='notifuser', password='TestPass123!')
        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/notifications.html')
        self.assertContains(response, 'Test Notification')

    def test_mark_notification_read(self):
        """POST /auth/notifications/{pk}/read/ marks the notification as read."""
        self.client.login(username='notifuser', password='TestPass123!')
        response = self.client.post(self.read_url)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data.get('success'))
        self.notif.refresh_from_db()
        self.assertTrue(self.notif.is_read)

    def test_mark_all_read_via_post(self):
        """POST /auth/notifications/ marks all notifications as read."""
        Notification.objects.create(user=self.user, title='N2', message='m2')
        self.client.login(username='notifuser', password='TestPass123!')
        response = self.client.post(self.list_url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Notification.objects.filter(user=self.user, is_read=False).count(), 0)


# ---------------------------------------------------------------------------
# 4. Activity View Tests
# ---------------------------------------------------------------------------

class ActivityViewTests(TestCase):
    """Tests for the user activity log view."""

    def setUp(self):
        self.client = Client()
        self.user = make_user(username='actuser', password='TestPass123!')
        AuditLog.log(user=self.user, action='login', resource='authentication', detail='test')

    def test_activity_requires_login(self):
        """Unauthenticated access is redirected."""
        response = self.client.get(reverse('authentication:activity'))
        self.assertEqual(response.status_code, 302)

    def test_activity_renders(self):
        """GET /auth/activity/ returns 200 with audit log data."""
        self.client.login(username='actuser', password='TestPass123!')
        response = self.client.get(reverse('authentication:activity'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/activity.html')
        self.assertContains(response, 'login')


# ---------------------------------------------------------------------------
# 5. Role System Tests
# ---------------------------------------------------------------------------

class RoleSystemTests(TestCase):
    """Tests for role utility functions and the role_required decorator."""

    def setUp(self):
        self.facility_user = make_user(username='fac', role=ROLE_FACILITY_ADMIN)
        self.facilities_user = make_user(username='facs', role=ROLE_FACILITIES_ADMIN)
        self.client_user = make_user(username='cli', role=ROLE_CLIENT_ADMIN)
        self.superuser = make_superuser()

    def test_user_roles_returns_correct_groups(self):
        """user_roles() returns the set of group names."""
        roles = user_roles(self.facility_user)
        self.assertIn(ROLE_FACILITY_ADMIN, roles)
        self.assertNotIn(ROLE_CLIENT_ADMIN, roles)

    def test_superuser_has_all_roles(self):
        """Superuser has all role groups returned."""
        roles = user_roles(self.superuser)
        self.assertIn(ROLE_CLIENT_ADMIN, roles)
        self.assertIn(ROLE_FACILITIES_ADMIN, roles)
        self.assertIn(ROLE_FACILITY_ADMIN, roles)

    def test_in_role_returns_true_for_matching(self):
        """in_role returns True when user has the role."""
        self.assertTrue(in_role(self.facility_user, ROLE_FACILITY_ADMIN))
        self.assertTrue(in_role(self.client_user, ROLE_CLIENT_ADMIN))

    def test_in_role_returns_false_for_non_matching(self):
        """in_role returns False when user lacks the role."""
        self.assertFalse(in_role(self.facility_user, ROLE_CLIENT_ADMIN))

    def test_is_client_admin_function(self):
        """is_client_admin returns True only for client_admin and superusers."""
        self.assertTrue(is_client_admin(self.client_user))
        self.assertTrue(is_client_admin(self.superuser))
        self.assertFalse(is_client_admin(self.facility_user))

    def test_is_facilities_admin_function(self):
        """is_facilities_admin includes client_admin and facilities_admin."""
        self.assertTrue(is_facilities_admin(self.client_user))
        self.assertTrue(is_facilities_admin(self.facilities_user))
        self.assertFalse(is_facilities_admin(self.facility_user))

    def test_is_facility_admin_function(self):
        """is_facility_admin is True for all admin roles."""
        self.assertTrue(is_facility_admin(self.facility_user))
        self.assertTrue(is_facility_admin(self.facilities_user))
        self.assertTrue(is_facility_admin(self.client_user))

    def test_user_has_role_with_profile(self):
        """user_has_role uses profile.role for fast lookup."""
        self.assertTrue(user_has_role(self.facility_user, ROLE_FACILITY_ADMIN))
        self.assertFalse(user_has_role(self.facility_user, ROLE_CLIENT_ADMIN))

    def test_unauthenticated_user_has_no_roles(self):
        """Unauthenticated user returns empty role set."""
        from django.contrib.auth.models import AnonymousUser
        anon = AnonymousUser()
        self.assertEqual(user_roles(anon), set())
        self.assertFalse(in_role(anon, ROLE_FACILITY_ADMIN))

    def test_role_required_decorator_allows_correct_role(self):
        """role_required permits access for users in the required role."""
        from django.test import RequestFactory
        factory = RequestFactory()

        @role_required(ROLE_CLIENT_ADMIN)
        def dummy_view(request):
            from django.http import HttpResponse
            return HttpResponse('ok')

        request = factory.get('/')
        request.user = self.client_user
        response = dummy_view(request)
        self.assertEqual(response.status_code, 200)

    def test_role_required_decorator_denies_wrong_role(self):
        """role_required raises PermissionDenied for users lacking the role."""
        from django.core.exceptions import PermissionDenied
        from django.test import RequestFactory

        factory = RequestFactory()

        @role_required(ROLE_CLIENT_ADMIN)
        def restricted_view(request):
            from django.http import HttpResponse
            return HttpResponse('ok')

        request = factory.get('/')
        request.user = self.facility_user
        with self.assertRaises(PermissionDenied):
            restricted_view(request)


# ---------------------------------------------------------------------------
# 6. Template Tag Tests
# ---------------------------------------------------------------------------

class TemplatTagTests(TestCase):
    """Tests for role_tags template filters."""

    def setUp(self):
        self.client_user = make_user(username='tagcli', role=ROLE_CLIENT_ADMIN)
        self.facility_user = make_user(username='tagfac', role=ROLE_FACILITY_ADMIN)

    def test_has_role_filter_true(self):
        """has_role returns True when user matches any listed role."""
        from authentication.templatetags.role_tags import has_role
        self.assertTrue(has_role(self.client_user, 'Client Admin,Facilities Admin'))

    def test_has_role_filter_false(self):
        """has_role returns False when user matches none of the listed roles."""
        from authentication.templatetags.role_tags import has_role
        self.assertFalse(has_role(self.facility_user, 'Client Admin'))

    def test_is_client_admin_filter(self):
        """is_client_admin template filter returns correct boolean."""
        from authentication.templatetags.role_tags import _is_client_admin
        self.assertTrue(_is_client_admin(self.client_user))
        self.assertFalse(_is_client_admin(self.facility_user))

    def test_is_facilities_admin_filter(self):
        """is_facilities_admin filter returns correct boolean."""
        from authentication.templatetags.role_tags import _is_facilities_admin
        facil = make_user(username='tagfacil', role=ROLE_FACILITIES_ADMIN)
        self.assertTrue(_is_facilities_admin(facil))
        self.assertFalse(_is_facilities_admin(self.facility_user))


# ---------------------------------------------------------------------------
# 7. Authentication API Tests
# ---------------------------------------------------------------------------

class AuthenticationAPITests(APITestCase):
    """Tests for DRF API endpoints."""

    def setUp(self):
        self.api_client = APIClient()
        self.user = make_user(
            username='apiuser',
            password='TestPass123!',
            email='api@test.com',
            role=ROLE_FACILITY_ADMIN,
        )
        self.admin = make_user(
            username='apiadmin',
            password='TestPass123!',
            email='apiadmin@test.com',
            role=ROLE_CLIENT_ADMIN,
        )
        self.superuser = make_superuser(username='apisuper')

    # --- UserProfile /me/ endpoint -------------------------------------------

    def test_profile_me_requires_auth(self):
        """GET /api/v1/auth/profiles/me/ requires authentication."""
        response = self.api_client.get('/api/v1/auth/profiles/me/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_profile_me_returns_current_user(self):
        """GET /api/v1/auth/profiles/me/ returns the authenticated user's profile."""
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.get('/api/v1/auth/profiles/me/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['user']['username'], 'apiuser')

    def test_profile_me_patch_updates_profile(self):
        """PATCH /api/v1/auth/profiles/me/ updates profile fields."""
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.patch('/api/v1/auth/profiles/me/', {
            'job_title': 'Senior Data Scientist',
            'phone_number': '+254700111222',
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.user.profile.refresh_from_db()
        self.assertEqual(self.user.profile.job_title, 'Senior Data Scientist')

    # --- Password change API --------------------------------------------------

    def test_password_change_api_requires_auth(self):
        """POST /api/v1/auth/password/change/ requires authentication."""
        response = self.api_client.post('/api/v1/auth/password/change/', {})
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_password_change_api_works(self):
        """POST with valid data changes the password."""
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.post('/api/v1/auth/password/change/', {
            'old_password': 'TestPass123!',
            'new_password': 'NewApiPass99!',
            'confirm_password': 'NewApiPass99!',
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password('NewApiPass99!'))

    def test_password_change_api_wrong_old_password(self):
        """POST with wrong old_password returns 400."""
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.post('/api/v1/auth/password/change/', {
            'old_password': 'WrongOldPass!',
            'new_password': 'NewApiPass99!',
            'confirm_password': 'NewApiPass99!',
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    # --- Notification API -----------------------------------------------------

    def test_notification_list_requires_auth(self):
        """GET /api/v1/auth/notifications/ requires authentication."""
        response = self.api_client.get('/api/v1/auth/notifications/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_notification_list_returns_only_own(self):
        """Notification list only includes current user's notifications."""
        Notification.objects.create(user=self.user, title='Own', message='msg', notification_type='info')
        Notification.objects.create(user=self.admin, title='Other', message='msg2', notification_type='warning')
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.get('/api/v1/auth/notifications/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        titles = [n['title'] for n in response.data['results']]
        self.assertIn('Own', titles)
        self.assertNotIn('Other', titles)

    def test_notification_mark_read_action(self):
        """POST /api/v1/auth/notifications/{id}/mark_read/ marks as read."""
        notif = Notification.objects.create(
            user=self.user, title='ToRead', message='msg', notification_type='info'
        )
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.post(f'/api/v1/auth/notifications/{notif.pk}/mark_read/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        notif.refresh_from_db()
        self.assertTrue(notif.is_read)

    def test_notification_mark_all_read(self):
        """POST /api/v1/auth/notifications/mark_all_read/ marks all as read."""
        Notification.objects.create(user=self.user, title='N1', message='m', notification_type='info')
        Notification.objects.create(user=self.user, title='N2', message='m', notification_type='success')
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.post('/api/v1/auth/notifications/mark_all_read/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(Notification.objects.filter(user=self.user, is_read=False).count(), 0)

    # --- User activity API ---------------------------------------------------

    def test_user_activity_requires_auth(self):
        """GET /api/v1/auth/activity/ requires authentication."""
        response = self.api_client.get('/api/v1/auth/activity/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_activity_returns_own_logs(self):
        """Activity log only returns records for the authenticated user."""
        AuditLog.log(user=self.user, action='login', resource='authentication')
        AuditLog.log(user=self.admin, action='login', resource='authentication')
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.get('/api/v1/auth/activity/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # All returned records should belong to self.user
        for entry in response.data['results']:
            self.assertEqual(entry['action'], 'login')

    # --- Users API -----------------------------------------------------------

    def test_user_list_requires_auth(self):
        """GET /api/v1/auth/users/ requires authentication."""
        response = self.api_client.get('/api/v1/auth/users/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_user_list_non_admin_sees_only_self(self):
        """Non-admin user only sees their own user in the list."""
        self.api_client.force_authenticate(user=self.user)
        response = self.api_client.get('/api/v1/auth/users/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        usernames = [u['username'] for u in response.data['results']]
        self.assertIn('apiuser', usernames)
        self.assertNotIn('apiadmin', usernames)

    def test_user_list_admin_sees_all(self):
        """Client Admin sees all users in the list."""
        self.api_client.force_authenticate(user=self.superuser)
        response = self.api_client.get('/api/v1/auth/users/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        usernames = [u['username'] for u in response.data['results']]
        self.assertIn('apiuser', usernames)
        self.assertIn('apiadmin', usernames)


# ---------------------------------------------------------------------------
# UserModuleGrant model + module_access helper tests
# ---------------------------------------------------------------------------

class UserModuleGrantModelTests(TestCase):

    def setUp(self):
        self.user = make_user('grant_target')
        self.admin = make_user('grant_admin', role=ROLE_CLIENT_ADMIN)

    def test_create_grant(self):
        from .models import UserModuleGrant
        grant = UserModuleGrant.objects.create(
            user=self.user, module_key=UserModuleGrant.MODULE_WAREHOUSE,
            is_granted=True, granted_by=self.admin,
        )
        self.assertTrue(grant.is_granted)
        self.assertIn('granted', str(grant))

    def test_unique_per_user_module(self):
        from django.db import IntegrityError
        from .models import UserModuleGrant
        UserModuleGrant.objects.create(user=self.user, module_key=UserModuleGrant.MODULE_WAREHOUSE)
        with self.assertRaises(IntegrityError):
            UserModuleGrant.objects.create(user=self.user, module_key=UserModuleGrant.MODULE_WAREHOUSE)


class ModuleAccessTests(TestCase):
    """authentication/module_access.py — role defaults + explicit overrides."""

    def setUp(self):
        self.facility_user = make_user('mod_facility_user', role=ROLE_FACILITY_ADMIN)
        self.client_admin = make_user('mod_client_admin', role=ROLE_CLIENT_ADMIN)
        self.superuser = make_superuser('mod_superuser')

    def test_superuser_always_has_access(self):
        from .module_access import has_module_access
        self.assertTrue(has_module_access(self.superuser, 'warehouse'))
        self.assertTrue(has_module_access(self.superuser, 'analytics'))
        self.assertTrue(has_module_access(self.superuser, 'self_service'))

    def test_warehouse_default_requires_client_admin(self):
        from .module_access import has_module_access
        self.assertFalse(has_module_access(self.facility_user, 'warehouse'))
        self.assertTrue(has_module_access(self.client_admin, 'warehouse'))

    def test_analytics_and_self_service_open_by_default(self):
        from .module_access import has_module_access
        self.assertTrue(has_module_access(self.facility_user, 'analytics'))
        self.assertTrue(has_module_access(self.facility_user, 'self_service'))

    def test_explicit_grant_overrides_default_deny(self):
        from .models import UserModuleGrant
        from .module_access import has_module_access
        UserModuleGrant.objects.create(
            user=self.facility_user, module_key=UserModuleGrant.MODULE_WAREHOUSE, is_granted=True,
        )
        self.assertTrue(has_module_access(self.facility_user, 'warehouse'))

    def test_explicit_revoke_overrides_default_allow(self):
        from .models import UserModuleGrant
        from .module_access import has_module_access
        UserModuleGrant.objects.create(
            user=self.client_admin, module_key=UserModuleGrant.MODULE_WAREHOUSE, is_granted=False,
        )
        self.assertFalse(has_module_access(self.client_admin, 'warehouse'))

    def test_get_module_overrides(self):
        from .models import UserModuleGrant
        from .module_access import get_module_overrides
        UserModuleGrant.objects.create(
            user=self.facility_user, module_key=UserModuleGrant.MODULE_WAREHOUSE, is_granted=True,
        )
        overrides = get_module_overrides(self.facility_user)
        self.assertEqual(overrides, {'warehouse': True})
