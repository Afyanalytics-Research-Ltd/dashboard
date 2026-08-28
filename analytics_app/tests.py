"""
Analytics app tests.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import Client as TestClient
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import AuditLog, Client
from .models import Dashboard

User = get_user_model()


def make_user(username='testuser', password='testpass123', is_superuser=False):
    user = User.objects.create_user(
        username=username, password=password, email=f'{username}@test.com'
    )
    if is_superuser:
        user.is_superuser = True
        user.is_staff = True
        user.save()
    return user


def make_client(name='Test Hospital', slug='test-hospital'):
    return Client.objects.create(name=name, slug=slug)


def make_dashboard(name='Revenue Dashboard', slug='revenue-dashboard', client=None, **kwargs):
    return Dashboard.objects.create(
        name=name,
        slug=slug,
        client=client,
        streamlit_url='http://localhost:8501/?dashboard=revenue',
        category=kwargs.get('category', 'financial'),
        is_active=kwargs.get('is_active', True),
        description=kwargs.get('description', ''),
        order=kwargs.get('order', 0),
    )


# =============================================================================
# MODEL TESTS
# =============================================================================

class DashboardModelTests(TestCase):
    """Test Dashboard model behaviour."""

    def setUp(self):
        self.client_obj = make_client()

    def test_create_dashboard(self):
        d = make_dashboard(client=self.client_obj)
        self.assertEqual(Dashboard.objects.count(), 1)
        self.assertEqual(d.name, 'Revenue Dashboard')

    def test_str_returns_name(self):
        d = make_dashboard(client=self.client_obj)
        self.assertEqual(str(d), 'Revenue Dashboard')

    def test_get_absolute_url(self):
        d = make_dashboard(client=self.client_obj)
        url = d.get_absolute_url()
        self.assertIn('/analytics/dashboards/', url)
        self.assertIn(d.slug, url)

    def test_increment_view_count(self):
        d = make_dashboard(client=self.client_obj)
        self.assertEqual(d.view_count, 0)
        d.increment_view_count()
        d.refresh_from_db()
        self.assertEqual(d.view_count, 1)

    def test_increment_view_count_multiple_times(self):
        d = make_dashboard(client=self.client_obj)
        for _ in range(5):
            d.increment_view_count()
        d.refresh_from_db()
        self.assertEqual(d.view_count, 5)

    def test_default_category_is_analytics(self):
        d = Dashboard.objects.create(
            name='Test', slug='test-def',
            streamlit_url='http://localhost:8501',
        )
        self.assertEqual(d.category, 'analytics')

    def test_ordering_by_order_then_name(self):
        make_dashboard(name='Zebra', slug='zebra', order=1)
        make_dashboard(name='Alpha', slug='alpha', order=2)
        make_dashboard(name='Middle', slug='middle', order=1)
        names = list(Dashboard.objects.values_list('name', flat=True))
        self.assertEqual(names[0], 'Middle')   # order=1 then alphabetical: Middle < Zebra
        self.assertEqual(names[2], 'Alpha')    # order=2

    def test_is_active_defaults_to_true(self):
        d = Dashboard.objects.create(name='X', slug='x', streamlit_url='http://localhost')
        self.assertTrue(d.is_active)

    def test_view_count_defaults_to_zero(self):
        d = Dashboard.objects.create(name='Y', slug='y', streamlit_url='http://localhost')
        self.assertEqual(d.view_count, 0)

    def test_dashboard_with_no_client(self):
        d = Dashboard.objects.create(name='Z', slug='z', streamlit_url='http://localhost')
        self.assertIsNone(d.client)

    def test_auto_slug_on_save(self):
        d = Dashboard(name='My New Dashboard', streamlit_url='http://localhost')
        d.save()
        self.assertEqual(d.slug, 'my-new-dashboard')

    def test_category_choices_are_valid(self):
        valid_categories = [c[0] for c in Dashboard.CATEGORY_CHOICES]
        for cat in valid_categories:
            d = Dashboard.objects.create(
                name=f'Test {cat}', slug=f'test-{cat}',
                streamlit_url='http://localhost', category=cat
            )
            self.assertEqual(d.category, cat)


# =============================================================================
# LIST VIEW TESTS
# =============================================================================

class DashboardListViewTests(TestCase):
    """Test DashboardListView."""

    def setUp(self):
        self.client_http = TestClient()
        self.user = make_user('listuser')
        self.superuser = make_user('superadmin', is_superuser=True)
        self.client_obj = make_client(name='Test Hosp', slug='test-hosp')
        # Patch profile so user has a client
        self.user.profile.client = self.client_obj
        self.user.profile.save()

    def _login(self, user):
        self.client_http.force_login(user)

    def test_redirects_unauthenticated(self):
        url = reverse('analytics:dashboard_list')
        resp = self.client_http.get(url)
        self.assertRedirects(resp, f'/auth/login/?next={url}', fetch_redirect_response=False)

    def test_list_shows_active_dashboards(self):
        make_dashboard(name='D1', slug='d1', client=self.client_obj)
        make_dashboard(name='D2', slug='d2', client=self.client_obj)
        make_dashboard(name='Inactive', slug='inactive', client=self.client_obj, is_active=False)
        self._login(self.user)
        with patch('analytics_app.views._sync_dashboards_for_client'):
            resp = self.client_http.get(reverse('analytics:dashboard_list'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'D1')
        self.assertContains(resp, 'D2')
        self.assertNotContains(resp, 'Inactive')

    def test_search_filters_by_name(self):
        make_dashboard(name='Revenue Board', slug='rev-board', client=self.client_obj)
        make_dashboard(name='Clinical KPIs', slug='clin-kpis', client=self.client_obj)
        self._login(self.user)
        with patch('analytics_app.views._sync_dashboards_for_client'):
            resp = self.client_http.get(reverse('analytics:dashboard_list') + '?q=Revenue')
        self.assertContains(resp, 'Revenue Board')
        self.assertNotContains(resp, 'Clinical KPIs')

    def test_filter_by_category(self):
        make_dashboard(name='Rev', slug='rev', client=self.client_obj, category='financial')
        make_dashboard(name='Clin', slug='clin', client=self.client_obj, category='clinical')
        self._login(self.user)
        with patch('analytics_app.views._sync_dashboards_for_client'):
            resp = self.client_http.get(reverse('analytics:dashboard_list') + '?category=financial')
        self.assertContains(resp, 'Rev')
        self.assertNotContains(resp, 'Clin')

    def test_pagination_12_per_page(self):
        for i in range(20):
            make_dashboard(name=f'D{i}', slug=f'd{i}', client=self.client_obj)
        self._login(self.user)
        with patch('analytics_app.views._sync_dashboards_for_client'):
            resp = self.client_http.get(reverse('analytics:dashboard_list'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.context['dashboards']), 12)

    def test_superuser_sees_all_dashboards(self):
        other_client = make_client(name='Other Org', slug='other-org')
        make_dashboard(name='My DB', slug='my-db', client=self.client_obj)
        make_dashboard(name='Other DB', slug='other-db', client=other_client)
        self._login(self.superuser)
        with patch('analytics_app.views._sync_dashboards_for_client'):
            resp = self.client_http.get(reverse('analytics:dashboard_list'))
        self.assertContains(resp, 'My DB')
        self.assertContains(resp, 'Other DB')

    def test_sidebar_section_is_analytics(self):
        self._login(self.user)
        with patch('analytics_app.views._sync_dashboards_for_client'):
            resp = self.client_http.get(reverse('analytics:dashboard_list'))
        self.assertEqual(resp.context['sidebar_section'], 'analytics')


# =============================================================================
# DETAIL VIEW TESTS
# =============================================================================

class DashboardDetailViewTests(TestCase):
    """Test DashboardDetailView."""

    def setUp(self):
        self.client_http = TestClient()
        self.user = make_user('detailuser')
        self.superuser = make_user('superdetail', is_superuser=True)
        self.client_obj = make_client(name='Detail Hosp', slug='detail-hosp')
        self.user.profile.client = self.client_obj
        self.user.profile.save()
        self.dashboard = make_dashboard(
            name='Clinical Overview', slug='clinical-overview', client=self.client_obj
        )

    def test_view_increments_view_count(self):
        self.client_http.force_login(self.user)
        self.assertEqual(self.dashboard.view_count, 0)
        resp = self.client_http.get(
            reverse('analytics:dashboard_view', kwargs={'slug': 'clinical-overview'})
        )
        self.assertEqual(resp.status_code, 200)
        self.dashboard.refresh_from_db()
        self.assertEqual(self.dashboard.view_count, 1)

    def test_view_creates_audit_log(self):
        self.client_http.force_login(self.user)
        self.client_http.get(
            reverse('analytics:dashboard_view', kwargs={'slug': 'clinical-overview'})
        )
        self.assertTrue(
            AuditLog.objects.filter(action='read', resource='dashboard').exists()
        )

    def test_404_for_inactive_dashboard(self):
        self.dashboard.is_active = False
        self.dashboard.save()
        self.client_http.force_login(self.user)
        resp = self.client_http.get(
            reverse('analytics:dashboard_view', kwargs={'slug': 'clinical-overview'})
        )
        self.assertEqual(resp.status_code, 404)

    def test_sidebar_section_is_analytics(self):
        self.client_http.force_login(self.user)
        resp = self.client_http.get(
            reverse('analytics:dashboard_view', kwargs={'slug': 'clinical-overview'})
        )
        self.assertEqual(resp.context['sidebar_section'], 'analytics')

    def test_wrong_client_raises_403(self):
        other_client = make_client(name='Another', slug='another')
        other_dashboard = make_dashboard(
            name='Other', slug='other-dashboard', client=other_client
        )
        self.client_http.force_login(self.user)
        resp = self.client_http.get(
            reverse('analytics:dashboard_view', kwargs={'slug': 'other-dashboard'})
        )
        self.assertEqual(resp.status_code, 403)

    def test_superuser_can_view_any_dashboard(self):
        other_client = make_client(name='AnotherOrg', slug='another-org')
        make_dashboard(name='Foreign', slug='foreign', client=other_client)
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get(
            reverse('analytics:dashboard_view', kwargs={'slug': 'foreign'})
        )
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# SYNC TESTS
# =============================================================================

class DashboardSyncTests(TestCase):
    """Test filesystem sync logic."""

    def setUp(self):
        self.client_http = TestClient()
        self.superuser = make_user('syncsuper', is_superuser=True)
        self.regular_user = make_user('syncuser')
        self.client_obj = make_client(name='Sync Hosp', slug='sync-hosp')

    def test_sync_requires_superuser(self):
        self.client_http.force_login(self.regular_user)
        resp = self.client_http.post(reverse('analytics:dashboard_sync'))
        self.assertEqual(resp.status_code, 403)

    def test_sync_creates_records_from_files(self):
        from analytics_app.views import _sync_dashboards_for_client
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake .py files
            for fname in ['revenue.py', 'clinical.py']:
                open(os.path.join(tmpdir, fname), 'w').close()
            with patch('analytics_app.views.os.path.join', return_value=tmpdir), \
                 patch('analytics_app.views.os.path.isdir', return_value=True), \
                 patch('analytics_app.views.os.listdir', return_value=['revenue.py', 'clinical.py']):
                result = _sync_dashboards_for_client('sync-hosp', self.client_obj)
        self.assertEqual(result['created'], 2)

    def test_sync_deactivates_removed_files(self):
        from analytics_app.views import _sync_dashboards_for_client
        # Pre-create a record that has no matching file
        make_dashboard(name='Old Dashboard', slug='old-dash', client=self.client_obj)
        with patch('analytics_app.views.os.path.join', return_value='/tmp/fake'), \
             patch('analytics_app.views.os.path.isdir', return_value=True), \
             patch('analytics_app.views.os.listdir', return_value=[]):
            result = _sync_dashboards_for_client('sync-hosp', self.client_obj)
        self.assertEqual(result['deactivated'], 1)
        Dashboard.objects.get(slug='old-dash').refresh_from_db()
        self.assertFalse(Dashboard.objects.get(slug='old-dash').is_active)


# =============================================================================
# API TESTS
# =============================================================================

class DashboardAPITests(TestCase):
    """Test DRF API endpoints."""

    def setUp(self):
        self.client_http = TestClient()
        self.user = make_user('apiuser')
        self.superuser = make_user('apisuper', is_superuser=True)
        self.client_obj = make_client(name='API Hosp', slug='api-hosp')
        self.user.profile.client = self.client_obj
        self.user.profile.save()
        self.dashboard = make_dashboard(
            name='API Dashboard', slug='api-dashboard', client=self.client_obj
        )

    def test_list_requires_authentication(self):
        resp = self.client_http.get('/api/v1/analytics/dashboards/')
        self.assertEqual(resp.status_code, 401)

    def test_list_returns_dashboards_for_user_client(self):
        self.client_http.force_login(self.user)
        resp = self.client_http.get('/api/v1/analytics/dashboards/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data['count'], 1)

    def test_retrieve_dashboard(self):
        self.client_http.force_login(self.user)
        resp = self.client_http.get(f'/api/v1/analytics/dashboards/{self.dashboard.pk}/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['slug'], 'api-dashboard')

    def test_create_requires_superuser(self):
        self.client_http.force_login(self.user)
        resp = self.client_http.post(
            '/api/v1/analytics/dashboards/',
            {'name': 'New', 'slug': 'new', 'streamlit_url': 'http://localhost:8501'},
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 403)

    def test_superuser_can_create_dashboard(self):
        self.client_http.force_login(self.superuser)
        resp = self.client_http.post(
            '/api/v1/analytics/dashboards/',
            {
                'name': 'Super New', 'slug': 'super-new',
                'streamlit_url': 'http://localhost:8501',
                'category': 'analytics',
            },
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 201)
        self.assertTrue(Dashboard.objects.filter(slug='super-new').exists())

    def test_search_by_name(self):
        self.client_http.force_login(self.user)
        resp = self.client_http.get('/api/v1/analytics/dashboards/?search=API')
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.json()['count'], 1)

    def test_stats_endpoint(self):
        self.client_http.force_login(self.user)
        resp = self.client_http.get('/api/v1/analytics/dashboards/stats/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('total', data)
        self.assertIn('by_category', data)

    def test_sync_api_requires_superuser(self):
        self.client_http.force_login(self.user)
        resp = self.client_http.post('/api/v1/analytics/dashboards/sync/')
        self.assertEqual(resp.status_code, 403)


# ---------------------------------------------------------------------------
# Dashboard.hidden_from_users — per-user visibility override
# ---------------------------------------------------------------------------

class HiddenFromUsersTests(TestCase):

    def setUp(self):
        self.client_obj = make_client(name='Hide Test Hosp', slug='hide-test-hosp')
        self.user = make_user('hide_target_user')
        self.user.profile.client = self.client_obj
        self.user.profile.save()
        self.other_user = make_user('hide_other_user')
        self.other_user.profile.client = self.client_obj
        self.other_user.profile.save()
        # Not using make_dashboard(): it hardcodes a "localhost"-style
        # streamlit_url, and DashboardListView.get() auto-deactivates any
        # dashboard with that URL pattern that isn't part of its filesystem
        # sync batch on every GET — an unrelated pre-existing side effect
        # that would silently flip is_active=False out from under this test.
        # A Redash-backed URL isn't swept up by that sync at all.
        self.dashboard = Dashboard.objects.create(
            name='Financial Report', slug='financial-report-hide-test',
            client=self.client_obj, redash_dashboard_url='https://redash.example.com/public/dashboards/abc123',
        )
        self.c = TestClient()

    def test_visible_by_default_in_list(self):
        self.c.force_login(self.user)
        resp = self.c.get(reverse('analytics:dashboard_list'))
        self.assertIn(self.dashboard, list(resp.context['dashboards']))

    def test_hidden_from_specific_user_disappears_from_list(self):
        self.dashboard.hidden_from_users.add(self.user)
        self.c.force_login(self.user)
        resp = self.c.get(reverse('analytics:dashboard_list'))
        self.assertNotIn(self.dashboard, list(resp.context['dashboards']))

    def test_still_visible_to_other_user_in_same_client(self):
        self.dashboard.hidden_from_users.add(self.user)
        self.c.force_login(self.other_user)
        resp = self.c.get(reverse('analytics:dashboard_list'))
        self.assertIn(self.dashboard, list(resp.context['dashboards']))

    def test_direct_url_blocked_when_hidden(self):
        self.dashboard.hidden_from_users.add(self.user)
        self.c.force_login(self.user)
        resp = self.c.get(reverse('analytics:dashboard_view', kwargs={'slug': self.dashboard.slug}))
        self.assertEqual(resp.status_code, 403)

    def test_direct_url_allowed_when_not_hidden(self):
        self.c.force_login(self.user)
        resp = self.c.get(reverse('analytics:dashboard_view', kwargs={'slug': self.dashboard.slug}))
        self.assertEqual(resp.status_code, 200)

    def test_superuser_unaffected_by_hiding(self):
        self.dashboard.hidden_from_users.add(self.user)
        superuser = make_user('hide_superuser', is_superuser=True)
        self.c.force_login(superuser)
        resp = self.c.get(reverse('analytics:dashboard_view', kwargs={'slug': self.dashboard.slug}))
        self.assertEqual(resp.status_code, 200)
