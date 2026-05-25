"""
Airflow UI tests.

All Airflow API calls are mocked so no live Airflow instance is required.
"""

from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import Client as TestClient
from django.test import TestCase
from django.urls import reverse

from core.models import AuditLog
from .models import DAGSummary

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


MOCK_DAGS = [
    {
        'dag_id': 'etl_pipeline_daily',
        'description': 'Daily ETL pipeline',
        'is_paused': False,
        'is_active': True,
        'latest_dag_run_start_date': None,
        'next_dagrun': None,
    },
    {
        'dag_id': 'reports_weekly',
        'description': 'Weekly reports',
        'is_paused': True,
        'is_active': False,
        'latest_dag_run_start_date': None,
        'next_dagrun': None,
    },
]

MOCK_RUNS = [
    {
        'dag_run_id': 'manual__2026-01-01T00:00:00+00:00',
        'dag_id': 'etl_pipeline_daily',
        'state': 'success',
        'start_date': '2026-01-01T00:00:00+00:00',
        'end_date': '2026-01-01T01:00:00+00:00',
        'execution_date': '2026-01-01T00:00:00+00:00',
        'logical_date': '2026-01-01T00:00:00+00:00',
        'run_type': 'manual',
    },
    {
        'dag_run_id': 'manual__2026-01-02T00:00:00+00:00',
        'dag_id': 'etl_pipeline_daily',
        'state': 'failed',
        'start_date': '2026-01-02T00:00:00+00:00',
        'end_date': '2026-01-02T00:30:00+00:00',
        'execution_date': '2026-01-02T00:00:00+00:00',
        'logical_date': '2026-01-02T00:00:00+00:00',
        'run_type': 'manual',
    },
]

MOCK_TASKS = [
    {
        'task_id': 'extract_data',
        'dag_id': 'etl_pipeline_daily',
        'dag_run_id': 'manual__2026-01-01T00:00:00+00:00',
        'state': 'success',
        'start_date': '2026-01-01T00:01:00+00:00',
        'end_date': '2026-01-01T00:10:00+00:00',
        'duration': 540.0,
        'try_number': 1,
    },
    {
        'task_id': 'transform_data',
        'dag_id': 'etl_pipeline_daily',
        'dag_run_id': 'manual__2026-01-01T00:00:00+00:00',
        'state': 'failed',
        'start_date': '2026-01-01T00:11:00+00:00',
        'end_date': '2026-01-01T00:12:00+00:00',
        'duration': 60.0,
        'try_number': 2,
    },
]


# =============================================================================
# MODEL TESTS
# =============================================================================

class DAGSummaryModelTests(TestCase):
    """Test DAGSummary model."""

    def test_create_dag_summary(self):
        s = DAGSummary.objects.create(
            dag_id='my_dag',
            total_runs=10, successful_runs=8, failed_runs=2,
        )
        self.assertEqual(DAGSummary.objects.count(), 1)
        self.assertEqual(str(s), 'my_dag')

    def test_success_rate_calculation(self):
        s = DAGSummary(total_runs=10, successful_runs=7, failed_runs=3)
        self.assertEqual(s.success_rate, 70.0)

    def test_success_rate_zero_runs(self):
        s = DAGSummary(total_runs=0, successful_runs=0, failed_runs=0)
        self.assertEqual(s.success_rate, 0)

    def test_success_rate_100_percent(self):
        s = DAGSummary(total_runs=5, successful_runs=5, failed_runs=0)
        self.assertEqual(s.success_rate, 100.0)

    def test_dag_id_is_unique(self):
        DAGSummary.objects.create(dag_id='unique_dag')
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            DAGSummary.objects.create(dag_id='unique_dag')

    def test_default_values(self):
        s = DAGSummary.objects.create(dag_id='defaults_dag')
        self.assertTrue(s.is_active)
        self.assertFalse(s.is_paused)
        self.assertEqual(s.total_runs, 0)
        self.assertEqual(s.successful_runs, 0)
        self.assertEqual(s.failed_runs, 0)


# =============================================================================
# SERVICE TESTS
# =============================================================================

class AirflowServiceTests(TestCase):
    """Test AirflowService methods (all API calls mocked)."""

    @patch('airflow_ui.services._airflow_request')
    def test_get_dags_returns_list(self, mock_req):
        from airflow_ui.services import AirflowService
        mock_req.return_value = {'dags': MOCK_DAGS}
        dags = AirflowService.get_dags()
        self.assertEqual(len(dags), 2)
        mock_req.assert_called_once_with('GET', '/dags')

    @patch('airflow_ui.services._airflow_request')
    def test_get_dags_returns_empty_on_bad_payload(self, mock_req):
        from airflow_ui.services import AirflowService
        mock_req.return_value = {'error': 'Unauthorized'}
        dags = AirflowService.get_dags()
        self.assertEqual(dags, [])

    @patch('airflow_ui.services._airflow_request')
    def test_get_dag_runs_returns_list(self, mock_req):
        from airflow_ui.services import AirflowService
        mock_req.return_value = {'dag_runs': MOCK_RUNS}
        runs = AirflowService.get_dag_runs('etl_pipeline_daily', limit=10)
        self.assertEqual(len(runs), 2)

    @patch('airflow_ui.services._airflow_request')
    def test_trigger_dag_sends_post(self, mock_req):
        from airflow_ui.services import AirflowService
        mock_req.return_value = {'dag_run_id': 'manual__xyz', 'state': 'queued'}
        result = AirflowService.trigger_dag('etl_pipeline_daily')
        self.assertIn('dag_run_id', result)
        call_args = mock_req.call_args
        self.assertEqual(call_args[0][0], 'POST')
        self.assertIn('/etl_pipeline_daily/dagRuns', call_args[0][1])

    @patch('airflow_ui.services._airflow_request')
    def test_get_task_instances_returns_list(self, mock_req):
        from airflow_ui.services import AirflowService
        mock_req.return_value = {'task_instances': MOCK_TASKS}
        tasks = AirflowService.get_task_instances('etl_pipeline_daily', 'manual__xyz')
        self.assertEqual(len(tasks), 2)

    @patch('airflow_ui.services.requests.post')
    def test_get_airflow_token_parses_access_token(self, mock_post):
        from airflow_ui.services import get_airflow_token
        mock_response = MagicMock()
        mock_response.json.return_value = {'access_token': 'tok-abc123'}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        token = get_airflow_token()
        self.assertEqual(token, 'tok-abc123')


# =============================================================================
# VIEW TESTS — PIPELINE DASHBOARD
# =============================================================================

class PipelineViewTests(TestCase):
    """Test PipelineDashboardView and related views."""

    def setUp(self):
        self.client_http = TestClient()
        self.superuser = make_user('psuper', is_superuser=True)
        self.regular_user = make_user('pregular')

    @patch('airflow_ui.views.AirflowService.get_dags')
    def test_superuser_can_access_pipeline_dashboard(self, mock_get_dags):
        mock_get_dags.return_value = MOCK_DAGS
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get(reverse('airflow:dashboard'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'etl_pipeline_daily')

    def test_regular_user_gets_403(self):
        self.client_http.force_login(self.regular_user)
        resp = self.client_http.get(reverse('airflow:dashboard'))
        self.assertEqual(resp.status_code, 403)

    def test_unauthenticated_redirects_to_login(self):
        resp = self.client_http.get(reverse('airflow:dashboard'))
        self.assertIn(resp.status_code, [302, 403])

    @patch('airflow_ui.views.AirflowService.get_dags')
    def test_dashboard_shows_dag_count(self, mock_get_dags):
        mock_get_dags.return_value = MOCK_DAGS
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get(reverse('airflow:dashboard'))
        self.assertEqual(resp.context['total_dags'], 2)
        self.assertEqual(resp.context['active_dags'], 1)

    @patch('airflow_ui.views.AirflowService.get_dags')
    def test_search_filters_dags(self, mock_get_dags):
        mock_get_dags.return_value = MOCK_DAGS
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get(reverse('airflow:dashboard') + '?q=weekly')
        self.assertEqual(resp.status_code, 200)
        # 'weekly' matches 'reports_weekly'
        page_obj = resp.context['page_obj']
        dag_ids = [d['dag_id'] for d in page_obj]
        self.assertIn('reports_weekly', dag_ids)
        self.assertNotIn('etl_pipeline_daily', dag_ids)

    @patch('airflow_ui.views.AirflowService.get_dags')
    def test_sidebar_section_is_pipelines(self, mock_get_dags):
        mock_get_dags.return_value = []
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get(reverse('airflow:dashboard'))
        self.assertEqual(resp.context['sidebar_section'], 'pipelines')

    @patch('airflow_ui.views.AirflowService.get_dags')
    def test_airflow_api_error_shows_warning(self, mock_get_dags):
        mock_get_dags.side_effect = Exception('Connection refused')
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get(reverse('airflow:dashboard'))
        self.assertEqual(resp.status_code, 200)
        # Should still render with empty list, not crash

    @patch('airflow_ui.views.AirflowService.get_dag_runs')
    def test_dag_detail_view(self, mock_runs):
        mock_runs.return_value = MOCK_RUNS
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get(
            reverse('airflow:dag_detail', kwargs={'dag_id': 'etl_pipeline_daily'})
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['dag_id'], 'etl_pipeline_daily')
        self.assertEqual(resp.context['total_runs'], 2)

    @patch('airflow_ui.views.AirflowService.get_dag_runs')
    def test_dag_detail_creates_audit_log(self, mock_runs):
        mock_runs.return_value = []
        self.client_http.force_login(self.superuser)
        self.client_http.get(
            reverse('airflow:dag_detail', kwargs={'dag_id': 'some_dag'})
        )
        self.assertTrue(
            AuditLog.objects.filter(action='read', resource='dag').exists()
        )


# =============================================================================
# VIEW TESTS — TRIGGER
# =============================================================================

class TriggerDAGViewTests(TestCase):
    """Test TriggerDAGView."""

    def setUp(self):
        self.client_http = TestClient()
        self.superuser = make_user('tsuper', is_superuser=True)
        self.regular_user = make_user('tregular')

    @patch('airflow_ui.views.AirflowService.trigger_dag')
    def test_trigger_post_creates_audit_log(self, mock_trigger):
        mock_trigger.return_value = {'dag_run_id': 'manual__abc123', 'state': 'queued'}
        self.client_http.force_login(self.superuser)
        self.client_http.post(
            reverse('airflow:trigger_dag', kwargs={'dag_id': 'etl_pipeline_daily'})
        )
        self.assertTrue(
            AuditLog.objects.filter(action='trigger', resource='dag').exists()
        )

    @patch('airflow_ui.views.AirflowService.trigger_dag')
    def test_trigger_post_redirects_to_dag_detail(self, mock_trigger):
        mock_trigger.return_value = {'dag_run_id': 'manual__xyz', 'state': 'queued'}
        self.client_http.force_login(self.superuser)
        resp = self.client_http.post(
            reverse('airflow:trigger_dag', kwargs={'dag_id': 'etl_pipeline_daily'})
        )
        self.assertRedirects(
            resp,
            reverse('airflow:dag_detail', kwargs={'dag_id': 'etl_pipeline_daily'}),
            fetch_redirect_response=False,
        )

    @patch('airflow_ui.views.AirflowService.trigger_dag')
    def test_trigger_api_error_shows_message(self, mock_trigger):
        mock_trigger.return_value = {'error': 'DAG not found in Airflow'}
        self.client_http.force_login(self.superuser)
        resp = self.client_http.post(
            reverse('airflow:trigger_dag', kwargs={'dag_id': 'bad_dag'})
        )
        # Should redirect without crashing
        self.assertEqual(resp.status_code, 302)

    def test_trigger_requires_superuser(self):
        self.client_http.force_login(self.regular_user)
        resp = self.client_http.post(
            reverse('airflow:trigger_dag', kwargs={'dag_id': 'etl_pipeline_daily'})
        )
        self.assertEqual(resp.status_code, 403)


# =============================================================================
# API TESTS
# =============================================================================

class AirflowAPITests(TestCase):
    """Test Airflow DRF API endpoints."""

    def setUp(self):
        self.client_http = TestClient()
        self.superuser = make_user('asuper', is_superuser=True)
        self.regular_user = make_user('aregular')

    @patch('airflow_ui.api.AirflowService.get_dags')
    def test_dag_list_api_requires_superuser(self, mock_dags):
        mock_dags.return_value = MOCK_DAGS
        self.client_http.force_login(self.regular_user)
        resp = self.client_http.get('/api/v1/pipelines/dags/')
        self.assertEqual(resp.status_code, 403)

    @patch('airflow_ui.api.AirflowService.get_dags')
    def test_dag_list_api_superuser_gets_data(self, mock_dags):
        mock_dags.return_value = MOCK_DAGS
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get('/api/v1/pipelines/dags/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['count'], 2)

    def test_dag_list_api_requires_auth(self):
        resp = self.client_http.get('/api/v1/pipelines/dags/')
        self.assertEqual(resp.status_code, 403)

    @patch('airflow_ui.api.AirflowService.trigger_dag')
    def test_trigger_api_superuser_succeeds(self, mock_trigger):
        mock_trigger.return_value = {'dag_run_id': 'manual__api123', 'state': 'queued'}
        self.client_http.force_login(self.superuser)
        resp = self.client_http.post('/api/v1/pipelines/dags/my_dag/trigger/')
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()['ok'])

    def test_dag_summary_list(self):
        DAGSummary.objects.create(dag_id='cached_dag', total_runs=5, successful_runs=4, failed_runs=1)
        self.client_http.force_login(self.superuser)
        resp = self.client_http.get('/api/v1/pipelines/summaries/')
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.json()['count'], 1)
