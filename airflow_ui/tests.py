"""
Airflow UI tests.

All Airflow HTTP traffic is mocked at the ``requests.Session`` level, so no
live Airflow instance is required.
"""

import json
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import Client as TestClient
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import AuditLog

from .client import AirflowClient, AirflowError, encode_segment, reset_client
from .models import DAGSummary
from .routes import ROUTES, match_route

User = get_user_model()

AIRFLOW_SETTINGS = dict(
    AIRFLOW_BASE_URL='http://airflow.test',
    AIRFLOW_USERNAME='u',
    AIRFLOW_PASSWORD='p',
    CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}},
)


def make_user(username='testuser', password='testpass123', is_superuser=False):
    user = User.objects.create_user(username=username, password=password, email=f'{username}@test.com')
    if is_superuser:
        user.is_superuser = True
        user.is_staff = True
        user.save()
    return user


def fake_response(status=200, payload=None, content_type='application/json', text=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    if text is not None:
        resp.content = text.encode()
    else:
        resp.content = json.dumps(payload).encode() if payload is not None else b''
    resp.headers = {'Content-Type': content_type}
    resp.json.side_effect = lambda: json.loads(resp.content)
    return resp


TOKEN_RESPONSE = fake_response(201, {'access_token': 'jwt-1'})


class AirflowTestCase(TestCase):
    """Fresh client + empty token cache for every test."""

    def setUp(self):
        cache.clear()
        reset_client()

    def tearDown(self):
        reset_client()


# =============================================================================
# MODEL
# =============================================================================

class DAGSummaryModelTest(TestCase):

    def test_success_rate(self):
        s = DAGSummary.objects.create(dag_id='d', total_runs=8, successful_runs=6)
        self.assertEqual(s.success_rate, 75.0)

    def test_success_rate_zero_runs(self):
        self.assertEqual(DAGSummary.objects.create(dag_id='d').success_rate, 0.0)

    def test_str(self):
        self.assertEqual(str(DAGSummary(dag_id='etl')), 'etl')


# =============================================================================
# ROUTE ALLOWLIST
# =============================================================================

class RouteAllowlistTest(TestCase):

    def test_covers_every_documented_endpoint(self):
        self.assertEqual(sum(len(m) for m in ROUTES.values()), 106)

    def test_literal_route_beats_template(self):
        self.assertEqual(match_route('/connections/test', 'POST'), ('/connections/test', True))
        self.assertEqual(match_route('/backfills/dry_run', 'POST'), ('/backfills/dry_run', True))

    def test_templated_routes(self):
        tpl, ok = match_route('/dags/a/dagRuns/x%2By/taskInstances/t/logs/1', 'GET')
        self.assertTrue(ok)
        self.assertTrue(tpl.endswith('/logs/{try_number}'))
        self.assertEqual(match_route('/dags/~/dagRuns', 'GET'), ('/dags/{dag_id}/dagRuns', True))

    def test_wrong_method(self):
        self.assertEqual(match_route('/dags', 'PUT'), ('/dags', False))

    def test_auth_routes_excluded(self):
        self.assertEqual(match_route('/auth/login', 'GET'), (None, False))
        self.assertEqual(match_route('/auth/token', 'POST'), (None, False))

    def test_unknown_route(self):
        self.assertEqual(match_route('/nope', 'GET'), (None, False))

    def test_encode_segment(self):
        self.assertEqual(encode_segment('scheduled__2026-10-21T00:00:00+00:00'),
                         'scheduled__2026-10-21T00%3A00%3A00%2B00%3A00')
        self.assertEqual(encode_segment('~'), '~')


# =============================================================================
# CLIENT
# =============================================================================

@override_settings(**AIRFLOW_SETTINGS)
class AirflowClientTest(AirflowTestCase):

    def make(self):
        return AirflowClient('http://airflow.test', 'u', 'p', max_retries=2)

    def test_login_then_reuse_cached_token(self):
        client = self.make()
        with patch.object(client.session, 'post', return_value=TOKEN_RESPONSE) as post, \
                patch.object(client.session, 'request', return_value=fake_response(200, {'dags': []})) as req:
            client.get_json('/dags')
            client.get_json('/dags')
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs['json'], {'username': 'u', 'password': 'p'})
        self.assertEqual(req.call_count, 2)
        self.assertEqual(req.call_args.kwargs['headers']['Authorization'], 'Bearer jwt-1')

    def test_token_shared_between_client_instances(self):
        a, b = self.make(), self.make()
        with patch.object(a.session, 'post', return_value=TOKEN_RESPONSE):
            a.token()
        with patch.object(b.session, 'post') as post:
            self.assertEqual(b.token(), 'jwt-1')
        post.assert_not_called()

    def test_relogin_once_on_expired_token(self):
        client = self.make()
        tokens = [fake_response(201, {'access_token': 'old'}), fake_response(201, {'access_token': 'new'})]
        responses = [fake_response(403, {'detail': 'Invalid JWT token'}), fake_response(200, {'ok': 1})]
        with patch.object(client.session, 'post', side_effect=tokens), \
                patch.object(client.session, 'request', side_effect=responses) as req:
            self.assertEqual(client.get_json('/dags'), {'ok': 1})
        self.assertEqual(req.call_args.kwargs['headers']['Authorization'], 'Bearer new')

    def test_permission_403_is_not_retried(self):
        client = self.make()
        with patch.object(client.session, 'post', return_value=TOKEN_RESPONSE) as post, \
                patch.object(client.session, 'request', return_value=fake_response(403, {'detail': 'not exposing config'})):
            resp = client.request('GET', '/config')
        self.assertEqual(resp.status_code, 403)
        post.assert_called_once()

    def test_bad_credentials_raise(self):
        client = self.make()
        with patch.object(client.session, 'post', return_value=fake_response(401, {'detail': 'Invalid credentials'})):
            with self.assertRaises(AirflowError) as ctx:
                client.get_json('/dags')
        self.assertEqual(ctx.exception.payload['upstream_status'], 401)

    @patch('airflow_ui.client.time.sleep')
    def test_get_retried_on_503(self, _sleep):
        client = self.make()
        with patch.object(client.session, 'post', return_value=TOKEN_RESPONSE), \
                patch.object(client.session, 'request', side_effect=[fake_response(503), fake_response(200, {'x': 1})]) as req:
            self.assertEqual(client.get_json('/dags'), {'x': 1})
        self.assertEqual(req.call_count, 2)

    @patch('airflow_ui.client.time.sleep')
    def test_post_not_retried_on_network_error(self, _sleep):
        client = self.make()
        with patch.object(client.session, 'post', return_value=TOKEN_RESPONSE), \
                patch.object(client.session, 'request', side_effect=requests.ConnectionError('boom')) as req:
            with self.assertRaises(AirflowError):
                client.request('POST', '/dags/x/dagRuns', json={})
        req.assert_called_once()

    def test_paginate_follows_offsets(self):
        client = self.make()
        pages = [
            fake_response(200, {'dags': [{'dag_id': 'a'}, {'dag_id': 'b'}], 'total_entries': 3}),
            fake_response(200, {'dags': [{'dag_id': 'c'}], 'total_entries': 3}),
        ]
        with patch.object(client.session, 'post', return_value=TOKEN_RESPONSE), \
                patch.object(client.session, 'request', side_effect=pages) as req:
            ids = [d['dag_id'] for d in client.paginate('/dags', 'dags', page_size=2)]
        self.assertEqual(ids, ['a', 'b', 'c'])
        self.assertEqual(req.call_args_list[1].kwargs['params']['offset'], 2)


# =============================================================================
# PROXY
# =============================================================================

@override_settings(**AIRFLOW_SETTINGS)
class ProxyTest(AirflowTestCase):

    def setUp(self):
        super().setUp()
        self.admin = make_user('admin', is_superuser=True)
        self.client = TestClient()
        self.client.force_login(self.admin)

    def call(self, method, path, upstream, **kwargs):
        with patch('requests.Session.post', return_value=TOKEN_RESPONSE), \
                patch('requests.Session.request', return_value=upstream) as req:
            resp = getattr(self.client, method)(f'/pipelines/api/v2/{path}', **kwargs)
        return resp, req

    def test_requires_login(self):
        resp = TestClient().get('/pipelines/api/v2/dags')
        self.assertEqual(resp.status_code, 401)

    def test_requires_superuser(self):
        c = TestClient()
        c.force_login(make_user('plain'))
        self.assertEqual(c.get('/pipelines/api/v2/dags').status_code, 403)

    def test_get_passthrough_with_repeated_params(self):
        resp, req = self.call('get', 'dags?tags=a&tags=b&limit=5', fake_response(200, {'dags': [], 'total_entries': 0}))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {'dags': [], 'total_entries': 0})
        self.assertEqual(req.call_args.args[:2], ('GET', 'http://airflow.test/api/v2/dags'))
        self.assertEqual(req.call_args.kwargs['params'], [('tags', 'a'), ('tags', 'b'), ('limit', '5')])
        self.assertEqual(resp['Cache-Control'], 'no-store')

    def test_run_id_reencoded(self):
        _, req = self.call('get', 'dags/d/dagRuns/scheduled__2026-10-21T00%3A00%3A00%2B00%3A00', fake_response(200, {}))
        self.assertEqual(req.call_args.args[1],
                         'http://airflow.test/api/v2/dags/d/dagRuns/scheduled__2026-10-21T00%3A00%3A00%2B00%3A00')

    def test_unknown_route_404(self):
        resp, req = self.call('get', 'not/a/route', fake_response(200, {}))
        self.assertEqual(resp.status_code, 404)
        req.assert_not_called()

    def test_auth_route_blocked(self):
        resp, req = self.call('post', 'auth/token', fake_response(200, {}), data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 404)
        req.assert_not_called()

    def test_method_not_allowed(self):
        resp, req = self.call('put', 'dags', fake_response(200, {}), data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 405)
        req.assert_not_called()

    def test_invalid_json_body(self):
        resp, req = self.call('post', 'dags/d/dagRuns', fake_response(200, {}), data='{nope', content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        req.assert_not_called()

    def test_upstream_error_relayed(self):
        resp, _ = self.call('get', 'dags/missing', fake_response(404, {'detail': 'DAG not found'}))
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()['detail'], 'DAG not found')

    def test_trigger_forwards_body_and_audits(self):
        body = {'logical_date': None, 'conf': {'k': 1}}
        resp, req = self.call('post', 'dags/etl/dagRuns', fake_response(200, {'dag_run_id': 'r1', 'state': 'queued'}),
                              data=json.dumps(body), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(json.loads(req.call_args.kwargs['data']), body)
        log = AuditLog.objects.get(resource='airflow')
        self.assertEqual(log.action, 'trigger')
        self.assertEqual(log.user, self.admin)
        self.assertIn('POST /api/v2/dags/etl/dagRuns -> 200', log.detail)

    def test_delete_audited(self):
        self.call('delete', 'variables/foo', fake_response(204))
        self.assertEqual(AuditLog.objects.get(resource='airflow').action, 'delete')

    def test_get_not_audited(self):
        self.call('get', 'dags', fake_response(200, {}))
        self.assertFalse(AuditLog.objects.filter(resource='airflow').exists())

    def test_plain_text_logs(self):
        resp, req = self.call('get', 'dags/d/dagRuns/r/taskInstances/t/logs/1', fake_response(200, text='line 1', content_type='text/plain'),
                              HTTP_ACCEPT='text/plain')
        self.assertEqual(resp.content, b'line 1')
        self.assertEqual(req.call_args.kwargs['headers']['Accept'], 'text/plain')

    def test_airflow_unreachable_502(self):
        with patch('requests.Session.post', side_effect=requests.ConnectionError('down')):
            resp = self.client.get('/pipelines/api/v2/dags')
        self.assertEqual(resp.status_code, 502)
        self.assertIn('Cannot reach Airflow', resp.json()['detail'])

    def test_csrf_enforced_for_mutations(self):
        c = TestClient(enforce_csrf_checks=True)
        c.force_login(self.admin)
        resp = c.post('/pipelines/api/v2/dags/d/dagRuns', data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 403)

    def test_spec_is_trimmed_and_cached(self):
        spec = {'paths': {'/api/v2/dags': {}, '/api/v2/auth/login': {}, '/ui/foo': {}}}
        with patch('airflow_ui.proxy.requests.get', return_value=fake_response(200, spec)) as get:
            first = self.client.get('/pipelines/api/spec/').json()
            self.client.get('/pipelines/api/spec/')
        self.assertEqual(list(first['paths']), ['/api/v2/dags'])
        get.assert_called_once()


# =============================================================================
# SPA SHELL
# =============================================================================

@override_settings(**AIRFLOW_SETTINGS, AIRFLOW_PUBLIC_URL='https://airflow.public')
class AppViewTest(TestCase):

    def setUp(self):
        self.client = TestClient()

    def test_superuser_gets_shell_with_config(self):
        self.client.force_login(make_user('admin', is_superuser=True))
        resp = self.client.get(reverse('airflow:dashboard'))
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, 'airflow/app.html')
        self.assertEqual(resp.context['sidebar_section'], 'pipelines')
        cfg = resp.context['airflow_config']
        self.assertEqual(cfg['api'], '/pipelines/api/v2/')
        self.assertEqual(cfg['airflowUrl'], 'https://airflow.public')
        self.assertContains(resp, 'id="af-config"')
        self.assertNotContains(resp, 'AIRFLOW_PASSWORD')

    def test_deep_links_serve_shell(self):
        self.client.force_login(make_user('admin', is_superuser=True))
        for path in ['/pipelines/dags', '/pipelines/dags/etl/runs/manual__2026-01-01T00:00:00+00:00/tasks/t/3',
                     '/pipelines/connections', '/pipelines/explorer']:
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_named_routes_still_reverse(self):
        self.assertEqual(reverse('airflow:dag_detail', kwargs={'dag_id': 'etl'}), '/pipelines/dags/etl/')
        self.assertEqual(reverse('airflow:dag_run_detail', kwargs={'dag_id': 'e', 'run_id': 'r'}), '/pipelines/dags/e/runs/r/')

    def test_regular_user_403(self):
        self.client.force_login(make_user('plain'))
        self.assertEqual(self.client.get('/pipelines/').status_code, 403)

    def test_anonymous_redirected(self):
        resp = self.client.get('/pipelines/dags')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/auth/login/', resp['Location'])


# =============================================================================
# DRF API (/api/v1/pipelines/)
# =============================================================================

@override_settings(**AIRFLOW_SETTINGS)
class PipelinesAPITest(AirflowTestCase):

    def setUp(self):
        super().setUp()
        self.admin = make_user('admin', is_superuser=True)
        self.client = TestClient()
        self.client.force_login(self.admin)

    @patch('airflow_ui.api.AirflowService.get_dags', return_value=[{'dag_id': 'a'}])
    def test_dag_list(self, _):
        resp = self.client.get('/api/v1/pipelines/dags/')
        self.assertEqual(resp.json(), {'count': 1, 'dags': [{'dag_id': 'a'}]})

    @patch('airflow_ui.api.AirflowService.get_dags', side_effect=AirflowError('down'))
    def test_dag_list_error_is_502(self, _):
        self.assertEqual(self.client.get('/api/v1/pipelines/dags/').status_code, 502)

    def test_non_superuser_forbidden(self):
        c = TestClient()
        c.force_login(make_user('plain'))
        self.assertEqual(c.get('/api/v1/pipelines/dags/').status_code, 403)

    @patch('airflow_ui.api.AirflowService.get_dag_runs')
    def test_runs_serialize_airflow3_fields(self, get_runs):
        get_runs.return_value = [{'dag_run_id': 'r', 'dag_id': 'd', 'state': 'success', 'note': None,
                                  'run_after': '2026-01-01T00:00:00Z', 'duration': 1.5, 'conf': {'a': 1}}]
        data = self.client.get('/api/v1/pipelines/dags/d/runs/?limit=500').json()
        get_runs.assert_called_once_with('d', limit=100)
        self.assertEqual(data['dag_runs'][0]['duration'], 1.5)
        self.assertEqual(data['dag_runs'][0]['conf'], {'a': 1})
        self.assertIsNone(data['dag_runs'][0]['note'])

    def test_runs_bad_limit(self):
        self.assertEqual(self.client.get('/api/v1/pipelines/dags/d/runs/?limit=x').status_code, 400)

    def test_trigger_sends_conf_and_null_logical_date(self):
        with patch('requests.Session.post', return_value=TOKEN_RESPONSE), \
                patch('requests.Session.request', return_value=fake_response(200, {'dag_run_id': 'mine', 'state': 'queued'})) as req:
            resp = self.client.post('/api/v1/pipelines/dags/etl/trigger/', {'conf': {'x': 1}, 'dag_run_id': 'mine'},
                                    content_type='application/json')
        self.assertTrue(resp.json()['ok'])
        body = req.call_args.kwargs['json']
        self.assertIsNone(body['logical_date'])
        self.assertEqual(body['conf'], {'x': 1})
        self.assertEqual(body['dag_run_id'], 'mine')
        self.assertTrue(AuditLog.objects.filter(action='trigger', resource_id='etl').exists())

    def test_trigger_rejects_non_object_conf(self):
        resp = self.client.post('/api/v1/pipelines/dags/etl/trigger/', {'conf': [1]}, content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_trigger_conflict_relayed(self):
        with patch('requests.Session.post', return_value=TOKEN_RESPONSE), \
                patch('requests.Session.request', return_value=fake_response(409, {'detail': 'exists'})):
            resp = self.client.post('/api/v1/pipelines/dags/etl/trigger/', {}, content_type='application/json')
        self.assertEqual(resp.status_code, 409)
        self.assertFalse(resp.json()['ok'])

    @patch('airflow_ui.api.AirflowService.get_dag_runs')
    @patch('airflow_ui.api.AirflowService.get_dag_stats')
    @patch('airflow_ui.api.AirflowService.get_dags')
    def test_sync_populates_summaries(self, get_dags, get_stats, get_runs):
        DAGSummary.objects.create(dag_id='gone')
        get_dags.return_value = [{'dag_id': 'etl', 'is_paused': True, 'description': 'x'}]
        get_stats.return_value = [{'dag_id': 'etl', 'stats': [{'state': 'success', 'count': 3}, {'state': 'failed', 'count': 1}]}]
        get_runs.return_value = [{'dag_id': 'etl', 'state': 'failed', 'run_after': '2026-01-01T00:00:00Z'}]
        resp = self.client.post('/api/v1/pipelines/summaries/sync/')
        self.assertEqual(resp.json()['synced'], 1)
        s = DAGSummary.objects.get(dag_id='etl')
        self.assertEqual((s.total_runs, s.successful_runs, s.failed_runs, s.last_run_state), (4, 3, 1, 'failed'))
        self.assertTrue(s.is_paused)
        self.assertFalse(DAGSummary.objects.get(dag_id='gone').is_active)


# =============================================================================
# BASE URL RESOLUTION (host vs Docker network, root vs /airflow path)
# =============================================================================

@override_settings(**AIRFLOW_SETTINGS)
class BaseUrlResolutionTest(AirflowTestCase):
    """The dashboard and datapipelines Airflow stacks serve the API under different paths."""

    def test_switches_to_airflow_prefix_when_root_404s(self):
        client = AirflowClient('http://localhost:8080', 'u', 'p')
        posts = [fake_response(404, {'detail': 'Not Found'}), fake_response(201, {'access_token': 'jwt-p'})]
        with patch.object(client.session, 'post', side_effect=posts) as post, \
                patch.object(client.session, 'request', return_value=fake_response(200, {'dags': []})) as req:
            client.get_json('/dags')
        self.assertEqual([c.args[0] for c in post.call_args_list],
                         ['http://localhost:8080/auth/token', 'http://localhost:8080/airflow/auth/token'])
        self.assertEqual(client.base_url, 'http://localhost:8080/airflow')
        self.assertEqual(req.call_args.args[1], 'http://localhost:8080/airflow/api/v2/dags')

    def test_drops_airflow_prefix_when_server_is_at_root(self):
        client = AirflowClient('http://localhost:8080/airflow/', 'u', 'p')
        posts = [fake_response(405, {'detail': 'Method Not Allowed'}), fake_response(201, {'access_token': 'jwt-r'})]
        with patch.object(client.session, 'post', side_effect=posts):
            self.assertEqual(client.token(), 'jwt-r')
        self.assertEqual(client.base_url, 'http://localhost:8080')

    def test_both_layouts_missing_reports_login_failure(self):
        client = AirflowClient('http://localhost:8080', 'u', 'p')
        with patch.object(client.session, 'post', return_value=fake_response(404, {'detail': 'Not Found'})):
            with self.assertRaises(AirflowError) as ctx:
                client.token()
        self.assertIn('login failed (404)', str(ctx.exception))
        self.assertEqual(client.base_url, 'http://localhost:8080')

    def test_bad_credentials_do_not_trigger_path_probing(self):
        client = AirflowClient('http://localhost:8080', 'u', 'wrong')
        with patch.object(client.session, 'post', return_value=fake_response(401, {'detail': 'Invalid'})) as post:
            with self.assertRaises(AirflowError):
                client.token()
        post.assert_called_once()

    def test_custom_path_has_no_alternate(self):
        client = AirflowClient('https://example.org/pipelines', 'u', 'p')
        with patch.object(client.session, 'post', return_value=fake_response(404, {})) as post:
            with self.assertRaises(AirflowError):
                client.token()
        post.assert_called_once()

    def test_unresolvable_docker_hostname_explains_the_fix(self):
        client = AirflowClient('http://airflow-apiserver:8080', 'u', 'p')
        err = requests.ConnectionError("Max retries exceeded (Caused by NameResolutionError(\"Failed to resolve "
                                       "'airflow-apiserver' ([Errno -2] Name or service not known)\"))")
        with patch.object(client.session, 'post', side_effect=err), \
                patch.dict('os.environ', {'AIRFLOW_HOST_PORT': '8081'}):
            with self.assertRaises(AirflowError) as ctx:
                client.token()
        msg = str(ctx.exception)
        self.assertIn('Cannot reach Airflow at http://airflow-apiserver:8080', msg)
        self.assertIn('only resolve inside Airflow', msg)
        self.assertIn('AIRFLOW_BASE_URL', msg)
        self.assertIn('http://localhost:8080', msg)
        self.assertIn('AIRFLOW_HOST_PORT=8081', msg)

    def test_other_connection_errors_have_no_docker_hint(self):
        client = AirflowClient('http://localhost:8080', 'u', 'p')
        with patch.object(client.session, 'post', side_effect=requests.ConnectionError('Connection refused')):
            with self.assertRaises(AirflowError) as ctx:
                client.token()
        self.assertNotIn('Docker', str(ctx.exception))

    def test_api_spec_falls_back_to_airflow_prefix_without_login(self):
        admin = make_user('specadmin', is_superuser=True)
        web = TestClient()
        web.force_login(admin)
        spec = fake_response(200, {'paths': {'/api/v2/dags': {'get': {}}, '/api/v2/auth/token': {}, '/ui/x': {}}})
        with override_settings(AIRFLOW_BASE_URL='http://localhost:8080'), \
                patch('requests.Session.post') as post, \
                patch('airflow_ui.proxy.requests.get', side_effect=[fake_response(404, {}), spec]) as get:
            resp = web.get('/pipelines/api/spec/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual([c.args[0] for c in get.call_args_list],
                         ['http://localhost:8080/openapi.json', 'http://localhost:8080/airflow/openapi.json'])
        self.assertEqual(list(resp.json()['paths']), ['/api/v2/dags'])
        post.assert_not_called()   # the spec is public; it must not depend on credentials

    def test_api_spec_reports_unreachable_airflow_as_502(self):
        admin = make_user('specadmin2', is_superuser=True)
        web = TestClient()
        web.force_login(admin)
        with patch('airflow_ui.proxy.requests.get', side_effect=requests.ConnectionError('Name or service not known')):
            resp = web.get('/pipelines/api/spec/')
        self.assertEqual(resp.status_code, 502)

    def test_candidate_bases(self):
        self.assertEqual(AirflowClient('http://h:8080', 'u', 'p').candidate_bases(), ['http://h:8080', 'http://h:8080/airflow'])
        self.assertEqual(AirflowClient('http://h:8080/airflow', 'u', 'p').candidate_bases(), ['http://h:8080/airflow', 'http://h:8080'])
        self.assertEqual(AirflowClient('https://x.org/pipes', 'u', 'p').candidate_bases(), ['https://x.org/pipes'])
