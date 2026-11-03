"""
Analytics Studio (redash_ui) tests. Redash HTTP is mocked at requests.Session.
"""

import json
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth import get_user_model
from django.test import Client as TestClient
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import AuditLog

from .client import RedashClient, RedashError, reset_client
from .models import QueryRevision
from .routes import ROUTES, match_route

User = get_user_model()

REDASH_SETTINGS = dict(REDASH_BASE_URL='http://redash.test', REDASH_API_URL='http://redash.test', REDASH_ADMIN_API_KEY='admin-key')


def make_user(username='u', superuser=False):
    user = User.objects.create_user(username=username, password='pw-123456', email=f'{username}@t.io')
    if superuser:
        user.is_superuser = user.is_staff = True
        user.save()
    return user


def fake(status=200, payload=None, content_type='application/json', body=None, headers=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.content = body if body is not None else (json.dumps(payload).encode() if payload is not None else b'')
    resp.headers = {'Content-Type': content_type, **(headers or {})}
    return resp


# =============================================================================
# ROUTES
# =============================================================================

class RouteTest(TestCase):

    def test_every_operation_from_route_table(self):
        self.assertEqual(sum(len(m) for m in ROUTES.values()), 123)

    def test_literal_beats_template(self):
        for path in ('/api/queries/search', '/api/queries/my', '/api/queries/archive', '/api/queries/favorites',
                     '/api/dashboards/my', '/api/dashboards/tags', '/api/data_sources/types'):
            self.assertEqual(match_route(path, 'GET'), (path, True), path)

    def test_templates(self):
        self.assertEqual(match_route('/api/queries/12', 'POST'), ('/api/queries/<query_id>', True))
        self.assertEqual(match_route('/api/widgets/7', 'DELETE'), ('/api/widgets/<widget_id>', True))
        self.assertEqual(match_route('/api/queries/3/results/9.csv', 'GET'), ('/api/queries/<query_id>/results/<query_result_id>.<filetype>', True))

    def test_filetype_restricted(self):
        self.assertTrue(match_route('/api/query_results/5.xlsx', 'GET')[1])
        self.assertEqual(match_route('/api/query_results/5.exe', 'GET'), (None, False))

    def test_acl_object_types_restricted(self):
        self.assertTrue(match_route('/api/dashboards/4/acl', 'GET')[1])
        self.assertEqual(match_route('/api/users/1/acl', 'GET'), (None, False))

    def test_method_not_allowed(self):
        self.assertEqual(match_route('/api/queries/1', 'PUT'), ('/api/queries/<query_id>', False))

    def test_unknown(self):
        self.assertEqual(match_route('/login', 'GET'), (None, False))
        self.assertEqual(match_route('/api/nope', 'GET'), (None, False))


# =============================================================================
# CLIENT
# =============================================================================

class ClientTest(TestCase):

    def make(self):
        return RedashClient('http://redash.test', 'k', max_retries=2)

    def test_key_header_and_no_redirect_follow(self):
        c = self.make()
        with patch.object(c.session, 'request', return_value=fake(200, {'ok': 1})) as req:
            self.assertEqual(c.get_json('/api/session'), {'ok': 1})
        self.assertEqual(req.call_args.kwargs['headers']['Authorization'], 'Key k')
        self.assertFalse(req.call_args.kwargs['allow_redirects'])

    def test_missing_key(self):
        with self.assertRaises(RedashError) as ctx:
            RedashClient('http://redash.test', '').request('GET', '/api/queries')
        self.assertEqual(ctx.exception.status_code, 503)

    def test_login_redirect_means_bad_key(self):
        c = self.make()
        with patch.object(c.session, 'request', return_value=fake(302, headers={'Location': '/login?next=x'})):
            with self.assertRaises(RedashError) as ctx:
                c.request('GET', '/api/queries')
        self.assertIn('API key', str(ctx.exception))

    @patch('redash_ui.client.time.sleep')
    def test_get_retried(self, _):
        c = self.make()
        with patch.object(c.session, 'request', side_effect=[requests.ConnectionError('x'), fake(503), fake(200, {})]) as req:
            self.assertTrue(c.request('GET', '/api/queries').ok)
        self.assertEqual(req.call_count, 3)

    @patch('redash_ui.client.time.sleep')
    def test_post_not_retried(self, _):
        c = self.make()
        with patch.object(c.session, 'request', side_effect=requests.ConnectionError('x')) as req:
            with self.assertRaises(RedashError):
                c.request('POST', '/api/query_results', data=b'{}')
        req.assert_called_once()


# =============================================================================
# PROXY
# =============================================================================

@override_settings(**REDASH_SETTINGS)
class ProxyTest(TestCase):

    def setUp(self):
        reset_client()
        self.admin = make_user('admin', superuser=True)
        self.client = TestClient()
        self.client.force_login(self.admin)

    def tearDown(self):
        reset_client()

    def call(self, method, path, upstream, **kw):
        with patch('requests.Session.request', return_value=upstream) as req:
            resp = getattr(self.client, method)(f'/redash/proxy/{path}', **kw)
        return resp, req

    def test_auth_required(self):
        self.assertEqual(TestClient().get('/redash/proxy/api/queries').status_code, 401)
        c = TestClient()
        c.force_login(make_user('plain'))
        self.assertEqual(c.get('/redash/proxy/api/queries').status_code, 403)

    def test_get_passthrough(self):
        resp, req = self.call('get', 'api/queries?page=2&tags=a&tags=b', fake(200, {'count': 0, 'results': []}))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(req.call_args.args[:2], ('GET', 'http://redash.test/api/queries'))
        self.assertEqual(req.call_args.kwargs['params'], [('page', '2'), ('tags', 'a'), ('tags', 'b')])
        self.assertEqual(req.call_args.kwargs['headers']['Authorization'], 'Key admin-key')
        self.assertEqual(resp['Cache-Control'], 'no-store')

    def test_status_json(self):
        _, req = self.call('get', 'status.json', fake(200, {'version': '10.1.0'}))
        self.assertEqual(req.call_args.args[1], 'http://redash.test/status.json')

    def test_download_headers_passthrough(self):
        resp, _ = self.call('get', 'api/query_results/5.csv', fake(200, body=b'a,b\n1,2\n', content_type='text/csv',
                                                                     headers={'Content-Disposition': 'attachment; filename="q.csv"'}))
        self.assertEqual(resp.content, b'a,b\n1,2\n')
        self.assertEqual(resp['Content-Type'], 'text/csv')
        self.assertIn('q.csv', resp['Content-Disposition'])

    def test_unknown_route_and_method(self):
        resp, req = self.call('get', 'login', fake())
        self.assertEqual(resp.status_code, 404)
        resp, req2 = self.call('put', 'api/queries/1', fake(), data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 405)
        req.assert_not_called()
        req2.assert_not_called()

    def test_segments_reencoded(self):
        _, req = self.call('get', 'api/dashboards/my%20dash', fake(200, {}))
        self.assertEqual(req.call_args.args[1], 'http://redash.test/api/dashboards/my%20dash')

    def test_invalid_json(self):
        resp, req = self.call('post', 'api/queries', fake(), data='{bad', content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        req.assert_not_called()

    def test_upstream_error_relayed(self):
        resp, _ = self.call('get', 'api/queries/999', fake(404, {'message': 'Couldn\'t find resource'}))
        self.assertEqual(resp.status_code, 404)
        self.assertIn('find resource', resp.json()['message'])

    def test_unreachable_is_502(self):
        with patch('requests.Session.request', side_effect=requests.ConnectionError('down')), patch('redash_ui.client.time.sleep'):
            resp = self.client.get('/redash/proxy/api/queries')
        self.assertEqual(resp.status_code, 502)
        self.assertIn('Cannot reach Redash', resp.json()['message'])

    def test_csrf_enforced(self):
        c = TestClient(enforce_csrf_checks=True)
        c.force_login(self.admin)
        self.assertEqual(c.post('/redash/proxy/api/queries', data='{}', content_type='application/json').status_code, 403)

    def test_audit_actions(self):
        self.call('post', 'api/dashboards', fake(200, {'id': 1}), data='{"name":"x"}', content_type='application/json')
        self.call('post', 'api/query_results', fake(200, {'job': {}}), data='{"query":"select 1"}', content_type='application/json')
        self.call('delete', 'api/widgets/3', fake(200, {}))
        self.call('post', 'api/dashboards/1', fake(200, {'id': 1}), data='{"name":"y"}', content_type='application/json')
        actions = list(AuditLog.objects.filter(resource='redash').order_by('id').values_list('action', flat=True))
        self.assertEqual(actions, ['create', 'query', 'delete', 'update'])

    def test_get_not_audited(self):
        self.call('get', 'api/queries', fake(200, {}))
        self.assertFalse(AuditLog.objects.filter(resource='redash').exists())

    def test_cannot_revoke_own_key_or_disable_key_owner(self):
        from django.core.cache import cache
        cache.delete('redash_ui:key_owner')
        session = fake(200, {'user': {'id': 1}})
        for path in ('api/users/1/regenerate_api_key', 'api/users/1/disable'):
            with patch('requests.Session.request', return_value=session) as req:
                resp = self.client.post(f'/redash/proxy/{path}', data='{}', content_type='application/json')
            self.assertEqual(resp.status_code, 409, path)
            self.assertIn('owns the API key', resp.json()['message'])
            self.assertTrue(all(c.args[1].endswith('/api/session') for c in req.call_args_list))

    def test_other_users_key_can_be_regenerated(self):
        from django.core.cache import cache
        cache.set('redash_ui:key_owner', 1)
        resp, req = self.call('post', 'api/users/5/regenerate_api_key', fake(200, {'api_key': 'new'}),
                              data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(req.call_args.args[1].endswith('/api/users/5/regenerate_api_key'))


# =============================================================================
# QUERY REVISIONS
# =============================================================================

@override_settings(**REDASH_SETTINGS)
class RevisionTest(TestCase):

    def setUp(self):
        reset_client()
        self.admin = make_user('admin', superuser=True)
        self.client = TestClient()
        self.client.force_login(self.admin)

    def save(self, path, sent, saved, status=200):
        with patch('requests.Session.request', return_value=fake(status, saved)):
            return self.client.post(f'/redash/proxy/{path}', data=json.dumps(sent), content_type='application/json')

    def test_create_and_save_recorded(self):
        self.save('api/queries', {'name': 'A', 'query': 'select 1', 'data_source_id': 1},
                  {'id': 7, 'name': 'A', 'query': 'select 1', 'version': 1, 'data_source_id': 1})
        self.save('api/queries/7', {'query': 'select 2', 'version': 1},
                  {'id': 7, 'name': 'A', 'query': 'select 2', 'version': 2, 'data_source_id': 1})
        revs = list(QueryRevision.objects.filter(query_id=7).values_list('version', 'query_text'))
        self.assertEqual(revs, [(2, 'select 2'), (1, 'select 1')])
        self.assertEqual(QueryRevision.objects.first().user, self.admin)

    def test_unchanged_text_not_duplicated(self):
        saved = {'id': 7, 'name': 'A', 'query': 'select 1', 'version': 1}
        self.save('api/queries/7', {'query': 'select 1'}, saved)
        self.save('api/queries/7', {'query': 'select 1'}, {**saved, 'version': 2})
        self.assertEqual(QueryRevision.objects.count(), 1)

    def test_tag_or_schedule_only_not_recorded(self):
        self.save('api/queries/7', {'tags': ['x']}, {'id': 7, 'name': 'A', 'query': 'q', 'version': 3})
        self.save('api/queries/7', {'schedule': None}, {'id': 7, 'name': 'A', 'query': 'q', 'version': 4})
        self.assertFalse(QueryRevision.objects.exists())

    def test_failed_save_not_recorded(self):
        self.save('api/queries/7', {'query': 'x', 'version': 1}, {'message': 'conflict'}, status=409)
        self.assertFalse(QueryRevision.objects.exists())

    def test_other_routes_not_recorded(self):
        self.save('api/queries/7/fork', {}, {'id': 8, 'name': 'Copy', 'query': 'q'})
        self.assertFalse(QueryRevision.objects.exists())

    def test_history_endpoint(self):
        QueryRevision.objects.create(query_id=3, version=1, name='n', query_text='a', user=self.admin)
        QueryRevision.objects.create(query_id=3, version=2, name='n', query_text='b', user=self.admin)
        QueryRevision.objects.create(query_id=4, version=1, name='other', query_text='z')
        data = self.client.get('/redash/history/3/').json()
        self.assertEqual([r['query'] for r in data['revisions']], ['b', 'a'])
        self.assertEqual(data['revisions'][0]['user'], 'admin')

    def test_history_requires_superuser(self):
        self.assertEqual(TestClient().get('/redash/history/3/').status_code, 401)
        c = TestClient()
        c.force_login(make_user('plain'))
        self.assertEqual(c.get('/redash/history/3/').status_code, 403)


# =============================================================================
# SPA SHELL
# =============================================================================

@override_settings(**REDASH_SETTINGS, REDASH_PUBLIC_URL='https://redash.public')
class StudioViewTest(TestCase):

    def test_superuser_shell(self):
        c = TestClient()
        c.force_login(make_user('admin', superuser=True))
        resp = c.get(reverse('redash_ui:home'))
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, 'redash_ui/app.html')
        cfg = resp.context['studio_config']
        self.assertEqual(cfg['api'], '/redash/proxy/')
        self.assertEqual(cfg['redashUrl'], 'https://redash.public')
        self.assertTrue(cfg['configured'])
        self.assertNotContains(resp, 'admin-key')

    def test_deep_links(self):
        c = TestClient()
        c.force_login(make_user('admin', superuser=True))
        for p in ('/redash/queries', '/redash/queries/12', '/redash/dashboards/start', '/redash/data-sources/new', '/redash/explorer'):
            self.assertEqual(c.get(p).status_code, 200, p)

    @override_settings(REDASH_ADMIN_API_KEY='')
    def test_unconfigured_flag(self):
        c = TestClient()
        c.force_login(make_user('admin', superuser=True))
        self.assertFalse(c.get('/redash/').context['studio_config']['configured'])

    def test_permissions(self):
        resp = TestClient().get('/redash/queries')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/auth/login/', resp['Location'])
        c = TestClient()
        c.force_login(make_user('plain'))
        self.assertEqual(c.get('/redash/').status_code, 403)
