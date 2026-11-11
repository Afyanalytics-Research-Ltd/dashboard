"""
Automation Studio (n8n_studio) backend tests. n8n HTTP is mocked at requests.Session.
"""

import gzip
import json
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client as TestClient
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import AuditLog
from n8n_studio import catalog
from n8n_studio.client import N8nClient, N8nError, N8nSession, reset_clients
from n8n_studio.endpoints import _first_trigger
from n8n_studio.models import WorkflowSnapshot, content_hash
from n8n_studio.routes import ROUTES, match_route, operation_count

User = get_user_model()

N8N = dict(N8N_API_URL='http://n8n.test', N8N_API_KEY='key-123', N8N_PUBLIC_URL='http://n8n.public',
           N8N_WEBHOOK_URL='http://hooks.public', N8N_STUDIO_EMAIL='', N8N_STUDIO_PASSWORD='')


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
    resp.ok = status < 400
    resp.json = lambda: json.loads(resp.content)
    return resp


WF = {'id': 'wf1', 'name': 'Orders', 'active': False, 'versionId': 'v1',
      'nodes': [{'id': 'n1', 'name': 'Manual', 'type': 'n8n-nodes-base.manualTrigger', 'typeVersion': 1, 'position': [0, 0], 'parameters': {}}],
      'connections': {}, 'settings': {'executionOrder': 'v1'}}


# =============================================================================
# ROUTES
# =============================================================================

class RouteTest(TestCase):

    def test_covers_every_operation_of_the_spec(self):
        # 72 paths / 114 operations in the 2.36.8 spec; /workflows/{id} and {workflowId} merged.
        self.assertEqual(operation_count(), 114)
        self.assertEqual(len(ROUTES), 71)

    def test_literal_segments_beat_parameters(self):
        self.assertEqual(match_route('/executions/stop', 'POST'), ('/executions/stop', True))
        self.assertEqual(match_route('/workflows/abc/history', 'GET'), ('/workflows/{workflowId}/history', True))
        self.assertEqual(match_route('/workflows/abc/tags', 'PUT'), ('/workflows/{workflowId}/tags', True))
        self.assertEqual(match_route('/workflows/abc/activate', 'POST'), ('/workflows/{id}/activate', True))
        self.assertEqual(match_route('/credentials/schema/slackApi', 'GET'), ('/credentials/schema/{credentialTypeName}', True))
        self.assertEqual(match_route('/data-tables/t1/rows/update', 'PATCH'), ('/data-tables/{dataTableId}/rows/update', True))

    def test_version_route_and_merged_workflow_route(self):
        self.assertEqual(match_route('/workflows/abc/v-123', 'GET'), ('/workflows/{id}/{versionId}', True))
        for method in ('GET', 'PUT', 'DELETE'):
            self.assertEqual(match_route('/workflows/abc', method), ('/workflows/{workflowId}', True))

    def test_method_not_allowed_and_unknown(self):
        self.assertEqual(match_route('/tags', 'DELETE'), ('/tags', False))
        self.assertEqual(match_route('/nope', 'GET'), (None, False))
        self.assertEqual(match_route('/workflows/a/b/c/d', 'GET'), (None, False))


# =============================================================================
# CLIENT
# =============================================================================

class ClientTest(TestCase):

    def test_sends_key_and_retries_idempotent(self):
        c = N8nClient('http://n8n.test/', 'k', max_retries=2)
        with patch.object(c.session, 'request', side_effect=[fake(503), fake(200, {'data': []})]) as req, patch('time.sleep'):
            resp = c.request('GET', '/workflows')
        self.assertTrue(resp.ok)
        self.assertEqual(req.call_count, 2)
        args, kwargs = req.call_args
        self.assertEqual(args[1], 'http://n8n.test/api/v1/workflows')
        self.assertEqual(kwargs['headers']['X-N8N-API-KEY'], 'k')

    def test_post_is_not_retried(self):
        c = N8nClient('http://n8n.test', 'k')
        with patch.object(c.session, 'request', return_value=fake(503)) as req:
            resp = c.request('POST', '/workflows', data=b'{}')
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(req.call_count, 1)

    def test_missing_key_and_bad_key(self):
        with self.assertRaises(N8nError) as ctx:
            N8nClient('http://n8n.test', '').request('GET', '/tags')
        self.assertEqual(ctx.exception.status_code, 503)
        c = N8nClient('http://n8n.test', 'k')
        with patch.object(c.session, 'request', return_value=fake(401, {'message': 'unauthorized'})):
            with self.assertRaises(N8nError) as ctx:
                c.request('GET', '/tags')
        self.assertEqual(ctx.exception.payload['code'], 'bad_key')

    def test_connection_error_becomes_n8n_error(self):
        c = N8nClient('http://n8n.test', 'k', max_retries=0)
        with patch.object(c.session, 'request', side_effect=requests.ConnectionError('boom')):
            with self.assertRaises(N8nError) as ctx:
                c.request('GET', '/tags')
        self.assertEqual(ctx.exception.payload['code'], 'unreachable')

    def test_iterate_follows_cursor(self):
        c = N8nClient('http://n8n.test', 'k')
        pages = [fake(200, {'data': [1, 2], 'nextCursor': 'c2'}), fake(200, {'data': [3], 'nextCursor': None})]
        with patch.object(c.session, 'request', side_effect=pages) as req:
            self.assertEqual(list(c.iterate('/workflows', limit=2)), [1, 2, 3])
        self.assertEqual(req.call_args_list[1][1]['params']['cursor'], 'c2')

    def test_session_relogs_in_on_401(self):
        s = N8nSession('http://n8n.test', 'a@b.c', 'pw')
        calls = [fake(200, {}), fake(401, {}), fake(200, {}), fake(200, {'data': {'executionId': '9'}})]
        with patch.object(s.session, 'request', side_effect=calls) as req:
            resp = s.request('POST', '/rest/workflows/x/run', json={})
        self.assertEqual(resp.json()['data']['executionId'], '9')
        urls = [c[0][1] for c in req.call_args_list]
        self.assertEqual(urls.count('http://n8n.test/rest/login'), 2)

    def test_session_cookie_usable_over_internal_http(self):
        # n8n behind HTTPS sets a Secure auth cookie; over the internal http URL
        # requests would drop it and every call after login would be a 401.
        def login_sets_secure_cookie(session):
            def fake_login(*args, **kwargs):
                session.cookies.set('n8n-auth', 'tok', domain='n8n.test', path='/', secure=True)
                return fake(200, {})
            return fake_login

        s = N8nSession('http://n8n.test', 'a@b.c', 'pw')
        with patch.object(s.session, 'request', side_effect=login_sets_secure_cookie(s.session)):
            s.login()
        self.assertEqual([c.secure for c in s.session.cookies], [False])
        req = requests.Request('GET', 'http://n8n.test/types/nodes.json')
        self.assertIn('n8n-auth=tok', s.session.prepare_request(req).headers.get('Cookie', ''))

        tls = N8nSession('https://n8n.test', 'a@b.c', 'pw')
        with patch.object(tls.session, 'request', side_effect=login_sets_secure_cookie(tls.session)):
            tls.login()
        self.assertEqual([c.secure for c in tls.session.cookies], [True])  # untouched over https

    def test_session_requires_configuration(self):
        with self.assertRaises(N8nError) as ctx:
            N8nSession('http://n8n.test', '', '').login()
        self.assertEqual(ctx.exception.payload['code'], 'login_not_configured')


# =============================================================================
# PROXY
# =============================================================================

@override_settings(**N8N)
class ProxyTest(TestCase):

    def setUp(self):
        reset_clients()
        self.admin = make_user('admin', superuser=True)
        self.client = TestClient()
        self.client.force_login(self.admin)

    def tearDown(self):
        reset_clients()

    def url(self, sub):
        return reverse('n8n_studio:proxy', kwargs={'subpath': sub})

    def test_requires_login_and_superuser(self):
        anon = TestClient()
        self.assertEqual(anon.get(self.url('workflows')).status_code, 401)
        member = TestClient()
        member.force_login(make_user('member'))
        self.assertEqual(member.get(self.url('workflows')).status_code, 403)

    def test_get_forwards_query_and_returns_body(self):
        with patch('requests.Session.request', return_value=fake(200, {'data': [WF], 'nextCursor': None})) as req:
            resp = self.client.get(self.url('workflows') + '?limit=5&active=true')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['data'][0]['id'], 'wf1')
        self.assertEqual(resp['Cache-Control'], 'no-store')
        args, kwargs = req.call_args
        self.assertEqual(args[1], 'http://n8n.test/api/v1/workflows')
        self.assertIn(('limit', '5'), kwargs['params'])

    def test_unknown_route_and_wrong_method(self):
        self.assertEqual(self.client.get(self.url('internal/secret')).status_code, 404)
        self.assertEqual(self.client.delete(self.url('tags')).status_code, 405)

    def test_path_traversal_rejected(self):
        resp = self.client.get(self.url('workflows/../users'))
        self.assertIn(resp.status_code, (400, 404))

    def test_rejects_invalid_json(self):
        resp = self.client.post(self.url('tags'), data='{nope', content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_upstream_errors_pass_through(self):
        with patch('requests.Session.request', return_value=fake(403, {'message': 'Your license does not allow for feat:variables.'})):
            resp = self.client.get(self.url('variables'))
        self.assertEqual(resp.status_code, 403)
        self.assertIn('feat:variables', resp.json()['message'])

    def test_unreachable_n8n_returns_502_json(self):
        with patch('requests.Session.request', side_effect=requests.ConnectionError('down')), patch('time.sleep'):
            resp = self.client.get(self.url('tags'))
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(resp.json()['code'], 'unreachable')

    @override_settings(N8N_API_KEY='')
    def test_not_configured(self):
        reset_clients()
        resp = self.client.get(self.url('tags'))
        self.assertEqual(resp.status_code, 503)

    def test_mutations_are_audited(self):
        with patch('requests.Session.request', return_value=fake(200, {'id': 't1', 'name': 'x'})):
            self.client.post(self.url('tags'), data=json.dumps({'name': 'x'}), content_type='application/json')
        log = AuditLog.objects.filter(resource='n8n').first()
        self.assertIsNotNone(log)
        self.assertEqual(log.action, 'create')
        self.assertIn('POST /tags', log.detail)

    def test_save_creates_snapshot_and_dedupes(self):
        saved = {**WF, 'versionId': 'v2'}
        with patch('requests.Session.request', return_value=fake(200, saved)):
            body = json.dumps({k: WF[k] for k in ('name', 'nodes', 'connections', 'settings')})
            self.client.put(self.url('workflows/wf1'), data=body, content_type='application/json')
            self.client.put(self.url('workflows/wf1'), data=body, content_type='application/json')
        snaps = WorkflowSnapshot.objects.filter(workflow_id='wf1')
        self.assertEqual(snaps.count(), 1, 'identical content is stored once')
        self.assertEqual(snaps.first().source, 'save')
        self.assertEqual(snaps.first().user, self.admin)

    def test_snapshot_source_header(self):
        with patch('requests.Session.request', return_value=fake(200, WF)):
            self.client.put(self.url('workflows/wf1'), data=json.dumps({'name': 'x', 'nodes': [], 'connections': {}, 'settings': {}}),
                            content_type='application/json', HTTP_X_STUDIO_SNAPSHOT='restore', HTTP_X_STUDIO_NOTE='rolled back')
        snap = WorkflowSnapshot.objects.get()
        self.assertEqual((snap.source, snap.note), ('restore', 'rolled back'))

    def test_delete_snapshots_first(self):
        responses = [fake(200, WF), fake(200, WF)]
        with patch('requests.Session.request', side_effect=responses) as req:
            resp = self.client.delete(self.url('workflows/wf1'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual([c[0][0] for c in req.call_args_list], ['GET', 'DELETE'])
        snap = WorkflowSnapshot.objects.get()
        self.assertEqual(snap.source, 'deleted')
        self.assertEqual(snap.data['nodes'][0]['name'], 'Manual')

    def test_binary_download_keeps_headers(self):
        gz = gzip.compress(b'package')
        with patch('requests.Session.request', return_value=fake(200, body=gz, content_type='application/gzip', headers={'Content-Disposition': 'attachment; filename="export.n8np"'})):
            resp = self.client.post(self.url('n8n-packages/export'), data=json.dumps({'workflowIds': ['wf1']}), content_type='application/json')
        self.assertEqual(resp.content, gz)
        self.assertEqual(resp['Content-Type'], 'application/gzip')
        self.assertIn('export.n8np', resp['Content-Disposition'])

    def test_package_import_is_multipart(self):
        upload = SimpleUploadedFile('p.n8np', b'gzbytes', content_type='application/gzip')
        with patch('requests.Session.request', return_value=fake(200, {'workflows': []})) as req:
            resp = self.client.post(self.url('n8n-packages/import'), data={'package': upload, 'workflowConflictPolicy': 'new-version'})
        self.assertEqual(resp.status_code, 200)
        kwargs = req.call_args[1]
        self.assertEqual(kwargs['files']['package'][0], 'p.n8np')
        self.assertEqual(kwargs['data']['workflowConflictPolicy'], 'new-version')
        self.assertNotIn('Content-Type', kwargs['headers'])

    def test_package_import_requires_file(self):
        self.assertEqual(self.client.post(self.url('n8n-packages/import'), data={'x': '1'}).status_code, 400)

    def test_csrf_enforced_for_mutations(self):
        c = TestClient(enforce_csrf_checks=True)
        c.force_login(self.admin)
        resp = c.post(self.url('tags'), data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 403)

    def test_delete_with_query_filter(self):
        with patch('requests.Session.request', return_value=fake(200, True)) as req:
            resp = self.client.delete(self.url('data-tables/t1/rows/delete') + '?filter=%7B%7D&dryRun=true')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(('dryRun', 'true'), req.call_args[1]['params'])


# =============================================================================
# STUDIO ENDPOINTS
# =============================================================================

@override_settings(**N8N)
class EndpointTest(TestCase):

    def setUp(self):
        reset_clients()
        catalog.reset()
        cache.clear()
        self.client = TestClient()
        self.client.force_login(make_user('admin', superuser=True))

    def tearDown(self):
        reset_clients()

    def test_endpoints_are_superuser_only(self):
        member = TestClient()
        member.force_login(make_user('m'))
        for name, kwargs in [('health', {}), ('catalog', {}), ('snapshots', {}), ('snapshots_deleted', {})]:
            self.assertEqual(member.get(reverse(f'n8n_studio:{name}', kwargs=kwargs)).status_code, 403, name)
        self.assertEqual(member.post(reverse('n8n_studio:webhook'), data='{}', content_type='application/json').status_code, 403)

    def test_catalog_index_is_compact_and_gzipped(self):
        resp = self.client.get(reverse('n8n_studio:catalog'), HTTP_ACCEPT_ENCODING='gzip')
        self.assertEqual(resp['Content-Encoding'], 'gzip')
        data = json.loads(gzip.decompress(resp.content))
        self.assertEqual(data['source'], 'bundled')
        names = {n['name'] for n in data['nodes']}
        self.assertIn('n8n-nodes-base.httpRequest', names)
        http = next(n for n in data['nodes'] if n['name'] == 'n8n-nodes-base.httpRequest')
        self.assertGreater(len(http['versions']), 3)
        self.assertNotIn('properties', http)
        self.assertTrue(next(n for n in data['nodes'] if n['name'] == 'n8n-nodes-base.webhook')['trigger'])
        self.assertGreater(len(data['credentials']), 300)
        self.assertLess(len(resp.content), 200_000)

    def test_catalog_node_detail(self):
        resp = self.client.get(reverse('n8n_studio:catalog_node', kwargs={'node_type': 'n8n-nodes-base.httpRequest'}))
        descs = resp.json()
        self.assertGreaterEqual(len(descs), 2)
        self.assertTrue(all('properties' in d for d in descs))
        self.assertEqual(self.client.get(reverse('n8n_studio:catalog_node', kwargs={'node_type': 'nope.nope'})).status_code, 404)

    def test_catalog_credential_detail(self):
        resp = self.client.get(reverse('n8n_studio:catalog_credential', kwargs={'cred_type': 'httpBasicAuth'}))
        self.assertEqual(resp.json()['name'], 'httpBasicAuth')

    @override_settings(N8N_STUDIO_EMAIL='a@b.c', N8N_STUDIO_PASSWORD='pw')
    def test_live_catalog_falls_back_to_bundled(self):
        reset_clients()
        catalog.reset()
        with patch('requests.Session.request', side_effect=requests.ConnectionError('x')), patch('time.sleep'):
            cat = catalog.get_catalog()
        self.assertEqual(cat.source, 'bundled')

    def test_icon_proxy_validates_and_caches(self):
        url = reverse('n8n_studio:icon', kwargs={'path': 'icons/n8n-nodes-base/dist/nodes/Slack/slack.svg'})
        with patch('requests.get', return_value=fake(200, body=b'<svg/>', content_type='image/svg+xml')) as get:
            r1 = self.client.get(url)
            r2 = self.client.get(url)
        self.assertEqual(r1.content, b'<svg/>')
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(get.call_count, 1)
        self.assertIn("default-src 'none'", r1['Content-Security-Policy'])
        self.assertEqual(self.client.get(reverse('n8n_studio:icon', kwargs={'path': 'rest/login'})).status_code, 400)
        self.assertEqual(self.client.get(reverse('n8n_studio:icon', kwargs={'path': 'icons/../../etc/passwd.svg'})).status_code, 400)

    def test_icon_rejects_non_images(self):
        url = reverse('n8n_studio:icon', kwargs={'path': 'icons/x/y.svg'})
        with patch('requests.get', return_value=fake(200, body=b'<html>', content_type='text/html')):
            self.assertEqual(self.client.get(url).status_code, 404)

    def test_health(self):
        with patch('requests.Session.request', side_effect=[fake(200, {'status': 'ok'}), fake(200, {'data': []})]):
            data = self.client.get(reverse('n8n_studio:health')).json()
        self.assertTrue(data['instance']['ok'])
        self.assertTrue(data['key']['ok'])
        self.assertEqual(data['operations'], 114)
        self.assertFalse(data['login']['configured'])

    def test_run_requires_login_config(self):
        resp = self.client.post(reverse('n8n_studio:run', kwargs={'workflow_id': 'wf1'}), data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()['code'], 'login_not_configured')

    @override_settings(N8N_STUDIO_EMAIL='a@b.c', N8N_STUDIO_PASSWORD='pw')
    def test_run_uses_saved_workflow_and_first_trigger(self):
        reset_clients()
        calls = [fake(200, WF), fake(200, {}), fake(200, {'data': {'executionId': '42'}})]
        with patch('requests.Session.request', side_effect=calls) as req:
            resp = self.client.post(reverse('n8n_studio:run', kwargs={'workflow_id': 'wf1'}), data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()['executionId'], '42')
        run_call = req.call_args_list[-1]
        self.assertTrue(run_call[0][1].endswith('/rest/workflows/wf1/run'))
        self.assertEqual(run_call[1]['json']['triggerToStartFrom'], {'name': 'Manual'})

    @override_settings(N8N_STUDIO_EMAIL='a@b.c', N8N_STUDIO_PASSWORD='pw')
    def test_run_with_unsaved_data_and_destination(self):
        reset_clients()
        calls = [fake(200, {}), fake(200, {'data': {'executionId': '43'}})]
        body = {'workflowData': WF, 'destinationNode': 'Manual'}
        with patch('requests.Session.request', side_effect=calls) as req:
            resp = self.client.post(reverse('n8n_studio:run', kwargs={'workflow_id': 'wf1'}), data=json.dumps(body), content_type='application/json')
        self.assertEqual(resp.json()['executionId'], '43')
        payload = req.call_args_list[-1][1]['json']
        self.assertEqual(payload['destinationNode'], {'nodeName': 'Manual', 'mode': 'inclusive'})
        self.assertNotIn('triggerToStartFrom', payload)

    def test_first_trigger_prefers_manual(self):
        wf = {'nodes': [{'name': 'Hook', 'type': 'n8n-nodes-base.webhook'}, {'name': 'Go', 'type': 'n8n-nodes-base.manualTrigger'}]}
        self.assertEqual(_first_trigger(wf), 'Go')
        self.assertEqual(_first_trigger({'nodes': [{'name': 'Hook', 'type': 'n8n-nodes-base.webhook'}]}), 'Hook')
        self.assertIsNone(_first_trigger({'nodes': [{'name': 'Set', 'type': 'n8n-nodes-base.set'}]}))
        self.assertIsNone(_first_trigger({'nodes': [{'name': 'Go', 'type': 'n8n-nodes-base.manualTrigger', 'disabled': True}]}))

    def test_webhook_tester(self):
        with patch('requests.request', return_value=fake(200, {'ok': True})) as req:
            resp = self.client.post(reverse('n8n_studio:webhook'), data=json.dumps({'path': 'orders', 'method': 'POST', 'body': {'a': 1}, 'mode': 'test'}), content_type='application/json')
        data = resp.json()
        self.assertEqual(data['status'], 200)
        self.assertEqual(data['json'], {'ok': True})
        self.assertEqual(data['url'], 'http://hooks.public/webhook-test/orders')
        self.assertEqual(req.call_args[0][1], 'http://n8n.test/webhook-test/orders')

    def test_webhook_tester_validates(self):
        post = lambda b: self.client.post(reverse('n8n_studio:webhook'), data=json.dumps(b), content_type='application/json')  # noqa: E731
        self.assertEqual(post({'path': '../rest/login'}).status_code, 400)
        self.assertEqual(post({'path': ''}).status_code, 400)
        self.assertEqual(post({'path': 'x', 'method': 'TRACE'}).status_code, 400)
        self.assertEqual(self.client.post(reverse('n8n_studio:webhook'), data='[1]', content_type='application/json').status_code, 400)

    def test_snapshot_listing_detail_and_recycle_bin(self):
        user = User.objects.get(username='admin')
        WorkflowSnapshot.capture(WF, source='save', user=user)
        WorkflowSnapshot.capture({**WF, 'id': 'gone', 'name': 'Gone'}, source='deleted', user=user)
        listing = self.client.get(reverse('n8n_studio:snapshots'), {'workflow': 'wf1'}).json()
        self.assertEqual(listing['total'], 1)
        self.assertNotIn('data', listing['data'][0])
        detail = self.client.get(reverse('n8n_studio:snapshot', kwargs={'pk': listing['data'][0]['id']})).json()
        self.assertEqual(detail['data']['name'], 'Orders')
        with patch('requests.Session.request', return_value=fake(200, {'data': [WF], 'nextCursor': None})):
            deleted = self.client.get(reverse('n8n_studio:snapshots_deleted')).json()
        self.assertEqual([d['workflowId'] for d in deleted['data']], ['gone'])

    def test_capture_all(self):
        page = fake(200, {'data': [WF, {**WF, 'id': 'wf2', 'name': 'Two'}], 'nextCursor': None})
        with patch('requests.Session.request', return_value=page):
            r1 = self.client.post(reverse('n8n_studio:snapshots_capture')).json()
            r2 = self.client.post(reverse('n8n_studio:snapshots_capture')).json()
        self.assertEqual((r1['workflows'], r1['captured']), (2, 2))
        self.assertEqual((r2['captured'], r2['unchanged']), (0, 2))


# =============================================================================
# MODEL, VIEW, COMMANDS
# =============================================================================

class SnapshotModelTest(TestCase):

    def test_hash_ignores_volatile_fields(self):
        a = {**WF, 'updatedAt': '1', 'versionId': 'a'}
        b = {**WF, 'updatedAt': '2', 'versionId': 'b'}
        self.assertEqual(content_hash(a), content_hash(b))
        self.assertNotEqual(content_hash(a), content_hash({**WF, 'name': 'Other'}))

    def test_capture_requires_id_and_nodes(self):
        self.assertIsNone(WorkflowSnapshot.capture({'name': 'x'}))
        self.assertIsNone(WorkflowSnapshot.capture({'id': '1', 'nodes': 'bad'}))
        snap = WorkflowSnapshot.capture(WF)
        self.assertEqual(snap.node_count, 1)
        self.assertIsNone(WorkflowSnapshot.capture(WF))
        self.assertIsNotNone(WorkflowSnapshot.capture(WF, force=True))


@override_settings(**N8N)
class ViewTest(TestCase):

    def test_shell_for_superuser_with_config(self):
        c = TestClient()
        c.force_login(make_user('admin', superuser=True))
        resp = c.get('/automations/workflows/abc')
        self.assertEqual(resp.status_code, 200)
        cfg = resp.context['studio_config']
        self.assertEqual(cfg['api'], '/automations/api/')
        self.assertEqual(cfg['n8nUrl'], 'http://n8n.public')
        self.assertTrue(cfg['configured'])
        self.assertFalse(cfg['canRun'])
        self.assertNotContains(resp, 'key-123')

    def test_anonymous_redirects_member_forbidden(self):
        self.assertEqual(TestClient().get('/automations/').status_code, 302)
        c = TestClient()
        c.force_login(make_user('m'))
        self.assertEqual(c.get('/automations/').status_code, 403)

    def test_command_snapshot(self):
        from io import StringIO

        from django.core.management import call_command
        reset_clients()
        out = StringIO()
        with patch('requests.Session.request', return_value=fake(200, {'data': [WF], 'nextCursor': None})):
            call_command('n8n_snapshot', stdout=out)
        self.assertIn('1 workflows: 1 captured', out.getvalue())
        reset_clients()

