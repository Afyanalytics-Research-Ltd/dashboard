"""
Model Studio (wandb_ui) tests. W&B HTTP is mocked at requests.Session.
"""

import json
from unittest.mock import MagicMock, patch

import requests
from django.contrib.auth import get_user_model
from django.test import Client as TestClient
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import AuditLog

from .client import WandbClient, WandbError, reset_client
from .gql import check_document
from .models import SavedView
from .operations import ALLOWED_MUTATIONS, ALLOWED_QUERIES, BLOCKED
from .proxy import _allowed_file_url

User = get_user_model()
WANDB = dict(WANDB_BASE_URL='http://wandb-local:8080', WANDB_HOST_URL='http://localhost:8090', WANDB_API_KEY='k123')


def make_user(username='u', superuser=False):
    user = User.objects.create_user(username=username, password='pw-123456', email=f'{username}@t.io')
    if superuser:
        user.is_superuser = user.is_staff = True
        user.save()
    return user


def fake(status=200, payload=None, content_type='application/json', body=None, headers=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.content = body if body is not None else json.dumps(payload or {}).encode()
    resp.headers = {'Content-Type': content_type, **(headers or {})}
    return resp


def gql(query, variables=None):
    return json.dumps({'query': query, 'variables': variables or {}})


# =============================================================================
# OPERATIONS ALLOWLIST + DOCUMENT VALIDATION
# =============================================================================

class OperationsTest(TestCase):

    def test_lists_are_disjoint_and_sized(self):
        self.assertEqual(len(ALLOWED_QUERIES), 67)
        self.assertEqual(len(ALLOWED_MUTATIONS), 248)
        self.assertFalse(BLOCKED & (ALLOWED_QUERIES | ALLOWED_MUTATIONS))

    def test_core_features_allowed(self):
        for q in ('project', 'projects', 'runs', 'artifact', 'artifactCollection', 'viewer', 'entity', 'views', 'serverInfo'):
            self.assertIn(q, ALLOWED_QUERIES)
        for m in ('upsertBucket', 'deleteRun', 'deleteRuns', 'moveRuns', 'upsertSweep', 'upsertView', 'addAliases',
                  'linkArtifact', 'createTeam', 'createUser', 'upsertModel', 'createEmailSubscription', 'deleteAlertSubscription'):
            self.assertIn(m, ALLOWED_MUTATIONS)

    def test_cloud_billing_and_catastrophic_blocked(self):
        for m in ('deleteOrganization', 'purgeUser', 'upgradeCustomerSubscription', 'attachCustomerPaymentMethod',
                  'createSandbox', 'submitSlurmCommand', 'generateLocalLicense'):
            self.assertIn(m, BLOCKED)


class CheckDocumentTest(TestCase):

    def test_plain_query(self):
        c = check_document('query { viewer { id } project(name: "p", entityName: "e") { id } }')
        self.assertTrue(c.ok)
        self.assertEqual(c.queries, ('viewer', 'project'))
        self.assertTrue(c.read_only)

    def test_mutation_collected(self):
        c = check_document('mutation M($id: ID!) { upsertBucket(input: {id: $id, notes: "x"}) { bucket { id } } }')
        self.assertTrue(c.ok)
        self.assertEqual(c.mutations, ('upsertBucket',))
        self.assertFalse(c.read_only)

    def test_blocked_mutation(self):
        c = check_document('mutation { deleteOrganization(input: {organizationId: "x"}) { success } }')
        self.assertFalse(c.ok)
        self.assertIn('deleteOrganization', c.error)

    def test_unknown_field(self):
        self.assertFalse(check_document('{ notAField }').ok)

    def test_alias_does_not_hide_field(self):
        c = check_document('mutation { safe: deleteOrganization(input: {organizationId: "x"}) { success } }')
        self.assertFalse(c.ok)

    def test_fragment_spread_and_inline_fragment_followed(self):
        doc = 'mutation { ...Bad } fragment Bad on Mutation { purgeUser(input: {username: "x"}) { user { id } } }'
        self.assertFalse(check_document(doc).ok)
        self.assertFalse(check_document('mutation { ... on Mutation { purgeUser(input: {username: "x"}) { user { id } } } }').ok)

    def test_recursive_fragments_terminate(self):
        doc = 'query { ...A } fragment A on Query { ...B viewer { id } } fragment B on Query { ...A }'
        c = check_document(doc)
        self.assertTrue(c.ok)
        self.assertEqual(c.queries, ('viewer',))

    def test_introspection_allowed(self):
        self.assertTrue(check_document('{ __schema { queryType { name } } }').ok)
        self.assertTrue(check_document('{ __typename }').ok)

    def test_subscription_rejected(self):
        self.assertFalse(check_document('subscription { viewer { id } }').ok)

    def test_syntax_error_and_empty(self):
        self.assertIn('syntax', check_document('query {').error)
        self.assertFalse(check_document('').ok)
        self.assertFalse(check_document(None).ok)
        self.assertFalse(check_document('fragment F on Query { viewer { id } }').ok)

    def test_size_limit(self):
        self.assertFalse(check_document('{ viewer { id } }' + ' ' * (300 * 1024)).ok)


# =============================================================================
# CLIENT
# =============================================================================

class ClientTest(TestCase):

    def test_basic_auth_and_no_key(self):
        c = WandbClient('http://w', 'key')
        with patch.object(c.session, 'request', return_value=fake(200, {'data': {}})) as req:
            c.graphql(b'{}', read_only=True)
        self.assertEqual(req.call_args.kwargs['auth'], ('api', 'key'))
        with self.assertRaises(WandbError) as ctx:
            WandbClient('http://w', '').graphql(b'{}', read_only=True)
        self.assertEqual(ctx.exception.status_code, 503)

    @patch('wandb_ui.client.time.sleep')
    def test_queries_retried_mutations_not(self, _):
        c = WandbClient('http://w', 'key')
        with patch.object(c.session, 'request', side_effect=[fake(503), fake(200)]) as req:
            self.assertTrue(c.graphql(b'{}', read_only=True).ok)
        self.assertEqual(req.call_count, 2)
        with patch.object(c.session, 'request', side_effect=requests.ConnectionError('x')) as req:
            with self.assertRaises(WandbError):
                c.graphql(b'{}', read_only=False)
        req.assert_called_once()

    def test_query_helper_raises_on_graphql_errors(self):
        c = WandbClient('http://w', 'key')
        with patch.object(c.session, 'request', return_value=fake(200, {'errors': [{'message': 'nope'}]})):
            with self.assertRaises(WandbError) as ctx:
                c.query('{ viewer { id } }')
        self.assertIn('nope', str(ctx.exception))


# =============================================================================
# PROXY
# =============================================================================

@override_settings(**WANDB)
class GraphqlProxyTest(TestCase):

    def setUp(self):
        reset_client()
        self.admin = make_user('admin', superuser=True)
        self.client = TestClient()
        self.client.force_login(self.admin)

    def tearDown(self):
        reset_client()

    def post(self, body, upstream=None):
        with patch('requests.Session.request', return_value=upstream or fake(200, {'data': {'viewer': {'id': '1'}}})) as req:
            resp = self.client.post('/models/graphql', data=body, content_type='application/json')
        return resp, req

    def test_auth(self):
        self.assertEqual(TestClient().post('/models/graphql', data=gql('{ viewer { id } }'), content_type='application/json').status_code, 401)
        c = TestClient()
        c.force_login(make_user('plain'))
        self.assertEqual(c.post('/models/graphql', data=gql('{ viewer { id } }'), content_type='application/json').status_code, 403)

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get('/models/graphql').status_code, 405)

    def test_query_forwarded(self):
        resp, req = self.post(gql('{ viewer { id } }'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(req.call_args.args[:2], ('POST', 'http://wandb-local:8080/graphql'))
        self.assertEqual(req.call_args.kwargs['auth'], ('api', 'k123'))
        self.assertEqual(json.loads(req.call_args.kwargs['data'])['query'], '{ viewer { id } }')
        self.assertFalse(AuditLog.objects.filter(resource='wandb').exists())

    def test_blocked_operation_not_forwarded(self):
        resp, req = self.post(gql('mutation { deleteOrganization(input: {organizationId: "o"}) { success } }'))
        self.assertEqual(resp.status_code, 403)
        self.assertIn('not available', resp.json()['errors'][0]['message'])
        req.assert_not_called()

    def test_bad_json_and_bad_document(self):
        resp, req = self.post('{nope')
        self.assertEqual(resp.status_code, 400)
        resp, req = self.post(gql('query {'))
        self.assertEqual(resp.status_code, 400)
        req.assert_not_called()

    def test_batch(self):
        body = json.dumps([{'query': '{ viewer { id } }'}, {'query': '{ serverInfo { availableVersion } }'}])
        resp, req = self.post(body, fake(200, [{'data': {}}, {'data': {}}]))
        self.assertEqual(resp.status_code, 200)
        bad = json.dumps([{'query': '{ viewer { id } }'}, {'query': 'mutation { purgeUser(input: {username: "x"}) { user { id } } }'}])
        resp, req = self.post(bad)
        self.assertEqual(resp.status_code, 403)
        req.assert_not_called()

    def test_mutation_audited(self):
        self.post(gql('mutation { deleteRun(input: {id: "r"}) { clientMutationId } }'))
        self.post(gql('mutation { upsertBucket(input: {notes: "n"}) { bucket { id } } }'))
        logs = list(AuditLog.objects.filter(resource='wandb').order_by('id').values_list('action', 'resource_id'))
        self.assertEqual(logs, [('delete', 'deleteRun'), ('update', 'upsertBucket')])

    def test_unreachable(self):
        with patch('requests.Session.request', side_effect=requests.ConnectionError('down')), patch('wandb_ui.client.time.sleep'):
            resp = self.client.post('/models/graphql', data=gql('{ viewer { id } }'), content_type='application/json')
        self.assertEqual(resp.status_code, 502)

    def test_csrf_enforced(self):
        c = TestClient(enforce_csrf_checks=True)
        c.force_login(self.admin)
        self.assertEqual(c.post('/models/graphql', data=gql('{ viewer { id } }'), content_type='application/json').status_code, 403)


@override_settings(**WANDB)
class LockoutGuardTest(TestCase):
    """Studio refuses to revoke its own W&B access (key in use, its user, its admin flag)."""

    # Real shapes from W&B 0.86: the node id of key local-wandb_v1_40Rv... embeds its public id.
    PUBLIC = 'local-wandb_v1_40Rv9u175Oew2nNIy2VJDJD67X2'
    NODE = ('QXBpS2V5VHlwZTphZzlrWlhaLWQyRnVaR0l0Ykc5allXeHlOZ3NTQmtGd2FVdGxlU0lxYkc5allXd3RkMkZ1WkdKZmRqRmZOREJT'
            'ZGpsMU1UYzFUMlYzTW01T1NYa3lWa3BFU2tRMk4xZ3lEQQ==')

    def setUp(self):
        from django.core.cache import cache
        reset_client()
        cache.set('wandb_ui:key_owner', {'user_id': 'VXNlcjo2', 'public_id': self.PUBLIC})
        self.client = TestClient()
        self.client.force_login(make_user('admin', superuser=True))

    def tearDown(self):
        from django.core.cache import cache
        cache.delete('wandb_ui:key_owner')

    def post(self, query, variables=None):
        with patch('requests.Session.request', return_value=fake(200, {'data': {}})) as req:
            resp = self.client.post('/models/graphql', data=gql(query, variables), content_type='application/json')
        return resp, req

    def assertBlocked(self, query, variables=None):
        resp, req = self.post(query, variables)
        self.assertEqual(resp.status_code, 409, query)
        self.assertIn('disconnect Model Studio', resp.json()['errors'][0]['message'])
        req.assert_not_called()

    def assertAllowed(self, query, variables=None):
        resp, req = self.post(query, variables)
        self.assertEqual(resp.status_code, 200, query)
        req.assert_called_once()

    def test_delete_key_in_use_blocked_via_variables_and_literal(self):
        self.assertBlocked('mutation D($i: DeleteApiKeyInput!) { deleteApiKey(input: $i) { success } }', {'i': {'id': self.NODE}})
        self.assertBlocked(f'mutation {{ deleteApiKey(input: {{id: "{self.NODE}"}}) {{ success }} }}')
        self.assertBlocked(f'mutation {{ x: deleteApiKeys(input: {{keys: [{{apiKeyId: "{self.PUBLIC}"}}]}}) {{ success }} }}')

    def test_other_keys_and_users_allowed(self):
        self.assertAllowed('mutation { deleteApiKey(input: {id: "QXBpS2V5VHlwZTpvdGhlcg=="}) { success } }')
        self.assertAllowed('mutation { deleteApiKeys(input: {keys: [{apiKeyId: "local-wandb_v1_SomeOtherKey"}]}) { success } }')
        self.assertAllowed('mutation { deleteUser(input: {id: "VXNlcjo5"}) { user { id } } }')
        self.assertAllowed('mutation { updateUser(input: {id: "VXNlcjo2", name: "New name"}) { user { id } } }')

    def test_key_owner_user_protected(self):
        self.assertBlocked('mutation { deleteUser(input: {id: "VXNlcjo2"}) { user { id } } }')
        self.assertBlocked('mutation { updateUser(input: {id: "VXNlcjo2", admin: false}) { user { id } } }')
        self.assertBlocked('mutation { updateUser(input: {admin: false}) { user { id } } }')
        self.assertBlocked('mutation { updateUserRole(input: {id: "VXNlcjo2", accountType: "USER"}) { user { id } } }')

    @override_settings(WANDB_API_KEY='local-wandb_v1_40Rv9u175Oew2nNIy2VJDJD67X2_SECRETPART')
    def test_owner_lookup_needs_only_viewer_id(self):
        """viewer.apiKeys needs a recent web login; the guard must work without it."""
        from django.core.cache import cache
        cache.delete('wandb_ui:key_owner')
        reset_client()
        viewer = fake(200, {'data': {'viewer': {'id': 'VXNlcjo2'}}})
        with patch('requests.Session.request', return_value=viewer) as req:
            resp = self.client.post('/models/graphql', data=gql(f'mutation {{ deleteApiKey(input: {{id: "{self.NODE}"}}) {{ success }} }}'),
                                    content_type='application/json')
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(req.call_count, 1)  # only the viewer lookup, never the delete
        self.assertNotIn('apiKeys', json.loads(req.call_args.kwargs['data'])['query'])
        self.assertEqual(cache.get('wandb_ui:key_owner'), {'user_id': 'VXNlcjo2', 'public_id': self.PUBLIC})

    def test_public_key_id(self):
        from .proxy import displayable_key_id, public_key_id
        self.assertEqual(public_key_id('local-wandb_v1_ABC_secret'), 'local-wandb_v1_ABC')
        self.assertEqual(public_key_id('a' * 40), 'a' * 40)
        # Never expose a classic key (its id IS the secret).
        self.assertEqual(displayable_key_id('a' * 40), '')
        self.assertEqual(displayable_key_id('local-wandb_v1_ABC_secret'), 'local-wandb_v1_ABC')
        self.assertEqual(displayable_key_id(''), '')

    @override_settings(WANDB_API_KEY='local-wandb_v1_PUBLICID_TOPSECRET')
    def test_page_exposes_only_public_part_of_new_keys(self):
        resp = self.client.get('/models/')
        self.assertEqual(resp.context['models_config']['apiKeyId'], 'local-wandb_v1_PUBLICID')
        self.assertNotContains(resp, 'TOPSECRET')


@override_settings(**WANDB)
class FileProxyTest(TestCase):

    def setUp(self):
        reset_client()
        self.client = TestClient()
        self.client.force_login(make_user('admin', superuser=True))

    def test_url_mapping(self):
        self.assertEqual(_allowed_file_url('http://localhost:8090/files/e/p/run/media/images/a.png?x=1'),
                         'http://wandb-local:8080/files/e/p/run/media/images/a.png?x=1')
        self.assertEqual(_allowed_file_url('http://wandb-local:8080/artifactsV2/default/x'), 'http://wandb-local:8080/artifactsV2/default/x')
        # Presigned local-files links on the loopback alias W&B actually returns.
        self.assertEqual(_allowed_file_url('http://127.0.0.1:8090/local-files/e/p/r/config.yaml?X-Amz-Signature=s'),
                         'http://wandb-local:8080/local-files/e/p/r/config.yaml?X-Amz-Signature=s')
        for bad in ('http://evil.com/files/x', 'file:///etc/passwd', 'http://localhost:8090/graphql', 'http://169.254.169.254/files/x',
                    'nonsense', 'http://127.0.0.1:9999/local-files/x', 'http://localhost:8090.evil.com/files/x', 'http://localhost:99999/files/x'):
            self.assertIsNone(_allowed_file_url(bad), bad)

    def test_streams_file_with_download_name(self):
        with patch('requests.Session.request', return_value=fake(200, body=b'PNG', content_type='image/png')) as req:
            resp = self.client.get('/models/file', {'url': 'http://localhost:8090/files/e/p/r/a.png', 'download': '1'})
        self.assertEqual(resp.content, b'PNG')
        self.assertEqual(resp['Content-Type'], 'image/png')
        self.assertIn('a.png', resp['Content-Disposition'])
        self.assertEqual(req.call_args.args[1], 'http://wandb-local:8080/files/e/p/r/a.png')

    def test_rejects_foreign_url(self):
        with patch('requests.Session.request') as req:
            resp = self.client.get('/models/file', {'url': 'http://example.com/files/x'})
        self.assertEqual(resp.status_code, 400)
        req.assert_not_called()


# =============================================================================
# SAVED VIEWS
# =============================================================================

class SavedViewTest(TestCase):

    def setUp(self):
        self.admin = make_user('admin', superuser=True)
        self.other = make_user('other', superuser=True)
        self.client = TestClient()
        self.client.force_login(self.admin)

    def test_crud(self):
        url = '/models/views/team/proj/'
        resp = self.client.post(url, data=json.dumps({'name': 'Best runs', 'state': {'filters': [1]}}), content_type='application/json')
        self.assertEqual(resp.status_code, 201)
        vid = resp.json()['id']
        self.assertEqual([v['name'] for v in self.client.get(url).json()['views']], ['Best runs'])
        resp = self.client.post(f'/models/views/{vid}/', data=json.dumps({'name': 'Renamed', 'state': {'x': 2}}), content_type='application/json')
        self.assertEqual(resp.json()['state'], {'x': 2})
        self.assertEqual(self.client.delete(f'/models/views/{vid}/').status_code, 200)
        self.assertFalse(SavedView.objects.exists())

    def test_private_views_hidden_from_others(self):
        SavedView.objects.create(entity='t', project='p', name='mine', shared=False, owner=self.other)
        SavedView.objects.create(entity='t', project='p', name='team', shared=True, owner=self.other)
        self.assertEqual([v['name'] for v in self.client.get('/models/views/t/p/').json()['views']], ['team'])
        private = SavedView.objects.get(name='mine')
        self.assertEqual(self.client.get(f'/models/views/{private.id}/').status_code, 404)

    def test_validation(self):
        url = '/models/views/t/p/'
        self.assertEqual(self.client.post(url, data='{}', content_type='application/json').status_code, 400)
        self.assertEqual(self.client.post(url, data=json.dumps({'name': 'x', 'state': []}), content_type='application/json').status_code, 400)


# =============================================================================
# SPA SHELL
# =============================================================================

@override_settings(**WANDB)
class StudioViewTest(TestCase):

    def test_shell_config(self):
        c = TestClient()
        c.force_login(make_user('admin', superuser=True))
        resp = c.get(reverse('wandb_ui:home'))
        self.assertEqual(resp.status_code, 200)
        cfg = resp.context['models_config']
        self.assertEqual(cfg['graphql'], '/models/graphql')
        self.assertEqual(cfg['wandbUrl'], 'http://localhost:8090')
        self.assertTrue(cfg['configured'])
        self.assertNotContains(resp, 'k123')
        for p in ('/models/projects', '/models/team/proj/runs/abc', '/models/explorer'):
            self.assertEqual(c.get(p).status_code, 200, p)

    def test_permissions(self):
        self.assertEqual(TestClient().get('/models/').status_code, 302)
        c = TestClient()
        c.force_login(make_user('plain'))
        self.assertEqual(c.get('/models/').status_code, 403)
