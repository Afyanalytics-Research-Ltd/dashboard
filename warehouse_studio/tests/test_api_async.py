"""The API decorator serves ASGI requests from a thread pool, so slow Snowflake calls don't serialise."""

import asyncio
import time
from types import SimpleNamespace
from unittest import mock

from django.http import JsonResponse
from django.test import RequestFactory, SimpleTestCase
from django.test.client import AsyncRequestFactory

from warehouse_studio import api

ADMIN = SimpleNamespace(is_authenticated=True, is_superuser=True, get_username=lambda: 'admin')


def slow_session(*_args, **_kwargs):
    time.sleep(0.6)
    return {'connected': True}


class AsgiConcurrencyTests(SimpleTestCase):

    def _asgi(self):
        req = AsyncRequestFactory().get('/warehouse-studio/api/session/')
        req.user = ADMIN
        return req

    def test_concurrent_requests_overlap(self):
        async def two():
            t0 = time.monotonic()
            responses = await asyncio.gather(api.session_view(self._asgi()), api.session_view(self._asgi()))
            return time.monotonic() - t0, responses

        with mock.patch.object(api.session_svc, 'info', side_effect=slow_session), \
                mock.patch.object(api.db_connections, 'close_all') as close_all:
            elapsed, responses = asyncio.run(two())
        self.assertTrue(all(isinstance(r, JsonResponse) and r.status_code == 200 for r in responses))
        self.assertLess(elapsed, 1.05, f'requests were serialised ({elapsed:.2f}s for two 0.6s calls)')
        self.assertEqual(close_all.call_count, 2)

    def test_method_not_allowed_before_auth(self):
        req = AsyncRequestFactory().delete('/warehouse-studio/api/session/')
        resp = asyncio.run(api.session_view(req))
        self.assertEqual(resp.status_code, 405)

    def test_wsgi_requests_stay_on_the_request_thread(self):
        req = RequestFactory().get('/warehouse-studio/api/session/')
        req.user = ADMIN
        with mock.patch.object(api.session_svc, 'info', return_value={'ok': 1}), \
                mock.patch.object(api.db_connections, 'close_all') as close_all:
            resp = asyncio.run(api.session_view(req))
        self.assertEqual(resp.status_code, 200)
        close_all.assert_not_called()

    def test_sync_handler_exposed(self):
        self.assertTrue(callable(api.session_view.sync_handler))
