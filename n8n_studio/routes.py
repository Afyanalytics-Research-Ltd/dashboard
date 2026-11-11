"""
Allowlist of n8n public API (``/api/v1``) routes the browser proxy may call.

Generated from ``/api/v1/openapi.yml`` of the running n8n 2.36.8 server (every
path and method in the spec). ``/workflows/{id}`` and ``/workflows/{workflowId}``
are the same route in the spec under two parameter names and are merged here.
Regenerate with ``python manage.py n8n_sync_routes`` after upgrading n8n.
"""

import re

N8N_SPEC_VERSION = '2.36.8'

ROUTES: dict[str, tuple[str, ...]] = {
    '/audit': ('POST',),
    '/settings/ldap': ('GET', 'PUT'),
    '/settings/ldap/sync': ('GET', 'POST'),
    '/settings/security-policy': ('GET', 'PUT'),
    '/settings/otel': ('GET', 'PUT'),
    '/settings/otel/test-trace': ('POST',),
    '/settings/sso/oidc': ('GET', 'PUT'),
    '/settings/sso/saml': ('GET', 'PUT'),
    '/credentials': ('GET', 'POST'),
    '/credentials/{id}': ('GET', 'PATCH', 'DELETE'),
    '/credentials/{id}/test': ('POST',),
    '/credentials/schema/{credentialTypeName}': ('GET',),
    '/credentials/{id}/transfer': ('PUT',),
    '/executions': ('GET',),
    '/executions/{id}': ('GET', 'DELETE'),
    '/executions/{id}/retry': ('POST',),
    '/executions/{id}/stop': ('POST',),
    '/executions/stop': ('POST',),
    '/executions/{id}/tags': ('GET', 'PUT'),
    '/tags': ('GET', 'POST'),
    '/tags/{id}': ('GET', 'PUT', 'DELETE'),
    '/workflows': ('GET', 'POST'),
    '/workflows/{id}/{versionId}': ('GET',),
    '/workflows/{id}/activate': ('POST',),
    '/workflows/{id}/deactivate': ('POST',),
    '/workflows/{id}/publish': ('POST',),
    '/workflows/{id}/unpublish': ('POST',),
    '/workflows/{id}/archive': ('POST',),
    '/workflows/{id}/unarchive': ('POST',),
    '/workflows/{id}/transfer': ('PUT',),
    '/workflows/{id}/test-runs': ('GET', 'POST'),
    '/workflows/{id}/test-runs/{runId}': ('GET',),
    '/workflows/{id}/test-runs/{runId}/cancel': ('POST',),
    '/workflows/{id}/test-runs/{runId}/test-cases': ('GET',),
    '/users': ('GET', 'POST'),
    '/users/{id}': ('GET', 'DELETE'),
    '/users/{id}/role': ('PATCH',),
    '/source-control/pull': ('POST',),
    '/variables': ('GET', 'POST'),
    '/variables/{id}': ('PUT', 'DELETE'),
    '/data-tables': ('GET', 'POST'),
    '/data-tables/{dataTableId}': ('GET', 'PATCH', 'DELETE'),
    '/data-tables/{dataTableId}/rows': ('GET', 'POST'),
    '/data-tables/{dataTableId}/rows/update': ('PATCH',),
    '/data-tables/{dataTableId}/rows/upsert': ('POST',),
    '/data-tables/{dataTableId}/rows/clear': ('DELETE',),
    '/data-tables/{dataTableId}/rows/delete': ('DELETE',),
    '/data-tables/{dataTableId}/columns': ('GET', 'POST'),
    '/data-tables/{dataTableId}/columns/{columnId}': ('PATCH', 'DELETE'),
    '/projects': ('GET', 'POST'),
    '/projects/{projectId}': ('PUT', 'DELETE'),
    '/projects/{projectId}/users': ('GET', 'POST'),
    '/projects/{projectId}/users/{userId}': ('PATCH', 'DELETE'),
    '/community-packages': ('GET', 'POST'),
    '/community-packages/{name}': ('PATCH', 'DELETE'),
    '/discover': ('GET',),
    '/insights/summary': ('GET',),
    '/projects/{projectId}/folders': ('GET', 'POST'),
    '/projects/{projectId}/folders/{folderId}': ('GET', 'PATCH', 'DELETE'),
    '/n8n-packages/export': ('POST',),
    '/n8n-packages/import': ('POST',),
    '/settings/log-streaming/event-types': ('GET',),
    '/settings/log-streaming/destinations': ('GET', 'POST'),
    '/settings/log-streaming/destinations/{id}/test': ('POST',),
    '/settings/log-streaming/destinations/{id}': ('GET', 'PUT', 'DELETE'),
    '/role-mapping-rules': ('POST',),
    '/roles': ('GET', 'POST'),
    '/roles/{slug}': ('GET', 'PUT'),
    '/workflows/{workflowId}': ('GET', 'PUT', 'DELETE'),
    '/workflows/{workflowId}/history': ('GET',),
    '/workflows/{workflowId}/tags': ('GET', 'PUT'),
}


def _compile(template: str) -> re.Pattern:
    out, pos = '^', 0
    for m in re.finditer(r'\{(\w+)\}', template):
        out += re.escape(template[pos:m.start()]) + r'[^/]+'
        pos = m.end()
    return re.compile(out + re.escape(template[pos:]) + '$')


_COMPILED = [(_compile(t), t, set(m)) for t, m in ROUTES.items()]
# Literal segments win over parameters: ``/workflows/{id}/history`` before
# ``/workflows/{id}/{versionId}``, ``/executions/stop`` before ``/executions/{id}``.
_COMPILED.sort(key=lambda r: (r[1].count('{'), -len(r[1])))


def match_route(path: str, method: str) -> tuple[str | None, bool]:
    """Return ``(template, method_allowed)``; ``template`` is ``None`` when nothing matches."""
    method = method.upper()
    first = None
    for regex, template, methods in _COMPILED:
        if regex.match(path):
            if method in methods:
                return template, True
            first = first or template
    return first, False


def operation_count() -> int:
    return sum(len(m) for m in ROUTES.values())
