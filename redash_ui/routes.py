"""
Allowlist of Redash API routes the browser proxy may call.

Extracted from the route table of the running Redash 10.1.0 server
(``redash/handlers/api.py`` plus the blueprint routes in ``admin.py``,
``authentication.py``, ``organization.py`` and ``queries.py``).
Paths are relative to the Redash base URL.
"""

import re

ROUTES: dict[str, tuple[str, ...]] = {
    '/api/alerts/<alert_id>': ('GET', 'POST', 'DELETE'),
    '/api/alerts/<alert_id>/mute': ('POST', 'DELETE'),
    '/api/alerts/<alert_id>/subscriptions': ('POST', 'GET'),
    '/api/alerts/<alert_id>/subscriptions/<subscriber_id>': ('DELETE',),
    '/api/alerts': ('POST', 'GET'),
    '/api/dashboards': ('GET', 'POST'),
    '/api/dashboards/<dashboard_id>': ('GET', 'POST', 'DELETE'),
    '/api/dashboards/public/<token>': ('GET',),
    '/api/dashboards/<dashboard_id>/share': ('POST', 'DELETE'),
    '/api/data_sources/types': ('GET',),
    '/api/data_sources': ('GET', 'POST'),
    '/api/data_sources/<data_source_id>/schema': ('GET',),
    '/api/databricks/databases/<data_source_id>': ('GET',),
    '/api/databricks/databases/<data_source_id>/<database_name>/tables': ('GET',),
    '/api/databricks/databases/<data_source_id>/<database_name>/columns/<table_name>': ('GET',),
    '/api/data_sources/<data_source_id>/pause': ('POST', 'DELETE'),
    '/api/data_sources/<data_source_id>/test': ('POST',),
    '/api/data_sources/<data_source_id>': ('GET', 'POST', 'DELETE'),
    '/api/groups': ('POST', 'GET'),
    '/api/groups/<group_id>': ('POST', 'GET', 'DELETE'),
    '/api/groups/<group_id>/members': ('POST', 'GET'),
    '/api/groups/<group_id>/members/<user_id>': ('DELETE',),
    '/api/groups/<group_id>/data_sources': ('POST', 'GET'),
    '/api/groups/<group_id>/data_sources/<data_source_id>': ('POST', 'DELETE'),
    '/api/events': ('POST', 'GET'),
    '/api/queries/favorites': ('GET',),
    '/api/queries/<query_id>/favorite': ('POST', 'DELETE'),
    '/api/dashboards/favorites': ('GET',),
    '/api/dashboards/<object_id>/favorite': ('POST', 'DELETE'),
    '/api/dashboards/my': ('GET',),
    '/api/queries/tags': ('GET',),
    '/api/dashboards/tags': ('GET',),
    '/api/queries/search': ('GET',),
    '/api/queries/recent': ('GET',),
    '/api/queries/archive': ('GET',),
    '/api/queries': ('POST', 'GET'),
    '/api/queries/my': ('GET',),
    '/api/queries/<query_id>/refresh': ('POST',),
    '/api/queries/<query_id>': ('POST', 'GET', 'DELETE'),
    '/api/queries/<query_id>/fork': ('POST',),
    '/api/queries/<query_id>/regenerate_api_key': ('POST',),
    '/api/<object_type>/<object_id>/acl': ('GET', 'POST', 'DELETE'),
    '/api/<object_type>/<object_id>/acl/<access_type>': ('GET',),
    '/api/query_results': ('POST',),
    '/api/queries/<query_id>/dropdown': ('GET',),
    '/api/queries/<query_id>/dropdowns/<dropdown_query_id>': ('GET',),
    '/api/query_results/<query_result_id>.<filetype>': ('POST', 'GET'),
    '/api/query_results/<query_result_id>': ('POST', 'GET'),
    '/api/queries/<query_id>/results': ('POST', 'GET'),
    '/api/queries/<query_id>/results.<filetype>': ('POST', 'GET'),
    '/api/queries/<query_id>/results/<query_result_id>.<filetype>': ('POST', 'GET'),
    '/api/jobs/<job_id>': ('GET', 'DELETE'),
    '/api/queries/<query_id>/jobs/<job_id>': ('GET', 'DELETE'),
    '/api/users': ('GET', 'POST'),
    '/api/users/<user_id>': ('GET', 'POST', 'DELETE'),
    '/api/users/<user_id>/invite': ('POST',),
    '/api/users/<user_id>/reset_password': ('POST',),
    '/api/users/<user_id>/regenerate_api_key': ('POST',),
    '/api/users/<user_id>/disable': ('POST', 'DELETE'),
    '/api/visualizations': ('POST',),
    '/api/visualizations/<visualization_id>': ('POST', 'DELETE'),
    '/api/widgets': ('POST',),
    '/api/widgets/<widget_id>': ('POST', 'DELETE'),
    '/api/destinations/types': ('GET',),
    '/api/destinations/<destination_id>': ('GET', 'POST', 'DELETE'),
    '/api/destinations': ('GET', 'POST'),
    '/api/query_snippets/<snippet_id>': ('GET', 'POST', 'DELETE'),
    '/api/query_snippets': ('POST', 'GET'),
    '/api/settings/organization': ('GET', 'POST'),
    '/api/admin/queries/outdated': ('GET',),
    '/api/admin/queries/rq_status': ('GET',),
    '/api/config': ('GET',),
    '/api/session': ('GET',),
    '/api/organization/status': ('GET',),
    '/api/queries/format': ('POST',),
    '/status.json': ('GET',),
}

# The generic ACL route only makes sense for these object types.
ACL_OBJECT_TYPES = ('queries', 'dashboards')


def _compile(template: str) -> re.Pattern:
    out, pos = '^', 0
    for m in re.finditer(r'<(\w+)>', template):
        out += re.escape(template[pos:m.start()])
        name = m.group(1)
        if name == 'object_type':
            out += '(?:' + '|'.join(ACL_OBJECT_TYPES) + ')'
        elif name == 'filetype':
            out += '(?:csv|tsv|json|xlsx)'
        else:
            out += r'[^/.]+'
        pos = m.end()
    return re.compile(out + re.escape(template[pos:]) + '$')


_COMPILED = [(_compile(t), t, set(m)) for t, m in ROUTES.items()]
# Literal routes win over templated ones (``/api/queries/search`` vs ``/api/queries/<query_id>``).
_COMPILED.sort(key=lambda r: r[1].count('<'))


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
