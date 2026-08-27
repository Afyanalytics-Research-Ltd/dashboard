"""
Intent detection and query handlers for the Self-Service Analytics chatbot.

Each handler receives (query, user, access_context) and returns a response dict:
    {
        'content': str,           # Markdown-ish text shown in the panel
        'data':    list | None,   # Optional structured records (for future charts)
        'intent':  str,           # Resolved intent name
    }
"""

import logging
import re

from django.db.models import Q

logger = logging.getLogger('self_service')

# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_INTENT_PATTERNS = {
    'help':        [r'\bhelp\b', r'\bwhat can you\b', r'\bcommands?\b', r'\bwhat do\b'],
    'summary':     [r'\bsummary\b', r'\boverall\b', r'\bhow are we\b', r'\bstatus\b', r'\boverview\b'],
    'facilities':  [r'\bfacil', r'\bhospitals?\b', r'\bclinics?\b', r'\bsites?\b', r'\bbranch\b'],
    'patients':    [r'\bpatients?\b', r'\badmissions?\b', r'\boutpatients?\b', r'\binpatients?\b', r'\bvisits?\b'],
    'staff':       [r'\bstaff\b', r'\bemployees?\b', r'\bworkers?\b', r'\bpersonnel\b', r'\bheadcount\b'],
    'financials':  [r'\bfinanci', r'\brevenue\b', r'\bbilling\b', r'\bcosts?\b', r'\bexpenses?\b', r'\bbudget\b'],
    # NOTE: deliberately does NOT match bare chart/graph/visuali[sz]e words —
    # those are already fully owned by agents/charts.py's is_pure_chart_request
    # / wants_visualization (checked earlier, in consumers.py:receive()). A
    # query like "show me the chart by sex" has real analytical content (the
    # "by sex" breakdown) and must reach the analytics agent, not get stolen
    # into this dashboard-listing bucket just because it says "chart".
    'dashboards':  [r'\bdashboards?\b', r'\breports?\b', r'\banalytics\b'],
    'operations':  [r'\boperati', r'\bperform', r'\befficiency\b', r'\bkpis?\b', r'\bmetrics?\b'],
}


def _detect_intent(query):
    q = query.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return 'general'


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_query(query, user, access_context):
    intent = _detect_intent(query)

    from .security import check_topic_access
    if intent not in ('help', 'general') and not check_topic_access(intent, access_context):
        denied_msg = (
            f"I'm sorry — **{intent}** data is outside your access level "
            f"as a *{access_context['role_display']}*.\n\n"
            f"You can ask about: {', '.join(f'**{t}**' for t in access_context['allowed_topics'])}."
        )
        return {'content': denied_msg, 'data': None, 'intent': intent}

    _handlers = {
        'help':       _handle_help,
        'summary':    _handle_summary,
        'facilities': _handle_facilities,
        'patients':   _handle_patients,
        'staff':      _handle_staff,
        'financials': _handle_financials,
        'dashboards': _handle_dashboards,
        'operations': _handle_operations,
        'general':    _handle_general,
    }

    handler = _handlers.get(intent, _handle_general)
    try:
        result = handler(query, user, access_context)
    except Exception:
        logger.exception("Handler error for intent '%s'", intent)
        result = {'content': 'An error occurred while fetching your data. Please try again.'}

    result['intent'] = intent
    return result


# ---------------------------------------------------------------------------
# Individual handlers
# ---------------------------------------------------------------------------

def _handle_help(query, user, ctx):
    topics = ctx['allowed_topics']
    lines = '\n'.join(f'• **{t.capitalize()}** — ask me about {t}' for t in topics)
    examples = (
        '• *Give me an overview of facility performance*\n'
        '• *How many dashboards are available?*\n'
        '• *Show me patient statistics*'
    )
    return {
        'content': (
            f"Here's what I can help you with as a **{ctx['role_display']}**:\n\n"
            f"{lines}\n\n**Example questions:**\n{examples}"
        ),
        'data': None,
    }


def _handle_summary(query, user, ctx):
    from core.models import Facility
    from analytics_app.models import Dashboard

    scope = ctx['row_scope']
    client = ctx.get('client')
    facility = ctx.get('facility')

    if scope == 'facility' and facility:
        facility_count = 1
        scope_label = facility.name
    elif client:
        facility_count = Facility.objects.filter(client=client, is_active=True).count()
        scope_label = client.name
    else:
        facility_count = Facility.objects.filter(is_active=True).count()
        scope_label = 'all clients'

    dashboard_count = Dashboard.objects.filter(is_active=True).count()

    return {
        'content': (
            f"**Platform Summary — {scope_label}**\n\n"
            f"• Active Facilities: **{facility_count}**\n"
            f"• Active Dashboards: **{dashboard_count}**\n\n"
            f"_Data is scoped to your role as {ctx['role_display']}._"
        ),
        'data': {
            'facility_count': facility_count,
            'dashboard_count': dashboard_count,
        },
    }


def _handle_facilities(query, user, ctx):
    from core.models import Facility

    scope = ctx['row_scope']

    if scope == 'facility':
        facility = ctx.get('facility')
        if facility:
            return {
                'content': (
                    f"**Your Facility**\n\n"
                    f"• **{facility.name}** — Status: {'Active' if facility.is_active else 'Inactive'}\n\n"
                    f"_Your access is limited to this single facility._"
                ),
                'data': [{'name': facility.name, 'is_active': facility.is_active}],
            }
        return {'content': 'No facility is assigned to your account.', 'data': None}

    client = ctx.get('client')
    qs = Facility.objects.filter(is_active=True)
    if client:
        qs = qs.filter(client=client)

    facilities = list(qs.values('name', 'is_active')[:20])
    if not facilities:
        return {'content': 'No active facilities found for your account.', 'data': None}

    lines = '\n'.join(
        f"• **{f['name']}** — {'Active' if f['is_active'] else 'Inactive'}"
        for f in facilities
    )
    return {
        'content': f"**Facilities ({len(facilities)} found)**\n\n{lines}",
        'data': facilities,
    }


def _handle_patients(query, user, ctx):
    return {
        'content': (
            "**Patient Statistics**\n\n"
            "Patient data is aggregated from your connected data warehouse. "
            "Detailed visualisations — scoped to your access level — are available in:\n\n"
            "• **Clinical Dashboard** — inpatient admissions & discharge stats\n"
            "• **Operational Dashboard** — outpatient visits & referrals\n\n"
            "_Patient identifiers are protected per your data governance policy._"
        ),
        'data': None,
    }


def _handle_staff(query, user, ctx):
    return {
        'content': (
            "**Staff & HR Information**\n\n"
            "Staffing data is sourced from your organisation's HR system. Key metrics:\n\n"
            "• Active headcount by department\n"
            "• Attendance and scheduling\n"
            "• Role distribution\n\n"
            "_For detailed staff analytics, visit the **Operational Dashboard**._"
        ),
        'data': None,
    }


def _handle_financials(query, user, ctx):
    return {
        'content': (
            "**Financial Overview**\n\n"
            "Revenue and billing analytics are available in the **Financial Dashboard**. "
            "If you need access to detailed financial breakdowns, please contact your "
            "Client Administrator.\n\n"
            "_Financial data requires elevated access permissions._"
        ),
        'data': None,
    }


def _handle_dashboards(query, user, ctx):
    from analytics_app.models import Dashboard

    client = ctx.get('client')
    qs = Dashboard.objects.filter(is_active=True)
    if client:
        qs = qs.filter(Q(client=client) | Q(is_public=True))

    items = list(qs.values('name', 'category', 'description', 'slug')[:12])
    if not items:
        return {'content': 'No active dashboards found for your account.', 'data': None}

    lines = '\n'.join(
        f"• **{d['name']}** *(_{d['category'].capitalize()}_)* — {d['description'] or 'Analytics dashboard'}"
        for d in items
    )
    return {
        'content': (
            f"**Available Dashboards ({len(items)} found)**\n\n{lines}\n\n"
            "_Visit the Analytics section to open any dashboard._"
        ),
        'data': items,
    }


def _handle_operations(query, user, ctx):
    return {
        'content': (
            "**Operational KPIs**\n\n"
            "Key performance indicators available for your facilities:\n\n"
            "• Bed occupancy rate\n"
            "• Average length of stay (ALOS)\n"
            "• Outpatient throughput\n"
            "• Department utilisation\n"
            "• Waiting times\n\n"
            "_For live KPIs and trend analysis, visit the **Operational Dashboard**._"
        ),
        'data': None,
    }


def _handle_general(query, user, ctx):
    topics = ', '.join(f'**{t}**' for t in ctx['allowed_topics'])
    return {
        'content': (
            "I didn't quite catch that. I can help you with:\n\n"
            f"{topics}\n\n"
            "Type **help** to see example questions."
        ),
        'data': None,
    }
