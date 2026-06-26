"""
Role-based data access controls for Self-Service Analytics.

Enforces three layers of security per the spec:
  - Row-Level Security  → row_scope limits which rows a user may see
  - Column-Level Security → denied_columns are stripped from every result set
  - Data Masking        → masked_columns are partially obscured for unauthorised roles
"""

from authentication.roles import (
    ROLE_CLIENT_ADMIN,
    ROLE_FACILITIES_ADMIN,
    ROLE_FACILITY_ADMIN,
    get_user_role,
)

# ---------------------------------------------------------------------------
# Role → access configuration
# ---------------------------------------------------------------------------

_ROLE_ACCESS = {
    ROLE_CLIENT_ADMIN: {
        'allowed_topics': [
            'summary', 'facilities', 'staff',
            'financials', 'patients', 'operations', 'dashboards',
        ],
        'denied_columns': [],
        'masked_columns': [],
        'row_scope': 'client',
        'display': 'Client Administrator',
    },
    ROLE_FACILITIES_ADMIN: {
        'allowed_topics': [
            'summary', 'facilities', 'staff',
            'patients', 'operations', 'dashboards',
        ],
        'denied_columns': ['salary', 'payroll', 'billing_code', 'financial_breakdown'],
        'masked_columns': ['national_id', 'patient_id'],
        'row_scope': 'multi_facility',
        'display': 'Facilities Administrator',
    },
    ROLE_FACILITY_ADMIN: {
        'allowed_topics': ['summary', 'facilities', 'staff', 'patients', 'operations', 'dashboards'],
        'denied_columns': [
            'salary', 'payroll', 'billing_code',
            'financial_breakdown', 'cross_facility_data',
        ],
        'masked_columns': ['national_id', 'patient_id', 'phone_number'],
        'row_scope': 'facility',
        'display': 'Facility Administrator',
    },
}

# Safe fallback for unknown / unauthenticated (should never reach consumer)
_FALLBACK = {
    'allowed_topics': [],
    'denied_columns': [],
    'masked_columns': [],
    'row_scope': 'none',
    'display': 'Guest',
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_user_access_context(user):
    """Build and return the access context dict for *user*.

    Superusers receive full Client Admin rights.
    """
    role = get_user_role(user)
    config = _ROLE_ACCESS.get(role, _FALLBACK)

    profile = getattr(user, 'profile', None)

    return {
        'role': role,
        'role_display': config['display'],
        'allowed_topics': list(config['allowed_topics']),
        'denied_columns': set(config['denied_columns']),
        'masked_columns': set(config['masked_columns']),
        'row_scope': config['row_scope'],
        'client': getattr(profile, 'client', None),
        'facility': getattr(profile, 'facility', None),
        'is_superuser': user.is_superuser,
    }


def check_topic_access(topic, access_context):
    """Return True if *access_context* permits querying *topic*."""
    if access_context.get('is_superuser'):
        return True
    return topic in access_context.get('allowed_topics', [])


def apply_data_filters(records, access_context):
    """Apply Column-Level Security and Data Masking to a list of row dicts.

    - Columns in denied_columns are removed entirely (CLS).
    - Columns in masked_columns are partially anonymised (masking).
    """
    denied = access_context['denied_columns']
    masked = access_context['masked_columns']

    filtered = []
    for row in records:
        clean = {}
        for key, value in row.items():
            if key in denied:
                continue
            if key in masked:
                clean[key] = _mask(value)
            else:
                clean[key] = value
        filtered.append(clean)
    return filtered


def _mask(value):
    """Partially obscure a sensitive field value."""
    if not value:  # handles None and empty string
        return value
    s = str(value)
    if len(s) <= 4:
        return '***'
    return s[:2] + '*' * (len(s) - 4) + s[-2:]
