"""
Role definitions and decorators for Afya DataHub.

Roles are implemented as Django Groups so they integrate with the standard
admin interface and DRF permission classes.  Use the constants below
everywhere instead of bare strings to avoid typos.

Hierarchy (most → least privileged):
    CLIENT_ADMIN     → full access, manages every facility under a client
    FACILITIES_ADMIN → manages multiple facilities (a region / cluster)
    FACILITY_ADMIN   → manages a single facility (default on signup)
"""

import logging
from functools import wraps

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------

ROLE_CLIENT_ADMIN = 'Client Admin'
ROLE_FACILITIES_ADMIN = 'Facilities Admin'
ROLE_FACILITY_ADMIN = 'Facility Admin'

ALL_ROLES = (
    ROLE_CLIENT_ADMIN,
    ROLE_FACILITIES_ADMIN,
    ROLE_FACILITY_ADMIN,
)

DEFAULT_ROLE = ROLE_FACILITY_ADMIN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def user_roles(user) -> set:
    """Return the set of Django Group names the user belongs to."""
    if not user or not user.is_authenticated:
        return set()
    if user.is_superuser:
        return set(ALL_ROLES)
    return set(user.groups.values_list('name', flat=True))


def in_role(user, *roles) -> bool:
    """True if the user is in *any* of the given roles."""
    if not roles:
        return False
    if user and user.is_authenticated and user.is_superuser:
        return True
    return bool(user_roles(user).intersection(roles))


def get_user_role(user) -> str:
    """
    Return the user's primary role string (profile-based fast lookup).
    Falls back to group-based lookup if profile doesn't exist.
    """
    if not user or not user.is_authenticated:
        return ROLE_FACILITY_ADMIN
    if user.is_superuser:
        return ROLE_CLIENT_ADMIN
    try:
        return user.profile.role
    except Exception:
        pass
    # Fallback: infer from Groups
    groups = user_roles(user)
    if ROLE_CLIENT_ADMIN in groups:
        return ROLE_CLIENT_ADMIN
    if ROLE_FACILITIES_ADMIN in groups:
        return ROLE_FACILITIES_ADMIN
    return ROLE_FACILITY_ADMIN


def user_has_role(user, *roles) -> bool:
    """Check if user has any of the given roles (profile + group aware)."""
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    try:
        return user.profile.role in roles
    except Exception:
        return in_role(user, *roles)


def is_client_admin(user) -> bool:
    return in_role(user, ROLE_CLIENT_ADMIN)


def is_facilities_admin(user) -> bool:
    return in_role(user, ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN)


def is_facility_admin(user) -> bool:
    return in_role(user, ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN, ROLE_FACILITY_ADMIN)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def role_required(*roles):
    """
    View decorator: require the user to be authenticated AND in at least one
    of the given roles. Raises PermissionDenied (403) otherwise.

    Usage:
        @role_required(ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN)
        def my_view(request): ...
    """
    if not roles:
        raise ValueError('role_required() requires at least one role.')

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def _wrapped(request, *args, **kwargs):
            if not in_role(request.user, *roles):
                logger.warning(
                    'Permission denied: user=%s path=%s required_roles=%s',
                    getattr(request.user, 'username', 'anonymous'),
                    request.path,
                    roles,
                )
                raise PermissionDenied('You do not have permission to access this page.')
            return view_func(request, *args, **kwargs)

        return _wrapped

    return decorator
