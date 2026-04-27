"""
Role definitions and decorators for the dashboard.

Roles are implemented as Django Groups so they integrate with the standard
admin interface. Use the constants below everywhere instead of bare strings
to avoid typos.

Hierarchy (most -> least privileged):
    CLIENT_ADMIN     -> full access, can manage every facility under a client
    FACILITIES_ADMIN -> manages multiple facilities (a region / cluster)
    FACILITY_ADMIN   -> manages a single facility, mostly read-only on shared tools
"""

from functools import wraps

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied

# --- Role constants ----------------------------------------------------------

ROLE_CLIENT_ADMIN = "Client Admin"
ROLE_FACILITIES_ADMIN = "Facilities Admin"
ROLE_FACILITY_ADMIN = "Facility Admin"

ALL_ROLES = (
    ROLE_CLIENT_ADMIN,
    ROLE_FACILITIES_ADMIN,
    ROLE_FACILITY_ADMIN,
)

DEFAULT_ROLE = ROLE_FACILITY_ADMIN


# --- Helpers -----------------------------------------------------------------


def user_roles(user):
    """Return the set of role names the user belongs to."""
    if not user or not user.is_authenticated:
        return set()
    if user.is_superuser:
        return set(ALL_ROLES)
    return set(user.groups.values_list("name", flat=True))


def in_role(user, *roles):
    """True if the user is in any of the given roles."""
    if not roles:
        return False
    if user and user.is_authenticated and user.is_superuser:
        return True
    return bool(user_roles(user).intersection(roles))


def is_client_admin(user):
    return in_role(user, ROLE_CLIENT_ADMIN)


def is_facilities_admin(user):
    return in_role(user, ROLE_FACILITIES_ADMIN)


def is_facility_admin(user):
    return in_role(user, ROLE_FACILITY_ADMIN)


# --- Decorators --------------------------------------------------------------


def role_required(*roles):
    """
    View decorator: require the user to be authenticated AND in at least one
    of the given roles. Raises PermissionDenied (403) otherwise.

    Usage:
        @role_required(ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN)
        def my_view(request): ...
    """
    if not roles:
        raise ValueError("role_required() requires at least one role")

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def _wrapped(request, *args, **kwargs):
            if not in_role(request.user, *roles):
                raise PermissionDenied(
                    "You don't have permission to access this page."
                )
            return view_func(request, *args, **kwargs)

        return _wrapped

    return decorator