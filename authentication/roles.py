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
    """Return the set of Django Group names the user belongs to.

    Superusers are treated as members of every role so that all
    role-based checks pass for them automatically.  Unauthenticated or
    ``None`` users get an empty set.

    Args:
        user: A Django User instance (or ``None``).

    Returns:
        A set of role name strings, e.g. ``{"Client Admin"}``.
        Returns an empty set for unauthenticated/anonymous users.
    """
    if not user or not user.is_authenticated:
        return set()
    if user.is_superuser:
        return set(ALL_ROLES)
    return set(user.groups.values_list('name', flat=True))


def in_role(user, *roles) -> bool:
    """Return ``True`` if the user is in *any* of the given roles.

    Non-technical explanation:
        Like checking whether a staff member holds any of a list of
        job titles.  If they have at least one match, they pass.

    Args:
        user: A Django User instance (or ``None``).
        *roles: One or more role name strings from ``ALL_ROLES`` to test
            against, e.g. ``in_role(user, ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN)``.

    Returns:
        ``True`` if the user is in at least one of ``roles`` (or is a
        superuser); ``False`` if ``roles`` is empty or no match is found.
    """
    if not roles:
        return False
    if user and user.is_authenticated and user.is_superuser:
        return True
    return bool(user_roles(user).intersection(roles))


def get_user_role(user) -> str:
    """Return the user's primary role string, preferring the profile for speed.

    The profile field (``user.profile.role``) is checked first to avoid a
    database JOIN on the Groups M2M table.  Falls back to group-based lookup
    if the profile is missing or raises an exception.

    Non-technical explanation:
        Quickly answers the question "what type of admin is this user?" —
        checking their profile card first for speed, then checking the
        permission groups as a backup.

    Args:
        user: A Django User instance (or ``None``).

    Returns:
        One of ``ROLE_CLIENT_ADMIN``, ``ROLE_FACILITIES_ADMIN``, or
        ``ROLE_FACILITY_ADMIN``.  Returns ``ROLE_FACILITY_ADMIN`` as the
        safe default for unauthenticated users.
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
    """Return ``True`` if the user has any of the given roles (profile + group aware).

    Checks the profile role field first (O(1) attribute access), then falls
    back to the group-based :func:`in_role` check.

    Args:
        user: A Django User instance (or ``None``).
        *roles: Role name strings to test against.

    Returns:
        ``True`` if the user matches at least one role; ``False`` otherwise.
    """
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    try:
        return user.profile.role in roles
    except Exception:
        return in_role(user, *roles)


def is_client_admin(user) -> bool:
    """Return ``True`` if the user is a Client Admin (or superuser).

    Args:
        user: A Django User instance.

    Returns:
        ``True`` if the user has the ``ROLE_CLIENT_ADMIN`` role.
    """
    return in_role(user, ROLE_CLIENT_ADMIN)


def is_facilities_admin(user) -> bool:
    """Return ``True`` if the user is a Facilities Admin or higher.

    Args:
        user: A Django User instance.

    Returns:
        ``True`` if the user has the ``ROLE_CLIENT_ADMIN`` or
        ``ROLE_FACILITIES_ADMIN`` role.
    """
    return in_role(user, ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN)


def is_facility_admin(user) -> bool:
    """Return ``True`` if the user has any admin role.

    Args:
        user: A Django User instance.

    Returns:
        ``True`` if the user has any of the three admin roles.
    """
    return in_role(user, ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN, ROLE_FACILITY_ADMIN)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def role_required(*roles):
    """Decorator: require authentication AND at least one of the given roles.

    Wraps a function-based view so that:
    - Unauthenticated users are redirected to the login page (via
      ``@login_required``).
    - Authenticated users who lack all listed roles receive a 403 Forbidden
      response (``PermissionDenied``).
    - Allowed users proceed to the view normally.

    A WARNING-level log entry is written for every rejected access attempt,
    including the username, URL, and required roles.

    Non-technical explanation:
        Like sticking a lock and a sign on a function.  The sign says
        "Client Admins and Facilities Admins only".  Anyone without the
        right badge gets a 403; anyone not logged in gets sent to the
        login page first.

    Args:
        *roles: One or more role name strings.  At least one is required;
            passing none raises :class:`ValueError`.

    Returns:
        A decorator that wraps the target view function.

    Raises:
        ValueError: If called with no roles (e.g. ``@role_required()``).

    Example:
        .. code-block:: python

            @role_required(ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN)
            def my_protected_view(request):
                ...
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
