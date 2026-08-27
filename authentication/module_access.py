"""Effective per-user module access, combining role defaults with explicit overrides.

Module access normally falls out of a user's role — e.g. only Client Admins
can reach the Warehouse module by default, matching the existing
``_is_warehouse_user()`` check in ``warehouse/views.py``/``warehouse/api.py``.
A facility administrator can grant or revoke access to a specific user via
the Permissions page (see ``core.views.PermissionsView``), recorded as a
:class:`authentication.models.UserModuleGrant` row. An explicit override
always wins over the role-based default.
"""

from __future__ import annotations

from authentication.models import UserModuleGrant
from authentication.roles import ROLE_CLIENT_ADMIN, user_has_role

MODULE_WAREHOUSE = UserModuleGrant.MODULE_WAREHOUSE
MODULE_ANALYTICS = UserModuleGrant.MODULE_ANALYTICS
MODULE_SELF_SERVICE = UserModuleGrant.MODULE_SELF_SERVICE

ALL_MODULES = tuple(key for key, _ in UserModuleGrant.MODULE_CHOICES)


def _default_module_access(user, module_key: str) -> bool:
    """Role-based default access, before any explicit override is applied.

    Mirrors the access rules each module already enforced before per-user
    grants existed:
      - warehouse: Client Admin (or superuser) only — see
        ``warehouse.views._is_warehouse_user`` / ``warehouse.api._is_warehouse_user``.
      - analytics / self_service: open to any authenticated user by default
        (their own existing client/facility scoping still applies elsewhere).
    """
    if module_key == MODULE_WAREHOUSE:
        return user_has_role(user, ROLE_CLIENT_ADMIN)
    if module_key in (MODULE_ANALYTICS, MODULE_SELF_SERVICE):
        return True
    return False


def has_module_access(user, module_key: str) -> bool:
    """Return this user's effective access to ``module_key``.

    Resolution order:
      1. Superusers always have access.
      2. An explicit :class:`UserModuleGrant` for this (user, module_key)
         wins, in either direction (grant or revoke).
      3. Otherwise, fall back to the role-based default.
    """
    if not user or not getattr(user, 'is_authenticated', False):
        return False
    if user.is_superuser:
        return True

    grant = UserModuleGrant.objects.filter(user=user, module_key=module_key).first()
    if grant is not None:
        return grant.is_granted

    return _default_module_access(user, module_key)


def get_module_overrides(user) -> dict[str, bool]:
    """Return ``{module_key: is_granted}`` for every explicit override this user has."""
    return dict(
        UserModuleGrant.objects.filter(user=user).values_list('module_key', 'is_granted')
    )
