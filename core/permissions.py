"""
DRF permission classes for role-based access control.
"""

from rest_framework.permissions import BasePermission, SAFE_METHODS

from authentication.roles import (
    ROLE_CLIENT_ADMIN,
    ROLE_FACILITIES_ADMIN,
    ROLE_FACILITY_ADMIN,
    in_role,
)


class IsClientAdmin(BasePermission):
    """Grants access only to users in the Client Admin role (or superusers)."""

    message = 'You must be a Client Admin to perform this action.'

    def has_permission(self, request, view) -> bool:
        return bool(
            request.user
            and request.user.is_authenticated
            and in_role(request.user, ROLE_CLIENT_ADMIN)
        )


class IsFacilitiesAdmin(BasePermission):
    """Grants access to Facilities Admin or higher."""

    message = 'You must be a Facilities Admin to perform this action.'

    def has_permission(self, request, view) -> bool:
        return bool(
            request.user
            and request.user.is_authenticated
            and in_role(request.user, ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN)
        )


class IsFacilityAdmin(BasePermission):
    """Grants access to any admin role."""

    message = 'You must have an admin role to perform this action.'

    def has_permission(self, request, view) -> bool:
        return bool(
            request.user
            and request.user.is_authenticated
            and in_role(
                request.user,
                ROLE_CLIENT_ADMIN,
                ROLE_FACILITIES_ADMIN,
                ROLE_FACILITY_ADMIN,
            )
        )


class IsOwnerOrAdmin(BasePermission):
    """
    Object-level: grants full access to superusers/admins,
    read-only to authenticated users, and write access only to the object's owner.

    The object must have a `user` attribute pointing to the owning user.
    """

    message = 'You do not have permission to modify this resource.'

    def has_permission(self, request, view) -> bool:
        return bool(request.user and request.user.is_authenticated)

    def has_object_permission(self, request, view, obj) -> bool:
        if request.user.is_superuser or in_role(request.user, ROLE_CLIENT_ADMIN):
            return True
        if request.method in SAFE_METHODS:
            return True
        owner = getattr(obj, 'user', None)
        return owner == request.user


class IsSuperuser(BasePermission):
    """Restricts access to superusers only."""

    message = 'Superuser access is required.'

    def has_permission(self, request, view) -> bool:
        return bool(request.user and request.user.is_authenticated and request.user.is_superuser)


class IsAdminOrReadOnly(BasePermission):
    """
    Read access for all authenticated users;
    write access only for Client Admins and superusers.
    """

    def has_permission(self, request, view) -> bool:
        if not (request.user and request.user.is_authenticated):
            return False
        if request.method in SAFE_METHODS:
            return True
        return request.user.is_superuser or in_role(request.user, ROLE_CLIENT_ADMIN)
