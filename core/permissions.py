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
    """Grants access only to users in the Client Admin role (or superusers).

    Non-technical explanation:
        Think of this as the VIP door check — only users wearing a "Client
        Admin" badge (or the platform's super-administrator) can get through.
        Everyone else is politely turned away with a 403 response.
    """

    message = 'You must be a Client Admin to perform this action.'

    def has_permission(self, request, view) -> bool:
        """Check whether the requesting user holds the Client Admin role.

        Args:
            request: The incoming HTTP request.
            view: The DRF view being accessed (not used here).

        Returns:
            ``True`` if the user is authenticated and is a Client Admin or
            superuser; ``False`` otherwise.
        """
        return bool(
            request.user
            and request.user.is_authenticated
            and in_role(request.user, ROLE_CLIENT_ADMIN)
        )


class IsFacilitiesAdmin(BasePermission):
    """Grants access to Facilities Admin or any higher role (Client Admin, superuser).

    Non-technical explanation:
        Allows in anyone who manages more than one facility — either a
        "Facilities Admin" (who oversees a cluster) or a "Client Admin"
        (who oversees the whole organisation).
    """

    message = 'You must be a Facilities Admin to perform this action.'

    def has_permission(self, request, view) -> bool:
        """Check whether the user holds Facilities Admin or a higher role.

        Args:
            request: The incoming HTTP request.
            view: The DRF view being accessed.

        Returns:
            ``True`` if the user is a Facilities Admin, Client Admin, or
            superuser; ``False`` otherwise.
        """
        return bool(
            request.user
            and request.user.is_authenticated
            and in_role(request.user, ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN)
        )


class IsFacilityAdmin(BasePermission):
    """Grants access to any admin role (Facility Admin, Facilities Admin, Client Admin, superuser).

    This is the broadest admin gate — it allows in any user who has been
    assigned at least the most basic admin role (Facility Admin).

    Non-technical explanation:
        Allows any staff member with any type of administrator badge,
        regardless of how senior.  If you manage at least one facility,
        you get in.
    """

    message = 'You must have an admin role to perform this action.'

    def has_permission(self, request, view) -> bool:
        """Check whether the user holds any admin role.

        Args:
            request: The incoming HTTP request.
            view: The DRF view being accessed.

        Returns:
            ``True`` if the user holds any of the three admin roles or is
            a superuser; ``False`` otherwise.
        """
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
    """Object-level permission: owners and admins can write; others get read-only.

    - Superusers and Client Admins: full read/write access to every object.
    - Any authenticated user: read-only (GET, HEAD, OPTIONS).
    - The object's owner (``obj.user == request.user``): full read/write to
      their own objects only.

    The object must have a ``user`` attribute pointing to the owning user.

    Non-technical explanation:
        Like a file cabinet where anyone can look through the folders, but
        only the folder's owner (or the manager) can add, change, or remove
        documents inside their own folder.
    """

    message = 'You do not have permission to modify this resource.'

    def has_permission(self, request, view) -> bool:
        """Allow any authenticated user at the view level.

        Object-level checks narrow this down further.

        Args:
            request: The incoming HTTP request.
            view: The DRF view being accessed.

        Returns:
            ``True`` if the user is authenticated; ``False`` otherwise.
        """
        return bool(request.user and request.user.is_authenticated)

    def has_object_permission(self, request, view, obj) -> bool:
        """Allow writes only to the object's owner, admins, and superusers.

        Args:
            request: The incoming HTTP request.
            view: The DRF view being accessed.
            obj: The model instance being checked (must have a ``user`` field).

        Returns:
            ``True`` if the action is allowed; ``False`` otherwise.
        """
        if request.user.is_superuser or in_role(request.user, ROLE_CLIENT_ADMIN):
            return True
        if request.method in SAFE_METHODS:
            return True
        owner = getattr(obj, 'user', None)
        return owner == request.user


class IsSuperuser(BasePermission):
    """Restricts access to Django superusers only.

    Used on endpoints that must never be accessible to any ordinary user,
    regardless of their role — for example, system-settings management.
    """

    message = 'Superuser access is required.'

    def has_permission(self, request, view) -> bool:
        """Allow only Django superusers.

        Args:
            request: The incoming HTTP request.
            view: The DRF view being accessed.

        Returns:
            ``True`` if the user is authenticated AND is a superuser.
        """
        return bool(request.user and request.user.is_authenticated and request.user.is_superuser)


class IsAdminOrReadOnly(BasePermission):
    """Read access for any authenticated user; writes restricted to Client Admins and superusers.

    This is the default permission class on most core ViewSets — it keeps
    the API open for read-only consumption while preventing accidental
    modifications by non-administrative staff.

    Non-technical explanation:
        Everyone logged in can read the data (like looking at a public
        noticeboard), but only administrators can make changes (like being
        the person who updates the noticeboard).
    """

    def has_permission(self, request, view) -> bool:
        """Allow reads for all authenticated users; writes for admins only.

        Args:
            request: The incoming HTTP request.
            view: The DRF view being accessed.

        Returns:
            ``True`` if the user is authenticated and the request is safe
            (GET/HEAD/OPTIONS), OR if the user is a Client Admin / superuser.
            ``False`` for unauthenticated requests or non-admin write attempts.
        """
        if not (request.user and request.user.is_authenticated):
            return False
        if request.method in SAFE_METHODS:
            return True
        return request.user.is_superuser or in_role(request.user, ROLE_CLIENT_ADMIN)
