"""
User profile model for Afya DataHub.

We hang an extensible UserProfile off the existing auth.User via OneToOneField.
Adding new fields is a normal makemigrations + migrate cycle.

Role choices are stored directly on the profile for fast lookup without
hitting the Groups M2M join. The user is ALSO added to the corresponding
Django Group (via the post_save signal) so that Group-based permission
checks (and the existing role decorators that use Groups) continue to work.
"""

import logging

from django.conf import settings
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

from authentication.roles import (
    ROLE_CLIENT_ADMIN,
    ROLE_FACILITIES_ADMIN,
    ROLE_FACILITY_ADMIN,
    in_role,
    user_roles,
)

logger = logging.getLogger(__name__)


class UserProfile(models.Model):
    """
    Extra fields attached to every user. One row per User; created
    automatically by the post_save signal below.

    Role hierarchy (most → least privileged):
        client_admin > facilities_admin > facility_admin
    """

    ROLE_CHOICES = [
        (ROLE_CLIENT_ADMIN, 'Client Admin'),
        (ROLE_FACILITIES_ADMIN, 'Facilities Admin'),
        (ROLE_FACILITY_ADMIN, 'Facility Admin'),
    ]

    ROLE_BADGE_COLORS = {
        ROLE_CLIENT_ADMIN: 'primary',
        ROLE_FACILITIES_ADMIN: 'info',
        ROLE_FACILITY_ADMIN: 'secondary',
    }

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='profile',
    )

    # --- Contact ---------------------------------------------------------------
    phone_number = models.CharField(
        max_length=32,
        blank=True,
        help_text='Primary contact number, including country code.',
    )

    # --- Tenancy / org context -------------------------------------------------
    client = models.ForeignKey(
        'core.Client',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='users',
        help_text='Client / organisation the user belongs to.',
    )
    facility = models.ForeignKey(
        'core.Facility',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='users',
        help_text='Facility the user is primarily assigned to.',
    )

    # --- Profile ---------------------------------------------------------------
    job_title = models.CharField(max_length=120, blank=True)
    bio = models.TextField(blank=True, max_length=500)
    avatar = models.ImageField(upload_to='avatars/%Y/%m/', blank=True, null=True)

    # --- Role ------------------------------------------------------------------
    role = models.CharField(
        max_length=30,
        choices=ROLE_CHOICES,
        default=ROLE_FACILITY_ADMIN,
        db_index=True,
    )

    # --- Security --------------------------------------------------------------
    is_verified = models.BooleanField(default=False)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)

    # --- Audit -----------------------------------------------------------------
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
        ordering = ['-created_at']

    def __str__(self) -> str:
        return f'{self.display_name} ({self.get_role_display()})'

    # --- Computed properties ---------------------------------------------------

    @property
    def display_name(self) -> str:
        """Return the user's full name, falling back to their username.

        Non-technical explanation:
            Produces a friendly name for display in the UI — "Jane Doe"
            if the full name is set, or "jdoe" if not.

        Returns:
            The full name (first + last) or the username if the full name
            is empty.
        """
        return self.user.get_full_name() or self.user.username

    @property
    def initials(self) -> str:
        """Return up to two initials (e.g. "JD") for use as an avatar fallback.

        Takes the first letter of the first name (or username) and the first
        letter of the last name.  If there is no last name, returns only one
        initial.

        Non-technical explanation:
            When a user hasn't uploaded a profile photo, the UI shows a
            coloured circle with their initials instead — like "JD" for
            Jane Doe.  This property produces those initials.

        Returns:
            A 1–2 character uppercase string.
        """
        first = (self.user.first_name[:1] or self.user.username[:1]).upper()
        last = (self.user.last_name[:1]).upper()
        return first + last if last else first

    @property
    def is_client_admin(self) -> bool:
        """Return ``True`` if this profile has Client Admin privileges.

        Superusers are always treated as Client Admins.

        Returns:
            ``True`` if role is ``ROLE_CLIENT_ADMIN`` or the user is a
            Django superuser.
        """
        return self.role == ROLE_CLIENT_ADMIN or self.user.is_superuser

    @property
    def is_facilities_admin(self) -> bool:
        """Return ``True`` if this profile has Facilities Admin or higher privileges.

        Returns:
            ``True`` if role is ``ROLE_CLIENT_ADMIN`` or
            ``ROLE_FACILITIES_ADMIN``, or the user is a superuser.
        """
        return self.role in (ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN) or self.user.is_superuser

    @property
    def role_display_badge(self) -> str:
        """Return the Bootstrap badge colour class for this user's role.

        Used by templates to render coloured role labels (e.g. "Client Admin"
        in blue, "Facility Admin" in grey).

        Returns:
            A Bootstrap contextual colour string, e.g. ``"primary"``,
            ``"info"``, or ``"secondary"``.  Defaults to ``"secondary"``
            for unknown roles.
        """
        return self.ROLE_BADGE_COLORS.get(self.role, 'secondary')

    # --- Role helpers (delegate to authentication.roles) -----------------------

    @property
    def roles(self) -> set:
        """Return the set of Django Group names this user belongs to.

        Delegates to :func:`authentication.roles.user_roles`.  Superusers
        are considered members of all groups.

        Returns:
            A set of role name strings, e.g.
            ``{"Client Admin", "Facilities Admin"}``.
        """
        return user_roles(self.user)

    @property
    def primary_role(self) -> str | None:
        """Return the most-privileged role this user holds, or ``None``.

        Iterates the role hierarchy from most to least privileged and
        returns the first match.  Useful for displaying a single "primary"
        badge in the UI without showing multiple role labels.

        Returns:
            The highest-priority role string the user has, e.g.
            ``"Client Admin"``, or ``None`` if the user has no assigned
            group roles.
        """
        for role in (ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN, ROLE_FACILITY_ADMIN):
            if role in self.roles:
                return role
        return None

    def has_role(self, *roles) -> bool:
        """Return ``True`` if the user belongs to any of the given roles.

        Args:
            *roles: One or more role name strings to check, e.g.
                ``ROLE_CLIENT_ADMIN``, ``ROLE_FACILITIES_ADMIN``.

        Returns:
            ``True`` if the user is in at least one of the supplied roles
            (or is a superuser); ``False`` otherwise.
        """
        return in_role(self.user, *roles)


# ---------------------------------------------------------------------------
# Signals — auto-create a profile whenever a User is created
# ---------------------------------------------------------------------------

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def _ensure_profile(sender, instance, created, **kwargs):
    """Auto-create a :class:`UserProfile` for every new User.

    Triggered by Django's ``post_save`` signal on the User model.  Uses
    ``get_or_create`` so it is idempotent — safe to call multiple times.

    Non-technical explanation:
        Whenever a new user account is created (even by a developer using
        the command line), this automatically creates their profile record
        behind the scenes — like setting up a desk for a new employee on
        their first day.

    Args:
        sender: The User model class (passed by the signal framework).
        instance: The User instance that was just saved.
        created: ``True`` if this is a new record, ``False`` for updates.
        **kwargs: Additional keyword arguments from the signal.
    """
    if created:
        UserProfile.objects.get_or_create(user=instance)
        logger.debug('UserProfile created for user pk=%s', instance.pk)


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def _save_profile(sender, instance, **kwargs):
    """Propagate User saves to the linked UserProfile to avoid stale data.

    Triggered on every User save (not just creation).  Re-saves the profile
    so any cached computed properties are refreshed.  Silently skips if the
    profile doesn't exist yet (e.g. during the initial creation signal
    before the profile has been committed).

    Args:
        sender: The User model class.
        instance: The User instance that was just saved.
        **kwargs: Includes ``created`` (bool) and other signal arguments.
    """
    if not kwargs.get('created') and hasattr(instance, 'profile'):
        try:
            instance.profile.save()
        except Exception:
            # Profile may not exist yet during the created signal; ignore.
            pass
