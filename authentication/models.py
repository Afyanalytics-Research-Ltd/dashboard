"""
User profile model.

We don't subclass AbstractUser because the project is already running and
auth.User is already migrated. Instead we hang an extensible UserProfile off
the existing User via OneToOneField — adding new fields is then a normal
makemigrations + migrate cycle, with no risk to existing user rows.

To add a new field: edit the class below, then run
    python manage.py makemigrations accounts
    python manage.py migrate accounts
"""

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


class UserProfile(models.Model):
    """
    Extra fields attached to every user. One row per User; created
    automatically by the post_save signal below.

    Add new fields directly to this class and run makemigrations.
    Examples are listed (commented) at the bottom — uncomment as needed.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )

    # --- Contact -------------------------------------------------------------
    phone_number = models.CharField(
        max_length=32,
        blank=True,
        help_text="Primary contact number, including country code.",
    )

    # --- Tenancy / org context ----------------------------------------------
    # These are typed as free-text so you don't have to introduce Client /
    # Facility models right now. Replace with FK fields once those exist.
    client = models.CharField(
        max_length=120,
        blank=True,
        help_text="Client / organization the user belongs to.",
    )
    facility = models.CharField(
        max_length=120,
        blank=True,
        help_text="Facility the user is primarily assigned to. "
                  "Leave blank for Facilities Admin / Client Admin.",
    )

    # --- Misc profile fields ------------------------------------------------
    job_title = models.CharField(max_length=120, blank=True)
    avatar = models.ImageField(upload_to="avatars/", blank=True, null=True)

    # --- Audit --------------------------------------------------------------
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # --- Add more fields here as needed -------------------------------------
    # Examples (uncomment and run makemigrations):
    #
    # date_of_birth   = models.DateField(blank=True, null=True)
    # timezone        = models.CharField(max_length=64, blank=True, default="Africa/Nairobi")
    # language        = models.CharField(max_length=8, blank=True, default="en")
    # whatsapp_opt_in = models.BooleanField(default=False)
    # last_login_ip   = models.GenericIPAddressField(blank=True, null=True)
    # notes           = models.TextField(blank=True)

    class Meta:
        verbose_name = "User profile"
        verbose_name_plural = "User profiles"

    def __str__(self):
        return f"{self.user.username} profile"

    # --- Role helpers (wrap accounts.roles for convenience) -----------------

    @property
    def roles(self):
        """Set of role names this user belongs to."""
        return user_roles(self.user)

    @property
    def primary_role(self):
        """
        The single most-privileged role for display purposes
        (Client Admin > Facilities Admin > Facility Admin).
        Returns None if the user has no role group.
        """
        for role in (ROLE_CLIENT_ADMIN, ROLE_FACILITIES_ADMIN, ROLE_FACILITY_ADMIN):
            if role in self.roles:
                return role
        return None

    def has_role(self, *roles):
        return in_role(self.user, *roles)


# --- Auto-create a profile whenever a User is created ------------------------


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def _ensure_profile(sender, instance, created, **kwargs):
    """Every User gets a UserProfile, even ones created via createsuperuser."""
    if created:
        UserProfile.objects.get_or_create(user=instance)