"""
Surface UserProfile inside the standard auth.User admin so role assignment
(Groups) and profile fields (phone, facility, …) live on the same page.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from django.contrib.auth.models import User

from authentication.models import UserProfile


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    fk_name = "user"
    fields = (
        "phone_number",
        "client",
        "facility",
        "job_title",
        "avatar",
        "created_at",
        "updated_at",
    )
    readonly_fields = ("created_at", "updated_at")
    extra = 0


class UserAdmin(DjangoUserAdmin):
    inlines = (UserProfileInline,)
    list_display = (
        "username",
        "email",
        "first_name",
        "last_name",
        "is_staff",
        "_roles",
        "_phone",
    )

    def _roles(self, obj):
        return ", ".join(sorted(obj.groups.values_list("name", flat=True))) or "—"

    _roles.short_description = "Roles"

    def _phone(self, obj):
        return getattr(getattr(obj, "profile", None), "phone_number", "") or "—"

    _phone.short_description = "Phone"


# Re-register User with our extended admin.
admin.site.unregister(User)
admin.site.register(User, UserAdmin)