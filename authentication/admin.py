"""
Django admin registrations for the authentication app.

UserProfile is exposed both:
  - As a standalone ModelAdmin (/admin/authentication/userprofile/)
  - As a StackedInline on the User change page
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from django.contrib.auth.models import User
from django.utils.html import format_html

from authentication.models import UserProfile


# ---------------------------------------------------------------------------
# UserProfile inline (shown on User admin page)
# ---------------------------------------------------------------------------

class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    fk_name = 'user'
    verbose_name = 'Profile'
    verbose_name_plural = 'Profile'
    fields = (
        'role',
        'phone_number',
        'job_title',
        'bio',
        'client',
        'facility',
        'avatar',
        'is_verified',
        'last_login_ip',
        'created_at',
        'updated_at',
    )
    readonly_fields = ('last_login_ip', 'created_at', 'updated_at')
    extra = 0


# ---------------------------------------------------------------------------
# Extended UserAdmin
# ---------------------------------------------------------------------------

class UserAdmin(DjangoUserAdmin):
    inlines = (UserProfileInline,)

    list_display = (
        'username',
        'email',
        'first_name',
        'last_name',
        'is_staff',
        '_role',
        '_client',
        '_phone',
        '_verified',
    )

    @admin.display(description='Role', ordering='profile__role')
    def _role(self, obj):
        try:
            role = obj.profile.get_role_display()
            colour = obj.profile.role_display_badge
            return format_html(
                '<span class="badge" style="background:#6c757d;padding:3px 8px;'
                'border-radius:12px;font-size:11px;">{}</span>',
                role,
            )
        except Exception:
            return '—'

    @admin.display(description='Client')
    def _client(self, obj):
        try:
            return obj.profile.client or '—'
        except Exception:
            return '—'

    @admin.display(description='Phone')
    def _phone(self, obj):
        try:
            return obj.profile.phone_number or '—'
        except Exception:
            return '—'

    @admin.display(description='Verified', boolean=True)
    def _verified(self, obj):
        try:
            return obj.profile.is_verified
        except Exception:
            return False


# Re-register User with our extended admin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)


# ---------------------------------------------------------------------------
# Standalone UserProfile admin
# ---------------------------------------------------------------------------

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = (
        'user',
        'role',
        'client',
        'facility',
        'phone_number',
        'is_verified',
        'created_at',
    )
    list_filter = ('role', 'is_verified', 'client', 'created_at')
    search_fields = (
        'user__username',
        'user__email',
        'user__first_name',
        'user__last_name',
        'phone_number',
    )
    readonly_fields = ('created_at', 'updated_at', 'last_login_ip')
    raw_id_fields = ('user', 'client', 'facility')
    ordering = ('-created_at',)

    fieldsets = (
        ('Identity', {
            'fields': ('user', 'role', 'is_verified'),
        }),
        ('Contact', {
            'fields': ('phone_number', 'job_title', 'bio', 'avatar'),
        }),
        ('Organisation', {
            'fields': ('client', 'facility'),
        }),
        ('Security & Audit', {
            'fields': ('last_login_ip', 'created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )
