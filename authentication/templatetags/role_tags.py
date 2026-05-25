"""
Template helpers for role checks in Afya DataHub.

Usage in templates:
    {% load role_tags %}

    {% if user|has_role:"Client Admin,Facilities Admin" %}
        ...visible to Client Admin and Facilities Admin only...
    {% endif %}

    {% if user|is_client_admin %} ... {% endif %}
    {% if user|is_facilities_admin %} ... {% endif %}
    {% if user|is_facility_admin %} ... {% endif %}

    {% user_role_badge %}
"""

from django import template
from django.utils.safestring import mark_safe

from authentication.roles import (
    in_role,
    is_client_admin,
    is_facilities_admin,
    is_facility_admin,
)

register = template.Library()


@register.filter(name='has_role')
def has_role(user, roles_str):
    """
    Return True if the user has any of the comma-separated roles.

    Usage: {% if user|has_role:"Client Admin,Facilities Admin" %}
    """
    if not roles_str:
        return False
    role_list = [r.strip() for r in roles_str.split(',') if r.strip()]
    return in_role(user, *role_list)


@register.filter(name='is_client_admin')
def _is_client_admin(user):
    """Return True if the user is a Client Admin or superuser."""
    return is_client_admin(user)


@register.filter(name='is_facilities_admin')
def _is_facilities_admin(user):
    """Return True if the user is Facilities Admin or higher."""
    return is_facilities_admin(user)


@register.filter(name='is_facility_admin')
def _is_facility_admin(user):
    """Return True if the user has any admin role."""
    return is_facility_admin(user)


@register.simple_tag(takes_context=True)
def user_role_badge(context):
    """
    Returns Bootstrap badge HTML for the current user's role.

    Usage: {% user_role_badge %}
    """
    user = context.get('user')
    if not user or not user.is_authenticated:
        return ''
    try:
        profile = user.profile
        colour = profile.role_display_badge
        label = profile.get_role_display()
        return mark_safe(
            f'<span class="badge bg-{colour}" style="font-size:11px;">{label}</span>'
        )
    except Exception:
        return ''


@register.filter(name='role_badge_color')
def role_badge_color(user):
    """Return the Bootstrap colour class string for the user's role."""
    try:
        return user.profile.role_display_badge
    except Exception:
        return 'secondary'
