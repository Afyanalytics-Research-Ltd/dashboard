"""
Template helpers for role checks.

Usage in templates:
    {% load role_tags %}

    {% if user|has_role:"Client Admin,Facilities Admin" %}
        ...visible to Client Admin and Facilities Admin only...
    {% endif %}

    {% if user|is_client_admin %} ... {% endif %}
"""

from django import template

from authentication.roles import (
    in_role,
    is_client_admin,
    is_facilities_admin,
    is_facility_admin,
)

register = template.Library()


@register.filter(name="has_role")
def has_role(user, roles):
    if not roles:
        return False
    role_list = [r.strip() for r in roles.split(",") if r.strip()]
    return in_role(user, *role_list)


@register.filter(name="is_client_admin")
def _is_client_admin(user):
    return is_client_admin(user)


@register.filter(name="is_facilities_admin")
def _is_facilities_admin(user):
    return is_facilities_admin(user)


@register.filter(name="is_facility_admin")
def _is_facility_admin(user):
    return is_facility_admin(user)