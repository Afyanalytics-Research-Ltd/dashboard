"""
Core platform models: Client, Facility, AuditLog, Notification, SystemSettings.
"""

from django.conf import settings
from django.db import models


class Client(models.Model):
    """A healthcare client / organisation using the platform."""

    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    logo = models.ImageField(
        upload_to='clients/logos/',
        null=True,
        blank=True,
        help_text='Client logo (PNG/JPG recommended).',
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = 'Client'
        verbose_name_plural = 'Clients'

    def __str__(self) -> str:
        return self.name

    @property
    def active_facilities_count(self) -> int:
        return self.facilities.filter(is_active=True).count()


class Facility(models.Model):
    """A physical or virtual facility belonging to a client."""

    client = models.ForeignKey(
        Client,
        on_delete=models.CASCADE,
        related_name='facilities',
    )
    name = models.CharField(max_length=200)
    slug = models.SlugField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['client', 'slug']]
        ordering = ['name']
        verbose_name = 'Facility'
        verbose_name_plural = 'Facilities'

    def __str__(self) -> str:
        return f'{self.client.name} — {self.name}'


class AuditLog(models.Model):
    """Immutable record of every significant user action."""

    ACTION_CHOICES = [
        ('create', 'Create'),
        ('read', 'Read'),
        ('update', 'Update'),
        ('delete', 'Delete'),
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('export', 'Export'),
        ('share', 'Share'),
        ('query', 'Query'),
        ('trigger', 'Trigger'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        on_delete=models.SET_NULL,
        related_name='audit_logs',
    )
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    resource = models.CharField(max_length=200)
    resource_id = models.CharField(max_length=100, blank=True)
    detail = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Audit Log'
        verbose_name_plural = 'Audit Logs'
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['action', 'timestamp']),
            models.Index(fields=['resource']),
        ]

    def __str__(self) -> str:
        username = self.user.username if self.user else 'anonymous'
        return f'{username} | {self.action} | {self.resource} | {self.timestamp:%Y-%m-%d %H:%M}'

    @classmethod
    def log(
        cls,
        user,
        action: str,
        resource: str,
        resource_id: str = '',
        detail: str = '',
        ip_address: str | None = None,
        user_agent: str = '',
    ) -> 'AuditLog':
        """Convenience factory method to create an audit log entry."""
        return cls.objects.create(
            user=user if (user and user.is_authenticated) else None,
            action=action,
            resource=resource,
            resource_id=str(resource_id),
            detail=detail,
            ip_address=ip_address,
            user_agent=user_agent,
        )


class Notification(models.Model):
    """In-app notification for a specific user."""

    TYPE_CHOICES = [
        ('info', 'Info'),
        ('success', 'Success'),
        ('warning', 'Warning'),
        ('danger', 'Danger'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='notifications',
    )
    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(
        max_length=20,
        choices=TYPE_CHOICES,
        default='info',
    )
    is_read = models.BooleanField(default=False)
    link = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Notification'
        verbose_name_plural = 'Notifications'

    def __str__(self) -> str:
        return f'{self.user.username} — {self.title}'

    def mark_read(self) -> None:
        self.is_read = True
        self.save(update_fields=['is_read'])

    @classmethod
    def send(
        cls,
        user,
        title: str,
        message: str,
        notification_type: str = 'info',
        link: str = '',
    ) -> 'Notification':
        """Create and return a notification for the given user."""
        return cls.objects.create(
            user=user,
            title=title,
            message=message,
            notification_type=notification_type,
            link=link,
        )


class SystemSettings(models.Model):
    """Key/value store for platform-wide configuration."""

    key = models.CharField(max_length=100, unique=True)
    value = models.JSONField(default=dict)
    description = models.TextField(blank=True)
    is_public = models.BooleanField(
        default=False,
        help_text='Public settings are visible to all authenticated users.',
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        on_delete=models.SET_NULL,
        related_name='system_settings_updates',
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['key']
        verbose_name = 'System Setting'
        verbose_name_plural = 'System Settings'

    def __str__(self) -> str:
        return self.key

    @classmethod
    def get(cls, key: str, default=None):
        """Retrieve a setting value by key."""
        try:
            return cls.objects.get(key=key).value
        except cls.DoesNotExist:
            return default

    @classmethod
    def set(cls, key: str, value, user=None, description: str = '') -> 'SystemSettings':
        """Create or update a setting."""
        obj, _ = cls.objects.update_or_create(
            key=key,
            defaults={
                'value': value,
                'updated_by': user,
                'description': description or '',
            },
        )
        return obj
