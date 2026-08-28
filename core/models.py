"""
Core platform models: Client, Facility, AuditLog, Notification, SystemSettings,
Ticket, TicketComment.
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
        """Return the number of currently active facilities for this client.

        Non-technical explanation:
            Counts how many of this organisation's hospital branches or clinics
            are currently switched on and accepting users — not counting any
            that have been deactivated.

        Returns:
            An integer >= 0.  Returns 0 if the client has no active facilities.
        """
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
    reporting_source_schema = models.CharField(
        max_length=100, blank=True,
        help_text='The exact value this facility\'s rows use in the '
                  'HOSPITALS.REPORTING "source_schema" column (e.g. "Kisumu") — '
                  'not necessarily related to the facility name above. Used to '
                  'scope synced Redash reporting queries to this facility.',
    )
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
        """Create and save a new audit log entry in one call.

        This is the preferred way to record user activity anywhere in the
        codebase — middleware, views, and API endpoints all use it.
        Unauthenticated (anonymous) users are stored with ``user=None``.

        Non-technical explanation:
            Every significant action a user takes — logging in, viewing a
            report, exporting data — gets written into a permanent record
            book (the audit log).  This method is the pen that does the
            writing.  You tell it *who* did *what* to *which thing*, and it
            stamps the entry with a timestamp automatically.

        Args:
            user: The Django User object performing the action.  Pass
                ``None`` or an unauthenticated user to record an anonymous
                action.
            action: One of the ``ACTION_CHOICES`` strings, e.g. ``"login"``,
                ``"create"``, ``"export"``.
            resource: Human-readable name of the thing being acted on, e.g.
                ``"dashboard"``, ``"authentication"``.
            resource_id: Optional identifier of the specific object, e.g.
                the slug ``"ksh-revenue"`` or a primary key ``"42"``.
            detail: Free-text description of what happened, e.g.
                ``"User logged in from 41.80.12.1"``.
            ip_address: The client's IP address (IPv4 or IPv6).
            user_agent: The ``User-Agent`` header string from the request.

        Returns:
            The newly created :class:`AuditLog` instance.
        """
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
        """Mark this notification as read and persist the change.

        Only updates the ``is_read`` field to avoid overwriting other
        concurrent changes to the same record.

        Non-technical explanation:
            Like tapping a notification bubble on your phone so the red
            dot disappears — it records that you've seen the message.
        """
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
        """Create and deliver an in-app notification to a user.

        Non-technical explanation:
            Drops a new message into the user's notification inbox — like
            sending a text message, but it appears inside the platform rather
            than on their phone.

        Args:
            user: The Django User who should receive the notification.
            title: Short heading shown in the notification list, e.g.
                ``"Dashboard sync complete"``.
            message: Body text with more detail.
            notification_type: Visual severity level — one of ``"info"``
                (blue), ``"success"`` (green), ``"warning"`` (yellow),
                ``"danger"`` (red).  Defaults to ``"info"``.
            link: Optional URL the user can click to navigate to related
                content, e.g. ``"/analytics/dashboards/ksh-revenue/"``.

        Returns:
            The newly created :class:`Notification` instance.
        """
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
        """Retrieve a platform setting value by its key name.

        Non-technical explanation:
            Works like looking up a word in a dictionary — you give the key
            (e.g. ``"max_export_rows"``), and you get back whatever value
            was stored for it.  If the key doesn't exist you get the
            ``default`` instead (so the app keeps running safely).

        Args:
            key: The setting name to look up, e.g. ``"maintenance_mode"``.
            default: Value to return when the key is not found.  Defaults
                to ``None``.

        Returns:
            The stored JSON value (could be a string, number, list, or
            dict) or ``default`` if the key is absent.
        """
        try:
            return cls.objects.get(key=key).value
        except cls.DoesNotExist:
            return default

    @classmethod
    def set(cls, key: str, value, user=None, description: str = '') -> 'SystemSettings':
        """Create or update a platform-wide setting.

        Uses ``update_or_create`` so it is safe to call repeatedly — if the
        key already exists it is updated; otherwise a new row is inserted.

        Non-technical explanation:
            Like writing an entry in a shared settings notebook.  If the
            page for that setting already exists, you update it; if not,
            you add a new page.  The notebook remembers who last changed
            each setting and when.

        Args:
            key: The unique name for this setting, e.g.
                ``"max_export_rows"``.  Will be normalised to lowercase
                with underscores by the serializer.
            value: Any JSON-serialisable value (string, number, list, dict).
            user: The Django User making the change (stored for auditing).
            description: Human-readable explanation of what this setting
                controls, stored alongside the value.

        Returns:
            The created-or-updated :class:`SystemSettings` instance.
        """
        obj, _ = cls.objects.update_or_create(
            key=key,
            defaults={
                'value': value,
                'updated_by': user,
                'description': description or '',
            },
        )
        return obj


class Ticket(models.Model):
    """A support ticket — an issue/error/complaint, a feature-improvement
    suggestion, or a brand-new feature request — raised by any authenticated
    user from anywhere in the platform.

    Non-technical explanation:
        A digital comment card. Whenever something's broken, could be
        better, or is completely missing, a user drops a card in one of
        three boxes (Issue, Suggestion, New Feature). The support team then
        works through the cards on the Support & Ticketing page, moving
        each one from Open through to Resolved.
    """

    TYPE_ISSUE = 'issue'
    TYPE_SUGGESTION = 'suggestion'
    TYPE_FEATURE = 'feature'
    TYPE_CHOICES = [
        (TYPE_ISSUE, 'Issue / Error / Complaint'),
        (TYPE_SUGGESTION, 'Feature Improvement Suggestion'),
        (TYPE_FEATURE, 'New Feature Request'),
    ]
    TYPE_ICONS = {
        TYPE_ISSUE: 'bi-exclamation-octagon-fill',
        TYPE_SUGGESTION: 'bi-lightbulb-fill',
        TYPE_FEATURE: 'bi-stars',
    }

    STATUS_OPEN = 'open'
    STATUS_IN_PROGRESS = 'in_progress'
    STATUS_RESOLVED = 'resolved'
    STATUS_CLOSED = 'closed'
    STATUS_CHOICES = [
        (STATUS_OPEN, 'Open'),
        (STATUS_IN_PROGRESS, 'In Progress'),
        (STATUS_RESOLVED, 'Resolved'),
        (STATUS_CLOSED, 'Closed'),
    ]
    STATUS_COLORS = {
        STATUS_OPEN: 'amber',
        STATUS_IN_PROGRESS: 'blue',
        STATUS_RESOLVED: 'teal',
        STATUS_CLOSED: 'cool',
    }

    PRIORITY_LOW = 'low'
    PRIORITY_MEDIUM = 'medium'
    PRIORITY_HIGH = 'high'
    PRIORITY_CRITICAL = 'critical'
    PRIORITY_CHOICES = [
        (PRIORITY_LOW, 'Low'),
        (PRIORITY_MEDIUM, 'Medium'),
        (PRIORITY_HIGH, 'High'),
        (PRIORITY_CRITICAL, 'Critical'),
    ]

    ticket_type = models.CharField(max_length=20, choices=TYPE_CHOICES, db_index=True)
    subject = models.CharField(max_length=200)
    description = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_OPEN, db_index=True)
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default=PRIORITY_MEDIUM)

    page_url = models.CharField(
        max_length=500, blank=True,
        help_text='The page the user was on when they submitted this (auto-captured).',
    )
    attachment = models.ImageField(upload_to='tickets/attachments/%Y/%m/', blank=True, null=True)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True,
        related_name='tickets_created',
    )
    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='tickets_assigned',
    )
    client = models.ForeignKey(
        Client, on_delete=models.SET_NULL, null=True, blank=True, related_name='tickets',
    )
    facility = models.ForeignKey(
        Facility, on_delete=models.SET_NULL, null=True, blank=True, related_name='tickets',
    )

    resolution_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Ticket'
        verbose_name_plural = 'Tickets'
        indexes = [
            models.Index(fields=['status', 'ticket_type']),
            models.Index(fields=['created_by', 'status']),
        ]

    def __str__(self) -> str:
        return f'[{self.get_ticket_type_display()}] {self.subject}'

    @property
    def status_color(self) -> str:
        """Return the Afya design-token colour family for this ticket's status."""
        return self.STATUS_COLORS.get(self.status, 'cool')

    @property
    def type_icon(self) -> str:
        """Return the Bootstrap Icon class for this ticket's type."""
        return self.TYPE_ICONS.get(self.ticket_type, 'bi-ticket-fill')

    def set_status(self, new_status: str, *, actor=None) -> None:
        """Transition this ticket to ``new_status`` and persist it.

        Stamps ``resolved_at`` the first time a ticket reaches
        ``STATUS_RESOLVED``, and notifies the ticket's creator of the
        status change (skipped if the actor making the change is the
        creator themselves, to avoid notifying someone about their own
        action).

        Non-technical explanation:
            Moves a ticket to a new column on the support board — e.g.
            from "Open" to "In Progress" — and, if it just moved to
            "Resolved" for the first time, stamps the moment it was fixed
            and lets the person who reported it know.

        Args:
            new_status: One of ``STATUS_CHOICES``.
            actor: The user making the change (used only to decide whether
                to notify the creator — pass ``None`` to always notify).
        """
        if new_status not in dict(self.STATUS_CHOICES):
            raise ValueError(f'Unknown ticket status: {new_status!r}')

        self.status = new_status
        update_fields = ['status', 'updated_at']
        if new_status == self.STATUS_RESOLVED and not self.resolved_at:
            from django.utils import timezone
            self.resolved_at = timezone.now()
            update_fields.append('resolved_at')
        self.save(update_fields=update_fields)

        if self.created_by and (actor is None or actor.pk != self.created_by.pk):
            Notification.send(
                self.created_by,
                title=f'Ticket updated: {self.subject}',
                message=f'Your ticket is now "{self.get_status_display()}".',
                notification_type='success' if new_status == self.STATUS_RESOLVED else 'info',
                link='/core/support/',
            )


class TicketComment(models.Model):
    """A single reply on a :class:`Ticket`'s thread.

    Non-technical explanation:
        One message in the back-and-forth conversation attached to a
        support ticket — like a comment thread, but scoped to that one
        issue/suggestion/feature request.
    """

    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='ticket_comments',
    )
    body = models.TextField()
    is_internal = models.BooleanField(
        default=False,
        help_text='Internal note visible to staff only, hidden from the ticket creator.',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        verbose_name = 'Ticket Comment'
        verbose_name_plural = 'Ticket Comments'

    def __str__(self) -> str:
        author_name = self.author.username if self.author else 'unknown'
        return f'{author_name} on #{self.ticket_id}: {self.body[:40]}'
