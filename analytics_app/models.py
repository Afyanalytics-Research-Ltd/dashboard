"""
Analytics app models.
"""

from django.db import models
from django.conf import settings
from django.utils.text import slugify


class Dashboard(models.Model):
    """A single analytics dashboard that users can view in the platform.

    Each dashboard is a named, categorised entry that either embeds a Streamlit
    app via ``streamlit_url`` or serves as a placeholder for future content.
    Dashboards belong to a :class:`core.models.Client` (and optionally a
    :class:`core.models.Facility`) so only users of that client can see them.

    Non-technical explanation:
        Think of a Dashboard like a framed report hanging on the wall of a
        healthcare facility's back-office.  Each frame has a name, a category
        (clinical, financial, etc.), and a picture inside (the Streamlit
        analytics).  Only staff from the right organisation can walk into that
        room and look at the frames.
    """

    CATEGORY_CHOICES = [
        ('clinical', 'Clinical'),
        ('financial', 'Financial'),
        ('operational', 'Operational'),
        ('analytics', 'Analytics'),
        ('reporting', 'Reporting'),
    ]

    name = models.CharField(max_length=500)
    description = models.TextField(blank=True)
    slug = models.SlugField(max_length=200, unique=True)
    client = models.ForeignKey(
        'core.Client',
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='dashboards',
    )
    facility = models.ForeignKey(
        'core.Facility',
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='dashboards',
    )
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default='analytics')
    streamlit_url = models.URLField(blank=True)
    thumbnail = models.ImageField(upload_to='dashboards/thumbnails/', blank=True, null=True)
    is_active = models.BooleanField(default=True)
    is_public = models.BooleanField(default=False)
    view_count = models.PositiveIntegerField(default=0)
    order = models.PositiveIntegerField(default=0)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True, on_delete=models.SET_NULL,
        related_name='created_dashboards',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['order', 'name']
        indexes = [
            models.Index(fields=['client', 'is_active']),
            models.Index(fields=['slug']),
        ]

    def __str__(self):
        return self.name

    def get_absolute_url(self) -> str:
        """Return the canonical URL for viewing this dashboard.

        Used by templates and serializers to build clickable links without
        hard-coding URL patterns.

        Returns:
            A relative URL string, e.g. ``"/analytics/dashboards/ksh-revenue/"``.
        """
        from django.urls import reverse
        return reverse('analytics:dashboard_view', kwargs={'slug': self.slug})

    def increment_view_count(self) -> None:
        """Atomically add 1 to this dashboard's view counter.

        Uses a database-level F-expression so concurrent requests do not
        overwrite each other's counts (avoids the read-modify-write race
        condition that a simple ``self.view_count += 1`` would introduce).

        Non-technical explanation:
            Every time someone opens this dashboard, we tick a counter up by
            one — like an old-fashioned turnstile at a museum entrance.  Using
            the database to do the counting means two people visiting at the
            same moment both get counted correctly.
        """
        Dashboard.objects.filter(pk=self.pk).update(view_count=models.F('view_count') + 1)

    def save(self, *args, **kwargs) -> None:
        """Persist the dashboard, auto-generating a URL-safe slug if none is set.

        The slug is derived from the dashboard name (e.g. "KSH Revenue" →
        ``"ksh-revenue"``).  It is truncated to 200 characters to stay within
        the database column limit.

        Args:
            *args: Passed through to the parent ``save``.
            **kwargs: Passed through to the parent ``save``.
        """
        if not self.slug:
            self.slug = slugify(self.name)[:200]
        super().save(*args, **kwargs)
