"""
Analytics app models.
"""

from django.db import models
from django.conf import settings
from django.utils.text import slugify


class Dashboard(models.Model):
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

    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('analytics:dashboard_view', kwargs={'slug': self.slug})

    def increment_view_count(self):
        Dashboard.objects.filter(pk=self.pk).update(view_count=models.F('view_count') + 1)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)[:200]
        super().save(*args, **kwargs)
