"""
Migration 0006: Rebuild Dashboard model with FK-based client/facility fields,
category, thumbnail, is_public, order, created_by, and description.

This replaces the old CharField-based client with a proper FK to core.Client.
"""

import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


def _deduplicate_slugs(apps, schema_editor):
    """Ensure all dashboard slugs are unique before adding unique constraint."""
    Dashboard = apps.get_model('analytics_app', 'Dashboard')
    seen = {}
    for dashboard in Dashboard.objects.order_by('id'):
        base = dashboard.slug or f'dashboard-{dashboard.pk}'
        slug = base
        counter = 1
        while slug in seen:
            slug = f'{base}-{counter}'
            counter += 1
        seen[slug] = True
        if slug != dashboard.slug:
            dashboard.slug = slug
            dashboard.save(update_fields=['slug'])


class Migration(migrations.Migration):

    dependencies = [
        ('analytics_app', '0005_dashboard_created_at_dashboard_updated_at'),
        ('core', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # 1. Add new nullable FK columns alongside old char column
        migrations.AddField(
            model_name='dashboard',
            name='client_fk',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='dashboards',
                to='core.client',
            ),
        ),
        migrations.AddField(
            model_name='dashboard',
            name='facility',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='dashboards',
                to='core.facility',
            ),
        ),
        migrations.AddField(
            model_name='dashboard',
            name='description',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='dashboard',
            name='category',
            field=models.CharField(
                choices=[
                    ('clinical', 'Clinical'),
                    ('financial', 'Financial'),
                    ('operational', 'Operational'),
                    ('analytics', 'Analytics'),
                    ('reporting', 'Reporting'),
                ],
                default='analytics',
                max_length=50,
            ),
        ),
        migrations.AddField(
            model_name='dashboard',
            name='thumbnail',
            field=models.ImageField(blank=True, null=True, upload_to='dashboards/thumbnails/'),
        ),
        migrations.AddField(
            model_name='dashboard',
            name='is_public',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='dashboard',
            name='view_count',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='dashboard',
            name='order',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='dashboard',
            name='created_by',
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='created_dashboards',
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        # 2. Widen name field
        migrations.AlterField(
            model_name='dashboard',
            name='name',
            field=models.CharField(max_length=500),
        ),
        # 3a. De-duplicate slugs before enforcing uniqueness
        migrations.RunPython(_deduplicate_slugs, migrations.RunPython.noop),
        # 3b. Make slug globally unique
        migrations.AlterField(
            model_name='dashboard',
            name='slug',
            field=models.SlugField(max_length=200, unique=True),
        ),
        # 4. Make streamlit_url optional
        migrations.AlterField(
            model_name='dashboard',
            name='streamlit_url',
            field=models.URLField(blank=True),
        ),
        # 5. Remove old char client field
        migrations.RemoveField(
            model_name='dashboard',
            name='client',
        ),
        # 6. Rename client_fk → client
        migrations.RenameField(
            model_name='dashboard',
            old_name='client_fk',
            new_name='client',
        ),
        # 7. Add indexes
        migrations.AddIndex(
            model_name='dashboard',
            index=models.Index(fields=['client', 'is_active'], name='dashboard_client_active_idx'),
        ),
        migrations.AddIndex(
            model_name='dashboard',
            index=models.Index(fields=['slug'], name='dashboard_slug_idx'),
        ),
        # 8. Update ordering
        migrations.AlterModelOptions(
            name='dashboard',
            options={'ordering': ['order', 'name']},
        ),
    ]
