from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('analytics_app', '0011_reportingquery_facility'),
    ]

    operations = [
        migrations.AddField(
            model_name='dashboard',
            name='hidden_from_users',
            field=models.ManyToManyField(blank=True, help_text='Users explicitly denied this dashboard even though it would otherwise be visible to their client — set by a facility administrator via the Permissions page.', related_name='hidden_dashboards', to=settings.AUTH_USER_MODEL),
        ),
    ]
