import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('authentication', '0002_alter_userprofile_options_userprofile_bio_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserModuleGrant',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('module_key', models.CharField(choices=[('warehouse', 'Warehouse (Snowflake SQL & schema browser)'), ('analytics', 'Analytics Dashboards'), ('self_service', 'AI Chatbot / Self-Service Query')], max_length=30)),
                ('is_granted', models.BooleanField(default=True, help_text='True = explicitly grant this module; False = explicitly revoke it.')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('granted_by', models.ForeignKey(blank=True, help_text='The administrator who set this override.', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='module_grants_issued', to=settings.AUTH_USER_MODEL)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='module_grants', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'User Module Grant',
                'verbose_name_plural': 'User Module Grants',
                'ordering': ['user__username', 'module_key'],
            },
        ),
        migrations.AddConstraint(
            model_name='usermodulegrant',
            constraint=models.UniqueConstraint(fields=('user', 'module_key'), name='unique_user_module_grant'),
        ),
    ]
