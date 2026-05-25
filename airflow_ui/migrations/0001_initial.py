"""
Initial migration for airflow_ui — creates DAGSummary.
"""

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='DAGSummary',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('dag_id', models.CharField(max_length=200, unique=True)),
                ('description', models.TextField(blank=True)),
                ('is_active', models.BooleanField(default=True)),
                ('is_paused', models.BooleanField(default=False)),
                ('last_run_state', models.CharField(blank=True, max_length=50)),
                ('last_run_at', models.DateTimeField(blank=True, null=True)),
                ('total_runs', models.PositiveIntegerField(default=0)),
                ('successful_runs', models.PositiveIntegerField(default=0)),
                ('failed_runs', models.PositiveIntegerField(default=0)),
                ('synced_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'DAG Summary',
                'verbose_name_plural': 'DAG Summaries',
                'ordering': ['dag_id'],
            },
        ),
    ]
