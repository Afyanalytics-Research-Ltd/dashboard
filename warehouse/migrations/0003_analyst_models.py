import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models

import warehouse.models


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('warehouse', '0002_trackedspreadsheet_client_trackedspreadsheet_created_by_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Workbook',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('file', models.FileField(upload_to=warehouse.models.workbook_upload_path)),
                ('original_name', models.CharField(max_length=255)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('overview', models.TextField(blank=True)),
                ('load_error', models.TextField(blank=True)),
                ('owner', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='workbooks', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Workbook',
                'verbose_name_plural': 'Workbooks',
                'ordering': ['-uploaded_at'],
            },
        ),
        migrations.CreateModel(
            name='Conversation',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('title', models.CharField(blank=True, max_length=255)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('transcript', models.JSONField(blank=True, default=list)),
                ('owner', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='analyst_conversations', to=settings.AUTH_USER_MODEL)),
                ('workbook', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='conversations', to='warehouse.workbook')),
            ],
            options={
                'verbose_name': 'Analyst Conversation',
                'verbose_name_plural': 'Analyst Conversations',
                'ordering': ['-updated_at'],
            },
        ),
        migrations.CreateModel(
            name='ChatMessage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('role', models.CharField(choices=[('user', 'User'), ('assistant', 'Assistant'), ('error', 'Error')], max_length=16)),
                ('content', models.TextField()),
                ('tool_calls', models.JSONField(blank=True, default=list)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('conversation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='messages', to='warehouse.conversation')),
            ],
            options={
                'verbose_name': 'Analyst Chat Message',
                'verbose_name_plural': 'Analyst Chat Messages',
                'ordering': ['created_at', 'id'],
            },
        ),
        migrations.CreateModel(
            name='Artifact',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('kind', models.CharField(choices=[('chart', 'Chart'), ('table', 'Table'), ('report', 'Report')], max_length=16)),
                ('title', models.CharField(max_length=255)),
                ('file', models.FileField(upload_to='warehouse/analyst/artifacts/%Y/%m/')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('conversation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='artifacts', to='warehouse.conversation')),
                ('message', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='artifacts', to='warehouse.chatmessage')),
            ],
            options={
                'verbose_name': 'Analyst Artifact',
                'verbose_name_plural': 'Analyst Artifacts',
                'ordering': ['created_at', 'id'],
            },
        ),
    ]
