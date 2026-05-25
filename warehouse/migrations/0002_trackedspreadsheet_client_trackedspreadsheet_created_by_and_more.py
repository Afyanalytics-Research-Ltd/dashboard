# Generated migration: add client/created_by FK to TrackedSpreadsheet;
# add SnowflakeQueryLog model.

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0001_initial"),
        ("warehouse", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # ── TrackedSpreadsheet: add client + created_by ─────────────────────
        migrations.AddField(
            model_name="trackedspreadsheet",
            name="client",
            field=models.ForeignKey(
                blank=True,
                help_text="Client this spreadsheet belongs to (optional).",
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="spreadsheets",
                to="core.client",
            ),
        ),
        migrations.AddField(
            model_name="trackedspreadsheet",
            name="created_by",
            field=models.ForeignKey(
                help_text="User who first tracked this spreadsheet.",
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="spreadsheets",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterModelOptions(
            name="trackedspreadsheet",
            options={
                "ordering": ["-updated_at"],
                "verbose_name": "Tracked Spreadsheet",
                "verbose_name_plural": "Tracked Spreadsheets",
            },
        ),

        # ── SnowflakeQueryLog ────────────────────────────────────────────────
        migrations.CreateModel(
            name="SnowflakeQueryLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("query", models.TextField()),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("success", "Success"),
                            ("error", "Error"),
                            ("pending", "Pending"),
                        ],
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("rows_returned", models.PositiveIntegerField(default=0)),
                ("execution_time_ms", models.PositiveIntegerField(default=0)),
                ("error_message", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="snowflake_queries",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Snowflake Query Log",
                "verbose_name_plural": "Snowflake Query Logs",
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="snowflakequerylog",
            index=models.Index(
                fields=["user", "created_at"],
                name="wh_sqlog_user_created_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="snowflakequerylog",
            index=models.Index(
                fields=["status", "created_at"],
                name="wh_sqlog_status_created_idx",
            ),
        ),
    ]
