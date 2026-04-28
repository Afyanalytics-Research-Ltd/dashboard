# Create your models here.
from django.db import models


class TrackedSpreadsheet(models.Model):
    """Local record of a spreadsheet the app has created or interacted with.

    The actual data lives on Google. This is just a convenience index so the
    web UI can list past sheets without having to remember the IDs.
    """

    spreadsheet_id = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=512, blank=True)
    web_view_link = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"{self.title or self.spreadsheet_id}"