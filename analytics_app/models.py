# models.py
from django.db import models

class Dashboard(models.Model):
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=False)
    client = models.CharField(max_length=200, null=True, blank=True)
    # THIS is the key part
    streamlit_url = models.URLField()
    slug = models.SlugField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_url(self):
        return f"/analytics/{self.slug}"