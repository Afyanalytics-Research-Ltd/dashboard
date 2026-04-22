# models.py
from django.db import models
class Dashboard(models.Model):
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=False)
    # THIS is the key part
    streamlit_url = models.URLField()
    slug = models.SlugField()

    def get_url(self):
        return f"/analytics/{self.slug}"