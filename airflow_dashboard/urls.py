
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    path("", TemplateView.as_view(template_name="home.html"), name="home"),
    path('datapipelines/', include('airflow_ui.urls')),
    path('dashboards/', include('analytics_app.urls')),
    path('warehouse/', include('warehouse.urls')),
    path('auth/', include('authentication.urls')),
    path('admin/', admin.site.urls),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)