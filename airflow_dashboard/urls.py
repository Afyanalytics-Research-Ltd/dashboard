"""
Afya DataHub - Root URL Configuration
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
    SpectacularRedocView,
)

urlpatterns = [
    # Landing page (unauthenticated users)
    path('', TemplateView.as_view(template_name='landing.html'), name='landing'),

    # Django admin
    path('admin/', admin.site.urls),

    # Application routes (HTML views)
    path('auth/', include('authentication.urls', namespace='authentication')),
    path('analytics/', include('analytics_app.urls', namespace='analytics')),
    path('warehouse/', include('warehouse.urls', namespace='warehouse')),
    path('pipelines/', include('airflow_ui.urls', namespace='airflow')),
    path('core/', include('core.urls', namespace='core')),
    path('settings/agents/', include('agents.urls', namespace='agents')),
    path('chat/', include('analytics_app.dashboards.ksh.facility_utilization.chat.urls')),
    path('forecast/', include('forecasting.urls')),
    path('analytics/chat/', include('self_service.urls', namespace='self_service')),

    # REST API v1
    path('api/v1/', include([
        path('auth/', include('authentication.api_urls')),
        path('analytics/', include('analytics_app.api_urls')),
        path('warehouse/', include('warehouse.api_urls')),
        path('pipelines/', include('airflow_ui.api_urls')),
        path('core/', include('core.api_urls')),
        path('agents/', include('agents.api_urls')),
        # OpenAPI schema + docs
        path('schema/', SpectacularAPIView.as_view(), name='api-schema'),
        path('docs/', SpectacularSwaggerView.as_view(url_name='api-schema'), name='api-docs'),
        path('redoc/', SpectacularRedocView.as_view(url_name='api-schema'), name='api-redoc'),
    ])),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Custom error handlers
handler403 = 'core.views.error_403'
handler404 = 'core.views.error_404'
handler500 = 'core.views.error_500'

# Admin site branding
admin.site.site_header = 'Afya DataHub Administration'
admin.site.site_title = 'Afya DataHub Admin'
admin.site.index_title = 'Platform Administration'
