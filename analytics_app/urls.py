"""
Analytics app HTML URL patterns.
"""

from django.urls import path

from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('dashboards/', views.DashboardListView.as_view(), name='dashboard_list'),
    path('dashboards/sync/', views.DashboardSyncView.as_view(), name='dashboard_sync'),
    path('dashboards/create/', views.DashboardCreateView.as_view(), name='dashboard_create'),
    path('dashboards/<slug:slug>/', views.DashboardDetailView.as_view(), name='dashboard_view'),
    path('dashboards/<slug:slug>/edit/', views.DashboardUpdateView.as_view(), name='dashboard_edit'),
    path('dashboards/<slug:slug>/delete/', views.DashboardDeleteView.as_view(), name='dashboard_delete'),
    path('reporting-queries/', views.ReportingQueryListView.as_view(), name='reporting_query_list'),
    path('reporting-queries/add/', views.ReportingQueryCreateView.as_view(), name='reporting_query_create'),
    path('reporting-queries/sync/', views.ReportingQuerySyncView.as_view(), name='reporting_query_sync'),
    path('reporting-queries/publish-all/', views.ReportingQueryPublishAllView.as_view(), name='reporting_query_publish_all'),
    path('redash-dashboards/create/', views.RedashDashboardStep1View.as_view(), name='redash_dashboard_create'),
    path('redash-dashboards/refresh-queries/', views.RedashDashboardRefreshQueriesView.as_view(), name='redash_dashboard_refresh_queries'),
    path('redash-dashboards/finalize/', views.RedashDashboardFinalizeView.as_view(), name='redash_dashboard_finalize'),
]
