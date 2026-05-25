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
]
