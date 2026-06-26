from django.urls import path

from .views import AccessContextView, ChatHistoryView

app_name = 'self_service'

urlpatterns = [
    path('history/', ChatHistoryView.as_view(), name='history'),
    path('access/', AccessContextView.as_view(), name='access'),
]
