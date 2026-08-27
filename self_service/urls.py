from django.urls import path

from .views import AccessContextView, ChatHistoryView, ChatSessionListView

app_name = 'self_service'

urlpatterns = [
    path('history/', ChatHistoryView.as_view(), name='history'),
    path('sessions/', ChatSessionListView.as_view(), name='sessions'),
    path('access/', AccessContextView.as_view(), name='access'),
]
