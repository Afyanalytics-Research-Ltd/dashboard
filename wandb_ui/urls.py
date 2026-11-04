"""
Model Studio URL patterns (mounted at ``/models/``).
"""

from django.urls import path, re_path

from . import proxy, views

app_name = 'wandb_ui'

app = views.ModelStudioView.as_view()

urlpatterns = [
    path('graphql', proxy.graphql_proxy, name='graphql'),
    path('file', proxy.file_proxy, name='file'),
    path('views/<str:entity>/<str:project>/', proxy.views_collection, name='views'),
    path('views/<int:view_id>/', proxy.view_detail, name='view'),

    path('', app, name='home'),
    re_path(r'^(?!graphql$|file$|views/).+$', app, name='app'),
]
