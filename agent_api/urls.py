from django.urls import path
from .views import AgentEndpoint

urlpatterns = [
    path('chat/', AgentEndpoint.as_view(), name='chat'),
]
