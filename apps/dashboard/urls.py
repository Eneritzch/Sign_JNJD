from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_view, name='index'),
    path('api/metrics/', views.metrics_api, name='metrics_api'),
]