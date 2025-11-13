from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.analytics, name='analytics'),
    path('api/save-stat/', views.save_recognition_stat, name='save_stat'),
    path('api/usage-stats/', views.get_usage_stats, name='get_usage_stats'),
]