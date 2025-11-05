from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    path('', views.recognition_view, name='index'),
    path('api/predict/', views.predict_api, name='predict_api'),
]