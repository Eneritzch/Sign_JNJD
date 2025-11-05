from django.urls import path
from . import views

app_name = 'challenge'

urlpatterns = [
    path('', views.challenge_view, name='index'),
    path('api/start/', views.start_challenge_api, name='start_api'),
    path('api/submit/', views.submit_challenge_api, name='submit_api'),
]