from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    # Página principal de la demo
    path('demo/', views.static, name='static'),
    
    # Reconocimiento estático 
    path('static/', views.static, name='static'),
    path('recognize-photo/', views.recognize_from_photo, name='recognize_photo'),
    
    # Reconocimiento dinámico 
    path('dynamic/', views.dynamic, name='dynamic'),
    path('recognize-dynamic/', views.recognize_dynamic, name='recognize_dynamic'),
]