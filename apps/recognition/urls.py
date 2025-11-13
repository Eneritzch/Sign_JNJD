from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    # Página principal de la demo
    path('demo/', views.live_demo, name='live_demo'),
    
    path('recognize/', views.recognize_from_photo, name='recognize_photo'),
    path('stats/', views.get_saved_photos_stats, name='photo_stats'), 
]