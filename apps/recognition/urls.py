from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    # Página principal de la demo
    path('demo/', views.live_demo, name='live_demo'),
    
    # API para señas estáticas (una sola imagen)
    path('api/predict-static/', views.predict_static, name='predict_static'),
    
    # API para señas dinámicas (secuencia de 20 frames)
    path('api/predict-dynamic/', views.predict_dynamic, name='predict_dynamic'),
]