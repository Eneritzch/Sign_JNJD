from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    # Página principal de la demo
    path('demo/', views.live_demo, name='live_demo'),
    
    # API para señas estáticas
    path('api/predict-static/', views.predict_static_realtime, name='predict_static'),

    # Dashboard de análisis
    path('report/', views.realtime_report, name='report'),
    
    # Endpoint de salud del sistema
    path('health/', views.system_health, name='system_health'),
    
    # Endpoint de debug de umbrales
    path('debug/thresholds/', views.debug_thresholds, name='debug_thresholds'),
    
    # API para señas dinámicas 
    #path('api/predict-dynamic/', views.predict_dynamic, name='predict_dynamic'),
]