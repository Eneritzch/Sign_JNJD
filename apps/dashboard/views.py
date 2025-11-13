import os
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# VISTA DE ANALÍTICA
# ============================================================================
def analytics(request):
    """
    Renderiza la página de analítica con datos de entrenamiento y uso
    """
    # Rutas a los archivos JSON de entrenamiento
    static_history_path = os.path.join(PROJECT_ROOT, 'models', 'history_mobilenetv2.json')
    dynamic_history_path = os.path.join(PROJECT_ROOT, 'models', 'history_dynamic.json')
    
    # Cargar historiales de entrenamiento
    static_history = None
    dynamic_history = None
    
    try:
        if os.path.exists(static_history_path):
            with open(static_history_path, 'r') as f:
                static_history = json.load(f)
    except Exception as e:
        print(f"Error cargando historial estático: {e}")
    
    try:
        if os.path.exists(dynamic_history_path):
            with open(dynamic_history_path, 'r') as f:
                dynamic_history = json.load(f)
    except Exception as e:
        print(f"Error cargando historial dinámico: {e}")
    
    context = {
        'static_history': json.dumps(static_history) if static_history else 'null',
        'dynamic_history': json.dumps(dynamic_history) if dynamic_history else 'null',
        'static_labels': json.dumps([
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y'
        ]),
        'dynamic_labels': json.dumps([
            "hola","adios","chao","bienvenido","buenos_dias","buenas_tardes","buenas_noches",
            "gracias","por_favor","mucho_gusto","como_estas","feliz","triste","enojado",
            "sorprendido","cansado","malo","bueno","nombre","familia","j","z"
        ])
    }
    
    return render(request, 'dashboard/analytics.html', context)


# ============================================================================
# API: GUARDAR ESTADÍSTICAS DE USO
# ============================================================================
@csrf_exempt
@require_http_methods(["POST"])
def save_recognition_stat(request):
    """
    Guarda estadísticas de reconocimiento en tiempo real
    
    Parámetros:
    - model_type: 'static' o 'dynamic'
    - predicted_class: Clase predicha
    - confidence: Confianza
    - timestamp: Marca de tiempo
    """
    try:
        model_type = request.POST.get('model_type')
        predicted_class = request.POST.get('predicted_class')
        confidence = float(request.POST.get('confidence', 0))
        
        # Ruta al archivo de estadísticas
        stats_file = os.path.join(PROJECT_ROOT, 'models', f'{model_type}_usage_stats.json')
        
        # Cargar stats existentes o crear nuevas
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        else:
            stats = {
                'total_recognitions': 0,
                'by_class': {},
                'confidence_history': [],
                'last_updated': None
            }
        
        # Actualizar estadísticas
        stats['total_recognitions'] += 1
        
        if predicted_class not in stats['by_class']:
            stats['by_class'][predicted_class] = {
                'count': 0,
                'avg_confidence': 0,
                'confidences': []
            }
        
        class_stats = stats['by_class'][predicted_class]
        class_stats['count'] += 1
        class_stats['confidences'].append(confidence)
        class_stats['avg_confidence'] = sum(class_stats['confidences']) / len(class_stats['confidences'])
        
        # Mantener solo las últimas 100 confianzas por clase
        if len(class_stats['confidences']) > 100:
            class_stats['confidences'] = class_stats['confidences'][-100:]
        
        # Historial de confianza global (últimas 50)
        stats['confidence_history'].append(confidence)
        if len(stats['confidence_history']) > 50:
            stats['confidence_history'] = stats['confidence_history'][-50:]
        
        from datetime import datetime
        stats['last_updated'] = datetime.now().isoformat()
        
        # Guardar
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return JsonResponse({
            'status': 'success',
            'message': 'Estadística guardada'
        })
    
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


# ============================================================================
# API: OBTENER ESTADÍSTICAS DE USO
# ============================================================================
@require_http_methods(["GET"])
def get_usage_stats(request):
    """
    Retorna estadísticas de uso en tiempo real
    """
    try:
        model_type = request.GET.get('model_type', 'static')
        stats_file = os.path.join(PROJECT_ROOT, 'models', f'{model_type}_usage_stats.json')
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            return JsonResponse({
                'status': 'success',
                'data': stats
            })
        else:
            return JsonResponse({
                'status': 'success',
                'data': {
                    'total_recognitions': 0,
                    'by_class': {},
                    'confidence_history': []
                }
            })
    
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)