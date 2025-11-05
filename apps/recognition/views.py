from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

@require_http_methods(["GET"])
def recognition_view(request):
    """Vista principal de reconocimiento"""
    context = {
        'page_title': 'Reconocimiento',
        'page_icon': 'video-camera',
    }
    return render(request, 'recognition/reconocimiento.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def predict_api(request):
    """API para predicción en tiempo real"""
    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        
        # Aquí irá la lógica del modelo CNN
        # Por ahora retornamos datos simulados
        
        prediction = {
            'letter': 'A',
            'confidence': 0.92,
            'success': True
        }
        return JsonResponse(prediction)
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=400)