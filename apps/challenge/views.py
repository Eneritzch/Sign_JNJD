from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

@require_http_methods(["GET"])
def challenge_view(request):
    """Vista principal del desafío"""
    context = {
        'page_title': 'Desafío',
        'page_icon': 'gamepad-2',
    }
    return render(request, 'challenge/index.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def start_challenge_api(request):
    """API para iniciar desafío"""
    try:
        data = json.loads(request.body)
        difficulty = data.get('difficulty', 'medium')
        
        challenge_data = {
            'id': 1,
            'letters': ['A', 'B', 'C', 'D', 'E'],
            'difficulty': difficulty,
            'success': True
        }
        return JsonResponse(challenge_data)
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def submit_challenge_api(request):
    """API para enviar resultado del desafío"""
    try:
        data = json.loads(request.body)
        score = data.get('score')
        accuracy = data.get('accuracy')
        
        result = {
            'saved': True,
            'success': True
        }
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=400)