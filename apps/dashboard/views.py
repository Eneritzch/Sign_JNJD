from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

@require_http_methods(["GET"])
def dashboard_view(request):
    """Vista principal del dashboard"""
    context = {
        'page_title': 'Analytics',
        'page_icon': 'chart-line',
    }
    return render(request, 'dashboard/index.html', context)

@require_http_methods(["GET"])
def metrics_api(request):
    """API para obtener métricas"""
    metrics = {
        'accuracy': 94.2,
        'total_samples': 5240,
        'error_rate': 5.8,
        'inference_time': 45,
        'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'accuracies': [96, 92, 94, 89, 95, 91, 93, 97, 90, 94]
    }
    return JsonResponse(metrics)