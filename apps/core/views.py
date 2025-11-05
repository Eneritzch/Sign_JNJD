from django.shortcuts import render
from django.views.decorators.http import require_http_methods

@require_http_methods(["GET"])
def home(request):
    """Vista principal - Dashboard"""
    # Detectar si está en modo oscuro (puedes ajustar esto según cómo manejes el tema)
    dark_mode = request.COOKIES.get('dark_mode', 'False') == 'True'
    # O si usas localStorage + JS, puedes pasar un contexto desde el template

    context = {
        'page_title': 'Dashboard',
        'page_icon': 'chart-bar',
        'dark_mode': dark_mode,
        'mode_text': 'Oscuro' if dark_mode else 'Claro',  # 👈 Esta es la clave
    }
    return render(request, 'home.html', context)