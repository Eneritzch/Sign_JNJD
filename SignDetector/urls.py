from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # Apps Routes
    path('', include('apps.core.urls')), 
    path('recognition/', include('apps.recognition.urls')),
    path('dashboard/', include('apps.dashboard.urls')),
    path('challenge/', include('apps.challenge.urls')),
]

# Agregar esto al final (fuera de urlpatterns)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)