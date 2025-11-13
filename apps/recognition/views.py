import os
import numpy as np
import cv2
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import tensorflow as tf
import base64
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN BÁSICA
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'mobilenetv2_static_ft.h5')

# Las fotos NO se guardan en disco - solo procesamiento en memoria

# ============================================================================
# CARGAR MODELO UNA SOLA VEZ
# ============================================================================
static_model = None
try:
    # Configurar TensorFlow para eficiencia
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    # Limitar memoria GPU si existe
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU configurada")
        except RuntimeError as e:
            print(f"⚠️ GPU warning: {e}")
    
    # Cargar modelo
    static_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    static_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Predicción dummy para inicializar
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    static_model.predict(dummy, verbose=0)
    
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")

# Etiquetas del modelo
STATIC_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y'
]

# ============================================================================
# PREPROCESAMIENTO DE IMAGEN
# ============================================================================
def preparar_imagen(img):
    """
    Preprocesa imagen para MobileNetV2:
    1. Convierte a cuadrado con padding
    2. Redimensiona a 224x224
    3. Normaliza [-1, 1]
    """
    h, w = img.shape[:2]
    
    # Hacer imagen cuadrada (padding)
    if h != w:
        size = max(h, w)
        cuadrado = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        cuadrado[y_offset:y_offset+h, x_offset:x_offset+w] = img
        img = cuadrado
    
    # Redimensionar a 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalización MobileNetV2: [-1, 1]
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    
    return img

# ============================================================================
# GUARDAR FOTO (OPCIONAL)
# ============================================================================
def guardar_foto_capturada(img_array, letra_detectada, confianza):
    """
    Guarda la foto capturada en disco para análisis posterior.
    Nombre: YYYY-MM-DD_HH-MM-SS_LETRA_CONFIANZA.jpg
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{letra_detectada}_{confianza:.1f}.jpg"
        filepath = os.path.join(PHOTOS_DIR, filename)
        
        cv2.imwrite(filepath, img_array)
        return filename
    except Exception as e:
        print(f"⚠️ Error guardando foto: {e}")
        return None

# ============================================================================
# ENDPOINT: RECONOCER SEÑA DESDE FOTO CAPTURADA
# ============================================================================
@csrf_exempt
@require_http_methods(["POST"])
def recognize_from_photo(request):
    """
    Recibe una foto en base64, la procesa y retorna la letra detectada.
    
    Parámetros esperados:
    - photo: string base64 de la imagen (formato: data:image/jpeg;base64,...)
    - save_photo: booleano opcional para guardar la foto en disco
    
    Retorna:
    - class: Letra detectada
    - confidence: Confianza en %
    - top3: Top 3 predicciones con sus confianzas
    - saved_as: Nombre del archivo guardado (si save_photo=true)
    """
    if not static_model:
        return JsonResponse({
            'error': 'Modelo no disponible',
            'status': 'error'
        }, status=500)
    
    try:
        # Obtener imagen en base64
        photo_data = request.POST.get('photo')
        save_photo = request.POST.get('save_photo', 'false').lower() == 'true'
        
        if not photo_data:
            return JsonResponse({
                'error': 'No se envió la foto',
                'status': 'error'
            }, status=400)
        
        # Decodificar base64
        if ',' in photo_data:
            photo_data = photo_data.split(',')[1]
        
        # Convertir a bytes
        image_bytes = base64.b64decode(photo_data)
        
        # Decodificar imagen con OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JsonResponse({
                'error': 'Imagen inválida',
                'status': 'error'
            }, status=400)
        
        # Convertir BGR a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocesar
        img_prep = preparar_imagen(img_rgb)
        img_batch = np.expand_dims(img_prep, axis=0)
        
        # Predecir
        predicciones = static_model.predict(img_batch, verbose=0)[0]
        
        # Resultado principal
        clase_id = int(np.argmax(predicciones))
        confianza = float(predicciones[clase_id]) * 100
        letra_detectada = STATIC_LABELS[clase_id]
        
        # Top 3 predicciones
        top3_indices = np.argsort(predicciones)[-3:][::-1]
        top3 = [
            {
                'letra': STATIC_LABELS[i],
                'confianza': round(float(predicciones[i]) * 100, 2)
            }
            for i in top3_indices
        ]
        
        # Guardar foto si se solicitó
        saved_filename = None
        if save_photo:
            saved_filename = guardar_foto_capturada(img, letra_detectada, confianza)
        
        # Retornar resultado
        response_data = {
            'status': 'success',
            'class': letra_detectada,
            'confidence': round(confianza, 2),
            'top3': top3,
            'message': f'Letra detectada: {letra_detectada} con {confianza:.1f}% de confianza'
        }
        
        if saved_filename:
            response_data['saved_as'] = saved_filename
        
        return JsonResponse(response_data)
    
    except Exception as e:
        print(f"❌ Error en reconocimiento: {e}")
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)

# ============================================================================
# VISTA PRINCIPAL
# ============================================================================
def live_demo(request):
    """Renderiza la página de captura y reconocimiento"""
    return render(request, 'recognition/live_demo.html')

# ============================================================================
# ENDPOINT: OBTENER ESTADÍSTICAS DE FOTOS GUARDADAS
# ============================================================================
@csrf_exempt
@require_http_methods(["GET"])
def get_saved_photos_stats(request):
    """
    Retorna estadísticas de las fotos guardadas:
    - Total de fotos
    - Letras más capturadas
    - Confianza promedio por letra
    """
    try:
        import glob
        
        photos = glob.glob(os.path.join(PHOTOS_DIR, "*.jpg"))
        
        if not photos:
            return JsonResponse({
                'total_photos': 0,
                'message': 'No hay fotos guardadas aún'
            })
        
        # Analizar nombres de archivo
        stats = {}
        for photo in photos:
            basename = os.path.basename(photo)
            parts = basename.split('_')
            
            if len(parts) >= 4:
                letra = parts[2]
                conf = float(parts[3].replace('.jpg', ''))
                
                if letra not in stats:
                    stats[letra] = {
                        'count': 0,
                        'confidences': []
                    }
                
                stats[letra]['count'] += 1
                stats[letra]['confidences'].append(conf)
        
        # Calcular promedios
        result = {}
        for letra, data in stats.items():
            result[letra] = {
                'count': data['count'],
                'avg_confidence': round(sum(data['confidences']) / len(data['confidences']), 2),
                'min_confidence': round(min(data['confidences']), 2),
                'max_confidence': round(max(data['confidences']), 2)
            }
        
        return JsonResponse({
            'total_photos': len(photos),
            'letters': result,
            'photos_directory': PHOTOS_DIR
        })
    
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)