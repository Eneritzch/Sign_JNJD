import os
import numpy as np
import cv2
from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import tensorflow as tf
import base64
import requests
from datetime import datetime
import json
from apps.dashboard.views import save_recognition_stat

# ============================================================================
# CONFIGURACIÓN BÁSICA
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'mobilenetv2_static_ft.h5')
DYNAMIC_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'cnn_bigru_dynamic.h5')

# PARA HTML DE STATIC

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
            print("GPU configurada")
        except RuntimeError as e:
            print(f"GPU warning: {e}")
    
    # Cargar modelo
    static_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    static_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Predicción dummy para inicializar
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    static_model.predict(dummy, verbose=0)
    
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar modelo: {e}")

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
# FUNCIÓN AUXILIAR PARA GUARDAR ESTADÍSTICAS
# ============================================================================
def _save_usage_stat(request: HttpRequest, model_type: str, predicted_class: str, confidence: float):
    """Guarda la estadística de uso llamando directamente a la función del dashboard."""
    try:
        # Crear un mock request para la función save_recognition_stat
        from django.http import HttpRequest as MockRequest
        mock_request = MockRequest()
        mock_request.method = 'POST'
        mock_request.POST = {
            'model_type': model_type,
            'predicted_class': predicted_class,
            'confidence': str(confidence)
        }

        # Llamar directamente a la función del dashboard
        response = save_recognition_stat(mock_request)
        if response.status_code == 200:
            print(f"Usage stat saved: {model_type} - {predicted_class} ({confidence:.2f}%)")
        else:
            print(f"Failed to save usage stat: {response.content}")
    except Exception as e:
        print(f"Could not save usage stat: {e}")
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
        
        # Retornar resultado
        response_data = {
            'status': 'success',
            'class': letra_detectada,
            'confidence': round(confianza, 2),
            'top3': top3,
            'message': f'Letra detectada: {letra_detectada} con {confianza:.1f}% de confianza'
        }
        
        # Guardar estadística de uso de forma asíncrona
        _save_usage_stat(request, 'static', letra_detectada, confianza)
        
        return JsonResponse(response_data)
    
    except Exception as e:
        print(f"Error en reconocimiento: {e}")
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)

# ============================================================================
# VISTA ESTÁTICA
# ============================================================================
def static(request):
    """Renderiza la página de captura y reconocimiento"""
    return render(request, 'recognition/static.html')




# ============================================================================
# PARA HTML DE DYNAMIC
# ============================================================================

NUM_FRAMES = 20  # Número de frames que espera tu modelo
IMG_SIZE = (224, 224)  # Tamaño de cada frame

# Tus clases (ajusta según tus carpetas de entrenamiento)
LABELS_DYNAMIC = ["hola","adios","chao","bienvenido","buenos_dias","buenas_tardes","buenas_noches",
                    "gracias","por_favor","mucho_gusto","como_estas","feliz","triste","enojado",
                    "sorprendido","cansado","malo","bueno","nombre","familia","j","z"]

# ============================================================================
# CARGAR MODELO DINÁMICO
# ============================================================================
dynamic_model = None
try:
    # Configurar TensorFlow
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"⚠️ GPU warning: {e}")
    
    dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH, compile=False)
    dynamic_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Warm-up prediction
    dummy = np.zeros((1, NUM_FRAMES, *IMG_SIZE, 3), dtype=np.float32)
    dynamic_model.predict(dummy, verbose=0)
    
    print(f"Modelo dinámico cargado - Input shape: {dynamic_model.input_shape}")
except Exception as e:
    print(f"Error cargando modelo dinámico: {e}")

# ============================================================================
# PREPROCESAMIENTO 
# ============================================================================
def efficientnet_preprocess(frame):
    """
    Preprocesamiento idéntico al de EfficientNetB0:
    - Redimensiona a 224x224
    - Aplica preprocess_input de EfficientNet (normalización específica)
    """
    # Redimensionar
    frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Convertir a float32
    frame = frame.astype(np.float32)
    
    # Aplicar el mismo preprocesamiento que EfficientNetB0
    # Rango: [0, 255] -> normalización específica de EfficientNet
    # Fórmula: (x - mean) / std donde mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    frame[..., 0] -= 123.675
    frame[..., 1] -= 116.28
    frame[..., 2] -= 103.53
    
    frame[..., 0] /= 58.395
    frame[..., 1] /= 57.12
    frame[..., 2] /= 57.375
    
    return frame

def procesar_secuencia_dinamica(frames_base64):

    frames_decodificados = []
    
    # Decodificar todos los frames
    for frame_b64 in frames_base64:
        try:
            if ',' in frame_b64:
                frame_b64 = frame_b64.split(',')[1]
            
            image_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # BGR a RGB (importante!)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames_decodificados.append(img_rgb)
        except Exception as e:
            print(f"Error decodificando frame: {e}")
            continue
    
    if len(frames_decodificados) == 0:
        raise ValueError("No se pudo decodificar ningún frame")
    
    total_frames = len(frames_decodificados)
    
    # Sampling uniforme para obtener exactamente NUM_FRAMES
    if total_frames != NUM_FRAMES:
        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        frames_decodificados = [frames_decodificados[i] for i in indices]
    
    # Preprocesar cada frame
    frames_procesados = []
    for frame in frames_decodificados:
        frame_prep = efficientnet_preprocess(frame)
        frames_procesados.append(frame_prep)
    
    return np.array(frames_procesados, dtype=np.float32)

# ============================================================================
# ENDPOINT: RECONOCER SEÑA DINÁMICA
# ============================================================================
@csrf_exempt
@require_http_methods(["POST"])
def recognize_dynamic(request):
    """
    Recibe secuencia de frames y predice con CNN+BiGRU.
    
    Parámetros:
    - frames: JSON array de base64 strings
    
    Retorna:
    - class: Seña detectada
    - confidence: Confianza %
    - top3: Top 3 predicciones
    - frames_info: Info de procesamiento
    """
    if not dynamic_model:
        return JsonResponse({
            'error': 'Modelo dinámico no disponible',
            'status': 'error'
        }, status=500)
    
    try:
        # Obtener frames
        frames_json = request.POST.get('frames')
        if not frames_json:
            return JsonResponse({
                'error': 'No se enviaron frames',
                'status': 'error'
            }, status=400)
        
        frames_list = json.loads(frames_json)
        
        if len(frames_list) == 0:
            return JsonResponse({
                'error': 'Lista de frames vacía',
                'status': 'error'
            }, status=400)
        
        frames_recibidos = len(frames_list)
        print(f"Frames recibidos: {frames_recibidos}")
        
        # Procesar secuencia
        secuencia = procesar_secuencia_dinamica(frames_list)
        print(f"Secuencia procesada: {secuencia.shape}")
        
        # Agregar batch dimension: (1, 20, 224, 224, 3)
        secuencia_batch = np.expand_dims(secuencia, axis=0)
        
        # Predecir
        predicciones = dynamic_model.predict(secuencia_batch, verbose=0)[0]
        
        # Resultado principal
        clase_id = int(np.argmax(predicciones))
        confianza = float(predicciones[clase_id]) * 100
        seña_detectada = LABELS_DYNAMIC[clase_id]
        
        print(f"Predicción: {seña_detectada} ({confianza:.2f}%)")
        
        # Top 3
        top3_indices = np.argsort(predicciones)[-3:][::-1]
        top3 = [
            {
                'seña': LABELS_DYNAMIC[i],
                'confianza': round(float(predicciones[i]) * 100, 2)
            }
            for i in top3_indices
        ]
        
        # Guardar estadística de uso
        _save_usage_stat(request, 'dynamic', seña_detectada, confianza)
        
        return JsonResponse({
            'status': 'success',
            'class': seña_detectada,
            'confidence': round(confianza, 2),
            'top3': top3,
            'frames_info': {
                'received': frames_recibidos,
                'used': NUM_FRAMES,
                'sampling': 'uniform' if frames_recibidos != NUM_FRAMES else 'none'
            },
            'message': f'Seña detectada: {seña_detectada} ({confianza:.1f}%)'
        })
        
    except Exception as e:
        print(f"Error en reconocimiento dinámico: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)

# ============================================================================
# VISTA DINÁMICO
# ============================================================================
def dynamic(request):
    """Renderiza página de reconocimiento dinámico"""
    context = {
        'num_frames': NUM_FRAMES,
        'labels': LABELS_DYNAMIC,
        'model_loaded': dynamic_model is not None
    }
    return render(request, 'recognition/dynamic.html', context)