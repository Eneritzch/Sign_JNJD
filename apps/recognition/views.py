# apps/recognition/views.py
import os
import numpy as np
from io import BytesIO
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# === Rutas ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet50_static.h5')
DYNAMIC_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn_bigru_dynamic.h5')

# === Flags de configuración (ajustables según entrenamiento real) ===
DYNAMIC_USE_EFF_PREPROCESS = True  # Si False, se usará escalado 0..1
DYNAMIC_TRANSPOSE = False          # Si True, transpone a [B, H, W, C, T]

# === Clases por modelo ===
CLASS_NAMES_STATIC = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y'
]

CLASS_NAMES_DYNAMIC = [
    "hola","adios","chao","bienvenido","buenos_dias","buenas_tardes","buenas_noches",
    "gracias","por_favor","mucho_gusto","como_estas","feliz","triste","enojado",
    "sorprendido","cansado","malo","bueno","nombre","familia","j","z"
]

# === Cache de modelos (lazy load) ===
_static_model_cache = None
_dynamic_model_cache = None


def _check_model_file(path, kind):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo {kind} no encontrado en: {path}")


def get_static_model():
    global _static_model_cache
    if _static_model_cache is None:
        _check_model_file(STATIC_MODEL_PATH, 'estático')
        try:
            _static_model_cache = load_model(STATIC_MODEL_PATH)
            print(f"[Recognition] Modelo estático cargado: {STATIC_MODEL_PATH}")
        except Exception as e:
            # Mensaje claro si faltan custom_objects
            raise RuntimeError(f"Error al cargar modelo estático: {e}")
    return _static_model_cache


def get_dynamic_model():
    global _dynamic_model_cache
    if _dynamic_model_cache is None:
        _check_model_file(DYNAMIC_MODEL_PATH, 'dinámico')
        try:
            _dynamic_model_cache = load_model(DYNAMIC_MODEL_PATH)
            print(f"[Recognition] Modelo dinámico cargado: {DYNAMIC_MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Error al cargar modelo dinámico: {e}")
    return _dynamic_model_cache


# === Vistas ===
def live_demo(request):
    return render(request, 'recognition/live_demo.html')


@csrf_exempt
@require_http_methods(["POST"])
def predict_static(request):
    try:
        image_file = request.FILES.get('frame')
        if not image_file:
            return JsonResponse({'error': 'No se recibió frame'}, status=400)

        # Preprocesamiento: 224x224 RGB -> ResNet preprocess
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = resnet_preprocess(image_array.astype(np.float32))

        model = get_static_model()
        predictions = model.predict(image_array, verbose=0)
        pred_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        return JsonResponse({
            'type': 'static',
            'class': CLASS_NAMES_STATIC[pred_idx],
            'confidence': round(confidence, 2)
        })
    except FileNotFoundError as e:
        return JsonResponse({'error': str(e)}, status=500)
    except RuntimeError as e:
        return JsonResponse({'error': str(e)}, status=500)
    except Exception as e:
        return JsonResponse({'error': f"Error en predict_static: {e}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_dynamic(request):
    try:
        frames = []
        for i in range(20):
            frame_file = request.FILES.get(f'frame_{i}')
            if not frame_file:
                return JsonResponse({'error': f'Falta frame_{i}'}, status=400)

            image = Image.open(frame_file).convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image)
            frames.append(image_array)

        # Array con forma [T, H, W, C] -> [1, T, H, W, C]
        frames = np.array(frames, dtype=np.float32)
        frames = np.expand_dims(frames, axis=0)

        # Opción de transponer si el modelo lo requiere
        if DYNAMIC_TRANSPOSE:
            # [B, T, H, W, C] -> [B, H, W, C, T]
            frames = np.transpose(frames, (0, 2, 3, 4, 1))

        # Preprocesamiento
        if DYNAMIC_USE_EFF_PREPROCESS:
            frames = eff_preprocess(frames)
        else:
            frames /= 255.0

        model = get_dynamic_model()
        predictions = model.predict(frames, verbose=0)
        pred_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        return JsonResponse({
            'type': 'dynamic',
            'prediction': CLASS_NAMES_DYNAMIC[pred_idx],
            'confidence': round(confidence, 2)
        })
    except FileNotFoundError as e:
        return JsonResponse({'error': str(e)}, status=500)
    except RuntimeError as e:
        return JsonResponse({'error': str(e)}, status=500)
    except Exception as e:
        return JsonResponse({'error': f"Error en predict_dynamic: {e}"}, status=500)