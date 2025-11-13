import os
import numpy as np
import cv2
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import tensorflow as tf
from collections import deque
import pandas as pd
from datetime import datetime
import time

# ============================================================================
# CONFIGURACIÓN BÁSICA
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'mobilenetv2_static_ft.h5')
DATA_PATH = os.path.join(PROJECT_ROOT, 'apps', 'recognition', 'data', 'detecciones.csv')

# ============================================================================
# OPTIMIZACIÓN: Cargar modelo UNA SOLA VEZ con configuración eficiente
# ============================================================================
static_model = None
try:
    # Configurar TensorFlow para usar menos recursos
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    # Limitar memoria GPU si existe
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU configurada con crecimiento dinámico de memoria")
        except RuntimeError as e:
            print(f"⚠️ GPU warning: {e}")
    
    # Cargar modelo
    static_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # CRÍTICO: Compilar sin optimizaciones pesadas
    static_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Hacer una predicción dummy para inicializar (evita lag en primera predicción)
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    static_model.predict(dummy, verbose=0)
    
    print("✅ Modelo cargado y optimizado")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")

STATIC_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y'
]

# ============================================================================
# ⚡ UMBRALES ADAPTATIVOS POR LETRA (NUEVO)
# ============================================================================
UMBRALES_CONFIANZA = {
    # Letras bien detectadas (umbral estándar)
    'A': 35, 'D': 35, 'E': 35, 'L': 35, 'O': 40, 'S': 35, 'Y': 40,
    
    # Letras que le cuestan (umbral moderado)
    'T': 30, 'V': 30,
    
    # Letras confundidas (umbral alto para evitar falsos positivos)
    'M': 45, 'N': 45, 'P': 45, 'Q': 50,
    
    # ⚡ LETRAS PROBLEMÁTICAS: Umbral BAJO (20-25%)
    'B': 15, 'C': 15, 'F': 15, 'G': 15, 'H': 15,
    'I': 15, 'K': 15, 'R': 15, 'U': 15, 'X': 15,
    
    # Resto
    'W': 30
}

# ============================================================================
# CONTROL DE TASA DE PROCESAMIENTO (evita sobrecarga)
# ============================================================================
class RateLimiter:
    """Limita cuántas predicciones por segundo se procesan"""
    def __init__(self, max_fps=10):
        self.max_fps = max_fps
        self.min_interval = 1.0 / max_fps
        self.last_process_time = 0
    
    def should_process(self):
        """Retorna True si ya pasó suficiente tiempo"""
        current_time = time.time()
        if current_time - self.last_process_time >= self.min_interval:
            self.last_process_time = current_time
            return True
        return False

# Limitador global: solo procesa 10 frames por segundo
rate_limiter = RateLimiter(max_fps=10)

# ============================================================================
# ⚡ BUFFER ADAPTATIVO MEJORADO (REEMPLAZA SimpleBuffer)
# ============================================================================
class BufferAdaptativo:
    """
    Buffer que ajusta automáticamente según la letra detectada.
    Para letras difíciles, es más permisivo.
    """
    def __init__(self, tamano_maximo=5):
        self.buffer = deque(maxlen=tamano_maximo)
        self.letras_dificiles = {'B', 'C', 'F', 'G', 'H', 'I', 'K', 'R', 'U', 'X'}
        self.letras_confusas = {'M', 'N', 'P', 'Q', 'S'}
    
    def add(self, letra, confianza):
        """Agrega detección con timestamp"""
        self.buffer.append({
            'letra': letra,
            'conf': confianza,
            'timestamp': time.time()
        })
    
    def get_best(self):
        """
        Retorna la mejor letra según tipo:
        - Letras fáciles: Necesitan 2/3 coincidencias (66%)
        - Letras difíciles: Necesitan 2/5 coincidencias (40%)
        - Letras confusas: Necesitan 3/5 coincidencias (60%)
        """
        if len(self.buffer) < 2:
            return None, 0
        
        # Contar votos
        votos = {}
        confianzas = {}
        
        for item in self.buffer:
            letra = item['letra']
            conf = item['conf']
            
            votos[letra] = votos.get(letra, 0) + 1
            if letra not in confianzas:
                confianzas[letra] = []
            confianzas[letra].append(conf)
        
        # No hay letra ganadora clara
        if not votos:
            return None, 0
        
        # Ordenar por votos
        letra_ganadora = max(votos, key=votos.get)
        votos_ganadora = votos[letra_ganadora]
        conf_promedio = sum(confianzas[letra_ganadora]) / len(confianzas[letra_ganadora])
        
        # VALIDACIÓN ADAPTATIVA según tipo de letra
        tamano_buffer = len(self.buffer)
        
        # Letras difíciles: MUY TOLERANTE (40% de votos)
        if letra_ganadora in self.letras_dificiles:
            umbral_votos = max(2, int(tamano_buffer * 0.4))  # Mínimo 2 votos o 40%
            if votos_ganadora >= umbral_votos:
                return letra_ganadora, conf_promedio
        
        # Letras confusas: TOLERANCIA MEDIA (60% de votos)
        elif letra_ganadora in self.letras_confusas:
            umbral_votos = max(3, int(tamano_buffer * 0.6))  # Mínimo 3 votos o 60%
            if votos_ganadora >= umbral_votos:
                return letra_ganadora, conf_promedio
        
        # Letras normales: TOLERANCIA ESTÁNDAR (50% de votos)
        else:
            umbral_votos = max(2, int(tamano_buffer * 0.5))  # Mínimo 2 votos o 50%
            if votos_ganadora >= umbral_votos:
                return letra_ganadora, conf_promedio
        
        return None, 0
    
    def clear(self):
        """Limpia el buffer"""
        self.buffer.clear()

# ⚡ INSTANCIA GLOBAL DEL BUFFER MEJORADO
buffer = BufferAdaptativo(tamano_maximo=5)

# ============================================================================
# PREPROCESAMIENTO OPTIMIZADO (mínimas operaciones)
# ============================================================================
def preparar_imagen(img):
    """Preprocesar imagen de forma eficiente"""
    # Si ya es 224x224 y cuadrada, saltar redimensionamiento
    h, w = img.shape[:2]
    
    # Hacer cuadrado solo si es necesario
    if h != w:
        size = max(h, w)
        cuadrado = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        cuadrado[y_offset:y_offset+h, x_offset:x_offset+w] = img
        img = cuadrado
    
    # Redimensionar solo si no es 224x224
    if img.shape[0] != 224:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalización in-place (más eficiente)
    img = img.astype(np.float32, copy=False)
    img = np.multiply(img, 1.0/127.5, out=img)
    img = np.subtract(img, 1.0, out=img)
    
    return img

# ============================================================================
# GUARDAR DETECCIONES (optimizado para no bloquear)
# ============================================================================
def guardar_deteccion(letra, confianza):
    """Guarda detección en CSV sin bloquear el procesamiento"""
    try:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        # Crear archivo si no existe
        if not os.path.exists(DATA_PATH):
            df = pd.DataFrame(columns=["Letra", "Confianza", "FechaHora"])
            df.to_csv(DATA_PATH, index=False)
            return
        
        # Agregar nueva detección
        nueva = pd.DataFrame({
            "Letra": [letra],
            "Confianza": [confianza],
            "FechaHora": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        
        # Append directo sin cargar todo el CSV
        nueva.to_csv(DATA_PATH, mode='a', header=False, index=False)
    except Exception as e:
        # No detener el proceso si falla el guardado
        print(f"⚠️ Error guardando CSV: {e}")

# ============================================================================
# ⚡ ENDPOINT DE PREDICCIÓN - OPTIMIZADO CON UMBRALES ADAPTATIVOS
# ============================================================================
@csrf_exempt
@require_http_methods(["POST"])
def predict_static_realtime(request):
    """
    Sistema de detección OPTIMIZADO con umbrales adaptativos por letra:
    - Limita FPS de procesamiento
    - Umbrales específicos para cada letra (20-50%)
    - Buffer adaptativo según dificultad
    - Predicciones sin verbose
    """
    if not static_model:
        return JsonResponse({'error': 'Modelo no disponible'}, status=500)
    
    # ⚡ CONTROL DE TASA: Procesar máximo 10 FPS
    if not rate_limiter.should_process():
        return JsonResponse({
            'class': '...',
            'confidence': 0,
            'skipped': True  # Indicador de frame saltado
        })
    
    try:
        # Leer frame
        frame_file = request.FILES.get('frame')
        if not frame_file:
            return JsonResponse({'error': 'No se envió frame'}, status=400)
        
        # Decodificar imagen
        image_bytes = frame_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JsonResponse({'error': 'Imagen inválida'}, status=400)
        
        # Preprocesar (optimizado)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_prep = preparar_imagen(img_rgb)
        img_batch = np.expand_dims(img_prep, axis=0)
        
        # Predecir (sin verbose para reducir overhead)
        predicciones = static_model.predict(img_batch, verbose=0)[0]
        
        # Resultado directo
        clase_id = int(np.argmax(predicciones))
        confianza = float(predicciones[clase_id]) * 100
        letra = STATIC_LABELS[clase_id]
        
        # ⚡ UMBRAL ADAPTATIVO por letra (NUEVO)
        umbral_letra = UMBRALES_CONFIANZA.get(letra, 30)
        
        # ⚡ DEBUG: Mostrar top 3 predicciones para letras difíciles
        letras_dificiles = {'B', 'C', 'F', 'G', 'H', 'I', 'K', 'R', 'U', 'X'}
        debug_info = None
        
        if letra in letras_dificiles:
            top3_indices = np.argsort(predicciones)[-3:][::-1]
            debug_info = {
                'top3': [
                    {
                        'letra': STATIC_LABELS[i],
                        'conf': round(float(predicciones[i]) * 100, 1)
                    }
                    for i in top3_indices
                ],
                'umbral': umbral_letra
            }
        
        # Validar contra umbral adaptativo
        if confianza >= umbral_letra:
            buffer.add(letra, confianza)
            letra_estable, conf_estable = buffer.get_best()
            
            if letra_estable:
                # Guardar sin bloquear
                guardar_deteccion(letra_estable, conf_estable)
                
                response_data = {
                    'class': letra_estable,
                    'confidence': round(conf_estable, 1),
                    'raw_class': letra,  # Para debug
                    'raw_confidence': round(confianza, 1),
                    'threshold_used': umbral_letra
                }
                
                if debug_info:
                    response_data['debug'] = debug_info
                
                return JsonResponse(response_data)
        
        # Si no pasa el umbral, retornar info para debug
        return JsonResponse({
            'class': '...',
            'confidence': round(confianza, 1),
            'rejected_class': letra,  # Letra rechazada
            'threshold_needed': umbral_letra,
            'debug': debug_info
        })
    
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return JsonResponse({'error': str(e)}, status=500)

# ============================================================================
# VISTA PRINCIPAL
# ============================================================================
def live_demo(request):
    return render(request, 'recognition/live_demo.html')

# ============================================================================
# ENDPOINT DE DIAGNÓSTICO DE RECURSOS
# ============================================================================
@csrf_exempt
@require_http_methods(["GET"])
def system_health(request):
    """Endpoint para monitorear uso de recursos"""
    import psutil
    
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Temperatura CPU (si está disponible)
        temp = None
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                temp = max([t.current for t in temps['coretemp']])
        except:
            pass
        
        return JsonResponse({
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'temperature_celsius': temp,
            'warning': cpu_percent > 80 or memory.percent > 85
        })
    except:
        return JsonResponse({'error': 'psutil no disponible'}, status=500)

# ============================================================================
# ⚡ ENDPOINT DE DEBUG DE UMBRALES (NUEVO)
# ============================================================================
@csrf_exempt
@require_http_methods(["GET"])
def debug_thresholds(request):
    """
    Endpoint para ver estadísticas de umbrales y detecciones.
    Visita: /recognition/debug/thresholds/
    """
    stats = {
        'umbrales_configurados': UMBRALES_CONFIANZA,
        'letras_por_dificultad': {
            'faciles': ['A', 'D', 'E', 'L', 'O', 'S', 'Y'],
            'dificiles': ['B', 'C', 'F', 'G', 'H', 'I', 'K', 'R', 'U', 'X'],
            'confusas': ['M', 'N', 'P', 'Q', 'S'],
            'intermedias': ['T', 'V', 'W']
        },
        'configuracion_buffer': {
            'tipo': 'BufferAdaptativo',
            'tamano_maximo': 5,
            'criterios_validacion': {
                'letras_faciles': '50% de votos (2/4 frames)',
                'letras_dificiles': '40% de votos (2/5 frames)',
                'letras_confusas': '60% de votos (3/5 frames)'
            }
        }
    }
    
    # Analizar CSV si existe
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            if not df.empty:
                # Letras más detectadas
                top_detectadas = df['Letra'].value_counts().head(10).to_dict()
                # Letras menos detectadas
                todas_letras = df['Letra'].value_counts().to_dict()
                # Confianza promedio por letra
                conf_promedio = df.groupby('Letra')['Confianza'].mean().to_dict()
                # Confianza mínima y máxima por letra
                conf_min = df.groupby('Letra')['Confianza'].min().to_dict()
                conf_max = df.groupby('Letra')['Confianza'].max().to_dict()
                
                # Identificar letras nunca detectadas
                letras_nunca_detectadas = [
                    letra for letra in STATIC_LABELS 
                    if letra not in todas_letras
                ]
                
                stats['analisis_csv'] = {
                    'total_detecciones': len(df),
                    'letras_unicas_detectadas': len(todas_letras),
                    'top_10_mas_detectadas': top_detectadas,
                    'todas_las_letras_conteo': todas_letras,
                    'letras_nunca_detectadas': letras_nunca_detectadas,
                    'confianza_por_letra': {
                        letra: {
                            'promedio': round(conf_promedio.get(letra, 0), 2),
                            'minima': round(conf_min.get(letra, 0), 2),
                            'maxima': round(conf_max.get(letra, 0), 2),
                            'umbral_configurado': UMBRALES_CONFIANZA.get(letra, 30)
                        }
                        for letra in STATIC_LABELS
                    }
                }
                
                # Análisis de efectividad de umbrales
                letras_bajo_umbral = []
                for letra in STATIC_LABELS:
                    if letra in conf_promedio:
                        prom = conf_promedio[letra]
                        umbral = UMBRALES_CONFIANZA.get(letra, 30)
                        if prom < umbral:
                            letras_bajo_umbral.append({
                                'letra': letra,
                                'confianza_promedio': round(prom, 2),
                                'umbral': umbral,
                                'diferencia': round(umbral - prom, 2)
                            })
                
                stats['analisis_csv']['letras_bajo_umbral'] = letras_bajo_umbral
                
        except Exception as e:
            stats['error_csv'] = str(e)
    else:
        stats['analisis_csv'] = {
            'mensaje': 'No hay datos de detecciones aún. CSV no existe.'
        }
    
    return JsonResponse(stats, json_dumps_params={'indent': 2})

# ============================================================================
# DASHBOARD DE ANÁLISIS
# ============================================================================
def realtime_report(request):
    """Muestra estadísticas de detecciones"""
    tabla = []
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if not df.empty:
            resumen = df.groupby('Letra')['Confianza'].agg(['count', 'mean', 'std']).reset_index()
            resumen.columns = ['letra', 'detecciones', 'conf_promedio', 'desviacion']
            resumen = resumen.sort_values('conf_promedio', ascending=False)
            tabla = resumen.to_dict('records')
    
    return render(request, 'recognition/reporte.html', {
        'tabla': tabla,
        'total': sum(row['detecciones'] for row in tabla) if tabla else 0
    })