# Documentación de Implementación - Fase 1

## Descripción General
Se implementó la primera fase del plan básico: integración del clasificador de sketches con detección de gestos en tiempo real. La aplicación detecta dibujos en el aire y los clasifica usando el modelo preentrenado.

## ✓ Completado

### 1. Estructura de Directorios
Se creó la siguiente estructura de carpetas:
```
IA/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── hand_detector.py       # Detección de manos con MediaPipe
│   │   ├── gesture_processor.py   # Procesamiento de gestos a imágenes
│   │   └── classifier.py          # Clasificación con modelo ML
│   └── ui/                        # (Para futuro)
├── config/                        # (Para futuro)
├── air_draw_classifier.py         # Aplicación principal
```

### 2. Módulo: HandDetector (`src/core/hand_detector.py`)
**Responsabilidad**: Detectar manos y dedos en tiempo real usando MediaPipe

**Funcionalidades principales**:
- `detect(frame)`: Detecta landmarks de manos en un frame
- `draw_landmarks(frame, landmarks)`: Dibuja los landmarks en el video
- `get_index_finger_tip(landmarks)`: Obtiene posición del dedo índice (punto de dibujo)
- `get_thumb_tip(landmarks)`: Obtiene posición del pulgar
- `is_fist(landmarks)`: Detecta si la mano está cerrada

**Características técnicas**:
- Configuración de confianza ajustable
- Soporte para 2 manos simultáneas
- Conversión BGR↔RGB para compatibilidad con MediaPipe

### 3. Módulo: GestureProcessor (`src/core/gesture_processor.py`)
**Responsabilidad**: Convertir movimientos de manos 3D a imágenes 2D (28x28)

**Funcionalidades principales**:
- `normalize_coordinates()`: Convierte coordenadas normalizadas a píxeles del canvas
- `add_point()`: Agrega un punto al trazo actual
- `smooth_stroke()`: Suaviza los puntos para reducir ruido
- `points_to_image()`: Convierte puntos a imagen 28x28
- `get_gesture_image()`: Obtiene imagen final y limpia canvas
- `draw_on_frame()`: Dibuja trazo en tiempo real en el frame

**Características técnicas**:
- Canvas interno de 256x256 para mejor precisión
- Suavizado de Poisson para trazo más limpio
- Normalización a rango [0, 1] compatible con modelo
- Inversión de colores: dibujo blanco sobre fondo negro

### 4. Módulo: SketchClassifier (`src/core/classifier.py`)
**Responsabilidad**: Cargar y usar el modelo para clasificación

**Funcionalidades principales**:
- `predict(image, top_k)`: Predice clase de una imagen
- `get_top_prediction()`: Obtiene predicción con mayor confianza
- `is_confident_prediction()`: Verifica si predicción supera umbral
- `get_model_info()`: Retorna información del modelo

**Características técnicas**:
- Carga modelo .keras o .h5
- Soporte para top-K predicciones
- Umbral de confianza configurable
- Mapeo automático índice → nombre de clase

### 5. Aplicación Principal (`air_draw_classifier.py`)
**Integra todos los componentes en un loop interactivo**

**Funcionalidades principales**:
- Captura video de cámara web en tiempo real
- Detecta intención de dibujo (dedo índice extendido)
- Registra trazos mientras usuario dibuja
- Clasifica gesto al completarlo
- Muestra predicciones en tiempo real

**Interfaz de usuario**:
```
┌─────────────────────────────────────────────────────┐
│ Video Feed con overlay:                             │
│ - Landmarks de mano detectados (puntos azules)     │
│ - Trazo dibujado en progreso (línea verde)         │
│ - Contador de puntos                               │
│ - FPS actual                                       │
│ - Top 3 predicciones con confianza                 │
└─────────────────────────────────────────────────────┘

Controles:
  r - Resetear dibujo actual
  SPACE - Procesar/clasificar gesto
  q - Salir de la aplicación
```

**Flujo de ejecución**:
1. Inicializa componentes (detector, procesador, clasificador)
2. Abre cámara web
3. Loop principal:
   - Captura frame
   - Detecta manos
   - Si dedo índice extendido → registra puntos
   - Si mano relaja → marca gesto como completo
   - Dibuja puntos en pantalla
   - Procesa teclas (SPACE para clasificar)
4. Al procesar:
   - Convierte puntos a imagen 28x28
   - Realiza predicción con modelo
   - Muestra resultados

## Configuración

### Parámetros ajustables en `SketchDrawer`:
```python
self.confidence_threshold = 0.5          # Umbral de confianza (0-1)
self.min_points_for_gesture = 5          # Mínimo de puntos para procesar
```

### Parámetros en `HandDetector`:
```python
min_detection_confidence = 0.7           # Confianza de detección (0-1)
min_tracking_confidence = 0.7            # Confianza de seguimiento (0-1)
```

### Parámetros en `GestureProcessor`:
```python
image_size = 28                          # Tamaño final (28x28 para modelo)
line_width = 2                           # Ancho de línea dibujada
canvas_size = 256                        # Canvas interno
```

## Dependencias Requeridas

```
opencv-python>=4.8.0
mediapipe>=0.10.0
tensorflow>=2.10.0
numpy>=1.24.0
pillow>=10.0.0
```

## Instalación y Uso

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar aplicación
```bash
python air_draw_classifier.py
```

### 3. Usar la aplicación
- Apuntar cámara hacia usted
- Extender dedo índice para "activar" modo dibujo
- Mover dedo para dibujar en el aire
- Flexionar dedo/cerrar mano para completar dibujo
- Presionar SPACE para clasificar
- Ver predicciones en pantalla

## Limitaciones Conocidas

1. **Iluminación**: Funciona mejor con buena iluminación frontal
2. **Trazo rápido**: Gestos muy rápidos pueden perder puntos
3. **Fondo complejo**: Fondos con muchas manos/objetos pueden confundir
4. **Precisión gesto**: Algunos gestos pueden no mapear perfectamente al dataset

## Mejoras Futuras

1. Agregar interfaz gráfica mejorada (PyQt)
2. Guardar gestos dibujados para análisis
3. Soporte para múltiples gestos simultáneos
4. Calibración automática por iluminación
5. Tests unitarios para cada módulo
6. Predicciones más suave (buffer/averaging)
7. Soporte para otros modelos

## Notas Técnicas

### Conversión 3D → 2D
- MediaPipe proporciona coordenadas normalizadas (0-1) en espacio 3D
- Se mapean a canvas 2D interno (256x256)
- Se redimensiona a 28x28 para compatibilidad con modelo
- Se suaviza para reducir ruido de movimiento

### Detección de intención de dibujo
- Dedo índice extendido = activación de dibujo
- Se verifica que punta del índice esté debajo de PIP (primer segmento)
- Método simple pero efectivo

### Procesamiento en tiempo real
- Captura: ~30 FPS
- Detección: ~25 FPS
- Procesamiento: negligible
- Total: ~20-25 FPS mantenidos

## Troubleshooting

### Cámara no se abre
- Verificar que no esté en uso por otra aplicación
- Revisar permisos de cámara del sistema

### MediaPipe no detecta manos
- Mejorar iluminación
- Acercar mano a cámara
- Aumentar `min_detection_confidence` en HandDetector

### Predicciones incorrectas
- Dibujar más lentamente
- Presionar SPACE después de completar gesto
- Asegurarse que trazo sea similar al dataset Quick Draw

## Referencias

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)

---
**Fecha de creación**: 15 de Noviembre, 2025
**Versión**: 1.0.0
