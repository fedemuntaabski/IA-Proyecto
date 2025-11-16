# Pictionary Live 🎨

Aplicación Python interactiva para jugar **Pictionary en vivo** usando detección de gestos con las manos y clasificación de sketches con inteligencia artificial.

## Características

- 🎥 **Captura en tiempo real**: Lee video de la cámara web
- ✋ **Detección de manos**: Usa MediaPipe para tracking de manos en 3D
- ✍️ **Acumulación de trazo**: Detecta cuando dibujas en el aire y acumula la trayectoria
- 🤖 **Inferencia en vivo**: Clasifica sketches usando modelos Keras/TensorFlow
- 📊 **Predicción visualizada**: Muestra top-1 y top-3 predicciones en pantalla
- 💾 **Logging automático**: Registra cada inferencia con timestamp
- 🖼️ **Captura de screenshots**: Guarda predicciones en `./predictions/`

## Requisitos

### Python
- Python 3.8+

### Dependencias principales
- `opencv-python` — captura y procesamiento de video
- `tensorflow` (o `tensorflow-cpu` para CPU) — cargar y ejecutar modelos Keras
- `mediapipe` — detección de manos (recomendado)
- `numpy` — procesamiento de arrays
- `ndjson` — lectura de archivos NDJSON (si se necesita explorar datos)

### Estructura de carpeta `IA`
Debe contener los siguientes archivos:
```
IA/
├── model_info.json                          # Metadatos del modelo
├── sketch_classifier_model.keras            # Modelo Keras (preferido)
├── sketch_classifier_model.h5               # Modelo alternativo (HDF5)
├── reduced_full_simplified_ambulance.ndjson # Datos de ejemplo
└── PictionaryTrainer.ipynb                  # Notebook de referencia (no se ejecuta)
```

#### Contenido de `model_info.json`
```json
{
  "input_shape": [28, 28, 1],           # (height, width, channels)
  "num_classes": 228,                    # Número de clases
  "classes": ["ambulance", "airplane", ...],  # Lista de etiquetas
  "test_accuracy": 0.8026,               # Accuracy del modelo
  "image_size": 28
}
```

#### Formato NDJSON
Cada línea es un JSON con un sketch:
```json
{
  "word": "ambulance",
  "drawing": [[[x1, x2, ...], [y1, y2, ...]], ...],  # Trazos (lista de lista de coordenadas)
  "recognized": true,
  "countrycode": "NL",
  ...
}
```

## Instalación

### 1. Clonar o descargar el repositorio
```bash
cd tu_repo
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# En Windows
venv\Scripts\activate
# En Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

O instalar manualmente:
```bash
pip install opencv-python tensorflow mediapipe numpy ndjson
```

**Nota sobre TensorFlow:**
- Para GPU: `pip install tensorflow` (requiere CUDA/cuDNN)
- Para CPU: `pip install tensorflow-cpu`

## Uso

### Ejecución básica
```bash
python src/pictionary_live.py --ia-dir ./IA
```

### Con opciones
```bash
# Habilitar logging DEBUG
python src/pictionary_live.py --ia-dir ./IA --debug

# Usar cámara 1 en lugar de 0
python src/pictionary_live.py --ia-dir ./IA --camera-id 1

# Validar modelo sin abrir cámara (dry-run)
python src/pictionary_live.py --ia-dir ./IA --dry-run
```

### Controles en vivo
- **Dibujar**: Levanta la mano y mueve el dedo índice en el aire
- **Pausa detección**: El trazo se clasifica automáticamente cuando paras ~200ms
- **`s`** — Guardar frame actual + predicción en `./predictions/`
- **`q`** — Salir

## Cómo funciona

1. **Inicialización**
   - Lee `model_info.json` (metadatos: tamaño, etiquetas)
   - Carga modelo desde `*.keras` (preferido) o `*.h5`

2. **Captura y detección**
   - Abre cámara web
   - Detecta landmarks de la mano con MediaPipe
   - Rastrea el dedo índice (landmark 8)

3. **Acumulación de trazo**
   - Almacena puntos (x, y) normalizados mientras detecta movimiento
   - Cuando detecta pausa (200ms sin movimiento), dispara inferencia

4. **Preprocesado**
   - Normaliza puntos a canvas 28×28
   - Dibuja trazo (líneas anti-aliased)
   - Normaliza valores a [0, 1]
   - Reshape a (28, 28, 1) para el modelo

5. **Inferencia**
   - Ejecuta `model.predict()`
   - Obtiene top-1 y top-3 predicciones
   - Muestra en overlay del video

6. **Logging**
   - Guarda cada predicción en `./inference.log`
   - Formato: `timestamp | etiqueta (prob%) | Top-3: ...`

## Salida

### Logs
- `logs/pictionary_YYYYMMDD_HHMMSS.log` — Log completo de ejecución (DEBUG/INFO)
- `inference.log` — Log de inferencias (timestamp, etiqueta, probabilidad)

### Capturas
- `predictions/frame_YYYYMMDD_HHMMSS_ffffff.png` — Frames guardados con `s`

## Arquitectura interna

```
PictionaryLive (aplicación principal)
├── ModelLoader         → Carga modelo y metadatos
├── HandTracker         → Detección de manos (MediaPipe)
├── StrokeAccumulator   → Acumula puntos del trazo
├── DrawingPreprocessor → Convierte trazo a imagen 28×28
└── [OpenCV UI]         → Renderizado en pantalla
```

## Dependencias opcionales

### MediaPipe (recomendado)
Para mejor detección de manos:
```bash
pip install mediapipe
```

Si no está disponible, el código intenta usar detección de contornos (fallback menos preciso).

## Troubleshooting

### "TensorFlow no está instalado"
```bash
pip install tensorflow
# O para CPU:
pip install tensorflow-cpu
```

### "Cámara no se abre"
- Verifica que no esté en uso por otra aplicación
- Intenta con `--camera-id 1` o mayor
- En Linux, asegúrate de tener permisos: `sudo usermod -a -G video $USER`

### "MediaPipe no disponible"
```bash
pip install mediapipe
```
Sin MediaPipe, el sistema usa fallback de movimiento (menos preciso).

### "model_info.json no encontrado"
- Verifica que la carpeta IA existe y está en la ruta correcta
- Usa: `python pictionary_live.py --ia-dir /ruta/a/IA`

### Bajo rendimiento en GPU
- Asegúrate de que TensorFlow detecta GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- En Windows, verifica CUDA/cuDNN

## Ejemplos de uso

### Juego simple
```bash
cd e:\IA
python src\pictionary_live.py --ia-dir .\IA
```

### Debugging
```bash
python src\pictionary_live.py --ia-dir ./IA --debug
# Verifica logs en logs/pictionary_*.log
```

### Validar configuración antes de jugar
```bash
python src\pictionary_live.py --ia-dir ./IA --dry-run
# Muestra: modelo cargado, clases disponibles, etc.
```

## Notas técnicas

### Preprocesado
- Puntos de entrada: normalizados a [0, 1] desde landmarks de MediaPipe
- Canvas: 28×28 (blanco = 255, trazo = 0)
- Normalización: [0, 255] → [0, 1]
- Shape final: (28, 28, 1) para modelo CNN

### Detección de pausa
- Umbral: 200ms sin nuevos puntos
- Mínimo de puntos: 5 (para evitar ruido)

### Modelo
- Entrada: (28, 28, 1) — escala de grises
- Salida: 228 clases (Quick, Draw! dataset)
- Accuracy: ~80.3%

## Licencia

Interno — Proyecto IA

## Autor

Generado con Copilot

---

¿Preguntas? Revisa los logs en `logs/` para más detalles.
