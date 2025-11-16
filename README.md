# IA Proyecto - Clasificador de Sketches para Pictionary

Clasificador de sketches basado en deep learning para un entrenador de IA en el juego Pictionary, con aplicación completa de dibujo en el aire.

## 🚀 Características

- **Clasificación en tiempo real**: 228+ clases usando CNN entrenada con Quick Draw dataset
- **Detección de manos**: Procesamiento avanzado con background subtraction y optical flow
- **Interfaz intuitiva**: UI mejorada con feedback visual, tooltips y controles contextuales
- **Multi-idioma**: Soporte completo para español e inglés con detección automática
- **Performance optimizada**: Procesamiento asíncrono, aceleración GPU y monitoring de FPS
- **Sistema robusto**: Fallback automático cuando TensorFlow no está disponible
- **Testing completo**: Framework de pruebas con cobertura unitaria e integración
- **✨ Sensibilidad Adaptativa**: Ajuste automático de thresholds según condiciones ambientales
- **✨ Compensación de Iluminación**: Normalización automática en diferentes condiciones de luz
- **✨ Monitor de Diagnóstico**: Chequeos de salud en tiempo real y recomendaciones
- **✨ Optimización de ROI**: Detección dinámica del área de búsqueda para mayor eficiencia
- **✨ Análisis de Calidad de Frame**: Optimización automática de resolución según FPS disponible

## 📦 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Compilar traducciones (opcional)
python compile_translations.py
```

## 🎮 Uso

```bash
# Aplicación principal
python main.py

# Ejecutar tests
python tests/test_runner.py
```

## 🎯 Controles

- `SPACE` - Forzar clasificación
- `r` - Limpiar dibujo
- `h` - Mostrar/ocultar ayuda
- `d` - Mostrar diagnóstico del sistema
- `q` - Salir

## 📁 Estructura del Proyecto

```
├── main.py                 # Aplicación principal
├── air_draw_classifier.py  # Versión simplificada
├── PictionaryTrainer.ipynb # Notebook de entrenamiento
├── src/core/              # Módulos principales
│   ├── detection/         # Detección de manos
│   ├── classification/    # Clasificación de sketches
│   ├── ui/               # Interfaz de usuario
│   └── utils/            # Utilidades (GPU, async, analytics)
├── tests/                # Framework de testing
├── locale/               # Traducciones
└── IA/                   # Modelos y datos
```

## 🔧 Requisitos

- Python 3.8+
- TensorFlow 2.10+ (opcional - funciona con fallback)
- OpenCV, NumPy, MediaPipe
- psutil (para monitoring del sistema)

## 📈 Mejoras Implementadas

### Detección y Iluminación
- ✅ **Compensación Automática de Iluminación**: Análisis de histograma por regiones, corrección gamma y CLAHE
- ✅ **Detección y Mitigación de Sombras**: Identificación automática de áreas sombreadas
- ✅ **Rangos HSV Adaptativos**: Ajuste dinámico según condiciones de luz

### Sensibilidad y Precisión
- ✅ **Sensibilidad Adaptativa**: Ajuste automático basado en calidad de frame, ruido y rendimiento
- ✅ **Análisis de Ruido**: Detección de ruido ambiental para mejorar detección
- ✅ **Estabilidad Multi-Frame**: Buffer circular y filtrado temporal para contornos estables

### Rendimiento y Optimización
- ✅ **Optimización de ROI**: Detección dinámica del área de búsqueda (Region of Interest)
- ✅ **Optimización de Resolución**: Ajuste automático de calidad según FPS disponible
- ✅ **GPU Acceleration**: Aceleración automática con TensorFlow
- ✅ **Procesamiento Asíncrono**: Clasificación en segundo plano sin bloqueos

### Monitoreo y Diagnóstico
- ✅ **Monitor de Diagnóstico**: Chequeos de salud del sistema en tiempo real
- ✅ **Health Check Completo**: Verificación de Python, dependencias, cámara, disco, memoria y permisos
- ✅ **Recomendaciones Dinámicas**: Sugerencias de optimización basadas en condiciones actuales
- ✅ **Análisis de Calidad de Frame**: Métricas de nitidez y contraste

### Gestos y Tracking
- ✅ **Análisis Avanzado de Gestos**: Tracking multi-mano con estados estables
- ✅ **Análisis de Movimiento**: Detección de velocidad, dirección y estabilidad de gestos
- ✅ **Análisis de Estabilidad de Contornos**: Seguimiento histórico para mejor precisión

### Configuración y Usabilidad
- ✅ **Configuración Avanzada**: Sistema de settings con validación y perfiles
- ✅ **Bootstrap del Sistema**: Inicialización automática con chequeos integrados
- ✅ **Diagnosticador en Tiempo Real**: Presionar 'D' para ver estado del sistema

## 🎯 Mejoras Futuras

Ver [`IMPROVEMENTS.md`](IMPROVEMENTS.md) para funcionalidades planificadas como MediaPipe integration, model quantization, y sistema de feedback.

## 📄 Licencia

Proyecto educativo - uso libre para fines de aprendizaje.