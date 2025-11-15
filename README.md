# IA Proyecto - Clasificador de Sketches para Pictionary

Este proyecto implementa un clasificador de sketches basado en deep learning para un entrenador de IA en el juego Pictionary, incluyendo una aplicación completa de dibujo en el aire.

## Descripción

El sistema utiliza una red neuronal convolucional (CNN) entrenada con el dataset Quick Draw de Google para reconocer dibujos en tiempo real. El modelo puede identificar 228 clases diferentes de objetos, desde "The Eiffel Tower" hasta "zebra".

## Características

- **Arquitectura**: CNN profunda con 4 bloques convolucionales
- **Precisión**: 80.26% en conjunto de prueba
- **Eficiencia**: Generadores de datos para manejo óptimo de memoria
- **Análisis progresivo**: Evalúa cuánto del dibujo necesita la IA para adivinar correctamente
- **Aplicación completa**: Detección de manos en tiempo real y dibujo en el aire
- **Mini-app simplificada**: Versión básica para uso rápido
- **Interfaz Mejorada**: UI intuitiva con tooltips, feedback visual y controles contextuales
- **Multi-idioma**: Soporte completo para español e inglés con detección automática
- **Performance Optimizada**: Procesamiento asíncrono, aceleración GPU y monitoring de FPS
- **Testing Framework**: Cobertura completa con tests unitarios e integración

## Aplicaciones Disponibles

### 1. Air Draw Classifier (Principal - main.py)
Aplicación completa para clasificación de dibujos en el aire:
- Detección de manos con background subtraction y optical flow
- Procesamiento de gestos en tiempo real
- Clasificación automática con estadísticas de sesión
- Interfaz intuitiva con controles simples

**Características principales:**
- Activación automática de cámara
- Detección robusta de movimientos de mano
- Interpretación de trazos en el espacio
- Clasificación de 228+ clases de objetos
- Estadísticas de rendimiento en tiempo real
- Sistema de fallback cuando TensorFlow no está disponible

## Archivos incluidos

- `PictionaryTrainer.ipynb`: Notebook completo de entrenamiento
- `sketch_classifier_model.h5`: Modelo entrenado (formato HDF5)
- `sketch_classifier_model.keras`: Modelo entrenado (formato Keras moderno)
- `model_info.json`: Metadatos del modelo (clases, precisión, parámetros)
- `IMPROVEMENTS.md`: Documento con mejoras futuras planificadas
- `src/core/`: Módulos principales del sistema
  - `hand_detector.py`: Detector de manos con técnicas avanzadas
  - `gesture_processor.py`: Procesador de gestos
  - `classifier.py`: Clasificador con sistema de fallback
  - `config_manager.py`: Sistema de configuración avanzada
  - `calibration_manager.py`: Sistema de calibración
  - `advanced_vision.py`: Procesamiento avanzado de visión

## Nuevas Funcionalidades (v2.0)

### 🎨 Interfaz de Usuario Mejorada
- **Feedback Visual Avanzado**: Indicadores de estado, barras de progreso y animaciones
- **Panel de Ayuda Contextual**: Información integrada que se puede mostrar/ocultar
- **Monitoring en Tiempo Real**: FPS, estadísticas de sesión y métricas de rendimiento
- **Tema Mejorado**: Colores consistentes y diseño profesional

### 🌐 Soporte Multi-Idioma
- **Detección Automática**: Detecta automáticamente el idioma del sistema
- **Traducción Completa**: UI, mensajes y nombres de clases en español e inglés
- **Sistema Extensible**: Fácil agregar nuevos idiomas
- **Fallback Robusto**: Funciona correctamente si faltan traducciones

### ⚡ Optimizaciones de Performance
- **Procesamiento Asíncrono**: Clasificación en segundo plano sin bloquear la UI
- **Aceleración GPU**: Configuración automática de CUDA/cuDNN cuando disponible
- **Memory Management**: Gestión optimizada de memoria y recursos
- **Monitoring de FPS**: Visualización en tiempo real del rendimiento

### 🧪 Framework de Testing
- **Cobertura Completa**: Tests para todos los módulos principales
- **Test Runner Simple**: Ejecutable sin dependencias externas
- **Integración Continua**: Preparado para CI/CD
- **Tests de Integración**: Validación del pipeline completo

## Requisitos

- Python 3.x
- TensorFlow/Keras (opcional - funciona con fallback)
- OpenCV
- NumPy, Matplotlib, PIL
- MediaPipe (opcional para funcionalidades avanzadas)
- polib (para compilación de traducciones)

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Compilar traducciones (opcional):
```bash
python compile_translations.py
```

3. Ejecutar la aplicación principal:
```bash
python main.py
```

4. Ejecutar tests (opcional):
```bash
python test_runner.py
# o con pytest
pytest
```

## Controles

### Air Draw Classifier (Principal)
- `SPACE` - Forzar nueva clasificación
- `r` - Limpiar dibujo actual
- `h` - Mostrar/ocultar panel de ayuda
- `q` - Salir

## Dataset

El proyecto utiliza el dataset "Quick, Draw!" de Google (no incluido en este repositorio).

## Arquitectura del Sistema

El sistema está organizado en módulos reutilizables:

- **Detección**: HandDetector con segmentación por color y técnicas avanzadas
- **Procesamiento**: GestureProcessor convierte movimientos en imágenes 28x28
- **Clasificación**: SketchClassifier con TensorFlow y fallback heurístico
- **Configuración**: ConfigManager para perfiles de usuario
- **Calibración**: CalibrationManager para ajuste automático

## Testing

El proyecto incluye un framework de testing completo:

### Ejecutar Tests

**Opción 1: Con pytest (recomendado)**
```bash
pip install pytest
pytest
```

**Opción 2: Test runner simple**
```bash
python test_runner.py
```

### Cobertura de Tests

- **Internacionalización**: Sistema de traducción multi-idioma
- **Configuración**: Gestión de perfiles de usuario
- **Procesamiento de Gestos**: Conversión de movimientos a imágenes
- **Clasificación**: Sistema de IA con fallback
- **Detección de Manos**: Procesamiento de visión computacional
- **Integración**: Pipeline completo de procesamiento

## Mejores Prácticas Implementadas

- **Modularidad**: Componentes independientes y reutilizables
- **Fallback**: Sistema funciona sin TensorFlow
- **Configuración**: Perfiles personalizables
- **Estabilidad**: Filtros temporales para detección robusta
- **Performance**: Procesamiento optimizado para tiempo real
- **Internacionalización**: Soporte multi-idioma
- **Testing**: Cobertura completa de funcionalidades
- **Async Processing**: Clasificación en segundo plano

## Mejoras Futuras

Para ver las mejoras planificadas y funcionalidades futuras, consulta el archivo [`IMPROVEMENTS.md`](IMPROVEMENTS.md).

## Licencia

Proyecto educativo - uso libre para fines de aprendizaje.