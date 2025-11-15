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

### 2. Air Draw Classifier Completo (air_draw_classifier.py)
Versión avanzada con todas las funcionalidades:
- Sistema de calibración interactiva
- Múltiples perfiles de configuración
- Interfaz completa con información detallada
- Configuración avanzada de detección y ML

## Archivos incluidos

- `PictionaryTrainer.ipynb`: Notebook completo de entrenamiento
- `sketch_classifier_model.h5`: Modelo entrenado (formato HDF5)
- `sketch_classifier_model.keras`: Modelo entrenado (formato Keras moderno)
- `model_info.json`: Metadatos del modelo (clases, precisión, parámetros)
- `air_draw_classifier.py`: Aplicación completa de dibujo en el aire
- `IMPROVEMENTS.md`: Documento con mejoras futuras planificadas
- `src/core/`: Módulos principales del sistema
  - `hand_detector.py`: Detector de manos con técnicas avanzadas
  - `gesture_processor.py`: Procesador de gestos
  - `classifier.py`: Clasificador con sistema de fallback
  - `config_manager.py`: Sistema de configuración avanzada
  - `calibration_manager.py`: Sistema de calibración
  - `advanced_vision.py`: Procesamiento avanzado de visión

## Requisitos

- Python 3.x
- TensorFlow/Keras (opcional - funciona con fallback)
- OpenCV
- NumPy, Matplotlib, PIL
- MediaPipe (opcional para funcionalidades avanzadas)

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar la aplicación principal:
```bash
python main.py
```

## Controles

### Air Draw Classifier (Principal)
- `SPACE` - Forzar nueva clasificación
- `r` - Limpiar dibujo actual
- `q` - Salir

### Air Draw Classifier (Completo)
- `r` - Resetear dibujo
- `c` - Recalibrar sistema
- `p` - Cambiar perfil de configuración
- `q` - Salir
- `SPACE` - Procesar dibujo actual

## Dataset

El proyecto utiliza el dataset "Quick, Draw!" de Google (no incluido en este repositorio).

## Arquitectura del Sistema

El sistema está organizado en módulos reutilizables:

- **Detección**: HandDetector con segmentación por color y técnicas avanzadas
- **Procesamiento**: GestureProcessor convierte movimientos en imágenes 28x28
- **Clasificación**: SketchClassifier con TensorFlow y fallback heurístico
- **Configuración**: ConfigManager para perfiles de usuario
- **Calibración**: CalibrationManager para ajuste automático

## Mejores Prácticas Implementadas

- **Modularidad**: Componentes independientes y reutilizables
- **Fallback**: Sistema funciona sin TensorFlow
- **Configuración**: Perfiles personalizables
- **Estabilidad**: Filtros temporales para detección robusta
- **Performance**: Procesamiento optimizado para tiempo real

## Mejoras Futuras

Para ver las mejoras planificadas y funcionalidades futuras, consulta el archivo [`IMPROVEMENTS.md`](IMPROVEMENTS.md).

## Licencia

Proyecto educativo - uso libre para fines de aprendizaje.