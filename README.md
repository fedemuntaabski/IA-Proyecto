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

- ✅ **GPU Acceleration**: Aceleración automática con TensorFlow
- ✅ **Async Processing**: Clasificación en segundo plano
- ✅ **Analytics Framework**: Métricas de uso y rendimiento
- ✅ **UI Mejorada**: Interfaz intuitiva y moderna
- ✅ **Multi-idioma**: Español e inglés con detección automática
- ✅ **Testing Framework**: Cobertura completa de funcionalidades

## 🎯 Mejoras Futuras

Ver [`IMPROVEMENTS.md`](IMPROVEMENTS.md) para funcionalidades planificadas como MediaPipe integration, model quantization, y sistema de feedback.

## 📄 Licencia

Proyecto educativo - uso libre para fines de aprendizaje.