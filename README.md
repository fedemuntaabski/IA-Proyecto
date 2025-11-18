# Pictionary Live 🎨

Aplicación Python interactiva para jugar **Pictionary en vivo** usando detección de gestos con las manos y clasificación de sketches con IA.

## 🎉 Nueva Interfaz PyQt6 Moderna

La aplicación ahora cuenta con una **interfaz gráfica profesional con PyQt6**:
- Diseño moderno y responsivo
- Mejor rendimiento (30-60 FPS)
- Tema cyberpunk personalizable
- Widgets interactivos avanzados

Ver [README_PYQT6.md](README_PYQT6.md) para documentación completa.

## Características

- 🎮 **UI PyQt6 Moderna**: Interfaz profesional con alto rendimiento
- 🎥 Captura en tiempo real desde cámara web
- ✋ Detección de manos con MediaPipe
- ✍️ Acumulación de trazos en el aire
- 🤖 Clasificación de sketches con TensorFlow/Keras
- 📊 Visualización de predicciones en tiempo real
- 🎨 Múltiples temas (Cyberpunk, Light, Dark)
- 🏆 Sistema de puntuación y rachas

## 🚀 Inicio Rápido

```bash
# Iniciar la aplicación (instala dependencias automáticamente)
python main.py

# Con opciones personalizadas
python main.py --camera 1 --theme dark --debug
```

## Requisitos

- Python 3.10, 3.11 o 3.12
- PyQt6 >= 6.5.0 (se instala automáticamente)
- Dependencias: `opencv-python`, `tensorflow`, `mediapipe`, `numpy`
- Carpeta `IA/` con `model_info.json`, modelo `.keras` o `.h5`

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/fedemuntaabski/IA-Proyecto.git
cd IA-Proyecto

# Las dependencias se instalan automáticamente en la primera ejecución
python main.py
```

## Configuración

La aplicación usa un sistema de configuración flexible basado en `config.yaml`. Este archivo permite personalizar todos los aspectos de la aplicación sin modificar código.

### Archivo de configuración

Copia y modifica `config.yaml` según tus necesidades:

```yaml
# Ejemplo de configuración personalizada
camera:
  width: 1280
  height: 720
  fps: 60

model:
  demo_mode: false  # Cambiar a false si tienes modelo entrenado

performance:
  async_processing: true
```

También puedes usar `config.example.yaml` como base para configuraciones específicas de desarrollo o producción.

### Validación de configuración

La configuración se valida automáticamente al iniciar la aplicación. Si hay errores, se mostrarán mensajes detallados con información sobre qué valores son inválidos.

## Testing

El proyecto incluye una suite completa de pruebas unitarias y de integración.

### Ejecutar pruebas
```bash
# Desde la raíz del proyecto
python src/run_tests.py
```

### Ejecutar pruebas manualmente
```bash
# Instalar dependencias de testing
pip install pytest pytest-mock pytest-cov

# Ejecutar todas las pruebas
pytest src/tests/ --cov=src --cov-report=html

# Ejecutar pruebas específicas
pytest src/tests/test_model.py -v
```

### Cobertura de pruebas
- ✅ **SketchClassifier**: Carga de modelo, predicciones, modo demo
- ✅ **HandDetector**: Detección de manos, cálculo de velocidad, dibujo
- ✅ **StrokeAccumulator**: Acumulación de trazos, detección de pausas
- ✅ **PictionaryLive**: Integración completa, validación de setup

## Uso

### Ejecución básica
```bash
python main.py
```

### Opciones disponibles
```bash
python main.py --theme light     # Tema claro
python main.py --theme dark      # Tema oscuro
python main.py --camera-id 1     # Cambiar cámara
python main.py --debug           # Logging detallado
```

### Controles del juego
- Dibuja en el aire con el dedo índice
- `Enter` — Predecir dibujo
- `Escape` — Salir del juego

## Troubleshooting

### Problemas comunes
- **Tkinter no encontrado**: Instala `python3-tk` (Linux) o reinstala Python con Tkinter (Windows/macOS)
- **Pillow no instalado**: `pip install pillow`
- **Interfaz no responde**: Verifica que tengas display gráfico disponible
- **Cámara no abre**: Prueba `--camera-id 1`
- **Modelo no carga**: Verifica carpeta `IA/`
- **Bajo rendimiento**: Usa `tensorflow-cpu` para CPU

### Diagnóstico
- **Errores de MediaPipe**: Actualiza protobuf: `pip install --upgrade protobuf`
- **Logs detallados**: Usa `--debug` para más información

## Licencia

Interno — Proyecto IA
