# Pictionary Live 🎨

Aplicación Python interactiva para jugar **Pictionary en vivo** usando detección de gestos con las manos y clasificación de sketches con IA.

## Características

- 🎥 Captura en tiempo real desde cámara web
- ✋ Detección de manos con MediaPipe
- ✍️ Acumulación de trazos en el aire
- 🤖 Clasificación de sketches con TensorFlow/Keras
- 📊 Visualización de predicciones en pantalla

## Requisitos

- Python 3.10+
- Dependencias: `opencv-python`, `tensorflow`, `mediapipe`, `numpy`
- Carpeta `IA/` con `model_info.json`, modelo `.keras` o `.h5`, y datos opcionales

## Instalación

1. Clona o descarga el repositorio
2. Crea entorno virtual: `python -m venv venv` y activa
3. Instala dependencias: `pip install -r src/requirements.txt`

**Nota**: Las dependencias han sido actualizadas para resolver warnings de deprecación y mejorar compatibilidad.

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

### Opciones
```bash
python main.py --debug          # Logging detallado
python main.py --camera-id 1    # Cambiar cámara
python main.py --dry-run        # Validar sin abrir cámara
```

### Controles
- Dibuja en el aire con el dedo índice
- `s` — Guardar captura
- `q` — Salir

## Troubleshooting

- **Cámara no abre**: Prueba `--camera-id 1`
- **Modelo no carga**: Verifica carpeta `IA/`
- **Bajo rendimiento**: Usa `tensorflow-cpu` para CPU

## Licencia

Interno — Proyecto IA
