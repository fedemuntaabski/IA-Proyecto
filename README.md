# Pictionary Live 🎨

Aplicación Python interactiva para jugar **Pictionary en vivo** usando detección de gestos con las manos y clasificación de sketches con IA.

## Características

- 🎥 Captura en tiempo real desde cámara web
- ✋ Detección de manos con MediaPipe
- ✍️ Acumulación de trazos en el aire
- 🤖 Clasificación de sketches con TensorFlow/Keras
- 📊 Visualización de predicciones en pantalla

## Requisitos

- Python 3.8+
- Dependencias: `opencv-python`, `tensorflow`, `mediapipe`, `numpy`
- Carpeta `IA/` con `model_info.json`, modelo `.keras` o `.h5`, y datos opcionales

## Instalación

1. Clona o descarga el repositorio
2. Crea entorno virtual: `python -m venv venv` y activa
3. Instala dependencias: `pip install opencv-python tensorflow mediapipe numpy`

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
