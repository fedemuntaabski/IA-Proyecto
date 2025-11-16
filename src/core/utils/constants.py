"""
Constantes del proyecto Air Draw Classifier.

Este módulo contiene todas las constantes utilizadas en el proyecto
para mantener consistencia y facilitar mantenimiento.
"""

# Dimensiones de imagen para el modelo de clasificación
MODEL_IMAGE_SIZE = 28

# Configuración de gestos
DEFAULT_LINE_WIDTH = 2
DEFAULT_CANVAS_SIZE = 256
MIN_POINTS_FOR_CLASSIFICATION = 10

# Configuración de aplicación
DEFAULT_RESOLUTION_WIDTH = 640
DEFAULT_RESOLUTION_HEIGHT = 480
DEFAULT_TARGET_FPS = 30
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_TOP_K_PREDICTIONS = 3

# Configuración de detección de manos
DEFAULT_MIN_HAND_AREA = 5000
DEFAULT_MAX_HAND_AREA = 50000
DEFAULT_STABILITY_THRESHOLD = 3
DEFAULT_MAX_HISTORY_SIZE = 10

# Configuración de UI
DEFAULT_FONT_SCALE = 0.6
DEFAULT_FONT_THICKNESS = 2
DEFAULT_BAR_HEIGHT = 80

# Colores BGR para UI - Tema moderno por defecto
COLOR_SUCCESS = (76, 175, 80)          # Verde moderno
COLOR_WARNING = (33, 150, 243)         # Azul moderno
COLOR_ERROR = (244, 67, 54)            # Rojo moderno
COLOR_INFO = (156, 39, 176)            # Púrpura moderno
COLOR_TEXT_PRIMARY = (33, 33, 33)      # Gris oscuro para texto
COLOR_ACCENT = (255, 152, 0)           # Naranja moderno
COLOR_BG_PRIMARY = (255, 255, 255)     # Blanco
COLOR_BG_SECONDARY = (240, 240, 240)   # Gris muy claro
COLOR_BORDER = (189, 189, 189)         # Gris medio

# Configuración de GPU
GPU_MEMORY_GROWTH = True
GPU_INTRA_OP_THREADS = 4
GPU_INTER_OP_THREADS = 4

# Configuración de logging (si se implementa en el futuro)
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configuración de internacionalización
DEFAULT_LANGUAGE = "es"
SUPPORTED_LANGUAGES = ["es", "en"]

# Configuración de calibración
CALIBRATION_SAMPLES_SKIN = 3
CALIBRATION_SAMPLES_BACKGROUND = 2
CALIBRATION_CONFIG_FILE = "config/calibration_config.json"

# Configuración de perfiles
DEFAULT_PROFILE_NAME = "default"
PROFILE_CONFIG_DIR = "config"
ACTIVE_PROFILE_FILE = "config/active_profile.txt"