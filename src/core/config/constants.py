"""
Constantes para el módulo de configuración.

Este archivo contiene todas las constantes utilizadas en la gestión
de configuración y perfiles de usuario.
"""

from pathlib import Path

# Configuración de perfiles
DEFAULT_PROFILE_NAME = "default"
PROFILE_CONFIG_DIR = "config"
ACTIVE_PROFILE_FILE = "active_profile.txt"

# Archivos de configuración
PROFILE_FILE_PREFIX = "profile_"
PROFILE_FILE_EXTENSION = ".json"

# Configuración de calibración
CALIBRATION_SAMPLES_REQUIRED = 50
CALIBRATION_SAMPLES_SKIN = 3
CALIBRATION_SAMPLES_BACKGROUND = 2
CALIBRATION_TIMEOUT_SECONDS = 30
CALIBRATION_CONFIG_FILE = "calibration_config.json"

# Configuración de UI por defecto
DEFAULT_UI_THEME = "dark"
DEFAULT_FONT_SCALE = 0.6
DEFAULT_BAR_HEIGHT = 40

# Configuración de detección por defecto
DEFAULT_MIN_HAND_AREA = 5000
DEFAULT_MAX_HAND_AREA = 50000
DEFAULT_STABILITY_THRESHOLD = 3

# Configuración de ML por defecto
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MODEL_PATH = "IA/sketch_classifier_model.keras"
DEFAULT_MODEL_INFO_PATH = "IA/model_info.json"

# Configuración de performance por defecto
DEFAULT_TARGET_FPS = 30
DEFAULT_RESOLUTION = (640, 480)