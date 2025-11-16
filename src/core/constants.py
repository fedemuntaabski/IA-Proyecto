"""
Constants - Constantes Globales del Proyecto.

Este módulo define todas las constantes utilizadas en el proyecto
para mantener consistencia y facilitar mantenimiento.
"""

from typing import Dict, Any

# Configuración de detección
DETECTION_CONFIG = {
    'MIN_HAND_AREA': 3000,
    'MAX_HAND_AREA': 30000,
    'DEFAULT_SKIN_LOWER': [0, 20, 70],
    'DEFAULT_SKIN_UPPER': [20, 255, 255],
    'STABILITY_THRESHOLD': 3,
    'MAX_HISTORY_SIZE': 10,
    'CONTOUR_SIMILARITY_THRESHOLD': 0.7,
    'CONFIDENCE_THRESHOLD': 0.6
}

# Configuración de procesamiento de frames
FRAME_PROCESSING_CONFIG = {
    'MIN_POINTS_FOR_CLASSIFICATION': 10,
    'DEFAULT_CONFIDENCE_THRESHOLD': 0.5,
    'TARGET_FPS': 30,
    'DEFAULT_RESOLUTION_WIDTH': 640,
    'DEFAULT_RESOLUTION_HEIGHT': 480
}

# Configuración de UI
UI_CONFIG = {
    'DEFAULT_SCALE': 1.0,
    'DEFAULT_VOLUME': 0.5,
    'FPS_UPDATE_INTERVAL': 1.0
}

# Configuración de análisis de sensibilidad
SENSITIVITY_CONFIG = {
    'BASE_SENSITIVITY': 0.6,
    'FRAME_QUALITY_HISTORY_SIZE': 30,
    'NOISE_LEVEL_HISTORY_SIZE': 30,
    'FPS_HISTORY_SIZE': 30,
    'ADAPTATION_SPEED': 0.3,
    'MIN_SENSITIVITY': 0.3,
    'MAX_SENSITIVITY': 1.2
}

# Configuración de iluminación
LIGHTING_CONFIG = {
    'REGION_GRID_SIZE': 4,
    'LIGHTING_HISTORY_SIZE': 30,
    'SHADOW_RATIO_THRESHOLD': 0.4,
    'BRIGHTNESS_LOW_THRESHOLD': 85,
    'BRIGHTNESS_HIGH_THRESHOLD': 170,
    'OVEREXPOSED_THRESHOLD': 240
}

# Configuración de diagnóstico
DIAGNOSTIC_CONFIG = {
    'REPORT_INTERVAL': 10.0,
    'WARNING_THRESHOLD': 0.8,
    'CRITICAL_THRESHOLD': 0.95,
    'HEALTH_CHECK_INTERVAL': 30.0
}

# Configuración de gestos
GESTURE_CONFIG = {
    'GESTURE_WINDOW_SIZE': 10,
    'MAX_GESTURES': 5,
    'CONTOUR_BUFFER_SIZE': 5,
    'MOVEMENT_THRESHOLD': 0.5,
    'FAST_MOVEMENT_THRESHOLD': 3.0
}

# Configuración de optimización de área
AREA_OPTIMIZATION_CONFIG = {
    'ROI_PADDING': 50,
    'MIN_ROI_RATIO': 0.2,
    'FRAME_QUALITY_WINDOW': 30,
    'RESOLUTION_SCALE_STEP': 0.1,
    'MIN_FPS_THRESHOLD': 15
}

# Configuración de salud del sistema
HEALTH_CHECK_CONFIG = {
    'PYTHON_MIN_VERSION': (3, 8),
    'REQUIRED_PACKAGES': [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('tensorflow', 'tensorflow'),
        ('psutil', 'psutil')
    ],
    'MIN_DISK_SPACE_GB': 1.0,
    'MAX_MEMORY_PERCENT': 80,
    'MAX_CPU_PERCENT': 80
}

# Configuración de logging y debugging
LOGGING_CONFIG = {
    'MAX_LOG_ENTRIES': 1000,
    'LOG_LEVEL': 'INFO',
    'ENABLE_PERFORMANCE_LOGGING': True,
    'ENABLE_ERROR_LOGGING': True
}

# Configuración de archivos y rutas
PATH_CONFIG = {
    'MODEL_PATH': 'IA/sketch_classifier_model.keras',
    'MODEL_INFO_PATH': 'IA/model_info.json',
    'SETTINGS_FILE': 'config/app_settings.json',
    'FEEDBACK_FILE': 'src/core/utils/feedback_data.json',
    'LOCALE_DIR': 'locale'
}

# Configuración de internacionalización
I18N_CONFIG = {
    'DEFAULT_LANGUAGE': 'es',
    'SUPPORTED_LANGUAGES': ['es', 'en'],
    'LOCALE_DOMAIN': 'messages'
}

# Configuración de testing
TEST_CONFIG = {
    'TEST_TIMEOUT': 30,
    'COVERAGE_THRESHOLD': 80,
    'ENABLE_INTEGRATION_TESTS': True,
    'ENABLE_PERFORMANCE_TESTS': True
}
