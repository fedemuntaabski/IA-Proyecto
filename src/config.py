"""
config.py - Configuración centralizada para Pictionary Live (LEGACY)
Este archivo mantiene compatibilidad con el código existente mientras usa el nuevo sistema de configuración.
"""

from config_manager import get_config

# Obtener configuración validada
app_config = get_config()

# ============================================================================
# CONFIGURACIÓN DE MEDIAPIPE
# ============================================================================

MEDIAPIPE_CONFIG = {
    "hands": app_config.mediapipe.hands.model_dump(),
    "pose": app_config.mediapipe.pose.model_dump()
}

# ============================================================================
# CONFIGURACIÓN DE CÁMARA
# ============================================================================

CAMERA_CONFIG = app_config.camera.model_dump()

# ============================================================================
# CONFIGURACIÓN DE TRAZO
# ============================================================================

STROKE_CONFIG = app_config.stroke.model_dump()

# ============================================================================
# CONFIGURACIÓN DE MODELO
# ============================================================================

MODEL_CONFIG = app_config.model.model_dump()

# ============================================================================
# CONFIGURACIÓN DE PREPROCESAMIENTO
# ============================================================================

PREPROCESSING_CONFIG = app_config.preprocessing.model_dump()

# ============================================================================
# CONFIGURACIÓN DE UI
# ============================================================================

UI_CONFIG = app_config.ui.model_dump()

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

LOGGING_CONFIG = app_config.logging.model_dump()

# ============================================================================
# CONFIGURACIÓN DE DETECCIÓN
# ============================================================================

DETECTION_CONFIG = app_config.detection.model_dump()

# ============================================================================
# CONFIGURACIÓN DE RENDIMIENTO
# ============================================================================

PERFORMANCE_CONFIG = app_config.performance.model_dump()
