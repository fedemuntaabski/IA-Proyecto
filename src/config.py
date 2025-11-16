"""
config.py - Configuración centralizada para Pictionary Live
"""

# ============================================================================
# CONFIGURACIÓN DE MEDIAPIPE
# ============================================================================

MEDIAPIPE_CONFIG = {
    # Detección de manos
    "hands": {
        "static_image_mode": False,
        "max_num_hands": 2,
        "model_complexity": 0,  # 0=ligero, 1=medio, 2=pesado (para mejor FPS)
        "min_detection_confidence": 0.2,  # Muy bajo para máxima detección
        "min_tracking_confidence": 0.2,   # Muy bajo para mejor seguimiento
    },
    # Detección de pose (DESACTIVADA por performance)
    "pose": {
        "enabled": False,  # Desactivar pose para mejorar FPS
    }
}

# ============================================================================
# CONFIGURACIÓN DE CÁMARA
# ============================================================================

CAMERA_CONFIG = {
    "width": 640,              # Reducido de 1280 para mejor FPS
    "height": 480,             # Reducido de 720 para mejor FPS
    "fps": 30,
    "buffer_size": 1,          # Minimizar latencia (un frame en buffer)
    "flip_horizontal": True,   # Espejo (más natural)
}

# ============================================================================
# CONFIGURACIÓN DE TRAZO
# ============================================================================

STROKE_CONFIG = {
    "pause_threshold_ms": 400,     # Tiempo de pausa para completar trazo
    "velocity_threshold": 0.002,   # Umbral mínimo de velocidad (más sensible)
    "min_points": 8,               # Puntos mínimos para validar trazo
    "max_stroke_age_ms": 3000,     # Máximo tiempo para un trazo activo
}

# ============================================================================
# CONFIGURACIÓN DE MODELO
# ============================================================================

MODEL_CONFIG = {
    "input_shape": [28, 28, 1],
    "demo_mode": True,             # Funcionar sin modelo TensorFlow
}

# ============================================================================
# CONFIGURACIÓN DE UI
# ============================================================================

UI_CONFIG = {
    "window_name": "Pictionary Live - Dibuja en el aire",
    "window_width": 1280,
    "window_height": 960,
    "show_fps": True,
    "show_diagnostics": True,
    "show_top_predictions": 3,
}

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

LOGGING_CONFIG = {
    "log_dir": "./logs",
    "inference_log_file": "./inference.log",
    "level_debug": "DEBUG",
    "level_info": "INFO",
}

# ============================================================================
# CONFIGURACIÓN DE DETECCIÓN
# ============================================================================

DETECTION_CONFIG = {
    "hand_index_finger_id": 8,     # ID del dedo índice en landmarks
    "hand_landmark_count": 21,     # Total de landmarks en mano
}

# ============================================================================
# CONFIGURACIÓN DE RENDIMIENTO
# ============================================================================

PERFORMANCE_CONFIG = {
    "skip_frames": 0,              # No saltar frames (mantener todos)
    "thread_workers": 1,           # Una CPU thread para detectar
    "enable_profiling": False,     # Desactivar profiling por defecto
}
