"""
config.py - Simplified Configuration (KISS Principle)

Loads configuration directly from config.yaml without complex validation.
"""

import yaml
from pathlib import Path

# Load config.yaml
_config_path = Path(__file__).parent.parent / "config.yaml"

try:
    with open(_config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    _config = {}

# ============================================================================
# MEDIAPIPE CONFIGURATION
# ============================================================================

MEDIAPIPE_CONFIG = _config.get("mediapipe", {
    "hands": {
        "static_image_mode": False,
        "max_num_hands": 1,
        "model_complexity": 0,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.7
    },
    "pose": {"enabled": False}
})

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

CAMERA_CONFIG = _config.get("camera", {
    "width": 640,
    "height": 480,
    "fps": 30,
    "buffer_size": 1,
    "flip_horizontal": True
})

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = _config.get("model", {
    "input_shape": [28, 28, 1],
    "demo_mode": False,
    "use_quantized_model": True,
    "prefer_gpu": True,
    "use_ensemble": 1
})

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = _config.get("ui", {
    "window_name": "Pictionary Live - Dibuja en el aire",
    "window_width": 900,
    "window_height": 675,
    "show_fps": True,
    "show_diagnostics": True,
    "show_top_predictions": 3
})

# ============================================================================
# DETECTION CONFIGURATION
# ============================================================================

DETECTION_CONFIG = _config.get("detection", {
    "hand_index_finger_id": 8,
    "hand_landmark_count": 21,
    "processing_resolution": 320
})

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

PERFORMANCE_CONFIG = _config.get("performance", {
    "skip_frames": 0,
    "thread_workers": 1,
    "enable_profiling": False,
    "async_processing": True
})
