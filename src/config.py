"""
config.py - Simplified Configuration (KISS Principle)

Loads configuration directly from config.yaml without complex validation.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def _safe_get(config: Dict, key: str, default: Any) -> Any:
    """Safely get config value with default fallback."""
    return config.get(key, default) if isinstance(config, dict) else default


# Load config.yaml
_config_path = Path(__file__).parent.parent / "config.yaml"
_config = {}

try:
    if _config_path.exists():
        with open(_config_path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f) or {}
    else:
        print(f"Info: config.yaml not found at {_config_path}, using defaults")
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}, using defaults")
    _config = {}

# Default configurations
_DEFAULTS = {
    "mediapipe": {
        "hands": {
            "static_image_mode": False,
            "max_num_hands": 1,
            "model_complexity": 0,
            "min_detection_confidence": 0.7,
            "min_tracking_confidence": 0.7
        },
        "pose": {"enabled": False}
    },
    "camera": {
        "width": 640,
        "height": 640,
        "fps": 30,
        "buffer_size": 1,
        "flip_horizontal": True
    },
    "model": {
        "input_shape": [28, 28, 1],
        "demo_mode": False,
        "use_quantized_model": True,
        "prefer_gpu": True,
        "use_ensemble": 1
    },
    "ui": {
        "window_name": "Pictionary Live - Dibuja en el aire",
        "window_width": 900,
        "window_height": 900,
        "show_fps": True,
        "show_diagnostics": True,
        "show_top_predictions": 3
    },
    "detection": {
        "hand_index_finger_id": 8,
        "hand_landmark_count": 21,
        "processing_resolution": 320
    },
    "performance": {
        "skip_frames": 0,
        "thread_workers": 1,
        "enable_profiling": False,
        "async_processing": True
    }
}

# ============================================================================
# PUBLIC CONFIGURATIONS
# ============================================================================

MEDIAPIPE_CONFIG = _safe_get(_config, "mediapipe", _DEFAULTS["mediapipe"])
CAMERA_CONFIG = _safe_get(_config, "camera", _DEFAULTS["camera"])
MODEL_CONFIG = _safe_get(_config, "model", _DEFAULTS["model"])
UI_CONFIG = _safe_get(_config, "ui", _DEFAULTS["ui"])
DETECTION_CONFIG = _safe_get(_config, "detection", _DEFAULTS["detection"])
PERFORMANCE_CONFIG = _safe_get(_config, "performance", _DEFAULTS["performance"])
