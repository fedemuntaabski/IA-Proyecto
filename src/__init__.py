"""
Módulo principal del proyecto de clasificador de sketches con detección de gestos.
"""

__version__ = "1.0.0"

# Exponer clases principales para facilitar imports
from .core.application_controller import ApplicationController
from .core.camera_manager import CameraManager
from .core.detection.hand_detector import HandDetector
from .core.classification.gesture_processor import GestureProcessor
from .core.classification.classifier import SketchClassifier
from .core.config.config_manager import ConfigManager
from .core.frame_processor import FrameProcessor
from .core.config.calibration_manager import CalibrationManager
from .core.ui.ui_manager import UIManager
from .core.i18n import i18n, _
