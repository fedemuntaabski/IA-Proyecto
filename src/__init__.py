"""
Módulo principal del proyecto de clasificador de sketches con detección de gestos.
"""

__version__ = "1.0.0"

# Exponer clases principales para facilitar imports
from .core.application_controller import ApplicationController
from .core.camera_manager import CameraManager
from .core.hand_detector import HandDetector
from .core.gesture_processor import GestureProcessor
from .core.classifier import SketchClassifier
from .core.config_manager import ConfigManager
from .core.frame_processor import FrameProcessor
from .core.calibration_manager import CalibrationManager
from .ui.ui_manager import UIManager
from .core.i18n import i18n, _
