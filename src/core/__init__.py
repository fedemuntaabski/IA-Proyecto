"""
Core components del proyecto.
"""
from .detection import HandDetector, AdvancedVisionProcessor, BackgroundSubtractionMethod, OpticalFlowMethod
from .classification import GestureProcessor, SketchClassifier
from .config import (ConfigManager, DetectionConfig, UIConfig, MLConfig, PerformanceConfig,
                     CalibrationManager, CalibrationUI)
from .ui import UIManager
from .utils import FPSCounter, calculate_average, clamp
from .application_controller import ApplicationController
from .frame_processor import FrameProcessor
from .camera_manager import CameraManager
from .i18n import i18n, _

__all__ = ['HandDetector', 'AdvancedVisionProcessor', 'BackgroundSubtractionMethod', 'OpticalFlowMethod',
           'GestureProcessor', 'SketchClassifier', 'ConfigManager', 'DetectionConfig', 'UIConfig',
           'MLConfig', 'PerformanceConfig', 'CalibrationManager', 'CalibrationUI', 'UIManager',
           'FPSCounter', 'calculate_average', 'clamp', 'ApplicationController', 'FrameProcessor',
           'CameraManager', 'i18n', '_']
