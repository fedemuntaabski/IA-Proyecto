"""
Core components del proyecto.
"""
from .hand_detector import HandDetector
from .gesture_processor import GestureProcessor
from .calibration_manager import CalibrationManager, CalibrationUI
from .config_manager import ConfigManager, DetectionConfig, UIConfig, MLConfig, PerformanceConfig
from .advanced_vision import AdvancedVisionProcessor, BackgroundSubtractionMethod, OpticalFlowMethod

# Import opcional del clasificador (solo si TensorFlow está disponible)
try:
    from .classifier import SketchClassifier
    _classifier_available = True
except ImportError:
    _classifier_available = False
    SketchClassifier = None

__all__ = ['HandDetector', 'GestureProcessor', 'CalibrationManager', 'CalibrationUI',
           'ConfigManager', 'DetectionConfig', 'UIConfig', 'MLConfig', 'PerformanceConfig',
           'AdvancedVisionProcessor', 'BackgroundSubtractionMethod', 'OpticalFlowMethod']

if _classifier_available:
    __all__.append('SketchClassifier')
