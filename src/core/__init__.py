"""
Core components del proyecto.
"""
from .hand_detector import HandDetector
from .gesture_processor import GestureProcessor

# Import opcional del clasificador (solo si TensorFlow está disponible)
try:
    from .classifier import SketchClassifier
    _classifier_available = True
except ImportError:
    _classifier_available = False
    SketchClassifier = None

__all__ = ['HandDetector', 'GestureProcessor']

if _classifier_available:
    __all__.append('SketchClassifier')
