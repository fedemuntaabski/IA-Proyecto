"""
Core components del proyecto.
"""
from .hand_detector import HandDetector
from .gesture_processor import GestureProcessor
from .classifier import SketchClassifier

__all__ = ['HandDetector', 'GestureProcessor', 'SketchClassifier']
