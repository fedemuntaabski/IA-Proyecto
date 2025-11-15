"""
Unit tests for the Air Draw Classifier project.

This module contains comprehensive tests for all core components.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.i18n import i18n, _, get_class_name_translation


class TestI18n:
    """Test cases for internationalization functionality."""

    def test_i18n_initialization(self):
        """Test that i18n system initializes correctly."""
        assert i18n.current_language is not None
        assert i18n.locale_dir.exists()

    def test_language_loading(self):
        """Test loading different languages."""
        # Test English
        assert i18n.load_language('en')
        assert i18n.get_current_language() == 'en'
        assert _('Mano detectada') == 'Hand detected'

        # Test Spanish
        assert i18n.load_language('es')
        assert i18n.get_current_language() == 'es'
        assert _('Mano detectada') == 'Mano detectada'

    def test_class_name_translation(self):
        """Test class name translations."""
        i18n.load_language('es')
        assert get_class_name_translation('circle') == 'círculo'
        assert get_class_name_translation('house') == 'casa'

        i18n.load_language('en')
        assert get_class_name_translation('circle') == 'circle'
        assert get_class_name_translation('house') == 'house'

    def test_fallback_translation(self):
        """Test fallback when translation is missing."""
        original_text = "NonExistentTranslationKey"
        assert _(original_text) == original_text

    def test_available_languages(self):
        """Test getting available languages."""
        languages = i18n.get_available_languages()
        assert 'en' in languages
        assert 'es' in languages


class TestConfigManager:
    """Test cases for configuration management."""

    def test_config_initialization(self):
        """Test ConfigManager initialization."""
        from core.config_manager import ConfigManager

        config_manager = ConfigManager()
        assert config_manager.current_profile is not None
        assert len(config_manager.profiles) > 0

    def test_profile_management(self):
        """Test profile creation and switching."""
        from core.config_manager import ConfigManager

        config_manager = ConfigManager()

        # Create test profile
        assert config_manager.create_profile("test_profile")

        # Switch to profile
        assert config_manager.set_active_profile("test_profile")
        assert config_manager.current_profile.name == "test_profile"

        # Clean up
        config_manager.delete_profile("test_profile")


class TestGestureProcessor:
    """Test cases for gesture processing."""

    def test_gesture_processor_initialization(self):
        """Test GestureProcessor initialization."""
        from core.gesture_processor import GestureProcessor

        processor = GestureProcessor()
        assert processor.stroke_points == []
        assert processor.image_size == 28

    def test_point_adding(self):
        """Test adding points to gesture."""
        from core.gesture_processor import GestureProcessor

        processor = GestureProcessor()

        # Add some points
        processor.add_point((0.5, 0.5), (100, 100))
        processor.add_point((0.6, 0.6), (100, 100))

        assert len(processor.stroke_points) == 2

    def test_gesture_image_generation(self):
        """Test gesture image generation."""
        from core.gesture_processor import GestureProcessor

        processor = GestureProcessor()

        # Add points to form a simple shape
        for i in range(10):
            x = 0.1 + (i * 0.8 / 9)  # Line from left to right
            processor.add_point((x, 0.5), (100, 100))

        image = processor.get_gesture_image()
        assert image is not None
        assert image.shape == (28, 28)


class TestSketchClassifier:
    """Test cases for sketch classification."""

    def test_classifier_initialization(self):
        """Test SketchClassifier initialization."""
        from core.classifier import SketchClassifier

        # Test with fallback mode (no model file needed)
        classifier = SketchClassifier(None, None, enable_fallback=True)
        assert classifier.enable_fallback
        assert not classifier.is_available()  # No model loaded

    def test_fallback_predictions(self):
        """Test fallback prediction mode."""
        from core.classifier import SketchClassifier

        classifier = SketchClassifier(None, None, enable_fallback=True)

        # Create dummy image
        dummy_image = np.random.rand(28, 28, 1).astype(np.float32)

        predictions = classifier.predict(dummy_image)
        assert predictions is not None
        assert len(predictions) > 0


class TestHandDetector:
    """Test cases for hand detection."""

    def test_hand_detector_initialization(self):
        """Test HandDetector initialization."""
        from core.hand_detector import HandDetector

        detector = HandDetector()
        assert detector.min_area > 0
        assert detector.max_area > detector.min_area

    def test_frame_processing(self):
        """Test basic frame processing."""
        from core.hand_detector import HandDetector

        detector = HandDetector()

        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        frame_rgb, contours, has_hands = detector.detect(dummy_frame)
        assert frame_rgb.shape == dummy_frame.shape
        assert isinstance(contours, list)
        assert isinstance(has_hands, bool)


# Integration tests
class TestIntegration:
    """Integration tests for component interaction."""

    def test_full_pipeline(self):
        """Test the full processing pipeline."""
        from core.gesture_processor import GestureProcessor
        from core.classifier import SketchClassifier

        # Create components
        processor = GestureProcessor()
        classifier = SketchClassifier(None, None, enable_fallback=True)

        # Add gesture points
        for i in range(20):
            x = 0.1 + (i * 0.8 / 19)
            processor.add_point((x, 0.5), (100, 100))

        # Process gesture
        gesture_image = processor.get_gesture_image()
        assert gesture_image is not None

        # Classify
        predictions = classifier.predict(gesture_image)
        assert predictions is not None
        assert len(predictions) > 0

        # Check prediction format
        class_name, confidence = predictions[0]
        assert isinstance(class_name, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])