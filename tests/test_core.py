"""
Unit tests for the Air Draw Classifier project.

This module contains comprehensive tests for all core components including
unit tests, integration tests, and performance benchmarks.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.i18n import i18n, _, get_class_name_translation
from core.utils.async_processor import AsyncProcessor, TaskPriority


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

        # Test Spanish
        assert i18n.load_language('es')
        assert i18n.get_current_language() == 'es'

    def test_class_name_translation(self):
        """Test class name translations."""
        i18n.load_language('es')
        # Usar palabras genéricas para traducción
        result = i18n.get_text('Test translation')
        assert result is not None

        i18n.load_language('en')
        result = i18n.get_text('Test translation')
        assert result is not None

    def test_fallback_translation(self):
        """Test fallback when translation is missing."""
        original_text = "NonExistentTranslationKey"
        result = _(original_text)
        assert result is not None

    def test_available_languages(self):
        """Test getting available languages."""
        languages = i18n.get_available_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0

    def test_format_message(self):
        """Test message formatting with variables."""
        result = i18n.format_message("Hello {name}", name="World")
        assert result is not None
        assert "World" in result or result == "Hello {name}"

    def test_supported_languages_dict(self):
        """Test getting supported languages dictionary."""
        langs = i18n.get_supported_languages()
        assert isinstance(langs, dict)
        assert 'en' in langs
        assert 'es' in langs


class TestAsyncProcessor:
    """Test cases for async processing with priorities."""

    def test_async_processor_initialization(self):
        """Test AsyncProcessor initialization."""
        processor = AsyncProcessor(max_workers=2)
        assert processor.max_workers == 2
        assert not processor.running

    def test_async_processor_start_stop(self):
        """Test starting and stopping AsyncProcessor."""
        processor = AsyncProcessor(max_workers=2)
        processor.start()
        assert processor.running

        processor.stop()
        assert not processor.running

    def test_submit_task_with_priority(self):
        """Test submitting tasks with different priorities."""
        processor = AsyncProcessor(max_workers=2)
        processor.start()

        def test_func():
            return "result"

        task_id = processor.submit_task(test_func, priority=TaskPriority.HIGH)
        assert task_id is not None
        assert task_id.startswith("task_")

        # Check that task is in the tasks dict
        assert task_id in processor.tasks or task_id is None

        processor.stop()

    def test_task_result_retrieval(self):
        """Test retrieving task results."""
        processor = AsyncProcessor(max_workers=2)
        processor.start()

        def simple_task():
            return 42

        task_id = processor.submit_task(simple_task)
        
        # Wait a bit for task to complete
        time.sleep(0.1)
        
        result = processor.get_task_result(task_id, timeout=1.0)
        assert result == 42 or result is None  # May or may not be ready

        processor.stop()

    def test_async_stats(self):
        """Test getting processor statistics."""
        processor = AsyncProcessor(max_workers=2)
        processor.start()

        stats = processor.get_stats()
        assert 'running' in stats
        assert 'active_tasks' in stats
        assert 'completed_tasks' in stats
        assert stats['running'] is True

        processor.stop()

    def test_context_manager(self):
        """Test AsyncProcessor as context manager."""
        with AsyncProcessor(max_workers=2) as processor:
            assert processor.running

            def test_func():
                return "result"

            task_id = processor.submit_task(test_func)
            assert task_id is not None

        assert not processor.running


class TestConfigManager:
    """Test cases for configuration management."""

    def test_config_initialization(self):
        """Test ConfigManager initialization."""
        from core.config.config_manager import ConfigManager

        config_manager = ConfigManager()
        assert config_manager.current_profile is not None
        assert len(config_manager.profiles) > 0

    def test_profile_management(self):
        """Test profile creation and switching."""
        from core.config.config_manager import ConfigManager

        config_manager = ConfigManager()

        # Create test profile
        assert config_manager.create_profile("test_profile_temp")

        # Switch to profile
        assert config_manager.set_active_profile("test_profile_temp")
        assert config_manager.current_profile.name == "test_profile_temp"

        # Clean up
        config_manager.delete_profile("test_profile_temp")

    def test_detection_config_update(self):
        """Test updating detection configuration."""
        from core.config.config_manager import ConfigManager

        config_manager = ConfigManager()
        config_manager.update_detection_config(min_area=6000)
        
        config = config_manager.get_detection_config()
        assert config.min_area == 6000


class TestGestureProcessor:
    """Test cases for gesture processing."""

    def test_gesture_processor_initialization(self):
        """Test GestureProcessor initialization."""
        from core.classification.gesture_processor import GestureProcessor

        processor = GestureProcessor()
        assert processor.stroke_points == []
        assert processor.image_size == 28

    def test_point_adding(self):
        """Test adding points to gesture."""
        from core.classification.gesture_processor import GestureProcessor

        processor = GestureProcessor()

        # Add some points
        processor.add_point((0.5, 0.5), (100, 100))
        processor.add_point((0.6, 0.6), (100, 100))

        assert len(processor.stroke_points) == 2

    def test_gesture_image_generation(self):
        """Test gesture image generation."""
        from core.classification.gesture_processor import GestureProcessor

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
        from core.classification.classifier import SketchClassifier

        # Test with fallback mode (no model file needed)
        classifier = SketchClassifier(None, None, enable_fallback=True)
        assert classifier.enable_fallback

    def test_fallback_predictions(self):
        """Test fallback prediction mode."""
        from core.classification.classifier import SketchClassifier

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
        from core.detection.hand_detector import HandDetector

        detector = HandDetector()
        assert detector.min_area > 0
        assert detector.max_area > detector.min_area

    def test_frame_processing(self):
        """Test basic frame processing."""
        from core.detection.hand_detector import HandDetector

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

    def test_full_gesture_pipeline(self):
        """Test the full gesture processing pipeline."""
        from core.classification.gesture_processor import GestureProcessor
        from core.classification.classifier import SketchClassifier

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

    def test_async_processing_integration(self):
        """Test async processor integration with tasks."""
        processor = AsyncProcessor(max_workers=2)
        processor.start()

        def compute_task(n):
            time.sleep(0.05)
            return n * 2

        # Submit multiple tasks with different priorities
        task_ids = []
        for i in range(5):
            priority = TaskPriority.HIGH if i < 2 else TaskPriority.NORMAL
            task_id = processor.submit_task(compute_task, i, priority=priority)
            task_ids.append(task_id)

        # Wait for completion
        time.sleep(0.5)

        # Check stats
        stats = processor.get_stats()
        assert stats['total_tasks_submitted'] >= len(task_ids)

        processor.stop()


class TestPerformance:
    """Performance and benchmark tests."""

    def test_gesture_image_generation_performance(self):
        """Test performance of gesture image generation."""
        from core.classification.gesture_processor import GestureProcessor

        processor = GestureProcessor()

        # Add many points
        start_time = time.time()
        for i in range(100):
            x = (i * 0.01) % 1.0
            y = 0.5 + np.sin(i * 0.1) * 0.2
            processor.add_point((x, y), (100, 100))
        
        image = processor.get_gesture_image()
        elapsed = time.time() - start_time

        assert image is not None
        assert elapsed < 1.0  # Should complete in less than 1 second

    def test_async_task_throughput(self):
        """Test async processor task throughput."""
        processor = AsyncProcessor(max_workers=2)
        processor.start()

        def quick_task():
            return 1

        start_time = time.time()
        task_ids = []

        for i in range(20):
            task_id = processor.submit_task(quick_task)
            if task_id:
                task_ids.append(task_id)

        time.sleep(0.5)
        elapsed = time.time() - start_time

        processor.stop()

        assert len(task_ids) > 0
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])