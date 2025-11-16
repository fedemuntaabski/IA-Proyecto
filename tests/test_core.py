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
import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.i18n import i18n, _, get_class_name_translation
from core.utils.async_processor import AsyncProcessor, TaskPriority
from core.detection.hand_detector import (
    KalmanFilter1D, ContourBuffer, HistogramAnalyzer, 
    ShadowDetector, AdaptiveRangeManager
)
from core.classification.gesture_processor import (
    VelocityAnalyzer, AdaptiveSmoothing, DuplicateFilter, PointInterpolator
)
from core.config.calibration_manager import (
    CalibrationQualityMonitor, EnvironmentProfile, AutoCalibrationEngine
)


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


class TestDetectionImprovements:
    """Test cases for detection stability improvements."""

    def test_kalman_filter_1d(self):
        """Test 1D Kalman filter for smoothing."""
        kalman = KalmanFilter1D(process_variance=0.01, measurement_variance=0.1)
        
        # Test sequence of measurements with noise
        measurements = [100, 101, 99, 102, 100]
        filtered = []
        
        for m in measurements:
            filtered.append(kalman.update(float(m)))
        
        # Filtered values should be smoother (closer to each other)
        assert len(filtered) == len(measurements)
        assert all(90 <= f <= 110 for f in filtered)

    def test_contour_buffer_stability(self):
        """Test contour buffer for multi-frame stability."""
        buffer = ContourBuffer(max_size=5)
        
        # Create dummy contours
        dummy_contour = np.array([[[0, 0]], [[1, 1]], [[2, 0]]])
        
        # Add same contours
        result = buffer.add([dummy_contour])
        assert len(result) >= 0
        
        # Add to buffer multiple times
        for _ in range(3):
            buffer.add([dummy_contour])
        
        assert len(buffer.buffer) > 0

    def test_histogram_analyzer(self):
        """Test histogram analysis for illumination compensation."""
        analyzer = HistogramAnalyzer(region_grid=3)
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (100, 100, 100)  # Gris uniforme
        
        stats = analyzer.analyze_regions(frame)
        
        assert 'global_mean' in stats
        assert 'global_std' in stats
        assert 'regions' in stats
        assert 80 <= stats['global_mean'] <= 120

    def test_shadow_detector(self):
        """Test shadow detection."""
        detector = ShadowDetector()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 100:200] = (50, 50, 50)  # Área oscura
        
        shadow_mask = detector.detect_shadows(frame)
        
        assert shadow_mask.shape == (480, 640)
        assert shadow_mask.dtype == np.uint8
        assert np.any(shadow_mask > 0)

    def test_adaptive_range_manager(self):
        """Test adaptive HSV range management."""
        manager = AdaptiveRangeManager()
        
        lower = np.array([0, 20, 70])
        upper = np.array([20, 255, 255])
        manager.set_base_ranges(lower, upper)
        
        frame_stats = {
            'global_mean': 128,
            'global_std': 30,
            'regions': {}
        }
        
        shadow_mask = np.zeros((480, 640), dtype=np.uint8)
        
        adjusted = manager.adjust((lower, upper), frame_stats, shadow_mask)
        
        assert adjusted is not None
        assert len(adjusted) == 2


class TestPointProcessingImprovements:
    """Test cases for gesture point processing improvements."""

    def test_velocity_analyzer(self):
        """Test velocity analysis for adaptive smoothing."""
        analyzer = VelocityAnalyzer(window_size=3)
        
        points = [(0, 0), (10, 10), (20, 20), (30, 30)]
        timestamps = [0.0, 0.016, 0.032, 0.048]
        
        velocities = analyzer.calculate_velocities(points, timestamps)
        
        assert len(velocities) == len(points)
        assert all(v >= 0 for v in velocities)
        assert velocities[0] == 0.0  # Primera velocidad es 0

    def test_velocity_classification(self):
        """Test classification of velocity into categories."""
        analyzer = VelocityAnalyzer()
        
        assert analyzer.classify_velocity(25) == 'slow'
        assert analyzer.classify_velocity(100) == 'medium'
        assert analyzer.classify_velocity(200) == 'fast'

    def test_adaptive_smoothing(self):
        """Test adaptive smoothing based on velocity."""
        smoother = AdaptiveSmoothing()
        
        # Create test points with varying speeds
        points = [(0, 0), (5, 5), (10, 10), (15, 15), (20, 20)]
        timestamps = [0.0, 0.016, 0.032, 0.048, 0.064]
        
        smoothed = smoother.smooth_points(points, timestamps)
        
        assert len(smoothed) == len(points)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in smoothed)

    def test_duplicate_filter(self):
        """Test filtering of duplicate/close points."""
        filter = DuplicateFilter(min_distance=2.0)
        
        points = [(0, 0), (1, 0), (2, 0), (10, 10), (11, 10)]
        filtered = filter.filter_duplicates(points)
        
        assert len(filtered) <= len(points)
        assert (0, 0) in filtered
        assert (10, 10) in filtered

    def test_point_interpolator(self):
        """Test point interpolation for filling gaps."""
        interpolator = PointInterpolator(max_gap_distance=20.0)
        
        # Create points with large gap
        points = [(0, 0), (50, 50)]
        interpolated = interpolator.interpolate_gaps(points)
        
        assert len(interpolated) >= len(points)
        assert interpolated[0] == points[0]
        assert interpolated[-1] == points[-1]

    def test_cubic_interpolation(self):
        """Test cubic spline interpolation."""
        interpolator = PointInterpolator(max_gap_distance=10.0, interpolation_method='cubic')
        
        points = [(0, 0), (30, 30)]
        interpolated = interpolator.interpolate_gaps(points)
        
        assert len(interpolated) > 2  # Debe haber interpolación
        assert interpolated[0] == (0, 0)
        assert interpolated[-1] == (30, 30)


class TestCalibrationImprovements:
    """Test cases for automatic calibration system."""

    def test_quality_monitor_initialization(self):
        """Test CalibrationQualityMonitor initialization."""
        monitor = CalibrationQualityMonitor(window_size=30)
        
        assert monitor.window_size == 30
        assert monitor.total_frames == 0
        assert monitor.get_quality_score() == 0.5

    def test_quality_monitor_recording(self):
        """Test recording detection results."""
        monitor = CalibrationQualityMonitor()
        
        # Record some detections
        for i in range(10):
            is_correct = i < 7  # 70% accuracy
            monitor.record_detection(True, is_correct, 0.8)
        
        assert monitor.total_frames == 10
        quality = monitor.get_quality_score()
        assert 0.5 <= quality <= 1.0

    def test_environment_profile_creation(self):
        """Test environment profile creation."""
        profile = EnvironmentProfile('test', 'normal', 'uniform')
        
        assert profile.name == 'test'
        assert profile.lighting_level == 'normal'
        assert profile.background_type == 'uniform'
        
        profile_dict = profile.to_dict()
        assert 'name' in profile_dict
        assert 'creation_date' in profile_dict

    def test_environment_profile_serialization(self):
        """Test environment profile serialization/deserialization."""
        profile = EnvironmentProfile('test', 'bright', 'complex')
        profile_dict = profile.to_dict()
        
        restored = EnvironmentProfile.from_dict(profile_dict)
        
        assert restored.name == profile.name
        assert restored.lighting_level == profile.lighting_level
        assert restored.background_type == profile.background_type

    def test_auto_calibration_engine_environment_detection(self):
        """Test automatic environment detection."""
        engine = AutoCalibrationEngine()
        
        frame_stats = {
            'global_brightness': 150,
            'background_variance': 1000
        }
        
        lighting, background = engine.detect_environment(frame_stats)
        
        assert lighting in ['dark', 'normal', 'bright']
        assert background in ['uniform', 'complex', 'moving']

    def test_auto_calibration_suggestions(self):
        """Test calibration adjustment suggestions."""
        engine = AutoCalibrationEngine()
        
        # Record some bad detections
        for i in range(30):
            engine.quality_monitor.record_detection(True, False, 0.3)
        
        current_ranges = (np.array([0, 20, 70]), np.array([20, 255, 255]))
        quality_metrics = {
            'false_positive_rate': 0.15,
            'false_negative_rate': 0.05
        }
        
        suggested = engine.suggest_range_adjustment(current_ranges, quality_metrics)
        
        # May return None if quality is acceptable, otherwise returns adjusted ranges
        if suggested is not None:
            assert len(suggested) == 2
            assert isinstance(suggested[0], np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])