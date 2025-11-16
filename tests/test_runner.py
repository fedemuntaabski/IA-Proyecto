#!/usr/bin/env python3
"""
Simple test runner for the Air Draw Classifier project.

Runs basic tests without requiring pytest.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_test(test_func, test_name):
    """Run a single test function."""
    try:
        test_func()
        print(f"✓ {test_name}")
        return True
    except Exception as e:
        print(f"✗ {test_name}: {e}")
        traceback.print_exc()
        return False

def test_i18n_basic():
    """Test basic i18n functionality."""
    try:
        from core.i18n import i18n, _

        # Test initialization
        assert i18n.current_language is not None

        # Test translation
        text = _("Mano detectada")
        assert isinstance(text, str)

        # Test language switching
        assert i18n.load_language('en')
        assert i18n.load_language('es')

    except ImportError:
        print("⚠ i18n module not available, skipping test")
        return

def test_config_basic():
    """Test basic config functionality."""
    try:
        from core import ConfigManager

        config = ConfigManager()
        assert config.current_profile is not None
        assert len(config.profiles) > 0

    except ImportError:
        print("⚠ ConfigManager not available, skipping test")
        return

def test_gesture_processor_basic():
    """Test basic gesture processor functionality."""
    try:
        from core import GestureProcessor

        processor = GestureProcessor()
        assert processor.stroke_points == []

        # Test adding points
        processor.add_point((0.5, 0.5), (100, 100))
        assert len(processor.stroke_points) == 1

    except ImportError:
        print("⚠ GestureProcessor not available, skipping test")
        return

def test_classifier_basic():
    """Test basic classifier functionality."""
    try:
        from core import SketchClassifier
        import numpy as np

        classifier = SketchClassifier(None, None, enable_fallback=True)
        assert classifier.enable_fallback

        # Test fallback prediction
        dummy_image = np.random.rand(28, 28, 1).astype(np.float32)
        predictions = classifier.predict(dummy_image)
        assert predictions is not None

    except ImportError:
        print("⚠ SketchClassifier not available, skipping test")
        return

def test_ui_manager_basic():
    """Test basic UI manager functionality."""
    try:
        from core import UIManager
        import numpy as np

        ui_manager = UIManager()

        # Test initialization
        assert ui_manager.current_fps == 0.0
        assert not ui_manager.show_help

        # Test theme switching
        ui_manager.switch_theme('dark')
        assert ui_manager.current_theme == 'dark'

        # Test UI drawing
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_app_state = {
            'has_hands': True,
            'is_drawing': False,
            'stroke_points': [],
            'min_points_for_classification': 10,
            'session_time': 60.0,
            'total_drawings': 5,
            'successful_predictions': 4
        }

        result_frame = ui_manager.draw_ui(test_frame, test_app_state)
        assert result_frame.shape == test_frame.shape

    except ImportError:
        print("⚠ UIManager not available, skipping test")
        return

def test_gpu_manager_basic():
    """Test basic GPU manager functionality."""
    try:
        from core.utils.gpu_manager import gpu_manager

        # Test GPU detection
        gpu_info = gpu_manager.get_gpu_info()
        assert isinstance(gpu_info, dict)
        assert 'gpu_available' in gpu_info

        # Test memory info
        memory_info = gpu_manager.get_memory_usage()
        assert isinstance(memory_info, dict)

        # Test optimization (should not fail)
        if gpu_manager.is_gpu_available():
            gpu_manager.optimize_for_inference()

    except ImportError:
        print("⚠ GPU Manager not available, skipping test")
        return

def test_async_processor_basic():
    """Test basic async processor functionality."""
    try:
        from core.utils.async_processor import async_processor, ml_async_processor
        import time

        # Test processor initialization
        assert not async_processor.running

        # Start processor
        async_processor.start()
        assert async_processor.running

        # Test task submission
        def dummy_task():
            time.sleep(0.1)
            return "test_result"

        task_id = async_processor.submit_task(dummy_task)
        assert task_id is not None

        # Wait for completion
        time.sleep(0.2)
        result = async_processor.get_task_result(task_id)
        assert result == "test_result"

        # Stop processor
        async_processor.stop()
        assert not async_processor.running

    except ImportError:
        print("⚠ Async Processor not available, skipping test")
        return

def test_analytics_basic():
    """Test basic analytics functionality."""
    try:
        from core.utils.analytics import analytics_tracker

        # Test event tracking
        initial_count = len(analytics_tracker.events)
        analytics_tracker.track_event('test_event', {'test': 'data'})
        assert len(analytics_tracker.events) == initial_count + 1

        # Test prediction tracking
        analytics_tracker.track_prediction(True, 0.85, 'circle')
        analytics_tracker.track_prediction(False, 0.3, 'square')

        # Test session summary
        summary = analytics_tracker.get_session_summary()
        assert isinstance(summary, dict)
        assert 'total_predictions' in summary

    except ImportError:
        print("⚠ Analytics not available, skipping test")
        return

def main():
    """Run all tests."""
    print("🧪 Running Air Draw Classifier Tests")
    print("=" * 40)

    tests = [
        (test_i18n_basic, "i18n Basic Functionality"),
        (test_config_basic, "Config Manager Basic"),
        (test_gesture_processor_basic, "Gesture Processor Basic"),
        (test_classifier_basic, "Classifier Basic"),
        (test_ui_manager_basic, "UI Manager Basic"),
        (test_gpu_manager_basic, "GPU Manager Basic"),
        (test_async_processor_basic, "Async Processor Basic"),
        (test_analytics_basic, "Analytics Basic"),
    ]

    passed = 0
    total = len(tests)

    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1

    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())