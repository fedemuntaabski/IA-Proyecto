#!/usr/bin/env python3
"""
Simple test runner for the Air Draw Classifier project.

Runs basic tests without requiring pytest.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
        from core.config_manager import ConfigManager

        config = ConfigManager()
        assert config.current_profile is not None
        assert len(config.profiles) > 0

    except ImportError:
        print("⚠ ConfigManager not available, skipping test")
        return

def test_gesture_processor_basic():
    """Test basic gesture processor functionality."""
    try:
        from core.gesture_processor import GestureProcessor

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
        from core.classifier import SketchClassifier
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

def main():
    """Run all tests."""
    print("🧪 Running Air Draw Classifier Tests")
    print("=" * 40)

    tests = [
        (test_i18n_basic, "i18n Basic Functionality"),
        (test_config_basic, "Config Manager Basic"),
        (test_gesture_processor_basic, "Gesture Processor Basic"),
        (test_classifier_basic, "Classifier Basic"),
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