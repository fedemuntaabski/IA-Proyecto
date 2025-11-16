"""
conftest.py - Configuración y fixtures comunes para pruebas
"""

import pytest
import logging
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_logger():
    """Fixture para logger mockeado."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def temp_ia_dir(tmp_path):
    """Fixture para directorio IA temporal con archivos mock."""
    ia_dir = tmp_path / "IA"
    ia_dir.mkdir()

    # Crear model_info.json mock
    model_info = {
        "input_shape": [28, 28, 1],
        "classes": ["apple", "banana", "car", "dog", "house"]
    }
    (ia_dir / "model_info.json").write_text(str(model_info).replace("'", '"'))

    return ia_dir


@pytest.fixture
def mock_tensorflow():
    """Fixture para mockear TensorFlow/Keras."""
    from unittest import mock
    with mock.patch.dict('sys.modules', {
        'tensorflow': Mock(),
        'keras': Mock(),
        'tensorflow.keras': Mock(),
    }):
        # Configurar mocks básicos
        tf_mock = Mock()
        keras_mock = Mock()
        tf_mock.keras = keras_mock
        tf_mock.config.list_physical_devices.return_value = []

        import sys
        sys.modules['tensorflow'] = tf_mock
        sys.modules['keras'] = keras_mock
        sys.modules['tensorflow.keras'] = keras_mock

        yield


@pytest.fixture
def sample_drawing():
    """Fixture para un dibujo de muestra (28x28x1)."""
    return np.random.rand(28, 28, 1).astype(np.float32)


@pytest.fixture
def sample_stroke_points():
    """Fixture para puntos de trazo de muestra."""
    return [
        (100, 200, 0.1),
        (105, 195, 0.15),
        (110, 190, 0.2),
        (115, 185, 0.25),
        (120, 180, 0.3),
    ]