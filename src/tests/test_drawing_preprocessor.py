"""
test_drawing_preprocessor.py - Pruebas unitarias para DrawingPreprocessor
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from unittest.mock import Mock, patch

from drawing_preprocessor import DrawingPreprocessor


class TestDrawingPreprocessor:
    """Pruebas para la clase DrawingPreprocessor."""

    @pytest.fixture
    def preprocessor_config(self):
        """Configuración de prueba para preprocessing."""
        return {
            "scale_factor": 0.8,
            "min_stroke_length": 0.05,
            "min_points": 5,
            "blur_kernel": 3,
            "blur_sigma": 0.5,
            "thickness_base": 2,
            "thickness_max": 4,
        }

    def test_init(self, preprocessor_config):
        """Prueba inicialización."""
        input_shape = (28, 28, 1)
        preprocessor = DrawingPreprocessor(input_shape, preprocessor_config)

        assert preprocessor.h == input_shape[0]
        assert preprocessor.w == input_shape[1]
        assert preprocessor.c == input_shape[2]

    def test_preprocess_valid_stroke(self, preprocessor_config):
        """Prueba preprocessing de trazo válido."""
        input_shape = (28, 28, 1)
        preprocessor = DrawingPreprocessor(input_shape, preprocessor_config)

        # Trazo válido
        stroke = [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3), (0.25, 0.35), (0.3, 0.4)]

        result = preprocessor.preprocess(stroke)

        assert result.shape == input_shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_preprocess_insufficient_points(self, preprocessor_config):
        """Prueba preprocessing con pocos puntos."""
        input_shape = (28, 28, 1)
        preprocessor = DrawingPreprocessor(input_shape, preprocessor_config)

        # Trazo con pocos puntos
        stroke = [(0.1, 0.2), (0.15, 0.25)]

        result = preprocessor.preprocess(stroke)

        # Debería retornar array vacío o manejar el caso
        assert result.shape == input_shape

    def test_preprocess_short_stroke(self, preprocessor_config):
        """Prueba preprocessing con trazo muy corto."""
        input_shape = (28, 28, 1)
        preprocessor = DrawingPreprocessor(input_shape, preprocessor_config)

        # Trazo muy corto
        stroke = [(0.1, 0.2), (0.101, 0.201)]

        result = preprocessor.preprocess(stroke)

        assert result.shape == input_shape

    def test_preprocess_empty_stroke(self, preprocessor_config):
        """Prueba preprocessing con trazo vacío."""
        input_shape = (28, 28, 1)
        preprocessor = DrawingPreprocessor(input_shape, preprocessor_config)

        stroke = []

        result = preprocessor.preprocess(stroke)

        assert result.shape == input_shape
        # Debería ser un canvas blanco (unos)
        assert np.all(result == 1)