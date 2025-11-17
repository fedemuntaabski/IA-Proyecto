"""
test_model.py - Pruebas unitarias para SketchClassifier
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from model import SketchClassifier


class TestSketchClassifier:
    """Pruebas para la clase SketchClassifier."""

    def test_init_without_tensorflow(self, temp_ia_dir, mock_logger):
        """Prueba inicialización sin TensorFlow disponible."""
        with patch('src.model.TENSORFLOW_AVAILABLE', False):
            classifier = SketchClassifier(str(temp_ia_dir), mock_logger)

            assert classifier.model is None
            assert classifier.input_shape == [28, 28, 1]
            assert len(classifier.get_labels()) == 5  # De model_info.json

    def test_init_with_tensorflow_and_model(self, temp_ia_dir, mock_logger, mock_tensorflow):
        """Prueba inicialización con TensorFlow y modelo disponible."""
        # Crear modelo mock .keras
        model_file = temp_ia_dir / "sketch_classifier_model.keras"
        model_file.write_text("mock model")

        with patch('src.model.TENSORFLOW_AVAILABLE', True):
            with patch('keras.models.load_model') as mock_load:
                mock_model = Mock()
                mock_load.return_value = mock_model

                classifier = SketchClassifier(str(temp_ia_dir), mock_logger, demo_mode=False)

                assert classifier.model == mock_model
                mock_load.assert_called_once_with(model_file)

    def test_load_model_info_success(self, temp_ia_dir, mock_logger):
        """Prueba carga exitosa de model_info.json."""
        classifier = SketchClassifier(str(temp_ia_dir), mock_logger)

        assert classifier.input_shape == [28, 28, 1]
        assert "apple" in classifier.get_labels()

    def test_load_model_info_missing_file(self, tmp_path, mock_logger):
        """Prueba carga cuando model_info.json no existe."""
        ia_dir = tmp_path / "IA"
        ia_dir.mkdir()

        classifier = SketchClassifier(str(ia_dir), mock_logger, demo_mode=False)

        assert classifier.input_shape == [28, 28, 1]  # Valores por defecto
        assert len(classifier.get_labels()) == 1

    def test_predict_with_model(self, temp_ia_dir, mock_logger, sample_drawing, mock_tensorflow):
        """Prueba predicción con modelo real."""
        # Verificar que el archivo se creó
        assert (temp_ia_dir / "model_info.json").exists()
        
        with patch('src.model.TENSORFLOW_AVAILABLE', True):
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[0.8, 0.1, 0.05, 0.03, 0.02]])
            delattr(mock_model, 'get_input_details')  # Para que use _predict_keras

            classifier = SketchClassifier(str(temp_ia_dir), mock_logger, demo_mode=False)
            classifier.model = mock_model

            label, conf, top3 = classifier.predict(sample_drawing)

            assert isinstance(label, str)
            assert isinstance(conf, float)
            assert len(top3) == 3

    def test_predict_demo_mode(self, temp_ia_dir, mock_logger, sample_drawing):
        """Prueba predicción en modo demo."""
        with patch('src.model.TENSORFLOW_AVAILABLE', False):
            classifier = SketchClassifier(str(temp_ia_dir), mock_logger, demo_mode=True)

            label, conf, top3 = classifier.predict(sample_drawing)

            assert isinstance(label, str)
            assert 0.5 <= conf <= 0.95
            assert len(top3) == 3

    def test_predict_error_handling(self, temp_ia_dir, mock_logger, sample_drawing, mock_tensorflow):
        """Prueba manejo de errores en predicción."""
        with patch('src.model.TENSORFLOW_AVAILABLE', True):
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[0.8, 0.1, 0.05, 0.03, 0.02]])
            delattr(mock_model, 'get_input_details')  # Para que use _predict_keras

            classifier = SketchClassifier(str(temp_ia_dir), mock_logger)
            classifier.model = mock_model

            label, conf, top3 = classifier.predict(sample_drawing)

            # Debería caer en modo demo
            assert isinstance(label, str)
            assert isinstance(conf, float)

    def test_get_input_shape(self, temp_ia_dir, mock_logger):
        """Prueba obtener forma de entrada."""
        classifier = SketchClassifier(str(temp_ia_dir), mock_logger)

        shape = classifier.get_input_shape()
        assert shape == (28, 28, 1)

    def test_get_labels(self, temp_ia_dir, mock_logger):
        """Prueba obtener etiquetas."""
        classifier = SketchClassifier(str(temp_ia_dir), mock_logger)

        labels = classifier.get_labels()
        assert isinstance(labels, list)
        assert len(labels) == 5