"""
test_app_integration.py - Pruebas de integración para PictionaryLive
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app import PictionaryLive
from config import MEDIAPIPE_CONFIG, CAMERA_CONFIG, STROKE_CONFIG, MODEL_CONFIG


class TestPictionaryLiveIntegration:
    """Pruebas de integración para la aplicación completa."""

    @pytest.fixture
    def ia_dir(self, tmp_path):
        """Directorio IA temporal con archivos necesarios."""
        ia_dir = tmp_path / "IA"
        ia_dir.mkdir()

        # Crear model_info.json
        model_info = {
            "input_shape": [28, 28, 1],
            "classes": ["apple", "banana", "car"]
        }
        (ia_dir / "model_info.json").write_text(str(model_info).replace("'", '"'))

        return ia_dir

    def test_init_success(self, ia_dir, mock_logger):
        """Prueba inicialización exitosa de la aplicación."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.side_effect = [640, 480, 30]  # width, height, fps

            app = PictionaryLive(
                ia_dir=str(ia_dir),
                camera_id=0,
                debug=False,
                dry_run=False
            )

            assert app.ia_dir == str(ia_dir)
            assert app.camera_id == 0
            assert not app.debug
            assert not app.dry_run

    def test_validate_setup_success(self, ia_dir, mock_logger):
        """Prueba validación exitosa del setup."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.side_effect = [640, 480, 30]

            app = PictionaryLive(str(ia_dir), 0, False, False)

            result = app._validate_setup()
            assert result

    def test_validate_setup_camera_fail(self, ia_dir, mock_logger):
        """Prueba validación cuando falla la cámara."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.isOpened.return_value = False

            app = PictionaryLive(str(ia_dir), 0, False, False)

            result = app._validate_setup()
            assert not result

    def test_run_dry_run(self, ia_dir, mock_logger, caplog):
        """Prueba modo dry-run."""
        with patch('cv2.VideoCapture'):
            app = PictionaryLive(str(ia_dir), 0, False, True)

            app.run()

            # Verificar que se imprimió información del modelo
            assert "Informacion del modelo" in caplog.text
            assert "Total de clases: 3" in caplog.text

    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_run_full_flow_simulation(self, mock_destroy, mock_waitkey, mock_imshow, ia_dir, mock_logger):
        """Prueba simulación del flujo completo de ejecución."""
        # Mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Frame 1
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Frame 2
            (False, None)  # Fin de frames
        ]

        with patch('cv2.VideoCapture', return_value=mock_cap):
            # Mock hand detector
            with patch('src.app.HandDetector') as mock_hand_detector_class:
                mock_detector = Mock()
                mock_detector.detect.return_value = {
                    "hand_landmarks": [(0.1, 0.2), (0.15, 0.25)],  # Mano detectada
                    "hand_confidence": 0.9,
                    "hand_velocity": 0.1,
                    "hands_count": 1
                }
                mock_detector.hands_detector = Mock()
                mock_detector.get_index_finger_position.return_value = (0.5, 0.6)
                mock_detector.draw_hand_landmarks.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                mock_hand_detector_class.return_value = mock_detector

                # Mock UI
                with patch('src.app.PictionaryUI') as mock_ui_class:
                    mock_ui = Mock()
                    mock_ui.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                    mock_ui_class.return_value = mock_ui

                    # Mock classifier
                    with patch('src.app.SketchClassifier') as mock_classifier_class:
                        mock_classifier = Mock()
                        mock_classifier.get_input_shape.return_value = (28, 28, 1)
                        mock_classifier.get_labels.return_value = ["apple", "banana"]
                        mock_classifier.predict.return_value = ("apple", 0.8, [("apple", 0.8), ("banana", 0.2)])
                        mock_classifier_class.return_value = mock_classifier

                        # Mock preprocessor
                        with patch('src.app.DrawingPreprocessor') as mock_preprocessor_class:
                            mock_preprocessor = Mock()
                            mock_preprocessor.preprocess.return_value = np.zeros((28, 28, 1), dtype=np.float32)
                            mock_preprocessor_class.return_value = mock_preprocessor

                            app = PictionaryLive(str(ia_dir), 0, False, False)

                            # Simular que waitKey retorna 'q' para salir
                            mock_waitkey.return_value = ord('q')

                            app.run()

                            # Verificar que se llamó a los componentes
                            assert mock_cap.read.call_count >= 2
                            assert mock_detector.detect.call_count >= 2
                            mock_classifier.predict.assert_called()

    def test_camera_init_success(self, ia_dir, mock_logger):
        """Prueba inicialización exitosa de cámara."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [640, 480, 30]

        with patch('cv2.VideoCapture', return_value=mock_cap):
            app = PictionaryLive(str(ia_dir), 0, False, False)

            result = app._init_camera()
            assert result
            assert app.cap == mock_cap

    def test_camera_init_failure(self, ia_dir, mock_logger):
        """Prueba fallo en inicialización de cámara."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False

        with patch('cv2.VideoCapture', return_value=mock_cap):
            app = PictionaryLive(str(ia_dir), 0, False, False)

            result = app._init_camera()
            assert not result

    def test_save_screenshot(self, ia_dir, mock_logger, tmp_path):
        """Prueba guardar captura de pantalla."""
        with patch('cv2.imwrite') as mock_imwrite:
            app = PictionaryLive(str(ia_dir), 0, False, False)

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            app._save_screenshot(frame)

            # Verificar que se llamó a imwrite
            mock_imwrite.assert_called_once()
            args = mock_imwrite.call_args[0]
            assert "predictions" in str(args[1])  # Path contiene 'predictions'
            assert args[1].suffix == ".png"