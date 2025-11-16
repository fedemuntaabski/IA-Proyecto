"""
test_hand_detector.py - Pruebas unitarias para HandDetector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from hand_detector import HandDetector


class TestHandDetector:
    """Pruebas para la clase HandDetector."""

    @pytest.fixture
    def hand_config(self):
        """Configuración de prueba para MediaPipe."""
        return {
            "static_image_mode": False,
            "max_num_hands": 2,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

    def test_init_without_mediapipe(self, hand_config, mock_logger):
        """Prueba inicialización sin MediaPipe disponible."""
        with patch('src.hand_detector.MEDIAPIPE_AVAILABLE', False):
            detector = HandDetector(hand_config, mock_logger)

            assert detector.hands_detector is None
            mock_logger.warning.assert_called_with("MediaPipe no disponible. Funcionando en modo demo.")

    def test_init_with_mediapipe(self, hand_config, mock_logger):
        """Prueba inicialización con MediaPipe disponible."""
        with patch('src.hand_detector.MEDIAPIPE_AVAILABLE', True):
            with patch('mediapipe.solutions.hands') as mock_mp_hands:
                mock_detector = Mock()
                mock_mp_hands.Hands.return_value = mock_detector

                detector = HandDetector(hand_config, mock_logger)

                assert detector.hands_detector == mock_detector
                mock_mp_hands.Hands.assert_called_once()

    def test_detect_no_detector(self, hand_config, mock_logger):
        """Prueba detección cuando no hay detector inicializado."""
        with patch('src.hand_detector.MEDIAPIPE_AVAILABLE', False):
            detector = HandDetector(hand_config, mock_logger)

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.detect(frame)

            expected = {
                "hand_landmarks": None,
                "hand_confidence": 0.0,
                "hand_velocity": 0.0,
                "hands_count": 0
            }
            assert result == expected

    def test_detect_with_hand(self, hand_config, mock_logger):
        """Prueba detección exitosa de mano."""
        with patch('src.hand_detector.MEDIAPIPE_AVAILABLE', True):
            # Mock MediaPipe
            mock_results = Mock()
            mock_hand_landmarks = Mock()
            mock_hand_landmarks.landmark = [
                Mock(x=0.1, y=0.2),  # Punto 0
                Mock(x=0.15, y=0.25),  # Punto 1
                # ... más puntos
            ] + [Mock(x=0.5, y=0.6) for _ in range(20)]  # Rellenar hasta 21

            mock_results.multi_hand_landmarks = [mock_hand_landmarks]
            mock_results.multi_handedness = [Mock(classification=Mock(score=0.9))]

            mock_detector = Mock()
            mock_detector.process.return_value = mock_results

            detector = HandDetector(hand_config, mock_logger)
            detector.hands_detector = mock_detector

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.detect(frame)

            assert result["hand_landmarks"] is not None
            assert len(result["hand_landmarks"]) == 21
            assert result["hand_confidence"] == 0.9
            assert result["hands_count"] == 1

    def test_detect_no_hand(self, hand_config, mock_logger):
        """Prueba detección cuando no hay mano."""
        with patch('src.hand_detector.MEDIAPIPE_AVAILABLE', True):
            mock_results = Mock()
            mock_results.multi_hand_landmarks = None

            mock_detector = Mock()
            mock_detector.process.return_value = mock_results

            detector = HandDetector(hand_config, mock_logger)
            detector.hands_detector = mock_detector

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.detect(frame)

            assert result["hand_landmarks"] is None
            assert result["hands_count"] == 0

    def test_detect_velocity_calculation(self, hand_config, mock_logger):
        """Prueba cálculo de velocidad."""
        with patch('src.hand_detector.MEDIAPIPE_AVAILABLE', True):
            # Primera detección
            mock_results1 = Mock()
            mock_hand_landmarks1 = Mock()
            mock_hand_landmarks1.landmark = [Mock(x=0.1, y=0.2) for _ in range(9)]  # Punto 8 en (0.1, 0.2)
            mock_results1.multi_hand_landmarks = [mock_hand_landmarks1]

            # Segunda detección
            mock_results2 = Mock()
            mock_hand_landmarks2 = Mock()
            mock_hand_landmarks2.landmark = [Mock(x=0.2, y=0.3) for _ in range(9)]  # Punto 8 en (0.2, 0.3)
            mock_results2.multi_hand_landmarks = [mock_hand_landmarks2]

            mock_detector = Mock()
            mock_detector.process.side_effect = [mock_results1, mock_results2]

            detector = HandDetector(hand_config, mock_logger)
            detector.hands_detector = mock_detector

            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Primera llamada
            result1 = detector.detect(frame)
            assert result1["hand_velocity"] == 0.0  # Primera vez, no hay velocidad

            # Segunda llamada
            result2 = detector.detect(frame)
            expected_velocity = ((0.2 - 0.1)**2 + (0.3 - 0.2)**2)**0.5
            assert abs(result2["hand_velocity"] - expected_velocity) < 0.001

    def test_draw_hand_landmarks_no_landmarks(self, hand_config, mock_logger):
        """Prueba dibujo cuando no hay landmarks."""
        detector = HandDetector(hand_config, mock_logger)
        detector.hand_landmarks = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.draw_hand_landmarks(frame)

        assert np.array_equal(result, frame)

    def test_draw_hand_landmarks_with_landmarks(self, hand_config, mock_logger):
        """Prueba dibujo de landmarks."""
        detector = HandDetector(hand_config, mock_logger)
        # Landmarks normalizados (0-1)
        detector.hand_landmarks = [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)] + [(0.0, 0.0)] * 18

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.draw_hand_landmarks(frame)

        # Verificar que se dibujó algo (no es todo ceros)
        assert not np.array_equal(result, frame)

    def test_get_index_finger_position_no_landmarks(self, hand_config, mock_logger):
        """Prueba obtener posición del índice sin landmarks."""
        detector = HandDetector(hand_config, mock_logger)
        detector.hand_landmarks = None

        pos = detector.get_index_finger_position()
        assert pos is None

    def test_get_index_finger_position_with_landmarks(self, hand_config, mock_logger):
        """Prueba obtener posición del índice con landmarks."""
        detector = HandDetector(hand_config, mock_logger)
        detector.hand_landmarks = [(0.0, 0.0)] * 9  # Hasta punto 8

        pos = detector.get_index_finger_position()
        assert pos == (0.0, 0.0)