"""
test_stroke_manager.py - Pruebas unitarias para StrokeAccumulator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import time
from unittest.mock import Mock

from stroke_manager import StrokeAccumulator


class TestStrokeAccumulator:
    """Pruebas para la clase StrokeAccumulator."""

    @pytest.fixture
    def stroke_config(self):
        """Configuración de prueba para trazos."""
        return {
            "pause_threshold_ms": 400,
            "velocity_threshold": 0.002,
            "min_points": 8,
            "max_stroke_age_ms": 3000,
        }

    def test_init(self, stroke_config, mock_logger):
        """Prueba inicialización."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)

        assert accumulator.config == stroke_config
        assert accumulator.logger == mock_logger
        assert accumulator.points == []
        assert not accumulator.stroke_active

    def test_reset(self, stroke_config, mock_logger):
        """Prueba reset del acumulador."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)
        accumulator.points = [(0.1, 0.2), (0.15, 0.25)]
        accumulator.stroke_active = True

        accumulator.reset()

        assert accumulator.points == []
        assert not accumulator.stroke_active
        assert accumulator.last_point_time is None

    def test_add_point_slow_movement_no_stroke(self, stroke_config, mock_logger):
        """Prueba agregar punto con movimiento lento (no inicia trazo)."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)

        result = accumulator.add_point(0.1, 0.2, velocity=0.001)  # Por debajo del threshold

        assert not result
        assert accumulator.points == []
        assert not accumulator.stroke_active

    def test_add_point_start_new_stroke(self, stroke_config, mock_logger):
        """Prueba iniciar nuevo trazo."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)

        result = accumulator.add_point(0.1, 0.2, velocity=0.01)  # Por encima del threshold

        assert not result
        assert accumulator.points == [(0.1, 0.2)]
        assert accumulator.stroke_active

    def test_add_point_continue_stroke(self, stroke_config, mock_logger):
        """Prueba continuar trazo existente."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)
        accumulator.add_point(0.1, 0.2, velocity=0.01)  # Iniciar

        result = accumulator.add_point(0.15, 0.25, velocity=0.01)  # Continuar

        assert not result
        assert len(accumulator.points) == 2
        assert accumulator.stroke_active

    def test_add_point_complete_stroke_by_pause(self, stroke_config, mock_logger):
        """Prueba completar trazo por pausa."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)

        # Iniciar trazo con suficientes puntos
        for i in range(8):
            accumulator.add_point(0.1 + i*0.01, 0.2, velocity=0.01)

        # Simular pausa (tiempo pasa)
        accumulator.last_significant_move_time = time.time() * 1000 - 500  # 500ms atrás

        result = accumulator.add_point(0.1, 0.2, velocity=0.001)  # Movimiento lento

        assert result  # Debería completar el trazo

    def test_add_point_complete_stroke_by_timeout(self, stroke_config, mock_logger):
        """Prueba completar trazo por timeout."""
        with patch('time.time', side_effect=[10.0, 15.0]):
            accumulator = StrokeAccumulator(stroke_config, mock_logger)

            # Iniciar trazo
            accumulator.add_point(0.1, 0.2, velocity=0.01)

            # Simular timeout
            accumulator.stroke_start_time = 6.0  # 10.0 - 4.0

            result = accumulator.add_point(0.15, 0.25, velocity=0.01)

            assert result  # Debería completar por timeout

    def test_get_stroke_insufficient_points(self, stroke_config, mock_logger):
        """Prueba obtener trazo con pocos puntos."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)
        accumulator.points = [(0.1, 0.2), (0.15, 0.25)]  # Solo 2 puntos

        stroke = accumulator.get_stroke()

        assert stroke is None

    def test_get_stroke_sufficient_points(self, stroke_config, mock_logger):
        """Prueba obtener trazo con suficientes puntos."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)
        points = [(0.1 + i*0.01, 0.2) for i in range(10)]
        accumulator.points = points

        stroke = accumulator.get_stroke()

        assert stroke == points

    def test_get_stroke_info_empty(self, stroke_config, mock_logger):
        """Prueba información de trazo vacío."""
        accumulator = StrokeAccumulator(stroke_config, mock_logger)

        info = accumulator.get_stroke_info()

        assert info["points_count"] == 0
        assert not info["is_active"]
        assert info["age_ms"] == 0

    def test_get_stroke_info_active(self, stroke_config, mock_logger):
        """Prueba información de trazo activo."""
        with patch('time.time', return_value=10.0):
            accumulator = StrokeAccumulator(stroke_config, mock_logger)
            accumulator.add_point(0.1, 0.2, velocity=0.01)
            accumulator.points = [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)]

            info = accumulator.get_stroke_info()

            assert info["points_count"] == 3
            assert info["is_active"]
            assert info["age_ms"] > 0