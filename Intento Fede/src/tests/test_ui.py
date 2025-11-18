"""
test_ui.py - Pruebas unitarias para PictionaryUI
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from ui import PictionaryUI


class TestPictionaryUI:
    """Pruebas para la clase PictionaryUI."""

    @pytest.fixture
    def ui_config(self):
        """Configuración de prueba para UI."""
        return {
            "window_name": "Test Window",
            "window_width": 800,
            "window_height": 600,
            "show_fps": True,
            "show_diagnostics": True,
            "show_top_predictions": 3,
        }

    def test_init(self, ui_config):
        """Prueba inicialización."""
        ui = PictionaryUI(ui_config)

        assert ui.config == ui_config
        assert ui.last_prediction is None
        assert ui.frame_count == 0

    def test_update_fps(self, ui_config):
        """Prueba actualización de FPS."""
        ui = PictionaryUI(ui_config)
        ui.last_time = 0.0  # Resetear tiempo inicial

        ui.update_fps(0.0)  # Tiempo inicial
        ui.update_fps(1.0)  # 1 segundo después

        # FPS debería ser calculado
        assert ui.fps > 0

    def test_render(self, ui_config):
        """Prueba renderizado."""
        ui = PictionaryUI(ui_config)

        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        original = frame.copy()
        prediction = ("apple", 0.8, [("apple", 0.8), ("banana", 0.2)])

        result = ui.render(
            frame,
            hand_detected=True,
            stroke_points=10,
            hand_velocity=0.1,
            prediction=prediction
        )

        assert result.shape == original.shape
        assert not np.array_equal(result, original)  # Debería haber cambios

    def test_fit_text_scale(self, ui_config):
        """Prueba ajuste de escala de texto."""
        ui = PictionaryUI(ui_config)

        scale = ui._fit_text_scale("Test Text", cv2.FONT_HERSHEY_SIMPLEX, 200, 1.0)

        assert 0.3 <= scale <= 1.0

    def test_draw_stroke_preview(self, ui_config):
        """Prueba dibujo de preview de trazo."""
        ui = PictionaryUI(ui_config)

        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        original = frame.copy()
        points = [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)]

        result = ui.draw_stroke_preview(frame, points)

        assert result.shape == original.shape
        assert not np.array_equal(result, original)