"""
UI Manager - Gestor de Interfaz de Usuario para Air Draw Classifier.

Este módulo maneja toda la lógica de dibujo de la interfaz gráfica,
manteniendo la separación de responsabilidades y siguiendo el principio KISS.
"""

import cv2
import numpy as np
import time
from typing import List, Optional, Tuple
from ..i18n import _
from ..utils import (COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_INFO, COLOR_TEXT_PRIMARY,
                     COLOR_ACCENT, COLOR_BG_PRIMARY, COLOR_BG_SECONDARY, COLOR_BORDER,
                     DEFAULT_BAR_HEIGHT)


class UIManager:
    """
    Gestor de interfaz de usuario para la aplicación de dibujo en el aire.

    Responsabilidades:
    - Dibujar elementos de UI en frames
    - Gestionar estado visual de la aplicación
    - Mostrar información contextual al usuario
    """

    def __init__(self):
        """Inicializa el gestor de UI."""
        # Tema de colores mejorado
        self.theme = {
            'bg_primary': COLOR_BG_PRIMARY,
            'bg_secondary': COLOR_BG_SECONDARY,
            'border': COLOR_BORDER,
            'text_primary': COLOR_TEXT_PRIMARY,
            'text_success': COLOR_SUCCESS,
            'text_warning': COLOR_WARNING,
            'text_error': COLOR_ERROR,
            'text_info': COLOR_INFO,
            'accent': COLOR_ACCENT
        }

        # Estado de UI
        self.show_help = True
        self.current_fps = 0.0

    def draw_ui(self, frame: np.ndarray, app_state: dict) -> np.ndarray:
        """
        Dibuja la interfaz completa en el frame.

        Args:
            frame: Frame de OpenCV donde dibujar
            app_state: Diccionario con el estado de la aplicación

        Returns:
            Frame con la UI dibujada
        """
        height, width = frame.shape[:2]

        # Dibujar barra superior
        self._draw_top_bar(frame, app_state)

        # Dibujar predicción si existe
        self._draw_prediction(frame, app_state)

        # Dibujar panel de ayuda si está habilitado
        if self.show_help:
            self._draw_help_panel(frame, app_state)

        return frame

    def _draw_top_bar(self, frame: np.ndarray, app_state: dict) -> None:
        """Dibuja la barra superior con información de estado."""
        height, width = frame.shape[:2]

        # Fondo de la barra
        bar_height = DEFAULT_BAR_HEIGHT
        for i in range(bar_height):
            alpha = i / bar_height
            color = tuple(int(self.theme['bg_primary'][j] * (1-alpha) + self.theme['bg_secondary'][j] * alpha) for j in range(3))
            cv2.line(frame, (0, i), (width, i), color)

        cv2.rectangle(frame, (0, 0), (width, bar_height), self.theme['border'], 1)

        # Estado de detección
        y_offset = 25
        has_hands = app_state.get('has_hands', False)
        stroke_points_count = len(app_state.get('stroke_points', []))

        if has_hands:
            cv2.circle(frame, (width-30, 20), 8, self.theme['text_success'], -1)
            cv2.circle(frame, (width-30, 20), 10, self.theme['text_success'], 2)
            status_text = f"👋 {_('Mano detectada')} | Puntos: {stroke_points_count}"
            status_color = self.theme['text_success']
            confidence_indicator = "●●●"
        else:
            cv2.circle(frame, (width-30, 20), 8, self.theme['text_error'], -1)
            cv2.circle(frame, (width-30, 20), 10, self.theme['text_error'], 2)
            status_text = _("Sin detección de manos")
            status_color = self.theme['text_error']
            confidence_indicator = "○○○"

        cv2.putText(frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        fps_color = self.theme['text_success'] if self.current_fps >= 25 else self.theme['text_warning'] if self.current_fps >= 15 else self.theme['text_error']
        cv2.putText(frame, fps_text, (width - 100, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # Barra de confianza
        cv2.putText(frame, f"Confianza: {confidence_indicator}", (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text_info'], 1)

        # Estado de dibujo
        y_offset += 35
        is_drawing = app_state.get('is_drawing', False)
        min_points = app_state.get('min_points_for_classification', 10)

        if is_drawing:
            pulse = int(time.time() * 4) % 2
            draw_color = self.theme['accent'] if pulse else self.theme['text_warning']
            draw_status = f"✏️ {_('DIBUJANDO...')}"
            # Barra de progreso
            progress = min(stroke_points_count / min_points, 1.0)
            bar_width = int(progress * 200)
            cv2.rectangle(frame, (10, y_offset + 5), (10 + bar_width, y_offset + 15), draw_color, -1)
            cv2.rectangle(frame, (10, y_offset + 5), (210, y_offset + 15), self.theme['border'], 1)
        else:
            draw_status = f"✅ {_('Listo para dibujar')}"
            draw_color = self.theme['text_success']

        cv2.putText(frame, draw_status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

    def _draw_prediction(self, frame: np.ndarray, app_state: dict) -> None:
        """Dibuja la predicción actual si existe."""
        prediction = app_state.get('last_prediction')
        if not prediction:
            return

        class_name, confidence, pred_time = prediction
        if time.time() - pred_time > 8.0:  # Expirar después de 8 segundos
            return

        height, width = frame.shape[:2]
        bar_height = DEFAULT_BAR_HEIGHT

        # Fondo para la predicción
        pred_bg_width = 280
        pred_bg_height = 50
        pred_x = width - pred_bg_width - 10
        pred_y = bar_height - pred_bg_height - 10

        cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_bg_width, pred_y + pred_bg_height),
                     self.theme['bg_secondary'], -1)
        cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_bg_width, pred_y + pred_bg_height),
                     self.theme['border'], 1)

        # Texto de predicción
        pred_text = f"🎯 {class_name}"
        conf_text = f"Confianza: {confidence:.1%}"

        cv2.putText(frame, pred_text, (pred_x + 10, pred_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme['text_success'], 2)
        cv2.putText(frame, conf_text, (pred_x + 10, pred_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text_info'], 1)

        # Barra de confianza visual
        conf_bar_width = int(confidence * (pred_bg_width - 20))
        cv2.rectangle(frame, (pred_x + 10, pred_y + 45), (pred_x + 10 + conf_bar_width, pred_y + 48),
                     self.theme['text_success'], -1)

    def _draw_help_panel(self, frame: np.ndarray, app_state: dict) -> None:
        """Dibuja el panel de ayuda contextual."""
        height, width = frame.shape[:2]

        help_y = height - 120
        cv2.rectangle(frame, (0, help_y), (width, height), self.theme['bg_secondary'], -1)
        cv2.rectangle(frame, (0, help_y), (width, height), self.theme['border'], 1)

        # Título
        cv2.putText(frame, _("AYUDA & CONTROLES"), (10, help_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text_primary'], 2)

        # Controles
        controls_y = help_y + 40
        controls = [
            ("ESPACIO", _("Forzar clasificación")),
            ("R", _("Limpiar dibujo")),
            ("Q", _("Salir de la app")),
            ("H", _("Mostrar/ocultar ayuda"))
        ]

        for i, (key, desc) in enumerate(controls):
            y_pos = controls_y + i * 18
            key_text = f"[{key}]"
            cv2.putText(frame, key_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['accent'], 1)
            cv2.putText(frame, desc, (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text_primary'], 1)

        # Tips
        tips = [
            _("Muestra tu mano completa a la cámara"),
            _("Dibuja con el dedo índice extendido"),
            _("Mantén buena iluminación para mejor detección")
        ]

        for i, tip in enumerate(tips):
            tip_y = height - 60 + i * 15
            cv2.putText(frame, tip, (10, tip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text_info'], 1)

        # Estadísticas
        self._draw_stats(frame, app_state, help_y, width)

    def _draw_stats(self, frame: np.ndarray, app_state: dict, help_y: int, width: int) -> None:
        """Dibuja las estadísticas de sesión."""
        stats_x = width - 200
        stats_y = help_y - 25

        session_time = app_state.get('session_time', 0)
        total_drawings = app_state.get('total_drawings', 0)
        successful_predictions = app_state.get('successful_predictions', 0)

        minutes, seconds = divmod(int(session_time), 60)

        stats = [
            f"{_('Tiempo')}: {minutes:02d}:{seconds:02d}",
            f"{_('Dibujos')}: {total_drawings}",
            f"{_('Éxitos')}: {successful_predictions}"
        ]

        cv2.rectangle(frame, (stats_x - 10, stats_y - 25), (width - 10, stats_y + len(stats) * 15),
                     self.theme['bg_secondary'], -1)
        cv2.rectangle(frame, (stats_x - 10, stats_y - 25), (width - 10, stats_y + len(stats) * 15),
                     self.theme['border'], 1)

        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (stats_x, stats_y + i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text_info'], 1)

    def toggle_help(self) -> None:
        """Alterna la visibilidad del panel de ayuda."""
        self.show_help = not self.show_help

    def update_fps(self, fps: float) -> None:
        """Actualiza el valor de FPS para mostrar."""
        self.current_fps = fps