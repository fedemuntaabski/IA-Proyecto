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
        self.show_help = False
        self.current_fps = 0.0

        # Sistema de tooltips mejorado
        self.tooltip_active = False
        self.tooltip_text = ""
        self.tooltip_position = (0, 0)
        self.tooltip_start_time = 0
        self.tooltip_duration = 3.0  # segundos

        # Sistema de tutorial
        self.tutorial_active = False
        self.tutorial_step = 0
        self.tutorial_steps = [
            _("¡Bienvenido! Muestra tu mano completa a la cámara"),
            _("Extiende tu dedo índice para empezar a dibujar"),
            _("Dibuja formas reconocibles (círculo, cuadrado, etc.)"),
            _("Presiona ESPACIO para clasificar tu dibujo"),
            _("¡Excelente! Presiona H para ver más controles")
        ]

        # Temas disponibles mejorados
        self.available_themes = {
            'default': self.theme.copy(),
            'dark': {
                'bg_primary': (20, 20, 30),
                'bg_secondary': (40, 40, 50),
                'border': (100, 100, 120),
                'text_primary': (220, 220, 240),
                'text_success': (100, 255, 100),
                'text_warning': (255, 200, 100),
                'text_error': (255, 100, 100),
                'text_info': (150, 150, 255),
                'accent': (255, 150, 255)
            },
            'high_contrast': {
                'bg_primary': (0, 0, 0),
                'bg_secondary': (50, 50, 50),
                'border': (255, 255, 255),
                'text_primary': (255, 255, 255),
                'text_success': (0, 255, 0),
                'text_warning': (255, 255, 0),
                'text_error': (255, 0, 0),
                'text_info': (0, 255, 255),
                'accent': (255, 0, 255)
            },
            'ocean': {
                'bg_primary': (40, 20, 10),
                'bg_secondary': (60, 40, 20),
                'border': (150, 100, 50),
                'text_primary': (200, 180, 150),
                'text_success': (100, 200, 150),
                'text_warning': (200, 150, 100),
                'text_error': (200, 100, 100),
                'text_info': (150, 180, 200),
                'accent': (100, 200, 255)
            },
            'neon': {
                'bg_primary': (10, 10, 20),
                'bg_secondary': (20, 20, 40),
                'border': (100, 50, 200),
                'text_primary': (50, 255, 150),
                'text_success': (0, 255, 100),
                'text_warning': (255, 100, 0),
                'text_error': (255, 0, 100),
                'text_info': (0, 200, 255),
                'accent': (200, 50, 255)
            }
        }
        self.current_theme = 'default'

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

        # Dibujar tutorial si está activo
        if self.tutorial_active:
            self._draw_tutorial(frame, app_state)

        # Dibujar tooltip si está activo
        if self.tooltip_active:
            self._draw_tooltip(frame)
            self.update_tooltip()

        # Mostrar tooltips contextuales automáticos
        self.show_contextual_tooltip(app_state)

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
        min_points = app_state.get('min_points_for_classification', 10)

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
        confidence_level = self._calculate_confidence_level(has_hands, stroke_points_count, min_points)
        confidence_colors = ['○', '●', '●', '●']  # Estados: vacío, bajo, medio, alto
        confidence_text = f"Confianza: {confidence_colors[min(confidence_level, 3)] * (confidence_level + 1)}"
        confidence_color = [self.theme['text_error'], self.theme['text_warning'],
                           self.theme['text_info'], self.theme['text_success']][min(confidence_level, 3)]
        cv2.putText(frame, confidence_text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)

        # Estado de dibujo
        y_offset += 35
        is_drawing = app_state.get('is_drawing', False)

        if is_drawing:
            # Efecto de pulso animado más sofisticado
            pulse_intensity = (np.sin(time.time() * 6) + 1) / 2  # Oscilación más rápida
            draw_color = tuple(int(self.theme['accent'][i] * (0.5 + pulse_intensity * 0.5)) for i in range(3))
            draw_status = f"✏️ {_('DIBUJANDO...')}"

            # Barra de progreso con gradiente
            progress = min(stroke_points_count / min_points, 1.0)
            bar_width = int(progress * 200)

            # Gradiente en la barra de progreso
            for i in range(bar_width):
                gradient_factor = i / 200.0
                bar_color = tuple(int(self.theme['text_success'][j] * gradient_factor +
                                    self.theme['accent'][j] * (1 - gradient_factor)) for j in range(3))
                cv2.line(frame, (10 + i, y_offset + 5), (10 + i, y_offset + 15), bar_color)

            cv2.rectangle(frame, (10, y_offset + 5), (210, y_offset + 15), self.theme['border'], 1)

            # Indicador de progreso numérico
            progress_text = f"{stroke_points_count}/{min_points}"
            cv2.putText(frame, progress_text, (220, y_offset + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text_info'], 1)

            # Partículas de "polvo" para efecto visual
            self._draw_drawing_particles(frame, stroke_points_count, y_offset + 20)
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
            ("H", _("Mostrar/ocultar ayuda")),
            ("T", _("Iniciar tutorial")),
            ("F1-F3", _("Cambiar tema (F1=Normal, F2=Oscuro, F3=Alto Contraste)"))
        ]

        for i, (key, desc) in enumerate(controls):
            y_pos = controls_y + i * 18
            key_text = f"[{key}]"
            cv2.putText(frame, key_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['accent'], 1)
            cv2.putText(frame, desc, (120, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text_primary'], 1)

        # Tips
        tips = [
            _("Muestra tu mano completa a la cámara"),
            _("Dibuja con el dedo índice extendido"),
            _("Mantén buena iluminación para mejor detección"),
            _("Presiona T para ver el tutorial interactivo"),
            _("Usa F1-F3 para cambiar temas de color")
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

    def show_tooltip(self, text: str, position: Tuple[int, int], duration: float = 3.0) -> None:
        """
        Muestra un tooltip en la posición especificada.

        Args:
            text: Texto del tooltip
            position: Posición (x, y) donde mostrar el tooltip
            duration: Duración en segundos
        """
        self.tooltip_active = True
        self.tooltip_text = text
        self.tooltip_position = position
        self.tooltip_start_time = time.time()
        self.tooltip_duration = duration

    def hide_tooltip(self) -> None:
        """Oculta el tooltip actual."""
        self.tooltip_active = False

    def update_tooltip(self) -> None:
        """Actualiza el estado del tooltip (llamar en cada frame)."""
        if self.tooltip_active and time.time() - self.tooltip_start_time > self.tooltip_duration:
            self.tooltip_active = False

    def start_tutorial(self) -> None:
        """Inicia el tutorial interactivo."""
        self.tutorial_active = True
        self.tutorial_step = 0

    def next_tutorial_step(self) -> None:
        """Avanza al siguiente paso del tutorial."""
        if self.tutorial_step < len(self.tutorial_steps) - 1:
            self.tutorial_step += 1
        else:
            self.end_tutorial()

    def end_tutorial(self) -> None:
        """Finaliza el tutorial."""
        self.tutorial_active = False
        self.tutorial_step = 0

    def switch_theme(self, theme_name: str) -> None:
        """
        Cambia el tema de colores.

        Args:
            theme_name: Nombre del tema ('default', 'dark', 'high_contrast')
        """
        if theme_name in self.available_themes:
            self.theme = self.available_themes[theme_name].copy()
            self.current_theme = theme_name

    def _draw_drawing_particles(self, frame: np.ndarray, point_count: int, y_offset: int) -> None:
        """Dibuja partículas decorativas durante el dibujo."""
        # Crear efecto de "polvo mágico" con puntos aleatorios
        np.random.seed(int(time.time() * 10) % (2**32 - 1))  # Semilla basada en tiempo para animación, limitada al rango válido

        for i in range(min(point_count // 2, 10)):  # Máximo 10 partículas
            x = np.random.randint(10, 250)
            y = y_offset + np.random.randint(-5, 5)
            size = np.random.randint(1, 3)
            alpha = np.random.random() * 0.5 + 0.5  # Transparencia variable

            color = tuple(int(self.theme['accent'][j] * alpha) for j in range(3))
            cv2.circle(frame, (x, y), size, color, -1)

    def _draw_tooltip(self, frame: np.ndarray) -> None:
        """Dibuja el tooltip actual si está activo."""
        if not self.tooltip_active:
            return

        x, y = self.tooltip_position
        text_size = cv2.getTextSize(self.tooltip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        padding = 8

        # Fondo del tooltip
        bg_width = text_size[0] + padding * 2
        bg_height = text_size[1] + padding * 2
        bg_x = max(0, x - bg_width // 2)
        bg_y = y - bg_height - 10

        # Asegurar que el tooltip quepa en la pantalla
        height, width = frame.shape[:2]
        bg_x = max(0, min(bg_x, width - bg_width))
        bg_y = max(0, bg_y)

        cv2.rectangle(frame, (bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height),
                     self.theme['bg_secondary'], -1)
        cv2.rectangle(frame, (bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height),
                     self.theme['border'], 1)

        # Texto del tooltip
        text_x = bg_x + padding
        text_y = bg_y + padding + text_size[1]
        cv2.putText(frame, self.tooltip_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text_primary'], 1)

    def _draw_tutorial(self, frame: np.ndarray, app_state: dict) -> None:
        """Dibuja el tutorial interactivo."""
        height, width = frame.shape[:2]

        # Overlay semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # Panel del tutorial
        tutorial_width = 400
        tutorial_height = 150
        tutorial_x = (width - tutorial_width) // 2
        tutorial_y = (height - tutorial_height) // 2

        cv2.rectangle(frame, (tutorial_x, tutorial_y),
                     (tutorial_x + tutorial_width, tutorial_y + tutorial_height),
                     self.theme['bg_primary'], -1)
        cv2.rectangle(frame, (tutorial_x, tutorial_y),
                     (tutorial_x + tutorial_width, tutorial_y + tutorial_height),
                     self.theme['accent'], 2)

        # Título
        title = _("Tutorial Interactivo")
        cv2.putText(frame, title, (tutorial_x + 20, tutorial_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme['text_primary'], 2)

        # Paso actual
        step_text = f"{_('Paso')} {self.tutorial_step + 1}/{len(self.tutorial_steps)}"
        cv2.putText(frame, step_text, (tutorial_x + tutorial_width - 100, tutorial_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text_info'], 1)

        # Contenido del paso
        current_step = self.tutorial_steps[self.tutorial_step]
        lines = self._wrap_text(current_step, tutorial_width - 40)

        for i, line in enumerate(lines):
            y_pos = tutorial_y + 60 + i * 25
            cv2.putText(frame, line, (tutorial_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text_primary'], 1)

        # Indicador de progreso
        progress = (self.tutorial_step + 1) / len(self.tutorial_steps)
        progress_width = int(progress * (tutorial_width - 40))
        cv2.rectangle(frame, (tutorial_x + 20, tutorial_y + tutorial_height - 30),
                     (tutorial_x + 20 + progress_width, tutorial_y + tutorial_height - 20),
                     self.theme['accent'], -1)
        cv2.rectangle(frame, (tutorial_x + 20, tutorial_y + tutorial_height - 30),
                     (tutorial_x + tutorial_width - 20, tutorial_y + tutorial_height - 20),
                     self.theme['border'], 1)

        # Instrucción
        instruction = _("Presiona ESPACIO para continuar, T para saltar tutorial")
        cv2.putText(frame, instruction, (tutorial_x + 20, tutorial_y + tutorial_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text_info'], 1)

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Envuelve el texto para que quepa en el ancho especificado."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]

            if text_size[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def _calculate_confidence_level(self, has_hands: bool, stroke_points: int, min_points: int) -> int:
        """
        Calcula el nivel de confianza basado en el estado actual.

        Returns:
            Nivel de confianza (0-3): 0=ninguno, 1=bajo, 2=medio, 3=alto
        """
        level = 0
        if has_hands:
            level += 1
        if stroke_points >= min_points // 2:
            level += 1
        if stroke_points >= min_points:
            level += 1
        return level

    def show_contextual_tooltip(self, app_state: dict, mouse_pos: Optional[Tuple[int, int]] = None) -> None:
        """
        Muestra tooltips contextuales basados en el estado de la aplicación.

        Args:
            app_state: Estado actual de la aplicación
            mouse_pos: Posición del mouse (si está disponible)
        """
        has_hands = app_state.get('has_hands', False)
        is_drawing = app_state.get('is_drawing', False)
        stroke_points_count = len(app_state.get('stroke_points', []))
        min_points = app_state.get('min_points_for_classification', 10)

        # Tooltip para detección de manos
        if not has_hands and not self.tooltip_active:
            self.show_tooltip(_("Muestra tu mano completa a la cámara"),
                            (50, 100), 2.0)
            return

        # Tooltip para empezar a dibujar
        if has_hands and not is_drawing and stroke_points_count == 0 and not self.tooltip_active:
            self.show_tooltip(_("Extiende tu dedo índice para empezar a dibujar"),
                            (50, 120), 3.0)
            return

        # Tooltip para progreso de dibujo
        if is_drawing and stroke_points_count < min_points and not self.tooltip_active:
            remaining = min_points - stroke_points_count
            self.show_tooltip(_("Continúa dibujando... necesitas {remaining} puntos más").format(remaining=remaining),
                            (50, 140), 2.0)
            return

        # Tooltip para clasificación lista
        if is_drawing and stroke_points_count >= min_points and not self.tooltip_active:
            self.show_tooltip(_("¡Perfecto! Presiona ESPACIO para clasificar"),
                            (50, 160), 3.0)
            return

    def handle_keyboard_shortcut(self, key: int) -> str:
        """
        Maneja atajos de teclado y retorna la acción correspondiente.

        Args:
            key: Código de tecla presionada

        Returns:
            Acción a realizar ('space', 'r', 'q', 'h', 't', 'f1', 'f2', 'f3', 'f4', 'f5', etc.)
        """
        key_char = chr(key).lower() if key < 256 else ''

        if key == 32:  # ESPACIO
            return 'space'
        elif key_char == 'r':
            return 'r'
        elif key_char == 'q':
            return 'q'
        elif key_char == 'h':
            return 'h'
        elif key_char == 't':
            return 't'
        elif key == 112:  # F1
            return 'f1'
        elif key == 113:  # F2
            return 'f2'
        elif key == 114:  # F3
            return 'f3'
        elif key == 115:  # F4
            return 'f4'
        elif key == 116:  # F5
            return 'f5'
        elif key == 27:  # ESC
            return 'escape'

        return ''

    def process_ui_action(self, action: str) -> None:
        """
        Procesa acciones de UI basadas en atajos de teclado.

        Args:
            action: Acción a procesar
        """
        if action == 'h':
            self.toggle_help()
        elif action == 't':
            if self.tutorial_active:
                self.end_tutorial()
            else:
                self.start_tutorial()
        elif action == 'f1':
            self.switch_theme('default')
        elif action == 'f2':
            self.switch_theme('dark')
        elif action == 'f3':
            self.switch_theme('high_contrast')
        elif action == 'f4':
            self.switch_theme('ocean')
        elif action == 'f5':
            self.switch_theme('neon')
        elif action == 'escape':
            if self.tutorial_active:
                self.end_tutorial()
            elif self.tooltip_active:
                self.hide_tooltip()

    def get_accessibility_info(self) -> str:
        """
        Retorna información de accesibilidad para lectores de pantalla.

        Returns:
            Texto descriptivo del estado actual de la UI
        """
        info_parts = []

        if self.tutorial_active:
            info_parts.append(f"Tutorial activo, paso {self.tutorial_step + 1} de {len(self.tutorial_steps)}")

        info_parts.append(f"Tema actual: {self.current_theme}")
        info_parts.append(f"Panel de ayuda: {'visible' if self.show_help else 'oculto'}")

        if self.tooltip_active:
            info_parts.append(f"Tooltip: {self.tooltip_text}")

        return ". ".join(info_parts)

    def get_available_themes(self) -> List[str]:
        """Retorna lista de temas disponibles."""
        return list(self.available_themes.keys())

    def draw_status_indicator(self, frame: np.ndarray, status: str, position: Tuple[int, int],
                             status_color: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Dibuja un indicador de estado personalizado.

        Args:
            frame: Frame donde dibujar
            status: Texto del estado
            position: Posición (x, y)
            status_color: Color BGR (usa color_info por defecto)
        """
        if status_color is None:
            status_color = self.theme['text_info']

        # Fondo redondeado simulado
        text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        padding = 8
        bg_width = text_size[0] + padding * 2
        bg_height = text_size[1] + padding * 2

        x, y = position
        cv2.rectangle(frame, (x, y), (x + bg_width, y + bg_height),
                     self.theme['bg_secondary'], -1)
        cv2.rectangle(frame, (x, y), (x + bg_width, y + bg_height),
                     status_color, 1)

        cv2.putText(frame, status, (x + padding, y + padding + text_size[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    def draw_progress_bar(self, frame: np.ndarray, progress: float, position: Tuple[int, int],
                         width: int = 200, height: int = 20) -> None:
        """
        Dibuja una barra de progreso mejorada.

        Args:
            frame: Frame donde dibujar
            progress: Valor de progreso (0.0-1.0)
            position: Posición (x, y) superior izquierda
            width: Ancho de la barra
            height: Alto de la barra
        """
        x, y = position
        progress = max(0.0, min(1.0, progress))

        # Fondo
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.theme['bg_secondary'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height),
                     self.theme['border'], 2)

        # Barra de progreso con gradiente
        progress_width = int(progress * width)
        if progress_width > 0:
            cv2.rectangle(frame, (x, y), (x + progress_width, y + height),
                         self.theme['text_success'], -1)

        # Texto de porcentaje
        percent_text = f"{int(progress * 100)}%"
        text_size = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2

        cv2.putText(frame, percent_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text_primary'], 1)