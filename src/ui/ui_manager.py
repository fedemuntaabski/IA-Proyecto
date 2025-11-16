"""
UI Manager - Gestor de Interfaz de Usuario para Air Draw Classifier.

Este módulo maneja toda la lógica de dibujo de la interfaz gráfica,
manteniendo la separación de responsabilidades y siguiendo el principio KISS.
"""

import cv2
import numpy as np
import time
from typing import List, Optional, Tuple
from ..core.i18n import _
from ..core.utils.constants import (COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_INFO, COLOR_TEXT_PRIMARY,
                       COLOR_ACCENT, COLOR_BG_PRIMARY, COLOR_BG_SECONDARY, COLOR_BORDER,
                       DEFAULT_BAR_HEIGHT)
# Import lazy para evitar problemas de importación circular
# from ..core.utils.feedback_manager import feedback_manager

def _get_feedback_manager():
    """Obtiene el feedback_manager de manera lazy para evitar importaciones circulares."""
    from ..core.utils.feedback_manager import feedback_manager
    return feedback_manager


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
        self.show_user_profile = False
        self.current_fps = 0.0
        
        # Estado de feedback
        self.feedback_mode = False
        self.feedback_buttons = []
        self.current_prediction_data = None
        
        # Mejoras de UI - Estado expandido
        self.show_advanced_info = False
        self.animation_frame = 0
        self.tutorial_mode = False

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
        self.animation_frame = (self.animation_frame + 1) % 60

        # Dibujar barra superior mejorada
        self._draw_top_bar(frame, app_state)

        # Dibujar predicción si existe
        self._draw_prediction(frame, app_state)

        # Dibujar panel de información en tiempo real
        self._draw_realtime_info(frame, app_state)

        # Dibujar panel de ayuda si está habilitado
        if self.show_help:
            self._draw_help_panel(frame, app_state)

        # Dibujar perfil de usuario si está habilitado
        if self.show_user_profile:
            frame = self.draw_user_profile(frame)

        return frame
    
    def _draw_realtime_info(self, frame: np.ndarray, app_state: dict) -> None:
        """Dibuja información en tiempo real en la esquina inferior izquierda."""
        height, width = frame.shape[:2]
        info_height = 90
        info_x, info_y = 10, height - info_height - 10
        
        # Fondo semi-transparente
        cv2.rectangle(frame, (info_x, info_y), (info_x + 250, info_y + info_height),
                     self.theme['bg_secondary'], -1)
        cv2.rectangle(frame, (info_x, info_y), (info_x + 250, info_y + info_height),
                     self.theme['border'], 2)
        
        # Información de dibujo
        stroke_points = len(app_state.get('stroke_points', []))
        min_points = app_state.get('min_points_for_classification', 10)
        
        y_offset = info_y + 20
        
        # Estado de puntos
        status_icon = "●" if stroke_points > 0 else "○"
        cv2.putText(frame, f"{status_icon} Puntos: {stroke_points}/{min_points}", 
                   (info_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.theme['text_success'] if stroke_points >= min_points else self.theme['text_warning'], 1)
        
        # Predicciones totales
        total_drawings = app_state.get('total_drawings', 0)
        successful = app_state.get('successful_predictions', 0)
        success_rate = (successful / total_drawings * 100) if total_drawings > 0 else 0
        
        y_offset += 20
        cv2.putText(frame, f"📊 Dibujos: {total_drawings} (✓ {successful})", 
                   (info_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.theme['text_info'], 1)
        
        # Tasa de éxito
        y_offset += 20
        success_color = self.theme['text_success'] if success_rate >= 70 else self.theme['text_warning']
        cv2.putText(frame, f"Éxito: {success_rate:.0f}%", 
                   (info_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   success_color, 1)
        
        # Async predicciones pendientes
        y_offset += 20
        pending = app_state.get('pending_predictions_count', 0)
        if pending > 0:
            cv2.putText(frame, f"⏳ Procesando: {pending}", 
                       (info_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.theme['accent'], 1)

    def _draw_top_bar(self, frame: np.ndarray, app_state: dict) -> None:
        """Dibuja la barra superior con información de estado mejorada."""
        height, width = frame.shape[:2]

        # Fondo de la barra con gradiente
        bar_height = DEFAULT_BAR_HEIGHT
        for i in range(bar_height):
            alpha = i / bar_height
            color = tuple(int(self.theme['bg_primary'][j] * (1-alpha) + self.theme['bg_secondary'][j] * alpha) for j in range(3))
            cv2.line(frame, (0, i), (width, i), color)

        cv2.rectangle(frame, (0, 0), (width, bar_height), self.theme['border'], 2)

        # Estado de detección con animación
        y_offset = 25
        has_hands = app_state.get('has_hands', False)
        stroke_points_count = len(app_state.get('stroke_points', []))

        # Indicador de mano detectada (pulsante)
        pulse = int(self.animation_frame / 15) % 2
        indicator_color = self.theme['text_success'] if pulse and has_hands else (self.theme['text_error'] if has_hands else self.theme['text_error'])
        
        if has_hands:
            cv2.circle(frame, (width-30, 20), 8, indicator_color, -1)
            cv2.circle(frame, (width-30, 20), 10, indicator_color, 2)
            status_text = f"👋 {_('Mano detectada')} | Pts: {stroke_points_count}"
            status_color = self.theme['text_success']
        else:
            cv2.circle(frame, (width-30, 20), 8, self.theme['text_warning'], -1)
            status_text = _("Mueve tu mano a la cámara")
            status_color = self.theme['text_warning']

        cv2.putText(frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # FPS con color dinámico
        fps_text = f"FPS: {self.current_fps:.0f}"
        if self.current_fps >= 30:
            fps_color = self.theme['text_success']
        elif self.current_fps >= 20:
            fps_color = self.theme['text_info']
        elif self.current_fps >= 15:
            fps_color = self.theme['text_warning']
        else:
            fps_color = self.theme['text_error']
        
        cv2.putText(frame, fps_text, (width - 120, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

        # Estado de dibujo con barra de progreso
        y_offset += 25
        is_drawing = app_state.get('is_drawing', False)
        min_points = app_state.get('min_points_for_classification', 10)

        if is_drawing:
            draw_status = f"✏️ {_('DIBUJANDO...')}"
            draw_color = self.theme['accent']
            
            # Barra de progreso mejorada
            progress = min(stroke_points_count / min_points, 1.0)
            bar_width = int(progress * 300)
            
            # Fondo de la barra
            cv2.rectangle(frame, (10, y_offset + 5), (310, y_offset + 15), self.theme['border'], 1)
            # Barra de progreso
            cv2.rectangle(frame, (10, y_offset + 5), (10 + bar_width, y_offset + 15), draw_color, -1)
            
            # Porcentaje
            progress_pct = f"{progress*100:.0f}%"
            cv2.putText(frame, progress_pct, (315, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
        else:
            draw_status = f"✅ {_('Listo para dibujar')}"
            draw_color = self.theme['text_success']

        cv2.putText(frame, draw_status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, draw_color, 2)

    def _draw_prediction(self, frame: np.ndarray, app_state: dict) -> None:
        """Dibuja la predicción actual con interfaz mejorada."""
        prediction = app_state.get('last_prediction')
        if not prediction:
            return

        class_name, confidence, pred_time = prediction
        if time.time() - pred_time > 8.0:  # Expirar después de 8 segundos
            self.feedback_mode = False
            self.feedback_buttons = []
            self.current_prediction_data = None
            return

        height, width = frame.shape[:2]
        bar_height = DEFAULT_BAR_HEIGHT

        # Fondo para la predicción mejorado
        pred_bg_width = 350
        pred_bg_height = 95
        pred_x = width - pred_bg_width - 15
        pred_y = bar_height + 10

        # Fondo semi-transparente con bordes redondeados (simulados)
        cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_bg_width, pred_y + pred_bg_height),
                     self.theme['bg_secondary'], -1)
        cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_bg_width, pred_y + pred_bg_height),
                     self.theme['border'], 2)

        # Ícono de predicción pulsante
        pulse = int(self.animation_frame / 10) % 2
        icon_color = self.theme['accent'] if pulse else self.theme['text_success']
        
        # Texto de predicción con tamaño aumentado
        pred_text = f"🎯 {class_name}"
        conf_text = f"Confianza: {confidence:.1%}"

        cv2.putText(frame, pred_text, (pred_x + 15, pred_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, icon_color, 2)
        cv2.putText(frame, conf_text, (pred_x + 15, pred_y + 48),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text_info'], 1)

        # Barra de confianza visual mejorada
        conf_bar_width = int(confidence * (pred_bg_width - 30))
        cv2.rectangle(frame, (pred_x + 15, pred_y + 55), (pred_x + 15 + conf_bar_width, pred_y + 62),
                     icon_color, -1)
        cv2.rectangle(frame, (pred_x + 15, pred_y + 55), (pred_x + pred_bg_width - 15, pred_y + 62),
                     self.theme['border'], 1)

        # Botones de feedback
        self._draw_feedback_buttons(frame, pred_x, pred_y + 65, pred_bg_width, class_name, confidence, app_state)

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
            ("C", _("Modo feedback")),
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

    def toggle_user_profile(self) -> None:
        """Alterna la visibilidad del perfil de usuario."""
        self.show_user_profile = not self.show_user_profile

    def update_fps(self, fps: float) -> None:
        """Actualiza el valor de FPS para mostrar."""
        self.current_fps = fps

    def _draw_feedback_buttons(self, frame: np.ndarray, x: int, y: int, width: int,
                              class_name: str, confidence: float, app_state: dict) -> None:
        """
        Dibuja los botones de feedback para corrección manual.

        Args:
            frame: Frame donde dibujar
            x, y: Posición del área de botones
            width: Ancho disponible
            class_name: Nombre de la clase predicha
            confidence: Confianza de la predicción
            app_state: Estado de la aplicación
        """
        # Solo mostrar botones si la confianza es baja o el usuario presiona una tecla
        show_buttons = (confidence < 0.8 or self.feedback_mode) and app_state.get('current_gesture_image')

        if not show_buttons:
            # Mostrar indicador de que se puede corregir
            hint_text = _("Presiona 'C' para corregir")
            cv2.putText(frame, hint_text, (x + 10, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text_info'], 1)
            return

        # Verificar que tenemos datos de gesto
        gesture_data = app_state.get('current_gesture_image', [])
        if not gesture_data:
            return

        # Activar modo feedback
        self.feedback_mode = True

        # Preparar datos para feedback
        self.current_prediction_data = {
            'original_prediction': class_name,
            'confidence': confidence,
            'gesture_data': gesture_data
        }

        # Botones
        button_width = 80
        button_height = 25
        button_spacing = 10

        buttons = [
            ("✓", _("Correcto"), self.theme['text_success']),
            ("✗", _("Incorrecto"), self.theme['text_error']),
        ]

        # Obtener sugerencias de corrección
        suggestions = _get_feedback_manager().get_correction_suggestions(class_name, limit=2)
        for suggestion in suggestions[:2]:  # Máximo 2 sugerencias
            buttons.append(("💡", suggestion, self.theme['accent']))

        # Dibujar botones
        self.feedback_buttons = []
        current_x = x + 10

        for i, (icon, label, color) in enumerate(buttons):
            if current_x + button_width > x + width - 10:
                break  # No hay espacio para más botones

            # Número del botón para teclado
            button_number = str(i + 1)

            # Fondo del botón
            cv2.rectangle(frame, (current_x, y), (current_x + button_width, y + button_height),
                         self.theme['bg_primary'], -1)
            cv2.rectangle(frame, (current_x, y), (current_x + button_width, y + button_height),
                         color, 2)

            # Texto del botón
            text = f"{button_number}. {icon} {label}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            text_x = current_x + (button_width - text_size[0]) // 2
            text_y = y + (button_height + text_size[1]) // 2

            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Guardar información del botón
            self.feedback_buttons.append({
                'rect': (current_x, y, current_x + button_width, y + button_height),
                'action': label,
                'icon': icon,
                'color': color,
                'number': button_number
            })

            current_x += button_width + button_spacing

        # Instrucciones
        instructions_y = y + button_height + 20
        cv2.putText(frame, _("Presiona el número del botón para seleccionar"),
                   (x + 10, instructions_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text_info'], 1)

    def handle_feedback_click(self, click_x: int, click_y: int) -> Optional[str]:
        """
        Maneja un clic en el área de feedback.

        Args:
            click_x, click_y: Coordenadas del clic

        Returns:
            Acción del botón clickeado, o None si no se clickeó ningún botón
        """
        if not self.feedback_mode or not self.feedback_buttons:
            return None

        for button in self.feedback_buttons:
            bx1, by1, bx2, by2 = button['rect']
            if bx1 <= click_x <= bx2 and by1 <= click_y <= by2:
                return button['action']

        return None

    def submit_feedback(self, correction: str) -> bool:
        """
        Envía una corrección de feedback.

        Args:
            correction: La corrección seleccionada

        Returns:
            True si se envió correctamente
        """
        if not self.current_prediction_data:
            return False

        data = self.current_prediction_data

        # Si es "Correcto", no necesitamos hacer nada más
        if correction == _("Correcto"):
            print("✅ Predicción confirmada como correcta")
            self._reset_feedback()
            return True

        # Si es "Incorrecto", pedir nueva clase por teclado
        if correction == _("Incorrecto"):
            print("❌ Predicción marcada como incorrecta")
            print("📝 Ingresa la clase correcta (o presiona ENTER para cancelar):")
            # Aquí activaríamos un modo de input de texto
            # Por ahora, solo registramos que fue incorrecta
            self._reset_feedback()
            return True

        # Si es una sugerencia, usar esa clase
        corrected_class = correction

        # Enviar feedback
        success = _get_feedback_manager().add_correction(
            original_prediction=data['original_prediction'],
            original_confidence=data['confidence'],
            corrected_class=corrected_class,
            gesture_image_data=data['gesture_data']
        )

        if success:
            print(f"📝 Feedback enviado: {data['original_prediction']} → {corrected_class}")
            
            # Mostrar información de gamificación
            user_level = _get_feedback_manager().get_user_level("current_user")
            print(f"🎮 Nivel: {user_level['level_name']} | Puntos: {user_level['points']} | Próximo: {user_level['points_to_next']} pts")

        self._reset_feedback()
        return success

    def toggle_feedback_mode(self) -> None:
        """Alterna el modo de feedback manualmente."""
        self.feedback_mode = not self.feedback_mode
        if not self.feedback_mode:
            self._reset_feedback()

    def handle_feedback_key(self, key: int) -> bool:
        """
        Maneja teclas cuando está en modo feedback.

        Args:
            key: Código de tecla

        Returns:
            True si se manejó la tecla, False en caso contrario
        """
        if not self.feedback_mode or not self.feedback_buttons:
            return False

        # Mapear números a botones
        key_map = {
            ord('1'): 0,  # Primer botón
            ord('2'): 1,  # Segundo botón
            ord('3'): 2,  # Tercer botón
            ord('4'): 3,  # Cuarto botón
        }

        if key in key_map and key_map[key] < len(self.feedback_buttons):
            button = self.feedback_buttons[key_map[key]]
            self.submit_feedback(button['action'])
            return True

        return False

    def draw_user_profile(self, frame: np.ndarray, user_id: str = "current_user") -> np.ndarray:
        """
        Dibuja el perfil de usuario con estadísticas de gamificación.

        Args:
            frame: Frame donde dibujar
            user_id: ID del usuario

        Returns:
            Frame con perfil dibujado
        """
        frame_copy = frame.copy()
        height, width = frame.shape[:2]

        # Fondo del perfil
        profile_width = 400
        profile_height = 200
        profile_x = (width - profile_width) // 2
        profile_y = (height - profile_height) // 2

        cv2.rectangle(frame_copy, (profile_x, profile_y),
                     (profile_x + profile_width, profile_y + profile_height),
                     self.theme['bg_secondary'], -1)
        cv2.rectangle(frame_copy, (profile_x, profile_y),
                     (profile_x + profile_width, profile_y + profile_height),
                     self.theme['border'], 2)

        # Título
        cv2.putText(frame_copy, _("PERFIL DE USUARIO"), (profile_x + 20, profile_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.theme['text_primary'], 2)

        # Obtener datos del usuario
        user_level = _get_feedback_manager().get_user_level(user_id)
        leaderboard = _get_feedback_manager().get_leaderboard(5)

        # Encontrar posición en leaderboard
        user_position = None
        for i, entry in enumerate(leaderboard):
            if entry['user_id'] == user_id:
                user_position = i + 1
                break

        y_offset = profile_y + 60
        line_height = 25

        # Información del nivel
        cv2.putText(frame_copy, f"🏆 {_('Nivel')}: {user_level['level_name']} ({user_level['level']})",
                   (profile_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text_success'], 1)

        y_offset += line_height
        cv2.putText(frame_copy, f"⭐ {_('Puntos')}: {user_level['points']}",
                   (profile_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text_info'], 1)

        y_offset += line_height
        cv2.putText(frame_copy, f"📝 {_('Correcciones')}: {user_level['total_corrections']}",
                   (profile_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['text_info'], 1)

        if user_position:
            y_offset += line_height
            cv2.putText(frame_copy, f"🏅 {_('Posición')}: #{user_position} de {len(leaderboard)}",
                       (profile_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme['accent'], 1)

        # Barra de progreso hacia siguiente nivel
        if user_level['points_to_next'] > 0:
            y_offset += line_height + 10
            progress_width = profile_width - 40
            current_level_points = user_level['points'] - self._get_level_threshold(user_level['level'])
            next_level_points = self._get_level_threshold(user_level['next_level']) - self._get_level_threshold(user_level['level'])

            if next_level_points > 0:
                progress = min(1.0, current_level_points / next_level_points)
                bar_width = int(progress * progress_width)

                cv2.rectangle(frame_copy, (profile_x + 20, y_offset), (profile_x + 20 + progress_width, y_offset + 10),
                             self.theme['border'], 1)
                cv2.rectangle(frame_copy, (profile_x + 20, y_offset), (profile_x + 20 + bar_width, y_offset + 10),
                             self.theme['accent'], -1)

                progress_text = f"{user_level['points_to_next']} pts para {user_level['next_level_name']}"
                cv2.putText(frame_copy, progress_text, (profile_x + 20, y_offset - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.theme['text_info'], 1)

        # Instrucción para cerrar
        cv2.putText(frame_copy, _("Presiona 'ESC' para cerrar"),
                   (profile_x + 20, profile_y + profile_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme['text_warning'], 1)

        return frame_copy

    def _get_level_threshold(self, level: int) -> int:
        """Obtiene el umbral de puntos para un nivel."""
        return _get_feedback_manager().gamification['levels'].get(level, {}).get('threshold', 0)
    
    def update_fps(self, fps: float) -> None:
        """
        Actualiza el valor de FPS para mostrar en la UI.
        
        Args:
            fps: Frames por segundo actuales
        """
        self.current_fps = fps