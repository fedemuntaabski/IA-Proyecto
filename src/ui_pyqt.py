"""
ui_pyqt.py - Interfaz de usuario moderna con PyQt6

UI principal para Pictionary Live usando PyQt6 con diseño cyberpunk moderno.
Mantiene toda la funcionalidad de la UI original con overlays mejorados.
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QGraphicsDropShadowEffect, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QEvent
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPainterPath, QScreen
import cv2


# Paleta Cyberpunk (compatible con la original)
COLORS = {
    "bg_panel": QColor(10, 20, 40),
    "bg_card": QColor(25, 45, 60),
    "accent": QColor(255, 255, 0),
    "accent2": QColor(255, 160, 0),
    "success": QColor(100, 255, 100),
    "text_main": QColor(235, 235, 235),
    "text_dim": QColor(180, 180, 190),
    "warning": QColor(0, 100, 255),
}


class VideoWidget(QLabel):
    """Widget personalizado para mostrar video con overlays."""
    
    # Señal para emitir puntos del mouse
    mouse_draw = pyqtSignal(float, float, bool)  # (x, y, is_drawing)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #0a1428; border-radius: 10px;")
        self.current_frame = None
        self.stroke_points = []
        self.mouse_drawing = False
        self.show_debug_guide = False  # Debug mode for drawing guide
        self.setMouseTracking(True)  # Habilitar tracking del mouse
        
    def set_frame(self, frame: np.ndarray):
        """Actualiza el frame mostrado."""
        self.current_frame = frame
        self.update()
    
    def set_stroke_points(self, points: List[Tuple[float, float]]):
        """Establece los puntos del trazo actual."""
        self.stroke_points = points
        self.update()
    
    def mousePressEvent(self, event):
        """Maneja el evento de presionar el mouse."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_drawing = True
            self._emit_mouse_position(event.position(), is_drawing=True)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Maneja el movimiento del mouse."""
        # Only draw if left button is being held down
        if self.mouse_drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            self._emit_mouse_position(event.position(), is_drawing=True)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Maneja el evento de soltar el mouse."""
        if event.button() == Qt.MouseButton.LeftButton and self.mouse_drawing:
            self.mouse_drawing = False
            self.mouse_draw.emit(0.0, 0.0, False)  # Señal de fin de trazo
        super().mouseReleaseEvent(event)
    
    def _emit_mouse_position(self, pos, is_drawing=True):
        """Emite la posición del mouse normalizada."""
        # Calcular posición en el frame (normalizada 0-1)
        if self.current_frame is None:
            return
        
        # Obtener dimensiones del frame escalado
        h, w = self.current_frame.shape[:2]
        aspect = w / h
        widget_aspect = self.width() / self.height()
        
        if widget_aspect > aspect:
            # Widget más ancho que el frame
            scaled_h = self.height()
            scaled_w = int(scaled_h * aspect)
        else:
            # Widget más alto que el frame
            scaled_w = self.width()
            scaled_h = int(scaled_w / aspect)
        
        x_offset = (self.width() - scaled_w) // 2
        y_offset = (self.height() - scaled_h) // 2
        
        # Convertir a coordenadas del frame
        x = pos.x() - x_offset
        y = pos.y() - y_offset
        
        # Normalizar a [0, 1]
        if 0 <= x <= scaled_w and 0 <= y <= scaled_h:
            norm_x = x / scaled_w
            norm_y = y / scaled_h
            self.mouse_draw.emit(norm_x, norm_y, is_drawing)
    
    def paintEvent(self, event):
        """Dibuja el frame y overlays."""
        super().paintEvent(event)
        
        if self.current_frame is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Convertir frame a QPixmap
        h, w, ch = self.current_frame.shape
        bytes_per_line = ch * w
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Escalar para ajustar al widget
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Centrar y dibujar frame
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        y_offset = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        # Draw debug guide if enabled (dotted gray box showing model input area)
        if self.show_debug_guide:
            self._draw_debug_guide(painter, x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height())
        
        # Dibujar trazos si existen
        if self.stroke_points and len(self.stroke_points) >= 2:
            self._draw_strokes(painter, x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height())
    
    def _draw_strokes(self, painter: QPainter, x_offset: int, y_offset: int, w: int, h: int):
        """Dibuja los trazos del usuario como línea negra de 8px."""
        # Línea negra de 8px (como se dibuja en el canvas)
        pen_main = QPen(QColor(0, 0, 0), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_main)
        
        path = QPainterPath()
        first_point = True
        for px, py in self.stroke_points:
            x = int(px * w) + x_offset
            y = int(py * h) + y_offset
            if first_point:
                path.moveTo(x, y)
                first_point = False
            else:
                path.lineTo(x, y)
        
        painter.drawPath(path)
        
        # Punto final en rojo para indicar posición actual del dedo
        if self.stroke_points:
            fx, fy = self.stroke_points[-1]
            fx = int(fx * w) + x_offset
            fy = int(fy * h) + y_offset
            painter.setBrush(QColor(255, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(fx - 6, fy - 6, 12, 12)
    
    def _draw_debug_guide(self, painter: QPainter, x_offset: int, y_offset: int, w: int, h: int):
        """
        Draws an enhanced guide showing the optimal drawing area for model prediction.
        
        This guide helps users understand:
        1. Where to draw for best results (centered area)
        2. How the model sees their drawing (28x28 grid overlay)
        3. The importance of keeping drawings centered
        """
        # Calculate the optimal drawing area (centered square)
        # The model works best with centered, square compositions
        min_dim = min(w, h)
        optimal_size = int(min_dim * 0.75)  # Use 75% of available space
        
        guide_x = x_offset + (w - optimal_size) // 2
        guide_y = y_offset + (h - optimal_size) // 2
        
        # Draw main guide rectangle (bright cyan for high visibility)
        pen_main = QPen(QColor(0, 255, 255, 255), 4, Qt.PenStyle.SolidLine)
        painter.setPen(pen_main)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(guide_x, guide_y, optimal_size, optimal_size)
        
        # Draw corner markers (emphasized corners)
        marker_len = 30
        pen_marker = QPen(QColor(0, 255, 255, 255), 6, Qt.PenStyle.SolidLine)
        painter.setPen(pen_marker)
        
        # Top-left corner
        painter.drawLine(guide_x, guide_y, guide_x + marker_len, guide_y)
        painter.drawLine(guide_x, guide_y, guide_x, guide_y + marker_len)
        
        # Top-right corner
        painter.drawLine(guide_x + optimal_size, guide_y, guide_x + optimal_size - marker_len, guide_y)
        painter.drawLine(guide_x + optimal_size, guide_y, guide_x + optimal_size, guide_y + marker_len)
        
        # Bottom-left corner
        painter.drawLine(guide_x, guide_y + optimal_size, guide_x + marker_len, guide_y + optimal_size)
        painter.drawLine(guide_x, guide_y + optimal_size, guide_x, guide_y + optimal_size - marker_len)
        
        # Bottom-right corner
        painter.drawLine(guide_x + optimal_size, guide_y + optimal_size, guide_x + optimal_size - marker_len, guide_y + optimal_size)
        painter.drawLine(guide_x + optimal_size, guide_y + optimal_size, guide_x + optimal_size, guide_y + optimal_size - marker_len)
        
        # Draw 28x28 grid overlay to show model resolution
        pen_grid = QPen(QColor(0, 200, 255, 80), 1, Qt.PenStyle.DotLine)
        painter.setPen(pen_grid)
        
        cell_size = optimal_size / 28  # 28x28 is the model's input size
        
        # Vertical grid lines
        for i in range(1, 28):
            x = guide_x + int(i * cell_size)
            painter.drawLine(x, guide_y, x, guide_y + optimal_size)
        
        # Horizontal grid lines
        for i in range(1, 28):
            y = guide_y + int(i * cell_size)
            painter.drawLine(guide_x, y, guide_x + optimal_size, y)
        
        # Draw center crosshair
        pen_cross = QPen(QColor(255, 255, 0, 150), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen_cross)
        center_x = guide_x + optimal_size // 2
        center_y = guide_y + optimal_size // 2
        painter.drawLine(guide_x, center_y, guide_x + optimal_size, center_y)
        painter.drawLine(center_x, guide_y, center_x, guide_y + optimal_size)
        
        # Minimal text label at bottom only
        text_bottom_y = guide_y + optimal_size + 22
        if text_bottom_y < y_offset + h - 10:
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(255, 255, 0, 220))
            painter.drawText(guide_x + 10, text_bottom_y, "Press . to toggle guide")
        
        # Semi-transparent overlay outside the optimal area to emphasize it
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 20, 30, 40))
        
        # Top overlay
        if guide_y > y_offset:
            painter.drawRect(x_offset, y_offset, w, guide_y - y_offset)
        
        # Bottom overlay
        if guide_y + optimal_size < y_offset + h:
            painter.drawRect(x_offset, guide_y + optimal_size, w, 
                           y_offset + h - (guide_y + optimal_size))
        
        # Left overlay
        if guide_x > x_offset:
            painter.drawRect(x_offset, guide_y, guide_x - x_offset, optimal_size)
        
        # Right overlay
        if guide_x + optimal_size < x_offset + w:
            painter.drawRect(guide_x + optimal_size, guide_y, 
                           x_offset + w - (guide_x + optimal_size), optimal_size)


class GameCard(QFrame):
    """Tarjeta para mostrar objetivo, timer y puntaje."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gameCard")
        self.setMinimumWidth(280)  # Reduced from 300
        self.setMaximumWidth(350)  # Reduced from 400
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura la UI de la tarjeta de juego."""
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)  # Reduced margins
        layout.setSpacing(10)  # Reduced spacing
        
        # Título
        title = QLabel("🎯 OBJETIVO")
        title.setObjectName("gameTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Separador
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("separator")
        layout.addWidget(separator)
        
        # Palabra objetivo (texto simple)
        self.target_label = QLabel("--")
        self.target_label.setObjectName("targetLabel")
        self.target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_label.setWordWrap(True)
        self.target_label.setMinimumHeight(50)  # Reduced from 60
        self.target_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.target_label.setStyleSheet("""
            QLabel#targetLabel {
                color: #ffff00;
                font-size: 24px;  /* Reduced from 28px */
                font-weight: bold;
                background-color: transparent;
                padding: 10px;  /* Reduced from 15px */
            }
        """)
        layout.addWidget(self.target_label)
        
        # Timer y Score en fila horizontal para ahorrar espacio vertical
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(10)
        
        # Timer
        timer_container = QFrame()
        timer_container.setObjectName("timerContainer")
        timer_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        timer_container.setStyleSheet("""
            QFrame#timerContainer {
                background-color: #0a1428;
                border-radius: 6px;  /* Reduced from 8px */
                border: 2px solid #00ffff;
                padding: 8px;  /* Reduced from 10px */
                margin: 0px;
            }
        """)
        timer_layout = QVBoxLayout()
        timer_layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        timer_title = QLabel("⏱️ TIEMPO")
        timer_title.setObjectName("timerTitle")
        timer_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        timer_title.setStyleSheet("color: #00ffff; font-size: 12px; font-weight: bold;")  # Reduced font
        timer_layout.addWidget(timer_title)
        
        self.timer_label = QLabel("02:00")
        self.timer_label.setObjectName("timerLabel")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("color: #00ffff; font-size: 28px; font-weight: bold;")  # Reduced from 36px
        timer_layout.addWidget(self.timer_label)
        timer_container.setLayout(timer_layout)
        stats_layout.addWidget(timer_container)
        
        # Puntaje
        score_container = QFrame()
        score_container.setObjectName("scoreContainer")
        score_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        score_container.setStyleSheet("""
            QFrame#scoreContainer {
                background-color: #0a1428;
                border-radius: 6px;  /* Reduced from 8px */
                border: 2px solid #64ff64;
                padding: 8px;  /* Reduced from 10px */
                margin: 0px;
            }
        """)
        score_layout = QVBoxLayout()
        score_layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        score_title = QLabel("🏆 PUNTAJE")
        score_title.setObjectName("scoreTitle")
        score_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_title.setStyleSheet("color: #64ff64; font-size: 12px; font-weight: bold;")  # Reduced font
        score_layout.addWidget(score_title)
        
        self.score_label = QLabel("0")
        self.score_label.setObjectName("scoreLabel")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("color: #64ff64; font-size: 36px; font-weight: bold;")  # Reduced from 48px
        score_layout.addWidget(self.score_label)
        score_container.setLayout(score_layout)
        stats_layout.addWidget(score_container)
        
        layout.addLayout(stats_layout)
        
        # No stretch to keep compact
        self.setLayout(layout)
        
        # Efecto de sombra/glow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)  # Reduced from 20
        shadow.setColor(QColor(0, 255, 255, 100))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
    
    def update_target(self, target: str):
        """Actualiza el objetivo mostrado."""
        self.target_label.setText(target.upper())
    
    def update_timer(self, seconds: int):
        """Actualiza el timer."""
        minutes = seconds // 60
        secs = seconds % 60
        self.timer_label.setText(f"{minutes:02d}:{secs:02d}")
        
        # Cambiar color según tiempo restante
        if seconds <= 30:
            self.timer_label.setStyleSheet("color: #ff6400;")
        elif seconds <= 60:
            self.timer_label.setStyleSheet("color: #ffa000;")
        else:
            self.timer_label.setStyleSheet("color: #00ffff;")
    
    def update_score(self, score: int):
        """Actualiza el puntaje."""
        self.score_label.setText(str(score))


class PredictionCard(QFrame):
    """Tarjeta moderna para mostrar predicciones."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("predictionCard")
        self.setMinimumWidth(280)  # Reduced from default
        self.setMaximumWidth(350)  # Reduced from default
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._setup_ui()
        
    def _setup_ui(self):
        """Configura la UI de la tarjeta."""
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)  # Reduced margins
        layout.setSpacing(8)  # Reduced spacing
        
        # Título
        title = QLabel("PREDICCIÓN")
        title.setObjectName("predictionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Separador
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("separator")
        layout.addWidget(separator)
        
        # Etiqueta principal
        self.label_text = QLabel("--")
        self.label_text.setObjectName("predictionLabel")
        self.label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_text.setWordWrap(True)
        self.label_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.label_text.setMinimumHeight(40)  # Set minimum height
        layout.addWidget(self.label_text)
        
        # Confianza
        self.confidence_text = QLabel("")
        self.confidence_text.setObjectName("confidenceText")
        self.confidence_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.confidence_text)
        
        # Top-3 (placeholder, se llenará dinámicamente)
        self.top3_container = QWidget()
        self.top3_layout = QVBoxLayout()
        self.top3_layout.setSpacing(2)  # Reduced spacing
        self.top3_container.setLayout(self.top3_layout)
        self.top3_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.top3_container)
        
        self.setLayout(layout)
        
        # Efecto de sombra/glow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)  # Reduced from 20
        shadow.setColor(QColor(0, 255, 255, 100))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
    
    def update_prediction(self, label: str, confidence: float, top3: List[Tuple[str, float]]):
        """Actualiza la predicción mostrada."""
        self.label_text.setText(label.upper())
        self.confidence_text.setText(f"{confidence*100:.1f}%")
        
        # Limpiar top3 anterior
        while self.top3_layout.count():
            item = self.top3_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Agregar nuevos top3
        for lbl, conf in top3[:3]:
            item_label = QLabel(f"• {lbl}: {conf*100:.1f}%")
            item_label.setObjectName("top3Item")
            self.top3_layout.addWidget(item_label)
    
    def clear(self):
        """Limpia la predicción."""
        self.label_text.setText("--")
        self.confidence_text.setText("")
        while self.top3_layout.count():
            item = self.top3_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class StatusBar(QFrame):
    """Barra de estado moderna con información del sistema."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusBar")
        self._setup_ui()
        
    def _setup_ui(self):
        """Configura la UI de la barra de estado."""
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        # FPS
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setObjectName("statusLabel")
        layout.addWidget(self.fps_label)
        
        layout.addStretch()
        
        # Estado de mano
        self.hand_label = QLabel("Mano: NO")
        self.hand_label.setObjectName("statusLabel")
        layout.addWidget(self.hand_label)
        
        layout.addStretch()
        
        # Puntos de trazo
        self.stroke_label = QLabel("Puntos: 0")
        self.stroke_label.setObjectName("statusLabel")
        layout.addWidget(self.stroke_label)
        
        self.setLayout(layout)
    
    def update_fps(self, fps: float):
        """Actualiza FPS."""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def update_hand(self, detected: bool):
        """Actualiza estado de mano."""
        self.hand_label.setText(f"Mano: {'SÍ' if detected else 'NO'}")
        if detected:
            self.hand_label.setStyleSheet("color: #64ff64;")
        else:
            self.hand_label.setStyleSheet("color: #b4b4be;")
    
    def update_stroke(self, points: int):
        """Actualiza puntos de trazo."""
        self.stroke_label.setText(f"Puntos: {points}")


class PictionaryUIQt(QMainWindow):
    """Ventana principal de Pictionary Live con PyQt6."""
    
    # Señales para comunicación thread-safe
    frame_ready = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str, float, list)
    clear_requested = pyqtSignal()
    mode_switched = pyqtSignal(bool)  # True=mano, False=mouse
    timer_expired = pyqtSignal()
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        self.last_prediction = None
        
        # Logger for debug messages
        import logging
        self.logger = logging.getLogger("PictionaryLive")
        
        # Estado del juego
        self.current_target = None
        self.score = 0
        self.time_remaining = 180  # 2 minutos
        self.game_paused = False
        
        self._setup_window()
        self._setup_ui()
        self._connect_signals()
        self._load_styles()
        
        # Timer para FPS (cada 500ms)
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._update_fps_display)
        self.fps_timer.start(500)
        
        # Timer del juego (cada 1 segundo) - INICIAR después de mostrar UI
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self._update_game_timer)
        
        # Connect resize event to ensure visibility
        self.installEventFilter(self)
    
    def _setup_window(self):
        """Configura la ventana principal con tamaño responsivo."""
        window_name = self.config.get('window_name', 'Pictionary Live - UI Moderna')
        
        # Get screen geometry for responsive sizing
        screen = QScreen.availableGeometry(self.screen())
        screen_width = screen.width()
        screen_height = screen.height()
        
        # Use 90% of screen size for better visibility on small screens
        default_width = int(screen_width * 0.9)
        default_height = int(screen_height * 0.9)
        
        window_width = min(self.config.get('window_width', 1400), default_width)
        window_height = min(self.config.get('window_height', 800), default_height)
        
        self.setWindowTitle(window_name)
        self.setMinimumSize(1000, 700)  # Increased minimum size for better visibility
        self.resize(window_width, window_height)
        
        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.move(x, y)
        
        # Aplicar color de fondo
        self.setStyleSheet(f"background-color: {COLORS['bg_panel'].name()};")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        main_layout.setSpacing(8)  # Reduced spacing
        
        # Header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Contenedor de video y predicción
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)  # Reduced spacing
        
        # Video - Priority component, gets more space
        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(self.video_widget, stretch=3)
        
        # Panel lateral (predicción + controles) - Can shrink if needed
        side_panel = self._create_side_panel()
        side_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(side_panel, stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)  # Allow content to expand
        
        # Barra de estado - Can be compressed
        self.status_bar = StatusBar()
        self.status_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(self.status_bar)
        
        # Footer con controles - Fixed height
        footer = self._create_footer()
        footer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(footer)
        
        central_widget.setLayout(main_layout)
    
    def _create_header(self) -> QWidget:
        """Crea el encabezado."""
        header = QFrame()
        header.setObjectName("header")
        header.setMinimumHeight(80)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)
        
        # Título
        title = QLabel("🎮 PICTIONARY LIVE")
        title.setObjectName("mainTitle")
        font = QFont("Segoe UI", 28, QFont.Weight.Bold)
        title.setFont(font)
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Indicador de estado
        self.state_indicator = QLabel("🟢 LISTO")
        self.state_indicator.setObjectName("stateIndicator")
        font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        self.state_indicator.setFont(font)
        layout.addWidget(self.state_indicator)
        
        header.setLayout(layout)
        return header
    
    def _create_side_panel(self) -> QWidget:
        """Crea el panel lateral con juego y predicción."""
        panel = QFrame()
        panel.setObjectName("sidePanel")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)  # Reduced spacing for better space usage
        
        # Tarjeta de juego (objetivo, timer, puntaje) - PRIORITY: Always visible
        self.game_card = GameCard()
        self.game_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)  # Allow shrinking
        layout.addWidget(self.game_card)
        
        # Tarjeta de predicción - Can be compressed if needed
        self.prediction_card = PredictionCard()
        self.prediction_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.prediction_card)
        
        # Botón de cambio de modo - Compact
        self.mode_button = QPushButton("🖱️ CAMBIAR A MOUSE")
        self.mode_button.setObjectName("modeButton")
        self.mode_button.setMinimumHeight(40)  # Reduced from 50
        self.mode_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.mode_button.clicked.connect(self._toggle_mode)
        self.mode_button.setToolTip("Alternar entre detección de manos y dibujo con mouse")
        layout.addWidget(self.mode_button)
        
        # Estado del modo actual
        self.current_mode = "hand"  # Por defecto modo mano
        
        # Minimal stretch to push everything up
        layout.addStretch(1)
        
        panel.setLayout(layout)
        return panel
    
    def _create_footer(self) -> QWidget:
        """Crea el footer con instrucciones."""
        footer = QFrame()
        footer.setObjectName("footer")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        self.instructions = QLabel("Q = Salir  |  C = Limpiar  |  S = Siguiente  |  R = Reiniciar  |  . = Guía")
        self.instructions.setObjectName("instructions")
        self.instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.instructions)
        
        footer.setLayout(layout)
        return footer
    
    def _connect_signals(self):
        """Conecta las señales."""
        self.frame_ready.connect(self._on_frame_ready)
        self.prediction_ready.connect(self._on_prediction_ready)
    
    def _load_styles(self):
        """Load QSS styles with fallback to minimal embedded styles."""
        try:
            # Try loading external QSS file
            qss_file = Path(__file__).parent / "styles_cyberpunk.qss"
            if qss_file.exists():
                with open(qss_file, 'r', encoding='utf-8') as f:
                    self.setStyleSheet(f.read())
                return
        except Exception as e:
            self.logger.warning(f"Could not load external styles: {e}")
        
        # Fallback to embedded minimal styles
        self.setStyleSheet(self._get_fallback_styles())
    
    def _get_fallback_styles(self) -> str:
        """Get minimal fallback styles."""
        return """
            QMainWindow { background-color: #0a1428; color: #ebebeb; }
            #mainTitle { color: #00ffff; }
            #stateIndicator { color: #64ff64; }
            #header { background-color: #192540; border-radius: 10px; }
            #predictionTitle { color: #00ffff; font-size: 18px; font-weight: bold; }
            #predictionLabel { color: #64ff64; font-size: 32px; font-weight: bold; }
            #confidenceText { color: #ebebeb; font-size: 20px; }
            #top3Item { color: #b4b4be; font-size: 14px; }
            #separator { background-color: #ffa000; max-height: 2px; }
            #statusLabel { color: #b4b4be; font-size: 13px; }
            #footer { background-color: #192540; border-radius: 8px; }
            #instructions { color: #b4b4be; font-size: 12px; }
            #modeButton { 
                background-color: #ffa000; 
                color: #0a1428; 
                font-size: 16px; 
                font-weight: bold; 
                border: none; 
                border-radius: 10px; 
                padding: 15px; 
            }
            #modeButton:hover { background-color: #ffb000; }
            #modeButton:pressed { background-color: #ff9000; }
        """
    
    def _update_fps_display(self):
        """Actualiza el display de FPS."""
        current_time = time.time()
        if current_time - self.last_time >= 0.5:
            elapsed = max(current_time - self.last_time, 1e-3)
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
            self.status_bar.update_fps(self.fps)
    
    def _on_frame_ready(self, frame: np.ndarray):
        """Maneja frames nuevos (thread-safe)."""
        self.video_widget.set_frame(frame)
        self.frame_count += 1
    
    def _on_prediction_ready(self, label: str, confidence: float, top3: list):
        """Maneja predicciones nuevas (thread-safe)."""
        self.last_prediction = (label, confidence, top3)
        self.prediction_card.update_prediction(label, confidence, top3)
        
        # Verificar si la predicción coincide con el objetivo
        if self.current_target and label.lower() == self.current_target.lower():
            self.score += 1
            self.game_card.update_score(self.score)
            # Limpiar canvas al acertar
            self.clear_requested.emit()
            # Seleccionar nuevo objetivo
            self.select_new_target()
    
    def update_frame(self, frame: np.ndarray):
        """Actualiza el frame (llamar desde thread de video)."""
        self.frame_ready.emit(frame)
    
    def update_prediction(self, label: str, confidence: float, top3: List[Tuple[str, float]]):
        """Actualiza la predicción (llamar desde thread de procesamiento)."""
        self.prediction_ready.emit(label, confidence, top3)
    
    def update_hand_detected(self, detected: bool):
        """Actualiza el estado de detección de mano."""
        self.status_bar.update_hand(detected)
    
    def update_stroke_points(self, points: List[Tuple[float, float]]):
        """Actualiza los puntos del trazo."""
        self.status_bar.update_stroke(len(points))
        self.video_widget.set_stroke_points(points)
    
    def set_state(self, state: str, color: str = "#64ff64"):
        """Establece el estado mostrado."""
        self.state_indicator.setText(state)
        self.state_indicator.setStyleSheet(f"color: {color};")
    
    def _toggle_mode(self):
        """Cambia entre modo mano y modo mouse."""
        if self.current_mode == "hand":
            self.current_mode = "mouse"
            self.mode_button.setText("☕ CAMBIAR A MANO")
            self.state_indicator.setText("🖱️ MODO MOUSE")
            self.state_indicator.setStyleSheet("color: #ffa000;")
            self.instructions.setText("Q = Salir  |  C = Limpiar  |  S = Siguiente  |  R = Reiniciar  |  . = Guía")
            self.mode_switched.emit(False)  # False = mouse
        else:
            self.current_mode = "hand"
            self.mode_button.setText("🖱️ CAMBIAR A MOUSE")
            self.state_indicator.setText("☕ MODO MANO")
            self.state_indicator.setStyleSheet("color: #64ff64;")
            self.instructions.setText("Q = Salir  |  C = Limpiar  |  S = Siguiente  |  R = Reiniciar  |  . = Guía")
            self.mode_switched.emit(True)  # True = hand
    
    def _update_game_timer(self):
        """Actualiza el timer del juego."""
        if self.time_remaining > 0:
            self.time_remaining -= 1
            self.game_card.update_timer(self.time_remaining)
        else:
            self.game_timer.stop()
            self._show_game_over_dialog()
    
    def select_new_target(self):
        """Selecciona un nuevo objetivo aleatorio."""
        # Esta función será asignada desde app_pyqt.py con la lista de labels
        if hasattr(self, '_select_new_target_func') and self._select_new_target_func:
            self._select_new_target_func()
    
    def set_select_new_target_func(self, func):
        """Asigna la función para seleccionar nuevo objetivo."""
        self._select_new_target_func = func
    
    def set_target(self, target: str):
        """Establece el objetivo actual."""
        self.current_target = target
        self.game_card.update_target(target)
    
    def reset_timer(self):
        """Reinicia el timer a 3 minutos."""
        self.time_remaining = 180
        self.game_card.update_timer(self.time_remaining)
        if not self.game_timer.isActive():
            self.game_timer.start(1000)
    
    def reset_score(self):
        """Reinicia el puntaje."""
        self.score = 0
        self.game_card.update_score(self.score)
    
    def _show_game_over_dialog(self):
        """Shows game over dialog with score and options to restart or exit."""
        # Pause the game - emit signal to pause camera
        self.game_paused = True
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("⏰ Time's Up!")
        msg_box.setText(f"<h2>Game Over!</h2><p style='font-size: 16px;'>Your final score: <b style='color: #64ff64; font-size: 24px;'>{self.score}</b></p>")
        msg_box.setInformativeText("What would you like to do?")
        
        # Custom buttons
        restart_btn = msg_box.addButton("🔄 Restart Game", QMessageBox.ButtonRole.AcceptRole)
        exit_btn = msg_box.addButton("❌ Exit", QMessageBox.ButtonRole.RejectRole)
        
        # Style the message box
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #192540;
                color: #ebebeb;
            }
            QMessageBox QLabel {
                color: #ebebeb;
                min-width: 300px;
            }
            QPushButton {
                background-color: #ffa000;
                color: #0a1428;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #ffb000;
            }
            QPushButton:pressed {
                background-color: #ff9000;
            }
        """)
        
        msg_box.exec()
        
        # Check which button was clicked
        if msg_box.clickedButton() == restart_btn:
            # Resume the game
            self.game_paused = False
            # Restart the game
            self.reset_timer()
            self.reset_score()
            self.clear_requested.emit()
            self.prediction_card.clear()
            self.select_new_target()
        else:
            # Exit the application
            self.close()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        # Don't process keys if game is paused
        if self.game_paused:
            return
            
        key = event.key()
        
        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            # Enter - Make prediction (silent, no trace)
            # Note: Actual prediction is triggered automatically
            pass
        elif key == Qt.Key.Key_Space or key == Qt.Key.Key_C:
            # Space/C - Clear canvas
            self.clear_requested.emit()
        elif key == Qt.Key.Key_Q:
            # Q - Quit application
            self.close()
        elif key == Qt.Key.Key_Period:
            # Period (.) - Toggle debug guide
            self.video_widget.show_debug_guide = not self.video_widget.show_debug_guide
            self.video_widget.update()  # Force repaint
        elif key == Qt.Key.Key_S:
            # S - Next target
            self.select_new_target()
        elif key == Qt.Key.Key_R:
            # R - Reset game
            self.reset_timer()
            self.reset_score()
            self.clear_requested.emit()
            self.prediction_card.clear()
            self.select_new_target()
    
    def eventFilter(self, obj, event):
        """Handle window resize events to ensure critical elements are visible."""
        if obj == self and event.type() == event.Type.Resize:
            # Ensure minimum sizes are respected
            min_width = 1000
            min_height = 700
            
            if self.width() < min_width or self.height() < min_height:
                # Force minimum size if too small
                new_width = max(self.width(), min_width)
                new_height = max(self.height(), min_height)
                self.resize(new_width, new_height)
                
        return super().eventFilter(obj, event)
