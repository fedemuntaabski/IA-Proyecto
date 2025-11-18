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
    QPushButton, QFrame, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPainterPath
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
            self._emit_mouse_position(event.position())
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Maneja el movimiento del mouse."""
        if self.mouse_drawing:
            self._emit_mouse_position(event.position())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Maneja el evento de soltar el mouse."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_drawing = False
            self.mouse_draw.emit(0.0, 0.0, False)  # Señal de fin de trazo
        super().mouseReleaseEvent(event)
    
    def _emit_mouse_position(self, pos):
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
            self.mouse_draw.emit(norm_x, norm_y, True)
    
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


class GameCard(QFrame):
    """Tarjeta para mostrar objetivo, timer y puntaje."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gameCard")
        self.setMinimumWidth(300)
        self.setMaximumWidth(400)
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura la UI de la tarjeta de juego."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
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
        self.target_label.setStyleSheet("color: #ffff00; font-size: 32px; font-weight: bold; padding: 15px;")
        layout.addWidget(self.target_label)
        
        # Timer
        timer_container = QFrame()
        timer_container.setObjectName("timerContainer")
        timer_layout = QVBoxLayout()
        timer_title = QLabel("⏱️ TIEMPO")
        timer_title.setObjectName("timerTitle")
        timer_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        timer_layout.addWidget(timer_title)
        
        self.timer_label = QLabel("02:00")
        self.timer_label.setObjectName("timerLabel")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("color: #00ffff; font-size: 36px; font-weight: bold;")
        timer_layout.addWidget(self.timer_label)
        timer_container.setLayout(timer_layout)
        layout.addWidget(timer_container)
        
        # Puntaje
        score_container = QFrame()
        score_container.setObjectName("scoreContainer")
        score_layout = QVBoxLayout()
        score_title = QLabel("🏆 PUNTAJE")
        score_title.setObjectName("scoreTitle")
        score_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(score_title)
        
        self.score_label = QLabel("0")
        self.score_label.setObjectName("scoreLabel")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("color: #64ff64; font-size: 48px; font-weight: bold;")
        score_layout.addWidget(self.score_label)
        score_container.setLayout(score_layout)
        layout.addWidget(score_container)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Efecto de sombra/glow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
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
            self.timer_label.setStyleSheet("color: #ff6400; font-size: 36px; font-weight: bold;")
        elif seconds <= 60:
            self.timer_label.setStyleSheet("color: #ffa000; font-size: 36px; font-weight: bold;")
        else:
            self.timer_label.setStyleSheet("color: #00ffff; font-size: 36px; font-weight: bold;")
    
    def update_score(self, score: int):
        """Actualiza el puntaje."""
        self.score_label.setText(str(score))


class PredictionCard(QFrame):
    """Tarjeta moderna para mostrar predicciones."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("predictionCard")
        self.setMinimumWidth(300)
        self.setMaximumWidth(400)
        self._setup_ui()
        
    def _setup_ui(self):
        """Configura la UI de la tarjeta."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
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
        layout.addWidget(self.label_text)
        
        # Confianza
        self.confidence_text = QLabel("")
        self.confidence_text.setObjectName("confidenceText")
        self.confidence_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.confidence_text)
        
        # Top-3 (placeholder, se llenará dinámicamente)
        self.top3_container = QWidget()
        self.top3_layout = QVBoxLayout()
        self.top3_container.setLayout(self.top3_layout)
        layout.addWidget(self.top3_container)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Efecto de sombra/glow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
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
        
        # Estado del juego
        self.current_target = None
        self.score = 0
        self.time_remaining = 120  # 2 minutos
        
        self._setup_window()
        self._setup_ui()
        self._connect_signals()
        self._load_styles()
        
        # Timer para FPS (cada 500ms)
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._update_fps_display)
        self.fps_timer.start(500)
        
        # Timer del juego (cada 1 segundo)
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self._update_game_timer)
        self.game_timer.start(1000)
    
    def _setup_window(self):
        """Configura la ventana principal."""
        self.setWindowTitle("PICTIONARY LIVE - UI Moderna")
        self.setMinimumSize(1280, 720)
        
        # Aplicar color de fondo
        self.setStyleSheet(f"background-color: {COLORS['bg_panel'].name()};")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Contenedor de video y predicción
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # Video
        self.video_widget = VideoWidget()
        content_layout.addWidget(self.video_widget, stretch=3)
        
        # Panel lateral (predicción + controles)
        side_panel = self._create_side_panel()
        content_layout.addWidget(side_panel, stretch=1)
        
        main_layout.addLayout(content_layout)
        
        # Barra de estado
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)
        
        # Footer con controles
        footer = self._create_footer()
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
        layout.setSpacing(15)
        
        # Tarjeta de juego (objetivo, timer, puntaje)
        self.game_card = GameCard()
        layout.addWidget(self.game_card)
        
        # Tarjeta de predicción
        self.prediction_card = PredictionCard()
        layout.addWidget(self.prediction_card)
        
        # Botón de cambio de modo
        self.mode_button = QPushButton("🖱️ CAMBIAR A MOUSE")
        self.mode_button.setObjectName("modeButton")
        self.mode_button.setMinimumHeight(50)
        self.mode_button.clicked.connect(self._toggle_mode)
        self.mode_button.setToolTip("Alternar entre detección de manos y dibujo con mouse")
        layout.addWidget(self.mode_button)
        
        # Estado del modo actual
        self.current_mode = "hand"  # Por defecto modo mano
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def _create_footer(self) -> QWidget:
        """Crea el footer con instrucciones."""
        footer = QFrame()
        footer.setObjectName("footer")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        self.instructions = QLabel("Q = Salir  |  C = Limpiar  |  S = Siguiente  |  R = Reiniciar")
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
        """Carga los estilos QSS optimizados."""
        try:
            # Intentar cargar el archivo QSS externo
            qss_file = Path(__file__).parent / "styles_cyberpunk.qss"
            if qss_file.exists():
                with open(qss_file, 'r', encoding='utf-8') as f:
                    styles = f.read()
                self.setStyleSheet(styles)
            else:
                # Fallback a estilos inline
                styles = """
                    QMainWindow { background-color: #0a1428; color: #ebebeb; }
                    #mainTitle { color: #00ffff; background: transparent; }
                    #stateIndicator { color: #64ff64; background: transparent; }
                    #header { background-color: #192540; border-radius: 10px; }
                    #sidePanel { background-color: transparent; }
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
                self.setStyleSheet(styles)
        except Exception as e:
            # Si falla la carga de estilos, continuar sin ellos
            print(f"Warning: No se pudieron cargar los estilos: {e}")
    
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
            self.mode_button.setText("✋ CAMBIAR A MANO")
            self.state_indicator.setText("🖱️ MODO MOUSE")
            self.state_indicator.setStyleSheet("color: #ffa000;")
            self.instructions.setText("Q = Salir  |  C = Limpiar  |  S = Siguiente  |  R = Reiniciar")
            self.mode_switched.emit(False)  # False = mouse
        else:
            self.current_mode = "hand"
            self.mode_button.setText("🖱️ CAMBIAR A MOUSE")
            self.state_indicator.setText("✋ MODO MANO")
            self.state_indicator.setStyleSheet("color: #64ff64;")
            self.instructions.setText("Q = Salir  |  C = Limpiar  |  S = Siguiente  |  R = Reiniciar")
            self.mode_switched.emit(True)  # True = hand
    
    def _update_game_timer(self):
        """Actualiza el timer del juego."""
        if self.time_remaining > 0:
            self.time_remaining -= 1
            self.game_card.update_timer(self.time_remaining)
        else:
            self.game_timer.stop()
            self.timer_expired.emit()
    
    def select_new_target(self):
        """Selecciona un nuevo objetivo aleatorio."""
        # Esta función será conectada desde app_pyqt.py con la lista de labels
        pass
    
    def set_target(self, target: str):
        """Establece el objetivo actual."""
        self.current_target = target
        self.game_card.update_target(target)
    
    def reset_timer(self):
        """Reinicia el timer a 2 minutos."""
        self.time_remaining = 120
        self.game_card.update_timer(self.time_remaining)
        if not self.game_timer.isActive():
            self.game_timer.start(1000)
    
    def reset_score(self):
        """Reinicia el puntaje."""
        self.score = 0
        self.game_card.update_score(self.score)
    
    def keyPressEvent(self, event):
        """Maneja eventos de teclado."""
        key = event.key()
        
        if key == Qt.Key.Key_Q:
            # Salir
            self.close()
        elif key == Qt.Key.Key_C:
            # C = Limpiar el tablero
            self.clear_requested.emit()
            self.prediction_card.clear()
        elif key == Qt.Key.Key_S:
            # S = Siguiente objetivo (sin limpiar)
            self.select_new_target()
        elif key == Qt.Key.Key_R:
            # R = Reiniciar juego completo
            self.reset_timer()
            self.reset_score()
            self.clear_requested.emit()
            self.prediction_card.clear()
            self.select_new_target()
        else:
            super().keyPressEvent(event)
