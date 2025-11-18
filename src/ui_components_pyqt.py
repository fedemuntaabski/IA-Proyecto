"""
ui_components_pyqt.py - Componentes modulares de UI con PyQt6

Componentes modernos para el modo de juego usando PyQt6,
siguiendo el principio de responsabilidad única.
"""

from typing import Callable, Optional, List, Tuple
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QImage, QPixmap, QPainter
import numpy as np
import cv2


class StyleManager:
    """Gestor de estilos y colores para la UI con PyQt6."""

    def __init__(self, theme: str = "cyberpunk"):
        self.theme = theme
        self.colors = self._get_color_scheme()
        self.fonts = self._get_font_config()

    def _get_color_scheme(self) -> dict:
        """Obtiene el esquema de colores según el tema."""
        schemes = {
            "cyberpunk": {
                "bg_primary": "#0a1428",
                "bg_secondary": "#192540",
                "accent_primary": "#00ffff",
                "accent_secondary": "#ffa000",
                "success": "#64ff64",
                "warning": "#ff6400",
                "text_primary": "#ebebeb",
                "text_secondary": "#b4b4be",
                "text_dim": "#6a6a6e",
            },
            "light": {
                "bg_primary": "#ffffff",
                "bg_secondary": "#f5f5f5",
                "accent_primary": "#0066cc",
                "accent_secondary": "#ff9900",
                "success": "#00aa00",
                "warning": "#ff3300",
                "text_primary": "#000000",
                "text_secondary": "#333333",
                "text_dim": "#999999",
            },
            "dark": {
                "bg_primary": "#1e1e1e",
                "bg_secondary": "#2d2d2d",
                "accent_primary": "#4a9eff",
                "accent_secondary": "#ff7043",
                "success": "#66bb6a",
                "warning": "#ef5350",
                "text_primary": "#ffffff",
                "text_secondary": "#e0e0e0",
                "text_dim": "#9e9e9e",
            },
        }
        return schemes.get(self.theme, schemes["cyberpunk"])

    def _get_font_config(self) -> dict:
        """Configuración de fuentes."""
        return {
            "family": "Segoe UI",
            "title": 36,
            "word": 48,
            "normal": 14,
            "small": 12,
        }

    def get_color(self, key: str) -> str:
        """Obtiene un color por clave."""
        return self.colors.get(key, "#ffffff")

    def get_font(self, size_key: str, bold: bool = False) -> QFont:
        """Obtiene una configuración de fuente."""
        size = self.fonts.get(size_key, self.fonts["normal"])
        weight = QFont.Weight.Bold if bold else QFont.Weight.Normal
        return QFont(self.fonts["family"], size, weight)

    def get_qss(self) -> str:
        """Obtiene el stylesheet QSS completo."""
        return f"""
            QMainWindow {{
                background-color: {self.get_color("bg_primary")};
            }}
            
            QFrame#headerFrame {{
                background-color: {self.get_color("bg_secondary")};
                border-radius: 10px;
                padding: 15px;
            }}
            
            QFrame#cameraFrame {{
                background-color: {self.get_color("bg_secondary")};
                border-radius: 10px;
                border: 2px solid {self.get_color("accent_primary")};
            }}
            
            QFrame#infoFrame {{
                background-color: {self.get_color("bg_secondary")};
                border-radius: 10px;
                padding: 15px;
            }}
            
            QLabel#titleLabel {{
                color: {self.get_color("accent_primary")};
            }}
            
            QLabel#scoreLabel {{
                color: {self.get_color("success")};
            }}
            
            QLabel#wordLabel {{
                color: {self.get_color("accent_secondary")};
            }}
            
            QLabel#predictionLabel {{
                color: {self.get_color("text_primary")};
            }}
            
            QLabel#predictionLabelCorrect {{
                color: {self.get_color("success")};
            }}
            
            QLabel#confidenceLabel {{
                color: {self.get_color("text_dim")};
            }}
            
            QLabel#stateLabel {{
                color: {self.get_color("text_secondary")};
            }}
            
            QPushButton#clearButton {{
                background-color: {self.get_color("warning")};
                color: {self.get_color("bg_primary")};
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            
            QPushButton#clearButton:hover {{
                background-color: #ff7400;
            }}
            
            QPushButton#nextButton {{
                background-color: {self.get_color("accent_secondary")};
                color: {self.get_color("bg_primary")};
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            
            QPushButton#nextButton:hover {{
                background-color: #ffb000;
            }}
        """


class HeaderComponent(QFrame):
    """Componente del encabezado de la aplicación."""

    def __init__(self, parent: QWidget, style_manager: StyleManager, logger: logging.Logger):
        super().__init__(parent)
        self.style = style_manager
        self.logger = logger
        self.setObjectName("headerFrame")
        self._build()

    def _build(self):
        """Construye el componente del encabezado."""
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Título
        self.title_label = QLabel("🎮 PICTIONARY LIVE - MODO JUEGO")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setFont(self.style.get_font("title", bold=True))
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Score
        self.score_label = QLabel("Puntuación: 0 | Racha: 0")
        self.score_label.setObjectName("scoreLabel")
        self.score_label.setFont(self.style.get_font("normal", bold=True))
        layout.addWidget(self.score_label)
        
        self.setLayout(layout)
        self.logger.debug("HeaderComponent construido")

    def update_score(self, score: int, streak: int):
        """Actualiza la etiqueta de puntuación."""
        self.score_label.setText(f"Puntuación: {score} | Racha: {streak}")


class CameraComponent(QFrame):
    """Componente de la sección de cámara con PyQt6."""
    
    frame_clicked = pyqtSignal()

    def __init__(self, parent: QWidget, style_manager: StyleManager, logger: logging.Logger):
        super().__init__(parent)
        self.style = style_manager
        self.logger = logger
        self.setObjectName("cameraFrame")
        self.current_frame = None
        self._build()

    def _build(self):
        """Construye el componente de cámara."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Label para mostrar video
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet(f"background-color: {self.style.get_color('bg_primary')};")
        layout.addWidget(self.camera_label)
        
        # Estado de conexión
        self.state_label = QLabel("🔴 Desconectado")
        self.state_label.setObjectName("stateLabel")
        self.state_label.setFont(self.style.get_font("normal"))
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.state_label)
        
        self.setLayout(layout)
        self.logger.debug("CameraComponent construido")

    def update_frame(self, frame: np.ndarray):
        """Actualiza el frame de la cámara."""
        try:
            self.current_frame = frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Escalar manteniendo aspecto
            pixmap = QPixmap.fromImage(q_image).scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.camera_label.setPixmap(pixmap)
        except Exception as e:
            self.logger.error(f"Error actualizando frame: {e}")

    def update_state(self, text: str, color: str):
        """Actualiza el estado de la cámara."""
        self.state_label.setText(text)
        self.state_label.setStyleSheet(f"color: {color};")


class GameInfoComponent(QFrame):
    """Componente de información del juego."""

    def __init__(self, parent: QWidget, style_manager: StyleManager, logger: logging.Logger):
        super().__init__(parent)
        self.style = style_manager
        self.logger = logger
        self.setObjectName("infoFrame")
        self._build()

    def _build(self):
        """Construye el componente de información del juego."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # Marco de palabra
        word_container = QFrame()
        word_container.setStyleSheet(f"""
            background-color: {self.style.get_color("bg_primary")};
            border-radius: 10px;
            padding: 20px;
        """)
        word_layout = QVBoxLayout()
        
        word_title = QLabel("📝 PALABRA ACTUAL")
        word_title.setFont(self.style.get_font("normal", bold=True))
        word_title.setStyleSheet(f"color: {self.style.get_color('accent_primary')}; background: transparent;")
        word_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        word_layout.addWidget(word_title)
        
        # Palabra grande
        self.word_label = QLabel("--")
        self.word_label.setObjectName("wordLabel")
        self.word_label.setFont(self.style.get_font("word", bold=True))
        self.word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.word_label.setWordWrap(True)
        word_layout.addWidget(self.word_label)
        
        word_container.setLayout(word_layout)
        main_layout.addWidget(word_container)
        
        # Marco de predicción
        pred_container = QFrame()
        pred_container.setStyleSheet(f"""
            background-color: {self.style.get_color("bg_primary")};
            border-radius: 10px;
            padding: 20px;
        """)
        pred_layout = QVBoxLayout()
        
        pred_title = QLabel("🤖 PREDICCIÓN")
        pred_title.setFont(self.style.get_font("normal", bold=True))
        pred_title.setStyleSheet(f"color: {self.style.get_color('accent_primary')}; background: transparent;")
        pred_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(pred_title)
        
        # Predicción principal
        self.prediction_label = QLabel("(Sin predicción)")
        self.prediction_label.setObjectName("predictionLabel")
        self.prediction_label.setFont(self.style.get_font("normal", bold=True))
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.prediction_label)
        
        # Confianza
        self.confidence_label = QLabel("")
        self.confidence_label.setObjectName("confidenceLabel")
        self.confidence_label.setFont(self.style.get_font("normal"))
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.confidence_label)
        
        # Top-3
        self.top3_label = QLabel("")
        self.top3_label.setFont(self.style.get_font("small"))
        self.top3_label.setStyleSheet(f"color: {self.style.get_color('text_dim')}; background: transparent;")
        self.top3_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.top3_label.setWordWrap(True)
        pred_layout.addWidget(self.top3_label)
        
        pred_container.setLayout(pred_layout)
        main_layout.addWidget(pred_container)
        
        main_layout.addStretch()
        
        self.setLayout(main_layout)
        
        # Efecto glow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 255, 255, 80))
        shadow.setOffset(0, 0)
        pred_container.setGraphicsEffect(shadow)
        
        self.logger.debug("GameInfoComponent construido")

    def update_word(self, word: str):
        """Actualiza la palabra mostrada."""
        self.word_label.setText(word.upper())

    def update_prediction(self, label: str, confidence: float, top3: List[Tuple[str, float]], is_correct: bool = False):
        """Actualiza la predicción."""
        if is_correct:
            self.prediction_label.setObjectName("predictionLabelCorrect")
            self.prediction_label.setStyleSheet(f"color: {self.style.get_color('success')}; background: transparent;")
        else:
            self.prediction_label.setObjectName("predictionLabel")
            self.prediction_label.setStyleSheet(f"color: {self.style.get_color('text_primary')}; background: transparent;")
        
        self.prediction_label.setText(label.upper())
        self.confidence_label.setText(f"Confianza: {confidence*100:.1f}%")

        top3_text = "Top-3:\n" + "\n".join([f"  • {l}: {p*100:.1f}%" for l, p in top3[:3]])
        self.top3_label.setText(top3_text)

    def clear_predictions(self):
        """Limpia las predicciones."""
        self.prediction_label.setText("(Sin predicción)")
        self.prediction_label.setStyleSheet(f"color: {self.style.get_color('text_secondary')}; background: transparent;")
        self.confidence_label.setText("")
        self.top3_label.setText("")


class FooterComponent(QFrame):
    """Componente de controles del pie de página."""
    
    clear_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    def __init__(self, parent: QWidget, style_manager: StyleManager, logger: logging.Logger):
        super().__init__(parent)
        self.style = style_manager
        self.logger = logger
        self._build()

    def _build(self):
        """Construye el componente de controles."""
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Botón limpiar
        clear_btn = QPushButton("🧹 LIMPIAR (L)")
        clear_btn.setObjectName("clearButton")
        clear_btn.setFont(self.style.get_font("normal", bold=True))
        clear_btn.clicked.connect(self.clear_clicked.emit)
        clear_btn.setMinimumHeight(50)
        layout.addWidget(clear_btn)
        
        # Botón siguiente
        next_btn = QPushButton("⏭️  SIGUIENTE (C)")
        next_btn.setObjectName("nextButton")
        next_btn.setFont(self.style.get_font("normal", bold=True))
        next_btn.clicked.connect(self.next_clicked.emit)
        next_btn.setMinimumHeight(50)
        layout.addWidget(next_btn)
        
        self.setLayout(layout)
        self.logger.debug("FooterComponent construido")
