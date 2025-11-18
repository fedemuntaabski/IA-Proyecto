"""
game_mode_pyqt.py - Modo de juego con interfaz PyQt6 moderna

Implementa el modo de juego donde se selecciona aleatoriamente una palabra,
el usuario la dibuja, el modelo predice, y se cambia la palabra si es correcta.
Versión modernizada con PyQt6.
"""

import random
import logging
import sys
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut
import numpy as np

from ui_components_pyqt import (
    StyleManager, HeaderComponent, CameraComponent, 
    GameInfoComponent, FooterComponent
)


class GameState(Enum):
    """Estados del juego."""
    WAITING_FOR_DRAW = "waiting"
    DRAWING = "drawing"
    PREDICTING = "predicting"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PAUSED = "paused"


@dataclass
class GameConfig:
    """Configuración del modo de juego."""
    window_width: int = 1400
    window_height: int = 800
    auto_next_delay_ms: int = 2000
    min_drawing_points: int = 10
    theme: str = "cyberpunk"


class GameModeQt(QMainWindow):
    """
    Modo de juego principal con interfaz PyQt6.
    
    Flujo:
    1. Se selecciona aleatoriamente una palabra de las disponibles
    2. Se muestra la palabra al usuario
    3. El usuario dibuja usando la cámara
    4. Al presionar "Predecir", el modelo predice
    5. Si es correcta, cambia automáticamente a una palabra nueva
    6. Si es incorrecta, muestra feedback y permite reintentar
    """
    
    # Señales
    prediction_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    
    def __init__(
        self,
        labels: List[str],
        predict_callback: Callable[[Optional[np.ndarray]], Tuple[str, float, List[Tuple[str, float]]]],
        config: Optional[GameConfig] = None,
        logger: Optional[logging.Logger] = None,
        clear_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Inicializa el modo de juego.
        
        Args:
            labels: Lista de palabras disponibles para el juego
            predict_callback: Función que predice un dibujo
            config: Configuración del juego
            logger: Logger para debugging
            clear_callback: Callback para limpiar trazos
        """
        super().__init__()
        
        self.labels = labels
        self.predict_callback = predict_callback
        self.config = config or GameConfig()
        self.logger = logger or self._setup_default_logger()
        self.clear_callback = clear_callback or (lambda: None)
        
        # Validar labels
        if not self.labels:
            raise ValueError("Se requiere al menos una palabra en labels")
        
        # Estado del juego
        self.current_word = None
        self.game_state = GameState.WAITING_FOR_DRAW
        self.score = 0
        self.streak = 0
        self.recent_words = []
        self.max_recent = 5
        
        # Gestor de estilos
        self.style_manager = StyleManager(self.config.theme)
        
        # Componentes UI
        self.header = None
        self.camera = None
        self.game_info = None
        self.footer = None
        
        # Inicializar UI
        self._setup_window()
        self._init_ui()
        self._setup_shortcuts()
        
        # Seleccionar primera palabra
        self._select_next_word()
        
        self.logger.info(f"GameMode PyQt6 inicializado con {len(self.labels)} palabras")
    
    def _setup_default_logger(self) -> logging.Logger:
        """Crea un logger por defecto."""
        logger = logging.getLogger("GameModeQt")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_window(self):
        """Configura la ventana principal."""
        self.setWindowTitle("Pictionary Live - Modo Juego (PyQt6)")
        self.setGeometry(100, 100, self.config.window_width, self.config.window_height)
        
        # Aplicar stylesheet del gestor de estilos
        self.setStyleSheet(self.style_manager.get_qss())
    
    def _init_ui(self):
        """Inicializa la interfaz de usuario usando componentes modulares."""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Encabezado
        self.header = HeaderComponent(central_widget, self.style_manager, self.logger)
        main_layout.addWidget(self.header)
        
        # Contenedor principal con dos columnas
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # Columna izquierda: Cámara
        self.camera = CameraComponent(central_widget, self.style_manager, self.logger)
        content_layout.addWidget(self.camera, stretch=3)
        
        # Columna derecha: Info del juego
        self.game_info = GameInfoComponent(central_widget, self.style_manager, self.logger)
        content_layout.addWidget(self.game_info, stretch=1)
        
        main_layout.addLayout(content_layout)
        
        # Pie de página: Controles
        self.footer = FooterComponent(central_widget, self.style_manager, self.logger)
        self.footer.clear_clicked.connect(self.clear_predictions)
        self.footer.next_clicked.connect(self._select_next_word)
        main_layout.addWidget(self.footer)
        
        central_widget.setLayout(main_layout)
    
    def _setup_shortcuts(self):
        """Configura atajos de teclado."""
        # L para limpiar
        QShortcut(QKeySequence('L'), self).activated.connect(self.clear_predictions)
        QShortcut(QKeySequence('l'), self).activated.connect(self.clear_predictions)
        
        # C para siguiente
        QShortcut(QKeySequence('C'), self).activated.connect(self._select_next_word)
        QShortcut(QKeySequence('c'), self).activated.connect(self._select_next_word)
        
        # Q para salir
        QShortcut(QKeySequence('Q'), self).activated.connect(self.close)
        QShortcut(QKeySequence('q'), self).activated.connect(self.close)
    
    @pyqtSlot()
    def _select_next_word(self):
        """Hace predicción si hay dibujo, luego selecciona la siguiente palabra aleatoria."""
        # Filtrar palabras no usadas recientemente
        available = [w for w in self.labels if w not in self.recent_words]
        
        # Si se acabaron las palabras nuevas, resetear el histórico
        if not available:
            self.recent_words = []
            available = self.labels
        
        # Seleccionar aleatoriamente
        self.current_word = random.choice(available)
        
        # Agregar al histórico
        self.recent_words.append(self.current_word)
        if len(self.recent_words) > self.max_recent:
            self.recent_words.pop(0)
        
        # Hacer predicción DEMO (sin dibujo real)
        try:
            label, conf, top3 = self.predict_callback(None)
            is_correct = False
            self.game_info.update_prediction(label, conf, top3, is_correct)
        except Exception as e:
            self.logger.error(f"Error en predicción demo: {e}")
        
        # Actualizar UI
        self.game_info.update_word(self.current_word.upper())
        self.game_state = GameState.WAITING_FOR_DRAW
        self._update_camera_state()
        
        # Limpiar predicciones previas y trazas
        self.clear_predictions()
        
        self.logger.info(f"Nueva palabra seleccionada: {self.current_word}")
    
    def _update_camera_state(self):
        """Actualiza el estado mostrado en la cámara."""
        state_map = {
            GameState.WAITING_FOR_DRAW: ("🟢 Esperando dibujo", self.style_manager.get_color("success")),
            GameState.DRAWING: ("🟡 Dibujando", self.style_manager.get_color("accent_secondary")),
            GameState.PREDICTING: ("🔵 Prediciendo", self.style_manager.get_color("accent_primary")),
            GameState.CORRECT: ("✅ ¡Correcto!", self.style_manager.get_color("success")),
            GameState.INCORRECT: ("❌ Incorrecto", self.style_manager.get_color("warning")),
            GameState.PAUSED: ("⏸️  Pausado", self.style_manager.get_color("text_dim")),
        }
        text, color = state_map.get(self.game_state, ("?", self.style_manager.get_color("text_dim")))
        self.camera.update_state(text, color)
    
    @pyqtSlot()
    def predict_drawing(self):
        """Realiza predicción del dibujo actual."""
        if self.game_state == GameState.PREDICTING:
            self.logger.warning("Ya se está realizando una predicción")
            return
        
        self.game_state = GameState.PREDICTING
        self._update_camera_state()
        
        # Ejecutar predicción
        self._do_prediction()
    
    def _do_prediction(self):
        """Ejecuta la predicción."""
        try:
            label, conf, top3 = self.predict_callback(None)
            
            # Verificar si es correcta
            is_correct = label.lower() == self.current_word.lower()
            self.game_info.update_prediction(label, conf, top3, is_correct)
            
            if is_correct:
                self.game_state = GameState.CORRECT
                self.score += 1
                self.streak += 1
                self.logger.info(f"✅ ¡Correcto! Racha: {self.streak}")
            else:
                self.game_state = GameState.INCORRECT
                self.streak = 0
                self.logger.info(f"❌ Incorrecto. La palabra era: {self.current_word}")
            
            self._update_camera_state()
            self.header.update_score(self.score, self.streak)
        
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            self.game_state = GameState.WAITING_FOR_DRAW
            self._update_camera_state()
    
    @pyqtSlot()
    def reset_game(self):
        """Reinicia el juego."""
        self.score = 0
        self.streak = 0
        self.recent_words = []
        self.header.update_score(self.score, self.streak)
        self.logger.info("Juego reiniciado")
    
    @pyqtSlot()
    def clear_predictions(self):
        """Limpia las predicciones mostradas en la pantalla."""
        self.game_info.clear_predictions()
        self.clear_callback()
        self.logger.info("Predicciones y trazas limpiadas")
    
    @pyqtSlot(str, float, list)
    def update_real_time_prediction(self, label: str, confidence: float, top3: List[Tuple[str, float]]):
        """
        Actualiza la predicción en tiempo real en la UI.
        
        Args:
            label: Etiqueta predicha
            confidence: Confianza de la predicción
            top3: Lista de top 3 predicciones
        """
        try:
            self.game_info.update_prediction(label, confidence, top3, False)
        except Exception as e:
            self.logger.error(f"Error actualizando predicción en tiempo real: {e}")
    
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        """
        Actualiza el frame de la cámara mostrado en la UI.
        
        Args:
            frame: Array de imagen en BGR (OpenCV)
        """
        try:
            self.camera.update_frame(frame)
        except Exception as e:
            self.logger.error(f"Error actualizando frame: {e}")
    
    def run(self):
        """Inicia la interfaz del juego."""
        self.logger.info("Iniciando interfaz del juego PyQt6")
        self.show()


def main():
    """Función principal para testing del GameMode."""
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("GameModeTest")
    
    # Labels de ejemplo
    labels = ["cat", "dog", "house", "tree", "car", "sun", "moon", "star"]
    
    # Callback de predicción demo
    def predict_demo(drawing: Optional[np.ndarray]) -> Tuple[str, float, List[Tuple[str, float]]]:
        import random
        label = random.choice(labels)
        conf = random.uniform(0.5, 0.95)
        top3 = [(random.choice(labels), random.uniform(0.1, 0.9)) for _ in range(3)]
        return label, conf, top3
    
    # Crear aplicación
    app = QApplication(sys.argv)
    
    # Crear ventana del juego
    game = GameModeQt(
        labels=labels,
        predict_callback=predict_demo,
        logger=logger
    )
    
    game.run()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
