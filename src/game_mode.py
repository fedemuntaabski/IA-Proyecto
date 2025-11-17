"""
game_mode.py - Modo de juego con interfaz Tkinter

Implementa el modo de juego donde se selecciona aleatoriamente una palabra,
el usuario la dibuja, el modelo predice, y se cambia la palabra si es correcta.
"""

import random
import logging
import threading
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

from ui_components import StyleManager, HeaderComponent, CameraComponent, GameInfoComponent, FooterComponent


class GameState(Enum):
    """Estados del juego."""
    WAITING_FOR_DRAW = "waiting"      # Esperando que el usuario dibuje
    DRAWING = "drawing"                # Usuario está dibujando
    PREDICTING = "predicting"          # Modelo está prediciendo
    CORRECT = "correct"                # Predicción correcta
    INCORRECT = "incorrect"            # Predicción incorrecta
    PAUSED = "paused"                  # Juego pausado


@dataclass
class GameConfig:
    """Configuración del modo de juego."""
    window_width: int = 1400
    window_height: int = 800
    font_family: str = "Segoe UI"
    font_size_title: int = 36
    font_size_word: int = 48
    font_size_normal: int = 14
    auto_next_delay_ms: int = 2000    # Milisegundos antes de cambiar palabra automáticamente
    min_drawing_points: int = 10      # Puntos mínimos para hacer predicción
    theme: str = "cyberpunk"          # Tema de colores: "cyberpunk", "light", "dark"


class GameMode:
    """
    Modo de juego principal con interfaz Tkinter.
    
    Flujo:
    1. Se selecciona aleatoriamente una palabra de las disponibles
    2. Se muestra la palabra al usuario
    3. El usuario dibuja usando la cámara
    4. Al presionar "Predecir", el modelo predice
    5. Si es correcta, cambia automáticamente a una palabra nueva
    6. Si es incorrecta, muestra feedback y permite reintentar
    """
    
    def __init__(
        self,
        labels: List[str],
        predict_callback: Callable[[], Tuple[str, float, List[Tuple[str, float]]]],
        config: Optional[GameConfig] = None,
        logger: Optional[logging.Logger] = None,
        clear_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Inicializa el modo de juego.
        
        Args:
            labels: Lista de palabras disponibles para el juego
            predict_callback: Función que predice un dibujo. Signature: (drawing) -> (label, conf, top3)
            config: Configuración del juego
            logger: Logger para debugging
        """
        self.labels = labels
        self.predict_callback = predict_callback
        self.config = config or GameConfig()
        self.logger = logger or self._setup_default_logger()
        
        # Validar labels
        if not self.labels:
            raise ValueError("Se requiere al menos una palabra en labels")
        
        # Estado del juego
        self.current_word = None
        self.game_state = GameState.WAITING_FOR_DRAW
        self.score = 0
        self.streak = 0
        self.recent_words = []  # Evitar repeticiones recientes
        self.max_recent = 5
        
        # Callbacks
        self.predict_callback = predict_callback
        self.clear_callback = clear_callback or (lambda: None)
        
        # Interfaz Tkinter
        self.root = tk.Tk()
        
        # Gestor de estilos
        self.style_manager = StyleManager(self.config.theme)
        
        self.root.title("Pictionary Live - Modo Juego")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        self.root.configure(bg=self.style_manager.get_color("bg_primary"))
        
        # Componentes UI
        self.header = None
        self.camera = None
        self.game_info = None
        self.footer = None
        
        # Control de threading para video
        self.video_thread = None
        self.video_running = False
        self.camera = None
        self.current_frame = None
        
        # Inicializar UI
        self._init_ui()
        
        # Seleccionar primera palabra
        self._select_next_word()
        
        self.logger.info(f"GameMode inicializado con {len(self.labels)} palabras disponibles")
    
    def _setup_default_logger(self) -> logging.Logger:
        """Crea un logger por defecto."""
        logger = logging.getLogger("GameMode")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_color(self, color_key: str) -> str:
        """Obtiene un color del gestor de estilos."""
        return self.style_manager.get_color(color_key)
    
    def _init_ui(self):
        """Inicializa la interfaz Tkinter usando componentes modulares."""
        # Marco principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Encabezado
        self.header = HeaderComponent(main_frame, self.style_manager, self.logger)
        self.header.build()
        
        # Contenedor principal con dos columnas
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Columna izquierda: Cámara
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.camera = CameraComponent(left_frame, self.style_manager, self.logger)
        self.camera.build()
        
        # Columna derecha: Info del juego
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self.game_info = GameInfoComponent(right_frame, self.style_manager, self.logger)
        self.game_info.build()
        
        # Pie de página: Controles
        self.footer = FooterComponent(
            main_frame,
            self.style_manager,
            self.logger,
            on_clear=self.clear_predictions,
            on_next=self._select_next_word
        )
        self.footer.build()
        
        # Bindings de teclas
        self.root.bind('<l>', lambda e: self.clear_predictions())
        self.root.bind('<L>', lambda e: self.clear_predictions())
        self.root.bind('<c>', lambda e: self._select_next_word())
        self.root.bind('<C>', lambda e: self._select_next_word())
    
    def _select_next_word(self):
        """Hace predicción si hay dibujo, luego selecciona la siguiente palabra aleatoria."""
        # Hacer predicción primero si hay dibujo
        try:
            label, conf, top3 = self.predict_callback()
            is_correct = label.lower() == self.current_word.lower()
            self.game_info.update_prediction(label, conf, top3, is_correct)
            
            # Verificar si es correcta
            if is_correct:
                self.game_state = GameState.CORRECT
                self.score += 1
                self.streak += 1
                self.logger.info(f"✅ ¡Correcto! Racha: {self.streak}")
            else:
                self.game_state = GameState.INCORRECT
                self.streak = 0
                self.logger.info(f"❌ Incorrecto. La palabra era: {self.current_word}")
            
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
            self.header.update_score(self.score, self.streak)
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            self.game_state = GameState.WAITING_FOR_DRAW
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
        
        # Ahora seleccionar siguiente palabra
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
        
        # Actualizar UI
        self.game_info.update_word(self.current_word.upper())
        self.game_state = GameState.WAITING_FOR_DRAW
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
        
        # Limpiar predicciones previas y trazas
        self.clear_predictions()
        
        self.logger.info(f"Nueva palabra seleccionada: {self.current_word}")
    
    def predict_drawing(self):
        """Realiza predicción del dibujo actual."""
        if self.game_state == GameState.PREDICTING:
            self.logger.warning("Ya se está realizando una predicción")
            return
        
        self.game_state = GameState.PREDICTING
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
        
        # Ejecutar predicción en thread separado para no bloquear UI
        thread = threading.Thread(target=self._do_prediction, daemon=True)
        thread.start()
    
    def _do_prediction(self):
        """Ejecuta la predicción (en thread separado)."""
        try:
            # Aquí se llamaría al modelo con el dibujo actual
            # Por ahora, simulamos
            # En la implementación real, esto vendría de los datos del dibujo
            
            # Llamar callback de predicción
            # Este método debería ser implementado por quien integre GameMode
            # Simulamos una predicción
            label, conf, top3 = self.predict_callback()  # Sin argumentos
            
            # Actualizar UI
            is_correct = label.lower() == self.current_word.lower()
            self.game_info.update_prediction(label, conf, top3, is_correct)
            
            # Verificar si es correcta
            if is_correct:
                self.game_state = GameState.CORRECT
                self.score += 1
                self.streak += 1
                self.logger.info(f"✅ ¡Correcto! Racha: {self.streak}")
                
                # No cambiar palabra automáticamente
            else:
                self.game_state = GameState.INCORRECT
                self.streak = 0
                self.logger.info(f"❌ Incorrecto. La palabra era: {self.current_word}")
            
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
            self.header.update_score(self.score, self.streak)
        
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            self.game_state = GameState.WAITING_FOR_DRAW
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
    
    def reset_game(self):
        """Reinicia el juego."""
        self.score = 0
        self.streak = 0
        self.recent_words = []
        self.header.update_score(self.score, self.streak)
        self.logger.info("Juego reiniciado")
    
    def clear_predictions(self):
        """Limpia las predicciones mostradas en la pantalla."""
        self.game_info.clear_predictions()
        self.clear_callback()  # Limpiar trazas dibujadas
        self.logger.info("Predicciones y trazas limpiadas")
    
    def quit_game(self):
        """Cierra la aplicación."""
        self.logger.info("Cerrando el juego")
        self.video_running = False
        if self.camera:
            self.camera.release()
        self.root.quit()
    
    def run(self):
        """Inicia la interfaz del juego."""
        self.logger.info("Iniciando interfaz del juego")
        self.root.mainloop()
    
    def update_frame(self, frame: np.ndarray):
        """
        Actualiza el frame de la cámara mostrado en la UI.
        
        Args:
            frame: Array de imagen en BGR (OpenCV)
        """
        try:
            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para que quepa en el label
            display_height = 400
            aspect_ratio = rgb_frame.shape[1] / rgb_frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            rgb_frame = cv2.resize(rgb_frame, (display_width, display_height))
            
            # Convertir a PIL Image
            pil_image = Image.fromarray(rgb_frame)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Actualizar componente de cámara
            self.camera.update_frame(tk_image)
        
        except Exception as e:
            self.logger.error(f"Error actualizando frame: {e}")
