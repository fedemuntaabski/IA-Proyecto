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


class ColorScheme:
    """Esquema de colores para la UI."""
    
    # Paleta Cyberpunk
    CYBERPUNK = {
        "bg_primary": "#0a1428",
        "bg_secondary": "#192540",
        "accent_primary": "#00ffff",
        "accent_secondary": "#ffa000",
        "success": "#64ff64",
        "warning": "#ff6400",
        "text_primary": "#ebebeb",
        "text_secondary": "#b4b4be",
        "text_dim": "#6a6a6e",
    }
    
    # Paleta Light
    LIGHT = {
        "bg_primary": "#ffffff",
        "bg_secondary": "#f5f5f5",
        "accent_primary": "#0066cc",
        "accent_secondary": "#ff9900",
        "success": "#00aa00",
        "warning": "#ff3300",
        "text_primary": "#000000",
        "text_secondary": "#333333",
        "text_dim": "#999999",
    }
    
    # Paleta Dark
    DARK = {
        "bg_primary": "#1e1e1e",
        "bg_secondary": "#2d2d2d",
        "accent_primary": "#4a9eff",
        "accent_secondary": "#ff7043",
        "success": "#66bb6a",
        "warning": "#ef5350",
        "text_primary": "#ffffff",
        "text_secondary": "#e0e0e0",
        "text_dim": "#9e9e9e",
    }
    
    @staticmethod
    def get_scheme(theme: str) -> dict:
        """Obtiene el esquema de colores según el tema."""
        schemes = {
            "cyberpunk": ColorScheme.CYBERPUNK,
            "light": ColorScheme.LIGHT,
            "dark": ColorScheme.DARK,
        }
        return schemes.get(theme, ColorScheme.CYBERPUNK)


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
        
        # Interfaz Tkinter
        self.root = tk.Tk()
        
        # Esquema de colores (definir ANTES de usar)
        self.colors = ColorScheme.get_scheme(self.config.theme)
        
        self.root.title("Pictionary Live - Modo Juego")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        self.root.configure(bg=self._get_color("bg_primary"))
        
        # Variables de UI
        self.camera_label = None
        self.word_label = None
        self.state_label = None
        self.score_label = None
        self.prediction_label = None
        self.confidence_label = None
        self.top3_label = None
        
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
        """Obtiene un color del esquema actual."""
        return self.colors.get(color_key, "#ffffff")
    
    def _init_ui(self):
        """Inicializa la interfaz Tkinter."""
        # Marco principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Encabezado
        self._init_header(main_frame)
        
        # Contenedor principal con dos columnas
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Columna izquierda: Cámara
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self._init_camera_section(left_frame)
        
        # Columna derecha: Info del juego
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self._init_game_info_section(right_frame)
        
        # Pie de página: Controles
        self._init_footer(main_frame)
    
    def _init_header(self, parent):
        """Inicializa la sección de encabezado."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Título
        title_label = tk.Label(
            header_frame,
            text="🎮 PICTIONARY LIVE - MODO JUEGO",
            font=(self.config.font_family, self.config.font_size_title, "bold"),
            bg=self._get_color("bg_primary"),
            fg=self._get_color("accent_primary"),
        )
        title_label.pack(side=tk.LEFT)
        
        # Score a la derecha
        self.score_label = tk.Label(
            header_frame,
            text="Puntuación: 0 | Racha: 0",
            font=(self.config.font_family, self.config.font_size_normal + 4, "bold"),
            bg=self._get_color("bg_primary"),
            fg=self._get_color("success"),
        )
        self.score_label.pack(side=tk.RIGHT)
    
    def _init_camera_section(self, parent):
        """Inicializa la sección de cámara."""
        # Marco de cámara
        camera_frame = tk.LabelFrame(
            parent,
            text="📷 CÁMARA",
            font=(self.config.font_family, self.config.font_size_normal, "bold"),
            bg=self._get_color("bg_secondary"),
            fg=self._get_color("accent_primary"),
            padx=5,
            pady=5,
        )
        camera_frame.pack(fill=tk.BOTH, expand=True)
        
        # Label para mostrar video
        self.camera_label = tk.Label(
            camera_frame,
            bg=self._get_color("bg_primary"),
            height=400,
            width=500,
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Estado de conexión
        self.state_label = tk.Label(
            camera_frame,
            text="🔴 Desconectado",
            font=(self.config.font_family, self.config.font_size_normal),
            bg=self._get_color("bg_primary"),
            fg=self._get_color("warning"),
        )
        self.state_label.pack(fill=tk.X, pady=(5, 0))
    
    def _init_game_info_section(self, parent):
        """Inicializa la sección de información del juego."""
        # Marco de palabra
        word_frame = tk.LabelFrame(
            parent,
            text="📝 PALABRA ACTUAL",
            font=(self.config.font_family, self.config.font_size_normal, "bold"),
            bg=self._get_color("bg_secondary"),
            fg=self._get_color("accent_primary"),
            padx=10,
            pady=10,
        )
        word_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        # Palabra grande
        self.word_label = tk.Label(
            word_frame,
            text="--",
            font=(self.config.font_family, self.config.font_size_word, "bold"),
            bg=self._get_color("bg_secondary"),
            fg=self._get_color("accent_secondary"),
            wraplength=300,
        )
        self.word_label.pack(fill=tk.BOTH, expand=True)
        
        # Marco de predicción
        pred_frame = tk.LabelFrame(
            parent,
            text="🤖 PREDICCIÓN",
            font=(self.config.font_family, self.config.font_size_normal, "bold"),
            bg=self._get_color("bg_secondary"),
            fg=self._get_color("accent_primary"),
            padx=10,
            pady=10,
        )
        pred_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Predicción principal
        self.prediction_label = tk.Label(
            pred_frame,
            text="(Sin predicción)",
            font=(self.config.font_family, self.config.font_size_normal + 6, "bold"),
            bg=self._get_color("bg_secondary"),
            fg=self._get_color("text_secondary"),
        )
        self.prediction_label.pack(fill=tk.X, pady=(0, 5))
        
        # Confianza
        self.confidence_label = tk.Label(
            pred_frame,
            text="",
            font=(self.config.font_family, self.config.font_size_normal),
            bg=self._get_color("bg_secondary"),
            fg=self._get_color("text_dim"),
        )
        self.confidence_label.pack(fill=tk.X)
        
        # Top-3
        self.top3_label = tk.Label(
            pred_frame,
            text="",
            font=(self.config.font_family, self.config.font_size_normal - 2),
            bg=self._get_color("bg_secondary"),
            fg=self._get_color("text_dim"),
            justify=tk.LEFT,
        )
        self.top3_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
    
    def _init_footer(self, parent):
        """Inicializa la sección de controles."""
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Botones
        predict_btn = tk.Button(
            footer_frame,
            text="🎯 PREDECIR (Enter)",
            command=self.predict_drawing,
            bg=self._get_color("accent_primary"),
            fg=self._get_color("bg_primary"),
            font=(self.config.font_family, self.config.font_size_normal, "bold"),
            padx=20,
            pady=10,
        )
        predict_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = tk.Button(
            footer_frame,
            text="🧹 LIMPIAR (L)",
            command=self.clear_predictions,
            bg=self._get_color("warning"),
            fg=self._get_color("bg_primary"),
            font=(self.config.font_family, self.config.font_size_normal, "bold"),
            padx=20,
            pady=10,
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        next_btn = tk.Button(
            footer_frame,
            text="⏭️  SIGUIENTE (C)",
            command=self._select_next_word,
            bg=self._get_color("accent_secondary"),
            fg=self._get_color("bg_primary"),
            font=(self.config.font_family, self.config.font_size_normal, "bold"),
            padx=20,
            pady=10,
        )
        next_btn.pack(side=tk.LEFT)
        
        # Binding de teclas
        self.root.bind('<Return>', lambda e: self.predict_drawing())
        self.root.bind('<l>', lambda e: self.clear_predictions())
        self.root.bind('<L>', lambda e: self.clear_predictions())
        self.root.bind('<c>', lambda e: self._select_next_word())
        self.root.bind('<C>', lambda e: self._select_next_word())
    
    def _select_next_word(self):
        """Selecciona la siguiente palabra aleatoria."""
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
        self.word_label.config(text=self.current_word.upper())
        self.game_state = GameState.WAITING_FOR_DRAW
        self._update_state_label()
        
        # Limpiar predicciones previas
        self.prediction_label.config(text="(Sin predicción)")
        self.confidence_label.config(text="")
        self.top3_label.config(text="")
        
        self.logger.info(f"Nueva palabra seleccionada: {self.current_word}")
    
    def predict_drawing(self):
        """Realiza predicción del dibujo actual."""
        if self.game_state == GameState.PREDICTING:
            self.logger.warning("Ya se está realizando una predicción")
            return
        
        self.game_state = GameState.PREDICTING
        self._update_state_label()
        
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
            self._update_prediction(label, conf, top3)
            
            # Verificar si es correcta
            is_correct = label.lower() == self.current_word.lower()
            
            if is_correct:
                self.game_state = GameState.CORRECT
                self.score += 1
                self.streak += 1
                self.logger.info(f"✅ ¡Correcto! Racha: {self.streak}")
                
                # Cambiar palabra automáticamente después de un tiempo
                self.root.after(self.config.auto_next_delay_ms, self._select_next_word)
            else:
                self.game_state = GameState.INCORRECT
                self.streak = 0
                self.logger.info(f"❌ Incorrecto. La palabra era: {self.current_word}")
            
            self._update_state_label()
            self._update_score_label()
        
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            self.game_state = GameState.WAITING_FOR_DRAW
            self._update_state_label()
    
    def _update_prediction(self, label: str, conf: float, top3: List[Tuple[str, float]]):
        """Actualiza los labels de predicción en la UI."""
        # Predicción principal
        color = self._get_color("success") if label.lower() == self.current_word.lower() else self._get_color("warning")
        self.prediction_label.config(text=label.upper(), fg=color)
        
        # Confianza
        self.confidence_label.config(text=f"Confianza: {conf*100:.1f}%")
        
        # Top-3
        top3_text = "Top-3:\n" + "\n".join([f"  • {l}: {p*100:.1f}%" for l, p in top3[:3]])
        self.top3_label.config(text=top3_text)
    
    def _update_state_label(self):
        """Actualiza el label de estado."""
        state_map = {
            GameState.WAITING_FOR_DRAW: ("🟢 Esperando dibujo", self._get_color("success")),
            GameState.DRAWING: ("🟡 Dibujando", self._get_color("accent_secondary")),
            GameState.PREDICTING: ("🔵 Prediciendo", self._get_color("accent_primary")),
            GameState.CORRECT: ("✅ ¡Correcto!", self._get_color("success")),
            GameState.INCORRECT: ("❌ Incorrecto", self._get_color("warning")),
            GameState.PAUSED: ("⏸️  Pausado", self._get_color("text_dim")),
        }
        
        text, color = state_map.get(self.game_state, ("?", self._get_color("text_dim")))
        self.state_label.config(text=text, fg=color)
    
    def _update_score_label(self):
        """Actualiza el label de puntuación."""
        self.score_label.config(
            text=f"Puntuación: {self.score} | Racha: {self.streak}"
        )
    
    def reset_game(self):
        """Reinicia el juego."""
        self.score = 0
        self.streak = 0
        self.recent_words = []
        self._update_score_label()
        self.logger.info("Juego reiniciado")
    
    def clear_predictions(self):
        """Limpia las predicciones mostradas en la pantalla."""
        self.prediction_label.config(text="(Sin predicción)")
        self.confidence_label.config(text="")
        self.top3_label.config(text="")
        self.logger.info("Predicciones limpiadas")
    
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
            
            # Actualizar label
            self.camera_label.config(image=tk_image)
            self.camera_label.image = tk_image  # Mantener referencia
        
        except Exception as e:
            self.logger.error(f"Error actualizando frame: {e}")
