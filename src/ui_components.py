"""
ui_components.py - Componentes modulares de la interfaz de usuario

Contiene clases para cada sección de la UI del modo de juego,
siguiendo el principio de responsabilidad única.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import logging


class StyleManager:
    """Gestor de estilos y colores para la UI."""

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

    def get_font(self, size_key: str, bold: bool = False) -> tuple:
        """Obtiene una configuración de fuente."""
        size = self.fonts.get(size_key, self.fonts["normal"])
        weight = "bold" if bold else "normal"
        return (self.fonts["family"], size, weight)


class HeaderComponent:
    """Componente del encabezado de la aplicación."""

    def __init__(self, parent: tk.Widget, style_manager: StyleManager, logger: logging.Logger):
        self.parent = parent
        self.style = style_manager
        self.logger = logger
        self.frame = None
        self.title_label = None
        self.score_label = None

    def build(self) -> tk.Frame:
        """Construye el componente del encabezado."""
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.X, pady=(0, 10))

        # Título
        self.title_label = tk.Label(
            self.frame,
            text="🎮 PICTIONARY LIVE - MODO JUEGO",
            font=self.style.get_font("title", bold=True),
            bg=self.style.get_color("bg_primary"),
            fg=self.style.get_color("accent_primary"),
        )
        self.title_label.pack(side=tk.LEFT)

        # Score
        self.score_label = tk.Label(
            self.frame,
            text="Puntuación: 0 | Racha: 0",
            font=self.style.get_font("normal", bold=True),
            bg=self.style.get_color("bg_primary"),
            fg=self.style.get_color("success"),
        )
        self.score_label.pack(side=tk.RIGHT)

        self.logger.debug("HeaderComponent construido")
        return self.frame

    def update_score(self, score: int, streak: int):
        """Actualiza la etiqueta de puntuación."""
        self.score_label.config(text=f"Puntuación: {score} | Racha: {streak}")


class CameraComponent:
    """Componente de la sección de cámara."""

    def __init__(self, parent: tk.Widget, style_manager: StyleManager, logger: logging.Logger):
        self.parent = parent
        self.style = style_manager
        self.logger = logger
        self.frame = None
        self.camera_label = None
        self.state_label = None

    def build(self) -> tk.Frame:
        """Construye el componente de cámara."""
        # Marco de cámara
        self.frame = tk.LabelFrame(
            self.parent,
            text="📷 CÁMARA",
            font=self.style.get_font("normal", bold=True),
            bg=self.style.get_color("bg_secondary"),
            fg=self.style.get_color("accent_primary"),
            padx=5,
            pady=5,
        )
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Label para mostrar video
        self.camera_label = tk.Label(
            self.frame,
            bg=self.style.get_color("bg_primary"),
            height=400,
            width=500,
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Estado de conexión
        self.state_label = tk.Label(
            self.frame,
            text="🔴 Desconectado",
            font=self.style.get_font("normal"),
            bg=self.style.get_color("bg_primary"),
            fg=self.style.get_color("warning"),
        )
        self.state_label.pack(fill=tk.X, pady=(5, 0))

        self.logger.debug("CameraComponent construido")
        return self.frame

    def update_frame(self, frame_image):
        """Actualiza el frame de la cámara."""
        self.camera_label.config(image=frame_image)
        self.camera_label.image = frame_image  # Mantener referencia

    def update_state(self, text: str, color: str):
        """Actualiza el estado de la cámara."""
        self.state_label.config(text=text, fg=color)


class GameInfoComponent:
    """Componente de información del juego."""

    def __init__(self, parent: tk.Widget, style_manager: StyleManager, logger: logging.Logger):
        self.parent = parent
        self.style = style_manager
        self.logger = logger
        self.frame = None
        self.word_label = None
        self.prediction_label = None
        self.confidence_label = None
        self.top3_label = None

    def build(self) -> tk.Frame:
        """Construye el componente de información del juego."""
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Marco de palabra
        word_frame = tk.LabelFrame(
            self.frame,
            text="📝 PALABRA ACTUAL",
            font=self.style.get_font("normal", bold=True),
            bg=self.style.get_color("bg_secondary"),
            fg=self.style.get_color("accent_primary"),
            padx=10,
            pady=10,
        )
        word_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        # Palabra grande
        self.word_label = tk.Label(
            word_frame,
            text="--",
            font=self.style.get_font("word", bold=True),
            bg=self.style.get_color("bg_secondary"),
            fg=self.style.get_color("accent_secondary"),
            wraplength=300,
        )
        self.word_label.pack(fill=tk.BOTH, expand=True)

        # Marco de predicción
        pred_frame = tk.LabelFrame(
            self.frame,
            text="🤖 PREDICCIÓN",
            font=self.style.get_font("normal", bold=True),
            bg=self.style.get_color("bg_secondary"),
            fg=self.style.get_color("accent_primary"),
            padx=10,
            pady=10,
        )
        pred_frame.pack(fill=tk.BOTH, expand=True)

        # Predicción principal
        self.prediction_label = tk.Label(
            pred_frame,
            text="(Sin predicción)",
            font=self.style.get_font("normal", bold=True),
            bg=self.style.get_color("bg_secondary"),
            fg=self.style.get_color("text_secondary"),
        )
        self.prediction_label.pack(fill=tk.X, pady=(0, 5))

        # Confianza
        self.confidence_label = tk.Label(
            pred_frame,
            text="",
            font=self.style.get_font("normal"),
            bg=self.style.get_color("bg_secondary"),
            fg=self.style.get_color("text_dim"),
        )
        self.confidence_label.pack(fill=tk.X)

        # Top-3
        self.top3_label = tk.Label(
            pred_frame,
            text="",
            font=self.style.get_font("small"),
            bg=self.style.get_color("bg_secondary"),
            fg=self.style.get_color("text_dim"),
            justify=tk.LEFT,
        )
        self.top3_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.logger.debug("GameInfoComponent construido")
        return self.frame

    def update_word(self, word: str):
        """Actualiza la palabra mostrada."""
        self.word_label.config(text=word.upper())

    def update_prediction(self, label: str, confidence: float, top3: list, is_correct: bool = False):
        """Actualiza la predicción."""
        if is_correct:
            color = self.style.get_color("success")
        else:
            color = self.style.get_color("text_primary")  # Color normal para tiempo real
        
        self.prediction_label.config(text=label.upper(), fg=color)
        self.confidence_label.config(text=f"Confianza: {confidence*100:.1f}%")

        top3_text = "Top-3:\n" + "\n".join([f"  • {l}: {p*100:.1f}%" for l, p in top3[:3]])
        self.top3_label.config(text=top3_text)

    def clear_predictions(self):
        """Limpia las predicciones."""
        self.prediction_label.config(text="(Sin predicción)", fg=self.style.get_color("text_secondary"))
        self.confidence_label.config(text="")
        self.top3_label.config(text="")


class FooterComponent:
    """Componente de controles del pie de página."""

    def __init__(self, parent: tk.Widget, style_manager: StyleManager, logger: logging.Logger,
                 on_clear: Callable, on_next: Callable):
        self.parent = parent
        self.style = style_manager
        self.logger = logger
        self.on_clear = on_clear
        self.on_next = on_next
        self.frame = None

    def build(self) -> tk.Frame:
        """Construye el componente de controles."""
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.X, pady=(10, 0))

        # Botones
        clear_btn = tk.Button(
            self.frame,
            text="🧹 LIMPIAR (L)",
            command=self.on_clear,
            bg=self.style.get_color("warning"),
            fg=self.style.get_color("bg_primary"),
            font=self.style.get_font("normal", bold=True),
            padx=20,
            pady=10,
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))

        next_btn = tk.Button(
            self.frame,
            text="⏭️  SIGUIENTE (C)",
            command=self.on_next,
            bg=self.style.get_color("accent_secondary"),
            fg=self.style.get_color("bg_primary"),
            font=self.style.get_font("normal", bold=True),
            padx=20,
            pady=10,
        )
        next_btn.pack(side=tk.LEFT)

        self.logger.debug("FooterComponent construido")
        return self.frame
