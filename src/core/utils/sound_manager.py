"""
Sound Manager - Gestor de Sonidos y Audio Feedback.

Este módulo proporciona funcionalidad de audio feedback para eventos de la aplicación.
"""

import os
from typing import Optional
from ..i18n import _


class SoundManager:
    """Gestor de sonidos para la aplicación."""

    SOUNDS = {
        'success': 'sounds/success.wav',
        'error': 'sounds/error.wav',
        'info': 'sounds/info.wav',
        'drawing_start': 'sounds/drawing_start.wav',
        'drawing_end': 'sounds/drawing_end.wav',
        'classification': 'sounds/classification.wav'
    }

    def __init__(self, enabled: bool = True, volume: float = 0.5):
        """
        Inicializa el gestor de sonidos.

        Args:
            enabled: Si los sonidos están habilitados
            volume: Volumen (0.0-1.0)
        """
        self.enabled = enabled
        self.volume = max(0.0, min(1.0, volume))
        self.sounds_dir = "resources/sounds"
        self._init_audio()

    def _init_audio(self) -> None:
        """Inicializa el sistema de audio."""
        try:
            import pygame
            pygame.mixer.init()
            self.pygame = pygame
            self.audio_available = True
        except (ImportError, Exception) as e:
            self.audio_available = False
            if not isinstance(e, ImportError):
                print(f"⚠️  {_('Audio no disponible')}: {e}")

    def play_sound(self, sound_name: str) -> None:
        """
        Reproduce un sonido si está disponible y habilitado.

        Args:
            sound_name: Nombre del sonido ('success', 'error', 'info', etc.)
        """
        if not self.enabled or not self.audio_available:
            return

        if sound_name not in self.SOUNDS:
            return

        try:
            sound_path = os.path.join(self.sounds_dir, sound_name + '.wav')
            if os.path.exists(sound_path):
                sound = self.pygame.mixer.Sound(sound_path)
                sound.set_volume(self.volume)
                sound.play()
        except Exception:
            pass  # Fallar silenciosamente si no puede reproducir

    def play_success(self) -> None:
        """Reproduce sonido de éxito."""
        self.play_sound('success')

    def play_error(self) -> None:
        """Reproduce sonido de error."""
        self.play_sound('error')

    def play_info(self) -> None:
        """Reproduce sonido de información."""
        self.play_sound('info')

    def play_drawing_start(self) -> None:
        """Reproduce sonido de inicio de dibujo."""
        self.play_sound('drawing_start')

    def play_drawing_end(self) -> None:
        """Reproduce sonido de fin de dibujo."""
        self.play_sound('drawing_end')

    def play_classification(self) -> None:
        """Reproduce sonido de clasificación."""
        self.play_sound('classification')

    def set_enabled(self, enabled: bool) -> None:
        """Habilita o deshabilita los sonidos."""
        self.enabled = enabled

    def set_volume(self, volume: float) -> None:
        """Establece el volumen (0.0-1.0)."""
        self.volume = max(0.0, min(1.0, volume))

    def is_available(self) -> bool:
        """Verifica si el audio está disponible."""
        return self.audio_available


# Instancia global del gestor de sonidos
sound_manager = SoundManager(enabled=True, volume=0.5)
