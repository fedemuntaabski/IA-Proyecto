"""
User Preferences Manager - Gestor de Preferencias del Usuario.

Este módulo gestiona las preferencias del usuario como tema, idioma,
volumen de sonidos, y otras configuraciones personales.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from ..i18n import _


class UserPreferences:
    """Gestor de preferencias de usuario con persistencia en archivo."""

    DEFAULT_PREFERENCES = {
        'theme': 'default',
        'language': 'es',
        'sound_enabled': True,
        'sound_volume': 0.5,
        'show_tutorials': True,
        'help_on_startup': False,
        'show_fps': True,
        'show_confidence': True,
        'auto_classify_threshold': 10,
        'keyboard_shortcuts': {
            'clear': 'r',
            'force_classify': 'space',
            'toggle_help': 'h',
            'toggle_profile': 'p',
            'quit': 'q'
        },
        'ui_scale': 1.0,
        'auto_save_drawings': False
    }

    def __init__(self, preferences_file: str = "config/user_preferences.json"):
        """
        Inicializa el gestor de preferencias.

        Args:
            preferences_file: Ruta al archivo de preferencias
        """
        self.preferences_file = Path(preferences_file)
        self.preferences_file.parent.mkdir(parents=True, exist_ok=True)
        self.preferences = self.DEFAULT_PREFERENCES.copy()
        self.load()

    def load(self) -> None:
        """Carga preferencias desde el archivo."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Fusionar con valores por defecto para permitir nuevas opciones
                    self.preferences = {**self.DEFAULT_PREFERENCES, **loaded}
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  {_('Error cargando preferencias')}: {e}")
                self.preferences = self.DEFAULT_PREFERENCES.copy()

    def save(self) -> None:
        """Guarda preferencias en el archivo."""
        try:
            with open(self.preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️  {_('Error guardando preferencias')}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene una preferencia.

        Args:
            key: Clave de la preferencia
            default: Valor por defecto si no existe

        Returns:
            Valor de la preferencia
        """
        return self.preferences.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Establece una preferencia.

        Args:
            key: Clave de la preferencia
            value: Nuevo valor
        """
        self.preferences[key] = value

    def get_all(self) -> Dict[str, Any]:
        """Retorna todas las preferencias."""
        return self.preferences.copy()

    def reset_to_defaults(self) -> None:
        """Reinicia las preferencias a valores por defecto."""
        self.preferences = self.DEFAULT_PREFERENCES.copy()
        self.save()

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Actualiza múltiples preferencias de una vez.

        Args:
            updates: Diccionario con actualizaciones
        """
        self.preferences.update(updates)

    def get_keyboard_shortcut(self, action: str) -> Optional[str]:
        """
        Obtiene la tecla de atajo para una acción.

        Args:
            action: Nombre de la acción

        Returns:
            Tecla de atajo o None
        """
        shortcuts = self.preferences.get('keyboard_shortcuts', {})
        return shortcuts.get(action)

    def set_keyboard_shortcut(self, action: str, key: str) -> None:
        """
        Establece la tecla de atajo para una acción.

        Args:
            action: Nombre de la acción
            key: Nueva tecla de atajo
        """
        if 'keyboard_shortcuts' not in self.preferences:
            self.preferences['keyboard_shortcuts'] = {}
        self.preferences['keyboard_shortcuts'][action] = key

    def get_theme(self) -> str:
        """Obtiene el tema actual."""
        return self.preferences.get('theme', 'default')

    def set_theme(self, theme: str) -> None:
        """Establece el tema."""
        self.preferences['theme'] = theme

    def get_language(self) -> str:
        """Obtiene el idioma actual."""
        return self.preferences.get('language', 'es')

    def set_language(self, language: str) -> None:
        """Establece el idioma."""
        self.preferences['language'] = language

    def is_sound_enabled(self) -> bool:
        """Verifica si los sonidos están habilitados."""
        return self.preferences.get('sound_enabled', True)

    def set_sound_enabled(self, enabled: bool) -> None:
        """Habilita o deshabilita los sonidos."""
        self.preferences['sound_enabled'] = enabled

    def get_sound_volume(self) -> float:
        """Obtiene el volumen de sonidos (0.0-1.0)."""
        return max(0.0, min(1.0, self.preferences.get('sound_volume', 0.5)))

    def set_sound_volume(self, volume: float) -> None:
        """Establece el volumen de sonidos (0.0-1.0)."""
        self.preferences['sound_volume'] = max(0.0, min(1.0, volume))

    def get_ui_scale(self) -> float:
        """Obtiene la escala de la UI."""
        return max(0.5, min(2.0, self.preferences.get('ui_scale', 1.0)))

    def set_ui_scale(self, scale: float) -> None:
        """Establece la escala de la UI."""
        self.preferences['ui_scale'] = max(0.5, min(2.0, scale))
