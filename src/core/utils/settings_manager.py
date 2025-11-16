"""
Settings Manager - Gestor de Configuración Avanzada.

Este módulo gestiona configuraciones avanzadas de la aplicación,
permitiendo personalización profunda del comportamiento.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from ..i18n import _


class SettingsManager:
    """Gestor de configuración avanzada de la aplicación."""

    DEFAULT_SETTINGS = {
        'performance': {
            'target_fps': 30,
            'max_frame_size': 640,
            'enable_async_processing': True,
            'enable_gpu_acceleration': True,
            'gpu_memory_growth': True,
            'enable_roi_optimization': True,
            'enable_frame_quality_optimization': True
        },
        'ui': {
            'theme': 'default',
            'scale': 1.0,
            'show_fps': True,
            'show_confidence': True,
            'help_on_startup': False,
            'show_diagnostics': False
        },
        'audio': {
            'enabled': True,
            'volume': 0.5
        },
        'classification': {
            'min_points': 10,
            'confidence_threshold': 0.5,
            'auto_retrain': False
        },
        'detection': {
            'min_hand_area': 5000,
            'max_hand_area': 50000,
            'enable_advanced_vision': True,
            'sensitivity': 0.6,
            'enable_adaptive_sensitivity': True,
            'enable_lighting_compensation': True
        },
        'privacy': {
            'auto_save_drawings': False,
            'collect_analytics': True,
            'collect_feedback': True
        },
        'accessibility': {
            'high_contrast': False,
            'increase_font_size': False,
            'screen_reader_support': False
        }
    }

    def __init__(self, settings_file: str = "config/app_settings.json"):
        """
        Inicializa el gestor de configuración.

        Args:
            settings_file: Ruta al archivo de configuración
        """
        self.settings_file = Path(settings_file)
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.load()

    def load(self) -> None:
        """Carga la configuración desde archivo."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Fusionar con valores por defecto
                    self.settings = self._deep_merge(
                        self.DEFAULT_SETTINGS.copy(), loaded
                    )
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  {_('Error cargando configuración')}: {e}")
                self.settings = self.DEFAULT_SETTINGS.copy()

    def save(self) -> None:
        """Guarda la configuración en archivo."""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️  {_('Error guardando configuración')}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene una configuración usando notación de puntos.

        Args:
            key: Clave con notación de puntos (ej: 'performance.target_fps')
            default: Valor por defecto

        Returns:
            Valor de la configuración
        """
        keys = key.split('.')
        value = self.settings

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """
        Establece una configuración usando notación de puntos.

        Args:
            key: Clave con notación de puntos (ej: 'performance.target_fps')
            value: Nuevo valor
        """
        keys = key.split('.')
        current = self.settings

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Obtiene una sección completa de configuración.

        Args:
            section: Nombre de la sección

        Returns:
            Diccionario con la sección
        """
        return self.settings.get(section, {}).copy()

    def reset_to_defaults(self) -> None:
        """Reinicia la configuración a valores por defecto."""
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.save()

    def validate(self) -> tuple[bool, list]:
        """
        Valida la configuración actual.

        Returns:
            Tupla de (válido, lista_de_errores)
        """
        errors = []

        # Validar performance
        target_fps = self.get('performance.target_fps', 30)
        if not isinstance(target_fps, int) or target_fps < 1 or target_fps > 120:
            errors.append(_("FPS target debe estar entre 1 y 120"))

        max_size = self.get('performance.max_frame_size', 640)
        if not isinstance(max_size, int) or max_size < 320 or max_size > 1920:
            errors.append(_("Tamaño máximo de frame debe estar entre 320 y 1920"))

        # Validar audio
        volume = self.get('audio.volume', 0.5)
        if not (0.0 <= volume <= 1.0):
            errors.append(_("Volumen debe estar entre 0.0 y 1.0"))

        # Validar clasificación
        min_points = self.get('classification.min_points', 10)
        if not isinstance(min_points, int) or min_points < 5 or min_points > 100:
            errors.append(_("Puntos mínimos debe estar entre 5 y 100"))

        # Validar UI
        scale = self.get('ui.scale', 1.0)
        if not (0.5 <= scale <= 2.0):
            errors.append(_("Escala UI debe estar entre 0.5 y 2.0"))

        return len(errors) == 0, errors

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        Fusiona dos diccionarios de forma profunda.

        Args:
            base: Diccionario base
            override: Diccionario a fusionar

        Returns:
            Diccionario fusionado
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def export(self, filename: str) -> None:
        """
        Exporta la configuración a un archivo.

        Args:
            filename: Nombre del archivo de exportación
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print(f"✅ {_('Configuración exportada a')} {filename}")
        except IOError as e:
            print(f"❌ {_('Error exportando configuración')}: {e}")

    def import_settings(self, filename: str) -> None:
        """
        Importa la configuración desde un archivo.

        Args:
            filename: Nombre del archivo a importar
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                imported = json.load(f)
                valid, errors = self.validate()

                if not valid:
                    print(f"❌ {_('Configuración inválida')}:")
                    for error in errors:
                        print(f"   • {error}")
                    return

                self.settings = self._deep_merge(
                    self.DEFAULT_SETTINGS.copy(), imported
                )
                self.save()
                print(f"✅ {_('Configuración importada desde')} {filename}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"❌ {_('Error importando configuración')}: {e}")

    def print_settings(self) -> None:
        """Imprime la configuración actual de forma formateada."""
        print("\n" + "=" * 60)
        print(f"⚙️  {_('CONFIGURACIÓN ACTUAL')}")
        print("=" * 60)

        for section, values in self.settings.items():
            print(f"\n📋 {section.upper()}:")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"  • {key}: {value}")
            else:
                print(f"  {values}")

        print("=" * 60 + "\n")


# Instancia global del gestor de configuración
settings_manager = SettingsManager()
