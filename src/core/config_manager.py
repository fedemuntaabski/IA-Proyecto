"""
Sistema de Configuración Avanzada para Air Draw Classifier.

Este módulo maneja la configuración persistente de la aplicación,
incluyendo perfiles de usuario, ajustes de detección, y preferencias.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class DetectionConfig:
    """Configuración de detección de manos."""
    min_area: int = 5000
    max_area: int = 50000
    stability_threshold: int = 3
    max_history_size: int = 10
    illumination_compensation: bool = True

    # Rangos HSV calibrados
    skin_lower: Optional[List[int]] = None
    skin_upper: Optional[List[int]] = None
    background_exclude_lower: Optional[List[int]] = None
    background_exclude_upper: Optional[List[int]] = None


@dataclass
class UIConfig:
    """Configuración de interfaz de usuario."""
    show_fps: bool = True
    show_confidence: bool = True
    show_hand_state: bool = True
    show_controls: bool = True
    theme: str = "dark"  # "dark", "light"

    # Colores (BGR)
    color_success: List[int] = field(default_factory=lambda: [0, 255, 0])
    color_warning: List[int] = field(default_factory=lambda: [0, 165, 255])
    color_error: List[int] = field(default_factory=lambda: [0, 0, 255])
    color_info: List[int] = field(default_factory=lambda: [255, 255, 0])


@dataclass
class MLConfig:
    """Configuración de machine learning."""
    enabled: bool = True
    model_path: str = "IA/sketch_classifier_model.keras"
    model_info_path: str = "IA/model_info.json"
    confidence_threshold: float = 0.5
    top_k_predictions: int = 3
    fallback_enabled: bool = True


@dataclass
class PerformanceConfig:
    """Configuración de performance."""
    target_fps: int = 30
    resolution_width: int = 640
    resolution_height: int = 480
    threading_enabled: bool = False
    gpu_acceleration: bool = True


@dataclass
class UserProfile:
    """Perfil de usuario con configuraciones personalizadas."""
    name: str
    detection: DetectionConfig
    ui: UIConfig
    ml: MLConfig
    performance: PerformanceConfig
    created_at: str
    last_modified: str
    calibration_completed: bool = False


class ConfigManager:
    """
    Administrador de configuración avanzada con perfiles de usuario.

    Atributos:
        config_dir: Directorio donde se almacenan las configuraciones
        current_profile: Perfil activo actualmente
        profiles: Diccionario de perfiles disponibles
    """

    def __init__(self, config_dir: str = "config"):
        """
        Inicializa el administrador de configuración.

        Args:
            config_dir: Directorio para archivos de configuración
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.profiles: Dict[str, UserProfile] = {}
        self.current_profile: Optional[UserProfile] = None
        self.default_profile_name = "default"

        # Cargar configuraciones existentes
        self._load_profiles()

        # Si no hay perfiles, crear uno por defecto
        if not self.profiles:
            self._create_default_profile()

        # Establecer perfil activo (el último usado o default)
        self._load_active_profile()

        print("✓ ConfigManager inicializado")

    def _create_default_profile(self) -> None:
        """Crea un perfil por defecto con configuraciones estándar."""
        default_profile = UserProfile(
            name=self.default_profile_name,
            detection=DetectionConfig(),
            ui=UIConfig(),
            ml=MLConfig(),
            performance=PerformanceConfig(),
            created_at=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            calibration_completed=False
        )

        self.profiles[self.default_profile_name] = default_profile
        self._save_profile(default_profile)
        print(f"✓ Perfil por defecto creado: {self.default_profile_name}")

    def _load_profiles(self) -> None:
        """Carga todos los perfiles desde archivos."""
        if not self.config_dir.exists():
            return

        for config_file in self.config_dir.glob("profile_*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convertir diccionarios anidados a objetos de configuración
                profile = UserProfile(
                    name=data['name'],
                    detection=DetectionConfig(**data.get('detection', {})),
                    ui=UIConfig(**data.get('ui', {})),
                    ml=MLConfig(**data.get('ml', {})),
                    performance=PerformanceConfig(**data.get('performance', {})),
                    created_at=data.get('created_at', datetime.now().isoformat()),
                    last_modified=data.get('last_modified', datetime.now().isoformat()),
                    calibration_completed=data.get('calibration_completed', False)
                )

                self.profiles[profile.name] = profile

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"⚠ Error cargando perfil {config_file}: {e}")

        print(f"✓ {len(self.profiles)} perfiles cargados")

    def _save_profile(self, profile: UserProfile) -> None:
        """Guarda un perfil en archivo."""
        profile.last_modified = datetime.now().isoformat()

        config_file = self.config_dir / f"profile_{profile.name}.json"

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                # Convertir objetos dataclass a diccionarios
                data = {
                    'name': profile.name,
                    'detection': asdict(profile.detection),
                    'ui': asdict(profile.ui),
                    'ml': asdict(profile.ml),
                    'performance': asdict(profile.performance),
                    'created_at': profile.created_at,
                    'last_modified': profile.last_modified,
                    'calibration_completed': profile.calibration_completed
                }
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠ Error guardando perfil {profile.name}: {e}")

    def _load_active_profile(self) -> None:
        """Carga el perfil activo desde archivo de estado."""
        state_file = self.config_dir / "active_profile.txt"

        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    profile_name = f.read().strip()

                if profile_name in self.profiles:
                    self.current_profile = self.profiles[profile_name]
                    print(f"✓ Perfil activo: {profile_name}")
                    return
            except Exception as e:
                print(f"⚠ Error cargando perfil activo: {e}")

        # Usar perfil por defecto
        self.current_profile = self.profiles.get(self.default_profile_name)
        if self.current_profile:
            self.set_active_profile(self.default_profile_name)

    def set_active_profile(self, profile_name: str) -> bool:
        """
        Establece el perfil activo.

        Args:
            profile_name: Nombre del perfil a activar

        Returns:
            True si el perfil fue activado exitosamente
        """
        if profile_name not in self.profiles:
            print(f"⚠ Perfil '{profile_name}' no encontrado")
            return False

        self.current_profile = self.profiles[profile_name]

        # Guardar estado
        state_file = self.config_dir / "active_profile.txt"
        try:
            with open(state_file, 'w') as f:
                f.write(profile_name)
        except Exception as e:
            print(f"⚠ Error guardando estado del perfil activo: {e}")

        print(f"✓ Perfil activo cambiado a: {profile_name}")
        return True

    def create_profile(self, name: str, base_profile: Optional[str] = None) -> bool:
        """
        Crea un nuevo perfil basado en uno existente o por defecto.

        Args:
            name: Nombre del nuevo perfil
            base_profile: Nombre del perfil base (opcional)

        Returns:
            True si el perfil fue creado exitosamente
        """
        if name in self.profiles:
            print(f"⚠ Perfil '{name}' ya existe")
            return False

        # Usar perfil base o por defecto
        base = self.profiles.get(base_profile or self.default_profile_name)
        if not base:
            print("⚠ No se puede crear perfil: perfil base no encontrado")
            return False

        # Crear nuevo perfil copiando configuración
        new_profile = UserProfile(
            name=name,
            detection=DetectionConfig(**asdict(base.detection)),
            ui=UIConfig(**asdict(base.ui)),
            ml=MLConfig(**asdict(base.ml)),
            performance=PerformanceConfig(**asdict(base.performance)),
            created_at=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            calibration_completed=False
        )

        self.profiles[name] = new_profile
        self._save_profile(new_profile)

        print(f"✓ Perfil '{name}' creado basado en '{base.name}'")
        return True

    def delete_profile(self, name: str) -> bool:
        """
        Elimina un perfil.

        Args:
            name: Nombre del perfil a eliminar

        Returns:
            True si el perfil fue eliminado exitosamente
        """
        if name not in self.profiles:
            print(f"⚠ Perfil '{name}' no encontrado")
            return False

        if name == self.default_profile_name:
            print("⚠ No se puede eliminar el perfil por defecto")
            return False

        # Si es el perfil activo, cambiar a default
        if self.current_profile and self.current_profile.name == name:
            self.set_active_profile(self.default_profile_name)

        # Eliminar archivo
        config_file = self.config_dir / f"profile_{name}.json"
        try:
            if config_file.exists():
                os.remove(config_file)
        except Exception as e:
            print(f"⚠ Error eliminando archivo de perfil: {e}")

        # Remover de memoria
        del self.profiles[name]

        print(f"✓ Perfil '{name}' eliminado")
        return True

    def update_detection_config(self, **kwargs) -> None:
        """Actualiza la configuración de detección del perfil activo."""
        if not self.current_profile:
            return

        for key, value in kwargs.items():
            if hasattr(self.current_profile.detection, key):
                setattr(self.current_profile.detection, key, value)

        self._save_profile(self.current_profile)
        print("✓ Configuración de detección actualizada")

    def update_ui_config(self, **kwargs) -> None:
        """Actualiza la configuración de UI del perfil activo."""
        if not self.current_profile:
            return

        for key, value in kwargs.items():
            if hasattr(self.current_profile.ui, key):
                setattr(self.current_profile.ui, key, value)

        self._save_profile(self.current_profile)
        print("✓ Configuración de UI actualizada")

    def update_ml_config(self, **kwargs) -> None:
        """Actualiza la configuración de ML del perfil activo."""
        if not self.current_profile:
            return

        for key, value in kwargs.items():
            if hasattr(self.current_profile.ml, key):
                setattr(self.current_profile.ml, key, value)

        self._save_profile(self.current_profile)
        print("✓ Configuración de ML actualizada")

    def get_detection_config(self) -> DetectionConfig:
        """Obtiene la configuración de detección del perfil activo."""
        return self.current_profile.detection if self.current_profile else DetectionConfig()

    def get_ui_config(self) -> UIConfig:
        """Obtiene la configuración de UI del perfil activo."""
        return self.current_profile.ui if self.current_profile else UIConfig()

    def get_ml_config(self) -> MLConfig:
        """Obtiene la configuración de ML del perfil activo."""
        return self.current_profile.ml if self.current_profile else MLConfig()

    def get_performance_config(self) -> PerformanceConfig:
        """Obtiene la configuración de performance del perfil activo."""
        return self.current_profile.performance if self.current_profile else PerformanceConfig()

    def list_profiles(self) -> List[str]:
        """Lista todos los perfiles disponibles."""
        return list(self.profiles.keys())

    def get_profile_info(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Obtiene información detallada de un perfil."""
        profile = self.profiles.get(name or (self.current_profile.name if self.current_profile else None))

        if not profile:
            return None

        return {
            'name': profile.name,
            'active': self.current_profile and self.current_profile.name == profile.name,
            'calibration_completed': profile.calibration_completed,
            'created_at': profile.created_at,
            'last_modified': profile.last_modified,
            'detection_config': asdict(profile.detection),
            'ui_config': asdict(profile.ui),
            'ml_config': asdict(profile.ml),
            'performance_config': asdict(profile.performance)
        }

    def export_profile(self, name: str, export_path: str) -> bool:
        """
        Exporta un perfil a un archivo JSON.

        Args:
            name: Nombre del perfil a exportar
            export_path: Ruta donde guardar el archivo

        Returns:
            True si la exportación fue exitosa
        """
        profile = self.profiles.get(name)
        if not profile:
            return False

        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.get_profile_info(name), f, indent=2, ensure_ascii=False)
            print(f"✓ Perfil '{name}' exportado a {export_path}")
            return True
        except Exception as e:
            print(f"⚠ Error exportando perfil: {e}")
            return False

    def import_profile(self, import_path: str, new_name: Optional[str] = None) -> bool:
        """
        Importa un perfil desde un archivo JSON.

        Args:
            import_path: Ruta del archivo a importar
            new_name: Nuevo nombre para el perfil (opcional)

        Returns:
            True si la importación fue exitosa
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Usar nombre del archivo o especificado
            profile_name = new_name or data.get('name', f"imported_{int(datetime.now().timestamp())}")

            if profile_name in self.profiles:
                print(f"⚠ Perfil '{profile_name}' ya existe")
                return False

            # Crear perfil desde datos importados
            profile = UserProfile(
                name=profile_name,
                detection=DetectionConfig(**data.get('detection_config', {})),
                ui=UIConfig(**data.get('ui_config', {})),
                ml=MLConfig(**data.get('ml_config', {})),
                performance=PerformanceConfig(**data.get('performance_config', {})),
                created_at=data.get('created_at', datetime.now().isoformat()),
                last_modified=datetime.now().isoformat(),
                calibration_completed=data.get('calibration_completed', False)
            )

            self.profiles[profile_name] = profile
            self._save_profile(profile)

            print(f"✓ Perfil '{profile_name}' importado desde {import_path}")
            return True

        except Exception as e:
            print(f"⚠ Error importando perfil: {e}")
            return False


# Instancia global para acceso fácil
config_manager = ConfigManager()


if __name__ == "__main__":
    # Test del sistema de configuración
    print("Testing ConfigManager...")

    # Listar perfiles
    print(f"Perfiles disponibles: {config_manager.list_profiles()}")

    # Obtener configuración actual
    detection = config_manager.get_detection_config()
    ui = config_manager.get_ui_config()
    ml = config_manager.get_ml_config()

    print(f"Detección - Área mínima: {detection.min_area}")
    print(f"UI - Tema: {ui.theme}")
    print(f"ML - Habilitado: {ml.enabled}")

    # Crear un perfil de prueba
    config_manager.create_profile("test_profile")
    print(f"Perfiles después de crear: {config_manager.list_profiles()}")

    # Cambiar configuración
    config_manager.update_detection_config(min_area=3000)
    config_manager.update_ui_config(theme="light")

    print("✓ ConfigManager test completado")