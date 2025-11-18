"""
config_manager.py - Gestión de configuración con validación usando Pydantic
"""

import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError


class MediaPipeHandsConfig(BaseModel):
    static_image_mode: bool = False
    max_num_hands: int = 2
    model_complexity: int = 0
    min_detection_confidence: float = Field(0.2, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(0.2, ge=0.0, le=1.0)


class MediaPipePoseConfig(BaseModel):
    enabled: bool = False


class MediaPipeConfig(BaseModel):
    hands: MediaPipeHandsConfig
    pose: MediaPipePoseConfig


class CameraConfig(BaseModel):
    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    fps: int = Field(30, gt=0)
    buffer_size: int = Field(1, ge=1)
    flip_horizontal: bool = True


class StrokeConfig(BaseModel):
    pause_threshold_ms: int = Field(400, gt=0)
    velocity_threshold: float = Field(0.002, gt=0.0)
    min_points: int = Field(8, gt=0)
    max_stroke_age_ms: int = Field(3000, gt=0)
    min_stroke_length: float = Field(0.05, gt=0.0)
    # Número de frames consecutivos de baja velocidad requeridos antes de cerrar trazos
    low_vel_consecutive: int = Field(3, gt=0)
    # Umbral mínimo de desplazamiento (en coordenadas normalizadas) que cuenta como movimiento
    min_movement_delta: float = Field(0.0005, gt=0.0)
    # Permite agregar puntos con desplazamiento pequeño aún si la velocidad medida es baja
    allow_low_velocity_append: bool = Field(False)


class PreprocessingConfig(BaseModel):
    scale_factor: float = Field(0.8, gt=0.0, le=1.0)
    min_stroke_length: float = Field(0.05, gt=0.0)
    min_points: int = Field(5, gt=0)
    blur_kernel: int = Field(3, ge=1, le=11)
    blur_sigma: float = Field(0.5, gt=0.0)
    thickness_base: int = Field(2, gt=0)
    thickness_max: int = Field(4, gt=0)


class ModelConfig(BaseModel):
    input_shape: List[int] = [28, 28, 1]
    demo_mode: bool = True
    use_quantized_model: bool = True
    prefer_gpu: bool = True


class UIConfig(BaseModel):
    window_name: str = "Pictionary Live - Dibuja en el aire"
    window_width: int = Field(1280, gt=0)
    window_height: int = Field(960, gt=0)
    show_fps: bool = True
    show_diagnostics: bool = True
    show_top_predictions: int = Field(3, ge=1)


class LoggingConfig(BaseModel):
    log_dir: str = "./logs"
    inference_log_file: str = "./inference/inference.log"
    level_debug: str = "DEBUG"
    level_info: str = "INFO"


class DetectionConfig(BaseModel):
    hand_index_finger_id: int = 8
    hand_landmark_count: int = 21
    processing_resolution: int = Field(320, gt=0)


class PerformanceConfig(BaseModel):
    skip_frames: int = 0
    thread_workers: int = Field(1, ge=1)
    enable_profiling: bool = False
    async_processing: bool = True


class AppConfig(BaseModel):
    mediapipe: MediaPipeConfig
    camera: CameraConfig
    stroke: StrokeConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    ui: UIConfig
    logging: LoggingConfig
    detection: DetectionConfig
    performance: PerformanceConfig


class ConfigManager:
    """Gestor de configuración con carga y validación."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Buscar config.yaml en el directorio padre del directorio src
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Optional[AppConfig] = None
        self.load_config()

    def load_config(self) -> AppConfig:
        """Carga y valida la configuración desde archivo YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self.config = AppConfig(**data)
            return self.config

        except yaml.YAMLError as e:
            raise ValueError(f"Error al parsear YAML: {e}")
        except ValidationError as e:
            raise ValueError(f"Error de validación en configuración: {e}")

    def get_config(self) -> AppConfig:
        """Retorna la configuración cargada."""
        if self.config is None:
            self.load_config()
        return self.config

    def reload_config(self) -> AppConfig:
        """Recarga la configuración desde el archivo."""
        self.config = None
        return self.load_config()


# Instancia global del gestor de configuración
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Función de conveniencia para obtener la configuración."""
    return config_manager.get_config()