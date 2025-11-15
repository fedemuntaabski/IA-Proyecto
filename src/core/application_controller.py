"""
Application Controller - Controlador Principal de la Aplicación.

Este módulo coordina todos los componentes de la aplicación,
manteniendo la lógica de negocio separada de la UI y el procesamiento.
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path

# Importar componentes modulares
from .detection import HandDetector
from .classification import GestureProcessor, SketchClassifier
from .config import ConfigManager
from .frame_processor import FrameProcessor
from .i18n import i18n, _
from .ui import UIManager
from .utils import (MIN_POINTS_FOR_CLASSIFICATION, DEFAULT_CONFIDENCE_THRESHOLD,
                    DEFAULT_RESOLUTION_WIDTH, DEFAULT_RESOLUTION_HEIGHT, DEFAULT_TARGET_FPS)


class ApplicationController:
    """
    Controlador principal que coordina todos los componentes de la aplicación.

    Responsabilidades:
    - Inicializar y configurar componentes
    - Gestionar el ciclo principal de la aplicación
    - Coordinar entre UI, procesamiento y lógica de negocio
    - Manejar eventos del usuario
    """

    def __init__(self, model_path: str = "IA/sketch_classifier_model.keras",
                 model_info_path: str = "IA/model_info.json"):
        """
        Inicializa el controlador de la aplicación.

        Args:
            model_path: Ruta al modelo de clasificación
            model_info_path: Ruta a la información del modelo
        """
        print("🚀 Iniciando Mini Air Draw Classifier...")
        print("=" * 50)

        # Inicializar internacionalización
        i18n.auto_detect_and_load()
        print(f"🌐 Idioma: {i18n.get_current_language().upper()}")

        # Configurar GPU
        self._setup_gpu_acceleration()

        # Inicializar configuración
        self.config_manager = ConfigManager()
        self.detection_config = self.config_manager.get_detection_config()
        self.ml_config = self.config_manager.get_ml_config()

        # Inicializar componentes
        self.hand_detector = HandDetector(min_area=5000, max_area=50000)
        self.gesture_processor = GestureProcessor(image_size=28)
        self.classifier = SketchClassifier(model_path, model_info_path, enable_fallback=True)

        # Configuración de la aplicación
        self.app_config = {
            'min_points_for_classification': MIN_POINTS_FOR_CLASSIFICATION,
            'confidence_threshold': self.ml_config.confidence_threshold,
            'target_fps': DEFAULT_TARGET_FPS,
            'resolution_width': DEFAULT_RESOLUTION_WIDTH,
            'resolution_height': DEFAULT_RESOLUTION_HEIGHT
        }

        # Inicializar procesador de frames
        self.frame_processor = FrameProcessor(
            self.hand_detector,
            self.gesture_processor,
            self.classifier,
            self.app_config
        )

        # Inicializar UI
        self.ui_manager = UIManager()

        # Estado de la aplicación
        self.session_start_time = time.time()
        self.frame_processor.session_start_time = self.session_start_time

        # Performance monitoring
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        print("✓ Componentes inicializados")
        print(f"  Detector de manos: {'Avanzado' if self.hand_detector.enable_advanced_vision else 'Básico'}")
        print(f"  Clasificador: {'Disponible' if self.classifier.is_available() else 'No disponible'}")
        print()

    def _setup_gpu_acceleration(self) -> None:
        """Configura aceleración GPU si está disponible."""
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"🎮 GPU detectada: {len(gpus)} GPU(s) disponible(s)")

                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                tf.config.set_visible_devices(gpus[0], 'GPU')
                print("✅ Aceleración GPU configurada")
            else:
                print("⚠ No se detectaron GPUs - usando CPU")
                tf.config.threading.set_intra_op_parallelism_threads(4)
                tf.config.threading.set_inter_op_parallelism_threads(4)

        except ImportError:
            print("⚠ TensorFlow no disponible - modo CPU")
        except Exception as e:
            print(f"⚠ Error configurando GPU: {e} - usando configuración por defecto")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame completo de la aplicación.

        Args:
            frame: Frame de OpenCV

        Returns:
            Frame procesado con UI
        """
        # Actualizar FPS
        self._update_fps()

        # Procesar frame
        processed_frame, app_state = self.frame_processor.process_frame(frame)

        # Actualizar estado de la aplicación
        app_state.update({
            'has_hands': len(self.frame_processor.gesture_processor.stroke_points) > 0 or
                        app_state.get('has_hands', False),
            'session_time': time.time() - self.session_start_time
        })

        # Actualizar UI con FPS
        self.ui_manager.update_fps(self.current_fps)

        # Dibujar UI
        final_frame = self.ui_manager.draw_ui(processed_frame, app_state)

        return final_frame

    def _update_fps(self) -> None:
        """Actualiza el cálculo de FPS."""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = current_time

    def handle_key_press(self, key: int) -> bool:
        """
        Maneja la presión de teclas.

        Args:
            key: Código de tecla de OpenCV

        Returns:
            True si debe salir, False para continuar
        """
        if key == ord('q'):
            print(f"\n👋 {_('Saliendo...')}")
            return True
        elif key == ord('r'):
            print(_("🧹 Limpiando dibujo..."))
            self.frame_processor.clear_drawing()
        elif key == ord(' ') and len(self.frame_processor.gesture_processor.stroke_points) > 0:
            self.frame_processor.force_classification()
        elif key == ord('h'):
            self.ui_manager.toggle_help()
            status = _("mostrada") if self.ui_manager.show_help else _("oculta")
            print(f"{'👁️' if self.ui_manager.show_help else '🙈'} {_('Ayuda')} {status}")

        return False

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la sesión actual.

        Returns:
            Diccionario con estadísticas
        """
        stats = self.frame_processor.get_statistics()
        success_rate = stats['success_rate']

        print("\n" + "="*50)
        print(_("ESTADÍSTICAS DE LA SESIÓN"))
        print("="*50)
        print(f"⏱️  {_('Duración')}: {stats['session_duration']:.1f} segundos")
        print(f"🎨 {_('Dibujos realizados')}: {stats['total_drawings']}")
        print(f"✅ {_('Predicciones exitosas')}: {stats['successful_predictions']}")
        print(f"📈 {_('Tasa de éxito')}: {success_rate:.1f}%")
        print(f"🤖 {_('Modo clasificador')}: {self.classifier.mode.upper()}")
        print("="*50)

        return stats

    def cleanup(self) -> None:
        """Limpia recursos de la aplicación."""
        if hasattr(self, 'hand_detector'):
            self.hand_detector.close()
        print(_("✅ Aplicación cerrada correctamente"))