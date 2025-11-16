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
from .ui.ui_manager import UIManager
from .utils import (MIN_POINTS_FOR_CLASSIFICATION, DEFAULT_CONFIDENCE_THRESHOLD,
                    DEFAULT_RESOLUTION_WIDTH, DEFAULT_RESOLUTION_HEIGHT, DEFAULT_TARGET_FPS)
from .utils.async_processor import async_processor
from .utils.analytics import analytics_tracker
from .utils.feedback_manager import feedback_manager
from .utils.model_retrainer import ModelRetrainer
from .utils.user_preferences import UserPreferences
from .utils.sound_manager import sound_manager
from .utils.statistics_tracker import statistics_tracker
from .utils.performance_monitor import performance_monitor
from .utils.settings_manager import settings_manager
from .utils.error_handler import handle_error


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

        # Cargar preferencias de usuario
        self.user_preferences = UserPreferences()
        print(f"🎨 Tema: {self.user_preferences.get_theme()}")

        # Cargar configuración avanzada
        self.settings = settings_manager
        valid, errors = self.settings.validate()
        if not valid:
            print(f"⚠️  {_('Errores en configuración')}:")
            for error in errors:
                print(f"   • {error}")

        # Configurar sonidos según preferencias
        sound_manager.set_enabled(self.user_preferences.is_sound_enabled())
        sound_manager.set_volume(self.user_preferences.get_sound_volume())

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
            'resolution_height': DEFAULT_RESOLUTION_HEIGHT,
            'async_processing': True  # Habilitar procesamiento asíncrono
        }

        # Inicializar procesamiento asíncrono
        async_processor.start()
        print("✓ Procesamiento asíncrono inicializado")

        # Inicializar analíticas
        analytics_tracker.track_event('app_start', {
            'model_path': model_path,
            'model_info_path': model_info_path,
            'timestamp': time.time()
        })

        # Inicializar procesador de frames
        self.frame_processor = FrameProcessor(
            self.hand_detector,
            self.gesture_processor,
            self.classifier,
            self.app_config
        )

        # Inicializar reentrenador de modelos
        feedback_file = "src/core/utils/feedback_data.json"
        self.model_retrainer = ModelRetrainer(
            feedback_file=feedback_file,
            model_path=model_path,
            classifier=self.classifier
        )
        print("✓ Model Retraining Pipeline inicializado")

        # Inicializar UI
        self.ui_manager = UIManager()
        # Aplicar tema guardado
        self.ui_manager.switch_theme(self.user_preferences.get_theme())

        # Estado de la aplicación
        self.session_start_time = time.time()
        self.frame_processor.session_start_time = self.session_start_time

        # Performance monitoring
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # Estado de mouse para feedback
        self.last_mouse_click = None

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

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Procesa un frame completo de la aplicación.

        Args:
            frame: Frame de OpenCV

        Returns:
            Tupla de (frame_procesado, app_state)
        """
        # Actualizar FPS
        self._update_fps()

        # Registrar frame en estadísticas y monitor de rendimiento
        statistics_tracker.record_frame(self.current_fps)
        performance_monitor.update(self.current_fps)

        # Rastrear frame procesado
        analytics_tracker.track_frame_processed(self.current_fps)

        # Procesar frame
        processed_frame, app_state = self.frame_processor.process_frame(frame)

        # Actualizar estado de la aplicación
        app_state.update({
            'has_hands': len(self.frame_processor.gesture_processor.stroke_points) > 0 or
                        app_state.get('has_hands', False),
            'session_time': time.time() - self.session_start_time,
            'current_gesture_image': self.frame_processor.gesture_processor.get_gesture_image_for_feedback(),
            'fps': self.current_fps
        })

        # Actualizar UI con FPS
        self.ui_manager.update_fps(self.current_fps)

        # Dibujar UI
        final_frame = self.ui_manager.draw_ui(processed_frame, app_state)

        return final_frame, app_state

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
            sound_manager.play_sound('info')
            return True
        elif key == ord('r'):
            print(_("🧹 Limpiando dibujo..."))
            sound_manager.play_sound('info')
            self.frame_processor.clear_drawing()
        elif key == ord(' ') and len(self.frame_processor.gesture_processor.stroke_points) > 0:
            sound_manager.play_classification()
            self.frame_processor.force_classification()
        elif key == ord('c') or key == ord('C'):
            self.ui_manager.toggle_feedback_mode()
            status = _("activado") if self.ui_manager.feedback_mode else _("desactivado")
            print(f"📝 Modo feedback {status}")
        elif hasattr(self.ui_manager, 'handle_feedback_key') and self.ui_manager.handle_feedback_key(key):
            # Tecla manejada por el sistema de feedback
            pass
        elif key == ord('h'):
            self.ui_manager.toggle_help()
            status = _("mostrada") if self.ui_manager.show_help else _("oculta")
            print(f"{'👁️' if self.ui_manager.show_help else '🙈'} {_('Ayuda')} {status}")
            sound_manager.play_info()
        elif key == ord('p') or key == ord('P'):
            self.ui_manager.toggle_user_profile()
            status = _("mostrado") if self.ui_manager.show_user_profile else _("oculto")
            print(f"👤 {_('Perfil de usuario')} {status}")
        elif key == 112:  # F1
            self.ui_manager.switch_theme('default')
            self.user_preferences.set_theme('default')
            self.user_preferences.save()
            print(f"🎨 {_('Tema')}: Default")
            sound_manager.play_info()
        elif key == 113:  # F2
            self.ui_manager.switch_theme('dark')
            self.user_preferences.set_theme('dark')
            self.user_preferences.save()
            print(f"🎨 {_('Tema')}: Dark")
            sound_manager.play_info()
        elif key == 114:  # F3
            self.ui_manager.switch_theme('high_contrast')
            self.user_preferences.set_theme('high_contrast')
            self.user_preferences.save()
            print(f"🎨 {_('Tema')}: Alto Contraste")
            sound_manager.play_info()
        elif key == 115:  # F4
            self.ui_manager.switch_theme('ocean')
            self.user_preferences.set_theme('ocean')
            self.user_preferences.save()
            print(f"🎨 {_('Tema')}: Ocean")
            sound_manager.play_info()
        elif key == 116:  # F5
            self.ui_manager.switch_theme('forest')
            self.user_preferences.set_theme('forest')
            self.user_preferences.save()
            print(f"🎨 {_('Tema')}: Forest")
            sound_manager.play_info()
        elif key == 117:  # F6 - Toggle sonidos
            enabled = not self.user_preferences.is_sound_enabled()
            self.user_preferences.set_sound_enabled(enabled)
            self.user_preferences.save()
            sound_manager.set_enabled(enabled)
            status = _("habilitados") if enabled else _("deshabilitados")
            print(f"🔊 {_('Sonidos')} {status}")
        elif key == ord('i'):  # Info de rendimiento
            performance_monitor.print_status()
        elif key == ord('s'):  # Settings
            self.settings.print_settings()
        elif key == 27:  # ESC key
            if self.ui_manager.show_user_profile:
                self.ui_manager.toggle_user_profile()
                print(f"👤 {_('Perfil de usuario oculto')}")

        return False

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la sesión actual.

        Returns:
            Diccionario con estadísticas
        """
        # Mostrar estado de rendimiento
        performance_monitor.print_status()

        # Finalizar sesión en rastreador
        session_stats = statistics_tracker.end_session()
        statistics_tracker.save()

        # Imprimir resumen
        statistics_tracker.print_summary()

        # Estadísticas por clase
        class_stats = statistics_tracker.get_class_statistics()
        if class_stats:
            print(f"📊 {_('Estadísticas por clase')}:")
            for class_name, stats in class_stats.items():
                print(f"  - {class_name}: {stats['count']} dibujos ({stats['success_rate']:.1f}% éxito)")

        # Sugerencias de optimización
        optimization_suggestions = performance_monitor.get_optimization_suggestions()
        if optimization_suggestions:
            print(f"\n⚡ {_('Sugerencias de optimización')}:")
            for suggestion in optimization_suggestions:
                print(f"  • {suggestion}")

        return session_stats

    def cleanup(self) -> None:
        """Limpia recursos de la aplicación."""
        # Guardar preferencias de usuario
        self.user_preferences.save()

        # Detener procesamiento asíncrono
        async_processor.stop()

        # Guardar analíticas
        analytics_tracker.save_events()

        # Registrar fin de sesión
        session_summary = analytics_tracker.get_session_summary()
        analytics_tracker.track_event('app_end', {
            'session_summary': session_summary,
            'timestamp': time.time()
        })
        analytics_tracker.save_events()

        if hasattr(self, 'hand_detector'):
            self.hand_detector.close()
        print(_("✅ Aplicación cerrada correctamente"))