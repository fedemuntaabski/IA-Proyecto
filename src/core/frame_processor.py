"""
Frame Processor - Procesador de Frames para Air Draw Classifier.

Este módulo maneja el procesamiento de frames de la cámara,
separando la lógica de visión computacional del controlador principal.
"""

import cv2
import numpy as np
import time
from typing import List, Optional, Tuple, Dict, Any
from .detection import HandDetector
from .classification import GestureProcessor, SketchClassifier
from .i18n import get_class_name_translation
from .utils import MIN_POINTS_FOR_CLASSIFICATION
from .utils.async_processor import ml_async_processor
from .utils.analytics import analytics_tracker
from .utils.sensitivity_manager import sensitivity_manager


class FrameProcessor:
    """
    Procesador de frames que maneja toda la lógica de visión computacional.

    Responsabilidades:
    - Procesar frames de la cámara
    - Detectar manos y gestos
    - Gestionar el estado del dibujo
    - Coordinar la clasificación
    """

    def __init__(self, hand_detector: HandDetector, gesture_processor: GestureProcessor,
                 classifier: SketchClassifier, config: dict):
        """
        Inicializa el procesador de frames.

        Args:
            hand_detector: Detector de manos
            gesture_processor: Procesador de gestos
            classifier: Clasificador de sketches
            config: Configuración de la aplicación
        """
        self.hand_detector = hand_detector
        self.gesture_processor = gesture_processor
        self.classifier = classifier

        # Configuración
        self.min_points_for_classification = config.get('min_points_for_classification', MIN_POINTS_FOR_CLASSIFICATION)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)

        # Estado del procesamiento
        self.is_drawing = False
        self.drawing_start_time = None
        self.last_prediction = None

        # Estado asíncrono
        self.pending_predictions = {}  # task_id -> (timestamp, gesture_image)
        self.async_enabled = config.get('async_processing', True)

        # Estadísticas
        self.total_drawings = 0
        self.successful_predictions = 0
        self.async_predictions = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procesa un frame completo con sensibilidad adaptativa.

        Args:
            frame: Frame de OpenCV (BGR)

        Returns:
            Tupla de (frame_procesado, estado_de_aplicacion)
        """
        # Voltear para efecto espejo
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # Analizar calidad del frame para sensibilidad adaptativa
        frame_quality = sensitivity_manager.analyze_frame_quality(frame)
        
        # Calcular sensibilidad actual y actualizar umbrales
        current_sensitivity = sensitivity_manager.calculate_current_sensitivity()
        sensitivity_manager.update_thresholds(current_sensitivity)

        # Verificar predicciones asíncronas completadas
        self._check_pending_predictions()

        # Detectar manos
        frame_rgb, contours, has_hands = self.hand_detector.detect(frame)

        # Medir ruido si hay máscara de detección
        if has_hands and contours:
            # Crear máscara simple para análisis de ruido
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, contours, -1, 255, -1)
            noise_level = sensitivity_manager.measure_noise_level(frame, mask)

        # Procesar gestos si hay manos
        if has_hands and contours:
            self._process_gesture(contours, (height, width))

        # Crear frame de visualización
        display_frame = frame.copy()

        # Dibujar landmarks si hay manos
        if has_hands and contours:
            display_frame = self.hand_detector.draw_landmarks(display_frame, contours)

        # Dibujar trazo actual
        if len(self.gesture_processor.stroke_points) > 0:
            display_frame = self.gesture_processor.draw_on_frame(
                display_frame, frame_shape=(height, width)
            )

        # Preparar estado de la aplicación
        app_state = self._get_app_state(current_sensitivity, frame_quality)

        return display_frame, app_state

    def _process_gesture(self, contours: List, frame_shape: Tuple[int, int]) -> None:
        """
        Procesa el gesto actual basado en los contornos detectados.

        Args:
            contours: Contornos detectados
            frame_shape: Forma del frame (height, width)
        """
        # Obtener posición del dedo índice
        index_pos = self.hand_detector.get_index_finger_tip(contours)

        if index_pos and self.hand_detector.is_drawing_gesture(contours):
            # Usuario está dibujando
            if not self.is_drawing:
                self._start_drawing()

            # Agregar punto al gesto
            normalized_pos = (index_pos[0] / frame_shape[1], index_pos[1] / frame_shape[0])
            self.gesture_processor.add_point(normalized_pos, frame_shape)
        else:
            # Usuario dejó de dibujar
            if self.is_drawing:
                self._stop_drawing()

    def _start_drawing(self) -> None:
        """Inicia una nueva sesión de dibujo."""
        self.is_drawing = True
        self.drawing_start_time = time.time()
        self.gesture_processor.clear()
        print("✏️  Comenzando dibujo...")

    def _stop_drawing(self) -> None:
        """Finaliza la sesión de dibujo actual."""
        self.is_drawing = False
        points_count = len(self.gesture_processor.stroke_points)

        if points_count >= self.min_points_for_classification:
            self._classify_current_gesture()
        else:
            print(f"⚠ Dibujo muy corto ({points_count} puntos)")

    def _classify_current_gesture(self) -> None:
        """Clasifica el gesto actual."""
        if len(self.gesture_processor.stroke_points) < self.min_points_for_classification:
            return

        print("🔍 Clasificando dibujo...")
        self.total_drawings += 1

        # Obtener imagen del gesto
        gesture_image = self.gesture_processor.get_gesture_image()

        if gesture_image is None:
            print("⚠ No se pudo procesar el gesto")
            return

        # Realizar clasificación
        if self.classifier.is_available():
            if self.async_enabled and hasattr(self.classifier, 'predict_async'):
                # Clasificación asíncrona
                task_id = self.classifier.predict_async(gesture_image, top_k=3)
                self.pending_predictions[task_id] = (time.time(), gesture_image)
                self.async_predictions += 1
                print(f"📤 Predicción asíncrona enviada (ID: {task_id})")
            else:
                # Clasificación síncrona
                predictions = self.classifier.predict(gesture_image, top_k=3)
                self._process_prediction_results(predictions)
        else:
            print("⚠ Clasificador no disponible")

        print()

    def _check_pending_predictions(self) -> None:
        """
        Verifica si hay predicciones asíncronas completadas y las procesa.
        """
        completed_tasks = []

        for task_id, (timestamp, gesture_image) in self.pending_predictions.items():
            if self.classifier.is_prediction_ready(task_id):
                predictions = self.classifier.get_prediction_result(task_id)
                if predictions:
                    self._process_prediction_results(predictions)
                    print(f"✅ Predicción asíncrona completada (ID: {task_id})")
                else:
                    print(f"⚠ Error en predicción asíncrona (ID: {task_id})")
                completed_tasks.append(task_id)

        # Limpiar tareas completadas
        for task_id in completed_tasks:
            del self.pending_predictions[task_id]

    def _process_prediction_results(self, predictions: List[Tuple[str, float]]) -> None:
        """
        Procesa los resultados de una predicción.

        Args:
            predictions: Lista de predicciones (clase, confianza)
        """
        if not predictions:
            analytics_tracker.track_error('prediction', 'No predictions returned')
            print("⚠ No se obtuvieron predicciones")
            return

        top_class, confidence = predictions[0]
        translated_class = get_class_name_translation(top_class)

        # Rastrear predicción
        success = confidence >= self.confidence_threshold
        analytics_tracker.track_prediction(success, confidence, top_class)

        # Contar predicción exitosa
        if confidence >= self.confidence_threshold:
            self.successful_predictions += 1

        self.last_prediction = (translated_class, confidence, time.time())

        print(f"🎯 Predicción: {translated_class} ({confidence:.1%})")

        if len(predictions) > 1:
            print("  Otras opciones:")
            for alt_class, alt_conf in predictions[1:2]:
                alt_translated = get_class_name_translation(alt_class)
                print(f"    {alt_translated} ({alt_conf:.1%})")

    def _get_app_state(self, current_sensitivity: float = None, frame_quality: float = None) -> Dict[str, Any]:
        """
        Obtiene el estado actual de la aplicación.

        Returns:
            Diccionario con el estado de la aplicación
        """
        return {
            'has_hands': False,  # Se actualizará en el loop principal
            'is_drawing': self.is_drawing,
            'stroke_points': self.gesture_processor.stroke_points,
            'last_prediction': self.last_prediction,
            'min_points_for_classification': self.min_points_for_classification,
            'total_drawings': self.total_drawings,
            'successful_predictions': self.successful_predictions,
            'async_predictions': self.async_predictions,
            'pending_predictions_count': len(self.pending_predictions),
            'async_enabled': self.async_enabled,
            'session_time': time.time() - getattr(self, 'session_start_time', time.time()),
            'current_sensitivity': current_sensitivity,
            'frame_quality': frame_quality
        }

    def clear_drawing(self) -> None:
        """Limpia el dibujo actual."""
        self.gesture_processor.clear()
        self.last_prediction = None
        self.is_drawing = False

    def force_classification(self) -> None:
        """Fuerza la clasificación del dibujo actual."""
        if len(self.gesture_processor.stroke_points) > 0:
            print("🔄 Forzando clasificación...")
            self._classify_current_gesture()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la sesión.

        Returns:
            Diccionario con estadísticas
        """
        session_duration = time.time() - getattr(self, 'session_start_time', time.time())
        success_rate = (self.successful_predictions / self.total_drawings * 100) if self.total_drawings > 0 else 0
        async_rate = (self.async_predictions / self.total_drawings * 100) if self.total_drawings > 0 else 0

        return {
            'session_duration': session_duration,
            'total_drawings': self.total_drawings,
            'successful_predictions': self.successful_predictions,
            'success_rate': success_rate,
            'async_predictions': self.async_predictions,
            'async_rate': async_rate,
            'pending_predictions': len(self.pending_predictions)
        }