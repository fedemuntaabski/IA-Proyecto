"""
Air Draw Classifier - Aplicación Principal

Aplicación completa para clasificación de dibujos en el aire que integra:
- Activación de cámara del dispositivo
- Detección de movimientos de manos en tiempo real
- Interpretación de movimientos como trazos/dibujos
- Clasificación automática de figuras dibujadas

Esta aplicación reutiliza componentes avanzados del proyecto para ofrecer
una experiencia completa de dibujo en el aire con IA.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Optional

# Importar componentes existentes
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.core.hand_detector import HandDetector
from src.core.gesture_processor import GestureProcessor
from src.core.classifier import SketchClassifier
from src.core.config_manager import ConfigManager


class AirDrawClassifier:
    """
    Aplicación principal para clasificación de dibujos en el aire.

    Esta clase integra todos los componentes del proyecto en una aplicación
    completa y fácil de usar para dibujo en el aire con clasificación IA.
    """

    def __init__(self, model_path: str = "IA/sketch_classifier_model.keras",
                 model_info_path: str = "IA/model_info.json"):
        """
        Inicializa la mini-app.

        Args:
            model_path: Ruta al modelo de clasificación
            model_info_path: Ruta a la información del modelo
        """
        print("🚀 Iniciando Mini Air Draw Classifier...")
        print("=" * 50)

        # Inicializar configuración
        self.config_manager = ConfigManager()
        self.detection_config = self.config_manager.get_detection_config()
        self.ml_config = self.config_manager.get_ml_config()

        # Inicializar componentes principales
        self.hand_detector = HandDetector(min_area=5000, max_area=50000)
        self.gesture_processor = GestureProcessor(image_size=28)
        self.classifier = SketchClassifier(model_path, model_info_path, enable_fallback=True)

        # Estado de la aplicación
        self.is_drawing = False
        self.last_prediction = None
        self.drawing_start_time = None
        self.session_start_time = time.time()
        self.total_drawings = 0
        self.successful_predictions = 0

        # Configuración simplificada
        self.min_points_for_classification = 10  # Mínimo puntos para intentar clasificar
        self.classification_cooldown = 2.0  # Segundos entre clasificaciones
        self.confidence_threshold = self.ml_config.confidence_threshold

        print("✓ Componentes inicializados")
        print(f"  Detector de manos: {'Avanzado' if self.hand_detector.enable_advanced_vision else 'Básico'}")
        print(f"  Clasificador: {'Disponible' if self.classifier.is_available() else 'No disponible'}")
        print()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame de la cámara.

        Args:
            frame: Frame de OpenCV (BGR)

        Returns:
            Frame procesado con visualizaciones
        """
        # Voltear para efecto espejo
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # Detectar manos
        frame_rgb, contours, has_hands = self.hand_detector.detect(frame)

        # Procesar si hay manos
        if has_hands and contours:
            # Obtener posición del dedo índice
            index_pos = self.hand_detector.get_index_finger_tip(contours)

            if index_pos and self.hand_detector.is_drawing_gesture(contours):
                # Usuario está dibujando
                if not self.is_drawing:
                    self.is_drawing = True
                    self.drawing_start_time = time.time()
                    self.gesture_processor.clear()
                    print("✏️  Comenzando dibujo...")

                # Agregar punto al gesto
                normalized_pos = (index_pos[0] / width, index_pos[1] / height)
                self.gesture_processor.add_point(normalized_pos, (height, width))
            else:
                # Usuario dejó de dibujar
                if self.is_drawing:
                    self.is_drawing = False
                    points_count = len(self.gesture_processor.stroke_points)

                    if points_count >= self.min_points_for_classification:
                        self._classify_current_gesture()
                    else:
                        print(f"⚠ Dibujo muy corto ({points_count} puntos)")

        # Crear frame de visualización
        display_frame = frame.copy()

        # Dibujar contornos si hay manos
        if has_hands and contours:
            display_frame = self.hand_detector.draw_landmarks(display_frame, contours)

        # Dibujar trazo actual
        if len(self.gesture_processor.stroke_points) > 0:
            display_frame = self.gesture_processor.draw_on_frame(
                display_frame,
                frame_shape=(height, width)
            )

        # Interfaz simplificada
        self._draw_simple_ui(display_frame, has_hands, contours)

        return display_frame

    def _classify_current_gesture(self):
        """Clasifica el gesto actual si es posible."""
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
            predictions = self.classifier.predict(gesture_image, top_k=3)

            if predictions:
                top_class, confidence = predictions[0]
                self.last_prediction = (top_class, confidence, time.time())

                # Contar predicción exitosa si supera el umbral
                if confidence >= self.confidence_threshold:
                    self.successful_predictions += 1

                print(f"🎯 Predicción: {top_class} ({confidence:.1%})")

                if len(predictions) > 1:
                    print("  Otras opciones:")
                    for alt_class, alt_conf in predictions[1:2]:  # Solo mostrar la segunda mejor
                        print(f"    {alt_class} ({alt_conf:.1%})")
            else:
                print("⚠ No se obtuvieron predicciones")
        else:
            print("⚠ Clasificador no disponible")

        print()

    def _show_session_stats(self):
        """Muestra estadísticas de la sesión actual."""
        session_duration = time.time() - self.session_start_time
        success_rate = (self.successful_predictions / self.total_drawings * 100) if self.total_drawings > 0 else 0

        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS DE LA SESIÓN")
        print("="*50)
        print(f"⏱️  Duración: {session_duration:.1f} segundos")
        print(f"🎨 Dibujos realizados: {self.total_drawings}")
        print(f"✅ Predicciones exitosas: {self.successful_predictions}")
        print(f"📈 Tasa de éxito: {success_rate:.1f}%")
        print(f"🤖 Modo clasificador: {self.classifier.mode.upper()}")
        print("="*50)

    def _draw_simple_ui(self, frame: np.ndarray, has_hands: bool, contours: List):
        """
        Dibuja una interfaz de usuario simplificada.

        Args:
            frame: Frame donde dibujar
            has_hands: Si se detectaron manos
            contours: Contornos detectados
        """
        height, width = frame.shape[:2]

        # Barra superior con información básica
        cv2.rectangle(frame, (0, 0), (width, 60), (30, 30, 30), -1)
        cv2.rectangle(frame, (0, 0), (width, 60), (200, 200, 200), 1)

        # Estado de detección
        if has_hands and contours:
            status_text = f"👋 Mano detectada | Puntos: {len(self.gesture_processor.stroke_points)}"
            status_color = (0, 255, 0)
        else:
            status_text = "Sin detección de manos"
            status_color = (0, 0, 255)

        cv2.putText(frame, status_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Estado de dibujo
        if self.is_drawing:
            draw_status = "✏️ DIBUJANDO..."
            draw_color = (0, 255, 255)
        else:
            draw_status = "Listo para dibujar"
            draw_color = (255, 165, 0)

        cv2.putText(frame, draw_status, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

        # Mostrar última predicción si existe y no ha expirado
        if self.last_prediction:
            class_name, confidence, pred_time = self.last_prediction
            if time.time() - pred_time < 5.0:  # Mostrar por 5 segundos
                pred_text = f"🎯 {class_name} ({confidence:.0%})"
                cv2.putText(frame, pred_text, (width - 250, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Controles en la parte inferior
        controls_y = height - 40
        cv2.rectangle(frame, (0, controls_y), (width, height), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, controls_y), (width, height), (150, 150, 150), 1)

        controls_text = "CONTROLES: [ESPACIO] Nueva clasificación | [R] Limpiar | [Q] Salir"
        cv2.putText(frame, controls_text, (10, height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Ejecuta el loop principal de la mini-app."""
        print("📹 Iniciando cámara...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: No se pudo acceder a la cámara")
            return

        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("✅ Cámara inicializada")
        print("\n🎮 Controles:")
        print("  ESPACIO: Forzar nueva clasificación")
        print("  R: Limpiar dibujo actual")
        print("  Q: Salir")
        print("\n💡 Instrucciones:")
        print("  1. Muestra tu mano a la cámara")
        print("  2. Dibuja una figura en el aire con el dedo índice")
        print("  3. La app intentará adivinar qué dibujaste")
        print("\n" + "=" * 50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar frame
                display_frame = self.process_frame(frame)

                # Mostrar frame
                cv2.imshow('Air Draw Classifier - IA Proyecto', display_frame)

                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n👋 Saliendo...")
                    break
                elif key == ord('r'):
                    print("🧹 Limpiando dibujo...")
                    self.gesture_processor.clear()
                    self.last_prediction = None
                    self.is_drawing = False
                elif key == ord(' ') and len(self.gesture_processor.stroke_points) > 0:
                    # Forzar clasificación
                    print("🔄 Forzando clasificación...")
                    self._classify_current_gesture()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            self._show_session_stats()
            print("✅ Aplicación cerrada correctamente")

    def close(self):
        """Cierra la aplicación y libera recursos."""
        if hasattr(self, 'hand_detector'):
            self.hand_detector.close()


def main():
    """Punto de entrada de la mini-app."""
    try:
        app = AirDrawClassifier()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠ Interrupción detectada")
    except Exception as e:
        print(f"❌ Error en la mini-app: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()