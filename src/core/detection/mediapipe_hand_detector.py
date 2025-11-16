"""
Detector de manos usando MediaPipe Hands.

Proporciona detección de manos más precisa y robusta usando MediaPipe,
mientras mantiene compatibilidad con la interfaz del HandDetector actual.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MediaPipeHandDetector:
    """
    Detector de manos usando MediaPipe Hands.

    Proporciona detección más precisa y robusta que el detector basado en OpenCV,
    con soporte para múltiples manos y landmarks detallados.
    """

    def __init__(self, min_area: int = 3000, max_area: int = 30000,
                 max_hands: int = 2, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Inicializa el detector MediaPipe.

        Args:
            min_area: Área mínima para considerar detección válida (compatibilidad)
            max_area: Área máxima para detección (compatibilidad)
            max_hands: Número máximo de manos a detectar
            min_detection_confidence: Confianza mínima para detección inicial
            min_tracking_confidence: Confianza mínima para tracking
        """
        self.min_area = min_area
        self.max_area = max_area
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Estado para compatibilidad con interfaz anterior
        self.prev_contours = []
        self.contour_history = []
        self.tracking_point = None
        self.stability_threshold = 3
        self.max_history_size = 10

        # Inicializar MediaPipe
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None
        self.available = False

        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                max_num_hands=max_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.available = True
            print("✓ MediaPipeHandDetector inicializado correctamente")
            print(f"  Max hands: {max_hands}")
            print(f"  Detection confidence: {min_detection_confidence}")
            print(f"  Tracking confidence: {min_tracking_confidence}")

        except ImportError as e:
            print(f"⚠ MediaPipe no disponible: {e}")
            print("  El detector funcionará en modo fallback")
            self.available = False

        except Exception as e:
            print(f"⚠ Error inicializando MediaPipeHandDetector: {e}")
            print("  El detector funcionará en modo fallback")
            self.available = False

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List], bool]:
        """
        Detecta manos en un frame usando MediaPipe.

        Args:
            frame: Frame de OpenCV (BGR)

        Returns:
            Tupla con (frame_rgb, contours, manos_detectadas)
            - contours: Lista de contornos aproximados para compatibilidad
        """
        if not self.available or self.hands is None:
            # Fallback: devolver frame original sin detecciones
            return frame, [], False

        try:
            # Convertir BGR a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar con MediaPipe
            results = self.hands.process(frame_rgb)

            # Convertir resultados a contornos para compatibilidad
            contours = []
            has_hands = False

            if results.multi_hand_landmarks:
                has_hands = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Convertir landmarks a contorno aproximado
                    contour = self._landmarks_to_contour(hand_landmarks, frame.shape)
                    if contour is not None:
                        contours.append(contour)

            return frame_rgb, contours, has_hands

        except Exception as e:
            logger.error(f"Error en detección MediaPipe: {e}")
            return frame, [], False

    def _landmarks_to_contour(self, hand_landmarks, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Convierte landmarks de MediaPipe a un contorno aproximado.

        Args:
            hand_landmarks: Landmarks de MediaPipe
            frame_shape: Forma del frame (height, width)

        Returns:
            Contorno aproximado como array de puntos
        """
        try:
            height, width = frame_shape[:2]
            points = []

            # Extraer puntos de los landmarks
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append([x, y])

            if len(points) < 3:
                return None

            # Crear contorno como array numpy
            contour = np.array(points, dtype=np.int32)

            # Calcular área aproximada para filtrado
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                return None

            return contour

        except Exception as e:
            logger.error(f"Error convirtiendo landmarks a contorno: {e}")
            return None

    def draw_landmarks(self, frame: np.ndarray, contours: List, vision_result=None) -> np.ndarray:
        """
        Dibuja landmarks de MediaPipe en el frame.

        Args:
            frame: Frame original
            contours: Lista de contornos (ignorado, usa landmarks internos)
            vision_result: Resultados adicionales (opcional)

        Returns:
            Frame con landmarks dibujados
        """
        if not self.available or self.mp_drawing is None:
            # Fallback: dibujar contornos básicos si existen
            frame_copy = frame.copy()
            if contours:
                cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 2)
            return frame_copy

        # Nota: Para dibujar landmarks de MediaPipe necesitaríamos
        # tener acceso a los resultados de process(). Por ahora,
        # dibujamos contornos aproximados
        frame_copy = frame.copy()

        if contours:
            # Dibujar contornos aproximados
            cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 2)

            # Dibujar bounding boxes
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Etiqueta
                cv2.putText(frame_copy, f"Mano MediaPipe {i+1}",
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Indicador de gesto de dibujo
                if self.is_drawing_gesture([cnt]):
                    cv2.putText(frame_copy, "GESTO DIBUJO", (x, y + h + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Dibujar punto de dedo índice
                    finger_tip = self.get_index_finger_tip([cnt])
                    if finger_tip:
                        cv2.circle(frame_copy, (int(finger_tip[0]), int(finger_tip[1])),
                                 8, (0, 255, 255), -1)
                        cv2.circle(frame_copy, (int(finger_tip[0]), int(finger_tip[1])),
                                 12, (0, 255, 255), 2)

        # Indicador de MediaPipe
        cv2.putText(frame_copy, "MEDIAPIPE HANDS", (frame.shape[1] - 180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        return frame_copy

    def get_index_finger_tip(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición del dedo índice usando landmarks de MediaPipe.

        Args:
            contours: Lista de contornos (usado para compatibilidad)

        Returns:
            Tupla (x, y) del dedo índice o None
        """
        # Nota: Esta implementación simplificada usa el punto más alto del contorno
        # Una implementación completa usaría los landmarks reales de MediaPipe
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        # Encontrar punto más alto (aproximación del dedo índice)
        if len(largest_contour) > 0:
            top_point = min(largest_contour, key=lambda p: p[0][1])[0]
            return (float(top_point[0]), float(top_point[1]))

        return None

    def is_drawing_gesture(self, contours: List) -> bool:
        """
        Determina si hay gesto de dibujo usando análisis de contornos.

        Args:
            contours: Lista de contornos detectados

        Returns:
            True si detecta gesto de dibujo
        """
        if not contours:
            return False

        # Análisis simplificado basado en forma del contorno
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        if perimeter > 0:
            # Calcular circularidad
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # Gesto de dibujo: forma extendida pero no circular
            return 0.3 < circularity < 0.7

        return False

    def get_thumb_tip(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición aproximada del pulgar.

        Args:
            contours: Lista de contornos

        Returns:
            Tupla (x, y) del pulgar o None
        """
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        # Punto más a la izquierda (aproximación del pulgar)
        if len(largest_contour) > 0:
            left_point = min(largest_contour, key=lambda p: p[0][0])[0]
            return (float(left_point[0]), float(left_point[1]))

        return None

    def is_fist(self, contours: List) -> bool:
        """
        Detecta si la mano está cerrada (puño).

        Args:
            contours: Lista de contornos

        Returns:
            True si detecta puño
        """
        if not contours:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # Puño cerrado: forma más circular
            return circularity > 0.7

        return False

    def close(self):
        """Cierra el detector y libera recursos."""
        if self.hands:
            self.hands.close()
            print("✓ MediaPipeHandDetector cerrado correctamente")

        # Limpiar historiales
        self.prev_contours = []
        self.contour_history = []
        self.tracking_point = None

    def is_available(self) -> bool:
        """Verifica si MediaPipe está disponible."""
        return self.available

    def get_detector_info(self) -> Dict[str, Any]:
        """
        Obtiene información del detector.

        Returns:
            Diccionario con información del detector
        """
        return {
            'type': 'MediaPipe Hands',
            'available': self.available,
            'max_hands': self.max_hands,
            'min_detection_confidence': self.min_detection_confidence,
            'min_tracking_confidence': self.min_tracking_confidence,
            'min_area': self.min_area,
            'max_area': self.max_area
        }