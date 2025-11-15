"""
Detector de manos usando MediaPipe.

Este módulo detecta las posiciones de las manos y dedos en tiempo real
usando el framework MediaPipe de Google.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class HandDetector:
    """
    Detecta manos y dedos usando MediaPipe Hands.
    
    Atributos:
        hands: Objeto de detección de manos de MediaPipe
        mp_drawing: Utilidades para dibujar landmarks
        min_detection_confidence: Confianza mínima para detectar mano
        min_tracking_confidence: Confianza mínima para seguimiento
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        """
        Inicializa el detector de manos.
        
        Args:
            min_detection_confidence: Confianza mínima para detección (0-1)
            min_tracking_confidence: Confianza mínima para seguimiento (0-1)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList], bool]:
        """
        Detecta manos en un frame de video.
        
        Args:
            frame: Frame de OpenCV (BGR)
            
        Returns:
            Tupla con (frame_rgb, landmarks, manos_detectadas)
        """
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar frame
        results = self.hands.process(frame_rgb)
        
        # Detectar si hay manos
        has_hands = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
        
        return frame_rgb, results.multi_hand_landmarks, has_hands
    
    def draw_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """
        Dibuja los landmarks de las manos en el frame.
        
        Args:
            frame: Frame en RGB
            landmarks: Landmarks de MediaPipe
            
        Returns:
            Frame con landmarks dibujados
        """
        if landmarks is None:
            return frame
        
        frame_copy = frame.copy()
        for hand_landmarks in landmarks:
            self.mp_drawing.draw_landmarks(
                frame_copy,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.drawing_spec,
                self.drawing_spec
            )
        
        return frame_copy
    
    def get_index_finger_tip(self, landmarks) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición del dedo índice (landmark 8).
        
        Args:
            landmarks: Lista de landmarks de una mano
            
        Returns:
            Tupla (x, y) normalizada (0-1) o None
        """
        if landmarks is None or len(landmarks.landmark) < 9:
            return None
        
        # Landmark 8 es la punta del dedo índice
        tip = landmarks.landmark[8]
        return (tip.x, tip.y)
    
    def get_thumb_tip(self, landmarks) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición del pulgar (landmark 4).
        
        Args:
            landmarks: Lista de landmarks de una mano
            
        Returns:
            Tupla (x, y) normalizada (0-1) o None
        """
        if landmarks is None or len(landmarks.landmark) < 5:
            return None
        
        # Landmark 4 es la punta del pulgar
        tip = landmarks.landmark[4]
        return (tip.x, tip.y)
    
    def is_fist(self, landmarks) -> bool:
        """
        Detecta si la mano está en puño.
        
        Args:
            landmarks: Lista de landmarks de una mano
            
        Returns:
            True si detecta puño, False en caso contrario
        """
        if landmarks is None or len(landmarks.landmark) < 21:
            return False
        
        # Calcular distancia entre dedos y palma
        palm_center = landmarks.landmark[9]  # Centro de la palma
        
        # Verificar si todos los dedos están cerca de la palma
        distances = []
        for i in [8, 12, 16, 20]:  # Puntas de dedos
            tip = landmarks.landmark[i]
            dist = np.sqrt((tip.x - palm_center.x)**2 + (tip.y - palm_center.y)**2)
            distances.append(dist)
        
        # Si el promedio de distancias es pequeño, es un puño
        return np.mean(distances) < 0.15
    
    def close(self):
        """Cierra el detector de manos."""
        self.hands.close()


if __name__ == "__main__":
    # Test básico
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb, landmarks, has_hands = detector.detect(frame)
        
        if has_hands:
            frame_rgb = detector.draw_landmarks(frame_rgb, landmarks)
        
        # Convertir de vuelta a BGR para display
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Detection', frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
