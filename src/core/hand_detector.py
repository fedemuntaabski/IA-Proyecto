"""
Detector de manos usando OpenCV (versión simplificada).

Esta versión usa técnicas básicas de visión por computadora sin dependencias
pesadas como MediaPipe/TensorFlow.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class HandDetector:
    """
    Detecta manos usando técnicas básicas de OpenCV.
    
    Atributos:
        min_area: Área mínima para considerar una detección válida
        max_area: Área máxima para detección
        skin_lower: Límite inferior del rango de color de piel (HSV)
        skin_upper: Límite superior del rango de color de piel (HSV)
    """
    
    def __init__(self, min_area: int = 3000, max_area: int = 30000):
        """
        Inicializa el detector de manos.
        
        Args:
            min_area: Área mínima en píxeles para detección
            max_area: Área máxima en píxeles para detección
        """
        self.min_area = min_area
        self.max_area = max_area
        
        # Rangos HSV para detección de piel (ajustables)
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        
        # Para tracking
        self.prev_contours = []
        self.tracking_point = None
        
        print("✓ HandDetector (OpenCV) inicializado")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List], bool]:
        """
        Detecta manos en un frame usando segmentación de piel.
        
        Args:
            frame: Frame de OpenCV (BGR)
            
        Returns:
            Tupla con (frame_rgb, contours, manos_detectadas)
        """
        # Convertir a HSV para mejor segmentación de piel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Crear máscara para piel
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Limpiar máscara
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Operaciones morfológicas para limpiar
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                valid_contours.append(cnt)
        
        # Detectar si hay manos
        has_hands = len(valid_contours) > 0
        
        return frame, valid_contours, has_hands
    
    def draw_landmarks(self, frame: np.ndarray, contours: List) -> np.ndarray:
        """
        Dibuja los contornos detectados en el frame.
        
        Args:
            frame: Frame original
            contours: Lista de contornos
            
        Returns:
            Frame con contornos dibujados
        """
        frame_copy = frame.copy()
        
        if contours:
            # Dibujar todos los contornos
            cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 2)
            
            # Dibujar bounding boxes
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return frame_copy
    
    def get_index_finger_tip(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición aproximada del "dedo índice" (punto más alto del contorno).
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            Tupla (x, y) en píxeles o None
        """
        if not contours:
            return None
        
        # Tomar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Encontrar el punto más alto (aproximación de dedo)
        hull = cv2.convexHull(largest_contour)
        
        if len(hull) > 0:
            # El punto más alto del convex hull
            top_point = min(hull, key=lambda p: p[0][1])[0]
            return (float(top_point[0]), float(top_point[1]))
        
        return None
    
    def get_thumb_tip(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición aproximada del "pulgar" (punto más a la izquierda).
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            Tupla (x, y) en píxeles o None
        """
        if not contours:
            return None
        
        # Tomar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Encontrar el punto más a la izquierda
        hull = cv2.convexHull(largest_contour)
        
        if len(hull) > 0:
            left_point = min(hull, key=lambda p: p[0][0])[0]
            return (float(left_point[0]), float(left_point[1]))
        
        return None
    
    def is_fist(self, contours: List) -> bool:
        """
        Detecta si la mano está cerrada (puño) basado en la compacidad del contorno.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            True si detecta puño, False en caso contrario
        """
        if not contours:
            return False
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter > 0:
            # Circularidad: cuanto más circular, más cerrado el puño
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            return circularity > 0.7  # Umbral para forma circular
        
        return False
    
    def close(self):
        """Método dummy para compatibilidad con la interfaz."""
        pass


if __name__ == "__main__":
    # Test básico
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        exit()
    
    print("Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Detectar
        frame_rgb, contours, has_hands = detector.detect(frame)
        
        # Dibujar
        frame_with_contours = detector.draw_contours(frame, contours)
        
        # Mostrar info
        if has_hands:
            cv2.putText(frame_with_contours, f"Manos detectadas: {len(contours)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_with_contours, "No se detectan manos", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Hand Detection (OpenCV)', frame_with_contours)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
