"""
drawing_preprocessor.py - Preprocesado de trazos para inferencia
"""

import cv2
import numpy as np
from typing import List, Tuple


class DrawingPreprocessor:
    """Preprocesa trazos para entrada del modelo."""
    
    def __init__(self, target_shape: Tuple[int, int, int]):
        """
        Inicializa el preprocesador.
        
        Args:
            target_shape: (height, width, channels)
        """
        self.h, self.w, self.c = target_shape
    
    def preprocess(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Convierte puntos del trazo a imagen preprocesada.
        
        Args:
            points: Lista de (x, y) normalizadas 0-1
        
        Returns:
            Array de forma target_shape listo para predicción
        """
        if not points or len(points) < 2:
            # Retornar imagen vacía (blanco)
            return np.ones((self.h, self.w, self.c), dtype=np.float32)
        
        # Crear canvas blanco
        canvas = np.ones((self.h, self.w), dtype=np.uint8) * 255
        
        # Convertir puntos a píxeles
        pixel_points = self._normalize_to_pixels(points)
        
        # Dibujar el trazo
        self._draw_stroke(canvas, pixel_points)
        
        # Invertir (fondo blanco -> negro, trazo blanco)
        canvas = 255 - canvas
        
        # Normalizar a [0, 1]
        canvas = canvas.astype(np.float32) / 255.0
        
        # Expandir a 3D si es necesario
        if self.c == 1:
            canvas = np.expand_dims(canvas, axis=-1)
        
        return canvas
    
    def _normalize_to_pixels(self, points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Convierte coordenadas normalizadas a píxeles."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        max_x = max(xs) if xs else 0
        max_y = max(ys) if ys else 0
        
        # Si están normalizados (0-1), convertir a píxeles
        if max_x <= 1.0 and max_y <= 1.0:
            pixel_points = [
                (int(x * self.w), int(y * self.h))
                for x, y in points
            ]
        else:
            # Asumidos en píxeles
            pixel_points = [(int(x), int(y)) for x, y in points]
        
        return pixel_points
    
    def _draw_stroke(self, canvas: np.ndarray, points: List[Tuple[int, int]]):
        """Dibuja el trazo en el canvas."""
        for i in range(1, len(points)):
            cv2.line(canvas, points[i - 1], points[i], 0, 2)  # Negro, grosor 2
