"""
drawing_preprocessor.py - Preprocesado de trazos para inferencia
"""

import cv2
import numpy as np
from typing import List, Tuple


class DrawingPreprocessor:
    """Preprocesa trazos para entrada del modelo."""
    
    def __init__(self, target_shape: Tuple[int, int, int], config: dict = None):
        """
        Inicializa el preprocesador.
        
        Args:
            target_shape: (height, width, channels)
            config: Configuración de preprocesamiento
        """
        self.h, self.w, self.c = target_shape
        self.config = config or {}
        
        # Parámetros configurables
        self.scale_factor = self.config.get("scale_factor", 0.8)
        self.min_stroke_length = self.config.get("min_stroke_length", 0.05)
        self.min_points = self.config.get("min_points", 5)
        self.blur_kernel = self.config.get("blur_kernel", 3)
        # Por defecto usar blur mínimo y trazos más delgados (coincide mejor con QuickDraw)
        self.blur_sigma = self.config.get("blur_sigma", 0.0)
        self.thickness_base = self.config.get("thickness_base", 1)
        self.thickness_max = self.config.get("thickness_max", 2)
        # Opcional: aplicar skeletonization (thinning) para normalizar grosor
        self.skeletonize = self.config.get("skeletonize", False)
    
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
        
        # Filtrar trazos de baja calidad
        if not self._is_valid_stroke(points):
            return np.ones((self.h, self.w, self.c), dtype=np.float32)
        
        # Centrar y escalar puntos
        scaled_points = self._center_and_scale_points(points)
        
        # Crear canvas blanco
        canvas = np.ones((self.h, self.w), dtype=np.uint8) * 255
        
        # Dibujar el trazo con suavizado
        self._draw_smooth_stroke(canvas, scaled_points)
        
        # No invertir colores: mantener trazo negro (0) sobre fondo blanco (255)
        
        # Normalizar a [0, 1]
        canvas = canvas.astype(np.float32) / 255.0
        
        # Expandir a 3D si es necesario
        if self.c == 1:
            canvas = np.expand_dims(canvas, axis=-1)
        
        return canvas
    
    def _is_valid_stroke(self, points: List[Tuple[float, float]]) -> bool:
        """Valida que el trazo tenga calidad suficiente."""
        if len(points) < self.min_points:
            return False
        
        # Calcular longitud total del trazo
        total_length = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            total_length += np.sqrt(dx*dx + dy*dy)
        
        # Trazo demasiado corto
        if total_length < self.min_stroke_length:
            return False
        
        # Calcular varianza para detectar ruido excesivo
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        var_x = np.var(xs)
        var_y = np.var(ys)
        
        # Si la varianza es muy baja, es un punto casi estático
        if var_x + var_y < 0.0001:
            return False
        
        return True
    
    def _center_and_scale_points(self, points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Centra y escala los puntos para ocupar mejor el canvas."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Evitar división por cero
        if width <= 0:
            width = 0.01
        if height <= 0:
            height = 0.01
        
        # Calcular escala para ocupar ~80% del canvas
        scale = self.scale_factor * min(self.w / width, self.h / height)
        
        # Centro del bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Transformar puntos: centrar y escalar
        scaled_points = []
        for x, y in points:
            new_x = (x - center_x) * scale + self.w / 2
            new_y = (y - center_y) * scale + self.h / 2
            # Asegurar que estén dentro del canvas
            new_x = max(0, min(self.w - 1, new_x))
            new_y = max(0, min(self.h - 1, new_y))
            scaled_points.append((int(new_x), int(new_y)))
        
        return scaled_points
    
    def _draw_smooth_stroke(self, canvas: np.ndarray, points: List[Tuple[int, int]]):
        """Dibuja el trazo con suavizado para reducir ruido."""
        if len(points) < 2:
            return
        
        # Dibujar líneas con grosor variable basado en velocidad (simulado)
        for i in range(1, len(points)):
            # Calcular distancia para simular grosor variable
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Grosor base 2, pero ajustado por distancia (simula velocidad)
            thickness = max(1, min(self.thickness_max, int(self.thickness_base + distance * 10)))
            
            cv2.line(canvas, points[i-1], points[i], 0, thickness)
        
        # Aplicar blur suave solo si se configuró kernel>1 y sigma>0
        if self.blur_kernel and self.blur_kernel > 1 and self.blur_sigma and self.blur_sigma > 0:
            canvas[:] = cv2.GaussianBlur(canvas, (self.blur_kernel, self.blur_kernel), self.blur_sigma)

        # Aplicar skeletonization si está activado
        if self.skeletonize:
            # Binarizar (0/255)
            _, bw = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
            # Convertir a uint8
            bw = bw.astype('uint8')
            # Skeletonization via algoritmo iterativo de thinning (morphology)
            skel = np.zeros(bw.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp = np.zeros(bw.shape, np.uint8)
            done = False
            img = bw.copy()
            while not done:
                eroded = cv2.erode(img, element)
                opened = cv2.dilate(eroded, element)
                temp = cv2.subtract(img, opened)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()
                if cv2.countNonZero(img) == 0:
                    done = True

            # skel tiene 255 para esqueleto
            canvas = skel
