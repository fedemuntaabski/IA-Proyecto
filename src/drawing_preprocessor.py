"""
drawing_preprocessor.py - RÉPLICA EXACTA del preprocesamiento de entrenamiento
"""

import cv2
import numpy as np
from typing import List, Tuple, Union


class DrawingPreprocessor:
    """Preprocesa trazos EXACTAMENTE como en strokes_to_image() del entrenamiento."""
    
    def __init__(self, target_shape: Tuple[int, int, int], config: dict = None):
        """
        Inicializa el preprocesador.
        
        Args:
            target_shape: (height, width, channels)
            config: Configuración (ignorada - usa valores exactos del entrenamiento)
        """
        self.h, self.w, self.c = target_shape
        
        # VALORES EXACTOS del entrenamiento - NO MODIFICAR
        self.intermediate_size = 256
        self.line_width = 2  # Exactamente como en entrenamiento
        self.min_points_per_stroke = 2
    
    def preprocess(self, input_data: Union[List[Tuple[float, float]], List[List[Tuple[float, float]]]]) -> np.ndarray:
        """
        Convierte trazo(s) a imagen lista para el modelo.
        
        Acepta dos formatos:
        1. Lista de puntos: [(x1,y1), (x2,y2), ...] - un solo trazo
        2. Lista de trazos: [[(x1,y1), ...], [(x1,y1), ...]] - múltiples trazos
        
        Args:
            input_data: Puntos normalizados [0, 1]
        
        Returns:
            Array numpy (h, w, c) listo para predicción
        """
        if not input_data:
            return np.zeros((self.h, self.w, self.c), dtype=np.float32)
        
        # Detectar formato
        if isinstance(input_data[0], (list, tuple)) and len(input_data[0]) == 2 and isinstance(input_data[0][0], (int, float)):
            # Formato 1: Lista simple de puntos
            strokes = [input_data]
        else:
            # Formato 2: Lista de trazos
            strokes = input_data
        
        try:
            return self._strokes_to_image_exact(strokes)
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return np.zeros((self.h, self.w, self.c), dtype=np.float32)
    
    def _strokes_to_image_exact(self, strokes: List[List[Tuple[float, float]]]) -> np.ndarray:
        """
        RÉPLICA EXACTA de strokes_to_image() del entrenamiento.
        Cada línea de código coincide con la función original.
        """
        # Create a white image (255 = blanco)
        canvas = np.ones((self.intermediate_size, self.intermediate_size), dtype=np.uint8) * 255
        
        # Draw each stroke
        for stroke in strokes:
            if len(stroke) < self.min_points_per_stroke:
                continue
            
            # Convertir puntos normalizados [0,1] a píxeles [0, 255]
            # CRÍTICO: El canvas de entrenamiento es 256x256, pero los píxeles van de 0-255
            pixel_points = []
            for x, y in stroke:
                # Escalar exactamente como en entrenamiento
                px = int(x * 255)
                py = int(y * 255)
                # Clamp a [0, 255]
                px = max(0, min(255, px))
                py = max(0, min(255, py))
                pixel_points.append((px, py))
            
            # Draw lines between consecutive points
            # fill=0 significa negro, width=2 es el grosor
            for i in range(len(pixel_points) - 1):
                cv2.line(canvas, 
                        pixel_points[i], 
                        pixel_points[i+1], 
                        0,  # Negro (fill=0 en PIL)
                        self.line_width,  # width=2
                        cv2.LINE_8)  # Sin anti-aliasing (PIL default)
        
        # Resize to target size
        # LANCZOS en PIL = INTER_LANCZOS4 en OpenCV
        resized = cv2.resize(canvas, (self.w, self.h), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = resized.astype(np.float32) / 255.0
        
        # Invert so drawing is white on black background
        img_array = 1.0 - img_array
        
        # Añadir dimensión de canal para el modelo
        if self.c == 1:
            img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
    
    def preprocess_from_app(self, drawing_strokes: List[List[Tuple[float, float]]]) -> np.ndarray:
        """
        Versión específica para llamar desde app.py con múltiples trazos.
        
        Args:
            drawing_strokes: Lista de trazos, donde cada trazo es [(x,y), ...]
        
        Returns:
            Imagen preprocesada
        """
        if not drawing_strokes:
            return np.zeros((self.h, self.w, self.c), dtype=np.float32)
        
        return self._strokes_to_image_exact(drawing_strokes)
    
    def visualize_preprocessing(self, input_data, show_stats=True) -> np.ndarray:
        """Visualización para debugging con estadísticas."""
        image = self.preprocess(input_data)
        
        # Convertir a uint8 y ampliar
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image_2d = image.squeeze()
        else:
            image_2d = image
        
        if show_stats:
            print(f"   Visualización:")
            print(f"   - Rango: [{image.min():.4f}, {image.max():.4f}]")
            print(f"   - Media: {image.mean():.4f}")
            print(f"   - Desv.Std: {image.std():.4f}")
        
        vis = (image_2d * 255).astype(np.uint8)
        vis_large = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Convertir a RGB
        vis_rgb = cv2.cvtColor(vis_large, cv2.COLOR_GRAY2RGB)
        
        # Añadir borde verde
        cv2.rectangle(vis_rgb, (0, 0), (255, 255), (0, 255, 0), 2)
        
        return vis_rgb