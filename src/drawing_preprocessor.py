"""
drawing_preprocessor.py - Preprocesamiento EXACTO del entrenamiento
"""

import cv2
import numpy as np
from typing import List, Tuple, Union


class DrawingPreprocessor:
    """
    Preprocesa trazos EXACTAMENTE como en el entrenamiento.
    CRÍTICO: Invierte colores al final (blanco sobre negro).
    """
    
    def __init__(self, target_shape: Tuple[int, int, int], config: dict = None):
        """
        Inicializa el preprocesador.
        
        Args:
            target_shape: (height, width, channels) - típicamente (28, 28, 1)
            config: Configuración opcional
        """
        self.h, self.w, self.c = target_shape
        
        # Configuración optimizada pero fiel al entrenamiento
        config = config or {}
        self.intermediate_size = 256  # Exacto como entrenamiento
        self.line_width = config.get('line_width', 8)  # 8px como el canvas de dibujo
        self.min_points_per_stroke = 2
        self.padding_percent = config.get('padding_percent', 0.12)
        self.use_antialiasing = config.get('use_antialiasing', True)
    
    def preprocess(self, input_data: Union[List[Tuple[float, float]], List[List[Tuple[float, float]]]]) -> np.ndarray:
        """
        Convierte trazo(s) a imagen lista para el modelo.
        
        Acepta dos formatos:
        1. Lista de puntos: [(x1,y1), (x2,y2), ...] - un solo trazo
        2. Lista de trazos: [[(x1,y1), ...], [(x1,y1), ...]] - múltiples trazos
        
        Args:
            input_data: Puntos normalizados [0, 1]
        
        Returns:
            Array numpy (h, w, c) normalizado [0, 1] con dibujo BLANCO sobre NEGRO
        """
        if not input_data:
            return np.zeros((self.h, self.w, self.c), dtype=np.float32)
        
        # Detectar y normalizar formato
        if isinstance(input_data[0], (list, tuple)) and len(input_data[0]) == 2 and isinstance(input_data[0][0], (int, float)):
            strokes = [input_data]  # Un solo trazo
        else:
            strokes = input_data  # Múltiples trazos
        
        try:
            return self._strokes_to_image(strokes)
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return np.zeros((self.h, self.w, self.c), dtype=np.float32)
    
    def _strokes_to_image(self, strokes: List[List[Tuple[float, float]]]) -> np.ndarray:
        """
        Convierte trazos a imagen replicando EXACTAMENTE el entrenamiento.
        
        Pipeline:
        1. Dibujar en canvas blanco 256x256 (negro = líneas)
        2. Calcular bounding box y centrar con padding
        3. Redimensionar a 28x28 con LANCZOS
        4. Normalizar a [0, 1]
        5. ⭐ INVERTIR (1.0 - img) para blanco sobre negro
        """
        # 1. Crear canvas BLANCO (255 = fondo, como en entrenamiento)
        canvas = np.ones((self.intermediate_size, self.intermediate_size), dtype=np.uint8) * 255
        
        # 2. Dibujar todos los trazos en NEGRO (0)
        all_points = []
        for stroke in strokes:
            if len(stroke) < self.min_points_per_stroke:
                continue
            
            # Convertir puntos normalizados [0,1] a píxeles [0, 255]
            pixel_points = []
            for x, y in stroke:
                # Escalar a [0, 255] como en entrenamiento
                px = int(np.clip(x * 255, 0, 255))
                py = int(np.clip(y * 255, 0, 255))
                pixel_points.append((px, py))
                all_points.append((px, py))
            
            # Dibujar líneas del trazo
            for i in range(1, len(pixel_points)):
                pt1 = pixel_points[i - 1]
                pt2 = pixel_points[i]
                
                if self.use_antialiasing:
                    cv2.line(canvas, pt1, pt2, 0, self.line_width, cv2.LINE_AA)
                else:
                    cv2.line(canvas, pt1, pt2, 0, self.line_width)
        
        if not all_points:
            # No hay trazos válidos - retornar fondo negro (post-inversión)
            result = np.zeros((self.h, self.w, self.c), dtype=np.float32)
            return result
        
        # 3. Calcular bounding box del contenido
        binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)[1]
        coords = cv2.findNonZero(binary)
        
        if coords is None:
            result = np.zeros((self.h, self.w, self.c), dtype=np.float32)
            return result
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # 4. Agregar padding
        max_dim = max(w, h)
        pad = int(max_dim * self.padding_percent)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(256 - x, w + 2 * pad)
        h = min(256 - y, h + 2 * pad)
        
        # 5. Recortar región de interés
        roi = canvas[y:y+h, x:x+w]
        
        # 6. Centrar en canvas cuadrado (mantener proporciones)
        max_dim = max(w, h)
        square_canvas = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
        
        offset_x = (max_dim - w) // 2
        offset_y = (max_dim - h) // 2
        square_canvas[offset_y:offset_y+h, offset_x:offset_x+w] = roi
        
        # 7. Redimensionar a 28x28 con LANCZOS (como entrenamiento con PIL)
        resized = cv2.resize(square_canvas, (self.w, self.h), interpolation=cv2.INTER_LANCZOS4)
        
        # 8. Normalizar a [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # 9. ⭐ INVERTIR COLORES (CRÍTICO) ⭐
        # Entrenamiento hace: 1.0 - img
        # Esto convierte: blanco (1.0) -> negro (0.0), negro (0.0) -> blanco (1.0)
        normalized = 1.0 - normalized
        
        # 10. Agregar dimensión de canal
        if self.c == 1:
            normalized = np.expand_dims(normalized, axis=-1)
        
        return normalized
    
    def visualize_preprocessing(self, strokes: List[List[Tuple[float, float]]], show_stats: bool = False) -> np.ndarray:
        """
        Genera visualización del preprocesamiento para debugging.
        
        Returns:
            Imagen RGB mostrando el resultado (blanco sobre negro como el modelo lo ve)
        """
        processed = self.preprocess(strokes)
        
        # Convertir a uint8 para visualización
        vis = (processed.squeeze() * 255).astype(np.uint8)
        
        # Convertir a RGB (ya está en formato correcto: blanco sobre negro)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
        
        if show_stats:
            num_strokes = len(strokes)
            total_points = sum(len(s) for s in strokes)
            white_pixels = np.sum(processed > 0.01)
            coverage = (white_pixels / (self.h * self.w)) * 100
            
            print(f"\n--- Estadísticas de Preprocesamiento ---")
            print(f"Trazos: {num_strokes}")
            print(f"Puntos totales: {total_points}")
            print(f"Píxeles blancos (contenido): {white_pixels}/{self.h * self.w}")
            print(f"Cobertura: {coverage:.1f}%")
            print(f"Forma de salida: {processed.shape}")
            print(f"Rango: [{processed.min():.3f}, {processed.max():.3f}]")
            print(f"Invertido: ✅ (blanco sobre negro)")
        
        return vis_rgb