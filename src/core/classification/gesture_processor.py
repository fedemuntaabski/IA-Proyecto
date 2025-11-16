"""
Procesador de gestos que convierte movimientos de manos en imágenes 28x28.

Este módulo toma las coordenadas de los dedos detectadas por MediaPipe
y las convierte en imágenes que puedan ser clasificadas por el modelo.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
from .constants import MODEL_IMAGE_SIZE, DEFAULT_LINE_WIDTH, DEFAULT_CANVAS_SIZE


class GestureProcessor:
    """
    Convierte gestos de manos en imágenes compatibles con el modelo (28x28).
    
    Atributos:
        image_size: Tamaño de la imagen de salida (default: 28)
        line_width: Ancho de línea para dibujar (default: 2)
        canvas_size: Tamaño del canvas interno de dibujo (default: 256)
    """
    
    def __init__(self, image_size: int = MODEL_IMAGE_SIZE, line_width: int = DEFAULT_LINE_WIDTH, canvas_size: int = DEFAULT_CANVAS_SIZE):
        """
        Inicializa el procesador de gestos.
        
        Args:
            image_size: Tamaño final de imagen (28x28 para modelo)
            line_width: Ancho de línea a dibujar
            canvas_size: Tamaño interno del canvas
        """
        self.image_size = image_size
        self.line_width = line_width
        self.canvas_size = canvas_size
        self.stroke_points: List[Tuple[float, float]] = []
    
    def normalize_coordinates(self, coord: Tuple[float, float], frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convierte coordenadas normalizadas de MediaPipe a píxeles del canvas.
        
        Args:
            coord: Tupla (x, y) normalizada (0-1)
            frame_shape: Tupla (height, width) del frame
            
        Returns:
            Tupla (x_pixel, y_pixel) en el canvas
        """
        x_norm, y_norm = coord
        height, width = frame_shape
        
        # Convertir a píxeles del frame
        x_pixel = int(x_norm * width)
        y_pixel = int(y_norm * height)
        
        # Escalar al canvas
        x_canvas = int((x_pixel / width) * self.canvas_size)
        y_canvas = int((y_pixel / height) * self.canvas_size)
        
        return x_canvas, y_canvas
    
    def add_point(self, coord: Tuple[float, float], frame_shape: Tuple[int, int]):
        """
        Agrega un punto al trazo actual.
        
        Args:
            coord: Coordenadas normalizadas (0-1)
            frame_shape: Forma del frame (height, width)
        """
        pixel_coord = self.normalize_coordinates(coord, frame_shape)
        self.stroke_points.append(pixel_coord)
    
    def smooth_stroke(self, points: List[Tuple[int, int]], window_size: int = 3) -> List[Tuple[int, int]]:
        """
        Suaviza los puntos del trazo para reducir ruido.
        
        Args:
            points: Lista de puntos (x, y)
            window_size: Tamaño de ventana para suavizado
            
        Returns:
            Lista de puntos suavizados
        """
        if len(points) < window_size:
            return points
        
        smoothed = []
        for i in range(len(points)):
            if i < window_size // 2 or i >= len(points) - window_size // 2:
                smoothed.append(points[i])
            else:
                # Promediar con puntos vecinos
                window = points[i - window_size // 2 : i + window_size // 2 + 1]
                avg_x = int(np.mean([p[0] for p in window]))
                avg_y = int(np.mean([p[1] for p in window]))
                smoothed.append((avg_x, avg_y))
        
        return smoothed
    
    def points_to_image(self, points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Convierte una lista de puntos en una imagen 28x28.
        
        Args:
            points: Lista de puntos (x, y)
            
        Returns:
            Array numpy 28x28 con valores 0-1 (normalizado)
        """
        # Crear imagen blanca
        img = Image.new('L', (self.canvas_size, self.canvas_size), color=255)
        draw = ImageDraw.Draw(img)
        
        # Dibujar trazo
        if len(points) > 1:
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=0, width=self.line_width)
        
        # Redimensionar a 28x28
        img_resized = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Convertir a numpy array y normalizar
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Invertir: dibujo blanco sobre fondo negro (como el dataset)
        img_array = 1.0 - img_array
        
        return img_array
    
    def get_gesture_image(self, as_list: bool = False) -> Optional[np.ndarray]:
        """
        Obtiene la imagen del gesto actual y limpia el canvas.
        
        Args:
            as_list: Si True, devuelve la imagen como lista de listas
            
        Returns:
            Array numpy 28x28 o lista de listas, o None si no hay puntos
        """
        if len(self.stroke_points) < 2:
            return None
        
        # Suavizar puntos
        smoothed_points = self.smooth_stroke(self.stroke_points)
        
        # Convertir a imagen
        image = self.points_to_image(smoothed_points)
        
        # Limpiar para próximo gesto
        self.clear()
        
        if as_list:
            return image.tolist()
        
        return image
    
    def clear(self):
        """Limpia el canvas y los puntos acumulados."""
        self.stroke_points = []
    
    def get_preview(self) -> Optional[np.ndarray]:
        """
        Obtiene una vista previa de lo dibujado hasta ahora sin limpiar.
        
        Returns:
            Array numpy 28x28 o None
        """
        if len(self.stroke_points) < 1:
            return None
        
        smoothed_points = self.smooth_stroke(self.stroke_points)
        return self.points_to_image(smoothed_points)
    
    def get_gesture_image_for_feedback(self) -> Optional[List[List[float]]]:
        """
        Obtiene la imagen del gesto actual como lista para feedback, sin limpiar.
        
        Returns:
            Lista de listas con la imagen 28x28, o None si no hay puntos
        """
        if len(self.stroke_points) < 2:
            return None
        
        # Suavizar puntos
        smoothed_points = self.smooth_stroke(self.stroke_points)
        
        # Convertir a imagen
        image = self.points_to_image(smoothed_points)
        
        # Devolver como lista sin limpiar
        return image.tolist()
    
    def draw_on_frame(self, frame: np.ndarray, points: Optional[List[Tuple[int, int]]] = None,
                      frame_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Dibuja los puntos del trazo en el frame actual.
        
        Args:
            frame: Frame de OpenCV
            points: Puntos a dibujar (si None, usa stroke_points)
            frame_shape: Forma del frame original
            
        Returns:
            Frame con trazo dibujado
        """
        frame_copy = frame.copy()
        
        if points is None:
            points = self.stroke_points
        
        if len(points) < 2:
            return frame_copy
        
        if frame_shape is None:
            frame_shape = frame.shape[:2]
        
        # Desnormalizar puntos para mostrar en frame
        height, width = frame_shape
        
        for i in range(len(points) - 1):
            # Convertir del canvas al frame
            x1 = int((points[i][0] / self.canvas_size) * width)
            y1 = int((points[i][1] / self.canvas_size) * height)
            x2 = int((points[i + 1][0] / self.canvas_size) * width)
            y2 = int((points[i + 1][1] / self.canvas_size) * height)
            
            # Dibujar línea
            cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame_copy, (x1, y1), 3, (0, 0, 255), -1)
        
        return frame_copy
