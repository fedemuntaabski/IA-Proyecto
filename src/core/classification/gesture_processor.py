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
from scipy.interpolate import CubicSpline


class VelocityAnalyzer:
    """Analizador de velocidad de movimiento."""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.timestamp_history = []
    
    def calculate_velocities(self, points: List[Tuple[int, int]], 
                           timestamps: List[float]) -> List[float]:
        """Calcula velocidades para cada punto."""
        if len(points) < 2 or len(timestamps) != len(points):
            return [0.0] * len(points)
        
        velocities = [0.0]
        
        for i in range(1, len(points)):
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > 0:
                distance = np.sqrt(
                    (points[i][0] - points[i-1][0])**2 + 
                    (points[i][1] - points[i-1][1])**2
                )
                velocity = distance / time_diff
            else:
                velocity = 0.0
            
            velocities.append(velocity)
        
        return velocities
    
    def classify_velocity(self, velocity: float) -> str:
        """Clasifica velocidad en categorías."""
        if velocity < 50:
            return 'slow'
        elif velocity < 150:
            return 'medium'
        else:
            return 'fast'


class AdaptiveSmoothing:
    """Sistema de suavizado adaptativo basado en velocidad."""
    
    def __init__(self):
        self.velocity_analyzer = VelocityAnalyzer()
        self.smoothing_levels = {
            'slow': {'window': 5, 'strength': 0.8},
            'medium': {'window': 3, 'strength': 0.6},
            'fast': {'window': 2, 'strength': 0.3}
        }
    
    def smooth_points(self, points: List[Tuple[int, int]], 
                     timestamps: List[float]) -> List[Tuple[int, int]]:
        """Suaviza puntos con nivel adaptativo."""
        if len(points) < 2:
            return points
        
        velocities = self.velocity_analyzer.calculate_velocities(points, timestamps)
        smoothed_points = []
        
        for i, point in enumerate(points):
            velocity = velocities[i] if i < len(velocities) else 0
            level = self.velocity_analyzer.classify_velocity(velocity)
            config = self.smoothing_levels[level]
            
            smoothed = self._apply_smoothing(point, smoothed_points, config, i, points)
            smoothed_points.append(smoothed)
        
        return smoothed_points
    
    def _apply_smoothing(self, point: Tuple[int, int], 
                        smoothed_points: List, config: dict, 
                        index: int, all_points: List) -> Tuple[int, int]:
        """Aplica suavizado según configuración."""
        window = config['window']
        strength = config['strength']
        
        if index < window // 2 or len(smoothed_points) < window // 2:
            return point
        
        # Usar puntos suavizados anteriores si están disponibles
        start_idx = max(0, len(smoothed_points) - window // 2)
        window_points = smoothed_points[start_idx:] + [point]
        
        avg_x = int(np.mean([p[0] for p in window_points]) * strength + 
                   point[0] * (1 - strength))
        avg_y = int(np.mean([p[1] for p in window_points]) * strength + 
                   point[1] * (1 - strength))
        
        return (avg_x, avg_y)


class DuplicateFilter:
    """Filtrado inteligente de puntos duplicados."""
    
    def __init__(self, min_distance: float = 2.0, 
                 direction_change_threshold: float = 30.0):
        self.min_distance = min_distance
        self.direction_change_threshold = direction_change_threshold
    
    def filter_duplicates(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filtra puntos muy cercanos preservando cambios de dirección."""
        if len(points) < 3:
            return points
        
        filtered = [points[0]]
        
        for i in range(1, len(points)):
            current = points[i]
            previous = filtered[-1]
            
            distance = self._calculate_distance(previous, current)
            
            if distance < self.min_distance:
                # Punto muy cercano, pero verificar si hay cambio de dirección
                if i > 0 and len(filtered) > 1:
                    direction_change = self._calculate_direction_change(
                        filtered[-2], previous, current
                    )
                    if direction_change > self.direction_change_threshold:
                        # Hay cambio de dirección importante, mantener punto
                        filtered.append(current)
                # Si no hay cambio importante, descartar punto
            else:
                filtered.append(current)
        
        return filtered
    
    def _calculate_distance(self, p1: Tuple[int, int], 
                          p2: Tuple[int, int]) -> float:
        """Calcula distancia euclidiana."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _calculate_direction_change(self, p1: Tuple[int, int], 
                                   p2: Tuple[int, int], 
                                   p3: Tuple[int, int]) -> float:
        """Calcula cambio de dirección en grados."""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg


class PointInterpolator:
    """Interpolación de puntos faltantes."""
    
    def __init__(self, max_gap_distance: float = 20.0, 
                 interpolation_method: str = 'cubic'):
        self.max_gap_distance = max_gap_distance
        self.method = interpolation_method
    
    def interpolate_gaps(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Interpola puntos faltantes en gaps."""
        if len(points) < 2:
            return points
        
        interpolated = [points[0]]
        
        for i in range(1, len(points)):
            prev_point = points[i-1]
            current_point = points[i]
            distance = self._calculate_distance(prev_point, current_point)
            
            if distance <= self.max_gap_distance:
                interpolated.append(current_point)
            else:
                # Interpolar puntos faltantes
                gap_points = self._interpolate_points(
                    prev_point, current_point, distance
                )
                interpolated.extend(gap_points)
                interpolated.append(current_point)
        
        return interpolated
    
    def _calculate_distance(self, p1: Tuple[int, int], 
                          p2: Tuple[int, int]) -> float:
        """Calcula distancia euclidiana."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _interpolate_points(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                           distance: float) -> List[Tuple[int, int]]:
        """Interpola puntos entre dos puntos."""
        num_points = int(distance / self.max_gap_distance)
        
        if self.method == 'cubic':
            return self._cubic_interpolate(p1, p2, num_points)
        else:
            return self._linear_interpolate(p1, p2, num_points)
    
    def _linear_interpolate(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                           num_points: int) -> List[Tuple[int, int]]:
        """Interpolación lineal."""
        points = []
        for i in range(1, num_points):
            t = i / (num_points + 1)
            x = int(p1[0] * (1 - t) + p2[0] * t)
            y = int(p1[1] * (1 - t) + p2[1] * t)
            points.append((x, y))
        
        return points
    
    def _cubic_interpolate(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                          num_points: int) -> List[Tuple[int, int]]:
        """Interpolación cúbica para trazos más suave."""
        try:
            # Crear puntos de control adicionales para suavidad
            x_vals = np.array([0, 1])
            y_x = np.array([p1[0], p2[0]])
            y_y = np.array([p1[1], p2[1]])
            
            # Usar linear si no hay suficientes puntos para cubic
            if len(x_vals) < 2:
                return self._linear_interpolate(p1, p2, num_points)
            
            cs_x = CubicSpline(x_vals, y_x, bc_type='natural')
            cs_y = CubicSpline(x_vals, y_y, bc_type='natural')
            
            points = []
            for i in range(1, num_points):
                t = i / (num_points + 1)
                x = int(cs_x(t))
                y = int(cs_y(t))
                points.append((x, y))
            
            return points
        except Exception:
            # Fallback a interpolación lineal
            return self._linear_interpolate(p1, p2, num_points)


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
        self.stroke_timestamps: List[float] = []
        
        # Herramientas avanzadas de procesamiento
        self.adaptive_smoothing = AdaptiveSmoothing()
        self.duplicate_filter = DuplicateFilter(min_distance=2.0)
        self.point_interpolator = PointInterpolator(max_gap_distance=20.0)
    
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
        import time
        pixel_coord = self.normalize_coordinates(coord, frame_shape)
        self.stroke_points.append(pixel_coord)
        self.stroke_timestamps.append(time.time())
    
    def smooth_stroke(self, points: List[Tuple[int, int]], window_size: int = 3) -> List[Tuple[int, int]]:
        """
        Suaviza los puntos del trazo para reducir ruido.
        
        Args:
            points: Lista de puntos (x, y)
            window_size: Tamaño de ventana para suavizado
            
        Returns:
            Lista de puntos suavizados
        """
        if len(points) < 2:
            return points
        
        # Filtrar duplicados
        points = self.duplicate_filter.filter_duplicates(points)
        
        # Interpolar gaps
        points = self.point_interpolator.interpolate_gaps(points)
        
        # Suavizado adaptativo
        if len(self.stroke_timestamps) >= len(points):
            timestamps = self.stroke_timestamps[-len(points):]
        else:
            # Crear timestamps si no están disponibles
            import time
            current_time = time.time()
            timestamps = [current_time - (len(points) - i) * 0.016 for i in range(len(points))]
        
        points = self.adaptive_smoothing.smooth_points(points, timestamps)
        
        return points
    
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
        self.stroke_timestamps = []
    
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
