"""
Area Optimization - Optimización de Área de Detección.

Este módulo proporciona herramientas para optimizar dinámicamente
el área de búsqueda y detección de manos.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from collections import deque


class ROIOptimizer:
    """Optimizador de Región de Interés (ROI) para detección de manos."""

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """
        Inicializa el optimizador de ROI.

        Args:
            frame_width: Ancho del frame
            frame_height: Alto del frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # ROI completo por defecto
        self.roi = {
            'x': 0,
            'y': 0,
            'width': frame_width,
            'height': frame_height
        }
        
        # Historial de detecciones
        self.detection_history = deque(maxlen=30)
        self.roi_history = deque(maxlen=10)
        
        # Parámetros de ajuste
        self.roi_padding = 50  # Padding alrededor de detecciones
        self.adaptation_speed = 0.3  # Velocidad de adaptación (0-1)
        self.min_roi_area_ratio = 0.2  # ROI mínimo es 20% del frame

    def update_with_detections(self, contours: List) -> Dict[str, int]:
        """
        Actualiza el ROI basado en detecciones recientes.

        Args:
            contours: Lista de contornos detectados

        Returns:
            Diccionario con nuevo ROI
        """
        if not contours:
            # Sin detecciones: ampliar ROI gradualmente
            self._expand_roi()
            return self.roi

        # Calcular bounding box de todas las detecciones
        all_points = []
        for cnt in contours:
            all_points.extend(cnt.reshape(-1, 2))

        if all_points:
            all_points = np.array(all_points)
            x_min = max(0, int(np.min(all_points[:, 0])) - self.roi_padding)
            y_min = max(0, int(np.min(all_points[:, 1])) - self.roi_padding)
            x_max = min(self.frame_width, int(np.max(all_points[:, 0])) + self.roi_padding)
            y_max = min(self.frame_height, int(np.max(all_points[:, 1])) + self.roi_padding)

            proposed_roi = {
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            }

            # Aplicar adaptación gradual
            self.roi = self._smooth_roi_transition(self.roi, proposed_roi)
            self.detection_history.append(True)
        else:
            self._expand_roi()
            self.detection_history.append(False)

        self.roi_history.append(self.roi.copy())
        return self.roi

    def _smooth_roi_transition(self, current_roi: Dict, target_roi: Dict) -> Dict:
        """
        Suaviza la transición entre ROIs.

        Args:
            current_roi: ROI actual
            target_roi: ROI objetivo

        Returns:
            ROI suavizado
        """
        smooth_roi = {}
        
        for key in ['x', 'y', 'width', 'height']:
            current_val = current_roi[key]
            target_val = target_roi[key]
            # Interpolación lineal
            smooth_roi[key] = int(
                current_val + (target_val - current_val) * self.adaptation_speed
            )

        return smooth_roi

    def _expand_roi(self) -> None:
        """Expande gradualmente el ROI cuando no hay detecciones."""
        expansion = int(self.frame_width * 0.05)  # 5% de expansión
        
        self.roi['x'] = max(0, self.roi['x'] - expansion)
        self.roi['y'] = max(0, self.roi['y'] - expansion)
        self.roi['width'] = min(
            self.frame_width - self.roi['x'],
            self.roi['width'] + 2 * expansion
        )
        self.roi['height'] = min(
            self.frame_height - self.roi['y'],
            self.roi['height'] + 2 * expansion
        )

    def get_roi_mask(self) -> np.ndarray:
        """
        Obtiene una máscara del ROI actual.

        Returns:
            Máscara binaria del ROI
        """
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        roi = self.roi
        x1 = max(0, roi['x'])
        y1 = max(0, roi['y'])
        x2 = min(self.frame_width, roi['x'] + roi['width'])
        y2 = min(self.frame_height, roi['y'] + roi['height'])
        
        mask[y1:y2, x1:x2] = 255
        
        return mask

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del ROI.

        Returns:
            Diccionario con estadísticas
        """
        roi_area = self.roi['width'] * self.roi['height']
        frame_area = self.frame_width * self.frame_height
        area_ratio = roi_area / frame_area

        detection_rate = 0.0
        if self.detection_history:
            detection_rate = sum(self.detection_history) / len(self.detection_history)

        return {
            'current_roi': self.roi.copy(),
            'roi_area': roi_area,
            'roi_area_ratio': area_ratio,
            'detection_rate': detection_rate,
            'roi_history_length': len(self.roi_history)
        }

    def reset(self) -> None:
        """Reinicia el ROI al tamaño completo del frame."""
        self.roi = {
            'x': 0,
            'y': 0,
            'width': self.frame_width,
            'height': self.frame_height
        }
        self.detection_history.clear()
        self.roi_history.clear()


class FrameQualityOptimizer:
    """Optimizador dinámico de calidad de frame."""

    def __init__(self):
        """Inicializa el optimizador de calidad."""
        self.resolution_history = deque(maxlen=30)
        self.quality_scores = deque(maxlen=30)
        self.current_scale = 1.0
        self.target_fps = 30
        self.min_fps_threshold = 15

    def analyze_and_optimize(self, frame: np.ndarray, current_fps: float) -> Tuple[np.ndarray, float]:
        """
        Analiza la calidad del frame y optimiza si es necesario.

        Args:
            frame: Frame de OpenCV
            current_fps: FPS actual

        Returns:
            Tupla de (frame_optimizado, escala_aplicada)
        """
        # Calcular score de calidad
        quality_score = self._calculate_quality_score(frame, current_fps)
        self.quality_scores.append(quality_score)

        # Ajustar resolución si es necesario
        optimized_frame = frame
        applied_scale = 1.0

        if current_fps < self.min_fps_threshold and self.current_scale < 1.0:
            # Reducir resolución si FPS es muy bajo
            applied_scale = max(0.5, self.current_scale - 0.1)
            height, width = frame.shape[:2]
            new_width = int(width * applied_scale)
            new_height = int(height * applied_scale)
            optimized_frame = cv2.resize(frame, (new_width, new_height))
            self.current_scale = applied_scale

        elif current_fps > self.target_fps and self.current_scale < 1.0:
            # Aumentar resolución si hay headroom
            applied_scale = min(1.0, self.current_scale + 0.05)
            height, width = frame.shape[:2]
            new_width = int(width * applied_scale)
            new_height = int(height * applied_scale)
            optimized_frame = cv2.resize(frame, (new_width, new_height))
            self.current_scale = applied_scale

        self.resolution_history.append(applied_scale)
        return optimized_frame, applied_scale

    def _calculate_quality_score(self, frame: np.ndarray, fps: float) -> float:
        """
        Calcula un score de calidad del frame.

        Args:
            frame: Frame de OpenCV
            fps: FPS actual

        Returns:
            Score de calidad (0.0 a 1.0)
        """
        # Factor de nitidez
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.uint8).var()
        sharpness = min(laplacian_var / 500.0, 1.0)

        # Factor de FPS
        fps_factor = min(fps / self.target_fps, 1.0)

        # Combinar factores
        quality_score = 0.7 * sharpness + 0.3 * fps_factor

        return quality_score

    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual de optimización.

        Returns:
            Diccionario con información de optimización
        """
        avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0.5
        avg_scale = np.mean(self.resolution_history) if self.resolution_history else 1.0

        return {
            'current_scale': self.current_scale,
            'average_scale': avg_scale,
            'average_quality': avg_quality,
            'optimization_active': self.current_scale < 1.0
        }


# Instancias globales
roi_optimizer = ROIOptimizer()
frame_quality_optimizer = FrameQualityOptimizer()
