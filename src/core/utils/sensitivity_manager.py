"""
Sensitivity Manager - Gestor de Sensibilidad Adaptativa.

Este módulo gestiona ajustes dinámicos de sensibilidad basados en
condiciones ambientales, rendimiento y feedback del usuario.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import deque
from .performance_monitor import performance_monitor
from ..constants import SENSITIVITY_CONFIG


class DynamicSensitivityManager:
    """Gestor de sensibilidad dinámica y adaptativa."""

    def __init__(self):
        """Inicializa el gestor de sensibilidad."""
        # Sensibilidad base (0.0 a 1.0)
        self.base_sensitivity = SENSITIVITY_CONFIG['BASE_SENSITIVITY']
        
        # Históricos para cálculo de tendencias
        self.frame_quality_history = deque(maxlen=SENSITIVITY_CONFIG['FRAME_QUALITY_HISTORY_SIZE'])
        self.noise_level_history = deque(maxlen=SENSITIVITY_CONFIG['NOISE_LEVEL_HISTORY_SIZE'])
        self.fps_history = deque(maxlen=SENSITIVITY_CONFIG['FPS_HISTORY_SIZE'])
        
        # Factores de ajuste
        self.performance_factor = 1.0
        self.environmental_factor = 1.0
        self.stability_factor = 1.0
        
        # Umbrales adaptativos
        self.current_thresholds = {
            'motion_threshold': 0.1,
            'area_threshold': 0.3,
            'contour_threshold': 0.4
        }

    def analyze_frame_quality(self, frame: np.ndarray) -> float:
        """
        Analiza la calidad del frame.

        Args:
            frame: Frame de OpenCV (BGR). Debe ser un array válido no vacío.

        Returns:
            Puntuación de calidad (0.0 a 1.0)

        Raises:
            ValueError: Si el frame es None o inválido
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return 0.0

        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Medir nitidez usando Laplaciano
            laplacian_var = cv2.Laplacian(gray, cv2.uint8).var()
            
            # Normalizar (valores típicos: 100-500+)
            sharpness_score = min(laplacian_var / 500.0, 1.0)

            # Medir contraste usando desviación estándar
            contrast_score = np.std(gray) / 255.0

            # Combinar puntuaciones
            quality_score = 0.6 * sharpness_score + 0.4 * contrast_score
            
            self.frame_quality_history.append(quality_score)
            return quality_score

        except cv2.error as e:
            print(f"Error analyzing frame quality: {e}")
            return 0.0

    def measure_noise_level(self, frame: np.ndarray, mask: np.ndarray) -> float:
        """
        Mide el nivel de ruido en el frame.

        Args:
            frame: Frame de OpenCV (BGR)
            mask: Máscara de segmentación binaria

        Returns:
            Nivel de ruido (0.0 a 1.0, donde 1.0 es muy ruidoso)

        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if frame is None or mask is None:
            raise ValueError("Frame and mask cannot be None")

        if mask.sum() == 0:
            return 0.5

        try:
            # Aplicar máscara
            masked = cv2.bitwise_and(frame, frame, mask=mask)

            # Convertir a escala de grises
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

            # Calcular gradientes para detectar ruido
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Magnitud de gradientes
            magnitude = np.sqrt(sobelx**2 + sobely**2)

            # Ruido típicamente tiene gradientes altos y erráticos
            noise_level = np.std(magnitude) / 100.0
            noise_level = min(noise_level, 1.0)

            self.noise_level_history.append(noise_level)
            return noise_level

        except cv2.error as e:
            print(f"Error measuring noise level: {e}")
            return 0.5

    def calculate_current_sensitivity(self) -> float:
        """
        Calcula la sensibilidad actual basada en factores dinámicos.

        Returns:
            Sensibilidad ajustada (0.0 a 1.0)
        """
        # Obtener factores de rendimiento
        perf_metrics = performance_monitor.update()

        # Factor de rendimiento: reduce sensibilidad si el rendimiento es bajo
        if 'fps' in perf_metrics:
            fps = perf_metrics['fps']
            if fps < 15:
                self.performance_factor = 0.7
            elif fps < 20:
                self.performance_factor = 0.85
            elif fps > 30:
                self.performance_factor = 1.0
            else:
                self.performance_factor = 0.9

        # Factor ambiental basado en calidad de frame
        if len(self.frame_quality_history) > 0:
            avg_quality = np.mean(self.frame_quality_history)
            if avg_quality < 0.3:
                self.environmental_factor = 0.7
            elif avg_quality < 0.5:
                self.environmental_factor = 0.85
            elif avg_quality > 0.8:
                self.environmental_factor = 1.1
            else:
                self.environmental_factor = 1.0

        # Factor de estabilidad basado en ruido
        if len(self.noise_level_history) > 0:
            avg_noise = np.mean(self.noise_level_history)
            if avg_noise > 0.7:
                self.stability_factor = 0.75
            elif avg_noise > 0.5:
                self.stability_factor = 0.85
            elif avg_noise < 0.2:
                self.stability_factor = 1.0
            else:
                self.stability_factor = 0.9

        # Calcular sensibilidad final
        current_sensitivity = (
            self.base_sensitivity *
            self.performance_factor *
            self.environmental_factor *
            self.stability_factor
        )

        # Limitar a rango válido
        return np.clip(current_sensitivity, 
                      SENSITIVITY_CONFIG['MIN_SENSITIVITY'],
                      SENSITIVITY_CONFIG['MAX_SENSITIVITY'])

    def update_thresholds(self, sensitivity: float) -> Dict[str, float]:
        """
        Actualiza los umbrales basado en sensibilidad actual.

        Args:
            sensitivity: Sensibilidad actual

        Returns:
            Diccionario con umbrales actualizados
        """
        # Sensibilidad más alta = umbrales más bajos
        self.current_thresholds = {
            'motion_threshold': 0.15 / sensitivity,
            'area_threshold': 0.5 / sensitivity,
            'contour_threshold': 0.6 / sensitivity,
            'stability_requirement': max(1, int(3 / sensitivity))
        }

        return self.current_thresholds

    def set_user_sensitivity(self, level: float) -> None:
        """
        Establece la sensibilidad según preferencia del usuario.

        Args:
            level: Nivel de sensibilidad (0.0 a 1.0)

        Raises:
            ValueError: Si el nivel está fuera del rango válido
        """
        if not (0.0 <= level <= 1.0):
            raise ValueError("Sensitivity level must be between 0.0 and 1.0")

        self.base_sensitivity = np.clip(level, 0.1, 1.0)

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Obtiene información de diagnóstico de sensibilidad.

        Returns:
            Diccionario con información de diagnóstico
        """
        return {
            'base_sensitivity': self.base_sensitivity,
            'current_sensitivity': self.calculate_current_sensitivity(),
            'performance_factor': self.performance_factor,
            'environmental_factor': self.environmental_factor,
            'stability_factor': self.stability_factor,
            'avg_frame_quality': np.mean(self.frame_quality_history) if self.frame_quality_history else 0.5,
            'avg_noise_level': np.mean(self.noise_level_history) if self.noise_level_history else 0.5,
            'current_thresholds': self.current_thresholds
        }


# Instancia global
sensitivity_manager = DynamicSensitivityManager()
