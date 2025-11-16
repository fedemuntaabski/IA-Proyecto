"""
Lighting Analysis - Análisis y Compensación de Iluminación Avanzada.

Este módulo proporciona herramientas para análisis, detección y compensación
de variaciones de iluminación en tiempo real.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from collections import deque


class LightingAnalyzer:
    """Analizador avanzado de iluminación."""

    def __init__(self, region_grid: int = 4):
        """
        Inicializa el analizador de iluminación.

        Args:
            region_grid: Tamaño de la malla de regiones para análisis
        """
        self.region_grid = region_grid
        self.lighting_history = deque(maxlen=30)
        self.dark_regions_history = deque(maxlen=30)

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Realiza análisis exhaustivo de iluminación.

        Args:
            frame: Frame de OpenCV (BGR)

        Returns:
            Diccionario con análisis de iluminación
        """
        if frame is None or frame.size == 0:
            return self._empty_analysis()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Análisis global
        global_mean = np.mean(gray)
        global_std = np.std(gray)
        
        # Histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Detectar si es demasiado oscuro o brillante
        is_dark = global_mean < 85
        is_bright = global_mean > 170
        is_overexposed = np.percentile(gray, 95) > 240

        # Análisis por regiones
        region_stats = self._analyze_regions(gray)

        # Detección de gradiente de luz
        light_gradient = self._detect_light_gradient(gray)

        # Detección de sombras
        shadow_ratio = self._detect_shadows(frame, gray)

        # Recomendación de compensación
        compensation = self._recommend_compensation(
            global_mean, is_dark, is_bright, is_overexposed, shadow_ratio
        )

        analysis = {
            'global_brightness': float(global_mean),
            'brightness_std': float(global_std),
            'is_dark': is_dark,
            'is_bright': is_bright,
            'is_overexposed': is_overexposed,
            'shadow_ratio': shadow_ratio,
            'light_gradient': light_gradient,
            'region_stats': region_stats,
            'compensation_recommended': compensation,
            'histogram_shape': 'normal' if 50 < global_mean < 200 else ('dark' if is_dark else 'bright')
        }

        self.lighting_history.append(analysis)
        return analysis

    def _analyze_regions(self, gray: np.ndarray) -> Dict[str, float]:
        """Analiza iluminación por regiones."""
        height, width = gray.shape
        region_height = height // self.region_grid
        region_width = width // self.region_grid

        regions = {}
        min_brightness = 255
        max_brightness = 0

        for i in range(self.region_grid):
            for j in range(self.region_grid):
                y1 = i * region_height
                y2 = (i + 1) * region_height if i < self.region_grid - 1 else height
                x1 = j * region_width
                x2 = (j + 1) * region_width if j < self.region_grid - 1 else width

                region = gray[y1:y2, x1:x2]
                brightness = np.mean(region)

                regions[f'region_{i}_{j}'] = float(brightness)

                min_brightness = min(min_brightness, int(brightness))
                max_brightness = max(max_brightness, int(brightness))

        regions['min_brightness'] = float(min_brightness)
        regions['max_brightness'] = float(max_brightness)
        regions['brightness_range'] = float(max_brightness - min_brightness)

        return regions

    def _detect_light_gradient(self, gray: np.ndarray) -> Dict[str, float]:
        """Detecta gradientes de luz significativos."""
        # Calcular gradientes
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        return {
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude)),
            'gradient_max': float(np.max(gradient_magnitude)),
            'has_significant_gradient': float(np.mean(gradient_magnitude)) > 30
        }

    def _detect_shadows(self, frame: np.ndarray, gray: np.ndarray) -> float:
        """
        Detecta presencia de sombras.

        Returns:
            Ratio de píxeles de sombra (0.0 a 1.0)
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango para sombras: baja saturación y bajo brillo
        lower_shadow = np.array([0, 0, 0])
        upper_shadow = np.array([180, 50, 100])

        shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
        shadow_ratio = np.count_nonzero(shadow_mask) / shadow_mask.size

        return float(shadow_ratio)

    def _recommend_compensation(self, brightness: float, is_dark: bool, 
                               is_bright: bool, is_overexposed: bool,
                               shadow_ratio: float) -> Dict[str, Any]:
        """
        Recomienda compensaciones de iluminación.

        Returns:
            Diccionario con recomendaciones
        """
        recommendation = {
            'apply_clahe': False,
            'apply_histogram_equalization': False,
            'apply_gamma_correction': False,
            'gamma_value': 1.0,
            'reduce_brightness': False,
            'increase_brightness': False,
            'advice': []
        }

        if is_dark:
            recommendation['apply_gamma_correction'] = True
            recommendation['gamma_value'] = 1.5  # Aumentar
            recommendation['advice'].append("Iluminación muy oscura: Aumentando brillo")
        elif is_bright and not is_overexposed:
            recommendation['apply_gamma_correction'] = True
            recommendation['gamma_value'] = 0.7  # Reducir
            recommendation['reduce_brightness'] = True
            recommendation['advice'].append("Iluminación muy brillante: Reduciendo brillo")
        elif is_overexposed:
            recommendation['apply_histogram_equalization'] = True
            recommendation['reduce_brightness'] = True
            recommendation['advice'].append("Exposición excesiva: Normalizando")

        if shadow_ratio > 0.4:
            recommendation['apply_clahe'] = True
            recommendation['advice'].append(f"Muchas sombras detectadas ({shadow_ratio:.0%})")

        if brightness < 100 and shadow_ratio > 0.3:
            recommendation['apply_clahe'] = True
            recommendation['apply_gamma_correction'] = True
            recommendation['advice'].append("Condiciones difíciles: Aplicando múltiples compensaciones")

        return recommendation

    def _empty_analysis(self) -> Dict[str, Any]:
        """Retorna un análisis vacío."""
        return {
            'global_brightness': 128.0,
            'brightness_std': 0.0,
            'is_dark': False,
            'is_bright': False,
            'is_overexposed': False,
            'shadow_ratio': 0.0,
            'light_gradient': {'gradient_mean': 0.0},
            'region_stats': {},
            'compensation_recommended': {'apply_clahe': False}
        }

    def get_average_brightness(self) -> float:
        """Obtiene el brillo promedio de los últimos frames."""
        if not self.lighting_history:
            return 128.0
        return np.mean([a['global_brightness'] for a in self.lighting_history])


class LightingCompensator:
    """Compensador de iluminación."""

    def __init__(self):
        """Inicializa el compensador."""
        self.analyzer = LightingAnalyzer(region_grid=4)

    def compensate_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compensa iluminación en un frame.

        Args:
            frame: Frame de OpenCV (BGR)

        Returns:
            Tupla de (frame_compensado, análisis)
        """
        analysis = self.analyzer.analyze_frame(frame)
        recommendation = analysis['compensation_recommended']

        compensated = frame.copy()

        # Aplicar compensaciones recomendadas
        if recommendation['apply_clahe']:
            compensated = self._apply_clahe(compensated)

        if recommendation['apply_gamma_correction']:
            compensated = self._apply_gamma_correction(
                compensated, recommendation['gamma_value']
            )

        if recommendation['apply_histogram_equalization']:
            compensated = self._apply_histogram_equalization(compensated)

        return compensated, analysis

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _apply_gamma_correction(self, frame: np.ndarray, gamma: float) -> np.ndarray:
        """Aplica corrección gamma."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)

        return cv2.LUT(frame, table)

    def _apply_histogram_equalization(self, frame: np.ndarray) -> np.ndarray:
        """Aplica ecualización de histograma."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l = cv2.equalizeHist(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Instancias globales
lighting_analyzer = LightingAnalyzer()
lighting_compensator = LightingCompensator()
