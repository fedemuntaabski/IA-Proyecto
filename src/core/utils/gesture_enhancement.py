"""
Enhanced Gesture Recognition - Reconocimiento de Gestos Mejorado.

Este módulo proporciona detección de gestos más robusta y confiable
con análisis avanzado de movimiento.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque


class GestureState:
    """Representa el estado actual de un gesto."""

    def __init__(self, gesture_type: str):
        """
        Inicializa el estado del gesto.

        Args:
            gesture_type: Tipo de gesto ('drawing', 'pointing', 'static', etc.)
        """
        self.gesture_type = gesture_type
        self.confidence = 0.0
        self.stability_score = 0.0
        self.duration = 0.0
        self.movement_direction = (0.0, 0.0)
        self.velocity = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'type': self.gesture_type,
            'confidence': self.confidence,
            'stability': self.stability_score,
            'duration': self.duration,
            'direction': self.movement_direction,
            'velocity': self.velocity
        }


class GestureAnalyzer:
    """Analizador avanzado de gestos."""

    def __init__(self, window_size: int = 10):
        """
        Inicializa el analizador de gestos.

        Args:
            window_size: Tamaño de la ventana de análisis
        """
        self.window_size = window_size
        self.position_history = deque(maxlen=window_size)
        self.velocity_history = deque(maxlen=window_size)
        self.gesture_state = GestureState('static')

    def analyze_movement(self, current_position: Tuple[float, float],
                        current_time: float = None) -> GestureState:
        """
        Analiza el movimiento actual.

        Args:
            current_position: Posición actual (x, y)
            current_time: Tiempo actual (para cálculo de velocidad)

        Returns:
            Estado del gesto analizado
        """
        self.position_history.append(current_position)

        if len(self.position_history) < 2:
            return self.gesture_state

        # Calcular movimiento reciente
        prev_position = list(self.position_history)[-2]
        movement = (
            current_position[0] - prev_position[0],
            current_position[1] - prev_position[1]
        )

        # Calcular velocidad
        velocity = np.sqrt(movement[0]**2 + movement[1]**2)
        self.velocity_history.append(velocity)

        # Determinar tipo de gesto
        if velocity < 0.5:
            gesture_type = 'static'
            confidence = 0.9
        elif velocity < 3.0:
            gesture_type = 'slow'
            confidence = 0.7
        else:
            gesture_type = 'fast'
            confidence = 0.85

        # Calcular stabilidad (consistencia)
        stability = self._calculate_stability()

        # Actualizar estado
        self.gesture_state.gesture_type = gesture_type
        self.gesture_state.confidence = confidence
        self.gesture_state.stability_score = stability
        self.gesture_state.movement_direction = self._get_dominant_direction()
        self.gesture_state.velocity = velocity

        return self.gesture_state

    def _calculate_stability(self) -> float:
        """Calcula la estabilidad del movimiento (0-1)."""
        if len(self.velocity_history) < 3:
            return 0.5

        velocities = list(self.velocity_history)
        variance = np.var(velocities)
        
        # Transformar varianza a score de estabilidad
        # Baja varianza = alta estabilidad
        stability = 1.0 / (1.0 + variance / 5.0)
        return np.clip(stability, 0.0, 1.0)

    def _get_dominant_direction(self) -> Tuple[float, float]:
        """Obtiene la dirección dominante del movimiento."""
        if len(self.position_history) < 5:
            return (0.0, 0.0)

        positions = list(self.position_history)
        first = positions[0]
        last = positions[-1]

        direction = (
            (last[0] - first[0]) / max(0.1, np.linalg.norm([
                last[0] - first[0], last[1] - first[1]
            ])),
            (last[1] - first[1]) / max(0.1, np.linalg.norm([
                last[0] - first[0], last[1] - first[1]
            ]))
        )

        return direction

    def reset(self) -> None:
        """Reinicia el analizador."""
        self.position_history.clear()
        self.velocity_history.clear()
        self.gesture_state = GestureState('static')


class MultiGestureTracker:
    """Rastreador de múltiples gestos simultáneamente."""

    def __init__(self, max_gestures: int = 5):
        """
        Inicializa el rastreador.

        Args:
            max_gestures: Número máximo de gestos a rastrear
        """
        self.max_gestures = max_gestures
        self.gesture_trackers: Dict[int, GestureAnalyzer] = {}
        self.gesture_lifetime = deque(maxlen=50)

    def track_hand(self, hand_id: int, position: Tuple[float, float]) -> GestureState:
        """
        Rastrea un gesto para una mano específica.

        Args:
            hand_id: ID de la mano
            position: Posición actual

        Returns:
            Estado del gesto
        """
        if hand_id not in self.gesture_trackers:
            if len(self.gesture_trackers) >= self.max_gestures:
                # Remover el más antiguo
                oldest_id = min(self.gesture_trackers.keys())
                del self.gesture_trackers[oldest_id]
            
            self.gesture_trackers[hand_id] = GestureAnalyzer()

        tracker = self.gesture_trackers[hand_id]
        gesture_state = tracker.analyze_movement(position)

        self.gesture_lifetime.append({
            'hand_id': hand_id,
            'gesture': gesture_state.to_dict()
        })

        return gesture_state

    def remove_hand(self, hand_id: int) -> None:
        """Elimina un rastreador de gesto."""
        if hand_id in self.gesture_trackers:
            del self.gesture_trackers[hand_id]

    def get_all_gestures(self) -> Dict[int, GestureState]:
        """Obtiene todos los gestos activos."""
        return {
            hand_id: tracker.gesture_state
            for hand_id, tracker in self.gesture_trackers.items()
        }


class ContourStabilityAnalyzer:
    """Analizador de estabilidad de contornos."""

    def __init__(self, window_size: int = 5):
        """
        Inicializa el analizador.

        Args:
            window_size: Tamaño de la ventana histórica
        """
        self.window_size = window_size
        self.contour_history = deque(maxlen=window_size)

    def analyze_contour(self, contour: np.ndarray) -> Dict[str, float]:
        """
        Analiza la estabilidad de un contorno.

        Args:
            contour: Contorno a analizar

        Returns:
            Diccionario con métricas de estabilidad
        """
        # Calcular propiedades del contorno
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return {'stability': 0.0, 'consistency': 0.0}

        # Agregar al historial
        self.contour_history.append({
            'area': area,
            'perimeter': perimeter,
            'timestamp': None
        })

        # Calcular consistencia
        consistency = self._calculate_consistency()
        stability = self._calculate_stability_from_history()

        return {
            'stability': stability,
            'consistency': consistency,
            'area': area,
            'perimeter': perimeter
        }

    def _calculate_consistency(self) -> float:
        """Calcula la consistencia del contorno."""
        if len(self.contour_history) < 2:
            return 0.5

        history = list(self.contour_history)
        areas = [h['area'] for h in history]

        # Calcular variación de área
        area_variance = np.var(areas) / (np.mean(areas) ** 2) if np.mean(areas) > 0 else 1.0
        area_variance = min(area_variance, 1.0)

        # Transformar a score de consistencia
        consistency = 1.0 - area_variance
        return np.clip(consistency, 0.0, 1.0)

    def _calculate_stability_from_history(self) -> float:
        """Calcula la estabilidad basada en historial."""
        if len(self.contour_history) < 3:
            return 0.5

        history = list(self.contour_history)
        perimeters = [h['perimeter'] for h in history]

        # Calcular variación de perímetro
        perimeter_variance = np.var(perimeters) / (np.mean(perimeters) ** 2) if np.mean(perimeters) > 0 else 1.0
        perimeter_variance = min(perimeter_variance, 1.0)

        # Transformar a score de estabilidad
        stability = 1.0 - (perimeter_variance * 0.5)
        return np.clip(stability, 0.0, 1.0)


# Instancias globales
gesture_analyzer = GestureAnalyzer()
multi_gesture_tracker = MultiGestureTracker()
contour_stability_analyzer = ContourStabilityAnalyzer()
