"""
Utilidades comunes para el proyecto Air Draw Classifier.

Este módulo contiene funciones auxiliares reutilizables.
"""

import time
from typing import List, Optional


class FPSCounter:
    """
    Contador de FPS para medir rendimiento en tiempo real.

    Atributos:
        fps_history: Lista de timestamps para calcular FPS promedio
        max_history: Número máximo de timestamps a mantener
    """

    def __init__(self, max_history: int = 30):
        """
        Inicializa el contador de FPS.

        Args:
            max_history: Número máximo de mediciones a mantener
        """
        self.fps_history: List[float] = []
        self.max_history = max_history

    def update(self) -> None:
        """Registra un nuevo frame para el cálculo de FPS."""
        current_time = time.time()
        self.fps_history.append(current_time)

        # Mantener solo el historial reciente
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)

    def get_fps(self) -> float:
        """
        Calcula el FPS promedio basado en el historial.

        Returns:
            FPS promedio o 0 si no hay suficientes datos
        """
        if len(self.fps_history) < 2:
            return 0.0

        # Calcular tiempo total y número de frames
        time_span = self.fps_history[-1] - self.fps_history[0]
        frame_count = len(self.fps_history) - 1  # -1 porque timestamps son entre frames

        if time_span <= 0:
            return 0.0

        return frame_count / time_span

    def reset(self) -> None:
        """Reinicia el contador de FPS."""
        self.fps_history.clear()


def calculate_average(values: List[float]) -> float:
    """
    Calcula el promedio de una lista de valores.

    Args:
        values: Lista de valores numéricos

    Returns:
        Promedio de los valores o 0 si la lista está vacía
    """
    return sum(values) / len(values) if values else 0.0


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Limita un valor entre un mínimo y máximo.

    Args:
        value: Valor a limitar
        min_value: Valor mínimo
        max_value: Valor máximo

    Returns:
        Valor limitado
    """
    return max(min_value, min(value, max_value))