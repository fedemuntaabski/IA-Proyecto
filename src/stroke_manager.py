"""
stroke_manager.py - Gestión de trazos y detección de gestos
"""

import logging
import time
from typing import List, Tuple, Optional


class StrokeAccumulator:
    """Acumula puntos del trazo y detecta su finalización."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        """
        Inicializa el acumulador de trazos.
        
        Args:
            config: Configuración de trazo (thresholds, etc)
            logger: Logger para debug
        """
        self.config = config
        self.logger = logger
        self.reset()
    
    def reset(self):
        """Reinicia el acumulador."""
        self.points = []
        self.last_point_time = None
        self.last_significant_move_time = None
        self.stroke_active = False
        self.stroke_start_time = None
    
    def add_point(self, x: float, y: float, velocity: float = 0.0) -> bool:
        """
        Agrega un punto al trazo.
        
        Args:
            x, y: Coordenadas normalizadas (0-1)
            velocity: Velocidad del movimiento
        
        Returns:
            True si se completó el trazo (pausa detectada)
        """
        current_time = time.time() * 1000  # ms
        
        # Umbral de velocidad para ignorar ruido
        velocity_threshold = self.config.get("velocity_threshold", 0.002)
        
        # Si movimiento lento/pausa
        if velocity < velocity_threshold:
            if self.stroke_active and len(self.points) >= self.config.get("min_points", 8):
                if self.last_significant_move_time:
                    time_since_move = current_time - self.last_significant_move_time
                    pause_threshold = self.config.get("pause_threshold_ms", 400)
                    if time_since_move > pause_threshold:
                        self.logger.debug(f"Trazo completado: {len(self.points)} puntos, pausa de {time_since_move:.0f}ms")
                        return True
            return False
        
        # Movimiento significativo detectado
        if self.last_point_time is None:
            # Iniciar nuevo trazo
            self.points = [(x, y)]
            self.last_point_time = current_time
            self.last_significant_move_time = current_time
            self.stroke_active = True
            self.stroke_start_time = current_time
            self.logger.debug("Nuevo trazo iniciado")
            return False
        
        # Agregar punto al trazo actual
        self.points.append((x, y))
        self.last_point_time = current_time
        self.last_significant_move_time = current_time
        
        # Verificar si el trazo es demasiado antiguo (timeout)
        max_age = self.config.get("max_stroke_age_ms", 3000)
        if self.stroke_start_time and (current_time - self.stroke_start_time) > max_age:
            if len(self.points) >= self.config.get("min_points", 8):
                self.logger.debug(f"Trazo finalizado por timeout: {len(self.points)} puntos")
                return True
        
        return False
    
    def get_stroke(self) -> Optional[List[Tuple[float, float]]]:
        """Retorna el trazo actual si tiene suficientes puntos."""
        min_points = self.config.get("min_points", 8)
        if len(self.points) >= min_points:
            return self.points.copy()
        return None
    
    def get_stroke_info(self) -> dict:
        """Retorna información sobre el trazo actual."""
        return {
            "points_count": len(self.points),
            "is_active": self.stroke_active,
            "age_ms": (time.time() * 1000 - self.stroke_start_time) if self.stroke_start_time else 0
        }
