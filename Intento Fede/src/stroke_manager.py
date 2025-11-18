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
        # Contador para detectar múltiples frames de baja velocidad consecutivos
        self.low_vel_count = 0
        # Por compatibilidad, si no se especifica en config usar 1 (comportamiento previo)
        self.low_vel_required = int(self.config.get("low_vel_consecutive", 1)) if isinstance(config, dict) else 1
        self.reset()
    
    def reset(self):
        """Reinicia el acumulador."""
        self.points = []
        self.last_point_time = None
        self.last_significant_move_time = None
        self.stroke_active = False
        self.stroke_start_time = None
        self.low_vel_count = 0
    
    def add_point(self, x: float, y: float, velocity: float = 0.0) -> bool:
        """
        Agrega un punto al trazo.
        
        Args:
            x, y: Coordenadas normalizadas (0-1)
            velocity: Velocidad del movimiento
        
        Returns:
            True si se completó el trazo (pausa detectada)
        """
        try:
            current_time = time.time() * 1000  # ms
            
            # Umbral de velocidad para ignorar ruido
            velocity_threshold = self.config.get("velocity_threshold", 0.01)
            # Umbral mínimo de desplazamiento para considerar movimiento aunque la velocidad sea baja
            min_movement_delta = self.config.get("min_movement_delta", 0.003)

            # Si movimiento lento/pausa
            if velocity < velocity_threshold:
                # Opcional: permitir registrar pequeños desplazamientos aun con velocidad baja
                allow_append = bool(self.config.get("allow_low_velocity_append", False))
                if allow_append and self.points:
                    last = self.points[-1]
                    dx = x - last[0]
                    dy = y - last[1]
                    dist = (dx*dx + dy*dy) ** 0.5
                    if dist > min_movement_delta:
                        # Considerar como movimiento significativo aunque la velocidad medida sea baja
                        self.points.append((x, y))
                        self.last_point_time = current_time
                        self.last_significant_move_time = current_time
                        self.low_vel_count = 0
                        return False

                # No hubo desplazamiento suficiente o no está habilitado -> aumentar contador de frames lentos
                self.low_vel_count += 1
                low_required = int(self.config.get("low_vel_consecutive", self.low_vel_required))
                if self.stroke_active and len(self.points) >= self.config.get("min_points", 8):
                    if self.last_significant_move_time:
                        time_since_move = current_time - self.last_significant_move_time
                        pause_threshold = self.config.get("pause_threshold_ms", 400)
                        # Requerir N frames lentos además del umbral temporal
                        if time_since_move > pause_threshold and self.low_vel_count >= low_required:
                            self.logger.debug(f"Trazo completado: {len(self.points)} puntos, pausa de {time_since_move:.0f}ms (con {self.low_vel_count} frames lentos)")
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
                # reset contador de baja velocidad
                self.low_vel_count = 0
                self.logger.debug("Nuevo trazo iniciado")
                return False
            
            # Agregar punto al trazo actual
            self.points.append((x, y))
            self.last_point_time = current_time
            self.last_significant_move_time = current_time
            # reset contador de baja velocidad
            self.low_vel_count = 0
            
            # Verificar si el trazo es demasiado antiguo (timeout) usando tiempo desde el último movimiento
            max_age = self.config.get("max_stroke_age_ms", 3000)
            if self.stroke_start_time:
                last_move_time = self.last_significant_move_time or self.stroke_start_time
                if (current_time - last_move_time) > max_age:
                    if len(self.points) >= self.config.get("min_points", 8):
                        self.logger.debug(f"Trazo finalizado por timeout (inactividad): {len(self.points)} puntos")
                        return True
            
            return False
        except Exception as e:
            self.logger.warning(f"Error en add_point: {e} - reseteando trazo")
            self.reset()
            return False
    
    def get_stroke(self) -> Optional[List[Tuple[float, float]]]:
        """Retorna el trazo actual si tiene suficientes puntos y calidad."""
        min_points = self.config.get("min_points", 8)
        if len(self.points) >= min_points and self._is_stroke_valid():
            return self.points.copy()
        return None
    
    def get_stroke_info(self) -> dict:
        """Retorna información sobre el trazo actual."""
        return {
            "points_count": len(self.points),
            "is_active": self.stroke_active,
            "age_ms": (time.time() * 1000 - self.stroke_start_time) if self.stroke_start_time else 0,
            "is_valid": self._is_stroke_valid() if len(self.points) >= self.config.get("min_points", 8) else False
        }
    
    def _is_stroke_valid(self) -> bool:
        """Valida la calidad del trazo actual."""
        if len(self.points) < 2:
            return False
        
        # Calcular longitud total
        total_length = 0.0
        for i in range(1, len(self.points)):
            dx = self.points[i][0] - self.points[i-1][0]
            dy = self.points[i][1] - self.points[i-1][1]
            total_length += (dx*dx + dy*dy) ** 0.5
        
        # Longitud mínima
        min_length = self.config.get("min_stroke_length", 0.05)
        if total_length < min_length:
            return False
        
        # Calcular varianza para detectar ruido
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        var_x = sum((x - sum(xs)/len(xs))**2 for x in xs) / len(xs)
        var_y = sum((y - sum(ys)/len(ys))**2 for y in ys) / len(ys)
        
        # Si la varianza es muy baja, es un punto casi estático
        if var_x + var_y < 0.0001:
            return False
        
        return True
