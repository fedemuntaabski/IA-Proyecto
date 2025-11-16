"""
ui.py - Interfaz gráfica y overlay
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class PictionaryUI:
    """Renderiza la interfaz gráfica de Pictionary Live."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa la UI.
        
        Args:
            config: Configuración de UI
        """
        self.config = config
        self.frame_count = 0
        self.last_time = 0
        self.fps = 0.0
        self.last_prediction = None
    
    def update_fps(self, current_time: float):
        """Actualiza el contador de FPS."""
        self.frame_count += 1
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
    
    def render(
        self,
        frame: np.ndarray,
        hand_detected: bool,
        stroke_points: int,
        hand_velocity: float,
        prediction: Optional[Tuple[str, float, List[Tuple[str, float]]]] = None
    ) -> np.ndarray:
        """
        Renderiza la UI en el frame.
        
        Args:
            frame: Frame para renderizar
            hand_detected: Si se detectó mano
            stroke_points: Número de puntos en el trazo actual
            hand_velocity: Velocidad actual de la mano
            prediction: (label, confianza, top3) o None
        
        Returns:
            Frame con UI renderizada
        """
        h, w = frame.shape[:2]
        
        # Actualizar predicción si existe
        if prediction:
            self.last_prediction = prediction
        
        # Panel superior semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Título
        cv2.putText(
            frame, "PICTIONARY LIVE", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
        )
        
        # Información de diagnostico
        diag_text = f"FPS: {self.fps:.1f} | Mano: {'✓' if hand_detected else '✗'} | Puntos: {stroke_points}"
        cv2.putText(
            frame, diag_text, (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2
        )
        
        # Velocidad de la mano
        if hand_detected and hand_velocity > 0:
            vel_text = f"Velocidad: {hand_velocity:.4f}"
            cv2.putText(
                frame, vel_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2
            )
        
        # Predicción (si existe)
        if self.last_prediction:
            label, conf, top3 = self.last_prediction
            pred_text = f"{label}: {conf:.1%}"
            
            # Panel de predicción a la derecha
            text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            x_pos = w - text_size[0] - 30
            
            cv2.rectangle(frame, (x_pos - 10, 15), (w - 10, 110), (50, 100, 150), -1)
            cv2.rectangle(frame, (x_pos - 10, 15), (w - 10, 110), (0, 200, 255), 2)
            
            cv2.putText(frame, "Predicción:", (x_pos, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(frame, pred_text, (x_pos, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Controles abajo
        controls = "q=SALIR | s=GUARDAR | Dibuja en el aire con el dedo indice"
        cv2.putText(
            frame, controls, (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )
        
        # Top-3 predicciones (abajo a la izquierda)
        if self.last_prediction and len(self.last_prediction[2]) > 0:
            top3_y = h - 100
            cv2.putText(frame, "Top 3:", (20, top3_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
            for i, (label, conf) in enumerate(self.last_prediction[2][:3]):
                y_offset = top3_y + 25 + (i * 22)
                cv2.putText(frame, f"{i+1}. {label}: {conf:.1%}", (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 150, 255), 1)
        
        return frame
    
    def draw_stroke_preview(self, frame: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
        """Dibuja vista previa del trazo en tiempo real."""
        if not points or len(points) < 2:
            return frame
        
        h, w = frame.shape[:2]
        pixel_points = [
            (int(p[0] * w), int(p[1] * h))
            for p in points
        ]
        
        # Dibujar línea de trazo
        for i in range(1, len(pixel_points)):
            cv2.line(frame, pixel_points[i - 1], pixel_points[i], (0, 0, 255), 3)
        
        return frame
