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
        prediction: Optional[Tuple[str, float, List[Tuple[str, float]]]] = None,
        hand_in_fist: bool = False,
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
            frame, "PICTIONARY LIVE", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3
        )
        
        # Información de diagnostico (separada en líneas)
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame, fps_text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2
        )
        
        mano_text = f"Mano: {'SI' if hand_detected else 'NO'}"
        cv2.putText(
            frame, mano_text, (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2
        )
        
        puntos_text = f"Puntos: {stroke_points}"
        cv2.putText(
            frame, puntos_text, (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2
        )
        
        # Línea divisoria
        cv2.line(frame, (0, 125), (w, 125), (255, 255, 255), 1)
        
        # Predicción (si existe)
        if self.last_prediction:
            label, conf, top3 = self.last_prediction
            pred_text = f"{label}: {conf:.1%}"
            
            # Panel de predicción a la derecha (mínimo x=400)
            text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            x_pos = max(w - text_size[0] - 30, 400)
            
            cv2.rectangle(frame, (x_pos - 10, 15), (w - 10, 115), (50, 100, 150), -1)
            cv2.rectangle(frame, (x_pos - 10, 15), (w - 10, 115), (0, 200, 255), 2)
            
            cv2.putText(frame, "Prediccion:", (x_pos, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(frame, pred_text, (x_pos, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Controles abajo
        controls = "q=SALIR | s=GUARDAR | Dibuja en el aire con el dedo indice"
        cv2.putText(
            frame, controls, (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        # Indicador visual cuando se detecta puño (dibujo pausado)
        if hand_in_fist:
            fist_text = "PUÑO: DIBUJO EN PAUSA"
            # Dibujar fondo semitransparente en rojo en la esquina superior derecha
            text_size = cv2.getTextSize(fist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            x_pos = max(w - text_size[0] - 30, 420)
            cv2.rectangle(frame, (x_pos - 10, 120), (w - 10, 160), (0, 0, 200), -1)
            cv2.putText(frame, fist_text, (x_pos, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top-3 predicciones (abajo a la izquierda)
        if self.last_prediction and len(self.last_prediction[2]) > 0:
            top3_y = h - 105
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
