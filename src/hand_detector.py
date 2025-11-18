"""
hand_detector.py - Detección de manos usando MediaPipe
"""

import cv2
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class HandDetector:
    """Detecta manos usando MediaPipe con optimización para performance."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Inicializa el detector de manos.
        
        Args:
            config: Configuración de MediaPipe para manos
            logger: Logger para debug
        """
        self.logger = logger
        self.config = config
        self.hands_detector = None
        self.hand_landmarks = None
        self.previous_hand_pos = None
        self.hand_velocity = 0.0
        
        # Configuración de optimización
        self.processing_resolution = self.config.get("processing_resolution", 320)
        self.async_processing = self.config.get("async_processing", True)
        
        try:
            if MEDIAPIPE_AVAILABLE:
                self._init_mediapipe()
            else:
                self.logger.warning("MediaPipe no disponible - funcionando en modo sin detección")
        except Exception as e:
            self.logger.error(f"Error al inicializar HandDetector: {e} - funcionando en modo sin detección")
            self.hands_detector = None
    
    def _init_mediapipe(self):
        """Inicializa MediaPipe Hands."""
        try:
            mp_hands = mp.solutions.hands
            self.hands_detector = mp_hands.Hands(
                static_image_mode=self.config.get("static_image_mode", False),
                max_num_hands=1,  # Solo una mano
                min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5),
            )
            self.logger.info("MediaPipe Hands inicializado")
            self.logger.info(f"  Thresholds: detection={self.config['min_detection_confidence']}, tracking={self.config['min_tracking_confidence']}")
        except Exception as e:
            self.logger.warning(f"Error inicializando MediaPipe: {e}")
            self.hands_detector = None
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detecta manos en el frame.
        
        Args:
            frame: Frame BGR de OpenCV
        
        Returns:
            Dict con landmarks, confianza y velocidad
        """
        if self.hands_detector is None:
            return {
                "hand_landmarks": None,
                "hand_confidence": 0.0,
                "hand_velocity": 0.0,
                "hands_count": 0
            }
        
        try:
            # Optimizar resolución para mejor performance
            h, w = frame.shape[:2]
            if h > self.processing_resolution:
                scale_factor = self.processing_resolution / h
                new_w = int(w * scale_factor)
                processing_frame = cv2.resize(frame, (new_w, self.processing_resolution))
            else:
                processing_frame = frame
                scale_factor = 1.0
            
            frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(frame_rgb)
            
            hand_landmarks = None
            hand_confidence = 0.0
            hand_velocity = 0.0
            hands_count = 0
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                hands_count = len(results.multi_hand_landmarks)
                # Usar la primera mano detectada
                raw_landmarks = results.multi_hand_landmarks[0].landmark
                
                # Escalar landmarks de vuelta al tamaño original
                hand_landmarks = [
                    (lm.x, lm.y) 
                    for lm in raw_landmarks
                ]
                
                # Usar confianza de detección
                if results.multi_handedness and len(results.multi_handedness) > 0:
                    try:
                        # Intentar acceso directo (versiones antiguas)
                        if hasattr(results.multi_handedness[0], 'score'):
                            hand_confidence = results.multi_handedness[0].score
                        # Intentar acceso anidado (versiones nuevas)
                        elif hasattr(results.multi_handedness[0], 'classification'):
                            hand_confidence = results.multi_handedness[0].classification.score
                        else:
                            hand_confidence = 0.0
                    except (AttributeError, IndexError) as e:
                        self.logger.warning(f"No se pudo obtener confianza de mano: {e}")
                        hand_confidence = 0.0
                
                # Calcular velocidad del dedo índice (punto 8)
                if len(hand_landmarks) > 8:
                    current_pos = hand_landmarks[8]
                    if self.previous_hand_pos:
                        dx = current_pos[0] - self.previous_hand_pos[0]
                        dy = current_pos[1] - self.previous_hand_pos[1]
                        hand_velocity = (dx ** 2 + dy ** 2) ** 0.5
                    self.previous_hand_pos = current_pos
                
                self.logger.info(
                    f"Mano(s) detectada(s): {hands_count}, "
                    f"confianza: {hand_confidence:.2f}, velocidad: {hand_velocity:.4f}"
                )
            else:
                self.logger.info("No se detectó mano en este frame")
                self.previous_hand_pos = None
            
            self.hand_landmarks = hand_landmarks
            self.hand_velocity = hand_velocity
            
            return {
                "hand_landmarks": hand_landmarks,
                "hand_confidence": hand_confidence,
                "hand_velocity": hand_velocity,
                "hands_count": hands_count
            }
        
        except Exception as e:
            self.logger.error(f"Error en detección de manos: {e}", exc_info=False)
            self.previous_hand_pos = None
            return {
                "hand_landmarks": None,
                "hand_confidence": 0.0,
                "hand_velocity": 0.0,
                "hands_count": 0
            }
    
    def draw_hand_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja los landmarks de la mano en el frame."""
        if not self.hand_landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        # Conexiones de mano (dedos)
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),           # Índice
            (0, 9), (9, 10), (10, 11), (11, 12),      # Corazón
            (0, 13), (13, 14), (14, 15), (15, 16),    # Anular
            (0, 17), (17, 18), (18, 19), (19, 20),    # Meñique
        ]
        
        # Dibujar conexiones
        for start, end in hand_connections:
            if start < len(self.hand_landmarks) and end < len(self.hand_landmarks):
                x1, y1 = self.hand_landmarks[start]
                x2, y2 = self.hand_landmarks[end]
                cv2.line(
                    frame,
                    (int(x1 * w), int(y1 * h)),
                    (int(x2 * w), int(y2 * h)),
                    (255, 0, 150),  # Magenta
                    2
                )
        
        # Dibujar puntos
        for i, (x, y) in enumerate(self.hand_landmarks):
            px, py = int(x * w), int(y * h)
            # Dedo índice (punto 8) en rojo brillante
            if i == 8:
                cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)
            else:
                cv2.circle(frame, (px, py), 4, (255, 0, 255), -1)
        
        return frame
    
    def get_index_finger_position(self) -> Optional[Tuple[float, float]]:
        """Retorna la posición del dedo índice (punto 8)."""
        if self.hand_landmarks and len(self.hand_landmarks) > 8:
            return self.hand_landmarks[8]
        return None

    def is_fist(self) -> bool:
        """
        Heurística simple para detectar puño cerrado usando landmarks.
        Retorna True si la mayoría de las puntas de los dedos (index, middle, ring, pinky)
        están plegadas respecto a sus articulaciones PIP.
        """
        if not self.hand_landmarks or len(self.hand_landmarks) < 21:
            return False

        # Índices de landmarks
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        folded = 0
        for tip_idx, pip_idx in zip(tips, pips):
            tip = self.hand_landmarks[tip_idx]
            pip = self.hand_landmarks[pip_idx]
            # Si la punta está más cerca del centro de la palma (mayor y en coordenada normalizada)
            # en la mayoría de las orientaciones tip.y > pip.y suele indicar dedo plegado (dependiendo de la orientación de la mano)
            try:
                if tip[1] > pip[1]:
                    folded += 1
            except Exception:
                continue

        # Considerar puño si 3 o más dedos plegados
        return folded >= 3
