"""
game_integration.py - Integración de la cámara con el modo de juego

Conecta la captura de video, el preprocessor, el modelo y la UI de juego
en un flujo coherente.
"""

import cv2
import logging
import threading
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

from game_mode import GameMode, GameConfig, GameState
from hand_detector import HandDetector
from stroke_manager import StrokeAccumulator
from drawing_preprocessor import DrawingPreprocessor
from model import SketchClassifier


@dataclass
class IntegrationConfig:
    """Configuración de integración cámara-juego."""
    camera_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    ia_dir: str = "./IA"
    debug: bool = False


class GameIntegration:
    """
    Integra todos los componentes (cámara, detección, modelo, UI juego)
    en una aplicación coherente.
    """
    
    def __init__(
        self,
        hand_detector: HandDetector,
        stroke_accumulator: StrokeAccumulator,
        preprocessor: DrawingPreprocessor,
        classifier: SketchClassifier,
        game_config: Optional[GameConfig] = None,
        integration_config: Optional[IntegrationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Inicializa la integración.
        
        Args:
            hand_detector: Detector de manos
            stroke_accumulator: Acumulador de trazos
            preprocessor: Preprocesador de dibujos
            classifier: Clasificador de sketches
            game_config: Config del juego (GameConfig)
            integration_config: Config de integración (IntegrationConfig)
            logger: Logger
        """
        self.hand_detector = hand_detector
        self.stroke_accumulator = stroke_accumulator
        self.preprocessor = preprocessor
        self.classifier = classifier
        
        self.game_config = game_config or GameConfig()
        self.integration_config = integration_config or IntegrationConfig()
        self.logger = logger or self._setup_default_logger()
        
        # Inicializar game mode
        labels = self.classifier.get_labels() if self.classifier else ["demo"]
        
        # Callback de predicción
        def predict_fn(drawing):
            return self.classifier.predict(drawing) if drawing is not None else ("demo", 0.5, [])
        
        self.game_mode = GameMode(
            labels=labels,
            predict_callback=predict_fn,
            config=self.game_config,
            logger=self.logger,
            clear_callback=self.clear_drawing,
        )
        
        # Estado de captura
        self.camera = None
        self.capturing = False
        self.capture_thread = None
        
        # Dibujo actual
        self.drawing_strokes = []
        self.hand_in_fist = False
        
        self.logger.info("GameIntegration inicializada")
    
    def _setup_default_logger(self) -> logging.Logger:
        """Crea un logger por defecto."""
        logger = logging.getLogger("GameIntegration")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _init_camera(self) -> bool:
        """Inicializa la cámara."""
        try:
            self.camera = cv2.VideoCapture(self.integration_config.camera_id)
            
            if not self.camera.isOpened():
                self.logger.error("No se pudo abrir la cámara")
                return False
            
            # Configurar cámara
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.integration_config.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.integration_config.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.integration_config.camera_fps)
            
            w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"Cámara inicializada: {w}x{h} @ {fps}FPS")
            return True
        
        except Exception as e:
            self.logger.error(f"Error inicializando cámara: {e}")
            return False
    
    def _capture_loop(self):
        """Loop principal de captura de video (ejecutado en thread)."""
        self.logger.info("Iniciando loop de captura")
        
        try:
            while self.capturing:
                try:
                    ret, frame = self.camera.read()
                    if not ret:
                        self.logger.warning("Error leyendo frame")
                        time.sleep(0.1)
                        continue
                    
                    # Voltear frame (espejo)
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    
                    # Detectar mano
                    try:
                        detection = self.hand_detector.detect(frame)
                        hand_landmarks = detection["hand_landmarks"]
                        hand_velocity = detection["hand_velocity"]
                    except Exception as e:
                        self.logger.debug(f"Error detectando mano: {e}")
                        hand_landmarks = None
                        hand_velocity = 0.0
                    
                    # Dibujar landmarks
                    try:
                        if self.hand_detector and hand_landmarks:
                            frame = self.hand_detector.draw_hand_landmarks(frame)
                    except Exception as e:
                        self.logger.debug(f"Error dibujando landmarks: {e}")
                    
                    # Procesar trazo si hay mano
                    is_fist = False
                    try:
                        if self.hand_detector and hasattr(self.hand_detector, 'is_fist'):
                            is_fist = self.hand_detector.is_fist()
                    except Exception:
                        pass
                    
                    # Actualizar game state según si hay mano
                    if hand_landmarks:
                        if self.game_mode.game_state == GameState.WAITING_FOR_DRAW:
                            self.game_mode.game_state = GameState.DRAWING
                            self.game_mode._update_state_label()
                    else:
                        if self.game_mode.game_state == GameState.DRAWING:
                            self.game_mode.game_state = GameState.WAITING_FOR_DRAW
                            self.game_mode._update_state_label()
                    
                    # Procesar trazos
                    if hand_landmarks and not is_fist:
                        try:
                            index_pos = self.hand_detector.get_index_finger_position()
                            if index_pos:
                                stroke_complete = self.stroke_accumulator.add_point(
                                    index_pos[0], index_pos[1], hand_velocity
                                )
                                
                                if stroke_complete:
                                    stroke = self.stroke_accumulator.get_stroke()
                                    if stroke:
                                        self.drawing_strokes.append(stroke)
                                    self.stroke_accumulator.reset()
                        except Exception as e:
                            self.logger.debug(f"Error procesando trazo: {e}")
                    elif is_fist:
                        # Completar trazo si se cierra la mano
                        if self.stroke_accumulator and self.stroke_accumulator.stroke_active:
                            stroke = self.stroke_accumulator.get_stroke()
                            if stroke:
                                self.drawing_strokes.append(stroke)
                            self.stroke_accumulator.reset()
                    else:
                        # Sin mano, resetear
                        if self.stroke_accumulator and self.stroke_accumulator.stroke_active:
                            try:
                                stroke_info = self.stroke_accumulator.get_stroke_info()
                                if stroke_info["age_ms"] > 3000:
                                    self.stroke_accumulator.reset()
                            except Exception:
                                pass
                    
                    # Dibujar trazos previos
                    if self.drawing_strokes:
                        for stroke in self.drawing_strokes:
                            if not stroke or len(stroke) < 2:
                                continue
                            pixel_points = [(int(p[0] * w), int(p[1] * h)) for p in stroke]
                            for i in range(1, len(pixel_points)):
                                cv2.line(frame, pixel_points[i-1], pixel_points[i], (0, 255, 255), 2)
                    
                    # Dibujar trazo activo
                    try:
                        if self.stroke_accumulator and self.stroke_accumulator.stroke_active:
                            stroke_info = self.stroke_accumulator.get_stroke_info()
                            if stroke_info["is_active"]:
                                for i in range(1, len(self.stroke_accumulator.points)):
                                    p1 = (int(self.stroke_accumulator.points[i-1][0] * w), 
                                          int(self.stroke_accumulator.points[i-1][1] * h))
                                    p2 = (int(self.stroke_accumulator.points[i][0] * w), 
                                          int(self.stroke_accumulator.points[i][1] * h))
                                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    except Exception as e:
                        self.logger.debug(f"Error dibujando trazo activo: {e}")
                    
                    # Actualizar game mode UI con el frame
                    try:
                        self.game_mode.update_frame(frame)
                    except Exception as e:
                        self.logger.debug(f"Error actualizando frame en game_mode: {e}")
                    
                    # Control de FPS
                    time.sleep(0.01)  # ~100 fps máximo
                
                except Exception as frame_e:
                    self.logger.error(f"Error en capture loop: {frame_e}")
                    time.sleep(0.1)
        
        except Exception as e:
            self.logger.error(f"Error crítico en capture loop: {e}")
        finally:
            if self.camera:
                self.camera.release()
            self.logger.info("Capture loop finalizado")
    
    def predict_current_drawing(self) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predice el dibujo actual acumulado.
        
        Returns:
            (label, confianza, top3)
        """
        try:
            if not self.drawing_strokes:
                self.logger.warning("No hay trazos para predecir")
                return ("vacío", 0.0, [])
            
            # Combinar todos los trazos
            combined = []
            for stroke in self.drawing_strokes:
                combined.extend(stroke)
            
            if not combined:
                return ("vacío", 0.0, [])
            
            # Preprocesar
            drawing = self.preprocessor.preprocess(combined)
            
            # Predecir
            label, conf, top3 = self.classifier.predict(drawing)
            
            self.logger.info(f"Predicción: {label} ({conf:.1%})")
            
            # Limpiar dibujo
            self.drawing_strokes = []
            
            return label, conf, top3
        
        except Exception as e:
            self.logger.error(f"Error prediciendo dibujo: {e}")
            return ("error", 0.0, [])
    
    def clear_drawing(self):
        """Limpia el dibujo actual (trazos acumulados)."""
        self.drawing_strokes = []
        if self.stroke_accumulator:
            self.stroke_accumulator.reset()
        self.logger.info("Dibujo limpiado")
    
    def run(self):
        """Inicia la aplicación integrada."""
        # Inicializar cámara
        if not self._init_camera():
            self.logger.error("No se pudo inicializar cámara")
            return
        
        # Actualizar callback de predicción en game_mode
        self.game_mode.predict_callback = self.predict_current_drawing
        
        # Iniciar thread de captura
        self.capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Iniciar UI (bloqueante)
        try:
            self.game_mode.run()
        except KeyboardInterrupt:
            self.logger.info("Interrupción por usuario")
        finally:
            self.capturing = False
            if self.capture_thread:
                self.capture_thread.join(timeout=5)
            if self.camera:
                self.camera.release()
            self.logger.info("Aplicación finalizada")
