"""
app.py - Aplicación principal de Pictionary Live
"""

import cv2
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from config import (
    MEDIAPIPE_CONFIG, CAMERA_CONFIG, STROKE_CONFIG, MODEL_CONFIG,
    UI_CONFIG, LOGGING_CONFIG, DETECTION_CONFIG
)
from hand_detector import HandDetector
from stroke_manager import StrokeAccumulator
from drawing_preprocessor import DrawingPreprocessor
from model import SketchClassifier
from ui import PictionaryUI
from logger_setup import setup_logging


class PictionaryLive:
    """Aplicación principal de Pictionary en vivo."""
    
    def __init__(self, ia_dir: str = "./IA", camera_id: int = 0, debug: bool = False, dry_run: bool = False):
        """
        Inicializa Pictionary Live.
        
        Args:
            ia_dir: Ruta a la carpeta con modelo
            camera_id: ID de cámara a usar
            debug: Habilitar logging DEBUG
            dry_run: Solo validar sin ejecutar
        """
        self.logger = setup_logging(debug=debug)
        self.logger.info("=" * 70)
        self.logger.info("PICTIONARY LIVE - Inicializando")
        self.logger.info("=" * 70)
        
        self.ia_dir = ia_dir
        self.camera_id = camera_id
        self.debug = debug
        self.dry_run = dry_run
        self.running = False
        self.cap = None
        
        # Inicializar componentes
        self.hand_detector = HandDetector(MEDIAPIPE_CONFIG["hands"], self.logger)
        self.stroke_accumulator = StrokeAccumulator(STROKE_CONFIG, self.logger)
        self.classifier = SketchClassifier(ia_dir, self.logger, demo_mode=MODEL_CONFIG["demo_mode"])
        self.preprocessor = DrawingPreprocessor(self.classifier.get_input_shape())
        self.ui = PictionaryUI(UI_CONFIG)
        
        # Validar setup
        if not self._validate_setup():
            raise RuntimeError("Validación de setup fallida")
        
        self.logger.info("Aplicación inicializada correctamente")
    
    def _validate_setup(self) -> bool:
        """Valida que todo está configurado correctamente."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("VALIDACIÓN DE SETUP")
        self.logger.info("=" * 70)
        
        checks = {
            "IA folder": Path(self.ia_dir).exists(),
            "model_info.json": (Path(self.ia_dir) / "model_info.json").exists(),
            "Hand Detector": self.hand_detector.hands_detector is not None,
            "Classifier": len(self.classifier.get_labels()) > 0,
        }
        
        for check_name, result in checks.items():
            status = "[OK]" if result else "[FAIL]"
            self.logger.info(f"  {status} {check_name}")
        
        # Validar cámara si no es dry-run
        if not self.dry_run:
            try:
                cap = cv2.VideoCapture(self.camera_id)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.logger.info(f"  [OK] Camara {self.camera_id} ({w}x{h})")
                    cap.release()
                else:
                    self.logger.warning(f"  [FAIL] Camara {self.camera_id}")
                    return False
            except Exception as e:
                self.logger.error(f"  [FAIL] Error al validar camara: {e}")
                return False
        
        self.logger.info("=" * 70 + "\n")
        return all(checks.values()) and self.hand_detector.hands_detector is not None
    
    def run_dry_run(self):
        """Modo de validación sin abrir cámara."""
        self.logger.info("\n[DRY RUN] Informacion del modelo:")
        self.logger.info(f"  Input shape: {self.classifier.get_input_shape()}")
        self.logger.info(f"  Total de clases: {len(self.classifier.get_labels())}")
        labels = self.classifier.get_labels()
        if labels:
            self.logger.info(f"  Primeras 10 clases: {labels[:10]}")
        self.logger.info("[DRY RUN] Setup validado correctamente - Listo para usar\n")
    
    def run(self):
        """Ejecuta la aplicación principal."""
        if self.dry_run:
            self.run_dry_run()
            return
        
        if not self._init_camera():
            self.logger.error("No se pudo inicializar la cámara")
            return
        
        self.logger.info("[VIVO] Iniciando captura. Presiona 'q' para salir, 's' para guardar.")
        self.running = True
        
        # Crear ventana
        cv2.namedWindow(UI_CONFIG["window_name"], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(UI_CONFIG["window_name"], UI_CONFIG["window_width"], UI_CONFIG["window_height"])
        
        inference_log = Path(LOGGING_CONFIG["inference_log_file"])
        last_frame_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("No se pudo leer frame")
                    break
                
                current_time = time.time()
                
                # Voltear frame (espejo)
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Detectar mano
                detection = self.hand_detector.detect(frame)
                hand_landmarks = detection["hand_landmarks"]
                hand_velocity = detection["hand_velocity"]
                hands_count = detection["hands_count"]
                
                # Dibujar landmarks
                frame = self.hand_detector.draw_hand_landmarks(frame)
                
                # Procesar trazo si hay mano
                if hand_landmarks:
                    index_pos = self.hand_detector.get_index_finger_position()
                    if index_pos:
                        stroke_complete = self.stroke_accumulator.add_point(
                            index_pos[0], index_pos[1], hand_velocity
                        )
                        
                        # Si trazo completado, realizar inferencia
                        if stroke_complete:
                            stroke = self.stroke_accumulator.get_stroke()
                            if stroke:
                                drawing = self.preprocessor.preprocess(stroke)
                                label, conf, top3 = self.classifier.predict(drawing)
                                
                                self.logger.info(f"Predicción: {label} ({conf:.1%})")
                                self.ui.last_prediction = (label, conf, top3)
                                
                                # Log
                                try:
                                    with open(inference_log, 'a', encoding='utf-8') as f:
                                        top3_str = "; ".join([f"{l}: {p:.1%}" for l, p in top3])
                                        f.write(f"{current_time:.0f} | {label} ({conf:.1%}) | Top-3: {top3_str}\n")
                                except:
                                    pass
                            
                            self.stroke_accumulator.reset()
                        
                        # Dibujar trazo en tiempo real
                        stroke_info = self.stroke_accumulator.get_stroke_info()
                        if stroke_info["is_active"]:
                            frame = self.ui.draw_stroke_preview(frame, self.stroke_accumulator.points)
                
                else:
                    # Sin mano detectada, resetear trazo si está muy viejo
                    if self.stroke_accumulator.stroke_active:
                        stroke_info = self.stroke_accumulator.get_stroke_info()
                        if stroke_info["age_ms"] > STROKE_CONFIG.get("max_stroke_age_ms", 3000):
                            self.stroke_accumulator.reset()
                
                # Actualizar UI
                self.ui.update_fps(current_time)
                frame = self.ui.render(
                    frame,
                    hand_detected=hand_landmarks is not None,
                    stroke_points=len(self.stroke_accumulator.points),
                    hand_velocity=hand_velocity,
                    prediction=self.ui.last_prediction
                )
                
                # Mostrar frame
                cv2.imshow(UI_CONFIG["window_name"], frame)
                
                # Procesar input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Usuario presionó 'q' - saliendo")
                    self.running = False
                elif key == ord('s'):
                    self._save_screenshot(frame)
        
        except KeyboardInterrupt:
            self.logger.info("Interrupción por usuario (Ctrl+C)")
        finally:
            self._cleanup()
    
    def _init_camera(self) -> bool:
        """Inicializa la cámara con configuración óptima."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return False
            
            # Configurar cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["height"])
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG["buffer_size"])
            
            # Obtener parámetros reales
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"Cámara inicializada: {w}x{h} @ {fps}FPS")
            return True
        except Exception as e:
            self.logger.error(f"Error al inicializar cámara: {e}")
            return False
    
    def _save_screenshot(self, frame):
        """Guarda una captura de pantalla."""
        pred_dir = Path("./predictions")
        pred_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = pred_dir / f"frame_{timestamp}.png"
        cv2.imwrite(str(path), frame)
        self.logger.info(f"[OK] Captura guardada: {path.name}")
    
    def _cleanup(self):
        """Limpia recursos."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("\nAplicación cerrada")
