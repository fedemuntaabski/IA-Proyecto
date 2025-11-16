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
    UI_CONFIG, LOGGING_CONFIG, DETECTION_CONFIG, PERFORMANCE_CONFIG
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
        
        # Inicializar componentes con manejo de errores
        self._init_components()
        
        # Validar setup con fallbacks
        if not self._validate_setup():
            self.logger.warning("Validación de setup incompleta - continuando con fallbacks")
        
        self.logger.info("Aplicación inicializada correctamente")
    
    def _init_components(self):
        """Inicializa componentes con manejo de errores y fallbacks."""
        try:
            # Combinar configuraciones para hand detector
            hand_config = {**MEDIAPIPE_CONFIG["hands"], **DETECTION_CONFIG, **PERFORMANCE_CONFIG}
            self.hand_detector = HandDetector(hand_config, self.logger)
        except Exception as e:
            self.logger.error(f"Error al inicializar HandDetector: {e}")
            self.hand_detector = None
        
        try:
            self.stroke_accumulator = StrokeAccumulator(STROKE_CONFIG, self.logger)
        except Exception as e:
            self.logger.error(f"Error al inicializar StrokeAccumulator: {e}")
            self.stroke_accumulator = None
        
        try:
            self.classifier = SketchClassifier(self.ia_dir, self.logger, demo_mode=MODEL_CONFIG["demo_mode"], config=MODEL_CONFIG)
        except Exception as e:
            self.logger.error(f"Error al inicializar SketchClassifier: {e}")
            # Fallback a modo demo
            self.classifier = SketchClassifier(self.ia_dir, self.logger, demo_mode=True, config=MODEL_CONFIG)
        
        try:
            input_shape = self.classifier.get_input_shape() if self.classifier else [28, 28, 1]
            self.preprocessor = DrawingPreprocessor(input_shape)
        except Exception as e:
            self.logger.error(f"Error al inicializar DrawingPreprocessor: {e}")
            self.preprocessor = None
        
        try:
            self.ui = PictionaryUI(UI_CONFIG)
        except Exception as e:
            self.logger.error(f"Error al inicializar PictionaryUI: {e}")
            self.ui = None
    
    def _validate_setup(self) -> bool:
        """Valida que todo está configurado correctamente, con fallbacks."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("VALIDACIÓN DE SETUP")
        self.logger.info("=" * 70)
        
        checks = {
            "IA folder": Path(self.ia_dir).exists(),
            "model_info.json": (Path(self.ia_dir) / "model_info.json").exists() if Path(self.ia_dir).exists() else False,
            "Hand Detector": self.hand_detector is not None and self.hand_detector.hands_detector is not None,
            "Classifier": self.classifier is not None and len(self.classifier.get_labels()) > 0,
            "Stroke Accumulator": self.stroke_accumulator is not None,
            "Preprocessor": self.preprocessor is not None,
            "UI": self.ui is not None,
        }
        
        all_ok = True
        for check_name, result in checks.items():
            status = "[OK]" if result else "[FAIL]"
            self.logger.info(f"  {status} {check_name}")
            if not result:
                all_ok = False
        
        # Validar cámara si no es dry-run
        if not self.dry_run:
            camera_ok = self._validate_camera()
            if not camera_ok:
                all_ok = False
        
        self.logger.info("=" * 70 + "\n")
        
        if not all_ok:
            self.logger.warning("Algunos componentes fallaron - la aplicación puede funcionar con limitaciones")
            self.logger.info("Componentes críticos: Hand Detector, Classifier, Camera")
        
        return all_ok
    
    def _validate_camera(self) -> bool:
        """Valida la cámara con reintentos."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                cap = cv2.VideoCapture(self.camera_id)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.logger.info(f"  [OK] Camara {self.camera_id} ({w}x{h})")
                    cap.release()
                    return True
                else:
                    self.logger.warning(f"  [FAIL] Camara {self.camera_id} - intento {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Esperar antes de reintentar
            except Exception as e:
                self.logger.error(f"  [FAIL] Error al validar camara {self.camera_id}: {e} - intento {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        self.logger.error(f"  [FAIL] Camara {self.camera_id} no disponible después de {max_retries} intentos")
        return False
    
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
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.logger.warning("No se pudo leer frame - intentando continuar")
                        time.sleep(0.1)  # Pequeña pausa antes de continuar
                        continue
                    
                    current_time = time.time()
                    
                    # Voltear frame (espejo)
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    
                    # Detectar mano con manejo de errores
                    try:
                        detection = self.hand_detector.detect(frame) if self.hand_detector else {"hand_landmarks": None, "hand_velocity": 0.0, "hands_count": 0}
                        hand_landmarks = detection["hand_landmarks"]
                        hand_velocity = detection["hand_velocity"]
                        hands_count = detection["hands_count"]
                    except Exception as e:
                        self.logger.warning(f"Error en detección de manos: {e} - continuando sin detección")
                        hand_landmarks = None
                        hand_velocity = 0.0
                        hands_count = 0
                    
                    # Dibujar landmarks si detector disponible
                    try:
                        if self.hand_detector:
                            frame = self.hand_detector.draw_hand_landmarks(frame)
                    except Exception as e:
                        self.logger.warning(f"Error al dibujar landmarks: {e}")
                    
                    # Procesar trazo si hay mano y componentes disponibles
                    if hand_landmarks and self.stroke_accumulator and self.preprocessor and self.classifier:
                        try:
                            index_pos = self.hand_detector.get_index_finger_position() if self.hand_detector else None
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
                                        if self.ui:
                                            self.ui.last_prediction = (label, conf, top3)
                                        
                                        # Log
                                        try:
                                            with open(inference_log, 'a', encoding='utf-8') as f:
                                                top3_str = "; ".join([f"{l}: {p:.1%}" for l, p in top3])
                                                f.write(f"{current_time:.0f} | {label} ({conf:.1%}) | Top-3: {top3_str}\n")
                                        except Exception as log_e:
                                            self.logger.warning(f"Error al escribir log: {log_e}")
                                    
                                    self.stroke_accumulator.reset()
                                
                                # Dibujar trazo en tiempo real
                                if self.stroke_accumulator and self.ui:
                                    stroke_info = self.stroke_accumulator.get_stroke_info()
                                    if stroke_info["is_active"]:
                                        frame = self.ui.draw_stroke_preview(frame, self.stroke_accumulator.points)
                        except Exception as e:
                            self.logger.warning(f"Error en procesamiento de trazo: {e} - continuando")
                    
                    elif self.stroke_accumulator and self.stroke_accumulator.stroke_active:
                        # Sin mano detectada, resetear trazo si está muy viejo
                        try:
                            stroke_info = self.stroke_accumulator.get_stroke_info()
                            if stroke_info["age_ms"] > STROKE_CONFIG.get("max_stroke_age_ms", 3000):
                                self.stroke_accumulator.reset()
                        except Exception as e:
                            self.logger.warning(f"Error al resetear trazo: {e}")
                    
                    # Actualizar UI
                    try:
                        if self.ui:
                            self.ui.update_fps(current_time)
                            frame = self.ui.render(
                                frame,
                                hand_detected=hand_landmarks is not None,
                                stroke_points=len(self.stroke_accumulator.points) if self.stroke_accumulator else 0,
                                hand_velocity=hand_velocity,
                                prediction=self.ui.last_prediction if self.ui else None
                            )
                    except Exception as e:
                        self.logger.warning(f"Error al renderizar UI: {e}")
                    
                    # Mostrar frame
                    cv2.imshow(UI_CONFIG["window_name"], frame)
                    
                    # Procesar input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Usuario presionó 'q' - saliendo")
                        self.running = False
                    elif key == ord('s'):
                        try:
                            self._save_screenshot(frame)
                        except Exception as e:
                            self.logger.warning(f"Error al guardar screenshot: {e}")
                    
                    # Verificar si la ventana fue cerrada con el botón X
                    try:
                        if cv2.getWindowProperty(UI_CONFIG["window_name"], cv2.WND_PROP_VISIBLE) < 1:
                            self.logger.info("Usuario cerró la ventana - saliendo")
                            self.running = False
                    except Exception as e:
                        self.logger.warning(f"Error al verificar ventana: {e}")
                
                except Exception as frame_e:
                    self.logger.error(f"Error en procesamiento de frame: {frame_e} - continuando")
                    time.sleep(0.1)  # Pausa para evitar loop infinito en caso de error persistente
        
        except KeyboardInterrupt:
            self.logger.info("Interrupción por usuario (Ctrl+C)")
        except Exception as run_e:
            self.logger.error(f"Error fatal en run loop: {run_e}")
        finally:
            self._cleanup()
    
    def _init_camera(self) -> bool:
        """Inicializa la cámara con configuración óptima y reintentos."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    self.logger.warning(f"Cámara {self.camera_id} no se abrió - intento {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Esperar más tiempo entre reintentos
                        continue
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
                self.logger.error(f"Error al inicializar cámara {self.camera_id}: {e} - intento {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        self.logger.error(f"No se pudo inicializar la cámara {self.camera_id} después de {max_retries} intentos")
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
