"""
app_pyqt.py - Aplicación principal con PyQt6

Integra la UI moderna PyQt6 con la lógica de detección de mano,
procesamiento de trazos y predicciones de IA. Optimizado para rendimiento.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
import cv2
import numpy as np

from config import (
    MEDIAPIPE_CONFIG, CAMERA_CONFIG, STROKE_CONFIG, MODEL_CONFIG,
    UI_CONFIG, DETECTION_CONFIG, PERFORMANCE_CONFIG, PREPROCESSING_CONFIG
)
from hand_detector import HandDetector
from stroke_manager import StrokeAccumulator
from drawing_preprocessor import DrawingPreprocessor
from model import SketchClassifier
from ui_pyqt import PictionaryUIQt
from logger_setup import setup_logging


class VideoThread(QThread):
    """Thread para captura y procesamiento de video."""
    
    frame_ready = pyqtSignal(np.ndarray)
    hand_detected = pyqtSignal(bool)
    stroke_updated = pyqtSignal(list)
    prediction_ready = pyqtSignal(str, float, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_id: int, components: dict, logger: logging.Logger):
        super().__init__()
        self.camera_id = camera_id
        self.components = components
        self.logger = logger
        self.running = False
        self.cap = None
        
        # Estado
        self.drawing_strokes = []
        self.hand_in_fist = False
        
    def run(self):
        """Loop principal de captura de video."""
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            self.error_occurred.emit("No se pudo abrir la cámara")
            return
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["height"])
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
        
        hand_detector = self.components.get("hand_detector")
        stroke_accumulator = self.components.get("stroke_accumulator")
        preprocessor = self.components.get("preprocessor")
        classifier = self.components.get("classifier")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Voltear frame (espejo)
            frame = cv2.flip(frame, 1)
            
            # Detectar mano
            hand_landmarks = None
            hand_velocity = 0.0
            is_fist = False
            
            if hand_detector:
                try:
                    detection = hand_detector.detect(frame)
                    hand_landmarks = detection["hand_landmarks"]
                    hand_velocity = detection["hand_velocity"]
                    
                    # Detectar puño
                    if hasattr(hand_detector, 'is_fist'):
                        is_fist = hand_detector.is_fist()
                    
                    # Dibujar landmarks
                    frame = hand_detector.draw_hand_landmarks(frame)
                    
                    self.hand_detected.emit(hand_landmarks is not None)
                except Exception as e:
                    self.logger.warning(f"Error en detección: {e}")
            
            # Procesar trazo
            if hand_landmarks and stroke_accumulator:
                try:
                    index_pos = hand_detector.get_index_finger_position()
                    if index_pos:
                        if is_fist:
                            if stroke_accumulator.stroke_active:
                                stroke = stroke_accumulator.get_stroke()
                                if stroke:
                                    self.drawing_strokes.append(stroke)
                                stroke_accumulator.reset()
                            self.hand_in_fist = True
                        else:
                            if self.hand_in_fist:
                                self.hand_in_fist = False
                            
                            stroke_complete = stroke_accumulator.add_point(
                                index_pos[0], index_pos[1], hand_velocity
                            )
                            
                            if stroke_complete:
                                stroke = stroke_accumulator.get_stroke()
                                if stroke:
                                    self.drawing_strokes.append(stroke)
                                stroke_accumulator.reset()
                except Exception as e:
                    self.logger.warning(f"Error procesando trazo: {e}")
            
            # Emitir trazo actual
            if stroke_accumulator:
                self.stroke_updated.emit(stroke_accumulator.points)
            
            # Emitir frame
            self.frame_ready.emit(frame)
            
            # Small delay
            time.sleep(0.01)
        
        if self.cap:
            self.cap.release()
    
    def stop(self):
        """Detiene el thread."""
        self.running = False
        self.wait()
    
    def predict_drawing(self):
        """Realiza predicción del dibujo actual."""
        preprocessor = self.components.get("preprocessor")
        classifier = self.components.get("classifier")
        stroke_accumulator = self.components.get("stroke_accumulator")
        
        if not (preprocessor and classifier):
            return
        
        try:
            # Añadir trazo activo
            if stroke_accumulator:
                stroke = stroke_accumulator.get_stroke()
                if stroke:
                    self.drawing_strokes.append(stroke)
                    stroke_accumulator.reset()
            
            # Combinar trazos
            combined = []
            for s in self.drawing_strokes:
                combined.extend(s)
            
            if combined:
                drawing = preprocessor.preprocess(combined)
                label, conf, top3 = classifier.predict(drawing)
                self.prediction_ready.emit(label, conf, top3)
                
                # Limpiar
                self.drawing_strokes = []
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
    
    def clear_strokes(self):
        """Limpia los trazos."""
        self.drawing_strokes = []
        stroke_accumulator = self.components.get("stroke_accumulator")
        if stroke_accumulator:
            stroke_accumulator.reset()


class PictionaryLiveQt:
    """Aplicación principal con PyQt6."""
    
    def __init__(self, ia_dir: str = "./IA", camera_id: int = 0, debug: bool = False):
        """
        Inicializa la aplicación.
        
        Args:
            ia_dir: Ruta a la carpeta con modelo
            camera_id: ID de cámara
            debug: Habilitar logging DEBUG
        """
        self.logger = setup_logging(debug=debug)
        
        if debug:
            self.logger.info("Pictionary Live PyQt6 - Inicializando...")
        
        self.ia_dir = ia_dir
        self.camera_id = camera_id
        self.debug = debug
        
        # Inicializar QApplication
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Pictionary Live")
        
        # Inicializar componentes (lazy loading para velocidad)
        self.components = self._init_components()
        
        # Crear UI
        self.ui = PictionaryUIQt(UI_CONFIG)
        
        # Crear thread de video
        self.video_thread = VideoThread(camera_id, self.components, self.logger)
        self._connect_signals()
        
        if debug:
            self.logger.info("Aplicación inicializada")
    
    def _init_components(self) -> dict:
        """Inicializa los componentes de procesamiento."""
        components = {}
        
        try:
            hand_config = {**MEDIAPIPE_CONFIG["hands"], **DETECTION_CONFIG, **PERFORMANCE_CONFIG}
            components["hand_detector"] = HandDetector(hand_config, self.logger)
        except Exception as e:
            self.logger.error(f"Error inicializando HandDetector: {e}")
            components["hand_detector"] = None
        
        try:
            components["stroke_accumulator"] = StrokeAccumulator(STROKE_CONFIG, self.logger)
        except Exception as e:
            self.logger.error(f"Error inicializando StrokeAccumulator: {e}")
            components["stroke_accumulator"] = None
        
        try:
            components["classifier"] = SketchClassifier(
                self.ia_dir, self.logger, 
                demo_mode=MODEL_CONFIG["demo_mode"], 
                config=MODEL_CONFIG
            )
        except Exception as e:
            self.logger.error(f"Error inicializando SketchClassifier: {e}")
            components["classifier"] = SketchClassifier(
                self.ia_dir, self.logger, 
                demo_mode=True, 
                config=MODEL_CONFIG
            )
        
        try:
            input_shape = components["classifier"].get_input_shape() if components["classifier"] else [28, 28, 1]
            components["preprocessor"] = DrawingPreprocessor(input_shape, PREPROCESSING_CONFIG)
        except Exception as e:
            self.logger.error(f"Error inicializando DrawingPreprocessor: {e}")
            components["preprocessor"] = None
        
        return components
    
    def _connect_signals(self):
        """Conecta las señales del thread con la UI."""
        self.video_thread.frame_ready.connect(self.ui.update_frame)
        self.video_thread.hand_detected.connect(self.ui.update_hand_detected)
        self.video_thread.stroke_updated.connect(self.ui.update_stroke_points)
        self.video_thread.prediction_ready.connect(self.ui.update_prediction)
        self.video_thread.error_occurred.connect(self._handle_error)
    
    def _handle_error(self, error: str):
        """Maneja errores del thread."""
        self.logger.error(f"Error del thread: {error}")
        self.ui.set_state(f"❌ Error: {error}", "#ff6400")
    
    def run(self):
        """Ejecuta la aplicación."""
        if self.debug:
            self.logger.info("Iniciando aplicación PyQt6")
        
        # Mostrar UI
        self.ui.show()
        
        # Iniciar thread de video
        self.video_thread.start()
        
        # Ejecutar aplicación
        exit_code = self.app.exec()
        
        # Limpiar
        self.video_thread.stop()
        
        if self.debug:
            self.logger.info("Aplicación cerrada")
        return exit_code


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pictionary Live PyQt6")
    parser.add_argument("--ia-dir", default="./IA", help="Directorio con el modelo")
    parser.add_argument("--camera", type=int, default=0, help="ID de cámara")
    parser.add_argument("--debug", action="store_true", help="Habilitar debug")
    
    args = parser.parse_args()
    
    app = PictionaryLiveQt(
        ia_dir=args.ia_dir,
        camera_id=args.camera,
        debug=args.debug
    )
    
    sys.exit(app.run())


if __name__ == '__main__':
    main()
