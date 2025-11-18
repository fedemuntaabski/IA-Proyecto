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
        
        # Canvas de dibujo (256x256 blanco para acumular trazos negros de 8px)
        self.drawing_canvas = np.ones((256, 256), dtype=np.uint8) * 255
        self.last_index_pos = None
        
        # Modo de entrada (hand/mouse)
        self.use_mouse_input = False
        
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
        
        # Verificar si el detector de manos está disponible
        if hand_detector is None or not hasattr(hand_detector, 'hands_detector') or hand_detector.hands_detector is None:
            self.use_mouse_input = True
            self.logger.warning("MediaPipe no disponible - usando entrada de mouse")
            # No emitir error, es un comportamiento esperado en Python 3.13+
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Voltear frame (espejo)
            frame = cv2.flip(frame, 1)
            
            # Modo con detección de manos
            if not self.use_mouse_input and hand_detector:
                # Detectar mano
                hand_landmarks = None
                hand_velocity = 0.0
                is_fist = False
                
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
                                # Puño cerrado: finalizar trazo
                                if stroke_accumulator.stroke_active:
                                    stroke = stroke_accumulator.get_stroke()
                                    if stroke:
                                        self.drawing_strokes.append(stroke)
                                    stroke_accumulator.reset()
                                self.hand_in_fist = True
                                self.last_index_pos = None
                            else:
                                if self.hand_in_fist:
                                    self.hand_in_fist = False
                                
                                # Dibujar línea negra de 8px en el canvas
                                px = int(index_pos[0] * 255)
                                py = int(index_pos[1] * 255)
                                
                                if self.last_index_pos is not None:
                                    cv2.line(self.drawing_canvas, self.last_index_pos, (px, py), 0, 8, cv2.LINE_AA)
                                
                                self.last_index_pos = (px, py)
                                
                                stroke_complete = stroke_accumulator.add_point(
                                    index_pos[0], index_pos[1], hand_velocity
                                )
                                
                                if stroke_complete:
                                    stroke = stroke_accumulator.get_stroke()
                                    if stroke:
                                        self.drawing_strokes.append(stroke)
                                    stroke_accumulator.reset()
                                    self.last_index_pos = None
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
    
    @pyqtSlot()
    def predict_drawing(self):
        """Realiza predicción del dibujo actual usando el canvas."""
        classifier = self.components.get("classifier")
        
        if not classifier:
            return
        
        try:
            # Verificar que hay contenido dibujado
            if np.sum(self.drawing_canvas < 250) == 0:
                self.logger.warning("Canvas vacío, nada que predecir")
                return
            
            # Preprocesar el canvas para el modelo
            # El canvas ya tiene líneas negras (0) sobre fondo blanco (255)
            # Necesitamos convertirlo al formato del modelo
            
            # 1. Calcular bounding box del contenido
            binary = cv2.threshold(self.drawing_canvas, 127, 255, cv2.THRESH_BINARY_INV)[1]
            coords = cv2.findNonZero(binary)
            
            if coords is None:
                self.logger.warning("No se encontró contenido en el canvas")
                return
            
            x, y, w, h = cv2.boundingRect(coords)
            
            # 2. Agregar padding del 12%
            max_dim = max(w, h)
            pad = int(max_dim * 0.12)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(256 - x, w + 2 * pad)
            h = min(256 - y, h + 2 * pad)
            
            # 3. Recortar región de interés
            roi = self.drawing_canvas[y:y+h, x:x+w]
            
            # 4. Centrar en canvas cuadrado
            max_dim = max(w, h)
            square_canvas = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
            offset_x = (max_dim - w) // 2
            offset_y = (max_dim - h) // 2
            square_canvas[offset_y:offset_y+h, offset_x:offset_x+w] = roi
            
            # 5. Redimensionar a 28x28
            resized = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_LANCZOS4)
            
            # 6. Normalizar e invertir (blanco sobre negro)
            normalized = resized.astype(np.float32) / 255.0
            normalized = 1.0 - normalized  # Invertir
            normalized = np.expand_dims(normalized, axis=-1)  # Agregar canal
            
            # Predecir
            label, conf, top3 = classifier.predict(normalized)
            self.prediction_ready.emit(label, conf, top3)
            
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
    
    @pyqtSlot()
    def clear_strokes(self):
        """Limpia los trazos y el canvas."""
        self.drawing_strokes = []
        self.drawing_canvas = np.ones((256, 256), dtype=np.uint8) * 255  # Resetear a blanco
        self.last_index_pos = None
        stroke_accumulator = self.components.get("stroke_accumulator")
        if stroke_accumulator:
            stroke_accumulator.reset()
    
    @pyqtSlot(float, float, bool)
    def process_mouse_input(self, x: float, y: float, is_drawing: bool):
        """Procesa entrada del mouse para dibujo."""
        if not is_drawing:
            # Fin de trazo
            self.last_index_pos = None
            stroke_accumulator = self.components.get("stroke_accumulator")
            if stroke_accumulator and stroke_accumulator.stroke_active:
                stroke = stroke_accumulator.get_stroke()
                if stroke:
                    self.drawing_strokes.append(stroke)
                stroke_accumulator.reset()
            return
        
        # Dibujar en el canvas
        px = int(x * 255)
        py = int(y * 255)
        
        if self.last_index_pos is not None:
            cv2.line(self.drawing_canvas, self.last_index_pos, (px, py), 0, 8, cv2.LINE_AA)
        
        self.last_index_pos = (px, py)
        
        # Agregar punto al acumulador
        stroke_accumulator = self.components.get("stroke_accumulator")
        if stroke_accumulator:
            stroke_accumulator.add_point(x, y, 0.05)  # Velocidad constante para mouse
            self.stroke_updated.emit(stroke_accumulator.points)


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
        
        # Conectar señales de UI a acciones del thread
        self.ui.predict_requested.connect(self.video_thread.predict_drawing)
        self.ui.clear_requested.connect(self.video_thread.clear_strokes)
        
        # Conectar señal del mouse para modo fallback
        self.ui.video_widget.mouse_draw.connect(self.video_thread.process_mouse_input)
    
    def _handle_error(self, error: str):
        """Maneja errores del thread."""
        self.logger.error(f"Error del thread: {error}")
        # Solo actualizar estado en la UI si es un error real
        if "Error" in error or "error" in error:
            self.ui.set_state(f"Error: {error}", "#ff6400")
    
    def run(self):
        """Ejecuta la aplicación."""
        if self.debug:
            self.logger.info("Iniciando aplicación PyQt6")
        
        # Mostrar UI
        self.ui.show()
        
        # Verificar modo de entrada y actualizar UI
        if not self.components.get("hand_detector") or \
           not hasattr(self.components["hand_detector"], 'hands_detector') or \
           self.components["hand_detector"].hands_detector is None:
            self.ui.set_state("🖊️ MODO MOUSE", "#ffa000")
        
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
