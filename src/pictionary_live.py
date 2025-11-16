#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pictionary_live.py
===================
Aplicación Python para jugar Pictionary en vivo usando detección de gestos con las manos.

Propósito:
- Captura video en tiempo real de la cámara.
- Detecta el trazo dibujado en el aire con las manos usando MediaPipe.
- Acumula la trayectoria del gesto y, cuando detecta una pausa, clasifica el sketch
  usando un modelo de red neuronal (Keras/TensorFlow).
- Muestra la predicción en pantalla y guarda logs de inferencia.

Uso:
    python pictionary_live.py --ia-dir ./IA
    python pictionary_live.py --ia-dir ./IA --camera-id 0 --debug
    python pictionary_live.py --ia-dir ./IA --dry-run

Dependencias:
    - opencv-python
    - tensorflow (o tensorflow-cpu)
    - mediapipe
    - numpy
    - ndjson
"""

import argparse
import cv2
import json
import logging
import numpy as np
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow no disponible. El programa funcionara en modo demo sin inferencia real.")

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configura logging con nivel INFO o DEBUG."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Crear directorio de logs si no existe
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("PictionaryLive")
    logger.setLevel(level)
    
    # Eliminar handlers previos
    logger.handlers.clear()
    
    # Handler para archivo
    log_file = logs_dir / f"pictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formato
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# CARGA Y VALIDACIÓN DE MODELO
# ============================================================================

class ModelLoader:
    """Carga y valida modelos de Keras desde la carpeta IA."""
    
    def __init__(self, ia_dir: str, logger: logging.Logger):
        self.ia_dir = Path(ia_dir)
        self.logger = logger
        
        if not self.ia_dir.exists():
            raise FileNotFoundError(f"Carpeta IA no encontrada: {self.ia_dir}")
        
        self.model = None
        self.model_info = None
        self.input_shape = None
        self.labels = []
        self.preprocessing_config = {}
    
    def load(self) -> bool:
        """Carga el modelo y metadatos. Retorna True si fue exitoso."""
        try:
            # Cargar metadata
            if not self._load_model_info():
                return False
            
            # Cargar modelo
            if not self._load_model():
                return False
            
            self.logger.info("Modelo cargado exitosamente. Input shape: {}".format(self.input_shape))
            self.logger.info("Clases disponibles: {}".format(len(self.labels)))
            return True
        
        except Exception as e:
            self.logger.error(f"Error al cargar modelo: {e}")
            return False
    
    def _load_model_info(self) -> bool:
        """Carga model_info.json."""
        model_info_path = self.ia_dir / "model_info.json"
        
        if not model_info_path.exists():
            self.logger.error("No se encontró model_info.json")
            return False
        
        try:
            with open(model_info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            
            self.input_shape = self.model_info.get("input_shape", [28, 28, 1])
            self.labels = self.model_info.get("classes", [])
            
            self.logger.info("Modelo info cargado: {} clases".format(len(self.labels)))
            return True
        
        except Exception as e:
            self.logger.error("Error al cargar model_info.json: {}".format(e))
            return False
    
    def _load_model(self) -> bool:
        """Carga el modelo (keras o h5)."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow no disponible. Funcionando en modo demo.")
            return True  # Permitir funcionamiento en modo demo
        
        # Preferir .keras, luego .h5
        model_paths = [
            self.ia_dir / "sketch_classifier_model.keras",
            self.ia_dir / "sketch_classifier_model.h5",
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                try:
                    self.model = keras.models.load_model(model_path)
                    self.logger.info("Modelo cargado desde: {}".format(model_path.name))
                    return True
                except Exception as e:
                    self.logger.warning("Error al cargar {}: {}".format(model_path.name, e))
        
        self.logger.warning("No se encontraron archivos de modelo. Funcionando en modo demo.")
        return True  # Permitir funcionamiento en modo demo
    
    def predict(self, drawing: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Realiza inferencia sobre un drawing.
        
        Args:
            drawing: array de entrada preprocesado
        
        Returns:
            (etiqueta_top1, probabilidad_top1, top3_predictions)
        """
        if not TENSORFLOW_AVAILABLE:
            # Modo demo: devolver predicción simulada
            import random
            if len(self.labels) > 0:
                # Elegir una clase aleatoria
                top1_idx = random.randint(0, min(10, len(self.labels) - 1))  # Primeras 10 clases
                top1_label = self.labels[top1_idx]
                top1_prob = random.uniform(0.3, 0.9)  # Probabilidad aleatoria
                
                # Top-3 aleatorias
                indices = list(range(min(10, len(self.labels))))
                random.shuffle(indices)
                top3_indices = indices[:3]
                top3 = [(self.labels[idx], random.uniform(0.1, top1_prob)) for idx in top3_indices]
                
                return top1_label, top1_prob, top3
            else:
                return "demo", 0.5, [("demo", 0.5), ("test", 0.3), ("sample", 0.2)]
        
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
        
        # Agregar batch dimension si es necesario
        if len(drawing.shape) == 3:
            drawing = np.expand_dims(drawing, axis=0)
        
        # Predicción
        predictions = self.model.predict(drawing, verbose=0)
        probs = predictions[0]
        
        # Top-1
        top1_idx = np.argmax(probs)
        top1_label = self.labels[top1_idx] if top1_idx < len(self.labels) else "Unknown"
        top1_prob = float(probs[top1_idx])
        
        # Top-3
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3 = [
            (self.labels[idx] if idx < len(self.labels) else "Unknown", float(probs[idx]))
            for idx in top3_indices
        ]
        
        return top1_label, top1_prob, top3


# ============================================================================
# DETECCIÓN DE CUERPO COMPLETO (MANOS + POSE)
# ============================================================================

class BodyTracker:
    """Detecta cuerpo completo: manos, brazos, cabeza, hombros usando MediaPipe."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        self.hands_detector = None
        self.pose_detector = None
        self.hand_landmarks = None
        self.pose_landmarks = None
        self.previous_hand_pos = None
        self.hand_velocity = 0.0
        
        if self.use_mediapipe:
            self._init_mediapipe()
        else:
            self.logger.warning("MediaPipe no disponible. Funcionando en modo demo.")
    
    def _init_mediapipe(self):
        """Inicializa MediaPipe Hands + Pose para detección de cuerpo completo."""
        try:
            # Detección de manos
            mp_hands = mp.solutions.hands
            self.hands_detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,  # Reducido de 0.7 para mejor tolerancia
                min_tracking_confidence=0.5,   # Reducido de 0.7 para mejor seguimiento
            )
            
            # Detección de pose (cuerpo completo: cabeza, hombros, brazos, etc.)
            mp_pose = mp.solutions.pose
            self.pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,  # Reducido de 0.6 para mejor tolerancia
                min_tracking_confidence=0.5,   # Reducido de 0.6 para mejor seguimiento
            )
            
            self.logger.info("MediaPipe Hands + Pose inicializados (cuerpo completo)")
            self.logger.info("  Umbrales de confianza: 0.5 (optimizados para detección en condiciones reales)")
        except Exception as e:
            self.logger.warning(f"Error inicializando MediaPipe: {e}")
            self.use_mediapipe = False
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detecta manos y pose en el frame.
        
        Returns:
            Dict con 'hand_landmarks', 'pose_landmarks', 'hand_velocity'
        """
        if not self.use_mediapipe:
            return {"hand_landmarks": None, "pose_landmarks": None, "hand_velocity": 0.0}
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detectar manos
            hand_results = self.hands_detector.process(frame_rgb)
            hand_landmarks = None
            hand_velocity = 0.0
            
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) > 0:
                hand_landmarks = [(lm.x, lm.y) for lm in hand_results.multi_hand_landmarks[0].landmark]
                
                # Calcular velocidad del dedo índice (landmark 8)
                if len(hand_landmarks) > 8:
                    current_pos = hand_landmarks[8]
                    if self.previous_hand_pos:
                        dx = current_pos[0] - self.previous_hand_pos[0]
                        dy = current_pos[1] - self.previous_hand_pos[1]
                        hand_velocity = (dx ** 2 + dy ** 2) ** 0.5  # Distancia euclidiana
                    self.previous_hand_pos = current_pos
                
                self.logger.debug(f"Mano detectada: {len(hand_landmarks)} landmarks, velocidad: {hand_velocity:.6f}")
            else:
                self.logger.debug("No se detectó mano en este frame")
            
            # Detectar pose (cuerpo)
            pose_results = self.pose_detector.process(frame_rgb)
            pose_landmarks = None
            if pose_results.pose_landmarks:
                pose_landmarks = {
                    'landmarks': [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks],
                    'visibility': [lm.visibility for lm in pose_results.pose_landmarks]
                }
                self.logger.debug(f"Pose detectada: {len(pose_results.pose_landmarks)} landmarks")
            else:
                self.logger.debug("No se detectó pose en este frame")
            
            self.hand_landmarks = hand_landmarks
            self.pose_landmarks = pose_landmarks
            self.hand_velocity = hand_velocity
            
            return {
                "hand_landmarks": hand_landmarks,
                "pose_landmarks": pose_landmarks,
                "hand_velocity": hand_velocity
            }
        except Exception as e:
            self.logger.error(f"Error en detección: {e}", exc_info=True)
            return {"hand_landmarks": None, "pose_landmarks": None, "hand_velocity": 0.0}
    
    def draw_full_body(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja detección de cuerpo completo (pose + manos)."""
        if not self.use_mediapipe:
            return frame
        
        h, w = frame.shape[:2]
        
        # Dibujar POSE (cuerpo: cabeza, hombros, brazos)
        if self.pose_landmarks:
            landmarks = self.pose_landmarks['landmarks']
            visibility = self.pose_landmarks['visibility']
            
            # Conexiones clave del cuerpo
            body_connections = [
                (11, 13), (13, 15),  # Brazo izquierdo
                (12, 14), (14, 16),  # Brazo derecho
                (11, 12),            # Conexión de hombros
                (0, 11), (0, 12),    # Cabeza - Hombros
                (23, 25), (25, 27),  # Pierna izquierda
                (24, 26), (26, 28),  # Pierna derecha
            ]
            
            # Dibujar conexiones
            for start, end in body_connections:
                if start < len(landmarks) and end < len(landmarks):
                    if visibility[start] > 0.3 and visibility[end] > 0.3:
                        x1, y1 = int(landmarks[start][0] * w), int(landmarks[start][1] * h)
                        x2, y2 = int(landmarks[end][0] * w), int(landmarks[end][1] * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 100), 3)
            
            # Dibujar puntos de cuerpo
            for idx, (x, y, z) in enumerate(landmarks):
                if visibility[idx] > 0.3:
                    px, py = int(x * w), int(y * h)
                    # Colores especiales para puntos clave
                    if idx in [0]:  # Cabeza
                        color = (255, 100, 0)
                    elif idx in [11, 12]:  # Hombros
                        color = (255, 200, 0)
                    elif idx in [13, 14, 15, 16]:  # Brazos
                        color = (0, 255, 100)
                    else:
                        color = (100, 100, 100)
                    cv2.circle(frame, (px, py), 5, color, -1)
        
        # Dibujar LANDMARKS DE MANOS
        if self.hand_landmarks:
            # Conexiones de mano
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
                    cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), 
                           (255, 0, 150), 2)
            
            # Dibujar puntos con énfasis en dedo índice
            for i, (x, y) in enumerate(self.hand_landmarks):
                px, py = int(x * w), int(y * h)
                if i == 8:  # Dedo índice (punta)
                    cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)  # Rojo brillante
                    cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)
                else:
                    cv2.circle(frame, (px, py), 4, (255, 0, 255), -1)
        
        return frame
    
    def get_index_finger_position(self) -> Optional[Tuple[float, float]]:
        """Obtiene la posición del dedo índice (point 8)."""
        if self.hand_landmarks and len(self.hand_landmarks) > 8:
            return self.hand_landmarks[8]
        return None


# ============================================================================
# ACUMULADOR DE TRAZO
# ============================================================================

class StrokeAccumulator:
    """Acumula puntos del trazo y detecta finales basado en movimiento y pausa."""
    
    def __init__(
        self,
        pause_threshold_ms: int = 300,
        velocity_threshold: float = 0.005,
        min_points: int = 10,
        logger: logging.Logger = None
    ):
        self.pause_threshold_ms = pause_threshold_ms
        self.velocity_threshold = velocity_threshold  # Umbral de velocidad para detectar trazo
        self.min_points = min_points
        self.logger = logger or logging.getLogger()
        
        self.points = []
        self.last_point_time = None
        self.last_significant_move_time = None
        self.stroke_active = False
    
    def reset(self):
        """Reinicia el acumulador."""
        self.points = []
        self.last_point_time = None
        self.last_significant_move_time = None
        self.stroke_active = False
    
    def add_point(self, x: float, y: float, velocity: float = 0.0) -> bool:
        """
        Agrega un punto al trazo considerando velocidad del dedo.
        
        Args:
            x, y: coordenadas normalizadas
            velocity: velocidad del movimiento (0-1)
        
        Returns:
            True si el trazo se completó (pausa detectada)
        """
        current_time = time.time() * 1000  # ms
        
        # Ignorar movimientos muy lentos (ruido)
        if velocity < self.velocity_threshold:
            # Si hay pausa sin movimiento significativo
            if self.stroke_active and len(self.points) >= self.min_points:
                if self.last_significant_move_time:
                    time_since_move = current_time - self.last_significant_move_time
                    if time_since_move > self.pause_threshold_ms:
                        return True  # Trazo completado
            return False
        
        # Movimiento significativo detectado
        if self.last_point_time is None:
            self.points = [(x, y)]
            self.last_point_time = current_time
            self.last_significant_move_time = current_time
            self.stroke_active = True
            return False
        
        # Agregar punto
        self.points.append((x, y))
        self.last_point_time = current_time
        self.last_significant_move_time = current_time
        
        return False
    
    def get_stroke(self) -> Optional[List[Tuple[float, float]]]:
        """Retorna el trazo actual si tiene suficientes puntos."""
        if len(self.points) >= self.min_points:
            return self.points.copy()
        return None


# ============================================================================
# PREPROCESADO DE DRAWING
# ============================================================================

class DrawingPreprocessor:
    """Preprocesa trazo para inferencia."""
    
    def __init__(self, target_shape: Tuple[int, int, int], logger: logging.Logger = None):
        """
        Args:
            target_shape: (height, width, channels) esperado por el modelo
            logger: logger para debug
        """
        self.target_shape = target_shape
        self.logger = logger or logging.getLogger()
        self.h, self.w, self.c = target_shape
    
    def preprocess(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Preprocesa puntos del trazo a formato de entrada del modelo.
        
        Args:
            points: lista de (x, y) normalizadas (0-1) o en píxeles
        
        Returns:
            array de forma target_shape listo para predicción
        """
        if not points or len(points) < 2:
            return self._create_empty_drawing()
        
        # Crear canvas en blanco
        canvas = np.ones((self.h, self.w), dtype=np.uint8) * 255
        
        # Normalizar y dibujar
        points_pixels = self._normalize_to_pixels(points)
        self._draw_stroke(canvas, points_pixels)
        
        # Invertir (fondo blanco -> negro) para que el trazo sea blanco
        canvas = 255 - canvas
        
        # Normalizar a [0, 1]
        canvas = canvas.astype(np.float32) / 255.0
        
        # Reshape a target_shape
        if self.c == 1:
            canvas = np.expand_dims(canvas, axis=-1)
        
        return canvas
    
    def _normalize_to_pixels(self, points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Normaliza puntos a píxeles en el canvas."""
        if not points:
            return []
        
        # Asumir que los puntos están en rango [0, 1] (normalizados)
        # Si están en píxeles, ajustar
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        # Detectar si están en rango 0-1 o en píxeles
        max_x = max(xs) if xs else 0
        max_y = max(ys) if ys else 0
        
        if max_x <= 1.0 and max_y <= 1.0:
            # Están normalizados
            pixel_points = [
                (int(x * self.w), int(y * self.h))
                for x, y in points
            ]
        else:
            # Están en píxeles, normalizar primero
            pixel_points = [
                (int(x), int(y))
                for x, y in points
            ]
        
        return pixel_points
    
    def _draw_stroke(self, canvas: np.ndarray, points: List[Tuple[int, int]]):
        """Dibuja el trazo sobre el canvas."""
        for i in range(1, len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            cv2.line(canvas, p1, p2, 0, 2)  # color=0 (negro), thickness=2
    
    def _create_empty_drawing(self) -> np.ndarray:
        """Crea un drawing vacío (blanco)."""
        drawing = np.ones((self.h, self.w, self.c), dtype=np.float32)
        return drawing


# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

def validate_setup(ia_dir: str, camera_id: int = 0, logger: logging.Logger = None) -> Tuple[bool, str]:
    """
    Valida que el setup está correcto antes de ejecutar la aplicación.
    
    Returns:
        (success: bool, message: str)
    """
    logger = logger or logging.getLogger()
    
    checks = []
    
    # 1. Validar folder IA
    ia_path = Path(ia_dir)
    if not ia_path.exists():
        checks.append(f"[X] Carpeta IA no encontrada: {ia_dir}")
    else:
        checks.append(f"[OK] Carpeta IA encontrada: {ia_dir}")
    
    # 2. Validar modelo info
    model_info_path = ia_path / "model_info.json"
    if not model_info_path.exists():
        checks.append(f"[X] model_info.json no encontrado")
    else:
        checks.append(f"[OK] model_info.json encontrado")
    
    # 3. Validar modelo (keras o h5)
    model_exists = (ia_path / "sketch_classifier_model.keras").exists() or \
                   (ia_path / "sketch_classifier_model.h5").exists()
    if not model_exists and TENSORFLOW_AVAILABLE:
        checks.append(f"[X] Modelo (keras/h5) no encontrado")
    else:
        checks.append(f"[OK] Modelo encontrado")
    
    # 4. Validar dependencias
    if not TENSORFLOW_AVAILABLE:
        checks.append(f"[!] TensorFlow no disponible (funcionara en modo demo)")
    else:
        checks.append(f"[OK] TensorFlow disponible")
    
    if not MEDIAPIPE_AVAILABLE:
        checks.append(f"[X] MediaPipe no disponible (esencial para deteccion)")
    else:
        checks.append(f"[OK] MediaPipe disponible")
    
    # 5. Validar camara
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            checks.append(f"[X] No se puede acceder a la camara {camera_id}")
        else:
            # Obtener propiedades
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            checks.append(f"[OK] Camara {camera_id} accesible (resolucion: {width}x{height}, FPS: {fps})")
        cap.release()
    except Exception as e:
        checks.append(f"[X] Error al acceder a la camara: {e}")
    
    # Imprimir reporte
    logger.info("\n" + "=" * 70)
    logger.info("VALIDACION DE SETUP")
    logger.info("=" * 70)
    for check in checks:
        logger.info(check)
    logger.info("=" * 70 + "\n")
    
    # Determinar éxito
    success = not any(check.startswith("[X]") for check in checks) and MEDIAPIPE_AVAILABLE
    
    if not success:
        message = "Setup invalido. Verifica los errores arriba."
    else:
        message = "Setup validado correctamente."
    
    return success, message


# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

class PictionaryLive:
    """Aplicación principal de Pictionary en vivo."""
    
    def __init__(
        self,
        ia_dir: str = "./IA",
        camera_id: int = 0,
        debug: bool = False,
        dry_run: bool = False
    ):
        self.logger = setup_logging(debug=debug)
        self.logger.info("=" * 70)
        self.logger.info("PICTIONARY LIVE - Inicializando")
        self.logger.info("=" * 70)
        
        self.ia_dir = ia_dir
        self.camera_id = camera_id
        self.debug = debug
        self.dry_run = dry_run
        
        # Componentes
        self.model_loader = None
        self.hand_tracker = None
        self.stroke_accumulator = None
        self.preprocessor = None
        
        # Estado
        self.running = False
        self.cap = None
        
        # Inferencia
        self.inference_log_path = Path("./inference.log")
        
        # Diagnóstico y rendimiento
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
        self.detection_stats = {
            'hands_detected': 0,
            'pose_detected': 0,
            'avg_hand_velocity': 0.0
        }
        
        # Validar y cargar
        if not self._initialize():
            raise RuntimeError("Inicialización fallida")
    
    def _initialize(self) -> bool:
        """Inicializa componentes."""
        try:
            # Cargar modelo
            self.model_loader = ModelLoader(self.ia_dir, self.logger)
            if not self.model_loader.load():
                return False
            
            # Hand tracker
            self.body_tracker = BodyTracker(self.logger)
            
            # Stroke accumulator con parámetros mejorados
            self.stroke_accumulator = StrokeAccumulator(
                pause_threshold_ms=500,      # Más tiempo para pausas naturales
                velocity_threshold=0.003,    # Más sensible a movimientos lentos
                min_points=10,               # Trazos más cortos válidos
                logger=self.logger
            )
            
            # Preprocessor
            self.preprocessor = DrawingPreprocessor(
                tuple(self.model_loader.input_shape),
                logger=self.logger
            )
            
            self.logger.info("Componentes inicializados")
            return True
        
        except Exception as e:
            self.logger.error(f"Error en inicialización: {e}")
            return False
    
    def run_dry_run(self):
        """Valida modelo y lista etiquetas sin abrir cámara."""
        self.logger.info("\n[DRY RUN] Validando configuración...")
        self.logger.info(f"  Modelo info shape: {self.model_loader.input_shape}")
        self.logger.info(f"  Total de clases: {len(self.model_loader.labels)}")
        self.logger.info(f"  Primeras 10 clases: {self.model_loader.labels[:10]}")
        self.logger.info("[DRY RUN] ✓ Todo listo")
    
    def run(self):
        """Ejecuta la aplicación principal."""
        if self.dry_run:
            self.run_dry_run()
            return
        
        if not self._init_camera():
            self.logger.error("No se pudo inicializar la cámara")
            return
        
        self.logger.info("\n[VIVO] Iniciando captura. Presiona 'q' para salir, 's' para guardar.")
        self.running = True
        
        # Crear ventana en fullscreen/maximizado
        cv2.namedWindow("Pictionary Live - Dibuja en el aire", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pictionary Live - Dibuja en el aire", 1280, 960)  # Ventana grande
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("No se pudo leer frame")
                    break
                
                # Reflejar horizontalmente (espejo)
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Detección de cuerpo completo
                detection_result = self.body_tracker.detect(frame)
                hand_landmarks = detection_result["hand_landmarks"]
                hand_velocity = detection_result["hand_velocity"]
                
                # Dibujar pose y manos
                frame = self.body_tracker.draw_full_body(frame)
                
                if hand_landmarks:
                    # Obtener posición del dedo índice
                    index_pos = self.body_tracker.get_index_finger_position()
                    
                    if index_pos:
                        # Agregar punto al trazo considerando velocidad
                        stroke_complete = self.stroke_accumulator.add_point(
                            index_pos[0], index_pos[1], hand_velocity
                        )
                        
                        if stroke_complete:
                            self.logger.info("Trazo completado ({} puntos)".format(len(self.stroke_accumulator.points)))
                            self._process_stroke()
                            self.stroke_accumulator.reset()
                        
                        # Dibujar trazo acumulado en rojo
                        if self.stroke_accumulator.points:
                            points_px = [
                                (int(p[0] * w), int(p[1] * h))
                                for p in self.stroke_accumulator.points
                            ]
                            for i in range(1, len(points_px)):
                                cv2.line(frame, points_px[i - 1], points_px[i],
                                       (0, 0, 255), 3)  # Rojo para trazo
                else:
                    # Sin manos detectadas
                    if self.stroke_accumulator.stroke_active and len(
                        self.stroke_accumulator.points) >= 5:
                        # Forzar fin de trazo después de 500ms sin detección
                        self.logger.info("Mano no detectada - finalizando trazo")
                        self._process_stroke()
                        self.stroke_accumulator.reset()
                
                # UI overlay mejorado
                self._draw_ui(frame)
                
                # Mostrar frame
                cv2.imshow("Pictionary Live - Dibuja en el aire", frame)
                
                # Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Usuario presionó 'q' - saliendo")
                    self.running = False
                elif key == ord('s'):
                    self._save_prediction(frame)
        
        except KeyboardInterrupt:
            self.logger.info("Interrupción por usuario")
        finally:
            self._cleanup()
    
    def _init_camera(self) -> bool:
        """Inicializa captura de cámara con configuración óptima."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"No se puede abrir cámara {self.camera_id}")
                return False
            
            # Configurar resolución óptima (1280x720)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Configurar otros parámetros
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimizar latencia
            
            # Obtener parámetros reales
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"Cámara {self.camera_id} abierta")
            self.logger.info(f"  Resolución: {width}x{height}")
            self.logger.info(f"  FPS: {fps}")
            self.logger.info(f"  Buffer mínimo para baja latencia")
            return True
        except Exception as e:
            self.logger.error(f"Error al inicializar cámara: {e}")
            return False
    
    def _process_stroke(self):
        """Procesa el trazo acumulado."""
        stroke = self.stroke_accumulator.get_stroke()
        if not stroke or len(stroke) < 5:
            return
        
        try:
            # Preprocesar
            drawing = self.preprocessor.preprocess(stroke)
            
            # Predicción
            label, prob, top3 = self.model_loader.predict(drawing)
            
            # Log
            self.logger.info(f"Predicción: {label} ({prob:.2%})")
            self._log_inference(label, prob, top3)
            
            # Guardar para UI
            self.last_prediction = {
                'label': label,
                'prob': prob,
                'top3': top3
            }
        
        except Exception as e:
            self.logger.error(f"Error en inferencia: {e}")
    
    def _draw_ui(self, frame: np.ndarray):
        """Dibuja UI overlay mejorado en el frame."""
        h, w = frame.shape[:2]
        
        # Actualizar FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        # Panel superior (información)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Título principal
        cv2.putText(frame, "PICTIONARY LIVE - Dibuja en el aire", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # FPS y diagnóstico
        diag_text = f"FPS: {self.fps:.1f} | Puntos trazo: {len(self.stroke_accumulator.points)} | Manos: {'SI' if self.body_tracker.hand_landmarks else 'NO'}"
        cv2.putText(frame, diag_text, (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2)
        
        # Información de velocidad si hay mano detectada
        if self.body_tracker.hand_velocity > 0:
            vel_text = f"Velocidad dedo: {self.body_tracker.hand_velocity:.4f}"
            cv2.putText(frame, vel_text, (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        
        # Última predicción (si existe)
        if hasattr(self, 'last_prediction') and self.last_prediction:
            pred = self.last_prediction
            pred_text = f"{pred['label']}: {pred['prob']:.1%}"
            
            # Panel derecho con predicción
            text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            x_pos = w - text_size[0] - 30
            
            # Fondo para predicción
            cv2.rectangle(frame, (x_pos - 10, 25), (w - 10, 95), (50, 100, 150), -1)
            cv2.rectangle(frame, (x_pos - 10, 25), (w - 10, 95), (0, 200, 255), 2)
            
            cv2.putText(frame, "Predicción:", (x_pos, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(frame, pred_text, (x_pos, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Mostrar top-3
            if pred['top3']:
                top3_y = 100
                cv2.putText(frame, "Top 3 predicciones:", (20, h - top3_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
                for i, (label, prob) in enumerate(pred['top3'][:3]):
                    y_offset = h - top3_y + 30 + (i * 25)
                    cv2.putText(frame, f"  {i+1}. {label}: {prob:.1%}", (30, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 1)
        
        # Controles (abajo)
        controls_text = "q=SALIR | s=GUARDAR | Dibuja con el dedo indice | Iluminacion frontal + fondo plano = Mejor deteccion"
        cv2.putText(frame, controls_text, (20, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def _log_inference(self, label: str, prob: float, top3: List[Tuple[str, float]]):
        """Guarda log de inferencia."""
        try:
            with open(self.inference_log_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                top3_str = "; ".join([f"{l}: {p:.2%}" for l, p in top3])
                f.write(f"{timestamp} | {label} ({prob:.2%}) | Top-3: {top3_str}\n")
        except Exception as e:
            self.logger.warning(f"Error al guardar log: {e}")
    
    def _save_prediction(self, frame: np.ndarray):
        """Guarda predicción actual."""
        pred_dir = Path("./predictions")
        pred_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = pred_dir / f"frame_{timestamp}.png"
        cv2.imwrite(str(frame_path), frame)
        
        self.logger.info(f"✓ Frame guardado: {frame_path}")
    
    def _cleanup(self):
        """Limpia recursos."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Aplicación cerrada")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

def main():
    """Función principal con argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Pictionary Live - Juega Pictionary dibujando en el aire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python pictionary_live.py --ia-dir ./IA
  python pictionary_live.py --ia-dir ./IA --debug
  python pictionary_live.py --ia-dir ./IA --dry-run
  python pictionary_live.py --ia-dir ./IA --camera-id 1

Requisitos mínimos para detección óptima:
  - Iluminación: Luz frontal blanca uniforme
  - Fondo: Plano y contrastante (verde/azul sólido)
  - Cámara: Resolución 1280x720 o superior, centrada
  - Distancia: 60-80 cm de la mano
  - Python: 3.10, 3.11 o 3.12
        """
    )
    
    parser.add_argument(
        "--ia-dir",
        type=str,
        default="./IA",
        help="Ruta a la carpeta IA con modelo y metadatos (default: ./IA)"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="ID de cámara a usar (default: 0, prueba 1, 2 si no funciona)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Habilitar logging DEBUG (más detallado)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validar modelo y setup sin abrir cámara"
    )
    
    args = parser.parse_args()
    
    try:
        # Crear logger para validación
        logger = setup_logging(debug=args.debug)
        
        # Validar setup ANTES de crear la app
        success, message = validate_setup(args.ia_dir, args.camera_id, logger)
        
        if not success:
            logger.error(message)
            sys.exit(1)
        
        logger.info(message)
        
        # Crear y ejecutar aplicación
        app = PictionaryLive(
            ia_dir=args.ia_dir,
            camera_id=args.camera_id,
            debug=args.debug,
            dry_run=args.dry_run
        )
        app.run()
    
    except Exception as e:
        logging.error(f"Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
