"""
model.py - Carga y gestión del modelo de clasificación
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class SketchClassifier:
    """Carga y realiza inferencia con el modelo de clasificación de sketches."""
    
    def __init__(self, ia_dir: str, logger: logging.Logger, demo_mode: bool = True):
        """
        Inicializa el clasificador.
        
        Args:
            ia_dir: Ruta a la carpeta IA con modelo
            logger: Logger
            demo_mode: Usar predicciones demo si no está disponible TensorFlow
        """
        self.ia_dir = Path(ia_dir)
        self.logger = logger
        self.demo_mode = demo_mode
        
        self.model = None
        self.input_shape = [28, 28, 1]
        self.labels = []
        
        if not self.ia_dir.exists():
            self.logger.warning(f"Carpeta IA no encontrada: {ia_dir}")
            return
        
        # Cargar metadata
        self._load_model_info()
        
        # Cargar modelo si TensorFlow disponible
        if TENSORFLOW_AVAILABLE:
            self._load_model()
    
    def _load_model_info(self) -> bool:
        """Carga model_info.json."""
        model_info_path = self.ia_dir / "model_info.json"
        
        if not model_info_path.exists():
            self.logger.warning("model_info.json no encontrado")
            return False
        
        try:
            with open(model_info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            self.input_shape = info.get("input_shape", [28, 28, 1])
            self.labels = info.get("classes", [])
            
            self.logger.info(f"Modelo info cargado: {len(self.labels)} clases")
            return True
        except Exception as e:
            self.logger.warning(f"Error al cargar model_info.json: {e}")
            return False
    
    def _load_model(self) -> bool:
        """Carga el modelo Keras."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.info("TensorFlow no disponible, usando modo demo")
            return False
        
        # Intentar keras primero, luego h5
        model_paths = [
            self.ia_dir / "sketch_classifier_model.keras",
            self.ia_dir / "sketch_classifier_model.h5",
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    self.logger.info(f"Modelo cargado: {model_path.name}")
                    return True
                except Exception as e:
                    self.logger.warning(f"Error al cargar {model_path.name}: {e}")
        
        self.logger.info("No se encontró modelo, usando predicciones demo")
        return False
    
    def predict(self, drawing: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Realiza predicción sobre un drawing.
        
        Args:
            drawing: Array preprocesado del trazo
        
        Returns:
            (etiqueta_top1, confianza, top3_predictions)
        """
        try:
            # Si tenemos modelo real
            if self.model is not None and TENSORFLOW_AVAILABLE:
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
            
            else:
                # Modo demo: predicción aleatoria
                return self._demo_predict()
        
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            return self._demo_predict()
    
    def _demo_predict(self) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Retorna predicción demo aleatoria."""
        import random
        
        if len(self.labels) > 0:
            # Elegir clase aleatoria de las primeras 100
            idx = random.randint(0, min(99, len(self.labels) - 1))
            top1_label = self.labels[idx]
            top1_prob = random.uniform(0.5, 0.95)
            
            # Top-3 aleatorio
            indices = list(range(min(100, len(self.labels))))
            random.shuffle(indices)
            top3_indices = indices[:3]
            top3 = [(self.labels[i], random.uniform(0.1, top1_prob)) for i in top3_indices]
            
            return top1_label, top1_prob, top3
        else:
            return "demo", 0.7, [("demo", 0.7), ("test", 0.5), ("sample", 0.3)]
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Retorna la forma esperada por el modelo."""
        return tuple(self.input_shape)
    
    def get_labels(self) -> List[str]:
        """Retorna la lista de etiquetas."""
        return self.labels.copy()
