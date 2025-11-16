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
    import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
except Exception as e:
    # Silenciar errores de importación como AttributeError
    print(f"AVISO: Error al importar TensorFlow: {e}")
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None


class SketchClassifier:
    """Carga y realiza inferencia con el modelo de clasificación de sketches."""
    
    def __init__(self, ia_dir: str, logger: logging.Logger, demo_mode: bool = True, config: Dict[str, Any] = None):
        """
        Inicializa el clasificador.
        
        Args:
            ia_dir: Ruta a la carpeta IA con modelo
            logger: Logger
            demo_mode: Usar predicciones demo si no está disponible TensorFlow
            config: Configuración del modelo
        """
        self.ia_dir = Path(ia_dir)
        self.logger = logger
        self.demo_mode = demo_mode
        self.config = config or {}
        
        self.model = None
        self.input_shape = [28, 28, 1]
        self.labels = []
        self.use_quantized = self.config.get("use_quantized_model", True)
        self.prefer_gpu = self.config.get("prefer_gpu", True)
        
        try:
            if not self.ia_dir.exists():
                self.logger.warning(f"Carpeta IA no encontrada: {ia_dir} - usando modo demo")
                self.demo_mode = True
                self.labels = ["demo"]  # Etiqueta por defecto para demo
                return
            
            # Cargar metadata con fallback
            if not self._load_model_info():
                self.logger.warning("No se pudo cargar model_info.json - usando valores por defecto")
                self.input_shape = [28, 28, 1]
                self.labels = ["demo"]  # Al menos una etiqueta para evitar errores
            
            # Cargar modelo si TensorFlow disponible y no demo_mode
            if TENSORFLOW_AVAILABLE and not self.demo_mode:
                if not self._load_model():
                    self.logger.warning("No se pudo cargar modelo - cambiando a modo demo")
                    self.demo_mode = True
            else:
                self.logger.info("Modo demo activado")
                self.demo_mode = True
        except Exception as e:
            self.logger.error(f"Error crítico en inicialización del clasificador: {e} - usando modo demo")
            self.demo_mode = True
            self.labels = ["demo"]
    
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
        """Carga el modelo Keras con optimizaciones de rendimiento."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.info("TensorFlow no disponible, usando modo demo")
            return False
        
        # Configurar dispositivo
        device = self._get_device()
        self.logger.info(f"Usando dispositivo para inferencia: {device}")
        
        # Intentar modelos en orden de preferencia: cuantizado primero, luego normal
        model_paths = []
        if self.use_quantized:
            model_paths.extend([
                self.ia_dir / "sketch_classifier_model_quantized.tflite",
                self.ia_dir / "sketch_classifier_model_quantized.keras",
                self.ia_dir / "sketch_classifier_model_quantized.h5",
            ])
        
        # Modelos normales
        model_paths.extend([
            self.ia_dir / "sketch_classifier_model.keras",
            self.ia_dir / "sketch_classifier_model.h5",
        ])
        
        for model_path in model_paths:
            if model_path.exists():
                try:
                    if model_path.suffix == ".tflite":
                        # Modelo cuantizado TFLite
                        self.model = self._load_tflite_model(model_path, device)
                    else:
                        # Modelo Keras normal
                        with tf.device(device):
                            if keras:
                                self.model = keras.models.load_model(model_path)
                            else:
                                self.model = tf.keras.models.load_model(model_path)
                    
                    model_type = "cuantizado (TFLite)" if "quantized" in model_path.name else "normal"
                    self.logger.info(f"Modelo {model_type} cargado: {model_path.name}")
                    return True
                except Exception as e:
                    self.logger.warning(f"Error al cargar {model_path.name}: {e}")
        
        self.logger.info("No se encontró modelo, usando predicciones demo")
        return False
    
    def _get_device(self) -> str:
        """Determina el dispositivo óptimo para inferencia."""
        if not TENSORFLOW_AVAILABLE or not self.prefer_gpu:
            return "/CPU:0"
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and len(gpus) > 0:
                # Verificar que la GPU esté disponible
                tf.config.experimental.set_memory_growth(gpus[0], True)
                self.logger.info(f"GPU disponible: {gpus[0].name}")
                return "/GPU:0"
            else:
                self.logger.info("GPU no disponible, usando CPU")
                return "/CPU:0"
        except Exception as e:
            self.logger.warning(f"Error configurando GPU: {e}, usando CPU")
            return "/CPU:0"
    
    def _load_tflite_model(self, model_path: Path, device: str):
        """Carga modelo TFLite cuantizado."""
        try:
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            self.logger.info("Modelo TFLite cargado exitosamente")
            return interpreter
        except Exception as e:
            self.logger.error(f"Error cargando modelo TFLite: {e}")
            raise
    
    def predict(self, drawing: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Realiza predicción sobre un drawing.
        
        Args:
            drawing: Array preprocesado del trazo
        
        Returns:
            (etiqueta_top1, confianza, top3_predictions)
        """
        try:
            # Si no hay modelo, usar demo
            if self.model is None or not TENSORFLOW_AVAILABLE:
                return self._demo_predict()

            # Predicción inicial
            if hasattr(self.model, 'invoke'):
                result = self._predict_tflite(drawing)
            else:
                result = self._predict_keras(drawing)

            # Si la confianza es baja, intentar la versión invertida de la imagen
            top1_label, top1_prob, top3 = result
            low_conf_threshold = float(self.config.get("low_conf_threshold", 0.35))
            if top1_prob < low_conf_threshold:
                try:
                    # Crear variante invertida (1 - pixel)
                    inverted = (1.0 - drawing).astype(drawing.dtype)
                    if hasattr(self.model, 'invoke'):
                        inv_result = self._predict_tflite(inverted)
                    else:
                        inv_result = self._predict_keras(inverted)

                    # Elegir la predicción con mayor probabilidad top1
                    if inv_result[1] > top1_prob:
                        result = inv_result
                        top1_label, top1_prob, top3 = result
                except Exception as e:
                    self.logger.debug(f"Error al predecir con variante invertida: {e}")

            # Ensemble de augmentaciones: rotaciones pequeñas y flips para mejorar robustez
            try:
                if float(self.config.get("use_ensemble", 1)) == 1:
                    aug_rots = [0, -10, 10]
                    candidates = [result]
                    for ang in aug_rots:
                        if ang == 0:
                            continue
                        # rotar la imagen en el centro usando OpenCV
                        try:
                            import cv2
                            img2d = drawing.squeeze()
                            h, w = img2d.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, ang, 1.0)
                            rot = cv2.warpAffine((img2d * 255).astype('uint8'), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
                            rot = rot.astype('float32') / 255.0
                            rot_img = np.expand_dims(rot, axis=-1)
                        except Exception:
                            # Fallback: skip rotation if OpenCV no disponible
                            continue
                        if hasattr(self.model, 'invoke'):
                            r = self._predict_tflite(rot_img)
                        else:
                            r = self._predict_keras(rot_img)
                        candidates.append(r)

                    # también probar versión invertida de la original si no ya probada
                    inv = (1.0 - drawing).astype(drawing.dtype)
                    if hasattr(self.model, 'invoke'):
                        inv_r = self._predict_tflite(inv)
                    else:
                        inv_r = self._predict_keras(inv)
                    candidates.append(inv_r)

                    # elegir la predicción con mayor top1 prob
                    best = result
                    for cand in candidates:
                        if cand[1] > best[1]:
                            best = cand
                    result = best
                    top1_label, top1_prob, top3 = result
            except Exception as e:
                # Ignorar fallos de ensemble
                self.logger.debug(f"Ensemble prediction failed: {e}")

            return result

        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            return self._demo_predict()
    
    def _predict_keras(self, drawing: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Predicción con modelo Keras."""
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
    
    def _predict_tflite(self, drawing: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Predicción con modelo TFLite cuantizado."""
        try:
            interpreter = self.model
            
            # Obtener detalles de entrada/salida
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Preparar input
            if len(drawing.shape) == 3:
                drawing = np.expand_dims(drawing, axis=0)
            
            # Asegurar tipo correcto
            input_dtype = input_details[0]['dtype']
            drawing = drawing.astype(input_dtype)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], drawing)
            
            # Invoke
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            probs = output_data[0]
            
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
            
        except Exception as e:
            self.logger.error(f"Error en predicción TFLite: {e}")
            raise
    
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
