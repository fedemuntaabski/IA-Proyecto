"""
Clasificador de sketches mejorado con sistema de fallback.

Este módulo carga el modelo preentrenado y realiza predicciones en imágenes de gestos,
con manejo robusto de errores y fallback cuando TensorFlow no está disponible.
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """
    Interfaz base para clasificadores de sketches.
    """

    @abstractmethod
    def predict(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Realiza predicción en una imagen."""
        pass

    @abstractmethod
    def get_top_prediction(self, image: np.ndarray) -> Tuple[str, float]:
        """Obtiene la predicción con mayor confianza."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el clasificador está disponible."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        pass


class TensorFlowClassifier(BaseClassifier):
    """
    Clasificador que usa TensorFlow/Keras.
    """

    def __init__(self, model_path: str = "IA/sketch_classifier_model.keras",
                 model_info_path: str = "IA/model_info.json"):
        """
        Inicializa el clasificador TensorFlow.

        Args:
            model_path: Ruta al archivo del modelo
            model_info_path: Ruta al archivo de información del modelo
        """
        self.model_path = Path(model_path)
        self.model_info_path = Path(model_info_path)
        self.model = None
        self.model_info = {}
        self.idx_to_label = {}
        self.available = False

        try:
            # Intentar importar TensorFlow
            import tensorflow as tf
            self.tf = tf

            # Cargar información del modelo
            self.model_info = self._load_model_info()

            # Crear mapeo de índice a etiqueta
            self.idx_to_label = {i: label for i, label in enumerate(self.model_info['classes'])}

            # Cargar modelo
            self.model = self._load_model()
            self.available = True

            print("✓ TensorFlowClassifier inicializado correctamente")

        except ImportError as e:
            print(f"⚠ TensorFlow no disponible: {e}")
            print("  El clasificador funcionará en modo fallback")
            self.available = False

        except Exception as e:
            print(f"⚠ Error inicializando TensorFlowClassifier: {e}")
            print("  El clasificador funcionará en modo fallback")
            self.available = False

    def _load_model_info(self) -> dict:
        """
        Carga la información del modelo desde JSON.
        """
        try:
            with open(self.model_info_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠ model_info.json no encontrado en {self.model_info_path}")
            return {
                'num_classes': 228,
                'classes': [f'Class_{i}' for i in range(228)],
                'input_shape': [28, 28, 1],
                'test_accuracy': 0.8
            }
        except Exception as e:
            print(f"⚠ Error cargando model_info.json: {e}")
            return {
                'num_classes': 228,
                'classes': [f'Class_{i}' for i in range(228)],
                'input_shape': [28, 28, 1],
                'test_accuracy': 0.8
            }

    def _load_model(self):
        """
        Carga el modelo de disco.
        """
        if not self.model_path.exists():
            # Intentar con .h5
            alt_path = self.model_path.with_suffix('.h5')
            if alt_path.exists():
                self.model_path = alt_path
            else:
                raise FileNotFoundError(f"Modelo no encontrado en {self.model_path}")

        print(f"Cargando modelo desde {self.model_path}...")
        model = self.tf.keras.models.load_model(str(self.model_path))
        print("✓ Modelo cargado correctamente")
        print(f"  Clases: {len(self.model_info['classes'])}")
        print(f"  Precisión en test: {self.model_info.get('test_accuracy', 'N/A')}")

        return model

    def predict(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Realiza una predicción en una imagen.
        """
        if not self.available or self.model is None:
            return [("No disponible", 0.0)]

        try:
            # Validar entrada
            if image.shape != (28, 28):
                raise ValueError(f"Imagen debe ser 28x28, recibida {image.shape}")

            # Preparar imagen para modelo
            img_batch = image.reshape(1, 28, 28, 1).astype(np.float32)

            # Predicción
            predictions = self.model.predict(img_batch, verbose=0)

            # Obtener índices ordenados por confianza
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]

            # Construir resultado
            results = []
            for idx in top_indices:
                class_name = self.idx_to_label.get(idx, f"Class_{idx}")
                confidence = float(predictions[0][idx])
                results.append((class_name, confidence))

            return results

        except Exception as e:
            print(f"⚠ Error en predicción TensorFlow: {e}")
            return [("Error", 0.0)]

    def get_top_prediction(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Obtiene la predicción con mayor confianza.
        """
        results = self.predict(image, top_k=1)
        if results:
            return results[0]
        return ("Unknown", 0.0)

    def is_available(self) -> bool:
        """Verifica si TensorFlow está disponible."""
        return self.available

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del modelo.
        """
        if not self.available:
            return {
                'status': 'unavailable',
                'reason': 'TensorFlow not available',
                'num_classes': 0,
                'test_accuracy': 0.0
            }

        return {
            'status': 'available',
            'framework': 'TensorFlow/Keras',
            'num_classes': len(self.model_info['classes']),
            'classes_sample': self.model_info['classes'][:10],
            'test_accuracy': self.model_info.get('test_accuracy', 'N/A'),
            'total_parameters': self.model.count_params() if self.model else 0,
            'input_shape': self.model_info.get('input_shape', [28, 28, 1])
        }


class FallbackClassifier(BaseClassifier):
    """
    Clasificador de fallback cuando TensorFlow no está disponible.
    Proporciona predicciones simuladas basadas en análisis de imagen.
    """

    def __init__(self):
        """
        Inicializa el clasificador de fallback.
        """
        # Clases comunes de QuickDraw para simulaciones realistas
        self.common_classes = [
            "circle", "square", "triangle", "star", "heart",
            "house", "tree", "car", "person", "dog",
            "cat", "bird", "fish", "flower", "sun"
        ]

        # Crear mapeo inverso
        self.class_to_idx = {cls: i for i, cls in enumerate(self.common_classes)}

        print("✓ FallbackClassifier inicializado (sin TensorFlow)")

    def _analyze_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analiza características básicas de la imagen para generar predicciones realistas.
        """
        features = {}

        # Estadísticas básicas
        features['mean_intensity'] = np.mean(image)
        features['std_intensity'] = np.std(image)
        features['max_intensity'] = np.max(image)
        features['min_intensity'] = np.min(image)

        # Análisis de forma (simplificado)
        # Contar píxeles activos
        active_pixels = np.sum(image > 0.1)
        total_pixels = image.size
        features['fill_ratio'] = active_pixels / total_pixels

        # Simetría horizontal
        left_half = image[:, :14]
        right_half = np.fliplr(image[:, 14:])
        symmetry_h = np.mean(np.abs(left_half - right_half))
        features['horizontal_symmetry'] = 1.0 / (1.0 + symmetry_h)

        # Simetría vertical
        top_half = image[:14, :]
        bottom_half = np.flipud(image[14:, :])
        symmetry_v = np.mean(np.abs(top_half - bottom_half))
        features['vertical_symmetry'] = 1.0 / (1.0 + symmetry_v)

        # Complejidad (basada en cambios de intensidad)
        grad_x = np.abs(np.gradient(image, axis=1))
        grad_y = np.abs(np.gradient(image, axis=0))
        features['complexity'] = np.mean(grad_x + grad_y)

        return features

    def _generate_prediction_from_features(self, features: Dict[str, float]) -> str:
        """
        Genera una predicción basada en las características analizadas.
        """
        # Lógica heurística para mapear características a clases comunes

        fill_ratio = features['fill_ratio']
        symmetry_h = features['horizontal_symmetry']
        symmetry_v = features['vertical_symmetry']
        complexity = features['complexity']

        # Formas cerradas (alto fill_ratio, alta simetría)
        if fill_ratio > 0.3 and (symmetry_h > 0.7 or symmetry_v > 0.7):
            if symmetry_h > 0.8 and symmetry_v > 0.8:
                return "circle"  # Muy simétrico
            elif symmetry_h > symmetry_v:
                return "square"  # Más simétrico horizontal
            else:
                return "triangle"  # Más simétrico vertical

        # Formas abiertas o lineales (bajo fill_ratio)
        elif fill_ratio < 0.2:
            if complexity > 0.1:
                return "star"  # Alta complejidad
            else:
                return "line"  # Baja complejidad

        # Formas orgánicas
        else:
            # Basado en complejidad y simetría
            if complexity > 0.15:
                return "tree"  # Alta complejidad
            elif symmetry_h > 0.6:
                return "heart"  # Simétrico horizontal
            else:
                return "house"  # Caso general

    def predict(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Genera predicciones simuladas basadas en análisis de imagen.
        """
        try:
            # Analizar características
            features = self._analyze_image_features(image)

            # Generar predicción principal
            main_prediction = self._generate_prediction_from_features(features)

            # Generar confianza basada en "certeza" de las características
            base_confidence = 0.4 + np.random.random() * 0.3  # 0.4-0.7

            # Modificar confianza basada en características
            confidence_modifier = (
                features['horizontal_symmetry'] * 0.2 +
                features['vertical_symmetry'] * 0.2 +
                (1.0 - abs(features['fill_ratio'] - 0.5)) * 0.1  # Mejor cuando fill_ratio ~ 0.5
            )

            main_confidence = min(0.85, base_confidence + confidence_modifier)

            # Crear lista de predicciones
            results = [(main_prediction, main_confidence)]

            # Agregar predicciones alternativas con menor confianza
            remaining_classes = [cls for cls in self.common_classes if cls != main_prediction]
            np.random.shuffle(remaining_classes)

            for i, alt_class in enumerate(remaining_classes[:top_k-1]):
                # Confianza decreciente para alternativas
                alt_confidence = main_confidence * (0.3 - i * 0.1) + np.random.random() * 0.1
                alt_confidence = max(0.05, alt_confidence)
                results.append((alt_class, alt_confidence))

            return results[:top_k]

        except Exception as e:
            print(f"⚠ Error en predicción fallback: {e}")
            return [("Error", 0.0)]

    def get_top_prediction(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Obtiene la predicción con mayor confianza.
        """
        results = self.predict(image, top_k=1)
        if results:
            return results[0]
        return ("Unknown", 0.0)

    def is_available(self) -> bool:
        """El clasificador de fallback siempre está disponible."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del modelo de fallback.
        """
        return {
            'status': 'available',
            'framework': 'Fallback (sin TensorFlow)',
            'num_classes': len(self.common_classes),
            'classes_sample': self.common_classes[:10],
            'test_accuracy': 'Simulado',
            'total_parameters': 0,
            'input_shape': [28, 28, 1],
            'description': 'Clasificador heurístico basado en análisis de imagen'
        }


class SketchClassifier:
    """
    Clasificador de sketches con sistema de fallback automático.

    Esta clase intenta usar TensorFlow primero, y automáticamente
    cambia a un clasificador de fallback si TensorFlow no está disponible.
    """

    def __init__(self, model_path: str = "IA/sketch_classifier_model.keras",
                 model_info_path: str = "IA/model_info.json",
                 enable_fallback: bool = True):
        """
        Inicializa el clasificador con sistema de fallback.

        Args:
            model_path: Ruta al modelo TensorFlow
            model_info_path: Ruta a la información del modelo
            enable_fallback: Si True, usa fallback cuando TensorFlow falla
        """
        self.enable_fallback = enable_fallback

        # Intentar inicializar clasificador TensorFlow
        self.tf_classifier = TensorFlowClassifier(model_path, model_info_path)

        # Inicializar fallback si está habilitado
        self.fallback_classifier = FallbackClassifier() if enable_fallback else None

        # Determinar cuál usar
        if self.tf_classifier.is_available():
            self.active_classifier = self.tf_classifier
            self.mode = "tensorflow"
            print("✓ SketchClassifier usando TensorFlow")
        elif self.fallback_classifier and enable_fallback:
            self.active_classifier = self.fallback_classifier
            self.mode = "fallback"
            print("✓ SketchClassifier usando modo fallback")
        else:
            self.active_classifier = None
            self.mode = "unavailable"
            print("⚠ SketchClassifier no disponible")

    def predict(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Realiza una predicción usando el clasificador activo.
        """
        if self.active_classifier:
            results = self.active_classifier.predict(image, top_k)

            # Agregar indicador de modo en el primer resultado
            if results and self.mode == "fallback":
                # Modificar el primer resultado para indicar que es fallback
                original_class, confidence = results[0]
                results[0] = (f"{original_class}*", confidence)

            return results

        return [("No disponible", 0.0)]

    def get_top_prediction(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Obtiene la predicción con mayor confianza.
        """
        if self.active_classifier:
            class_name, confidence = self.active_classifier.get_top_prediction(image)

            if self.mode == "fallback":
                class_name = f"{class_name}*"

            return class_name, confidence

        return ("No disponible", 0.0)

    def is_confident_prediction(self, image: np.ndarray, confidence_threshold: float = 0.5) -> bool:
        """
        Verifica si la predicción supera un umbral de confianza.
        """
        _, confidence = self.get_top_prediction(image)
        return confidence >= confidence_threshold

    def is_available(self) -> bool:
        """Verifica si algún clasificador está disponible."""
        return self.active_classifier is not None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del clasificador activo.
        """
        if self.active_classifier:
            info = self.active_classifier.get_model_info()
            info['active_mode'] = self.mode
            info['fallback_available'] = self.fallback_classifier is not None
            return info

        return {
            'status': 'unavailable',
            'active_mode': 'none',
            'reason': 'No classifier available'
        }

    def switch_to_fallback(self) -> bool:
        """
        Fuerza el cambio al modo fallback si está disponible.
        """
        if self.fallback_classifier:
            self.active_classifier = self.fallback_classifier
            self.mode = "fallback"
            print("✓ Cambiado a modo fallback")
            return True

        print("⚠ Modo fallback no disponible")
        return False

    def switch_to_tensorflow(self) -> bool:
        """
        Fuerza el cambio al modo TensorFlow si está disponible.
        """
        if self.tf_classifier.is_available():
            self.active_classifier = self.tf_classifier
            self.mode = "tensorflow"
            print("✓ Cambiado a modo TensorFlow")
            return True

        print("⚠ TensorFlow no disponible")
        return False


# Funciones de compatibilidad para código existente
def create_classifier(model_path: str = "IA/sketch_classifier_model.keras",
                     model_info_path: str = "IA/model_info.json") -> SketchClassifier:
    """
    Crea un clasificador con configuración por defecto.
    Función de compatibilidad para código existente.
    """
    return SketchClassifier(model_path, model_info_path)
