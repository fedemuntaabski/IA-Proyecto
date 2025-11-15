"""
Clasificador de sketches usando el modelo entrenado.

Este módulo carga el modelo preentrenado y realiza predicciones en imágenes de gestos.
"""

import json
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import tensorflow as tf


class SketchClassifier:
    """
    Clasifica sketches usando el modelo CNN entrenado.
    
    Atributos:
        model: Modelo TensorFlow/Keras cargado
        model_info: Información del modelo (clases, parámetros)
        idx_to_label: Mapeo de índice a nombre de clase
    """
    
    def __init__(self, model_path: str = "IA/sketch_classifier_model.keras", 
                 model_info_path: str = "IA/model_info.json"):
        """
        Inicializa el clasificador cargando el modelo.
        
        Args:
            model_path: Ruta al archivo del modelo (.keras o .h5)
            model_info_path: Ruta al archivo de información del modelo
        """
        self.model_path = Path(model_path)
        self.model_info_path = Path(model_info_path)
        
        # Cargar información del modelo
        self.model_info = self._load_model_info()
        
        # Crear mapeo de índice a etiqueta
        self.idx_to_label = {i: label for i, label in enumerate(self.model_info['classes'])}
        
        # Cargar modelo
        self.model = self._load_model()
    
    def _load_model_info(self) -> dict:
        """
        Carga la información del modelo desde JSON.
        
        Returns:
            Diccionario con información del modelo
        """
        try:
            with open(self.model_info_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: model_info.json no encontrado en {self.model_info_path}")
            return {
                'num_classes': 228,
                'classes': [f'Class_{i}' for i in range(228)],
                'input_shape': [28, 28, 1],
                'test_accuracy': 0.8
            }
    
    def _load_model(self):
        """
        Carga el modelo de disco.
        
        Returns:
            Modelo TensorFlow cargado
        """
        if not self.model_path.exists():
            # Intentar con .h5
            alt_path = self.model_path.with_suffix('.h5')
            if alt_path.exists():
                self.model_path = alt_path
            else:
                raise FileNotFoundError(f"Modelo no encontrado en {self.model_path}")
        
        print(f"Cargando modelo desde {self.model_path}...")
        model = tf.keras.models.load_model(str(self.model_path))
        print(f"✓ Modelo cargado correctamente")
        print(f"  Clases: {len(self.model_info['classes'])}")
        print(f"  Precisión en test: {self.model_info.get('test_accuracy', 'N/A')}")
        
        return model
    
    def predict(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Realiza una predicción en una imagen.
        
        Args:
            image: Array numpy 28x28 con valores 0-1
            top_k: Número de top predicciones a retornar
            
        Returns:
            Lista de tuplas (clase, confianza) ordenadas por confianza
        """
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
            class_name = self.idx_to_label[idx]
            confidence = float(predictions[0][idx])
            results.append((class_name, confidence))
        
        return results
    
    def get_top_prediction(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Obtiene la predicción con mayor confianza.
        
        Args:
            image: Array numpy 28x28 con valores 0-1
            
        Returns:
            Tupla (clase, confianza)
        """
        results = self.predict(image, top_k=1)
        if results:
            return results[0]
        return ("Unknown", 0.0)
    
    def is_confident_prediction(self, image: np.ndarray, confidence_threshold: float = 0.5) -> bool:
        """
        Verifica si la predicción supera un umbral de confianza.
        
        Args:
            image: Array numpy 28x28
            confidence_threshold: Umbral mínimo de confianza (0-1)
            
        Returns:
            True si confianza >= umbral, False en caso contrario
        """
        _, confidence = self.get_top_prediction(image)
        return confidence >= confidence_threshold
    
    def get_model_info(self) -> dict:
        """
        Retorna información del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            'num_classes': len(self.model_info['classes']),
            'classes_sample': self.model_info['classes'][:10],  # Primeras 10 clases
            'test_accuracy': self.model_info.get('test_accuracy', 'N/A'),
            'total_parameters': self.model.count_params(),
            'input_shape': self.model_info.get('input_shape', [28, 28, 1])
        }


if __name__ == "__main__":
    # Test básico
    classifier = SketchClassifier()
    
    # Crear imagen de prueba (valores aleatorios)
    test_image = np.random.rand(28, 28).astype(np.float32)
    
    # Predecir
    print("\nPredicciones:")
    results = classifier.predict(test_image, top_k=5)
    for class_name, confidence in results:
        print(f"  {class_name}: {confidence:.2%}")
    
    # Información del modelo
    print("\nInformación del modelo:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
