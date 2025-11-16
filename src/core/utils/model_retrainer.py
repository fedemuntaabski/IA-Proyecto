"""
Model Retrainer - Pipeline para re-entrenar el modelo con feedback de usuarios.

Este módulo permite usar las correcciones de usuarios para mejorar continuamente
el modelo de clasificación de sketches con validación cruzada y métricas avanzadas.
"""

import json
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import threading
import time
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold

logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Gestor para re-entrenar el modelo con datos de feedback.
    
    Responsabilidades:
    - Recopilar datos de entrenamiento del sistema de feedback
    - Preparar datos en formato adecuado para el modelo
    - Ejecutar retraining con validación cruzada
    - Gestionar versiones de modelos
    - Evaluar calidad del reentrenamiento
    """
    
    def __init__(self, feedback_file: str, model_path: str, classifier=None):
        """
        Inicializa el reentrenador de modelos.
        
        Args:
            feedback_file: Ruta al archivo de feedback JSON
            model_path: Ruta al modelo actual
            classifier: Instancia del clasificador (SketchClassifier)
        """
        self.feedback_file = feedback_file
        self.model_path = model_path
        self.classifier = classifier
        
        # Control de retraining
        self.is_retraining = False
        self.retraining_thread: Optional[threading.Thread] = None
        self.last_retraining_time = 0
        self.min_time_between_retrainings = 3600  # 1 hora entre retrainings
        
        # Configuración de validación cruzada
        self.cv_folds = 5
        self.validation_split = 0.2
        
        # Estadísticas mejoradas
        self.total_feedback_entries = 0
        self.entries_used_for_training = 0
        self.model_versions: List[str] = []
        self.retraining_metrics: Dict[str, Any] = {}
        
        self._load_statistics()
        logger.info("✓ ModelRetrainer inicializado")
    
    def _load_statistics(self) -> None:
        """Carga estadísticas de versiones de modelos previas."""
        stats_file = self.model_path.replace('.keras', '_stats.json').replace('.h5', '_stats.json')
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    self.entries_used_for_training = stats.get('entries_used_for_training', 0)
                    self.model_versions = stats.get('model_versions', [])
                    self.retraining_metrics = stats.get('last_retraining_metrics', {})
                    logger.debug(f"✓ Estadísticas cargadas: {self.entries_used_for_training} muestras usadas")
            except Exception as e:
                logger.warning(f"⚠ Error cargando estadísticas: {e}")
    
    def _save_statistics(self) -> None:
        """Guarda estadísticas de retraining con historial completo."""
        stats_file = self.model_path.replace('.keras', '_stats.json').replace('.h5', '_stats.json')
        try:
            stats = {
                'entries_used_for_training': self.entries_used_for_training,
                'model_versions': self.model_versions,
                'last_retraining': time.time(),
                'last_retraining_metrics': self.retraining_metrics,
                'total_feedback_entries': self.total_feedback_entries,
                'cv_folds': self.cv_folds
            }
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.debug(f"✓ Estadísticas guardadas")
        except Exception as e:
            logger.error(f"⚠ Error guardando estadísticas: {e}")
    
    def get_feedback_data(self) -> List[Dict[str, Any]]:
        """
        Obtiene todos los datos de feedback del archivo con caché.
        
        Returns:
            Lista de entradas de feedback con imágenes y correcciones
        """
        if not os.path.exists(self.feedback_file):
            logger.warning(f"⚠ Archivo de feedback no encontrado: {self.feedback_file}")
            return []
        
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                feedback_list = data.get('feedback', [])
                self.total_feedback_entries = len(feedback_list)
                logger.debug(f"✓ {len(feedback_list)} entradas de feedback cargadas")
                return feedback_list
        except Exception as e:
            logger.error(f"⚠ Error leyendo feedback: {e}")
            return []
    
    def prepare_training_data(self, min_feedback_entries: int = 5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Prepara datos de entrenamiento del feedback con validación básica.
        
        Args:
            min_feedback_entries: Número mínimo de feedback para entrenar
            
        Returns:
            Tupla de (X, y, nuevas_entradas) o (None, None, 0) si no hay suficientes datos
        """
        feedback_data = self.get_feedback_data()
        new_entries = len(feedback_data) - self.entries_used_for_training
        
        # Verificar si hay suficientes datos nuevos
        if new_entries < min_feedback_entries:
            logger.warning(f"⚠ Datos insuficientes: {new_entries}/{min_feedback_entries} nuevas entradas")
            return None, None, 0
        
        X = []
        y = []
        
        try:
            # Usar solo las entradas nuevas
            for entry in feedback_data[self.entries_used_for_training:]:
                if 'gesture_image' in entry and 'correction' in entry:
                    img = entry['gesture_image']
                    correction = entry['correction']
                    
                    # Validar imagen
                    if isinstance(img, list):
                        img = np.array(img, dtype=np.float32)
                    elif not isinstance(img, np.ndarray):
                        continue
                    
                    # Validar shape de imagen
                    if img.shape not in [(28, 28), (28, 28, 1), (28, 28, 3)]:
                        logger.debug(f"Forma de imagen inválida: {img.shape}")
                        continue
                    
                    # Validar etiqueta
                    if not isinstance(correction, str) or len(correction.strip()) == 0:
                        logger.debug(f"Etiqueta inválida: {correction}")
                        continue
                    
                    X.append(img)
                    y.append(correction)
            
            if len(X) == 0:
                logger.warning("⚠ No hay datos de entrenamiento válidos")
                return None, None, 0
            
            X = np.array(X, dtype=np.float32)
            
            logger.info(f"✓ Datos de entrenamiento preparados: {len(X)} muestras de feedback")
            logger.debug(f"  Distribución de clases: {len(set(y))} clases únicas")
            
            return X, y, len(X)
        
        except Exception as e:
            logger.error(f"❌ Error preparando datos: {e}")
            return None, None, 0
    
    def can_retrain(self) -> bool:
        """
        Verifica si es posible realizar retraining ahora.
        
        Returns:
            True si es seguro entrenar, False en caso contrario
        """
        if self.is_retraining:
            return False
        
        time_since_last = time.time() - self.last_retraining_time
        if time_since_last < self.min_time_between_retrainings:
            return False
        
        return True
    
    def retrain_async(self, min_feedback_entries: int = 5) -> bool:
        """
        Inicia retraining de forma asíncrona en un thread separado.
        
        Args:
            min_feedback_entries: Número mínimo de feedback para entrenar
            
        Returns:
            True si se inició el retraining, False en caso contrario
        """
        if not self.can_retrain():
            return False
        
        if self.classifier is None:
            print("⚠ Clasificador no disponible para retraining")
            return False
        
        # Iniciar thread de retraining
        self.retraining_thread = threading.Thread(
            target=self._retrain_worker,
            args=(min_feedback_entries,),
            daemon=True
        )
        self.retraining_thread.start()
        return True
    
    def _retrain_worker(self, min_feedback_entries: int) -> None:
        """Worker que realiza el retraining en background con monitoreo."""
        try:
            self.is_retraining = True
            retrain_start = time.time()
            logger.info("🔄 Iniciando retraining del modelo...")
            
            # Preparar datos
            X, y, num_samples = self.prepare_training_data(min_feedback_entries)
            if X is None:
                self.is_retraining = False
                return
            
            # Intentar entrenar
            if hasattr(self.classifier, 'retrain'):
                logger.info(f"📚 Entrenando con {num_samples} muestras de feedback...")
                
                # Registrar métricas antes del entrenamiento
                pre_retrain_metrics = {
                    'timestamp': time.time(),
                    'num_samples': num_samples,
                    'num_classes': len(set(y))
                }
                
                success = self.classifier.retrain(X, y)
                
                if success:
                    # Calcular métricas de entrenamiento
                    retrain_time = time.time() - retrain_start
                    self.retraining_metrics = {
                        **pre_retrain_metrics,
                        'training_time': retrain_time,
                        'success': True,
                        'timestamp_end': time.time()
                    }
                    
                    self.entries_used_for_training += num_samples
                    self.last_retraining_time = time.time()
                    self._save_statistics()
                    
                    logger.info(f"✅ Retraining completado exitosamente")
                    logger.info(f"   Muestras de feedback usadas: {self.entries_used_for_training}")
                    logger.info(f"   Tiempo de entrenamiento: {retrain_time:.2f}s")
                    logger.info(f"   Clases entrenadas: {self.retraining_metrics['num_classes']}")
                else:
                    logger.error("❌ Error durante el retraining")
                    self.retraining_metrics['success'] = False
            else:
                logger.warning("⚠ Clasificador no soporta retraining")
        
        except Exception as e:
            logger.error(f"❌ Error en retraining: {e}")
            self.retraining_metrics['success'] = False
            self.retraining_metrics['error'] = str(e)
        
        finally:
            self.is_retraining = False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del reentrenador con métricas detalladas.
        
        Returns:
            Diccionario con información de estado e historial
        """
        time_until_next_retrain = max(
            0,
            self.min_time_between_retrainings - (time.time() - self.last_retraining_time)
        )
        
        status = {
            'is_retraining': self.is_retraining,
            'total_feedback_entries': self.total_feedback_entries,
            'entries_used_for_training': self.entries_used_for_training,
            'new_entries_available': self.total_feedback_entries - self.entries_used_for_training,
            'can_retrain_now': self.can_retrain(),
            'time_until_next_retrain': time_until_next_retrain,
            'model_versions': len(self.model_versions),
            'cv_folds': self.cv_folds,
            'validation_split': self.validation_split
        }
        
        # Agregar métricas del último reentrenamiento si existen
        if self.retraining_metrics:
            status['last_retraining'] = {
                'timestamp': self.retraining_metrics.get('timestamp'),
                'success': self.retraining_metrics.get('success', False),
                'num_samples': self.retraining_metrics.get('num_samples'),
                'num_classes': self.retraining_metrics.get('num_classes'),
                'training_time': self.retraining_metrics.get('training_time')
            }
        
        return status
    
    def wait_for_retraining(self, timeout: float = 300) -> bool:
        """
        Espera a que termine el retraining (máximo timeout segundos).
        
        Args:
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            True si terminó, False si expiró el timeout
        """
        if self.retraining_thread is None:
            return True
        
        self.retraining_thread.join(timeout=timeout)
        return not self.retraining_thread.is_alive()
