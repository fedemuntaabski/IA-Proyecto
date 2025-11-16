"""
Gestión de aceleración GPU para TensorFlow/Keras.

Este módulo configura y gestiona la aceleración GPU para mejorar
el rendimiento de las operaciones de machine learning.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import psutil
import platform

logger = logging.getLogger(__name__)

class GPUManager:
    """
    Administrador de aceleración GPU para TensorFlow.

    Detecta GPUs disponibles, configura memoria, y optimiza
    el rendimiento de las operaciones de ML.
    """

    def __init__(self):
        """Inicializa el administrador de GPU."""
        self.gpu_available = False
        self.gpu_info = {}
        self.memory_configured = False
        self.tensorflow_available = False

        # Detectar TensorFlow
        self._detect_tensorflow()

        # Detectar GPUs
        self._detect_gpus()

        # Configurar GPU si está disponible
        if self.gpu_available:
            self._configure_gpu()

    def _detect_tensorflow(self) -> None:
        """Detecta si TensorFlow está disponible."""
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            self.tf_version = tf.__version__
            logger.info(f"✓ TensorFlow {self.tf_version} detectado")
        except ImportError:
            self.tensorflow_available = False
            logger.warning("⚠ TensorFlow no está disponible")
        except Exception as e:
            self.tensorflow_available = False
            logger.error(f"⚠ Error detectando TensorFlow: {e}")

    def _detect_gpus(self) -> None:
        """Detecta GPUs disponibles en el sistema."""
        if not self.tensorflow_available:
            return

        try:
            import tensorflow as tf

            # Detectar GPUs físicas
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu_available = True
                self.gpu_info = {
                    'count': len(gpus),
                    'devices': [gpu.name for gpu in gpus],
                    'memory_limit': None,
                    'memory_growth': None
                }
                logger.info(f"✓ {len(gpus)} GPU(s) detectada(s): {[gpu.name for gpu in gpus]}")
            else:
                self.gpu_available = False
                logger.info("ℹ No se detectaron GPUs")

        except Exception as e:
            self.gpu_available = False
            logger.error(f"⚠ Error detectando GPUs: {e}")

    def _configure_gpu(self) -> None:
        """Configura la GPU para uso óptimo."""
        if not self.tensorflow_available or not self.gpu_available:
            return

        try:
            import tensorflow as tf

            # Configurar crecimiento de memoria
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"✓ Configurado crecimiento de memoria para {gpu.name}")
                except RuntimeError as e:
                    logger.warning(f"⚠ No se pudo configurar crecimiento de memoria: {e}")

            # Configurar política de memoria
            if hasattr(tf.config.experimental, 'set_visible_devices'):
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    try:
                        tf.config.set_visible_devices(gpus[0], 'GPU')  # Usar primera GPU
                        logger.info("✓ GPU primaria configurada como visible")
                    except RuntimeError as e:
                        logger.warning(f"⚠ Error configurando dispositivo visible: {e}")

            self.memory_configured = True
            logger.info("✓ Configuración de GPU completada")

        except Exception as e:
            logger.error(f"⚠ Error configurando GPU: {e}")

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada sobre las GPUs disponibles.

        Returns:
            Diccionario con información de GPU
        """
        if not self.tensorflow_available:
            return {'status': 'tensorflow_unavailable'}

        try:
            import tensorflow as tf

            info = {
                'tensorflow_version': self.tf_version,
                'gpu_available': self.gpu_available,
                'gpu_count': len(tf.config.list_physical_devices('GPU')) if self.gpu_available else 0,
                'cpu_count': os.cpu_count(),
                'platform': platform.system(),
                'memory_configured': self.memory_configured
            }

            if self.gpu_available:
                # Información detallada de GPU
                gpus = tf.config.list_physical_devices('GPU')
                gpu_details = []

                for i, gpu in enumerate(gpus):
                    gpu_detail = {
                        'index': i,
                        'name': gpu.name,
                        'device_type': gpu.device_type
                    }
                    gpu_details.append(gpu_detail)

                info['gpu_details'] = gpu_details

                # Información de memoria (si está disponible)
                try:
                    if hasattr(tf.config.experimental, 'get_memory_info'):
                        memory_info = tf.config.experimental.get_memory_info('GPU:0')
                        info['memory_info'] = memory_info
                except:
                    pass

            return info

        except Exception as e:
            logger.error(f"Error obteniendo información de GPU: {e}")
            return {'status': 'error', 'error': str(e)}

    def optimize_for_inference(self) -> None:
        """
        Optimiza TensorFlow para inferencia rápida.

        Configura opciones específicas para mejorar el rendimiento
        de las operaciones de predicción.
        """
        if not self.tensorflow_available:
            return

        try:
            import tensorflow as tf

            # Configuraciones de optimización
            tf.config.optimizer.set_jit(True)  # XLA compilation
            tf.config.optimizer.set_experimental_options({
                'auto_mixed_precision': True,
                'layout_optimizer': True,
                'constant_folding': True,
                'shape_optimization': True,
                'remapping': True,
                'arithmetic_optimization': True,
                'dependency_optimization': True,
                'loop_optimization': True,
                'function_optimization': True,
                'debug_stripper': True,
                'disable_model_pruning': False,
                'scoped_allocator_optimization': True,
                'pin_to_host_optimization': True,
                'implementation_selector': True,
                'disable_meta_optimizer': False,
            })

            logger.info("✓ Optimizaciones de inferencia aplicadas")

        except Exception as e:
            logger.warning(f"⚠ Error aplicando optimizaciones: {e}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Obtiene información de uso de memoria.

        Returns:
            Diccionario con información de memoria
        """
        memory_info = {
            'system_memory': self._get_system_memory(),
            'gpu_memory': {}
        }

        if self.tensorflow_available and self.gpu_available:
            try:
                import tensorflow as tf

                # Intentar obtener información de memoria GPU
                if hasattr(tf.config.experimental, 'get_memory_info'):
                    gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                    memory_info['gpu_memory'] = {
                        'current': gpu_memory.current,
                        'peak': gpu_memory.peak,
                        'limit': getattr(gpu_memory, 'limit', None)
                    }
            except Exception as e:
                logger.debug(f"No se pudo obtener información de memoria GPU: {e}")

        return memory_info

    def _get_system_memory(self) -> Dict[str, Any]:
        """Obtiene información de memoria del sistema."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percentage': memory.percent
            }
        except Exception as e:
            logger.debug(f"Error obteniendo memoria del sistema: {e}")
            return {'error': str(e)}

    def is_gpu_available(self) -> bool:
        """Verifica si GPU está disponible y configurada."""
        return self.gpu_available and self.memory_configured

    def get_performance_tips(self) -> List[str]:
        """
        Obtiene consejos de rendimiento basados en la configuración actual.

        Returns:
            Lista de consejos para mejorar el rendimiento
        """
        tips = []

        if not self.tensorflow_available:
            tips.append("Instalar TensorFlow para mejorar el rendimiento de ML")
            return tips

        if not self.gpu_available:
            tips.append("No se detectó GPU - considera usar una GPU NVIDIA con CUDA")
            tips.append("Asegúrate de que los drivers de GPU estén instalados")
        else:
            tips.append("GPU detectada - el rendimiento debería ser óptimo")

        if self.memory_configured:
            tips.append("Configuración de memoria GPU aplicada correctamente")
        else:
            tips.append("Configuración de memoria GPU podría optimizarse")

        # Consejos generales
        tips.extend([
            "Usa lotes pequeños para inferencia en tiempo real",
            "Considera cuantización del modelo para mejor rendimiento",
            "Monitorea el uso de memoria durante operación"
        ])

        return tips

    def benchmark_gpu(self, duration_seconds: int = 5) -> Dict[str, Any]:
        """
        Ejecuta un benchmark simple de GPU.

        Args:
            duration_seconds: Duración del benchmark en segundos

        Returns:
            Resultados del benchmark
        """
        if not self.tensorflow_available:
            return {'status': 'tensorflow_unavailable'}

        try:
            import tensorflow as tf
            import time
            import numpy as np

            start_time = time.time()

            # Benchmark simple: multiplicación de matrices
            operations = 0
            while time.time() - start_time < duration_seconds:
                # Crear matrices aleatorias
                a = tf.random.normal([100, 100])
                b = tf.random.normal([100, 100])

                # Operación de multiplicación
                c = tf.matmul(a, b)
                operations += 1

                # Forzar ejecución
                _ = c.numpy()

            end_time = time.time()
            elapsed = end_time - start_time

            return {
                'status': 'completed',
                'duration': elapsed,
                'operations': operations,
                'ops_per_second': operations / elapsed,
                'device': 'GPU' if self.gpu_available else 'CPU'
            }

        except Exception as e:
            logger.error(f"Error en benchmark: {e}")
            return {'status': 'error', 'error': str(e)}


# Instancia global
gpu_manager = GPUManager()