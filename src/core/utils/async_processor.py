"""
Procesamiento asíncrono para operaciones de ML.

Este módulo proporciona capacidades de procesamiento en segundo plano
para mejorar la responsiveness de la aplicación durante operaciones
de machine learning intensivas.
"""

import asyncio
import threading
import time
from typing import Callable, Any, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor
import queue
import logging
from collections import deque
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Prioridades de tareas."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class SpecializedWorkerPool:
    """Pool de workers especializados por tipo de tarea."""
    
    def __init__(self):
        self.pools = {
            'prediction': ThreadPoolExecutor(max_workers=2, thread_name_prefix="ML-Predict"),
            'preprocessing': ThreadPoolExecutor(max_workers=2, thread_name_prefix="ML-Preprocess"),
            'training': ThreadPoolExecutor(max_workers=1, thread_name_prefix="ML-Train"),
            'general': ThreadPoolExecutor(max_workers=2, thread_name_prefix="ML-General")
        }
    
    def submit(self, task_type: str, func: Callable, *args, **kwargs):
        """Envía tarea al pool especializado."""
        pool = self.pools.get(task_type, self.pools['general'])
        return pool.submit(func, *args, **kwargs)
    
    def shutdown(self):
        """Cierra todos los pools."""
        for pool in self.pools.values():
            pool.shutdown(wait=True)


class ResultCache:
    """Cache inteligente de resultados con expiración."""
    
    def __init__(self, max_size: int = 100, ttl: float = 3600.0):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live en segundos
        self.timestamps = {}
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Genera clave de cache basada en función y argumentos."""
        try:
            key_data = json.dumps({
                'func': func_name,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }, default=str)
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return None
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache si existe y no expiró."""
        if key not in self.cache:
            return None
        
        # Verificar expiración
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Almacena valor en cache."""
        if len(self.cache) >= self.max_size:
            # Remover entrada más antigua
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Limpia todo el cache."""
        self.cache.clear()
        self.timestamps.clear()


class PriorityTaskQueue:
    """Cola de tareas con sistema de prioridades mejorado."""
    
    def __init__(self, max_size: int = 100):
        self.queue = {
            TaskPriority.CRITICAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque()
        }
        self.max_size = max_size
        self.total_tasks = 0
    
    def put(self, task_id: str, priority: TaskPriority) -> bool:
        """Agrega tarea a la cola."""
        if self.total_tasks >= self.max_size:
            return False
        
        self.queue[priority].append(task_id)
        self.total_tasks += 1
        return True
    
    def get_next(self) -> Optional[str]:
        """Obtiene la siguiente tarea según prioridad."""
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            if self.queue[priority]:
                task_id = self.queue[priority].popleft()
                self.total_tasks -= 1
                return task_id
        
        return None
    
    def size(self) -> int:
        """Retorna tamaño total de la cola."""
        return self.total_tasks

class AsyncProcessor:
    """
    Procesador asíncrono para operaciones de ML.

    Permite ejecutar operaciones intensivas en segundo plano
    sin bloquear la interfaz de usuario.
    """

    def __init__(self, max_workers: int = 2):
        """
        Inicializa el procesador asíncrono.

        Args:
            max_workers: Número máximo de hilos de trabajo
        """
        self.max_workers = max_workers
        
        # Pool de workers especializados
        self.worker_pool = SpecializedWorkerPool()
        self.executor = self.worker_pool.pools['general']
        
        self.loop = None
        self.tasks = {}
        self.task_counter = 0
        
        # Cola de prioridades mejorada
        self.priority_queue = PriorityTaskQueue(max_size=100)
        self.task_priority = {}

        # Cache inteligente de resultados
        self.result_cache = ResultCache(max_size=100, ttl=3600.0)
        
        # Cola para resultados
        self.result_queue = queue.Queue()

        # Estadísticas mejoradas
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'cached_results': 0,
            'average_execution_time': 0.0,
            'peak_active_tasks': 0,
            'execution_times': deque(maxlen=100)
        }

        # Estado
        self.running = False
        self.worker_thread = None

        logger.info(f"✓ AsyncProcessor inicializado con {max_workers} workers (pool especializado)")

    def start(self) -> None:
        """Inicia el procesador asíncrono."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("✓ AsyncProcessor iniciado")

    def stop(self) -> None:
        """Detiene el procesador asíncrono."""
        if not self.running:
            return

        self.running = False
        self.worker_pool.shutdown()

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

        logger.info("✓ AsyncProcessor detenido")

    def submit_task(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, 
                   task_type: str = 'general', use_cache: bool = True, **kwargs) -> str:
        """
        Envía una tarea para ejecución asíncrona con soporte a prioridades y cache.

        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            priority: Prioridad de la tarea
            task_type: Tipo de tarea (prediction, preprocessing, training, general)
            use_cache: Si usar cache para resultados
            **kwargs: Argumentos nombrados

        Returns:
            ID único de la tarea
        """
        if not self.running:
            raise RuntimeError("AsyncProcessor no está ejecutándose")

        # Verificar cache si está habilitado
        if use_cache:
            cache_key = self.result_cache._generate_key(func.__name__, args, kwargs)
            if cache_key:
                cached_result = self.result_cache.get(cache_key)
                if cached_result is not None:
                    # Crear pseudo-task para cache hit
                    task_id = f"task_{self.task_counter}_cached"
                    self.task_counter += 1
                    self.tasks[task_id] = {
                        'result': cached_result,
                        'cached': True,
                        'func_name': func.__name__
                    }
                    self.stats['cached_results'] += 1
                    logger.debug(f"✓ Cache hit para {func.__name__}")
                    return task_id

        task_id = f"task_{self.task_counter}"
        self.task_counter += 1

        # Enviar al pool especializado
        future = self.worker_pool.submit(task_type, func, *args, **kwargs)

        # Almacenar información de la tarea
        start_time = time.time()
        self.tasks[task_id] = {
            'future': future,
            'submitted_at': start_time,
            'func_name': func.__name__,
            'priority': priority,
            'execution_time': None,
            'cached': False,
            'cache_key': cache_key if use_cache else None,
            'use_cache': use_cache
        }
        
        self.task_priority[task_id] = priority
        self.priority_queue.put(task_id, priority)
        self.stats['total_tasks'] += 1

        logger.debug(f"✓ Tarea {task_id} enviada: {func.__name__} (prioridad: {priority.name}, tipo: {task_type})")
        return task_id

    def get_task_result(self, task_id: str, timeout: float = 0.1) -> Optional[Any]:
        """
        Obtiene el resultado de una tarea si está disponible.

        Args:
            task_id: ID de la tarea
            timeout: Tiempo máximo de espera en segundos

        Returns:
            Resultado de la tarea o None si no está listo
        """
        if task_id not in self.tasks:
            return None

        task_info = self.tasks[task_id]
        
        # Manejo de cache hit
        if task_info.get('cached', False):
            result = task_info['result']
            del self.tasks[task_id]
            return result

        future = task_info['future']

        try:
            # Verificar si está listo sin bloquear
            if future.done():
                result = future.result(timeout=timeout)
                
                # Registrar tiempo de ejecución
                execution_time = time.time() - task_info['submitted_at']
                task_info['execution_time'] = execution_time
                self.stats['execution_times'].append(execution_time)
                self.stats['completed_tasks'] += 1
                
                # Actualizar promedio
                if self.stats['execution_times']:
                    self.stats['average_execution_time'] = sum(self.stats['execution_times']) / len(self.stats['execution_times'])
                
                # Guardar en cache si está habilitado
                if task_info['use_cache'] and task_info['cache_key']:
                    self.result_cache.set(task_info['cache_key'], result)
                
                # Limpiar tarea completada
                del self.tasks[task_id]
                if task_id in self.task_priority:
                    del self.task_priority[task_id]
                
                return result
            else:
                return None

        except Exception as e:
            logger.error(f"Error obteniendo resultado de tarea {task_id}: {e}")
            self.stats['failed_tasks'] += 1
            del self.tasks[task_id]
            if task_id in self.task_priority:
                del self.task_priority[task_id]
            raise e

    def is_task_done(self, task_id: str) -> bool:
        """
        Verifica si una tarea ha terminado.

        Args:
            task_id: ID de la tarea

        Returns:
            True si la tarea terminó (exitosa o con error)
        """
        if task_id not in self.tasks:
            return True

        task_info = self.tasks[task_id]
        
        if task_info.get('cached', False):
            return True

        return task_info['future'].done()

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea si es posible.

        Args:
            task_id: ID de la tarea

        Returns:
            True si la tarea fue cancelada
        """
        if task_id not in self.tasks:
            return False

        task_info = self.tasks[task_id]
        
        if task_info.get('cached', False):
            del self.tasks[task_id]
            return True

        future = task_info['future']
        cancelled = future.cancel()

        if cancelled:
            self.stats['cancelled_tasks'] += 1
            del self.tasks[task_id]
            if task_id in self.task_priority:
                del self.task_priority[task_id]
            logger.debug(f"✓ Tarea {task_id} cancelada")

        return cancelled

    def get_active_tasks(self) -> List[str]:
        """
        Obtiene lista de tareas activas.

        Returns:
            Lista de IDs de tareas activas
        """
        return list(self.tasks.keys())

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información detallada de una tarea.

        Args:
            task_id: ID de la tarea

        Returns:
            Diccionario con información de la tarea
        """
        if task_id not in self.tasks:
            return None

        task_info = self.tasks[task_id]
        
        if task_info.get('cached', False):
            return {
                'task_id': task_id,
                'func_name': task_info['func_name'],
                'done': True,
                'cached': True,
                'success': True
            }

        future = task_info['future']

        info = {
            'task_id': task_id,
            'func_name': task_info['func_name'],
            'submitted_at': task_info['submitted_at'],
            'running_time': time.time() - task_info['submitted_at'],
            'done': future.done(),
            'cancelled': future.cancelled(),
            'cached': False
        }

        if future.done() and not future.cancelled():
            try:
                future.exception(timeout=0.001)
                info['success'] = True
            except Exception:
                info['success'] = False

        return info

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del procesador.

        Returns:
            Diccionario con estadísticas mejoradas
        """
        active_tasks = len(self.tasks)
        
        if active_tasks > self.stats['peak_active_tasks']:
            self.stats['peak_active_tasks'] = active_tasks

        return {
            'running': self.running,
            'max_workers': self.max_workers,
            'active_tasks': active_tasks,
            'total_tasks_submitted': self.stats['total_tasks'],
            'completed_tasks': self.stats['completed_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'cancelled_tasks': self.stats['cancelled_tasks'],
            'cached_results': self.stats['cached_results'],
            'peak_active_tasks': self.stats['peak_active_tasks'],
            'average_execution_time': self.stats['average_execution_time'],
            'cache_stats': {
                'size': len(self.result_cache.cache),
                'max_size': self.result_cache.max_size,
                'ttl': self.result_cache.ttl
            },
            'queue_stats': {
                'pending': self.priority_queue.size(),
                'max_size': self.priority_queue.max_size
            }
        }

    def _worker_loop(self) -> None:
        """Bucle principal del worker thread."""
        logger.debug("Worker loop iniciado")

        while self.running:
            try:
                # Procesar resultados pendientes
                while not self.result_queue.empty():
                    try:
                        result = self.result_queue.get_nowait()
                    except queue.Empty:
                        break

                # Pequeña pausa para no consumir CPU
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error en worker loop: {e}")
                break

        logger.debug("Worker loop terminado")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class MLAsyncProcessor:
    """
    Procesador especializado para operaciones de ML.

    Proporciona una interfaz de alto nivel para operaciones
    comunes de machine learning de manera asíncrona.
    """

    def __init__(self, async_processor: AsyncProcessor):
        """
        Inicializa el procesador de ML.

        Args:
            async_processor: Instancia de AsyncProcessor
        """
        self.async_processor = async_processor
        self.prediction_cache = {}
        self.cache_max_size = 100

    def predict_async(self, classifier, image: Any, top_k: int = 3) -> str:
        """
        Realiza una predicción de manera asíncrona.

        Args:
            classifier: Clasificador a usar
            image: Imagen a clasificar
            top_k: Número de predicciones a retornar

        Returns:
            ID de la tarea asíncrona
        """
        def _predict_task():
            return classifier.predict(image, top_k)

        return self.async_processor.submit_task(
            _predict_task, 
            priority=TaskPriority.HIGH,
            task_type='prediction',
            use_cache=True
        )

    def preprocess_async(self, preprocessor_func: Callable, data: Any) -> str:
        """
        Ejecuta preprocesamiento de manera asíncrona.

        Args:
            preprocessor_func: Función de preprocesamiento
            data: Datos a preprocesar

        Returns:
            ID de la tarea asíncrona
        """
        def _preprocess_task():
            return preprocessor_func(data)

        return self.async_processor.submit_task(
            _preprocess_task,
            priority=TaskPriority.NORMAL,
            task_type='preprocessing',
            use_cache=True
        )

    def batch_predict_async(self, classifier, images: List[Any], batch_size: int = 4) -> List[str]:
        """
        Realiza predicciones por lotes de manera asíncrona.

        Args:
            classifier: Clasificador a usar
            images: Lista de imágenes
            batch_size: Tamaño del lote

        Returns:
            Lista de IDs de tareas
        """
        task_ids = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            def _batch_predict_task(batch=batch):
                results = []
                for img in batch:
                    result = classifier.predict(img, top_k=1)[0]
                    results.append(result)
                return results

            task_id = self.async_processor.submit_task(
                _batch_predict_task,
                priority=TaskPriority.HIGH,
                task_type='prediction',
                use_cache=True
            )
            task_ids.append(task_id)

        return task_ids

    def get_prediction_result(self, task_id: str) -> Optional[Any]:
        """
        Obtiene el resultado de una predicción asíncrona.

        Args:
            task_id: ID de la tarea

        Returns:
            Resultado de la predicción o None
        """
        return self.async_processor.get_task_result(task_id)

    def is_prediction_ready(self, task_id: str) -> bool:
        """
        Verifica si una predicción asíncrona está lista.

        Args:
            task_id: ID de la tarea

        Returns:
            True si la predicción está lista
        """
        return self.async_processor.is_task_done(task_id)

    def cancel_prediction(self, task_id: str) -> bool:
        """
        Cancela una predicción asíncrona.

        Args:
            task_id: ID de la tarea

        Returns:
            True si fue cancelada
        """
        return self.async_processor.cancel_task(task_id)

    def get_processor_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del procesador de ML.

        Returns:
            Diccionario con estadísticas
        """
        stats = self.async_processor.get_stats()
        stats['cache_size'] = len(self.prediction_cache)
        stats['cache_max_size'] = self.cache_max_size
        return stats


# Instancia global
async_processor = AsyncProcessor()
ml_async_processor = MLAsyncProcessor(async_processor)