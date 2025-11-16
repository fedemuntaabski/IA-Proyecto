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

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Prioridades de tareas."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

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
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ML-Worker")
        self.loop = None
        self.tasks = {}
        self.task_counter = 0
        
        # Manejo de prioridades mejorado
        self.priority_queue = deque()  # Para tareas con prioridad
        self.task_priority = {}  # Mapeo de task_id a prioridad
        self.max_queue_size = 100  # Límite de tareas en cola

        # Cola para resultados
        self.result_queue = queue.Queue()

        # Estadísticas mejoradas
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'average_execution_time': 0.0,
            'peak_active_tasks': 0,
            'execution_times': deque(maxlen=100)  # Últimas 100 ejecuciones
        }

        # Estado
        self.running = False
        self.worker_thread = None

        logger.info(f"✓ AsyncProcessor inicializado con {max_workers} workers")

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
        self.executor.shutdown(wait=True)

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

        logger.info("✓ AsyncProcessor detenido")

    def submit_task(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, **kwargs) -> str:
        """
        Envía una tarea para ejecución asíncrona con soporte a prioridades.

        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            priority: Prioridad de la tarea
            **kwargs: Argumentos nombrados

        Returns:
            ID único de la tarea
        """
        if not self.running:
            raise RuntimeError("AsyncProcessor no está ejecutándose")

        # Verificar límite de cola
        if len(self.tasks) >= self.max_queue_size:
            logger.warning(f"⚠ Cola de tareas llena ({len(self.tasks)}/{self.max_queue_size})")
            return None

        task_id = f"task_{self.task_counter}"
        self.task_counter += 1

        # Crear future
        future = self.executor.submit(func, *args, **kwargs)

        # Almacenar información de la tarea
        start_time = time.time()
        self.tasks[task_id] = {
            'future': future,
            'submitted_at': start_time,
            'func_name': func.__name__,
            'priority': priority,
            'execution_time': None
        }
        
        self.task_priority[task_id] = priority
        self.stats['total_tasks'] += 1

        logger.debug(f"✓ Tarea {task_id} enviada: {func.__name__} (prioridad: {priority.name})")
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

        return self.tasks[task_id]['future'].done()

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

        future = self.tasks[task_id]['future']
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
        future = task_info['future']

        info = {
            'task_id': task_id,
            'func_name': task_info['func_name'],
            'submitted_at': task_info['submitted_at'],
            'running_time': time.time() - task_info['submitted_at'],
            'done': future.done(),
            'cancelled': future.cancelled()
        }

        if future.done() and not future.cancelled():
            try:
                # Intentar obtener excepción si hubo error
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
        completed_tasks = self.stats['completed_tasks']
        
        # Actualizar pico de tareas activas
        if active_tasks > self.stats['peak_active_tasks']:
            self.stats['peak_active_tasks'] = active_tasks

        return {
            'running': self.running,
            'max_workers': self.max_workers,
            'active_tasks': active_tasks,
            'total_tasks_submitted': self.stats['total_tasks'],
            'completed_tasks': completed_tasks,
            'failed_tasks': self.stats['failed_tasks'],
            'cancelled_tasks': self.stats['cancelled_tasks'],
            'peak_active_tasks': self.stats['peak_active_tasks'],
            'average_execution_time': self.stats['average_execution_time'],
            'queue_size_ratio': active_tasks / self.max_queue_size,
            'task_info': {tid: self.get_task_info(tid) for tid in self.tasks.keys()}
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
                        # Aquí se podría manejar callbacks si fuera necesario
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

        return self.async_processor.submit_task(_predict_task)

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

        return self.async_processor.submit_task(_preprocess_task)

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

            task_id = self.async_processor.submit_task(_batch_predict_task)
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