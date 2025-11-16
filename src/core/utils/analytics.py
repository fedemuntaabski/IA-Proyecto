"""
Analytics Framework - Framework básico de analíticas.

Este módulo proporciona capacidades básicas de analíticas para rastrear
el uso de la aplicación, rendimiento y posibles problemas.
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import logging

logger = logging.getLogger(__name__)

class AnalyticsEvent:
    """
    Representa un evento de analíticas.
    """

    def __init__(self, event_type: str, data: Dict[str, Any], timestamp: Optional[float] = None):
        """
        Inicializa un evento de analíticas.

        Args:
            event_type: Tipo de evento
            data: Datos del evento
            timestamp: Timestamp del evento (opcional)
        """
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el evento a diccionario.

        Returns:
            Diccionario con los datos del evento
        """
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'data': self.data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalyticsEvent':
        """
        Crea un evento desde un diccionario.

        Args:
            data: Diccionario con datos del evento

        Returns:
            Instancia de AnalyticsEvent
        """
        return cls(
            event_type=data['event_type'],
            data=data['data'],
            timestamp=data['timestamp']
        )


class AnalyticsTracker:
    """
    Rastreador básico de analíticas para la aplicación.
    """

    def __init__(self, storage_path: str = "analytics.json", max_events: int = 1000):
        """
        Inicializa el rastreador de analíticas.

        Args:
            storage_path: Ruta donde almacenar los eventos
            max_events: Número máximo de eventos a mantener
        """
        self.storage_path = Path(storage_path)
        self.max_events = max_events
        self.events: List[AnalyticsEvent] = []
        self.session_start = time.time()
        self.lock = threading.Lock()

        # Estadísticas de sesión
        self.session_stats = {
            'total_frames': 0,
            'total_predictions': 0,
            'successful_predictions': 0,
            'errors': 0,
            'avg_fps': 0.0,
            'memory_peak': 0.0
        }

        # Cargar eventos existentes
        self._load_events()

        logger.info(f"✓ AnalyticsTracker inicializado (máx {max_events} eventos)")

    def track_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Registra un evento de analíticas.

        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        event = AnalyticsEvent(event_type, data)

        with self.lock:
            self.events.append(event)

            # Mantener límite de eventos
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

        logger.debug(f"📊 Evento registrado: {event_type}")

    def track_frame_processed(self, fps: float) -> None:
        """
        Registra el procesamiento de un frame.

        Args:
            fps: FPS actual
        """
        self.session_stats['total_frames'] += 1
        self.session_stats['avg_fps'] = (
            (self.session_stats['avg_fps'] * (self.session_stats['total_frames'] - 1)) + fps
        ) / self.session_stats['total_frames']

        # Monitorear memoria
        memory_mb = psutil.virtual_memory().used / (1024 * 1024)
        self.session_stats['memory_peak'] = max(self.session_stats['memory_peak'], memory_mb)

    def track_prediction(self, success: bool, confidence: float, class_name: str) -> None:
        """
        Registra una predicción.

        Args:
            success: Si la predicción fue exitosa
            confidence: Nivel de confianza
            class_name: Clase predicha
        """
        self.session_stats['total_predictions'] += 1

        if success:
            self.session_stats['successful_predictions'] += 1

        self.track_event('prediction', {
            'success': success,
            'confidence': confidence,
            'class_name': class_name,
            'timestamp': time.time()
        })

    def track_error(self, error_type: str, error_message: str, context: Optional[Dict] = None) -> None:
        """
        Registra un error.

        Args:
            error_type: Tipo de error
            error_message: Mensaje de error
            context: Contexto adicional del error
        """
        self.session_stats['errors'] += 1

        self.track_event('error', {
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {},
            'timestamp': time.time()
        })

        logger.warning(f"⚠ Error registrado: {error_type} - {error_message}")

    def track_performance_metric(self, metric_name: str, value: float) -> None:
        """
        Registra una métrica de rendimiento.

        Args:
            metric_name: Nombre de la métrica
            value: Valor de la métrica
        """
        self.track_event('performance', {
            'metric_name': metric_name,
            'value': value,
            'timestamp': time.time()
        })

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de la sesión actual.

        Returns:
            Diccionario con estadísticas de la sesión
        """
        session_duration = time.time() - self.session_start
        success_rate = (
            self.session_stats['successful_predictions'] / self.session_stats['total_predictions'] * 100
            if self.session_stats['total_predictions'] > 0 else 0
        )

        return {
            'session_duration': session_duration,
            'total_frames': self.session_stats['total_frames'],
            'total_predictions': self.session_stats['total_predictions'],
            'successful_predictions': self.session_stats['successful_predictions'],
            'success_rate': success_rate,
            'errors': self.session_stats['errors'],
            'avg_fps': self.session_stats['avg_fps'],
            'memory_peak_mb': self.session_stats['memory_peak'],
            'events_recorded': len(self.events)
        }

    def get_events_by_type(self, event_type: str, hours: int = 24) -> List[AnalyticsEvent]:
        """
        Obtiene eventos de un tipo específico en las últimas horas.

        Args:
            event_type: Tipo de evento
            hours: Horas hacia atrás para buscar

        Returns:
            Lista de eventos
        """
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            return [
                event for event in self.events
                if event.event_type == event_type and event.timestamp >= cutoff_time
            ]

    def get_error_summary(self, hours: int = 24) -> Dict[str, int]:
        """
        Obtiene un resumen de errores en las últimas horas.

        Args:
            hours: Horas hacia atrás para buscar

        Returns:
            Diccionario con conteo de errores por tipo
        """
        errors = self.get_events_by_type('error', hours)
        summary = {}

        for event in errors:
            error_type = event.data.get('error_type', 'unknown')
            summary[error_type] = summary.get(error_type, 0) + 1

        return summary

    def save_events(self) -> None:
        """
        Guarda los eventos en disco.
        """
        try:
            events_data = [event.to_dict() for event in self.events]

            with open(self.storage_path, 'w') as f:
                json.dump({
                    'events': events_data,
                    'session_stats': self.session_stats,
                    'last_save': time.time()
                }, f, indent=2)

            logger.debug(f"✓ Eventos guardados en {self.storage_path}")

        except Exception as e:
            logger.error(f"Error guardando eventos: {e}")

    def _load_events(self) -> None:
        """
        Carga eventos desde disco.
        """
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Cargar eventos
            self.events = [
                AnalyticsEvent.from_dict(event_data)
                for event_data in data.get('events', [])
            ]

            # Cargar estadísticas si existen
            if 'session_stats' in data:
                self.session_stats.update(data['session_stats'])

            logger.info(f"✓ Eventos cargados desde {self.storage_path} ({len(self.events)} eventos)")

        except Exception as e:
            logger.error(f"Error cargando eventos: {e}")
            self.events = []

    def cleanup_old_events(self, days: int = 7) -> int:
        """
        Limpia eventos antiguos.

        Args:
            days: Días de antigüedad para eliminar

        Returns:
            Número de eventos eliminados
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        original_count = len(self.events)

        with self.lock:
            self.events = [event for event in self.events if event.timestamp >= cutoff_time]

        removed_count = original_count - len(self.events)

        if removed_count > 0:
            logger.info(f"🧹 Eliminados {removed_count} eventos antiguos")

        return removed_count


# Instancia global
analytics_tracker = AnalyticsTracker()