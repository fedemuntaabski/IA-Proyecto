"""
Performance Monitor - Monitor de Rendimiento y Recursos.

Este módulo monitorea el rendimiento de la aplicación, uso de memoria,
y proporciona recomendaciones para optimización.
"""

import psutil
import time
from typing import Dict, Any, Optional
from ..i18n import _


class PerformanceMonitor:
    """Monitor de rendimiento y recursos del sistema."""

    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        """
        Inicializa el monitor de rendimiento.

        Args:
            warning_threshold: Umbral de advertencia para uso de recursos (0-1)
            critical_threshold: Umbral crítico para uso de recursos (0-1)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.last_check_time = time.time()
        self.check_interval = 5.0  # Verificar cada 5 segundos

        # Métricas
        self.memory_history = []
        self.cpu_history = []
        self.fps_history = []
        self.max_history_size = 100

        # Estado
        self.memory_warning = False
        self.cpu_warning = False

    def update(self, fps: Optional[float] = None) -> Dict[str, Any]:
        """
        Actualiza las métricas de rendimiento.

        Args:
            fps: FPS actual (opcional)

        Returns:
            Diccionario con métricas actuales
        """
        current_time = time.time()
        metrics = {}

        # Registrar FPS si se proporciona
        if fps is not None:
            self.fps_history.append(fps)
            if len(self.fps_history) > self.max_history_size:
                self.fps_history.pop(0)
            metrics['fps'] = fps

        # Verificar recursos cada intervalo
        if current_time - self.last_check_time >= self.check_interval:
            self.last_check_time = current_time

            # Memoria
            memory_percent = psutil.virtual_memory().percent / 100.0
            self.memory_history.append(memory_percent)
            if len(self.memory_history) > self.max_history_size:
                self.memory_history.pop(0)
            metrics['memory_percent'] = memory_percent

            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            self.cpu_history.append(cpu_percent)
            if len(self.cpu_history) > self.max_history_size:
                self.cpu_history.pop(0)
            metrics['cpu_percent'] = cpu_percent

            # Verificar umbrales
            self._check_thresholds(memory_percent, cpu_percent)

        return metrics

    def _check_thresholds(self, memory_percent: float, cpu_percent: float) -> None:
        """Verifica si se han alcanzado umbrales de recursos."""
        # Memoria
        if memory_percent >= self.critical_threshold:
            self.memory_warning = True
        elif memory_percent < self.critical_threshold * 0.9:
            self.memory_warning = False

        # CPU
        if cpu_percent >= self.critical_threshold:
            self.cpu_warning = True
        elif cpu_percent < self.critical_threshold * 0.9:
            self.cpu_warning = False

    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del rendimiento.

        Returns:
            Diccionario con estado y recomendaciones
        """
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB

            status = {
                'system_memory_percent': memory_info.percent,
                'system_memory_total_gb': memory_info.total / (1024**3),
                'system_memory_available_gb': memory_info.available / (1024**3),
                'process_memory_mb': process_memory,
                'cpu_percent': cpu_percent,
                'warnings': [],
                'recommendations': []
            }

            # Generar advertencias y recomendaciones
            if memory_info.percent > self.critical_threshold * 100:
                status['warnings'].append(_("Memoria del sistema muy alta"))
                status['recommendations'].append(_("Cierra otras aplicaciones"))

            if cpu_percent > self.critical_threshold * 100:
                status['warnings'].append(_("Uso de CPU muy alto"))
                status['recommendations'].append(_("Reduce la resolución o complejidad"))

            if process_memory > 500:  # MB
                status['recommendations'].append(_("Considera reiniciar la aplicación"))

            # FPS promedio
            if self.fps_history:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                status['average_fps'] = avg_fps

                if avg_fps < 15:
                    status['warnings'].append(_("FPS bajo"))
                    status['recommendations'].append(_("Verifica la iluminación y resolución"))

            return status

        except Exception as e:
            return {'error': str(e)}

    def print_status(self) -> None:
        """Imprime el estado del rendimiento formateado."""
        status = self.get_status()

        if 'error' in status:
            print(f"⚠️  {_('Error obteniendo estado')}: {status['error']}")
            return

        print("\n" + "=" * 60)
        print(f"⚙️  {_('ESTADO DE RENDIMIENTO')}")
        print("=" * 60)

        print(f"💾 {_('Memoria')}: {status['system_memory_percent']:.1f}% "
              f"({status['system_memory_available_gb']:.1f}GB disponible)")
        print(f"🔧 {_('CPU')}: {status['cpu_percent']:.1f}%")
        print(f"📊 {_('Proceso')}: {status['process_memory_mb']:.1f} MB")

        if 'average_fps' in status:
            print(f"🎬 {_('FPS promedio')}: {status['average_fps']:.1f}")

        if status['warnings']:
            print(f"\n⚠️  {_('Advertencias')}:")
            for warning in status['warnings']:
                print(f"   • {warning}")

        if status['recommendations']:
            print(f"\n💡 {_('Recomendaciones')}:")
            for rec in status['recommendations']:
                print(f"   • {rec}")

        print("=" * 60 + "\n")

    def get_average_fps(self) -> float:
        """Obtiene el FPS promedio."""
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

    def get_average_memory(self) -> float:
        """Obtiene el uso promedio de memoria."""
        return sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0.0

    def get_average_cpu(self) -> float:
        """Obtiene el uso promedio de CPU."""
        return sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0.0

    def should_optimize(self) -> bool:
        """Determina si debe optimizarse el rendimiento."""
        return self.memory_warning or self.cpu_warning

    def get_optimization_suggestions(self) -> list:
        """
        Obtiene sugerencias de optimización.

        Returns:
            Lista de sugerencias
        """
        suggestions = []

        # Basado en FPS
        if self.get_average_fps() < 20:
            suggestions.append(_("Reduce la resolución de la cámara"))
            suggestions.append(_("Desactiva procesamiento avanzado de visión"))

        # Basado en memoria
        if self.get_average_memory() > self.warning_threshold:
            suggestions.append(_("Limpia el caché de predicciones"))
            suggestions.append(_("Reduce el tamaño del historial"))

        # Basado en CPU
        if self.get_average_cpu() > self.warning_threshold:
            suggestions.append(_("Usa modelos más pequeños"))
            suggestions.append(_("Activa el uso de GPU si está disponible"))

        return suggestions


# Instancia global del monitor de rendimiento
performance_monitor = PerformanceMonitor()
