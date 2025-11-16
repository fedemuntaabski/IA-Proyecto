"""
Diagnostic Monitor - Monitor de Diagnóstico en Tiempo Real.

Este módulo proporciona herramientas para monitoreo y diagnóstico de
la aplicación durante la ejecución.
"""

import time
from typing import Dict, Any, List
from collections import deque
from .performance_monitor import performance_monitor
from .sensitivity_manager import sensitivity_manager
from .lighting_analysis import lighting_analyzer


class DiagnosticMonitor:
    """Monitor de diagnóstico en tiempo real."""

    def __init__(self, window_size: int = 100):
        """
        Inicializa el monitor de diagnóstico.

        Args:
            window_size: Tamaño de la ventana histórica
        """
        self.window_size = window_size
        self.event_history = deque(maxlen=window_size)
        self.issue_log = deque(maxlen=50)
        self.last_report_time = time.time()
        self.report_interval = 10.0  # Reportar cada 10 segundos

    def track_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Registra un evento de diagnóstico.

        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        self.event_history.append({
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        })

    def log_issue(self, severity: str, component: str, message: str) -> None:
        """
        Registra un problema detectado.

        Args:
            severity: 'info', 'warning', o 'critical'
            component: Componente afectado
            message: Descripción del problema
        """
        self.issue_log.append({
            'timestamp': time.time(),
            'severity': severity,
            'component': component,
            'message': message
        })

    def generate_report(self, force: bool = False) -> Dict[str, Any]:
        """
        Genera un reporte completo de diagnóstico.

        Args:
            force: Si True, genera reporte aunque no sea el intervalo

        Returns:
            Diccionario con el reporte
        """
        current_time = time.time()
        
        if not force and (current_time - self.last_report_time < self.report_interval):
            return {}

        self.last_report_time = current_time

        # Recopilar métricas de todos los monitores
        report = {
            'timestamp': current_time,
            'performance': self._get_performance_metrics(),
            'sensitivity': sensitivity_manager.get_diagnostics(),
            'lighting': self._get_lighting_metrics(),
            'issues': self._get_critical_issues(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento."""
        return {
            'avg_fps': performance_monitor.get_average_fps(),
            'avg_memory_percent': performance_monitor.get_average_memory() * 100,
            'avg_cpu_percent': performance_monitor.get_average_cpu() * 100,
            'memory_warning': performance_monitor.memory_warning,
            'cpu_warning': performance_monitor.cpu_warning
        }

    def _get_lighting_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de iluminación."""
        return {
            'avg_brightness': lighting_analyzer.get_average_brightness(),
            'recent_analyses': list(lighting_analyzer.lighting_history)[-5:]
            if lighting_analyzer.lighting_history else []
        }

    def _get_critical_issues(self) -> List[Dict[str, Any]]:
        """Obtiene problemas críticos recientes."""
        critical = []
        
        for issue in reversed(self.issue_log):
            if issue['severity'] == 'critical':
                critical.append(issue)
                if len(critical) >= 5:
                    break

        return critical

    def _generate_recommendations(self) -> List[str]:
        """Genera recomendaciones basadas en diagnóstico."""
        recommendations = []

        # Recomendaciones de rendimiento
        perf = self._get_performance_metrics()
        if perf['avg_fps'] < 20:
            recommendations.append("Rendimiento bajo: considera reducir resolución o desactivar visión avanzada")

        if perf['avg_memory_percent'] > 80:
            recommendations.append("Uso de memoria alto: considera reiniciar la aplicación")

        # Recomendaciones de iluminación
        lighting = self._get_lighting_metrics()
        if lighting['avg_brightness'] < 85:
            recommendations.append("Iluminación insuficiente: aumenta luz ambiente")
        elif lighting['avg_brightness'] > 200:
            recommendations.append("Iluminación excesiva: reduce luz o ajusta ángulo de cámara")

        # Recomendaciones de sensibilidad
        sensitivity = sensitivity_manager.get_diagnostics()
        if sensitivity['avg_noise_level'] > 0.7:
            recommendations.append("Ruido alto: limpia la cámara o mejora iluminación")

        return recommendations

    def get_status_summary(self) -> str:
        """
        Obtiene un resumen del estado actual.

        Returns:
            String con el resumen formateado
        """
        report = self.generate_report(force=True)
        
        if not report:
            return "Monitor: Sistema operativo correctamente"

        perf = report['performance']
        sens = report['sensitivity']
        lighting = report['lighting']

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    DIAGNÓSTICO DEL SISTEMA                  ║
╠══════════════════════════════════════════════════════════════╣
║ 🎬 FPS: {perf['avg_fps']:.1f} | 💾 Memoria: {perf['avg_memory_percent']:.1f}% | 🔧 CPU: {perf['avg_cpu_percent']:.1f}%
║ 🌞 Brillo: {lighting['avg_brightness']:.0f} | 📊 Sensibilidad: {sens['current_sensitivity']:.2f}
║ 🎯 Calidad: {sens['avg_frame_quality']:.0%} | 🔊 Ruido: {sens['avg_noise_level']:.0%}
╠══════════════════════════════════════════════════════════════╣
"""

        if report['issues']:
            summary += "║ ⚠️  PROBLEMAS DETECTADOS:\n"
            for issue in report['issues'][:3]:
                summary += f"║   • {issue['component']}: {issue['message']}\n"
            summary += "╠══════════════════════════════════════════════════════════════╣\n"

        if report['recommendations']:
            summary += "║ 💡 RECOMENDACIONES:\n"
            for rec in report['recommendations'][:3]:
                summary += f"║   • {rec}\n"

        summary += "╚══════════════════════════════════════════════════════════════╝"

        return summary

    def health_check(self) -> bool:
        """
        Realiza un chequeo de salud del sistema.

        Returns:
            True si el sistema está saludable, False si hay problemas críticos
        """
        perf = self._get_performance_metrics()

        # Criterios para sistema saludable
        critical_fps = perf['avg_fps'] > 10
        critical_memory = perf['avg_memory_percent'] < 95
        critical_cpu = perf['avg_cpu_percent'] < 95

        return critical_fps and critical_memory and critical_cpu


# Instancia global
diagnostic_monitor = DiagnosticMonitor()
