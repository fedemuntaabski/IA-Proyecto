"""
Statistics Tracker - Rastreador de Estadísticas Mejorado.

Este módulo proporciona seguimiento detallado de estadísticas de la aplicación
para análisis de desempeño y uso del usuario.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from ..i18n import _


class StatisticsTracker:
    """Rastreador de estadísticas de sesión y aplicación."""

    def __init__(self, stats_file: str = "data/session_stats.json"):
        """
        Inicializa el rastreador de estadísticas.

        Args:
            stats_file: Ruta al archivo de estadísticas
        """
        self.stats_file = Path(stats_file)
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

        self.session_stats = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration': 0,
            'total_frames': 0,
            'total_drawings': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_fps': 0,
            'confidence_scores': [],
            'predictions_by_class': {},
            'errors': [],
            'device_info': {}
        }

    def record_frame(self, fps: float) -> None:
        """Registra que se procesó un frame."""
        self.session_stats['total_frames'] += 1
        self.session_stats['average_fps'] = fps

    def record_drawing(self, confidence: float, class_name: str, success: bool) -> None:
        """
        Registra un dibujo y su clasificación.

        Args:
            confidence: Nivel de confianza de la predicción
            class_name: Nombre de la clase predicha
            success: Si la predicción fue exitosa
        """
        self.session_stats['total_drawings'] += 1
        self.session_stats['confidence_scores'].append(confidence)

        if success:
            self.session_stats['successful_predictions'] += 1
        else:
            self.session_stats['failed_predictions'] += 1

        # Registrar predicción por clase
        if class_name not in self.session_stats['predictions_by_class']:
            self.session_stats['predictions_by_class'][class_name] = {'count': 0, 'success': 0}

        self.session_stats['predictions_by_class'][class_name]['count'] += 1
        if success:
            self.session_stats['predictions_by_class'][class_name]['success'] += 1

    def record_error(self, error_type: str, message: str) -> None:
        """
        Registra un error que ocurrió.

        Args:
            error_type: Tipo de error
            message: Mensaje de error
        """
        self.session_stats['errors'].append({
            'type': error_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    def set_device_info(self, device_info: Dict[str, Any]) -> None:
        """
        Establece información del dispositivo.

        Args:
            device_info: Diccionario con información del dispositivo
        """
        self.session_stats['device_info'] = device_info

    def end_session(self) -> Dict[str, Any]:
        """
        Finaliza la sesión y retorna el resumen.

        Returns:
            Diccionario con estadísticas finales
        """
        self.session_stats['end_time'] = datetime.now().isoformat()

        # Calcular duración
        start = datetime.fromisoformat(self.session_stats['start_time'])
        end = datetime.fromisoformat(self.session_stats['end_time'])
        self.session_stats['duration'] = (end - start).total_seconds()

        # Calcular confianza promedio
        if self.session_stats['confidence_scores']:
            avg_confidence = sum(self.session_stats['confidence_scores']) / len(
                self.session_stats['confidence_scores']
            )
            self.session_stats['average_confidence'] = avg_confidence

        # Calcular tasa de éxito
        total = (
            self.session_stats['successful_predictions'] +
            self.session_stats['failed_predictions']
        )
        if total > 0:
            success_rate = (
                self.session_stats['successful_predictions'] / total * 100
            )
            self.session_stats['success_rate'] = success_rate

        return self.session_stats.copy()

    def save(self) -> None:
        """Guarda las estadísticas en archivo."""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_stats, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️  {_('Error guardando estadísticas')}: {e}")

    def get_summary(self) -> str:
        """
        Genera un resumen formateado de las estadísticas.

        Returns:
            Resumen como string
        """
        stats = self.session_stats
        summary = []

        summary.append("📊 " + _("RESUMEN DE ESTADÍSTICAS"))
        summary.append("=" * 50)

        # Información temporal
        duration = stats.get('duration', 0)
        minutes, seconds = divmod(int(duration), 60)
        summary.append(f"⏱️  {_('Duración')}: {minutes:02d}:{seconds:02d}")

        # Información de frames
        summary.append(f"🎬 {_('Frames procesados')}: {stats.get('total_frames', 0)}")
        summary.append(f"📈 {_('FPS promedio')}: {stats.get('average_fps', 0):.1f}")

        # Información de dibujos
        summary.append(f"🎨 {_('Dibujos realizados')}: {stats.get('total_drawings', 0)}")
        success = stats.get('successful_predictions', 0)
        total_pred = success + stats.get('failed_predictions', 0)
        summary.append(f"✅ {_('Predicciones exitosas')}: {success}/{total_pred}")

        if total_pred > 0:
            rate = (success / total_pred * 100)
            summary.append(f"📊 {_('Tasa de éxito')}: {rate:.1f}%")

        # Confianza promedio
        if 'average_confidence' in stats:
            summary.append(f"🎯 {_('Confianza promedio')}: {stats['average_confidence']:.2%}")

        # Errores registrados
        if stats.get('errors'):
            summary.append(f"⚠️  {_('Errores')}: {len(stats['errors'])}")

        summary.append("=" * 50)

        return "\n".join(summary)

    def print_summary(self) -> None:
        """Imprime el resumen de estadísticas."""
        print("\n" + self.get_summary() + "\n")

    def get_class_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene estadísticas por clase.

        Returns:
            Diccionario con estadísticas por clase
        """
        class_stats = {}

        for class_name, stats in self.session_stats.get('predictions_by_class', {}).items():
            total = stats['count']
            success = stats['success']
            success_rate = (success / total * 100) if total > 0 else 0

            class_stats[class_name] = {
                'count': total,
                'success': success,
                'success_rate': success_rate
            }

        return class_stats

    def get_error_summary(self) -> Dict[str, int]:
        """
        Obtiene resumen de errores.

        Returns:
            Diccionario contando errores por tipo
        """
        error_summary = {}

        for error in self.session_stats.get('errors', []):
            error_type = error.get('type', 'unknown')
            error_summary[error_type] = error_summary.get(error_type, 0) + 1

        return error_summary


# Instancia global del rastreador de estadísticas
statistics_tracker = StatisticsTracker()
