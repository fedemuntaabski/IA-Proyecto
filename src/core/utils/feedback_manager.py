"""
Feedback System - Sistema de Feedback para Corrección Manual de Predicciones.

Este módulo permite a los usuarios corregir predicciones incorrectas del clasificador,
acumulando datos para mejorar el modelo con el tiempo.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
from datetime import datetime

from ..i18n import _
from .analytics import analytics_tracker


@dataclass
class FeedbackEntry:
    """
    Entrada de feedback de un usuario.

    Representa una corrección manual de una predicción del clasificador.
    """
    timestamp: float
    original_prediction: str
    original_confidence: float
    corrected_class: str
    gesture_image_data: List[List[float]]  # Imagen del gesto como lista de listas
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    quality_score: float = 1.0  # Calidad de la corrección (0-1)
    validated: bool = False  # Si ha sido validada por un experto


class FeedbackManager:
    """
    Gestor del sistema de feedback para corrección manual de predicciones.

    Permite a los usuarios corregir predicciones incorrectas y acumula
    datos para mejorar el modelo de clasificación.
    """

    def __init__(self, feedback_dir: str = "feedback_data"):
        """
        Inicializa el gestor de feedback.

        Args:
            feedback_dir: Directorio donde almacenar los datos de feedback
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)

        # Archivo de datos de feedback
        self.feedback_file = self.feedback_dir / "feedback_entries.json"
        self.stats_file = self.feedback_dir / "feedback_stats.json"
        self.user_stats_file = self.feedback_dir / "user_stats.json"

        # Estado en memoria
        self.feedback_entries: List[FeedbackEntry] = []
        self.user_scores: Dict[str, float] = {}  # Puntuaciones de usuarios
        self.class_corrections: Dict[str, Dict[str, int]] = {}  # Correcciones por clase

        # Lock para acceso thread-safe
        self.lock = threading.Lock()

        # Estadísticas
        self.stats = {
            'total_corrections': 0,
            'unique_users': 0,
            'corrections_by_class': {},
            'accuracy_improvements': [],
            'last_updated': time.time()
        }

        # Cargar datos existentes
        self._load_feedback_data()
        self._load_stats()
        self._load_user_stats()

        # Sistema de gamificación
        self.gamification = {
            'points_per_correction': 10,
            'points_per_validation': 5,
            'bonus_accuracy_streak': 50,  # puntos extra cada 10 correcciones consecutivas
            'levels': {
                0: {'name': 'Principiante', 'threshold': 0},
                1: {'name': 'Contribuidor', 'threshold': 50},
                2: {'name': 'Experto', 'threshold': 200},
                3: {'name': 'Maestro', 'threshold': 500},
                4: {'name': 'Leyenda', 'threshold': 1000}
            }
        }

        # Estadísticas de gamificación
        self.user_stats = {}
        self.leaderboard = []

        print(f"✓ Sistema de feedback inicializado - {len(self.feedback_entries)} entradas cargadas")

    def add_correction(self, original_prediction: str, original_confidence: float,
                      corrected_class: str, gesture_image_data: List[List[float]],
                      user_id: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """
        Agrega una nueva corrección de feedback.

        Args:
            original_prediction: Clase predicha originalmente
            original_confidence: Confianza de la predicción original
            corrected_class: Clase corregida por el usuario
            gesture_image_data: Datos de la imagen del gesto
            user_id: ID opcional del usuario
            session_id: ID opcional de la sesión

        Returns:
            True si se agregó correctamente
        """
        try:
            with self.lock:
                # Crear entrada de feedback
                entry = FeedbackEntry(
                    timestamp=time.time(),
                    original_prediction=original_prediction,
                    original_confidence=original_confidence,
                    corrected_class=corrected_class,
                    gesture_image_data=gesture_image_data,
                    user_id=user_id,
                    session_id=session_id
                )

                # Agregar a la lista
                self.feedback_entries.append(entry)

                # Actualizar estadísticas
                self._update_stats(entry)

                # Otorgar puntos por la corrección
                user_id = entry.user_id or "anonymous"
                self.award_points(user_id, self.gamification['points_per_correction'], "correction")

                # Guardar inmediatamente
                self._save_feedback_data()

                # Rastrear en analytics
                analytics_tracker.track_event('feedback_correction', {
                    'original_prediction': original_prediction,
                    'corrected_class': corrected_class,
                    'confidence': original_confidence,
                    'user_id': user_id,
                    'timestamp': entry.timestamp
                })

                print(f"✅ Corrección agregada: {original_prediction} → {corrected_class}")
                return True

        except Exception as e:
            print(f"❌ Error agregando corrección: {e}")
            analytics_tracker.track_error('feedback', f'Error adding correction: {e}')
            return False

    def get_correction_suggestions(self, current_prediction: str, limit: int = 3) -> List[str]:
        """
        Obtiene sugerencias de corrección basadas en el historial.

        Args:
            current_prediction: Predicción actual
            limit: Número máximo de sugerencias

        Returns:
            Lista de clases sugeridas para corrección
        """
        if current_prediction not in self.class_corrections:
            return []

        corrections = self.class_corrections[current_prediction]
        sorted_corrections = sorted(corrections.items(), key=lambda x: x[1], reverse=True)

        return [class_name for class_name, count in sorted_corrections[:limit]]

    def get_user_score(self, user_id: str) -> float:
        """
        Obtiene la puntuación de calidad de un usuario.

        Args:
            user_id: ID del usuario

        Returns:
            Puntuación de calidad (0-1)
        """
        return self.user_scores.get(user_id, 0.5)  # Default medio si no hay historial

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema de feedback.

        Returns:
            Diccionario con estadísticas
        """
        with self.lock:
            return self.stats.copy()

    def export_training_data(self, output_file: Optional[str] = None) -> str:
        """
        Exporta datos de feedback para re-entrenamiento del modelo.

        Args:
            output_file: Archivo de salida (opcional)

        Returns:
            Ruta al archivo exportado
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.feedback_dir / f"training_data_{timestamp}.json"

        try:
            # Filtrar solo entradas validadas y de alta calidad
            training_data = []
            for entry in self.feedback_entries:
                if entry.quality_score >= 0.7 and entry.validated:
                    training_data.append({
                        'image': entry.gesture_image_data,
                        'label': entry.corrected_class,
                        'original_prediction': entry.original_prediction,
                        'confidence': entry.original_confidence,
                        'user_id': entry.user_id,
                        'timestamp': entry.timestamp
                    })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Datos de entrenamiento exportados: {len(training_data)} muestras a {output_file}")
            return str(output_file)

        except Exception as e:
            print(f"❌ Error exportando datos de entrenamiento: {e}")
            return ""

    def validate_correction(self, entry_index: int, quality_score: float,
                           validator_id: Optional[str] = None) -> bool:
        """
        Valida una corrección manualmente (para expertos).

        Args:
            entry_index: Índice de la entrada a validar
            quality_score: Puntuación de calidad (0-1)
            validator_id: ID del validador

        Returns:
            True si se validó correctamente
        """
        try:
            with self.lock:
                if 0 <= entry_index < len(self.feedback_entries):
                    entry = self.feedback_entries[entry_index]
                    entry.quality_score = quality_score
                    entry.validated = quality_score >= 0.8

                    # Actualizar puntuación del usuario
                    if entry.user_id:
                        current_score = self.user_scores.get(entry.user_id, 0.5)
                        # Promedio ponderado
                        self.user_scores[entry.user_id] = (current_score + quality_score) / 2

                    self._save_feedback_data()

                    analytics_tracker.track_event('feedback_validation', {
                        'entry_index': entry_index,
                        'quality_score': quality_score,
                        'validator_id': validator_id,
                        'timestamp': time.time()
                    })

                    print(f"✅ Corrección validada: calidad {quality_score:.2f}")
                    return True

        except Exception as e:
            print(f"❌ Error validando corrección: {e}")

        return False

    def _update_stats(self, entry: FeedbackEntry) -> None:
        """Actualiza las estadísticas con una nueva entrada."""
        self.stats['total_corrections'] += 1

        # Actualizar correcciones por clase
        if entry.original_prediction not in self.stats['corrections_by_class']:
            self.stats['corrections_by_class'][entry.original_prediction] = {}

        if entry.corrected_class not in self.stats['corrections_by_class'][entry.original_prediction]:
            self.stats['corrections_by_class'][entry.original_prediction][entry.corrected_class] = 0

        self.stats['corrections_by_class'][entry.original_prediction][entry.corrected_class] += 1

        # Actualizar usuarios únicos
        if entry.user_id:
            user_set = set()
            for e in self.feedback_entries:
                if e.user_id:
                    user_set.add(e.user_id)
            self.stats['unique_users'] = len(user_set)

        # Actualizar class_corrections para sugerencias
        if entry.original_prediction not in self.class_corrections:
            self.class_corrections[entry.original_prediction] = {}

        if entry.corrected_class not in self.class_corrections[entry.original_prediction]:
            self.class_corrections[entry.original_prediction][entry.corrected_class] = 0

        self.class_corrections[entry.original_prediction][entry.corrected_class] += 1

        self.stats['last_updated'] = time.time()
        self._save_stats()

    def _load_feedback_data(self) -> None:
        """Carga los datos de feedback desde el archivo."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feedback_entries = [FeedbackEntry(**entry) for entry in data]
                    print(f"✓ Cargadas {len(self.feedback_entries)} entradas de feedback")
            except Exception as e:
                print(f"⚠ Error cargando datos de feedback: {e}")

    def _save_feedback_data(self) -> None:
        """Guarda los datos de feedback al archivo."""
        try:
            data = [asdict(entry) for entry in self.feedback_entries]
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error guardando datos de feedback: {e}")

    def _load_user_stats(self) -> None:
        """Carga las estadísticas de usuario desde el archivo."""
        if self.user_stats_file.exists():
            try:
                with open(self.user_stats_file, 'r', encoding='utf-8') as f:
                    self.user_stats.update(json.load(f))
                    print(f"✓ Cargadas estadísticas de {len(self.user_stats)} usuarios")
            except Exception as e:
                print(f"⚠ Error cargando estadísticas de usuario: {e}")

    def _load_stats(self) -> None:
        """Carga las estadísticas desde el archivo."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.stats.update(json.load(f))
            except Exception as e:
                print(f"⚠ Error cargando estadísticas: {e}")

    def _save_stats(self) -> None:
        """Guarda las estadísticas al archivo."""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error guardando estadísticas: {e}")

    def _save_user_stats(self) -> None:
        """Guarda las estadísticas de usuario al archivo."""
        try:
            with open(self.user_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error guardando estadísticas de usuario: {e}")

    def cleanup_old_entries(self, max_age_days: int = 365) -> int:
        """
        Limpia entradas de feedback antiguas.

        Args:
            max_age_days: Edad máxima en días para mantener entradas

        Returns:
            Número de entradas eliminadas
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        original_count = len(self.feedback_entries)

        self.feedback_entries = [entry for entry in self.feedback_entries if entry.timestamp > cutoff_time]

        removed_count = original_count - len(self.feedback_entries)
        if removed_count > 0:
            self._save_feedback_data()
            print(f"🧹 Limpiadas {removed_count} entradas de feedback antiguas")

        return removed_count

    def get_user_level(self, user_id: str) -> Dict[str, Any]:
        """
        Obtiene el nivel y estadísticas de un usuario.

        Args:
            user_id: ID del usuario

        Returns:
            Diccionario con información del nivel
        """
        points = self.user_stats.get(user_id, {}).get('points', 0)

        current_level = 0
        next_level_threshold = 0

        for level, info in self.gamification['levels'].items():
            if points >= info['threshold']:
                current_level = level
            else:
                next_level_threshold = info['threshold']
                break

        next_level = current_level + 1
        if next_level not in self.gamification['levels']:
            next_level = current_level
            next_level_threshold = points  # Ya está en el nivel máximo

        points_to_next = next_level_threshold - points

        return {
            'level': current_level,
            'level_name': self.gamification['levels'][current_level]['name'],
            'points': points,
            'points_to_next': max(0, points_to_next),
            'next_level': next_level,
            'next_level_name': self.gamification['levels'].get(next_level, {}).get('name', 'Máximo'),
            'total_corrections': self.user_stats.get(user_id, {}).get('corrections', 0),
            'accuracy_rate': self.user_stats.get(user_id, {}).get('accuracy_rate', 0.0)
        }

    def award_points(self, user_id: str, points: int, reason: str = "correction"):
        """
        Otorga puntos a un usuario.

        Args:
            user_id: ID del usuario
            points: Puntos a otorgar
            reason: Razón de los puntos
        """
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                'points': 0,
                'corrections': 0,
                'validations': 0,
                'accuracy_streak': 0,
                'last_activity': time.time()
            }

        self.user_stats[user_id]['points'] += points
        self.user_stats[user_id]['last_activity'] = time.time()

        if reason == 'correction':
            self.user_stats[user_id]['corrections'] += 1
        elif reason == 'validation':
            self.user_stats[user_id]['validations'] += 1

        # Actualizar leaderboard
        self._update_leaderboard()

        # Guardar estadísticas de usuario
        self._save_user_stats()

        # Rastrear en analytics
        analytics_tracker.track_event('gamification_points', {
            'user_id': user_id,
            'points_awarded': points,
            'reason': reason,
            'total_points': self.user_stats[user_id]['points']
        })

    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene el leaderboard de usuarios.

        Args:
            limit: Número máximo de usuarios a retornar

        Returns:
            Lista de usuarios ordenados por puntos
        """
        return self.leaderboard[:limit]

    def _update_leaderboard(self):
        """Actualiza el leaderboard."""
        self.leaderboard = []
        for user_id, stats in self.user_stats.items():
            level_info = self.get_user_level(user_id)
            self.leaderboard.append({
                'user_id': user_id,
                'points': stats['points'],
                'level': level_info['level'],
                'level_name': level_info['level_name'],
                'corrections': stats['corrections'],
                'last_activity': stats['last_activity']
            })

        # Ordenar por puntos descendente
        self.leaderboard.sort(key=lambda x: x['points'], reverse=True)


# Instancia global del gestor de feedback
feedback_manager = FeedbackManager()