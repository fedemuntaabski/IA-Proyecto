#!/usr/bin/env python3
"""
Script de prueba para el sistema de feedback y gamificación.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.core.utils.feedback_manager import FeedbackManager

def test_feedback_system():
    """Prueba el sistema de feedback y gamificación."""
    print("🧪 Probando sistema de feedback y gamificación...")
    print("=" * 50)

    # Crear instancia local para evitar problemas de importación
    feedback_manager = FeedbackManager()

    # Test básico de funcionalidad
    user_level = feedback_manager.get_user_level('test_user')
    print(f"🏆 Nivel inicial del usuario: {user_level['level_name']} (Nivel {user_level['level']})")
    print(f"⭐ Puntos iniciales: {user_level['points']}")
    print(f"📝 Correcciones iniciales: {user_level['total_corrections']}")
    print()

    # Agregar algunas correcciones de prueba
    print("📝 Agregando correcciones de prueba...")
    feedback_manager.add_correction('circle', 0.5, 'square', [[1,2,3]], user_id='test_user')
    feedback_manager.add_correction('triangle', 0.6, 'circle', [[4,5,6]], user_id='test_user')
    feedback_manager.add_correction('square', 0.7, 'triangle', [[7,8,9]], user_id='test_user')
    print("✅ Correcciones agregadas")
    print()

    # Verificar nivel después de correcciones
    user_level_after = feedback_manager.get_user_level('test_user')
    print(f"🏆 Nivel después de correcciones: {user_level_after['level_name']} (Nivel {user_level_after['level']})")
    print(f"⭐ Puntos después: {user_level_after['points']}")
    print(f"📝 Correcciones totales: {user_level_after['total_corrections']}")
    print(f"🎯 Puntos para siguiente nivel: {user_level_after['points_to_next']}")
    print()

    # Verificar leaderboard
    leaderboard = feedback_manager.get_leaderboard(5)
    print("🏅 Leaderboard (Top 5):")
    for i, entry in enumerate(leaderboard, 1):
        print(f"  {i}. {entry['user_id']}: {entry['points']} pts (Nivel {entry['level']})")
    print()

    # Verificar sugerencias de corrección
    suggestions = feedback_manager.get_correction_suggestions('circle', limit=3)
    print(f"💡 Sugerencias de corrección para 'circle': {suggestions}")
    print()

    print("✅ Prueba del sistema de feedback completada exitosamente!")
    print("=" * 50)

if __name__ == "__main__":
    test_feedback_system()