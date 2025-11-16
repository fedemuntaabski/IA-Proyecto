#!/usr/bin/env python
"""
Test script - Verificar que todos los módulos nuevos están correctamente integrados.
"""

import sys
sys.path.insert(0, '.')

def test_modules():
    """Prueba la carga de los módulos nuevos."""
    print("\n🧪 Iniciando pruebas de módulos...")
    print("=" * 60)

    # Test 1: UserPreferences
    try:
        from src.core.utils.user_preferences import UserPreferences
        prefs = UserPreferences()
        print(f"✅ UserPreferences cargado")
        print(f"   - Tema actual: {prefs.get_theme()}")
        print(f"   - Idioma: {prefs.get_language()}")
        print(f"   - Sonidos habilitados: {prefs.is_sound_enabled()}")
    except Exception as e:
        print(f"❌ Error en UserPreferences: {e}")

    # Test 2: SoundManager
    try:
        from src.core.utils.sound_manager import sound_manager
        print(f"✅ SoundManager cargado")
        print(f"   - Audio disponible: {sound_manager.is_available()}")
        print(f"   - Volumen: {sound_manager.volume}")
    except Exception as e:
        print(f"❌ Error en SoundManager: {e}")

    # Test 3: ErrorHandler
    try:
        from src.core.utils.error_handler import ErrorMessages
        print(f"✅ ErrorHandler cargado")
        print(f"   - Mensajes disponibles: {len(ErrorMessages.MESSAGES)}")
    except Exception as e:
        print(f"❌ Error en ErrorHandler: {e}")

    # Test 4: StatisticsTracker
    try:
        from src.core.utils.statistics_tracker import statistics_tracker
        print(f"✅ StatisticsTracker cargado")
        print(f"   - Frames registrados: {statistics_tracker.session_stats['total_frames']}")
    except Exception as e:
        print(f"❌ Error en StatisticsTracker: {e}")

    # Test 5: PerformanceMonitor
    try:
        from src.core.utils.performance_monitor import performance_monitor
        print(f"✅ PerformanceMonitor cargado")
        status = performance_monitor.get_status()
        print(f"   - Memoria del sistema: {status.get('system_memory_percent', 'N/A')}%")
    except Exception as e:
        print(f"❌ Error en PerformanceMonitor: {e}")

    # Test 6: SettingsManager
    try:
        from src.core.utils.settings_manager import settings_manager
        print(f"✅ SettingsManager cargado")
        valid, errors = settings_manager.validate()
        print(f"   - Configuración válida: {'Sí' if valid else 'No'}")
        if errors:
            for error in errors[:2]:
                print(f"   - Error: {error}")
    except Exception as e:
        print(f"❌ Error en SettingsManager: {e}")

    # Test 7: Constantes actualizadas
    try:
        from src.core.utils.constants import COLOR_SUCCESS, COLOR_ERROR, DEFAULT_BAR_HEIGHT
        print(f"✅ Constantes cargadas correctamente")
        print(f"   - Color éxito: {COLOR_SUCCESS}")
        print(f"   - Alto de barra: {DEFAULT_BAR_HEIGHT}px")
    except Exception as e:
        print(f"❌ Error en Constantes: {e}")

    print("=" * 60)
    print("✅ Pruebas completadas\n")


if __name__ == "__main__":
    test_modules()
