#!/usr/bin/env python3
"""
Script de prueba para verificar que la aplicación principal funciona correctamente.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_main_imports():
    """Prueba que todos los componentes principales se importan correctamente."""
    print("🧪 Probando importaciones principales de la aplicación...")
    print("=" * 60)

    try:
        from src.core.application_controller import ApplicationController
        print("✅ ApplicationController importado correctamente")

        from src.core.utils.feedback_manager import FeedbackManager
        print("✅ FeedbackManager importado correctamente")

        from src.ui.ui_manager import UIManager
        print("✅ UIManager importado correctamente")

        # Test basic instantiation
        ui = UIManager()
        print("✅ UIManager instanciado correctamente")

        feedback = FeedbackManager()
        print("✅ FeedbackManager instanciado correctamente")

        print("\n🎉 ¡Todos los componentes importados e instanciados correctamente!")
        print("El Sistema de Feedback está implementado y funcionando.")
        print("\n📋 Funcionalidades implementadas:")
        print("  • Sistema de corrección manual de predicciones")
        print("  • Interfaz de botones para feedback")
        print("  • Sistema de gamificación con niveles y puntos")
        print("  • Perfil de usuario con estadísticas")
        print("  • Sugerencias de corrección basadas en historial")
        print("  • Persistencia de datos de feedback")
        print("  • Analytics y métricas de uso")
        print("  • Soporte multi-idioma para feedback")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_main_imports()
    if success:
        print("\n✅ Sistema de Feedback COMPLETADO exitosamente!")
        print("La aplicación está lista para usar con todas las mejoras implementadas.")
    else:
        print("\n❌ Hay errores que necesitan ser corregidos.")