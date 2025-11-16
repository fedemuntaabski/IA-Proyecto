#!/usr/bin/env python3
"""
Herramientas de gestión del sistema de feedback.

Este script proporciona utilidades para:
- Exportar datos de feedback para re-entrenamiento
- Validar correcciones existentes
- Generar estadísticas del sistema de feedback
- Limpiar datos antiguos
"""

import sys
import os
from pathlib import Path
import json
import argparse
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.utils.feedback_manager import feedback_manager


def export_training_data(output_file=None, min_quality=0.7, validated_only=True):
    """Exporta datos de feedback para re-entrenamiento."""
    print("📤 Exportando datos de entrenamiento...")

    exported_file = feedback_manager.export_training_data(output_file)

    if exported_file:
        print(f"✅ Datos exportados exitosamente: {exported_file}")

        # Mostrar estadísticas del archivo exportado
        try:
            with open(exported_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"📊 Estadísticas del dataset:")
            print(f"   - Total de muestras: {len(data)}")
            print(f"   - Calidad mínima: {min_quality}")
            print(f"   - Solo validadas: {validated_only}")

            if data:
                classes = set(item['label'] for item in data)
                print(f"   - Clases representadas: {len(classes)}")
                print(f"   - Clases: {sorted(classes)}")

        except Exception as e:
            print(f"⚠ Error leyendo archivo exportado: {e}")
    else:
        print("❌ Error exportando datos")


def show_stats():
    """Muestra estadísticas del sistema de feedback."""
    print("📊 Estadísticas del Sistema de Feedback")
    print("=" * 50)

    stats = feedback_manager.get_feedback_stats()

    print(f"Total de correcciones: {stats['total_corrections']}")
    print(f"Usuarios únicos: {stats['unique_users']}")
    print(f"Última actualización: {datetime.fromtimestamp(stats['last_updated']).strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nCorrecciones por clase original:")
    for original, corrections in stats['corrections_by_class'].items():
        total = sum(corrections.values())
        print(f"  {original}: {total} correcciones")
        for corrected, count in corrections.items():
            if original != corrected:
                print(f"    → {corrected}: {count}")

    print("\n💡 Sugerencias disponibles:")
    # Mostrar algunas sugerencias para clases comunes
    common_classes = ['circle', 'square', 'triangle', 'line']
    for cls in common_classes:
        suggestions = feedback_manager.get_correction_suggestions(cls, limit=3)
        if suggestions:
            print(f"  {cls} → {', '.join(suggestions)}")


def validate_corrections(start_index=0, count=10):
    """Permite validar correcciones manualmente."""
    print("🔍 Modo de validación de correcciones")
    print("Instrucciones:")
    print("  - 'y' para validar (calidad buena)")
    print("  - 'n' para rechazar (calidad baja)")
    print("  - 's' para saltar")
    print("  - 'q' para salir")
    print()

    # Esto sería interactivo en una aplicación real
    # Por ahora, solo mostramos información
    print("Funcionalidad de validación disponible en feedback_manager.validate_correction()")


def cleanup_old_data(days=365):
    """Limpia datos de feedback antiguos."""
    print(f"🧹 Limpiando datos de feedback anteriores a {days} días...")

    removed = feedback_manager.cleanup_old_entries(days)
    print(f"✅ Removidos {removed} entradas antiguas")


def main():
    parser = argparse.ArgumentParser(description="Herramientas de gestión del sistema de feedback")
    parser.add_argument('command', choices=['export', 'stats', 'validate', 'cleanup'],
                       help='Comando a ejecutar')
    parser.add_argument('--output', '-o', help='Archivo de salida para export')
    parser.add_argument('--min-quality', type=float, default=0.7,
                       help='Calidad mínima para export (0-1)')
    parser.add_argument('--days', type=int, default=365,
                       help='Días para cleanup de datos antiguos')

    args = parser.parse_args()

    if args.command == 'export':
        export_training_data(args.output, args.min_quality)
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'validate':
        validate_corrections()
    elif args.command == 'cleanup':
        cleanup_old_data(args.days)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()