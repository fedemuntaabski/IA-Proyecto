#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Punto de entrada principal para Pictionary Live

Este archivo es el único punto de entrada de la aplicación.
Maneja la instalación automática de dependencias y el lanzamiento de la app.
"""

import os
# Reducir logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Verificar e instalar dependencias
from dependencies import main as check_dependencies
check_dependencies()

# Importar config_manager para validar configuración al inicio
try:
    from config_manager import get_config
    config = get_config()
    print(f"[INFO] Configuración cargada correctamente desde config.yaml")
except Exception as e:
    print(f"[ERROR] Error al cargar configuración: {e}")
    print("[INFO] Verifica que config.yaml existe y es válido")
    sys.exit(1)

# Importar app después de verificar dependencias y configuración
from app import PictionaryLive


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Pictionary Live - Juega Pictionary dibujando en el aire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py
  python main.py --ia-dir ./IA
  python main.py --debug
  python main.py --dry-run
  python main.py --camera-id 1

Requisitos para mejor detección:
  - Iluminación: Luz frontal blanca uniforme
  - Fondo: Sólido y contrastante (preferentemente verde o azul)
  - Cámara: Resolución 640x480 o superior
  - Distancia: 50-80 cm de la mano
  - Dedos: Índice extendido y visible
        """
    )
    
    parser.add_argument(
        "--ia-dir",
        type=str,
        default="./IA",
        help="Ruta a la carpeta IA con modelo (default: ./IA)"
    )
    
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="ID de cámara (default: 0, prueba 1 o 2 si no funciona)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Habilitar logging detallado (DEBUG)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validar setup sin abrir cámara"
    )
    
    args = parser.parse_args()
    
    try:
        # Crear y ejecutar aplicación
        app = PictionaryLive(
            ia_dir=args.ia_dir,
            camera_id=args.camera_id,
            debug=args.debug,
            dry_run=args.dry_run
        )
        app.run()
    
    except RuntimeError as e:
        print(f"\n[ERROR] Error de configuración: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        print("\nSugerencias:")
        print("- Verifica que la carpeta IA existe y contiene model_info.json")
        print("- Asegúrate de que la cámara no esté en uso por otra aplicación")
        print("- Instala dependencias faltantes: pip install -r src/requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Aplicación finalizada por usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error fatal inesperado: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        print("\nPor favor, revisa los logs en logs/ para más detalles.")
        sys.exit(1)


if __name__ == "__main__":
    main()
