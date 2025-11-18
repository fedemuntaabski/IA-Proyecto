#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Punto de entrada principal para Pictionary Live

Lanza la aplicación con interfaz PyQt6 moderna.
"""

import os
# Reducir logs de TensorFlow y deshabilitar GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import argparse
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Verificar e instalar dependencias (silencioso)
from dependencies import main as check_dependencies
check_dependencies()


def main():
    """Función principal - lanza directamente PyQt6."""
    parser = argparse.ArgumentParser(
        description="Pictionary Live - Dibuja en el aire con IA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--ia-dir", type=str, default="./IA", 
                       help="Ruta a la carpeta con el modelo")
    parser.add_argument("--camera", type=int, default=0, 
                       help="ID de cámara (default: 0)")
    parser.add_argument("--debug", action="store_true", 
                       help="Habilitar logging detallado")
    parser.add_argument("--theme", type=str, 
                       choices=["cyberpunk", "light", "dark"], 
                       default="cyberpunk",
                       help="Tema de colores (default: cyberpunk)")

    args = parser.parse_args()

    try:
        from app_pyqt import PictionaryLiveQt
        
        # Crear y ejecutar aplicación PyQt6
        app = PictionaryLiveQt(
            ia_dir=args.ia_dir,
            camera_id=args.camera,
            debug=args.debug
        )
        
        sys.exit(app.run())
        
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Instala PyQt6: pip install PyQt6>=6.5.0")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAplicación finalizada")
        sys.exit(0)
    except Exception as e:
        print(f"Error inesperado: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
