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
import logging
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
# from app import PictionaryLive  # Eliminado - solo modo de juego


def run_game_mode(args):
    """
    Ejecuta el modo de juego con interfaz Tkinter.
    
    Args:
        args: Argumentos parseados del parser
    """
    print("[INFO] Iniciando modo de juego con interfaz Tkinter...")
    
    try:
        # Importar componentes del modo de juego
        from config import (
            MEDIAPIPE_CONFIG, CAMERA_CONFIG, STROKE_CONFIG, MODEL_CONFIG,
            PREPROCESSING_CONFIG, DETECTION_CONFIG, PERFORMANCE_CONFIG
        )
        from hand_detector import HandDetector
        from stroke_manager import StrokeAccumulator
        from drawing_preprocessor import DrawingPreprocessor
        from model import SketchClassifier
        from logger_setup import setup_logging
        from game_mode import GameConfig
        from game_integration import GameIntegration, IntegrationConfig
        
        # Configurar logging
        logger = setup_logging(debug=args.debug)
        
        logger.info("=" * 70)
        logger.info("PICTIONARY LIVE - MODO JUEGO")
        logger.info("=" * 70)
        
        # Inicializar componentes con manejo de errores
        logger.info("\n[INIT] Inicializando componentes...")
        
        # Hand Detector
        hand_config = {**MEDIAPIPE_CONFIG["hands"], **DETECTION_CONFIG, **PERFORMANCE_CONFIG}
        hand_detector = HandDetector(hand_config, logger)
        logger.info("  ✓ HandDetector inicializado")
        
        # Stroke Accumulator
        stroke_accumulator = StrokeAccumulator(STROKE_CONFIG, logger)
        logger.info("  ✓ StrokeAccumulator inicializado")
        
        # Classifier
        classifier = SketchClassifier(args.ia_dir, logger, demo_mode=MODEL_CONFIG.get("demo_mode", True), config=MODEL_CONFIG)
        logger.info(f"  ✓ SketchClassifier inicializado ({len(classifier.get_labels())} clases)")
        
        # Preprocessor
        input_shape = classifier.get_input_shape() if classifier else [28, 28, 1]
        preprocessor = DrawingPreprocessor(input_shape, PREPROCESSING_CONFIG)
        logger.info("  ✓ DrawingPreprocessor inicializado")
        
        # Game Config
        game_config = GameConfig(theme=args.theme)
        logger.info(f"  ✓ GameConfig inicializado (tema: {args.theme})")
        
        # Integration Config
        integration_config = IntegrationConfig(
            camera_id=args.camera_id,
            ia_dir=args.ia_dir,
            debug=args.debug
        )
        logger.info("  ✓ IntegrationConfig inicializado")
        
        # Game Integration
        logger.info("\n[INIT] Inicializando GameIntegration...")
        game_integration = GameIntegration(
            hand_detector=hand_detector,
            stroke_accumulator=stroke_accumulator,
            preprocessor=preprocessor,
            classifier=classifier,
            game_config=game_config,
            integration_config=integration_config,
            logger=logger,
        )
        logger.info("  ✓ GameIntegration inicializada")
        
        # Iniciar aplicación
        logger.info("\n[START] Iniciando aplicación...")
        logger.info("=" * 70)
        game_integration.run()
        
    except ImportError as e:
        print(f"[ERROR] Error al importar módulos del modo de juego: {e}", file=sys.stderr)
        print("Sugerencias:")
        print("- Asegúrate de que todos los archivos del modo de juego existen")
        print("- Verifica que pillow esté instalado: pip install pillow")
        print("- Revisa que Tkinter esté disponible en tu instalación de Python")
        raise
    except Exception as e:
        logger = logging.getLogger("GameMode") if 'logger' in locals() else None
        if logger:
            logger.error(f"Error fatal en modo de juego: {e}", exc_info=True)
        else:
            print(f"[ERROR] Error fatal en modo de juego: {e}", file=sys.stderr)
        raise


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Pictionary Live - Juega Pictionary dibujando en el aire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                           # Iniciar el juego
  python main.py --theme light             # Tema claro
  python main.py --theme dark              # Tema oscuro
  python main.py --camera-id 1             # Cámara específica
  python main.py --debug                   # Logging detallado

Características:
  - Interfaz Tkinter con predicción de palabras
  - Selección aleatoria de palabras del modelo
  - Sistema de puntuación y rachas
  - Múltiples temas disponibles

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
        "--theme",
        type=str,
        choices=["cyberpunk", "light", "dark"],
        default="cyberpunk",
        help="Tema de colores para el modo de juego (default: cyberpunk)"
    )

    args = parser.parse_args()

    try:
        # Ejecutar modo de juego
        run_game_mode(args)

    except RuntimeError as e:
        print(f"\n[ERROR] Error de configuración: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        print("\nSugerencias:")
        print("- Verifica que la carpeta IA existe y contiene model_info.json")
        print("- Asegúrate de que la cámara no esté en uso por otra aplicación")
        print("- Instala dependencias faltantes: pip install -r src/requirements.txt")
        print("- Verifica que Tkinter esté instalado")
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
