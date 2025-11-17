"""
game_main.py - Punto de entrada para el modo de juego

Inicializa todos los componentes y ejecuta la aplicación integrada.
"""

import sys
import logging
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

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


def main():
    """Función principal."""
    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description="Pictionary Live - Modo Juego con Tkinter"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="ID de la cámara a usar (default: 0)"
    )
    parser.add_argument(
        "--ia-dir",
        type=str,
        default="./IA",
        help="Ruta a la carpeta IA con el modelo (default: ./IA)"
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["cyberpunk", "light", "dark"],
        default="cyberpunk",
        help="Tema de colores (default: cyberpunk)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Habilitar modo debug con logging detallado"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logger = setup_logging(debug=args.debug)
    
    logger.info("=" * 70)
    logger.info("PICTIONARY LIVE - MODO JUEGO")
    logger.info("=" * 70)
    
    try:
        # Inicializar componentes
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
        
    except KeyboardInterrupt:
        logger.info("\n[STOP] Interrupción por usuario (Ctrl+C)")
    except Exception as e:
        logger.error(f"[ERROR] Error fatal: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("=" * 70)
        logger.info("Aplicación finalizada")


if __name__ == "__main__":
    main()
