#!/usr/bin/env python3
"""
test_model_loading.py - Script para probar la carga del modelo
"""

import os
import sys
from pathlib import Path

# Configurar entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_loading():
    """Prueba cargar el modelo directamente."""
    print("Probando carga del modelo...")

    try:
        from model import SketchClassifier
        import logging

        # Crear logger
        logger = logging.getLogger("TestModel")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
        logger.addHandler(handler)

        # Intentar cargar el modelo
        ia_dir = Path(__file__).parent / "IA"
        classifier = SketchClassifier(str(ia_dir), logger, demo_mode=False)

        if classifier.model is not None:
            print("✅ MODELO CARGADO EXITOSAMENTE")
            print(f"   Forma de entrada: {classifier.get_input_shape()}")
            print(f"   Número de clases: {len(classifier.get_labels())}")

            # Probar una predicción
            import numpy as np
            dummy_drawing = np.random.rand(28, 28, 1).astype(np.float32)
            label, conf, top3 = classifier.predict(dummy_drawing)
            print(f"   Predicción de prueba: {label} ({conf:.1%})")

            return True
        else:
            print("❌ Modelo no cargado - usando modo demo")
            return False

    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)