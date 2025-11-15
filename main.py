"""
Air Draw Classifier - Aplicación Principal

Aplicación completa para clasificación de dibujos en el aire que integra:
- Activación de cámara del dispositivo
- Detección de movimientos de manos en tiempo real
- Interpretación de movimientos como trazos/dibujos
- Clasificación automática de figuras dibujadas

Esta aplicación reutiliza componentes avanzados del proyecto para ofrecer
una experiencia completa de dibujo en el aire con IA.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Optional
import threading
import queue

# Importar componentes modulares
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.core.application_controller import ApplicationController
from src.core.camera_manager import CameraManager
from src.core.i18n import _


class AirDrawClassifier:
    """
    Aplicación principal para clasificación de dibujos en el aire.

    Esta clase mantiene la interfaz pública original pero delega
    la lógica a componentes modulares especializados.
    """

    def __init__(self, model_path: str = "IA/sketch_classifier_model.keras",
                 model_info_path: str = "IA/model_info.json"):
        """
        Inicializa la mini-app.

        Args:
            model_path: Ruta al modelo de clasificación
            model_info_path: Ruta a la información del modelo
        """
        # Delegar inicialización al controlador
        self.controller = ApplicationController(model_path, model_info_path)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame de la cámara.

        Args:
            frame: Frame de OpenCV (BGR)

        Returns:
            Frame procesado con visualizaciones
        """
        return self.controller.process_frame(frame)

    def run(self):
        """Ejecuta el loop principal de la mini-app."""
        # Crear gestor de cámara
        camera_manager = CameraManager(
            width=self.controller.app_config['resolution_width'],
            height=self.controller.app_config['resolution_height']
        )

        # Inicializar cámara
        if not camera_manager.initialize_camera():
            return

        # Ejecutar loop principal
        with camera_manager:
            camera_manager.run_main_loop(
                frame_processor=self.process_frame,
                key_handler=self.controller.handle_key_press
            )

        # Mostrar estadísticas finales
        self.controller.get_session_statistics()
        self.controller.cleanup()

    def close(self):
        """Cierra la aplicación y libera recursos."""
        self.controller.cleanup()


def main():
    """Punto de entrada de la mini-app."""
    try:
        app = AirDrawClassifier()
        app.run()
    except KeyboardInterrupt:
        print(f"\n⚠ {_('Interrupción detectada')}")
    except Exception as e:
        print(f"❌ {_('Error en la mini-app')}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()