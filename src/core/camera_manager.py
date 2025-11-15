"""
Camera Manager - Gestor de Captura de Video.

Este módulo maneja la captura de video de la cámara y el loop principal
de procesamiento de frames, separando esta responsabilidad del main.py.
"""

import cv2
import numpy as np
from typing import Optional, Callable
from .i18n import _


class CameraManager:
    """
    Gestor de captura de video y loop principal.

    Responsabilidades:
    - Inicializar y configurar la cámara
    - Gestionar el loop principal de captura y procesamiento
    - Manejar eventos de teclado
    - Gestionar limpieza de recursos
    """

    def __init__(self, width: int = 640, height: int = 480):
        """
        Inicializa el gestor de cámara.

        Args:
            width: Ancho de la resolución de video
            height: Alto de la resolución de video
        """
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def initialize_camera(self) -> bool:
        """
        Inicializa la cámara.

        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        print(_("📹 Iniciando cámara..."))
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print(_("❌ Error: No se pudo acceder a la cámara"))
            return False

        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print(_("✅ Cámara inicializada"))
        return True

    def run_main_loop(self, frame_processor: Callable[[np.ndarray], np.ndarray],
                     key_handler: Callable[[int], bool],
                     window_title: str = "Air Draw Classifier - IA Proyecto") -> None:
        """
        Ejecuta el loop principal de captura y procesamiento.

        Args:
            frame_processor: Función que procesa cada frame
            key_handler: Función que maneja eventos de teclado
            window_title: Título de la ventana
        """
        if not self.cap:
            return

        print("\n🎮 Controles:")
        print(_("ESPACIO: Forzar nueva clasificación"))
        print(_("R: Limpiar dibujo actual"))
        print(_("H: Mostrar/ocultar ayuda"))
        print(_("Q: Salir"))
        print("\n💡 Instrucciones:")
        print(_("1. Muestra tu mano a la cámara"))
        print(_("2. Dibuja una figura en el aire con el dedo índice"))
        print(_("3. La app intentará adivinar qué dibujaste"))
        print("\n" + "=" * 50)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Procesar frame
                display_frame = frame_processor(frame)

                # Mostrar frame
                cv2.imshow(window_title, display_frame)

                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF

                if key_handler(key):
                    break

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Libera los recursos de la cámara."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()