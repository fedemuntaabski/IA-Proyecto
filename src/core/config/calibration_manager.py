"""
Sistema de Calibración Automática para Detección de Manos.

Este módulo implementa un sistema de calibración que permite ajustar
automáticamente los parámetros de detección de piel basándose en
muestras del usuario.
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from .constants import CALIBRATION_SAMPLES_SKIN, CALIBRATION_SAMPLES_BACKGROUND, CALIBRATION_CONFIG_FILE


class CalibrationManager:
    """
    Maneja la calibración automática de parámetros para detección de piel.

    Atributos:
        config_file: Archivo donde guardar la configuración
        skin_samples: Lista de muestras de piel tomadas
        background_samples: Lista de muestras de fondo tomadas
        calibrated_ranges: Rangos HSV calibrados
    """

    def __init__(self, config_file: str = CALIBRATION_CONFIG_FILE):
        """
        Inicializa el administrador de calibración.

        Args:
            config_file: Archivo de configuración para guardar calibración
        """
        self.config_file = Path(__file__).parent.parent.parent / config_file
        self.skin_samples: List[np.ndarray] = []
        self.background_samples: List[np.ndarray] = []
        self.calibrated_ranges: Optional[Dict[str, Tuple]] = None
        self.is_calibrated = False

        # Cargar configuración existente si existe
        self._load_calibration()

        print("✓ CalibrationManager inicializado")

    def _load_calibration(self) -> None:
        """Carga configuración de calibración desde archivo."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)

                self.calibrated_ranges = data.get('ranges', {})
                self.is_calibrated = data.get('is_calibrated', False)

                print(f"✓ Configuración cargada desde {self.config_file}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠ Error cargando configuración: {e}")
                self.is_calibrated = False
        else:
            print("ℹ No se encontró configuración previa")

    def _save_calibration(self) -> None:
        """Guarda la configuración actual en archivo."""
        data = {
            'is_calibrated': self.is_calibrated,
            'ranges': self.calibrated_ranges or {},
            'timestamp': str(np.datetime64('now'))
        }

        try:
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Configuración guardada en {self.config_file}")
        except Exception as e:
            print(f"⚠ Error guardando configuración: {e}")

    def add_skin_sample(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """
        Agrega una muestra de piel desde una región de interés.

        Args:
            frame: Frame de la cámara (BGR)
            roi: Región de interés (x, y, w, h)

        Returns:
            True si la muestra fue agregada exitosamente
        """
        try:
            x, y, w, h = roi

            # Validar límites
            h_frame, w_frame = frame.shape[:2]
            x_end, y_end = x + w, y + h

            if x < 0 or y < 0 or x_end > w_frame or y_end > h_frame:
                print("⚠ Región de interés fuera de límites")
                return False

            # Extraer región
            roi_img = frame[y:y_end, x:x_end]

            # Convertir a HSV
            hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

            # Agregar muestra
            self.skin_samples.append(hsv_roi)

            print(f"✓ Muestra de piel agregada ({len(self.skin_samples)} total)")
            return True

        except Exception as e:
            print(f"⚠ Error agregando muestra de piel: {e}")
            return False

    def add_background_sample(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """
        Agrega una muestra de fondo desde una región de interés.

        Args:
            frame: Frame de la cámara (BGR)
            roi: Región de interés (x, y, w, h)

        Returns:
            True si la muestra fue agregada exitosamente
        """
        try:
            x, y, w, h = roi

            # Validar límites
            h_frame, w_frame = frame.shape[:2]
            x_end, y_end = x + w, y + h

            if x < 0 or y < 0 or x_end > w_frame or y_end > h_frame:
                print("⚠ Región de interés fuera de límites")
                return False

            # Extraer región
            roi_img = frame[y:y_end, x:x_end]

            # Convertir a HSV
            hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

            # Agregar muestra
            self.background_samples.append(hsv_roi)

            print(f"✓ Muestra de fondo agregada ({len(self.background_samples)} total)")
            return True

        except Exception as e:
            print(f"⚠ Error agregando muestra de fondo: {e}")
            return False

    def calculate_skin_ranges(self) -> Dict[str, Tuple]:
        """
        Calcula los rangos óptimos de piel basándose en las muestras tomadas.

        Returns:
            Diccionario con rangos HSV (lower, upper) para piel
        """
        if len(self.skin_samples) == 0:
            print("⚠ No hay muestras de piel para calcular rangos")
            return {}

        try:
            # Combinar todas las muestras
            all_pixels = np.concatenate([sample.reshape(-1, 3) for sample in self.skin_samples])

            # Calcular estadísticas
            h_mean, h_std = np.mean(all_pixels[:, 0]), np.std(all_pixels[:, 0])
            s_mean, s_std = np.mean(all_pixels[:, 1]), np.std(all_pixels[:, 1])
            v_mean, v_std = np.mean(all_pixels[:, 2]), np.std(all_pixels[:, 2])

            # Calcular rangos con margen de seguridad (2-3 desviaciones estándar)
            h_margin = max(5, 2.5 * h_std)  # Mínimo 5 para estabilidad
            s_margin = max(20, 2.5 * s_std)
            v_margin = max(30, 2.5 * v_std)

            # Rangos finales
            ranges = {
                'skin_lower': (
                    max(0, int(h_mean - h_margin)),
                    max(20, int(s_mean - s_margin)),
                    max(40, int(v_mean - v_margin))
                ),
                'skin_upper': (
                    min(179, int(h_mean + h_margin)),
                    min(255, int(s_mean + s_margin)),
                    min(255, int(v_mean + v_margin))
                )
            }

            print("✓ Rangos de piel calculados:")
            print(f"  HSV Lower: {ranges['skin_lower']}")
            print(f"  HSV Upper: {ranges['skin_upper']}")

            return ranges

        except Exception as e:
            print(f"⚠ Error calculando rangos de piel: {e}")
            return {}

    def calculate_background_ranges(self) -> Dict[str, Tuple]:
        """
        Calcula rangos para excluir fondo basado en muestras.

        Returns:
            Diccionario con rangos HSV para excluir fondo
        """
        if len(self.background_samples) == 0:
            print("ℹ No hay muestras de fondo")
            return {}

        try:
            # Combinar todas las muestras
            all_pixels = np.concatenate([sample.reshape(-1, 3) for sample in self.background_samples])

            # Calcular estadísticas
            h_mean, h_std = np.mean(all_pixels[:, 0]), np.std(all_pixels[:, 0])
            s_mean, s_std = np.mean(all_pixels[:, 1]), np.std(all_pixels[:, 1])
            v_mean, v_std = np.mean(all_pixels[:, 2]), np.std(all_pixels[:, 2])

            # Rangos para excluir fondo (invertidos)
            ranges = {
                'background_exclude_lower': (
                    max(0, int(h_mean - 3 * h_std)),
                    max(0, int(s_mean - 3 * s_std)),
                    max(0, int(v_mean - 3 * v_std))
                ),
                'background_exclude_upper': (
                    min(179, int(h_mean + 3 * h_std)),
                    min(255, int(s_mean + 3 * s_std)),
                    min(255, int(v_mean + 3 * v_std))
                )
            }

            print("✓ Rangos de fondo calculados para exclusión")
            return ranges

        except Exception as e:
            print(f"⚠ Error calculando rangos de fondo: {e}")
            return {}

    def calibrate(self) -> bool:
        """
        Realiza la calibración completa basada en muestras tomadas.

        Returns:
            True si la calibración fue exitosa
        """
        if len(self.skin_samples) < 2:
            print("⚠ Se necesitan al menos 2 muestras de piel para calibrar")
            return False

        try:
            # Calcular rangos
            skin_ranges = self.calculate_skin_ranges()
            bg_ranges = self.calculate_background_ranges()

            if not skin_ranges:
                print("⚠ No se pudieron calcular rangos de piel")
                return False

            # Combinar rangos
            self.calibrated_ranges = {**skin_ranges, **bg_ranges}
            self.is_calibrated = True

            # Guardar configuración
            self._save_calibration()

            print("✓ Calibración completada exitosamente")
            return True

        except Exception as e:
            print(f"⚠ Error durante calibración: {e}")
            return False

    def get_calibrated_ranges(self) -> Optional[Dict[str, Tuple]]:
        """
        Obtiene los rangos calibrados actuales.

        Returns:
            Diccionario con rangos HSV o None si no calibrado
        """
        return self.calibrated_ranges if self.is_calibrated else None

    def reset_calibration(self) -> None:
        """Resetea la calibración y elimina configuración guardada."""
        self.skin_samples.clear()
        self.background_samples.clear()
        self.calibrated_ranges = None
        self.is_calibrated = False

        if self.config_file.exists():
            try:
                os.remove(self.config_file)
                print("✓ Configuración eliminada")
            except Exception as e:
                print(f"⚠ Error eliminando configuración: {e}")

    def get_calibration_status(self) -> Dict:
        """
        Obtiene el estado actual de la calibración.

        Returns:
            Diccionario con información del estado
        """
        return {
            'is_calibrated': self.is_calibrated,
            'skin_samples_count': len(self.skin_samples),
            'background_samples_count': len(self.background_samples),
            'has_config_file': self.config_file.exists(),
            'ranges_available': self.calibrated_ranges is not None
        }

    def apply_illumination_compensation(self, ranges: Dict[str, Tuple],
                                       frame_brightness: float) -> Dict[str, Tuple]:
        """
        Ajusta rangos basándose en la iluminación del frame actual.

        Args:
            ranges: Rangos base calibrados
            frame_brightness: Brillo promedio del frame (0-255)

        Returns:
            Rangos ajustados por iluminación
        """
        if not ranges or 'skin_lower' not in ranges or 'skin_upper' not in ranges:
            return ranges

        try:
            # Factor de compensación basado en brillo
            # Si el frame es más oscuro, expandir rangos V
            # Si es más brillante, contraer rangos V
            brightness_factor = frame_brightness / 128.0  # 128 es brillo medio

            adjusted_ranges = ranges.copy()

            # Ajustar componente V (brillo)
            lower_h, lower_s, lower_v = ranges['skin_lower']
            upper_h, upper_s, upper_v = ranges['skin_upper']

            if brightness_factor < 1.0:  # Frame oscuro
                # Expandir rango V hacia abajo
                new_lower_v = max(0, lower_v - int(20 * (1 - brightness_factor)))
                adjusted_ranges['skin_lower'] = (lower_h, lower_s, new_lower_v)
            else:  # Frame brillante
                # Expandir rango V hacia arriba
                new_upper_v = min(255, upper_v + int(20 * (brightness_factor - 1)))
                adjusted_ranges['skin_upper'] = (upper_h, upper_s, new_upper_v)

            return adjusted_ranges

        except Exception as e:
            print(f"⚠ Error en compensación de iluminación: {e}")
            return ranges


class CalibrationUI:
    """
    Interfaz de usuario para el proceso de calibración.
    """

    def __init__(self, calibration_manager: CalibrationManager):
        """
        Inicializa la interfaz de calibración.

        Args:
            calibration_manager: Instancia del administrador de calibración
        """
        self.calibration_manager = calibration_manager
        self.selecting_roi = False
        self.roi_start = None
        self.roi_end = None
        self.current_step = "skin"  # "skin" o "background"

    def draw_calibration_interface(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja la interfaz de calibración en el frame.

        Args:
            frame: Frame original

        Returns:
            Frame con interfaz dibujada
        """
        display_frame = frame.copy()
        height, width = frame.shape[:2]

        # Información de estado
        status = self.calibration_manager.get_calibration_status()

        # Título
        cv2.putText(display_frame, "CALIBRACION - Paso a paso",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Instrucciones según paso actual
        if self.current_step == "skin":
            cv2.putText(display_frame, f"Muestras de piel: {status['skin_samples_count']}/{CALIBRATION_SAMPLES_SKIN}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, "1. Coloca mano en area verde y presiona ESPACIO",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "2. Repite 3 veces en diferentes posiciones",
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Área de muestreo (verde)
            cv2.rectangle(display_frame, (width//3, height//3),
                         (2*width//3, 2*height//3), (0, 255, 0), 2)

        elif self.current_step == "background":
            cv2.putText(display_frame, f"Muestras de fondo: {status['background_samples_count']}/{CALIBRATION_SAMPLES_BACKGROUND}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, "3. Coloca mano fuera del area y presiona ESPACIO",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "4. Repite 2 veces",
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Área de muestreo (roja)
            cv2.rectangle(display_frame, (width//4, height//4),
                         (3*width//4, 3*height//4), (0, 0, 255), 2)

        # Controles
        cv2.putText(display_frame, "ESPACIO: Tomar muestra | R: Reset | C: Calibrar | Q: Salir",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display_frame

    def handle_key_press(self, key: int, frame: np.ndarray) -> str:
        """
        Maneja eventos de teclado durante calibración.

        Args:
            key: Código de tecla presionada
            frame: Frame actual

        Returns:
            Comando resultante ("continue", "calibrate", "reset", "quit")
        """
        if key == ord(' '):  # ESPACIO
            return self._take_sample(frame)
        elif key == ord('r'):  # R
            self.calibration_manager.reset_calibration()
            self.current_step = "skin"
            print("✓ Calibración reseteada")
            return "continue"
        elif key == ord('c'):  # C
            if self._can_calibrate():
                success = self.calibration_manager.calibrate()
                return "calibrate" if success else "continue"
            else:
                print("⚠ Insuficientes muestras para calibrar")
                return "continue"
        elif key == ord('q'):  # Q
            return "quit"

        return "continue"

    def _take_sample(self, frame: np.ndarray) -> str:
        """Toma una muestra según el paso actual."""
        height, width = frame.shape[:2]

        if self.current_step == "skin":
            # Área central para piel
            roi = (width//3, height//3, width//3, height//3)
            success = self.calibration_manager.add_skin_sample(frame, roi)

            if success:
                status = self.calibration_manager.get_calibration_status()
                if status['skin_samples_count'] >= 3:
                    self.current_step = "background"
                    print("✓ Muestras de piel completadas. Ahora fondo.")

        elif self.current_step == "background":
            # Área amplia para fondo
            roi = (width//4, height//4, width//2, height//2)
            success = self.calibration_manager.add_background_sample(frame, roi)

            if success:
                status = self.calibration_manager.get_calibration_status()
                if status['background_samples_count'] >= 2:
                    print("✓ Muestras de fondo completadas. Presiona 'C' para calibrar.")

        return "continue"

    def _can_calibrate(self) -> bool:
        """Verifica si hay suficientes muestras para calibrar."""
        status = self.calibration_manager.get_calibration_status()
        return (status['skin_samples_count'] >= CALIBRATION_SAMPLES_SKIN and
                status['background_samples_count'] >= CALIBRATION_SAMPLES_BACKGROUND)