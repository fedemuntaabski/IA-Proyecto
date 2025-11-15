"""
Algoritmos avanzados de visión por computadora para detección de gestos.

Este módulo implementa técnicas avanzadas como background subtraction,
optical flow y análisis de movimiento para mejorar la detección de gestos.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class BackgroundSubtractionMethod(Enum):
    """Métodos disponibles para background subtraction."""
    MOG2 = "mog2"
    KNN = "knn"
    GMG = "gmg"
    CNT = "cnt"
    GSOC = "gsoc"


class OpticalFlowMethod(Enum):
    """Métodos disponibles para optical flow."""
    LUCAS_KANADE = "lucas_kanade"
    FARNEBACK = "farneback"
    TVL1 = "tvl1"


@dataclass
class MotionVector:
    """Representa un vector de movimiento."""
    x: float
    y: float
    magnitude: float
    angle: float


@dataclass
class GestureMotion:
    """Información de movimiento de un gesto."""
    dominant_direction: float  # Ángulo dominante en grados
    average_speed: float       # Velocidad promedio
    motion_vectors: List[MotionVector]
    stability_score: float    # 0-1, qué tan estable es el movimiento
    gesture_type: str         # "drawing", "pointing", "static", "noise"


class AdvancedVisionProcessor:
    """
    Procesador avanzado de visión con background subtraction y optical flow.

    Esta clase combina múltiples técnicas de visión por computadora para
    mejorar la detección y análisis de gestos de dibujo en el aire.
    """

    def __init__(self,
                 bg_method: BackgroundSubtractionMethod = BackgroundSubtractionMethod.MOG2,
                 flow_method: OpticalFlowMethod = OpticalFlowMethod.LUCAS_KANADE,
                 learning_rate: float = 0.001,
                 history_length: int = 10):
        """
        Inicializa el procesador avanzado de visión.

        Args:
            bg_method: Método de background subtraction
            flow_method: Método de optical flow
            learning_rate: Tasa de aprendizaje para background subtraction
            history_length: Longitud del historial de frames
        """
        self.bg_method = bg_method
        self.flow_method = flow_method
        self.learning_rate = learning_rate
        self.history_length = history_length

        # Inicializar background subtractor
        self.bg_subtractor = self._create_bg_subtractor()

        # Inicializar optical flow
        self.flow_detector = self._create_flow_detector()

        # Historial de frames para análisis temporal
        self.frame_history = []
        self.flow_history = []

        # Estado del procesamiento
        self.is_initialized = False
        self.prev_frame = None
        self.prev_gray = None

        # Parámetros de análisis
        self.min_motion_threshold = 0.5
        self.max_motion_threshold = 50.0
        self.stability_threshold = 0.7

        print("✓ AdvancedVisionProcessor inicializado")
        print(f"  Background subtraction: {bg_method.value}")
        print(f"  Optical flow: {flow_method.value}")

    def _create_bg_subtractor(self):
        """Crea el background subtractor según el método seleccionado."""
        if self.bg_method == BackgroundSubtractionMethod.MOG2:
            return cv2.createBackgroundSubtractorMOG2(
                history=100,
                varThreshold=16,
                detectShadows=True
            )
        elif self.bg_method == BackgroundSubtractionMethod.KNN:
            return cv2.createBackgroundSubtractorKNN(
                history=100,
                dist2Threshold=400.0,
                detectShadows=True
            )
        elif self.bg_method == BackgroundSubtractionMethod.GMG:
            subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
                initializationFrames=20,
                decisionThreshold=0.8
            )
            return subtractor
        elif self.bg_method == BackgroundSubtractionMethod.CNT:
            return cv2.bgsegm.createBackgroundSubtractorCNT(
                minPixelStability=15,
                useHistory=True,
                maxPixelStability=15*60,
                isParallel=True
            )
        elif self.bg_method == BackgroundSubtractionMethod.GSOC:
            return cv2.bgsegm.createBackgroundSubtractorGSOC(
                mc=0.5,
                nSamples=20,
                replaceRate=0.003,
                propagationRate=0.01,
                hitsThreshold=32,
                alpha=0.01,
                beta=0.0022,
                blinkingSupressionDecay=0.1,
                blinkingSupressionMultiplier=0.1,
                noiseRemovalThresholdFacBG=0.0004,
                noiseRemovalThresholdFacFG=0.0008
            )
        else:
            raise ValueError(f"Método de background subtraction no soportado: {self.bg_method}")

    def _create_flow_detector(self):
        """Crea el detector de optical flow según el método seleccionado."""
        if self.flow_method == OpticalFlowMethod.LUCAS_KANADE:
            # Lucas-Kanade parameters
            self.lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            return None  # Se inicializa en cada frame
        elif self.flow_method == OpticalFlowMethod.FARNEBACK:
            return None  # Se usa cv2.calcOpticalFlowFarneback
        elif self.flow_method == OpticalFlowMethod.TVL1:
            return cv2.optflow.DualTVL1OpticalFlow_create()
        else:
            raise ValueError(f"Método de optical flow no soportado: {self.flow_method}")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Procesa un frame usando técnicas avanzadas de visión.

        Args:
            frame: Frame RGB de la cámara

        Returns:
            Diccionario con resultados del procesamiento
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Suavizar para reducir ruido
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Inicializar si es necesario
            if not self.is_initialized:
                self._initialize_processing(gray)
                return self._get_empty_result()

            # Aplicar background subtraction
            fg_mask = self.bg_subtractor.apply(gray, learningRate=self.learning_rate)

            # Limpiar máscara
            fg_mask = self._clean_foreground_mask(fg_mask)

            # Calcular optical flow
            flow = self._calculate_optical_flow(gray)

            # Analizar movimiento
            motion_analysis = self._analyze_motion(flow, fg_mask)

            # Actualizar historial
            self._update_history(gray, flow)

            # Preparar resultado
            result = {
                'foreground_mask': fg_mask,
                'optical_flow': flow,
                'motion_analysis': motion_analysis,
                'frame_gray': gray,
                'processing_success': True
            }

            return result

        except Exception as e:
            print(f"⚠ Error en procesamiento avanzado de visión: {e}")
            return self._get_empty_result()

    def _initialize_processing(self, gray: np.ndarray):
        """Inicializa el procesamiento con el primer frame."""
        self.prev_gray = gray.copy()
        self.is_initialized = True

        # Inicializar puntos de tracking para Lucas-Kanade
        if self.flow_method == OpticalFlowMethod.LUCAS_KANADE:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )

    def _clean_foreground_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        """
        Limpia y mejora la máscara de foreground.

        Args:
            fg_mask: Máscara de foreground cruda

        Returns:
            Máscara limpia y procesada
        """
        # Aplicar operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Erosión para eliminar ruido
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)

        # Dilatación para conectar componentes
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Filtrar por tamaño de componentes
        fg_mask = self._filter_components_by_size(fg_mask)

        return fg_mask

    def _filter_components_by_size(self, mask: np.ndarray,
                                  min_area: int = 100,
                                  max_area: int = 50000) -> np.ndarray:
        """
        Filtra componentes conectados por tamaño.

        Args:
            mask: Máscara binaria
            min_area: Área mínima para mantener componente
            max_area: Área máxima para mantener componente

        Returns:
            Máscara filtrada
        """
        # Encontrar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8, ltype=cv2.CV_32S
        )

        # Crear máscara filtrada
        filtered_mask = np.zeros_like(mask)

        for i in range(1, num_labels):  # Saltar fondo (etiqueta 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                filtered_mask[labels == i] = 255

        return filtered_mask

    def _calculate_optical_flow(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Calcula el optical flow entre frames consecutivos.

        Args:
            gray: Frame actual en escala de grises

        Returns:
            Array de optical flow o None si falla
        """
        try:
            if self.flow_method == OpticalFlowMethod.LUCAS_KANADE:
                if self.prev_points is not None and len(self.prev_points) > 0:
                    # Calcular optical flow
                    new_points, status, error = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.prev_points, None, **self.lk_params
                    )

                    # Filtrar puntos buenos
                    good_new = new_points[status == 1]
                    good_old = self.prev_points[status == 1]

                    # Actualizar puntos para siguiente frame
                    self.prev_points = good_new.reshape(-1, 1, 2)

                    # Crear array de flow completo
                    flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)

                    # Solo puntos tracked tienen flow
                    for old_pt, new_pt in zip(good_old, good_new):
                        old_pt = old_pt.ravel()
                        new_pt = new_pt.ravel()
                        dx, dy = new_pt - old_pt

                        # Asignar flow a región alrededor del punto
                        x, y = int(old_pt[0]), int(old_pt[1])
                        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                            # Región de influencia
                            radius = 5
                            y_min, y_max = max(0, y-radius), min(gray.shape[0], y+radius)
                            x_min, x_max = max(0, x-radius), min(gray.shape[1], x+radius)
                            flow[y_min:y_max, x_min:x_max] = [dx, dy]

                    return flow
                else:
                    # Re-inicializar puntos
                    self.prev_points = cv2.goodFeaturesToTrack(
                        gray,
                        maxCorners=100,
                        qualityLevel=0.3,
                        minDistance=7,
                        blockSize=7
                    )
                    return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)

            elif self.flow_method == OpticalFlowMethod.FARNEBACK:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=0
                )
                return flow

            elif self.flow_method == OpticalFlowMethod.TVL1:
                flow = self.flow_detector.calc(self.prev_gray, gray, None)
                return flow

        except Exception as e:
            print(f"⚠ Error calculando optical flow: {e}")
            return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)

    def _analyze_motion(self, flow: np.ndarray, fg_mask: np.ndarray) -> GestureMotion:
        """
        Analiza el movimiento en el flow y máscara de foreground.

        Args:
            flow: Array de optical flow
            fg_mask: Máscara de foreground

        Returns:
            Análisis de movimiento del gesto
        """
        try:
            # Calcular magnitud y ángulo del flow
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Aplicar máscara de foreground
            if fg_mask is not None:
                magnitude = cv2.bitwise_and(magnitude.astype(np.uint8), fg_mask)
                angle = cv2.bitwise_and(angle.astype(np.uint8), fg_mask)

            # Filtrar movimientos pequeños
            motion_mask = magnitude > self.min_motion_threshold
            motion_mask &= magnitude < self.max_motion_threshold

            # Extraer vectores de movimiento significativos
            motion_vectors = []
            y_coords, x_coords = np.where(motion_mask)

            for y, x in zip(y_coords[::10], x_coords[::10]):  # Sample every 10th point
                mag = magnitude[y, x]
                ang = angle[y, x]
                motion_vectors.append(MotionVector(
                    x=float(x), y=float(y),
                    magnitude=float(mag),
                    angle=float(np.degrees(ang))
                ))

            if not motion_vectors:
                return GestureMotion(
                    dominant_direction=0.0,
                    average_speed=0.0,
                    motion_vectors=[],
                    stability_score=0.0,
                    gesture_type="static"
                )

            # Calcular dirección dominante
            angles = [v.angle for v in motion_vectors]
            magnitudes = [v.magnitude for v in motion_vectors]

            # Usar promedio ponderado por magnitud
            dominant_direction = np.average(angles, weights=magnitudes)

            # Calcular velocidad promedio
            average_speed = np.mean(magnitudes)

            # Calcular estabilidad (consistencia en dirección)
            angle_std = np.std(angles)
            stability_score = max(0.0, 1.0 - angle_std / 90.0)  # Normalizar a 0-1

            # Clasificar tipo de gesto
            gesture_type = self._classify_gesture_type(
                average_speed, stability_score, len(motion_vectors)
            )

            return GestureMotion(
                dominant_direction=dominant_direction,
                average_speed=average_speed,
                motion_vectors=motion_vectors,
                stability_score=stability_score,
                gesture_type=gesture_type
            )

        except Exception as e:
            print(f"⚠ Error analizando movimiento: {e}")
            return GestureMotion(
                dominant_direction=0.0,
                average_speed=0.0,
                motion_vectors=[],
                stability_score=0.0,
                gesture_type="error"
            )

    def _classify_gesture_type(self, speed: float, stability: float,
                              num_vectors: int) -> str:
        """
        Clasifica el tipo de gesto basado en características de movimiento.

        Args:
            speed: Velocidad promedio
            stability: Puntuación de estabilidad (0-1)
            num_vectors: Número de vectores de movimiento

        Returns:
            Tipo de gesto clasificado
        """
        # Umbrales para clasificación
        min_drawing_speed = 2.0
        min_stability_drawing = 0.6
        min_vectors_drawing = 20

        if (speed > min_drawing_speed and
            stability > min_stability_drawing and
            num_vectors > min_vectors_drawing):
            return "drawing"

        elif speed > 1.0 and num_vectors > 10:
            return "pointing"

        elif speed < 0.5 and num_vectors < 5:
            return "static"

        else:
            return "noise"

    def _update_history(self, gray: np.ndarray, flow: np.ndarray):
        """Actualiza el historial de frames y flow."""
        self.frame_history.append(gray.copy())
        self.flow_history.append(flow.copy() if flow is not None else None)

        # Mantener tamaño máximo del historial
        if len(self.frame_history) > self.history_length:
            self.frame_history.pop(0)
            self.flow_history.pop(0)

        # Actualizar frame previo
        self.prev_gray = gray.copy()

    def _get_empty_result(self) -> Dict[str, Any]:
        """Retorna un resultado vacío para casos de error."""
        return {
            'foreground_mask': None,
            'optical_flow': None,
            'motion_analysis': GestureMotion(
                dominant_direction=0.0,
                average_speed=0.0,
                motion_vectors=[],
                stability_score=0.0,
                gesture_type="uninitialized"
            ),
            'frame_gray': None,
            'processing_success': False
        }

    def get_motion_visualization(self, frame: np.ndarray,
                               motion_analysis: GestureMotion) -> np.ndarray:
        """
        Crea una visualización del movimiento detectado.

        Args:
            frame: Frame original
            motion_analysis: Análisis de movimiento

        Returns:
            Frame con visualización de movimiento
        """
        vis_frame = frame.copy()

        try:
            # Dibujar vectores de movimiento
            for vector in motion_analysis.motion_vectors[:50]:  # Limitar para performance
                x, y = int(vector.x), int(vector.y)
                dx = int(vector.magnitude * np.cos(np.radians(vector.angle)) * 0.1)
                dy = int(vector.magnitude * np.sin(np.radians(vector.angle)) * 0.1)

                # Color basado en magnitud
                color_intensity = min(255, int(vector.magnitude * 10))
                color = (0, color_intensity, 255 - color_intensity)

                cv2.arrowedLine(vis_frame, (x, y), (x + dx, y + dy), color, 1, tipLength=0.3)

            # Dibujar información de gesto
            info_text = f"Gesto: {motion_analysis.gesture_type}"
            stability_text = f"Estabilidad: {motion_analysis.stability_score:.2f}"
            speed_text = f"Velocidad: {motion_analysis.average_speed:.1f}"

            cv2.putText(vis_frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 0), 2)
            cv2.putText(vis_frame, stability_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 0), 2)
            cv2.putText(vis_frame, speed_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 0), 2)

        except Exception as e:
            print(f"⚠ Error en visualización de movimiento: {e}")

        return vis_frame

    def reset(self):
        """Reinicia el estado del procesador."""
        self.frame_history.clear()
        self.flow_history.clear()
        self.is_initialized = False
        self.prev_frame = None
        self.prev_gray = None

        # Re-crear background subtractor
        self.bg_subtractor = self._create_bg_subtractor()

        print("✓ AdvancedVisionProcessor reiniciado")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del procesamiento."""
        return {
            'bg_method': self.bg_method.value,
            'flow_method': self.flow_method.value,
            'history_length': len(self.frame_history),
            'is_initialized': self.is_initialized,
            'learning_rate': self.learning_rate
        }


# Funciones de compatibilidad y utilidades
def create_advanced_vision_processor(bg_method: str = "mog2",
                                   flow_method: str = "lucas_kanade") -> AdvancedVisionProcessor:
    """
    Crea un procesador avanzado de visión con configuración por defecto.

    Args:
        bg_method: Método de background subtraction
        flow_method: Método de optical flow

    Returns:
        Instancia configurada de AdvancedVisionProcessor
    """
    bg_enum = BackgroundSubtractionMethod(bg_method.lower())
    flow_enum = OpticalFlowMethod(flow_method.lower())

    return AdvancedVisionProcessor(bg_method=bg_enum, flow_method=flow_enum)