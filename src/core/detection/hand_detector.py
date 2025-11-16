"""
Detector de manos usando OpenCV con técnicas avanzadas de visión.

Esta versión combina técnicas básicas de visión por computadora con
algoritmos avanzados como background subtraction y optical flow.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque
from .advanced_vision import AdvancedVisionProcessor, BackgroundSubtractionMethod, OpticalFlowMethod
from ..constants import DETECTION_CONFIG


class KalmanFilter1D:
    """Filtro de Kalman simplificado para 1D."""
    
    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = None
        self.estimate_error = 1.0
    
    def update(self, measurement: float) -> float:
        if self.estimate is None:
            self.estimate = measurement
            return measurement
        
        # Predicción
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # Actualización
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate


class ContourBuffer:
    """Buffer circular de contornos para votación mayoritaria."""
    
    def __init__(self, max_size: int = None):
        self.buffer = deque(maxlen=max_size or DETECTION_CONFIG['CONTOUR_BUFFER_SIZE'])
        self.confidence_threshold = DETECTION_CONFIG['CONTOUR_CONFIDENCE_THRESHOLD']
    
    def add(self, contours: List) -> List:
        """Agrega contornos al buffer y devuelve los más estables."""
        self.buffer.append(contours)
        
        if len(self.buffer) < 3:
            return contours
        
        # Encontrar contornos que aparecen consistentemente
        stable_contours = []
        
        for ref_contour in contours:
            matches = 0
            for historical_contours in self.buffer:
                for hist_contour in historical_contours:
                    if self._contours_match(ref_contour, hist_contour):
                        matches += 1
                        break
            
            if matches / len(self.buffer) >= self.confidence_threshold:
                stable_contours.append(ref_contour)
        
        return stable_contours if stable_contours else list(self.buffer[-1])
    
    def _contours_match(self, cnt1, cnt2) -> bool:
        """Verifica si dos contornos son similares."""
        area1 = cv2.contourArea(cnt1)
        area2 = cv2.contourArea(cnt2)
        
        if area1 == 0 or area2 == 0:
            return False
        
        area_ratio = min(area1, area2) / max(area1, area2)
        return area_ratio > 0.7


class HistogramAnalyzer:
    """Analizador de histograma para compensación de iluminación."""
    
    def __init__(self, region_grid: int = None):
        self.region_grid = region_grid or DETECTION_CONFIG['HISTOGRAM_GRID_SIZE']
        self.histogram_cache = None
    
    def analyze_regions(self, frame: np.ndarray) -> Dict[str, float]:
        """Analiza histograma por regiones del frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        region_height = height // self.region_grid
        region_width = width // self.region_grid
        
        region_stats = {
            'global_mean': np.mean(gray),
            'global_std': np.std(gray),
            'regions': {}
        }
        
        for i in range(self.region_grid):
            for j in range(self.region_grid):
                y1 = i * region_height
                y2 = (i + 1) * region_height if i < self.region_grid - 1 else height
                x1 = j * region_width
                x2 = (j + 1) * region_width if j < self.region_grid - 1 else width
                
                region = gray[y1:y2, x1:x2]
                key = f'region_{i}_{j}'
                region_stats['regions'][key] = np.mean(region)
        
        self.histogram_cache = region_stats
        return region_stats


class ShadowDetector:
    """Detector de sombras usando análisis de color."""
    
    def detect_shadows(self, frame: np.ndarray) -> np.ndarray:
        """Detecta sombras y crea máscara."""
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Sombras típicamente tienen baja saturación y bajo valor
        lower_shadow = np.array(DETECTION_CONFIG['SHADOW_HSV_LOWER'])
        upper_shadow = np.array(DETECTION_CONFIG['SHADOW_HSV_UPPER'])
        
        shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
        
        # Aplicar morphology para limpiar
        kernel = np.ones(DETECTION_CONFIG['SHADOW_MORPHOLOGY_KERNEL'], np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=DETECTION_CONFIG['SHADOW_MORPHOLOGY_ITERATIONS'])
        
        return shadow_mask


class AdaptiveRangeManager:
    """Gestiona rangos HSV adaptativos basado en condiciones."""
    
    def __init__(self):
        self.base_lower = None
        self.base_upper = None
        self.adjustment_history = deque(maxlen=10)
    
    def set_base_ranges(self, lower: np.ndarray, upper: np.ndarray):
        """Establece rangos base."""
        self.base_lower = lower.copy()
        self.base_upper = upper.copy()
    
    def adjust(self, current_ranges: Tuple, histogram_stats: Dict, shadow_mask: np.ndarray) -> Tuple:
        """Ajusta rangos basado en condiciones."""
        current_lower, current_upper = current_ranges
        
        if self.base_lower is None or self.base_upper is None:
            self.base_lower = current_lower.copy()
            self.base_upper = current_upper.copy()
        
        adjusted_lower = current_lower.copy().astype(np.float32)
        adjusted_upper = current_upper.copy().astype(np.float32)
        
        # Ajuste basado en brillo global
        global_mean = histogram_stats['global_mean']
        brightness_factor = global_mean / 128.0
        
        if brightness_factor < 0.7:  # Muy oscuro
            adjusted_lower[2] = max(0, adjusted_lower[2] - 20)
            adjusted_upper[2] = min(255, adjusted_upper[2] + 10)
        elif brightness_factor > 1.3:  # Muy brillante
            adjusted_lower[2] = max(0, adjusted_lower[2] + 10)
            adjusted_upper[2] = min(255, adjusted_upper[2] - 20)
        
        # Ajuste basado en sombras
        shadow_ratio = np.count_nonzero(shadow_mask) / shadow_mask.size
        if shadow_ratio > 0.3:  # Muchas sombras
            adjusted_lower[1] = max(0, adjusted_lower[1] - 15)  # Reducir requisito de saturación
        
        return (adjusted_lower.astype(np.uint8), adjusted_upper.astype(np.uint8))


class HandDetector:
    """
    Detecta manos usando técnicas básicas de OpenCV.
    
    Atributos:
        min_area: Área mínima para considerar una detección válida
        max_area: Área máxima para detección
        skin_lower: Límite inferior del rango de color de piel (HSV)
        skin_upper: Límite superior del rango de color de piel (HSV)
    """
    
    def __init__(self, min_area: int = None, max_area: int = None,
                 skin_lower: Optional[Tuple[int, int, int]] = None,
                 skin_upper: Optional[Tuple[int, int, int]] = None,
                 enable_advanced_vision: bool = True,
                 use_mediapipe: bool = False):
        """
        Inicializa el detector de manos con técnicas avanzadas.

        Args:
            min_area: Área mínima en píxeles para detección (opcional, usa config por defecto)
            max_area: Área máxima en píxeles para detección (opcional, usa config por defecto)
            skin_lower: Límite inferior del rango de color de piel (HSV) - opcional
            skin_upper: Límite superior del rango de color de piel (HSV) - opcional
            enable_advanced_vision: Si True, usa algoritmos avanzados de visión
            use_mediapipe: Si True, usa MediaPipe en lugar de OpenCV

        Raises:
            ValueError: Si los parámetros de área son inválidos
        """
        # Validar parámetros de área
        self.min_area = min_area or DETECTION_CONFIG['DEFAULT_MIN_AREA']
        self.max_area = max_area or DETECTION_CONFIG['DEFAULT_MAX_AREA']

        if self.min_area <= 0 or self.max_area <= 0:
            raise ValueError("Min and max area must be positive")
        if self.min_area >= self.max_area:
            raise ValueError("Min area must be less than max area")

        self.use_mediapipe = use_mediapipe

        # Rangos HSV para detección de piel (usar valores por defecto o personalizados)
        self.skin_lower = np.array(skin_lower if skin_lower else DETECTION_CONFIG['DEFAULT_SKIN_LOWER'], dtype=np.uint8)
        self.skin_upper = np.array(skin_upper if skin_upper else DETECTION_CONFIG['DEFAULT_SKIN_UPPER'], dtype=np.uint8)

        # Para tracking temporal
        self.prev_contours = []
        self.contour_history = []  # Historial de contornos para estabilidad
        self.tracking_point = None
        self.stability_threshold = DETECTION_CONFIG['STABILITY_THRESHOLD']
        self.max_history_size = DETECTION_CONFIG['MAX_HISTORY_SIZE']

        # Estabilidad multi-frame
        self.contour_buffer = ContourBuffer(max_size=DETECTION_CONFIG['CONTOUR_BUFFER_SIZE'])
        self.kalman_filters = {}  # Filtros Kalman por contorno
        self.hysteresis_state = {}  # Estado de hysteresis

        # Compensación de iluminación avanzada
        self.histogram_analyzer = HistogramAnalyzer(region_grid=DETECTION_CONFIG['HISTOGRAM_GRID_SIZE'])
        self.shadow_detector = ShadowDetector()
        self.adaptive_ranges = AdaptiveRangeManager()

        # Inicializar detector apropiado
        if use_mediapipe:
            try:
                from .mediapipe_hand_detector import MediaPipeHandDetector
                self.detector = MediaPipeHandDetector(
                    min_area=self.min_area,
                    max_area=self.max_area
                )
                self.enable_advanced_vision = False  # MediaPipe no necesita visión avanzada
                print("✓ HandDetector usando MediaPipe Hands")
            except ImportError as e:
                print(f"⚠ MediaPipe no disponible: {e}")
                print("  Cambiando a detector OpenCV...")
                self.use_mediapipe = False
                self._init_opencv_detector(enable_advanced_vision)
        else:
            self._init_opencv_detector(enable_advanced_vision)
    
    def _init_opencv_detector(self, enable_advanced_vision: bool):
        """
        Inicializa el detector OpenCV tradicional.
        
        Args:
            enable_advanced_vision: Si usar algoritmos avanzados
        """
        # Procesador avanzado de visión
        self.enable_advanced_vision = enable_advanced_vision
        self.vision_processor = None
        
        if enable_advanced_vision:
            try:
                self.vision_processor = AdvancedVisionProcessor(
                    bg_method=BackgroundSubtractionMethod.MOG2,
                    flow_method=OpticalFlowMethod.LUCAS_KANADE
                )
                print("✓ Procesador avanzado de visión habilitado")
            except Exception as e:
                print(f"⚠ Error inicializando procesador avanzado: {e}")
                print("  Continuando con detección básica")
                self.enable_advanced_vision = False
        
        print(f"✓ HandDetector (OpenCV) inicializado - Rangos: {tuple(self.skin_lower)} a {tuple(self.skin_upper)}")
        if self.enable_advanced_vision:
            print("  Modo: Avanzado (Background Subtraction + Optical Flow)")
        else:
            print("  Modo: Básico (Segmentación por color)")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List], bool]:
        """
        Detecta manos en un frame usando el detector configurado.

        Args:
            frame: Frame de OpenCV (BGR). Debe ser un array válido no vacío.

        Returns:
            Tupla con (frame_rgb, contours, manos_detectadas)

        Raises:
            ValueError: Si el frame es None o inválido
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Frame must be a valid non-empty numpy array")

        if self.use_mediapipe and hasattr(self, 'detector'):
            # Usar MediaPipe detector
            return self.detector.detect(frame)
        else:
            # Usar detector OpenCV tradicional
            return self._detect_opencv(frame)
    
    def _detect_opencv(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List], bool]:
        """
        Detección de manos usando el detector OpenCV tradicional.

        Args:
            frame: Frame de OpenCV (BGR)

        Returns:
            Tupla con (frame_rgb, contours, manos_detectadas)
        """
        try:
            # Importar compensador de iluminación
            from ..utils.lighting_analysis import lighting_compensator

            # Aplicar compensación de iluminación automática
            frame_compensated, lighting_analysis = lighting_compensator.compensate_frame(frame)

            # Procesamiento avanzado si está habilitado
            vision_result = None
            if self.enable_advanced_vision and self.vision_processor:
                try:
                    vision_result = self.vision_processor.process_frame(frame_compensated)
                except Exception as e:
                    print(f"⚠ Error en procesamiento avanzado: {e}")
                    vision_result = None

            # Análisis de histograma y detección de sombras
            histogram_stats = self.histogram_analyzer.analyze_regions(frame_compensated)
            shadow_mask = self.shadow_detector.detect_shadows(frame_compensated)

            # Calcular brillo promedio del frame para compensación
            gray = cv2.cvtColor(frame_compensated, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)

            # Obtener rangos adaptados
            current_ranges = (self.skin_lower, self.skin_upper)
            adjusted_ranges = self.adaptive_ranges.adjust(current_ranges, histogram_stats, shadow_mask)
            adjusted_lower, adjusted_upper = adjusted_ranges

            # Convertir a HSV
            hsv = cv2.cvtColor(frame_compensated, cv2.COLOR_BGR2HSV)

            # Crear máscara con rangos ajustados
            mask = cv2.inRange(hsv, adjusted_lower, adjusted_upper)

            # Limpiar máscara
            mask = cv2.medianBlur(mask, DETECTION_CONFIG['MEDIAN_BLUR_SIZE'])
            mask = cv2.GaussianBlur(mask, DETECTION_CONFIG['GAUSSIAN_BLUR_SIZE'], 0)

            # Operaciones morfológicas para limpiar
            kernel = np.ones(DETECTION_CONFIG['MORPHOLOGY_KERNEL_SIZE'], np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=DETECTION_CONFIG['MORPH_OPEN_ITERATIONS'])
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=DETECTION_CONFIG['MORPH_CLOSE_ITERATIONS'])

            # Integrar resultados del procesamiento avanzado
            if vision_result and vision_result['processing_success']:
                fg_mask = vision_result['foreground_mask']
                motion_analysis = vision_result['motion_analysis']

                if fg_mask is not None:
                    # Combinar máscaras: usar AND entre segmentación por color y background subtraction
                    combined_mask = cv2.bitwise_and(mask, fg_mask)

                    # Si hay movimiento significativo, dar más peso al background subtraction
                    if motion_analysis.gesture_type in ["drawing", "pointing"]:
                        # Usar principalmente background subtraction con algo de color
                        combined_mask = cv2.bitwise_or(
                            cv2.bitwise_and(mask, fg_mask),
                            cv2.bitwise_and(mask, fg_mask, mask=fg_mask.astype(np.uint8) // DETECTION_CONFIG['MASK_COMBINE_RATIO'])
                        )
                    else:
                        # Usar principalmente segmentación por color
                        combined_mask = cv2.bitwise_or(mask, fg_mask.astype(np.uint8) // DETECTION_CONFIG['MASK_COMBINE_RATIO'])

                    mask = combined_mask

            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filtrar contornos por área
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.min_area < area < self.max_area:
                    valid_contours.append(cnt)

            # Aplicar buffer de estabilidad multi-frame
            buffered_contours = self.contour_buffer.add(valid_contours)

            # Aplicar filtros de estabilidad mejorados
            stable_contours = self._filter_stable_contours_advanced(buffered_contours, vision_result)

            # Detectar si hay manos (usando contornos estables)
            has_hands = len(stable_contours) > 0

            return frame, stable_contours, has_hands

        except cv2.error as e:
            print(f"Error in OpenCV detection: {e}")
            return frame, [], False
        except Exception as e:
            print(f"Unexpected error in hand detection: {e}")
            return frame, [], False

    def _calculate_contour_similarity(self, contour1, contour2) -> float:
        """
        Calcula similitud entre dos contornos basada en área y posición.
        
        Args:
            contour1, contour2: Contornos a comparar
            
        Returns:
            Puntuación de similitud (0-1, donde 1 es idéntico)
        """
        try:
            # Calcular áreas
            area1 = cv2.contourArea(contour1)
            area2 = cv2.contourArea(contour2)
            
            if area1 == 0 or area2 == 0:
                return 0.0
            
            # Área similarity (0-1)
            area_ratio = min(area1, area2) / max(area1, area2)
            
            # Centro de masa similarity
            M1 = cv2.moments(contour1)
            M2 = cv2.moments(contour2)
            
            if M1['m00'] == 0 or M2['m00'] == 0:
                return area_ratio
            
            cx1 = M1['m10'] / M1['m00']
            cy1 = M1['m01'] / M1['m00']
            cx2 = M2['m10'] / M2['m00']
            cy2 = M2['m01'] / M2['m00']
            
            # Distancia euclidiana entre centros
            center_distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # Normalizar distancia (usando constantes de frame)
            max_distance = np.sqrt(DETECTION_CONFIG['FRAME_WIDTH']**2 + DETECTION_CONFIG['FRAME_HEIGHT']**2)
            position_similarity = 1.0 - min(center_distance / max_distance, 1.0)
            
            # Puntuación combinada
            return DETECTION_CONFIG['AREA_SIMILARITY_WEIGHT'] * area_ratio + DETECTION_CONFIG['POSITION_SIMILARITY_WEIGHT'] * position_similarity
            
        except Exception:
            return 0.0
    
    def _filter_stable_contours(self, current_contours: List) -> List:
        """
        Filtra contornos que han sido estables en frames consecutivos.
        
        Args:
            current_contours: Contornos del frame actual
            
        Returns:
            Lista de contornos estables
        """
        if len(self.contour_history) < self.stability_threshold:
            # Agregar al historial y devolver todos (primera vez)
            self.contour_history.append(current_contours)
            if len(self.contour_history) > self.max_history_size:
                self.contour_history.pop(0)
            return current_contours
        
        stable_contours = []
        
        for current_contour in current_contours:
            # Contar cuántos frames consecutivos tiene un contorno similar
            consecutive_matches = 0
            
            for historical_contours in reversed(self.contour_history[-self.stability_threshold:]):
                best_match = max(historical_contours, 
                               key=lambda h: self._calculate_contour_similarity(current_contour, h),
                               default=None)
                
                if best_match is not None:
                    similarity = self._calculate_contour_similarity(current_contour, best_match)
                    if similarity > DETECTION_CONFIG['CONTOUR_SIMILARITY_THRESHOLD']:  # Umbral de similitud
                        consecutive_matches += 1
                    else:
                        break
                else:
                    break
            
            # Si tiene suficientes matches consecutivos, es estable
            if consecutive_matches >= self.stability_threshold - 1:
                stable_contours.append(current_contour)
        
        # Actualizar historial
        self.contour_history.append(current_contours)
        if len(self.contour_history) > self.max_history_size:
            self.contour_history.pop(0)
        
        return stable_contours
    
    def _filter_stable_contours_advanced(self, current_contours: List, vision_result=None) -> List:
        """
        Filtra contornos usando estabilidad temporal mejorada con información de movimiento.
        
        Args:
            current_contours: Contornos del frame actual
            vision_result: Resultados del procesamiento avanzado de visión
            
        Returns:
            Lista de contornos estables
        """
        # Usar filtrado básico si no hay resultados avanzados
        if not vision_result or not vision_result['processing_success']:
            return self._filter_stable_contours(current_contours)
        
        motion_analysis = vision_result['motion_analysis']
        
        # Si hay movimiento significativo, ser más permisivo con estabilidad
        stability_threshold = self.stability_threshold
        if motion_analysis.gesture_type in ["drawing", "pointing"]:
            stability_threshold = max(1, stability_threshold - 1)  # Reducir requerimiento
        
        if len(self.contour_history) < stability_threshold:
            # Agregar al historial y devolver todos (primera vez)
            self.contour_history.append(current_contours)
            if len(self.contour_history) > self.max_history_size:
                self.contour_history.pop(0)
            return current_contours
        
        stable_contours = []
        
        for current_contour in current_contours:
            # Contar cuántos frames consecutivos tiene un contorno similar
            consecutive_matches = 0
            
            for historical_contours in reversed(self.contour_history[-stability_threshold:]):
                best_match = max(historical_contours, 
                               key=lambda h: self._calculate_contour_similarity(current_contour, h),
                               default=None)
                
                if best_match is not None:
                    similarity = self._calculate_contour_similarity(current_contour, best_match)
                    if similarity > DETECTION_CONFIG['CONTOUR_SIMILARITY_THRESHOLD']:  # Umbral de similitud
                        consecutive_matches += 1
                    else:
                        break
                else:
                    break
            
            # Si tiene suficientes matches consecutivos, es estable
            if consecutive_matches >= stability_threshold - 1:
                stable_contours.append(current_contour)
        
        # Actualizar historial
        self.contour_history.append(current_contours)
        if len(self.contour_history) > self.max_history_size:
            self.contour_history.pop(0)
        
        return stable_contours
    
    def _determine_hand_state(self, contour) -> str:
        """
        Determina el estado de la mano basado en análisis del contorno.
        
        Args:
            contour: Contorno de la mano
            
        Returns:
            "open", "closed", o "unknown"
        """
        try:
            # Calcular propiedades del contorno
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                return "unknown"
            
            # Circularidad: medida de qué tan "redonda" es la forma
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Convex hull para detectar concavidades
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area == 0:
                return "unknown"
            
            # Solidity: área del contorno / área del convex hull
            solidity = area / hull_area
            
            # Análisis de forma
            # Mano cerrada: alta circularidad, alta solidity
            # Mano abierta: baja circularidad, baja solidity (más "extendida")
            
            if circularity > DETECTION_CONFIG['CLOSED_HAND_CIRCULARITY_THRESHOLD'] and solidity > DETECTION_CONFIG['CLOSED_HAND_SOLIDITY_THRESHOLD']:
                return "closed"
            elif circularity < DETECTION_CONFIG['OPEN_HAND_CIRCULARITY_THRESHOLD'] and solidity < DETECTION_CONFIG['OPEN_HAND_SOLIDITY_THRESHOLD']:
                return "open"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    def draw_landmarks(self, frame: np.ndarray, contours: List, vision_result=None) -> np.ndarray:
        """
        Dibuja los contornos detectados con información adicional y visualización de movimiento.
        
        Args:
            frame: Frame original
            contours: Lista de contornos
            vision_result: Resultados del procesamiento avanzado de visión
            
        Returns:
            Frame con contornos e información dibujados
        """
        if self.use_mediapipe and hasattr(self, 'detector'):
            # Usar MediaPipe detector
            return self.detector.draw_landmarks(frame, contours, vision_result)
        else:
            # Usar implementación OpenCV tradicional
            return self._draw_landmarks_opencv(frame, contours, vision_result)
    
    def _draw_landmarks_opencv(self, frame: np.ndarray, contours: List, vision_result=None) -> np.ndarray:
        """
        Dibuja los contornos detectados usando OpenCV tradicional.
        
        Args:
            frame: Frame original
            contours: Lista de contornos
            vision_result: Resultados del procesamiento avanzado de visión
            
        Returns:
            Frame con contornos e información dibujados
        """
        frame_copy = frame.copy()
        
        if contours:
            # Dibujar todos los contornos
            cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 2)
            
            # Dibujar bounding boxes e información adicional
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Estado de la mano
                hand_state = self._determine_hand_state(cnt)
                state_color = {
                    "open": (0, 255, 0),
                    "closed": (0, 0, 255),
                    "unknown": (128, 128, 128)
                }.get(hand_state, (128, 128, 128))
                
                # Etiqueta con estado
                cv2.putText(frame_copy, f"Mano {i+1}: {hand_state.upper()}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
                
                # Indicador de gesto de dibujo
                if self.is_drawing_gesture([cnt]):
                    cv2.putText(frame_copy, "GESTO DIBUJO", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Dibujar punto de dedo índice
                    finger_tip = self.get_index_finger_tip([cnt])
                    if finger_tip:
                        cv2.circle(frame_copy, (int(finger_tip[0]), int(finger_tip[1])), 
                                 8, (0, 255, 255), -1)
                        cv2.circle(frame_copy, (int(finger_tip[0]), int(finger_tip[1])), 
                                 12, (0, 255, 255), 2)
        
        # Agregar visualización de movimiento si está disponible
        if vision_result and vision_result['processing_success']:
            motion_analysis = vision_result['motion_analysis']
            
            # Información de movimiento
            motion_info = f"Movimiento: {motion_analysis.gesture_type.upper()}"
            cv2.putText(frame_copy, motion_info, (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            stability_info = f"Estabilidad: {motion_analysis.stability_score:.2f}"
            cv2.putText(frame_copy, stability_info, (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            speed_info = f"Velocidad: {motion_analysis.average_speed:.1f}"
            cv2.putText(frame_copy, speed_info, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Indicador de modo avanzado
            cv2.putText(frame_copy, "MODO AVANZADO", (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame_copy
    
    def get_index_finger_tip(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición aproximada del "dedo índice" (punto más alto del contorno).
        Mejorado para contornos estables.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            Tupla (x, y) en píxeles o None
        """
        if self.use_mediapipe and hasattr(self, 'detector'):
            # Usar MediaPipe detector
            return self.detector.get_index_finger_tip(contours)
        else:
            # Usar implementación OpenCV tradicional
            return self._get_index_finger_tip_opencv(contours)
    
    def _get_index_finger_tip_opencv(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición aproximada del "dedo índice" usando OpenCV tradicional.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            Tupla (x, y) en píxeles o None
        """
        if not contours:
            return None
        
        # Tomar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Para contornos estables, usar el punto más alto del convex hull
        hull = cv2.convexHull(largest_contour)
        
        if len(hull) > 0:
            # El punto más alto del convex hull (aproximación de dedo)
            top_point = min(hull, key=lambda p: p[0][1])[0]
            return (float(top_point[0]), float(top_point[1]))
        
        return None
    
    def is_drawing_gesture(self, contours: List) -> bool:
        """
        Determina si la mano está en gesto de dibujo usando análisis avanzado.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            True si está en gesto de dibujo
        """
        if self.use_mediapipe and hasattr(self, 'detector'):
            # Usar MediaPipe detector
            return self.detector.is_drawing_gesture(contours)
        else:
            # Usar implementación OpenCV tradicional
            return self._is_drawing_gesture_opencv(contours)
    
    def _is_drawing_gesture_opencv(self, contours: List) -> bool:
        """
        Determina si la mano está en gesto de dibujo usando análisis OpenCV tradicional.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            True si está en gesto de dibujo
        """
        if not contours:
            return False
        
        largest_contour = max(contours, key=cv2.contourArea)
        hand_state = self._determine_hand_state(largest_contour)
        
        # Gesto de dibujo: mano abierta con buena extensión
        if hand_state == "open":
            # Verificar que no sea demasiado circular (sería puño) ni demasiado irregular
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Debe ser lo suficientemente "extendida" pero no un círculo perfecto
                return DETECTION_CONFIG['DRAWING_GESTURE_MIN_CIRCULARITY'] < circularity < DETECTION_CONFIG['DRAWING_GESTURE_MAX_CIRCULARITY']
        
        return False
    
    def get_thumb_tip(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición aproximada del "pulgar" (punto más a la izquierda).
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            Tupla (x, y) en píxeles o None
        """
        if self.use_mediapipe and hasattr(self, 'detector'):
            # Usar MediaPipe detector
            return self.detector.get_thumb_tip(contours)
        else:
            # Usar implementación OpenCV tradicional
            return self._get_thumb_tip_opencv(contours)
    
    def _get_thumb_tip_opencv(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición aproximada del "pulgar" usando OpenCV tradicional.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            Tupla (x, y) en píxeles o None
        """
        if not contours:
            return None
        
        # Tomar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Encontrar el punto más a la izquierda
        hull = cv2.convexHull(largest_contour)
        
        if len(hull) > 0:
            left_point = min(hull, key=lambda p: p[0][0])[0]
            return (float(left_point[0]), float(left_point[1]))
        
        return None
    
    def is_fist(self, contours: List) -> bool:
        """
        Detecta si la mano está cerrada (puño) usando análisis de estado.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            True si detecta puño, False en caso contrario
        """
        if self.use_mediapipe and hasattr(self, 'detector'):
            # Usar MediaPipe detector
            return self.detector.is_fist(contours)
        else:
            # Usar implementación OpenCV tradicional
            return self._is_fist_opencv(contours)
    
    def _is_fist_opencv(self, contours: List) -> bool:
        """
        Detecta si la mano está cerrada (puño) usando análisis OpenCV tradicional.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            True si detecta puño, False en caso contrario
        """
        if not contours:
            return False
        
        # Tomar el contorno más grande (mano principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Usar nueva lógica de estado de mano
        hand_state = self._determine_hand_state(largest_contour)
        
        return hand_state == "closed"
    
    def close(self):
        """Cierra el detector y libera recursos."""
        # Cerrar detector MediaPipe si existe
        if self.use_mediapipe and hasattr(self, 'detector'):
            try:
                self.detector.close()
                print("✓ MediaPipe detector cerrado")
            except Exception as e:
                print(f"⚠ Error cerrando MediaPipe detector: {e}")
        
        # Cerrar procesador de visión avanzada
        if hasattr(self, 'vision_processor') and self.vision_processor:
            try:
                self.vision_processor.reset()
                print("✓ Procesador de visión avanzada cerrado")
            except Exception as e:
                print(f"⚠ Error cerrando procesador de visión: {e}")
        
        # Limpiar historiales
        self.prev_contours = []
        self.contour_history = []
        self.tracking_point = None
