"""
Detector de manos usando OpenCV con técnicas avanzadas de visión.

Esta versión combina técnicas básicas de visión por computadora con
algoritmos avanzados como background subtraction y optical flow.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from .advanced_vision import AdvancedVisionProcessor, BackgroundSubtractionMethod, OpticalFlowMethod


class HandDetector:
    """
    Detecta manos usando técnicas básicas de OpenCV.
    
    Atributos:
        min_area: Área mínima para considerar una detección válida
        max_area: Área máxima para detección
        skin_lower: Límite inferior del rango de color de piel (HSV)
        skin_upper: Límite superior del rango de color de piel (HSV)
    """
    
    def __init__(self, min_area: int = 3000, max_area: int = 30000,
                 skin_lower: Optional[Tuple[int, int, int]] = None,
                 skin_upper: Optional[Tuple[int, int, int]] = None,
                 enable_advanced_vision: bool = True):
        """
        Inicializa el detector de manos con técnicas avanzadas.
        
        Args:
            min_area: Área mínima en píxeles para detección
            max_area: Área máxima en píxeles para detección
            skin_lower: Límite inferior del rango de color de piel (HSV) - opcional
            skin_upper: Límite superior del rango de color de piel (HSV) - opcional
            enable_advanced_vision: Si True, usa algoritmos avanzados de visión
        """
        self.min_area = min_area
        self.max_area = max_area
        
        # Rangos HSV para detección de piel (usar valores por defecto o personalizados)
        self.skin_lower = np.array(skin_lower if skin_lower else [0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array(skin_upper if skin_upper else [20, 255, 255], dtype=np.uint8)
        
        # Para tracking temporal
        self.prev_contours = []
        self.contour_history = []  # Historial de contornos para estabilidad
        self.tracking_point = None
        self.stability_threshold = 3  # Frames consecutivos para confirmar detección
        self.max_history_size = 10  # Máximo frames en historial
        
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
        Detecta manos en un frame usando técnicas avanzadas de visión.

        Combina segmentación por color con background subtraction y optical flow
        para una detección más robusta y precisa.

        Args:
            frame: Frame de OpenCV (BGR)

        Returns:
            Tupla con (frame_rgb, contours, manos_detectadas)
        """
        # Procesamiento avanzado si está habilitado
        vision_result = None
        if self.enable_advanced_vision and self.vision_processor:
            try:
                vision_result = self.vision_processor.process_frame(frame)
            except Exception as e:
                print(f"⚠ Error en procesamiento avanzado: {e}")
                vision_result = None

        # Calcular brillo promedio del frame para compensación
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        # Aplicar compensación de iluminación si tenemos rangos calibrados
        current_lower = self.skin_lower
        current_upper = self.skin_upper

        # Compensación simple: ajustar rango V basado en brillo
        brightness_factor = avg_brightness / 128.0  # 128 es brillo medio

        if brightness_factor < 0.8:  # Frame oscuro
            # Expandir rango V hacia abajo
            adjusted_lower = current_lower.copy()
            adjusted_upper = current_upper.copy()
            v_lower = max(0, current_lower[2] - int(15 * (1 - brightness_factor)))
            adjusted_lower[2] = v_lower
        elif brightness_factor > 1.2:  # Frame brillante
            # Expandir rango V hacia arriba
            adjusted_lower = current_lower.copy()
            adjusted_upper = current_upper.copy()
            v_upper = min(255, current_upper[2] + int(15 * (brightness_factor - 1)))
            adjusted_upper[2] = v_upper
        else:
            adjusted_lower = current_lower
            adjusted_upper = current_upper

        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Crear máscara con rangos ajustados
        mask = cv2.inRange(hsv, adjusted_lower, adjusted_upper)

        # Limpiar máscara
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Operaciones morfológicas para limpiar
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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
                        cv2.bitwise_and(mask, fg_mask, mask=fg_mask.astype(np.uint8) // 2)
                    )
                else:
                    # Usar principalmente segmentación por color
                    combined_mask = cv2.bitwise_or(mask, fg_mask.astype(np.uint8) // 4)

                mask = combined_mask

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos por área
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                valid_contours.append(cnt)

        # Aplicar filtros de estabilidad mejorados
        stable_contours = self._filter_stable_contours_advanced(valid_contours, vision_result)

        # Detectar si hay manos (usando contornos estables)
        has_hands = len(stable_contours) > 0

        return frame, stable_contours, has_hands
    
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
            
            # Normalizar distancia (asumiendo frame de 640x480)
            max_distance = np.sqrt(640**2 + 480**2)
            position_similarity = 1.0 - min(center_distance / max_distance, 1.0)
            
            # Puntuación combinada
            return 0.6 * area_ratio + 0.4 * position_similarity
            
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
                    if similarity > 0.7:  # Umbral de similitud
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
                    if similarity > 0.7:  # Umbral de similitud
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
            
            if circularity > 0.75 and solidity > 0.85:
                return "closed"
            elif circularity < 0.6 and solidity < 0.8:
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
                return 0.3 < circularity < 0.7
        
        return False
    
    def get_thumb_tip(self, contours: List) -> Optional[Tuple[float, float]]:
        """
        Obtiene la posición aproximada del "pulgar" (punto más a la izquierda).
        
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
        if not contours:
            return False
        
        # Tomar el contorno más grande (mano principal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Usar nueva lógica de estado de mano
        hand_state = self._determine_hand_state(largest_contour)
        
        return hand_state == "closed"
    
    def close(self):
        """Cierra el detector y libera recursos."""
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
