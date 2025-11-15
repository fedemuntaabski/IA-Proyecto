"""
Aplicación principal: Clasificador de Sketches en Tiempo Real

Este script integra todos los componentes para detectar dibujos en el aire
y clasificarlos en tiempo real.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List

# Importar módulos locales
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.core.hand_detector import HandDetector
from src.core.gesture_processor import GestureProcessor
from src.core.calibration_manager import CalibrationManager, CalibrationUI
# from src.core.classifier import SketchClassifier  # Comentado temporalmente


class SketchDrawer:
    """
    Aplicación principal que integra detección de manos, procesamiento de gestos y clasificación.
    """
    
    def __init__(self, model_path: str = "IA/sketch_classifier_model.keras",
                 model_info_path: str = "IA/model_info.json"):
        """
        Inicializa la aplicación.
        
        Args:
            model_path: Ruta al modelo
            model_info_path: Ruta a la información del modelo
        """
        print("Inicializando aplicación...")
        
        # Inicializar sistema de calibración
        self.calibration_manager = CalibrationManager()
        self.calibration_ui = CalibrationUI(self.calibration_manager)
        
        # Verificar si ya está calibrado
        if self.calibration_manager.is_calibrated:
            print("✓ Sistema calibrado previamente")
            calibrated_ranges = self.calibration_manager.get_calibrated_ranges()
            if calibrated_ranges:
                # Usar rangos calibrados
                skin_lower = calibrated_ranges.get('skin_lower', (0, 20, 70))
                skin_upper = calibrated_ranges.get('skin_upper', (20, 255, 255))
                self.hand_detector = HandDetector(
                    min_area=5000, max_area=50000,
                    skin_lower=skin_lower, skin_upper=skin_upper
                )
            else:
                # Fallback a valores por defecto
                self.hand_detector = HandDetector(min_area=5000, max_area=50000)
        else:
            print("⚠ Sistema no calibrado - usar valores por defecto")
            self.hand_detector = HandDetector(min_area=5000, max_area=50000)
        
        self.gesture_processor = GestureProcessor(image_size=28)
        # self.classifier = SketchClassifier(model_path, model_info_path)  # Comentado temporalmente
        
        # Estado
        self.is_drawing = False
        self.last_position = None
        self.predictions = []
        self.fps_counter = []
        self.gesture_complete = False
        
        # Configuración
        self.confidence_threshold = 0.5
        self.min_points_for_gesture = 5
        
        print("✓ Aplicación inicializada correctamente\n")
        # print("Información del modelo:")
        # info = self.classifier.get_model_info()
        # for key, value in info.items():
        #     print(f"  {key}: {value}")
        print("⚠ Modo debug: Clasificación deshabilitada")
        print()
    
    def is_drawing_gesture(self, contours: List) -> bool:
        """
        Determina si la mano está en gesto de dibujo usando el HandDetector mejorado.
        
        Args:
            contours: Lista de contornos detectados
            
        Returns:
            True si está en gesto de dibujo
        """
        return self.hand_detector.is_drawing_gesture(contours)
    
    def run_calibration(self) -> bool:
        """
        Ejecuta el proceso de calibración interactiva.
        
        Returns:
            True si la calibración fue exitosa, False en caso contrario
        """
        print("\n" + "="*60)
        print("CALIBRACIÓN DEL SISTEMA")
        print("="*60)
        print("Este proceso ajustará automáticamente la detección de piel")
        print("para tu iluminación y tono de piel específicos.")
        print()
        print("INSTRUCCIONES:")
        print("1. Asegúrate de tener buena iluminación")
        print("2. Sostén tu mano en el área verde y presiona ESPACIO")
        print("3. Repite 3 veces en diferentes posiciones")
        print("4. Luego toma 2 muestras de fondo fuera del área")
        print("5. Presiona 'C' para calibrar")
        print()
        print("Presiona cualquier tecla para comenzar...")
        print("="*60)
        
        # Esperar input del usuario
        cv2.waitKey(0)
        
        # Inicializar cámara para calibración
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Iniciando proceso de calibración...")
        print("Presiona 'Q' en cualquier momento para cancelar")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Voltear para efecto espejo
                frame = cv2.flip(frame, 1)
                
                # Dibujar interfaz de calibración
                display_frame = self.calibration_ui.draw_calibration_interface(frame)
                
                # Mostrar frame
                cv2.imshow('Calibración - Air Draw Classifier', display_frame)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                command = self.calibration_ui.handle_key_press(key, frame)
                
                if command == "quit":
                    print("Calibración cancelada por el usuario")
                    break
                elif command == "calibrate":
                    print("✓ Calibración completada exitosamente!")
                    
                    # Actualizar detector con nuevos rangos
                    calibrated_ranges = self.calibration_manager.get_calibrated_ranges()
                    if calibrated_ranges:
                        skin_lower = calibrated_ranges.get('skin_lower')
                        skin_upper = calibrated_ranges.get('skin_upper')
                        
                        # Recrear detector con rangos calibrados
                        self.hand_detector = HandDetector(
                            min_area=5000, max_area=50000,
                            skin_lower=skin_lower, skin_upper=skin_upper
                        )
                        print("✓ Detector actualizado con rangos calibrados")
                    
                    cv2.waitKey(2000)  # Mostrar mensaje de éxito
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return False

    def run(self):
        """Ejecuta el loop principal de la aplicación."""
        print("Iniciando cámara...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Cámara inicializada\n")
        print("Controles:")
        print("  r - Resetear dibujo")
        print("  c - Recalibrar sistema")
        print("  q - Salir")
        print("  SPACE - Procesar dibujo actual\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Voltear frame para efecto espejo
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                
                # Detectar manos
                frame_rgb, contours, has_hands = self.hand_detector.detect(frame)
                
                # Procesar si hay manos detectadas
                if has_hands and contours:
                    # Obtener posición del dedo índice
                    index_pos = self.hand_detector.get_index_finger_tip(contours)
                    
                    if index_pos and self.is_drawing_gesture(contours):
                        # Usuario está dibujando
                        if not self.is_drawing:
                            self.is_drawing = True
                            self.gesture_processor.clear()
                            print("▶ Iniciando dibujo...")
                        
                        # Agregar punto (convertir a coordenadas normalizadas)
                        normalized_pos = (index_pos[0] / width, index_pos[1] / height)
                        self.gesture_processor.add_point(normalized_pos, (height, width))
                        self.last_position = index_pos
                    else:
                        # Usuario dejó de dibujar
                        if self.is_drawing:
                            self.is_drawing = False
                            self.gesture_complete = True
                            print("⏹ Dibujo completado - Presionar SPACE para clasificar")
                
                # Preparar display
                display_frame = frame.copy()
                
                # Dibujar contornos si hay manos
                if has_hands and contours:
                    display_frame = self.hand_detector.draw_landmarks(display_frame, contours)
                
                # Dibujar puntos del gesto en progreso
                if len(self.gesture_processor.stroke_points) > 0:
                    display_frame = self.gesture_processor.draw_on_frame(
                        display_frame, 
                        frame_shape=(height, width)
                    )
                
                # Información en pantalla mejorada
                height, width = frame.shape[:2]
                
                # Barra de estado superior
                status_y = 30
                cv2.rectangle(display_frame, (10, 10), (width-10, 50), (50, 50, 50), -1)
                cv2.rectangle(display_frame, (10, 10), (width-10, 50), (255, 255, 255), 1)
                
                # Estado de calibración
                calib_status = "CALIBRADO" if self.calibration_manager.is_calibrated else "SIN CALIBRAR"
                calib_color = (0, 255, 0) if self.calibration_manager.is_calibrated else (0, 165, 255)
                cv2.putText(display_frame, f"Estado: {calib_status}", (20, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
                
                # FPS
                fps_color = (0, 255, 0) if self.get_fps() > 20 else (0, 165, 255)
                cv2.putText(display_frame, f"FPS: {self.get_fps():.1f}", (200, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
                
                # Estado de dibujo
                draw_status = "DIBUJANDO" if self.is_drawing else "LISTO"
                draw_color = (0, 255, 0) if self.is_drawing else (255, 165, 0)
                cv2.putText(display_frame, f"Modo: {draw_status}", (350, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                
                # Información de detección mejorada
                if has_hands and contours:
                    # Estado de la mano
                    is_fist = self.hand_detector.is_fist(contours)
                    hand_state = "CERRADA" if is_fist else "ABIERTA"
                    hand_color = (0, 0, 255) if is_fist else (0, 255, 0)
                    
                    # Confianza basada en estabilidad del historial
                    stability = min(len(self.hand_detector.contour_history), self.hand_detector.stability_threshold)
                    confidence_level = "ALTA" if stability >= self.hand_detector.stability_threshold else "MEDIA"
                    confidence_color = (0, 255, 0) if stability >= self.hand_detector.stability_threshold else (0, 165, 255)
                    
                    cv2.putText(display_frame, f"Manos: {len(contours)} | Estado: {hand_state}", (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
                    cv2.putText(display_frame, f"Estabilidad: {confidence_level} ({stability}/{self.hand_detector.stability_threshold})", (20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)
                else:
                    cv2.putText(display_frame, "Manos: 0 | Sin detección", (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Contador de puntos si está dibujando
                if self.is_drawing:
                    points_count = len(self.gesture_processor.stroke_points)
                    cv2.putText(display_frame, f"Puntos dibujados: {points_count}", (20, 105), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Mensaje de completado
                if self.gesture_complete:
                    cv2.putText(display_frame, "GESTO COMPLETADO - Presiona ESPACIO para procesar", 
                               (width//2 - 200, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Mostrar predicciones
                if self.predictions:
                    y_pos = 60
                    cv2.putText(display_frame, "Top 3 predicciones:", 
                               (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    for i, (class_name, confidence) in enumerate(self.predictions[:3]):
                        text = f"{i+1}. {class_name}: {confidence:.1%}"
                        cv2.putText(display_frame, text, (width - 250, 60 + i * 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Panel de controles inferior
                controls_y = height - 80
                cv2.rectangle(display_frame, (10, controls_y - 10), (width-10, height-10), (30, 30, 30), -1)
                cv2.rectangle(display_frame, (10, controls_y - 10), (width-10, height-10), (200, 200, 200), 1)
                
                controls_text = "CONTROLES: [ESPACIO] Procesar | [R] Reset | [C] Recalibrar | [Q] Salir"
                cv2.putText(display_frame, controls_text, (20, controls_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nSaliendo...")
                    break
                elif key == ord('r'):
                    print("Reseteando dibujo...")
                    self.gesture_processor.clear()
                    self.predictions = []
                    self.gesture_complete = False
                    self.is_drawing = False
                elif key == ord('c'):
                    print("Iniciando recalibración...")
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Ejecutar calibración
                    if self.run_calibration():
                        print("✓ Recalibración completada. Reiniciando aplicación...")
                    else:
                        print("⚠ Recalibración fallida. Continuando con configuración actual...")
                    
                    # Reinicializar cámara
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                elif key == ord(' ') and (self.gesture_complete or self.gesture_processor.stroke_points):
                    # Procesar gesto
                    self.process_gesture()
                
                # Actualizar FPS
                self.update_fps()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            print("✓ Aplicación cerrada")
    
    def process_gesture(self):
        """Procesa el gesto dibujado y realiza clasificación."""
        if len(self.gesture_processor.stroke_points) < self.min_points_for_gesture:
            print(f"⚠ Dibujo muy corto ({len(self.gesture_processor.stroke_points)} puntos)")
            return
        
        print("\n🔍 Procesando gesto...")
        
        # Obtener imagen
        gesture_image = self.gesture_processor.get_gesture_image()
        
        if gesture_image is None:
            print("⚠ No se pudo procesar el gesto")
            return
        
        # DEBUG: Solo mostrar información sin clasificar
        print(f"✓ Imagen generada: {gesture_image.shape}")
        print(f"  Rango de valores: [{gesture_image.min():.2f}, {gesture_image.max():.2f}]")
        print("⚠ Clasificación deshabilitada (TensorFlow no disponible)")
        
        self.predictions = [("Debug", 0.5), ("Sin modelo", 0.3)]  # Predicciones dummy
        self.gesture_complete = False
        print()
    
    def get_fps(self) -> float:
        """Calcula FPS promedio."""
        if not self.fps_counter:
            return 0
        
        if len(self.fps_counter) > 30:
            self.fps_counter.pop(0)
        
        fps = len(self.fps_counter) / (time.time() - self.fps_counter[0] + 1e-6)
        return fps
    
    def update_fps(self):
        """Actualiza contador de FPS."""
        self.fps_counter.append(time.time())


def main():
    """Punto de entrada de la aplicación."""
    app = SketchDrawer()
    
    # Verificar calibración
    if not app.calibration_manager.is_calibrated:
        print("🔧 Sistema no calibrado. Iniciando proceso de calibración...")
        calibration_success = app.run_calibration()
        
        if not calibration_success:
            print("⚠ Calibración fallida o cancelada. Usando configuración por defecto.")
            print("Puedes recalibrar más tarde presionando 'R' en el menú principal.")
        else:
            print("✓ Sistema calibrado exitosamente!")
    
    # Ejecutar aplicación principal
    app.run()


if __name__ == "__main__":
    main()
