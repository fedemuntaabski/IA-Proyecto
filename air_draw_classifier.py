"""
Aplicación principal: Clasificador de Sketches en Tiempo Real

Este script integra todos los componentes para detectar dibujos en el aire
y clasificarlos en tiempo real.
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Importar módulos locales
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.core.hand_detector import HandDetector
from src.core.gesture_processor import GestureProcessor
from src.core.classifier import SketchClassifier


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
        
        # Inicializar componentes
        self.hand_detector = HandDetector(min_detection_confidence=0.7)
        self.gesture_processor = GestureProcessor(image_size=28)
        self.classifier = SketchClassifier(model_path, model_info_path)
        
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
        print("Información del modelo:")
        info = self.classifier.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
    
    def is_drawing_gesture(self, hand_landmarks) -> bool:
        """
        Determina si la mano está en gesto de dibujo (dedo índice extendido).
        
        Args:
            hand_landmarks: Landmarks de MediaPipe
            
        Returns:
            True si está en gesto de dibujo
        """
        # Dedo índice (landmark 8) debe estar extendido
        # Pulgar (landmark 4) debe estar cerrado
        
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]  # PIP del índice
        
        # Si la punta del índice está por debajo del PIP, está extendido
        return index_tip.y < index_pip.y
    
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
                frame_rgb, landmarks, has_hands = self.hand_detector.detect(frame)
                
                # Procesar si hay manos detectadas
                if has_hands and landmarks:
                    # Tomar primera mano detectada
                    hand = landmarks[0]
                    
                    # Obtener posición del dedo índice
                    index_pos = self.hand_detector.get_index_finger_tip(hand)
                    
                    if index_pos and self.is_drawing_gesture(hand):
                        # Usuario está dibujando
                        if not self.is_drawing:
                            self.is_drawing = True
                            self.gesture_processor.clear()
                            print("▶ Iniciando dibujo...")
                        
                        # Agregar punto
                        self.gesture_processor.add_point(index_pos, (height, width))
                        self.last_position = index_pos
                    else:
                        # Usuario dejó de dibujar
                        if self.is_drawing:
                            self.is_drawing = False
                            self.gesture_complete = True
                            print("⏹ Dibujo completado - Presionar SPACE para clasificar")
                
                # Preparar display
                display_frame = frame.copy()
                
                # Dibujar landmarks si hay manos
                if has_hands and landmarks:
                    frame_rgb = self.hand_detector.draw_landmarks(frame_rgb, landmarks)
                    display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Dibujar puntos del gesto en progreso
                if len(self.gesture_processor.stroke_points) > 0:
                    display_frame = self.gesture_processor.draw_on_frame(
                        display_frame, 
                        frame_shape=(height, width)
                    )
                
                # Información en pantalla
                info_text = f"FPS: {self.get_fps():.1f}"
                if self.is_drawing:
                    info_text += " | DIBUJANDO"
                    points_count = len(self.gesture_processor.stroke_points)
                    cv2.putText(display_frame, f"Puntos: {points_count}", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Mostrar predicciones
                if self.predictions:
                    y_pos = 60
                    cv2.putText(display_frame, "Top 3 predicciones:", 
                               (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    for i, (class_name, confidence) in enumerate(self.predictions[:3]):
                        text = f"{i+1}. {class_name}: {confidence:.1%}"
                        cv2.putText(display_frame, text, (width - 250, 60 + i * 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Mostrar frame
                cv2.imshow('Sketch Classifier - Air Draw', display_frame)
                
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
        
        # Clasificar
        predictions = self.classifier.predict(gesture_image, top_k=3)
        
        # Verificar confianza
        top_class, top_confidence = predictions[0]
        
        if top_confidence >= self.confidence_threshold:
            print(f"✓ Predicción: {top_class} ({top_confidence:.1%})")
        else:
            print(f"⚠ Baja confianza: {top_class} ({top_confidence:.1%})")
        
        # Mostrar todas las predicciones
        print("  Top 3:")
        for i, (class_name, confidence) in enumerate(predictions):
            print(f"    {i+1}. {class_name}: {confidence:.1%}")
        
        self.predictions = predictions
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
    app.run()


if __name__ == "__main__":
    main()
