"""
prediction_tracer.py - Slow-motion prediction drawing visualization

Shows the predicted object being drawn slowly on camera overlay.
Helps users learn how to draw each category correctly.
"""

import cv2
import numpy as np
import math
import random
from typing import Optional, Tuple, List
from PyQt6.QtCore import QThread, pyqtSignal
import time


class PredictionTracer(QThread):
    """
    Thread that animates the predicted drawing slowly.
    
    Shows how to draw the predicted object stroke by stroke.
    """
    
    # Signals
    drawing_update = pyqtSignal(list, str, float)  # Emits (strokes_so_far, label, progress)
    finished = pyqtSignal()
    
    def __init__(self, label: str, confidence: float, frame_width: int, frame_height: int):
        """
        Initialize tracer.
        
        Args:
            label: Predicted label to draw
            confidence: Prediction confidence
            frame_width: Camera frame width
            frame_height: Camera frame height
        """
        super().__init__()
        self.label = label
        self.confidence = confidence
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.is_running = False
    
    def run(self):
        """Animate the drawing slowly, stroke by stroke."""
        self.is_running = True
        
        # Generate strokes for this label
        strokes = self._generate_strokes_for_label(self.label)
        
        if not strokes:
            self.finished.emit()
            return
        
        # Animate each stroke progressively
        total_strokes = len(strokes)
        
        for stroke_idx in range(total_strokes + 1):
            if not self.is_running:
                break
            
            # Send strokes drawn so far
            strokes_so_far = strokes[:stroke_idx]
            progress = stroke_idx / total_strokes if total_strokes > 0 else 1.0
            
            self.drawing_update.emit(strokes_so_far, self.label, progress)
            
            # Delay between strokes (slower for better visibility)
            time.sleep(0.8)  # 800ms per stroke
        
        self.finished.emit()
        self.is_running = False
    
    def _generate_strokes_for_label(self, label: str) -> List[List[Tuple[int, int]]]:
        """
        Generate drawing strokes for a label.
        
        Args:
            label: Label name
            
        Returns:
            List of strokes, where each stroke is a list of (x, y) points
        """
        # Center of frame
        cx = self.frame_width // 2
        cy = self.frame_height // 2
        scale = min(self.frame_width, self.frame_height) // 4  # Drawing size
        
        label_lower = label.lower()
        random.seed(hash(label))  # Consistent for same label
        
        # Return strokes based on category
        if 'cat' in label_lower:
            return self._strokes_cat(cx, cy, scale)
        elif 'dog' in label_lower:
            return self._strokes_dog(cx, cy, scale)
        elif 'bird' in label_lower:
            return self._strokes_bird(cx, cy, scale)
        elif 'fish' in label_lower:
            return self._strokes_fish(cx, cy, scale)
        elif 'car' in label_lower or 'truck' in label_lower:
            return self._strokes_car(cx, cy, scale)
        elif 'airplane' in label_lower or 'plane' in label_lower:
            return self._strokes_airplane(cx, cy, scale)
        elif 'house' in label_lower:
            return self._strokes_house(cx, cy, scale)
        elif 'tree' in label_lower:
            return self._strokes_tree(cx, cy, scale)
        elif 'flower' in label_lower:
            return self._strokes_flower(cx, cy, scale)
        elif 'star' in label_lower:
            return self._strokes_star(cx, cy, scale)
        elif 'heart' in label_lower:
            return self._strokes_heart(cx, cy, scale)
        elif 'circle' in label_lower or 'ball' in label_lower or 'sun' in label_lower:
            return self._strokes_circle(cx, cy, scale)
        elif 'square' in label_lower or 'box' in label_lower:
            return self._strokes_square(cx, cy, scale)
        else:
            # Generic simple pattern
            return self._strokes_generic(cx, cy, scale)
    
    # Stroke generation methods (each returns list of strokes)
    
    def _strokes_cat(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Cat face strokes."""
        return [
            self._circle_points(cx, cy, int(s * 0.6), 40),  # Head
            [(cx - s//3, cy - s//3), (cx - s//2, cy - int(s * 0.8))],  # Left ear
            [(cx - s//3, cy - s//3), (cx - s//5, cy - int(s * 0.7))],  # Left ear inner
            [(cx + s//3, cy - s//3), (cx + s//2, cy - int(s * 0.8))],  # Right ear
            [(cx + s//3, cy - s//3), (cx + s//5, cy - int(s * 0.7))],  # Right ear inner
            self._circle_points(cx - s//5, cy - s//10, s//15, 20),  # Left eye
            self._circle_points(cx + s//5, cy - s//10, s//15, 20),  # Right eye
            [(cx, cy + s//10), (cx - s//8, cy + s//5)],  # Nose left
            [(cx, cy + s//10), (cx + s//8, cy + s//5)],  # Nose right
        ]
    
    def _strokes_dog(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Dog strokes."""
        return [
            self._ellipse_points(cx, cy, int(s * 0.8), int(s * 0.5), 40),  # Body
            self._circle_points(cx + s//3, cy - s//4, int(s * 0.45), 30),  # Head
            self._ellipse_points(cx + s//8, cy - int(s * 0.55), s//6, int(s * 0.3), 20),  # Left ear
            self._ellipse_points(cx + int(s * 0.58), cy - int(s * 0.55), s//6, int(s * 0.3), 20),  # Right ear
            [(cx - s//4, cy + s//3), (cx - s//4, cy + int(s * 0.75))],  # Front left leg
            [(cx, cy + s//3), (cx, cy + int(s * 0.75))],  # Front right leg
            [(cx + s//4, cy + s//3), (cx + s//4, cy + int(s * 0.75))],  # Back leg
        ]
    
    def _strokes_bird(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Bird strokes."""
        return [
            self._ellipse_points(cx, cy, int(s * 0.5), int(s * 0.4), 30),  # Body
            self._circle_points(cx + s//3, cy - s//4, int(s * 0.25), 25),  # Head
            [(cx + int(s * 0.58), cy - s//6), (cx + int(s * 0.85), cy - s//6)],  # Beak
            self._arc_points(cx - s//8, cy, int(s * 0.35), int(s * 0.25), 0, 180, 20),  # Wing
            [(cx - s//3, cy), (cx - int(s * 0.7), cy - s//6)],  # Tail up
            [(cx - s//3, cy + s//8), (cx - int(s * 0.7), cy + s//8)],  # Tail mid
        ]
    
    def _strokes_fish(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Fish strokes."""
        return [
            self._ellipse_points(cx, cy, int(s * 0.6), int(s * 0.35), 35),  # Body
            [(cx - int(s * 0.6), cy), (cx - int(s * 0.9), cy - s//3)],  # Tail top
            [(cx - int(s * 0.6), cy), (cx - int(s * 0.9), cy + s//3)],  # Tail bottom
            [(cx - int(s * 0.9), cy - s//3), (cx - int(s * 0.9), cy + s//3)],  # Tail connect
            self._circle_points(cx + s//4, cy - s//10, s//20, 15),  # Eye
        ]
    
    def _strokes_car(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Car strokes."""
        return [
            [(cx - int(s * 0.55), cy), (cx + int(s * 0.55), cy),
             (cx + int(s * 0.55), cy + s//3), (cx - int(s * 0.55), cy + s//3),
             (cx - int(s * 0.55), cy)],  # Body
            [(cx - s//4, cy - s//5), (cx - s//4, cy),
             (cx + s//4, cy), (cx + s//4, cy - s//5),
             (cx - s//4, cy - s//5)],  # Roof
            self._circle_points(cx - s//3, cy + s//3, s//7, 25),  # Left wheel
            self._circle_points(cx + s//3, cy + s//3, s//7, 25),  # Right wheel
        ]
    
    def _strokes_airplane(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Airplane strokes."""
        return [
            [(cx - int(s * 0.6), cy), (cx + int(s * 0.6), cy)],  # Fuselage
            [(cx - s//4, cy), (cx - s//4, cy - int(s * 0.5)),
             (cx + s//4, cy - int(s * 0.5)), (cx + s//4, cy)],  # Wings
            [(cx + int(s * 0.5), cy), (cx + int(s * 0.5), cy - s//3)],  # Tail vert
            [(cx + int(s * 0.35), cy - s//3), (cx + int(s * 0.65), cy - s//3)],  # Tail horiz
        ]
    
    def _strokes_house(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """House strokes."""
        return [
            [(cx - int(s * 0.5), cy), (cx + int(s * 0.5), cy),
             (cx + int(s * 0.5), cy + int(s * 0.65)), (cx - int(s * 0.5), cy + int(s * 0.65)),
             (cx - int(s * 0.5), cy)],  # Base
            [(cx - int(s * 0.6), cy), (cx, cy - s//2), (cx + int(s * 0.6), cy)],  # Roof
            [(cx - s//6, cy + s//5), (cx - s//6, cy + int(s * 0.65)),
             (cx + s//6, cy + int(s * 0.65)), (cx + s//6, cy + s//5),
             (cx - s//6, cy + s//5)],  # Door
            [(cx + s//4, cy + s//6), (cx + s//4, cy + s//3),
             (cx + int(s * 0.45), cy + s//3), (cx + int(s * 0.45), cy + s//6),
             (cx + s//4, cy + s//6)],  # Window
        ]
    
    def _strokes_tree(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Tree strokes."""
        return [
            [(cx - s//12, cy + s//5), (cx - s//12, cy + int(s * 0.7)),
             (cx + s//12, cy + int(s * 0.7)), (cx + s//12, cy + s//5),
             (cx - s//12, cy + s//5)],  # Trunk
            self._circle_points(cx, cy - s//5, int(s * 0.6), 40),  # Leaves
        ]
    
    def _strokes_flower(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Flower strokes."""
        strokes = [
            [(cx, cy + s//5), (cx, cy + int(s * 0.7))],  # Stem
            self._circle_points(cx, cy, s//10, 20),  # Center
        ]
        # Petals
        for i in range(5):
            angle = i * 72
            x = cx + int(s * 0.25 * math.cos(math.radians(angle)))
            y = cy + int(s * 0.25 * math.sin(math.radians(angle)))
            strokes.append(self._circle_points(x, y, s//7, 20))
        return strokes
    
    def _strokes_star(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Star strokes (5-pointed)."""
        points = []
        for i in range(5):
            angle = i * 144 - 90
            x = cx + int(s * 0.6 * math.cos(math.radians(angle)))
            y = cy + int(s * 0.6 * math.sin(math.radians(angle)))
            points.append((x, y))
        
        # Connect points in star pattern
        strokes = []
        for i in range(5):
            strokes.append([points[i], points[(i + 2) % 5]])
        return strokes
    
    def _strokes_heart(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Heart strokes."""
        return [
            self._circle_points(cx - s//4, cy - s//5, int(s * 0.3), 25),  # Left bump
            self._circle_points(cx + s//4, cy - s//5, int(s * 0.3), 25),  # Right bump
            [(cx - int(s * 0.4), cy - s//10), (cx, cy + int(s * 0.5))],  # Left side
            [(cx + int(s * 0.4), cy - s//10), (cx, cy + int(s * 0.5))],  # Right side
        ]
    
    def _strokes_circle(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Circle strokes."""
        return [self._circle_points(cx, cy, int(s * 0.6), 50)]
    
    def _strokes_square(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Square strokes."""
        half = int(s * 0.6)
        return [[(cx - half, cy - half), (cx + half, cy - half),
                 (cx + half, cy + half), (cx - half, cy + half),
                 (cx - half, cy - half)]]
    
    def _strokes_generic(self, cx, cy, s) -> List[List[Tuple[int, int]]]:
        """Generic pattern."""
        return [
            self._circle_points(cx, cy, int(s * 0.5), 40),
            [(cx, cy - int(s * 0.5)), (cx, cy + int(s * 0.5))],
            [(cx - int(s * 0.5), cy), (cx + int(s * 0.5), cy)],
        ]
    
    # Helper methods for generating point sequences
    
    def _circle_points(self, cx: int, cy: int, radius: int, num_points: int = 50) -> List[Tuple[int, int]]:
        """Generate points for a circle."""
        points = []
        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * math.pi
            x = cx + int(radius * math.cos(angle))
            y = cy + int(radius * math.sin(angle))
            points.append((x, y))
        return points
    
    def _ellipse_points(self, cx: int, cy: int, rx: int, ry: int, num_points: int = 50) -> List[Tuple[int, int]]:
        """Generate points for an ellipse."""
        points = []
        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * math.pi
            x = cx + int(rx * math.cos(angle))
            y = cy + int(ry * math.sin(angle))
            points.append((x, y))
        return points
    
    def _arc_points(self, cx: int, cy: int, rx: int, ry: int, 
                   start_deg: float, end_deg: float, num_points: int = 30) -> List[Tuple[int, int]]:
        """Generate points for an arc."""
        points = []
        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)
        for i in range(num_points + 1):
            t = i / num_points
            angle = start_rad + (end_rad - start_rad) * t
            x = cx + int(rx * math.cos(angle))
            y = cy + int(ry * math.sin(angle))
            points.append((x, y))
        return points
    
    def stop(self):
        """Stop the animation."""
        self.is_running = False


class PredictionTracerDialog:
    """
    Manager for prediction tracer visualization.
    
    Integrates with main UI to show prediction drawing animations.
    """
    
    def __init__(self):
        """Initialize manager."""
        self.current_tracer: Optional[PredictionTracer] = None
        self.is_active = False
    
    def start_trace(self, label: str, confidence: float, 
                   frame_width: int, frame_height: int,
                   drawing_callback) -> None:
        """
        Start prediction trace animation.
        
        Args:
            label: Predicted label
            confidence: Prediction confidence
            frame_width: Camera frame width
            frame_height: Camera frame height
            drawing_callback: Function to call with drawing updates
        """
        # Stop any existing trace
        self.stop_trace()
        
        # Create and start new tracer
        self.current_tracer = PredictionTracer(label, confidence, frame_width, frame_height)
        self.current_tracer.drawing_update.connect(drawing_callback)
        self.current_tracer.finished.connect(self._on_trace_finished)
        self.current_tracer.finished.connect(lambda: drawing_callback([], label, 1.0))  # Clear on finish
        self.current_tracer.start()
        self.is_active = True
    
    def stop_trace(self):
        """Stop current trace."""
        if self.current_tracer is not None:
            self.current_tracer.stop()
            self.current_tracer.wait()
            self.current_tracer = None
        self.is_active = False
    
    def _on_trace_finished(self):
        """Handle trace completion."""
        self.is_active = False
