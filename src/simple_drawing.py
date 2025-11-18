"""
simple_drawing.py - Unified Drawing System (KISS Principle)

Single module handling all drawing logic:
- Point collection with index finger
- Stroke completion on fist close
- Rendering with 8px green lines
- Preprocessing for ML model (28x28)

No complexity, no state machines, just simple point tracking.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from threading import Lock


class SimpleDrawing:
    """
    Unified drawing system following KISS principle.
    
    Handles:
    - Drawing with index finger (8px green line)
    - Completing stroke on fist close
    - Rendering on video frames
    - Preprocessing for ML model
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize drawing system.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.width = width
        self.height = height
        
        # Current stroke being drawn
        self.current_points: List[Tuple[float, float]] = []
        
        # Completed strokes saved on screen
        self.saved_strokes: List[List[Tuple[float, float]]] = []
        
        # Thread safety
        self._lock = Lock()
        
        # Visual config
        self.line_color = (0, 255, 0)  # Green (BGR)
        self.line_thickness = 8
        self.finger_indicator_color = (0, 0, 255)  # Red (BGR)
        self.finger_indicator_radius = 10
        
        # Minimum points for valid stroke
        self.min_points = 5
    
    def add_point(self, x: float, y: float) -> None:
        """
        Add point to current stroke (called when index finger visible).
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
        """
        with self._lock:
            # Validate coordinates
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return
            
            self.current_points.append((x, y))
    
    def save_current_stroke(self) -> bool:
        """
        Save current stroke (called when fist closes).
        
        Returns:
            True if stroke was saved, False if invalid
        """
        with self._lock:
            if len(self.current_points) >= self.min_points:
                # Save the stroke
                self.saved_strokes.append(self.current_points.copy())
                self.current_points = []
                return True
            else:
                # Discard insufficient points
                self.current_points = []
                return False
    
    def clear_all(self) -> None:
        """Clear all strokes (current + saved)."""
        with self._lock:
            self.current_points = []
            self.saved_strokes = []
    
    def has_drawing(self) -> bool:
        """Check if there's any saved drawing."""
        with self._lock:
            return len(self.saved_strokes) > 0
    
    def get_stroke_count(self) -> int:
        """Get number of saved strokes."""
        with self._lock:
            return len(self.saved_strokes)
    
    def render_on_frame(self, frame: np.ndarray, finger_pos: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Render all strokes on video frame.
        
        Args:
            frame: BGR video frame
            finger_pos: Optional (x, y) normalized finger position for indicator
            
        Returns:
            Frame with drawings rendered
        """
        with self._lock:
            h, w = frame.shape[:2]
            
            # Render saved strokes (persistent)
            for stroke in self.saved_strokes:
                self._render_stroke(frame, stroke, w, h)
            
            # Render current stroke being drawn
            if len(self.current_points) > 0:
                self._render_stroke(frame, self.current_points, w, h)
            
            # Render finger position indicator
            if finger_pos is not None:
                fx, fy = finger_pos
                if 0 <= fx <= 1 and 0 <= fy <= 1:
                    px = int(fx * w)
                    py = int(fy * h)
                    cv2.circle(frame, (px, py), self.finger_indicator_radius, 
                             self.finger_indicator_color, -1)
            
            return frame
    
    def _render_stroke(self, frame: np.ndarray, points: List[Tuple[float, float]], 
                      w: int, h: int) -> None:
        """
        Render a single stroke on frame.
        
        Args:
            frame: Frame to draw on
            points: List of normalized (x, y) coordinates
            w: Frame width
            h: Frame height
        """
        if len(points) < 2:
            return
        
        # Convert to pixel coordinates
        pixel_points = [(int(x * w), int(y * h)) for x, y in points]
        
        # Draw lines between consecutive points
        for i in range(len(pixel_points) - 1):
            cv2.line(frame, pixel_points[i], pixel_points[i + 1], 
                    self.line_color, self.line_thickness, cv2.LINE_AA)
    
    def get_preprocessed_image(self) -> Optional[np.ndarray]:
        """
        Get preprocessed image for ML model (28x28x1).
        
        Returns:
            Preprocessed numpy array or None if no drawing
        """
        with self._lock:
            if len(self.saved_strokes) == 0:
                return None
            
            try:
                # Create high-res canvas for rendering
                canvas_size = 256
                canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255  # White
                
                # Render all saved strokes in black
                for stroke in self.saved_strokes:
                    if len(stroke) < 2:
                        continue
                    
                    # Convert to pixel coordinates
                    pixel_points = [
                        (int(x * canvas_size), int(y * canvas_size)) 
                        for x, y in stroke
                    ]
                    
                    # Draw stroke
                    for i in range(len(pixel_points) - 1):
                        cv2.line(canvas, pixel_points[i], pixel_points[i + 1], 
                                0, self.line_thickness, cv2.LINE_AA)
                
                # Check if anything was drawn
                if np.sum(canvas < 250) < 30:
                    return None
                
                # Find bounding box and crop
                binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)[1]
                coords = cv2.findNonZero(binary)
                
                if coords is None:
                    return None
                
                x, y, w, h = cv2.boundingRect(coords)
                
                # Add 12% padding
                max_dim = max(w, h)
                pad = int(max_dim * 0.12)
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(canvas_size - x, w + 2 * pad)
                h = min(canvas_size - y, h + 2 * pad)
                
                # Extract ROI
                roi = canvas[y:y+h, x:x+w]
                
                # Center in square
                max_dim = max(w, h)
                square = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
                offset_x = (max_dim - w) // 2
                offset_y = (max_dim - h) // 2
                square[offset_y:offset_y+h, offset_x:offset_x+w] = roi
                
                # Resize to 28x28
                resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_LANCZOS4)
                
                # Normalize and invert (white on black for model)
                normalized = resized.astype(np.float32) / 255.0
                normalized = 1.0 - normalized  # Invert to white on black
                normalized = np.expand_dims(normalized, axis=-1)  # Add channel
                
                return normalized
                
            except Exception:
                return None
