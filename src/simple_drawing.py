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
        self.line_thickness = 3  # Optimal for 28x28 model (was 8)
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
            if not self.saved_strokes:
                return None
            
            try:
                canvas = self._render_to_canvas()
                if canvas is None:
                    return None
                
                # Process canvas to model input
                return self._process_canvas_to_model_input(canvas)
                
            except Exception:
                return None
    
    def _render_to_canvas(self, canvas_size: int = 256) -> Optional[np.ndarray]:
        """Render strokes to high-res canvas."""
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        
        for stroke in self.saved_strokes:
            if len(stroke) < 2:
                continue
            
            pixel_points = [(int(x * canvas_size), int(y * canvas_size)) for x, y in stroke]
            
            for i in range(len(pixel_points) - 1):
                cv2.line(canvas, pixel_points[i], pixel_points[i + 1], 
                        0, 3, cv2.LINE_AA)  # Fixed thickness for consistent model input
        
        # Validate drawing
        if np.sum(canvas < 250) < 30:
            return None
        
        return canvas
    
    def _process_canvas_to_model_input(self, canvas: np.ndarray) -> Optional[np.ndarray]:
        """
        Process canvas to 28x28 model input with improved preprocessing.
        
        This method ensures the drawing is properly centered and normalized
        regardless of where on the canvas it was drawn.
        """
        # Find and crop content
        binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)[1]
        coords = cv2.findNonZero(binary)
        
        if coords is None:
            return None
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Calculate padding (20% of the drawing size for better centering)
        max_dim = max(w, h)
        pad = int(max_dim * 0.20)
        
        # Apply padding with bounds checking
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(canvas.shape[1], x + w + pad)
        y2 = min(canvas.shape[0], y + h + pad)
        
        # Extract ROI with padding
        roi = canvas[y1:y2, x1:x2]
        
        # Create square canvas centered
        roi_h, roi_w = roi.shape
        square_size = max(roi_h, roi_w)
        
        # Ensure minimum size for better quality
        square_size = max(square_size, 100)
        
        # Create white square canvas
        square = np.ones((square_size, square_size), dtype=np.uint8) * 255
        
        # Calculate offsets to center the ROI
        offset_x = (square_size - roi_w) // 2
        offset_y = (square_size - roi_h) // 2
        
        # Paste ROI centered in square
        square[offset_y:offset_y+roi_h, offset_x:offset_x+roi_w] = roi
        
        # Resize to 28x28 with high-quality interpolation
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize: 0=white background, 1=black lines
        # This matches the training data format
        normalized = 1.0 - (resized.astype(np.float32) / 255.0)
        
        # Add channel dimension
        return np.expand_dims(normalized, axis=-1)
