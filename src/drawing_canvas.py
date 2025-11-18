"""
drawing_canvas.py - Drawing canvas for model prediction

Manages the internal canvas used for ML model predictions.
Keeps drawing history and provides preprocessing for inference.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from threading import Lock


class DrawingCanvas:
    """
    Manages drawing canvas for ML predictions.
    Follows KISS principle - simple point accumulation and rendering.
    """
    
    def __init__(self, size: int = 256):
        """
        Initialize drawing canvas.
        
        Args:
            size: Canvas size (square)
        """
        self.size = size
        self.canvas = self._create_blank_canvas()
        self.last_point = None
        self._lock = Lock()  # Thread safety for canvas operations
    
    def _create_blank_canvas(self) -> np.ndarray:
        """Create blank white canvas."""
        return np.ones((self.size, self.size), dtype=np.uint8) * 255
    
    def add_point(self, x: float, y: float, is_drawing: bool):
        """
        Add drawing point to canvas.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            is_drawing: Whether currently drawing
        """
        with self._lock:  # Thread-safe access to canvas
            try:
                if not is_drawing:
                    # End of stroke
                    self.last_point = None
                    return
                
                # Validate coordinates
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    return
                
                # Convert to canvas coordinates
                cx = int(x * (self.size - 1))
                cy = int(y * (self.size - 1))
                
                # Clamp to canvas bounds
                cx = max(0, min(cx, self.size - 1))
                cy = max(0, min(cy, self.size - 1))
                
                current_point = (cx, cy)
                
                # Draw line from last point
                if self.last_point is not None:
                    cv2.line(
                        self.canvas,
                        self.last_point,
                        current_point,
                        0,  # Black line
                        8,
                        cv2.LINE_AA
                    )
                
                self.last_point = current_point
            except Exception as e:
                # Silently handle errors to prevent crashes
                self.last_point = None
    
    def clear(self):
        """Clear the canvas."""
        with self._lock:  # Thread-safe clear
            self.canvas = self._create_blank_canvas()
            self.last_point = None
    
    def is_empty(self) -> bool:
        """
        Check if canvas is empty.
        
        Returns:
            True if no drawing on canvas
        """
        with self._lock:  # Thread-safe read
            # Count non-white pixels
            return np.sum(self.canvas < 250) < 30
    
    def get_preprocessed_for_model(self) -> Optional[np.ndarray]:
        """
        Get preprocessed canvas ready for model inference.
        
        Returns:
            Preprocessed image (28x28x1) or None if empty
        """
        with self._lock:  # Thread-safe preprocessing
            # Check if empty (inline to avoid deadlock)
            if np.sum(self.canvas < 250) < 30:
                return None
            
            try:
                # 1. Find bounding box of drawing
                binary = cv2.threshold(self.canvas, 127, 255, cv2.THRESH_BINARY_INV)[1]
                coords = cv2.findNonZero(binary)
                
                if coords is None:
                    return None
                
                x, y, w, h = cv2.boundingRect(coords)
                
                # 2. Add 12% padding
                max_dim = max(w, h)
                pad = int(max_dim * 0.12)
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(self.size - x, w + 2 * pad)
                h = min(self.size - y, h + 2 * pad)
                
                # 3. Extract ROI
                roi = self.canvas[y:y+h, x:x+w]
                
                # 4. Center in square canvas
                max_dim = max(w, h)
                square = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
                offset_x = (max_dim - w) // 2
                offset_y = (max_dim - h) // 2
                square[offset_y:offset_y+h, offset_x:offset_x+w] = roi
                
                # 5. Resize to 28x28
                resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_LANCZOS4)
                
                # 6. Normalize and invert (white on black for model)
                normalized = resized.astype(np.float32) / 255.0
                normalized = 1.0 - normalized  # Invert
                normalized = np.expand_dims(normalized, axis=-1)  # Add channel dimension
                
                return normalized
            
            except Exception as e:
                return None
    
    def get_canvas_copy(self) -> np.ndarray:
        """
        Get copy of current canvas.
        
        Returns:
            Copy of canvas array
        """
        with self._lock:  # Thread-safe copy
            return self.canvas.copy()
