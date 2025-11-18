"""
drawing_system.py - Clean Drawing and Processing System

A robust, thread-safe system for capturing hand-drawn strokes and processing them
for ML model prediction. Follows best practices with clear separation of concerns.

Architecture:
- DrawingStroke: Represents a single continuous stroke with points
- DrawingCanvas: Manages multiple strokes and rendering
- DrawingProcessor: Handles preprocessing for ML model inference
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from threading import Lock
from dataclasses import dataclass, field
from enum import Enum


class StrokeState(Enum):
    """State of a drawing stroke."""
    ACTIVE = "active"      # Currently being drawn
    COMPLETED = "completed"  # Drawing finished, ready for rendering
    PROCESSED = "processed"  # Already sent for prediction


@dataclass
class DrawingStroke:
    """
    Represents a single continuous stroke drawn by the user.
    
    Attributes:
        points: List of (x, y) normalized coordinates (0-1)
        state: Current state of the stroke
        color: RGB color tuple for rendering
        thickness: Line thickness in pixels
    """
    points: List[Tuple[float, float]] = field(default_factory=list)
    state: StrokeState = StrokeState.ACTIVE
    color: Tuple[int, int, int] = (0, 255, 0)  # Green by default
    thickness: int = 8
    
    def add_point(self, x: float, y: float) -> None:
        """
        Add a point to the stroke.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
        """
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return  # Ignore invalid coordinates
        
        self.points.append((x, y))
    
    def complete(self) -> None:
        """Mark stroke as completed."""
        self.state = StrokeState.COMPLETED
    
    def is_valid(self) -> bool:
        """Check if stroke has enough points to be meaningful."""
        return len(self.points) >= 2


class DrawingCanvas:
    """
    Thread-safe canvas for managing drawing strokes and rendering.
    
    This class handles:
    - Multiple stroke management
    - Real-time rendering with visual feedback
    - Thread-safe operations for concurrent access
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize the drawing canvas.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
        """
        self.width = width
        self.height = height
        
        # Stroke management
        self.strokes: List[DrawingStroke] = []
        self.current_stroke: Optional[DrawingStroke] = None
        
        # Thread safety
        self._lock = Lock()
        
        # Visual feedback
        self.show_hand_position = True
        self.hand_position: Optional[Tuple[int, int]] = None
    
    def start_stroke(self, x: float, y: float) -> None:
        """
        Start a new drawing stroke.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
        """
        with self._lock:
            # Complete previous stroke if exists
            if self.current_stroke is not None:
                self.current_stroke.complete()
                if self.current_stroke.is_valid():
                    self.strokes.append(self.current_stroke)
            
            # Start new stroke
            self.current_stroke = DrawingStroke()
            self.current_stroke.add_point(x, y)
    
    def add_point(self, x: float, y: float) -> None:
        """
        Add a point to the current stroke.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
        """
        with self._lock:
            if self.current_stroke is not None:
                self.current_stroke.add_point(x, y)
            else:
                # Auto-start stroke if none exists
                self.start_stroke(x, y)
            
            # Update hand position for visual feedback
            px = int(x * self.width)
            py = int(y * self.height)
            self.hand_position = (px, py)
    
    def complete_current_stroke(self) -> bool:
        """
        Complete the current stroke (called when hand closes).
        
        Returns:
            True if a valid stroke was completed, False otherwise
        """
        with self._lock:
            if self.current_stroke is None:
                return False
            
            self.current_stroke.complete()
            
            if self.current_stroke.is_valid():
                self.strokes.append(self.current_stroke)
                self.current_stroke = None
                return True
            else:
                # Discard invalid stroke
                self.current_stroke = None
                return False
    
    def has_strokes(self) -> bool:
        """Check if canvas has any completed strokes."""
        with self._lock:
            return len(self.strokes) > 0
    
    def get_stroke_count(self) -> int:
        """Get number of completed strokes."""
        with self._lock:
            return len(self.strokes)
    
    def clear_all(self) -> None:
        """Clear all strokes and reset canvas."""
        with self._lock:
            self.strokes.clear()
            self.current_stroke = None
            self.hand_position = None
    
    def render_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Render all strokes onto a video frame.
        
        Args:
            frame: BGR video frame to draw on
            
        Returns:
            Frame with strokes rendered
        """
        with self._lock:
            # Make a copy to avoid modifying original
            output = frame.copy()
            h, w = output.shape[:2]
            
            # Render completed strokes
            for stroke in self.strokes:
                self._render_stroke(output, stroke, w, h)
            
            # Render current stroke being drawn
            if self.current_stroke is not None and len(self.current_stroke.points) > 0:
                self._render_stroke(output, self.current_stroke, w, h)
            
            # Render hand position indicator
            if self.show_hand_position and self.hand_position is not None:
                px, py = self.hand_position
                # Draw a red circle at finger tip
                cv2.circle(output, (px, py), 12, (0, 0, 255), -1)
                cv2.circle(output, (px, py), 14, (255, 255, 255), 2)
            
            return output
    
    def _render_stroke(self, frame: np.ndarray, stroke: DrawingStroke, 
                      frame_w: int, frame_h: int) -> None:
        """
        Render a single stroke on the frame.
        
        Args:
            frame: Frame to draw on
            stroke: Stroke to render
            frame_w: Frame width
            frame_h: Frame height
        """
        if len(stroke.points) < 2:
            return
        
        # Convert normalized points to pixel coordinates
        pixel_points = [
            (int(x * frame_w), int(y * frame_h))
            for x, y in stroke.points
        ]
        
        # Draw lines between consecutive points
        for i in range(len(pixel_points) - 1):
            pt1 = pixel_points[i]
            pt2 = pixel_points[i + 1]
            cv2.line(frame, pt1, pt2, stroke.color, stroke.thickness, cv2.LINE_AA)


class DrawingProcessor:
    """
    Processes completed strokes for ML model prediction.
    
    Converts drawing strokes into preprocessed images suitable for
    the sketch classification model.
    """
    
    def __init__(self, model_input_size: Tuple[int, int] = (28, 28)):
        """
        Initialize the processor.
        
        Args:
            model_input_size: Expected input size for ML model (width, height)
        """
        self.model_input_size = model_input_size
        self.render_size = 256  # Internal rendering resolution
        self._lock = Lock()
    
    def process_strokes(self, strokes: List[DrawingStroke]) -> Optional[np.ndarray]:
        """
        Process strokes into a preprocessed image for the model.
        
        Args:
            strokes: List of completed strokes to process
            
        Returns:
            Preprocessed image array (model_input_size + channel dimension)
            or None if strokes are invalid
        """
        with self._lock:
            if not strokes or len(strokes) == 0:
                return None
            
            # Step 1: Render strokes to internal canvas
            canvas = self._render_strokes_to_canvas(strokes)
            
            if canvas is None:
                return None
            
            # Step 2: Find bounding box and crop
            cropped = self._crop_to_content(canvas)
            
            if cropped is None:
                return None
            
            # Step 3: Resize to model input size
            resized = cv2.resize(
                cropped, 
                self.model_input_size,
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Step 4: Normalize and prepare for model
            normalized = resized.astype(np.float32) / 255.0
            normalized = 1.0 - normalized  # Invert: white background -> black
            normalized = np.expand_dims(normalized, axis=-1)  # Add channel dimension
            
            return normalized
    
    def _render_strokes_to_canvas(self, strokes: List[DrawingStroke]) -> Optional[np.ndarray]:
        """
        Render strokes to a clean white canvas.
        
        Args:
            strokes: Strokes to render
            
        Returns:
            Grayscale canvas with strokes rendered in black
        """
        # Create white canvas
        canvas = np.ones((self.render_size, self.render_size), dtype=np.uint8) * 255
        
        drawn_anything = False
        
        for stroke in strokes:
            if len(stroke.points) < 2:
                continue
            
            # Convert normalized points to canvas coordinates
            pixel_points = [
                (
                    int(x * (self.render_size - 1)),
                    int(y * (self.render_size - 1))
                )
                for x, y in stroke.points
            ]
            
            # Draw stroke in black
            for i in range(len(pixel_points) - 1):
                pt1 = pixel_points[i]
                pt2 = pixel_points[i + 1]
                cv2.line(canvas, pt1, pt2, 0, stroke.thickness, cv2.LINE_AA)
                drawn_anything = True
        
        return canvas if drawn_anything else None
    
    def _crop_to_content(self, canvas: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop canvas to the bounding box of drawn content with padding.
        
        Args:
            canvas: Canvas with drawing
            
        Returns:
            Cropped and centered square image
        """
        # Find non-white pixels
        binary = cv2.threshold(canvas, 250, 255, cv2.THRESH_BINARY_INV)[1]
        coords = cv2.findNonZero(binary)
        
        if coords is None:
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add 12% padding
        max_dim = max(w, h)
        pad = int(max_dim * 0.12)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(canvas.shape[1] - x, w + 2 * pad)
        h = min(canvas.shape[0] - y, h + 2 * pad)
        
        # Extract ROI
        roi = canvas[y:y+h, x:x+w]
        
        # Center in square
        max_dim = max(w, h)
        square = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
        offset_x = (max_dim - w) // 2
        offset_y = (max_dim - h) // 2
        square[offset_y:offset_y+h, offset_x:offset_x+w] = roi
        
        return square


class DrawingStateMachine:
    """
    Manages the state transitions of the drawing system.
    
    States:
    - IDLE: No drawing, waiting for hand
    - DRAWING: Actively drawing with index finger
    - STROKE_COMPLETE: Hand closed, stroke saved
    - READY_FOR_PREDICTION: Strokes available for processing
    """
    
    class State(Enum):
        IDLE = "idle"
        DRAWING = "drawing"
        STROKE_COMPLETE = "stroke_complete"
        READY_FOR_PREDICTION = "ready"
    
    def __init__(self):
        """Initialize state machine."""
        self.state = self.State.IDLE
        self._lock = Lock()
    
    def on_hand_open_with_finger(self) -> bool:
        """
        Hand detected with index finger extended.
        
        Returns:
            True if should start/continue drawing
        """
        with self._lock:
            if self.state == self.State.IDLE:
                self.state = self.State.DRAWING
                return True
            elif self.state in [self.State.DRAWING, self.State.STROKE_COMPLETE]:
                self.state = self.State.DRAWING
                return True
            return False
    
    def on_hand_closed(self) -> bool:
        """
        Hand closed (fist detected).
        
        Returns:
            True if should complete stroke
        """
        with self._lock:
            if self.state == self.State.DRAWING:
                self.state = self.State.STROKE_COMPLETE
                return True
            return False
    
    def on_stroke_saved(self) -> None:
        """Stroke has been saved to canvas."""
        with self._lock:
            if self.state == self.State.STROKE_COMPLETE:
                self.state = self.State.READY_FOR_PREDICTION
    
    def on_prediction_requested(self) -> bool:
        """
        User requested prediction.
        
        Returns:
            True if ready to predict
        """
        with self._lock:
            return self.state == self.State.READY_FOR_PREDICTION
    
    def reset(self) -> None:
        """Reset to idle state."""
        with self._lock:
            self.state = self.State.IDLE
    
    def get_state(self) -> State:
        """Get current state."""
        with self._lock:
            return self.state
