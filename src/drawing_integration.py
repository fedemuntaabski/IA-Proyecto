"""
drawing_integration.py - Integration layer for drawing system

Connects the drawing system with camera, hand detection, and ML model.
Handles the complete flow from hand tracking to prediction.
"""

import logging
import numpy as np
from typing import Optional, Tuple, List
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMutex, Qt

from drawing_system import DrawingCanvas, DrawingProcessor, DrawingStateMachine, DrawingStroke
from hand_detector import HandDetector
from model import SketchClassifier


class DrawingController(QObject):
    """
    Central controller for the drawing and prediction system.
    
    Coordinates between:
    - Hand detection (input)
    - Drawing canvas (visualization)
    - Drawing processor (preprocessing)
    - ML model (prediction)
    - UI (output)
    
    Thread-safe and follows single responsibility principle.
    """
    
    # Signals for UI updates
    stroke_completed = pyqtSignal()  # Emitted when user closes hand
    prediction_ready = pyqtSignal(str, float, list)  # label, confidence, top3
    canvas_cleared = pyqtSignal()
    
    def __init__(
        self,
        hand_detector: HandDetector,
        model: Optional[SketchClassifier],
        canvas_width: int = 640,
        canvas_height: int = 480,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the drawing controller.
        
        Args:
            hand_detector: Hand detection instance
            model: ML model for prediction
            canvas_width: Drawing canvas width
            canvas_height: Drawing canvas height
            logger: Logger instance
        """
        super().__init__()
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Core components
        self.hand_detector = hand_detector
        self.model = model
        self.canvas = DrawingCanvas(canvas_width, canvas_height)
        self.processor = DrawingProcessor()
        self.state_machine = DrawingStateMachine()
        
        # Thread safety for prediction
        self.prediction_mutex = QMutex()
        self.is_predicting = False
        
        # State tracking
        self.last_finger_pos: Optional[Tuple[float, float]] = None
        self.was_fist_closed = False
    
    def process_hand_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a video frame with hand detection and drawing.
        
        Args:
            frame: BGR video frame from camera
            
        Returns:
            Frame with drawing overlays and hand landmarks
        """
        try:
            # Detect hand
            detection = self.hand_detector.detect(frame)
            hand_landmarks = detection.get("hand_landmarks")
            
            if hand_landmarks:
                # Draw hand landmarks on frame
                frame = self.hand_detector.draw_hand_landmarks(frame)
                
                # Get index finger position
                finger_pos = self.hand_detector.get_index_finger_position()
                
                # Check if hand is closed (fist)
                is_fist = self._check_fist_safely()
                
                if finger_pos and not is_fist:
                    # Drawing mode: index finger extended
                    self._handle_drawing(finger_pos)
                    self.was_fist_closed = False
                elif is_fist and not self.was_fist_closed:
                    # Fist closed: complete stroke
                    self._handle_fist_close()
                    self.was_fist_closed = True
                elif not is_fist and self.was_fist_closed:
                    # Fist opened: reset flag
                    self.was_fist_closed = False
            else:
                # No hand detected: reset state
                self.last_finger_pos = None
            
            # Render drawing on frame
            frame = self.canvas.render_on_frame(frame)
            
            return frame
        
        except Exception as e:
            self.logger.error(f"Error processing hand frame: {e}", exc_info=True)
            return frame
    
    def _check_fist_safely(self) -> bool:
        """
        Safely check if hand is closed.
        
        Returns:
            True if fist is closed, False otherwise
        """
        try:
            return self.hand_detector.is_fist()
        except Exception as e:
            self.logger.warning(f"Error checking fist: {e}")
            return False
    
    def _handle_drawing(self, finger_pos: Tuple[float, float]) -> None:
        """
        Handle drawing with index finger.
        
        Args:
            finger_pos: Normalized (x, y) position of index finger
        """
        x, y = finger_pos
        
        # Add point to canvas
        self.canvas.add_point(x, y)
        
        # Update state machine
        self.state_machine.on_hand_open_with_finger()
        
        self.last_finger_pos = finger_pos
    
    def _handle_fist_close(self) -> None:
        """Handle hand closing (fist detected)."""
        # Complete current stroke
        stroke_saved = self.canvas.complete_current_stroke()
        
        if stroke_saved:
            self.logger.info(f"Stroke completed! Total strokes: {self.canvas.get_stroke_count()}")
            
            # Update state machine
            if self.state_machine.on_hand_closed():
                self.state_machine.on_stroke_saved()
            
            # Emit signal for UI feedback
            self.stroke_completed.emit()
            
            # Auto-predict if enabled
            self._auto_predict()
        else:
            self.logger.debug("Stroke discarded (too few points)")
    
    def _auto_predict(self) -> None:
        """Automatically run prediction after stroke completion."""
        if self.canvas.has_strokes():
            self.request_prediction()
    
    @pyqtSlot()
    def request_prediction(self) -> None:
        """
        Request prediction on current drawing.
        Thread-safe with mutex protection.
        """
        try:
            if not self.model:
                self.logger.warning("No model available for prediction")
                return
            
            if not self.canvas.has_strokes():
                self.logger.debug("No strokes to predict")
                return
            
            # Prevent concurrent predictions (TensorFlow not thread-safe!)
            if not self.prediction_mutex.tryLock():
                self.logger.warning("Prediction already in progress")
                return
            
            try:
                if self.is_predicting:
                    return
                
                self.is_predicting = True
                
                # Get all strokes
                with self.canvas._lock:
                    strokes = self.canvas.strokes.copy()
                
                # Process strokes to image
                processed_img = self.processor.process_strokes(strokes)
                
                if processed_img is None:
                    self.logger.warning("Failed to process strokes")
                    return
                
                # Run prediction
                label, confidence, top3 = self.model.predict(processed_img)
                
                self.logger.info(f"Prediction: {label} ({confidence:.2%})")
                
                # Emit results
                self.prediction_ready.emit(label, confidence, top3)
                
            finally:
                self.is_predicting = False
                self.prediction_mutex.unlock()
        
        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            if self.prediction_mutex.tryLock():
                self.prediction_mutex.unlock()
            self.is_predicting = False
    
    @pyqtSlot()
    def clear_drawing(self) -> None:
        """Clear all strokes and reset state."""
        self.canvas.clear_all()
        self.state_machine.reset()
        self.last_finger_pos = None
        self.was_fist_closed = False
        self.canvas_cleared.emit()
        self.logger.info("Drawing cleared")
    
    def get_stroke_count(self) -> int:
        """Get number of completed strokes."""
        return self.canvas.get_stroke_count()
    
    def has_drawing(self) -> bool:
        """Check if there's any drawing on canvas."""
        return self.canvas.has_strokes()
