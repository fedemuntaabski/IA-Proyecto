"""
camera_handler.py - Camera capture and processing handler

Clean implementation following KISS and DRY principles.
Handles both hand detection and mouse input modes.
"""

import cv2
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from PyQt6.QtCore import QThread, pyqtSignal


class CameraHandler(QThread):
    """
    Handles camera capture and frame processing.
    Supports both hand tracking and mouse modes.
    """
    
    # Signals
    frame_ready = pyqtSignal(np.ndarray)  # Emits processed frame
    hand_detected = pyqtSignal(bool)  # Hand presence status
    drawing_point = pyqtSignal(float, float, bool)  # x, y, is_drawing (normalized 0-1)
    error_occurred = pyqtSignal(str)  # Error messages
    camera_ready = pyqtSignal()  # Emitted when camera is successfully opened
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize camera handler.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Target FPS
            logger: Logger instance
        """
        super().__init__()
        
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = logger or logging.getLogger(__name__)
        
        # Camera state
        self.cap = None
        self.running = False
        self.mode = "hand"  # "hand" or "mouse"
        
        # Hand detector callback (injected)
        self.hand_detector = None
        
        # Drawing overlay (RGBA with alpha channel)
        self.overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Drawing state
        self.last_point = None
        self.is_fist = False
    
    def set_hand_detector(self, detector):
        """
        Inject hand detector instance.
        
        Args:
            detector: HandDetector instance with detect() method
        """
        self.hand_detector = detector
    
    def set_mode(self, mode: str):
        """
        Switch between hand and mouse modes.
        
        Args:
            mode: Either "hand" or "mouse"
        """
        if mode not in ["hand", "mouse"]:
            self.logger.warning(f"Invalid mode: {mode}. Using 'hand'")
            mode = "hand"
        
        self.mode = mode
        self.clear_overlay()
        self.logger.info(f"Mode switched to: {mode.upper()}")
    
    def clear_overlay(self):
        """Clear the drawing overlay."""
        self.overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.last_point = None
    
    def run(self):
        """Main capture loop (runs in thread)."""
        self.logger.info("Camera thread run() method called")
        self.running = True
        
        # Try different camera backends for Windows compatibility
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Any")
        ]
        
        self.cap = None
        for backend, name in backends:
            self.logger.info(f"Trying camera backend: {name}")
            self.cap = cv2.VideoCapture(self.camera_id, backend)
            if self.cap.isOpened():
                self.logger.info(f"Camera opened successfully with {name} backend")
                break
            else:
                self.logger.warning(f"Failed to open camera with {name} backend")
                if self.cap:
                    self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            self.error_occurred.emit("Failed to open camera with any backend")
            self.logger.error("Camera not available with any backend")
            self.running = False
            return
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify camera properties
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
        self.camera_ready.emit()
        
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                self.logger.warning(f"Failed to grab frame (failure {consecutive_failures}/{max_consecutive_failures})")
                if consecutive_failures >= max_consecutive_failures:
                    self.error_occurred.emit("Camera lost connection - too many frame failures")
                    break
                continue
            else:
                consecutive_failures = 0
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                self.logger.info(f"Processed {frame_count} frames")
            
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Process based on mode
            if self.mode == "hand":
                frame = self._process_hand_mode(frame)
            
            # Apply drawing overlay
            frame = self._apply_overlay(frame)
            
            # Emit frame
            self.frame_ready.emit(frame)
        
        # Cleanup
        if self.cap:
            self.cap.release()
        self.logger.info("Camera closed")
    
    def _process_hand_mode(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame with hand detection.
        
        Args:
            frame: Input BGR frame
        
        Returns:
            Frame with hand landmarks drawn
        """
        if not self.hand_detector:
            return frame
        
        try:
            # Detect hand
            detection = self.hand_detector.detect(frame)
            hand_landmarks = detection.get("hand_landmarks")
            
            # Emit detection status
            self.hand_detected.emit(hand_landmarks is not None)
            
            if hand_landmarks:
                # Draw landmarks on frame
                frame = self.hand_detector.draw_hand_landmarks(frame)
                
                # Get index finger position (normalized 0-1)
                index_pos = self.hand_detector.get_index_finger_position()
                
                # Check if fist is closed
                is_fist = self.hand_detector.is_fist() if hasattr(self.hand_detector, 'is_fist') else False
                
                if index_pos:
                    self._handle_drawing_point(index_pos, is_fist)
            else:
                # No hand detected - reset state
                self._reset_drawing_state()
        
        except Exception as e:
            self.logger.warning(f"Hand detection error: {e}")
        
        return frame
    
    def _handle_drawing_point(self, pos: Tuple[float, float], is_fist: bool):
        """
        Handle a drawing point from hand or mouse.
        
        Args:
            pos: Normalized position (x, y) in range [0, 1]
            is_fist: Whether hand is closed (ends stroke)
        """
        x, y = pos
        
        # Convert to pixel coordinates
        px = int(x * self.width)
        py = int(y * self.height)
        current_point = (px, py)
        
        if is_fist:
            # Fist closed - end current stroke
            self.drawing_point.emit(x, y, False)
            self._reset_drawing_state()
        else:
            # Drawing - emit point and draw line
            self.drawing_point.emit(x, y, True)
            
            if self.last_point is not None:
                # Draw line on overlay
                cv2.line(
                    self.overlay,
                    self.last_point,
                    current_point,
                    (0, 255, 0, 255),  # Green with full opacity
                    8,
                    cv2.LINE_AA
                )
            
            self.last_point = current_point
    
    def process_mouse_point(self, x: float, y: float, is_drawing: bool):
        """
        Process mouse input point.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            is_drawing: Whether mouse button is pressed
        """
        if self.mode != "mouse":
            return
        
        if not is_drawing:
            # Mouse released - end stroke
            self.drawing_point.emit(0.0, 0.0, False)
            self._reset_drawing_state()
        else:
            # Mouse pressed - draw
            px = int(x * self.width)
            py = int(y * self.height)
            current_point = (px, py)
            
            self.drawing_point.emit(x, y, True)
            
            if self.last_point is not None:
                cv2.line(
                    self.overlay,
                    self.last_point,
                    current_point,
                    (0, 255, 0, 255),
                    8,
                    cv2.LINE_AA
                )
            
            self.last_point = current_point
    
    def _reset_drawing_state(self):
        """Reset drawing state."""
        self.last_point = None
        self.is_fist = False
    
    def _apply_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply drawing overlay to frame.
        
        Args:
            frame: Input BGR frame
        
        Returns:
            Frame with overlay applied
        """
        # Create mask from alpha channel
        mask = self.overlay[:, :, 3] > 0
        
        # Apply overlay where mask is true
        frame[mask] = self.overlay[mask, :3]
        
        return frame
    
    def stop(self):
        """Stop the camera thread."""
        self.logger.info("Stopping camera thread...")
        self.running = False
        self.wait()  # Wait for thread to finish
        self.logger.info("Camera thread stopped")
