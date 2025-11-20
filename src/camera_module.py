"""
camera_module.py - Clean camera capture module

Simplified camera handler focused solely on video capture and frame processing.
Drawing logic is delegated to the drawing system.
"""

import cv2
import logging
import numpy as np
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal


class CameraCapture(QThread):
    """
    Handles camera capture and basic frame processing.
    
    Responsibilities:
    - Camera initialization and configuration
    - Frame capture loop
    - Frame emission to main thread
    - Error handling and recovery
    """
    
    # Signals
    frame_ready = pyqtSignal(np.ndarray)  # Emits BGR frames
    camera_ready = pyqtSignal()  # Camera successfully initialized
    camera_error = pyqtSignal(str)  # Error messages
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize camera capture.
        
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
        self.mirror_frame = True  # Flip horizontally for natural interaction
    
    def run(self) -> None:
        """Main capture loop (runs in separate thread)."""
        self.logger.info("Starting camera capture thread")
        self.running = True
        
        # Try different backends for Windows compatibility
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Any Available")
        ]
        
        # Initialize camera
        for backend, name in backends:
            self.logger.info(f"Trying {name} backend...")
            self.cap = cv2.VideoCapture(self.camera_id, backend)
            
            if self.cap.isOpened():
                self.logger.info(f"Camera opened with {name}")
                break
            else:
                if self.cap:
                    self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            error_msg = "Failed to open camera with any backend"
            self.logger.error(error_msg)
            self.camera_error.emit(error_msg)
            self.running = False
            return
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Log actual settings
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.logger.info(f"Camera: {actual_w}x{actual_h} @ {actual_fps}fps")
        
        # Signal ready
        self.camera_ready.emit()
        
        # Capture loop
        frame_count = 0
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                self.logger.warning(f"Frame capture failed ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    self.camera_error.emit("Too many frame capture failures")
                    break
                
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            frame_count += 1
            
            # Mirror frame for natural interaction
            if self.mirror_frame:
                frame = cv2.flip(frame, 1)
            
            # Crop to square (1:1 ratio) by taking center portion
            h, w = frame.shape[:2]
            if h != w:
                # Get the smaller dimension
                size = min(h, w)
                # Calculate crop offsets to center the crop
                start_y = (h - size) // 2
                start_x = (w - size) // 2
                frame = frame[start_y:start_y + size, start_x:start_x + size]
                
                # Resize to target dimensions if needed
                if size != self.width or size != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
            
            # Emit frame to main thread
            self.frame_ready.emit(frame)
            
            # Log progress periodically
            if frame_count % 300 == 0:
                self.logger.debug(f"Processed {frame_count} frames")
        
        # Cleanup
        if self.cap:
            self.cap.release()
        
        self.logger.info("Camera capture thread stopped")
    
    def stop(self) -> None:
        """Stop the camera capture thread."""
        self.logger.info("Stopping camera...")
        self.running = False
        self.wait(5000)  # Wait max 5 seconds for thread
        self._cleanup()
        self.logger.info("Camera stopped")
    
    def _cleanup(self) -> None:
        """Clean up camera resources."""
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                self.logger.warning(f"Error releasing camera: {e}")
