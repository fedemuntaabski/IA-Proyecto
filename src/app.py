"""
app.py - Main Application (Clean Architecture)

Single application class integrating:
- Camera capture
- Hand detection
- Simple drawing system
- ML model prediction
- UI display
"""

import sys
import logging
import json
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSlot, Qt

from camera_module import CameraCapture
from simple_drawing import SimpleDrawing
from hand_detector import HandDetector
from model import SketchClassifier
from ui_pyqt import PictionaryUIQt
from logger_setup import setup_logging
from config import MEDIAPIPE_CONFIG, CAMERA_CONFIG, DETECTION_CONFIG, PERFORMANCE_CONFIG, MODEL_CONFIG, UI_CONFIG


class PictionaryApp(QObject):
    """
    Main application coordinating all components.
    
    Simplified architecture following KISS principle.
    """
    
    def __init__(
        self,
        ia_dir: str = "./IA",
        camera_id: int = 0,
        debug: bool = False
    ):
        """
        Initialize application.
        
        Args:
            ia_dir: Path to ML model directory
            camera_id: Camera device ID
            debug: Enable debug logging
        """
        super().__init__()
        
        self.logger = setup_logging(debug=debug)
        self.ia_dir = ia_dir
        self.debug = debug
        
        # Qt Application
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Pictionary Live")
        
        # Load labels
        self.labels = self._load_labels()
        
        # Initialize components
        self.logger.info("Initializing components...")
        self.hand_detector = self._init_hand_detector()
        self.model = self._init_model()
        self.drawing = SimpleDrawing(CAMERA_CONFIG["width"], CAMERA_CONFIG["height"])
        self.camera = self._init_camera(camera_id)
        self.ui = PictionaryUIQt(UI_CONFIG)
        
        # Connect signals
        self._connect_signals()
        
        # State tracking
        self.was_fist_closed = False
        self.current_mode = "hand"  # Track current input mode
        
        # No prediction tracer needed
        
        # Set initial target
        self._select_random_target()
        
        self.logger.info("Application initialized")
    
    def _load_labels(self):
        """Load class labels from model_info.json."""
        try:
            model_info_path = Path(self.ia_dir) / "model_info.json"
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            labels = model_info.get("classes", [])
            self.logger.info(f"Loaded {len(labels)} class labels")
            return labels
        except Exception as e:
            self.logger.error(f"Error loading labels: {e}")
            return ["cat", "dog", "house", "tree", "car"]
    
    def _init_hand_detector(self) -> Optional[HandDetector]:
        """Initialize hand detector."""
        try:
            config = {**MEDIAPIPE_CONFIG["hands"], **DETECTION_CONFIG, **PERFORMANCE_CONFIG}
            detector = HandDetector(config, self.logger)
            self.logger.info("Hand detector initialized")
            return detector
        except Exception as e:
            self.logger.error(f"Hand detector init failed: {e}")
            return None
    
    def _init_model(self) -> Optional[SketchClassifier]:
        """Initialize ML model."""
        try:
            model = SketchClassifier(self.ia_dir, self.logger, demo_mode=False, config=MODEL_CONFIG)
            if model.load_model():
                self.logger.info("ML model loaded")
                return model
            return None
        except Exception as e:
            self.logger.error(f"Model init failed: {e}")
            return None
    
    def _init_camera(self, camera_id: int) -> CameraCapture:
        """Initialize camera."""
        return CameraCapture(
            camera_id=camera_id,
            width=CAMERA_CONFIG["width"],
            height=CAMERA_CONFIG["height"],
            fps=CAMERA_CONFIG["fps"],
            logger=self.logger
        )
    
    def _connect_signals(self) -> None:
        """Connect Qt signals with appropriate thread safety."""
        # Camera signals - QueuedConnection for thread safety
        queued = Qt.ConnectionType.QueuedConnection
        signals = [
            (self.camera.frame_ready, self._process_frame, queued),
            (self.camera.camera_ready, self._on_camera_ready, queued),
            (self.camera.camera_error, self._on_camera_error, queued),
            (self.ui.clear_requested, self._clear_drawing, None),
            (self.ui.mode_switched, self._on_mode_switched, None),
            (self.ui.video_widget.mouse_draw, self._handle_mouse_draw, None),
        ]
        
        for signal, slot, connection_type in signals:
            if connection_type:
                signal.connect(slot, connection_type)
            else:
                signal.connect(slot)
        
        self.ui.set_select_new_target_func(self._select_random_target)
        self.logger.info("Signals connected")
    
    @pyqtSlot(object)
    def _process_frame(self, frame) -> None:
        """
        Process video frame with hand detection and drawing.
        
        Args:
            frame: BGR video frame from camera
        """
        try:
            # Don't process if game is paused
            if self.ui.game_paused:
                self.ui.update_frame(frame)
                return
            
            finger_pos = None
            
            # Only process hand input in hand mode
            if self.current_mode == "hand":
                # Detect hand
                detection = self.hand_detector.detect(frame)
                hand_landmarks = detection.get("hand_landmarks")
                
                if hand_landmarks:
                    # Draw hand landmarks
                    frame = self.hand_detector.draw_hand_landmarks(frame)
                    
                    # Get finger position
                    finger_pos = self.hand_detector.get_index_finger_position()
                    
                    # Check if fist
                    is_fist = self._check_fist_safely()
                    
                    if finger_pos and not is_fist:
                        # Drawing: add point
                        self.drawing.add_point(finger_pos[0], finger_pos[1])
                        self.was_fist_closed = False
                        
                    elif is_fist and not self.was_fist_closed:
                        # Fist closed: save stroke
                        if self.drawing.save_current_stroke():
                            count = self.drawing.get_stroke_count()
                            self.logger.info(f"Stroke saved! Total: {count}")
                            
                            # Auto-predict
                            self._predict_drawing()
                        
                        self.was_fist_closed = True
                        
                    elif not is_fist and self.was_fist_closed:
                        # Fist opened: reset flag
                        self.was_fist_closed = False
            
            # Render drawing on frame
            frame = self.drawing.render_on_frame(frame, finger_pos)
            
            # Update UI
            self.ui.update_frame(frame)
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}", exc_info=self.debug)
            # Continue processing - don't crash on single frame error
    
    def _check_fist_safely(self) -> bool:
        """Safely check if fist is closed."""
        try:
            return self.hand_detector.is_fist()
        except Exception:
            return False
    
    def _predict_drawing(self) -> None:
        """Run prediction on current drawing."""
        try:
            if not self.model or not self.drawing.has_drawing():
                return
            
            # Get preprocessed image
            img = self.drawing.get_preprocessed_image()
            if img is None:
                return
            
            # Predict
            label, confidence, top3 = self.model.predict(img)
            self.logger.info(f"Prediction: {label} ({confidence:.2%})")
            
            # Update UI
            self.ui.update_prediction(label, confidence, top3)
            
            # Check if correct
            if self.ui.current_target and label.lower() == self.ui.current_target.lower():
                self.logger.info(f"🎯 Correct! {label}")
                
        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
    
    @pyqtSlot(bool)
    def _on_mode_switched(self, use_hand: bool) -> None:
        """Handle mode switch between hand and mouse."""
        self.current_mode = "hand" if use_hand else "mouse"
        self.logger.info(f"Mode switched to: {self.current_mode}")
        # Clear drawing when switching modes
        self._clear_drawing()
    
    @pyqtSlot()
    def _clear_drawing(self) -> None:
        """Clear all drawings."""
        self.drawing.clear_all()
        self.logger.info("Drawing cleared")
    
    @pyqtSlot(float, float, bool)
    def _handle_mouse_draw(self, x: float, y: float, is_drawing: bool) -> None:
        """Handle mouse drawing input (only in mouse mode)."""
        # Don't process if game is paused
        if self.ui.game_paused:
            return
            
        # Only process mouse input in mouse mode
        if self.current_mode != "mouse":
            return
            
        if is_drawing:
            # Add point while drawing
            self.drawing.add_point(x, y)
        else:
            # Mouse released - save stroke and predict
            if self.drawing.save_current_stroke():
                count = self.drawing.get_stroke_count()
                self.logger.info(f"Mouse stroke saved! Total: {count}")
                self._predict_drawing()
    
    @pyqtSlot()
    def _on_camera_ready(self) -> None:
        """Camera ready callback."""
        self.ui.set_state("✋ DRAWING MODE", "#64ff64")
        self.logger.info("Camera ready")
    
    @pyqtSlot(str)
    def _on_camera_error(self, error: str) -> None:
        """Camera error callback."""
        self.logger.error(f"Camera error: {error}")
        self.ui.set_state(f"❌ Error: {error}", "#ff6400")
    
    def _select_random_target(self) -> None:
        """Select random target word."""
        import random
        if self.labels:
            target = random.choice(self.labels)
            self.ui.set_target(target)
            self.logger.info(f"New target: {target}")
    
    def run(self) -> int:
        """
        Run the application.
        
        Returns:
            Exit code
        """
        try:
            self.logger.info("Starting application...")
            
            # Show UI
            self.ui.show()
            self.ui.activateWindow()
            self.ui.raise_()
            self.ui.set_state("🔄 STARTING...", "#ffa000")
            self.ui.reset_timer()
            
            # Start camera
            self.camera.start()
            
            # Run Qt event loop
            exit_code = self.app.exec()
            self.logger.info(f"Application exited: {exit_code}")
            
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            exit_code = 1
        
        finally:
            self.logger.info("Cleanup...")
            self.camera.stop()
        
        return exit_code
