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
        
        # Prediction tracer
        from prediction_tracer import PredictionTracerDialog
        self.tracer_dialog = PredictionTracerDialog()
        self.trace_strokes = []  # Current trace strokes being animated
        self.trace_label = ""
        self.trace_progress = 0.0
        
        # Set initial target
        self._select_random_target()
        
        # Pass labels to UI for examples viewer
        self.ui.set_available_labels(self.labels)
        
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
            (self.ui.trace_prediction_requested, self._trace_prediction, None),
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
            
            # Render trace animation if active
            if self.tracer_dialog.is_active and self.trace_strokes:
                frame = self._render_trace_on_frame(frame)
            
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
    
    @pyqtSlot()
    def _trace_prediction(self) -> None:
        """Run slow-motion prediction drawing visualization."""
        try:
            if not self.model or not self.drawing.has_drawing():
                self.logger.warning("No drawing to trace")
                return
            
            # Get preprocessed image
            img = self.drawing.get_preprocessed_image()
            if img is None:
                self.logger.warning("Could not preprocess image")
                return
            
            # Predict
            label, confidence, top3 = self.model.predict(img)
            self.logger.info(f"Tracing prediction: {label} ({confidence:.2%})")
            
            # Start trace animation - draw the predicted label
            self.tracer_dialog.start_trace(
                label, 
                confidence * 100,  # Convert to percentage
                self.camera.width,
                self.camera.height,
                self._on_trace_drawing_update
            )
            
        except Exception as e:
            self.logger.error(f"Trace prediction error: {e}", exc_info=True)
    
    def _on_trace_drawing_update(self, strokes, label, progress):
        """Handle trace animation drawing update."""
        # Store trace data for rendering on next frame
        self.trace_strokes = strokes
        self.trace_label = label
        self.trace_progress = progress
    
    def _render_trace_on_frame(self, frame):
        """
        Render trace animation strokes on frame.
        
        Args:
            frame: Video frame to draw on
            
        Returns:
            Frame with trace overlay
        """
        import cv2
        
        # Draw each stroke in cyan (bright and visible)
        for stroke in self.trace_strokes:
            if len(stroke) < 2:
                continue
            
            # Draw lines connecting points
            for i in range(len(stroke) - 1):
                pt1 = stroke[i]
                pt2 = stroke[i + 1]
                cv2.line(frame, pt1, pt2, (255, 200, 0), 4)  # Cyan thick line
        
        # Add info overlay
        h, w = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 90), (20, 30, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Drawing Prediction Tutorial", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        # Label
        text = f"Drawing: {self.trace_label.upper()}"
        cv2.putText(frame, text, (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Progress bar
        bar_width = w - 40
        bar_height = 8
        bar_x, bar_y = 20, 75
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(bar_width * self.trace_progress), bar_y + bar_height),
                     (100, 200, 255), -1)
        
        return frame
    
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
            self.tracer_dialog.stop_trace()
            self.camera.stop()
        
        return exit_code
