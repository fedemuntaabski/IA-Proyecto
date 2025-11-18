"""
app_main.py - Main application with new drawing system

Clean architecture integrating:
- Camera capture
- Hand detection  
- Drawing system
- ML model prediction
- UI display
"""

import sys
import logging
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSlot, Qt

from camera_module import CameraCapture
from drawing_integration import DrawingController
from hand_detector import HandDetector
from model import SketchClassifier
from ui_pyqt import PictionaryUIQt
from logger_setup import setup_logging
from config import MEDIAPIPE_CONFIG, CAMERA_CONFIG, DETECTION_CONFIG, PERFORMANCE_CONFIG, MODEL_CONFIG, UI_CONFIG


class PictionaryApplication(QObject):
    """
    Main application class coordinating all components.
    
    Architecture:
    - CameraCapture: Video input
    - HandDetector: Hand tracking
    - DrawingController: Drawing + prediction logic
    - UI: Display and user interaction
    """
    
    def __init__(
        self,
        ia_dir: str = "./IA",
        camera_id: int = 0,
        debug: bool = False
    ):
        """
        Initialize the Pictionary application.
        
        Args:
            ia_dir: Path to ML model directory
            camera_id: Camera device ID
            debug: Enable debug logging
        """
        super().__init__()
        
        self.logger = setup_logging(debug=debug)
        self.ia_dir = ia_dir
        self.debug = debug
        
        # Initialize Qt Application
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Pictionary Live - New Drawing System")
        
        # Load model labels
        self.labels = self._load_labels()
        
        # Initialize components
        self.logger.info("Initializing components...")
        self.hand_detector = self._init_hand_detector()
        self.model = self._init_model()
        self.camera = self._init_camera(camera_id)
        self.ui = PictionaryUIQt(UI_CONFIG)
        
        # Initialize drawing controller
        self.drawing_controller = DrawingController(
            hand_detector=self.hand_detector,
            model=self.model,
            canvas_width=CAMERA_CONFIG["width"],
            canvas_height=CAMERA_CONFIG["height"],
            logger=self.logger
        )
        
        # Connect all components
        self._connect_signals()
        
        # Set initial target
        self._select_random_target()
        
        self.logger.info("Application initialized successfully")
    
    def _load_labels(self):
        """Load class labels from model_info.json."""
        try:
            import json
            from pathlib import Path
            
            model_info_path = Path(self.ia_dir) / "model_info.json"
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            labels = model_info.get("classes", [])
            self.logger.info(f"Loaded {len(labels)} class labels")
            return labels
        
        except Exception as e:
            self.logger.error(f"Error loading labels: {e}")
            return ["cat", "dog", "house", "tree", "car"]  # Fallback
    
    def _init_hand_detector(self) -> Optional[HandDetector]:
        """Initialize MediaPipe hand detector."""
        try:
            hand_config = {
                **MEDIAPIPE_CONFIG["hands"],
                **DETECTION_CONFIG,
                **PERFORMANCE_CONFIG
            }
            detector = HandDetector(hand_config, self.logger)
            self.logger.info("Hand detector initialized")
            return detector
        except Exception as e:
            self.logger.error(f"Failed to initialize hand detector: {e}")
            return None
    
    def _init_model(self) -> Optional[SketchClassifier]:
        """Initialize ML model."""
        try:
            model = SketchClassifier(
                self.ia_dir,
                self.logger,
                demo_mode=False,
                config=MODEL_CONFIG
            )
            
            # Load model
            if model.load_model():
                self.logger.info("ML model loaded successfully")
                return model
            else:
                self.logger.warning("Model loading failed, running without prediction")
                return None
        
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            return None
    
    def _init_camera(self, camera_id: int) -> CameraCapture:
        """Initialize camera capture."""
        camera = CameraCapture(
            camera_id=camera_id,
            width=CAMERA_CONFIG["width"],
            height=CAMERA_CONFIG["height"],
            fps=CAMERA_CONFIG["fps"],
            logger=self.logger
        )
        return camera
    
    def _connect_signals(self) -> None:
        """Connect all Qt signals between components."""
        
        # Camera -> Frame processing (use QueuedConnection for thread safety)
        self.camera.frame_ready.connect(
            self._process_frame,
            Qt.ConnectionType.QueuedConnection
        )
        
        self.camera.camera_ready.connect(
            self._on_camera_ready,
            Qt.ConnectionType.QueuedConnection
        )
        
        self.camera.camera_error.connect(
            self._on_camera_error,
            Qt.ConnectionType.QueuedConnection
        )
        
        # Drawing controller -> UI
        self.drawing_controller.prediction_ready.connect(
            self._on_prediction,
            Qt.ConnectionType.QueuedConnection
        )
        
        self.drawing_controller.stroke_completed.connect(
            self._on_stroke_completed,
            Qt.ConnectionType.QueuedConnection
        )
        
        # UI -> Actions
        self.ui.clear_requested.connect(self.drawing_controller.clear_drawing)
        self.ui.set_select_new_target_func(self._select_random_target)
        
        self.logger.info("Signals connected")
    
    @pyqtSlot(object)  # np.ndarray
    def _process_frame(self, frame) -> None:
        """
        Process video frame with hand detection and drawing.
        
        Args:
            frame: BGR video frame from camera
        """
        try:
            # Process frame with drawing system
            processed_frame = self.drawing_controller.process_hand_frame(frame)
            
            # Update UI
            self.ui.update_frame(processed_frame)
        
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}", exc_info=True)
    
    @pyqtSlot()
    def _on_camera_ready(self) -> None:
        """Handle camera ready signal."""
        self.ui.set_state("✋ DRAWING MODE", "#64ff64")
        self.logger.info("Camera ready - drawing mode active")
    
    @pyqtSlot(str)
    def _on_camera_error(self, error: str) -> None:
        """Handle camera errors."""
        self.logger.error(f"Camera error: {error}")
        self.ui.set_state(f"❌ Error: {error}", "#ff6400")
    
    @pyqtSlot(str, float, list)
    def _on_prediction(self, label: str, confidence: float, top3: list) -> None:
        """
        Handle prediction results.
        
        Args:
            label: Predicted class label
            confidence: Confidence score
            top3: Top 3 predictions
        """
        self.ui.update_prediction(label, confidence, top3)
        
        # Check if matches target
        if self.ui.current_target and label.lower() == self.ui.current_target.lower():
            self.logger.info(f"🎯 Correct! Predicted: {label}")
            # Could add score increment here
    
    @pyqtSlot()
    def _on_stroke_completed(self) -> None:
        """Handle stroke completion (hand closed)."""
        stroke_count = self.drawing_controller.get_stroke_count()
        self.logger.info(f"Stroke completed! Total: {stroke_count}")
        # Could add UI feedback here
    
    def _select_random_target(self) -> None:
        """Select a random target word."""
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
            self.ui.set_state("🔄 STARTING CAMERA...", "#ffa000")
            self.ui.reset_timer()
            
            # Start camera
            self.camera.start()
            
            # Run Qt event loop
            exit_code = self.app.exec()
            self.logger.info(f"Application exited with code: {exit_code}")
            
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            exit_code = 1
        
        finally:
            self.logger.info("Cleaning up...")
            self.camera.stop()
            self.logger.info("Cleanup complete")
        
        return exit_code
