"""
app_integration.py - Clean integration layer for Pictionary Live

Connects camera, drawing canvas, ML model, and UI following KISS/DRY principles.
"""

import sys
import logging
from typing import Optional, List, Tuple
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import pyqtSlot, QObject

from camera_handler import CameraHandler
from drawing_canvas import DrawingCanvas
from hand_detector import HandDetector
from model import SketchClassifier
from ui_pyqt import PictionaryUIQt
from logger_setup import setup_logging
from config import MEDIAPIPE_CONFIG, CAMERA_CONFIG, DETECTION_CONFIG, PERFORMANCE_CONFIG, MODEL_CONFIG, UI_CONFIG


class PictionaryApp(QObject):
    """
    Main application class - integrates all components.
    Follows KISS principle with clear separation of concerns.
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
            ia_dir: Path to model directory
            camera_id: Camera device ID
            debug: Enable debug logging
        """
        super().__init__()
        
        self.logger = setup_logging(debug=debug)
        self.ia_dir = ia_dir
        self.debug = debug
        
        # Initialize Qt Application
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Pictionary Live")
        
        # Load model labels
        self.labels = self._load_labels()
        
        # Initialize components
        self.camera = self._init_camera(camera_id)
        self.canvas = DrawingCanvas(size=256)
        self.hand_detector = self._init_hand_detector()
        self.classifier = self._init_classifier()
        self.ui = PictionaryUIQt(UI_CONFIG)
        
        # Connect components
        self._connect_components()
        
        # Set initial target
        self._select_random_target()
        
        self.logger.info("Application initialized")
    
    def _load_labels(self) -> List[str]:
        """Load class labels from model_info.json."""
        try:
            import json
            from pathlib import Path
            
            model_info_path = Path(self.ia_dir) / "model_info.json"
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            labels = model_info.get("classes", [])
            self.logger.info(f"Loaded {len(labels)} labels")
            return labels
        
        except Exception as e:
            self.logger.error(f"Error loading labels: {e}")
            return ["cat", "dog", "house", "tree", "car"]  # Fallback
    
    def _init_camera(self, camera_id: int) -> CameraHandler:
        """Initialize camera handler."""
        camera = CameraHandler(
            camera_id=camera_id,
            width=CAMERA_CONFIG["width"],
            height=CAMERA_CONFIG["height"],
            fps=CAMERA_CONFIG["fps"],
            logger=self.logger
        )
        return camera
    
    def _init_hand_detector(self) -> Optional[HandDetector]:
        """Initialize hand detector."""
        try:
            hand_config = {
                **MEDIAPIPE_CONFIG["hands"],
                **DETECTION_CONFIG,
                **PERFORMANCE_CONFIG
            }
            detector = HandDetector(hand_config, self.logger)
            return detector
        except Exception as e:
            self.logger.error(f"Failed to initialize hand detector: {e}")
            return None
    
    def _init_classifier(self) -> Optional[SketchClassifier]:
        """Initialize ML classifier."""
        try:
            classifier = SketchClassifier(
                self.ia_dir,
                self.logger,
                demo_mode=False,
                config=MODEL_CONFIG
            )
            
            # Force model loading
            if classifier.load_model():
                self.logger.info("Model loaded successfully")
                return classifier
            else:
                self.logger.error("Failed to load model")
                return None
        
        except Exception as e:
            self.logger.error(f"Failed to initialize classifier: {e}")
            return None
    
    def _connect_components(self):
        """Connect all component signals and callbacks."""
        # Inject hand detector into camera
        if self.hand_detector:
            self.camera.set_hand_detector(self.hand_detector)
        
        # Camera -> UI
        self.camera.frame_ready.connect(self.ui.update_frame)
        self.camera.hand_detected.connect(self.ui.update_hand_detected)
        self.camera.error_occurred.connect(self._handle_error)
        self.camera.camera_ready.connect(self._on_camera_ready)
        
        # Camera -> Canvas (drawing points)
        self.camera.drawing_point.connect(self._on_drawing_point)
        
        # UI -> Camera (mouse input)
        self.ui.video_widget.mouse_draw.connect(self.camera.process_mouse_point)
        
        # UI -> Actions
        self.ui.clear_requested.connect(self._clear_all)
        self.ui.mode_switched.connect(self._switch_mode)
        self.ui.set_select_new_target_func(self._select_random_target)
    
    @pyqtSlot()
    def _on_camera_ready(self):
        """Handle camera ready event."""
        self.ui.set_state("✋ HAND MODE", "#64ff64")
        self.logger.info("Camera ready, switching to hand mode")
    
    @pyqtSlot(float, float, bool)
    def _on_drawing_point(self, x: float, y: float, is_drawing: bool):
        """
        Handle drawing point from camera or mouse.
        
        Args:
            x: Normalized x coordinate
            y: Normalized y coordinate
            is_drawing: Whether actively drawing
        """
        # Add to canvas
        self.canvas.add_point(x, y, is_drawing)
        
        # Predict if stroke ended and canvas has content
        if not is_drawing and not self.canvas.is_empty():
            self._predict_drawing()
    
    def _predict_drawing(self):
        """Run prediction on current canvas."""
        if not self.classifier:
            return
        
        # Get preprocessed image
        img = self.canvas.get_preprocessed_for_model()
        if img is None:
            return
        
        try:
            # Predict
            label, confidence, top3 = self.classifier.predict(img)
            
            # Update UI
            self.ui.update_prediction(label, confidence, top3)
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
    
    @pyqtSlot()
    def _clear_all(self):
        """Clear all drawing state."""
        self.canvas.clear()
        self.camera.clear_overlay()
        self.logger.info("Canvas cleared")
    
    @pyqtSlot(bool)
    def _switch_mode(self, use_hand: bool):
        """
        Switch between hand and mouse modes.
        
        Args:
            use_hand: True for hand mode, False for mouse mode
        """
        mode = "hand" if use_hand else "mouse"
        self.camera.set_mode(mode)
        self._clear_all()
    
    def _select_random_target(self):
        """Select random target from labels."""
        import random
        
        if self.labels:
            target = random.choice(self.labels)
            self.ui.set_target(target)
            self.logger.info(f"New target: {target}")
    
    def _handle_error(self, error: str):
        """Handle errors from components."""
        self.logger.error(f"Component error: {error}")
        if "Error" in error or "error" in error:
            self.ui.set_state(f"Error: {error}", "#ff6400")
    
    def run(self) -> int:
        """
        Run the application.
        
        Returns:
            Exit code
        """
        try:
            self.logger.info("Setting up UI...")
            # Setup UI
            self.ui.show()
            self.ui.activateWindow()
            self.ui.raise_()
            self.ui.set_state("🔄 INICIANDO CÁMARA...", "#ffa000")
            self.ui.reset_timer()
            self.logger.info("UI setup complete")
            
            self.logger.info("Starting camera...")
            # Start camera
            self.camera.start()
            self.logger.info("Camera started")
            
            self.logger.info("Starting Qt event loop...")
            # Run Qt event loop
            exit_code = self.app.exec()
            self.logger.info(f"Qt event loop exited with code: {exit_code}")
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            exit_code = 1
        
        finally:
            self.logger.info("Cleaning up...")
            # Cleanup
            self.camera.stop()
            self.logger.info("Cleanup complete")
        
        return exit_code
