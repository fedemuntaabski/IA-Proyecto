"""
examples_viewer.py - Examples viewer for drawing labels

Shows example drawings for each label to help users understand what to draw.
"""

from typing import List
import math
import random
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QGridLayout, QLineEdit
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QPen, QImage


class ExamplesViewer(QDialog):
    """
    Dialog to show example drawings for each label.
    
    Helps users understand what each label looks like.
    """
    
    def __init__(self, labels: List[str], parent=None):
        """
        Initialize examples viewer.
        
        Args:
            labels: List of all available labels
            parent: Parent widget
        """
        super().__init__(parent)
        self.labels = sorted(labels)  # Sort alphabetically
        self.filtered_labels = self.labels.copy()
        
        self.setWindowTitle("Drawing Examples - Press E to close")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        self._setup_ui()
        self._apply_styles()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("Drawing Examples")
        header.setObjectName("examplesHeader")
        header_font = QFont("Segoe UI", 24, QFont.Weight.Bold)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Info text
        info = QLabel(f"Total: {len(self.labels)} categories available")
        info.setObjectName("examplesInfo")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)
        
        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setObjectName("searchLabel")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type to filter categories...")
        self.search_box.setObjectName("searchBox")
        self.search_box.textChanged.connect(self._filter_labels)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box, stretch=1)
        layout.addLayout(search_layout)
        
        # Scroll area for labels
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Container for grid
        self.container = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(15)
        self.container.setLayout(self.grid_layout)
        
        scroll.setWidget(self.container)
        layout.addWidget(scroll, stretch=1)
        
        # Footer with buttons
        footer_layout = QHBoxLayout()
        
        self.count_label = QLabel(f"Showing: {len(self.filtered_labels)}")
        self.count_label.setObjectName("countLabel")
        footer_layout.addWidget(self.count_label)
        
        footer_layout.addStretch()
        
        close_btn = QPushButton("Close (E or ESC)")
        close_btn.setObjectName("closeButton")
        close_btn.clicked.connect(self.close)
        close_btn.setMinimumSize(150, 40)
        footer_layout.addWidget(close_btn)
        
        layout.addLayout(footer_layout)
        
        self.setLayout(layout)
        
        # Populate grid
        self._populate_grid()
    
    def _populate_grid(self):
        """Populate grid with label cards."""
        # Clear existing items
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add cards in grid (4 columns)
        columns = 4
        for index, label in enumerate(self.filtered_labels):
            row = index // columns
            col = index % columns
            
            card = self._create_label_card(label)
            self.grid_layout.addWidget(card, row, col)
        
        # Update count
        self.count_label.setText(f"Showing: {len(self.filtered_labels)} of {len(self.labels)}")
    
    def _create_label_card(self, label: str) -> QWidget:
        """
        Create a card widget for a label.
        
        Args:
            label: Label name
            
        Returns:
            Card widget
        """
        card = QWidget()
        card.setObjectName("labelCard")
        card.setMinimumSize(200, 180)
        card.setMaximumSize(300, 220)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Example drawing
        example_widget = self._create_example_sketch(label)
        layout.addWidget(example_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Label name
        name_label = QLabel(label.upper())
        name_label.setObjectName("labelName")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(True)
        name_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        name_label.setFont(name_font)
        layout.addWidget(name_label)
        
        card.setLayout(layout)
        return card
    
    def _create_example_sketch(self, label: str) -> QLabel:
        """
        Create sketch-like example drawing that mimics how the neural network sees it.
        
        The model processes drawings as 28x28 grayscale images with:
        - White background (0)
        - Black lines (1)  
        - Centered and normalized
        
        Args:
            label: Label name
            
        Returns:
            Label widget with sketch matching model's view
        """
        # Render size matches actual model input (28x28) scaled up for visibility
        model_size = 28
        display_size = 120  # Scale up for display
        
        # Create canvas at model size (28x28)
        canvas = np.ones((model_size, model_size), dtype=np.uint8) * 255  # White background
        
        # Consistent patterns based on label
        random.seed(hash(label))
        cx, cy = model_size // 2, model_size // 2
        
        # Draw category-specific sketches AT MODEL SCALE
        label_lower = label.lower()
        
        # Scale factor for coordinates (everything relative to 28x28)
        s = model_size / 28.0  # Should be 1.0, but keeps code clear
        
        # Draw simplified shapes that work at 28x28 resolution
        # Animals
        if 'cat' in label_lower:
            self._draw_cat_simple(canvas, cx, cy, s)
        elif 'dog' in label_lower:
            self._draw_dog_simple(canvas, cx, cy, s)
        elif 'bird' in label_lower:
            self._draw_bird_simple(canvas, cx, cy, s)
        elif 'fish' in label_lower:
            self._draw_fish_simple(canvas, cx, cy, s)
        elif 'elephant' in label_lower:
            self._draw_elephant_simple(canvas, cx, cy, s)
        
        # Shapes and simple objects
        elif any(w in label_lower for w in ['circle', 'ball', 'moon', 'sun']):
            cv2.circle(canvas, (cx, cy), int(8*s), 0, 2)
        elif 'star' in label_lower:
            self._draw_star_simple(canvas, cx, cy, s)
        elif 'heart' in label_lower:
            self._draw_heart_simple(canvas, cx, cy, s)
        elif any(w in label_lower for w in ['square', 'box', 'book']):
            cv2.rectangle(canvas, (int(cx-8*s), int(cy-6*s)), 
                         (int(cx+8*s), int(cy+6*s)), 0, 2)
        
        # Vehicles
        elif 'car' in label_lower or 'truck' in label_lower:
            self._draw_car_simple(canvas, cx, cy, s)
        elif 'airplane' in label_lower or 'plane' in label_lower:
            self._draw_airplane_simple(canvas, cx, cy, s)
        
        # Nature
        elif 'tree' in label_lower:
            self._draw_tree_simple(canvas, cx, cy, s)
        elif 'flower' in label_lower:
            self._draw_flower_simple(canvas, cx, cy, s)
        
        # Buildings/Structures
        elif 'house' in label_lower:
            self._draw_house_simple(canvas, cx, cy, s)
        
        # Objects
        elif 'umbrella' in label_lower:
            self._draw_umbrella_simple(canvas, cx, cy, s)
        
        else:
            # Generic sketch - simple circle
            cv2.circle(canvas, (cx, cy), int(7*s), 0, 2)
        
        # Scale up to display size using NEAREST neighbor to show pixelation
        # This helps users understand the model sees low-resolution images
        scaled = cv2.resize(canvas, (display_size, display_size), 
                           interpolation=cv2.INTER_NEAREST)
        
        # Convert to QPixmap
        height, width = scaled.shape
        bytes_per_line = width
        q_image = QImage(scaled.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        
        label_widget = QLabel()
        label_widget.setPixmap(pixmap)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        return label_widget
    
    # Simplified drawing helper methods for 28x28 resolution
    # These are optimized to work at the model's native resolution
    
    def _draw_cat_simple(self, canvas, cx, cy, s):
        """Draw simple cat face at 28x28 scale."""
        # Head circle
        cv2.circle(canvas, (cx, cy), int(7*s), 0, 1)
        # Ears (triangles approximated with lines)
        cv2.line(canvas, (int(cx-5*s), int(cy-5*s)), (int(cx-7*s), int(cy-9*s)), 0, 1)
        cv2.line(canvas, (int(cx+5*s), int(cy-5*s)), (int(cx+7*s), int(cy-9*s)), 0, 1)
        # Eyes
        cv2.circle(canvas, (int(cx-3*s), int(cy-2*s)), 1, 0, -1)
        cv2.circle(canvas, (int(cx+3*s), int(cy-2*s)), 1, 0, -1)
    
    def _draw_dog_simple(self, canvas, cx, cy, s):
        """Draw simple dog at 28x28 scale."""
        # Body ellipse
        cv2.ellipse(canvas, (cx, int(cy+2*s)), (int(7*s), int(5*s)), 0, 0, 360, 0, 1)
        # Head circle
        cv2.circle(canvas, (int(cx+4*s), int(cy-3*s)), int(4*s), 0, 1)
        # Legs
        cv2.line(canvas, (int(cx-3*s), int(cy+7*s)), (int(cx-3*s), int(cy+11*s)), 0, 1)
        cv2.line(canvas, (int(cx+3*s), int(cy+7*s)), (int(cx+3*s), int(cy+11*s)), 0, 1)
    
    def _draw_bird_simple(self, canvas, cx, cy, s):
        """Draw simple bird at 28x28 scale."""
        # Body
        cv2.ellipse(canvas, (cx, cy), (int(5*s), int(4*s)), 0, 0, 360, 0, 1)
        # Head
        cv2.circle(canvas, (int(cx+4*s), int(cy-3*s)), int(3*s), 0, 1)
        # Wing
        cv2.line(canvas, (int(cx-2*s), cy), (int(cx-7*s), int(cy-2*s)), 0, 1)
    
    def _draw_fish_simple(self, canvas, cx, cy, s):
        """Draw simple fish at 28x28 scale."""
        # Body ellipse
        cv2.ellipse(canvas, (cx, cy), (int(6*s), int(3*s)), 0, 0, 360, 0, 1)
        # Tail triangle
        pts = np.array([[int(cx-6*s), cy], [int(cx-9*s), int(cy-3*s)], [int(cx-9*s), int(cy+3*s)]], np.int32)
        cv2.polylines(canvas, [pts], True, 0, 1)
        # Eye
        cv2.circle(canvas, (int(cx+3*s), cy), 1, 0, -1)
    
    def _draw_elephant_simple(self, canvas, cx, cy, s):
        """Draw simple elephant at 28x28 scale."""
        # Body
        cv2.ellipse(canvas, (cx, cy), (int(7*s), int(5*s)), 0, 0, 360, 0, 1)
        # Head
        cv2.circle(canvas, (int(cx+5*s), int(cy-4*s)), int(4*s), 0, 1)
        # Trunk (curved line)
        cv2.line(canvas, (int(cx+7*s), int(cy-2*s)), (int(cx+9*s), int(cy+4*s)), 0, 1)
        # Legs
        cv2.line(canvas, (int(cx-3*s), int(cy+5*s)), (int(cx-3*s), int(cy+10*s)), 0, 1)
        cv2.line(canvas, (int(cx+3*s), int(cy+5*s)), (int(cx+3*s), int(cy+10*s)), 0, 1)
    
    def _draw_star_simple(self, canvas, cx, cy, s):
        """Draw 5-point star at 28x28 scale."""
        # Simple star with 5 points
        r = int(8*s)
        pts = []
        for i in range(5):
            a = i * 144 - 90  # degrees
            x = cx + int(r * math.cos(math.radians(a)))
            y = cy + int(r * math.sin(math.radians(a)))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        # Draw star by connecting every other point
        for i in range(5):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[(i+2)%5]), 0, 1)
    
    def _draw_heart_simple(self, canvas, cx, cy, s):
        """Draw heart at 28x28 scale."""
        # Two circles at top
        cv2.circle(canvas, (int(cx-3*s), int(cy-2*s)), int(3*s), 0, 1)
        cv2.circle(canvas, (int(cx+3*s), int(cy-2*s)), int(3*s), 0, 1)
        # V shape at bottom
        cv2.line(canvas, (int(cx-6*s), int(cy-1*s)), (cx, int(cy+6*s)), 0, 1)
        cv2.line(canvas, (int(cx+6*s), int(cy-1*s)), (cx, int(cy+6*s)), 0, 1)
    
    def _draw_car_simple(self, canvas, cx, cy, s):
        """Draw car at 28x28 scale."""
        # Body rectangle
        cv2.rectangle(canvas, (int(cx-8*s), int(cy)), (int(cx+8*s), int(cy+4*s)), 0, 1)
        # Cabin rectangle
        cv2.rectangle(canvas, (int(cx-4*s), int(cy-3*s)), (int(cx+4*s), int(cy)), 0, 1)
        # Wheels
        cv2.circle(canvas, (int(cx-5*s), int(cy+4*s)), int(2*s), 0, 1)
        cv2.circle(canvas, (int(cx+5*s), int(cy+4*s)), int(2*s), 0, 1)
    
    def _draw_airplane_simple(self, canvas, cx, cy, s):
        """Draw airplane at 28x28 scale."""
        # Fuselage (horizontal line)
        cv2.line(canvas, (int(cx-8*s), cy), (int(cx+8*s), cy), 0, 1)
        # Wings (cross)
        cv2.line(canvas, (int(cx-4*s), cy), (int(cx-4*s), int(cy-6*s)), 0, 1)
        cv2.line(canvas, (int(cx+4*s), cy), (int(cx+4*s), int(cy-6*s)), 0, 1)
        cv2.line(canvas, (int(cx-4*s), int(cy-6*s)), (int(cx+4*s), int(cy-6*s)), 0, 1)
    
    def _draw_tree_simple(self, canvas, cx, cy, s):
        """Draw tree at 28x28 scale."""
        # Trunk
        cv2.rectangle(canvas, (int(cx-2*s), int(cy+2*s)), (int(cx+2*s), int(cy+8*s)), 0, -1)
        # Foliage (circle)
        cv2.circle(canvas, (cx, int(cy-2*s)), int(6*s), 0, 1)
    
    def _draw_flower_simple(self, canvas, cx, cy, s):
        """Draw flower at 28x28 scale."""
        # Stem
        cv2.line(canvas, (cx, int(cy+2*s)), (cx, int(cy+10*s)), 0, 1)
        # Petals (5 small circles around center)
        for i in range(5):
            a = i * 72  # degrees
            x = cx + int(4*s * math.cos(math.radians(a)))
            y = int(cy + 4*s * math.sin(math.radians(a)))
            cv2.circle(canvas, (x, y), int(2*s), 0, 1)
        # Center
        cv2.circle(canvas, (cx, cy), int(2*s), 0, -1)
    
    def _draw_house_simple(self, canvas, cx, cy, s):
        """Draw house at 28x28 scale."""
        # Walls
        cv2.rectangle(canvas, (int(cx-6*s), int(cy)), (int(cx+6*s), int(cy+8*s)), 0, 1)
        # Roof (triangle)
        pts = np.array([[int(cx-7*s), int(cy)], [cx, int(cy-6*s)], [int(cx+7*s), int(cy)]], np.int32)
        cv2.polylines(canvas, [pts], True, 0, 1)
        # Door
        cv2.rectangle(canvas, (int(cx-2*s), int(cy+3*s)), (int(cx+2*s), int(cy+8*s)), 0, 1)
    
    def _draw_umbrella_simple(self, canvas, cx, cy, s):
        """Draw umbrella at 28x28 scale."""
        # Canopy (arc approximated with ellipse)
        cv2.ellipse(canvas, (cx, int(cy-2*s)), (int(8*s), int(4*s)), 0, 0, 180, 0, 1)
        # Handle (vertical line with curve)
        cv2.line(canvas, (cx, int(cy-2*s)), (cx, int(cy+8*s)), 0, 1)
        cv2.ellipse(canvas, (cx, int(cy+8*s)), (int(2*s), int(2*s)), 0, 180, 360, 0, 1)
    
    # Old drawing methods (kept for backward compatibility but not used)
    def _draw_cat(self, p, cx, cy):
        """Draw cat face."""
        p.drawEllipse(cx - 28, cy - 23, 56, 50)
        # Ears
        p.drawLine(cx - 23, cy - 23, cx - 28, cy - 38)
        p.drawLine(cx - 23, cy - 23, cx - 15, cy - 33)
        p.drawLine(cx + 23, cy - 23, cx + 28, cy - 38)
        p.drawLine(cx + 23, cy - 23, cx + 15, cy - 33)
        # Eyes
        p.drawEllipse(cx - 14, cy - 8, 7, 10)
        p.drawEllipse(cx + 7, cy - 8, 7, 10)
        # Nose
        p.drawLine(cx, cy + 5, cx - 4, cy + 9)
        p.drawLine(cx, cy + 5, cx + 4, cy + 9)
    
    def _draw_dog(self, p, cx, cy):
        """Draw dog."""
        p.drawEllipse(cx - 28, cy - 8, 56, 36)
        p.drawEllipse(cx + 13, cy - 28, 32, 32)
        # Ears
        p.drawEllipse(cx + 8, cy - 33, 14, 23)
        p.drawEllipse(cx + 32, cy - 33, 14, 23)
        # Legs
        p.drawLine(cx - 18, cy + 23, cx - 18, cy + 42)
        p.drawLine(cx, cy + 23, cx, cy + 42)
        p.drawLine(cx + 18, cy + 23, cx + 18, cy + 42)
    
    def _draw_bird(self, p, cx, cy):
        """Draw bird."""
        p.drawEllipse(cx - 18, cy - 8, 36, 28)
        p.drawEllipse(cx + 13, cy - 18, 18, 18)
        p.drawLine(cx + 31, cy - 9, cx + 42, cy - 9)
        p.drawArc(cx - 13, cy - 4, 23, 18, 0, 180 * 16)
        p.drawLine(cx - 18, cy, cx - 32, cy - 9)
        p.drawLine(cx - 18, cy + 4, cx - 32, cy + 4)
    
    def _draw_fish(self, p, cx, cy):
        """Draw fish."""
        p.drawEllipse(cx - 23, cy - 13, 46, 26)
        p.drawLine(cx - 23, cy, cx - 37, cy - 13)
        p.drawLine(cx - 23, cy, cx - 37, cy + 13)
        p.drawLine(cx - 37, cy - 13, cx - 37, cy + 13)
        p.drawEllipse(cx + 9, cy - 4, 5, 5)
    
    def _draw_elephant(self, p, cx, cy):
        """Draw elephant."""
        p.drawEllipse(cx - 28, cy - 13, 56, 42)
        p.drawEllipse(cx + 13, cy - 28, 32, 37)
        p.drawArc(cx + 28, cy - 9, 18, 37, -90 * 16, 180 * 16)
        p.drawEllipse(cx + 4, cy - 32, 23, 28)
        # Legs
        p.drawLine(cx - 18, cy + 23, cx - 18, cy + 42)
        p.drawLine(cx, cy + 23, cx, cy + 42)
        p.drawLine(cx + 18, cy + 23, cx + 18, cy + 42)
    
    def _draw_butterfly(self, p, cx, cy):
        """Draw butterfly."""
        p.drawLine(cx, cy - 18, cx, cy + 18)
        p.drawEllipse(cx - 32, cy - 23, 28, 23)
        p.drawEllipse(cx + 4, cy - 23, 28, 23)
        p.drawEllipse(cx - 28, cy, 23, 18)
        p.drawEllipse(cx + 5, cy, 23, 18)
    
    def _draw_snake(self, p, cx, cy):
        """Draw snake."""
        path_points = [
            (cx - 35, cy), (cx - 20, cy - 20), (cx, cy - 15),
            (cx + 20, cy - 25), (cx + 30, cy - 10), (cx + 25, cy + 10)
        ]
        for i in range(len(path_points) - 1):
            p.drawLine(path_points[i][0], path_points[i][1],
                      path_points[i + 1][0], path_points[i + 1][1])
    
    def _draw_star(self, p, cx, cy, r):
        """Draw 5-point star."""
        points = []
        for i in range(5):
            a = i * 144 - 90
            x = cx + r * math.cos(math.radians(a))
            y = cy + r * math.sin(math.radians(a))
            points.append((int(x), int(y)))
        for i in range(5):
            p.drawLine(points[i][0], points[i][1],
                      points[(i + 2) % 5][0], points[(i + 2) % 5][1])
    
    def _draw_heart(self, p, cx, cy):
        """Draw heart."""
        p.drawEllipse(cx - 23, cy - 18, 23, 23)
        p.drawEllipse(cx, cy - 18, 23, 23)
        p.drawLine(cx - 23, cy - 4, cx, cy + 23)
        p.drawLine(cx + 23, cy - 4, cx, cy + 23)
    
    def _draw_car(self, p, cx, cy):
        """Draw car."""
        p.drawRect(cx - 32, cy, 64, 18)
        p.drawRect(cx - 18, cy - 13, 36, 13)
        p.drawEllipse(cx - 23, cy + 13, 14, 14)
        p.drawEllipse(cx + 9, cy + 13, 14, 14)
    
    def _draw_airplane(self, p, cx, cy):
        """Draw airplane."""
        p.drawLine(cx - 32, cy, cx + 32, cy)
        p.drawLine(cx - 18, cy, cx - 18, cy - 23)
        p.drawLine(cx - 18, cy - 23, cx + 18, cy - 23)
        p.drawLine(cx + 18, cy - 23, cx + 18, cy)
        p.drawLine(cx + 28, cy, cx + 28, cy - 13)
        p.drawLine(cx + 23, cy - 13, cx + 32, cy - 13)
    
    def _draw_bicycle(self, p, cx, cy):
        """Draw bicycle."""
        p.drawEllipse(cx - 32, cy + 4, 23, 23)
        p.drawEllipse(cx + 9, cy + 4, 23, 23)
        p.drawLine(cx - 20, cy + 15, cx + 20, cy + 15)
        p.drawLine(cx, cy - 9, cx - 20, cy + 15)
        p.drawLine(cx, cy - 9, cx + 20, cy + 15)
        p.drawLine(cx, cy - 9, cx + 9, cy - 13)
    
    def _draw_train(self, p, cx, cy):
        """Draw train."""
        p.drawRect(cx - 32, cy - 13, 64, 32)
        p.drawRect(cx - 23, cy - 23, 18, 10)
        p.drawEllipse(cx - 23, cy + 18, 14, 14)
        p.drawEllipse(cx + 9, cy + 18, 14, 14)
    
    def _draw_tree(self, p, cx, cy):
        """Draw tree."""
        p.drawRect(cx - 7, cy + 9, 14, 28)
        p.drawEllipse(cx - 23, cy - 28, 46, 46)
    
    def _draw_flower(self, p, cx, cy):
        """Draw flower."""
        p.drawLine(cx, cy + 9, cx, cy + 37)
        for i in range(5):
            a = i * 72
            x = cx + 14 * math.cos(math.radians(a))
            y = cy + 14 * math.sin(math.radians(a))
            p.drawEllipse(int(x) - 7, int(y) - 7, 14, 14)
        p.drawEllipse(cx - 5, cy - 5, 10, 10)
    
    def _draw_cloud(self, p, cx, cy):
        """Draw cloud."""
        p.drawEllipse(cx - 37, cy - 18, 28, 28)
        p.drawEllipse(cx - 18, cy - 23, 32, 32)
        p.drawEllipse(cx, cy - 18, 28, 28)
    
    def _draw_house(self, p, cx, cy):
        """Draw house."""
        p.drawRect(cx - 28, cy, 56, 37)
        p.drawLine(cx - 32, cy, cx, cy - 23)
        p.drawLine(cx + 32, cy, cx, cy - 23)
        p.drawRect(cx - 9, cy + 13, 18, 23)
        p.drawRect(cx + 9, cy + 9, 11, 11)
    
    def _draw_umbrella(self, p, cx, cy):
        """Draw umbrella."""
        p.drawArc(cx - 32, cy - 18, 64, 37, 0, 180 * 16)
        p.drawLine(cx, cy, cx, cy + 32)
        p.drawArc(cx - 7, cy + 28, 14, 14, 180 * 16, 180 * 16)
    
    def _draw_cup(self, p, cx, cy):
        """Draw cup."""
        p.drawLine(cx - 18, cy - 18, cx - 23, cy + 18)
        p.drawLine(cx + 18, cy - 18, cx + 23, cy + 18)
        p.drawLine(cx - 23, cy + 18, cx + 23, cy + 18)
        p.drawArc(cx - 18, cy - 23, 36, 10, 0, 180 * 16)
        p.drawArc(cx + 18, cy - 5, 18, 23, -90 * 16, 180 * 16)
    
    def _draw_guitar(self, p, cx, cy):
        """Draw guitar."""
        p.drawEllipse(cx - 23, cy + 5, 46, 32)
        p.drawRect(cx - 7, cy - 28, 14, 37)
        p.drawEllipse(cx - 14, cy - 37, 28, 18)
    
    def _draw_generic(self, p, cx, cy, label):
        """Draw generic pattern."""
        random.seed(hash(label))
        for _ in range(3):
            a = random.randint(0, 360)
            r = random.randint(15, 32)
            x = cx + r * math.cos(math.radians(a))
            y = cy + r * math.sin(math.radians(a))
            p.drawLine(cx, cy, int(x), int(y))
        p.drawEllipse(cx - 18, cy - 18, 36, 36)
    
    def _filter_labels(self, text: str):
        """Filter labels based on search text."""
        search = text.lower().strip()
        if not search:
            self.filtered_labels = self.labels.copy()
        else:
            self.filtered_labels = [
                label for label in self.labels
                if search in label.lower()
            ]
        self._populate_grid()
    
    def _apply_styles(self):
        """Apply custom styles."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1a2332;
                color: #e0e0e0;
            }
            #examplesHeader {
                color: #00ffff;
                padding: 10px;
            }
            #examplesInfo {
                color: #a0a0a0;
                font-size: 13px;
            }
            #searchLabel {
                color: #00ffff;
                font-size: 14px;
                font-weight: bold;
            }
            #searchBox {
                background-color: #2d3d4f;
                color: #ffffff;
                border: 2px solid #00ffff;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
            #labelCard {
                background-color: #243447;
                border: 2px solid #3a4d62;
                border-radius: 8px;
                padding: 10px;
            }
            #labelCard:hover {
                border-color: #00ffff;
                background-color: #2d3d4f;
            }
            #labelName {
                color: #00ffff;
                font-size: 11px;
            }
            #countLabel {
                color: #a0a0a0;
                font-size: 13px;
            }
            #closeButton {
                background-color: #00ffff;
                color: #000000;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }
            #closeButton:hover {
                background-color: #00cccc;
            }
        """)
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() in (Qt.Key.Key_E, Qt.Key.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)
