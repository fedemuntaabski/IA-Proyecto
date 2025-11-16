"""
Constantes para el módulo de clasificación.

Este archivo contiene todas las constantes utilizadas en el procesamiento
y clasificación de sketches.
"""

# Tamaño de imagen del modelo (28x28 para MNIST-like)
MODEL_IMAGE_SIZE = 28

# Configuración de dibujo por defecto
DEFAULT_LINE_WIDTH = 2
DEFAULT_CANVAS_SIZE = 256

# Umbrales de clasificación
MIN_POINTS_FOR_CLASSIFICATION = 10
CONFIDENCE_THRESHOLD = 0.5

# Configuración de suavizado
SMOOTHING_WINDOW_SIZE = 3

# Colores para visualización
DRAW_COLOR = (0, 255, 0)  # Verde para dibujo
POINT_COLOR = (0, 0, 255)  # Rojo para puntos
BACKGROUND_COLOR = (255, 255, 255)  # Blanco para fondo