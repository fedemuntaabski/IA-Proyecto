"""
Pictionary Live - Aplicación para jugar Pictionary dibujando en el aire

Estructura del módulo src:
- config.py: Configuración centralizada
- hand_detector.py: Detección de manos
- stroke_manager.py: Gestión de trazos
- drawing_preprocessor.py: Preprocesado de datos
- model.py: Carga del modelo
- ui.py: Interfaz gráfica
- app.py: Aplicación principal
- logger_setup.py: Configuración de logging
- dependencies.py: Verificador de dependencias
"""

__version__ = "2.0"
__author__ = "Pictionary Live Team"

__all__ = [
    "config",
    "hand_detector",
    "stroke_manager",
    "drawing_preprocessor",
    "model",
    "ui",
    "app",
    "logger_setup",
    "dependencies",
]
