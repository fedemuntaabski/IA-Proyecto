"""
logger_setup.py - Configuración de logging
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Configura logging con nivel INFO o DEBUG.
    
    Args:
        debug: Si True, usar nivel DEBUG
    
    Returns:
        Logger configurado
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Crear directorio de logs
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Crear logger
    logger = logging.getLogger("PictionaryLive")
    logger.setLevel(level)
    
    # Eliminar handlers previos
    logger.handlers.clear()
    
    # Handler para archivo
    log_file = logs_dir / f"pictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formato
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
