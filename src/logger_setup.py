"""
logger_setup.py - Configuración de logging
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logging(debug: bool = False, log_name: str = "pictionary", log_dir: str = "./logs") -> logging.Logger:
    """
    Configure logging with INFO or DEBUG level.
    
    Args:
        debug: Use DEBUG level if True
        log_name: Base name for log file
        log_dir: Directory for log files
    
    Returns:
        Configured logger
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    logs_path = Path(log_dir)
    logs_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("PictionaryLive")
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_path / f"{log_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Shared formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
