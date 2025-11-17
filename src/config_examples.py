"""
config_examples.py - Ejemplos de configuración avanzada del modo de juego

Este archivo contiene ejemplos de diferentes configuraciones para
personalizar el comportamiento del modo de juego.
"""

from src.game_mode import GameConfig, ColorScheme
from src.game_integration import IntegrationConfig


# ============================================================================
# CONFIGURACIONES DE JUEGO
# ============================================================================

# Configuración: Juego Casual
CONFIG_CASUAL = GameConfig(
    window_width=1400,
    window_height=800,
    font_size_title=36,
    font_size_word=48,
    font_size_normal=14,
    auto_next_delay_ms=3000,     # 3 segundos antes de cambiar
    min_drawing_points=5,         # Menos puntos requeridos
    theme="light",
)

# Configuración: Juego Competitivo
CONFIG_COMPETITIVE = GameConfig(
    window_width=1600,
    window_height=900,
    font_size_title=42,
    font_size_word=64,            # Palabra más grande
    font_size_normal=16,
    auto_next_delay_ms=1000,      # 1 segundo apenas
    min_drawing_points=15,        # Más puntos requeridos
    theme="cyberpunk",
)

# Configuración: Juego Relajado
CONFIG_RELAXED = GameConfig(
    window_width=1200,
    window_height=700,
    font_size_title=30,
    font_size_word=40,
    font_size_normal=12,
    auto_next_delay_ms=5000,      # 5 segundos
    min_drawing_points=3,
    theme="dark",
)

# Configuración: Pantalla Grande (proyector)
CONFIG_LARGE_SCREEN = GameConfig(
    window_width=3840,
    window_height=2160,
    font_size_title=96,
    font_size_word=144,
    font_size_normal=48,
    auto_next_delay_ms=2000,
    min_drawing_points=20,
    theme="cyberpunk",
)

# Configuración: Pantalla Pequeña (netbook)
CONFIG_SMALL_SCREEN = GameConfig(
    window_width=800,
    window_height=600,
    font_size_title=24,
    font_size_word=32,
    font_size_normal=10,
    auto_next_delay_ms=2000,
    min_drawing_points=8,
    theme="light",
)


# ============================================================================
# CONFIGURACIONES DE INTEGRACIÓN
# ============================================================================

# Configuración: Alta Calidad (Full HD)
CONFIG_INT_HIGH_QUALITY = IntegrationConfig(
    camera_id=0,
    camera_width=1920,
    camera_height=1080,
    camera_fps=30,
    ia_dir="./IA",
    debug=False,
)

# Configuración: Optimizada para Performance (HD)
CONFIG_INT_PERFORMANCE = IntegrationConfig(
    camera_id=0,
    camera_width=1280,
    camera_height=720,
    camera_fps=30,
    ia_dir="./IA",
    debug=False,
)

# Configuración: Bajo Rendimiento (VGA)
CONFIG_INT_LOW = IntegrationConfig(
    camera_id=0,
    camera_width=640,
    camera_height=480,
    camera_fps=15,
    ia_dir="./IA",
    debug=False,
)

# Configuración: Debug (máximo detalle)
CONFIG_INT_DEBUG = IntegrationConfig(
    camera_id=0,
    camera_width=1280,
    camera_height=720,
    camera_fps=30,
    ia_dir="./IA",
    debug=True,
)


# ============================================================================
# COMBINACIONES RECOMENDADAS
# ============================================================================

PRESET_CASUAL_HOME = {
    "game_config": CONFIG_CASUAL,
    "integration_config": CONFIG_INT_PERFORMANCE,
    "description": "Juego casual en casa, bien equilibrado"
}

PRESET_COMPETITIVE_LAN = {
    "game_config": CONFIG_COMPETITIVE,
    "integration_config": CONFIG_INT_HIGH_QUALITY,
    "description": "Juego competitivo con máxima calidad"
}

PRESET_EVENT_PROJECTOR = {
    "game_config": CONFIG_LARGE_SCREEN,
    "integration_config": CONFIG_INT_HIGH_QUALITY,
    "description": "Proyector grande para eventos"
}

PRESET_LAPTOP_TRAVEL = {
    "game_config": CONFIG_SMALL_SCREEN,
    "integration_config": CONFIG_INT_PERFORMANCE,
    "description": "Portátil de viaje, pantalla pequeña"
}

PRESET_OLD_COMPUTER = {
    "game_config": CONFIG_RELAXED,
    "integration_config": CONFIG_INT_LOW,
    "description": "Computadora antigua, recursos limitados"
}


# ============================================================================
# UTILIDAD: FUNCIÓN PARA USAR PRESETS
# ============================================================================

def get_preset(preset_name: str):
    """
    Obtiene un preset de configuración por nombre.
    
    Args:
        preset_name: Uno de:
            - "casual_home"
            - "competitive_lan"
            - "event_projector"
            - "laptop_travel"
            - "old_computer"
    
    Returns:
        Dict con game_config, integration_config y descripción
    
    Ejemplo:
        preset = get_preset("casual_home")
        game_config = preset["game_config"]
        integration_config = preset["integration_config"]
    """
    presets = {
        "casual_home": PRESET_CASUAL_HOME,
        "competitive_lan": PRESET_COMPETITIVE_LAN,
        "event_projector": PRESET_EVENT_PROJECTOR,
        "laptop_travel": PRESET_LAPTOP_TRAVEL,
        "old_computer": PRESET_OLD_COMPUTER,
    }
    
    if preset_name not in presets:
        raise ValueError(f"Preset desconocido: {preset_name}. Disponibles: {list(presets.keys())}")
    
    return presets[preset_name]


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo 1: Usar un preset
    print("Disponibles presets:")
    presets = [
        "casual_home",
        "competitive_lan",
        "event_projector",
        "laptop_travel",
        "old_computer",
    ]
    for preset in presets:
        config = get_preset(preset)
        print(f"  • {preset}: {config['description']}")
    
    # Ejemplo 2: Cargar preset específico
    print("\nCargando preset 'casual_home':")
    preset = get_preset("casual_home")
    print(f"  Tema: {preset['game_config'].theme}")
    print(f"  Resolución cámara: {preset['integration_config'].camera_width}x{preset['integration_config'].camera_height}")
    
    # Ejemplo 3: Personalizar configuración
    print("\nPersonalizando configuración:")
    custom_config = GameConfig(
        window_width=1400,
        window_height=800,
        theme="cyberpunk",
        font_size_word=56,  # Palabra más grande
        auto_next_delay_ms=1500,  # 1.5 segundos
    )
    print(f"  Configuración personalizada creada")
    print(f"  Tema: {custom_config.theme}")
    print(f"  Tamaño palabra: {custom_config.font_size_word}")
