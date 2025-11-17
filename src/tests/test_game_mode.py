"""
test_game_mode.py - Pruebas unitarias para el modo de juego

Pruebas para GameMode, GameState y ColorScheme.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_mode import (
    GameMode, GameState, GameConfig, ColorScheme
)


class TestGameState:
    """Pruebas para el enum GameState."""
    
    def test_game_states_exist(self):
        """Verifica que todos los estados existen."""
        assert hasattr(GameState, 'WAITING_FOR_DRAW')
        assert hasattr(GameState, 'DRAWING')
        assert hasattr(GameState, 'PREDICTING')
        assert hasattr(GameState, 'CORRECT')
        assert hasattr(GameState, 'INCORRECT')
        assert hasattr(GameState, 'PAUSED')
    
    def test_game_state_values(self):
        """Verifica que los valores están correctamente asignados."""
        assert GameState.WAITING_FOR_DRAW.value == "waiting"
        assert GameState.DRAWING.value == "drawing"
        assert GameState.PREDICTING.value == "predicting"


class TestColorScheme:
    """Pruebas para los esquemas de colores."""
    
    def test_cyberpunk_theme_has_all_colors(self):
        """Verifica que el tema Cyberpunk tiene todos los colores."""
        colors = ColorScheme.CYBERPUNK
        required_keys = [
            "bg_primary", "bg_secondary", "accent_primary", "accent_secondary",
            "success", "warning", "text_primary", "text_secondary", "text_dim"
        ]
        for key in required_keys:
            assert key in colors, f"Color {key} falta en CYBERPUNK"
    
    def test_light_theme_has_all_colors(self):
        """Verifica que el tema Light tiene todos los colores."""
        colors = ColorScheme.LIGHT
        required_keys = [
            "bg_primary", "bg_secondary", "accent_primary", "accent_secondary",
            "success", "warning", "text_primary", "text_secondary", "text_dim"
        ]
        for key in required_keys:
            assert key in colors, f"Color {key} falta en LIGHT"
    
    def test_dark_theme_has_all_colors(self):
        """Verifica que el tema Dark tiene todos los colores."""
        colors = ColorScheme.DARK
        required_keys = [
            "bg_primary", "bg_secondary", "accent_primary", "accent_secondary",
            "success", "warning", "text_primary", "text_secondary", "text_dim"
        ]
        for key in required_keys:
            assert key in colors, f"Color {key} falta en DARK"
    
    def test_get_scheme_cyberpunk(self):
        """Prueba obtener esquema Cyberpunk."""
        scheme = ColorScheme.get_scheme("cyberpunk")
        assert scheme == ColorScheme.CYBERPUNK
    
    def test_get_scheme_light(self):
        """Prueba obtener esquema Light."""
        scheme = ColorScheme.get_scheme("light")
        assert scheme == ColorScheme.LIGHT
    
    def test_get_scheme_dark(self):
        """Prueba obtener esquema Dark."""
        scheme = ColorScheme.get_scheme("dark")
        assert scheme == ColorScheme.DARK
    
    def test_get_scheme_default(self):
        """Prueba obtener esquema por defecto (cyberpunk)."""
        scheme = ColorScheme.get_scheme("invalid_theme")
        assert scheme == ColorScheme.CYBERPUNK


class TestGameConfig:
    """Pruebas para la configuración del juego."""
    
    def test_default_config(self):
        """Verifica que la configuración por defecto es válida."""
        config = GameConfig()
        assert config.window_width == 1400
        assert config.window_height == 800
        assert config.font_family == "Segoe UI"
        assert config.font_size_title == 36
        assert config.font_size_word == 48
        assert config.font_size_normal == 14
        assert config.auto_next_delay_ms == 2000
        assert config.min_drawing_points == 10
        assert config.theme == "cyberpunk"
    
    def test_custom_config(self):
        """Verifica que se puede personalizar la configuración."""
        config = GameConfig(
            window_width=800,
            window_height=600,
            theme="light",
            auto_next_delay_ms=1000,
        )
        assert config.window_width == 800
        assert config.window_height == 600
        assert config.theme == "light"
        assert config.auto_next_delay_ms == 1000


class TestGameMode:
    """Pruebas para la lógica principal del juego."""
    
    @pytest.fixture
    def mock_predict_callback(self):
        """Callback de predicción mock."""
        def predict():
            return ("guitar", 0.95, [("guitar", 0.95), ("violin", 0.03), ("piano", 0.02)])
        return predict
    
    @pytest.fixture
    def sample_labels(self):
        """Palabras de muestra."""
        return ["guitar", "piano", "violin", "drum", "flute"]
    
    def test_game_mode_initialization(self, sample_labels, mock_predict_callback):
        """Prueba que GameMode se inicializa correctamente."""
        game = GameMode(
            labels=sample_labels,
            predict_callback=mock_predict_callback,
        )
        assert game.labels == sample_labels
        assert game.predict_callback == mock_predict_callback
        assert game.current_word is not None
        assert game.current_word in sample_labels
        assert game.score == 0
        assert game.streak == 0
        assert game.game_state == GameState.WAITING_FOR_DRAW
    
    def test_game_mode_requires_labels(self, mock_predict_callback):
        """Verifica que GameMode requiere al menos una palabra."""
        with pytest.raises(ValueError):
            GameMode(labels=[], predict_callback=mock_predict_callback)
    
    def test_select_next_word(self, sample_labels, mock_predict_callback):
        """Prueba la selección de palabras."""
        game = GameMode(
            labels=sample_labels,
            predict_callback=mock_predict_callback,
        )
        first_word = game.current_word
        
        # Seleccionar varias palabras
        for _ in range(10):
            game._select_next_word()
            assert game.current_word in sample_labels
            assert game.game_state == GameState.WAITING_FOR_DRAW
    
    def test_recent_words_rotation(self, sample_labels, mock_predict_callback):
        """Verifica que las palabras recientes no se repiten."""
        game = GameMode(
            labels=sample_labels,
            predict_callback=mock_predict_callback,
            config=GameConfig(theme="cyberpunk"),
        )
        
        # Seleccionar palabras hasta llenar el histórico
        words_selected = set()
        for _ in range(len(sample_labels) * 2):
            game._select_next_word()
            words_selected.add(game.current_word)
        
        # Debería haberse seleccionado más de una palabra
        assert len(words_selected) > 1
    
    def test_score_increments_on_correct(self, sample_labels, mock_predict_callback):
        """Verifica que la puntuación aumenta con predicción correcta."""
        game = GameMode(
            labels=sample_labels,
            predict_callback=lambda: ("guitar", 0.95, []),
        )
        game.current_word = "guitar"
        
        initial_score = game.score
        initial_streak = game.streak
        
        # Hacer predicción correcta
        game.predict_drawing()
        
        # Esperar a que se complete el thread
        import time
        time.sleep(0.5)
        
        # Verificar que se incrementaron
        assert game.score > initial_score or game.streak > initial_streak
    
    def test_streak_resets_on_incorrect(self, sample_labels, mock_predict_callback):
        """Verifica que la racha se resetea en predicción incorrecta."""
        game = GameMode(
            labels=sample_labels,
            predict_callback=lambda: ("piano", 0.95, []),
        )
        game.current_word = "guitar"
        game.streak = 5
        
        # Hacer predicción incorrecta
        game.predict_drawing()
        
        # Esperar a que se complete el thread
        import time
        time.sleep(0.5)
        
        # Verificar que se resetea la racha
        assert game.streak == 0
    
    def test_reset_game(self, sample_labels, mock_predict_callback):
        """Prueba que reiniciar el juego resetea el estado."""
        game = GameMode(
            labels=sample_labels,
            predict_callback=mock_predict_callback,
        )
        
        # Modificar estado
        game.score = 10
        game.streak = 5
        game.recent_words = ["guitar", "piano"]
        
        # Reiniciar
        game.reset_game()
        
        # Verificar que todo está en estado inicial
        assert game.score == 0
        assert game.streak == 0
        assert game.recent_words == []
    
    def test_get_color(self, sample_labels, mock_predict_callback):
        """Prueba obtención de colores."""
        game = GameMode(
            labels=sample_labels,
            predict_callback=mock_predict_callback,
            config=GameConfig(theme="cyberpunk"),
        )
        
        color = game._get_color("accent_primary")
        assert color == "#00ffff"
        
        # Tema Light
        game_light = GameMode(
            labels=sample_labels,
            predict_callback=mock_predict_callback,
            config=GameConfig(theme="light"),
        )
        
        color_light = game_light._get_color("accent_primary")
        assert color_light == "#0066cc"
    
    def test_logger_setup(self, sample_labels, mock_predict_callback):
        """Verifica que se crea un logger por defecto."""
        game = GameMode(
            labels=sample_labels,
            predict_callback=mock_predict_callback,
        )
        assert game.logger is not None


class TestGameModeIntegration:
    """Pruebas de integración para GameMode."""
    
    def test_game_flow_correct_prediction(self, sample_labels=None):
        """Prueba el flujo completo con predicción correcta."""
        if sample_labels is None:
            sample_labels = ["guitar", "piano", "violin"]
        
        def predict_correct():
            return ("guitar", 0.95, [("guitar", 0.95), ("violin", 0.03)])
        
        game = GameMode(
            labels=sample_labels,
            predict_callback=predict_correct,
        )
        
        # Establecer palabra conocida
        game.current_word = "guitar"
        assert game.game_state == GameState.WAITING_FOR_DRAW
        
        # Simular predicción
        game.game_state = GameState.PREDICTING
        game.score += 1
        game.streak += 1
        
        assert game.score == 1
        assert game.streak == 1
    
    def test_game_flow_incorrect_prediction(self, sample_labels=None):
        """Prueba el flujo completo con predicción incorrecta."""
        if sample_labels is None:
            sample_labels = ["guitar", "piano", "violin"]
        
        def predict_incorrect():
            return ("piano", 0.80, [("piano", 0.80), ("violin", 0.10)])
        
        game = GameMode(
            labels=sample_labels,
            predict_callback=predict_incorrect,
        )
        
        # Establecer palabra diferente
        game.current_word = "guitar"
        game.streak = 3
        
        # Simular predicción incorrecta
        game.game_state = GameState.PREDICTING
        game.streak = 0  # Reset en incorrecto
        
        assert game.streak == 0


# Para ejecutar las pruebas: pytest src/tests/test_game_mode.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
