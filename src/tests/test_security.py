"""
test_security.py - Pruebas unitarias para el módulo de seguridad
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile

from security import (
    SecurityError, validate_path_safety, sanitize_filename, validate_json_data,
    check_dependencies_vulnerabilities, setup_secure_environment, log_security_event,
    validate_ia_directory, validate_camera_id
)


class TestSecurity:
    """Pruebas para funciones de seguridad."""

    def test_validate_path_safety_valid_relative(self, tmp_path):
        """Prueba validación de ruta relativa válida."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        test_file = base_dir / "test.txt"
        test_file.write_text("test")

        result = validate_path_safety("test.txt", base_dir=base_dir)
        assert result == test_file

    def test_validate_path_safety_path_traversal(self, tmp_path):
        """Prueba detección de path traversal."""
        with pytest.raises(SecurityError, match="path traversal"):
            validate_path_safety("../etc/passwd", base_dir=tmp_path)

    def test_validate_path_safety_dangerous_chars(self, tmp_path):
        """Prueba detección de caracteres peligrosos."""
        with pytest.raises(SecurityError, match="caracteres peligrosos"):
            validate_path_safety("test<file>.txt", base_dir=tmp_path)

    def test_validate_path_safety_absolute_not_allowed(self, tmp_path):
        """Prueba que rutas absolutas no sean permitidas por defecto."""
        with pytest.raises(SecurityError, match="absolutas no permitidas"):
            validate_path_safety("C:\\Windows\\system32", base_dir=tmp_path, allow_absolute=False)

    def test_sanitize_filename_valid(self):
        """Prueba sanitización de nombre de archivo válido."""
        result = sanitize_filename("test_file.txt")
        assert result == "test_file.txt"

    def test_sanitize_filename_dangerous_chars(self):
        """Prueba sanitización eliminando caracteres peligrosos."""
        result = sanitize_filename("test<file>.txt")
        assert result == "testfile.txt"

    def test_sanitize_filename_empty_after_sanitize(self):
        """Prueba sanitización que resulta en nombre vacío."""
        with pytest.raises(SecurityError, match="inválido después de sanitizar"):
            sanitize_filename("<<<>>>")

    def test_validate_json_data_valid(self):
        """Prueba validación de datos JSON válidos."""
        data = {"key": "value", "list": [1, 2, 3]}
        result = validate_json_data(data)
        assert result == data

    def test_validate_json_data_invalid_key(self):
        """Prueba validación de claves JSON inválidas."""
        with pytest.raises(SecurityError, match="deben ser strings"):
            validate_json_data({123: "value"})

    def test_validate_json_data_long_string(self):
        """Prueba validación de strings muy largos."""
        long_string = "a" * 10001
        with pytest.raises(SecurityError, match="demasiado largo"):
            validate_json_data({"key": long_string})

    def test_validate_json_data_invalid_type(self):
        """Prueba validación de tipos no permitidos."""
        with pytest.raises(SecurityError, match="no permitido"):
            validate_json_data({"key": set([1, 2, 3])})

    @patch('importlib.metadata.version')
    def test_check_dependencies_vulnerabilities_no_safety(self, mock_version):
        """Prueba verificación de vulnerabilidades sin safety instalado."""
        mock_version.side_effect = lambda pkg: {
            'tensorflow': '2.18.0',
            'protobuf': '4.25.3'
        }.get(pkg, '1.0.0')

        vulnerabilities = check_dependencies_vulnerabilities()
        # Debería encontrar versiones no recomendadas o ninguna
        assert isinstance(vulnerabilities, list)

    def test_setup_secure_environment(self):
        """Prueba configuración de entorno seguro."""
        with patch.dict('os.environ', {}, clear=True):
            setup_secure_environment()
            # Verificar que TF_CPP_MIN_LOG_LEVEL esté configurado
            import os
            assert 'TF_CPP_MIN_LOG_LEVEL' in os.environ

    def test_log_security_event_with_logger(self):
        """Prueba logging de eventos de seguridad."""
        import logging
        logger = logging.getLogger('test')
        with patch.object(logger, 'warning') as mock_warning:
            log_security_event("test_event", {"detail": "value"}, logger)
            mock_warning.assert_called_once()

    def test_validate_ia_directory_valid(self, tmp_path):
        """Prueba validación de directorio IA válido."""
        ia_dir = tmp_path / "IA"
        ia_dir.mkdir()
        (ia_dir / "model_info.json").write_text('{"labels": ["test"]}')

        result = validate_ia_directory(str(ia_dir))
        assert result == ia_dir

    def test_validate_ia_directory_invalid_path(self):
        """Prueba validación de directorio IA con path traversal."""
        with pytest.raises(SecurityError):
            validate_ia_directory("../invalid")

    def test_validate_camera_id_valid(self):
        """Prueba validación de ID de cámara válido."""
        result = validate_camera_id(1)
        assert result == 1

    def test_validate_camera_id_invalid(self):
        """Prueba validación de ID de cámara inválido."""
        with pytest.raises(SecurityError, match="inválido"):
            validate_camera_id(15)

    def test_validate_camera_id_not_int(self):
        """Prueba validación de ID de cámara no entero."""
        with pytest.raises(SecurityError, match="inválido"):
            validate_camera_id("1")