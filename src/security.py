"""
security.py - Módulo de seguridad para Pictionary Live

Proporciona funciones para validar inputs, sanitizar datos,
y verificar vulnerabilidades en dependencias.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import pkg_resources


class SecurityError(Exception):
    """Excepción para errores de seguridad."""
    pass


def validate_path_safety(path: str, base_dir: Optional[Path] = None, allow_absolute: bool = False) -> Path:
    """
    Valida que una ruta sea segura y no contenga ataques de path traversal.

    Args:
        path: Ruta a validar
        base_dir: Directorio base para rutas relativas
        allow_absolute: Si permitir rutas absolutas

    Returns:
        Path: Ruta validada y resuelta

    Raises:
        SecurityError: Si la ruta es insegura
    """
    if not path or not isinstance(path, str):
        raise SecurityError("Ruta inválida: debe ser una cadena no vacía")

    # Verificar caracteres peligrosos primero
    dangerous_chars = ['<', '>', '|', '"', '*', '?']
    if any(char in path for char in dangerous_chars):
        raise SecurityError(f"Ruta contiene caracteres peligrosos: {path}")

    # Verificar path traversal (..) en la ruta original
    if ".." in path:
        raise SecurityError(f"Ruta contiene path traversal: {path}")

    # Convertir a Path
    p = Path(path)

    # Si es absoluta y no se permite, error
    if p.is_absolute() and not allow_absolute:
        raise SecurityError(f"Rutas absolutas no permitidas: {path}")

    # Resolver la ruta
    try:
        if p.is_absolute():
            resolved = p.resolve()
        else:
            # Para rutas relativas, resolver desde base_dir si se proporciona
            if base_dir:
                resolved = (base_dir / p).resolve()
            else:
                resolved = p.resolve()
    except Exception as e:
        raise SecurityError(f"Error al resolver ruta '{path}': {e}")

    # Verificar que el directorio padre existe
    if not resolved.parent.exists():
        raise SecurityError(f"Directorio padre no existe: {resolved.parent}")

    # Si hay base_dir y no es absoluta, verificar que esté dentro del base_dir
    if base_dir and not p.is_absolute():
        try:
            resolved.relative_to(base_dir.resolve())
        except ValueError:
            raise SecurityError(f"Ruta no está dentro del directorio base: {path}")

    return resolved


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza un nombre de archivo eliminando caracteres peligrosos.

    Args:
        filename: Nombre de archivo a sanitizar

    Returns:
        str: Nombre sanitizado
    """
    if not filename:
        raise SecurityError("Nombre de archivo vacío")

    # Remover caracteres peligrosos
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)

    # Limitar longitud
    if len(sanitized) > 255:
        sanitized = sanitized[:255]

    # Asegurar que no esté vacío después de sanitizar
    if not sanitized.strip():
        raise SecurityError("Nombre de archivo inválido después de sanitizar")

    return sanitized.strip()


def validate_json_data(data: Any) -> Any:
    """
    Valida y sanitiza datos JSON para prevenir inyección.

    Args:
        data: Datos a validar

    Returns:
        Datos validados

    Raises:
        SecurityError: Si los datos son peligrosos
    """
    if isinstance(data, dict):
        validated = {}
        for k, v in data.items():
            if not isinstance(k, str):
                raise SecurityError("Claves JSON deben ser strings")
            # Validar clave
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', k):
                raise SecurityError(f"Clave JSON inválida: {k}")
            validated[k] = validate_json_data(v)
        return validated
    elif isinstance(data, list):
        return [validate_json_data(item) for item in data]
    elif isinstance(data, (str, int, float, bool)) or data is None:
        # Para strings, verificar longitud razonable
        if isinstance(data, str) and len(data) > 10000:
            raise SecurityError("String demasiado largo en datos JSON")
        return data
    else:
        raise SecurityError(f"Tipo de dato no permitido en JSON: {type(data)}")


def check_dependencies_vulnerabilities() -> List[Dict[str, Any]]:
    """
    Verifica vulnerabilidades conocidas en dependencias instaladas.

    Returns:
        Lista de vulnerabilidades encontradas
    """
    vulnerabilities = []

    try:
        # Intentar usar safety si está disponible
        import safety
        result = safety.check()
        if result:
            vulnerabilities.extend(result)
    except ImportError:
        # Fallback: verificar manualmente versiones críticas
        pass
    except Exception as e:
        logging.warning(f"Error al verificar vulnerabilidades con safety: {e}")

    # Verificar versiones críticas manualmente
    critical_packages = {
        'tensorflow': '2.18.0',  # Versión conocida segura
        'protobuf': '4.25.3',   # Compatible con TF
        'opencv-python': '4.5.0',
        'mediapipe': '0.10.0'
    }

    try:
        import pkg_resources
        for pkg, safe_version in critical_packages.items():
            try:
                installed = pkg_resources.get_distribution(pkg).version
                if installed != safe_version:
                    vulnerabilities.append({
                        'package': pkg,
                        'installed': installed,
                        'recommended': safe_version,
                        'severity': 'medium',
                        'description': f'Versión no recomendada para {pkg}'
                    })
            except pkg_resources.DistributionNotFound:
                continue
    except Exception as e:
        logging.warning(f"Error al verificar versiones manualmente: {e}")

    return vulnerabilities


def setup_secure_environment():
    """
    Configura el entorno para mayor seguridad.
    """
    # Deshabilitar ejecución de código remoto en pickle (si se usa)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ya está en main.py

    # Configuraciones adicionales de seguridad
    # Limitar threads si es necesario
    # os.environ['OMP_NUM_THREADS'] = '1'  # Para evitar fork bombs en algunos casos


def log_security_event(event: str, details: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """
    Registra eventos de seguridad.

    Args:
        event: Tipo de evento
        details: Detalles del evento
        logger: Logger a usar
    """
    if logger:
        logger.warning(f"SECURITY EVENT: {event} - {details}")
    else:
        print(f"[SECURITY] {event}: {details}")


# Funciones de validación específicas para el proyecto

def validate_ia_directory(ia_dir: str) -> Path:
    """
    Valida el directorio IA.

    Args:
        ia_dir: Ruta al directorio IA

    Returns:
        Path validado
    """
    base_dir = Path.cwd()
    validated_path = validate_path_safety(ia_dir, base_dir=base_dir, allow_absolute=True)

    if not validated_path.is_dir():
        raise SecurityError(f"El directorio IA no existe o no es un directorio: {validated_path}")

    # Verificar archivos críticos
    required_files = ['model_info.json']
    for file in required_files:
        if not (validated_path / file).exists():
            log_security_event("missing_critical_file", {"file": str(validated_path / file)})

    return validated_path


def validate_camera_id(camera_id: int) -> int:
    """
    Valida el ID de cámara.

    Args:
        camera_id: ID de cámara

    Returns:
        ID validado
    """
    if not isinstance(camera_id, int) or camera_id < 0 or camera_id > 10:
        raise SecurityError(f"ID de cámara inválido: {camera_id} (debe ser 0-10)")

    return camera_id