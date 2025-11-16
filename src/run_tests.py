#!/usr/bin/env python3
"""
run_tests.py - Ejecuta la suite completa de pruebas
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Ejecuta las pruebas con pytest."""
    print("🚀 Ejecutando suite de pruebas para Pictionary Live")
    print("=" * 60)

    # Verificar que estamos en el directorio correcto
    if not Path("src/tests").exists():
        print("❌ Error: Ejecutar desde la raíz del proyecto (donde está src/)")
        sys.exit(1)

    # Instalar dependencias de test si no están
    try:
        import pytest
        import pytest_mock
    except ImportError:
        print("📦 Instalando dependencias de testing...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "pytest>=7.0.0", "pytest-mock>=3.10.0"
        ])

    # Ejecutar pruebas
    cmd = [
        sys.executable, "-m", "pytest",
        "src/tests/",
        "--tb=short",
        "--verbose",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n✅ Todas las pruebas pasaron exitosamente!")
        else:
            print(f"\n❌ Algunas pruebas fallaron (código: {result.returncode})")
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Ejecución interrumpida por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error ejecutando pruebas: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()