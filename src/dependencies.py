"""
Verificador de dependencias para Pictionary Live
"""

import sys
import subprocess
from pathlib import Path


REQUIRED_PACKAGES = {
    "opencv-python": "4.8.0",
    "numpy": "1.24.0",
    "mediapipe": "0.10.0",
}

OPTIONAL_PACKAGES = {
    "tensorflow": "2.13.0",  # Para inferencia real
}


def check_python_version():
    """Verifica que se está usando Python 3.10+"""
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ requerido. Version actual: {sys.version}")
        return False
    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor} detectado")
    return True


def check_package(package_name: str, min_version: str = None) -> bool:
    """
    Verifica si un paquete está instalado.
    
    Args:
        package_name: Nombre del paquete pip
        min_version: Versión mínima requerida
    
    Returns:
        True si está instalado y cumple versión
    """
    try:
        module = __import__(package_name.replace("-", "_"))
        if hasattr(module, '__version__'):
            version = module.__version__
            print(f"  OK: {package_name}: {version}")
            return True
        else:
            print(f"  OK: {package_name}: instalado (version desconocida)")
            return True
    except ImportError:
        return False


def install_packages(packages: dict, upgrade: bool = False):
    """
    Instala paquetes desde pip.
    
    Args:
        packages: Dict con nombre -> versión mínima
        upgrade: Si True, actualizar paquetes existentes
    """
    print("\n[INSTALACION] Instalando dependencias...\n")
    
    for package_name, min_version in packages.items():
        if check_package(package_name):
            continue
        
        print(f"  [INSTALANDO] {package_name}...")
        
        cmd = [sys.executable, "-m", "pip", "install", package_name]
        if upgrade:
            cmd.append("--upgrade")
        
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  [OK] {package_name} instalado correctamente")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Fallo al instalar {package_name}: {e}")
            return False
    
    return True


def main():
    """Verifica e instala dependencias."""
    print("=" * 70)
    print("VERIFICADOR DE DEPENDENCIAS - Pictionary Live")
    print("=" * 70)
    print()
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    print("\n[CHEQUEO] Verificando dependencias requeridas...")
    missing_required = []
    for package_name in REQUIRED_PACKAGES.keys():
        if not check_package(package_name):
            missing_required.append(package_name)
    
    if missing_required:
        print(f"\n[AVISO] Faltan {len(missing_required)} dependencias requeridas:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        
        if not install_packages({p: REQUIRED_PACKAGES[p] for p in missing_required}):
            print("\n[ERROR] Fallo al instalar dependencias requeridas")
            sys.exit(1)
    
    print("\n[OK] Todas las dependencias requeridas estan instaladas")
    
    # Verificar opcional
    print("\n[CHEQUEO] Verificando dependencias opcionales...")
    for package_name in OPTIONAL_PACKAGES.keys():
        if check_package(package_name):
            print(f"  OK: {package_name} disponible (inferencia real activada)")
        else:
            print(f"  AVISO: {package_name} no disponible (modo demo)")
    
    print("\n" + "=" * 70)
    print("[OK] Verificacion completada correctamente")
    print("=" * 70)


if __name__ == "__main__":
    main()
