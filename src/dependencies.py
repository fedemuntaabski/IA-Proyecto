"""
Verificador de dependencias para Pictionary Live
"""

import sys
import subprocess
from pathlib import Path

try:
    from importlib.metadata import version
except ImportError:
    # Fallback para versiones antiguas de Python
    from importlib_metadata import version


REQUIRED_PACKAGES = {
    "opencv-python": ">=4.5.0",
    "numpy": ">=1.24.0",
    "mediapipe": ">=0.10.0",  # REQUERIDO - detección de manos
}

OPTIONAL_PACKAGES = {
    "tensorflow": ">=2.17.0",  # Para inferencia real, compatible con protobuf
}


def check_python_version():
    """Verifica que se está usando Python 3.10-3.12 (REQUERIDO para MediaPipe)"""
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ requerido. Versión actual: {sys.version}")
        return False
    
    if sys.version_info >= (3, 13):
        print("ERROR CRÍTICO: Python 3.13+ NO es compatible con MediaPipe")
        print("       Esta aplicación REQUIERE detección de manos con MediaPipe")
        print("       Por favor instale Python 3.10, 3.11 o 3.12")
        print("\nPara instalar Python 3.12:")
        print("  1. Descargue desde https://www.python.org/downloads/")
        print("  2. Instale Python 3.12.x")
        print("  3. Ejecute: py -3.12 main.py")
        return False
    
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"OK: Python {version_str} detectado (Compatible con MediaPipe)")
    return True


def check_package(package_name: str, version_spec: str = None) -> bool:
    """
    Verify if a package is installed.
    
    Args:
        package_name: Package name
        version_spec: Version spec (optional, informational only)
    
    Returns:
        True if installed
    """
    try:
        module = __import__(package_name.replace("-", "_"))
        version_str = getattr(module, '__version__', 'unknown')
        print(f"  OK: {package_name}: {version_str}")
        return True
    except ImportError:
        return False
    except Exception as e:
        # Handle import errors (e.g., TensorFlow AttributeError)
        print(f"  AVISO: {package_name}: error al importar ({e}), asumiendo instalado")
        return True


def get_package_version(package_name: str) -> str:
    """Get installed package version using pip show."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", package_name], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
        return "unknown"
    except Exception:
        return "unknown"


def install_packages(packages: dict, upgrade: bool = False):
    """
    Instala paquetes desde pip usando rangos.
    
    Args:
        packages: Dict con nombre -> especificación de versión
        upgrade: Si True, actualizar paquetes existentes
    """
    print("\n[INSTALACION] Instalando dependencias...\n")
    
    for package_name, version_spec in packages.items():
        if check_package(package_name, version_spec):
            continue
        
        print(f"  [INSTALANDO] {package_name}{version_spec}...")
        
        cmd = [sys.executable, "-m", "pip", "install", f"{package_name}{version_spec}"]
        if upgrade:
            cmd.append("--upgrade")
        
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  [OK] {package_name} instalado correctamente")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Fallo al instalar {package_name}: {e}")
            return False
    
    return True


def install_from_requirements():
    """Instala desde requirements.txt si existe."""
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        print(f"\n[INSTALACION] Instalando desde {req_file.name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  [OK] Dependencias instaladas desde requirements.txt")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [AVISO] Fallo al instalar desde requirements.txt (exit code {e.returncode}), continuando con instalación individual")
            return False
    else:
        print("\n[AVISO] requirements.txt no encontrado, usando rangos directos")
        return False


def validate_compatibility():
    """Validate compatibility between critical packages."""
    print("\n[VALIDACION] Verificando compatibilidad de versiones...")
    
    try:
        versions = {
            "protobuf": get_package_version("protobuf"),
            "tensorflow": get_package_version("tensorflow") if check_package("tensorflow") else None,
            "mediapipe": get_package_version("mediapipe") if check_package("mediapipe") else None,
        }
        
        # Report available packages
        available = [f"{k} {v}" for k, v in versions.items() if v]
        print(f"  OK: Paquetes disponibles: {', '.join(available)}")
        
        return True
    except Exception as e:
        print(f"  AVISO: Error en validación: {e}")
        return True  # Don't block on validation failure


def main():
    """Verifica e instala dependencias."""
    print("=" * 70)
    print("VERIFICADOR DE DEPENDENCIAS - Pictionary Live")
    print("=" * 70)
    print()
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Intentar instalar desde requirements.txt primero
    if not install_from_requirements():
        print("\n[CHEQUEO] Verificando dependencias requeridas...")
        missing_required = []
        for package_name, version_spec in REQUIRED_PACKAGES.items():
            if not check_package(package_name, version_spec):
                missing_required.append(package_name)
        
        if missing_required:
            print(f"\n[AVISO] Faltan {len(missing_required)} dependencias requeridas:")
            for pkg in missing_required:
                print(f"  - {pkg}")
            
            if not install_packages({p: REQUIRED_PACKAGES[p] for p in missing_required}):
                print("\n[ERROR] Fallo al instalar dependencias requeridas")
                sys.exit(1)
    
    print("\n[OK] Todas las dependencias requeridas están instaladas")
    
    # Validar compatibilidad
    validate_compatibility()
    
    # Verificar opcional
    print("\n[CHEQUEO] Verificando dependencias opcionales...")
    for package_name, version_spec in OPTIONAL_PACKAGES.items():
        if check_package(package_name, version_spec):
            print(f"  OK: {package_name} disponible (inferencia real activada)")
        else:
            print(f"  AVISO: {package_name} no disponible (modo demo)")
    
    # Verificar MediaPipe (CRÍTICO)
    if check_package("mediapipe"):
        print("  OK: mediapipe disponible (deteccion de manos activada)")
    else:
        print("  ERROR: mediapipe no disponible - funcionalidad critica")
    
    print("\n" + "=" * 70)
    print("[OK] Verificación completada correctamente")
    print("=" * 70)


if __name__ == "__main__":
    main()
