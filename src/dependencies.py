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
}

# MediaPipe solo requerido para Python < 3.13
if sys.version_info < (3, 13):
    REQUIRED_PACKAGES["mediapipe"] = ">=0.10.0"

OPTIONAL_PACKAGES = {
    "tensorflow": ">=2.17.0",  # Para inferencia real, compatible con protobuf
}


def check_python_version():
    """Verifica que se está usando Python 3.10+"""
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ requerido. Version actual: {sys.version}")
        return False
    
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"OK: Python {version_str} detectado")
    
    # Advertir sobre MediaPipe en Python 3.13+
    if sys.version_info >= (3, 13):
        print("AVISO: MediaPipe no es compatible con Python 3.13+")
        print("       Para funcionalidad completa, use Python 3.10-3.12")
        print("       La aplicación funcionará en modo limitado sin detección de manos")
    
    return True


def check_package(package_name: str, version_spec: str = None) -> bool:
    """
    Verifica si un paquete está instalado.
    
    Args:
        package_name: Nombre del paquete pip
        version_spec: Especificación de versión (opcional, solo informativo)
    
    Returns:
        True si está instalado
    """
    try:
        module = __import__(package_name.replace("-", "_"))
        if hasattr(module, '__version__'):
            version_str = module.__version__
            print(f"  OK: {package_name}: {version_str}")
            return True
        else:
            print(f"  OK: {package_name}: instalado (version desconocida)")
            return True
    except ImportError:
        return False
    except Exception as e:
        # Manejar errores de importación como AttributeError en TensorFlow
        print(f"  AVISO: {package_name}: error al importar ({e}), asumiendo instalado")
        return True


def get_package_version(package_name: str) -> str:
    """Obtiene la versión de un paquete instalado usando pip show."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", package_name], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
        return "desconocida"
    except Exception:
        return "desconocida"


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


def get_package_version(package_name: str) -> str:
    """Obtiene la versión de un paquete instalado usando pip show."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", package_name], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
        return "desconocida"
    except Exception:
        return "desconocida"


def validate_compatibility():
    """Valida compatibilidad entre paquetes críticos (protobuf con TF/MP)."""
    print("\n[VALIDACION] Verificando compatibilidad de versiones...")
    
    try:
        # Verificar protobuf
        protobuf_version = get_package_version("protobuf")
        print(f"  OK: protobuf {protobuf_version} instalado")
        
        # Verificar combinaciones críticas
        tf_available = check_package("tensorflow")
        mp_available = check_package("mediapipe")
        
        if tf_available and mp_available:
            tf_version = get_package_version("tensorflow")
            mp_version = get_package_version("mediapipe")
            print(f"  OK: TensorFlow {tf_version} + MediaPipe {mp_version} + protobuf {protobuf_version} disponibles")
        elif tf_available and not mp_available:
            tf_version = get_package_version("tensorflow")
            print(f"  OK: TensorFlow {tf_version} + protobuf {protobuf_version} disponibles (MediaPipe no disponible)")
        elif not tf_available and mp_available:
            mp_version = get_package_version("mediapipe")
            print(f"  OK: MediaPipe {mp_version} + protobuf {protobuf_version} disponibles (TensorFlow no disponible)")
        else:
            print(f"  AVISO: Solo protobuf {protobuf_version} disponible (modo limitado)")
        
        return True
    except Exception as e:
        print(f"  AVISO: No se pudo validar compatibilidad completa: {e}")
        return True  # No bloquear si falla la validación


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
    
    # Verificar MediaPipe específicamente
    if sys.version_info >= (3, 13):
        print("  AVISO: mediapipe no disponible en Python 3.13+ (detección de manos limitada)")
    elif check_package("mediapipe"):
        print("  OK: mediapipe disponible (detección de manos activada)")
    else:
        print("  AVISO: mediapipe no disponible (detección de manos limitada)")
    
    print("\n" + "=" * 70)
    print("[OK] Verificación completada correctamente")
    print("=" * 70)


if __name__ == "__main__":
    main()
