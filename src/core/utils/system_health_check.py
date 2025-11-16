"""
System Health Check - Verificación de Salud del Sistema.

Este módulo realiza verificaciones exhaustivas del sistema y
proporciona recomendaciones de configuración óptima.
"""

import sys
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path


class HealthCheckResult:
    """Resultado de un chequeo de salud."""

    def __init__(self, check_name: str):
        """
        Inicializa el resultado.

        Args:
            check_name: Nombre del chequeo
        """
        self.check_name = check_name
        self.status = 'unknown'  # 'pass', 'warning', 'fail'
        self.message = ''
        self.details = {}
        self.recommendations = []

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'name': self.check_name,
            'status': self.status,
            'message': self.message,
            'details': self.details,
            'recommendations': self.recommendations
        }


class SystemHealthChecker:
    """Verificador de salud del sistema."""

    def __init__(self):
        """Inicializa el verificador."""
        self.results: List[HealthCheckResult] = []

    def run_all_checks(self) -> List[HealthCheckResult]:
        """
        Ejecuta todos los chequeos de salud.

        Returns:
            Lista de resultados de chequeos
        """
        self.results = []

        # Ejecutar chequeos
        self._check_python_version()
        self._check_dependencies()
        self._check_camera()
        self._check_disk_space()
        self._check_memory()
        self._check_file_permissions()
        self._check_model_files()

        return self.results

    def _check_python_version(self) -> None:
        """Verifica la versión de Python."""
        result = HealthCheckResult('Python Version')

        version_info = sys.version_info
        version_string = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

        if version_info.major >= 3 and version_info.minor >= 7:
            result.status = 'pass'
            result.message = f"Python {version_string} OK"
        else:
            result.status = 'fail'
            result.message = f"Python {version_string} demasiado antiguo"
            result.recommendations.append("Instalar Python 3.8 o superior")

        result.details['version'] = version_string

        self.results.append(result)

    def _check_dependencies(self) -> None:
        """Verifica dependencias críticas."""
        result = HealthCheckResult('Dependencies')

        required_packages = [
            ('cv2', 'opencv-python'),
            ('numpy', 'numpy'),
            ('tensorflow', 'tensorflow'),
            ('psutil', 'psutil')
        ]

        missing = []
        for module_name, package_name in required_packages:
            try:
                __import__(module_name)
            except ImportError:
                missing.append(package_name)

        if not missing:
            result.status = 'pass'
            result.message = "Todas las dependencias OK"
        else:
            result.status = 'fail'
            result.message = f"Faltan dependencias: {', '.join(missing)}"
            result.recommendations.append(f"pip install {' '.join(missing)}")

        result.details['required'] = required_packages
        result.details['missing'] = missing

        self.results.append(result)

    def _check_camera(self) -> None:
        """Verifica disponibilidad de cámara."""
        result = HealthCheckResult('Camera')

        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                result.status = 'pass'
                result.message = "Cámara disponible"

                # Intentar leer un frame
                ret, frame = cap.read()
                if ret:
                    result.details['resolution'] = f"{frame.shape[1]}x{frame.shape[0]}"
                    result.details['format'] = 'OK'
                else:
                    result.status = 'warning'
                    result.message = "Cámara detectada pero no puede capturar frames"
                    result.recommendations.append("Verificar permisos de cámara")

                cap.release()
            else:
                result.status = 'fail'
                result.message = "Cámara no disponible"
                result.recommendations.append("Conectar una cámara USB o verificar la cámara integrada")

        except Exception as e:
            result.status = 'fail'
            result.message = f"Error: {str(e)}"

        self.results.append(result)

    def _check_disk_space(self) -> None:
        """Verifica espacio en disco."""
        result = HealthCheckResult('Disk Space')

        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            free_gb = free / (1024**3)

            if free_gb > 1.0:
                result.status = 'pass'
                result.message = f"Espacio suficiente ({free_gb:.1f} GB disponible)"
            else:
                result.status = 'warning'
                result.message = f"Espacio bajo ({free_gb:.1f} GB disponible)"
                result.recommendations.append("Liberar espacio en disco")

            result.details['total_gb'] = total / (1024**3)
            result.details['used_gb'] = used / (1024**3)
            result.details['free_gb'] = free_gb

        except Exception as e:
            result.status = 'warning'
            result.message = f"No se pudo verificar: {str(e)}"

        self.results.append(result)

    def _check_memory(self) -> None:
        """Verifica memoria disponible."""
        result = HealthCheckResult('Memory')

        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if memory.percent < 80:
                result.status = 'pass'
                result.message = f"Memoria OK ({memory.percent:.0f}% utilizada)"
            else:
                result.status = 'warning'
                result.message = f"Memoria alta ({memory.percent:.0f}% utilizada)"
                result.recommendations.append("Cerrar aplicaciones innecesarias")

            result.details['total_gb'] = memory.total / (1024**3)
            result.details['available_gb'] = available_gb
            result.details['percent_used'] = memory.percent

        except Exception as e:
            result.status = 'warning'
            result.message = f"No se pudo verificar: {str(e)}"

        self.results.append(result)

    def _check_file_permissions(self) -> None:
        """Verifica permisos de archivo."""
        result = HealthCheckResult('File Permissions')

        dirs_to_check = [
            'config',
            'data',
            'IA',
            'locale'
        ]

        issues = []
        for dir_name in dirs_to_check:
            try:
                path = Path(dir_name)
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)

                # Intentar crear un archivo temporal
                test_file = path / '.health_check_test'
                test_file.touch()
                test_file.unlink()

            except PermissionError:
                issues.append(f"No hay permisos en {dir_name}")
            except Exception as e:
                issues.append(f"Error en {dir_name}: {str(e)}")

        if not issues:
            result.status = 'pass'
            result.message = "Permisos OK"
        else:
            result.status = 'warning'
            result.message = f"Problemas de permisos: {len(issues)}"
            result.recommendations.extend(issues)

        self.results.append(result)

    def _check_model_files(self) -> None:
        """Verifica existencia de archivos del modelo."""
        result = HealthCheckResult('Model Files')

        required_files = [
            'IA/sketch_classifier_model.keras',
            'IA/model_info.json'
        ]

        missing = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing.append(file_path)

        if not missing:
            result.status = 'pass'
            result.message = "Archivos del modelo presentes"
        else:
            result.status = 'warning'
            result.message = f"Faltan archivos: {', '.join(missing)}"
            result.recommendations.append("Entrenar o descargar los archivos del modelo")

        result.details['required'] = required_files
        result.details['missing'] = missing

        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de los chequeos.

        Returns:
            Diccionario con el resumen
        """
        passed = sum(1 for r in self.results if r.status == 'pass')
        warnings = sum(1 for r in self.results if r.status == 'warning')
        failures = sum(1 for r in self.results if r.status == 'fail')

        overall_status = 'pass' if failures == 0 else ('warning' if warnings > 0 else 'fail')

        return {
            'overall_status': overall_status,
            'passed': passed,
            'warnings': warnings,
            'failures': failures,
            'total': len(self.results),
            'results': [r.to_dict() for r in self.results]
        }

    def print_report(self) -> None:
        """Imprime un reporte formateado."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("REPORTE DE SALUD DEL SISTEMA")
        print("="*60)

        for result in self.results:
            status_symbol = {
                'pass': '✅',
                'warning': '⚠️ ',
                'fail': '❌'
            }.get(result.status, '❓')

            print(f"\n{status_symbol} {result.check_name}")
            print(f"   {result.message}")

            if result.recommendations:
                print("   Recomendaciones:")
                for rec in result.recommendations:
                    print(f"     • {rec}")

        print("\n" + "="*60)
        print(f"RESUMEN: {summary['passed']} OK, {summary['warnings']} Advertencias, {summary['failures']} Errores")
        print("="*60 + "\n")


# Instancia global
system_health_checker = SystemHealthChecker()
