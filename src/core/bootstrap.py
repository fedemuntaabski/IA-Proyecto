"""
Bootstrap Module - Inicialización y Configuración del Sistema.

Este módulo maneja la inicialización completa del sistema,
incluyendo chequeos de salud y optimización de configuración.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import sys


def initialize_system(run_health_check: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """
    Inicializa el sistema con todas las verificaciones necesarias.

    Args:
        run_health_check: Si ejecutar chequeos de salud
        verbose: Si imprimir información detallada

    Returns:
        Diccionario con información de inicialización
    """
    init_info = {
        'success': False,
        'health_check_passed': False,
        'warnings': [],
        'errors': [],
        'config': {}
    }

    # Crear directorios necesarios
    dirs = ['config', 'data', 'feedback_data', 'locale', 'IA']
    for dir_name in dirs:
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)

    # Ejecutar chequeos de salud si está habilitado
    if run_health_check:
        try:
            from .src.core.utils.system_health_check import system_health_checker

            health_results = system_health_checker.run_all_checks()
            summary = system_health_checker.get_summary()

            init_info['health_check_passed'] = summary['failures'] == 0
            init_info['health_summary'] = summary

            if verbose:
                system_health_checker.print_report()

            # Recopilar problemas
            for result in health_results:
                if result.status == 'warning':
                    init_info['warnings'].append(result.message)
                elif result.status == 'fail':
                    init_info['errors'].append(result.message)

        except Exception as e:
            init_info['warnings'].append(f"No se pudo ejecutar chequeos de salud: {str(e)}")

    # Cargar configuración
    try:
        from .src.core.utils.settings_manager import settings_manager

        settings_manager.load()
        valid, errors = settings_manager.validate()

        if not valid:
            init_info['warnings'].extend(errors)
        else:
            if verbose:
                print("✓ Configuración validada correctamente")

        init_info['config'] = settings_manager.get_section('detection')

    except Exception as e:
        init_info['errors'].append(f"Error cargando configuración: {str(e)}")

    # Verificar disponibilidad de dependencias críticas
    critical_deps = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow'
    }

    missing_deps = []
    for module_name, package_name in critical_deps.items():
        try:
            __import__(module_name)
            if verbose:
                print(f"✓ {package_name} disponible")
        except ImportError:
            missing_deps.append(package_name)
            init_info['errors'].append(f"Dependencia crítica faltante: {package_name}")

    # Determinar éxito general
    init_info['success'] = len(init_info['errors']) == 0

    if verbose:
        print("\n" + "="*60)
        if init_info['success']:
            print("✅ INICIALIZACIÓN COMPLETADA")
        else:
            print("❌ INICIALIZACIÓN CON ERRORES")
            if init_info['errors']:
                print("\nErrores:")
                for error in init_info['errors']:
                    print(f"  • {error}")
        if init_info['warnings']:
            print("\nAdvertencias:")
            for warning in init_info['warnings']:
                print(f"  • {warning}")
        print("="*60 + "\n")

    return init_info


def apply_performance_optimizations() -> Dict[str, Any]:
    """
    Aplica optimizaciones de rendimiento basadas en disponibilidad de recursos.

    Returns:
        Diccionario con optimizaciones aplicadas
    """
    optimizations = {
        'applied': [],
        'gpu_enabled': False,
        'roi_optimization': False,
        'async_processing': True
    }

    try:
        import tensorflow as tf

        # Detectar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            optimizations['gpu_enabled'] = True
            optimizations['applied'].append("GPU acceleration enabled")
        else:
            # Optimizar CPU
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(4)
            optimizations['applied'].append("CPU optimization applied")

    except Exception as e:
        optimizations['applied'].append(f"Warning: Could not apply GPU optimization - {str(e)}")

    # Aplicar optimizaciones de ROI
    try:
        from .src.core.utils.settings_manager import settings_manager
        if settings_manager.get('performance.enable_roi_optimization', True):
            optimizations['roi_optimization'] = True
            optimizations['applied'].append("ROI optimization enabled")
    except Exception:
        pass

    return optimizations


if __name__ == "__main__":
    # Si se ejecuta directamente
    init_result = initialize_system(run_health_check=True, verbose=True)

    if init_result['success']:
        perf_opts = apply_performance_optimizations()
        print("Performance Optimizations Applied:")
        for opt in perf_opts['applied']:
            print(f"  ✓ {opt}")
        sys.exit(0)
    else:
        sys.exit(1)
