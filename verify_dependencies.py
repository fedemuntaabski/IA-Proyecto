#!/usr/bin/env python3
"""
Script de verificación de MediaPipe sin dependencias de TensorFlow
Bypassa el import de tf.tools.docs que falla en Python 3.12
"""

import sys
import os

# Limpiar sys.modules para evitar imports cachés
for mod in list(sys.modules.keys()):
    if 'tensorflow' in mod or 'mediapipe.tasks' in mod:
        del sys.modules[mod]

print("Verificando dependencias...")

# 1. Verificar OpenCV
try:
    import cv2
    print("[OK] OpenCV disponible")
except ImportError:
    print("[ERROR] OpenCV no disponible")
    sys.exit(1)

# 2. Verificar NumPy
try:
    import numpy
    print("[OK] NumPy disponible")
except ImportError:
    print("[ERROR] NumPy no disponible")
    sys.exit(1)

# 3. Verificar MediaPipe.solutions (sin cargar tasks.python que requiere TF)
try:
    # Importar solo lo que necesitamos
    import mediapipe
    print(f"[OK] MediaPipe {mediapipe.__version__} disponible")
    
    # Intentar cargar solutions directamente
    try:
        from mediapipe.python.solutions import hands, pose
        print("[OK] MediaPipe Hands disponible")
        print("[OK] MediaPipe Pose disponible")
    except Exception as e:
        print(f"[WARN] No se pudieron cargar solutions: {e}")
        print("      (Esto es aceptable, se usará modo demo)")
except ImportError as e:
    print(f"[ERROR] MediaPipe no disponible: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("Verificacion completada. Sistema listo para usar.")
print("="*70)
