"""
Script para diagnosticar problemas de preprocesamiento
Ejecutar: python test_preprocessing.py
"""

import numpy as np
import cv2
import json
from pathlib import Path

def test_preprocessing():
    """Prueba el preprocesamiento con diferentes configuraciones."""
    
    # Cargar modelo info
    with open("IA/model_info.json", 'r') as f:
        model_info = json.load(f)
    
    print("="*60)
    print("DIAGNÓSTICO DE PREPROCESAMIENTO")
    print("="*60)
    print(f"Input shape del modelo: {model_info['input_shape']}")
    print(f"Clases: {len(model_info['classes'])}")
    print()
    
    # Crear un trazo de prueba simple (una línea diagonal)
    test_points = [(i/100, i/100) for i in range(0, 100, 2)]
    
    print("Probando diferentes configuraciones...")
    print()
    
    # Configuración 1: Fondo negro, trazos blancos (como captura actual)
    print("1️⃣  CONFIGURACIÓN ACTUAL (captura en vivo):")
    img1 = create_test_image(test_points, invert=False)
    print(f"   Rango: [{img1.min():.3f}, {img1.max():.3f}]")
    print(f"   Promedio: {img1.mean():.3f}")
    
    # Crear directorio si no existe
    diag_dir = Path("diagnosis_results")
    diag_dir.mkdir(exist_ok=True)
    
    capture_path = diag_dir / "test_capture_format.png"
    cv2.imwrite(str(capture_path), (img1.squeeze() * 255).astype('uint8'))
    print(f"   ✅ Guardado: {capture_path}")
    print()
    
    # Configuración 2: Fondo blanco, trazos negros (Quick Draw estándar)
    print("2️⃣  FORMATO QUICK DRAW (entrenamiento típico):")
    img2 = create_test_image(test_points, invert=True)
    print(f"   Rango: [{img2.min():.3f}, {img2.max():.3f}]")
    print(f"   Promedio: {img2.mean():.3f}")
    
    quickdraw_path = diag_dir / "test_quickdraw_format.png"
    cv2.imwrite(str(quickdraw_path), (img2.squeeze() * 255).astype('uint8'))
    print(f"   ✅ Guardado: {quickdraw_path}")
    print()
    
    print("="*60)
    print("RESULTADO:")
    print("="*60)
    print("Compara las imágenes generadas con las de tu entrenamiento:")
    print()
    print(f"  📁 {capture_path}   - Tu formato actual")
    print(f"  📁 {quickdraw_path} - Formato Quick Draw")
    print()
    print("Si tus imágenes de ENTRENAMIENTO se parecen a:")
    print(f"  → {capture_path.name}: usar invert_colors = False")
    print(f"  → {quickdraw_path.name}: usar invert_colors = True")
    print()
    
    # Verificar si hay ejemplos de entrenamiento
    check_training_examples()


def create_test_image(points, invert=False):
    """Crea imagen de prueba con los puntos dados."""
    canvas = np.zeros((256, 256), dtype=np.uint8)
    
    pixel_points = [(int(x*256), int(y*256)) for x, y in points]
    
    for i in range(1, len(pixel_points)):
        cv2.line(canvas, pixel_points[i-1], pixel_points[i], 255, 3, cv2.LINE_AA)
    
    # Redimensionar a 28x28
    resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalizar
    image = resized.astype(np.float32) / 255.0
    
    # Invertir si es necesario
    if invert:
        image = 1.0 - image
    
    return np.expand_dims(image, axis=-1)


def check_training_examples():
    """Verifica si hay ejemplos del entrenamiento."""
    print("="*60)
    print("BUSCANDO EJEMPLOS DE ENTRENAMIENTO...")
    print("="*60)
    
    possible_paths = [
        "IA/training_examples/",
        "IA/samples/",
        "data/samples/",
        "training/samples/",
    ]
    
    found = False
    for path in possible_paths:
        p = Path(path)
        if p.exists():
            images = list(p.glob("*.png")) + list(p.glob("*.jpg"))
            if images:
                print(f"✅ Encontrados ejemplos en: {path}")
                print(f"   Total: {len(images)} imágenes")
                
                # Analizar primera imagen
                img = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    avg = img.mean()
                    print(f"   Promedio de píxeles: {avg:.1f}/255")
                    if avg > 200:
                        print("   → Parece fondo BLANCO (típico Quick Draw)")
                    elif avg < 50:
                        print("   → Parece fondo NEGRO")
                    else:
                        print("   → Formato ambiguo, verificar manualmente")
                
                found = True
                break
    
    if not found:
        print("⚠️  No se encontraron ejemplos de entrenamiento")
        print("   Revisa manualmente tus imágenes de entrenamiento")
    
    print()


if __name__ == "__main__":
    test_preprocessing()