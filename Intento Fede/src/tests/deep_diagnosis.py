"""
deep_diagnosis.py - Diagnóstico profundo del problema de preprocesamiento
Ejecutar: python deep_diagnosis.py
"""

import numpy as np
import cv2
import json
from pathlib import Path
import sys
sys.path.append('./src')

from drawing_preprocessor import DrawingPreprocessor
from model import SketchClassifier
import logging

# Para simular la función de entrenamiento
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except:
    PIL_AVAILABLE = False

def strokes_to_image_training(strokes, image_size=28, line_width=2):
    """
    EXACTAMENTE la función que usaste en el entrenamiento.
    """
    if not PIL_AVAILABLE:
        print("❌ PIL no disponible")
        return None
        
    # Create a white image
    img = Image.new('L', (256, 256), color=255)
    draw = ImageDraw.Draw(img)
    
    # Draw each stroke
    for stroke in strokes:
        if len(stroke) < 2 or len(stroke[0]) < 2:
            continue
        
        x_coords = stroke[0]
        y_coords = stroke[1]
        
        # Draw lines between consecutive points
        for i in range(len(x_coords) - 1):
            draw.line(
                [(x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1])],
                fill=0,
                width=line_width
            )
    
    # Resize to target size
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Invert so drawing is white on black background (better for CNNs)
    img_array = 1.0 - img_array
    
    return img_array


def convert_normalized_to_stroke_format(points):
    """
    Convierte puntos normalizados (0-1) al formato de strokes de Quick Draw.
    
    Args:
        points: [(x1, y1), (x2, y2), ...] donde x,y están en [0, 1]
    
    Returns:
        [[x_coords], [y_coords]] donde coords están en [0, 255]
    """
    if not points:
        return [[], []]
    
    x_coords = [int(p[0] * 255) for p in points]
    y_coords = [int(p[1] * 255) for p in points]
    
    return [x_coords, y_coords]


def compare_preprocessing_methods():
    """Compara el método de entrenamiento vs el de captura en vivo."""
    
    print("="*80)
    print("DIAGNÓSTICO PROFUNDO: COMPARACIÓN DE PREPROCESAMIENTO")
    print("="*80)
    
    # Cargar modelo
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    
    classifier = SketchClassifier("./IA", logger, demo_mode=False)
    print(f"\n✅ Modelo cargado: {len(classifier.get_labels())} clases\n")
    
    # Inicializar preprocesador actual
    config = {
        "intermediate_size": 256,
        "line_thickness": 8,
        "padding_ratio": 0.15,
        "use_antialiasing": True,
        "apply_blur": True,
        "blur_kernel": 3,
        "blur_sigma": 1.0,
    }
    
    preprocessor = DrawingPreprocessor((28, 28, 1), config)
    
    # Crear dibujos de prueba
    test_cases = {
        "Reloj (círculo con manecillas)": create_clock_points(),
        "Círculo simple": create_circle_points(),
        "Casa": create_house_points(),
        "Estrella": create_star_points(),
    }
    
    results_dir = Path("diagnosis_results")
    results_dir.mkdir(exist_ok=True)
    
    for name, points in test_cases.items():
        print("\n" + "="*80)
        print(f"📝 TESTEANDO: {name}")
        print("="*80)
        
        # MÉTODO 1: Preprocesador de captura en vivo (actual)
        print("\n1️⃣  MÉTODO DE CAPTURA EN VIVO (tu código actual):")
        img_live = preprocessor.preprocess(points)
        
        print(f"   Shape: {img_live.shape}")
        print(f"   Dtype: {img_live.dtype}")
        print(f"   Rango: [{img_live.min():.4f}, {img_live.max():.4f}]")
        print(f"   Media: {img_live.mean():.4f}")
        print(f"   Desv.Std: {img_live.std():.4f}")
        
        # Predecir
        label1, conf1, top3_1 = classifier.predict(img_live)
        print(f"   🎯 Predicción: '{label1}' ({conf1:.1%})")
        print(f"   📊 Top 3:")
        for i, (lbl, c) in enumerate(top3_1[:3], 1):
            print(f"      {i}. {lbl}: {c:.1%}")
        
        # Guardar visualización
        vis_live = (img_live.squeeze() * 255).astype('uint8')
        vis_live_large = cv2.resize(vis_live, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(results_dir / f"{name.replace(' ', '_')}_LIVE.png"), vis_live_large)
        
        # MÉTODO 2: Simulación exacta del entrenamiento
        if PIL_AVAILABLE:
            print("\n2️⃣  MÉTODO DE ENTRENAMIENTO (función original):")
            
            # Convertir a formato de strokes
            strokes = [convert_normalized_to_stroke_format(points)]
            img_train = strokes_to_image_training(strokes, image_size=28, line_width=2)
            
            if img_train is not None:
                # Añadir dimensión de canal
                img_train = np.expand_dims(img_train, axis=-1)
                
                print(f"   Shape: {img_train.shape}")
                print(f"   Dtype: {img_train.dtype}")
                print(f"   Rango: [{img_train.min():.4f}, {img_train.max():.4f}]")
                print(f"   Media: {img_train.mean():.4f}")
                print(f"   Desv.Std: {img_train.std():.4f}")
                
                # Predecir
                label2, conf2, top3_2 = classifier.predict(img_train)
                print(f"   🎯 Predicción: '{label2}' ({conf2:.1%})")
                print(f"   📊 Top 3:")
                for i, (lbl, c) in enumerate(top3_2[:3], 1):
                    print(f"      {i}. {lbl}: {c:.1%}")
                
                # Guardar visualización
                vis_train = (img_train.squeeze() * 255).astype('uint8')
                vis_train_large = cv2.resize(vis_train, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(results_dir / f"{name.replace(' ', '_')}_TRAIN.png"), vis_train_large)
                
                # COMPARACIÓN
                print("\n3️⃣  COMPARACIÓN:")
                print(f"   Diferencia en media: {abs(img_live.mean() - img_train.mean()):.4f}")
                print(f"   Diferencia en std: {abs(img_live.std() - img_train.std()):.4f}")
                print(f"   MSE entre imágenes: {np.mean((img_live - img_train)**2):.6f}")
                
                # Diferencia visual
                diff = np.abs(img_live.squeeze() - img_train.squeeze())
                diff_vis = (diff * 255).astype('uint8')
                diff_vis_large = cv2.resize(diff_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(results_dir / f"{name.replace(' ', '_')}_DIFF.png"), diff_vis_large)
                
                if conf1 != conf2 or label1 != label2:
                    print(f"   ⚠️  PREDICCIONES DIFERENTES!")
                    print(f"       Live: {label1} ({conf1:.1%})")
                    print(f"       Train: {label2} ({conf2:.1%})")
                else:
                    print(f"   ✅ Predicciones idénticas")
    
    print("\n" + "="*80)
    print("✅ DIAGNÓSTICO COMPLETADO")
    print("="*80)
    print(f"\n📁 Imágenes guardadas en: {results_dir}/")
    print("\nCompara visualmente:")
    print("  - *_LIVE.png  = Lo que captura tu sistema")
    print("  - *_TRAIN.png = Cómo se veían en el entrenamiento")
    print("  - *_DIFF.png  = Diferencia entre ambos")
    print("\nSi *_DIFF.png tiene mucho blanco, hay diferencias grandes.")


def create_clock_points():
    """Crea un reloj simple: círculo + dos manecillas."""
    points = []
    
    # Círculo
    num_points = 40
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = 0.5 + 0.3 * np.cos(angle)
        y = 0.5 + 0.3 * np.sin(angle)
        points.append((x, y))
    
    # Separación para manecillas
    points.append((0.5, 0.5))  # Centro
    
    # Manecilla de hora (a las 3)
    points.append((0.5, 0.5))
    points.append((0.7, 0.5))
    
    # Volver al centro
    points.append((0.5, 0.5))
    
    # Manecilla de minutos (a las 12)
    points.append((0.5, 0.5))
    points.append((0.5, 0.25))
    
    return points


def create_circle_points(num_points=50):
    """Crea puntos de un círculo."""
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = 0.5 + 0.3 * np.cos(angle)
        y = 0.5 + 0.3 * np.sin(angle)
        points.append((x, y))
    return points


def create_house_points():
    """Crea una casa simple."""
    points = []
    
    # Base del cuadrado
    base = [(0.2, 0.5), (0.8, 0.5), (0.8, 0.8), (0.2, 0.8), (0.2, 0.5)]
    points.extend(base)
    
    # Separación
    points.append((0.2, 0.5))
    
    # Techo triangular
    points.append((0.2, 0.5))
    points.append((0.5, 0.2))
    points.append((0.8, 0.5))
    
    return points


def create_star_points(num_points=10):
    """Crea una estrella de 5 puntas."""
    points = []
    outer_radius = 0.35
    inner_radius = 0.15
    
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points - np.pi / 2
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius
        x = 0.5 + r * np.cos(angle)
        y = 0.5 + r * np.sin(angle)
        points.append((x, y))
    
    points.append(points[0])
    return points


if __name__ == "__main__":
    if not PIL_AVAILABLE:
        print("⚠️  PIL/Pillow no está instalado. Instala con: pip install pillow")
        print("    Esto es necesario para simular la función de entrenamiento.\n")
    
    compare_preprocessing_methods()