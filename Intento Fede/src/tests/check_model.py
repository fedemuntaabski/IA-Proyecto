"""
check_model.py - Verifica las clases disponibles y prueba la predicción raw
"""

import numpy as np
import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

try:
    from model import SketchClassifier
    from drawing_preprocessor import DrawingPreprocessor
    import logging
except ImportError as e:
    print(f"Error importando: {e}")
    sys.exit(1)


def check_available_classes():
    """Verifica qué clases tiene el modelo."""
    print("="*80)
    print("VERIFICACIÓN DE CLASES DEL MODELO")
    print("="*80)
    
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    
    try:
        classifier = SketchClassifier("./IA", logger, demo_mode=False)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None
    
    labels = classifier.get_labels()
    print(f"\n✓ Modelo cargado con {len(labels)} clases\n")
    
    # Buscar clases relacionadas con "house"
    house_related = [l for l in labels if 'house' in l.lower()]
    print("🏠 Clases relacionadas con 'house':")
    if house_related:
        for label in house_related:
            print(f"   - {label}")
    else:
        print("   ❌ NO SE ENCONTRÓ 'house' en las clases del modelo")
        print("   ⚠️  ESTE ES EL PROBLEMA!")
    
    # Buscar clases que podrían confundirse
    similar = ['beard', 'face', 'moustache', 'The Mona Lisa', 'triangle', 'tent']
    print("\n🔍 Clases que el modelo predijo o relacionadas:")
    for label in similar:
        if label in labels:
            idx = labels.index(label)
            print(f"   ✓ {label} (índice: {idx})")
    
    # Mostrar todas las clases
    print(f"\n📋 TODAS LAS CLASES ({len(labels)}):")
    print("="*80)
    
    # Ordenar alfabéticamente
    sorted_labels = sorted(labels)
    
    # Mostrar en columnas
    cols = 4
    for i in range(0, len(sorted_labels), cols):
        row = sorted_labels[i:i+cols]
        print("   ".join(f"{j+i+1:3d}. {lbl:20s}" for j, lbl in enumerate(row)))
    
    return classifier, labels


def test_raw_prediction():
    """Prueba la predicción directa con el modelo."""
    print("\n" + "="*80)
    print("PRUEBA DE PREDICCIÓN RAW")
    print("="*80)
    
    # Cargar el dibujo del usuario
    stroke_file = "test_stroke_data.json"
    if not os.path.exists(stroke_file):
        print(f"❌ No se encontró {stroke_file}")
        return
    
    with open(stroke_file, 'r') as f:
        strokes_data = json.load(f)
    
    strokes = [[(float(p[0]), float(p[1])) for p in stroke] for stroke in strokes_data]
    print(f"✓ Cargados {len(strokes)} trazos")
    
    # Setup
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    
    classifier = SketchClassifier("./IA", logger, demo_mode=False)
    preprocessor = DrawingPreprocessor((28, 28, 1))
    
    # Preprocesar
    img = preprocessor.preprocess(strokes)
    
    print(f"\n📊 Imagen preprocesada:")
    print(f"   Shape: {img.shape}")
    print(f"   Dtype: {img.dtype}")
    print(f"   Rango: [{img.min():.4f}, {img.max():.4f}]")
    print(f"   Media: {img.mean():.4f}")
    print(f"   Desv. Std: {img.std():.4f}")
    
    # Predecir y obtener probabilidades raw
    try:
        # Expandir dimensión de batch
        img_batch = np.expand_dims(img, axis=0)
        print(f"   Batch shape: {img_batch.shape}")
        
        # Predecir
        predictions = classifier.model.predict(img_batch, verbose=0)
        print(f"\n🎯 Predicciones raw:")
        print(f"   Shape: {predictions.shape}")
        print(f"   Suma de probabilidades: {predictions.sum():.4f}")
        
        # Top 10 predicciones
        top_indices = np.argsort(predictions[0])[::-1][:10]
        labels = classifier.get_labels()
        
        print(f"\n📊 Top 10 Predicciones:")
        print("-"*60)
        for i, idx in enumerate(top_indices, 1):
            label = labels[idx]
            prob = predictions[0][idx]
            bar = "█" * int(prob * 50)
            print(f"   {i:2d}. {label:25s} {prob:6.2%} {bar}")
        
        # Verificar si hay algo relacionado con "house" en el top 20
        print(f"\n🔍 Buscando 'house' en top 20:")
        top20_indices = np.argsort(predictions[0])[::-1][:20]
        house_found = False
        for idx in top20_indices:
            label = labels[idx]
            if 'house' in label.lower():
                prob = predictions[0][idx]
                rank = list(top20_indices).index(idx) + 1
                print(f"   ✓ Encontrado '{label}' en posición {rank} ({prob:.2%})")
                house_found = True
        
        if not house_found:
            print(f"   ❌ 'house' no está en el top 20")
            print(f"   ⚠️  El modelo probablemente NO fue entrenado con 'house'")
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        import traceback
        traceback.print_exc()


def suggest_solutions(labels):
    """Sugiere soluciones basadas en las clases disponibles."""
    print("\n" + "="*80)
    print("💡 SOLUCIONES SUGERIDAS")
    print("="*80)
    
    has_house = any('house' in l.lower() for l in labels)
    
    if not has_house:
        print("\n❌ PROBLEMA CONFIRMADO: El modelo NO tiene la clase 'house'")
        print("\n📋 Opciones:")
        print("\n1. RE-ENTRENAR EL MODELO incluyendo 'house':")
        print("   - Descarga datos de Quick Draw para 'house'")
        print("   - Agrega 'house' a tu lista de clases")
        print("   - Re-entrena el modelo")
        
        print("\n2. USAR CLASES DISPONIBLES:")
        print("   - Dibuja objetos que el modelo SÍ conoce")
        print("   - Revisa la lista completa arriba")
        
        print("\n3. VERIFICAR EL ENTRENAMIENTO:")
        print("   - ¿Usaste el dataset correcto?")
        print("   - ¿El archivo model_info.json tiene las clases correctas?")
        
        # Verificar model_info.json
        model_info_path = Path("./IA/model_info.json")
        if model_info_path.exists():
            print("\n📄 Contenido de model_info.json:")
            try:
                with open(model_info_path, 'r') as f:
                    info = json.load(f)
                    print(f"   Clases en archivo: {len(info.get('labels', []))}")
                    if 'house' in info.get('labels', []):
                        print("   ✓ 'house' ESTÁ en model_info.json")
                    else:
                        print("   ❌ 'house' NO está en model_info.json")
            except Exception as e:
                print(f"   Error leyendo archivo: {e}")
    else:
        print("\n✓ El modelo tiene 'house', pero no lo está prediciendo correctamente")
        print("\n📋 Posibles causas:")
        print("   1. Datos de entrenamiento de 'house' muy diferentes a tu estilo")
        print("   2. Pocos ejemplos de 'house' en el entrenamiento")
        print("   3. Problema de normalización de entrada")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔬 VERIFICACIÓN COMPLETA DEL MODELO")
    print("="*80)
    
    # Verificar clases
    result = check_available_classes()
    
    if result:
        classifier, labels = result
        
        # Probar predicción
        test_raw_prediction()
        
        # Sugerir soluciones
        suggest_solutions(labels)
    
    print("\n" + "="*80)
    print("✅ VERIFICACIÓN COMPLETADA")
    print("="*80)