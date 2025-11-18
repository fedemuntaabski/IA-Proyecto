"""
test_live_diagnostic_enhanced.py - Diagnóstico con comparación de ejemplos del dataset
Muestra: Tu dibujo | Lo que el modelo ve | Lo que el modelo espera
"""

import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from typing import List, Tuple, Any

# ======================================================
# CORRECCIÓN DE RUTA
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

try:
    from ui import PictionaryUI
    from drawing_preprocessor import DrawingPreprocessor
    from model import SketchClassifier
    import logging
except ImportError as e:
    print(f"Error: No se pudo importar módulos. Asegúrate de que estén en: {parent_dir}")
    print(f"Detalles: {e}")
    sys.exit(1)


def load_strokes(filename: str) -> List[List[Tuple[float, float]]]:
    """Carga los trazos desde un archivo JSON y los convierte a tuplas."""
    if not os.path.exists(filename):
        print(f"Error: Archivo de trazos '{filename}' no encontrado.")
        print("¡Debes dibujarlo y presionar 'S' primero!")
        return []
        
    with open(filename, 'r') as f:
        strokes_list = json.load(f)
    
    return [[(float(p[0]), float(p[1])) for p in stroke] for stroke in strokes_list]


def load_dataset_sample(label: str, sample_idx: int = 0, max_samples: int = 5):
    """
    Carga ejemplos del dataset de entrenamiento para la clase dada.
    
    Args:
        label: Nombre de la clase (ej: 'house', 'clock', 'flower')
        sample_idx: Índice inicial del ejemplo
        max_samples: Número máximo de ejemplos a cargar
    
    Returns:
        Lista de imágenes numpy array (28, 28) o lista vacía
    """
    # Buscar en posibles ubicaciones del dataset
    possible_paths = [
        Path(f"./data/processed/{label}_train.npy"),
        Path(f"./data/{label}_train.npy"),
        Path(f"./quickdraw_data/{label}.npy"),
        Path(f"./IA/data/{label}.npy"),
        Path(f"../data/{label}.npy"),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                print(f"   📂 Encontrado: {path}")
                data = np.load(path)
                print(f"   📊 Dataset shape: {data.shape}")
                
                samples = []
                for i in range(sample_idx, min(sample_idx + max_samples, len(data))):
                    sample = data[i]
                    
                    # Normalizar formato
                    if sample.shape == (784,):
                        sample = sample.reshape(28, 28)
                    
                    # Normalizar rango [0, 1]
                    if sample.max() > 1.0:
                        sample = sample.astype(np.float32) / 255.0
                    
                    samples.append(sample)
                
                print(f"   ✅ Cargados {len(samples)} ejemplos")
                return samples
                
            except Exception as e:
                print(f"   ⚠️  Error cargando {path}: {e}")
    
    print(f"   ❌ No se encontró dataset para '{label}'")
    return []


def create_comparison_visualization(user_drawing, model_input, predicted_label, 
                                   dataset_samples, confidence, top3):
    """
    Crea una visualización completa comparando:
    1. Dibujo del usuario (vista previa UI)
    2. Lo que el modelo recibe (28x28 procesado)
    3. Ejemplos del dataset de la clase predicha
    
    Args:
        user_drawing: Frame con el dibujo del usuario
        model_input: Imagen procesada (28, 28, 1)
        predicted_label: Etiqueta predicha
        dataset_samples: Lista de ejemplos del dataset
        confidence: Confianza de la predicción
        top3: Top 3 predicciones [(label, conf), ...]
    
    Returns:
        Imagen combinada
    """
    img_size = 280
    padding = 15
    text_height = 100
    
    # ===== COLUMNA 1: DIBUJO DEL USUARIO =====
    col1 = cv2.resize(user_drawing, (img_size, img_size))
    
    # ===== COLUMNA 2: INPUT DEL MODELO =====
    model_vis = (model_input.squeeze() * 255).astype('uint8')
    model_vis = cv2.resize(model_vis, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    model_vis = cv2.cvtColor(model_vis, cv2.COLOR_GRAY2RGB)
    
    # ===== COLUMNA 3+: EJEMPLOS DEL DATASET =====
    dataset_cols = []
    if dataset_samples:
        for sample in dataset_samples[:3]:  # Máximo 3 ejemplos
            sample_vis = (sample * 255).astype('uint8')
            sample_vis = cv2.resize(sample_vis, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            sample_vis = cv2.cvtColor(sample_vis, cv2.COLOR_GRAY2RGB)
            dataset_cols.append(sample_vis)
    
    # Si no hay ejemplos, crear placeholder
    if not dataset_cols:
        placeholder = np.ones((img_size, img_size, 3), dtype=np.uint8) * 40
        cv2.putText(placeholder, "Dataset no", (30, img_size//2 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.putText(placeholder, "encontrado", (30, img_size//2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        dataset_cols.append(placeholder)
    
    # ===== AGREGAR TÍTULOS Y BORDES =====
    all_cols = []
    border_size = 3
    
    # Columna 1
    header1 = np.ones((text_height, img_size, 3), dtype=np.uint8) * 30
    cv2.putText(header1, "TU DIBUJO", (10, 35), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header1, "(Vista previa UI)", (10, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    col1_bordered = cv2.copyMakeBorder(col1, border_size, border_size, border_size, border_size, 
                                       cv2.BORDER_CONSTANT, value=(0, 255, 255))
    # Asegurar que el header y la columna tengan el mismo ancho
    col1_bordered_resized = cv2.resize(col1_bordered, (img_size, img_size))
    col1_full = np.vstack([header1, col1_bordered_resized])
    all_cols.append(col1_full)
    
    # Columna 2
    header2 = np.ones((text_height, img_size, 3), dtype=np.uint8) * 30
    cv2.putText(header2, "MODELO VE", (10, 35), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(header2, f"(28x28 procesado)", (10, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    col2_bordered = cv2.copyMakeBorder(model_vis, border_size, border_size, border_size, border_size, 
                                       cv2.BORDER_CONSTANT, value=(0, 255, 0))
    col2_bordered_resized = cv2.resize(col2_bordered, (img_size, img_size))
    col2_full = np.vstack([header2, col2_bordered_resized])
    all_cols.append(col2_full)
    
    # Columnas 3+: Ejemplos del dataset
    for i, dataset_col in enumerate(dataset_cols):
        header = np.ones((text_height, img_size, 3), dtype=np.uint8) * 30
        cv2.putText(header, f"DATASET: {predicted_label.upper()}", (10, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
        cv2.putText(header, f"Ejemplo #{i+1}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        col_bordered = cv2.copyMakeBorder(dataset_col, border_size, border_size, border_size, border_size, 
                                          cv2.BORDER_CONSTANT, value=(255, 165, 0))
        col_bordered_resized = cv2.resize(col_bordered, (img_size, img_size))
        col_full = np.vstack([header, col_bordered_resized])
        all_cols.append(col_full)
    
    # ===== COMBINAR COLUMNAS =====
    combined = all_cols[0]
    for col in all_cols[1:]:
        pad = np.ones((combined.shape[0], padding, 3), dtype=np.uint8) * 20
        combined = np.hstack([combined, pad, col])
    
    # ===== AGREGAR PANEL DE PREDICCIONES =====
    info_height = 120
    info_panel = np.ones((info_height, combined.shape[1], 3), dtype=np.uint8) * 25
    
    # Predicción principal
    pred_text = f"PREDICCION: {predicted_label.upper()}"
    conf_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.4 else (0, 0, 255)
    cv2.putText(info_panel, pred_text, (20, 35), 
               cv2.FONT_HERSHEY_DUPLEX, 1.0, conf_color, 2, cv2.LINE_AA)
    
    conf_text = f"Confianza: {confidence:.1%}"
    cv2.putText(info_panel, conf_text, (20, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Top 3
    top3_text = "Top 3: "
    for i, (label, conf) in enumerate(top3[:3]):
        top3_text += f"{i+1}. {label} ({conf:.0%})  "
    cv2.putText(info_panel, top3_text, (20, 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # ===== COMBINAR TODO =====
    pad_top = np.ones((padding, combined.shape[1], 3), dtype=np.uint8) * 20
    final = np.vstack([info_panel, pad_top, combined])
    
    return final


def run_live_diagnostic_test(stroke_filename: str = 'test_stroke_data.json'):
    """
    Carga trazos reales, los preprocesa, predice y muestra comparación completa.
    """
    
    # Cargar trazos
    real_strokes = load_strokes(stroke_filename)
    if not real_strokes:
        return

    print(f"\n{'='*80}")
    print(f"DIAGNÓSTICO EN VIVO: Análisis Completo")
    print(f"{'='*80}")
    print(f"✓ Cargados {len(real_strokes)} trazos desde {stroke_filename}")
    
    # 1. SETUP
    W, H = 640, 480 
    TARGET_SHAPE = (28, 28, 1)

    ui = PictionaryUI(config={})
    preprocessor = DrawingPreprocessor(target_shape=TARGET_SHAPE)
    
    # Cargar modelo
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    
    print("\n📦 Cargando modelo...")
    try:
        classifier = SketchClassifier("./IA", logger, demo_mode=False)
        print(f"✓ Modelo cargado: {len(classifier.get_labels())} clases disponibles")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # 2. GENERAR VISTA PREVIA (Lo que el usuario ve)
    print("\n🎨 Generando vista previa del dibujo...")
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    preview_frame = frame.copy()
    
    all_points_flat = [p for stroke in real_strokes for p in stroke]
    preview_frame = ui.draw_stroke_preview(preview_frame, all_points_flat)
    
    draw_size = 350
    start_y, start_x = (H - draw_size) // 2, (W - draw_size) // 2
    user_draw_vis = preview_frame[start_y:start_y + draw_size, start_x:start_x + draw_size].copy()

    # 3. PREPROCESAR (Lo que el modelo ve)
    print("\n🔧 Preprocesando para el modelo...")
    model_input = preprocessor.preprocess(real_strokes)
    
    print(f"   Shape: {model_input.shape}")
    print(f"   Rango: [{model_input.min():.3f}, {model_input.max():.3f}]")
    print(f"   Media: {model_input.mean():.3f}")
    print(f"   Píxeles blancos: {np.sum(model_input > 0.01)}/{model_input.size}")
    
    # 4. PREDECIR
    print("\n🤖 Realizando predicción...")
    predicted_label, confidence, top3 = classifier.predict(model_input)
    
    print(f"\n{'='*80}")
    print(f"🎯 RESULTADO DE PREDICCIÓN")
    print(f"{'='*80}")
    print(f"   Predicción: {predicted_label}")
    print(f"   Confianza: {confidence:.1%}")
    print(f"\n   Top 3:")
    for i, (label, conf) in enumerate(top3[:3], 1):
        print(f"      {i}. {label}: {conf:.1%}")
    print(f"{'='*80}\n")
    
    # 5. CARGAR EJEMPLOS DEL DATASET
    print(f"📚 Cargando ejemplos del dataset para '{predicted_label}'...")
    dataset_samples = load_dataset_sample(predicted_label, sample_idx=0, max_samples=3)
    
    # 6. CREAR VISUALIZACIÓN COMPARATIVA
    print("\n🖼️  Generando visualización comparativa...")
    comparison = create_comparison_visualization(
        user_draw_vis, 
        model_input, 
        predicted_label, 
        dataset_samples, 
        confidence, 
        top3
    )
    
    # 7. MOSTRAR
    window_name = "DIAGNÓSTICO COMPLETO: Tu Dibujo vs Dataset"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, comparison)
    
    print("\n✅ Visualización lista!")
    print("   👀 Compara tu dibujo con los ejemplos del dataset")
    print("   📊 ¿Son similares? Si no, ajusta tu técnica de dibujo")
    print("\n   Presiona cualquier tecla para cerrar...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Guardar resultado
    output_dir = Path("diagnosis_results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"comparison_{predicted_label}.png"
    cv2.imwrite(str(output_path), comparison)
    print(f"\n💾 Resultado guardado en: {output_path}")


if __name__ == '__main__':
    run_live_diagnostic_test()