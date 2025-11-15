# IA Proyecto - Clasificador de Sketches para Pictionary

Este proyecto implementa un clasificador de sketches basado en deep learning para un entrenador de IA en el juego Pictionary.

## Descripción

El sistema utiliza una red neuronal convolucional (CNN) entrenada con el dataset Quick Draw de Google para reconocer dibujos en tiempo real. El modelo puede identificar 228 clases diferentes de objetos, desde "The Eiffel Tower" hasta "zebra".

## Características

- **Arquitectura**: CNN profunda con 4 bloques convolucionales
- **Precisión**: 80.26% en conjunto de prueba
- **Eficiencia**: Generadores de datos para manejo óptimo de memoria
- **Análisis progresivo**: Evalúa cuánto del dibujo necesita la IA para adivinar correctamente

## Archivos incluidos

- `PictionaryTrainer.ipynb`: Notebook completo de entrenamiento
- `sketch_classifier_model.h5`: Modelo entrenado (formato HDF5)
- `sketch_classifier_model.keras`: Modelo entrenado (formato Keras moderno)
- `model_info.json`: Metadatos del modelo (clases, precisión, parámetros)

## Requisitos

- Python 3.x
- TensorFlow/Keras
- NumPy, Matplotlib, PIL

## Uso

1. Ejecutar el notebook `PictionaryTrainer.ipynb` para entrenar el modelo
2. Los modelos entrenados están listos para inferencia

## Dataset

El proyecto utiliza el dataset "Quick, Draw!" de Google (no incluido en este repositorio).

## Licencia

Proyecto educativo - uso libre para fines de aprendizaje.