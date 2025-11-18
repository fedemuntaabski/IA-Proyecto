# Nuevas Funcionalidades - Pictionary Live

## Resumen de Cambios

Se han implementado las siguientes mejoras al modo de juego de Pictionary Live:

### 1. ⏱️ Timer de 2 Minutos
- **Ubicación**: Panel derecho de la interfaz, arriba del puntaje
- **Funcionalidad**: Cuenta regresiva desde 2:00 minutos
- **Características**:
  - Se actualiza cada segundo
  - Cambia de color según tiempo restante:
    - 🔵 Cyan (> 60 segundos)
    - 🟠 Naranja (30-60 segundos)
    - 🔴 Rojo (< 30 segundos)

### 2. 🎯 Objetivo Aleatorio
- **Ubicación**: Panel derecho, en un cuadrito destacado con fondo naranja
- **Funcionalidad**: 
  - Muestra una **palabra aleatoria** que el usuario debe dibujar
  - Las palabras se cargan desde `IA/model_info.json` (228 clases disponibles)
  - Cambia con la tecla **S** (siguiente)
  - Al acertar, automáticamente se selecciona un nuevo objetivo

### 3. 🏆 Sistema de Puntaje
- **Ubicación**: Panel derecho, debajo del timer
- **Funcionalidad**:
  - Comienza en 0
  - Se incrementa automáticamente en +1 cada vez que la predicción coincide con el objetivo
  - La comparación es case-insensitive
  - Muestra el puntaje en verde con fuente grande
  - **Predicción automática continua**: El sistema predice mientras dibujas y al cerrar la mano

### 4. ✋ Comportamiento del Puño
- **IMPORTANTE**: Cerrar el puño **SOLO guarda el trazo actual**
- **NO borra** el dibujo
- **NO hace predicción** automáticamente
- El dibujo persiste en pantalla hasta que uses la tecla **C** para limpiar

### 5. 🎨 Sistema de Dibujo Refactorizado

**Dibuja con el dedo índice:**
- El trazo se **pinta en tiempo real** siguiendo el movimiento del dedo índice
- El dibujo aparece como una **línea verde brillante** superpuesta al video de la cámara
- Es completamente visible y persistente

**Cuando cierras la mano:**
- El trazo actual se **guarda** automáticamente
- El dibujo **NO se borra** - permanece visible en la pantalla
- Puedes abrir la mano y seguir dibujando (acumulando más trazos)

**El dibujo solo se elimina cuando:**
1. Presionas **C** (limpiar manualmente)
2. El modelo **predice correctamente** lo que dibujaste (acierta el objetivo)

**Modos de dibujo:**
- **Modo Mano**: 
  - Mano abierta con dedo índice extendido: dibuja línea verde en tiempo real
  - Cierra el puño: guarda el trazo (el dibujo persiste visible)
  - Abre la mano: continúa dibujando sobre lo anterior
  
- **Modo Mouse**:
  - Click y arrastra: dibuja línea verde en tiempo real
  - Suelta el click: guarda el trazo (el dibujo persiste visible)
  - Click de nuevo: continúa dibujando sobre lo anterior

## Controles del Teclado

| Tecla | Acción |
|-------|--------|
| **Q** | Salir de la aplicación |
| **C** | **Limpiar el tablero** de la cámara |
| **S** | **Siguiente objetivo** (cambia la palabra sin limpiar) |
| **R** | Reiniciar el juego (resetea timer, puntaje, limpia y nueva palabra) |

**Nota**: La predicción se realiza **automáticamente** mientras dibujas y al guardar cada trazo.

## Flujo del Juego

1. **Inicio**: Se muestra un objetivo aleatorio (palabra) y el timer comienza en 2:00
2. **Dibujar con dedo índice**: 
   - **Modo Mano**: Extiende el dedo índice y muévelo por la pantalla → aparece línea verde siguiendo tu trazo
   - **Modo Mouse**: Click y arrastra → aparece línea verde siguiendo el cursor
   - **Predicción automática**: El sistema predice continuamente mientras dibujas
3. **Guardar trazo**:
   - **Modo Mano**: Cierra el puño → el trazo se guarda y queda visible en pantalla
   - **Modo Mouse**: Suelta el click → el trazo se guarda y queda visible en pantalla
   - Se realiza una predicción automática al guardar
4. **Continuar dibujando**: Abre la mano (o haz click) y sigue dibujando sobre lo anterior
5. **Acierto automático**: Si la predicción coincide con el objetivo:
   - Se suma 1 punto automáticamente
   - Se selecciona un nuevo objetivo automáticamente
   - **El dibujo se borra automáticamente** (canvas limpio para el nuevo objetivo)
6. **Limpiar**: Presiona **C** cuando quieras borrar el dibujo manualmente
7. **Siguiente**: Presiona **S** si quieres cambiar de palabra sin limpiar el dibujo
8. **Continuar**: El usuario sigue dibujando hasta que se acabe el tiempo
9. **Reiniciar**: Presionar **R** para comenzar un nuevo juego completo

## Estructura de la Interfaz

```
┌─────────────────────────────────────────────────────────────┐
│  🎮 PICTIONARY LIVE          🟢 ✋ MODO MANO                │
├────────────────────────────────┬────────────────────────────┤
│                                │  🎯 OBJETIVO               │
│                                │  ┌──────────────────────┐  │
│                                │  │      AMBULANCE       │  │
│         CÁMARA                 │  └──────────────────────┘  │
│     (dibujos persistentes)     │                            │
│                                │  ⏱️ TIEMPO                 │
│                                │     02:00                  │
│                                │                            │
│                                │  🏆 PUNTAJE                │
│                                │       0                    │
│                                │                            │
│                                │  🤖 PREDICCIÓN             │
│                                │  (Presiona P para pred.)   │
├────────────────────────────────┴────────────────────────────┤
│  Q=Salir | C=Limpiar | S=Siguiente | P=Predecir | R=Reinic.│
└─────────────────────────────────────────────────────────────┘
```

## Archivos Modificados

1. **src/ui_pyqt.py**:
   - Nueva clase `GameCard` para mostrar objetivo, timer y puntaje
   - Método `_update_game_timer()` para actualizar el timer
   - Métodos `set_target()`, `reset_timer()`, `reset_score()`
   - Teclas: C=Limpiar, S=Siguiente, P=Predecir, R=Reiniciar
   - Nueva señal `predict_requested` para predicción manual
   - Limpieza automática al acertar el objetivo

2. **src/app_pyqt.py**:
   - **Sistema de dibujo refactorizado**:
     - `overlay_canvas`: Canvas RGBA (640x480) que se superpone al video para mostrar trazos en verde
     - `drawing_canvas`: Canvas interno (256x256) para predicción del modelo
     - El dibujo se pinta en tiempo real siguiendo el dedo índice
     - Los trazos persisten visiblemente hasta limpiar con C o acertar
   - **Predicción automática continua**:
     - Predice cada 10 frames mientras el usuario dibuja (si hay suficiente contenido)
     - Predice automáticamente al cerrar el puño (guardar trazo)
     - Usa el modelo `sketch_classifier_model.h5/keras` en tiempo real
   - Método `_load_labels()` para cargar etiquetas desde model_info.json
   - Método `_select_random_target()` para seleccionar objetivo aleatorio
   - Cerrar puño **guarda el trazo + predice automáticamente**
   - Limpiar canvas al acertar objetivo o presionar C
   - Modo mouse actualizado para usar overlay_canvas

3. **src/config_manager.py**:
   - Ruta automática a config.yaml en el directorio del proyecto

## Notas Técnicas

- Las 228 etiquetas se cargan desde `IA/model_info.json`
- La comparación entre predicción y objetivo es case-insensitive
- El timer se actualiza mediante `QTimer` con intervalo de 1 segundo
- El puntaje persiste durante toda la sesión hasta presionar R
- **Sistema de doble canvas**:
  - `overlay_canvas` (640x480 RGBA): Muestra trazos verdes sobre el video en tiempo real
  - `drawing_canvas` (256x256 grayscale): Canvas interno para predicción del modelo
- **Predicción automática continua**:
  - Se ejecuta cada 10 frames mientras dibujas (si hay más de 100 píxeles dibujados)
  - Se ejecuta automáticamente al cerrar el puño (guardar trazo, si hay más de 50 píxeles)
  - Usa el modelo `sketch_classifier_model.h5` o `sketch_classifier_model.keras`
  - Compara automáticamente con el objetivo aleatorio
  - Suma +1 punto automáticamente al acertar
- **El dibujo persiste visiblemente** hasta:
  - Presionar C (limpiar manual)
  - Acertar la predicción (limpia automáticamente)
- **Cerrar el puño = Guardar trazo + Predecir automáticamente**
- Los trazos se dibujan con líneas verdes (0, 255, 0) de grosor 8px sobre el video

## Ejecución

```powershell
# Usar Python 3.12 (compatible con MediaPipe)
py -3.12 main.py
```

## Resumen de Teclas

```
┌─────────┬──────────────────────────────────────┐
│  Tecla  │  Acción                              │
├─────────┼──────────────────────────────────────┤
│    Q    │  Salir                               │
│    C    │  Limpiar tablero                     │
│    S    │  Siguiente objetivo (nueva palabra)  │
│    R    │  Reiniciar juego completo            │
└─────────┴──────────────────────────────────────┘

Comportamiento del Dibujo:
  • Dedo índice extendido  → Dibuja línea verde en tiempo real
  • Sistema predice        → Automáticamente mientras dibujas
  • Cierra puño            → Guarda trazo + predice
  • Abre mano              → Continúa dibujando
  • Acierta predicción     → +1 punto, borra automáticamente, nueva palabra
  • Presiona C             → Borra manualmente
```
