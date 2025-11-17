import cv2
import numpy as np
import time # Añadimos time para obtener la hora actual para el cálculo de FPS
from typing import List, Tuple, Optional, Dict, Any


# ==========================
# Paleta Cyberpunk (violeta / rosa neon)
# ==========================
COLORS = {
    # Fondos (Dark Blue/Cian Base)
    "bg_panel": (10, 20, 40),        # Azul oscuro profundo para el fondo principal
    "bg_card": (25, 45, 60),         # Azul grisáceo oscuro para tarjetas y paneles
    
    # Acentos Neón (Cian, Verde, Azul Eléctrico)
    "accent": (255, 255, 0),         # Cian brillante / Amarillo Neón (Máximo B y G)
    "accent2": (255, 160, 0),        # Azul Eléctrico (Sustituto del violeta claro)
    "success": (100, 255, 100),      # Verde Agua Neón (Para la predicción principal exitosa)

    # Texto y Alertas
    "text_main": (235, 235, 235), 
    "text_dim": (180, 180, 190),
    "warning": (0, 100, 255),
}

# Fuentes
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
FONT_REG = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


def rounded_rectangle(img, pt1, pt2, color, radius=12, thickness=-1, alpha=1.0):
    """
    Dibuja un rectangulo con esquinas redondeadas usando circulos y rectangulos.
    Si alpha<1.0, fusiona con transparencia.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 <= x1 or y2 <= y1:
        return img

    overlay = img.copy()

    # rect central
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    # esquinas
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    
    # Fusionar con transparencia
    # Nota: Si thickness es -1, queremos el relleno. Si es > 0, es solo borde.
    # Esta implementación del borde es compleja y no se usará aquí directamente 
    # para el borde exterior, pero mantenemos la lógica de fusión para la transparencia
    if alpha < 1.0:
        img[:] = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)
    else:
        # Si no hay transparencia (alpha=1.0), solo superponer el relleno
        if thickness == -1:
            img[y1:y2, x1:x2] = overlay[y1:y2, x1:x2]

    return img


def draw_glow_rect(img, pt1, pt2, color, glow_width=8):
    """
    Dibuja un 'glow' (resplandor) alrededor de un rectangulo mediante varias capas
    semitransparentes. color en BGR.
    """
    for i in range(glow_width, 0, -2):
        # Reducir el alpha base para un glow mas sutil
        alpha = (i / (glow_width + 2)) * 0.10 
        c = tuple(int(min(255, max(0, v))) for v in color)
        # Ajustamos el radio para que sea mas suave
        rounded_rectangle(img, (pt1[0]-i, pt1[1]-i), (pt2[0]+i, pt2[1]+i), c, radius=14 + i//2, thickness=-1, alpha=alpha)


class PictionaryUI:
    """Renderiza la interfaz grafica de Pictionary Live - estilo Cyberpunk."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.frame_count = 0
        self.last_time = time.time() # Usamos time.time() real
        self.fps = 0.0
        self.last_prediction = None

    # --------------------------------------------------------
    def update_fps(self, current_time: float):
        """Actualiza contador de FPS."""
        self.frame_count += 1
        # Usamos 0.5 segundos para una actualizacion mas responsiva
        if current_time - self.last_time >= 0.5: 
            elapsed = max(current_time - self.last_time, 1e-3)
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time

    # --------------------------------------------------------
    def _fit_text_scale(self, text: str, font, max_width: int, base_scale: float = 0.9) -> float:
        """
        Ajusta la escala del texto para que quepa en max_width.
        """
        scale = base_scale
        th = cv2.getTextSize(text, font, scale, 2)[0][0]
        # Bucle mas eficiente y con un limite de escala logico
        while th > max_width and scale > 0.3:
            scale -= 0.02 # Paso mas fino para ajuste
            th = cv2.getTextSize(text, font, scale, 2)[0][0]
        return max(scale, 0.3)

    # --------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        hand_detected: bool,
        stroke_points: int,
        hand_velocity: float,
        prediction: Optional[Tuple[str, float, List[Tuple[str, float]]]] = None,
        hand_in_fist: bool = False,
    ) -> np.ndarray:

        h, w = frame.shape[:2]

        # Guardar prediccion persistente
        if prediction:
            self.last_prediction = prediction

        # ======================================================
        # Fondo sutil: Oscurecer frame para que la UI resalte (SOLO UN POCO)
        # Esto permite que la camara se vea.
        # ======================================================
        # Aplica un overlay muy ligero sobre el video de la camara (alpha=0.2)
        overlay_bg = frame.copy()
        cv2.rectangle(overlay_bg, (0, 0), (w, h), COLORS["bg_panel"], -1)
        frame[:] = cv2.addWeighted(frame, 0.8, overlay_bg, 0.2, 0) # 80% camara, 20% overlay oscuro

        # ======================================================
        # HEADER holografico
        # ======================================================
        header_h = max(96, int(h * 0.13))
        # Fondo de la barra de encabezado
        rounded_rectangle(frame, (8, 8), (w - 8, header_h), COLORS["bg_card"], radius=14, thickness=-1, alpha=0.95)

        # ligero 'hologram strip' con gradient vertical simulado por lineas
        for i in range(6):
            x_start = 18 + i * 12
            cv2.line(frame, (x_start, 20), (x_start + int((w - 40) * 0.2), 20 + header_h//2),
                     COLORS["accent2"], 1, cv2.LINE_AA)

        # TITULO (sin tildes ni simbolos especiales)
        title = "PICTIONARY LIVE"
        # Usamos una escala fija y la limitamos para consistencia
        title_scale = 1.1 
        cv2.putText(frame, title, (30, 44),
                    FONT_BOLD, title_scale, COLORS["accent"], 2, cv2.LINE_AA)


        # Estado mano a la derecha del header, ajustado para no superponer con el FPS
        hand_txt = "Mano detectada: SI" if hand_detected else "Mano detectada: NO"
        # Ajustamos las coordenadas Y para que esten separadas
        cv2.putText(frame, hand_txt, (w - 300, 44), FONT_REG, 0.6, COLORS["text_dim"], 1, cv2.LINE_AA)

        # FPS grande pero comedido
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (w - 300, 74), FONT_REG, 0.6, COLORS["accent2"], 1, cv2.LINE_AA)

        # linea divisoria fina
        cv2.line(frame, (12, header_h + 2), (w - 12, header_h + 2), COLORS["accent2"], 1, cv2.LINE_AA)

        # ======================================================
        # TARJETA DERECHA: Prediccion (si existe)
        # ======================================================
        if self.last_prediction:
            label, conf, top3 = self.last_prediction
            # Dimensiones ajustadas para una mejor visualizacion
            card_w = max(280, int(w * 0.20)) # Ancho ligeramente mayor
            card_h = max(180, int(h * 0.25)) # Alto para acomodar el Top3 sin solapamiento
            x2 = w - 18
            x1 = x2 - card_w
            y1 = header_h + 18
            y2 = y1 + card_h

            # glow exterior
            draw_glow_rect(frame, (x1, y1), (x2, y2), COLORS["accent"], glow_width=10)

            # Fondo de la tarjeta
            rounded_rectangle(frame, (x1, y1), (x2, y2), COLORS["bg_card"], radius=12, thickness=-1, alpha=0.92)

            # cabecera de la tarjeta
            # Aumentamos el tamaño y color del titulo de la tarjeta
            cv2.putText(frame, "Prediccion", (x1 + 16, y1 + 30),
                        FONT_BOLD, 0.6, COLORS["accent"], 1, cv2.LINE_AA)
            
            # Linea divisoria dentro de la tarjeta
            cv2.line(frame, (x1 + 10, y1 + 40), (x2 - 10, y1 + 40), COLORS["accent2"], 1)

            # etiqueta principal (la palabra con mayor confianza)
            label_clean = label.replace("ñ", "n")
            # Dejamos un espacio considerable para la palabra principal y su escala
            lab_scale = self._fit_text_scale(label_clean, FONT_BOLD, max_width=card_w - 40, base_scale=1.5)
            # DIBUJO DE LA PALABRA PRINCIPAL
            cv2.putText(frame, label_clean.upper(), (x1 + 16, y1 + 80), 
                        FONT_BOLD, lab_scale, COLORS["success"], 2, cv2.LINE_AA) 

            # confianza (debajo de la palabra principal)
            conf_txt = f"{conf*100:.1f}%"
            conf_scale = 0.8 # Aumentamos el tamaño para que sea mas visible
            # DIBUJO DEL PORCENTAJE DE CONFIANZA
            cv2.putText(frame, conf_txt, (x1 + 16, y1 + 115), 
                        FONT_BOLD, conf_scale, COLORS["text_main"], 1, cv2.LINE_AA)

            
       # ======================================================
        # CONTROLES INFERIORES (texto compacto para no solapar)
        # ======================================================
        controls = "Q = salir   |   S = guardar   |   Dibuja con tu dedo indice"
        # Cambiamos la escala base de 0.55 a 0.45 para reducir el tamaño
        ctrl_scale = self._fit_text_scale(controls, FONT_REG, max_width=int(w * 0.9), base_scale=0.45)
        # Ajustamos la posicion vertical de h - 18 a h - 14 para moverlo un poco arriba
        cv2.putText(frame, controls, (20, h - 14), FONT_REG, ctrl_scale, COLORS["text_dim"], 1, cv2.LINE_AA)
        # ======================================================
        # ALERTA: Mano en puño
        # ======================================================
        if hand_in_fist:
            msg = "MANO CERRADA - DIBUJO EN PAUSA"
            # calcular caja segun tamaño del texto
            t_scale = 0.6
            tsize = cv2.getTextSize(msg, FONT_BOLD, t_scale, 2)[0]
            
            # Posicionamiento: centrada horizontalmente y justo debajo del header
            rx_padding = 24
            ry_padding = 10
            rx1 = (w - tsize[0]) // 2 - rx_padding
            ry1 = header_h + 18
            rx2 = rx1 + tsize[0] + 2 * rx_padding
            ry2 = ry1 + tsize[1] + 2 * ry_padding

            draw_glow_rect(frame, (rx1, ry1), (rx2, ry2), COLORS["warning"], glow_width=8)
            rounded_rectangle(frame, (rx1, ry1), (rx2, ry2), COLORS["bg_card"], radius=10, thickness=-1, alpha=0.9)
            # Dibujo del texto de la alerta
            cv2.putText(frame, msg, (rx1 + rx_padding, ry1 + tsize[1] + ry_padding), FONT_BOLD, t_scale, COLORS["text_main"], 1, cv2.LINE_AA)

        return frame

    # --------------------------------------------------------
    def draw_stroke_preview(self, frame: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Dibuja vista previa del trazo con estilo cyberpunk.
        'points' son coordenadas normalizadas en [0,1] relativas a (w,h).
        """
        if not points or len(points) < 2:
            return frame

        h, w = frame.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y in points]

        # lineas principales (mas gruesas)
        for i in range(1, len(pts)):
            thickness = 3 
            cv2.line(frame, pts[i - 1], pts[i], COLORS["accent"], thickness)

        # brillo extra con lines finas violeta (superpuesto)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], COLORS["accent2"], 1)

        # punto final con glow
        fx, fy = pts[-1]
        for r in range(12, 4, -3): # Circulos de glow mas grandes
            # Reducir el alpha del glow para que sea sutil
            alpha = (r / 15.0) * 0.15 
            overlay = frame.copy()
            cv2.circle(overlay, (fx, fy), r, COLORS["accent"], -1)
            # Fusionar con transparencia
            frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        # Punto central del final del trazo
        cv2.circle(frame, (fx, fy), 2, COLORS["text_main"], -1) 

        return frame


# ======================================================
# EJEMPLO DE USO (Para probar la interfaz)
# ======================================================

def run_pictionary_live():
    """Funcion de prueba para inicializar y correr la interfaz de la camara."""
    
    # Simular una configuracion basica
    config = {"camera_id": 0}
    ui = PictionaryUI(config)

    # Simular datos de prediccion
    # Cambiamos 'arm' y 'dog' por algo que se veia en tu imagen
    mock_prediction = ("guitar", 0.65, [
        ("arm", 0.35),
        ("dog", 0.18),
        ("hat", 0.10)
    ])
    
    # Inicializar camara
    cap = cv2.VideoCapture(config["camera_id"])
    if not cap.isOpened():
        print("Error: No se pudo abrir la camara.")
        return

    # Intento de configurar resolucion (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Estado de simulacion
    hand_detected = True
    hand_in_fist = False
    stroke_points = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            break
        
        # Reflejar la imagen para que se sienta como un espejo (comun en camaras web)
        frame = cv2.flip(frame, 1)

        # 1. Actualizar FPS
        ui.update_fps(time.time())
        
        # 2. Renderizar la interfaz
        # Usamos la prediccion simulada para que se muestre
        display_frame = ui.render(
            frame=frame,
            hand_detected=hand_detected,
            stroke_points=len(stroke_points),
            hand_velocity=0.0,
            prediction=mock_prediction,
            hand_in_fist=hand_in_fist
        )

        # 3. Dibujar el trazo (simulado)
        # Puedes simular el trazo aqui si quieres
        # ui.draw_stroke_preview(display_frame, stroke_points)

        # 4. Mostrar
        cv2.imshow("Pictionary Live - Cyberpunk", display_frame)

        # 5. Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # Q o ESC para salir
            break
        
        # Simular deteccion de mano para probar el estado
        if key == ord('t'):
            hand_detected = not hand_detected
            print(f"Hand Detected toggled to: {hand_detected}")
            
        # Simular puño para probar la alerta
        if key == ord('f'):
            hand_in_fist = not hand_in_fist
            print(f"Hand In Fist toggled to: {hand_in_fist}")
            
        # Simular guardar (simplemente un mensaje)
        if key == ord('s'):
            print("Guardando imagen...")

    cap.release()
    cv2.destroyAllWindows()

# Ejecuta la funcion de prueba si el script es el principal
if __name__ == '__main__':
    run_pictionary_live()