
#-------------------------- Video --------------------------
from ultralytics import RTDETR
import cv2
import collections

# Cargar el modelo
model = RTDETR('model_balanceado.pt')

# Ruta del archivo de video
video_path = "henrymnoche12.mp4"

# Inicializar el video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10  # Si FPS es 0, usar 10 por defecto
print(f"FPS del video: {fps}")

# Obtener las dimensiones del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializar el escritor de video para guardar el video procesado
output_video_path = "henrymnoche12_output.mp4" 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Archivo de salida
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Parámetros de análisis
WINDOW_SECONDS = 1.5  # Analizar cada 1.5 segundos
FRAME_WINDOW = int(fps * WINDOW_SECONDS)  # Cantidad de frames en la ventana
print(f"Ventana de tiempo: {WINDOW_SECONDS} segundos ({FRAME_WINDOW} frames)")
PERCLOS_THRESHOLD = 30  # Somnolencia si más del 30% son closed eye

# Cola para almacenar el historial de detecciones en la ventana
eye_states = collections.deque(maxlen=FRAME_WINDOW)  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Termina si ya no hay más frames

    # Detección con el modelo
    results = model(frame)

    # Verificar si `results` es lista y obtener el primer elemento
    if isinstance(results, list) and len(results) > 0:
        results = results[0]

    eyes_closed = 0  # Asumimos que los ojos están abiertos

    # Si hay detecciones
    if hasattr(results, 'boxes'):
        for box in results.boxes:
            cls = int(box.cls.item())  
            confidence = box.conf.item()  
            x1, y1, x2, y2 = map(int, box.xyxy[0])  

            # Dibujar cajas y etiquetas
            if confidence > 0.5:
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                label = f"{'open eye' if cls == 0 else 'closed eye'}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if cls == 1:  
                    eyes_closed = 1  

    # Agregar el estado actual de los ojos a la cola
    eye_states.append(eyes_closed)
    print(f"Estado de los ojos: {eye_states}")

    # Calcular PERCLOS en la última ventana de tiempo
    perc = (sum(eye_states) / len(eye_states)) * 100 if eye_states else 0
    # Detectar somnolencia si el PERCLOS supera el umbral
    somnolence_detected = perc >= PERCLOS_THRESHOLD
    # Mostrar información en pantalla
    cv2.putText(frame, f"PERCLOS: {perc:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"{'Somnolencia detectada' if somnolence_detected else ' '}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Guardar el frame en el video de salida
    out.write(frame)
    # Mostrar el frame
    cv2.imshow("Detector de somnolencia", frame)

    # Si presionamos 'q', salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
