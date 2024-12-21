import cv2
import numpy as np

def triple_frame_differencing():
    # Cargar el video
    cap = cv2.VideoCapture(0)
    
    # Leer los primeros dos fotogramas
    ret, frame1 = cap.read()
    if not ret:
        print("No se pudo leer el video.")
        return
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    ret, frame2 = cap.read()
    if not ret:
        print("No hay suficientes fotogramas en el video.")
        return
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    while True:
        # Leer el siguiente fotograma
        ret, frame3 = cap.read()
        if not ret:
            break
        frame3_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        
        # Calcular diferencias consecutivas
        diff1 = cv2.absdiff(frame2_gray, frame1_gray)
        diff2 = cv2.absdiff(frame3_gray, frame2_gray)
        
        # Intersección de las diferencias
        motion = cv2.bitwise_and(diff1, diff2)
        
        # Aplicar un umbral para resaltar áreas de movimiento
        _, thresh = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)
        
        # Mostrar la diferencia umbralizada y el fotograma original
        cv2.imshow("Movimiento Detectado (Triple Frame Differencing)", thresh)
        cv2.imshow("Fotograma Actual", frame3)
        
        # Actualizar los fotogramas anteriores
        frame1_gray = frame2_gray
        frame2_gray = frame3_gray
        
        # Salir con la tecla 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Ruta del video

triple_frame_differencing()
