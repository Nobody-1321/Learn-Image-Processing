import cv2
import numpy as np

def frame_differencing():
    # Cargar el video
    cap = cv2.VideoCapture(0)
    
    # Leer el primer fotograma
    ret, prev_frame = cap.read()
    if not ret:
        print("No se pudo leer el video.")
        return

    # Convertir el primer fotograma a escala de grises
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        # Leer el siguiente fotograma
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Convertir el fotograma actual a escala de grises
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular la diferencia absoluta entre el fotograma actual y el anterior
        diff = cv2.absdiff(curr_gray, prev_gray)
        
        # Aplicar un umbral para resaltar las áreas de movimiento
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Mostrar la diferencia umbralizada y el fotograma original
        cv2.imshow("Movimiento Detectado (Frame Differencing)", thresh)
        cv2.imshow("Fotograma Actual", curr_frame)
        
        # Actualizar el fotograma anterior
        prev_gray = curr_gray.copy()
        
        # Salir con la tecla 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Ruta del video
frame_differencing()
